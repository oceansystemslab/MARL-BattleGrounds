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
from marl_battlegrounds.core.env import reset
from marl_battlegrounds.core.types import (
    AGENT_FEATURE_ACTIVE,
    AGENT_FEATURE_ALIVE,
    AGENT_FEATURE_BASIC_INTERACTION_RADIUS,
    AGENT_FEATURE_CLASS_ID,
    AGENT_FEATURE_MOVEMENT_SPEED,
    AGENT_FEATURE_OBSERVATION_RADIUS,
    AGENT_FEATURE_RADIUS,
    AGENT_FEATURE_TEAM_ID,
    AGENT_FEATURE_ULTIMATE_INTERACTION_RADIUS,
    AGENT_FEATURE_X,
    AGENT_FEATURE_Y,
    CLASS_NEUTRAL,
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
    "BASIC_INTERACTION_RADIUS_BY_CLASS",
    "BASIC_DAMAGE_BY_CLASS",
    "BASIC_HEALING_BY_CLASS",
    "ULTIMATE_INTERACTION_RADIUS_BY_CLASS",
    "OBSERVATION_RADIUS_BY_CLASS",
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
    ("OBSERVATION_RADIUS_BY_CLASS", "get_observation_radius_by_class_ids"),
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
    return EnvConfig(
        team_size=team_size,
        max_steps=1000,
        map_width=20.0,
        map_height=12.0,
        obstacles=_empty_obstacles(),
        initial_class_ids=class_ids,
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
    _assert_float_array_close(
        state.slow_multipliers,
        jnp.ones((MAX_AGENT_SLOTS, NUM_SLOW_CHANNELS), dtype=jnp.float32),
        "slow_multipliers",
    )
    _assert_array_equal(
        state.stun_durations,
        jnp.zeros((MAX_AGENT_SLOTS, NUM_STUN_CHANNELS), dtype=jnp.int32),
        "stun_durations",
    )
    _assert_array_equal(
        state.anti_heal_durations,
        jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
        "anti_heal_durations",
    )
    _assert_float_array_close(
        state.anti_heal_multipliers,
        jnp.ones((MAX_AGENT_SLOTS,), dtype=jnp.float32),
        "anti_heal_multipliers",
    )
    _assert_array_equal(
        state.damage_amplification_durations,
        jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
        "damage_amplification_durations",
    )
    _assert_float_array_close(
        state.damage_amplification_multipliers,
        jnp.ones((MAX_AGENT_SLOTS,), dtype=jnp.float32),
        "damage_amplification_multipliers",
    )
    _assert_array_equal(
        state.blessing_of_freedom_durations,
        jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
        "blessing_of_freedom_durations",
    )


def _assert_self_features_project_state(obs: Observation, state: EnvState) -> None:
    """Assert self observation rows expose current state feature columns."""
    assert obs.self_features.shape == (MAX_AGENT_SLOTS, SELF_FEATURES)

    _assert_float_array_close(
        obs.self_features[:, AGENT_FEATURE_X : AGENT_FEATURE_Y + 1],
        state.agent_positions,
        "obs.self_features.positions",
    )
    _assert_float_array_close(
        obs.self_features[:, AGENT_FEATURE_RADIUS],
        state.agent_radii,
        "obs.self_features.radius",
    )
    _assert_float_array_close(
        obs.self_features[:, AGENT_FEATURE_TEAM_ID],
        state.team_ids.astype(jnp.float32),
        "obs.self_features.team_id",
    )
    _assert_float_array_close(
        obs.self_features[:, AGENT_FEATURE_ACTIVE],
        state.active_mask.astype(jnp.float32),
        "obs.self_features.active",
    )
    _assert_float_array_close(
        obs.self_features[:, AGENT_FEATURE_ALIVE],
        state.alive_mask.astype(jnp.float32),
        "obs.self_features.alive",
    )
    _assert_float_array_close(
        obs.self_features[:, AGENT_FEATURE_CLASS_ID],
        state.class_ids.astype(jnp.float32),
        "obs.self_features.class_id",
    )
    _assert_float_array_close(
        obs.self_features[:, AGENT_FEATURE_MOVEMENT_SPEED],
        state.movement_speeds,
        "obs.self_features.movement_speed",
    )
    _assert_float_array_close(
        obs.self_features[:, AGENT_FEATURE_OBSERVATION_RADIUS],
        state.observation_radii,
        "obs.self_features.observation_radius",
    )
    _assert_float_array_close(
        obs.self_features[:, AGENT_FEATURE_BASIC_INTERACTION_RADIUS],
        state.basic_interaction_radii,
        "obs.self_features.basic_interaction_radius",
    )
    _assert_float_array_close(
        obs.self_features[:, AGENT_FEATURE_ULTIMATE_INTERACTION_RADIUS],
        state.ultimate_interaction_radii,
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
            combat.MOVEMENT_SPEED_BY_CLASS[ROGUE_CLASS_ID]
            > combat.MOVEMENT_SPEED_BY_CLASS[non_rogue_class_ids]
        )
    )


def test_warrior_basic_and_ultimate_interaction_radii_are_separate() -> None:
    assert bool(
        combat.ULTIMATE_INTERACTION_RADIUS_BY_CLASS[WARRIOR_CLASS_ID]
        > combat.BASIC_INTERACTION_RADIUS_BY_CLASS[WARRIOR_CLASS_ID]
    )


def test_mage_burst_is_representable_as_no_target_self_buff() -> None:
    assert bool(combat.ULTIMATE_INTERACTION_RADIUS_BY_CLASS[MAGE_CLASS_ID] == 0)
    assert combat.MAGE_ULT_DAMAGE_DURATION_TICKS > 0
    assert combat.MAGE_ULT_DAMAGE_MULTIPLIER > 1.0


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


def test_passive_and_support_defaults_are_scalar_parameters() -> None:
    assert combat.MAGE_DAMAGE_AURA_RADIUS > 0.0
    assert combat.MAGE_DAMAGE_AURA_MULTIPLIER > 1.0

    assert combat.WARRIOR_MITIGATION_AURA_RADIUS > 0.0
    assert 0.0 < combat.WARRIOR_MITIGATION_AURA_MULTIPLIER < 1.0

    assert combat.PRIEST_ULT_HEAL_AMOUNT > 0.0
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

    _assert_array_equal(state.active_mask, expected_active_mask, "active_mask")
    _assert_array_equal(state.alive_mask, expected_active_mask, "alive_mask")
    _assert_array_equal(state.class_ids, expected_class_ids, "class_ids")

    _assert_float_array_close(
        state.agent_radii,
        combat.BODY_RADIUS_BY_CLASS[expected_class_ids],
        "agent_radii",
    )
    _assert_float_array_close(
        state.movement_speeds,
        combat.MOVEMENT_SPEED_BY_CLASS[expected_class_ids],
        "movement_speeds",
    )
    _assert_float_array_close(
        state.observation_radii,
        combat.OBSERVATION_RADIUS_BY_CLASS[expected_class_ids],
        "observation_radii",
    )
    _assert_float_array_close(
        state.basic_interaction_radii,
        combat.BASIC_INTERACTION_RADIUS_BY_CLASS[expected_class_ids],
        "basic_interaction_radii",
    )
    _assert_float_array_close(
        state.ultimate_interaction_radii,
        combat.ULTIMATE_INTERACTION_RADIUS_BY_CLASS[expected_class_ids],
        "ultimate_interaction_radii",
    )
    _assert_float_array_close(
        state.max_health,
        combat.MAX_HEALTH_BY_CLASS[expected_class_ids],
        "max_health",
    )
    _assert_float_array_close(
        state.current_health,
        state.max_health,
        "current_health",
    )

    _assert_reset_combat_state_is_inert(state)
    _assert_self_features_project_state(obs, state)

    assert not bool(jnp.any(action_mask.move[~expected_active_mask]))
    assert not bool(jnp.any(action_mask.target[~expected_active_mask]))
    assert not bool(jnp.any(action_mask.use_ultimate[~expected_active_mask]))


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

    state, obs, _, _ = reset(
        _config(team_size=team_size, class_ids=class_ids),
        jax.random.key(42),
    )

    expected_active_mask = _expected_active_mask(team_size)
    expected_class_ids = _expected_resolved_class_ids(class_ids, team_size)
    inactive_mask = jnp.logical_not(expected_active_mask)

    _assert_array_equal(state.class_ids, expected_class_ids, "class_ids")

    assert bool(jnp.all(state.class_ids[inactive_mask] == NEUTRAL_CLASS_ID))
    assert bool(jnp.all(state.agent_radii[inactive_mask] == 0.0))
    assert bool(jnp.all(state.movement_speeds[inactive_mask] == 0.0))
    assert bool(jnp.all(state.observation_radii[inactive_mask] == 0.0))
    assert bool(jnp.all(state.basic_interaction_radii[inactive_mask] == 0.0))
    assert bool(jnp.all(state.ultimate_interaction_radii[inactive_mask] == 0.0))
    assert bool(jnp.all(state.max_health[inactive_mask] == 0.0))
    assert bool(jnp.all(state.current_health[inactive_mask] == 0.0))

    _assert_self_features_project_state(obs, state)


def test_reset_starts_ultimate_remaining_cooldowns_at_zero_not_catalog_durations() -> (
    None
):
    state, _, _, _ = reset(_config(team_size=5), jax.random.key(42))

    assert bool(jnp.all(state.ultimate_cooldowns == 0))

    active_mask = _expected_active_mask(team_size=5)
    active_class_ids = state.class_ids[active_mask]
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

    state, obs, _, _ = reset(
        _config(team_size=5, class_ids=class_ids),
        jax.random.key(42),
    )

    expected_observation_radii = combat.OBSERVATION_RADIUS_BY_CLASS[class_ids]

    _assert_float_array_close(
        state.observation_radii,
        expected_observation_radii,
        "state.observation_radii",
    )
    _assert_float_array_close(
        obs.self_features[:, AGENT_FEATURE_OBSERVATION_RADIUS],
        expected_observation_radii,
        "obs.self_features.observation_radius",
    )


def test_observation_relation_unit_feature_tables_keep_agent_schema_width() -> None:
    state, obs, _, _ = reset(_config(team_size=3), jax.random.key(42))

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

    _assert_self_features_project_state(obs, state)
