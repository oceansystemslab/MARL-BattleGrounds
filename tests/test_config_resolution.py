"""Resolved episode-profile contracts for Milestone 5 Step 2 CP3B."""

from typing import cast

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from marl_battlegrounds.core import combat
from marl_battlegrounds.core.config import resolve_agent_profile
from marl_battlegrounds.core.types import (
    HUNTER_CLASS_ID,
    MAGE_CLASS_ID,
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    MAX_OBSTACLE_SLOTS,
    NEUTRAL_CLASS_ID,
    NO_TEAM_ID,
    OBSTACLE_FEATURES,
    PRIEST_CLASS_ID,
    ROGUE_CLASS_ID,
    TEAM_A_ID,
    TEAM_B_ID,
    WARRIOR_CLASS_ID,
    EnvConfig,
    ResolvedAgentProfile,
)

_CANONICAL_ROSTER = jnp.asarray(
    (
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
    ),
    dtype=jnp.int32,
)


def _expected_active_mask(team_sizes: tuple[int, int]) -> Array:
    local_slots = jnp.arange(MAX_AGENTS_PER_TEAM, dtype=jnp.int32)
    return jnp.concatenate(
        (local_slots < team_sizes[0], local_slots < team_sizes[1]), axis=0
    )


def _expected_team_ids(team_sizes: tuple[int, int]) -> Array:
    active_mask = _expected_active_mask(team_sizes)
    slot_team_ids = jnp.concatenate(
        (
            jnp.full((MAX_AGENTS_PER_TEAM,), TEAM_A_ID, dtype=jnp.int32),
            jnp.full((MAX_AGENTS_PER_TEAM,), TEAM_B_ID, dtype=jnp.int32),
        ),
        axis=0,
    )
    return jnp.where(active_mask, slot_team_ids, NO_TEAM_ID).astype(jnp.int32)


@pytest.mark.parametrize(
    ("requested_class_ids", "team_sizes"),
    (
        pytest.param(_CANONICAL_ROSTER, (5, 5), id="canonical-mirrored-5v5"),
        pytest.param(
            jnp.asarray(
                (
                    MAGE_CLASS_ID,
                    PRIEST_CLASS_ID,
                    HUNTER_CLASS_ID,
                    ROGUE_CLASS_ID,
                    WARRIOR_CLASS_ID,
                    ROGUE_CLASS_ID,
                    HUNTER_CLASS_ID,
                    MAGE_CLASS_ID,
                    PRIEST_CLASS_ID,
                    WARRIOR_CLASS_ID,
                ),
                dtype=jnp.int32,
            ),
            (1, 2),
            id="asymmetric-mage-vs-rogue-hunter",
        ),
        pytest.param(
            jnp.asarray(
                (
                    HUNTER_CLASS_ID,
                    HUNTER_CLASS_ID,
                    MAGE_CLASS_ID,
                    ROGUE_CLASS_ID,
                    PRIEST_CLASS_ID,
                    WARRIOR_CLASS_ID,
                    WARRIOR_CLASS_ID,
                    WARRIOR_CLASS_ID,
                    HUNTER_CLASS_ID,
                    MAGE_CLASS_ID,
                ),
                dtype=jnp.int32,
            ),
            (2, 3),
            id="duplicate-class-asymmetric-composition",
        ),
        pytest.param(
            jnp.full((MAX_AGENT_SLOTS,), ROGUE_CLASS_ID, dtype=jnp.int32),
            (0, 1),
            id="padded-non-neutral-roster",
        ),
    ),
)
def test_resolve_agent_profile_establishes_fixed_slots_and_neutral_padding(
    requested_class_ids: Array,
    team_sizes: tuple[int, int],
) -> None:
    resolved = resolve_agent_profile(
        requested_class_ids,
        jnp.asarray(team_sizes, dtype=jnp.int32),
    )
    expected_active_mask = _expected_active_mask(team_sizes)
    expected_class_ids = jnp.where(
        expected_active_mask, requested_class_ids, NEUTRAL_CLASS_ID
    ).astype(jnp.int32)

    assert resolved.class_ids.shape == (MAX_AGENT_SLOTS,)
    assert resolved.class_ids.dtype == jnp.int32
    assert bool(jnp.array_equal(resolved.class_ids, expected_class_ids))

    assert resolved.team_ids.shape == (MAX_AGENT_SLOTS,)
    assert resolved.team_ids.dtype == jnp.int32
    assert bool(jnp.array_equal(resolved.team_ids, _expected_team_ids(team_sizes)))

    assert resolved.active_mask.shape == (MAX_AGENT_SLOTS,)
    assert resolved.active_mask.dtype == jnp.bool_
    assert bool(jnp.array_equal(resolved.active_mask, expected_active_mask))


@pytest.mark.parametrize(
    ("profile_field", "catalog"),
    (
        ("agent_radii", combat.BODY_RADIUS_BY_CLASS),
        ("base_movement_speeds", combat.BASE_MOVEMENT_SPEED_BY_CLASS),
        ("observation_radii", combat.OBSERVATION_RADIUS_BY_CLASS),
        ("basic_interaction_radii", combat.BASIC_INTERACTION_RADIUS_BY_CLASS),
        ("ultimate_interaction_radii", combat.ULTIMATE_INTERACTION_RADIUS_BY_CLASS),
        ("max_health", combat.MAX_HEALTH_BY_CLASS),
    ),
)
def test_resolved_agent_profile_stats_match_combat_catalogs(
    profile_field: str,
    catalog: Array,
) -> None:
    resolved = resolve_agent_profile(
        _CANONICAL_ROSTER,
        jnp.asarray((3, 2), dtype=jnp.int32),
    )
    actual = cast(Array, getattr(resolved, profile_field))
    expected = catalog[resolved.class_ids]

    assert actual.shape == (MAX_AGENT_SLOTS,)
    assert actual.dtype == jnp.float32
    assert bool(jnp.array_equal(actual, expected))


def test_resolved_profile_uses_tuned_melee_basic_radii() -> None:
    resolved = resolve_agent_profile(
        _CANONICAL_ROSTER,
        jnp.asarray((5, 5), dtype=jnp.int32),
    )

    expected = jnp.asarray(
        (5.0, 1.5, 5.5, 1.5, 5.0, 5.0, 1.5, 5.5, 1.5, 5.0),
        dtype=jnp.float32,
    )

    assert bool(jnp.array_equal(resolved.basic_interaction_radii, expected))


def test_resolved_agent_profile_names_base_speed_without_competing_alias() -> None:
    assert "base_movement_speeds" in ResolvedAgentProfile._fields
    assert "movement_speeds" not in ResolvedAgentProfile._fields


def test_resolve_agent_profile_is_stable_under_jit_and_pytree_flattening() -> None:
    team_sizes = jnp.asarray((2, 4), dtype=jnp.int32)
    eager = resolve_agent_profile(_CANONICAL_ROSTER, team_sizes)
    jitted = cast(
        ResolvedAgentProfile,
        jax.jit(resolve_agent_profile)(_CANONICAL_ROSTER, team_sizes),
    )

    assert jax.tree_util.tree_structure(eager) == jax.tree_util.tree_structure(jitted)
    eager_leaves = jax.tree_util.tree_leaves(eager)
    jitted_leaves = jax.tree_util.tree_leaves(jitted)
    assert len(eager_leaves) == len(ResolvedAgentProfile._fields)

    for eager_leaf, jitted_leaf in zip(eager_leaves, jitted_leaves, strict=True):
        assert bool(jnp.array_equal(eager_leaf, jitted_leaf))


def test_env_config_packages_only_the_resolved_profile_for_roster_facts() -> None:
    profile = resolve_agent_profile(
        _CANONICAL_ROSTER,
        jnp.asarray((3, 3), dtype=jnp.int32),
    )
    config = EnvConfig(
        max_steps=1000,
        map_width=20.0,
        map_height=12.0,
        obstacles=jnp.zeros((MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES), dtype=jnp.float32),
        agent_profile=profile,
        initial_agent_positions=jnp.asarray(
            (
                (0.5, 0.5),
                (2.5, 0.5),
                (4.5, 0.5),
                (0.0, 0.0),
                (0.0, 0.0),
                (0.5, 7.5),
                (2.5, 7.5),
                (4.5, 7.5),
                (0.0, 0.0),
                (0.0, 0.0),
            ),
            dtype=jnp.float32,
        ),
        ordinary_movement_distance_scale=1.0,
    )

    assert "agent_profile" in EnvConfig._fields
    assert "team_size" not in EnvConfig._fields
    assert "initial_class_ids" not in EnvConfig._fields
    assert config.agent_profile is profile

    def _max_health_from_config(resolved_config: EnvConfig) -> Array:
        return resolved_config.agent_profile.max_health

    jitted_max_health = cast(
        Array,
        jax.jit(_max_health_from_config)(config),
    )
    assert bool(jnp.array_equal(jitted_max_health, profile.max_health))
