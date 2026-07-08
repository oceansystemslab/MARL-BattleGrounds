"""Combat contract tests for Milestone 5 class setup."""

import jax
import jax.numpy as jnp
from jax import Array

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
