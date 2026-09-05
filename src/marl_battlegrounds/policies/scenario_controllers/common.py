"""Small deterministic target and static-world movement primitives."""

import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.axis_mappings import (
    UNIT_DIRECTION_VECTOR_BY_MOVEMENT_ACTION_ARRAY,
)
from marl_battlegrounds.core.geometry import project_movement_with_geometry
from marl_battlegrounds.core.types import (
    AGENT_FEATURE_ACTIVE,
    AGENT_FEATURE_ALIVE,
    AGENT_FEATURE_CURRENT_HEALTH,
    AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED,
    AGENT_FEATURE_MAX_HEALTH,
    AGENT_FEATURE_RADIUS,
    AGENT_FEATURE_X,
    AGENT_FEATURE_Y,
    CONTEXT_FEATURE_MAP_HEIGHT,
    CONTEXT_FEATURE_MAP_WIDTH,
    MAX_AGENT_SLOTS,
    MOVE_STAY,
    ActionMask,
    Observation,
)

MINIMUM_MOVEMENT_FRACTION = 0.1


def centers(features: Array) -> Array:
    """Read observed world-space centers from one or more unit rows."""
    return features[..., jnp.asarray([AGENT_FEATURE_X, AGENT_FEATURE_Y])]


def living_candidates(features: Array, visible: Array) -> Array:
    """Reject hidden, inactive, dead, and zero-health candidate rows."""
    return (
        visible
        & (features[:, AGENT_FEATURE_ACTIVE] > 0)
        & (features[:, AGENT_FEATURE_ALIVE] > 0)
        & (features[:, AGENT_FEATURE_CURRENT_HEALTH] > 0)
    )


def lowest_health_row(
    features: Array, eligible: Array, *, break_ties_by_max_health: bool = False
) -> Array:
    """Choose by absolute HP, optional maximum HP, then ascending row/slot.

    The fixed ally and enemy axes are each ordered by global slot. Callers
    retain their eligibility mask to distinguish an empty set from row zero.
    """
    health = features[:, AGENT_FEATURE_CURRENT_HEALTH]
    tied = eligible & (health == jnp.min(jnp.where(eligible, health, jnp.inf)))
    if break_ties_by_max_health:
        max_health = features[:, AGENT_FEATURE_MAX_HEALTH]
        tied &= max_health == jnp.min(jnp.where(tied, max_health, jnp.inf))
    return jnp.argmax(tied).astype(jnp.int32)


def nearest_row(features: Array, eligible: Array, origin: Array) -> Array:
    """Choose nearest center; exact ties use ascending row/global slot."""
    distance_squared = jnp.sum(jnp.square(centers(features) - origin), axis=-1)
    return jnp.argmin(jnp.where(eligible, distance_squared, jnp.inf)).astype(jnp.int32)


def refine_movement(
    observation: Observation, action_mask: ActionMask, intended_direction: Array
) -> Array:
    """Pick the closest legal direction with useful static-world displacement.

    Eight independent hypothetical moves occupy the first eight geometry slots.
    No body pair participates; the final two slots are inert padding. This uses
    precisely the simulator's bounded obstacle/boundary projection, not a new
    collision implementation or a prediction of other agents' actions.
    """
    origin = centers(observation.self_features)
    speed = observation.self_features[AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED]
    radius = observation.self_features[AGENT_FEATURE_RADIUS]
    directions = UNIT_DIRECTION_VECTOR_BY_MOVEMENT_ACTION_ARRAY[1:]
    positions = jnp.broadcast_to(origin, (MAX_AGENT_SLOTS, 2))
    deltas = jnp.zeros_like(positions).at[:8].set(directions * speed)
    active = jnp.arange(MAX_AGENT_SLOTS) < 8
    no_bodies = jnp.zeros(MAX_AGENT_SLOTS, dtype=jnp.bool_)
    projected = project_movement_with_geometry(
        positions,
        jnp.full(MAX_AGENT_SLOTS, radius, dtype=jnp.float32),
        deltas,
        active,
        active,
        observation.context_features[CONTEXT_FEATURE_MAP_WIDTH],
        observation.context_features[CONTEXT_FEATURE_MAP_HEIGHT],
        observation.map_obstacle_features,
        no_bodies,
        no_bodies,
        agent_agent_overlap_projection_passes=0,
    )[:8]
    displacement = jnp.sqrt(jnp.sum(jnp.square(projected - origin), axis=-1))
    direction_length = jnp.sqrt(jnp.sum(jnp.square(intended_direction)))
    alignment = directions @ (
        intended_direction / jnp.where(direction_length > 0, direction_length, 1.0)
    )
    admissible = (
        action_mask.move_mask[1:]
        & (speed > 0)
        & (direction_length > 0)
        & (displacement >= MINIMUM_MOVEMENT_FRACTION * speed)
    )
    best = 1 + jnp.argmax(jnp.where(admissible, alignment, -jnp.inf))
    return jnp.where(jnp.any(admissible), best, MOVE_STAY).astype(jnp.int32)
