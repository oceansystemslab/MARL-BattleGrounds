"""Small deterministic targeting and general/specialist movement primitives."""

import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.axis_mappings import (
    UNIT_DIRECTION_VECTOR_BY_MOVEMENT_ACTION_ARRAY,
)
from marl_battlegrounds.core.geometry import (
    GEOMETRY_TOLERANCE,
    project_movement_with_geometry,
)
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
BYPASS_MINIMUM_CONTACT_ANGLE_DEGREES = 45


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


def _body_clear_moves(
    origin: Array,
    endpoints: Array,
    radius: Array,
    bodies: Array,
    body_mask: Array,
) -> Array:
    """Screen eight displacement segments against fixed observed body discs.

    This is a conservative local steering preference, not simulator collision
    resolution. Existing overlaps allow only outward motion; the target and
    self have already been removed from ``body_mask`` by the caller.
    """
    travel = endpoints - origin
    start = origin - centers(bodies)
    travel_squared = jnp.sum(jnp.square(travel), axis=-1)
    initial_alignment = jnp.sum(travel[:, None, :] * start[None, :, :], axis=-1)
    fraction = jnp.clip(
        -initial_alignment / jnp.where(travel_squared > 0, travel_squared, 1)[:, None],
        0,
        1,
    )
    closest = start[None, :, :] + fraction[:, :, None] * travel[:, None, :]
    closest_squared = jnp.sum(jnp.square(closest), axis=-1)
    start_squared = jnp.sum(jnp.square(start), axis=-1)
    end_squared = jnp.sum(
        jnp.square(endpoints[:, None, :] - centers(bodies)[None, :, :]), axis=-1
    )
    required_squared = jnp.square(radius + bodies[:, AGENT_FEATURE_RADIUS])
    escaping = (initial_alignment >= 0) & (end_squared > start_squared)
    clear = jnp.where(
        start_squared < required_squared,
        escaping,
        closest_squared >= required_squared,
    )
    return jnp.all(~body_mask[None, :] | clear, axis=1)


def _body_bypass_moves(
    origin: Array,
    endpoints: Array,
    radius: Array,
    bodies: Array,
    body_mask: Array,
) -> Array:
    """Admit clear segments and shoulder contact at least 45 degrees from head-on.

    Incidence uses the first contact normal, not the initial direction to a
    distant body. Tiny solver overlap counts as contact; deeper overlap retains
    strict outward escape. This changes steering only, never physical radii.
    """
    travel = endpoints - origin
    start = origin - centers(bodies)
    travel_squared = jnp.sum(jnp.square(travel), axis=-1)[:, None]
    safe_travel_squared = jnp.where(travel_squared > 0, travel_squared, 1)
    alignment = jnp.sum(travel[:, None, :] * start[None, :, :], axis=-1)
    start_squared = jnp.sum(jnp.square(start), axis=-1)
    required_radius = radius + bodies[:, AGENT_FEATURE_RADIUS]
    required_squared = jnp.square(required_radius)

    closest_time = jnp.clip(-alignment / safe_travel_squared, 0, 1)
    closest = start[None, :, :] + closest_time[:, :, None] * travel[:, None, :]
    clear = jnp.sum(jnp.square(closest), axis=-1) >= required_squared

    discriminant = jnp.square(alignment) - travel_squared * (
        start_squared - required_squared
    )
    first_contact_time = jnp.clip(
        (-alignment - jnp.sqrt(jnp.maximum(discriminant, 0))) / safe_travel_squared,
        0,
        1,
    )
    start_distance = jnp.sqrt(start_squared)
    touching = start_distance <= required_radius
    first_contact_time = jnp.where(
        touching,
        0,
        first_contact_time,
    )
    normal = start[None, :, :] + first_contact_time[:, :, None] * travel[:, None, :]
    inward = -jnp.sum(travel[:, None, :] * normal, axis=-1)
    lateral = (
        travel[:, None, 0] * normal[:, :, 1] - travel[:, None, 1] * normal[:, :, 0]
    )
    # At 45 degrees the lateral and inward components are equal. Comparing
    # them directly also avoids rounding a tiny inward step into a clear path.
    glancing = jnp.abs(lateral) >= inward
    end_squared = jnp.sum(
        jnp.square(endpoints[:, None, :] - centers(bodies)[None, :, :]), axis=-1
    )
    escaping = (alignment >= 0) & (end_squared > start_squared)
    admissible = jnp.where(
        start_distance < required_radius - GEOMETRY_TOLERANCE,
        escaping,
        (~touching & clear) | glancing,
    )
    return jnp.all(~body_mask[None, :] | admissible, axis=1)


def _body_aware_move(  # pyright: ignore[reportUnusedFunction]
    observation: Observation,
    action_mask: ActionMask,
    prey_center: Array,
    bodies: Array,
    body_mask: Array,
    *,
    allow_glancing_contact: bool = False,
) -> Array:
    """Choose the closest safe moving endpoint, even if a detour retreats."""
    origin = centers(observation.self_features)
    speed = observation.self_features[AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED]
    radius = observation.self_features[AGENT_FEATURE_RADIUS]
    directions = UNIT_DIRECTION_VECTOR_BY_MOVEMENT_ACTION_ARRAY[1:]
    positions = jnp.broadcast_to(origin, (MAX_AGENT_SLOTS, 2))
    deltas = jnp.zeros_like(positions).at[:8].set(directions * speed)
    active = jnp.arange(MAX_AGENT_SLOTS) < 8
    no_bodies = jnp.zeros(MAX_AGENT_SLOTS, dtype=jnp.bool_)
    endpoints = project_movement_with_geometry(
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
    displacement = jnp.sqrt(jnp.sum(jnp.square(endpoints - origin), axis=-1))
    body_screen = _body_bypass_moves if allow_glancing_contact else _body_clear_moves
    admissible = (
        action_mask.move_mask[1:]
        & (speed > 0)
        & (displacement >= MINIMUM_MOVEMENT_FRACTION * speed)
        & body_screen(origin, endpoints, radius, bodies, body_mask)
    )
    distance_squared = jnp.sum(jnp.square(endpoints - prey_center), axis=-1)
    move = 1 + jnp.argmin(jnp.where(admissible, distance_squared, jnp.inf))
    return jnp.where(jnp.any(admissible), move, MOVE_STAY).astype(jnp.int32)
