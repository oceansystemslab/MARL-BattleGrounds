"""Shared JAX-compatible geometry helpers for the simulator core."""

from typing import cast

import jax
import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.types import (
    OBSTACLE_FEATURE_ACTIVE,
    OBSTACLE_FEATURE_RADIUS,
    OBSTACLE_FEATURE_TYPE,
    OBSTACLE_FEATURE_X,
    OBSTACLE_FEATURE_Y,
    OBSTACLE_TYPE_PILLAR,
)

GEOMETRY_EPSILON = 1e-6
GEOMETRY_TOLERANCE = 1e-5
DEFAULT_MOVEMENT_SUBSTEPS = 4
DEFAULT_AGENT_PROJECTION_PASSES = 4

# Private geometry helpers ---


def _project_overlapping_pillar(
    distance_vector: Array,
    pillar_center: Array,
    distance_between_centers: Array,
    sum_of_radii: Array | float,
    center: Array,
) -> Array:
    """Project a disc that is known to overlap an active pillar."""
    has_measured_direction = distance_between_centers > GEOMETRY_EPSILON

    return cast(
        Array,
        jax.lax.cond(
            has_measured_direction,
            _project_pillar_with_measured_direction,
            _project_pillar_with_fallback_direction,
            distance_vector,
            pillar_center,
            distance_between_centers,
            sum_of_radii,
            center,
        ),
    )


def _project_pillar_with_measured_direction(
    distance_vector: Array,
    pillar_center: Array,
    distance_between_centers: Array,
    sum_of_radii: Array | float,
    center: Array,
) -> Array:
    """Project an overlapping disc using the measured center-to-center direction."""
    del center

    unit_direction = distance_vector / distance_between_centers

    return pillar_center + sum_of_radii * unit_direction


def _project_pillar_with_fallback_direction(
    distance_vector: Array,
    pillar_center: Array,
    distance_between_centers: Array,
    sum_of_radii: Array | float,
    center: Array,
) -> Array:
    """Project an overlapping disc using a deterministic fallback direction."""
    del distance_vector, distance_between_centers, center

    fallback_direction = jnp.array([1.0, 0.0], dtype=jnp.float32)

    return pillar_center + sum_of_radii * fallback_direction


def _keep_pillar_projection_center(
    distance_vector: Array,
    pillar_center: Array,
    distance_between_centers: Array,
    sum_of_radii: Array | float,
    center: Array,
) -> Array:
    """Return the original center when pillar projection is unnecessary."""
    del distance_vector, pillar_center, distance_between_centers, sum_of_radii

    return center


def _project_disc_out_of_active_pillar(
    center: Array,
    radius: Array | float,
    obstacle: Array,
) -> Array:
    """Project a disc out of an already-validated active pillar row."""
    pillar_center = jnp.stack(
        (
            obstacle[OBSTACLE_FEATURE_X],
            obstacle[OBSTACLE_FEATURE_Y],
        )
    )
    pillar_radius = obstacle[OBSTACLE_FEATURE_RADIUS]

    distance_vector = center - pillar_center
    distance_between_centers = cast(Array, jnp.linalg.norm(distance_vector))
    sum_of_radii = radius + pillar_radius
    projection_needed = cast(bool, distance_between_centers < sum_of_radii)

    return cast(
        Array,
        jax.lax.cond(
            projection_needed,
            _project_overlapping_pillar,
            _keep_pillar_projection_center,
            distance_vector,
            pillar_center,
            distance_between_centers,
            sum_of_radii,
            center,
        ),
    )


def _keep_obstacle_projection_center(
    center: Array,
    radius: Array | float,
    obstacle: Array,
) -> Array:
    """Return the original center for inactive or non-pillar obstacle rows."""
    del radius, obstacle

    return center


# Public geometry kernels ---


def project_disc_to_bounds(
    center: Array,
    radius: Array | float,
    map_width: Array | float,
    map_height: Array | float,
) -> Array:
    """Project a disc center inside the rectangular map bounds.

    Args:
        center: Disc center with shape ``(2,)``.
        radius: Disc radius.
        map_width: Width of the rectangular map.
        map_height: Height of the rectangular map.

    Returns:
        Disc center clipped to the inward-shrunk valid center region.
    """
    center_x = jnp.clip(center[0], min=radius, max=map_width - radius)
    center_y = jnp.clip(center[1], min=radius, max=map_height - radius)

    return jnp.stack((center_x, center_y))


def project_disc_out_of_pillar(
    center: Array,
    radius: Array | float,
    obstacle: Array,
) -> Array:
    """Project a disc center out of one active pillar obstacle row.

    If the obstacle row is inactive or is not a pillar, the input center is returned
    unchanged. If the row is an active pillar and the disc overlaps it, the center is
    moved to the closest non-overlapping position along a deterministic outward
    direction.

    Args:
        center: Disc center with shape ``(2,)``.
        radius: Disc radius.
        obstacle: Obstacle row with shape ``(OBSTACLE_FEATURES,)``.

    Returns:
        Projected disc center with shape ``(2,)``.
    """
    is_active_pillar = jnp.logical_and(
        obstacle[OBSTACLE_FEATURE_TYPE] == OBSTACLE_TYPE_PILLAR,
        obstacle[OBSTACLE_FEATURE_ACTIVE] == 1.0,
    )

    return cast(
        Array,
        jax.lax.cond(
            is_active_pillar,
            _project_disc_out_of_active_pillar,
            _keep_obstacle_projection_center,
            center,
            radius,
            obstacle,
        ),
    )


def project_disc_out_of_wall() -> Array:
    """
    Input: one center position, one agent radius, one obstacle row.
    Output: one center position.
    Responsibility: if the active obstacle is a wall and the disc overlaps the
    rotated rectangle, move the disc to non-overlap. If the row is inactive or not
    a wall, leave the position unchanged.
    Invariant: wall rotation uses the row's theta field and local-frame geometry.
    """
    raise NotImplementedError


def project_disc_out_of_obstacle() -> Array:
    """
    Input: one center position, one agent radius, one obstacle row.
    Output: one center position.
    Responsibility: dispatch between pillar, wall, and inactive/no-op obstacle
    cases using JAX-compatible masking rather than Python branching on JAX values.
    """
    raise NotImplementedError


def project_disc_out_of_obstacles() -> Array:
    """
    Input: one center position, one agent radius, full obstacle array.
    Output: one center position.
    Responsibility: _project one disc against every active obstacle slot with a
    fixed iteration pattern.
    """
    raise NotImplementedError


def resolve_agent_agent_overlaps() -> Array:
    """
    Input: all positions, all radii, active mask, alive mask, and fixed pass count.
    Output: all positions.
    Responsibility: resolve hard non-overlap between active alive agents. Inactive,
    padded, or dead slots must not push or be pushed.
    Invariant: same-team and enemy collisions use the same body-blocking rule.
    """
    raise NotImplementedError


def project_movement_with_geometry() -> Array:
    """
    Input: all current positions, radii, intended displacements, active mask,
    alive mask, map dimensions, obstacle array, fixed substep count, and fixed
    agent-projection pass count.
    Output: all projected positions.
    Responsibility: split intended movement into fixed substeps, apply map and
    obstacle projection to active alive agents, and run fixed-pass agent-agent
    non-overlap.
    Invariant: inactive or dead slots preserve their original positions.
    """
    raise NotImplementedError


def segment_intersects_circle() -> Array:
    """
    Input: segment start, segment end, circle center, circle radius.
    Output: scalar bool JAX array.
    Responsibility: detect whether a closed line segment intersects a blocking
    circle, including tangent or near-tangent contact.
    """
    raise NotImplementedError


def segment_intersects_rotated_rect() -> Array:
    """
    Input: segment start, segment end, one obstacle row representing a wall.
    Output: scalar bool JAX array.
    Responsibility: detect whether a closed line segment intersects a rotated
    rectangle.
    """
    raise NotImplementedError


def has_clear_line_of_sight() -> Array:
    """
    Input: segment start, segment end, full obstacle array.
    Output: scalar bool JAX array.
    Responsibility: return NotImplementedError false if any active pillar or wall
    blocks the segment.
    Invariant: agents are not inputs and do not block LOS in v1.
    Future use: observation construction and targetability must consume this
    helper, or a documented wrapper around it, so LOS-gated visibility and
    targetability cannot drift apart.
    """
    raise NotImplementedError
