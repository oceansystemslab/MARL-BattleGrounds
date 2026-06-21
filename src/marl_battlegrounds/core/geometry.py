"""Shared JAX-compatible geometry helpers for simulator movement and LOS."""

from typing import cast

import jax
import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.types import (
    OBSTACLE_FEATURE_ACTIVE,
    OBSTACLE_FEATURE_HEIGHT,
    OBSTACLE_FEATURE_RADIUS,
    OBSTACLE_FEATURE_THETA,
    OBSTACLE_FEATURE_TYPE,
    OBSTACLE_FEATURE_WIDTH,
    OBSTACLE_FEATURE_X,
    OBSTACLE_FEATURE_Y,
    OBSTACLE_TYPE_PILLAR,
    OBSTACLE_TYPE_WALL,
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
    """Resolve an agent-pillar overlap using a measured or fallback direction."""
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
    """Move an overlapping agent center to pillar tangency along the radial line."""
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
    """Move a coincident agent center out of a pillar along a fixed world axis."""
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
    """Keep an agent center unchanged when it does not overlap the pillar."""
    del distance_vector, pillar_center, distance_between_centers, sum_of_radii

    return center


def _project_disc_out_of_active_pillar(
    center: Array,
    radius: Array | float,
    obstacle: Array,
) -> Array:
    """Resolve collision between one agent disc and one active circular pillar."""
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


def _create_2d_rotation_matrix(theta: Array | float) -> Array:
    """Create the 2D rotation used to move between world and wall frames."""
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)

    return jnp.array(
        [
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta],
        ],
        dtype=jnp.float32,
    )


def _project_inside_disc_out_of_active_wall(
    agent_center_wall_local: Array,
    nearest_point_to_agent_center: Array,
    wall_x_bounds: Array,
    wall_y_bounds: Array,
    radius: Array | float,
) -> Array:
    """Push an agent center that is inside wall geometry through the nearest face."""
    del nearest_point_to_agent_center

    distance_to_left_face = jnp.abs(agent_center_wall_local[0] - wall_x_bounds[0])
    distance_to_right_face = jnp.abs(agent_center_wall_local[0] - wall_x_bounds[1])
    distance_to_bottom_face = jnp.abs(agent_center_wall_local[1] - wall_y_bounds[0])
    distance_to_top_face = jnp.abs(agent_center_wall_local[1] - wall_y_bounds[1])

    distances_to_faces = jnp.array(
        [
            distance_to_left_face,
            distance_to_right_face,
            distance_to_bottom_face,
            distance_to_top_face,
        ],
        dtype=jnp.float32,
    )
    nearest_face_index = jnp.argmin(distances_to_faces)

    branches = (
        _project_inside_disc_to_left_face_of_wall,
        _project_inside_disc_to_right_face_of_wall,
        _project_inside_disc_to_bottom_face_of_wall,
        _project_inside_disc_to_top_face_of_wall,
    )

    return cast(
        Array,
        jax.lax.switch(
            nearest_face_index,
            branches,
            agent_center_wall_local,
            wall_x_bounds,
            wall_y_bounds,
            radius,
        ),
    )


def _project_inside_disc_to_left_face_of_wall(
    agent_center_wall_local: Array,
    wall_x_bounds: Array,
    wall_y_bounds: Array,
    radius: Array | float,
) -> Array:
    """Place an inside-wall agent just outside the wall's local left face."""
    del wall_y_bounds

    return jnp.array(
        (
            wall_x_bounds[0] - radius,
            agent_center_wall_local[1],
        ),
        dtype=jnp.float32,
    )


def _project_inside_disc_to_right_face_of_wall(
    agent_center_wall_local: Array,
    wall_x_bounds: Array,
    wall_y_bounds: Array,
    radius: Array | float,
) -> Array:
    """Place an inside-wall agent just outside the wall's local right face."""
    del wall_y_bounds

    return jnp.array(
        (
            wall_x_bounds[1] + radius,
            agent_center_wall_local[1],
        ),
        dtype=jnp.float32,
    )


def _project_inside_disc_to_bottom_face_of_wall(
    agent_center_wall_local: Array,
    wall_x_bounds: Array,
    wall_y_bounds: Array,
    radius: Array | float,
) -> Array:
    """Place an inside-wall agent just outside the wall's local bottom face."""
    del wall_x_bounds

    return jnp.array(
        (
            agent_center_wall_local[0],
            wall_y_bounds[0] - radius,
        ),
        dtype=jnp.float32,
    )


def _project_inside_disc_to_top_face_of_wall(
    agent_center_wall_local: Array,
    wall_x_bounds: Array,
    wall_y_bounds: Array,
    radius: Array | float,
) -> Array:
    """Place an inside-wall agent just outside the wall's local top face."""
    del wall_x_bounds

    return jnp.array(
        (
            agent_center_wall_local[0],
            wall_y_bounds[1] + radius,
        ),
        dtype=jnp.float32,
    )


def _project_outside_disc_out_of_active_wall(
    agent_center_wall_local: Array,
    nearest_point_to_agent_center: Array,
    wall_x_bounds: Array,
    wall_y_bounds: Array,
    radius: Array | float,
) -> Array:
    """Resolve collision for an agent center outside a wall rectangle.

    The center is already outside the wall in wall-local coordinates. If the
    agent radius still overlaps the wall's solid area, move the center outward
    along the closest-point normal until the disc is tangent to the wall.
    """
    del wall_x_bounds, wall_y_bounds

    wall_to_agent_center_vector = (
        agent_center_wall_local - nearest_point_to_agent_center
    )
    distance_to_wall = cast(
        Array,
        jnp.linalg.norm(wall_to_agent_center_vector),
    )
    projection_needed = radius > distance_to_wall

    return cast(
        Array,
        jax.lax.cond(
            projection_needed,
            _project_overlapping_outside_disc_out_of_active_wall,
            _keep_wall_projection_center,
            agent_center_wall_local,
            wall_to_agent_center_vector,
            distance_to_wall,
            radius,
        ),
    )


def _project_overlapping_outside_disc_out_of_active_wall(
    agent_center_wall_local: Array,
    wall_to_agent_center_vector: Array,
    distance_to_wall: Array,
    radius: Array | float,
) -> Array:
    """Move an overlapping outside-wall agent along the wall contact normal."""
    direction_vector = wall_to_agent_center_vector / distance_to_wall
    violation_magnitude = radius - distance_to_wall

    return agent_center_wall_local + violation_magnitude * direction_vector


def _keep_wall_projection_center(
    agent_center_wall_local: Array,
    wall_to_agent_center_vector: Array,
    distance_to_wall: Array,
    radius: Array | float,
) -> Array:
    """Keep a wall-local agent center unchanged when there is no wall overlap."""
    del wall_to_agent_center_vector, distance_to_wall, radius

    return agent_center_wall_local


def _project_disc_out_of_active_wall(
    center: Array,
    radius: Array | float,
    obstacle: Array,
) -> Array:
    """Resolve collision between one agent disc and one active rotated wall."""
    wall_center = jnp.stack(
        (
            obstacle[OBSTACLE_FEATURE_X],
            obstacle[OBSTACLE_FEATURE_Y],
        )
    )
    wall_width = obstacle[OBSTACLE_FEATURE_WIDTH]
    wall_height = obstacle[OBSTACLE_FEATURE_HEIGHT]
    wall_theta = obstacle[OBSTACLE_FEATURE_THETA]

    world_to_wall = _create_2d_rotation_matrix(-wall_theta)
    agent_center_wall_local = world_to_wall @ (center - wall_center)

    wall_x_bounds = jnp.array(
        [-wall_width / 2.0, wall_width / 2.0],
        dtype=jnp.float32,
    )
    wall_y_bounds = jnp.array(
        [-wall_height / 2.0, wall_height / 2.0],
        dtype=jnp.float32,
    )

    nearest_x_to_agent_center = jnp.clip(
        agent_center_wall_local[0],
        min=wall_x_bounds[0],
        max=wall_x_bounds[1],
    )
    nearest_y_to_agent_center = jnp.clip(
        agent_center_wall_local[1],
        min=wall_y_bounds[0],
        max=wall_y_bounds[1],
    )
    nearest_point_to_agent_center = jnp.stack(
        (
            nearest_x_to_agent_center,
            nearest_y_to_agent_center,
        )
    )

    center_is_inside_or_near_wall = jnp.allclose(
        nearest_point_to_agent_center,
        agent_center_wall_local,
        atol=GEOMETRY_TOLERANCE,
        rtol=0.0,
    )

    center_is_outside_wall = jnp.logical_not(center_is_inside_or_near_wall)

    new_center_wall_local = cast(
        Array,
        jax.lax.cond(
            center_is_outside_wall,
            _project_outside_disc_out_of_active_wall,
            _project_inside_disc_out_of_active_wall,
            agent_center_wall_local,
            nearest_point_to_agent_center,
            wall_x_bounds,
            wall_y_bounds,
            radius,
        ),
    )

    wall_to_world = world_to_wall.T

    return (wall_to_world @ new_center_wall_local) + wall_center


def _keep_obstacle_projection_center(
    center: Array,
    radius: Array | float,
    obstacle: Array,
) -> Array:
    """Keep an agent center unchanged for inactive or non-matching obstacles."""
    del radius, obstacle

    return center


# Public geometry kernels ---


def project_disc_to_bounds(
    center: Array,
    radius: Array | float,
    map_width: Array | float,
    map_height: Array | float,
) -> Array:
    """Keep an agent disc fully inside the rectangular map.

    Args:
        center: Agent center with shape ``(2,)`` in world coordinates.
        radius: Agent body radius.
        map_width: Width of the rectangular battleground.
        map_height: Height of the rectangular battleground.

    Returns:
        Agent center clipped to the valid region where the full body remains
        inside the map.
    """
    center_x = jnp.clip(center[0], min=radius, max=map_width - radius)
    center_y = jnp.clip(center[1], min=radius, max=map_height - radius)

    return jnp.stack((center_x, center_y))


def project_disc_out_of_pillar(
    center: Array,
    radius: Array | float,
    obstacle: Array,
) -> Array:
    """Resolve one agent's collision against one circular pillar obstacle.

    Inactive rows and non-pillar rows are padding or different geometry types, so
    they leave the agent center unchanged. Active pillars are solid circular map
    blockers. When the agent body overlaps the pillar, the center is moved to the
    nearest tangent position along the pillar-to-agent direction. If both centers
    coincide exactly, a deterministic fallback direction avoids NaNs.

    Args:
        center: Agent center with shape ``(2,)`` in world coordinates.
        radius: Agent body radius.
        obstacle: One obstacle row using the shared obstacle feature layout.

    Returns:
        Agent center with shape ``(2,)`` after pillar collision projection.
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


def project_disc_out_of_wall(
    center: Array,
    radius: Array | float,
    obstacle: Array,
) -> Array:
    """Resolve one agent's collision against one rotated wall obstacle.

    Inactive rows and non-wall rows leave the agent center unchanged. Active
    walls are solid rotated rectangles. The helper transforms the agent center
    into the wall's local frame, resolves overlap against the axis-aligned local
    rectangle, and transforms the projected center back into world coordinates.

    Args:
        center: Agent center with shape ``(2,)`` in world coordinates.
        radius: Agent body radius.
        obstacle: One obstacle row using the shared obstacle feature layout.

    Returns:
        Agent center with shape ``(2,)`` after wall collision projection.
    """
    is_active_wall = jnp.logical_and(
        obstacle[OBSTACLE_FEATURE_TYPE] == OBSTACLE_TYPE_WALL,
        obstacle[OBSTACLE_FEATURE_ACTIVE] == 1.0,
    )

    return cast(
        Array,
        jax.lax.cond(
            is_active_wall,
            _project_disc_out_of_active_wall,
            _keep_obstacle_projection_center,
            center,
            radius,
            obstacle,
        ),
    )


def project_disc_out_of_obstacle() -> Array:
    """Resolve one agent against one obstacle row.

    This will dispatch between pillar projection, wall projection, and no-op
    padding behavior using JAX-compatible control flow.
    """
    raise NotImplementedError


def project_disc_out_of_obstacles() -> Array:
    """Resolve one agent against every static obstacle slot in the map config.

    This will apply obstacle projection through a fixed iteration pattern so the
    helper remains usable inside JIT-compiled transition code.
    """
    raise NotImplementedError


def resolve_agent_agent_overlaps() -> Array:
    """Resolve body blocking between active alive agents.

    Allied and enemy agents both occupy physical space. This helper will enforce
    hard non-overlap for active alive slots while leaving padded, inactive, and
    dead slots fixed.
    """
    raise NotImplementedError


def project_movement_with_geometry() -> Array:
    """Apply intended movement while preserving map and body-blocking constraints.

    This will split movement into fixed substeps, project active alive agents
    against bounds and obstacles, and run fixed-pass agent-agent overlap
    resolution. Inactive or dead slots must preserve their original positions.
    """
    raise NotImplementedError


def segment_intersects_circle() -> Array:
    """Return whether a line-of-sight segment intersects a circular pillar.

    Tangent and near-tangent contact should count as blocked LOS for the
    simulator's visibility and targetability rules.
    """
    raise NotImplementedError


def segment_intersects_rotated_rect() -> Array:
    """Return whether a line-of-sight segment intersects a rotated wall.

    The segment should be evaluated in the wall's local frame so rotated and
    axis-aligned walls share the same blocking semantics.
    """
    raise NotImplementedError


def has_clear_line_of_sight() -> Array:
    """Return whether static map geometry leaves a clear line of sight.

    Active pillars and active walls block LOS. Agents are deliberately not inputs
    because MARL-BattleGrounds v1 agents block movement but not line of sight.
    Observation construction and targetability should consume this shared helper
    so visibility, masks, and future debug tooling cannot drift apart.
    """
    raise NotImplementedError
