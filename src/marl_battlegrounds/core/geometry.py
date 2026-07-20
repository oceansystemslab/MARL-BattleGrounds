"""Shared JAX-compatible geometry helpers for simulator movement and LOS."""

from typing import cast

import jax
import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.types import (
    MAX_AGENT_SLOTS,
    MAX_OBSTACLE_SLOTS,
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
# Milestone 4 defaults are implementation-local, not public schema. With the
# Step 1 placeholder movement speed of 1.0 and default agent radius of 0.5, four
# movement substeps advance 0.25 world units at a time, which is small enough for
# the early deterministic maps without doubling projection cost. Four projection
# passes meaningfully reduce ordinary body-blocking residuals while keeping JAX
# control flow fixed and cheap. Revisit these after Step 3 movement integration
# profiling, especially if class mechanics introduce faster movement.
DEFAULT_MOVEMENT_SUBSTEPS = 4
DEFAULT_AGENT_PROJECTION_PASSES = 4

__all__ = (
    "has_clear_line_of_sight",
    "project_movement_with_geometry",
)

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

    # Inside-wall projection is ambiguous at corners and at the rectangle center;
    # choose the nearest local face so the result is deterministic.
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
    """Resolve outside-wall overlap in wall-local coordinates."""
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

    # Work in the wall's local frame so rotated walls reduce to axis-aligned
    # rectangle projection.
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

    # If clamping did not move the point, the center is inside or on the wall.
    center_is_inside_or_on_wall = jnp.array_equal(
        nearest_point_to_agent_center, agent_center_wall_local
    )

    center_is_outside_wall = jnp.logical_not(center_is_inside_or_on_wall)

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


def _obstacle_blocks_line_of_sight(
    agent_center_a: Array,
    agent_center_b: Array,
    obstacle: Array,
) -> Array:
    """Return whether one padded obstacle row blocks a LOS segment."""
    is_active = jnp.equal(obstacle[OBSTACLE_FEATURE_ACTIVE], 1.0)

    return cast(
        Array,
        jax.lax.cond(
            is_active,
            _active_obstacle_blocks_line_of_sight,
            _inactive_or_none_obstacle,
            agent_center_a,
            agent_center_b,
            obstacle,
        ),
    )


def _active_obstacle_blocks_line_of_sight(
    agent_center_a: Array,
    agent_center_b: Array,
    obstacle: Array,
) -> Array:
    """Dispatch active obstacle LOS blocking by obstacle type."""
    idx = obstacle[OBSTACLE_FEATURE_TYPE].astype(jnp.int32)
    branches = [_inactive_or_none_obstacle, _pillar_dispatcher, _wall_dispatcher]

    return cast(
        Array,
        jax.lax.switch(
            idx,
            branches,
            agent_center_a,
            agent_center_b,
            obstacle,
        ),
    )


def _pillar_dispatcher(
    agent_center_a: Array,
    agent_center_b: Array,
    obstacle: Array,
) -> Array:
    """Evaluate LOS blocking for one active circular pillar row."""
    pillar_center = jnp.stack(
        (obstacle[OBSTACLE_FEATURE_X], obstacle[OBSTACLE_FEATURE_Y]),
        dtype=jnp.float32,
    )
    pillar_radius = obstacle[OBSTACLE_FEATURE_RADIUS]

    return _segment_intersects_circle(
        agent_center_a,
        agent_center_b,
        pillar_center,
        pillar_radius,
    )


def _wall_dispatcher(
    agent_center_a: Array,
    agent_center_b: Array,
    obstacle: Array,
) -> Array:
    """Evaluate LOS blocking for one active rotated wall row."""
    wall_center = jnp.stack(
        (obstacle[OBSTACLE_FEATURE_X], obstacle[OBSTACLE_FEATURE_Y]),
        dtype=jnp.float32,
    )
    wall_width = obstacle[OBSTACLE_FEATURE_WIDTH]
    wall_height = obstacle[OBSTACLE_FEATURE_HEIGHT]
    wall_theta = obstacle[OBSTACLE_FEATURE_THETA]

    return _segment_intersects_rotated_rect(
        agent_center_a,
        agent_center_b,
        wall_center,
        wall_width,
        wall_height,
        wall_theta,
    )


def _inactive_or_none_obstacle(
    agent_center_a: Array,
    agent_center_b: Array,
    obstacle: Array,
) -> Array:
    """Return no LOS blocking for inactive, none, or padding obstacle rows."""
    del agent_center_a, agent_center_b, obstacle

    return jnp.array(False)


def _return_original_positions(
    agent_positions: Array,
    distance_between_agents: Array | float,
    agent_a_index: int,
    agent_b_index: int,
    agent_radii: Array,
) -> Array:
    """Keep positions unchanged when an agent pair does not need projection."""
    del distance_between_agents, agent_a_index, agent_b_index, agent_radii
    return agent_positions


def _resolve_agent_agent_overlap(
    agent_positions: Array,
    distance_between_agents: Array | float,
    agent_a_index: int,
    agent_b_index: int,
    agent_radii: Array,
) -> Array:
    """Symmetrically separate one overlapping active-alive agent pair."""
    agent_center_a = agent_positions[agent_a_index]
    agent_center_b = agent_positions[agent_b_index]

    displacement_a_from_b = agent_center_a - agent_center_b
    centers_are_coincident = distance_between_agents <= GEOMETRY_EPSILON

    # Coincident centers have no geometric normal; use a fixed axis to keep the
    # projection finite and deterministic.
    safe_distance = jnp.where(
        centers_are_coincident,
        1.0,
        distance_between_agents,
    )

    fallback_direction_a = jnp.array(
        (1.0, 0.0),
        dtype=agent_positions.dtype,
    )

    direction_vector_a = jnp.where(
        centers_are_coincident,
        fallback_direction_a,
        displacement_a_from_b / safe_distance,
    )
    direction_vector_b = -direction_vector_a

    sum_of_radii = agent_radii[agent_a_index] + agent_radii[agent_b_index]

    degree_of_violation = jnp.where(
        centers_are_coincident, sum_of_radii, sum_of_radii - distance_between_agents
    )

    updated_agent_a_center = agent_center_a + direction_vector_a * (
        degree_of_violation / 2.0
    )
    updated_agent_b_center = agent_center_b + direction_vector_b * (
        degree_of_violation / 2.0
    )

    agent_positions = agent_positions.at[agent_a_index].set(updated_agent_a_center)
    agent_positions = agent_positions.at[agent_b_index].set(updated_agent_b_center)

    return agent_positions


# Private projection kernels ---


def _project_disc_to_bounds(
    center: Array,
    radius: Array | float,
    map_width: Array | float,
    map_height: Array | float,
) -> Array:
    """Keep an agent disc fully inside the rectangular map."""
    center_x = jnp.clip(center[0], min=radius, max=map_width - radius)
    center_y = jnp.clip(center[1], min=radius, max=map_height - radius)

    return jnp.stack((center_x, center_y))


def _project_disc_out_of_pillar(
    center: Array,
    radius: Array | float,
    obstacle: Array,
) -> Array:
    """Resolve one agent against one circular pillar obstacle row."""
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


def _project_disc_out_of_wall(
    center: Array,
    radius: Array | float,
    obstacle: Array,
) -> Array:
    """Resolve one agent against one rotated wall obstacle row."""
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


def _project_disc_out_of_obstacle(
    center: Array,
    radius: Array | float,
    obstacle: Array,
) -> Array:
    """Resolve one agent against one padded obstacle row."""
    # Obstacle type values are part of the fixed obstacle-row schema:
    # 0 = none, 1 = pillar, 2 = wall.
    idx = obstacle[OBSTACLE_FEATURE_TYPE].astype(jnp.int32)

    branches = [
        _keep_obstacle_projection_center,
        _project_disc_out_of_pillar,
        _project_disc_out_of_wall,
    ]
    return cast(Array, jax.lax.switch(idx, branches, center, radius, obstacle))


def _project_disc_out_of_obstacles(
    center: Array,
    radius: Array | float,
    obstacles: Array,  # (MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES)
) -> Array:
    """Resolve one agent against every static obstacle slot."""

    def _project_disc_out_of_obstacle_wrapper(
        i: int,
        current_center: Array,
    ) -> Array:
        """Project the carried center against obstacle slot i."""
        return _project_disc_out_of_obstacle(current_center, radius, obstacles[i])

    # Projection is sequential: each obstacle sees the center produced by the
    # preceding obstacle slot.
    return cast(
        Array,
        jax.lax.fori_loop(
            0, MAX_OBSTACLE_SLOTS, _project_disc_out_of_obstacle_wrapper, center
        ),
    )


def _resolve_agent_agent_overlaps(
    agent_positions: Array,
    agent_radii: Array,
    active_mask: Array,
    alive_mask: Array,
    projection_passes: int = DEFAULT_AGENT_PROJECTION_PASSES,
) -> Array:
    """Reduce active-alive body overlap with fixed pairwise passes."""
    participants = jnp.logical_and(active_mask, alive_mask)

    def _projection_pass(
        pass_index: int,
        current_agent_positions: Array,
    ) -> Array:
        """Run one fixed sweep over all agent pairs."""
        del pass_index

        def _resolve_pairs_for_agent(
            agent_a_index: int,
            current_positions: Array,
        ) -> Array:
            """Resolve pairs anchored at one agent slot."""

            def _resolve_pair(
                agent_b_index: int,
                pair_positions: Array,
            ) -> Array:
                """Resolve one ordered agent pair when both slots participate."""
                # The triangular sweep visits each pair once without constructing
                # a dynamic pair list, which keeps JAX shapes static.
                pair_participants = jnp.logical_and(
                    participants[agent_a_index],
                    participants[agent_b_index],
                )

                distance_between_agents = cast(
                    Array,
                    jnp.linalg.norm(
                        pair_positions[agent_a_index] - pair_positions[agent_b_index]
                    ),
                )

                sum_of_radii = agent_radii[agent_a_index] + agent_radii[agent_b_index]
                pair_overlaps = sum_of_radii > distance_between_agents
                should_resolve_pair = jnp.logical_and(
                    pair_participants,
                    pair_overlaps,
                )

                return cast(
                    Array,
                    jax.lax.cond(
                        should_resolve_pair,
                        _resolve_agent_agent_overlap,
                        _return_original_positions,
                        pair_positions,
                        distance_between_agents,
                        agent_a_index,
                        agent_b_index,
                        agent_radii,
                    ),
                )

            return cast(
                Array,
                jax.lax.fori_loop(
                    lower=agent_a_index + 1,
                    upper=MAX_AGENT_SLOTS,
                    body_fun=_resolve_pair,
                    init_val=current_positions,
                ),
            )

        return cast(
            Array,
            jax.lax.fori_loop(
                lower=0,
                upper=MAX_AGENT_SLOTS,
                body_fun=_resolve_pairs_for_agent,
                init_val=current_agent_positions,
            ),
        )

    return cast(
        Array,
        jax.lax.fori_loop(
            lower=0,
            upper=projection_passes,
            body_fun=_projection_pass,
            init_val=agent_positions,
        ),
    )


# Private LOS kernels ---


def _segment_intersects_circle(
    segment_start: Array,
    segment_end: Array,
    circle_center: Array,
    circle_radius: Array | float,
) -> Array:
    """Return whether a finite segment intersects a circle."""
    v = segment_end - segment_start
    u = circle_center - segment_start

    # Project the circle center onto the finite segment; the denominator is
    # guarded so zero-length segments become point-vs-circle checks.
    alpha = jnp.dot(u, v) / jnp.maximum(jnp.dot(v, v), GEOMETRY_EPSILON)
    alpha_clipped = jnp.clip(alpha, 0.0, 1.0)

    closest_point = segment_start + alpha_clipped * v

    diff = closest_point - circle_center
    distance_sq = jnp.dot(diff, diff)

    return distance_sq <= (circle_radius + GEOMETRY_TOLERANCE) ** 2


def _segment_intersects_rotated_rect(
    segment_start: Array,
    segment_end: Array,
    rectangle_center: Array,
    width: Array | float,
    height: Array | float,
    theta: Array | float,
) -> Array:
    """Return whether a finite segment intersects a rotated rectangle."""
    # Rotate the query segment into rectangle-local space, then use a slab test
    # against the axis-aligned local bounds.
    world_to_local = _create_2d_rotation_matrix(-theta)
    segment_start_local = world_to_local @ (segment_start - rectangle_center)
    segment_end_local = world_to_local @ (segment_end - rectangle_center)

    max_y, min_y, max_x, min_x = (height / 2, -height / 2, width / 2, -width / 2)

    v_x = segment_end_local[0] - segment_start_local[0]
    v_y = segment_end_local[1] - segment_start_local[1]

    vertical_line_segment = jnp.abs(v_x) <= GEOMETRY_EPSILON
    horizontal_line_segment = jnp.abs(v_y) <= GEOMETRY_EPSILON

    x_inside_rectangle_bounds = jnp.logical_and(
        min_x <= segment_start_local[0],
        segment_start_local[0] <= max_x,
    )
    y_inside_rectangle_bounds = jnp.logical_and(
        min_y <= segment_start_local[1],
        segment_start_local[1] <= max_y,
    )

    safe_v_x = jnp.where(vertical_line_segment, 1.0, v_x)
    safe_v_y = jnp.where(horizontal_line_segment, 1.0, v_y)

    alpha_x_1 = (min_x - segment_start_local[0]) / safe_v_x
    alpha_x_2 = (max_x - segment_start_local[0]) / safe_v_x
    alpha_y_1 = (min_y - segment_start_local[1]) / safe_v_y
    alpha_y_2 = (max_y - segment_start_local[1]) / safe_v_y

    alpha_entry_x = jnp.minimum(alpha_x_1, alpha_x_2)
    alpha_exit_x = jnp.maximum(alpha_x_1, alpha_x_2)
    alpha_entry_y = jnp.minimum(alpha_y_1, alpha_y_2)
    alpha_exit_y = jnp.maximum(alpha_y_1, alpha_y_2)

    # Parallel segments either span the entire slab when already inside that axis
    # or miss it entirely; infinities encode those two cases without branching.
    alpha_entry_x = jnp.where(
        vertical_line_segment,
        jnp.where(x_inside_rectangle_bounds, -jnp.inf, jnp.inf),
        alpha_entry_x,
    )
    alpha_exit_x = jnp.where(
        vertical_line_segment,
        jnp.where(x_inside_rectangle_bounds, jnp.inf, -jnp.inf),
        alpha_exit_x,
    )
    alpha_entry_y = jnp.where(
        horizontal_line_segment,
        jnp.where(y_inside_rectangle_bounds, -jnp.inf, jnp.inf),
        alpha_entry_y,
    )
    alpha_exit_y = jnp.where(
        horizontal_line_segment,
        jnp.where(y_inside_rectangle_bounds, jnp.inf, -jnp.inf),
        alpha_exit_y,
    )

    entry_points = jnp.array((alpha_entry_x, alpha_entry_y))
    exit_points = jnp.array((alpha_exit_x, alpha_exit_y))

    return jnp.maximum(0.0, jnp.max(entry_points)) <= jnp.minimum(
        1.0,
        jnp.min(exit_points),
    )


# Public geometry API ---


def project_movement_with_geometry(
    agent_positions: Array,
    agent_radii: Array,
    intended_movement_deltas: Array,
    active_mask: Array,
    alive_mask: Array,
    map_width: Array | float,
    map_height: Array | float,
    obstacles: Array,
    agent_agent_overlap_projection_passes: int = 1,
    collision_projection_passes: int = DEFAULT_AGENT_PROJECTION_PASSES,
    movement_substeps: int = DEFAULT_MOVEMENT_SUBSTEPS,
) -> Array:
    """Project intended per-slot movement through shared geometry constraints.

    This helper is the public array-level geometry primitive that ``env.step``
    will call in the next milestone step. It accepts already-computed movement
    deltas; action-id semantics belong outside this module. Final committed
    positions prioritize static world validity. Agent-agent body blocking uses a
    deterministic fixed-pass projection, which fully separates ordinary feasible
    cases and may leave bounded residual overlap in pinned or crowded cases.

    Args:
        agent_positions: Current slot-aligned centers with shape
            ``(MAX_AGENT_SLOTS, 2)``.
        agent_radii: Slot-aligned body radii with shape ``(MAX_AGENT_SLOTS,)``.
        intended_movement_deltas: Slot-aligned movement deltas with shape
            ``(MAX_AGENT_SLOTS, 2)``.
        active_mask: Boolean mask for real, non-padding agent slots.
        alive_mask: Boolean mask for currently alive agent slots.
        map_width: Width of the rectangular battleground.
        map_height: Height of the rectangular battleground.
        obstacles: Fixed obstacle table with shape
            ``(MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES)``.
        agent_agent_overlap_projection_passes: Fixed pairwise overlap sweeps
            inside each collision projection pass.
        collision_projection_passes: Fixed passes that compose boundary,
            obstacle, and agent-agent projection during each movement substep.
        movement_substeps: Fixed number of movement increments used to apply
            the provided deltas.

    Returns:
        Slot-aligned projected positions. Inactive and dead slots preserve their
        original positions.

    """
    original_agent_positions = agent_positions

    participant_mask = jnp.logical_and(active_mask, alive_mask)[:, None]

    substep_deltas = intended_movement_deltas / movement_substeps
    masked_substep_deltas = jnp.where(
        participant_mask,
        substep_deltas,
        jnp.zeros_like(substep_deltas),
    )

    project_disc_to_bounds_vmap = jax.vmap(
        _project_disc_to_bounds,
        in_axes=(0, 0, None, None),
        out_axes=0,
    )

    project_disc_out_of_obstacles_vmap = jax.vmap(
        _project_disc_out_of_obstacles,
        in_axes=(0, 0, None),
        out_axes=0,
    )

    def _project_to_bounds(candidate_positions: Array) -> Array:
        """Project all slots into map bounds."""
        return project_disc_to_bounds_vmap(
            candidate_positions,
            agent_radii,
            map_width,
            map_height,
        )

    def _project_out_of_obstacles(candidate_positions: Array) -> Array:
        """Project all slots out of static obstacles."""
        return project_disc_out_of_obstacles_vmap(
            candidate_positions,
            agent_radii,
            obstacles,
        )

    def _project_collision_pass(
        pass_index: int,
        current_positions: Array,
    ) -> Array:
        """Run one fixed collision-composition pass."""
        del pass_index

        current_positions = _project_to_bounds(current_positions)
        current_positions = _project_out_of_obstacles(current_positions)

        return _resolve_agent_agent_overlaps(
            current_positions,
            agent_radii,
            active_mask,
            alive_mask,
            agent_agent_overlap_projection_passes,
        )

    def _project_movement_substep(
        substep_index: int,
        current_positions: Array,
    ) -> Array:
        """Apply and project one fixed movement substep."""
        del substep_index

        current_positions = current_positions + masked_substep_deltas

        current_positions = cast(
            Array,
            jax.lax.fori_loop(
                0,
                collision_projection_passes,
                _project_collision_pass,
                current_positions,
            ),
        )

        # Agent-agent projection can push a slot back into static geometry. The
        # final cleanup makes map bounds and obstacles the hard committed state.
        current_positions = _project_out_of_obstacles(current_positions)
        current_positions = _project_to_bounds(current_positions)

        return jnp.where(
            participant_mask,
            current_positions,
            original_agent_positions,
        )

    return cast(
        Array,
        jax.lax.fori_loop(
            0,
            movement_substeps,
            _project_movement_substep,
            original_agent_positions,
        ),
    )


def has_clear_line_of_sight(
    agent_center_a: Array,
    agent_center_b: Array,
    obstacles: Array,
) -> Array:
    """Return whether static map geometry leaves a clear line of sight.

    Active pillars and active walls block LOS. Agents are deliberately not inputs
    because MARL-BattleGrounds v1 agents block movement but not line of sight.
    Observation construction and targetability should consume this shared helper
    so visibility, masks, and future debug tooling cannot drift apart.

    Args:
        agent_center_a: LOS segment start point with shape ``(2,)``.
        agent_center_b: LOS segment end point with shape ``(2,)``.
        obstacles: Fixed obstacle table with shape
            ``(MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES)``.

    Returns:
        Scalar boolean JAX array. ``True`` means no active static obstacle blocks
        the segment; ``False`` means at least one active pillar or wall blocks it.

    """
    obstacle_blocks_line_of_sight_vmap = jax.vmap(
        _obstacle_blocks_line_of_sight,
        in_axes=(None, None, 0),
        out_axes=0,
    )

    blocked_by_obstacle = obstacle_blocks_line_of_sight_vmap(
        agent_center_a,
        agent_center_b,
        obstacles,
    )

    return jnp.logical_not(jnp.any(blocked_by_obstacle))
