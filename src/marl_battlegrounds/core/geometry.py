"""Own physical projection, body-obstacle overlap and static line of sight.

Ordinary movement guards the full travelled segment. Charge may cross geometry
and instead recovers its arrival before endpoint contact correction. Both use
the same fixed-slot geometry solver. Line of sight checks static pillars and
walls; agents block movement but not sight.

Public helpers consume already validated world coordinates, radii, masks and
obstacle rows. Numerical queries use JAX and draw no randomness. Projection
loop counts are static controls; data values stay dynamic. Host configuration
validation, action meanings and episode timing remain outside this module."""

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from marl_battlegrounds.core.types import (
    MAX_AGENT_SLOTS,
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
_STATIC_TRAVEL_MARGIN = 2e-6
DEFAULT_MOVEMENT_SUBSTEPS = 4
# This is a literal count. Static policy probes explicitly use four rounds.
DEFAULT_AGENT_PROJECTION_PASSES = 28
# DEFAULT_AGENT_PROJECTION_PASSES = 64
# Enable 64 if training reveals excessive body overlap.
# Cost: 28 was about 2.2-2.3x as fast in isolated RTX 5090 movement/Charge
# checks (batches 1, 64 and 1024); this is not a full-training speedup.

__all__ = (
    "disc_overlaps_obstacle",
    "has_clear_line_of_sight",
    "project_charge_endpoints_with_geometry",
    "project_movement_with_geometry",
)

# These fixed tables give each body its nine pair contributions.
_FIRST, _SECOND = np.triu_indices(MAX_AGENT_SLOTS, 1)
_PAIR_INDEX = np.zeros((MAX_AGENT_SLOTS, MAX_AGENT_SLOTS - 1), dtype=np.int32)
_PAIR_SIGN = np.zeros((MAX_AGENT_SLOTS, MAX_AGENT_SLOTS - 1), dtype=np.float32)
for _actor in range(MAX_AGENT_SLOTS):
    _ids = np.flatnonzero((_actor == _FIRST) | (_actor == _SECOND))
    _PAIR_INDEX[_actor] = _ids
    _PAIR_SIGN[_actor] = np.where(_FIRST[_ids] == _actor, 1.0, -1.0)


def _create_2d_rotation_matrix(theta: Array | float) -> Array:
    """Return the float32 (2, 2) rotation for a scalar angle in radians.

    Positive angles rotate local vectors into the world frame. Passing the
    negative wall angle performs the inverse transform used by shape tests.
    """
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)

    return jnp.array(
        [
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta],
        ],
        dtype=jnp.float32,
    )


def _disc_overlaps_active_pillar(
    center: Array,
    radius: Array | float,
    obstacle: Array,
) -> Array:
    """Test strict overlap against one known active pillar.

    The float32 center (2,), scalar radius and obstacle row (8,) are already
    validated. Return a bool scalar; exact tangency is not an overlap.
    """
    pillar_center = jnp.stack(
        (
            obstacle[OBSTACLE_FEATURE_X],
            obstacle[OBSTACLE_FEATURE_Y],
        )
    )
    center_delta = center - pillar_center
    center_distance = jnp.hypot(center_delta[0], center_delta[1])
    minimum_distance = radius + obstacle[OBSTACLE_FEATURE_RADIUS]
    return center_distance < minimum_distance


def _disc_overlaps_active_wall(
    center: Array,
    radius: Array | float,
    obstacle: Array,
) -> Array:
    """Test overlap against one known active rotated wall.

    Transform the center (2,) into the wall frame and measure distance to the
    nearest rectangle point. A center inside or on the rectangle overlaps.
    Outside centers use GEOMETRY_EPSILON so float32 rotation noise does not
    turn legal tangency into penetration. Return a bool scalar.
    """
    wall_center = jnp.stack(
        (
            obstacle[OBSTACLE_FEATURE_X],
            obstacle[OBSTACLE_FEATURE_Y],
        )
    )
    world_to_wall = _create_2d_rotation_matrix(-obstacle[OBSTACLE_FEATURE_THETA])
    center_wall_local = world_to_wall @ (center - wall_center)
    wall_half_width = obstacle[OBSTACLE_FEATURE_WIDTH] / 2.0
    wall_half_height = obstacle[OBSTACLE_FEATURE_HEIGHT] / 2.0
    nearest_point = jnp.stack(
        (
            jnp.clip(center_wall_local[0], -wall_half_width, wall_half_width),
            jnp.clip(center_wall_local[1], -wall_half_height, wall_half_height),
        )
    )
    center_is_inside_or_on_wall = jnp.array_equal(nearest_point, center_wall_local)
    wall_delta = center_wall_local - nearest_point
    distance_to_wall = jnp.hypot(wall_delta[0], wall_delta[1])
    # A world->local rotation round-trip can move an exact tangent by a few
    # float32 ULPs. Treat only separation violations beyond the geometry
    # epsilon as wall overlap so legal tangency is representation-stable.
    outside_overlap = distance_to_wall < radius - GEOMETRY_EPSILON
    return jnp.logical_or(center_is_inside_or_on_wall, outside_overlap)


def _disc_does_not_overlap_obstacle(
    center: Array,
    radius: Array | float,
    obstacle: Array,
) -> Array:
    """Return a false bool scalar for the inactive or empty obstacle branch.

    The three unused operands keep this branch compatible with lax.cond and
    lax.switch. No body or obstacle values are inspected.
    """
    del center, radius, obstacle
    return jnp.array(False)


def _active_disc_obstacle_overlap(
    center: Array,
    radius: Array | float,
    obstacle: Array,
) -> Array:
    """Select the overlap test for one validated active obstacle row.

    Type zero is empty, one is a pillar and two is a wall. The type is dynamic
    JAX data; host validation must reject unsupported categories beforehand.
    """
    obstacle_type = obstacle[OBSTACLE_FEATURE_TYPE].astype(jnp.int32)
    branches = (
        _disc_does_not_overlap_obstacle,
        _disc_overlaps_active_pillar,
        _disc_overlaps_active_wall,
    )
    return cast(
        Array,
        jax.lax.switch(obstacle_type, branches, center, radius, obstacle),
    )


def disc_overlaps_obstacle(
    center: Array,
    radius: Array | float,
    obstacle: Array,
) -> Array:
    """Test whether one disc overlaps one active obstacle.

    Parameters
    ----------
    center : jax.Array
        Float32 (2,) world coordinates [x, y].
    radius : float or jax.Array
        Nonnegative scalar body radius in world units.
    obstacle : jax.Array
        One validated float32 row (8,) using EnvConfig.obstacles columns.

    Returns
    -------
    jax.Array
        Bool scalar. False for an inactive row or legal tangency. Pillars use
        strict distance overlap; wall tests allow GEOMETRY_EPSILON for small
        rotation errors and treat a center inside the wall as overlapping.

    Notes
    -----
    This pure JAX predicate checks one row. Use vmap for multiple bodies or
    obstacles. It does not validate categories or test map bounds. Core config
    validation owns those checks before compiled execution.
    """
    return cast(
        Array,
        jax.lax.cond(
            obstacle[OBSTACLE_FEATURE_ACTIVE] == 1.0,
            _active_disc_obstacle_overlap,
            _disc_does_not_overlap_obstacle,
            center,
            radius,
            obstacle,
        ),
    )


def _obstacle_blocks_line_of_sight(
    agent_center_a: Array,
    agent_center_b: Array,
    obstacle: Array,
) -> Array:
    """Test one obstacle row only when its active flag is exactly one.

    Endpoints are float32 (2,) and the validated row is float32 (8,).
    Return a bool scalar; inactive padding cannot block the segment.
    """
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
    """Select the segment test for one validated active obstacle type.

    The dynamic type selects empty, pillar or wall behavior with lax.switch.
    Type validation belongs to the host configuration boundary.
    """
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
    """Read one pillar row and test its circle against the finite sight segment.

    Extract the world center and radius from the validated obstacle schema.
    Both endpoints are float32 (2,); the result is a bool scalar.
    """
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
    """Read one wall row and test its rotated rectangle against the sight segment.

    Width and height use world units; theta uses radians. Both endpoints are
    float32 (2,); the result is a bool scalar.
    """
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
    """Return a false bool scalar for an empty line-of-sight branch.

    Matching unused operands let lax.cond and lax.switch share the same call
    shape as active obstacle tests without special handling for padded rows.
    """
    del agent_center_a, agent_center_b, obstacle

    return jnp.array(False)


def _segment_intersects_circle(
    segment_start: Array,
    segment_end: Array,
    circle_center: Array,
    circle_radius: Array | float,
) -> Array:
    """Test a finite segment against a circle, including tangent contact.

    Endpoints and circle center are float32 (2,); radius is scalar. The nearest
    point is clamped to the segment. GEOMETRY_TOLERANCE expands the circle;
    the guarded projection also handles a zero-length segment. Return bool ().
    """
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
    """Test a finite segment against a rotated rectangle, including its boundary.

    Endpoints and center have shape (2,). Width and height are positive world
    lengths; theta is a scalar angle in radians. Work in the rectangle frame
    and intersect the allowed intervals on both axes. Parallel axes use guarded
    divisors and infinite intervals so the JAX path needs no Python branch.
    """
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


def has_clear_line_of_sight(
    agent_center_a: Array,
    agent_center_b: Array,
    obstacles: Array,
) -> Array:
    """Return whether static obstacles leave a clear center-to-center segment.

    Parameters
    ----------
    agent_center_a : jax.Array
        Float32 (2,) world coordinates of the segment start.
    agent_center_b : jax.Array
        Float32 (2,) world coordinates of the segment end.
    obstacles : jax.Array
        Validated float32 (32, 8) table in the EnvConfig.obstacles schema.

    Returns
    -------
    jax.Array
        Bool scalar: True if no active pillar or wall blocks the finite segment.
        Circle contact includes GEOMETRY_TOLERANCE; wall boundaries also block.
        Inactive rows never block.

    Notes
    -----
    Agents block movement but do not block sight, so bodies are not inputs.
    Observations and target masks share this pure JAX helper. It maps over
    obstacle rows internally; use an outer vmap for more segments or games.
    It does not enforce observation radius, participation or shield rules.
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


def _wall_frame(positions: Array, obstacles: Array) -> tuple[Array, Array, Array]:
    """Express each body center in each obstacle's rotated local frame.

    Float32 positions (10, 2) and obstacle rows (32, 8) produce local centers
    (10, 32, 2), then cosine and sine arrays (32,). The transform also runs for
    padded rows; downstream shape masks decide which rows contribute.
    """
    delta = (
        positions[:, None, :]
        - obstacles[None, :, OBSTACLE_FEATURE_X : OBSTACLE_FEATURE_Y + 1]
    )
    cosine = jnp.cos(obstacles[:, OBSTACLE_FEATURE_THETA])
    sine = jnp.sin(obstacles[:, OBSTACLE_FEATURE_THETA])
    local = jnp.stack(
        (
            delta[..., 0] * cosine + delta[..., 1] * sine,
            delta[..., 1] * cosine - delta[..., 0] * sine,
        ),
        axis=-1,
    )
    return (local, cosine, sine)


def _static_positions_are_valid(
    positions: Array,
    radii: Array,
    width: Array | float,
    height: Array | float,
    obstacles: Array,
    origin: float = 0.0,
) -> Array:
    """Check body endpoints against map bounds and actual obstacle shapes.

    Positions (10, 2) and radii (10,) produce a bool (10,) validity mask.
    Scalar width and height extend from origin, which defaults to 0.0.
    GEOMETRY_EPSILON allows rounding-sized contact error. This independent
    check does not trust the correction direction and does not test other bodies.
    """
    local, _, _ = _wall_frame(positions, obstacles)
    half = (
        obstacles[None, :, OBSTACLE_FEATURE_WIDTH : OBSTACLE_FEATURE_HEIGHT + 1] * 0.5
    )
    outside = local - jnp.clip(local, -half, half)
    wall_distance = jnp.hypot(outside[..., 0], outside[..., 1])
    wall_valid = wall_distance >= radii[:, None] - GEOMETRY_EPSILON
    centre_delta = (
        positions[:, None, :]
        - obstacles[None, :, OBSTACLE_FEATURE_X : OBSTACLE_FEATURE_Y + 1]
    )
    pillar_distance = jnp.hypot(centre_delta[..., 0], centre_delta[..., 1])
    pillar_valid = (
        pillar_distance
        >= radii[:, None]
        + obstacles[None, :, OBSTACLE_FEATURE_RADIUS]
        - GEOMETRY_EPSILON
    )
    shape_valid = jnp.where(
        obstacles[None, :, OBSTACLE_FEATURE_TYPE] == OBSTACLE_TYPE_WALL,
        wall_valid,
        pillar_valid,
    )
    known = (obstacles[None, :, OBSTACLE_FEATURE_TYPE] == OBSTACLE_TYPE_WALL) | (
        obstacles[None, :, OBSTACLE_FEATURE_TYPE] == OBSTACLE_TYPE_PILLAR
    )
    shape_valid |= (obstacles[None, :, OBSTACLE_FEATURE_ACTIVE] != 1) | ~known
    bounds_valid = jnp.all(
        positions >= origin + radii[:, None] - GEOMETRY_EPSILON, axis=-1
    ) & jnp.all(
        positions
        <= origin + jnp.asarray([width, height]) - radii[:, None] + GEOMETRY_EPSILON,
        axis=-1,
    )
    return bounds_valid & jnp.all(shape_valid, axis=1)


def _box_entry(start: Array, delta: Array, half: Array) -> Array:
    """Find each segment's first entry into an expanded axis-aligned rectangle.

    Start, delta and half extents broadcast with a final coordinate axis of
    length two. Return fractions with that axis removed; one means no guarded
    crossing. Parallel motion and motion leaving a near face remain admissible.
    """
    moving = delta != 0
    divisor = jnp.where(moving, delta, 1)
    first = (-half - start) / divisor
    second = (half - start) / divisor
    near = jnp.where(moving, jnp.minimum(first, second), -jnp.inf)
    far = jnp.where(moving, jnp.maximum(first, second), jnp.inf)
    entry = jnp.maximum(near[..., 0], near[..., 1])
    leave = jnp.minimum(far[..., 0], far[..., 1])
    parallel_clear = jnp.any(~moving & (jnp.abs(start) >= half), axis=-1)
    crossed = ~parallel_clear & (leave > jnp.maximum(entry, 0)) & (entry < 1)
    near_face = half - GEOMETRY_EPSILON
    first_inside = start + jnp.clip(entry, 0, 1)[..., None] * delta
    last_inside = start + jnp.clip(leave, 0, 1)[..., None] * delta
    leaving_near_face = jnp.any(
        (first_inside >= near_face) & (last_inside >= near_face)
        | (-first_inside >= near_face) & (-last_inside >= near_face),
        axis=-1,
    )
    crossed &= ~leaving_near_face
    return jnp.where(crossed, jnp.maximum(entry, 0), 1)


def _circle_entry(start: Array, delta: Array, radius: Array) -> Array:
    """Find each segment's first circle entry beyond the allowed contact error.

    Start and delta end in coordinate axis two; radius broadcasts over the
    remaining axes. Return fractions in [0, 1], using one for no guarded entry.
    Zero movement, tangent contact and travel away from the circle do not count
    as a crossing. GEOMETRY_EPSILON defines the tolerated inner boundary.
    """
    a = delta[..., 0] ** 2 + delta[..., 1] ** 2
    b = start[..., 0] * delta[..., 0] + start[..., 1] * delta[..., 1]
    divisor = jnp.where(a > 0, a, 1)
    cross = start[..., 0] * delta[..., 1] - start[..., 1] * delta[..., 0]
    chord_squared = (radius**2 * a - cross * cross) / divisor
    root = jnp.sqrt(jnp.maximum(chord_squared, 0) / divisor)
    middle = -b / divisor
    entry = middle - root
    leave = middle + root
    crossed = (a > 0) & (chord_squared > 0) & (b < 0)
    crossed &= (leave > jnp.maximum(entry, 0)) & (entry < 1)
    end = start + delta
    nearest_squared = jnp.where(
        middle <= 0,
        start[..., 0] ** 2 + start[..., 1] ** 2,
        jnp.where(
            middle >= 1, end[..., 0] ** 2 + end[..., 1] ** 2, cross * cross / divisor
        ),
    )
    inner_radius = jnp.maximum(radius - GEOMETRY_EPSILON, 0)
    crossed &= nearest_squared < inner_radius**2
    return jnp.where(crossed, jnp.maximum(entry, 0), 1)


def _obstacle_entry_fractions(
    start: Array, intended: Array, radii: Array, obstacles: Array
) -> Array:
    """Find first full-disc entry for every body and obstacle pair.

    Start and intended endpoints are (10, 2), radii are (10,), and obstacles
    are (32, 8). Return fractions (10, 32). Walls combine expanded face strips
    and rounded corners; pillars expand by body radius. Outward travel from a
    small existing overlap may escape. Callers mask inactive obstacle rows.
    """
    local, cosine, sine = _wall_frame(start, obstacles)
    world_delta = intended - start
    delta = jnp.stack(
        (
            world_delta[:, None, 0] * cosine + world_delta[:, None, 1] * sine,
            world_delta[:, None, 1] * cosine - world_delta[:, None, 0] * sine,
        ),
        axis=-1,
    )
    half = (
        obstacles[None, :, OBSTACLE_FEATURE_WIDTH : OBSTACLE_FEATURE_HEIGHT + 1] * 0.5
    )
    horizontal = half + radii[:, None, None] * jnp.asarray([1.0, 0.0])
    vertical = half + radii[:, None, None] * jnp.asarray([0.0, 1.0])
    box_fraction = jnp.minimum(
        _box_entry(local, delta, horizontal), _box_entry(local, delta, vertical)
    )
    signs = jnp.asarray([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    corners = half[..., None, :] * signs
    corner_fraction = _circle_entry(
        local[..., None, :] - corners, delta[..., None, :], radii[:, None, None]
    )
    wall_fraction = jnp.minimum(box_fraction, jnp.min(corner_fraction, axis=-1))
    relative = (
        start[:, None, :]
        - obstacles[None, :, OBSTACLE_FEATURE_X : OBSTACLE_FEATURE_Y + 1]
    )
    pillar_fraction = _circle_entry(
        relative,
        world_delta[:, None, :],
        radii[:, None] + obstacles[None, :, OBSTACLE_FEATURE_RADIUS],
    )
    wall_offset = local - jnp.clip(local, -half, half)
    wall_distance = jnp.hypot(wall_offset[..., 0], wall_offset[..., 1])
    wall_dot = wall_offset[..., 0] * delta[..., 0] + wall_offset[..., 1] * delta[..., 1]
    wall_escape = (
        (wall_distance > 0) & (wall_distance < radii[:, None]) & (wall_dot >= 0)
    )
    pillar_distance = jnp.hypot(relative[..., 0], relative[..., 1])
    pillar_dot = (
        relative[..., 0] * world_delta[:, None, 0]
        + relative[..., 1] * world_delta[:, None, 1]
    )
    pillar_escape = (
        (pillar_distance > obstacles[None, :, OBSTACLE_FEATURE_RADIUS])
        & (
            pillar_distance
            < obstacles[None, :, OBSTACLE_FEATURE_RADIUS] + radii[:, None]
        )
        & (pillar_dot >= 0)
    )
    wall = obstacles[None, :, OBSTACLE_FEATURE_TYPE] == OBSTACLE_TYPE_WALL
    active = (obstacles[None, :, OBSTACLE_FEATURE_ACTIVE] == 1) & (
        wall | (obstacles[None, :, OBSTACLE_FEATURE_TYPE] == OBSTACLE_TYPE_PILLAR)
    )
    escape = active & jnp.where(wall, wall_escape, pillar_escape)
    return jnp.where(escape, 1, jnp.where(wall, wall_fraction, pillar_fraction))


def _swept_fraction(
    start: Array,
    proposed: Array,
    radii: Array,
    width: Array | float,
    height: Array | float,
    obstacles: Array,
    origin: float = 0.0,
) -> Array:
    """Find the earliest guarded obstacle or map-bound contact for each body.

    Start and proposed endpoints (10, 2) produce fractions (10,) in [0, 1].
    One permits the whole segment. Radii expand obstacles and inset the map
    bounds; unknown or inactive obstacle rows do not constrain this query.
    Width and height extend from the scalar origin, default 0.0.
    """
    world_delta = proposed - start
    fraction = _obstacle_entry_fractions(start, proposed, radii, obstacles)
    active = (obstacles[None, :, OBSTACLE_FEATURE_ACTIVE] == 1) & (
        (obstacles[None, :, OBSTACLE_FEATURE_TYPE] == OBSTACLE_TYPE_WALL)
        | (obstacles[None, :, OBSTACLE_FEATURE_TYPE] == OBSTACLE_TYPE_PILLAR)
    )
    fraction = jnp.min(jnp.where(active, fraction, 1), axis=1)
    size = origin + jnp.asarray([width, height])
    divisor = jnp.where(world_delta != 0, world_delta, 1)
    bound_fraction = jnp.where(
        world_delta > 0,
        (size - radii[:, None] - start) / divisor,
        (origin + radii[:, None] - start) / divisor,
    )
    bound_fraction = jnp.where(world_delta != 0, bound_fraction, 1)
    return jnp.clip(jnp.minimum(fraction, jnp.min(bound_fraction, axis=-1)), 0, 1)


def _balanced_sum(values: Array, axis: int) -> Array:
    """Sum one fixed-size axis using an explicit pairwise reduction tree.

    Pad the chosen axis to a power of two with zeros, then add pairs in a fixed
    order. Axis and its nonempty size are static at tracing time. This avoids
    a backend choosing a different reduction order for the contact sums.
    """
    values = jnp.moveaxis(values, axis, -1)
    size = values.shape[-1]
    padded = 1 << (size - 1).bit_length()
    values = jnp.pad(values, [(0, 0)] * (values.ndim - 1) + [(0, padded - size)])
    while padded > 1:
        values = values[..., ::2] + values[..., 1::2]
        padded //= 2
    return values[..., 0]


def _sum_contact_corrections(values: Array, axis: int) -> Array:
    """Sum signed contact values in a repeatable order after rows are permuted.

    Sort along the static axis, reduce positive and negative magnitudes in
    separate fixed trees, then subtract. Exact zero is normalized to positive
    zero. The result removes the summed axis and retains the other dimensions.
    """
    ordered = jnp.sort(values, axis=axis, stable=False)
    positive = jnp.maximum(ordered, 0)
    negative = jnp.maximum(-jnp.flip(ordered, axis=axis), 0)
    result = _balanced_sum(positive, axis) - _balanced_sum(negative, axis)
    return jnp.where(result == 0, jnp.zeros_like(result), result)


def _unit(vector: Array) -> tuple[Array, Array]:
    """Return unit directions and lengths for vectors with final axis two.

    The direction keeps the input shape; lengths drop the coordinate axis.
    A zero vector stays zero because its divisor is replaced by one.
    """
    length = jnp.hypot(vector[..., 0], vector[..., 1])
    return (vector / jnp.where(length > 0, length, 1)[..., None], length)


def _body_corrections(
    positions: Array,
    radii: Array,
    participant: Array,
    start: Array,
    intended: Array,
    tie_direction: Array | None = None,
    recovering_pairs: Array | None = None,
) -> Array:
    """Share each participating pair's overlap correction between both bodies.

    Positions, start and intended are float32 (10, 2); radii and participation
    are (10,). The optional tie directions have shape (45, 2); optional recovery
    flags have shape (45,), matching the fixed unordered-pair table. Return
    corrections (10, 2). Prefer the incoming contact side unless a newly
    blocking overlap needs recovery; previous positions and intent break ties.
    """
    first, second = (jnp.asarray(_FIRST), jnp.asarray(_SECOND))
    difference = positions[first] - positions[second]
    previous = start[first] - start[second]
    radius = radii[first] + radii[second]
    _, previous_distance = _unit(previous)
    change = difference - previous
    _, change_distance = _unit(change)
    clearance = jnp.maximum(previous_distance - radius, 0)
    fraction = jnp.minimum(
        clearance / jnp.where(change_distance > 0, change_distance, 1), 1
    )
    direction = previous + fraction[:, None] * change
    direction = jnp.where((fraction == 1)[:, None], difference, direction)
    direction = jnp.where(
        jnp.any(direction != 0, axis=-1)[:, None], direction, previous
    )
    intent = intended[first] - intended[second]
    direction = jnp.where(jnp.any(direction != 0, axis=-1)[:, None], direction, intent)
    direction = jnp.where(
        jnp.any(direction != 0, axis=-1)[:, None], direction, difference
    )
    if tie_direction is not None:
        direction = jnp.where(
            jnp.any(direction != 0, axis=-1)[:, None], direction, tie_direction
        )
    if recovering_pairs is not None:
        # A newly blocking overlap has no valid incoming contact side to retain.
        recover = (
            recovering_pairs
            & (previous_distance < radius)
            & jnp.any(difference != 0, axis=-1)
        )
        direction = jnp.where(recover[:, None], difference, direction)
    normal, length = _unit(direction)
    separation = normal[:, 0] * difference[:, 0] + normal[:, 1] * difference[:, 1]
    depth = jnp.maximum(radius - separation, 0)
    active = participant[first] & participant[second] & (length > 0)
    response = jnp.where(active[:, None], normal * (depth * 0.5)[:, None], 0)
    contributions = (
        response[jnp.asarray(_PAIR_INDEX)] * jnp.asarray(_PAIR_SIGN)[..., None]
    )
    return _sum_contact_corrections(contributions, axis=1)


def _obstacle_corrections(
    positions: Array, radii: Array, obstacles: Array, start: Array
) -> Array:
    """Add wall and pillar separation corrections for each body endpoint.

    Positions and previous start positions are (10, 2), radii are (10,), and
    obstacles are (32, 8). Previous approach breaks equal wall-face or central
    pillar ties. Only known active obstacles contribute to the returned (10, 2)
    sum. Map bounds and path safety are checked by the surrounding projection.
    """
    local, cosine, sine = _wall_frame(positions, obstacles)
    previous, _, _ = _wall_frame(start, obstacles)
    half = (
        obstacles[None, :, OBSTACLE_FEATURE_WIDTH : OBSTACLE_FEATURE_HEIGHT + 1] * 0.5
    )
    outside, distance = _unit(local - jnp.clip(local, -half, half))
    faces = jnp.asarray(
        [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]], positions.dtype
    )
    face_distance = jnp.stack(
        (
            half[..., 0] + local[..., 0],
            half[..., 0] - local[..., 0],
            half[..., 1] + local[..., 1],
            half[..., 1] - local[..., 1],
        ),
        axis=-1,
    )
    nearest = face_distance == jnp.min(face_distance, axis=-1, keepdims=True)
    approach = previous - local
    preference = (
        approach[..., 0, None] * faces[:, 0] + approach[..., 1, None] * faces[:, 1]
    )
    preference = jnp.where(nearest, preference, -jnp.inf)
    chosen = nearest & (preference == jnp.max(preference, axis=-1, keepdims=True))
    # These values are only -1, 0 or 1. Every partial sum is exact.
    inside, _ = _unit(jnp.sum(chosen[..., None] * faces, axis=-2))
    inside_depth = (
        half[..., 0] * jnp.abs(inside[..., 0])
        + half[..., 1] * jnp.abs(inside[..., 1])
        - local[..., 0] * inside[..., 0]
        - local[..., 1] * inside[..., 1]
        + radii[:, None]
    )
    local_push = jnp.where(
        (distance > 0)[..., None],
        outside * jnp.maximum(radii[:, None] - distance, 0)[..., None],
        inside * jnp.maximum(inside_depth, 0)[..., None],
    )
    wall_push = jnp.stack(
        (
            local_push[..., 0] * cosine - local_push[..., 1] * sine,
            local_push[..., 0] * sine + local_push[..., 1] * cosine,
        ),
        axis=-1,
    )
    delta = (
        positions[:, None, :]
        - obstacles[None, :, OBSTACLE_FEATURE_X : OBSTACLE_FEATURE_Y + 1]
    )
    pillar_normal, pillar_distance = _unit(delta)
    prior_normal, _ = _unit(
        start[:, None, :]
        - obstacles[None, :, OBSTACLE_FEATURE_X : OBSTACLE_FEATURE_Y + 1]
    )
    pillar_normal = jnp.where(
        (pillar_distance > 0)[..., None], pillar_normal, prior_normal
    )
    pillar_push = (
        pillar_normal
        * jnp.maximum(
            radii[:, None]
            + obstacles[None, :, OBSTACLE_FEATURE_RADIUS]
            - pillar_distance,
            0,
        )[..., None]
    )
    pillar = obstacles[None, :, OBSTACLE_FEATURE_TYPE] == OBSTACLE_TYPE_PILLAR
    active = (obstacles[None, :, OBSTACLE_FEATURE_ACTIVE] == 1) & (
        pillar | (obstacles[None, :, OBSTACLE_FEATURE_TYPE] == OBSTACLE_TYPE_WALL)
    )
    correction = jnp.where(pillar[..., None], pillar_push, wall_push)
    return _sum_contact_corrections(jnp.where(active[..., None], correction, 0), axis=1)


def _commit_static_safe_positions(
    start: Array,
    proposed: Array,
    radii: Array,
    enabled: Array,
    width: Array | float,
    height: Array | float,
    obstacles: Array,
    origin: float = 0.0,
) -> Array:
    """Commit a guarded segment endpoint or keep that body's starting point.

    Start and proposed positions are (10, 2); radii and enabled masks are (10,).
    Clamp to radius-aware map bounds, shorten travel before first static contact,
    then independently check the endpoint. Disabled or invalid rows keep start.
    Width and height extend from origin, default 0.0. This checks static geometry,
    not whether the finite body-contact solver removed every body overlap.
    """
    proposed = jnp.clip(
        proposed,
        origin + radii[:, None],
        origin + jnp.asarray([width, height]) - radii[:, None],
    )
    fraction = _swept_fraction(start, proposed, radii, width, height, obstacles, origin)
    delta = proposed - start
    _, length = _unit(delta)
    safe_fraction = jnp.maximum(
        fraction - _STATIC_TRAVEL_MARGIN / jnp.maximum(length, _STATIC_TRAVEL_MARGIN), 0
    )
    accepted = start + safe_fraction[:, None] * delta
    accepted = jnp.where((fraction >= 1)[:, None], proposed, accepted)
    valid = _static_positions_are_valid(
        accepted, radii, width, height, obstacles, origin
    )
    return jnp.where((enabled & valid)[:, None], accepted, start)


def _slide_corrections(
    start: Array, proposed: Array, radii: Array, obstacles: Array
) -> Array:
    """Turn a blocked proposal along the shape at its first obstacle contact.

    Start and proposed positions (10, 2), radii (10,) and obstacle rows (32, 8)
    produce corrections (10, 2). Use the contact normal of the actual wall or
    pillar, then sum active contributions. The caller still guards the resulting
    segment; this correction alone does not commit a position.
    """
    fraction = _obstacle_entry_fractions(start, proposed, radii, obstacles)
    delta = proposed - start
    local, cosine, sine = _wall_frame(start, obstacles)
    local_delta = jnp.stack(
        (
            delta[:, None, 0] * cosine + delta[:, None, 1] * sine,
            delta[:, None, 1] * cosine - delta[:, None, 0] * sine,
        ),
        axis=-1,
    )
    contact = local + fraction[..., None] * local_delta
    half = (
        obstacles[None, :, OBSTACLE_FEATURE_WIDTH : OBSTACLE_FEATURE_HEIGHT + 1] * 0.5
    )
    local_normal, wall_distance = _unit(contact - jnp.clip(contact, -half, half))
    wall_normal = jnp.stack(
        (
            local_normal[..., 0] * cosine - local_normal[..., 1] * sine,
            local_normal[..., 0] * sine + local_normal[..., 1] * cosine,
        ),
        axis=-1,
    )
    pillar_normal, pillar_distance = _unit(
        start[:, None, :]
        - obstacles[None, :, OBSTACLE_FEATURE_X : OBSTACLE_FEATURE_Y + 1]
        + fraction[..., None] * delta[:, None, :]
    )
    pillar = obstacles[None, :, OBSTACLE_FEATURE_TYPE] == OBSTACLE_TYPE_PILLAR
    depth = jnp.where(
        pillar,
        radii[:, None] + obstacles[None, :, OBSTACLE_FEATURE_RADIUS] - pillar_distance,
        radii[:, None] - wall_distance,
    )
    normal = jnp.where(pillar[..., None], pillar_normal, wall_normal)
    remainder = (1 - fraction)[..., None] * delta[:, None, :]
    inward = depth - (
        remainder[..., 0] * normal[..., 0] + remainder[..., 1] * normal[..., 1]
    )
    blocked = (
        (obstacles[None, :, OBSTACLE_FEATURE_ACTIVE] == 1)
        & (pillar | (obstacles[None, :, OBSTACLE_FEATURE_TYPE] == OBSTACLE_TYPE_WALL))
        & (fraction < 1)
    )
    return _sum_contact_corrections(
        jnp.where(blocked[..., None], normal * jnp.maximum(inward, 0)[..., None], 0),
        axis=1,
    )


def _project_geometry(
    agent_positions: Array,
    agent_radii: Array,
    intended_movement_deltas: Array,
    active_mask: Array,
    alive_mask: Array,
    map_width: Array | float,
    map_height: Array | float,
    obstacles: Array,
    always_participates_in_agent_agent_collision: Array,
    participates_in_agent_agent_collision_at_final_position: Array,
    agent_agent_overlap_projection_passes: int,
    collision_projection_passes: int,
    movement_substeps: int,
    endpoint_mode: bool,
) -> Array:
    """Run the shared movement or Charge solver with fixed loop controls.

    Array contracts match the two public projection helpers. Positive substeps
    and nonnegative pass counts are static Python integers; invalid counts raise
    ValueError before numerical execution. Endpoint mode is also static.
    Charge recovers its arrival before resolving contacts. Ordinary movement
    splits travel into substeps and uses each substep's fixed start for guards.
    The result is positions (10, 2); the input arrays are not mutated.
    """
    if (
        movement_substeps < 1
        or min(agent_agent_overlap_projection_passes, collision_projection_passes) < 0
    ):
        raise ValueError("Substeps must be positive and pass counts nonnegative.")
    enabled = active_mask & alive_mask
    size = jnp.asarray([map_width, map_height], agent_positions.dtype)
    first, second = (jnp.asarray(_FIRST), jnp.asarray(_SECOND))
    origin = 0.0
    deltas = jnp.where(
        enabled[:, None], intended_movement_deltas / movement_substeps, 0
    )
    budget = collision_projection_passes

    def static_project(position: Array, start: Array) -> Array:
        """Correct bounds and obstacle penetration using one fixed starting snapshot.

        Return positions (10, 2), retaining start for disabled rows. This closure
        shares the parent geometry inputs; segment safety is checked separately.
        """
        bounded = jnp.clip(
            position, origin + agent_radii[:, None], size - agent_radii[:, None]
        )
        corrected = bounded + _obstacle_corrections(
            bounded, agent_radii, obstacles, start
        )
        return jnp.where(enabled[:, None], corrected, start)

    def substep(index: Array, current: Array) -> Array:
        """Commit one movement increment from fixed starting positions.

        The scalar index selects the earlier or final contact mask. The final
        mask replaces the earlier mask, including in a one-substep call.
        Endpoint mode first recovers Charge arrivals. Both paths return (10, 2)
        positions after bounded contact rounds and guarded static correction.
        """
        intended = current + deltas
        start = current
        tie_direction = None
        if endpoint_mode:
            # Charge uses its arrival scene. Earlier positions only break ties.
            previous = current[first] - current[second]
            tie_direction = jnp.where(
                jnp.any(previous != 0, axis=-1)[:, None],
                previous,
                deltas[first] - deltas[second],
            )

            def recover_static(_: int, position: Array) -> Array:
                """Recover one Charge arrival using its original approach.

                The loop index is unused. Positions (10, 2) are the loop
                carry. This endpoint phase does not require a safe travel
                segment from the pre-Charge location.
                """
                return static_project(position, current)

            recovered = cast(
                Array, jax.lax.fori_loop(0, budget, recover_static, intended)
            )
            valid = _static_positions_are_valid(
                recovered, agent_radii, map_width, map_height, obstacles, origin
            )
            intended = jnp.where((enabled & valid)[:, None], recovered, current)
            start = intended

        def correct_static(position: Array) -> Array:
            """Correct a candidate endpoint and add its first-contact sliding response.

            This closure reads the substep's fixed start. It returns a proposal with
            shape (10, 2); the caller then guards the entire correction segment.
            """
            corrected = static_project(position, start)
            return corrected + _slide_corrections(
                start, corrected, agent_radii, obstacles
            )

        participant = enabled & jnp.where(
            index == movement_substeps - 1,
            participates_in_agent_agent_collision_at_final_position,
            always_participates_in_agent_agent_collision,
        )
        recovering_pairs = None
        if not endpoint_mode:
            joining = participant & ~always_participates_in_agent_agent_collision
            recovering_pairs = joining[first] | joining[second]
        initial_static = static_project(intended, start)
        initial_slide = initial_static + _slide_corrections(
            start, initial_static, agent_radii, obstacles
        )
        initial = _commit_static_safe_positions(
            start,
            initial_slide,
            agent_radii,
            enabled,
            map_width,
            map_height,
            obstacles,
            origin,
        )

        def round_step(position: Array, _: None) -> tuple[Array, None]:
            """Run one body-contact round followed by guarded static correction.

            Positions (10, 2) are the scan carry. The second argument and output are
            None so intermediate positions are not collected as a scan history.
            """
            body = position
            if agent_agent_overlap_projection_passes > 0:

                def separate_bodies(_: int, current_position: Array) -> Array:
                    """Apply one body-separation sweep to positions (10, 2).

                    The loop index is unused. All pair corrections read
                    the same current positions. Their sum then updates
                    the loop carry.
                    """
                    return current_position + _body_corrections(
                        current_position,
                        agent_radii,
                        participant,
                        start,
                        intended,
                        tie_direction,
                        recovering_pairs,
                    )

                body = cast(
                    Array,
                    jax.lax.fori_loop(
                        0,
                        agent_agent_overlap_projection_passes,
                        separate_bodies,
                        position,
                    ),
                )
            corrected = correct_static(body)
            accepted = _commit_static_safe_positions(
                start,
                corrected,
                agent_radii,
                enabled,
                map_width,
                map_height,
                obstacles,
                origin,
            )
            return (accepted, None)

        result, _ = jax.lax.scan(round_step, initial, None, length=budget)
        result = jnp.where(enabled[:, None], result, agent_positions)
        return result

    def advance(current: Array, index: Array) -> tuple[Array, None]:
        """Carry positions through one physical substep without collecting history.

        Current positions are (10, 2); index selects the substep. Return the next
        positions and None for the outer scan's unused output.
        """
        return (substep(index, current), None)

    result, _ = jax.lax.scan(advance, agent_positions, jnp.arange(movement_substeps))
    return result


def project_movement_with_geometry(
    agent_positions: Array,
    agent_radii: Array,
    intended_movement_deltas: Array,
    active_mask: Array,
    alive_mask: Array,
    map_width: Array | float,
    map_height: Array | float,
    obstacles: Array,
    always_participates_in_agent_agent_collision: Array,
    participates_in_agent_agent_collision_at_final_position: Array,
    agent_agent_overlap_projection_passes: int = 1,
    collision_projection_passes: int = DEFAULT_AGENT_PROJECTION_PASSES,
    movement_substeps: int = DEFAULT_MOVEMENT_SUBSTEPS,
) -> Array:
    """Move bodies through bounds, obstacles and other participating bodies.

    Each physical substep keeps one starting point. Numerical rounds correct the
    proposed endpoint, and only the guarded straight segment is committed. Static
    travel must remain safe; crowded bodies may retain some overlap after the
    finite contact budget.

    Parameters
    ----------
    agent_positions : jax.Array
        Float32 (10, 2) starting world [x, y] coordinates.
    agent_radii : jax.Array
        Float32 (10,) nonnegative body radii in world units.
    intended_movement_deltas : jax.Array
        Float32 (10, 2) requested world displacements for this call.
    active_mask : jax.Array
        Bool (10,) configured membership.
    alive_mask : jax.Array
        Bool (10,) current alive status. Dead and unused rows retain their input
        positions and do not move or participate in body contacts.
    map_width, map_height : float or jax.Array
        Positive finite scalar map dimensions in world units, with origin (0, 0).
    obstacles : jax.Array
        Validated float32 (32, 8) EnvConfig.obstacles table.
    always_participates_in_agent_agent_collision : jax.Array
        Bool (10,) body-contact participation before the final substep. The
        final-position mask replaces this mask on the final substep. Charge
        has only one substep, so this mask does not select Charge contacts.
    participates_in_agent_agent_collision_at_final_position : jax.Array
        Bool (10,) body-contact participation on the final substep. This mask
        replaces the earlier mask; the two are not ORed together. Every
        selected participant must also be active and alive.
    agent_agent_overlap_projection_passes : int, default 1
        Static nonnegative number of body-separation sweeps per collision round.
        Zero skips body correction.
    collision_projection_passes : int, default 28
        Static nonnegative number of collision rounds, used as a literal count.

    movement_substeps : int, default 4
        Static positive number of physical increments sharing the requested travel.

    Returns
    -------
    jax.Array
        Float32 (10, 2) final world positions. Inputs are not mutated.

    Raises
    ------
    ValueError
        movement_substeps is below one, or either pass count is negative.

    Notes
    -----
    This pure JAX helper handles one game. Use vmap for separate games and keep
    loop controls static under jit. Configuration validation owns dtype, shape and
    geometry validity. The caller owns action meanings, status and shield timing;
    this helper neither advances those values nor draws randomness.
    """
    return _project_geometry(
        agent_positions,
        agent_radii,
        intended_movement_deltas,
        active_mask,
        alive_mask,
        map_width,
        map_height,
        obstacles,
        always_participates_in_agent_agent_collision,
        participates_in_agent_agent_collision_at_final_position,
        agent_agent_overlap_projection_passes,
        collision_projection_passes,
        movement_substeps,
        False,
    )


def project_charge_endpoints_with_geometry(
    agent_positions: Array,
    agent_radii: Array,
    intended_movement_deltas: Array,
    active_mask: Array,
    alive_mask: Array,
    map_width: Array | float,
    map_height: Array | float,
    obstacles: Array,
    always_participates_in_agent_agent_collision: Array,
    participates_in_agent_agent_collision_at_final_position: Array,
    agent_agent_overlap_projection_passes: int = 1,
    collision_projection_passes: int = DEFAULT_AGENT_PROJECTION_PASSES,
) -> Array:
    """Place simultaneous Charge arrivals and resolve their endpoint contacts.

    Requested relocation may pass intervening bodies or obstacles. Recover a
    statically valid arrival first, then use that scene for body contacts and the
    guarded correction path. Older physical separation only breaks a direction
    tie. Ordinary movement later starts from these positions with its already
    chosen voluntary action.

    Parameters
    ----------
    agent_positions : jax.Array
        Float32 (10, 2) starting world [x, y] coordinates.
    agent_radii : jax.Array
        Float32 (10,) nonnegative body radii in world units.
    intended_movement_deltas : jax.Array
        Float32 (10, 2) requested world displacements for this call.
    active_mask : jax.Array
        Bool (10,) configured membership.
    alive_mask : jax.Array
        Bool (10,) current alive status. Dead and unused rows retain their input
        positions and do not move or participate in body contacts.
    map_width, map_height : float or jax.Array
        Positive finite scalar map dimensions in world units, with origin (0, 0).
    obstacles : jax.Array
        Validated float32 (32, 8) EnvConfig.obstacles table.
    always_participates_in_agent_agent_collision : jax.Array
        Bool (10,) body-contact participation before the final substep. The
        final-position mask replaces this mask on the final substep. Charge
        has only one substep, so this mask does not select Charge contacts.
    participates_in_agent_agent_collision_at_final_position : jax.Array
        Bool (10,) body-contact participation on the final substep. This mask
        replaces the earlier mask; the two are not ORed together. Every
        selected participant must also be active and alive.
    agent_agent_overlap_projection_passes : int, default 1
        Static nonnegative number of body-separation sweeps per collision round.
        Zero skips body correction.
    collision_projection_passes : int, default 28
        Static nonnegative number of collision rounds, used as a literal count.

    Returns
    -------
    jax.Array
        Float32 (10, 2) final world positions after one endpoint step. Arrival
        recovery uses collision_projection_passes rounds as well as the later
        contact solver. A finite budget may leave some body overlap.

    Raises
    ------
    ValueError
        Either pass count is negative.

    Notes
    -----
    This pure JAX helper handles one game and draws no randomness. Use vmap for
    games and keep pass counts static under jit. The caller decides which actors
    Charge, their intended displacement and contact masks. No action, health,
    status or cooldown state is changed here.
    """
    return _project_geometry(
        agent_positions,
        agent_radii,
        intended_movement_deltas,
        active_mask,
        alive_mask,
        map_width,
        map_height,
        obstacles,
        always_participates_in_agent_agent_collision,
        participates_in_agent_agent_collision_at_final_position,
        agent_agent_overlap_projection_passes,
        collision_projection_passes,
        1,
        True,
    )
