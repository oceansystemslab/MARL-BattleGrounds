"""Share deterministic target and movement choices between reactive controllers.

These helpers read permitted observations and exact action masks. Target helpers
choose rows; movement helpers compare possible moves using Core's existing static
geometry functions. They neither change collision rules nor predict other actors'
actions. The BETA movement path treats observed bodies as stationary obstacles
when choosing a move. All choices remain the controller's preferences, not a
guarantee of the next physical outcome.
"""

import jax.numpy as jnp
from jax import Array, vmap

from marl_battlegrounds.core.axis_mappings import (
    UNIT_DIRECTION_VECTOR_BY_MOVEMENT_ACTION_ARRAY,
)
from marl_battlegrounds.core.geometry import (
    GEOMETRY_TOLERANCE,
    disc_overlaps_obstacle,
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
    OBSTACLE_FEATURE_ACTIVE,
    OBSTACLE_FEATURE_HEIGHT,
    OBSTACLE_FEATURE_THETA,
    OBSTACLE_FEATURE_TYPE,
    OBSTACLE_FEATURE_WIDTH,
    OBSTACLE_FEATURE_X,
    OBSTACLE_FEATURE_Y,
    OBSTACLE_TYPE_WALL,
    ActionMask,
    Observation,
)

MINIMUM_MOVEMENT_FRACTION = 0.1
BYPASS_MINIMUM_CONTACT_ANGLE_DEGREES = 45


def centers(features: Array) -> Array:
    """Read world-space (x, y) centers from unit-feature rows.

    Parameters
    ----------
    features : Array
        One feature row of shape (58,) or several rows of shape (..., 58).

    Returns
    -------
    Array
        The selected position columns, shape (..., 2), with the input dtype and
        all leading axes preserved. Positions use the map's world units.

    Notes
    -----
    The caller must use visibility and activity masks before treating a row as
    a known position. This helper does not distinguish a hidden zero row from
    a visible unit at the origin.
    """
    return features[..., jnp.asarray([AGENT_FEATURE_X, AGENT_FEATURE_Y])]


def living_candidates(features: Array, visible: Array) -> Array:
    """Select visible, active, living candidates with positive health.

    Parameters
    ----------
    features : Array
        Candidate features of shape (N, 58), in the caller's row order.
    visible : Array
        Boolean visibility for the same N candidates, shape (N,).

    Returns
    -------
    Array
        Boolean (N,) mask. A row is True only when visibility, activity, life
        state and positive current health all agree.

    Notes
    -----
    The helper reads supplied features without changing them or querying Core.
    The caller owns the observation's information permissions.
    """
    return (
        visible
        & (features[:, AGENT_FEATURE_ACTIVE] > 0)
        & (features[:, AGENT_FEATURE_ALIVE] > 0)
        & (features[:, AGENT_FEATURE_CURRENT_HEALTH] > 0)
    )


def lowest_health_row(
    features: Array, eligible: Array, *, break_ties_by_max_health: bool = False
) -> Array:
    """Choose the eligible row with the lowest current health.

    Parameters
    ----------
    features : Array
        Candidate features of shape (N, 58). Health is measured in health points,
        not as a fraction of maximum health.
    eligible : Array
        Boolean (N,) mask of candidates the caller permits.
    break_ties_by_max_health : bool, default=False
        If True, equal current health is resolved by the lower maximum health.
        Otherwise skip that comparison. Keep this Python choice static in JAX.

    Returns
    -------
    Array
        Scalar int32 row index. Remaining exact ties use the lowest row index.
        If no row is eligible, the result is 0; it does not mean row 0 is valid.

    Notes
    -----
    Check whether eligible contains any True values before using the result.
    Ally and enemy observation rows retain their own fixed order. Inputs are
    unchanged and no random tie-break is made.
    """
    health = features[:, AGENT_FEATURE_CURRENT_HEALTH]
    tied = eligible & (health == jnp.min(jnp.where(eligible, health, jnp.inf)))
    if break_ties_by_max_health:
        max_health = features[:, AGENT_FEATURE_MAX_HEALTH]
        tied &= max_health == jnp.min(jnp.where(tied, max_health, jnp.inf))
    return jnp.argmax(tied).astype(jnp.int32)


def nearest_row(features: Array, eligible: Array, origin: Array) -> Array:
    """Choose the eligible candidate whose center is nearest to the origin.

    Parameters
    ----------
    features : Array
        Candidate features of shape (N, 58).
    eligible : Array
        Boolean (N,) mask of permitted candidates.
    origin : Array
        World-space (x, y) point, shape (2,), in the same units as the features.

    Returns
    -------
    Array
        Scalar int32 row index. Exact distance ties choose the lowest row.
        An empty eligible set returns 0; the caller must check eligibility.

    Notes
    -----
    Distances compare centers and do not subtract body radii or test reachability.
    Inputs are unchanged.
    """
    distance_squared = jnp.sum(jnp.square(centers(features) - origin), axis=-1)
    return jnp.argmin(jnp.where(eligible, distance_squared, jnp.inf)).astype(jnp.int32)


def refine_movement(
    observation: Observation,
    action_mask: ActionMask,
    intended_direction: Array,
    *,
    approach: Array | bool = False,
) -> Array:
    """Choose a legal direction that makes useful progress through static geometry.

    Parameters
    ----------
    observation : Observation
        One actor's current observation. Position, body radius, movement speed,
        map bounds and visible map geometry supply the movement calculation.
    action_mask : ActionMask
        The actor's exact mask for the same decision.
    intended_direction : Array
        World-space displacement toward the intended goal, shape (2,).
        Direction ranking normalizes it, but wall steering uses its endpoint
        at the current position plus this displacement. Scaling it can change
        approach steering. Zero intent keeps Stay.
    approach : Array or bool, default=False
        Whether movement approaches a goal. True allows the bounded wall-end
        steering rule; retreat movement does not enter that rule.

    Returns
    -------
    Array
        Scalar int32 movement category. Prefer the best direction alignment
        among admitted moves; exact ties use the lower movement category.
        Return Stay when no useful legal move remains.

    Notes
    -----
    Core's static obstacle and map-bound projection evaluates eight possible moves
    without body-pair contacts. A move needs at least MINIMUM_MOVEMENT_FRACTION of
    the actor's movement speed as displacement, with additional progress checks
    during wall steering. This is a one-decision preference, not a route planner
    or a prediction of simultaneous body movement. The helper is JAX-compatible
    and changes no state.
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
        collision_projection_passes=4,
    )[:8]
    displacement = jnp.sqrt(jnp.sum(jnp.square(projected - origin), axis=-1))
    direction_length = jnp.sqrt(jnp.sum(jnp.square(intended_direction)))
    admissible = (
        action_mask.move_mask[1:]
        & (speed > 0)
        & (direction_length > 0)
        & (displacement >= MINIMUM_MOVEMENT_FRACTION * speed)
    )
    intended_direction, steering, end_direction, end_height = _wall_steering(
        observation,
        origin + intended_direction,
        approach,
        candidate_origins=jnp.where(admissible[:, None], projected, origin),
    )
    direction_length = jnp.sqrt(jnp.sum(jnp.square(intended_direction)))
    alignment = directions @ (
        intended_direction / jnp.where(direction_length > 0, direction_length, 1.0)
    )
    phase_progress = (projected - origin) @ intended_direction
    away_from_end = (projected[:, 1] - origin[1]) * end_direction
    clears_end = (projected[:, 1] - end_height) * end_direction
    preserves_clearance = (away_from_end >= -GEOMETRY_TOLERANCE) | (
        clears_end >= -GEOMETRY_TOLERANCE
    )
    preferred = admissible & (
        ~steering
        | ((phase_progress >= MINIMUM_MOVEMENT_FRACTION * speed) & preserves_clearance)
    )
    admissible = jnp.where(
        steering & ~jnp.any(preferred),
        admissible & (phase_progress > GEOMETRY_TOLERANCE) & preserves_clearance,
        preferred,
    )
    best = 1 + jnp.argmax(jnp.where(admissible, alignment, -jnp.inf))
    return jnp.where(jnp.any(admissible), best, MOVE_STAY).astype(jnp.int32)


def _wall_steering(
    observation: Observation,
    goal: Array,
    approach_enabled: Array | bool,
    *,
    candidate_origins: Array | None = None,
    bodies: Array | None = None,
    body_mask: Array | None = None,
) -> tuple[Array, Array, Array, Array]:
    """Choose a temporary direction around a nearby vertical wall end.

    Parameters
    ----------
    observation : Observation
        One actor's current permitted position, body size, movement state and map.
    goal : Array
        World-space destination, shape (2,).
    approach_enabled : Array or bool
        Scalar Boolean enabling this approach-only steering rule.
    candidate_origins : Array or None, default=None
        Optional possible next centers, shape (N, 2). Callers replace inadmissible
        candidates with the current center before passing them here.
    bodies : Array or None, default=None
        Optional observed unit rows of shape (N, 58). Supplying bodies selects
        BETA's wall grouping and boundary-passage handling.
    body_mask : Array or None, default=None
        Boolean (N,) blocker mask, paired with bodies. Pass both for body-aware
        steering. The caller excludes self, prey and unseen bodies.

    Returns
    -------
    direction : Array
        Float32 (2,) preferred direction. Without active steering, this is
        goal minus the actor's current center.
    active : Array
        Scalar Boolean saying a nearby wall steering rule applies.
    end_direction : Array
        Scalar direction toward the selected wall end: -1 south, 1 north,
        or 0 for no selected end. Active steering with 0 means neither end fits.
    end_height : Array
        Selected wall-end y coordinate in world units. Use it only when the
        steering and end-selection flags make it relevant.

    Notes
    -----
    This rule handles axis-aligned walls taller than they are wide, including
    quarter-turn obstacle encodings. It prefers a fitting south end, with a north
    alternative. The body-aware branch considers directly overlapping expanded
    wall groups, not an unbounded route search. It carries no route memory and
    changes neither physical geometry nor observations. Possible moves still
    need the caller's legality and progress checks.
    """
    own = observation.self_features
    origin = centers(own)
    radius = own[AGENT_FEATURE_RADIUS]
    obstacles = observation.map_obstacle_features
    theta = obstacles[:, OBSTACLE_FEATURE_THETA]
    cos_theta, sin_theta = jnp.abs(jnp.cos(theta)), jnp.abs(jnp.sin(theta))
    width = jnp.where(
        cos_theta >= sin_theta,
        obstacles[:, OBSTACLE_FEATURE_WIDTH],
        obstacles[:, OBSTACLE_FEATURE_HEIGHT],
    )
    height = jnp.where(
        cos_theta >= sin_theta,
        obstacles[:, OBSTACLE_FEATURE_HEIGHT],
        obstacles[:, OBSTACLE_FEATURE_WIDTH],
    )
    x, y = obstacles[:, OBSTACLE_FEATURE_X], obstacles[:, OBSTACLE_FEATURE_Y]
    west, east = x - width / 2 - radius, x + width / 2 + radius
    south, north = y - height / 2 - radius, y + height / 2 + radius
    tolerance = GEOMETRY_TOLERANCE
    east_goal = goal[0] >= x + width / 2 - tolerance
    west_goal = goal[0] <= x - width / 2 + tolerance
    exit_direction = jnp.where(east_goal, 1.0, -1.0)
    entry = jnp.where(east_goal, west, east)
    band = width + 2 * radius
    cross_progress = (origin[0] - entry) * exit_direction
    south_fits = south >= radius - tolerance
    north_fits = north <= (
        observation.context_features[CONTEXT_FEATURE_MAP_HEIGHT] - radius + tolerance
    )
    if bodies is None:
        exit_x = jnp.where(east_goal, east, west)
        exit_centers = jnp.stack(
            (jnp.stack((exit_x, south), axis=-1), jnp.stack((exit_x, north), axis=-1)),
            axis=1,
        )

        def exit_clear(center: Array) -> Array:
            """Check whether an actor-sized disc at this exit overlaps any map obstacle.

            The enclosing call supplies the body radius and observed obstacle array.
            This checks static exit clearance, not other actors or an entire route.
            """
            return ~jnp.any(
                vmap(disc_overlaps_obstacle, in_axes=(None, None, 0))(
                    center, radius, obstacles
                )
            )

        clear = vmap(vmap(exit_clear))(exit_centers)
        south_fits &= clear[:, 0]
        north_fits &= clear[:, 1]
    end_direction = jnp.where(south_fits, -1.0, jnp.where(north_fits, 1.0, 0.0))
    end = jnp.where(south_fits, south, north)
    end_progress = (origin[1] - end) * end_direction
    alongside = (
        (cross_progress >= -band - tolerance)
        & (cross_progress <= tolerance)
        & (origin[1] >= south - tolerance)
        & (origin[1] <= north + tolerance)
    )
    crossing = (
        (end_direction != 0)
        & (end_progress >= -tolerance)
        & (end_progress <= band + tolerance)
        & (cross_progress >= -band - tolerance)
        & (cross_progress < band - tolerance)
    )
    if bodies is not None:
        lateral_near = (cross_progress >= -band - tolerance) & (
            cross_progress <= band + tolerance
        )
        alongside = (
            lateral_near
            & (origin[1] >= south - tolerance)
            & (origin[1] <= north + tolerance)
        )
        crossing = lateral_near & (
            ((origin[1] >= north - tolerance) & (origin[1] <= north + band + tolerance))
            | (
                (origin[1] <= south + tolerance)
                & (origin[1] >= south - band - tolerance)
            )
        )
        if candidate_origins is not None:
            probe_cross = (candidate_origins[:, 0, None] - entry) * exit_direction
            probe_lateral = (probe_cross >= -band - tolerance) & (
                probe_cross <= band + tolerance
            )
            probe_near = (
                probe_lateral
                & (candidate_origins[:, 1, None] >= south - band - tolerance)
                & (candidate_origins[:, 1, None] <= north + band + tolerance)
            )
            alongside |= jnp.any(probe_near, axis=0)
    eligible = (
        (obstacles[:, OBSTACLE_FEATURE_ACTIVE] > 0)
        & (obstacles[:, OBSTACLE_FEATURE_TYPE] == OBSTACLE_TYPE_WALL)
        & ((sin_theta <= tolerance) | (cos_theta <= tolerance))
        & (height > width)
        & (east_goal | west_goal)
        & (alongside | crossing)
        & approach_enabled
        & (own[AGENT_FEATURE_ACTIVE] > 0)
        & (own[AGENT_FEATURE_ALIVE] > 0)
        & (own[AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED] > 0)
        & jnp.any(goal != origin)
    )
    eligibility_south = south + (radius if bodies is not None else 0.0)
    eligibility_north = north - (radius if bodies is not None else 0.0)
    same_clear_end = (
        (origin[1] <= eligibility_south + tolerance)
        & (goal[1] <= eligibility_south + tolerance)
    ) | (
        (origin[1] >= eligibility_north - tolerance)
        & (goal[1] >= eligibility_north - tolerance)
    )
    eligible &= ~same_clear_end
    if bodies is not None:
        anchor_members = (
            (west[None, :] < east[:, None] - tolerance)
            & (east[None, :] > west[:, None] + tolerance)
            & (south[None, :] < north[:, None] - tolerance)
            & (north[None, :] > south[:, None] + tolerance)
            & (obstacles[None, :, OBSTACLE_FEATURE_ACTIVE] > 0)
            & (obstacles[None, :, OBSTACLE_FEATURE_TYPE] == OBSTACLE_TYPE_WALL)
            & ((sin_theta[None, :] <= tolerance) | (cos_theta[None, :] <= tolerance))
            & (height[None, :] > width[None, :])
        )
        anchor_left = jnp.min(jnp.where(anchor_members, west[None, :], jnp.inf), axis=1)
        anchor_right = jnp.max(
            jnp.where(anchor_members, east[None, :], -jnp.inf), axis=1
        )
        anchor_exit = jnp.where(goal[0] >= anchor_right - tolerance, 1.0, -1.0)
        anchor_far_face = jnp.where(anchor_exit > 0, anchor_right, anchor_left)
        eligible &= (origin[0] - anchor_far_face) * anchor_exit < -tolerance
    row = jnp.argmin(
        jnp.where(eligible, (x - origin[0]) ** 2 + (y - origin[1]) ** 2, jnp.inf)
    )
    active = jnp.any(eligible)
    if bodies is not None and body_mask is not None:
        touching = (
            (west < east[row] - tolerance)
            & (east > west[row] + tolerance)
            & (south < north[row] - tolerance)
            & (north > south[row] + tolerance)
        )
        members = (
            touching
            & (obstacles[:, OBSTACLE_FEATURE_ACTIVE] > 0)
            & (obstacles[:, OBSTACLE_FEATURE_TYPE] == OBSTACLE_TYPE_WALL)
            & ((sin_theta <= tolerance) | (cos_theta <= tolerance))
            & (height > width)
        )
        low = jnp.where(active, jnp.min(jnp.where(members, south, jnp.inf)), origin[1])
        high = jnp.where(
            active, jnp.max(jnp.where(members, north, -jnp.inf)), origin[1]
        )
        left = jnp.where(active, jnp.min(jnp.where(members, west, jnp.inf)), origin[0])
        right = jnp.where(
            active, jnp.max(jnp.where(members, east, -jnp.inf)), origin[0]
        )
        group_exit_direction = jnp.where(goal[0] >= right - tolerance, 1.0, -1.0)
        group_far_face = jnp.where(group_exit_direction > 0, right, left)
        active &= (origin[0] - group_far_face) * group_exit_direction < -tolerance
        body_centers = centers(bodies)
        closest_x = jnp.clip(body_centers[:, 0], left, right)
        required_radius = radius + bodies[:, AGENT_FEATURE_RADIUS]

        def occupied(height: Array) -> Array:
            """Check whether a permitted body blocks this wall-end passage.

            The enclosing body-aware branch supplies the group width, body radii and
            Boolean blocker mask. The scalar height uses map world units.
            """
            distance_squared = (body_centers[:, 0] - closest_x) ** 2 + (
                body_centers[:, 1] - height
            ) ** 2
            return jnp.any(body_mask & (distance_squared < required_radius**2))

        occupied_ends = vmap(occupied)(jnp.stack((low, high)))
        map_top = observation.context_features[CONTEXT_FEATURE_MAP_HEIGHT] - radius
        lower_fits = low >= radius - tolerance
        upper_fits = (
            high
            <= observation.context_features[CONTEXT_FEATURE_MAP_HEIGHT]
            - radius
            + tolerance
        )
        lower_available = lower_fits & ~((low <= radius + tolerance) & occupied_ends[0])
        if candidate_origins is not None:
            lower_available |= lower_fits & jnp.any(
                (
                    (candidate_origins[:, 0] - origin[0]) * group_exit_direction
                    >= MINIMUM_MOVEMENT_FRACTION
                    * own[AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED]
                )
                & (candidate_origins[:, 1] <= origin[1] + tolerance)
                & (candidate_origins[:, 1] <= low + radius + tolerance)
            )
        upper_available = upper_fits & ~(
            (high >= map_top - tolerance) & occupied_ends[1]
        )
        boundary_end = jnp.where(lower_fits, -1.0, jnp.where(upper_fits, 1.0, 0.0))
        selected_end = jnp.where(
            lower_available, -1.0, jnp.where(upper_available, 1.0, boundary_end)
        )
        height = jnp.where(selected_end < 0, low, high)
        horizontal = (origin[1] - height) * selected_end >= -tolerance
        if candidate_origins is not None:
            horizontal |= jnp.any(
                (
                    (candidate_origins[:, 0] - origin[0]) * group_exit_direction
                    >= MINIMUM_MOVEMENT_FRACTION
                    * own[AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED]
                )
                & (
                    ((candidate_origins[:, 1] - origin[1]) * selected_end >= -tolerance)
                    | ((candidate_origins[:, 1] - height) * selected_end >= -tolerance)
                )
            )
        direction = jnp.array(
            [
                jnp.where(horizontal, group_exit_direction, 0.0),
                jnp.where(horizontal, 0.0, selected_end),
            ],
            dtype=jnp.float32,
        )
        direction = jnp.where(selected_end != 0, direction, 0.0)
        return (
            jnp.where(active, direction, goal - origin),
            active,
            jnp.where(active, selected_end, 0.0),
            height,
        )
    horizontal = end_progress[row] >= -tolerance
    if candidate_origins is not None:
        horizontal |= jnp.any(
            (
                (candidate_origins[:, 0] - origin[0]) * exit_direction[row]
                >= MINIMUM_MOVEMENT_FRACTION
                * own[AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED]
            )
            & (
                (end_progress[row] >= -radius - tolerance)
                | (
                    (candidate_origins[:, 1] - end[row]) * end_direction[row]
                    >= -tolerance
                )
            )
            & (
                (
                    (candidate_origins[:, 1] - origin[1]) * end_direction[row]
                    >= -tolerance
                )
                | (
                    (candidate_origins[:, 1] - end[row]) * end_direction[row]
                    >= -tolerance
                )
            )
        )
    phase_direction = jnp.array(
        [
            jnp.where(horizontal, exit_direction[row], 0.0),
            jnp.where(horizontal, 0.0, end_direction[row]),
        ],
        dtype=jnp.float32,
    )
    phase_direction = jnp.where(end_direction[row] != 0, phase_direction, 0.0)
    return (
        jnp.where(active, phase_direction, goal - origin),
        active,
        jnp.where(active, end_direction[row], 0.0),
        end[row],
    )


def _body_bypass_moves(
    origin: Array,
    endpoints: Array,
    radius: Array,
    bodies: Array,
    body_mask: Array,
) -> Array:
    """Mark candidate segments that avoid head-on contact with observed bodies.

    Parameters
    ----------
    origin : Array
        Starting center, shape (2,), in world units.
    endpoints : Array
        Candidate next centers, shape (M, 2).
    radius : Array
        Scalar radius of the moving actor.
    bodies : Array
        Observed body-feature rows, shape (N, 58).
    body_mask : Array
        Boolean (N,) blocker mask. False rows cannot reject a candidate.

    Returns
    -------
    Array
        Boolean (M,) mask. Every selected blocker must admit the segment:
        clear travel or glancing first contact is allowed. A deeper initial
        overlap requires travel that does not deepen it and ends farther away.

    Notes
    -----
    Glancing contact is at least 45 degrees from head-on, using the first contact
    normal. Tiny initial overlaps within Core's geometry tolerance count as
    contact. Bodies are assumed stationary for this preference. The function
    does not perform the simulation's collision resolution or alter body radii.
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
) -> Array:
    """Choose BETA's movement toward prey while checking observed body contacts.

    Parameters
    ----------
    observation : Observation
        The moving actor's current permitted observation.
    action_mask : ActionMask
        Its exact current action mask.
    prey_center : Array
        Observed prey center, shape (2,), in world units.
    bodies : Array
        Candidate blocker features of shape (N, 58).
    body_mask : Array
        Boolean (N,) mask of visible living blockers, excluding self and prey.

    Returns
    -------
    Array
        Scalar int32 movement category. Among admitted moves, prefer progress
        along active wall steering or a closer endpoint to the prey. Remaining
        ties use endpoint distance and then the lower movement category.
        Return Stay if no admitted move remains.

    Notes
    -----
    Static endpoints come from Core's geometry helper, without body-pair solving.
    Contact checks treat other observed bodies as stationary. Wall steering can
    permit a temporary move away from prey. No history, hidden positions, other
    actors' selected actions or extra random draws are used.
    """
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
        collision_projection_passes=4,
    )[:8]
    displacement = jnp.sqrt(jnp.sum(jnp.square(endpoints - origin), axis=-1))
    admissible = (
        action_mask.move_mask[1:]
        & (speed > 0)
        & (displacement >= MINIMUM_MOVEMENT_FRACTION * speed)
        & _body_bypass_moves(origin, endpoints, radius, bodies, body_mask)
    )
    distance_squared = jnp.sum(jnp.square(endpoints - prey_center), axis=-1)
    direction, steering, end_direction, end_height = _wall_steering(
        observation,
        prey_center,
        True,
        candidate_origins=jnp.where(admissible[:, None], endpoints, origin),
        bodies=bodies,
        body_mask=body_mask,
    )
    phase_progress = (endpoints - origin) @ direction
    away_from_end = (endpoints[:, 1] - origin[1]) * end_direction
    clears_end = (endpoints[:, 1] - end_height) * end_direction
    preferred = admissible & (
        ~steering
        | (
            (phase_progress >= MINIMUM_MOVEMENT_FRACTION * speed)
            & (
                (away_from_end >= -GEOMETRY_TOLERANCE)
                | (clears_end >= -GEOMETRY_TOLERANCE)
            )
        )
    )
    escape = steering & (end_direction != 0) & ~jnp.any(preferred)
    admissible = jnp.where(escape, admissible, preferred)
    score = jnp.where(steering & ~escape, directions @ direction, -distance_squared)
    best_score = jnp.max(jnp.where(admissible, score, -jnp.inf))
    best_alignment = admissible & (score == best_score)
    move = 1 + jnp.argmin(jnp.where(best_alignment, distance_squared, jnp.inf))
    return jnp.where(jnp.any(admissible), move, MOVE_STAY).astype(jnp.int32)
