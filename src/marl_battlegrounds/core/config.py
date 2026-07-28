"""Host-facing episode configuration and fixed-slot profile resolution."""

import math

import jax.numpy as jnp
import numpy as np
from jax import Array

from marl_battlegrounds.core.combat import (
    get_base_movement_speed_by_class_ids,
    get_basic_interaction_radius_by_class_ids,
    get_body_radius_by_class_ids,
    get_max_health_by_class_ids,
    get_observation_radius_by_class_ids,
    get_ultimate_interaction_radius_by_class_ids,
)
from marl_battlegrounds.core.geometry import disc_overlaps_obstacle
from marl_battlegrounds.core.types import (
    ENVIRONMENT_DIMENSIONS,
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    MAX_OBSTACLE_SLOTS,
    NEUTRAL_CLASS_ID,
    NO_TEAM_ID,
    NUM_CLASSES,
    OBSTACLE_FEATURE_ACTIVE,
    OBSTACLE_FEATURE_HEIGHT,
    OBSTACLE_FEATURE_RADIUS,
    OBSTACLE_FEATURE_THETA,
    OBSTACLE_FEATURE_TYPE,
    OBSTACLE_FEATURE_WIDTH,
    OBSTACLE_FEATURE_X,
    OBSTACLE_FEATURE_Y,
    OBSTACLE_FEATURES,
    OBSTACLE_TYPE_PILLAR,
    OBSTACLE_TYPE_WALL,
    TEAM_A_ID,
    TEAM_B_ID,
    EnvConfig,
    ResolvedAgentProfile,
)

_INT32_MAX = int(np.iinfo(np.int32).max)
_FLOAT32_MAX = float(np.finfo(np.float32).max)


def resolve_agent_profile(
    requested_class_ids: Array, team_sizes: Array
) -> ResolvedAgentProfile:
    """Resolve requested team rosters into immutable padded slot arrays.

    ``requested_class_ids`` has shape ``(MAX_AGENT_SLOTS,)`` and ``team_sizes``
    has shape ``(2,)``. Padded slots receive neutral class/catalog values and
    ``NO_TEAM_ID``; active team blocks receive their explicit public team IDs.
    Host-side validation of the complete resolved configuration belongs to
    :func:`validate_env_config`.
    """
    team_local_indices = jnp.arange(MAX_AGENTS_PER_TEAM)

    team_a_active_mask = team_local_indices < team_sizes[0]
    team_b_active_mask = team_local_indices < team_sizes[1]

    active_mask = jnp.hstack((team_a_active_mask, team_b_active_mask))
    class_ids = jnp.where(active_mask, requested_class_ids, NEUTRAL_CLASS_ID)

    team_a_team_ids = jnp.where(team_a_active_mask, TEAM_A_ID, NO_TEAM_ID)
    team_b_team_ids = jnp.where(team_b_active_mask, TEAM_B_ID, NO_TEAM_ID)
    team_ids = jnp.hstack((team_a_team_ids, team_b_team_ids), dtype=jnp.int32)

    agent_radii = get_body_radius_by_class_ids(class_ids)
    base_movement_speeds = get_base_movement_speed_by_class_ids(class_ids)
    observation_radii = get_observation_radius_by_class_ids(class_ids)
    basic_interaction_radii = get_basic_interaction_radius_by_class_ids(class_ids)
    ultimate_interaction_radii = get_ultimate_interaction_radius_by_class_ids(class_ids)
    max_health = get_max_health_by_class_ids(class_ids)

    return ResolvedAgentProfile(
        class_ids,
        team_ids,
        active_mask,
        agent_radii,
        base_movement_speeds,
        observation_radii,
        basic_interaction_radii,
        ultimate_interaction_radii,
        max_health,
    )


def _require_jax_array(
    value: object,
    *,
    field_name: str,
    expected_shape: tuple[int, ...],
    expected_dtype: object,
) -> Array:
    """Return one JAX array after enforcing its public storage contract."""
    if not isinstance(value, Array):
        raise TypeError(
            f"{field_name} must be a jax.Array, not {type(value).__name__}."
        )
    if value.shape != expected_shape:
        raise ValueError(
            f"{field_name} must have shape {expected_shape}, not {value.shape}."
        )
    if value.dtype != expected_dtype:
        raise TypeError(
            f"{field_name} must have dtype {expected_dtype}, not {value.dtype}."
        )
    return value


def _require_finite_array(value: Array, *, field_name: str) -> None:
    """Reject nonfinite host configuration values before JAX tracing."""
    if not bool(np.all(np.isfinite(np.asarray(value)))):
        raise ValueError(f"{field_name} must contain only finite values.")


def _validate_obstacles(obstacles: object) -> Array:
    """Validate the fixed padded obstacle-table schema."""
    obstacle_array = _require_jax_array(
        obstacles,
        field_name="obstacles",
        expected_shape=(MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES),
        expected_dtype=jnp.float32,
    )
    _require_finite_array(obstacle_array, field_name="obstacles")
    host_obstacles = np.asarray(obstacle_array)

    active_values = host_obstacles[:, OBSTACLE_FEATURE_ACTIVE]
    if not bool(np.all(np.isin(active_values, (0.0, 1.0)))):
        raise ValueError("obstacles.active values must be exactly 0.0 or 1.0.")

    inactive_rows = active_values == 0.0
    if not bool(np.all(host_obstacles[inactive_rows] == 0.0)):
        raise ValueError("inactive obstacle rows must be entirely zero.")

    active_indices = np.flatnonzero(active_values == 1.0)
    for obstacle_index in active_indices:
        obstacle = host_obstacles[obstacle_index]
        obstacle_type = obstacle[OBSTACLE_FEATURE_TYPE]

        if obstacle_type == float(OBSTACLE_TYPE_PILLAR):
            if obstacle[OBSTACLE_FEATURE_RADIUS] <= 0.0:
                raise ValueError(
                    f"obstacles[{obstacle_index}] pillar radius must be greater than 0."
                )
            unused_wall_features = obstacle[
                [
                    OBSTACLE_FEATURE_WIDTH,
                    OBSTACLE_FEATURE_HEIGHT,
                    OBSTACLE_FEATURE_THETA,
                ]
            ]
            if not bool(np.all(unused_wall_features == 0.0)):
                raise ValueError(
                    f"obstacles[{obstacle_index}] pillar wall fields must be zero."
                )
        elif obstacle_type == float(OBSTACLE_TYPE_WALL):
            if obstacle[OBSTACLE_FEATURE_WIDTH] <= 0.0:
                raise ValueError(
                    f"obstacles[{obstacle_index}] wall width must be greater than 0."
                )
            if obstacle[OBSTACLE_FEATURE_HEIGHT] <= 0.0:
                raise ValueError(
                    f"obstacles[{obstacle_index}] wall height must be greater than 0."
                )
            if obstacle[OBSTACLE_FEATURE_RADIUS] != 0.0:
                raise ValueError(
                    f"obstacles[{obstacle_index}] wall radius must be zero."
                )
        else:
            raise ValueError(
                f"obstacles[{obstacle_index}].type must be "
                f"{OBSTACLE_TYPE_PILLAR} (pillar) or {OBSTACLE_TYPE_WALL} "
                f"(wall), not {obstacle_type}."
            )

    return obstacle_array


def _validate_agent_profile(profile: object) -> ResolvedAgentProfile:
    """Validate one fully resolved fixed-slot agent profile."""
    if type(profile) is not ResolvedAgentProfile:
        raise TypeError(
            "agent_profile must be a ResolvedAgentProfile, not "
            f"{type(profile).__name__}."
        )

    class_ids = _require_jax_array(
        profile.class_ids,
        field_name="agent_profile.class_ids",
        expected_shape=(MAX_AGENT_SLOTS,),
        expected_dtype=jnp.int32,
    )
    team_ids = _require_jax_array(
        profile.team_ids,
        field_name="agent_profile.team_ids",
        expected_shape=(MAX_AGENT_SLOTS,),
        expected_dtype=jnp.int32,
    )
    active_mask = _require_jax_array(
        profile.active_mask,
        field_name="agent_profile.active_mask",
        expected_shape=(MAX_AGENT_SLOTS,),
        expected_dtype=jnp.bool_,
    )

    float_profile_fields = (
        ("agent_radii", profile.agent_radii),
        ("base_movement_speeds", profile.base_movement_speeds),
        ("observation_radii", profile.observation_radii),
        ("basic_interaction_radii", profile.basic_interaction_radii),
        ("ultimate_interaction_radii", profile.ultimate_interaction_radii),
        ("max_health", profile.max_health),
    )
    for field_name, field_value in float_profile_fields:
        validated_value = _require_jax_array(
            field_value,
            field_name=f"agent_profile.{field_name}",
            expected_shape=(MAX_AGENT_SLOTS,),
            expected_dtype=jnp.float32,
        )
        _require_finite_array(validated_value, field_name=f"agent_profile.{field_name}")

    host_class_ids = np.asarray(class_ids)
    host_team_ids = np.asarray(team_ids)
    host_active_mask = np.asarray(active_mask)

    if not bool(
        np.all((host_class_ids >= NEUTRAL_CLASS_ID) & (host_class_ids < NUM_CLASSES))
    ):
        raise ValueError(
            "agent_profile.class_ids contains a value outside the class catalog."
        )
    if bool(np.any(host_active_mask & (host_class_ids == NEUTRAL_CLASS_ID))):
        raise ValueError(
            "active agent_profile.class_ids rows must use a non-neutral class."
        )
    if bool(np.any(~host_active_mask & (host_class_ids != NEUTRAL_CLASS_ID))):
        raise ValueError(
            "inactive agent_profile.class_ids rows must use the neutral class."
        )

    valid_team_ids = (NO_TEAM_ID, TEAM_A_ID, TEAM_B_ID)
    if not bool(np.all(np.isin(host_team_ids, valid_team_ids))):
        raise ValueError("agent_profile.team_ids contains an invalid team id.")

    expected_active_team_ids = np.concatenate(
        (
            np.full(MAX_AGENTS_PER_TEAM, TEAM_A_ID, dtype=np.int32),
            np.full(MAX_AGENTS_PER_TEAM, TEAM_B_ID, dtype=np.int32),
        )
    )
    if bool(np.any(host_active_mask & (host_team_ids != expected_active_team_ids))):
        raise ValueError(
            "active agent_profile.team_ids rows must match their fixed team block."
        )
    if bool(np.any(~host_active_mask & (host_team_ids != NO_TEAM_ID))):
        raise ValueError("inactive agent_profile.team_ids rows must use NO_TEAM_ID.")

    for team_name, team_active_mask in (
        ("team A", host_active_mask[:MAX_AGENTS_PER_TEAM]),
        ("team B", host_active_mask[MAX_AGENTS_PER_TEAM:]),
    ):
        if bool(np.any(np.diff(team_active_mask.astype(np.int8)) > 0)):
            raise ValueError(
                f"agent_profile.active_mask {team_name} rows must be a contiguous "
                "active prefix."
            )

    expected_catalog_fields = (
        ("agent_radii", get_body_radius_by_class_ids(class_ids)),
        (
            "base_movement_speeds",
            get_base_movement_speed_by_class_ids(class_ids),
        ),
        ("observation_radii", get_observation_radius_by_class_ids(class_ids)),
        (
            "basic_interaction_radii",
            get_basic_interaction_radius_by_class_ids(class_ids),
        ),
        (
            "ultimate_interaction_radii",
            get_ultimate_interaction_radius_by_class_ids(class_ids),
        ),
        ("max_health", get_max_health_by_class_ids(class_ids)),
    )
    for field_name, expected_values in expected_catalog_fields:
        actual_values = np.asarray(getattr(profile, field_name))
        if not bool(np.array_equal(actual_values, np.asarray(expected_values))):
            raise ValueError(
                f"agent_profile.{field_name} must match the resolved class catalog."
            )

    return profile


def _validate_agent_obstacle_clearance(
    positions: Array,
    *,
    obstacles: Array,
    profile: ResolvedAgentProfile,
) -> None:
    """Reject starts intersecting an obstacle under authoritative M4 geometry."""
    host_obstacles = np.asarray(obstacles)
    active_agent_indices = np.flatnonzero(np.asarray(profile.active_mask))
    max_active_radius = (
        float(np.max(np.asarray(profile.agent_radii)[active_agent_indices]))
        if active_agent_indices.size
        else 0.0
    )

    for obstacle_index in np.flatnonzero(
        host_obstacles[:, OBSTACLE_FEATURE_ACTIVE] == 1.0
    ):
        obstacle = host_obstacles[obstacle_index]
        obstacle_type = obstacle[OBSTACLE_FEATURE_TYPE]
        if obstacle_type == float(OBSTACLE_TYPE_PILLAR):
            x_extent = float(obstacle[OBSTACLE_FEATURE_RADIUS]) + max_active_radius
            y_extent = x_extent
        else:
            half_width = float(obstacle[OBSTACLE_FEATURE_WIDTH]) / 2.0
            half_height = float(obstacle[OBSTACLE_FEATURE_HEIGHT]) / 2.0
            # The L1 bound is conservative for every rotation angle and avoids
            # host/JAX range-reduction differences for very large finite theta.
            x_extent = half_width + half_height + max_active_radius
            y_extent = x_extent

        x_limit = abs(float(obstacle[OBSTACLE_FEATURE_X])) + x_extent
        y_limit = abs(float(obstacle[OBSTACLE_FEATURE_Y])) + y_extent
        if x_limit > _FLOAT32_MAX or y_limit > _FLOAT32_MAX:
            raise ValueError(
                f"obstacles[{obstacle_index}] geometry plus active agent radius "
                "must remain representable by float32."
            )

        for agent_index in active_agent_indices:
            overlaps = disc_overlaps_obstacle(
                positions[agent_index],
                profile.agent_radii[agent_index],
                obstacles[obstacle_index],
            )
            if bool(overlaps):
                obstacle_name = (
                    "pillar" if obstacle_type == float(OBSTACLE_TYPE_PILLAR) else "wall"
                )
                raise ValueError(
                    f"initial_agent_positions[{agent_index}] overlaps active "
                    f"{obstacle_name} obstacles[{obstacle_index}]."
                )


def _validate_initial_agent_positions(
    positions: object,
    *,
    map_width: float,
    map_height: float,
    obstacles: Array,
    profile: ResolvedAgentProfile,
) -> None:
    """Validate fixed reset centers against roster and M4 geometry contracts."""
    position_array = _require_jax_array(
        positions,
        field_name="initial_agent_positions",
        expected_shape=(MAX_AGENT_SLOTS, ENVIRONMENT_DIMENSIONS),
        expected_dtype=jnp.float32,
    )
    _require_finite_array(position_array, field_name="initial_agent_positions")

    host_positions = np.asarray(position_array)
    host_active_mask = np.asarray(profile.active_mask)
    host_radii = np.asarray(profile.agent_radii)

    if not bool(np.all(host_positions[~host_active_mask] == 0.0)):
        raise ValueError("inactive initial_agent_positions rows must be exactly zero.")

    active_indices = np.flatnonzero(host_active_mask)
    for agent_index in active_indices:
        center = host_positions[agent_index]
        radius = float(host_radii[agent_index])
        if not (
            radius <= float(center[0]) <= map_width - radius
            and radius <= float(center[1]) <= map_height - radius
        ):
            raise ValueError(
                f"initial_agent_positions[{agent_index}] places the active body "
                "outside radius-adjusted map bounds."
            )

    _validate_agent_obstacle_clearance(
        position_array,
        obstacles=obstacles,
        profile=profile,
    )

    for pair_position, agent_a_index in enumerate(active_indices):
        for agent_b_index in active_indices[pair_position + 1 :]:
            center_delta = host_positions[agent_a_index] - host_positions[agent_b_index]
            center_distance = math.hypot(float(center_delta[0]), float(center_delta[1]))
            minimum_distance = float(
                host_radii[agent_a_index] + host_radii[agent_b_index]
            )
            if center_distance < minimum_distance:
                raise ValueError(
                    "initial_agent_positions active bodies overlap at slots "
                    f"{agent_a_index} and {agent_b_index}."
                )


def validate_env_config(config: EnvConfig) -> None:
    """Validate a resolved episode configuration before it reaches JAX core code.

    The validator is deliberately host-only. Official builders call it before
    passing ``config`` to :func:`marl_battlegrounds.core.env.reset`; traced reset
    and step code may therefore consume a compact, already-resolved contract.

    Raises:
        TypeError: A field has the wrong Python/JAX type or array dtype.
        ValueError: A field violates a shape, domain, padding, or geometry rule.
    """
    if type(config) is not EnvConfig:
        raise TypeError(f"config must be an EnvConfig, not {type(config).__name__}.")

    if type(config.max_steps) is not int:
        raise TypeError(
            f"max_steps must be an int, not {type(config.max_steps).__name__}."
        )
    if config.max_steps <= 0:
        raise ValueError(f"max_steps must be greater than 0, not {config.max_steps}.")
    if config.max_steps > _INT32_MAX:
        raise ValueError(
            f"max_steps must be at most {_INT32_MAX}, not {config.max_steps}."
        )

    for field_name, dimension in (
        ("map_width", config.map_width),
        ("map_height", config.map_height),
    ):
        if type(dimension) is not float:
            raise TypeError(
                f"{field_name} must be a float, not {type(dimension).__name__}."
            )
        if not math.isfinite(dimension):
            raise ValueError(f"{field_name} must be finite, not {dimension}.")
        if dimension <= 0.0:
            raise ValueError(f"{field_name} must be greater than 0, not {dimension}.")
        if dimension > _FLOAT32_MAX:
            raise ValueError(
                f"{field_name} must be at most {_FLOAT32_MAX}, not {dimension}."
            )

    if type(config.ordinary_movement_distance_scale) is not float:
        raise TypeError(
            "ordinary_movement_distance_scale must be a float, not "
            f"{type(config.ordinary_movement_distance_scale).__name__}."
        )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        execution_movement_scale = np.float32(config.ordinary_movement_distance_scale)
    if not bool(np.isfinite(execution_movement_scale)):
        raise ValueError(
            "ordinary_movement_distance_scale must remain finite after "
            "conversion to float32."
        )
    if not 0.0 < execution_movement_scale <= 1.0:
        raise ValueError(
            "ordinary_movement_distance_scale must remain in (0.0, 1.0] "
            "after conversion to float32."
        )

    obstacles = _validate_obstacles(config.obstacles)
    profile = _validate_agent_profile(config.agent_profile)
    _validate_initial_agent_positions(
        config.initial_agent_positions,
        map_width=config.map_width,
        map_height=config.map_height,
        obstacles=obstacles,
        profile=profile,
    )
