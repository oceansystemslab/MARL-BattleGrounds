"""Host-facing configuration, fixed-slot profile, and state validation."""

import math

import jax.numpy as jnp
import numpy as np
from jax import Array

from marl_battlegrounds.core.combat import (
    HUNTER_BASIC_SLOW_DURATION_TICKS,
    HUNTER_TRAP_STUN_DURATION_TICKS,
    MAGE_BURST_DAMAGE_DURATION_TICKS,
    PRIEST_HEAL_SPEED_FLOOR_DURATION_TICKS,
    ROGUE_POISON_ANTI_HEAL_DURATION_TICKS,
    ROGUE_POISON_SLOW_DURATION_TICKS,
    ROGUE_POISON_STUN_DURATION_TICKS,
    WARRIOR_CHARGE_SLOW_DURATION_TICKS,
    WARRIOR_CHARGE_STUN_DURATION_TICKS,
    get_base_movement_speed_by_class_ids,
    get_basic_interaction_radius_by_class_ids,
    get_body_radius_by_class_ids,
    get_max_health_by_class_ids,
    get_observation_radius_by_class_ids,
    get_ultimate_cooldown_by_class_ids,
    get_ultimate_interaction_radius_by_class_ids,
)
from marl_battlegrounds.core.geometry import (
    GEOMETRY_TOLERANCE,
    disc_overlaps_obstacle,
)
from marl_battlegrounds.core.types import (
    ENVIRONMENT_DIMENSIONS,
    MAGE_CLASS_ID,
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    MAX_OBSTACLE_SLOTS,
    NEUTRAL_CLASS_ID,
    NO_TEAM_ID,
    NUM_CLASSES,
    NUM_MOVE_ACTIONS,
    NUM_SLOW_CHANNELS,
    NUM_STUN_CHANNELS,
    NUM_TARGET_ACTIONS,
    NUM_TEAMS,
    NUM_ULTIMATE_ACTIONS,
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
    EnvState,
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
    agent_radii: Array,
    field_name: str,
) -> None:
    """Reject configured bodies intersecting authoritative M4 obstacles."""
    host_obstacles = np.asarray(obstacles)
    host_radii = np.asarray(agent_radii)
    max_body_radius = float(np.max(host_radii)) if host_radii.size else 0.0

    for obstacle_index in np.flatnonzero(
        host_obstacles[:, OBSTACLE_FEATURE_ACTIVE] == 1.0
    ):
        obstacle = host_obstacles[obstacle_index]
        obstacle_type = obstacle[OBSTACLE_FEATURE_TYPE]
        if obstacle_type == float(OBSTACLE_TYPE_PILLAR):
            x_extent = float(obstacle[OBSTACLE_FEATURE_RADIUS]) + max_body_radius
            y_extent = x_extent
        else:
            half_width = float(obstacle[OBSTACLE_FEATURE_WIDTH]) / 2.0
            half_height = float(obstacle[OBSTACLE_FEATURE_HEIGHT]) / 2.0
            # The L1 bound is conservative for every rotation angle and avoids
            # host/JAX range-reduction differences for very large finite theta.
            x_extent = half_width + half_height + max_body_radius
            y_extent = x_extent

        x_limit = abs(float(obstacle[OBSTACLE_FEATURE_X])) + x_extent
        y_limit = abs(float(obstacle[OBSTACLE_FEATURE_Y])) + y_extent
        if x_limit > _FLOAT32_MAX or y_limit > _FLOAT32_MAX:
            raise ValueError(
                f"obstacles[{obstacle_index}] geometry plus configured body radius "
                "must remain representable by float32."
            )

        for agent_index in range(positions.shape[0]):
            overlaps = disc_overlaps_obstacle(
                positions[agent_index],
                agent_radii[agent_index],
                obstacles[obstacle_index],
            )
            if bool(overlaps):
                team_index, team_local_index = divmod(agent_index, MAX_AGENTS_PER_TEAM)
                obstacle_name = (
                    "pillar" if obstacle_type == float(OBSTACLE_TYPE_PILLAR) else "wall"
                )
                raise ValueError(
                    f"{field_name}[{team_index}, {team_local_index}] overlaps active "
                    f"{obstacle_name} obstacles[{obstacle_index}]."
                )


def _validate_team_spawn_pad_positions(
    positions: object,
    *,
    map_width: float,
    map_height: float,
    obstacles: Array,
    profile: ResolvedAgentProfile,
) -> None:
    """Validate all spawn pads and the complete fallback-body formation."""
    position_array = _require_jax_array(
        positions,
        field_name="team_spawn_pad_positions",
        expected_shape=(NUM_TEAMS, MAX_AGENTS_PER_TEAM, ENVIRONMENT_DIMENSIONS),
        expected_dtype=jnp.float32,
    )
    _require_finite_array(position_array, field_name="team_spawn_pad_positions")

    host_positions = np.asarray(position_array).reshape(
        MAX_AGENT_SLOTS, ENVIRONMENT_DIMENSIONS
    )
    host_active_mask = np.asarray(profile.active_mask)
    host_radii = np.asarray(profile.agent_radii)

    # Every configured pad remains a real lifecycle location, including pad rows
    # whose corresponding roster slot is inactive. Validate each pad for the
    # largest body that can belong to its configured team.
    fallback_radii_by_team = np.zeros((NUM_TEAMS,), dtype=np.float32)
    for team_index in range(NUM_TEAMS):
        team_start = team_index * MAX_AGENTS_PER_TEAM
        team_stop = team_start + MAX_AGENTS_PER_TEAM
        team_active_radii = host_radii[team_start:team_stop][
            host_active_mask[team_start:team_stop]
        ]
        if team_active_radii.size:
            fallback_radii_by_team[team_index] = np.max(team_active_radii)
    pad_body_radii = np.repeat(fallback_radii_by_team, MAX_AGENTS_PER_TEAM)

    for pad_index in range(MAX_AGENT_SLOTS):
        center = host_positions[pad_index]
        radius = float(pad_body_radii[pad_index])
        team_index, team_local_index = divmod(pad_index, MAX_AGENTS_PER_TEAM)
        if not (
            radius <= float(center[0]) <= map_width - radius
            and radius <= float(center[1]) <= map_height - radius
        ):
            raise ValueError(
                "team_spawn_pad_positions"
                f"[{team_index}, {team_local_index}] places a configured same-team "
                "body outside radius-adjusted map bounds."
            )

    _validate_agent_obstacle_clearance(
        position_array.reshape(MAX_AGENT_SLOTS, ENVIRONMENT_DIMENSIONS),
        obstacles=obstacles,
        agent_radii=jnp.asarray(pad_body_radii, dtype=jnp.float32),
        field_name="team_spawn_pad_positions",
    )

    for pad_a_index in range(MAX_AGENT_SLOTS):
        for pad_b_index in range(pad_a_index + 1, MAX_AGENT_SLOTS):
            center_delta = host_positions[pad_a_index] - host_positions[pad_b_index]
            center_distance = math.hypot(float(center_delta[0]), float(center_delta[1]))
            minimum_distance = float(
                pad_body_radii[pad_a_index] + pad_body_radii[pad_b_index]
            )
            if center_distance < minimum_distance:
                team_a_index, team_a_local_index = divmod(
                    pad_a_index, MAX_AGENTS_PER_TEAM
                )
                team_b_index, team_b_local_index = divmod(
                    pad_b_index, MAX_AGENTS_PER_TEAM
                )
                raise ValueError(
                    "team_spawn_pad_positions fallback bodies overlap at pads "
                    f"[{team_a_index}, {team_a_local_index}] and "
                    f"[{team_b_index}, {team_b_local_index}]."
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

    if type(config.spawn_shield_duration_steps) is not int:
        raise TypeError(
            "spawn_shield_duration_steps must be an int, not "
            f"{type(config.spawn_shield_duration_steps).__name__}."
        )
    if config.spawn_shield_duration_steps < 0:
        raise ValueError(
            "spawn_shield_duration_steps must be nonnegative, not "
            f"{config.spawn_shield_duration_steps}."
        )
    if config.spawn_shield_duration_steps > _INT32_MAX:
        raise ValueError(
            f"spawn_shield_duration_steps must be at most {_INT32_MAX}, not "
            f"{config.spawn_shield_duration_steps}."
        )

    if type(config.spawn_shield_movement_speed) is not float:
        raise TypeError(
            "spawn_shield_movement_speed must be a float, not "
            f"{type(config.spawn_shield_movement_speed).__name__}."
        )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        execution_spawn_shield_speed = np.float32(config.spawn_shield_movement_speed)
    if not bool(np.isfinite(execution_spawn_shield_speed)):
        raise ValueError(
            "spawn_shield_movement_speed must remain finite after conversion "
            "to float32."
        )
    if execution_spawn_shield_speed <= 0.0:
        raise ValueError(
            "spawn_shield_movement_speed must remain greater than 0 after "
            "conversion to float32."
        )

    obstacles = _validate_obstacles(config.obstacles)
    profile = _validate_agent_profile(config.agent_profile)
    _validate_team_spawn_pad_positions(
        config.team_spawn_pad_positions,
        map_width=config.map_width,
        map_height=config.map_height,
        obstacles=obstacles,
        profile=profile,
    )


def _validate_state_positions(
    positions: Array,
    *,
    config: EnvConfig,
    active_mask: np.ndarray,
) -> None:
    """Validate runtime positions against hard static-geometry rules."""
    host_positions = np.asarray(positions)
    host_radii = np.asarray(config.agent_profile.agent_radii)

    if not bool(np.all(host_positions[~active_mask] == 0.0)):
        raise ValueError("inactive agent_positions rows must be exactly zero.")

    for agent_index in np.flatnonzero(active_mask):
        center = host_positions[agent_index]
        radius = float(host_radii[agent_index])
        if not (
            radius - GEOMETRY_TOLERANCE
            <= float(center[0])
            <= config.map_width - radius + GEOMETRY_TOLERANCE
            and radius - GEOMETRY_TOLERANCE
            <= float(center[1])
            <= config.map_height - radius + GEOMETRY_TOLERANCE
        ):
            raise ValueError(
                f"agent_positions[{agent_index}] places the active body outside "
                "radius-adjusted map bounds."
            )

        for obstacle_index in np.flatnonzero(
            np.asarray(config.obstacles)[:, OBSTACLE_FEATURE_ACTIVE] == 1.0
        ):
            if bool(
                disc_overlaps_obstacle(
                    positions[agent_index],
                    config.agent_profile.agent_radii[agent_index],
                    config.obstacles[obstacle_index],
                )
            ):
                raise ValueError(
                    f"agent_positions[{agent_index}] overlaps active "
                    f"obstacles[{obstacle_index}]."
                )

    # Agent-agent projection deliberately has a fixed-pass residual contract.
    # A strict pairwise-separation check would reject legitimate crowded or
    # boundary-pinned simulator outputs, so runtime state validation enforces
    # only the kernel's hard static-geometry guarantees here.


def _validate_scenario_living_body_clearance(
    positions: Array,
    *,
    config: EnvConfig,
    alive_mask: Array,
) -> None:
    """Reject positive-area overlap between living bodies in a curated start."""
    host_positions = np.asarray(positions)
    host_radii = np.asarray(config.agent_profile.agent_radii)
    active_and_alive = np.asarray(config.agent_profile.active_mask) & np.asarray(
        alive_mask
    )
    living_indices = np.flatnonzero(active_and_alive)

    for pair_position, agent_a_index in enumerate(living_indices):
        for agent_b_index in living_indices[pair_position + 1 :]:
            center_delta = host_positions[agent_a_index] - host_positions[agent_b_index]
            center_distance = math.hypot(float(center_delta[0]), float(center_delta[1]))
            minimum_distance = float(
                host_radii[agent_a_index] + host_radii[agent_b_index]
            )
            if center_distance < minimum_distance:
                raise ValueError(
                    "curated scenario living bodies overlap at slots "
                    f"{agent_a_index} and {agent_b_index}."
                )


def _validate_nonnegative_bounded_integer_array(
    values: Array,
    *,
    field_name: str,
    upper_bounds: np.ndarray | int,
) -> np.ndarray:
    """Return host integer values after enforcing a closed duration domain."""
    host_values = np.asarray(values)
    if bool(np.any(host_values < 0)):
        raise ValueError(f"{field_name} must contain only nonnegative values.")
    if bool(np.any(host_values > upper_bounds)):
        raise ValueError(f"{field_name} exceeds its catalog maximum.")
    return host_values


def _validate_previous_action_domain(
    values: Array,
    *,
    field_name: str,
    category_count: int,
) -> np.ndarray:
    """Return host action history after enforcing one categorical domain."""
    host_values = np.asarray(values)
    if bool(np.any((host_values < 0) | (host_values >= category_count))):
        raise ValueError(
            f"{field_name} must contain categories in [0, {category_count})."
        )
    return host_values


def validate_env_state(config: EnvConfig, state: EnvState) -> None:
    """Validate one runtime-representable state at a host-owned boundary.

    ``config`` must already have passed :func:`validate_env_config`. This
    function is deliberately host-only: replay readers and tooling call it
    before consuming simulator snapshots. It must never run from ``reset``,
    ``step``, ``jax.jit``, ``jax.vmap``, or ``jax.lax.scan``.

    Dead agents retain valid cooldown and accepted-action history. Their health
    and transient statuses are canonical zeros. This boundary cannot infer
    collision provenance from one snapshot, so it accepts living-body residuals
    emitted by the fixed-pass solver. Curated starts must additionally pass
    :func:`validate_scenario_initial_state`.

    Raises:
        TypeError: The container, array storage, or dtype is invalid.
        ValueError: A shape, domain, lifecycle, padding, or geometry invariant
            is invalid.
    """
    if type(config) is not EnvConfig:
        raise TypeError(f"config must be an EnvConfig, not {type(config).__name__}.")
    if type(state) is not EnvState:
        raise TypeError(f"state must be an EnvState, not {type(state).__name__}.")

    step_count = _require_jax_array(
        state.step_count,
        field_name="step_count",
        expected_shape=(),
        expected_dtype=jnp.int32,
    )
    positions = _require_jax_array(
        state.agent_positions,
        field_name="agent_positions",
        expected_shape=(MAX_AGENT_SLOTS, ENVIRONMENT_DIMENSIONS),
        expected_dtype=jnp.float32,
    )
    alive_mask = _require_jax_array(
        state.alive_mask,
        field_name="alive_mask",
        expected_shape=(MAX_AGENT_SLOTS,),
        expected_dtype=jnp.bool_,
    )
    current_health = _require_jax_array(
        state.current_health,
        field_name="current_health",
        expected_shape=(MAX_AGENT_SLOTS,),
        expected_dtype=jnp.float32,
    )
    ultimate_cooldowns = _require_jax_array(
        state.ultimate_cooldowns,
        field_name="ultimate_cooldowns",
        expected_shape=(MAX_AGENT_SLOTS,),
        expected_dtype=jnp.int32,
    )
    slow_durations = _require_jax_array(
        state.slow_durations,
        field_name="slow_durations",
        expected_shape=(MAX_AGENT_SLOTS, NUM_SLOW_CHANNELS),
        expected_dtype=jnp.int32,
    )
    stun_durations = _require_jax_array(
        state.stun_durations,
        field_name="stun_durations",
        expected_shape=(MAX_AGENT_SLOTS, NUM_STUN_CHANNELS),
        expected_dtype=jnp.int32,
    )
    rogue_anti_heal_durations = _require_jax_array(
        state.rogue_poison_anti_heal_durations,
        field_name="rogue_poison_anti_heal_durations",
        expected_shape=(MAX_AGENT_SLOTS,),
        expected_dtype=jnp.int32,
    )
    mage_burst_durations = _require_jax_array(
        state.mage_burst_damage_amplification_durations,
        field_name="mage_burst_damage_amplification_durations",
        expected_shape=(MAX_AGENT_SLOTS,),
        expected_dtype=jnp.int32,
    )
    priest_freedom_durations = _require_jax_array(
        state.priest_blessing_of_freedom_slow_floor_durations,
        field_name="priest_blessing_of_freedom_slow_floor_durations",
        expected_shape=(MAX_AGENT_SLOTS,),
        expected_dtype=jnp.int32,
    )
    spawn_shield_durations = _require_jax_array(
        state.spawn_shield_durations,
        field_name="spawn_shield_durations",
        expected_shape=(MAX_AGENT_SLOTS,),
        expected_dtype=jnp.int32,
    )
    previous_move_actions = _require_jax_array(
        state.previous_timestep_move_actions,
        field_name="previous_timestep_move_actions",
        expected_shape=(MAX_AGENT_SLOTS,),
        expected_dtype=jnp.int32,
    )
    previous_target_actions = _require_jax_array(
        state.previous_timestep_select_target_actions,
        field_name="previous_timestep_select_target_actions",
        expected_shape=(MAX_AGENT_SLOTS,),
        expected_dtype=jnp.int32,
    )
    previous_ultimate_actions = _require_jax_array(
        state.previous_timestep_use_ultimate_actions,
        field_name="previous_timestep_use_ultimate_actions",
        expected_shape=(MAX_AGENT_SLOTS,),
        expected_dtype=jnp.int32,
    )
    has_previous_action = _require_jax_array(
        state.has_previous_timestep_joint_action,
        field_name="has_previous_timestep_joint_action",
        expected_shape=(),
        expected_dtype=jnp.bool_,
    )

    _require_finite_array(positions, field_name="agent_positions")
    _require_finite_array(current_health, field_name="current_health")

    if int(np.asarray(step_count)) < 0:
        raise ValueError("step_count must be nonnegative.")

    configured_active = np.asarray(config.agent_profile.active_mask)
    host_alive = np.asarray(alive_mask)
    if bool(np.any(host_alive & ~configured_active)):
        raise ValueError("alive_mask may be true only for configured active slots.")

    active_and_alive = configured_active & host_alive
    active_and_dead = configured_active & ~host_alive
    inactive = ~configured_active

    host_health = np.asarray(current_health)
    host_max_health = np.asarray(config.agent_profile.max_health)
    if bool(np.any(host_health[active_and_alive] <= 0.0)):
        raise ValueError("active living current_health must be strictly positive.")
    if bool(np.any(host_health[active_and_alive] > host_max_health[active_and_alive])):
        raise ValueError("active living current_health must not exceed max_health.")
    if bool(np.any(host_health[active_and_dead] != 0.0)):
        raise ValueError("active dead current_health must be exactly zero.")
    if bool(np.any(host_health[inactive] != 0.0)):
        raise ValueError("inactive current_health rows must be exactly zero.")

    raw_host_cooldowns = np.asarray(ultimate_cooldowns)
    if bool(np.any(raw_host_cooldowns[inactive] != 0)):
        raise ValueError("inactive ultimate_cooldowns rows must be exactly zero.")
    _validate_nonnegative_bounded_integer_array(
        ultimate_cooldowns,
        field_name="ultimate_cooldowns",
        upper_bounds=np.asarray(
            get_ultimate_cooldown_by_class_ids(config.agent_profile.class_ids)
        ),
    )
    slow_maxima = np.asarray(
        (
            WARRIOR_CHARGE_SLOW_DURATION_TICKS,
            HUNTER_BASIC_SLOW_DURATION_TICKS,
            ROGUE_POISON_SLOW_DURATION_TICKS,
        ),
        dtype=np.int32,
    )
    host_slow_durations = _validate_nonnegative_bounded_integer_array(
        slow_durations,
        field_name="slow_durations",
        upper_bounds=slow_maxima[None, :],
    )
    stun_maxima = np.asarray(
        (
            WARRIOR_CHARGE_STUN_DURATION_TICKS,
            HUNTER_TRAP_STUN_DURATION_TICKS,
            ROGUE_POISON_STUN_DURATION_TICKS,
        ),
        dtype=np.int32,
    )
    host_stun_durations = _validate_nonnegative_bounded_integer_array(
        stun_durations,
        field_name="stun_durations",
        upper_bounds=stun_maxima[None, :],
    )
    host_rogue_anti_heal_durations = _validate_nonnegative_bounded_integer_array(
        rogue_anti_heal_durations,
        field_name="rogue_poison_anti_heal_durations",
        upper_bounds=ROGUE_POISON_ANTI_HEAL_DURATION_TICKS,
    )
    host_mage_burst_durations = _validate_nonnegative_bounded_integer_array(
        mage_burst_durations,
        field_name="mage_burst_damage_amplification_durations",
        upper_bounds=MAGE_BURST_DAMAGE_DURATION_TICKS,
    )
    host_class_ids = np.asarray(config.agent_profile.class_ids)
    if bool(np.any(host_mage_burst_durations[host_class_ids != MAGE_CLASS_ID] != 0)):
        raise ValueError(
            "mage_burst_damage_amplification_durations may be positive only "
            "for configured Mage slots."
        )
    host_priest_freedom_durations = _validate_nonnegative_bounded_integer_array(
        priest_freedom_durations,
        field_name="priest_blessing_of_freedom_slow_floor_durations",
        upper_bounds=PRIEST_HEAL_SPEED_FLOOR_DURATION_TICKS,
    )

    transient_status_families = (
        ("slow_durations", host_slow_durations),
        ("stun_durations", host_stun_durations),
        (
            "rogue_poison_anti_heal_durations",
            host_rogue_anti_heal_durations,
        ),
        (
            "mage_burst_damage_amplification_durations",
            host_mage_burst_durations,
        ),
        (
            "priest_blessing_of_freedom_slow_floor_durations",
            host_priest_freedom_durations,
        ),
    )
    for field_name, host_values in transient_status_families:
        if bool(np.any(host_values[active_and_dead] != 0)):
            raise ValueError(f"active dead {field_name} rows must be exactly zero.")
        if bool(np.any(host_values[inactive] != 0)):
            raise ValueError(f"inactive {field_name} rows must be exactly zero.")

    host_spawn_shield_durations = np.asarray(spawn_shield_durations)
    if bool(np.any(host_spawn_shield_durations < 0)):
        raise ValueError("spawn_shield_durations must contain only nonnegative values.")
    if bool(np.any(host_spawn_shield_durations > config.spawn_shield_duration_steps)):
        raise ValueError(
            "spawn_shield_durations must not exceed spawn_shield_duration_steps."
        )
    if bool(np.any(host_spawn_shield_durations[active_and_dead] != 0)):
        raise ValueError(
            "active dead spawn_shield_durations rows must be exactly zero."
        )
    if bool(np.any(host_spawn_shield_durations[inactive] != 0)):
        raise ValueError("inactive spawn_shield_durations rows must be exactly zero.")
    if bool(
        np.any(
            (host_spawn_shield_durations > 0) & np.any(host_stun_durations > 0, axis=-1)
        )
    ):
        raise ValueError(
            "spawn_shield_durations and stun_durations cannot both be positive "
            "for the same slot."
        )

    host_previous_move_actions = _validate_previous_action_domain(
        previous_move_actions,
        field_name="previous_timestep_move_actions",
        category_count=NUM_MOVE_ACTIONS,
    )
    host_previous_target_actions = _validate_previous_action_domain(
        previous_target_actions,
        field_name="previous_timestep_select_target_actions",
        category_count=NUM_TARGET_ACTIONS,
    )
    host_previous_ultimate_actions = _validate_previous_action_domain(
        previous_ultimate_actions,
        field_name="previous_timestep_use_ultimate_actions",
        category_count=NUM_ULTIMATE_ACTIONS,
    )
    previous_action_families = (
        ("previous_timestep_move_actions", host_previous_move_actions),
        ("previous_timestep_select_target_actions", host_previous_target_actions),
        ("previous_timestep_use_ultimate_actions", host_previous_ultimate_actions),
    )
    for field_name, host_values in previous_action_families:
        if bool(np.any(host_values[inactive] != 0)):
            raise ValueError(f"inactive {field_name} rows must be exactly zero.")
        if not bool(np.asarray(has_previous_action)) and bool(np.any(host_values != 0)):
            raise ValueError(
                f"{field_name} must be zero when "
                "has_previous_timestep_joint_action is false."
            )

    _validate_state_positions(
        positions,
        config=config,
        active_mask=configured_active,
    )


def validate_scenario_initial_state(config: EnvConfig, state: EnvState) -> None:
    """Validate a curated start and its official transition provenance.

    Runtime snapshots may contain deterministic fixed-pass collision residuals,
    but authored scenario starts have no such transition provenance. Tangency is
    legal, and preserved corpses remain nonphysical for pairwise clearance.
    Authored previous-action history must also be compatible with the current
    spawn-shield counters; general runtime validation deliberately remains
    permissive so externally supplied mask provenance can still be inspected.
    """
    validate_env_state(config, state)

    if bool(np.asarray(state.has_previous_timestep_joint_action)):
        shielded_slots = np.asarray(state.spawn_shield_durations) > 0
        previous_target_actions = np.asarray(
            state.previous_timestep_select_target_actions
        )
        previous_ultimate_actions = np.asarray(
            state.previous_timestep_use_ultimate_actions
        )

        shielded_source_has_combat_history = np.logical_and(
            shielded_slots,
            np.logical_or(
                previous_target_actions != 0,
                previous_ultimate_actions != 0,
            ),
        )
        if bool(np.any(shielded_source_has_combat_history)):
            source_slot = int(np.flatnonzero(shielded_source_has_combat_history)[0])
            raise ValueError(
                "A shielded scenario source must have target-none and "
                f"no-Ultimate in previous action history; slot {source_slot} "
                "has nonneutral combat history."
            )

        for source_slot, target_action in enumerate(previous_target_actions):
            if target_action == 0:
                continue

            source_team_start = (
                source_slot // MAX_AGENTS_PER_TEAM
            ) * MAX_AGENTS_PER_TEAM
            candidate_index = int(target_action) - 1
            if candidate_index < MAX_AGENTS_PER_TEAM:
                recipient_slot = source_team_start + candidate_index
            else:
                opposing_team_start = (
                    MAX_AGENTS_PER_TEAM if source_team_start == 0 else 0
                )
                recipient_slot = (
                    opposing_team_start + candidate_index - MAX_AGENTS_PER_TEAM
                )

            if shielded_slots[recipient_slot]:
                raise ValueError(
                    "Scenario previous action history cannot target a currently "
                    f"shielded recipient; source slot {source_slot} resolves to "
                    f"recipient slot {recipient_slot}."
                )

    _validate_scenario_living_body_clearance(
        state.agent_positions,
        config=config,
        alive_mask=state.alive_mask,
    )
