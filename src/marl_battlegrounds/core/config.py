"""Resolve fixed-slot profiles and validate concrete Core inputs on the host.

resolve_agent_profile performs a pure JAX catalog lookup with team padding.
maximum_team_deathmatch_score_increment and maximum_team_deathmatch_score_threshold
own the Team Deathmatch score bounds, which depend on the Red Zone depth.
The validators check complete configurations, product movement calibration,
runtime states and authored starts. Runtime states allow the geometry solver's
remaining body overlap; authored living starts require strict clearance.

Validation reads array values through NumPy or scalar conversion and may copy
or synchronize device data. Keep it at configuration, loading and tooling
boundaries, outside compiled reset/step loops. These checks do not change inputs
or create policies, episodes, logs or replay files."""

import math
from typing import Final

import jax.numpy as jnp
import numpy as np
from jax import Array

from marl_battlegrounds.core import combat
from marl_battlegrounds.core.axis_mappings import target_action_to_global_slot
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
    get_ooc_delay_steps_by_class_ids,
    get_ooc_health_regen_fraction_per_step_by_class_ids,
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
    TASK_MODE_CTF,
    TASK_MODE_KOTH,
    TASK_MODE_NEUTRAL,
    TASK_MODE_TDM,
    TEAM_A_ID,
    TEAM_B_ID,
    TEAM_DEATHMATCH_POINTS_PER_DEATH,
    TEAM_DEATHMATCH_POINTS_PER_RED_ZONE_DEATH,
    EnvConfig,
    EnvState,
    ResolvedAgentProfile,
)

_INT32_MAX = int(np.iinfo(np.int32).max)
_FLOAT32_MAX = float(np.finfo(np.float32).max)
_FLOAT32_SMALLEST_NORMAL = np.finfo(np.float32).tiny
_MAX_EXACT_FLOAT32_INTEGER = 2**24

# Product sessions use one movement calibration so researchers cannot change
# policy-relevant dynamics from a presentation surface. Generic ``EnvConfig``
# construction deliberately retains the full validated ``(0.0, 1.0]`` domain
# for tests and explicitly experimental callers.
CANONICAL_PRODUCT_MOVEMENT_SCALE: Final = 1.0


def maximum_team_deathmatch_score_increment(red_zone_depth: float) -> int:
    """Return the most Team Deathmatch points one team can gain in one step.

    Parameters
    ----------
    red_zone_depth : float
        A validated Python float depth (EnvConfig.team_deathmatch_red_zone_depth),
        never a traced JAX value.

    Returns
    -------
    int
        5 at depth 0.0 (five enemies die, one point each). 10 at a positive
        depth (five enemies die inside their own Red Zone, two points each).

    Notes
    -----
    Host only. The one owner of this bound; the score threshold and state
    checks derive from it.
    """
    points = (
        TEAM_DEATHMATCH_POINTS_PER_RED_ZONE_DEATH
        if red_zone_depth > 0.0
        else TEAM_DEATHMATCH_POINTS_PER_DEATH
    )
    return MAX_AGENTS_PER_TEAM * points


def maximum_team_deathmatch_score_threshold(red_zone_depth: float) -> int:
    """Return the largest valid Team Deathmatch score threshold for a depth.

    Parameters
    ----------
    red_zone_depth : float
        A validated Python float depth, never a traced JAX value.

    Returns
    -------
    int
        2**24 - (increment - 1), where increment is
        maximum_team_deathmatch_score_increment(red_zone_depth): 16,777,212 at
        depth 0.0 and 16,777,207 at a positive depth. A score can pass the
        threshold by at most increment - 1 in its final step, so every reachable
        score stays at most 2**24 and exact in float32 context features.

    Notes
    -----
    Host only.
    """
    return _MAX_EXACT_FLOAT32_INTEGER - (
        maximum_team_deathmatch_score_increment(red_zone_depth) - 1
    )


def resolve_agent_profile(
    requested_class_ids: Array, team_sizes: Array
) -> ResolvedAgentProfile:
    """Resolve requested team rosters into fixed-slot class and capability arrays.

    Parameters
    ----------
    requested_class_ids : jax.Array
        Int32 (10,) requested IDs in Team A then Team B order. Active slots
        use classes 1..5; unused slots are replaced by neutral class zero.
    team_sizes : jax.Array
        Int32 (2,) sizes in 0..5 for Team A and Team B. Each size selects a
        prefix within that team's five fixed slots.

    Returns
    -------
    ResolvedAgentProfile
        Ten-slot arrays with bool membership, int32 classes/team IDs/recovery
        delays, and float32 catalog capabilities. Unused slots have neutral
        class and team IDs and zero capabilities.

    Notes
    -----
    This numerical helper is pure JAX and may be mapped across games. It does
    not validate roster inputs or the resulting configuration. Host builders
    must call validate_env_config before reset; catalog indexing is not an
    input-validation mechanism. No randomness or position sampling occurs.
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
    ooc_delay_steps = get_ooc_delay_steps_by_class_ids(class_ids)
    ooc_health_regen_fraction_per_step = (
        get_ooc_health_regen_fraction_per_step_by_class_ids(class_ids)
    )

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
        ooc_delay_steps,
        ooc_health_regen_fraction_per_step,
    )


def _require_jax_array(
    value: object,
    *,
    field_name: str,
    expected_shape: tuple[int, ...],
    expected_dtype: object,
) -> Array:
    """Return an array after checking its exact storage type, shape and dtype.

    field_name labels errors; expected_shape and expected_dtype are the owned
    schema. Reject non-JAX storage or wrong dtype with TypeError, and a wrong
    shape with ValueError. No conversion or array-value read is performed here.
    """
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
    """Reject NaN or infinity in an already shape-checked JAX array.

    Reading through NumPy may synchronize and copy device values. field_name
    labels ValueError. Keep this host check outside JAX transforms.
    """
    if not bool(np.all(np.isfinite(np.asarray(value)))):
        raise ValueError(f"{field_name} must contain only finite values.")


def _validate_obstacles(obstacles: object) -> Array:
    """Check one float32 (32, 8) padded obstacle table and return the same array.

    Require finite values, active flags zero or one, all-zero inactive rows,
    positive pillar radii or wall dimensions, and zero unused shape columns.
    Raise TypeError for wrong storage/dtype and ValueError for invalid shape
    or geometry fields. Body clearance is checked separately on the host.
    """
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


def _validate_recovery_catalogs() -> tuple[Array, Array]:
    """Check the two class-recovery catalogs before resolving configuration rules.

    Return the original delay int32 (6,) and regeneration float32 (6,) arrays.
    Delays are in 0..16,777,216, fractions are in [0, 1], and neutral entries
    are zero. Host reads raise TypeError for storage/dtype errors or ValueError
    for bad shapes, nonfinite values, bounds or neutral entries.
    """
    delay_catalog = _require_jax_array(
        combat.OUT_OF_COMBAT_DELAY_STEPS_BY_CLASS,
        field_name="OUT_OF_COMBAT_DELAY_STEPS_BY_CLASS",
        expected_shape=(NUM_CLASSES,),
        expected_dtype=jnp.int32,
    )
    regeneration_fraction_catalog = _require_jax_array(
        combat.OUT_OF_COMBAT_HEALTH_REGENERATION_FRACTION_PER_STEP_BY_CLASS,
        field_name=("OUT_OF_COMBAT_HEALTH_REGENERATION_FRACTION_PER_STEP_BY_CLASS"),
        expected_shape=(NUM_CLASSES,),
        expected_dtype=jnp.float32,
    )
    _require_finite_array(
        delay_catalog,
        field_name="OUT_OF_COMBAT_DELAY_STEPS_BY_CLASS",
    )
    _require_finite_array(
        regeneration_fraction_catalog,
        field_name=("OUT_OF_COMBAT_HEALTH_REGENERATION_FRACTION_PER_STEP_BY_CLASS"),
    )

    host_delays = np.asarray(delay_catalog)
    if bool(np.any((host_delays < 0) | (host_delays > _MAX_EXACT_FLOAT32_INTEGER))):
        raise ValueError(
            "OUT_OF_COMBAT_DELAY_STEPS_BY_CLASS values must be in "
            f"[0, {_MAX_EXACT_FLOAT32_INTEGER}]."
        )

    host_regeneration_fractions = np.asarray(regeneration_fraction_catalog)
    if bool(
        np.any(
            (host_regeneration_fractions < 0.0) | (host_regeneration_fractions > 1.0)
        )
    ):
        raise ValueError(
            "OUT_OF_COMBAT_HEALTH_REGENERATION_FRACTION_PER_STEP_BY_CLASS "
            "values must be in [0.0, 1.0]."
        )

    if int(host_delays[NEUTRAL_CLASS_ID]) != 0:
        raise ValueError("OUT_OF_COMBAT_DELAY_STEPS_BY_CLASS neutral row must be zero.")
    if float(host_regeneration_fractions[NEUTRAL_CLASS_ID]) != 0.0:
        raise ValueError(
            "OUT_OF_COMBAT_HEALTH_REGENERATION_FRACTION_PER_STEP_BY_CLASS "
            "neutral row must be zero."
        )

    return delay_catalog, regeneration_fraction_catalog


def _validate_agent_profile(
    profile: object,
    *,
    recovery_delay_catalog: Array,
    recovery_regeneration_fraction_catalog: Array,
) -> ResolvedAgentProfile:
    """Check a resolved roster against fixed team blocks and catalog capabilities.

    profile must be a ResolvedAgentProfile with ten-slot JAX arrays. The two
    supplied recovery catalogs have already passed their catalog checks. Active
    classes, team IDs and prefix padding must agree; every capability must
    exactly equal its class-catalog value. Return the unchanged profile.
    Host checks raise TypeError for storage/dtype/container errors and ValueError
    for shape, domain, padding or catalog mismatches.
    """
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
    out_of_combat_delay_steps = _require_jax_array(
        profile.out_of_combat_delay_steps,
        field_name="agent_profile.out_of_combat_delay_steps",
        expected_shape=(MAX_AGENT_SLOTS,),
        expected_dtype=jnp.int32,
    )

    float_profile_fields = (
        ("agent_radii", profile.agent_radii),
        ("base_movement_speeds", profile.base_movement_speeds),
        ("observation_radii", profile.observation_radii),
        ("basic_interaction_radii", profile.basic_interaction_radii),
        ("ultimate_interaction_radii", profile.ultimate_interaction_radii),
        ("max_health", profile.max_health),
        (
            "out_of_combat_health_regen_fraction_per_step",
            profile.out_of_combat_health_regen_fraction_per_step,
        ),
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
    host_out_of_combat_delay_steps = np.asarray(out_of_combat_delay_steps)
    host_out_of_combat_regeneration_fractions = np.asarray(
        profile.out_of_combat_health_regen_fraction_per_step
    )

    if bool(
        np.any(
            (host_out_of_combat_delay_steps < 0)
            | (host_out_of_combat_delay_steps > _MAX_EXACT_FLOAT32_INTEGER)
        )
    ):
        raise ValueError(
            "agent_profile.out_of_combat_delay_steps values must be in "
            f"[0, {_MAX_EXACT_FLOAT32_INTEGER}]."
        )
    if bool(
        np.any(
            (host_out_of_combat_regeneration_fractions < 0.0)
            | (host_out_of_combat_regeneration_fractions > 1.0)
        )
    ):
        raise ValueError(
            "agent_profile.out_of_combat_health_regen_fraction_per_step "
            "values must be in [0.0, 1.0]."
        )
    if bool(np.any(host_out_of_combat_delay_steps[~host_active_mask] != 0)):
        raise ValueError(
            "inactive agent_profile.out_of_combat_delay_steps rows must be zero."
        )
    if bool(
        np.any(host_out_of_combat_regeneration_fractions[~host_active_mask] != 0.0)
    ):
        raise ValueError(
            "inactive "
            "agent_profile.out_of_combat_health_regen_fraction_per_step "
            "rows must be zero."
        )

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
        (
            "out_of_combat_delay_steps",
            recovery_delay_catalog[class_ids],
        ),
        (
            "out_of_combat_health_regen_fraction_per_step",
            recovery_regeneration_fraction_catalog[class_ids],
        ),
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
    """Reject body positions that overlap an active obstacle.

    Positions (N, 2), matching radii (N,) and obstacles (32, 8) are already
    validated arrays. Check that obstacle extent plus the largest radius fits
    float32, then reuse the authoritative disc-overlap predicate for each pair.
    Host reads may synchronize device work; ValueError names field_name and
    the failing body/obstacle. Tangency follows the shared geometry predicate.
    """
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
    """Check all ten pads as a complete fallback-body formation.

    positions must be float32 (2, 5, 2), in Team A then Team B order. For each
    team, use its largest configured radius on all five pads, including unused
    slots; an empty team uses radius zero. Check finite coordinates, map bounds,
    obstacles and pairwise body clearance. Return None; raise TypeError for
    wrong storage/dtype or ValueError for a shape or geometry violation.
    """
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


def _validate_red_zone_depth(config: EnvConfig) -> None:
    """Check team_deathmatch_red_zone_depth against the task and map width.

    Parameters
    ----------
    config : EnvConfig
        One scalar configuration whose task_mode and map_width have already
        passed their own checks.

    Returns
    -------
    None
        The depth is a Python float that is 0.0, or a positive value whose
        float32 value Core can use exactly as scored.

    Raises
    ------
    TypeError
        The depth is not exactly a Python float. Booleans, ints and NumPy
        scalars are rejected, as for map_width.
    ValueError
        The depth is not finite; has a negative sign (including -0.0, so
        "off" has one encoding); is not 0.0 in neutral mode; or, when
        positive, its float32 value is not finite, is below the smallest
        normal float32 (JAX treats smaller values as zero, which would
        silently switch the rule off) or exceeds the float32 map_width.

    Notes
    -----
    Host-only scalar checks; no arrays are read. The comparisons use the
    float32 values that reset, step and scoring use, so a Python value such
    as 12.000000000000002 on a width-12 map is accepted because it becomes
    12.0 in float32.
    """
    depth = config.team_deathmatch_red_zone_depth
    if type(depth) is not float:
        raise TypeError(
            "team_deathmatch_red_zone_depth must be a float, not "
            f"{type(depth).__name__}."
        )
    if not math.isfinite(depth):
        raise ValueError(f"team_deathmatch_red_zone_depth must be finite, not {depth}.")
    if math.copysign(1.0, depth) < 0.0:
        raise ValueError(
            f"team_deathmatch_red_zone_depth must be nonnegative, not {depth}."
        )
    if config.task_mode == TASK_MODE_NEUTRAL and depth != 0.0:
        raise ValueError(
            f"team_deathmatch_red_zone_depth must be 0.0 in neutral mode, not {depth}."
        )
    if depth == 0.0:
        return
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        execution_depth = np.float32(depth)
        execution_width = np.float32(config.map_width)
    if not bool(np.isfinite(execution_depth)):
        raise ValueError(
            "team_deathmatch_red_zone_depth must remain finite after conversion "
            "to float32."
        )
    if execution_depth < _FLOAT32_SMALLEST_NORMAL:
        raise ValueError(
            "team_deathmatch_red_zone_depth must remain at least "
            f"{float(_FLOAT32_SMALLEST_NORMAL)} after conversion to float32; "
            f"smaller positive values underflow, not {depth}."
        )
    if execution_depth > execution_width:
        raise ValueError(
            "team_deathmatch_red_zone_depth must not exceed map_width after "
            f"conversion to float32, not {depth} with map_width {config.map_width}."
        )


def validate_env_config(config: EnvConfig) -> None:
    """Validate one resolved episode configuration on the host before reset.

    Parameters
    ----------
    config : EnvConfig
        One scalar configuration. Python int/float settings and JAX array
        fields must meet the types, bounds and shapes documented by EnvConfig.
        The supported modes are neutral and Team Deathmatch.

    Returns
    -------
    None
        Success means the configuration, class catalogs, roster, obstacle table
        and complete spawn-pad formation satisfy the current Core contract.
        The input is not changed.

    Raises
    ------
    TypeError
        A container, Python scalar, JAX array or dtype is wrong.
    ValueError
        A shape, finite-value check, bound, mode, padding, catalog agreement or
        geometry rule fails. Reserved KOTH and CTF modes are rejected.

    Notes
    -----
    This reads array values on the host and may synchronize device work. Call
    it in builders before jit, vmap or scan, never inside a rollout. Generic
    scientific configurations may use a validated movement scale in (0, 1];
    validate_product_env_config adds the product's fixed calibration. The Red
    Zone depth is checked after the map size and before the Team Deathmatch
    threshold range, so a config with several faults reports the first in
    that order.
    """
    if type(config) is not EnvConfig:
        raise TypeError(f"config must be an EnvConfig, not {type(config).__name__}.")

    if type(config.task_mode) is not int:
        raise TypeError(
            f"task_mode must be an int, not {type(config.task_mode).__name__}."
        )
    if config.task_mode in (TASK_MODE_KOTH, TASK_MODE_CTF):
        raise ValueError(
            f"task_mode {config.task_mode} is reserved but is not implemented."
        )
    if config.task_mode not in (TASK_MODE_NEUTRAL, TASK_MODE_TDM):
        raise ValueError(
            "task_mode must identify an available mode: "
            f"{TASK_MODE_NEUTRAL} (neutral) or {TASK_MODE_TDM} "
            f"(Team Deathmatch), not {config.task_mode}."
        )

    if type(config.team_deathmatch_score_threshold) is not int:
        raise TypeError(
            "team_deathmatch_score_threshold must be an int, not "
            f"{type(config.team_deathmatch_score_threshold).__name__}."
        )
    if (
        config.task_mode == TASK_MODE_NEUTRAL
        and config.team_deathmatch_score_threshold != 0
    ):
        raise ValueError(
            "team_deathmatch_score_threshold must be zero in neutral mode, "
            f"not {config.team_deathmatch_score_threshold}."
        )

    if type(config.max_steps) is not int:
        raise TypeError(
            f"max_steps must be an int, not {type(config.max_steps).__name__}."
        )
    if config.max_steps <= 0:
        raise ValueError(f"max_steps must be greater than 0, not {config.max_steps}.")
    if config.max_steps > _MAX_EXACT_FLOAT32_INTEGER:
        raise ValueError(
            "max_steps must be at most "
            f"{_MAX_EXACT_FLOAT32_INTEGER}, not {config.max_steps}."
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

    _validate_red_zone_depth(config)
    # The Team Deathmatch threshold range is checked after the depth because
    # the largest safe threshold depends on it.
    maximum_threshold = maximum_team_deathmatch_score_threshold(
        config.team_deathmatch_red_zone_depth
    )
    if config.task_mode == TASK_MODE_TDM and not (
        1 <= config.team_deathmatch_score_threshold <= maximum_threshold
    ):
        raise ValueError(
            "team_deathmatch_score_threshold must be in "
            f"[1, {maximum_threshold}] in Team Deathmatch, "
            f"not {config.team_deathmatch_score_threshold}."
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

    team_respawn_wave_period_step_count = _require_jax_array(
        config.team_respawn_wave_period_step_count,
        field_name="team_respawn_wave_period_step_count",
        expected_shape=(NUM_TEAMS,),
        expected_dtype=jnp.int32,
    )
    if bool(np.any(np.asarray(team_respawn_wave_period_step_count) <= 0)):
        raise ValueError(
            "team_respawn_wave_period_step_count must contain only positive values."
        )

    recovery_delay_catalog, recovery_regeneration_fraction_catalog = (
        _validate_recovery_catalogs()
    )
    obstacles = _validate_obstacles(config.obstacles)
    profile = _validate_agent_profile(
        config.agent_profile,
        recovery_delay_catalog=recovery_delay_catalog,
        recovery_regeneration_fraction_catalog=(recovery_regeneration_fraction_catalog),
    )
    if config.task_mode == TASK_MODE_TDM:
        host_active_mask = np.asarray(profile.active_mask)
        team_a_has_active_member = bool(np.any(host_active_mask[:MAX_AGENTS_PER_TEAM]))
        team_b_has_active_member = bool(np.any(host_active_mask[MAX_AGENTS_PER_TEAM:]))
        if not team_a_has_active_member or not team_b_has_active_member:
            raise ValueError(
                "Team Deathmatch requires at least one configured active member "
                "on each team."
            )
    _validate_team_spawn_pad_positions(
        config.team_spawn_pad_positions,
        map_width=config.map_width,
        map_height=config.map_height,
        obstacles=obstacles,
        profile=profile,
    )


def validate_product_env_config(config: EnvConfig) -> None:
    """Validate one product episode with the fixed movement calibration.

    Parameters
    ----------
    config : EnvConfig
        One concrete scalar episode configuration with the EnvConfig contract.

    Returns
    -------
    None
        The full generic configuration check passed and the ordinary movement
        scale equals the product value 1.0. The input is not changed.

    Raises
    ------
    TypeError
        A generic configuration type or dtype check fails.
    ValueError
        A generic configuration check fails or the movement scale is not 1.0.

    Notes
    -----
    This is host-only and may read device arrays. Explicit scientific tests
    using other supported scales should use validate_env_config instead.
    """
    validate_env_config(config)
    if config.ordinary_movement_distance_scale != CANONICAL_PRODUCT_MOVEMENT_SCALE:
        raise ValueError(
            "product ordinary_movement_distance_scale must equal "
            f"{CANONICAL_PRODUCT_MOVEMENT_SCALE:.2f}, not "
            f"{config.ordinary_movement_distance_scale}."
        )


def _validate_state_positions(
    positions: Array,
    *,
    config: EnvConfig,
    active_mask: np.ndarray,
) -> None:
    """Check a snapshot's hard static-geometry and unused-position rules.

    Positions are float32 (10, 2); active_mask is a host bool (10,) array and
    config is already valid. Unused positions must be zero. Configured bodies
    must fit radius-adjusted map bounds within GEOMETRY_TOLERANCE and satisfy
    the shared obstacle predicate. ValueError reports a violation.
    Runtime body overlap is allowed because a fixed-pass solver may retain
    crowded residuals; only authored starts require strict living-body clearance.
    """
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
    """Reject overlapping configured living bodies in an authored start.

    Positions (10, 2) and alive_mask (10,) have already passed state validation.
    Read them on the host and compare each living pair using catalog radii.
    Tangency is allowed and corpses do not participate. Raise ValueError for
    positive-area overlap; runtime snapshots use the looser residual contract.
    """
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
    """Return a host array after checking a closed nonnegative duration range.

    Storage and integer dtype were checked by the caller. upper_bounds is a
    scalar or broadcastable host array; both endpoints are inclusive. A host
    read may synchronize. ValueError names field_name when a bound is violated.
    """
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
    """Return host action-history values after checking one category range.

    The caller has checked array shape and int32 dtype. Require categories in
    [0, category_count), with field_name used in ValueError. This checks IDs,
    not whether an old mask admitted the action; that mask is not available here.
    """
    host_values = np.asarray(values)
    if bool(np.any((host_values < 0) | (host_values >= category_count))):
        raise ValueError(
            f"{field_name} must contain categories in [0, {category_count})."
        )
    return host_values


def validate_env_state(config: EnvConfig, state: EnvState) -> None:
    """Validate one runtime snapshot at a host-owned boundary.

    Parameters
    ----------
    config : EnvConfig
        Matching scalar configuration, already checked by validate_env_config.
    state : EnvState
        One ten-slot snapshot. Positions and health are float32; membership
        flags are bool; counters and categorical history are int32. Shapes
        and field meanings are documented on EnvState.

    Returns
    -------
    None
        Storage, shapes, bounds, lifecycle relationships, padding and hard
        static geometry are valid. Neither input is changed.

    Raises
    ------
    TypeError
        A container, JAX array storage type or dtype is wrong.
    ValueError
        A shape, finite-value, domain, lifecycle, padding or geometry rule fails.

    Notes
    -----
    Replay readers and tools use this host-only check. It may copy/synchronize
    device data and must stay outside reset, step, jit, vmap and scan. Dead
    agents may retain cooldowns and accepted-action history, but their health
    and transient statuses are zero. A snapshot cannot prove collision history,
    so living-body residuals from the fixed-pass solver are allowed. Authored
    starts also need validate_scenario_initial_state. This check does not rebuild
    an old action mask or prove that submitted external mask history was correct.
    Team Deathmatch scores may reach the threshold plus
    maximum_team_deathmatch_score_increment(depth) - 1 (K + 4 at depth 0,
    K + 9 at a positive Red Zone depth).
    """
    if type(config) is not EnvConfig:
        raise TypeError(f"config must be an EnvConfig, not {type(config).__name__}.")
    if type(state) is not EnvState:
        raise TypeError(f"state must be an EnvState, not {type(state).__name__}.")

    team_deathmatch_scores = _require_jax_array(
        state.team_deathmatch_scores,
        field_name="team_deathmatch_scores",
        expected_shape=(NUM_TEAMS,),
        expected_dtype=jnp.int32,
    )
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
    team_respawn_wave_countdowns = _require_jax_array(
        state.team_respawn_wave_countdowns,
        field_name="team_respawn_wave_countdowns",
        expected_shape=(NUM_TEAMS,),
        expected_dtype=jnp.int32,
    )
    spawn_shield_durations = _require_jax_array(
        state.spawn_shield_durations,
        field_name="spawn_shield_durations",
        expected_shape=(MAX_AGENT_SLOTS,),
        expected_dtype=jnp.int32,
    )
    steps_until_out_of_combat = _require_jax_array(
        state.steps_until_out_of_combat,
        field_name="steps_until_out_of_combat",
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

    host_team_deathmatch_scores = np.asarray(team_deathmatch_scores)
    if bool(np.any(host_team_deathmatch_scores < 0)):
        raise ValueError("team_deathmatch_scores must contain only nonnegative values.")
    if config.task_mode == TASK_MODE_NEUTRAL:
        if bool(np.any(host_team_deathmatch_scores != 0)):
            raise ValueError(
                "team_deathmatch_scores must be exactly zero in neutral mode."
            )
    else:
        maximum_reachable_score = (
            config.team_deathmatch_score_threshold
            + maximum_team_deathmatch_score_increment(
                config.team_deathmatch_red_zone_depth
            )
            - 1
        )
        if bool(np.any(host_team_deathmatch_scores > maximum_reachable_score)):
            raise ValueError(
                "team_deathmatch_scores must not exceed the reachable terminal "
                f"bound {maximum_reachable_score}."
            )

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

    host_team_respawn_wave_countdowns = np.asarray(team_respawn_wave_countdowns)
    if bool(np.any(host_team_respawn_wave_countdowns < 0)):
        raise ValueError(
            "team_respawn_wave_countdowns must contain only nonnegative values."
        )
    if bool(
        np.any(
            host_team_respawn_wave_countdowns
            >= np.asarray(config.team_respawn_wave_period_step_count)
        )
    ):
        raise ValueError(
            "team_respawn_wave_countdowns must be strictly less than "
            "team_respawn_wave_period_step_count."
        )

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

    host_steps_until_out_of_combat = np.asarray(steps_until_out_of_combat)
    if bool(np.any(host_steps_until_out_of_combat < 0)):
        raise ValueError(
            "steps_until_out_of_combat must contain only nonnegative values."
        )
    if bool(np.any(host_steps_until_out_of_combat[active_and_dead] != 0)):
        raise ValueError(
            "active dead steps_until_out_of_combat rows must be exactly zero."
        )
    if bool(np.any(host_steps_until_out_of_combat[inactive] != 0)):
        raise ValueError(
            "inactive steps_until_out_of_combat rows must be exactly zero."
        )
    if bool(
        np.any(
            host_steps_until_out_of_combat
            > np.asarray(config.agent_profile.out_of_combat_delay_steps)
        )
    ):
        raise ValueError(
            "steps_until_out_of_combat exceeds its resolved per-slot delay."
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
    """Validate an authored start under the stricter scenario rules.

    Parameters
    ----------
    config : EnvConfig
        Matching scalar configuration, already checked by validate_env_config.
    state : EnvState
        Authored ten-slot snapshot satisfying the ordinary runtime schema.
        A Team Deathmatch start must be below both the horizon and score limit.

    Returns
    -------
    None
        Runtime checks pass, living bodies do not overlap, and any accepted
        action history agrees with the current shield-related restrictions.
        Inputs are not changed and the simulator is not advanced.

    Raises
    ------
    TypeError
        A runtime container, storage or dtype check fails.
    ValueError
        A runtime check, Team Deathmatch start limit, shield-history rule or
        living-body clearance rule fails.
    AssertionError
        An internal target mapping returns None for a validated nonzero category.

    Notes
    -----
    This is host-only and may read device values. Authored starts have no solver
    history to justify overlapping living bodies; exact tangency is allowed,
    and corpses do not block that pairwise check. Shielded sources require
    neutral combat history and history may not target a shielded recipient.
    General runtime validation deliberately does not impose those history rules
    so snapshots with external mask provenance can still be inspected.
    """
    validate_env_state(config, state)

    if config.task_mode == TASK_MODE_TDM:
        if int(np.asarray(state.step_count)) >= config.max_steps:
            raise ValueError(
                "Team Deathmatch scenario step_count must be strictly below max_steps."
            )
        if bool(
            np.any(
                np.asarray(state.team_deathmatch_scores)
                >= config.team_deathmatch_score_threshold
            )
        ):
            raise ValueError(
                "Team Deathmatch scenario scores must be strictly below the score "
                "threshold."
            )

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

            recipient_slot = target_action_to_global_slot(
                source_slot,
                int(target_action),
            )
            if recipient_slot is None:
                raise AssertionError("A nonzero target action resolved to target-none.")

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
