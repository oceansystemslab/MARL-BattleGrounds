"""Exhaustive host-side resolved environment configuration validation."""
# pyright: reportUnknownArgumentType=false, reportUnknownLambdaType=false

from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from marl_battlegrounds.core import combat
from marl_battlegrounds.core.config import (
    CANONICAL_PRODUCT_MOVEMENT_SCALE,
    resolve_agent_profile,
    validate_env_config,
    validate_product_env_config,
)
from marl_battlegrounds.core.env import reset, step
from marl_battlegrounds.core.geometry import (
    GEOMETRY_EPSILON,
    GEOMETRY_TOLERANCE,
    disc_overlaps_obstacle,
    project_movement_with_geometry,
)
from marl_battlegrounds.core.types import (
    ENVIRONMENT_DIMENSIONS,
    MAGE_CLASS_ID,
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    MAX_OBSTACLE_SLOTS,
    MOVE_STAY,
    NO_TEAM_ID,
    NUM_TEAMS,
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
    Action,
    ActionMask,
    EnvConfig,
    EnvState,
    Info,
    Observation,
    ResolvedAgentProfile,
)


def _empty_obstacles() -> Array:
    return jnp.zeros((MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES), dtype=jnp.float32)


def _pillar(*, x: float = 5.0, y: float = 4.0, radius: float = 1.0) -> Array:
    obstacle = jnp.zeros((OBSTACLE_FEATURES,), dtype=jnp.float32)
    obstacle = obstacle.at[OBSTACLE_FEATURE_TYPE].set(OBSTACLE_TYPE_PILLAR)
    obstacle = obstacle.at[OBSTACLE_FEATURE_X].set(x)
    obstacle = obstacle.at[OBSTACLE_FEATURE_Y].set(y)
    obstacle = obstacle.at[OBSTACLE_FEATURE_RADIUS].set(radius)
    return obstacle.at[OBSTACLE_FEATURE_ACTIVE].set(1.0)


def _wall(
    *,
    x: float = 8.0,
    y: float = 4.0,
    width: float = 2.0,
    height: float = 1.0,
    theta: float = 0.5,
) -> Array:
    obstacle = jnp.zeros((OBSTACLE_FEATURES,), dtype=jnp.float32)
    obstacle = obstacle.at[OBSTACLE_FEATURE_TYPE].set(OBSTACLE_TYPE_WALL)
    obstacle = obstacle.at[OBSTACLE_FEATURE_X].set(x)
    obstacle = obstacle.at[OBSTACLE_FEATURE_Y].set(y)
    obstacle = obstacle.at[OBSTACLE_FEATURE_WIDTH].set(width)
    obstacle = obstacle.at[OBSTACLE_FEATURE_HEIGHT].set(height)
    obstacle = obstacle.at[OBSTACLE_FEATURE_THETA].set(theta)
    return obstacle.at[OBSTACLE_FEATURE_ACTIVE].set(1.0)


def _spawn_pad_positions() -> Array:
    """Return five real, statically valid pad locations for each team."""
    return jnp.asarray(
        (
            (0.5, 0.5),
            (2.5, 0.5),
            (4.5, 0.5),
            (6.5, 0.5),
            (8.5, 0.5),
            (0.5, 7.5),
            (2.5, 7.5),
            (4.5, 7.5),
            (6.5, 7.5),
            (8.5, 7.5),
        ),
        dtype=jnp.float32,
    ).reshape(NUM_TEAMS, MAX_AGENTS_PER_TEAM, ENVIRONMENT_DIMENSIONS)


def _valid_config(
    *,
    team_sizes: tuple[int, int] = (2, 2),
    task_mode: int = TASK_MODE_NEUTRAL,
    team_deathmatch_score_threshold: int = 0,
    obstacles: Array | None = None,
    spawn_pad_positions: Array | None = None,
    spawn_shield_duration_steps: int = 3,
    spawn_shield_movement_speed: float = 2.0,
    team_respawn_wave_period_step_count: Array | None = None,
) -> EnvConfig:
    profile = resolve_agent_profile(
        jnp.full((MAX_AGENT_SLOTS,), MAGE_CLASS_ID, dtype=jnp.int32),
        jnp.asarray(team_sizes, dtype=jnp.int32),
    )
    return EnvConfig(
        task_mode=task_mode,
        team_deathmatch_score_threshold=team_deathmatch_score_threshold,
        max_steps=100,
        map_width=12.0,
        map_height=8.0,
        obstacles=_empty_obstacles() if obstacles is None else obstacles,
        agent_profile=profile,
        ordinary_movement_distance_scale=1.0,
        team_spawn_pad_positions=(
            _spawn_pad_positions()
            if spawn_pad_positions is None
            else spawn_pad_positions
        ),
        spawn_shield_duration_steps=spawn_shield_duration_steps,
        spawn_shield_movement_speed=spawn_shield_movement_speed,
        team_respawn_wave_period_step_count=(
            jnp.asarray((5, 7), dtype=jnp.int32)
            if team_respawn_wave_period_step_count is None
            else team_respawn_wave_period_step_count
        ),
    )


def _replace_config(config: EnvConfig, **changes: object) -> EnvConfig:
    return config._replace(**changes)


def _replace_profile(config: EnvConfig, **changes: object) -> EnvConfig:
    profile = config.agent_profile._replace(**changes)
    return config._replace(agent_profile=profile)


def _assert_pytrees_equal(left: object, right: object) -> None:
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    assert len(left_leaves) == len(right_leaves)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        assert bool(jnp.array_equal(left_leaf, right_leaf))


def _public_geometry_projection(config: EnvConfig, center: Array) -> Array:
    positions = (
        jnp.zeros((MAX_AGENT_SLOTS, ENVIRONMENT_DIMENSIONS), dtype=jnp.float32)
        .at[0]
        .set(center)
    )
    return project_movement_with_geometry(
        agent_positions=positions,
        agent_radii=config.agent_profile.agent_radii,
        intended_movement_deltas=jnp.zeros_like(positions),
        active_mask=config.agent_profile.active_mask,
        alive_mask=config.agent_profile.active_mask,
        map_width=config.map_width,
        map_height=config.map_height,
        obstacles=config.obstacles,
        always_participates_in_agent_agent_collision=(config.agent_profile.active_mask),
        participates_in_agent_agent_collision_at_final_position=(
            config.agent_profile.active_mask
        ),
    )[0]


def _wall_local_to_world(obstacle: Array, local_center: tuple[float, float]) -> Array:
    theta = obstacle[OBSTACLE_FEATURE_THETA]
    cosine = jnp.cos(theta)
    sine = jnp.sin(theta)
    rotation = jnp.asarray(((cosine, -sine), (sine, cosine)), dtype=jnp.float32)
    wall_center = obstacle[jnp.asarray((OBSTACLE_FEATURE_X, OBSTACLE_FEATURE_Y))]
    return rotation @ jnp.asarray(local_center, dtype=jnp.float32) + wall_center


def test_validation_inventory_covers_current_public_schemas() -> None:
    assert EnvConfig._fields == (
        "task_mode",
        "team_deathmatch_score_threshold",
        "max_steps",
        "map_width",
        "map_height",
        "obstacles",
        "agent_profile",
        "ordinary_movement_distance_scale",
        "team_spawn_pad_positions",
        "spawn_shield_duration_steps",
        "spawn_shield_movement_speed",
        "team_respawn_wave_period_step_count",
    )
    assert ResolvedAgentProfile._fields == (
        "class_ids",
        "team_ids",
        "active_mask",
        "agent_radii",
        "base_movement_speeds",
        "observation_radii",
        "basic_interaction_radii",
        "ultimate_interaction_radii",
        "max_health",
        "out_of_combat_delay_steps",
        "out_of_combat_health_regen_fraction_per_step",
    )


def test_valid_empty_and_obstacle_bearing_configs_return_none_without_mutation() -> (
    None
):
    obstacles = _empty_obstacles().at[0].set(_pillar()).at[1].set(_wall())
    for config in (_valid_config(), _valid_config(obstacles=obstacles)):
        before = config
        assert validate_env_config(config) is None
        _assert_pytrees_equal(config, before)


def test_validator_rejects_wrong_top_level_and_profile_types() -> None:
    with pytest.raises(TypeError, match="config"):
        validate_env_config(cast(Any, object()))
    with pytest.raises(TypeError, match="agent_profile"):
        validate_env_config(
            _replace_config(_valid_config(), agent_profile=cast(Any, ()))
        )


@pytest.mark.parametrize(
    "task_mode",
    (True, False, 0.0, np.int32(0), jnp.asarray(0), "0"),
    ids=("true", "false", "float", "numpy-int", "jax-scalar", "string"),
)
def test_task_mode_requires_an_exact_python_integer(task_mode: object) -> None:
    with pytest.raises(TypeError, match="task_mode must be an int"):
        validate_env_config(_replace_config(_valid_config(), task_mode=task_mode))


@pytest.mark.parametrize(
    "task_mode",
    (TASK_MODE_KOTH, TASK_MODE_CTF),
    ids=("king-of-the-hill", "capture-the-flag"),
)
def test_reserved_task_modes_remain_unavailable(task_mode: int) -> None:
    with pytest.raises(ValueError, match="reserved but is not implemented"):
        validate_env_config(_replace_config(_valid_config(), task_mode=task_mode))


@pytest.mark.parametrize("task_mode", (-1, 4, 99))
def test_unknown_task_modes_are_rejected(task_mode: int) -> None:
    with pytest.raises(ValueError, match="available mode"):
        validate_env_config(_replace_config(_valid_config(), task_mode=task_mode))


@pytest.mark.parametrize(
    "threshold",
    (True, False, 1.0, np.int32(1), jnp.asarray(1), "1"),
    ids=("true", "false", "float", "numpy-int", "jax-scalar", "string"),
)
def test_team_deathmatch_threshold_requires_an_exact_python_integer(
    threshold: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="team_deathmatch_score_threshold must be an int",
    ):
        validate_env_config(
            _replace_config(
                _valid_config(),
                team_deathmatch_score_threshold=threshold,
            )
        )


@pytest.mark.parametrize("threshold", (-1, 1))
def test_neutral_mode_requires_zero_team_deathmatch_threshold(
    threshold: int,
) -> None:
    with pytest.raises(ValueError, match="must be zero in neutral mode"):
        validate_env_config(_valid_config(team_deathmatch_score_threshold=threshold))


@pytest.mark.parametrize(
    "threshold",
    (1, 2**24 - 4),
    ids=("minimum", "maximum-exact-context-value"),
)
def test_team_deathmatch_threshold_accepts_its_complete_domain(
    threshold: int,
) -> None:
    assert (
        validate_env_config(
            _valid_config(
                task_mode=TASK_MODE_TDM,
                team_deathmatch_score_threshold=threshold,
            )
        )
        is None
    )


@pytest.mark.parametrize("threshold", (-1, 0, 2**24 - 3))
def test_team_deathmatch_threshold_rejects_values_outside_its_domain(
    threshold: int,
) -> None:
    with pytest.raises(ValueError, match="must be in"):
        validate_env_config(
            _valid_config(
                task_mode=TASK_MODE_TDM,
                team_deathmatch_score_threshold=threshold,
            )
        )


@pytest.mark.parametrize(
    "team_sizes",
    ((0, 0), (0, 1), (1, 0)),
    ids=("both-empty", "team-a-empty", "team-b-empty"),
)
def test_team_deathmatch_requires_one_active_member_per_team(
    team_sizes: tuple[int, int],
) -> None:
    with pytest.raises(ValueError, match="at least one configured active member"):
        validate_env_config(
            _valid_config(
                team_sizes=team_sizes,
                task_mode=TASK_MODE_TDM,
                team_deathmatch_score_threshold=1,
            )
        )


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "error_type", "message"),
    (
        ("max_steps", True, TypeError, "max_steps"),
        ("max_steps", np.int32(100), TypeError, "max_steps"),
        ("max_steps", 0, ValueError, "max_steps"),
        ("max_steps", 2**24 + 1, ValueError, "max_steps"),
        ("map_width", 12, TypeError, "map_width"),
        ("map_width", np.float32(12.0), TypeError, "map_width"),
        ("map_width", np.float64(12.0), TypeError, "map_width"),
        ("map_width", 0.0, ValueError, "map_width"),
        ("map_width", float("nan"), ValueError, "map_width"),
        ("map_width", float("inf"), ValueError, "map_width"),
        (
            "map_width",
            float(jnp.finfo(jnp.float32).max) * 2.0,
            ValueError,
            "map_width",
        ),
        ("map_height", 8, TypeError, "map_height"),
        ("map_height", -1.0, ValueError, "map_height"),
        ("map_height", float("nan"), ValueError, "map_height"),
        ("map_height", float("inf"), ValueError, "map_height"),
    ),
)
def test_invalid_scalar_fields_fail_early(
    field_name: str,
    invalid_value: object,
    error_type: type[Exception],
    message: str,
) -> None:
    config = _replace_config(_valid_config(), **{field_name: invalid_value})
    with pytest.raises(error_type, match=message):
        validate_env_config(config)


@pytest.mark.parametrize("max_steps", (1, 2**24), ids=("minimum", "float32-exact-max"))
def test_max_steps_accepts_exact_float32_integer_domain(max_steps: int) -> None:
    """Accept both closed endpoints of the public exact-integer horizon."""
    assert (
        validate_env_config(_replace_config(_valid_config(), max_steps=max_steps))
        is None
    )


@pytest.mark.parametrize(
    "movement_scale",
    (
        pytest.param(1.0, id="product-canonical"),
        pytest.param(0.1, id="noncanonical-experimental-tenth"),
        pytest.param(0.375, id="noncanonical-experimental-additional"),
        pytest.param(
            float(np.nextafter(np.float32(0.0), np.float32(1.0))),
            id="noncanonical-experimental-subnormal",
        ),
    ),
)
def test_movement_scale_accepts_positive_finite_float32_execution_values(
    movement_scale: float,
) -> None:
    """Prove validation accepts the complete documented execution domain."""
    config = _replace_config(
        _valid_config(),
        ordinary_movement_distance_scale=movement_scale,
    )
    before = config

    assert validate_env_config(config) is None
    _assert_pytrees_equal(config, before)


def test_product_config_requires_the_canonical_movement_scale() -> None:
    canonical = _valid_config()
    experimental = _replace_config(
        canonical,
        ordinary_movement_distance_scale=0.375,
    )

    assert CANONICAL_PRODUCT_MOVEMENT_SCALE == 1.0
    assert validate_product_env_config(canonical) is None
    assert validate_env_config(experimental) is None
    with pytest.raises(ValueError, match=r"must equal 1\.00"):
        validate_product_env_config(experimental)


@pytest.mark.parametrize(
    ("invalid_scale", "error_type"),
    (
        pytest.param(True, TypeError, id="bool"),
        pytest.param(1, TypeError, id="integer"),
        pytest.param(np.float32(0.1), TypeError, id="numpy-float32"),
        pytest.param(np.float64(0.1), TypeError, id="numpy-float64"),
        pytest.param(0.0, ValueError, id="zero"),
        pytest.param(-0.1, ValueError, id="negative"),
        pytest.param(1.0001, ValueError, id="above-one"),
        pytest.param(float("nan"), ValueError, id="nan"),
        pytest.param(float("inf"), ValueError, id="positive-infinity"),
        pytest.param(float("-inf"), ValueError, id="negative-infinity"),
        pytest.param(1e100, ValueError, id="float32-overflow"),
        pytest.param(
            float(np.nextafter(np.float32(0.0), np.float32(1.0))) / 2.0,
            ValueError,
            id="float32-underflow-to-zero",
        ),
    ),
)
def test_movement_scale_rejects_wrong_types_and_invalid_float32_values(
    invalid_scale: object,
    error_type: type[Exception],
) -> None:
    """Prove the host boundary rejects scales that cannot authorize movement."""
    config = _replace_config(
        _valid_config(),
        ordinary_movement_distance_scale=invalid_scale,
    )

    with pytest.raises(error_type, match="ordinary_movement_distance_scale"):
        validate_env_config(config)


@pytest.mark.parametrize(
    "duration_steps",
    (
        pytest.param(0, id="disabled"),
        pytest.param(1, id="single-transition"),
        pytest.param(3, id="official"),
        pytest.param(int(np.iinfo(np.int32).max), id="int32-maximum"),
    ),
)
def test_spawn_shield_duration_accepts_its_complete_host_domain(
    duration_steps: int,
) -> None:
    """Accept every representative nonnegative int32-representable duration."""
    assert (
        validate_env_config(_valid_config(spawn_shield_duration_steps=duration_steps))
        is None
    )


@pytest.mark.parametrize(
    ("invalid_duration", "error_type"),
    (
        pytest.param(True, TypeError, id="bool"),
        pytest.param(np.int32(3), TypeError, id="numpy-int32"),
        pytest.param(3.0, TypeError, id="float"),
        pytest.param(-1, ValueError, id="negative"),
        pytest.param(int(np.iinfo(np.int32).max) + 1, ValueError, id="overflow"),
    ),
)
def test_spawn_shield_duration_rejects_wrong_types_and_invalid_values(
    invalid_duration: object,
    error_type: type[Exception],
) -> None:
    """Keep the public host integer safe for the int32 runtime counter."""
    with pytest.raises(error_type, match="spawn_shield_duration_steps"):
        validate_env_config(
            _replace_config(
                _valid_config(),
                spawn_shield_duration_steps=invalid_duration,
            )
        )


@pytest.mark.parametrize(
    "movement_speed",
    (
        pytest.param(2.0, id="official"),
        pytest.param(0.1, id="nonexact-binary-fraction"),
        pytest.param(
            float(np.nextafter(np.float32(0.0), np.float32(1.0))),
            id="minimum-positive-float32",
        ),
        pytest.param(float(np.finfo(np.float32).max), id="float32-maximum"),
    ),
)
def test_spawn_shield_speed_accepts_positive_finite_float32_execution_values(
    movement_speed: float,
) -> None:
    """Accept host floats that remain finite and positive in the JAX core."""
    assert (
        validate_env_config(_valid_config(spawn_shield_movement_speed=movement_speed))
        is None
    )


@pytest.mark.parametrize(
    ("invalid_speed", "error_type"),
    (
        pytest.param(True, TypeError, id="bool"),
        pytest.param(2, TypeError, id="integer"),
        pytest.param(np.float32(2.0), TypeError, id="numpy-float32"),
        pytest.param(np.float64(2.0), TypeError, id="numpy-float64"),
        pytest.param(0.0, ValueError, id="zero"),
        pytest.param(-1.0, ValueError, id="negative"),
        pytest.param(float("nan"), ValueError, id="nan"),
        pytest.param(float("inf"), ValueError, id="positive-infinity"),
        pytest.param(float("-inf"), ValueError, id="negative-infinity"),
        pytest.param(1e100, ValueError, id="float32-overflow"),
        pytest.param(
            float(np.nextafter(np.float32(0.0), np.float32(1.0))) / 2.0,
            ValueError,
            id="float32-underflow-to-zero",
        ),
    ),
)
def test_spawn_shield_speed_rejects_wrong_types_and_invalid_float32_values(
    invalid_speed: object,
    error_type: type[Exception],
) -> None:
    """Reject speeds that cannot produce a positive finite float32 delta."""
    with pytest.raises(error_type, match="spawn_shield_movement_speed"):
        validate_env_config(
            _replace_config(
                _valid_config(),
                spawn_shield_movement_speed=invalid_speed,
            )
        )


@pytest.mark.parametrize(
    "period_step_counts",
    (
        pytest.param(
            jnp.asarray((1, 1), dtype=jnp.int32),
            id="wave-every-transition",
        ),
        pytest.param(
            jnp.asarray((1, 7), dtype=jnp.int32),
            id="asymmetric-periods",
        ),
        pytest.param(
            jnp.asarray(
                (np.iinfo(np.int32).max, np.iinfo(np.int32).max - 1),
                dtype=jnp.int32,
            ),
            id="int32-boundary",
        ),
    ),
)
def test_respawn_wave_period_accepts_positive_int32_team_values(
    period_step_counts: Array,
) -> None:
    """Accept positive per-team periods across the complete int32 domain."""
    assert (
        validate_env_config(
            _valid_config(
                team_respawn_wave_period_step_count=period_step_counts,
            )
        )
        is None
    )


@pytest.mark.parametrize(
    ("invalid_periods", "error_type"),
    (
        pytest.param(cast(Any, []), TypeError, id="python-list"),
        pytest.param(
            np.asarray((3, 5), dtype=np.int32),
            TypeError,
            id="numpy-array",
        ),
        pytest.param(
            jnp.asarray(3, dtype=jnp.int32),
            ValueError,
            id="scalar",
        ),
        pytest.param(
            jnp.ones((NUM_TEAMS, 1), dtype=jnp.int32),
            ValueError,
            id="wrong-shape",
        ),
        pytest.param(
            jnp.asarray((3.0, 5.0), dtype=jnp.float32),
            TypeError,
            id="wrong-dtype",
        ),
    ),
)
def test_respawn_wave_period_enforces_exact_jax_storage(
    invalid_periods: object,
    error_type: type[Exception],
) -> None:
    """Reject host containers and shape or dtype drift at the public boundary."""
    with pytest.raises(error_type, match="team_respawn_wave_period_step_count"):
        validate_env_config(
            _replace_config(
                _valid_config(),
                team_respawn_wave_period_step_count=invalid_periods,
            )
        )


@pytest.mark.parametrize(
    "invalid_periods",
    (
        pytest.param((0, 1), id="team-a-zero"),
        pytest.param((1, 0), id="team-b-zero"),
        pytest.param((-1, 1), id="team-a-negative"),
        pytest.param((1, -1), id="team-b-negative"),
    ),
)
def test_respawn_wave_period_rejects_nonpositive_team_values(
    invalid_periods: tuple[int, int],
) -> None:
    """Require every team clock period to make forward progress."""
    periods = jnp.asarray(invalid_periods, dtype=jnp.int32)
    with pytest.raises(ValueError, match="positive"):
        validate_env_config(_valid_config(team_respawn_wave_period_step_count=periods))


@pytest.mark.parametrize(
    ("invalid_value", "error_type", "message"),
    (
        (cast(Any, []), TypeError, "obstacles"),
        (
            jnp.zeros((MAX_OBSTACLE_SLOTS - 1, OBSTACLE_FEATURES), dtype=jnp.float32),
            ValueError,
            "shape",
        ),
        (
            jnp.zeros((MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES), dtype=jnp.int32),
            TypeError,
            "dtype",
        ),
    ),
)
def test_obstacle_storage_contract_is_enforced(
    invalid_value: object, error_type: type[Exception], message: str
) -> None:
    with pytest.raises(error_type, match=message):
        validate_env_config(_replace_config(_valid_config(), obstacles=invalid_value))


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda obstacles: obstacles.at[0, OBSTACLE_FEATURE_X].set(jnp.nan),
            "finite",
        ),
        (
            lambda obstacles: obstacles.at[0, OBSTACLE_FEATURE_ACTIVE].set(0.5),
            "active",
        ),
        (
            lambda obstacles: obstacles.at[0, OBSTACLE_FEATURE_X].set(1.0),
            "inactive obstacle rows",
        ),
        (
            lambda obstacles: (
                obstacles.at[0].set(_pillar()).at[0, OBSTACLE_FEATURE_TYPE].set(9.0)
            ),
            "type",
        ),
        (
            lambda obstacles: obstacles.at[0].set(_pillar(radius=0.0)),
            "pillar radius",
        ),
        (
            lambda obstacles: (
                obstacles.at[0].set(_pillar()).at[0, OBSTACLE_FEATURE_WIDTH].set(1.0)
            ),
            "pillar wall fields",
        ),
        (
            lambda obstacles: obstacles.at[0].set(_wall(width=0.0)),
            "wall width",
        ),
        (
            lambda obstacles: obstacles.at[0].set(_wall(height=0.0)),
            "wall height",
        ),
        (
            lambda obstacles: (
                obstacles.at[0].set(_wall()).at[0, OBSTACLE_FEATURE_RADIUS].set(1.0)
            ),
            "wall radius",
        ),
    ),
)
def test_obstacle_value_contract_is_enforced(
    mutate: Callable[[Array], Array], message: str
) -> None:
    config = _replace_config(_valid_config(), obstacles=mutate(_empty_obstacles()))
    with pytest.raises(ValueError, match=message):
        validate_env_config(config)


@pytest.mark.parametrize(
    ("catalog_name", "invalid_catalog", "error_type"),
    (
        pytest.param(
            "OUT_OF_COMBAT_DELAY_STEPS_BY_CLASS",
            cast(Any, []),
            TypeError,
            id="delay-non-jax",
        ),
        pytest.param(
            "OUT_OF_COMBAT_DELAY_STEPS_BY_CLASS",
            jnp.zeros((5,), dtype=jnp.int32),
            ValueError,
            id="delay-shape",
        ),
        pytest.param(
            "OUT_OF_COMBAT_DELAY_STEPS_BY_CLASS",
            jnp.zeros((6,), dtype=jnp.float32),
            TypeError,
            id="delay-dtype",
        ),
        pytest.param(
            "OUT_OF_COMBAT_HEALTH_REGENERATION_FRACTION_PER_STEP_BY_CLASS",
            cast(Any, np.zeros((6,), dtype=np.float32)),
            TypeError,
            id="regeneration-non-jax",
        ),
        pytest.param(
            "OUT_OF_COMBAT_HEALTH_REGENERATION_FRACTION_PER_STEP_BY_CLASS",
            jnp.zeros((5,), dtype=jnp.float32),
            ValueError,
            id="regeneration-shape",
        ),
        pytest.param(
            "OUT_OF_COMBAT_HEALTH_REGENERATION_FRACTION_PER_STEP_BY_CLASS",
            jnp.zeros((6,), dtype=jnp.int32),
            TypeError,
            id="regeneration-dtype",
        ),
    ),
)
def test_recovery_catalogs_require_exact_jax_storage(
    monkeypatch: pytest.MonkeyPatch,
    catalog_name: str,
    invalid_catalog: object,
    error_type: type[Exception],
) -> None:
    """Reject catalog container, shape, and dtype drift at the host boundary."""
    config = _valid_config()
    monkeypatch.setattr(combat, catalog_name, invalid_catalog)

    with pytest.raises(error_type, match=catalog_name):
        validate_env_config(config)


@pytest.mark.parametrize(
    ("catalog_name", "invalid_catalog", "message"),
    (
        pytest.param(
            "OUT_OF_COMBAT_DELAY_STEPS_BY_CLASS",
            combat.OUT_OF_COMBAT_DELAY_STEPS_BY_CLASS.at[1].set(-1),
            r"\[0, 16777216\]",
            id="delay-negative",
        ),
        pytest.param(
            "OUT_OF_COMBAT_DELAY_STEPS_BY_CLASS",
            combat.OUT_OF_COMBAT_DELAY_STEPS_BY_CLASS.at[1].set(2**24 + 1),
            r"\[0, 16777216\]",
            id="delay-above-float32-exact-max",
        ),
        pytest.param(
            "OUT_OF_COMBAT_DELAY_STEPS_BY_CLASS",
            combat.OUT_OF_COMBAT_DELAY_STEPS_BY_CLASS.at[0].set(1),
            "neutral row",
            id="delay-nonneutral-padding",
        ),
        pytest.param(
            "OUT_OF_COMBAT_HEALTH_REGENERATION_FRACTION_PER_STEP_BY_CLASS",
            (
                combat.OUT_OF_COMBAT_HEALTH_REGENERATION_FRACTION_PER_STEP_BY_CLASS.at[
                    1
                ].set(jnp.nan)
            ),
            "finite",
            id="regeneration-nan",
        ),
        pytest.param(
            "OUT_OF_COMBAT_HEALTH_REGENERATION_FRACTION_PER_STEP_BY_CLASS",
            (
                combat.OUT_OF_COMBAT_HEALTH_REGENERATION_FRACTION_PER_STEP_BY_CLASS.at[
                    1
                ].set(jnp.inf)
            ),
            "finite",
            id="regeneration-infinity",
        ),
        pytest.param(
            "OUT_OF_COMBAT_HEALTH_REGENERATION_FRACTION_PER_STEP_BY_CLASS",
            (
                combat.OUT_OF_COMBAT_HEALTH_REGENERATION_FRACTION_PER_STEP_BY_CLASS.at[
                    1
                ].set(-0.01)
            ),
            r"\[0\.0, 1\.0\]",
            id="regeneration-negative",
        ),
        pytest.param(
            "OUT_OF_COMBAT_HEALTH_REGENERATION_FRACTION_PER_STEP_BY_CLASS",
            (
                combat.OUT_OF_COMBAT_HEALTH_REGENERATION_FRACTION_PER_STEP_BY_CLASS.at[
                    1
                ].set(1.01)
            ),
            r"\[0\.0, 1\.0\]",
            id="regeneration-above-one",
        ),
        pytest.param(
            "OUT_OF_COMBAT_HEALTH_REGENERATION_FRACTION_PER_STEP_BY_CLASS",
            (
                combat.OUT_OF_COMBAT_HEALTH_REGENERATION_FRACTION_PER_STEP_BY_CLASS.at[
                    0
                ].set(0.01)
            ),
            "neutral row",
            id="regeneration-nonneutral-padding",
        ),
    ),
)
def test_recovery_catalogs_enforce_value_domains_and_neutral_rows(
    monkeypatch: pytest.MonkeyPatch,
    catalog_name: str,
    invalid_catalog: Array,
    message: str,
) -> None:
    """Reject recovery tuning values outside their versioned catalog contract."""
    config = _valid_config()
    monkeypatch.setattr(combat, catalog_name, invalid_catalog)

    with pytest.raises(ValueError, match=message):
        validate_env_config(config)


def test_valid_alternate_recovery_catalogs_resolve_and_validate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep catalog values tunable while enforcing profile/catalog agreement."""
    alternate_delays = jnp.asarray((0, 1, 2, 3, 4, 5), dtype=jnp.int32)
    alternate_regeneration_fractions = jnp.asarray(
        (0.0, 0.01, 0.02, 0.03, 0.04, 1.0),
        dtype=jnp.float32,
    )
    monkeypatch.setattr(
        combat,
        "OUT_OF_COMBAT_DELAY_STEPS_BY_CLASS",
        alternate_delays,
    )
    monkeypatch.setattr(
        combat,
        "OUT_OF_COMBAT_HEALTH_REGENERATION_FRACTION_PER_STEP_BY_CLASS",
        alternate_regeneration_fractions,
    )

    assert validate_env_config(_valid_config(team_sizes=(5, 5))) is None


@pytest.mark.parametrize(
    ("field_name", "expected_dtype"),
    (
        ("class_ids", jnp.int32),
        ("team_ids", jnp.int32),
        ("active_mask", jnp.bool_),
        ("agent_radii", jnp.float32),
        ("base_movement_speeds", jnp.float32),
        ("observation_radii", jnp.float32),
        ("basic_interaction_radii", jnp.float32),
        ("ultimate_interaction_radii", jnp.float32),
        ("max_health", jnp.float32),
        ("out_of_combat_delay_steps", jnp.int32),
        ("out_of_combat_health_regen_fraction_per_step", jnp.float32),
    ),
)
def test_every_profile_field_rejects_wrong_storage(
    field_name: str, expected_dtype: object
) -> None:
    with pytest.raises(TypeError, match=field_name):
        validate_env_config(
            _replace_profile(_valid_config(), **{field_name: cast(Any, [])})
        )

    with pytest.raises(ValueError, match=field_name):
        validate_env_config(
            _replace_profile(
                _valid_config(),
                **{field_name: jnp.zeros((MAX_AGENT_SLOTS - 1,), dtype=jnp.float32)},
            )
        )

    wrong_dtype = jnp.int32 if expected_dtype == jnp.float32 else jnp.float32
    with pytest.raises(TypeError, match=field_name):
        validate_env_config(
            _replace_profile(
                _valid_config(),
                **{field_name: jnp.zeros((MAX_AGENT_SLOTS,), dtype=wrong_dtype)},
            )
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "agent_radii",
        "base_movement_speeds",
        "observation_radii",
        "basic_interaction_radii",
        "ultimate_interaction_radii",
        "max_health",
        "out_of_combat_health_regen_fraction_per_step",
    ),
)
def test_float_profile_fields_reject_dtype_nonfinite_and_catalog_drift(
    field_name: str,
) -> None:
    config = _valid_config()
    with pytest.raises(TypeError, match=field_name):
        validate_env_config(
            _replace_profile(
                config,
                **{field_name: jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32)},
            )
        )

    actual = cast(Array, getattr(config.agent_profile, field_name))
    with pytest.raises(ValueError, match=field_name):
        validate_env_config(
            _replace_profile(config, **{field_name: actual.at[0].set(jnp.nan)})
        )
    with pytest.raises(ValueError, match=field_name):
        validate_env_config(
            _replace_profile(config, **{field_name: actual.at[0].add(1.0)})
        )


@pytest.mark.parametrize(
    ("invalid_delay_steps", "message"),
    (
        pytest.param(
            lambda values: values.at[0].set(-1),
            r"\[0, 16777216\]",
            id="negative",
        ),
        pytest.param(
            lambda values: values.at[0].set(2**24 + 1),
            r"\[0, 16777216\]",
            id="above-float32-exact-max",
        ),
        pytest.param(
            lambda values: values.at[0].add(1),
            "resolved class catalog",
            id="catalog-drift",
        ),
    ),
)
def test_profile_recovery_delay_rejects_domain_and_catalog_drift(
    invalid_delay_steps: Callable[[Array], Array],
    message: str,
) -> None:
    """Enforce exact per-slot delay values resolved from the class catalog."""
    config = _valid_config()
    invalid_values = invalid_delay_steps(config.agent_profile.out_of_combat_delay_steps)

    with pytest.raises(ValueError, match=message):
        validate_env_config(
            _replace_profile(
                config,
                out_of_combat_delay_steps=invalid_values,
            )
        )


@pytest.mark.parametrize(
    ("replacement", "message"),
    (
        pytest.param(-0.01, r"\[0\.0, 1\.0\]", id="negative"),
        pytest.param(0.05, "resolved class catalog", id="catalog-drift"),
    ),
)
def test_profile_recovery_regeneration_rejects_range_and_catalog_drift(
    replacement: float,
    message: str,
) -> None:
    """Enforce bounded rates and exact per-slot catalog resolution."""
    config = _valid_config()
    invalid_values = (
        config.agent_profile.out_of_combat_health_regen_fraction_per_step.at[0].set(
            replacement
        )
    )

    with pytest.raises(ValueError, match=message):
        validate_env_config(
            _replace_profile(
                config,
                out_of_combat_health_regen_fraction_per_step=invalid_values,
            )
        )


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    (
        pytest.param(
            "out_of_combat_delay_steps",
            1,
            id="delay",
        ),
        pytest.param(
            "out_of_combat_health_regen_fraction_per_step",
            0.25,
            id="regeneration",
        ),
    ),
)
def test_inactive_profile_recovery_rows_must_be_canonical_zero(
    field_name: str,
    replacement: int | float,
) -> None:
    """Reject hidden recovery configuration in fixed-slot padding."""
    config = _valid_config(team_sizes=(1, 1))
    inactive_slot = 1
    values = cast(Array, getattr(config.agent_profile, field_name))
    invalid_values = values.at[inactive_slot].set(replacement)

    with pytest.raises(ValueError, match=rf"inactive agent_profile\.{field_name}"):
        validate_env_config(_replace_profile(config, **{field_name: invalid_values}))


def test_profile_categorical_and_padding_invariants_are_enforced() -> None:
    config = _valid_config(team_sizes=(2, 1))
    cases = (
        _replace_profile(
            config,
            class_ids=config.agent_profile.class_ids.at[0].set(99),
        ),
        _replace_profile(
            config,
            class_ids=config.agent_profile.class_ids.at[0].set(0),
        ),
        _replace_profile(
            config,
            class_ids=config.agent_profile.class_ids.at[2].set(MAGE_CLASS_ID),
        ),
        _replace_profile(
            config,
            team_ids=config.agent_profile.team_ids.at[0].set(99),
        ),
        _replace_profile(
            config,
            team_ids=config.agent_profile.team_ids.at[0].set(TEAM_A_ID + 1),
        ),
        _replace_profile(
            config,
            team_ids=config.agent_profile.team_ids.at[2].set(TEAM_A_ID),
        ),
        _replace_profile(
            config,
            active_mask=config.agent_profile.active_mask.at[0]
            .set(False)
            .at[2]
            .set(True),
        ),
    )
    for invalid_config in cases:
        with pytest.raises(ValueError, match="agent_profile"):
            validate_env_config(invalid_config)


def test_profile_active_rows_must_be_contiguous_team_prefixes() -> None:
    config = _valid_config(team_sizes=(2, 1))
    class_ids = config.agent_profile.class_ids.at[1].set(0).at[2].set(MAGE_CLASS_ID)
    active_mask = config.agent_profile.active_mask.at[1].set(False).at[2].set(True)
    profile = resolve_agent_profile(
        class_ids,
        jnp.asarray((3, 1), dtype=jnp.int32),
    )._replace(
        team_ids=config.agent_profile.team_ids.at[1]
        .set(NO_TEAM_ID)
        .at[2]
        .set(TEAM_A_ID),
        active_mask=active_mask,
    )

    with pytest.raises(ValueError, match="contiguous active prefix"):
        validate_env_config(config._replace(agent_profile=profile))


@pytest.mark.parametrize(
    ("invalid_positions", "error_type", "message"),
    (
        (cast(Any, []), TypeError, "team_spawn_pad_positions"),
        (
            jnp.zeros(
                (
                    NUM_TEAMS,
                    MAX_AGENTS_PER_TEAM - 1,
                    ENVIRONMENT_DIMENSIONS,
                ),
                dtype=jnp.float32,
            ),
            ValueError,
            "shape",
        ),
        (
            jnp.zeros(
                (NUM_TEAMS, MAX_AGENTS_PER_TEAM, ENVIRONMENT_DIMENSIONS),
                dtype=jnp.int32,
            ),
            TypeError,
            "dtype",
        ),
    ),
)
def test_spawn_pad_storage_contract_is_enforced(
    invalid_positions: object, error_type: type[Exception], message: str
) -> None:
    with pytest.raises(error_type, match=message):
        validate_env_config(
            _replace_config(
                _valid_config(),
                team_spawn_pad_positions=invalid_positions,
            )
        )


def test_spawn_pads_reject_nonfinite_bounds_and_fallback_formation_overlap() -> None:
    config = _valid_config(team_sizes=(2, 1))
    positions = config.team_spawn_pad_positions
    invalid_cases = (
        (positions.at[0, 0, 0].set(jnp.nan), "finite"),
        (positions.at[0, 0, 0].set(0.49), "bounds"),
        (positions.at[0, 1].set(positions[0, 0]), "fallback bodies overlap"),
        (
            positions.at[0, 4].set(positions[0, 0]),
            "fallback bodies overlap",
        ),
    )
    for invalid_positions, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            validate_env_config(
                config._replace(team_spawn_pad_positions=invalid_positions)
            )


def test_inactive_roster_slots_do_not_turn_real_spawn_pads_into_padding() -> None:
    """Keep all ten configured pad rows even when ordinary reset zeros padding."""
    config = _valid_config(team_sizes=(1, 1))
    assert bool(jnp.all(config.team_spawn_pad_positions != 0.0))
    assert validate_env_config(config) is None

    state, *_ = reset(config, jax.random.key(10))
    assert bool(
        jnp.all(
            state.agent_positions[jnp.logical_not(config.agent_profile.active_mask)]
            == 0.0
        )
    )


def test_every_spawn_pad_is_valid_for_its_largest_configured_team_body() -> None:
    """Validate dormant pad locations with a real same-team fallback body."""
    config = _valid_config(team_sizes=(1, 0))
    invalid_pads = config.team_spawn_pad_positions.at[0, 4, 0].set(0.49)

    with pytest.raises(ValueError, match=r"team_spawn_pad_positions.*bounds"):
        validate_env_config(config._replace(team_spawn_pad_positions=invalid_pads))


def test_spawn_pads_reject_pillar_and_rotated_wall_overlap() -> None:
    pillar_config = _valid_config(
        team_sizes=(1, 0),
        obstacles=_empty_obstacles().at[0].set(_pillar(x=2.0, y=2.0)),
    )
    pillar_positions = pillar_config.team_spawn_pad_positions.at[0, 0].set(
        jnp.asarray((2.0, 2.0), dtype=jnp.float32)
    )
    with pytest.raises(ValueError, match="pillar"):
        validate_env_config(
            pillar_config._replace(team_spawn_pad_positions=pillar_positions)
        )

    wall_config = _valid_config(
        team_sizes=(1, 0),
        obstacles=_empty_obstacles().at[0].set(_wall(theta=0.7)),
    )
    wall_positions = wall_config.team_spawn_pad_positions.at[0, 0].set(
        jnp.asarray((8.0, 4.0), dtype=jnp.float32)
    )
    with pytest.raises(ValueError, match="wall"):
        validate_env_config(
            wall_config._replace(team_spawn_pad_positions=wall_positions)
        )


@pytest.mark.parametrize(
    "obstacle",
    (
        pytest.param(
            _pillar(x=2.0e38, y=2.0e38, radius=3.0e38),
            id="pillar",
        ),
        pytest.param(
            _wall(
                x=2.0e38,
                y=2.0e38,
                width=3.0e38,
                height=3.0e38,
                theta=0.7,
            ),
            id="rotated-wall",
        ),
    ),
)
def test_extreme_finite_obstacle_projection_extent_is_rejected(
    obstacle: Array,
) -> None:
    config = _valid_config(
        team_sizes=(1, 0),
        obstacles=_empty_obstacles().at[0].set(obstacle),
    )

    with pytest.raises(ValueError, match="representable by float32"):
        validate_env_config(config)


def test_extreme_finite_pillar_overlap_uses_overflow_safe_distance() -> None:
    obstacle = _pillar(x=2.0e38, y=2.0e38, radius=1.0e38)
    config = _valid_config(
        team_sizes=(1, 0),
        obstacles=_empty_obstacles().at[0].set(obstacle),
    )
    positions = config.team_spawn_pad_positions.at[0, 0].set(
        jnp.asarray((1.5e38, 1.5e38), dtype=jnp.float32)
    )
    config = config._replace(
        map_width=3.0e38,
        map_height=3.0e38,
        team_spawn_pad_positions=positions,
    )

    with pytest.raises(ValueError, match="pillar"):
        validate_env_config(config)


def test_bounds_pillar_wall_and_agent_tangency_are_legal() -> None:
    profile = resolve_agent_profile(
        jnp.full((MAX_AGENT_SLOTS,), MAGE_CLASS_ID, dtype=jnp.int32),
        jnp.asarray((2, 0), dtype=jnp.int32),
    )
    obstacles = _empty_obstacles()
    obstacles = obstacles.at[0].set(_pillar(x=4.0, y=2.0, radius=1.0))
    obstacles = obstacles.at[1].set(
        _wall(x=8.0, y=2.0, width=2.0, height=1.0, theta=0.0)
    )
    positions = _spawn_pad_positions()
    positions = positions.at[0, 0].set(jnp.asarray((0.5, 0.5), dtype=jnp.float32))
    positions = positions.at[0, 1].set(jnp.asarray((1.5, 0.5), dtype=jnp.float32))
    config = EnvConfig(
        task_mode=0,
        team_deathmatch_score_threshold=0,
        max_steps=10,
        map_width=12.0,
        map_height=8.0,
        obstacles=obstacles,
        agent_profile=profile,
        ordinary_movement_distance_scale=1.0,
        team_spawn_pad_positions=positions,
        spawn_shield_duration_steps=3,
        spawn_shield_movement_speed=2.0,
        team_respawn_wave_period_step_count=jnp.asarray((5, 7), dtype=jnp.int32),
    )
    assert validate_env_config(config) is None

    pillar_tangent = (
        positions.at[0, 0]
        .set(jnp.asarray((2.5, 2.0), dtype=jnp.float32))
        .at[0, 1]
        .set(jnp.asarray((0.5, 0.5), dtype=jnp.float32))
    )
    assert (
        validate_env_config(config._replace(team_spawn_pad_positions=pillar_tangent))
        is None
    )

    wall_tangent = (
        positions.at[0, 0]
        .set(jnp.asarray((8.0, 3.0), dtype=jnp.float32))
        .at[0, 1]
        .set(jnp.asarray((0.5, 0.5), dtype=jnp.float32))
    )
    assert (
        validate_env_config(config._replace(team_spawn_pad_positions=wall_tangent))
        is None
    )


@pytest.mark.parametrize(
    ("center_distance", "is_valid"),
    (
        pytest.param(1.5, True, id="exact-tangency"),
        pytest.param(1.5 + GEOMETRY_EPSILON / 2.0, True, id="epsilon-separated"),
        pytest.param(1.5 - GEOMETRY_EPSILON / 2.0, False, id="epsilon-overlap"),
        pytest.param(1.5 - 2.0 * GEOMETRY_TOLERANCE, False, id="tolerance-overlap"),
    ),
)
def test_pillar_validation_matches_public_m4_projection(
    center_distance: float, is_valid: bool
) -> None:
    obstacle = _pillar(x=4.0, y=4.0, radius=1.0)
    obstacles = _empty_obstacles().at[0].set(obstacle)
    base_config = _valid_config(team_sizes=(1, 0), obstacles=obstacles)
    center = jnp.asarray((4.0 + center_distance, 4.0), dtype=jnp.float32)
    config = base_config._replace(
        team_spawn_pad_positions=base_config.team_spawn_pad_positions.at[0, 0].set(
            center
        )
    )
    projected_center = _public_geometry_projection(config, center)
    overlaps = bool(
        disc_overlaps_obstacle(
            center,
            config.agent_profile.agent_radii[0],
            obstacle,
        )
    )

    if is_valid:
        assert not overlaps
        assert validate_env_config(config) is None
        assert bool(jnp.array_equal(projected_center, center))
    else:
        assert overlaps
        with pytest.raises(ValueError, match="pillar"):
            validate_env_config(config)
        assert not bool(jnp.array_equal(projected_center, center))


@pytest.mark.parametrize(
    ("local_center", "is_valid"),
    (
        pytest.param(
            (0.0, 1.0),
            True,
            id="rotated-face-exact-tangency",
        ),
        pytest.param(
            (0.0, 1.0 + GEOMETRY_EPSILON),
            True,
            id="rotated-face-epsilon-separated",
        ),
        pytest.param(
            (0.0, 1.0 - 2.0 * GEOMETRY_TOLERANCE),
            False,
            id="rotated-face-tolerance-overlap",
        ),
        pytest.param(
            (0.0, 1.0 + 2.0 * GEOMETRY_TOLERANCE),
            True,
            id="rotated-face-tolerance-separated",
        ),
        pytest.param((1.0, 0.0), False, id="rotated-edge-center-contact"),
        pytest.param(
            (
                1.0 + 0.5 / 2.0**0.5,
                0.5 + 0.5 / 2.0**0.5,
            ),
            True,
            id="rotated-corner-exact-tangency",
        ),
        pytest.param(
            (
                1.0 + (0.5 - 2.0 * GEOMETRY_TOLERANCE) / 2.0**0.5,
                0.5 + (0.5 - 2.0 * GEOMETRY_TOLERANCE) / 2.0**0.5,
            ),
            False,
            id="rotated-corner-tolerance-overlap",
        ),
        pytest.param(
            (
                1.0 + (0.5 + 2.0 * GEOMETRY_TOLERANCE) / 2.0**0.5,
                0.5 + (0.5 + 2.0 * GEOMETRY_TOLERANCE) / 2.0**0.5,
            ),
            True,
            id="rotated-corner-tolerance-separated",
        ),
    ),
)
@pytest.mark.parametrize("wall_theta", (0.31, 0.7, -1.1))
def test_rotated_wall_validation_matches_public_m4_projection(
    local_center: tuple[float, float], is_valid: bool, wall_theta: float
) -> None:
    obstacle = _wall(x=6.0, y=4.0, width=2.0, height=1.0, theta=wall_theta)
    obstacles = _empty_obstacles().at[0].set(obstacle)
    base_config = _valid_config(team_sizes=(1, 0), obstacles=obstacles)
    center = _wall_local_to_world(obstacle, local_center)
    config = base_config._replace(
        team_spawn_pad_positions=base_config.team_spawn_pad_positions.at[0, 0].set(
            center
        )
    )
    projected_center = _public_geometry_projection(config, center)
    overlaps = bool(
        disc_overlaps_obstacle(
            center,
            config.agent_profile.agent_radii[0],
            obstacle,
        )
    )

    if is_valid:
        assert not overlaps
        assert validate_env_config(config) is None
        assert bool(jnp.array_equal(projected_center, center))
    else:
        assert overlaps
        with pytest.raises(ValueError, match="wall"):
            validate_env_config(config)
        assert not bool(jnp.array_equal(projected_center, center))


@pytest.mark.parametrize(
    ("wall_theta", "center"),
    (
        pytest.param(0.7, (1.1, 1.3), id="lower-left-non-roundtrip"),
        pytest.param(0.31, (0.75, 0.75), id="lower-left-quarter-grid"),
        pytest.param(-1.1, (2.2, 6.4), id="upper-left-non-roundtrip"),
    ),
)
def test_distant_rotated_wall_clearance_ignores_projection_roundoff(
    wall_theta: float,
    center: tuple[float, float],
) -> None:
    obstacle = _wall(x=8.0, y=4.0, width=2.0, height=1.0, theta=wall_theta)
    config = _valid_config(
        team_sizes=(1, 0),
        obstacles=_empty_obstacles().at[0].set(obstacle),
    )
    agent_center = jnp.asarray(center, dtype=jnp.float32)
    config = config._replace(
        team_spawn_pad_positions=config.team_spawn_pad_positions.at[0, 0].set(
            agent_center
        )
    )

    assert not bool(
        disc_overlaps_obstacle(
            agent_center,
            config.agent_profile.agent_radii[0],
            obstacle,
        )
    )
    assert validate_env_config(config) is None
    assert bool(
        jnp.array_equal(_public_geometry_projection(config, agent_center), agent_center)
    )


def test_reset_is_key_independent_jittable_and_stay_needs_no_repair() -> None:
    obstacles = _empty_obstacles().at[0].set(_pillar()).at[1].set(_wall())
    config = _valid_config(team_sizes=(5, 5), obstacles=obstacles)
    validate_env_config(config)

    first = reset(config, jax.random.key(1))
    second = reset(config, jax.random.key(999))
    jitted = cast(
        tuple[EnvState, Observation, ActionMask, Info],
        jax.jit(reset)(config, jax.random.key(7)),
    )
    _assert_pytrees_equal(first, second)
    _assert_pytrees_equal(first, jitted)
    configured_positions = config.team_spawn_pad_positions.reshape(
        MAX_AGENT_SLOTS, ENVIRONMENT_DIMENSIONS
    )
    expected_reset_positions = jnp.where(
        config.agent_profile.active_mask[:, None],
        configured_positions,
        0.0,
    )
    assert bool(jnp.array_equal(first[0].agent_positions, expected_reset_positions))

    stay_action = Action(
        move=jnp.full((MAX_AGENT_SLOTS,), MOVE_STAY, dtype=jnp.int32),
        select_target=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
        use_ultimate=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
    )
    next_state, *_ = step(
        config,
        first[0],
        first[2],
        stay_action,
        jax.random.key(3),
    )
    assert bool(jnp.array_equal(next_state.agent_positions, expected_reset_positions))
