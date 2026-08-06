"""Host-only validation for runtime snapshots and curated scenario starts."""
# pyright: reportUnknownArgumentType=false
# pyright: reportPrivateUsage=false

from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
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
    get_ultimate_cooldown_by_class_ids,
)
from marl_battlegrounds.core.config import (
    resolve_agent_profile,
    validate_env_state,
    validate_scenario_initial_state,
)
from marl_battlegrounds.core.env import (
    _build_observation_and_action_mask,
    reset,
    step,
)
from marl_battlegrounds.core.geometry import GEOMETRY_TOLERANCE
from marl_battlegrounds.core.types import (
    ENVIRONMENT_DIMENSIONS,
    HUNTER_CLASS_ID,
    MAGE_CLASS_ID,
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    MAX_OBSTACLE_SLOTS,
    MOVE_EAST,
    MOVE_STAY,
    MOVE_WEST,
    NUM_MOVE_ACTIONS,
    NUM_SLOW_CHANNELS,
    NUM_STUN_CHANNELS,
    NUM_TARGET_ACTIONS,
    NUM_TEAMS,
    NUM_ULTIMATE_ACTIONS,
    OBSTACLE_FEATURE_ACTIVE,
    OBSTACLE_FEATURE_HEIGHT,
    OBSTACLE_FEATURE_RADIUS,
    OBSTACLE_FEATURE_TYPE,
    OBSTACLE_FEATURE_WIDTH,
    OBSTACLE_FEATURE_X,
    OBSTACLE_FEATURE_Y,
    OBSTACLE_FEATURES,
    OBSTACLE_TYPE_PILLAR,
    OBSTACLE_TYPE_WALL,
    PRIEST_CLASS_ID,
    ROGUE_CLASS_ID,
    WARRIOR_CLASS_ID,
    Action,
    EnvConfig,
    EnvState,
)

_CANONICAL_TEAM_CLASSES = (
    MAGE_CLASS_ID,
    WARRIOR_CLASS_ID,
    HUNTER_CLASS_ID,
    ROGUE_CLASS_ID,
    PRIEST_CLASS_ID,
)

_STATE_ARRAY_FIELDS = (
    ("step_count", (), jnp.int32),
    (
        "agent_positions",
        (MAX_AGENT_SLOTS, ENVIRONMENT_DIMENSIONS),
        jnp.float32,
    ),
    ("alive_mask", (MAX_AGENT_SLOTS,), jnp.bool_),
    ("current_health", (MAX_AGENT_SLOTS,), jnp.float32),
    ("ultimate_cooldowns", (MAX_AGENT_SLOTS,), jnp.int32),
    (
        "slow_durations",
        (MAX_AGENT_SLOTS, NUM_SLOW_CHANNELS),
        jnp.int32,
    ),
    (
        "stun_durations",
        (MAX_AGENT_SLOTS, NUM_STUN_CHANNELS),
        jnp.int32,
    ),
    ("rogue_poison_anti_heal_durations", (MAX_AGENT_SLOTS,), jnp.int32),
    (
        "mage_burst_damage_amplification_durations",
        (MAX_AGENT_SLOTS,),
        jnp.int32,
    ),
    (
        "priest_blessing_of_freedom_slow_floor_durations",
        (MAX_AGENT_SLOTS,),
        jnp.int32,
    ),
    ("team_respawn_wave_countdowns", (NUM_TEAMS,), jnp.int32),
    ("spawn_shield_durations", (MAX_AGENT_SLOTS,), jnp.int32),
    ("previous_timestep_move_actions", (MAX_AGENT_SLOTS,), jnp.int32),
    (
        "previous_timestep_select_target_actions",
        (MAX_AGENT_SLOTS,),
        jnp.int32,
    ),
    (
        "previous_timestep_use_ultimate_actions",
        (MAX_AGENT_SLOTS,),
        jnp.int32,
    ),
    ("has_previous_timestep_joint_action", (), jnp.bool_),
)


def _empty_obstacles() -> Array:
    """Return a canonical inactive obstacle table."""
    return jnp.zeros((MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES), dtype=jnp.float32)


def _pillar(*, x: float = 15.0, y: float = 10.0, radius: float = 1.0) -> Array:
    """Return one active circular obstacle row."""
    obstacle = jnp.zeros((OBSTACLE_FEATURES,), dtype=jnp.float32)
    obstacle = obstacle.at[OBSTACLE_FEATURE_TYPE].set(OBSTACLE_TYPE_PILLAR)
    obstacle = obstacle.at[OBSTACLE_FEATURE_X].set(x)
    obstacle = obstacle.at[OBSTACLE_FEATURE_Y].set(y)
    obstacle = obstacle.at[OBSTACLE_FEATURE_RADIUS].set(radius)
    return obstacle.at[OBSTACLE_FEATURE_ACTIVE].set(1.0)


def _wall(
    *,
    x: float = 15.0,
    y: float = 10.0,
    width: float = 2.0,
    height: float = 1.0,
) -> Array:
    """Return one active axis-aligned wall obstacle row."""
    obstacle = jnp.zeros((OBSTACLE_FEATURES,), dtype=jnp.float32)
    obstacle = obstacle.at[OBSTACLE_FEATURE_TYPE].set(OBSTACLE_TYPE_WALL)
    obstacle = obstacle.at[OBSTACLE_FEATURE_X].set(x)
    obstacle = obstacle.at[OBSTACLE_FEATURE_Y].set(y)
    obstacle = obstacle.at[OBSTACLE_FEATURE_WIDTH].set(width)
    obstacle = obstacle.at[OBSTACLE_FEATURE_HEIGHT].set(height)
    return obstacle.at[OBSTACLE_FEATURE_ACTIVE].set(1.0)


def _valid_config(
    *,
    team_sizes: tuple[int, int] = (5, 5),
    obstacles: Array | None = None,
) -> EnvConfig:
    """Build a deterministic catalog-valid config for state validation."""
    requested_classes = jnp.asarray(
        _CANONICAL_TEAM_CLASSES + _CANONICAL_TEAM_CLASSES,
        dtype=jnp.int32,
    )
    profile = resolve_agent_profile(
        requested_classes,
        jnp.asarray(team_sizes, dtype=jnp.int32),
    )
    spawn_pad_positions = jnp.asarray(
        (
            (2.0, 2.0),
            (2.0, 5.0),
            (2.0, 8.0),
            (2.0, 11.0),
            (2.0, 14.0),
            (28.0, 2.0),
            (28.0, 5.0),
            (28.0, 8.0),
            (28.0, 11.0),
            (28.0, 14.0),
        ),
        dtype=jnp.float32,
    ).reshape(2, MAX_AGENTS_PER_TEAM, ENVIRONMENT_DIMENSIONS)
    return EnvConfig(
        max_steps=100,
        map_width=30.0,
        map_height=20.0,
        obstacles=_empty_obstacles() if obstacles is None else obstacles,
        agent_profile=profile,
        ordinary_movement_distance_scale=1.0,
        team_spawn_pad_positions=spawn_pad_positions,
        spawn_shield_duration_steps=3,
        spawn_shield_movement_speed=2.0,
        team_respawn_wave_period_step_count=jnp.asarray(
            (5, 7),
            dtype=jnp.int32,
        ),
    )


def _valid_state(
    *,
    team_sizes: tuple[int, int] = (5, 5),
    obstacles: Array | None = None,
) -> tuple[EnvConfig, EnvState]:
    """Return a reset-produced official config/state pair."""
    config = _valid_config(team_sizes=team_sizes, obstacles=obstacles)
    state, _, _, _ = reset(config, jax.random.key(0))
    return config, state


def _replace_slot_value(
    state: EnvState,
    field_name: str,
    *,
    slot: int,
    value: int | float | bool,
    channel: int | None = None,
) -> EnvState:
    """Replace one vector or matrix entry in a state field."""
    field = cast(Array, getattr(state, field_name))
    if channel is None:
        updated = field.at[slot].set(value)
    else:
        updated = field.at[slot, channel].set(value)
    return state._replace(**{field_name: updated})


def _assert_pytrees_equal(left: object, right: object) -> None:
    """Assert exact equality without relying on container identity."""
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    assert len(left_leaves) == len(right_leaves)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        assert bool(jnp.array_equal(left_leaf, right_leaf))


def test_valid_living_and_dead_states_return_none_without_mutation() -> None:
    """Accept official states while preserving every input leaf."""
    config, living_state = _valid_state()
    shielded_living_state = living_state._replace(
        spawn_shield_durations=living_state.spawn_shield_durations.at[0].set(
            config.spawn_shield_duration_steps
        )
    )
    dead_slot = 0
    dead_state = living_state._replace(
        alive_mask=living_state.alive_mask.at[dead_slot].set(False),
        current_health=living_state.current_health.at[dead_slot].set(0.0),
        ultimate_cooldowns=living_state.ultimate_cooldowns.at[dead_slot].set(
            get_ultimate_cooldown_by_class_ids(
                config.agent_profile.class_ids[dead_slot]
            )
        ),
        previous_timestep_move_actions=(
            living_state.previous_timestep_move_actions.at[dead_slot].set(MOVE_EAST)
        ),
        previous_timestep_select_target_actions=(
            living_state.previous_timestep_select_target_actions.at[dead_slot].set(0)
        ),
        previous_timestep_use_ultimate_actions=(
            living_state.previous_timestep_use_ultimate_actions.at[dead_slot].set(1)
        ),
        has_previous_timestep_joint_action=jnp.asarray(True),
    )

    for state in (living_state, shielded_living_state, dead_state):
        before = state
        assert validate_env_state(config, state) is None
        _assert_pytrees_equal(state, before)


def test_validator_rejects_wrong_top_level_types() -> None:
    """Reject non-contract config and state containers before field access."""
    config, state = _valid_state()
    with pytest.raises(TypeError, match="config"):
        validate_env_state(cast(Any, object()), state)
    with pytest.raises(TypeError, match="state"):
        validate_env_state(config, cast(Any, object()))


@pytest.mark.parametrize(
    ("field_name", "expected_shape", "expected_dtype"),
    _STATE_ARRAY_FIELDS,
)
def test_every_state_leaf_rejects_nonarray_shape_and_dtype_drift(
    field_name: str,
    expected_shape: tuple[int, ...],
    expected_dtype: object,
) -> None:
    """Enforce exact JAX storage for every public EnvState leaf."""
    config, state = _valid_state()

    with pytest.raises(TypeError, match=field_name):
        validate_env_state(
            config,
            state._replace(**{field_name: cast(Any, [])}),
        )

    with pytest.raises(ValueError, match=field_name):
        validate_env_state(
            config,
            state._replace(**{field_name: jnp.zeros((1,), dtype=jnp.float32)}),
        )

    wrong_dtype = jnp.int32 if expected_dtype == jnp.float32 else jnp.float32
    with pytest.raises(TypeError, match=field_name):
        validate_env_state(
            config,
            state._replace(
                **{field_name: jnp.zeros(expected_shape, dtype=wrong_dtype)}
            ),
        )


@pytest.mark.parametrize(
    ("field_name", "nonfinite_value"),
    (
        pytest.param("agent_positions", jnp.nan, id="position-nan"),
        pytest.param("agent_positions", jnp.inf, id="position-infinity"),
        pytest.param("current_health", jnp.nan, id="health-nan"),
        pytest.param("current_health", jnp.inf, id="health-infinity"),
    ),
)
def test_float_state_fields_reject_nonfinite_values(
    field_name: str,
    nonfinite_value: float,
) -> None:
    """Reject nonfinite geometry and health before simulator entry."""
    config, state = _valid_state()
    field = cast(Array, getattr(state, field_name))
    invalid_field = (
        field.at[0, 0].set(nonfinite_value)
        if field.ndim == 2
        else field.at[0].set(nonfinite_value)
    )

    with pytest.raises(ValueError, match=field_name):
        validate_env_state(config, state._replace(**{field_name: invalid_field}))


def test_step_count_must_be_nonnegative() -> None:
    """Reject a transition-start step count outside the nonnegative domain."""
    config, state = _valid_state()
    with pytest.raises(ValueError, match="step_count"):
        validate_env_state(
            config,
            state._replace(step_count=jnp.asarray(-1, dtype=jnp.int32)),
        )


def test_liveness_must_be_a_subset_of_configured_activity() -> None:
    """Reject a live flag on canonical inactive padding."""
    config, state = _valid_state(team_sizes=(1, 1))
    invalid_state = state._replace(
        alive_mask=state.alive_mask.at[1].set(True),
        current_health=state.current_health.at[1].set(1.0),
    )
    with pytest.raises(ValueError, match="alive_mask"):
        validate_env_state(config, invalid_state)


def test_reset_respawn_wave_countdowns_match_each_team_period() -> None:
    """Accept the official reset state at each period's inclusive upper bound."""
    config, state = _valid_state()

    assert bool(
        jnp.array_equal(
            state.team_respawn_wave_countdowns,
            config.team_respawn_wave_period_step_count - 1,
        )
    )
    assert validate_env_state(config, state) is None


@pytest.mark.parametrize(
    ("period_step_counts", "countdowns"),
    (
        pytest.param((1, 1), (0, 0), id="wave-every-transition"),
        pytest.param((1, 7), (0, 6), id="asymmetric-upper-bounds"),
        pytest.param((5, 7), (2, 0), id="independent-interior-and-due"),
        pytest.param(
            (np.iinfo(np.int32).max, np.iinfo(np.int32).max),
            (np.iinfo(np.int32).max - 1, np.iinfo(np.int32).max - 1),
            id="int32-boundary",
        ),
    ),
)
def test_respawn_wave_countdowns_accept_each_team_domain(
    period_step_counts: tuple[int, int],
    countdowns: tuple[int, int],
) -> None:
    """Accept independent countdowns in each team's half-open period domain."""
    config, state = _valid_state()
    config = config._replace(
        team_respawn_wave_period_step_count=jnp.asarray(
            period_step_counts,
            dtype=jnp.int32,
        )
    )
    state = state._replace(
        team_respawn_wave_countdowns=jnp.asarray(countdowns, dtype=jnp.int32)
    )

    assert validate_env_state(config, state) is None


@pytest.mark.parametrize(
    ("period_step_counts", "countdowns", "message"),
    (
        pytest.param((5, 7), (-1, 0), "nonnegative", id="team-a-negative"),
        pytest.param((5, 7), (0, -1), "nonnegative", id="team-b-negative"),
        pytest.param(
            (5, 7),
            (5, 0),
            "strictly less",
            id="team-a-equals-period",
        ),
        pytest.param(
            (5, 7),
            (0, 7),
            "strictly less",
            id="team-b-equals-period",
        ),
        pytest.param(
            (5, 7),
            (6, 0),
            "strictly less",
            id="team-a-exceeds-period",
        ),
        pytest.param(
            (5, 7),
            (0, 8),
            "strictly less",
            id="team-b-exceeds-period",
        ),
    ),
)
def test_respawn_wave_countdowns_reject_values_outside_each_team_period(
    period_step_counts: tuple[int, int],
    countdowns: tuple[int, int],
    message: str,
) -> None:
    """Reject underflow and elementwise values outside the half-open domain."""
    config, state = _valid_state()
    config = config._replace(
        team_respawn_wave_period_step_count=jnp.asarray(
            period_step_counts,
            dtype=jnp.int32,
        )
    )
    state = state._replace(
        team_respawn_wave_countdowns=jnp.asarray(countdowns, dtype=jnp.int32)
    )

    with pytest.raises(ValueError, match=message):
        validate_env_state(config, state)


@pytest.mark.parametrize(
    ("duration", "counter"),
    (
        pytest.param(0, 0, id="disabled"),
        pytest.param(1, 1, id="single-transition"),
        pytest.param(3, 3, id="official-maximum"),
        pytest.param(7, 4, id="larger-config-intermediate"),
    ),
)
def test_spawn_shield_counter_accepts_configured_living_domain(
    duration: int,
    counter: int,
) -> None:
    """Accept any living counter in the closed configured duration range."""
    config, state = _valid_state()
    config = config._replace(spawn_shield_duration_steps=duration)
    state = state._replace(
        spawn_shield_durations=state.spawn_shield_durations.at[0].set(counter)
    )

    assert validate_env_state(config, state) is None


@pytest.mark.parametrize(
    ("counter", "message"),
    (
        pytest.param(-1, "nonnegative", id="negative"),
        pytest.param(4, "spawn_shield_duration_steps", id="above-configured-duration"),
    ),
)
def test_spawn_shield_counter_rejects_values_outside_configured_domain(
    counter: int,
    message: str,
) -> None:
    """Reject underflow and counters exceeding the public config authority."""
    config, state = _valid_state()
    invalid_state = state._replace(
        spawn_shield_durations=state.spawn_shield_durations.at[0].set(counter)
    )

    with pytest.raises(ValueError, match=message):
        validate_env_state(config, invalid_state)


def test_spawn_shield_counter_must_be_zero_for_dead_and_inactive_slots() -> None:
    """Prevent protection memory from surviving death or entering padding."""
    config, state = _valid_state(team_sizes=(1, 1))
    dead_state = state._replace(
        alive_mask=state.alive_mask.at[0].set(False),
        current_health=state.current_health.at[0].set(0.0),
        spawn_shield_durations=state.spawn_shield_durations.at[0].set(1),
    )
    with pytest.raises(ValueError, match="dead spawn_shield_durations"):
        validate_env_state(config, dead_state)

    inactive_state = state._replace(
        spawn_shield_durations=state.spawn_shield_durations.at[1].set(1)
    )
    with pytest.raises(ValueError, match="inactive spawn_shield_durations"):
        validate_env_state(config, inactive_state)


def test_disabled_spawn_shield_requires_all_counter_rows_to_be_zero() -> None:
    """Make duration zero a complete ablation rather than a partial state."""
    config, state = _valid_state()
    disabled_config = config._replace(spawn_shield_duration_steps=0)
    invalid_state = state._replace(
        spawn_shield_durations=state.spawn_shield_durations.at[0].set(1)
    )

    with pytest.raises(ValueError, match="spawn_shield_duration_steps"):
        validate_env_state(disabled_config, invalid_state)


@pytest.mark.parametrize(
    "stun_channels",
    (
        pytest.param((0,), id="warrior-charge"),
        pytest.param((1,), id="hunter-trap"),
        pytest.param((2,), id="rogue-poison"),
        pytest.param(tuple(range(NUM_STUN_CHANNELS)), id="multiple-channels"),
    ),
)
@pytest.mark.parametrize(
    "spawn_shield_duration",
    (
        pytest.param(1, id="one-step-remaining"),
        pytest.param(3, id="configured-maximum"),
    ),
)
def test_spawn_shield_and_stun_cannot_coexist_on_one_living_slot(
    stun_channels: tuple[int, ...],
    spawn_shield_duration: int,
) -> None:
    """Reject lifecycle states that combine shielding with any active stun."""
    config, state = _valid_state()
    stun_durations = state.stun_durations
    for channel in stun_channels:
        stun_durations = stun_durations.at[0, channel].set(1)
    invalid_state = state._replace(
        stun_durations=stun_durations,
        spawn_shield_durations=state.spawn_shield_durations.at[0].set(
            spawn_shield_duration
        ),
    )

    with pytest.raises(
        ValueError,
        match="spawn_shield_durations and stun_durations",
    ):
        validate_env_state(config, invalid_state)


def test_spawn_shield_stun_exclusion_preserves_each_independent_state() -> None:
    """Accept shielding without stun and stun without shielding."""
    config, state = _valid_state()
    shielded_state = state._replace(
        spawn_shield_durations=state.spawn_shield_durations.at[0].set(
            config.spawn_shield_duration_steps
        )
    )
    stunned_state = state._replace(
        stun_durations=state.stun_durations.at[0].set(
            jnp.ones((NUM_STUN_CHANNELS,), dtype=jnp.int32)
        )
    )

    assert validate_env_state(config, shielded_state) is None
    assert validate_env_state(config, stunned_state) is None


@pytest.mark.parametrize(
    "health_case",
    (
        pytest.param("zero", id="alive-zero"),
        pytest.param("negative", id="alive-negative"),
        pytest.param("above-maximum", id="alive-above-maximum"),
    ),
)
def test_living_health_must_be_strictly_positive_and_bounded(
    health_case: str,
) -> None:
    """Enforce the official living-health half of the lifecycle invariant."""
    config, state = _valid_state()
    if health_case == "zero":
        invalid_health = 0.0
    elif health_case == "negative":
        invalid_health = -1.0
    else:
        invalid_health = float(config.agent_profile.max_health[0]) + 1.0
    invalid_state = _replace_slot_value(
        state,
        "current_health",
        slot=0,
        value=invalid_health,
    )

    with pytest.raises(ValueError, match="current_health"):
        validate_env_state(config, invalid_state)


def test_dead_health_must_be_exactly_zero() -> None:
    """Reject an active corpse carrying positive health."""
    config, state = _valid_state()
    invalid_state = state._replace(alive_mask=state.alive_mask.at[0].set(False))
    with pytest.raises(ValueError, match="dead current_health"):
        validate_env_state(config, invalid_state)


@pytest.mark.parametrize(
    ("field_name", "channel"),
    (
        pytest.param("slow_durations", 0, id="slow"),
        pytest.param("stun_durations", 0, id="stun"),
        pytest.param(
            "rogue_poison_anti_heal_durations",
            None,
            id="rogue-anti-heal",
        ),
        pytest.param(
            "mage_burst_damage_amplification_durations",
            None,
            id="mage-burst",
        ),
        pytest.param(
            "priest_blessing_of_freedom_slow_floor_durations",
            None,
            id="priest-freedom",
        ),
    ),
)
def test_dead_transient_statuses_must_be_zero(
    field_name: str,
    channel: int | None,
) -> None:
    """Reject every transient status family on an active corpse."""
    config, state = _valid_state()
    dead_state = state._replace(
        alive_mask=state.alive_mask.at[0].set(False),
        current_health=state.current_health.at[0].set(0.0),
    )
    invalid_state = _replace_slot_value(
        dead_state,
        field_name,
        slot=0,
        channel=channel,
        value=1,
    )

    with pytest.raises(ValueError, match=field_name):
        validate_env_state(config, invalid_state)


@pytest.mark.parametrize(
    ("field_name", "channel"),
    (
        pytest.param("ultimate_cooldowns", None, id="cooldown"),
        pytest.param("slow_durations", 0, id="slow"),
        pytest.param("stun_durations", 0, id="stun"),
        pytest.param(
            "rogue_poison_anti_heal_durations",
            None,
            id="rogue-anti-heal",
        ),
        pytest.param(
            "mage_burst_damage_amplification_durations",
            None,
            id="mage-burst",
        ),
        pytest.param(
            "priest_blessing_of_freedom_slow_floor_durations",
            None,
            id="priest-freedom",
        ),
    ),
)
def test_cooldown_and_status_durations_must_be_nonnegative(
    field_name: str,
    channel: int | None,
) -> None:
    """Reject negative tick counters in every lifecycle family."""
    config, state = _valid_state()
    invalid_state = _replace_slot_value(
        state,
        field_name,
        slot=0,
        channel=channel,
        value=-1,
    )

    with pytest.raises(ValueError, match=field_name):
        validate_env_state(config, invalid_state)


@pytest.mark.parametrize(
    ("slot", "class_id"),
    tuple(enumerate(_CANONICAL_TEAM_CLASSES)),
)
def test_each_class_cooldown_rejects_values_above_its_catalog_maximum(
    slot: int,
    class_id: int,
) -> None:
    """Use the configured actor class, not one global cooldown ceiling."""
    config, state = _valid_state()
    assert int(config.agent_profile.class_ids[slot]) == class_id
    maximum = int(
        get_ultimate_cooldown_by_class_ids(config.agent_profile.class_ids[slot])
    )
    invalid_state = _replace_slot_value(
        state,
        "ultimate_cooldowns",
        slot=slot,
        value=maximum + 1,
    )

    with pytest.raises(ValueError, match="ultimate_cooldowns"):
        validate_env_state(config, invalid_state)


@pytest.mark.parametrize(
    ("slot", "class_id"),
    tuple(enumerate(_CANONICAL_TEAM_CLASSES)),
)
def test_each_class_cooldown_catalog_maximum_is_inclusive(
    slot: int,
    class_id: int,
) -> None:
    """Accept the full cooldown value that an Ultimate starts with."""
    config, state = _valid_state()
    assert int(config.agent_profile.class_ids[slot]) == class_id
    maximum = get_ultimate_cooldown_by_class_ids(config.agent_profile.class_ids[slot])
    valid_state = _replace_slot_value(
        state,
        "ultimate_cooldowns",
        slot=slot,
        value=int(maximum),
    )
    assert validate_env_state(config, valid_state) is None


@pytest.mark.parametrize(
    ("field_name", "channel", "maximum"),
    (
        pytest.param(
            "slow_durations",
            0,
            WARRIOR_CHARGE_SLOW_DURATION_TICKS,
            id="warrior-charge-slow",
        ),
        pytest.param(
            "slow_durations",
            1,
            HUNTER_BASIC_SLOW_DURATION_TICKS,
            id="hunter-basic-slow",
        ),
        pytest.param(
            "slow_durations",
            2,
            ROGUE_POISON_SLOW_DURATION_TICKS,
            id="rogue-poison-slow",
        ),
        pytest.param(
            "stun_durations",
            0,
            WARRIOR_CHARGE_STUN_DURATION_TICKS,
            id="warrior-charge-stun",
        ),
        pytest.param(
            "stun_durations",
            1,
            HUNTER_TRAP_STUN_DURATION_TICKS,
            id="hunter-trap-stun",
        ),
        pytest.param(
            "stun_durations",
            2,
            ROGUE_POISON_STUN_DURATION_TICKS,
            id="rogue-poison-stun",
        ),
        pytest.param(
            "rogue_poison_anti_heal_durations",
            None,
            ROGUE_POISON_ANTI_HEAL_DURATION_TICKS,
            id="rogue-anti-heal",
        ),
        pytest.param(
            "mage_burst_damage_amplification_durations",
            None,
            MAGE_BURST_DAMAGE_DURATION_TICKS,
            id="mage-burst",
        ),
        pytest.param(
            "priest_blessing_of_freedom_slow_floor_durations",
            None,
            PRIEST_HEAL_SPEED_FLOOR_DURATION_TICKS,
            id="priest-freedom",
        ),
    ),
)
def test_each_status_channel_rejects_values_above_its_catalog_maximum(
    field_name: str,
    channel: int | None,
    maximum: int,
) -> None:
    """Enforce every independent public duration ceiling."""
    config, state = _valid_state()
    invalid_state = _replace_slot_value(
        state,
        field_name,
        slot=0,
        channel=channel,
        value=maximum + 1,
    )

    with pytest.raises(ValueError, match=field_name):
        validate_env_state(config, invalid_state)


def test_catalog_duration_maxima_are_inclusive_for_living_agents() -> None:
    """Accept fresh full durations rather than treating maxima as overflow."""
    config, state = _valid_state()
    valid_state = state._replace(
        slow_durations=state.slow_durations.at[0].set(
            jnp.asarray(
                (
                    WARRIOR_CHARGE_SLOW_DURATION_TICKS,
                    HUNTER_BASIC_SLOW_DURATION_TICKS,
                    ROGUE_POISON_SLOW_DURATION_TICKS,
                ),
                dtype=jnp.int32,
            )
        ),
        stun_durations=state.stun_durations.at[0].set(
            jnp.asarray(
                (
                    WARRIOR_CHARGE_STUN_DURATION_TICKS,
                    HUNTER_TRAP_STUN_DURATION_TICKS,
                    ROGUE_POISON_STUN_DURATION_TICKS,
                ),
                dtype=jnp.int32,
            )
        ),
        rogue_poison_anti_heal_durations=(
            state.rogue_poison_anti_heal_durations.at[0].set(
                ROGUE_POISON_ANTI_HEAL_DURATION_TICKS
            )
        ),
        mage_burst_damage_amplification_durations=(
            state.mage_burst_damage_amplification_durations.at[0].set(
                MAGE_BURST_DAMAGE_DURATION_TICKS
            )
        ),
        priest_blessing_of_freedom_slow_floor_durations=(
            state.priest_blessing_of_freedom_slow_floor_durations.at[0].set(
                PRIEST_HEAL_SPEED_FLOOR_DURATION_TICKS
            )
        ),
    )
    assert validate_env_state(config, valid_state) is None


def test_mage_burst_duration_is_owned_only_by_configured_mages() -> None:
    """Reject source-local Burst memory on a class that cannot create it."""
    config, state = _valid_state()
    non_mage_slot = 1
    assert int(config.agent_profile.class_ids[non_mage_slot]) != MAGE_CLASS_ID
    invalid_state = state._replace(
        mage_burst_damage_amplification_durations=(
            state.mage_burst_damage_amplification_durations.at[non_mage_slot].set(1)
        )
    )

    with pytest.raises(ValueError, match="configured Mage slots"):
        validate_env_state(config, invalid_state)


@pytest.mark.parametrize(
    ("field_name", "channel"),
    (
        pytest.param("current_health", None, id="health"),
        pytest.param("ultimate_cooldowns", None, id="cooldown"),
        pytest.param("slow_durations", 0, id="slow"),
        pytest.param("stun_durations", 0, id="stun"),
        pytest.param(
            "rogue_poison_anti_heal_durations",
            None,
            id="rogue-anti-heal",
        ),
        pytest.param(
            "mage_burst_damage_amplification_durations",
            None,
            id="mage-burst",
        ),
        pytest.param(
            "priest_blessing_of_freedom_slow_floor_durations",
            None,
            id="priest-freedom",
        ),
        pytest.param(
            "previous_timestep_move_actions",
            None,
            id="previous-move",
        ),
        pytest.param(
            "previous_timestep_select_target_actions",
            None,
            id="previous-target",
        ),
        pytest.param(
            "previous_timestep_use_ultimate_actions",
            None,
            id="previous-ultimate",
        ),
    ),
)
def test_inactive_dynamic_rows_must_remain_canonical_zero(
    field_name: str,
    channel: int | None,
) -> None:
    """Reject nonzero dynamic memory in configured padding."""
    config, state = _valid_state(team_sizes=(1, 1))
    inactive_slot = 1
    invalid_state = _replace_slot_value(
        state,
        field_name,
        slot=inactive_slot,
        channel=channel,
        value=1,
    )

    with pytest.raises(ValueError, match=field_name):
        validate_env_state(config, invalid_state)


def test_inactive_positions_must_remain_canonical_zero() -> None:
    """Reject hidden geometry in a configured padding row."""
    config, state = _valid_state(team_sizes=(1, 1))
    invalid_state = state._replace(
        agent_positions=state.agent_positions.at[1].set(
            jnp.asarray((1.0, 1.0), dtype=jnp.float32)
        )
    )
    with pytest.raises(ValueError, match="inactive agent_positions"):
        validate_env_state(config, invalid_state)


@pytest.mark.parametrize(
    ("field_name", "category_count"),
    (
        pytest.param(
            "previous_timestep_move_actions",
            NUM_MOVE_ACTIONS,
            id="move",
        ),
        pytest.param(
            "previous_timestep_select_target_actions",
            NUM_TARGET_ACTIONS,
            id="target",
        ),
        pytest.param(
            "previous_timestep_use_ultimate_actions",
            NUM_ULTIMATE_ACTIONS,
            id="ultimate",
        ),
    ),
)
@pytest.mark.parametrize("boundary", ("below", "above"))
def test_previous_action_categories_must_remain_in_domain(
    field_name: str,
    category_count: int,
    boundary: str,
) -> None:
    """Reject malformed accepted-action history at either domain boundary."""
    config, state = _valid_state()
    invalid_category = -1 if boundary == "below" else category_count
    state = state._replace(has_previous_timestep_joint_action=jnp.asarray(True))
    invalid_state = _replace_slot_value(
        state,
        field_name,
        slot=0,
        value=invalid_category,
    )

    with pytest.raises(ValueError, match=field_name):
        validate_env_state(config, invalid_state)


@pytest.mark.parametrize(
    "field_name",
    (
        "previous_timestep_move_actions",
        "previous_timestep_select_target_actions",
        "previous_timestep_use_ultimate_actions",
    ),
)
def test_absent_previous_action_requires_zero_history(field_name: str) -> None:
    """Keep reset's scalar validity flag coherent with every history head."""
    config, state = _valid_state()
    invalid_state = _replace_slot_value(
        state,
        field_name,
        slot=0,
        value=1,
    )

    with pytest.raises(
        ValueError,
        match="has_previous_timestep_joint_action",
    ):
        validate_env_state(config, invalid_state)


@pytest.mark.parametrize(
    "combat_head",
    (
        pytest.param("target", id="target"),
        pytest.param("ultimate", id="ultimate"),
    ),
)
def test_curated_scenario_rejects_shielded_source_combat_history(
    combat_head: str,
) -> None:
    """Keep authored shielded-source history movement-only."""
    config, state = _valid_state()
    previous_targets = state.previous_timestep_select_target_actions
    previous_ultimates = state.previous_timestep_use_ultimate_actions
    if combat_head == "target":
        previous_targets = previous_targets.at[0].set(2)
    else:
        previous_ultimates = previous_ultimates.at[0].set(1)
    inconsistent_state = state._replace(
        spawn_shield_durations=state.spawn_shield_durations.at[0].set(2),
        previous_timestep_select_target_actions=previous_targets,
        previous_timestep_use_ultimate_actions=previous_ultimates,
        has_previous_timestep_joint_action=jnp.asarray(True),
    )

    assert validate_env_state(config, inconsistent_state) is None
    with pytest.raises(ValueError, match="shielded scenario source"):
        validate_scenario_initial_state(config, inconsistent_state)


@pytest.mark.parametrize(
    ("source_slot", "target_action", "shielded_recipient_slot"),
    (
        pytest.param(0, 2, 1, id="team-a-ally"),
        pytest.param(0, 1 + MAX_AGENTS_PER_TEAM, 5, id="team-a-enemy"),
        pytest.param(5, 2, 6, id="team-b-ally"),
        pytest.param(5, 1 + MAX_AGENTS_PER_TEAM, 0, id="team-b-enemy"),
    ),
)
def test_curated_scenario_rejects_history_targeting_shielded_recipient(
    source_slot: int,
    target_action: int,
    shielded_recipient_slot: int,
) -> None:
    """Reject impossible authored target provenance for either team relation."""
    config, state = _valid_state()
    inconsistent_state = state._replace(
        spawn_shield_durations=state.spawn_shield_durations.at[
            shielded_recipient_slot
        ].set(2),
        previous_timestep_select_target_actions=(
            state.previous_timestep_select_target_actions.at[source_slot].set(
                target_action
            )
        ),
        has_previous_timestep_joint_action=jnp.asarray(True),
    )

    assert validate_env_state(config, inconsistent_state) is None
    with pytest.raises(ValueError, match="currently shielded recipient"):
        validate_scenario_initial_state(config, inconsistent_state)


@pytest.mark.parametrize(
    "recipient_condition",
    (
        pytest.param("out-of-range", id="out-of-range"),
        pytest.param("hidden", id="hidden"),
        pytest.param("dead", id="dead"),
    ),
)
def test_curated_scenario_preserves_nonshielded_historical_target_identity(
    recipient_condition: str,
) -> None:
    """Do not reinterpret accepted target history from current observability."""
    obstacles = None
    if recipient_condition == "hidden":
        obstacles = (
            _empty_obstacles().at[0].set(_wall(x=15.0, y=2.0, width=2.0, height=4.0))
        )
    config, state = _valid_state(obstacles=obstacles)
    recipient_slot = MAX_AGENTS_PER_TEAM
    historical_state = state._replace(
        previous_timestep_select_target_actions=(
            state.previous_timestep_select_target_actions.at[0].set(
                1 + MAX_AGENTS_PER_TEAM
            )
        ),
        has_previous_timestep_joint_action=jnp.asarray(True),
    )
    if recipient_condition == "dead":
        historical_state = historical_state._replace(
            alive_mask=historical_state.alive_mask.at[recipient_slot].set(False),
            current_health=historical_state.current_health.at[recipient_slot].set(0.0),
        )

    assert validate_env_state(config, historical_state) is None
    assert validate_scenario_initial_state(config, historical_state) is None


@pytest.mark.parametrize(
    ("axis", "side"),
    (
        pytest.param(0, "lower", id="left"),
        pytest.param(0, "upper", id="right"),
        pytest.param(1, "lower", id="bottom"),
        pytest.param(1, "upper", id="top"),
    ),
)
@pytest.mark.parametrize("is_dead", (False, True), ids=("living", "dead"))
def test_active_living_and_dead_positions_obey_map_bounds(
    axis: int,
    side: str,
    is_dead: bool,
) -> None:
    """Apply the same hard static bounds to living agents and preserved corpses."""
    config, state = _valid_state()
    if is_dead:
        state = state._replace(
            alive_mask=state.alive_mask.at[0].set(False),
            current_health=state.current_health.at[0].set(0.0),
        )
    radius = float(config.agent_profile.agent_radii[0])
    map_extent = config.map_width if axis == 0 else config.map_height
    invalid_coordinate = (
        radius - 2.0 * GEOMETRY_TOLERANCE
        if side == "lower"
        else map_extent - radius + 2.0 * GEOMETRY_TOLERANCE
    )
    invalid_state = state._replace(
        agent_positions=state.agent_positions.at[0, axis].set(invalid_coordinate)
    )

    with pytest.raises(ValueError, match="map bounds"):
        validate_env_state(config, invalid_state)


@pytest.mark.parametrize(
    ("axis", "side"),
    (
        pytest.param(0, "lower", id="left"),
        pytest.param(0, "upper", id="right"),
        pytest.param(1, "lower", id="bottom"),
        pytest.param(1, "upper", id="top"),
    ),
)
def test_map_bound_roundoff_within_geometry_tolerance_is_accepted(
    axis: int,
    side: str,
) -> None:
    """Do not reject the documented float32 residual around hard boundaries."""
    config, state = _valid_state()
    radius = float(config.agent_profile.agent_radii[0])
    map_extent = config.map_width if axis == 0 else config.map_height
    residual_coordinate = (
        radius - GEOMETRY_TOLERANCE / 2.0
        if side == "lower"
        else map_extent - radius + GEOMETRY_TOLERANCE / 2.0
    )
    residual_state = state._replace(
        agent_positions=state.agent_positions.at[0, axis].set(residual_coordinate)
    )
    assert validate_env_state(config, residual_state) is None


@pytest.mark.parametrize(
    ("obstacle", "overlapping_center"),
    (
        pytest.param(_pillar(), (15.0, 10.0), id="pillar"),
        pytest.param(_wall(), (15.0, 10.0), id="wall"),
    ),
)
@pytest.mark.parametrize("is_dead", (False, True), ids=("living", "dead"))
def test_active_living_and_dead_positions_must_clear_static_obstacles(
    obstacle: Array,
    overlapping_center: tuple[float, float],
    is_dead: bool,
) -> None:
    """Reject obstacle overlap for living agents and preserved corpses."""
    obstacles = _empty_obstacles().at[0].set(obstacle)
    config, state = _valid_state(obstacles=obstacles)
    if is_dead:
        state = state._replace(
            alive_mask=state.alive_mask.at[0].set(False),
            current_health=state.current_health.at[0].set(0.0),
        )
    invalid_state = state._replace(
        agent_positions=state.agent_positions.at[0].set(
            jnp.asarray(overlapping_center, dtype=jnp.float32)
        )
    )

    with pytest.raises(ValueError, match="obstacles"):
        validate_env_state(config, invalid_state)


def test_static_obstacle_tangency_is_accepted() -> None:
    """Reuse the authoritative positive-overlap predicate where tangency is legal."""
    obstacle = _pillar()
    obstacles = _empty_obstacles().at[0].set(obstacle)
    config, state = _valid_state(obstacles=obstacles)
    tangent_x = (
        float(obstacle[OBSTACLE_FEATURE_X])
        + float(obstacle[OBSTACLE_FEATURE_RADIUS])
        + float(config.agent_profile.agent_radii[0])
    )
    tangent_state = state._replace(
        agent_positions=state.agent_positions.at[0].set(
            jnp.asarray(
                (tangent_x, float(obstacle[OBSTACLE_FEATURE_Y])),
                dtype=jnp.float32,
            )
        )
    )
    assert validate_env_state(config, tangent_state) is None


def test_runtime_snapshot_accepts_residual_while_curated_start_rejects_it() -> None:
    """Separate runtime snapshot validity from strict curated-start validity."""
    config, state = _valid_state()
    residual_positions = (
        state.agent_positions.at[0]
        .set(jnp.asarray((4.0, 4.0), dtype=jnp.float32))
        .at[1]
        .set(jnp.asarray((4.75, 4.0), dtype=jnp.float32))
    )
    residual_state = state._replace(agent_positions=residual_positions)

    center_distance = float(
        jnp.linalg.norm(
            residual_state.agent_positions[0] - residual_state.agent_positions[1]
        )
    )
    sum_of_radii = float(
        config.agent_profile.agent_radii[0] + config.agent_profile.agent_radii[1]
    )
    assert center_distance < sum_of_radii
    assert validate_env_state(config, residual_state) is None
    with pytest.raises(ValueError, match="curated scenario living bodies overlap"):
        validate_scenario_initial_state(config, residual_state)


def test_curated_scenario_living_body_tangency_is_accepted() -> None:
    """Treat exact living-body tangency as a legal authored start."""
    config, state = _valid_state()
    radius_sum = float(
        config.agent_profile.agent_radii[0] + config.agent_profile.agent_radii[1]
    )
    tangent_positions = (
        state.agent_positions.at[0]
        .set(jnp.asarray((4.0, 4.0), dtype=jnp.float32))
        .at[1]
        .set(jnp.asarray((4.0 + radius_sum, 4.0), dtype=jnp.float32))
    )
    tangent_state = state._replace(agent_positions=tangent_positions)

    assert validate_scenario_initial_state(config, tangent_state) is None


def test_preserved_corpse_may_overlap_a_living_body() -> None:
    """Treat preserved corpses as nonphysical for pairwise state validity."""
    config, state = _valid_state()
    dead_slot = 0
    living_slot = 1
    corpse_overlap_state = state._replace(
        alive_mask=state.alive_mask.at[dead_slot].set(False),
        current_health=state.current_health.at[dead_slot].set(0.0),
        agent_positions=state.agent_positions.at[dead_slot].set(
            state.agent_positions[living_slot]
        ),
    )
    assert validate_env_state(config, corpse_overlap_state) is None
    assert validate_scenario_initial_state(config, corpse_overlap_state) is None


def test_runtime_validator_accepts_public_death_and_corpse_successors() -> None:
    """Validate actual public successor states without entering traced execution."""
    config = _valid_config(team_sizes=(1, 1))
    spawn_pad_positions = (
        config.team_spawn_pad_positions.at[0, 0]
        .set(jnp.asarray((2.0, 2.0), dtype=jnp.float32))
        .at[1, 0]
        .set(jnp.asarray((4.0, 2.0), dtype=jnp.float32))
    )
    config = config._replace(team_spawn_pad_positions=spawn_pad_positions)
    state, _, _, _ = reset(config, jax.random.key(21))
    target_slot = MAX_AGENTS_PER_TEAM
    state = state._replace(current_health=state.current_health.at[target_slot].set(1.0))
    _, choosing_mask = _build_observation_and_action_mask(state, config)
    lethal_action = Action(
        move=jnp.full((MAX_AGENT_SLOTS,), MOVE_STAY, dtype=jnp.int32),
        select_target=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32)
        .at[0]
        .set(1 + MAX_AGENTS_PER_TEAM),
        use_ultimate=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
    )

    dead_state, _, _, _, dead_mask, _ = step(
        config,
        state,
        choosing_mask,
        lethal_action,
        jax.random.key(22),
    )
    assert not bool(dead_state.alive_mask[target_slot])
    assert validate_env_state(config, dead_state) is None

    canonical_action = Action(
        move=jnp.full((MAX_AGENT_SLOTS,), MOVE_STAY, dtype=jnp.int32),
        select_target=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
        use_ultimate=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
    )
    corpse_state, *_ = step(
        config,
        dead_state,
        dead_mask,
        canonical_action,
        jax.random.key(23),
    )
    assert validate_env_state(config, corpse_state) is None


def test_runtime_validator_accepts_an_actual_boundary_residual_successor() -> None:
    """Accept fixed-pass overlap only at the runtime/replay snapshot boundary."""
    config = _valid_config(team_sizes=(1, 1))
    moving_slot = MAX_AGENTS_PER_TEAM
    spawn_pad_positions = (
        config.team_spawn_pad_positions.at[0, 0]
        .set(jnp.asarray((0.5, 10.0), dtype=jnp.float32))
        .at[1, 0]
        .set(jnp.asarray((2.0, 10.0), dtype=jnp.float32))
    )
    config = config._replace(team_spawn_pad_positions=spawn_pad_positions)
    state, _, action_mask, _ = reset(config, jax.random.key(31))
    movement = (
        jnp.full((MAX_AGENT_SLOTS,), MOVE_STAY, dtype=jnp.int32)
        .at[moving_slot]
        .set(MOVE_WEST)
    )
    action = Action(
        move=movement,
        select_target=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
        use_ultimate=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
    )

    next_state, *_ = step(
        config,
        state,
        action_mask,
        action,
        jax.random.key(32),
    )
    center_distance = float(
        jnp.linalg.norm(
            next_state.agent_positions[0] - next_state.agent_positions[moving_slot]
        )
    )
    radius_sum = float(
        config.agent_profile.agent_radii[0]
        + config.agent_profile.agent_radii[moving_slot]
    )

    assert center_distance < radius_sum
    assert validate_env_state(config, next_state) is None
    with pytest.raises(ValueError, match="curated scenario living bodies overlap"):
        validate_scenario_initial_state(config, next_state)
