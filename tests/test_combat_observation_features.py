"""Combat observation contracts through Milestone 5 Step 6 Checkpoint 1."""
# pyright: reportPrivateUsage=false

from collections.abc import Sequence
from typing import cast

import jax
import jax.numpy as jnp
import pytest
from jax import Array

import marl_battlegrounds.core.combat as combat
import marl_battlegrounds.core.types as core_types
from marl_battlegrounds.core.config import resolve_agent_profile
from marl_battlegrounds.core.env import (
    _build_observation_and_action_mask,
    reset,
    step,
)
from marl_battlegrounds.core.types import (
    AGENT_FEATURE_ANTI_HEAL_ROGUE_POISON_DURATION,
    AGENT_FEATURE_ANTI_HEAL_ROGUE_POISON_MULTIPLIER,
    AGENT_FEATURE_BASE_MOVEMENT_SPEED,
    AGENT_FEATURE_CAPABILITY_ANTI_HEAL_ROGUE_POISON_DURATION,
    AGENT_FEATURE_CAPABILITY_ANTI_HEAL_ROGUE_POISON_MULTIPLIER,
    AGENT_FEATURE_CAPABILITY_BASIC_DAMAGE,
    AGENT_FEATURE_CAPABILITY_BASIC_HEALING,
    AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER,
    AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_AURA_RADIUS,
    AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_BURST_DURATION,
    AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_BURST_MULTIPLIER,
    AGENT_FEATURE_CAPABILITY_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER,
    AGENT_FEATURE_CAPABILITY_DAMAGE_MITIGATION_WARRIOR_AURA_RADIUS,
    AGENT_FEATURE_CAPABILITY_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_DURATION,
    AGENT_FEATURE_CAPABILITY_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_FRACTION,
    AGENT_FEATURE_CAPABILITY_SLOW_HUNTER_BASIC_DURATION,
    AGENT_FEATURE_CAPABILITY_SLOW_HUNTER_BASIC_MULTIPLIER,
    AGENT_FEATURE_CAPABILITY_SLOW_ROGUE_POISON_DURATION,
    AGENT_FEATURE_CAPABILITY_SLOW_ROGUE_POISON_MULTIPLIER,
    AGENT_FEATURE_CAPABILITY_SLOW_WARRIOR_CHARGE_DURATION,
    AGENT_FEATURE_CAPABILITY_SLOW_WARRIOR_CHARGE_MULTIPLIER,
    AGENT_FEATURE_CAPABILITY_STUN_HUNTER_TRAP_DURATION,
    AGENT_FEATURE_CAPABILITY_STUN_ROGUE_POISON_DURATION,
    AGENT_FEATURE_CAPABILITY_STUN_WARRIOR_CHARGE_DURATION,
    AGENT_FEATURE_CAPABILITY_ULTIMATE_COOLDOWN_DURATION,
    AGENT_FEATURE_CAPABILITY_ULTIMATE_DAMAGE,
    AGENT_FEATURE_CAPABILITY_ULTIMATE_HEALING,
    AGENT_FEATURE_CURRENT_HEALTH,
    AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER,
    AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_BURST_DURATION,
    AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER,
    AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED,
    AGENT_FEATURE_MAX_HEALTH,
    AGENT_FEATURE_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_DURATION,
    AGENT_FEATURE_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_FRACTION,
    AGENT_FEATURE_SLOW_HUNTER_BASIC_DURATION,
    AGENT_FEATURE_SLOW_HUNTER_BASIC_MULTIPLIER,
    AGENT_FEATURE_SLOW_ROGUE_POISON_DURATION,
    AGENT_FEATURE_SLOW_ROGUE_POISON_MULTIPLIER,
    AGENT_FEATURE_SLOW_WARRIOR_CHARGE_DURATION,
    AGENT_FEATURE_SLOW_WARRIOR_CHARGE_MULTIPLIER,
    AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION,
    AGENT_FEATURE_ULTIMATE_COOLDOWN_REMAINING,
    HUNTER_CLASS_ID,
    MAGE_CLASS_ID,
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    MAX_OBSTACLE_SLOTS,
    MOVE_EAST,
    MOVE_STAY,
    MOVE_WEST,
    OBSTACLE_FEATURES,
    PRIEST_CLASS_ID,
    ROGUE_CLASS_ID,
    SELF_FEATURES,
    SLOW_CHANNEL_HUNTER_BASIC,
    SLOW_CHANNEL_ROGUE_POISON,
    SLOW_CHANNEL_WARRIOR_CHARGE,
    STUN_CHANNEL_HUNTER_TRAP,
    UNIT_FEATURES,
    WARRIOR_CLASS_ID,
    Action,
    ActionMask,
    EnvConfig,
    EnvState,
    Observation,
)

_ROSTER = jnp.asarray(
    (
        MAGE_CLASS_ID,
        WARRIOR_CLASS_ID,
        HUNTER_CLASS_ID,
        ROGUE_CLASS_ID,
        PRIEST_CLASS_ID,
        MAGE_CLASS_ID,
        WARRIOR_CLASS_ID,
        HUNTER_CLASS_ID,
        ROGUE_CLASS_ID,
        PRIEST_CLASS_ID,
    ),
    dtype=jnp.int32,
)

_FEATURE_NAMES_IN_ORDER: tuple[str, ...] = (
    "AGENT_FEATURE_X",
    "AGENT_FEATURE_Y",
    "AGENT_FEATURE_RADIUS",
    "AGENT_FEATURE_TEAM_ID",
    "AGENT_FEATURE_ACTIVE",
    "AGENT_FEATURE_ALIVE",
    "AGENT_FEATURE_CLASS_ID",
    "AGENT_FEATURE_BASE_MOVEMENT_SPEED",
    "AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED",
    "AGENT_FEATURE_OBSERVATION_RADIUS",
    "AGENT_FEATURE_BASIC_INTERACTION_RADIUS",
    "AGENT_FEATURE_ULTIMATE_INTERACTION_RADIUS",
    "AGENT_FEATURE_CURRENT_HEALTH",
    "AGENT_FEATURE_MAX_HEALTH",
    "AGENT_FEATURE_ULTIMATE_COOLDOWN_REMAINING",
    "AGENT_FEATURE_SLOW_WARRIOR_CHARGE_DURATION",
    "AGENT_FEATURE_SLOW_HUNTER_BASIC_DURATION",
    "AGENT_FEATURE_SLOW_ROGUE_POISON_DURATION",
    "AGENT_FEATURE_SLOW_WARRIOR_CHARGE_MULTIPLIER",
    "AGENT_FEATURE_SLOW_HUNTER_BASIC_MULTIPLIER",
    "AGENT_FEATURE_SLOW_ROGUE_POISON_MULTIPLIER",
    "AGENT_FEATURE_STUN_WARRIOR_CHARGE_DURATION",
    "AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION",
    "AGENT_FEATURE_STUN_ROGUE_POISON_DURATION",
    "AGENT_FEATURE_ANTI_HEAL_ROGUE_POISON_DURATION",
    "AGENT_FEATURE_ANTI_HEAL_ROGUE_POISON_MULTIPLIER",
    "AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_BURST_DURATION",
    "AGENT_FEATURE_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_DURATION",
    "AGENT_FEATURE_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_FRACTION",
    "AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER",
    "AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER",
    "AGENT_FEATURE_CAPABILITY_BASIC_DAMAGE",
    "AGENT_FEATURE_CAPABILITY_BASIC_HEALING",
    "AGENT_FEATURE_CAPABILITY_ULTIMATE_COOLDOWN_DURATION",
    "AGENT_FEATURE_CAPABILITY_SLOW_WARRIOR_CHARGE_DURATION",
    "AGENT_FEATURE_CAPABILITY_SLOW_HUNTER_BASIC_DURATION",
    "AGENT_FEATURE_CAPABILITY_SLOW_ROGUE_POISON_DURATION",
    "AGENT_FEATURE_CAPABILITY_SLOW_WARRIOR_CHARGE_MULTIPLIER",
    "AGENT_FEATURE_CAPABILITY_SLOW_HUNTER_BASIC_MULTIPLIER",
    "AGENT_FEATURE_CAPABILITY_SLOW_ROGUE_POISON_MULTIPLIER",
    "AGENT_FEATURE_CAPABILITY_STUN_WARRIOR_CHARGE_DURATION",
    "AGENT_FEATURE_CAPABILITY_STUN_HUNTER_TRAP_DURATION",
    "AGENT_FEATURE_CAPABILITY_STUN_ROGUE_POISON_DURATION",
    "AGENT_FEATURE_CAPABILITY_ANTI_HEAL_ROGUE_POISON_DURATION",
    "AGENT_FEATURE_CAPABILITY_ANTI_HEAL_ROGUE_POISON_MULTIPLIER",
    "AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_BURST_DURATION",
    "AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_BURST_MULTIPLIER",
    "AGENT_FEATURE_CAPABILITY_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_DURATION",
    "AGENT_FEATURE_CAPABILITY_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_FRACTION",
    "AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_AURA_RADIUS",
    "AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER",
    "AGENT_FEATURE_CAPABILITY_DAMAGE_MITIGATION_WARRIOR_AURA_RADIUS",
    "AGENT_FEATURE_CAPABILITY_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER",
    "AGENT_FEATURE_CAPABILITY_ULTIMATE_HEALING",
    "AGENT_FEATURE_CAPABILITY_ULTIMATE_DAMAGE",
)


def _config(team_sizes: tuple[int, int] = (5, 5)) -> EnvConfig:
    profile = resolve_agent_profile(_ROSTER, jnp.asarray(team_sizes, dtype=jnp.int32))
    positions = jnp.asarray(
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
    )
    return EnvConfig(
        max_steps=1000,
        map_width=20.0,
        map_height=12.0,
        obstacles=jnp.zeros((MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES), dtype=jnp.float32),
        agent_profile=profile,
        initial_agent_positions=jnp.where(profile.active_mask[:, None], positions, 0.0),
    )


def _action(
    *, east_slots: Sequence[int] = (), west_slots: Sequence[int] = ()
) -> Action:
    """Return a no-combat joint action with selected horizontal movement."""
    move = jnp.full((MAX_AGENT_SLOTS,), MOVE_STAY, dtype=jnp.int32)
    for slot in east_slots:
        move = move.at[slot].set(MOVE_EAST)
    for slot in west_slots:
        move = move.at[slot].set(MOVE_WEST)
    return Action(
        move=move,
        select_target=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
        use_ultimate=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
    )


def _expected_owned_payload(
    config: EnvConfig, class_id: int, payload: float | int | Array
) -> Array:
    owns_payload = jnp.logical_and(
        config.agent_profile.active_mask,
        config.agent_profile.class_ids == class_id,
    )
    return jnp.where(owns_payload, payload, 0.0).astype(jnp.float32)


def _observation_after_step(
    config: EnvConfig,
    state: EnvState,
    action_mask: ActionMask,
    action: Action,
    key: Array,
) -> Observation:
    return step(config, state, action_mask, action, key)[1]


def _current_action_mask(config: EnvConfig, state: EnvState) -> ActionMask:
    """Return the action mask paired with an explicitly built test state."""
    _, action_mask = _build_observation_and_action_mask(state, config)
    return action_mask


def test_shared_agent_feature_schema_is_contiguous_and_duration_first() -> None:
    indices = tuple(getattr(core_types, name) for name in _FEATURE_NAMES_IN_ORDER)

    assert SELF_FEATURES == UNIT_FEATURES == len(_FEATURE_NAMES_IN_ORDER) == 55
    assert indices == tuple(range(55))
    assert not hasattr(
        core_types, "AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_BURST_MULTIPLIER"
    )


def test_reset_rows_expose_profile_state_and_neutral_attached_values() -> None:
    config = _config()
    state, observation, *_ = reset(config, jax.random.key(1))
    rows = observation.self_features

    assert rows.shape == (MAX_AGENT_SLOTS, SELF_FEATURES)
    assert rows.dtype == jnp.float32
    assert bool(
        jnp.array_equal(
            rows[:, AGENT_FEATURE_BASE_MOVEMENT_SPEED],
            config.agent_profile.base_movement_speeds,
        )
    )
    assert bool(
        jnp.array_equal(
            rows[:, AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED],
            config.agent_profile.base_movement_speeds,
        )
    )
    assert bool(
        jnp.array_equal(rows[:, AGENT_FEATURE_CURRENT_HEALTH], state.current_health)
    )
    assert bool(
        jnp.array_equal(
            rows[:, AGENT_FEATURE_MAX_HEALTH], config.agent_profile.max_health
        )
    )
    assert bool(
        jnp.array_equal(
            rows[:, AGENT_FEATURE_ULTIMATE_COOLDOWN_REMAINING],
            state.ultimate_cooldowns.astype(jnp.float32),
        )
    )

    multiplicative_columns = (
        AGENT_FEATURE_SLOW_WARRIOR_CHARGE_MULTIPLIER,
        AGENT_FEATURE_SLOW_HUNTER_BASIC_MULTIPLIER,
        AGENT_FEATURE_SLOW_ROGUE_POISON_MULTIPLIER,
        AGENT_FEATURE_ANTI_HEAL_ROGUE_POISON_MULTIPLIER,
    )
    assert bool(jnp.all(rows[:, jnp.asarray(multiplicative_columns)] == 1.0))
    assert bool(
        jnp.all(
            rows[:, AGENT_FEATURE_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_FRACTION] == 0.0
        )
    )


def test_attached_aura_features_follow_slot_aligned_team_geometry() -> None:
    """Prove self rows expose the modifiers attached to each produced slot."""
    config = _config((2, 2))
    state, *_ = reset(config, jax.random.key(11))
    positions = state.agent_positions
    positions = positions.at[0].set(jnp.asarray((2.0, 2.0), dtype=jnp.float32))
    positions = positions.at[1].set(jnp.asarray((3.0, 2.0), dtype=jnp.float32))
    positions = positions.at[MAX_AGENTS_PER_TEAM].set(
        jnp.asarray((12.0, 8.0), dtype=jnp.float32)
    )
    positions = positions.at[MAX_AGENTS_PER_TEAM + 1].set(
        jnp.asarray((13.0, 8.0), dtype=jnp.float32)
    )
    state = state._replace(agent_positions=positions)

    observation, _ = _build_observation_and_action_mask(state, config)
    mage_attached = observation.self_features[
        :, AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER
    ]
    warrior_attached = observation.self_features[
        :, AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER
    ]

    active_slots = jnp.asarray((0, 1, MAX_AGENTS_PER_TEAM, MAX_AGENTS_PER_TEAM + 1))
    inactive_slots = jnp.asarray((2, 3, 4, 7, 8, 9))
    assert bool(
        jnp.all(mage_attached[active_slots] == combat.MAGE_DAMAGE_AURA_MULTIPLIER)
    )
    assert bool(
        jnp.all(
            warrior_attached[active_slots]
            == combat.WARRIOR_DAMAGE_MITIGATION_AURA_MULTIPLIER
        )
    )
    assert bool(jnp.all(mage_attached[inactive_slots] == 1.0))
    assert bool(jnp.all(warrior_attached[inactive_slots] == 1.0))


@pytest.mark.parametrize(
    ("start_distance", "move_west", "expected_multiplier"),
    (
        pytest.param(2.25, True, combat.MAGE_DAMAGE_AURA_MULTIPLIER, id="enters"),
        pytest.param(1.75, False, 1.0, id="leaves"),
    ),
)
def test_returned_aura_features_use_post_movement_positions(
    start_distance: float,
    move_west: bool,
    expected_multiplier: float,
) -> None:
    """Prove returned attached modifiers describe the produced next state."""
    config = _config((2, 1))
    state, *_ = reset(config, jax.random.key(12))
    positions = state.agent_positions
    positions = positions.at[0].set(jnp.asarray((4.0, 4.0), dtype=jnp.float32))
    positions = positions.at[1].set(
        jnp.asarray((4.0 + start_distance, 4.0), dtype=jnp.float32)
    )
    positions = positions.at[MAX_AGENTS_PER_TEAM].set(
        jnp.asarray((14.0, 8.0), dtype=jnp.float32)
    )
    state = state._replace(agent_positions=positions)
    action = _action(west_slots=(1,)) if move_west else _action(east_slots=(1,))

    next_state, observation, *_ = step(
        config,
        state,
        _current_action_mask(config, state),
        action,
        jax.random.key(13),
    )

    next_distance = cast(
        Array,
        jnp.linalg.norm(next_state.agent_positions[1] - next_state.agent_positions[0]),
    )
    assert bool(next_distance <= combat.MAGE_DAMAGE_AURA_RADIUS) is (
        expected_multiplier == combat.MAGE_DAMAGE_AURA_MULTIPLIER
    )
    assert (
        observation.self_features[
            1, AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER
        ]
        == expected_multiplier
    )


def test_capabilities_are_class_owned_payloads_not_cooldown_availability() -> None:
    config = _config()
    state, first_observation, *_ = reset(config, jax.random.key(2))
    state = state._replace(
        ultimate_cooldowns=jnp.arange(MAX_AGENT_SLOTS, dtype=jnp.int32) + 3
    )
    _, second_observation, *_ = step(
        config,
        state,
        _current_action_mask(config, state),
        _action(),
        jax.random.key(3),
    )

    expected_catalog_columns = (
        (AGENT_FEATURE_CAPABILITY_BASIC_DAMAGE, combat.BASIC_DAMAGE_BY_CLASS),
        (AGENT_FEATURE_CAPABILITY_BASIC_HEALING, combat.BASIC_HEALING_BY_CLASS),
        (
            AGENT_FEATURE_CAPABILITY_ULTIMATE_COOLDOWN_DURATION,
            combat.ULTIMATE_COOLDOWN_BY_CLASS,
        ),
    )
    for column, catalog in expected_catalog_columns:
        expected = catalog[config.agent_profile.class_ids].astype(jnp.float32)
        assert bool(
            jnp.array_equal(second_observation.self_features[:, column], expected)
        )
        assert bool(
            jnp.array_equal(
                second_observation.self_features[:, column],
                first_observation.self_features[:, column],
            )
        )

    owned_payloads = (
        (
            AGENT_FEATURE_CAPABILITY_SLOW_WARRIOR_CHARGE_DURATION,
            WARRIOR_CLASS_ID,
            combat.WARRIOR_CHARGE_SLOW_DURATION_TICKS,
        ),
        (
            AGENT_FEATURE_CAPABILITY_SLOW_HUNTER_BASIC_DURATION,
            HUNTER_CLASS_ID,
            combat.HUNTER_BASIC_SLOW_DURATION_TICKS,
        ),
        (
            AGENT_FEATURE_CAPABILITY_SLOW_ROGUE_POISON_DURATION,
            ROGUE_CLASS_ID,
            combat.ROGUE_POISON_SLOW_DURATION_TICKS,
        ),
        (
            AGENT_FEATURE_CAPABILITY_SLOW_WARRIOR_CHARGE_MULTIPLIER,
            WARRIOR_CLASS_ID,
            combat.WARRIOR_CHARGE_SLOW_MULTIPLIER,
        ),
        (
            AGENT_FEATURE_CAPABILITY_SLOW_HUNTER_BASIC_MULTIPLIER,
            HUNTER_CLASS_ID,
            combat.HUNTER_BASIC_SLOW_MULTIPLIER,
        ),
        (
            AGENT_FEATURE_CAPABILITY_SLOW_ROGUE_POISON_MULTIPLIER,
            ROGUE_CLASS_ID,
            combat.ROGUE_POISON_SLOW_MULTIPLIER,
        ),
        (
            AGENT_FEATURE_CAPABILITY_STUN_WARRIOR_CHARGE_DURATION,
            WARRIOR_CLASS_ID,
            combat.WARRIOR_CHARGE_STUN_DURATION_TICKS,
        ),
        (
            AGENT_FEATURE_CAPABILITY_STUN_HUNTER_TRAP_DURATION,
            HUNTER_CLASS_ID,
            combat.HUNTER_TRAP_STUN_DURATION_TICKS,
        ),
        (
            AGENT_FEATURE_CAPABILITY_STUN_ROGUE_POISON_DURATION,
            ROGUE_CLASS_ID,
            combat.ROGUE_POISON_STUN_DURATION_TICKS,
        ),
        (
            AGENT_FEATURE_CAPABILITY_ANTI_HEAL_ROGUE_POISON_DURATION,
            ROGUE_CLASS_ID,
            combat.ROGUE_POISON_ANTI_HEAL_DURATION_TICKS,
        ),
        (
            AGENT_FEATURE_CAPABILITY_ANTI_HEAL_ROGUE_POISON_MULTIPLIER,
            ROGUE_CLASS_ID,
            combat.ROGUE_POISON_ANTI_HEAL_MULTIPLIER,
        ),
        (
            AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_BURST_DURATION,
            MAGE_CLASS_ID,
            combat.MAGE_BURST_DAMAGE_DURATION_TICKS,
        ),
        (
            AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_BURST_MULTIPLIER,
            MAGE_CLASS_ID,
            combat.MAGE_BURST_DAMAGE_MULTIPLIER,
        ),
        (
            AGENT_FEATURE_CAPABILITY_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_DURATION,
            PRIEST_CLASS_ID,
            combat.PRIEST_HEAL_SPEED_FLOOR_DURATION_TICKS,
        ),
        (
            AGENT_FEATURE_CAPABILITY_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_FRACTION,
            PRIEST_CLASS_ID,
            combat.PRIEST_HEAL_SPEED_FLOOR,
        ),
        (
            AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_AURA_RADIUS,
            MAGE_CLASS_ID,
            combat.MAGE_DAMAGE_AURA_RADIUS,
        ),
        (
            AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER,
            MAGE_CLASS_ID,
            combat.MAGE_DAMAGE_AURA_MULTIPLIER,
        ),
        (
            AGENT_FEATURE_CAPABILITY_DAMAGE_MITIGATION_WARRIOR_AURA_RADIUS,
            WARRIOR_CLASS_ID,
            combat.WARRIOR_DAMAGE_MITIGATION_AURA_RADIUS,
        ),
        (
            AGENT_FEATURE_CAPABILITY_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER,
            WARRIOR_CLASS_ID,
            combat.WARRIOR_DAMAGE_MITIGATION_AURA_MULTIPLIER,
        ),
        (
            AGENT_FEATURE_CAPABILITY_ULTIMATE_HEALING,
            PRIEST_CLASS_ID,
            combat.ULTIMATE_HEALING_BY_CLASS[PRIEST_CLASS_ID],
        ),
        (
            AGENT_FEATURE_CAPABILITY_ULTIMATE_DAMAGE,
            WARRIOR_CLASS_ID,
            combat.ULTIMATE_DAMAGE_BY_CLASS[WARRIOR_CLASS_ID],
        ),
    )
    for column, class_id, payload in owned_payloads:
        assert bool(
            jnp.array_equal(
                second_observation.self_features[:, column],
                _expected_owned_payload(config, class_id, payload),
            )
        )


def test_attached_statuses_and_effective_speed_follow_duration_state() -> None:
    config = _config()
    state, *_ = reset(config, jax.random.key(4))
    separated_positions = jnp.asarray(
        (
            (2.0, 2.0),
            (5.0, 2.0),
            (8.0, 2.0),
            (11.0, 2.0),
            (14.0, 2.0),
            (2.0, 8.0),
            (5.0, 8.0),
            (8.0, 8.0),
            (11.0, 8.0),
            (14.0, 8.0),
        ),
        dtype=jnp.float32,
    )
    state = state._replace(
        agent_positions=separated_positions,
        slow_durations=(
            state.slow_durations.at[0, SLOW_CHANNEL_WARRIOR_CHARGE]
            .set(2)
            .at[1, SLOW_CHANNEL_HUNTER_BASIC]
            .set(3)
            .at[2, SLOW_CHANNEL_ROGUE_POISON]
            .set(4)
        ),
        rogue_poison_anti_heal_durations=(
            state.rogue_poison_anti_heal_durations.at[3].set(5)
        ),
        mage_burst_damage_amplification_durations=(
            state.mage_burst_damage_amplification_durations.at[4].set(6)
        ),
        priest_blessing_of_freedom_slow_floor_durations=(
            state.priest_blessing_of_freedom_slow_floor_durations.at[0].set(7)
        ),
    )

    next_state, observation, *_ = step(
        config,
        state,
        _current_action_mask(config, state),
        _action(east_slots=(0, 1, 2)),
        jax.random.key(5),
    )
    rows = observation.self_features

    assert rows[0, AGENT_FEATURE_SLOW_WARRIOR_CHARGE_DURATION] == 1.0
    assert rows[1, AGENT_FEATURE_SLOW_HUNTER_BASIC_DURATION] == 2.0
    assert rows[2, AGENT_FEATURE_SLOW_ROGUE_POISON_DURATION] == 3.0
    assert (
        rows[0, AGENT_FEATURE_SLOW_WARRIOR_CHARGE_MULTIPLIER]
        == combat.WARRIOR_CHARGE_SLOW_MULTIPLIER
    )
    assert (
        rows[1, AGENT_FEATURE_SLOW_HUNTER_BASIC_MULTIPLIER]
        == combat.HUNTER_BASIC_SLOW_MULTIPLIER
    )
    assert (
        rows[2, AGENT_FEATURE_SLOW_ROGUE_POISON_MULTIPLIER]
        == combat.ROGUE_POISON_SLOW_MULTIPLIER
    )
    assert rows[3, AGENT_FEATURE_ANTI_HEAL_ROGUE_POISON_DURATION] == 4.0
    assert (
        rows[3, AGENT_FEATURE_ANTI_HEAL_ROGUE_POISON_MULTIPLIER]
        == combat.ROGUE_POISON_ANTI_HEAL_MULTIPLIER
    )
    assert rows[4, AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_BURST_DURATION] == 5.0
    assert rows[0, AGENT_FEATURE_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_DURATION] == 6.0
    assert (
        rows[0, AGENT_FEATURE_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_FRACTION]
        == combat.PRIEST_HEAL_SPEED_FLOOR
    )

    expected_observed_effective = combat.derive_effective_movement_speeds(
        next_state.slow_durations,
        next_state.priest_blessing_of_freedom_slow_floor_durations,
        next_state.stun_durations,
        config.agent_profile.base_movement_speeds,
        jnp.logical_and(config.agent_profile.active_mask, next_state.alive_mask),
    )
    assert bool(
        jnp.array_equal(
            rows[:, AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED],
            expected_observed_effective,
        )
    )
    expected_effective = combat.derive_effective_movement_speeds(
        state.slow_durations,
        state.priest_blessing_of_freedom_slow_floor_durations,
        state.stun_durations,
        config.agent_profile.base_movement_speeds,
        jnp.logical_and(config.agent_profile.active_mask, state.alive_mask),
    )
    displacement = next_state.agent_positions - state.agent_positions
    assert bool(jnp.isclose(displacement[0, 0], expected_effective[0]))
    assert bool(jnp.isclose(displacement[1, 0], expected_effective[1]))
    assert bool(jnp.isclose(displacement[2, 0], expected_effective[2]))


def test_stun_zeroes_effective_speed_in_self_and_visible_unit_rows() -> None:
    """Prove public effective speed agrees with current stun and visibility."""
    config = _config((1, 2))
    state, *_ = reset(config, jax.random.key(14))
    visible_enemy_slot = MAX_AGENTS_PER_TEAM
    hidden_enemy_slot = MAX_AGENTS_PER_TEAM + 1
    positions = state.agent_positions
    positions = positions.at[0].set(jnp.asarray((2.0, 2.0), dtype=jnp.float32))
    positions = positions.at[visible_enemy_slot].set(
        jnp.asarray((3.0, 2.0), dtype=jnp.float32)
    )
    positions = positions.at[hidden_enemy_slot].set(
        jnp.asarray((19.0, 11.0), dtype=jnp.float32)
    )
    state = state._replace(
        agent_positions=positions,
        stun_durations=state.stun_durations.at[
            visible_enemy_slot, STUN_CHANNEL_HUNTER_TRAP
        ].set(2),
    )

    observation, _ = _build_observation_and_action_mask(state, config)

    assert config.agent_profile.base_movement_speeds[visible_enemy_slot] > 0.0
    assert (
        observation.self_features[
            visible_enemy_slot, AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED
        ]
        == 0.0
    )
    assert (
        observation.self_features[
            visible_enemy_slot, AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION
        ]
        == 2.0
    )
    assert bool(observation.enemy_visibility_mask[0, 0])
    assert (
        observation.enemy_unit_features[0, 0, AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED]
        == 0.0
    )
    assert (
        observation.enemy_unit_features[0, 0, AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION]
        == 2.0
    )
    assert not bool(observation.enemy_visibility_mask[0, 1])
    assert bool(jnp.all(observation.enemy_unit_features[0, 1] == 0.0))


def test_visible_candidates_match_shared_rows_and_hidden_rows_are_fully_zero() -> None:
    config = _config((1, 2))
    state, *_ = reset(config, jax.random.key(6))
    positions = state.agent_positions
    positions = positions.at[0].set(jnp.asarray((2.0, 2.0), dtype=jnp.float32))
    positions = positions.at[MAX_AGENTS_PER_TEAM].set(
        jnp.asarray((3.0, 2.0), dtype=jnp.float32)
    )
    positions = positions.at[MAX_AGENTS_PER_TEAM + 1].set(
        jnp.asarray((19.0, 11.0), dtype=jnp.float32)
    )
    state = state._replace(agent_positions=positions)

    _, observation, *_ = step(
        config,
        state,
        _current_action_mask(config, state),
        _action(),
        jax.random.key(7),
    )

    assert bool(observation.enemy_visibility_mask[0, 0])
    assert not bool(observation.enemy_visibility_mask[0, 1])
    assert bool(
        jnp.array_equal(
            observation.enemy_unit_features[0, 0],
            observation.self_features[MAX_AGENTS_PER_TEAM],
        )
    )
    assert bool(jnp.all(observation.enemy_unit_features[0, 1] == 0.0))
    assert observation.enemy_unit_features[0, 1].shape == (UNIT_FEATURES,)


@pytest.mark.parametrize(
    "inactive_class_id",
    (
        pytest.param(MAGE_CLASS_ID, id="mage"),
        pytest.param(WARRIOR_CLASS_ID, id="warrior"),
        pytest.param(PRIEST_CLASS_ID, id="priest"),
    ),
)
def test_capabilities_remain_zero_for_inactive_non_neutral_padding(
    inactive_class_id: int,
) -> None:
    """Prove every capability column honors activity, even for malformed padding."""
    config = _config((1, 1))
    padded_slot = 1
    profile = config.agent_profile._replace(
        class_ids=config.agent_profile.class_ids.at[padded_slot].set(inactive_class_id)
    )
    config = config._replace(agent_profile=profile)
    _, observation, *_ = reset(config, jax.random.key(8))

    assert bool(jnp.all(observation.self_features[padded_slot, 31:] == 0.0))


def test_cp4_observation_contract_is_jit_stable() -> None:
    config = _config((2, 2))
    state, *_ = reset(config, jax.random.key(9))
    current_action_mask = _current_action_mask(config, state)
    eager = _observation_after_step(
        config,
        state,
        current_action_mask,
        _action(),
        jax.random.key(10),
    )
    compiled = cast(
        Observation,
        jax.jit(_observation_after_step)(
            config,
            state,
            current_action_mask,
            _action(),
            jax.random.key(10),
        ),
    )

    assert jax.tree_util.tree_structure(eager) == jax.tree_util.tree_structure(compiled)
    for eager_leaf, compiled_leaf in zip(
        jax.tree_util.tree_leaves(eager),
        jax.tree_util.tree_leaves(compiled),
        strict=True,
    ):
        assert bool(jnp.array_equal(eager_leaf, compiled_leaf))
