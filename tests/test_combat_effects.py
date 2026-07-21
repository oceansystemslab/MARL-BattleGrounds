"""Semantic proofs for accepted basic combat effects in Milestone 5 Step 5."""
# pyright: reportPrivateUsage=false

from collections.abc import Sequence
from typing import cast

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from marl_battlegrounds.core.combat import (
    BASIC_DAMAGE_BY_CLASS,
    BASIC_HEALING_BY_CLASS,
    HUNTER_BASIC_SLOW_DURATION_TICKS,
    HUNTER_BASIC_SLOW_MULTIPLIER,
    MAGE_BURST_DAMAGE_MULTIPLIER,
    MAGE_DAMAGE_AURA_MULTIPLIER,
    MAGE_DAMAGE_AURA_RADIUS,
    PRIEST_HEAL_SPEED_FLOOR,
    PRIEST_HEAL_SPEED_FLOOR_DURATION_TICKS,
    ROGUE_POISON_ANTI_HEAL_MULTIPLIER,
    ROGUE_POISON_SLOW_MULTIPLIER,
    ULTIMATE_COOLDOWN_BY_CLASS,
    ULTIMATE_DAMAGE_BY_CLASS,
    WARRIOR_CHARGE_SLOW_MULTIPLIER,
    WARRIOR_DAMAGE_MITIGATION_AURA_MULTIPLIER,
)
from marl_battlegrounds.core.config import resolve_agent_profile
from marl_battlegrounds.core.env import (
    _build_observation_and_action_mask,
    _compute_global_pairwise_distances_from_agent_positions,
    _derive_aura_damage_multipliers,
    step,
)
from marl_battlegrounds.core.types import (
    AGENT_FEATURE_CURRENT_HEALTH,
    AGENT_FEATURE_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_DURATION,
    AGENT_FEATURE_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_FRACTION,
    AGENT_FEATURE_SLOW_HUNTER_BASIC_DURATION,
    AGENT_FEATURE_SLOW_HUNTER_BASIC_MULTIPLIER,
    AGENT_FEATURE_STUN_WARRIOR_CHARGE_DURATION,
    ENVIRONMENT_DIMENSIONS,
    HUNTER_CLASS_ID,
    MAGE_CLASS_ID,
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    MAX_OBSTACLE_SLOTS,
    MOVE_EAST,
    MOVE_STAY,
    MOVE_WEST,
    NEUTRAL_CLASS_ID,
    NUM_SLOW_CHANNELS,
    NUM_STUN_CHANNELS,
    OBSTACLE_FEATURE_ACTIVE,
    OBSTACLE_FEATURE_RADIUS,
    OBSTACLE_FEATURE_TYPE,
    OBSTACLE_FEATURE_X,
    OBSTACLE_FEATURE_Y,
    OBSTACLE_FEATURES,
    OBSTACLE_TYPE_PILLAR,
    PRIEST_CLASS_ID,
    ROGUE_CLASS_ID,
    SLOW_CHANNEL_HUNTER_BASIC,
    SLOW_CHANNEL_ROGUE_POISON,
    SLOW_CHANNEL_WARRIOR_CHARGE,
    STUN_CHANNEL_WARRIOR_CHARGE,
    WARRIOR_CLASS_ID,
    Action,
    ActionMask,
    DoneFlags,
    EnvConfig,
    EnvState,
    Info,
    Observation,
    Reward,
)

_TEAM_A_ACTOR = 0
_TEAM_A_ALLY = 1
_TEAM_B_ACTOR = MAX_AGENTS_PER_TEAM
_FIRST_ALLY_TARGET = 1
_FIRST_ENEMY_TARGET = 1 + MAX_AGENTS_PER_TEAM


def _empty_obstacles() -> Array:
    """Return an inactive fixed-size obstacle table."""
    return jnp.zeros((MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES), dtype=jnp.float32)


def _blocking_pillar(*, x: float, y: float, radius: float = 0.75) -> Array:
    """Return one active pillar at the supplied center."""
    obstacles = _empty_obstacles()
    obstacles = obstacles.at[0, OBSTACLE_FEATURE_TYPE].set(OBSTACLE_TYPE_PILLAR)
    obstacles = obstacles.at[0, OBSTACLE_FEATURE_X].set(x)
    obstacles = obstacles.at[0, OBSTACLE_FEATURE_Y].set(y)
    obstacles = obstacles.at[0, OBSTACLE_FEATURE_RADIUS].set(radius)
    return obstacles.at[0, OBSTACLE_FEATURE_ACTIVE].set(1.0)


def _requested_roster(*class_rows: tuple[int, int]) -> Array:
    """Return a padded class roster with selected slot overrides."""
    roster = jnp.full((MAX_AGENT_SLOTS,), NEUTRAL_CLASS_ID, dtype=jnp.int32)
    for slot, class_id in class_rows:
        roster = roster.at[slot].set(class_id)
    return roster


def _default_positions(team_sizes: tuple[int, int]) -> Array:
    """Place active team blocks on two clear, non-overlapping vertical lines."""
    positions = jnp.zeros((MAX_AGENT_SLOTS, ENVIRONMENT_DIMENSIONS), dtype=jnp.float32)
    for local_slot in range(team_sizes[0]):
        positions = positions.at[local_slot].set(
            jnp.asarray((2.0, 2.0 + 1.5 * local_slot), dtype=jnp.float32)
        )
    for local_slot in range(team_sizes[1]):
        positions = positions.at[MAX_AGENTS_PER_TEAM + local_slot].set(
            jnp.asarray((7.0, 2.0 + 1.5 * local_slot), dtype=jnp.float32)
        )
    return positions


def _scenario(
    *class_rows: tuple[int, int],
    team_sizes: tuple[int, int] = (1, 1),
    positions: Array | None = None,
    obstacles: Array | None = None,
    observation_radius: float = 10.0,
    basic_radius: float = 10.0,
    ultimate_radius: float = 10.0,
) -> tuple[EnvConfig, EnvState]:
    """Build a deterministic fixed-slot combat scenario."""
    profile = resolve_agent_profile(
        _requested_roster(*class_rows),
        jnp.asarray(team_sizes, dtype=jnp.int32),
    )
    profile = profile._replace(
        observation_radii=jnp.where(
            profile.active_mask, observation_radius, 0.0
        ).astype(jnp.float32),
        basic_interaction_radii=jnp.where(
            profile.active_mask, basic_radius, 0.0
        ).astype(jnp.float32),
        ultimate_interaction_radii=jnp.where(
            profile.active_mask, ultimate_radius, 0.0
        ).astype(jnp.float32),
    )
    config = EnvConfig(
        max_steps=100,
        map_width=20.0,
        map_height=12.0,
        obstacles=_empty_obstacles() if obstacles is None else obstacles,
        agent_profile=profile,
        initial_agent_positions=(
            _default_positions(team_sizes) if positions is None else positions
        ),
    )
    state = EnvState(
        step_count=jnp.array(0, dtype=jnp.int32),
        agent_positions=(
            _default_positions(team_sizes) if positions is None else positions
        ),
        alive_mask=profile.active_mask,
        current_health=profile.max_health,
        ultimate_cooldowns=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
        slow_durations=jnp.zeros((MAX_AGENT_SLOTS, NUM_SLOW_CHANNELS), dtype=jnp.int32),
        stun_durations=jnp.zeros((MAX_AGENT_SLOTS, NUM_STUN_CHANNELS), dtype=jnp.int32),
        rogue_poison_anti_heal_durations=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
        mage_burst_damage_amplification_durations=jnp.zeros(
            (MAX_AGENT_SLOTS,), dtype=jnp.int32
        ),
        priest_blessing_of_freedom_slow_floor_durations=jnp.zeros(
            (MAX_AGENT_SLOTS,), dtype=jnp.int32
        ),
    )
    return config, state


def _joint_action(
    *rows: tuple[int, int, int, int],
) -> Action:
    """Return a canonical joint action with selected actor overrides.

    Each row is ``(actor_slot, move, target, use_ultimate)``.
    """
    move = jnp.full((MAX_AGENT_SLOTS,), MOVE_STAY, dtype=jnp.int32)
    target = jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32)
    ultimate = jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32)
    for actor_slot, move_action, target_action, use_ultimate in rows:
        move = move.at[actor_slot].set(move_action)
        target = target.at[actor_slot].set(target_action)
        ultimate = ultimate.at[actor_slot].set(use_ultimate)
    return Action(move=move, select_target=target, use_ultimate=ultimate)


def _current_action_mask(config: EnvConfig, state: EnvState) -> ActionMask:
    """Return the authoritative action mask paired with a test state."""
    _, action_mask = _build_observation_and_action_mask(state, config)
    return action_mask


def _step(
    config: EnvConfig,
    state: EnvState,
    action: Action,
) -> tuple[EnvState, Observation, ActionMask]:
    """Advance one transition and return the state-facing public outputs."""
    next_state, observation, _, _, next_action_mask, _ = step(
        config,
        state,
        _current_action_mask(config, state),
        action,
        jax.random.key(17),
    )
    return next_state, observation, next_action_mask


def _aura_multipliers(config: EnvConfig, state: EnvState) -> tuple[Array, Array]:
    """Derive both aura vectors from a test scenario's current snapshot."""
    distances = _compute_global_pairwise_distances_from_agent_positions(
        state.agent_positions
    )
    return _derive_aura_damage_multipliers(config, distances, state.alive_mask)


def _assert_only_health_changed(
    before: EnvState,
    after: EnvState,
    expected_health: Array,
) -> None:
    """Assert health resolution and fields outside basic-passive ownership."""
    assert bool(jnp.array_equal(after.current_health, expected_health))
    assert int(after.step_count) == int(before.step_count) + 1
    assert bool(jnp.array_equal(after.agent_positions, before.agent_positions))
    assert bool(jnp.array_equal(after.alive_mask, before.alive_mask))
    assert bool(jnp.array_equal(after.ultimate_cooldowns, before.ultimate_cooldowns))
    assert bool(
        jnp.array_equal(after.stun_durations, jnp.maximum(before.stun_durations - 1, 0))
    )
    assert bool(
        jnp.array_equal(
            after.rogue_poison_anti_heal_durations,
            jnp.maximum(before.rogue_poison_anti_heal_durations - 1, 0),
        )
    )
    assert bool(
        jnp.array_equal(
            after.mage_burst_damage_amplification_durations,
            jnp.maximum(before.mage_burst_damage_amplification_durations - 1, 0),
        )
    )


def test_aura_derivation_is_neutral_without_an_emitter() -> None:
    """Prove ordinary damage classes do not create passive modifiers."""
    config, state = _scenario(
        (_TEAM_A_ACTOR, HUNTER_CLASS_ID),
        (_TEAM_B_ACTOR, ROGUE_CLASS_ID),
    )

    mage_multipliers, warrior_multipliers = _aura_multipliers(config, state)

    assert mage_multipliers.shape == warrior_multipliers.shape == (MAX_AGENT_SLOTS,)
    assert mage_multipliers.dtype == warrior_multipliers.dtype == jnp.float32
    assert bool(jnp.all(mage_multipliers == 1.0))
    assert bool(jnp.all(warrior_multipliers == 1.0))


def test_mage_aura_includes_self_and_boundary_ally_but_excludes_others() -> None:
    """Prove inclusive range, team ownership, and self-benefit semantics."""
    positions = _default_positions((3, 1))
    positions = positions.at[0].set(jnp.asarray((4.0, 4.0), dtype=jnp.float32))
    positions = positions.at[1].set(
        jnp.asarray((4.0 + MAGE_DAMAGE_AURA_RADIUS, 4.0), dtype=jnp.float32)
    )
    positions = positions.at[2].set(
        jnp.asarray((4.1 + MAGE_DAMAGE_AURA_RADIUS, 4.0), dtype=jnp.float32)
    )
    positions = positions.at[_TEAM_B_ACTOR].set(
        jnp.asarray((4.0, 4.0), dtype=jnp.float32)
    )
    config, state = _scenario(
        (0, MAGE_CLASS_ID),
        (1, HUNTER_CLASS_ID),
        (2, ROGUE_CLASS_ID),
        (_TEAM_B_ACTOR, HUNTER_CLASS_ID),
        team_sizes=(3, 1),
        positions=positions,
    )

    mage_multipliers, _ = _aura_multipliers(config, state)

    expected = jnp.ones((MAX_AGENT_SLOTS,), dtype=jnp.float32)
    expected = expected.at[jnp.asarray((0, 1))].set(MAGE_DAMAGE_AURA_MULTIPLIER)
    assert bool(jnp.array_equal(mage_multipliers, expected))


@pytest.mark.parametrize(
    "emitter_state",
    [
        pytest.param("dead", id="dead-emitter"),
        pytest.param("inactive", id="inactive-emitter"),
    ],
)
def test_nonparticipating_mage_does_not_emit_an_aura(emitter_state: str) -> None:
    """Prove both liveness and configured activity gate aura emission."""
    config, state = _scenario(
        (0, MAGE_CLASS_ID),
        (1, HUNTER_CLASS_ID),
        (_TEAM_B_ACTOR, HUNTER_CLASS_ID),
        team_sizes=(2, 1),
    )
    if emitter_state == "dead":
        state = state._replace(alive_mask=state.alive_mask.at[0].set(False))
    else:
        profile = config.agent_profile._replace(
            active_mask=config.agent_profile.active_mask.at[0].set(False)
        )
        config = config._replace(agent_profile=profile)

    mage_multipliers, _ = _aura_multipliers(config, state)

    assert bool(jnp.all(mage_multipliers == 1.0))


def test_zero_health_living_mage_still_emits_an_aura() -> None:
    """Prove liveness, rather than current health, owns participation."""
    config, state = _scenario(
        (0, MAGE_CLASS_ID),
        (1, HUNTER_CLASS_ID),
        (_TEAM_B_ACTOR, HUNTER_CLASS_ID),
        team_sizes=(2, 1),
    )
    state = state._replace(current_health=state.current_health.at[0].set(0.0))

    mage_multipliers, _ = _aura_multipliers(config, state)

    assert mage_multipliers[0] == MAGE_DAMAGE_AURA_MULTIPLIER
    assert mage_multipliers[1] == MAGE_DAMAGE_AURA_MULTIPLIER


def test_duplicate_auras_stack_multiplicatively_and_ignore_emitter_order() -> None:
    """Prove duplicate Mage and Warrior emitters multiply deterministically."""
    class_rows = (
        (0, MAGE_CLASS_ID),
        (1, WARRIOR_CLASS_ID),
        (2, MAGE_CLASS_ID),
        (3, WARRIOR_CLASS_ID),
        (_TEAM_B_ACTOR, HUNTER_CLASS_ID),
    )
    positions = _default_positions((4, 1))
    positions = positions.at[0].set(jnp.asarray((3.0, 3.0), dtype=jnp.float32))
    positions = positions.at[1].set(jnp.asarray((3.0, 3.5), dtype=jnp.float32))
    positions = positions.at[2].set(jnp.asarray((3.0, 4.0), dtype=jnp.float32))
    positions = positions.at[3].set(jnp.asarray((3.0, 4.5), dtype=jnp.float32))
    config, state = _scenario(*class_rows, team_sizes=(4, 1), positions=positions)
    first = _aura_multipliers(config, state)

    swapped_class_ids = (
        config.agent_profile.class_ids.at[0]
        .set(WARRIOR_CLASS_ID)
        .at[1]
        .set(MAGE_CLASS_ID)
        .at[2]
        .set(WARRIOR_CLASS_ID)
        .at[3]
        .set(MAGE_CLASS_ID)
    )
    second_config = config._replace(
        agent_profile=config.agent_profile._replace(class_ids=swapped_class_ids)
    )
    second = _aura_multipliers(second_config, state)

    expected_mage = MAGE_DAMAGE_AURA_MULTIPLIER**2
    expected_warrior = WARRIOR_DAMAGE_MITIGATION_AURA_MULTIPLIER**2
    assert bool(jnp.allclose(first[0][:4], expected_mage))
    assert bool(jnp.allclose(first[1][:4], expected_warrior))
    assert bool(jnp.array_equal(first[0], second[0]))
    assert bool(jnp.array_equal(first[1], second[1]))


def test_aura_derivation_is_independent_of_los_and_observation_radius() -> None:
    """Prove aura attachment depends only on spatial range and participation."""
    positions = _default_positions((2, 1))
    positions = positions.at[0].set(jnp.asarray((2.0, 2.0), dtype=jnp.float32))
    positions = positions.at[1].set(jnp.asarray((3.0, 2.0), dtype=jnp.float32))
    config, state = _scenario(
        (0, MAGE_CLASS_ID),
        (1, HUNTER_CLASS_ID),
        (_TEAM_B_ACTOR, HUNTER_CLASS_ID),
        team_sizes=(2, 1),
        positions=positions,
        obstacles=_blocking_pillar(x=2.5, y=2.0, radius=0.25),
        observation_radius=0.25,
    )

    mage_multipliers, _ = _aura_multipliers(config, state)

    assert mage_multipliers[1] == MAGE_DAMAGE_AURA_MULTIPLIER


def test_aura_derivation_matches_jit() -> None:
    """Prove fixed-shape aura reduction is stable under compilation."""
    config, state = _scenario(
        (0, MAGE_CLASS_ID),
        (1, WARRIOR_CLASS_ID),
        (_TEAM_B_ACTOR, HUNTER_CLASS_ID),
        team_sizes=(2, 1),
    )
    distances = _compute_global_pairwise_distances_from_agent_positions(
        state.agent_positions
    )

    eager = _derive_aura_damage_multipliers(config, distances, state.alive_mask)
    compiled = cast(
        tuple[Array, Array],
        jax.jit(_derive_aura_damage_multipliers)(config, distances, state.alive_mask),
    )

    assert bool(jnp.array_equal(eager[0], compiled[0]))
    assert bool(jnp.array_equal(eager[1], compiled[1]))


def test_mage_aura_amplifies_allied_damage_but_not_healing() -> None:
    """Prove the outgoing aura modifies damage contributions exclusively."""
    config, state = _scenario(
        (0, MAGE_CLASS_ID),
        (1, HUNTER_CLASS_ID),
        (2, PRIEST_CLASS_ID),
        (_TEAM_B_ACTOR, HUNTER_CLASS_ID),
        team_sizes=(3, 1),
    )
    state = state._replace(current_health=state.current_health.at[2].add(-20.0))
    action = _joint_action(
        (1, MOVE_STAY, _FIRST_ENEMY_TARGET, 0),
        (2, MOVE_STAY, 3, 0),
    )

    next_state, _, _ = _step(config, state, action)

    expected_health = state.current_health
    expected_health = expected_health.at[_TEAM_B_ACTOR].add(
        -BASIC_DAMAGE_BY_CLASS[HUNTER_CLASS_ID] * MAGE_DAMAGE_AURA_MULTIPLIER
    )
    expected_health = expected_health.at[2].add(BASIC_HEALING_BY_CLASS[PRIEST_CLASS_ID])
    _assert_only_health_changed(state, next_state, expected_health)


def test_mage_burst_applies_only_to_its_owner_and_stacks_with_aura() -> None:
    """Prove the duration-derived buff is actor-local and multiplicative."""
    config, state = _scenario(
        (0, MAGE_CLASS_ID),
        (1, HUNTER_CLASS_ID),
        (_TEAM_B_ACTOR, HUNTER_CLASS_ID),
        team_sizes=(2, 1),
    )
    state = state._replace(
        mage_burst_damage_amplification_durations=(
            state.mage_burst_damage_amplification_durations.at[0].set(2)
        )
    )
    action = _joint_action(
        (0, MOVE_STAY, _FIRST_ENEMY_TARGET, 0),
        (1, MOVE_STAY, _FIRST_ENEMY_TARGET, 0),
    )

    next_state, _, _ = _step(config, state, action)

    expected_damage = (
        BASIC_DAMAGE_BY_CLASS[MAGE_CLASS_ID]
        * MAGE_BURST_DAMAGE_MULTIPLIER
        * MAGE_DAMAGE_AURA_MULTIPLIER
        + BASIC_DAMAGE_BY_CLASS[HUNTER_CLASS_ID] * MAGE_DAMAGE_AURA_MULTIPLIER
    )
    expected_health = state.current_health.at[_TEAM_B_ACTOR].add(-expected_damage)
    _assert_only_health_changed(state, next_state, expected_health)


def test_warrior_aura_mitigates_allied_incoming_damage() -> None:
    """Prove mitigation is selected by the final global recipient slot."""
    config, state = _scenario(
        (0, WARRIOR_CLASS_ID),
        (1, HUNTER_CLASS_ID),
        (_TEAM_B_ACTOR, HUNTER_CLASS_ID),
        team_sizes=(2, 1),
    )
    action = _joint_action(
        (_TEAM_B_ACTOR, MOVE_STAY, _FIRST_ENEMY_TARGET + 1, 0),
    )

    next_state, _, _ = _step(config, state, action)

    expected_health = state.current_health.at[1].add(
        -BASIC_DAMAGE_BY_CLASS[HUNTER_CLASS_ID]
        * WARRIOR_DAMAGE_MITIGATION_AURA_MULTIPLIER
    )
    _assert_only_health_changed(state, next_state, expected_health)


def test_rogue_anti_heal_reduces_healing_without_reducing_damage() -> None:
    """Prove anti-heal is a recipient-side healing modifier only."""
    config, state = _scenario(
        (0, PRIEST_CLASS_ID),
        (1, HUNTER_CLASS_ID),
        (_TEAM_B_ACTOR, HUNTER_CLASS_ID),
        team_sizes=(2, 1),
    )
    state = state._replace(
        current_health=state.current_health.at[1].add(-30.0),
        rogue_poison_anti_heal_durations=(
            state.rogue_poison_anti_heal_durations.at[1].set(3)
        ),
    )
    action = _joint_action(
        (0, MOVE_STAY, _FIRST_ALLY_TARGET + 1, 0),
        (_TEAM_B_ACTOR, MOVE_STAY, _FIRST_ENEMY_TARGET + 1, 0),
    )

    next_state, _, _ = _step(config, state, action)

    expected_health = state.current_health.at[1].add(
        -BASIC_DAMAGE_BY_CLASS[HUNTER_CLASS_ID]
        + BASIC_HEALING_BY_CLASS[PRIEST_CLASS_ID] * ROGUE_POISON_ANTI_HEAL_MULTIPLIER
    )
    _assert_only_health_changed(state, next_state, expected_health)


def test_outgoing_amplification_and_incoming_mitigation_compose() -> None:
    """Prove actor-side and recipient-side aura modifiers both apply once."""
    config, state = _scenario(
        (0, MAGE_CLASS_ID),
        (1, HUNTER_CLASS_ID),
        (_TEAM_B_ACTOR, WARRIOR_CLASS_ID),
        (_TEAM_B_ACTOR + 1, HUNTER_CLASS_ID),
        team_sizes=(2, 2),
    )
    action = _joint_action(
        (1, MOVE_STAY, _FIRST_ENEMY_TARGET + 1, 0),
    )

    next_state, _, _ = _step(config, state, action)

    expected_health = state.current_health.at[_TEAM_B_ACTOR + 1].add(
        -BASIC_DAMAGE_BY_CLASS[HUNTER_CLASS_ID]
        * MAGE_DAMAGE_AURA_MULTIPLIER
        * WARRIOR_DAMAGE_MITIGATION_AURA_MULTIPLIER
    )
    _assert_only_health_changed(state, next_state, expected_health)


def test_hunter_basic_slow_is_observed_before_its_affected_decision() -> None:
    """Prove a duration-one Hunter slow governs one later decision window."""
    config, state = _scenario(
        (_TEAM_A_ACTOR, HUNTER_CLASS_ID),
        (_TEAM_B_ACTOR, HUNTER_CLASS_ID),
    )
    state = state._replace(
        current_health=state.current_health.at[_TEAM_B_ACTOR].set(0.0)
    )
    action = _joint_action(
        (_TEAM_A_ACTOR, MOVE_STAY, _FIRST_ENEMY_TARGET, 0),
        (_TEAM_B_ACTOR, MOVE_EAST, 0, 0),
    )

    applied_state, applied_observation, _ = _step(config, state, action)

    application_displacement = (
        applied_state.agent_positions[_TEAM_B_ACTOR]
        - state.agent_positions[_TEAM_B_ACTOR]
    )
    assert bool(
        jnp.isclose(
            application_displacement[0],
            config.agent_profile.base_movement_speeds[_TEAM_B_ACTOR],
        )
    )
    assert application_displacement[1] == 0.0
    assert applied_state.current_health[_TEAM_B_ACTOR] == 0.0
    assert (
        applied_state.slow_durations[_TEAM_B_ACTOR, SLOW_CHANNEL_HUNTER_BASIC]
        == HUNTER_BASIC_SLOW_DURATION_TICKS
    )
    assert (
        applied_observation.self_features[
            _TEAM_B_ACTOR, AGENT_FEATURE_SLOW_HUNTER_BASIC_DURATION
        ]
        == HUNTER_BASIC_SLOW_DURATION_TICKS
    )
    assert (
        applied_observation.self_features[
            _TEAM_B_ACTOR, AGENT_FEATURE_SLOW_HUNTER_BASIC_MULTIPLIER
        ]
        == HUNTER_BASIC_SLOW_MULTIPLIER
    )

    expired_state, expired_observation, _ = _step(
        config,
        applied_state,
        _joint_action((_TEAM_B_ACTOR, MOVE_EAST, 0, 0)),
    )

    governed_displacement = (
        expired_state.agent_positions[_TEAM_B_ACTOR]
        - applied_state.agent_positions[_TEAM_B_ACTOR]
    )
    assert bool(
        jnp.isclose(
            governed_displacement[0],
            config.agent_profile.base_movement_speeds[_TEAM_B_ACTOR]
            * HUNTER_BASIC_SLOW_MULTIPLIER,
        )
    )
    assert expired_state.slow_durations[_TEAM_B_ACTOR, SLOW_CHANNEL_HUNTER_BASIC] == 0
    assert (
        expired_observation.self_features[
            _TEAM_B_ACTOR, AGENT_FEATURE_SLOW_HUNTER_BASIC_DURATION
        ]
        == 0.0
    )


def test_fresh_charge_stun_preserves_precommitted_movement_then_becomes_public() -> (
    None
):
    """Prove fresh control cannot cancel an action chosen before it was visible."""
    config, state = _scenario(
        (_TEAM_A_ACTOR, WARRIOR_CLASS_ID),
        (_TEAM_B_ACTOR, HUNTER_CLASS_ID),
    )
    action = _joint_action(
        (_TEAM_A_ACTOR, MOVE_STAY, _FIRST_ENEMY_TARGET, 1),
        (_TEAM_B_ACTOR, MOVE_EAST, 0, 0),
    )

    applied_state, applied_observation, applied_mask = _step(config, state, action)

    application_displacement = (
        applied_state.agent_positions[_TEAM_B_ACTOR]
        - state.agent_positions[_TEAM_B_ACTOR]
    )
    assert bool(
        jnp.isclose(
            application_displacement[0],
            config.agent_profile.base_movement_speeds[_TEAM_B_ACTOR],
        )
    )
    assert applied_state.stun_durations[_TEAM_B_ACTOR, STUN_CHANNEL_WARRIOR_CHARGE] == 1
    assert (
        applied_observation.self_features[
            _TEAM_B_ACTOR, AGENT_FEATURE_STUN_WARRIOR_CHARGE_DURATION
        ]
        == 1.0
    )
    assert (
        applied_observation.enemy_unit_features[
            _TEAM_A_ACTOR, 0, AGENT_FEATURE_STUN_WARRIOR_CHARGE_DURATION
        ]
        == 1.0
    )
    assert not bool(jnp.any(applied_mask.select_target_mask[_TEAM_B_ACTOR, 1:]))

    expired_state, expired_observation, expired_mask = _step(
        config,
        applied_state,
        _joint_action(),
    )

    assert expired_state.stun_durations[_TEAM_B_ACTOR, STUN_CHANNEL_WARRIOR_CHARGE] == 0
    assert (
        expired_observation.self_features[
            _TEAM_B_ACTOR, AGENT_FEATURE_STUN_WARRIOR_CHARGE_DURATION
        ]
        == 0.0
    )
    assert bool(jnp.any(expired_mask.select_target_mask[_TEAM_B_ACTOR, 1:]))


@pytest.mark.parametrize(
    ("has_existing_slows", "expected_speed_multiplier"),
    (
        pytest.param(True, PRIEST_HEAL_SPEED_FLOOR, id="raises-slowed-ally-to-floor"),
        pytest.param(False, 1.0, id="does-not-accelerate-unslowed-ally"),
    ),
)
def test_priest_freedom_is_observed_before_its_affected_decision(
    has_existing_slows: bool,
    expected_speed_multiplier: float,
) -> None:
    """Prove full-health healing grants a floor for one later decision."""
    config, state = _scenario(
        (0, PRIEST_CLASS_ID),
        (1, HUNTER_CLASS_ID),
        (_TEAM_B_ACTOR, HUNTER_CLASS_ID),
        team_sizes=(2, 1),
    )
    if has_existing_slows:
        state = state._replace(
            slow_durations=(
                state.slow_durations.at[1, SLOW_CHANNEL_WARRIOR_CHARGE]
                .set(3)
                .at[1, SLOW_CHANNEL_ROGUE_POISON]
                .set(4)
            )
        )
    action = _joint_action(
        (0, MOVE_STAY, _FIRST_ALLY_TARGET + 1, 0),
        (1, MOVE_EAST, 0, 0),
    )

    applied_state, applied_observation, _ = _step(config, state, action)

    application_displacement = (
        applied_state.agent_positions[1] - state.agent_positions[1]
    )
    application_speed_multiplier = (
        WARRIOR_CHARGE_SLOW_MULTIPLIER * ROGUE_POISON_SLOW_MULTIPLIER
        if has_existing_slows
        else 1.0
    )
    assert bool(
        jnp.isclose(
            application_displacement[0],
            config.agent_profile.base_movement_speeds[1] * application_speed_multiplier,
        )
    )
    assert bool(
        jnp.array_equal(
            applied_state.slow_durations,
            jnp.maximum(state.slow_durations - 1, 0),
        )
    )
    assert (
        applied_state.priest_blessing_of_freedom_slow_floor_durations[1]
        == PRIEST_HEAL_SPEED_FLOOR_DURATION_TICKS
    )
    assert (
        applied_observation.self_features[
            1, AGENT_FEATURE_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_DURATION
        ]
        == PRIEST_HEAL_SPEED_FLOOR_DURATION_TICKS
    )
    assert (
        applied_observation.self_features[
            1, AGENT_FEATURE_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_FRACTION
        ]
        == PRIEST_HEAL_SPEED_FLOOR
    )

    expired_state, expired_observation, _ = _step(
        config,
        applied_state,
        _joint_action((1, MOVE_EAST, 0, 0)),
    )

    governed_displacement = (
        expired_state.agent_positions[1] - applied_state.agent_positions[1]
    )
    assert bool(
        jnp.isclose(
            governed_displacement[0],
            config.agent_profile.base_movement_speeds[1] * expected_speed_multiplier,
        )
    )
    assert expired_state.priest_blessing_of_freedom_slow_floor_durations[1] == 0
    assert (
        expired_observation.self_features[
            1, AGENT_FEATURE_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_DURATION
        ]
        == 0.0
    )


def test_passive_refresh_never_shortens_and_only_owned_channels_tick() -> None:
    """Prove source-local max refresh within the complete duration lifecycle."""
    recipient = _TEAM_B_ACTOR + 1
    config, state = _scenario(
        (_TEAM_A_ACTOR, HUNTER_CLASS_ID),
        (_TEAM_B_ACTOR, PRIEST_CLASS_ID),
        (recipient, HUNTER_CLASS_ID),
        team_sizes=(1, 2),
    )
    slow_durations = state.slow_durations
    slow_durations = slow_durations.at[recipient, SLOW_CHANNEL_WARRIOR_CHARGE].set(5)
    slow_durations = slow_durations.at[recipient, SLOW_CHANNEL_HUNTER_BASIC].set(4)
    slow_durations = slow_durations.at[recipient, SLOW_CHANNEL_ROGUE_POISON].set(3)
    state = state._replace(
        slow_durations=slow_durations,
        ultimate_cooldowns=state.ultimate_cooldowns.at[recipient].set(7),
        stun_durations=state.stun_durations.at[recipient, 0].set(2),
        rogue_poison_anti_heal_durations=(
            state.rogue_poison_anti_heal_durations.at[recipient].set(5)
        ),
        mage_burst_damage_amplification_durations=(
            state.mage_burst_damage_amplification_durations.at[recipient].set(6)
        ),
        priest_blessing_of_freedom_slow_floor_durations=(
            state.priest_blessing_of_freedom_slow_floor_durations.at[recipient].set(4)
        ),
    )
    action = _joint_action(
        (_TEAM_A_ACTOR, MOVE_STAY, _FIRST_ENEMY_TARGET + 1, 0),
        (_TEAM_B_ACTOR, MOVE_STAY, _FIRST_ALLY_TARGET + 1, 0),
    )

    next_state, _, _ = _step(config, state, action)

    assert next_state.slow_durations[recipient, SLOW_CHANNEL_HUNTER_BASIC] == 3
    assert next_state.slow_durations[recipient, SLOW_CHANNEL_WARRIOR_CHARGE] == 4
    assert next_state.slow_durations[recipient, SLOW_CHANNEL_ROGUE_POISON] == 2
    assert next_state.priest_blessing_of_freedom_slow_floor_durations[recipient] == 3
    assert bool(
        jnp.array_equal(
            next_state.ultimate_cooldowns,
            jnp.maximum(state.ultimate_cooldowns - 1, 0),
        )
    )
    assert bool(
        jnp.array_equal(
            next_state.stun_durations,
            jnp.maximum(state.stun_durations - 1, 0),
        )
    )
    assert bool(
        jnp.array_equal(
            next_state.rogue_poison_anti_heal_durations,
            jnp.maximum(state.rogue_poison_anti_heal_durations - 1, 0),
        )
    )
    assert bool(
        jnp.array_equal(
            next_state.mage_burst_damage_amplification_durations,
            jnp.maximum(state.mage_burst_damage_amplification_durations - 1, 0),
        )
    )


@pytest.mark.parametrize(
    "invalidity",
    (
        pytest.param("target-none", id="target-none"),
        pytest.param("wrong-relation", id="wrong-relation"),
        pytest.param("blocked-los", id="blocked-los"),
        pytest.param("stunned", id="stunned"),
        pytest.param("ultimate-lane", id="ultimate-lane"),
    ),
)
def test_invalid_hunter_basic_does_not_slow_independent_movement(
    invalidity: str,
) -> None:
    """Prove rejected or ultimate-lane interactions cannot trigger the passive."""
    team_sizes = (2, 1) if invalidity == "wrong-relation" else (1, 1)
    obstacles = _blocking_pillar(x=4.5, y=2.0) if invalidity == "blocked-los" else None
    config, state = _scenario(
        (_TEAM_A_ACTOR, HUNTER_CLASS_ID),
        (_TEAM_A_ALLY, HUNTER_CLASS_ID),
        (_TEAM_B_ACTOR, HUNTER_CLASS_ID),
        team_sizes=team_sizes,
        obstacles=obstacles,
    )
    target_action = _FIRST_ENEMY_TARGET
    use_ultimate = 0
    if invalidity == "target-none":
        target_action = 0
    elif invalidity == "wrong-relation":
        target_action = _FIRST_ALLY_TARGET + 1
    elif invalidity == "stunned":
        state = state._replace(stun_durations=state.stun_durations.at[0, 0].set(1))
    elif invalidity == "ultimate-lane":
        use_ultimate = 1
    action = _joint_action(
        (_TEAM_A_ACTOR, MOVE_STAY, target_action, use_ultimate),
        (_TEAM_B_ACTOR, MOVE_EAST, 0, 0),
    )

    next_state, _, _ = _step(config, state, action)

    displacement = (
        next_state.agent_positions[_TEAM_B_ACTOR] - state.agent_positions[_TEAM_B_ACTOR]
    )
    assert bool(
        jnp.isclose(
            displacement[0],
            config.agent_profile.base_movement_speeds[_TEAM_B_ACTOR],
        )
    )
    assert next_state.slow_durations[_TEAM_B_ACTOR, SLOW_CHANNEL_HUNTER_BASIC] == 0


@pytest.mark.parametrize(
    "invalidity",
    (
        pytest.param("target-none", id="target-none"),
        pytest.param("wrong-relation", id="wrong-relation"),
        pytest.param("out-of-range", id="out-of-range"),
        pytest.param("stunned", id="stunned"),
        pytest.param("ultimate-lane", id="ultimate-lane"),
    ),
)
def test_invalid_priest_basic_does_not_grant_freedom(
    invalidity: str,
) -> None:
    """Prove Priest Freedom requires an accepted no-ultimate healing basic."""
    basic_radius = 0.5 if invalidity == "out-of-range" else 10.0
    config, state = _scenario(
        (0, PRIEST_CLASS_ID),
        (1, HUNTER_CLASS_ID),
        (_TEAM_B_ACTOR, HUNTER_CLASS_ID),
        team_sizes=(2, 1),
        basic_radius=basic_radius,
    )
    state = state._replace(
        slow_durations=state.slow_durations.at[1, SLOW_CHANNEL_ROGUE_POISON].set(3)
    )
    target_action = _FIRST_ALLY_TARGET + 1
    use_ultimate = 0
    if invalidity == "target-none":
        target_action = 0
    elif invalidity == "wrong-relation":
        target_action = _FIRST_ENEMY_TARGET
    elif invalidity == "stunned":
        state = state._replace(stun_durations=state.stun_durations.at[0, 0].set(1))
    elif invalidity == "ultimate-lane":
        use_ultimate = 1
    action = _joint_action(
        (0, MOVE_STAY, target_action, use_ultimate),
        (1, MOVE_EAST, 0, 0),
    )

    next_state, _, _ = _step(config, state, action)

    displacement = next_state.agent_positions[1] - state.agent_positions[1]
    expected_displacement = (
        config.agent_profile.base_movement_speeds[1] * ROGUE_POISON_SLOW_MULTIPLIER
    )
    assert bool(jnp.isclose(displacement[0], expected_displacement))
    assert next_state.priest_blessing_of_freedom_slow_floor_durations[1] == 0


def test_fresh_hunter_slow_does_not_retroactively_change_boundary_projection() -> None:
    """Prove geometry consumes current rather than newly applied slow truth."""
    positions = _default_positions((1, 1))
    positions = positions.at[_TEAM_A_ACTOR].set(
        jnp.asarray((13.2, 2.0), dtype=jnp.float32)
    )
    positions = positions.at[_TEAM_B_ACTOR].set(
        jnp.asarray((18.6, 2.0), dtype=jnp.float32)
    )
    config, state = _scenario(
        (_TEAM_A_ACTOR, HUNTER_CLASS_ID),
        (_TEAM_B_ACTOR, HUNTER_CLASS_ID),
        positions=positions,
    )
    action = _joint_action(
        (_TEAM_A_ACTOR, MOVE_STAY, _FIRST_ENEMY_TARGET, 0),
        (_TEAM_B_ACTOR, MOVE_EAST, 0, 0),
    )

    next_state, _, _ = _step(config, state, action)

    assert next_state.agent_positions[_TEAM_B_ACTOR, 0] == 19.5
    assert (
        next_state.slow_durations[_TEAM_B_ACTOR, SLOW_CHANNEL_HUNTER_BASIC]
        == HUNTER_BASIC_SLOW_DURATION_TICKS
    )


def test_duplicate_hunter_applications_refresh_once_without_stacking_strength() -> None:
    """Prove simultaneous same-source applications remain order-independent."""
    config, state = _scenario(
        (0, HUNTER_CLASS_ID),
        (1, HUNTER_CLASS_ID),
        (_TEAM_B_ACTOR, HUNTER_CLASS_ID),
        team_sizes=(2, 1),
    )
    action = _joint_action(
        (0, MOVE_STAY, _FIRST_ENEMY_TARGET, 0),
        (1, MOVE_STAY, _FIRST_ENEMY_TARGET, 0),
        (_TEAM_B_ACTOR, MOVE_EAST, 0, 0),
    )

    next_state, _, _ = _step(config, state, action)

    displacement = (
        next_state.agent_positions[_TEAM_B_ACTOR] - state.agent_positions[_TEAM_B_ACTOR]
    )
    assert bool(jnp.isclose(displacement[0], 1.0))
    assert next_state.current_health[_TEAM_B_ACTOR] == (
        state.current_health[_TEAM_B_ACTOR] - 2 * BASIC_DAMAGE_BY_CLASS[HUNTER_CLASS_ID]
    )
    assert (
        next_state.slow_durations[_TEAM_B_ACTOR, SLOW_CHANNEL_HUNTER_BASIC]
        == HUNTER_BASIC_SLOW_DURATION_TICKS
    )


def test_duplicate_priest_applications_refresh_one_freedom_floor() -> None:
    """Prove simultaneous Priest basics grant one non-stacking movement floor."""
    config, state = _scenario(
        (0, PRIEST_CLASS_ID),
        (1, PRIEST_CLASS_ID),
        (2, HUNTER_CLASS_ID),
        (_TEAM_B_ACTOR, HUNTER_CLASS_ID),
        team_sizes=(3, 1),
    )
    state = state._replace(
        slow_durations=(
            state.slow_durations.at[2, SLOW_CHANNEL_WARRIOR_CHARGE]
            .set(3)
            .at[2, SLOW_CHANNEL_ROGUE_POISON]
            .set(3)
        )
    )
    action = _joint_action(
        (0, MOVE_STAY, _FIRST_ALLY_TARGET + 2, 0),
        (1, MOVE_STAY, _FIRST_ALLY_TARGET + 2, 0),
        (2, MOVE_EAST, 0, 0),
    )

    next_state, _, _ = _step(config, state, action)

    displacement = next_state.agent_positions[2] - state.agent_positions[2]
    assert bool(
        jnp.isclose(
            displacement[0],
            WARRIOR_CHARGE_SLOW_MULTIPLIER * ROGUE_POISON_SLOW_MULTIPLIER,
        )
    )
    assert bool(
        jnp.array_equal(
            next_state.slow_durations,
            jnp.maximum(state.slow_durations - 1, 0),
        )
    )
    assert (
        next_state.priest_blessing_of_freedom_slow_floor_durations[2]
        == PRIEST_HEAL_SPEED_FLOOR_DURATION_TICKS
    )


def test_jitted_step_matches_eager_decision_epoch_passive_semantics() -> None:
    """Prove decision-epoch passive timing remains JIT-stable."""
    config, state = _scenario(
        (0, HUNTER_CLASS_ID),
        (_TEAM_B_ACTOR, PRIEST_CLASS_ID),
        (_TEAM_B_ACTOR + 1, HUNTER_CLASS_ID),
        team_sizes=(1, 2),
    )
    recipient = _TEAM_B_ACTOR + 1
    state = state._replace(
        slow_durations=state.slow_durations.at[
            recipient, SLOW_CHANNEL_ROGUE_POISON
        ].set(3)
    )
    action = _joint_action(
        (0, MOVE_STAY, _FIRST_ENEMY_TARGET + 1, 0),
        (_TEAM_B_ACTOR, MOVE_STAY, _FIRST_ALLY_TARGET + 1, 0),
        (recipient, MOVE_EAST, 0, 0),
    )
    current_mask = _current_action_mask(config, state)

    eager_outputs = step(config, state, current_mask, action, jax.random.key(31))
    compiled_outputs = cast(
        tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info],
        jax.jit(step)(config, state, current_mask, action, jax.random.key(31)),
    )

    for eager_leaf, compiled_leaf in zip(
        jax.tree_util.tree_leaves(eager_outputs),
        jax.tree_util.tree_leaves(compiled_outputs),
        strict=True,
    ):
        assert bool(jnp.array_equal(eager_leaf, compiled_leaf))


def test_scanned_repeated_hunter_hits_refresh_for_each_next_movement() -> None:
    """Prove repeated duration-one applications govern each later scan step."""
    horizon = 3
    config, state = _scenario(
        (_TEAM_A_ACTOR, HUNTER_CLASS_ID),
        (_TEAM_B_ACTOR, HUNTER_CLASS_ID),
    )
    action = _joint_action(
        (_TEAM_A_ACTOR, MOVE_STAY, _FIRST_ENEMY_TARGET, 0),
        (_TEAM_B_ACTOR, MOVE_EAST, 0, 0),
    )
    keys = jax.random.split(jax.random.key(37), horizon)

    def _scan_step(
        carry: tuple[EnvState, ActionMask],
        step_key: Array,
    ) -> tuple[tuple[EnvState, ActionMask], tuple[Array, Array, Array]]:
        current_state, current_mask = carry
        next_state, _, _, _, next_mask, _ = step(
            config,
            current_state,
            current_mask,
            action,
            step_key,
        )
        return (next_state, next_mask), (
            next_state.current_health,
            next_state.agent_positions,
            next_state.slow_durations[:, SLOW_CHANNEL_HUNTER_BASIC],
        )

    def _rollout(
        initial_state: EnvState,
        initial_mask: ActionMask,
        scan_keys: Array,
    ) -> tuple[tuple[EnvState, ActionMask], tuple[Array, Array, Array]]:
        """Run the repeated-passive scenario in one fixed-shape scan."""
        return jax.lax.scan(
            _scan_step,
            (initial_state, initial_mask),
            scan_keys,
        )

    (_, _), (health_history, position_history, hunter_duration_history) = cast(
        tuple[tuple[EnvState, ActionMask], tuple[Array, Array, Array]],
        jax.jit(_rollout)(state, _current_action_mask(config, state), keys),
    )

    expected_health = state.current_health[_TEAM_B_ACTOR] - (
        BASIC_DAMAGE_BY_CLASS[HUNTER_CLASS_ID]
        * jnp.arange(1, horizon + 1, dtype=jnp.float32)
    )
    expected_x = state.agent_positions[_TEAM_B_ACTOR, 0] + jnp.asarray(
        (
            1.0,
            1.0 + HUNTER_BASIC_SLOW_MULTIPLIER,
            1.0 + 2 * HUNTER_BASIC_SLOW_MULTIPLIER,
        ),
        dtype=jnp.float32,
    )
    assert health_history.shape == (horizon, MAX_AGENT_SLOTS)
    assert position_history.shape == (
        horizon,
        MAX_AGENT_SLOTS,
        ENVIRONMENT_DIMENSIONS,
    )
    assert hunter_duration_history.dtype == jnp.int32
    assert bool(
        jnp.allclose(health_history[:, _TEAM_B_ACTOR], expected_health, atol=1e-5)
    )
    assert bool(
        jnp.allclose(position_history[:, _TEAM_B_ACTOR, 0], expected_x, atol=1e-5)
    )
    assert bool(
        jnp.all(
            hunter_duration_history[:, _TEAM_B_ACTOR]
            == HUNTER_BASIC_SLOW_DURATION_TICKS
        )
    )
    assert bool(jnp.all(hunter_duration_history.at[:, _TEAM_B_ACTOR].set(0) == 0))


@pytest.mark.parametrize(
    "actor_class_id",
    [
        pytest.param(MAGE_CLASS_ID, id="mage"),
        pytest.param(WARRIOR_CLASS_ID, id="warrior"),
        pytest.param(HUNTER_CLASS_ID, id="hunter"),
        pytest.param(ROGUE_CLASS_ID, id="rogue"),
    ],
)
def test_each_damage_class_applies_its_catalog_payload(actor_class_id: int) -> None:
    """Prove every offensive basic starts from its class catalog damage."""
    config, state = _scenario(
        (_TEAM_A_ACTOR, actor_class_id),
        (_TEAM_B_ACTOR, MAGE_CLASS_ID),
    )

    next_state, _, _ = _step(
        config,
        state,
        _joint_action(
            (_TEAM_A_ACTOR, MOVE_STAY, _FIRST_ENEMY_TARGET, 0),
        ),
    )

    outgoing_multiplier = (
        MAGE_DAMAGE_AURA_MULTIPLIER if actor_class_id == MAGE_CLASS_ID else 1.0
    )
    expected_health = state.current_health.at[_TEAM_B_ACTOR].add(
        -BASIC_DAMAGE_BY_CLASS[actor_class_id] * outgoing_multiplier
    )
    _assert_only_health_changed(state, next_state, expected_health)


@pytest.mark.parametrize(
    ("target_action", "recipient_slot"),
    [
        pytest.param(_FIRST_ALLY_TARGET, _TEAM_A_ACTOR, id="self"),
        pytest.param(_FIRST_ALLY_TARGET + 1, _TEAM_A_ALLY, id="ally"),
    ],
)
def test_priest_basic_heals_self_and_allies(
    target_action: int,
    recipient_slot: int,
) -> None:
    """Prove Priest healing uses the same stable allied candidate mapping."""
    config, state = _scenario(
        (_TEAM_A_ACTOR, PRIEST_CLASS_ID),
        (_TEAM_A_ALLY, MAGE_CLASS_ID),
        (_TEAM_B_ACTOR, MAGE_CLASS_ID),
        team_sizes=(2, 1),
    )
    state = state._replace(
        current_health=state.current_health.at[recipient_slot].add(-20.0)
    )

    next_state, _, _ = _step(
        config,
        state,
        _joint_action((_TEAM_A_ACTOR, MOVE_STAY, target_action, 0)),
    )

    expected_health = state.current_health.at[recipient_slot].add(
        BASIC_HEALING_BY_CLASS[PRIEST_CLASS_ID]
    )
    _assert_only_health_changed(state, next_state, expected_health)


_STABLE_TARGET_CASES: Sequence[tuple[int, int, int, int]] = (
    *((_TEAM_A_ACTOR, PRIEST_CLASS_ID, 1 + local, local) for local in range(5)),
    *((_TEAM_B_ACTOR, PRIEST_CLASS_ID, 1 + local, 5 + local) for local in range(5)),
    *((_TEAM_A_ACTOR, HUNTER_CLASS_ID, 6 + local, 5 + local) for local in range(5)),
    *((_TEAM_B_ACTOR, HUNTER_CLASS_ID, 6 + local, local) for local in range(5)),
)


@pytest.mark.parametrize(
    ("actor_slot", "actor_class_id", "target_action", "recipient_slot"),
    [
        pytest.param(
            actor_slot,
            actor_class_id,
            target_action,
            recipient_slot,
            id=f"actor-{actor_slot}-target-{target_action}-slot-{recipient_slot}",
        )
        for actor_slot, actor_class_id, target_action, recipient_slot in (
            _STABLE_TARGET_CASES
        )
    ],
)
def test_relation_local_targets_route_to_stable_global_slots(
    actor_slot: int,
    actor_class_id: int,
    target_action: int,
    recipient_slot: int,
) -> None:
    """Prove all ally/enemy candidate categories preserve fixed roster identity."""
    class_rows = [(slot, HUNTER_CLASS_ID) for slot in range(MAX_AGENT_SLOTS)]
    class_rows[actor_slot] = (actor_slot, actor_class_id)
    config, state = _scenario(*class_rows, team_sizes=(5, 5))
    state = state._replace(current_health=state.current_health - 20.0)

    next_state, _, _ = _step(
        config,
        state,
        _joint_action((actor_slot, MOVE_STAY, target_action, 0)),
    )

    payload = (
        BASIC_HEALING_BY_CLASS[actor_class_id]
        if actor_class_id == PRIEST_CLASS_ID
        else -BASIC_DAMAGE_BY_CLASS[actor_class_id]
    )
    expected_health = state.current_health.at[recipient_slot].add(payload)
    _assert_only_health_changed(state, next_state, expected_health)


@pytest.mark.parametrize(
    "actor_slots",
    [
        pytest.param((0, 1), id="early-team-block-slots"),
        pytest.param((3, 4), id="late-team-block-slots"),
    ],
)
def test_focus_fire_is_independent_of_contributor_slots(
    actor_slots: tuple[int, int],
) -> None:
    """Prove multiple offensive basics reduce once by their aggregate payload."""
    first_actor, second_actor = actor_slots
    config, state = _scenario(
        (first_actor, HUNTER_CLASS_ID),
        (second_actor, ROGUE_CLASS_ID),
        (_TEAM_B_ACTOR, HUNTER_CLASS_ID),
        team_sizes=(5, 1),
    )
    action = _joint_action(
        (first_actor, MOVE_STAY, _FIRST_ENEMY_TARGET, 0),
        (second_actor, MOVE_STAY, _FIRST_ENEMY_TARGET, 0),
    )

    next_state, _, _ = _step(config, state, action)

    expected_damage = (
        BASIC_DAMAGE_BY_CLASS[HUNTER_CLASS_ID] + BASIC_DAMAGE_BY_CLASS[ROGUE_CLASS_ID]
    )
    expected_health = state.current_health.at[_TEAM_B_ACTOR].add(-expected_damage)
    _assert_only_health_changed(state, next_state, expected_health)


def test_duplicate_priest_healing_aggregates_on_one_recipient() -> None:
    """Prove multiple Priest basics add before the single health clamp."""
    config, state = _scenario(
        (0, PRIEST_CLASS_ID),
        (1, PRIEST_CLASS_ID),
        (2, MAGE_CLASS_ID),
        (_TEAM_B_ACTOR, MAGE_CLASS_ID),
        team_sizes=(3, 1),
    )
    state = state._replace(current_health=state.current_health.at[2].add(-30.0))
    action = _joint_action(
        (0, MOVE_STAY, 3, 0),
        (1, MOVE_STAY, 3, 0),
    )

    next_state, _, _ = _step(config, state, action)

    expected_health = state.current_health.at[2].add(
        2 * BASIC_HEALING_BY_CLASS[PRIEST_CLASS_ID]
    )
    _assert_only_health_changed(state, next_state, expected_health)


def test_mixed_damage_and_healing_aggregate_before_lower_clamp() -> None:
    """Distinguish aggregate-then-clamp from ordered per-actor mutation."""
    config, state = _scenario(
        (_TEAM_A_ACTOR, ROGUE_CLASS_ID),
        (_TEAM_B_ACTOR, HUNTER_CLASS_ID),
        (_TEAM_B_ACTOR + 1, PRIEST_CLASS_ID),
        team_sizes=(1, 2),
    )
    state = state._replace(
        current_health=state.current_health.at[_TEAM_B_ACTOR].set(3.0)
    )
    action = _joint_action(
        (_TEAM_A_ACTOR, MOVE_STAY, _FIRST_ENEMY_TARGET, 0),
        (_TEAM_B_ACTOR + 1, MOVE_STAY, _FIRST_ALLY_TARGET, 0),
    )

    next_state, _, _ = _step(config, state, action)

    expected_health = state.current_health.at[_TEAM_B_ACTOR].set(0.0)
    _assert_only_health_changed(state, next_state, expected_health)


def test_healing_clamps_once_at_resolved_max_health() -> None:
    """Prove accepted healing cannot exceed the recipient's resolved bound."""
    config, state = _scenario(
        (_TEAM_A_ACTOR, PRIEST_CLASS_ID),
        (_TEAM_B_ACTOR, MAGE_CLASS_ID),
    )

    next_state, _, _ = _step(
        config,
        state,
        _joint_action((_TEAM_A_ACTOR, MOVE_STAY, _FIRST_ALLY_TARGET, 0)),
    )

    _assert_only_health_changed(state, next_state, state.current_health)


def test_zero_health_but_alive_target_remains_bounded_without_death_semantics() -> None:
    """Prove Checkpoint 1 clamps health without changing liveness or rewards."""
    config, state = _scenario(
        (_TEAM_A_ACTOR, MAGE_CLASS_ID),
        (_TEAM_B_ACTOR, MAGE_CLASS_ID),
    )
    state = state._replace(
        current_health=state.current_health.at[_TEAM_B_ACTOR].set(0.0)
    )
    action = _joint_action(
        (_TEAM_A_ACTOR, MOVE_STAY, _FIRST_ENEMY_TARGET, 0),
    )

    next_state, _, rewards, done_flags, _, info = step(
        config,
        state,
        _current_action_mask(config, state),
        action,
        jax.random.key(19),
    )

    _assert_only_health_changed(state, next_state, state.current_health)
    assert bool(next_state.alive_mask[_TEAM_B_ACTOR])
    assert bool(jnp.all(rewards.rewards == 0.0))
    assert not bool(done_flags.terminated)
    assert not bool(done_flags.truncated)
    assert len(info) == 0


def test_legal_movement_survives_an_invalid_combat_pair() -> None:
    """Prove combat rejection cannot discard independently legal movement."""
    config, state = _scenario(
        (_TEAM_A_ACTOR, MAGE_CLASS_ID),
        (_TEAM_A_ALLY, MAGE_CLASS_ID),
        (_TEAM_B_ACTOR, MAGE_CLASS_ID),
        team_sizes=(2, 1),
    )
    action = _joint_action(
        (_TEAM_A_ACTOR, MOVE_EAST, _FIRST_ALLY_TARGET + 1, 0),
    )

    next_state, _, _ = _step(config, state, action)

    assert (
        next_state.agent_positions[_TEAM_A_ACTOR, 0]
        > state.agent_positions[_TEAM_A_ACTOR, 0]
    )
    assert bool(jnp.array_equal(next_state.current_health, state.current_health))


def test_invalid_ultimate_pair_cannot_fall_back_to_a_valid_basic() -> None:
    """Prove target and ultimate heads are rejected as one authoritative pair."""
    config, state = _scenario(
        (_TEAM_A_ACTOR, MAGE_CLASS_ID),
        (_TEAM_B_ACTOR, MAGE_CLASS_ID),
    )

    next_state, _, _ = _step(
        config,
        state,
        _joint_action((_TEAM_A_ACTOR, MOVE_STAY, _FIRST_ENEMY_TARGET, 1)),
    )

    _assert_only_health_changed(state, next_state, state.current_health)


def test_accepted_ultimate_lane_does_not_also_apply_a_basic() -> None:
    """Prove an ultimate applies its own payload without also applying a basic."""
    config, state = _scenario(
        (_TEAM_A_ACTOR, WARRIOR_CLASS_ID),
        (_TEAM_B_ACTOR, MAGE_CLASS_ID),
    )
    current_mask = _current_action_mask(config, state)
    assert bool(
        current_mask.select_target_use_ultimate_joint_mask[
            _TEAM_A_ACTOR, _FIRST_ENEMY_TARGET, 1
        ]
    )

    next_state, _, _ = _step(
        config,
        state,
        _joint_action((_TEAM_A_ACTOR, MOVE_STAY, _FIRST_ENEMY_TARGET, 1)),
    )

    expected_health = state.current_health.at[_TEAM_B_ACTOR].add(
        -ULTIMATE_DAMAGE_BY_CLASS[WARRIOR_CLASS_ID]
    )
    assert bool(jnp.array_equal(next_state.current_health, expected_health))
    assert (
        next_state.ultimate_cooldowns[_TEAM_A_ACTOR]
        == ULTIMATE_COOLDOWN_BY_CLASS[WARRIOR_CLASS_ID]
    )


def test_target_none_no_ultimate_is_effect_inert() -> None:
    """Prove the canonical combat no-op creates no contribution."""
    config, state = _scenario(
        (_TEAM_A_ACTOR, MAGE_CLASS_ID),
        (_TEAM_B_ACTOR, MAGE_CLASS_ID),
    )

    next_state, _, _ = _step(config, state, _joint_action())

    _assert_only_health_changed(state, next_state, state.current_health)


@pytest.mark.parametrize(
    "actor_state",
    [
        pytest.param("dead", id="dead-actor"),
        pytest.param("inactive", id="inactive-actor"),
    ],
)
def test_nonacting_actor_submissions_remain_physically_inert(actor_state: str) -> None:
    """Prove canonical mask fallback never grants dead or padded slots agency."""
    if actor_state == "dead":
        actor_slot = _TEAM_A_ACTOR
        config, state = _scenario(
            (actor_slot, MAGE_CLASS_ID),
            (_TEAM_B_ACTOR, MAGE_CLASS_ID),
        )
        state = state._replace(alive_mask=state.alive_mask.at[actor_slot].set(False))
    else:
        actor_slot = _TEAM_A_ALLY
        config, state = _scenario(
            (_TEAM_A_ACTOR, MAGE_CLASS_ID),
            (actor_slot, MAGE_CLASS_ID),
            (_TEAM_B_ACTOR, MAGE_CLASS_ID),
        )
    action = _joint_action(
        (actor_slot, MOVE_EAST, _FIRST_ENEMY_TARGET, 0),
    )

    next_state, _, _ = _step(config, state, action)

    assert bool(
        jnp.array_equal(
            next_state.agent_positions[actor_slot], state.agent_positions[actor_slot]
        )
    )
    assert bool(jnp.array_equal(next_state.current_health, state.current_health))


@pytest.mark.parametrize(
    "invalidity",
    [
        pytest.param("wrong-relation", id="wrong-relation"),
        pytest.param("out-of-range", id="out-of-range"),
        pytest.param("blocked-los", id="blocked-los"),
        pytest.param("stunned-actor", id="stunned-actor"),
        pytest.param("dead-candidate", id="dead-candidate"),
        pytest.param("inactive-candidate", id="inactive-candidate"),
    ],
)
def test_invalid_basic_attempts_create_no_health_effect(invalidity: str) -> None:
    """Prove every established legality gate reaches the same accepted no-op."""
    team_sizes = (2, 1) if invalidity == "wrong-relation" else (1, 1)
    obstacles = _blocking_pillar(x=4.5, y=2.0) if invalidity == "blocked-los" else None
    basic_radius = 2.0 if invalidity == "out-of-range" else 10.0
    config, state = _scenario(
        (_TEAM_A_ACTOR, MAGE_CLASS_ID),
        (_TEAM_A_ALLY, MAGE_CLASS_ID),
        (_TEAM_B_ACTOR, MAGE_CLASS_ID),
        team_sizes=team_sizes,
        obstacles=obstacles,
        basic_radius=basic_radius,
    )

    target_action = _FIRST_ENEMY_TARGET
    if invalidity == "wrong-relation":
        target_action = _FIRST_ALLY_TARGET + 1
    elif invalidity == "stunned-actor":
        state = state._replace(
            stun_durations=state.stun_durations.at[
                _TEAM_A_ACTOR, STUN_CHANNEL_WARRIOR_CHARGE
            ].set(1)
        )
    elif invalidity == "dead-candidate":
        state = state._replace(alive_mask=state.alive_mask.at[_TEAM_B_ACTOR].set(False))
    elif invalidity == "inactive-candidate":
        target_action = _FIRST_ENEMY_TARGET + 1

    next_state, _, _ = _step(
        config,
        state,
        _joint_action((_TEAM_A_ACTOR, MOVE_STAY, target_action, 0)),
    )

    _assert_only_health_changed(state, next_state, state.current_health)


@pytest.mark.parametrize(
    ("starts_in_range", "move_action", "expects_damage", "next_target_is_legal"),
    [
        pytest.param(False, MOVE_EAST, False, True, id="entering-range-is-too-late"),
        pytest.param(True, MOVE_WEST, True, False, id="leaving-range-does-not-cancel"),
    ],
)
def test_basic_legality_uses_pre_movement_state(
    starts_in_range: bool,
    move_action: int,
    expects_damage: bool,
    next_target_is_legal: bool,
) -> None:
    """Prove same-tick movement cannot change pre-state combat acceptance."""
    positions = _default_positions((1, 1))
    actor_x = 2.0 if starts_in_range else 1.0
    positions = positions.at[_TEAM_A_ACTOR].set(
        jnp.asarray((actor_x, 2.0), dtype=jnp.float32)
    )
    positions = positions.at[_TEAM_B_ACTOR].set(
        jnp.asarray((7.0, 2.0), dtype=jnp.float32)
    )
    config, state = _scenario(
        (_TEAM_A_ACTOR, HUNTER_CLASS_ID),
        (_TEAM_B_ACTOR, HUNTER_CLASS_ID),
        positions=positions,
        basic_radius=5.0,
    )
    action = _joint_action(
        (_TEAM_A_ACTOR, move_action, _FIRST_ENEMY_TARGET, 0),
    )

    next_state, _, next_mask = _step(config, state, action)

    expected_health = state.current_health
    if expects_damage:
        expected_health = expected_health.at[_TEAM_B_ACTOR].add(
            -BASIC_DAMAGE_BY_CLASS[HUNTER_CLASS_ID]
        )
    assert bool(jnp.array_equal(next_state.current_health, expected_health))
    assert (
        bool(
            next_mask.select_target_use_ultimate_joint_mask[
                _TEAM_A_ACTOR, _FIRST_ENEMY_TARGET, 0
            ]
        )
        is next_target_is_legal
    )


def test_post_state_observation_projects_updated_health() -> None:
    """Prove returned observations are built from the packaged next state."""
    config, state = _scenario(
        (_TEAM_A_ACTOR, MAGE_CLASS_ID),
        (_TEAM_B_ACTOR, MAGE_CLASS_ID),
    )

    next_state, observation, _ = _step(
        config,
        state,
        _joint_action((_TEAM_A_ACTOR, MOVE_STAY, _FIRST_ENEMY_TARGET, 0)),
    )

    assert (
        observation.self_features[_TEAM_B_ACTOR, AGENT_FEATURE_CURRENT_HEALTH]
        == (next_state.current_health[_TEAM_B_ACTOR])
    )


def test_jitted_step_matches_eager_accepted_effects() -> None:
    """Prove accepted routing, aggregation, and clamping are JIT-stable."""
    config, state = _scenario(
        (_TEAM_A_ACTOR, MAGE_CLASS_ID),
        (_TEAM_B_ACTOR, MAGE_CLASS_ID),
        (_TEAM_B_ACTOR + 1, PRIEST_CLASS_ID),
        team_sizes=(1, 2),
    )
    state = state._replace(
        current_health=state.current_health.at[_TEAM_B_ACTOR].set(20.0)
    )
    action = _joint_action(
        (_TEAM_A_ACTOR, MOVE_STAY, _FIRST_ENEMY_TARGET, 0),
        (_TEAM_B_ACTOR + 1, MOVE_STAY, _FIRST_ALLY_TARGET, 0),
    )
    current_mask = _current_action_mask(config, state)

    eager_outputs = step(config, state, current_mask, action, jax.random.key(23))
    compiled_outputs = cast(
        tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info],
        jax.jit(step)(config, state, current_mask, action, jax.random.key(23)),
    )

    for eager_leaf, compiled_leaf in zip(
        jax.tree_util.tree_leaves(eager_outputs),
        jax.tree_util.tree_leaves(compiled_outputs),
        strict=True,
    ):
        assert bool(jnp.array_equal(eager_leaf, compiled_leaf))


def test_scanned_rollout_reuses_each_post_state_mask_for_repeated_basics() -> None:
    """Prove accepted basic effects remain stable in a compiled temporal carry."""
    horizon = 3
    config, state = _scenario(
        (_TEAM_A_ACTOR, MAGE_CLASS_ID),
        (_TEAM_B_ACTOR, MAGE_CLASS_ID),
    )
    action = _joint_action(
        (_TEAM_A_ACTOR, MOVE_STAY, _FIRST_ENEMY_TARGET, 0),
    )
    keys = jax.random.split(jax.random.key(29), horizon)

    def _scan_step(
        carry: tuple[EnvState, ActionMask],
        step_key: Array,
    ) -> tuple[tuple[EnvState, ActionMask], Array]:
        current_state, current_mask = carry
        next_state, _, _, _, next_mask, _ = step(
            config,
            current_state,
            current_mask,
            action,
            step_key,
        )
        return (next_state, next_mask), next_state.current_health

    def _rollout(
        initial_state: EnvState,
        initial_mask: ActionMask,
        scan_keys: Array,
    ) -> tuple[tuple[EnvState, ActionMask], Array]:
        """Run one fixed-horizon rollout under ``jax.lax.scan``."""
        return jax.lax.scan(_scan_step, (initial_state, initial_mask), scan_keys)

    (final_state, _), health_history = cast(
        tuple[tuple[EnvState, ActionMask], Array],
        jax.jit(_rollout)(state, _current_action_mask(config, state), keys),
    )

    damage = BASIC_DAMAGE_BY_CLASS[MAGE_CLASS_ID] * MAGE_DAMAGE_AURA_MULTIPLIER
    expected_history = state.current_health[_TEAM_B_ACTOR] - damage * jnp.arange(
        1, horizon + 1, dtype=jnp.float32
    )
    assert bool(
        jnp.allclose(
            health_history[:, _TEAM_B_ACTOR],
            expected_history,
            rtol=0.0,
            atol=1e-5,
        )
    )
    assert bool(
        jnp.isclose(
            final_state.current_health[_TEAM_B_ACTOR],
            expected_history[-1],
            rtol=0.0,
            atol=1e-5,
        )
    )
