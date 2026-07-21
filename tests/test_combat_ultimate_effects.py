"""Semantic proofs for Milestone 5 Step 6 ultimate effects and lifecycles."""
# pyright: reportPrivateUsage=false

from typing import cast

import jax
import jax.numpy as jnp
import pytest
from jax import Array

import marl_battlegrounds.core.combat as combat
from marl_battlegrounds.core.config import resolve_agent_profile
from marl_battlegrounds.core.env import (
    _build_observation_and_action_mask,
    _compute_global_pairwise_distances_from_agent_positions,
    _derive_aura_damage_multipliers,
    step,
)
from marl_battlegrounds.core.types import (
    AGENT_FEATURE_ANTI_HEAL_ROGUE_POISON_DURATION,
    AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER,
    AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_BURST_DURATION,
    AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER,
    AGENT_FEATURE_SLOW_ROGUE_POISON_DURATION,
    AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION,
    ENVIRONMENT_DIMENSIONS,
    HUNTER_CLASS_ID,
    MAGE_CLASS_ID,
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    MAX_OBSTACLE_SLOTS,
    MOVE_STAY,
    NEUTRAL_CLASS_ID,
    NUM_CLASSES,
    NUM_SLOW_CHANNELS,
    NUM_STUN_CHANNELS,
    OBSTACLE_FEATURES,
    PRIEST_CLASS_ID,
    ROGUE_CLASS_ID,
    SLOW_CHANNEL_HUNTER_BASIC,
    SLOW_CHANNEL_ROGUE_POISON,
    SLOW_CHANNEL_WARRIOR_CHARGE,
    STUN_CHANNEL_HUNTER_TRAP,
    STUN_CHANNEL_ROGUE_POISON,
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

_TEAM_A_FIRST_SLOT = 0
_TEAM_A_SECOND_SLOT = 1
_TEAM_B_FIRST_SLOT = MAX_AGENTS_PER_TEAM
_TEAM_B_SECOND_SLOT = MAX_AGENTS_PER_TEAM + 1
_SELF_TARGET = 1
_SECOND_ALLY_TARGET = 2
_FIRST_ENEMY_TARGET = 1 + MAX_AGENTS_PER_TEAM
_SECOND_ENEMY_TARGET = _FIRST_ENEMY_TARGET + 1


def _empty_obstacles() -> Array:
    """Return an inactive fixed-size obstacle table."""
    return jnp.zeros((MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES), dtype=jnp.float32)


def _requested_roster(*class_rows: tuple[int, int]) -> Array:
    """Return a padded roster with explicit class assignments by global slot."""
    roster = jnp.full((MAX_AGENT_SLOTS,), NEUTRAL_CLASS_ID, dtype=jnp.int32)
    for slot, class_id in class_rows:
        roster = roster.at[slot].set(class_id)
    return roster


def _default_positions(team_sizes: tuple[int, int]) -> Array:
    """Place active team blocks on separated non-overlapping vertical lines."""
    positions = jnp.zeros((MAX_AGENT_SLOTS, ENVIRONMENT_DIMENSIONS), dtype=jnp.float32)
    for local_slot in range(team_sizes[0]):
        positions = positions.at[local_slot].set(
            jnp.asarray((2.0, 2.0 + 1.25 * local_slot), dtype=jnp.float32)
        )
    for local_slot in range(team_sizes[1]):
        positions = positions.at[MAX_AGENTS_PER_TEAM + local_slot].set(
            jnp.asarray((8.0, 2.0 + 1.25 * local_slot), dtype=jnp.float32)
        )
    return positions


def _scenario(
    *class_rows: tuple[int, int],
    team_sizes: tuple[int, int],
    positions: Array | None = None,
) -> tuple[EnvConfig, EnvState]:
    """Build a deterministic combat scenario with permissive interaction radii."""
    profile = resolve_agent_profile(
        _requested_roster(*class_rows),
        jnp.asarray(team_sizes, dtype=jnp.int32),
    )
    profile = profile._replace(
        base_movement_speeds=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.float32),
        observation_radii=jnp.where(profile.active_mask, 20.0, 0.0).astype(jnp.float32),
        basic_interaction_radii=jnp.where(profile.active_mask, 20.0, 0.0).astype(
            jnp.float32
        ),
        ultimate_interaction_radii=jnp.where(profile.active_mask, 20.0, 0.0).astype(
            jnp.float32
        ),
    )
    config = EnvConfig(
        max_steps=100,
        map_width=20.0,
        map_height=12.0,
        obstacles=_empty_obstacles(),
        agent_profile=profile,
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


def _joint_action(*rows: tuple[int, int, int]) -> Action:
    """Return a no-movement joint action with selected combat overrides.

    Each override is ``(actor_slot, target_action, use_ultimate)``.
    """
    targets = jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32)
    ultimate_uses = jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32)
    for actor_slot, target_action, use_ultimate in rows:
        targets = targets.at[actor_slot].set(target_action)
        ultimate_uses = ultimate_uses.at[actor_slot].set(use_ultimate)
    return Action(
        move=jnp.full((MAX_AGENT_SLOTS,), MOVE_STAY, dtype=jnp.int32),
        select_target=targets,
        use_ultimate=ultimate_uses,
    )


def _current_action_mask(config: EnvConfig, state: EnvState) -> ActionMask:
    """Return the authoritative action mask paired with a test state."""
    return _build_observation_and_action_mask(state, config)[1]


def _step(
    config: EnvConfig,
    state: EnvState,
    action: Action,
) -> tuple[EnvState, Observation, ActionMask]:
    """Advance one transition and return state-facing public outputs."""
    next_state, observation, _, _, next_mask, _ = step(
        config,
        state,
        _current_action_mask(config, state),
        action,
        jax.random.key(23),
    )
    return next_state, observation, next_mask


def _aura_multipliers(config: EnvConfig, state: EnvState) -> tuple[Array, Array]:
    """Derive current Mage and Warrior aura vectors for a scenario."""
    distances = _compute_global_pairwise_distances_from_agent_positions(
        state.agent_positions
    )
    return _derive_aura_damage_multipliers(config, distances, state.alive_mask)


def test_ultimate_health_catalogs_are_exact_class_aligned_and_jit_stable() -> None:
    """Prove both ultimate health catalogs are complete authoritative tables."""
    expected_damage = jnp.asarray((0.0, 0.0, 16.0, 0.0, 0.0, 0.0), dtype=jnp.float32)
    expected_healing = jnp.asarray((0.0, 0.0, 0.0, 0.0, 0.0, 75.0), dtype=jnp.float32)
    class_ids = jnp.arange(NUM_CLASSES, dtype=jnp.int32)

    assert combat.ULTIMATE_DAMAGE_BY_CLASS.shape == (NUM_CLASSES,)
    assert combat.ULTIMATE_HEALING_BY_CLASS.shape == (NUM_CLASSES,)
    assert combat.ULTIMATE_DAMAGE_BY_CLASS.dtype == jnp.float32
    assert combat.ULTIMATE_HEALING_BY_CLASS.dtype == jnp.float32
    assert bool(jnp.array_equal(combat.ULTIMATE_DAMAGE_BY_CLASS, expected_damage))
    assert bool(jnp.array_equal(combat.ULTIMATE_HEALING_BY_CLASS, expected_healing))
    assert bool(
        jnp.array_equal(
            combat.get_ultimate_damage_by_class_ids(class_ids), expected_damage
        )
    )
    assert bool(
        jnp.array_equal(
            combat.get_ultimate_healing_by_class_ids(class_ids), expected_healing
        )
    )
    assert bool(
        jnp.array_equal(
            cast(
                Array,
                jax.jit(combat.get_ultimate_damage_by_class_ids)(class_ids),
            ),
            expected_damage,
        )
    )
    assert bool(
        jnp.array_equal(
            cast(
                Array,
                jax.jit(combat.get_ultimate_healing_by_class_ids)(class_ids),
            ),
            expected_healing,
        )
    )


def test_three_duplicate_auras_are_bounded_and_observation_consistent() -> None:
    """Prove duplicate aura reductions clip once and feed attached features."""
    positions = _default_positions((4, 4))
    for slot, position in enumerate(((2.0, 2.0), (2.0, 2.5), (2.0, 3.0), (2.0, 3.5))):
        positions = positions.at[slot].set(jnp.asarray(position, dtype=jnp.float32))
    for offset, position in enumerate(((8.0, 2.0), (8.0, 2.5), (8.0, 3.0), (8.0, 3.5))):
        positions = positions.at[MAX_AGENTS_PER_TEAM + offset].set(
            jnp.asarray(position, dtype=jnp.float32)
        )
    config, state = _scenario(
        (0, MAGE_CLASS_ID),
        (1, MAGE_CLASS_ID),
        (2, MAGE_CLASS_ID),
        (3, HUNTER_CLASS_ID),
        (5, WARRIOR_CLASS_ID),
        (6, WARRIOR_CLASS_ID),
        (7, WARRIOR_CLASS_ID),
        (8, HUNTER_CLASS_ID),
        team_sizes=(4, 4),
        positions=positions,
    )

    eager_mage, eager_warrior = _aura_multipliers(config, state)
    distances = _compute_global_pairwise_distances_from_agent_positions(
        state.agent_positions
    )
    compiled_mage, compiled_warrior = cast(
        tuple[Array, Array],
        jax.jit(_derive_aura_damage_multipliers)(config, distances, state.alive_mask),
    )
    observation, _ = _build_observation_and_action_mask(state, config)

    assert bool(
        jnp.allclose(eager_mage[:4], combat.MAGE_DAMAGE_AURA_MULTIPLIER_CEILING)
    )
    assert bool(
        jnp.allclose(
            eager_warrior[5:9],
            combat.WARRIOR_DAMAGE_MITIGATION_AURA_MULTIPLIER_FLOOR,
        )
    )
    assert bool(jnp.array_equal(eager_mage, compiled_mage))
    assert bool(jnp.array_equal(eager_warrior, compiled_warrior))
    assert (
        observation.self_features[
            3, AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER
        ]
        == combat.MAGE_DAMAGE_AURA_MULTIPLIER_CEILING
    )
    assert (
        observation.self_features[
            8, AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER
        ]
        == combat.WARRIOR_DAMAGE_MITIGATION_AURA_MULTIPLIER_FLOOR
    )


def test_warrior_ultimates_route_symmetrically_for_both_teams() -> None:
    """Prove relation-local enemy selections resolve to stable global slots."""
    positions = _default_positions((2, 2))
    positions = positions.at[_TEAM_A_SECOND_SLOT].set(
        jnp.asarray((2.0, 5.0), dtype=jnp.float32)
    )
    positions = positions.at[_TEAM_B_SECOND_SLOT].set(
        jnp.asarray((8.0, 5.0), dtype=jnp.float32)
    )
    config, state = _scenario(
        (0, WARRIOR_CLASS_ID),
        (1, HUNTER_CLASS_ID),
        (5, WARRIOR_CLASS_ID),
        (6, HUNTER_CLASS_ID),
        team_sizes=(2, 2),
        positions=positions,
    )
    action = _joint_action(
        (_TEAM_A_FIRST_SLOT, _SECOND_ENEMY_TARGET, 1),
        (_TEAM_B_FIRST_SLOT, _SECOND_ENEMY_TARGET, 1),
    )

    next_state, _, _ = _step(config, state, action)

    assert (
        next_state.current_health[_TEAM_A_SECOND_SLOT]
        == state.current_health[_TEAM_A_SECOND_SLOT]
        - combat.ULTIMATE_DAMAGE_BY_CLASS[WARRIOR_CLASS_ID]
    )
    assert (
        next_state.current_health[_TEAM_B_SECOND_SLOT]
        == state.current_health[_TEAM_B_SECOND_SLOT]
        - combat.ULTIMATE_DAMAGE_BY_CLASS[WARRIOR_CLASS_ID]
    )


def test_priest_ultimates_route_to_self_and_allies_for_both_teams() -> None:
    """Prove Holy Word shares the stable relation-local recipient mapping."""
    config, state = _scenario(
        (0, PRIEST_CLASS_ID),
        (1, HUNTER_CLASS_ID),
        (5, PRIEST_CLASS_ID),
        (6, HUNTER_CLASS_ID),
        team_sizes=(2, 2),
    )
    current_health = state.current_health
    current_health = current_health.at[_TEAM_A_FIRST_SLOT].set(10.0)
    current_health = current_health.at[_TEAM_B_SECOND_SLOT].set(10.0)
    state = state._replace(current_health=current_health)
    action = _joint_action(
        (_TEAM_A_FIRST_SLOT, _SELF_TARGET, 1),
        (_TEAM_B_FIRST_SLOT, _SECOND_ALLY_TARGET, 1),
    )

    next_state, _, _ = _step(config, state, action)

    assert next_state.current_health[_TEAM_A_FIRST_SLOT] == 85.0
    assert next_state.current_health[_TEAM_B_SECOND_SLOT] == 85.0


@pytest.mark.parametrize(
    ("actor_class_id", "target_action"),
    (
        pytest.param(MAGE_CLASS_ID, 0, id="mage-burst"),
        pytest.param(HUNTER_CLASS_ID, _FIRST_ENEMY_TARGET, id="hunter-trap"),
        pytest.param(ROGUE_CLASS_ID, _FIRST_ENEMY_TARGET, id="rogue-poison"),
    ),
)
def test_zero_health_payload_ultimates_change_no_health_but_start_cooldown(
    actor_class_id: int,
    target_action: int,
) -> None:
    """Prove zero-payload ultimates remain accepted cooldown-bearing actions."""
    config, state = _scenario(
        (0, actor_class_id),
        (5, HUNTER_CLASS_ID),
        team_sizes=(1, 1),
    )

    next_state, _, next_mask = _step(
        config, state, _joint_action((0, target_action, 1))
    )

    assert bool(jnp.array_equal(next_state.current_health, state.current_health))
    assert (
        next_state.ultimate_cooldowns[0]
        == combat.ULTIMATE_COOLDOWN_BY_CLASS[actor_class_id]
    )
    assert not bool(next_mask.use_ultimate_mask[0, 1])


def test_fresh_mage_burst_first_amplifies_next_transition_damage() -> None:
    """Prove Burst is observed before its first damage-modifying decision."""
    config, state = _scenario(
        (0, MAGE_CLASS_ID),
        (5, HUNTER_CLASS_ID),
        team_sizes=(1, 1),
    )
    config = config._replace(
        agent_profile=config.agent_profile._replace(
            observation_radii=config.agent_profile.observation_radii.at[5].set(0.5)
        )
    )

    applied_state, applied_observation, _ = _step(
        config,
        state,
        _joint_action((0, 0, 1)),
    )

    assert bool(jnp.array_equal(applied_state.current_health, state.current_health))
    assert applied_state.mage_burst_damage_amplification_durations[0] == (
        combat.MAGE_BURST_DAMAGE_DURATION_TICKS
    )
    assert (
        applied_observation.self_features[
            0, AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_BURST_DURATION
        ]
        == combat.MAGE_BURST_DAMAGE_DURATION_TICKS
    )
    assert not bool(applied_observation.enemy_visibility_mask[5, 0])
    assert (
        applied_observation.enemy_unit_features[
            5, 0, AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_BURST_DURATION
        ]
        == 0.0
    )

    governed_state, governed_observation, _ = _step(
        config,
        applied_state,
        _joint_action((0, _FIRST_ENEMY_TARGET, 0)),
    )

    expected_damage = (
        combat.BASIC_DAMAGE_BY_CLASS[MAGE_CLASS_ID]
        * combat.MAGE_BURST_DAMAGE_MULTIPLIER
        * combat.MAGE_DAMAGE_AURA_MULTIPLIER
    )
    assert governed_state.current_health[5] == (
        applied_state.current_health[5] - expected_damage
    )
    assert governed_state.mage_burst_damage_amplification_durations[0] == (
        combat.MAGE_BURST_DAMAGE_DURATION_TICKS - 1
    )
    assert (
        governed_observation.self_features[
            0, AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_BURST_DURATION
        ]
        == combat.MAGE_BURST_DAMAGE_DURATION_TICKS - 1
    )


def test_charge_damage_uses_burst_mage_aura_and_warrior_mitigation() -> None:
    """Prove outgoing and incoming modifiers compose around ultimate damage."""
    positions = _default_positions((2, 2))
    positions = positions.at[0].set(jnp.asarray((2.0, 2.0), dtype=jnp.float32))
    positions = positions.at[1].set(jnp.asarray((2.0, 3.0), dtype=jnp.float32))
    positions = positions.at[5].set(jnp.asarray((8.0, 2.0), dtype=jnp.float32))
    positions = positions.at[6].set(jnp.asarray((8.0, 3.0), dtype=jnp.float32))
    config, state = _scenario(
        (0, WARRIOR_CLASS_ID),
        (1, MAGE_CLASS_ID),
        (5, HUNTER_CLASS_ID),
        (6, WARRIOR_CLASS_ID),
        team_sizes=(2, 2),
        positions=positions,
    )
    state = state._replace(
        mage_burst_damage_amplification_durations=(
            state.mage_burst_damage_amplification_durations.at[0].set(2)
        )
    )

    next_state, _, _ = _step(config, state, _joint_action((0, _FIRST_ENEMY_TARGET, 1)))

    expected_damage = (
        combat.ULTIMATE_DAMAGE_BY_CLASS[WARRIOR_CLASS_ID]
        * combat.MAGE_BURST_DAMAGE_MULTIPLIER
        * combat.MAGE_DAMAGE_AURA_MULTIPLIER
        * combat.WARRIOR_DAMAGE_MITIGATION_AURA_MULTIPLIER
    )
    assert bool(
        jnp.isclose(
            state.current_health[_TEAM_B_FIRST_SLOT]
            - next_state.current_health[_TEAM_B_FIRST_SLOT],
            expected_damage,
        )
    )


def test_holy_word_uses_pre_state_anti_heal_and_still_starts_cooldown() -> None:
    """Prove incoming anti-heal modifies ultimate healing before one clamp."""
    config, state = _scenario(
        (0, PRIEST_CLASS_ID),
        (1, HUNTER_CLASS_ID),
        (5, ROGUE_CLASS_ID),
        team_sizes=(2, 1),
    )
    state = state._replace(
        current_health=state.current_health.at[1].set(10.0),
        rogue_poison_anti_heal_durations=(
            state.rogue_poison_anti_heal_durations.at[1].set(2)
        ),
    )

    next_state, _, _ = _step(config, state, _joint_action((0, _SECOND_ALLY_TARGET, 1)))

    expected_health = 10.0 + (
        combat.ULTIMATE_HEALING_BY_CLASS[PRIEST_CLASS_ID]
        * combat.ROGUE_POISON_ANTI_HEAL_MULTIPLIER
    )
    assert next_state.current_health[1] == expected_health
    assert (
        next_state.ultimate_cooldowns[0]
        == combat.ULTIMATE_COOLDOWN_BY_CLASS[PRIEST_CLASS_ID]
    )


def test_fresh_rogue_anti_heal_first_modifies_next_transition_healing() -> None:
    """Prove Poison is public before it reduces a later incoming heal."""
    config, state = _scenario(
        (0, ROGUE_CLASS_ID),
        (5, PRIEST_CLASS_ID),
        (6, HUNTER_CLASS_ID),
        team_sizes=(1, 2),
    )
    state = state._replace(current_health=state.current_health.at[6].set(50.0))
    application_action = _joint_action(
        (0, _SECOND_ENEMY_TARGET, 1),
        (5, _SECOND_ALLY_TARGET, 0),
    )

    applied_state, applied_observation, _ = _step(
        config,
        state,
        application_action,
    )

    assert applied_state.current_health[6] == (
        50.0 + combat.BASIC_HEALING_BY_CLASS[PRIEST_CLASS_ID]
    )
    assert applied_state.rogue_poison_anti_heal_durations[6] == (
        combat.ROGUE_POISON_ANTI_HEAL_DURATION_TICKS
    )
    assert (
        applied_observation.self_features[
            6, AGENT_FEATURE_ANTI_HEAL_ROGUE_POISON_DURATION
        ]
        == combat.ROGUE_POISON_ANTI_HEAL_DURATION_TICKS
    )

    governed_state, _, _ = _step(
        config,
        applied_state,
        _joint_action((5, _SECOND_ALLY_TARGET, 0)),
    )

    assert governed_state.current_health[6] == (
        applied_state.current_health[6]
        + combat.BASIC_HEALING_BY_CLASS[PRIEST_CLASS_ID]
        * combat.ROGUE_POISON_ANTI_HEAL_MULTIPLIER
    )
    assert governed_state.rogue_poison_anti_heal_durations[6] == (
        combat.ROGUE_POISON_ANTI_HEAL_DURATION_TICKS - 1
    )


def test_full_health_holy_word_is_clamped_but_still_starts_cooldown() -> None:
    """Prove cooldown cost follows acceptance rather than effective healing."""
    config, state = _scenario(
        (0, PRIEST_CLASS_ID),
        (1, HUNTER_CLASS_ID),
        (5, ROGUE_CLASS_ID),
        team_sizes=(2, 1),
    )

    next_state, _, _ = _step(config, state, _joint_action((0, _SECOND_ALLY_TARGET, 1)))

    assert bool(jnp.array_equal(next_state.current_health, state.current_health))
    assert (
        next_state.ultimate_cooldowns[0]
        == combat.ULTIMATE_COOLDOWN_BY_CLASS[PRIEST_CLASS_ID]
    )


def test_mixed_ultimate_damage_and_healing_net_before_single_clamp() -> None:
    """Prove simultaneous opposing payloads resolve from one pre-state snapshot."""
    config, state = _scenario(
        (0, PRIEST_CLASS_ID),
        (1, HUNTER_CLASS_ID),
        (5, WARRIOR_CLASS_ID),
        team_sizes=(2, 1),
    )
    state = state._replace(current_health=state.current_health.at[1].set(10.0))
    action = _joint_action(
        (0, _SECOND_ALLY_TARGET, 1),
        (5, _SECOND_ENEMY_TARGET, 1),
    )

    next_state, _, _ = _step(config, state, action)

    expected_health = (
        10.0
        + combat.ULTIMATE_HEALING_BY_CLASS[PRIEST_CLASS_ID]
        - combat.ULTIMATE_DAMAGE_BY_CLASS[WARRIOR_CLASS_ID]
    )
    assert next_state.current_health[1] == expected_health


@pytest.mark.parametrize("starting_cooldown", (1, 7))
def test_existing_cooldown_ticks_once_without_accepted_replacement(
    starting_cooldown: int,
) -> None:
    """Prove cooldown counters decrement once and remain nonnegative."""
    config, state = _scenario(
        (0, WARRIOR_CLASS_ID),
        (5, HUNTER_CLASS_ID),
        team_sizes=(1, 1),
    )
    state = state._replace(
        ultimate_cooldowns=state.ultimate_cooldowns.at[0].set(starting_cooldown)
    )

    next_state, _, _ = _step(config, state, _joint_action())

    assert next_state.ultimate_cooldowns[0] == starting_cooldown - 1


@pytest.mark.parametrize(
    ("actor_class_id", "target_action"),
    (
        pytest.param(MAGE_CLASS_ID, 0, id="mage"),
        pytest.param(WARRIOR_CLASS_ID, _FIRST_ENEMY_TARGET, id="warrior"),
        pytest.param(HUNTER_CLASS_ID, _FIRST_ENEMY_TARGET, id="hunter"),
        pytest.param(ROGUE_CLASS_ID, _FIRST_ENEMY_TARGET, id="rogue"),
        pytest.param(PRIEST_CLASS_ID, _SELF_TARGET, id="priest"),
    ),
)
def test_every_class_starts_full_cooldown_without_same_tick_decrement(
    actor_class_id: int,
    target_action: int,
) -> None:
    """Prove accepted ultimate use starts the exact class catalog duration."""
    config, state = _scenario(
        (0, actor_class_id),
        (5, HUNTER_CLASS_ID),
        team_sizes=(1, 1),
    )

    next_state, _, next_mask = _step(
        config, state, _joint_action((0, target_action, 1))
    )

    assert (
        next_state.ultimate_cooldowns[0]
        == combat.ULTIMATE_COOLDOWN_BY_CLASS[actor_class_id]
    )
    assert not bool(next_mask.use_ultimate_mask[0, 1])


def test_jitted_step_matches_eager_ultimate_health_and_cooldown_outputs() -> None:
    """Prove compiled execution preserves meaningful Checkpoint 1 outputs."""
    config, state = _scenario(
        (0, WARRIOR_CLASS_ID),
        (5, HUNTER_CLASS_ID),
        team_sizes=(1, 1),
    )
    action = _joint_action((0, _FIRST_ENEMY_TARGET, 1))
    current_mask = _current_action_mask(config, state)
    eager = step(config, state, current_mask, action, jax.random.key(31))
    compiled = cast(
        tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info],
        jax.jit(step)(config, state, current_mask, action, jax.random.key(31)),
    )

    assert jax.tree_util.tree_structure(eager) == jax.tree_util.tree_structure(compiled)
    for eager_leaf, compiled_leaf in zip(
        jax.tree_util.tree_leaves(eager),
        jax.tree_util.tree_leaves(compiled),
        strict=True,
    ):
        assert bool(jnp.array_equal(eager_leaf, compiled_leaf))


def test_scanned_repeated_submission_applies_once_then_ticks_cooldown() -> None:
    """Prove each produced mask gates the next scanned transition."""
    config, initial_state = _scenario(
        (0, WARRIOR_CLASS_ID),
        (5, HUNTER_CLASS_ID),
        team_sizes=(1, 1),
    )
    initial_mask = _current_action_mask(config, initial_state)
    action = _joint_action((0, _FIRST_ENEMY_TARGET, 1))

    def scan_body(
        carry: tuple[EnvState, ActionMask], _: None
    ) -> tuple[tuple[EnvState, ActionMask], tuple[Array, Array]]:
        current_state, current_mask = carry
        (
            next_state,
            _observation,
            _reward,
            _done_flags,
            next_mask,
            _info,
        ) = step(
            config,
            current_state,
            current_mask,
            action,
            jax.random.key(37),
        )
        outputs = (
            next_state.current_health[_TEAM_B_FIRST_SLOT],
            next_state.ultimate_cooldowns[_TEAM_A_FIRST_SLOT],
        )
        return (next_state, next_mask), outputs

    _, (health_history, cooldown_history) = jax.lax.scan(
        scan_body,
        (initial_state, initial_mask),
        xs=None,
        length=3,
    )

    expected_health = (
        initial_state.current_health[_TEAM_B_FIRST_SLOT]
        - (combat.ULTIMATE_DAMAGE_BY_CLASS[WARRIOR_CLASS_ID])
    )
    assert bool(jnp.all(health_history == expected_health))
    assert bool(
        jnp.array_equal(
            cooldown_history,
            jnp.asarray((30, 29, 28), dtype=jnp.int32),
        )
    )


def test_every_status_ultimate_updates_only_its_owned_source_or_recipient() -> None:
    """Prove all four status ultimates use one source-local recipient route."""
    config, state = _scenario(
        (0, MAGE_CLASS_ID),
        (1, WARRIOR_CLASS_ID),
        (2, HUNTER_CLASS_ID),
        (3, ROGUE_CLASS_ID),
        (5, HUNTER_CLASS_ID),
        (6, HUNTER_CLASS_ID),
        (7, HUNTER_CLASS_ID),
        (8, HUNTER_CLASS_ID),
        team_sizes=(4, 4),
    )
    action = _joint_action(
        (0, 0, 1),
        (1, _FIRST_ENEMY_TARGET, 1),
        (2, _SECOND_ENEMY_TARGET, 1),
        (3, _FIRST_ENEMY_TARGET + 2, 1),
    )

    next_state, _, _ = _step(config, state, action)

    assert next_state.mage_burst_damage_amplification_durations[0] == (
        combat.MAGE_BURST_DAMAGE_DURATION_TICKS
    )
    assert next_state.slow_durations[5, SLOW_CHANNEL_WARRIOR_CHARGE] == (
        combat.WARRIOR_CHARGE_SLOW_DURATION_TICKS
    )
    assert (
        next_state.stun_durations[5, STUN_CHANNEL_WARRIOR_CHARGE]
        == combat.WARRIOR_CHARGE_STUN_DURATION_TICKS
    )
    assert next_state.stun_durations[6, STUN_CHANNEL_HUNTER_TRAP] == (
        combat.HUNTER_TRAP_STUN_DURATION_TICKS
    )
    assert next_state.slow_durations[7, SLOW_CHANNEL_ROGUE_POISON] == (
        combat.ROGUE_POISON_SLOW_DURATION_TICKS
    )
    assert (
        next_state.stun_durations[7, STUN_CHANNEL_ROGUE_POISON]
        == combat.ROGUE_POISON_STUN_DURATION_TICKS
    )
    assert next_state.rogue_poison_anti_heal_durations[7] == (
        combat.ROGUE_POISON_ANTI_HEAL_DURATION_TICKS
    )
    assert bool(jnp.all(next_state.slow_durations[8] == 0))
    assert bool(jnp.all(next_state.stun_durations[8] == 0))
    assert bool(jnp.all(next_state.ultimate_cooldowns[:4] == 30))


def test_status_ultimates_route_symmetrically_from_team_b() -> None:
    """Prove team-B relation-local targets map into the stable team-A block."""
    config, state = _scenario(
        (0, HUNTER_CLASS_ID),
        (1, HUNTER_CLASS_ID),
        (5, HUNTER_CLASS_ID),
        (6, ROGUE_CLASS_ID),
        team_sizes=(2, 2),
    )
    action = _joint_action(
        (5, _FIRST_ENEMY_TARGET, 1),
        (6, _SECOND_ENEMY_TARGET, 1),
    )

    next_state, _, _ = _step(config, state, action)

    assert next_state.stun_durations[0, STUN_CHANNEL_HUNTER_TRAP] == (
        combat.HUNTER_TRAP_STUN_DURATION_TICKS
    )
    assert next_state.slow_durations[1, SLOW_CHANNEL_ROGUE_POISON] == (
        combat.ROGUE_POISON_SLOW_DURATION_TICKS
    )
    assert next_state.rogue_poison_anti_heal_durations[1] == (
        combat.ROGUE_POISON_ANTI_HEAL_DURATION_TICKS
    )
    assert bool(jnp.all(next_state.slow_durations[0] == 0))
    assert (
        next_state.stun_durations[1, STUN_CHANNEL_ROGUE_POISON]
        == combat.ROGUE_POISON_STUN_DURATION_TICKS
    )
    assert bool(jnp.all(next_state.stun_durations[1, :2] == 0))


def test_holy_word_applies_no_status_channels() -> None:
    """Prove Priest ultimate healing has no hidden status side effect."""
    config, state = _scenario(
        (0, PRIEST_CLASS_ID),
        (1, HUNTER_CLASS_ID),
        (5, ROGUE_CLASS_ID),
        team_sizes=(2, 1),
    )
    state = state._replace(current_health=state.current_health.at[1].set(10.0))

    next_state, _, _ = _step(
        config,
        state,
        _joint_action((0, _SECOND_ALLY_TARGET, 1)),
    )

    assert bool(jnp.all(next_state.slow_durations == 0))
    assert bool(jnp.all(next_state.stun_durations == 0))
    assert bool(jnp.all(next_state.rogue_poison_anti_heal_durations == 0))
    assert bool(jnp.all(next_state.mage_burst_damage_amplification_durations == 0))
    assert bool(
        jnp.all(next_state.priest_blessing_of_freedom_slow_floor_durations == 0)
    )


def test_duplicate_status_sources_refresh_once_without_duration_stacking() -> None:
    """Prove duplicate accepted sources reduce to one fixed-duration refresh."""
    config, state = _scenario(
        (0, ROGUE_CLASS_ID),
        (1, ROGUE_CLASS_ID),
        (5, HUNTER_CLASS_ID),
        team_sizes=(2, 1),
    )
    action = _joint_action(
        (0, _FIRST_ENEMY_TARGET, 1),
        (1, _FIRST_ENEMY_TARGET, 1),
    )

    next_state, _, _ = _step(config, state, action)

    assert next_state.slow_durations[5, SLOW_CHANNEL_ROGUE_POISON] == (
        combat.ROGUE_POISON_SLOW_DURATION_TICKS
    )
    assert next_state.rogue_poison_anti_heal_durations[5] == (
        combat.ROGUE_POISON_ANTI_HEAL_DURATION_TICKS
    )
    assert (
        next_state.stun_durations[5, STUN_CHANNEL_ROGUE_POISON]
        == combat.ROGUE_POISON_STUN_DURATION_TICKS
    )


def test_refresh_never_shortens_and_preserves_concurrent_channels() -> None:
    """Prove each application refreshes only its owned irreducible duration."""
    config, state = _scenario(
        (0, MAGE_CLASS_ID),
        (1, WARRIOR_CLASS_ID),
        (2, HUNTER_CLASS_ID),
        (3, ROGUE_CLASS_ID),
        (5, HUNTER_CLASS_ID),
        (6, HUNTER_CLASS_ID),
        (7, HUNTER_CLASS_ID),
        (8, HUNTER_CLASS_ID),
        team_sizes=(4, 4),
    )
    state = state._replace(
        mage_burst_damage_amplification_durations=(
            state.mage_burst_damage_amplification_durations.at[0].set(8)
        ),
        slow_durations=(
            state.slow_durations.at[5, SLOW_CHANNEL_WARRIOR_CHARGE]
            .set(9)
            .at[5, SLOW_CHANNEL_HUNTER_BASIC]
            .set(6)
            .at[7, SLOW_CHANNEL_ROGUE_POISON]
            .set(8)
        ),
        stun_durations=(
            state.stun_durations.at[5, STUN_CHANNEL_WARRIOR_CHARGE]
            .set(3)
            .at[6, STUN_CHANNEL_HUNTER_TRAP]
            .set(7)
            .at[7, STUN_CHANNEL_ROGUE_POISON]
            .set(4)
        ),
        rogue_poison_anti_heal_durations=(
            state.rogue_poison_anti_heal_durations.at[7].set(9)
        ),
        priest_blessing_of_freedom_slow_floor_durations=(
            state.priest_blessing_of_freedom_slow_floor_durations.at[8].set(4)
        ),
    )
    action = _joint_action(
        (0, 0, 1),
        (1, _FIRST_ENEMY_TARGET, 1),
        (2, _SECOND_ENEMY_TARGET, 1),
        (3, _FIRST_ENEMY_TARGET + 2, 1),
    )

    next_state, _, _ = _step(config, state, action)

    assert next_state.mage_burst_damage_amplification_durations[0] == 7
    assert next_state.slow_durations[5, SLOW_CHANNEL_WARRIOR_CHARGE] == 8
    assert next_state.slow_durations[5, SLOW_CHANNEL_HUNTER_BASIC] == 5
    assert next_state.stun_durations[5, STUN_CHANNEL_WARRIOR_CHARGE] == 2
    assert next_state.stun_durations[6, STUN_CHANNEL_HUNTER_TRAP] == 6
    assert next_state.slow_durations[7, SLOW_CHANNEL_ROGUE_POISON] == 7
    assert next_state.stun_durations[7, STUN_CHANNEL_ROGUE_POISON] == 3
    assert next_state.rogue_poison_anti_heal_durations[7] == 8
    assert next_state.priest_blessing_of_freedom_slow_floor_durations[8] == 3


@pytest.mark.parametrize(
    ("damage_actor_class_id", "uses_ultimate"),
    (
        pytest.param(MAGE_CLASS_ID, 0, id="damaging-basic"),
        pytest.param(WARRIOR_CLASS_ID, 1, id="charge-damage"),
    ),
)
def test_accepted_raw_damage_breaks_only_pre_existing_hunter_trap(
    damage_actor_class_id: int,
    uses_ultimate: int,
) -> None:
    """Prove accepted raw damage clears Trap without erasing other stuns."""
    config, state = _scenario(
        (0, damage_actor_class_id),
        (5, HUNTER_CLASS_ID),
        team_sizes=(1, 1),
    )
    state = state._replace(
        stun_durations=(
            state.stun_durations.at[5, STUN_CHANNEL_WARRIOR_CHARGE]
            .set(3)
            .at[5, STUN_CHANNEL_HUNTER_TRAP]
            .set(4)
            .at[5, STUN_CHANNEL_ROGUE_POISON]
            .set(3)
        )
    )

    next_state, _, _ = _step(
        config,
        state,
        _joint_action((0, _FIRST_ENEMY_TARGET, uses_ultimate)),
    )

    assert next_state.stun_durations[5, STUN_CHANNEL_HUNTER_TRAP] == 0
    assert next_state.stun_durations[5, STUN_CHANNEL_WARRIOR_CHARGE] == 2
    assert next_state.stun_durations[5, STUN_CHANNEL_ROGUE_POISON] == 2


def test_accepted_damage_breaks_trap_at_zero_health_without_effective_loss() -> None:
    """Prove Trap break follows accepted raw damage, not final health delta."""
    config, state = _scenario(
        (0, MAGE_CLASS_ID),
        (5, HUNTER_CLASS_ID),
        team_sizes=(1, 1),
    )
    state = state._replace(
        current_health=state.current_health.at[5].set(0.0),
        stun_durations=state.stun_durations.at[5, STUN_CHANNEL_HUNTER_TRAP].set(4),
    )

    current_mask = _current_action_mask(config, state)
    assert not bool(jnp.any(current_mask.select_target_mask[5, 1:]))

    next_state, _, next_mask = _step(
        config,
        state,
        _joint_action((0, _FIRST_ENEMY_TARGET, 0)),
    )

    assert next_state.current_health[5] == 0.0
    assert next_state.stun_durations[5, STUN_CHANNEL_HUNTER_TRAP] == 0
    assert bool(jnp.any(next_mask.select_target_mask[5, 1:]))


def test_non_damage_effects_do_not_break_pre_existing_hunter_trap() -> None:
    """Prove accepted healing and zero-damage Poison only tick existing Trap."""
    config, state = _scenario(
        (0, PRIEST_CLASS_ID),
        (1, HUNTER_CLASS_ID),
        (5, ROGUE_CLASS_ID),
        team_sizes=(2, 1),
    )
    state = state._replace(
        current_health=state.current_health.at[1].set(10.0),
        stun_durations=(
            state.stun_durations.at[1, STUN_CHANNEL_HUNTER_TRAP]
            .set(4)
            .at[5, STUN_CHANNEL_HUNTER_TRAP]
            .set(4)
        ),
    )
    action = _joint_action(
        (0, _SECOND_ALLY_TARGET, 1),
        (5, _FIRST_ENEMY_TARGET, 1),
    )

    next_state, _, _ = _step(config, state, action)

    assert next_state.stun_durations[1, STUN_CHANNEL_HUNTER_TRAP] == 3
    assert next_state.stun_durations[5, STUN_CHANNEL_HUNTER_TRAP] == 3


def test_new_trap_survives_damage_that_breaks_the_pre_existing_trap() -> None:
    """Prove break precedes same-transition Trap refresh deterministically."""
    config, state = _scenario(
        (0, MAGE_CLASS_ID),
        (1, HUNTER_CLASS_ID),
        (5, ROGUE_CLASS_ID),
        team_sizes=(2, 1),
    )
    state = state._replace(
        stun_durations=state.stun_durations.at[5, STUN_CHANNEL_HUNTER_TRAP].set(7)
    )
    action = _joint_action(
        (0, _FIRST_ENEMY_TARGET, 0),
        (1, _FIRST_ENEMY_TARGET, 1),
    )

    next_state, _, _ = _step(config, state, action)

    assert next_state.stun_durations[5, STUN_CHANNEL_HUNTER_TRAP] == (
        combat.HUNTER_TRAP_STUN_DURATION_TICKS
    )


def test_rejected_ultimate_applies_no_status_and_existing_durations_tick() -> None:
    """Prove cooldown rejection reaches the common accepted no-op lifecycle."""
    config, state = _scenario(
        (0, ROGUE_CLASS_ID),
        (5, HUNTER_CLASS_ID),
        team_sizes=(1, 1),
    )
    state = state._replace(
        ultimate_cooldowns=state.ultimate_cooldowns.at[0].set(1),
        slow_durations=state.slow_durations.at[5, SLOW_CHANNEL_HUNTER_BASIC].set(3),
    )

    next_state, _, _ = _step(
        config,
        state,
        _joint_action((0, _FIRST_ENEMY_TARGET, 1)),
    )

    assert bool(jnp.all(next_state.slow_durations[5, [0, 2]] == 0))
    assert next_state.slow_durations[5, SLOW_CHANNEL_HUNTER_BASIC] == 2
    assert bool(jnp.all(next_state.stun_durations[5] == 0))
    assert next_state.rogue_poison_anti_heal_durations[5] == 0
    assert next_state.ultimate_cooldowns[0] == 0


def test_complete_lifecycle_ticks_nonnegative_and_observation_uses_next_state() -> None:
    """Prove zero, one, and multi-tick durations update every public surface."""
    config, state = _scenario(
        (0, MAGE_CLASS_ID),
        (1, HUNTER_CLASS_ID),
        (5, ROGUE_CLASS_ID),
        team_sizes=(2, 1),
    )
    state = state._replace(
        slow_durations=(
            state.slow_durations.at[0, SLOW_CHANNEL_WARRIOR_CHARGE]
            .set(1)
            .at[1, SLOW_CHANNEL_ROGUE_POISON]
            .set(3)
        ),
        stun_durations=state.stun_durations.at[0, STUN_CHANNEL_HUNTER_TRAP].set(2),
        rogue_poison_anti_heal_durations=(
            state.rogue_poison_anti_heal_durations.at[5].set(3)
        ),
        mage_burst_damage_amplification_durations=(
            state.mage_burst_damage_amplification_durations.at[0].set(1)
        ),
        priest_blessing_of_freedom_slow_floor_durations=(
            state.priest_blessing_of_freedom_slow_floor_durations.at[1].set(1)
        ),
    )

    next_state, observation, next_mask = _step(config, state, _joint_action())

    assert bool(jnp.all(next_state.slow_durations >= 0))
    assert next_state.slow_durations[0, SLOW_CHANNEL_WARRIOR_CHARGE] == 0
    assert next_state.slow_durations[1, SLOW_CHANNEL_ROGUE_POISON] == 2
    assert next_state.stun_durations[0, STUN_CHANNEL_HUNTER_TRAP] == 1
    assert next_state.rogue_poison_anti_heal_durations[5] == 2
    assert next_state.mage_burst_damage_amplification_durations[0] == 0
    assert next_state.priest_blessing_of_freedom_slow_floor_durations[1] == 0
    assert observation.self_features[1, AGENT_FEATURE_SLOW_ROGUE_POISON_DURATION] == 2.0
    assert observation.self_features[0, AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION] == 1.0
    assert (
        observation.self_features[5, AGENT_FEATURE_ANTI_HEAL_ROGUE_POISON_DURATION]
        == 2.0
    )
    assert (
        observation.self_features[
            0, AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_BURST_DURATION
        ]
        == 0.0
    )
    assert not bool(jnp.any(next_mask.select_target_mask[0, 1:]))


def test_jitted_step_matches_eager_status_application_and_trap_break() -> None:
    """Prove compiled execution preserves mixed lifecycle ordering."""
    config, state = _scenario(
        (0, MAGE_CLASS_ID),
        (1, HUNTER_CLASS_ID),
        (5, ROGUE_CLASS_ID),
        team_sizes=(2, 1),
    )
    state = state._replace(
        stun_durations=state.stun_durations.at[5, STUN_CHANNEL_HUNTER_TRAP].set(6)
    )
    action = _joint_action(
        (0, _FIRST_ENEMY_TARGET, 0),
        (1, _FIRST_ENEMY_TARGET, 1),
    )
    current_mask = _current_action_mask(config, state)

    eager = step(config, state, current_mask, action, jax.random.key(41))
    compiled = cast(
        tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info],
        jax.jit(step)(config, state, current_mask, action, jax.random.key(41)),
    )

    for eager_leaf, compiled_leaf in zip(
        jax.tree_util.tree_leaves(eager),
        jax.tree_util.tree_leaves(compiled),
        strict=True,
    ):
        assert bool(jnp.array_equal(eager_leaf, compiled_leaf))


def test_scanned_status_application_occurs_once_then_every_duration_ticks() -> None:
    """Prove mask reuse and complete lifecycle remain stable under scan."""
    config, initial_state = _scenario(
        (0, HUNTER_CLASS_ID),
        (5, ROGUE_CLASS_ID),
        team_sizes=(1, 1),
    )
    initial_mask = _current_action_mask(config, initial_state)
    action = _joint_action((0, _FIRST_ENEMY_TARGET, 1))

    def scan_body(
        carry: tuple[EnvState, ActionMask], _: None
    ) -> tuple[tuple[EnvState, ActionMask], tuple[Array, Array, Array]]:
        current_state, current_mask = carry
        (
            next_state,
            observation,
            _reward,
            _done_flags,
            next_mask,
            _info,
        ) = step(
            config,
            current_state,
            current_mask,
            action,
            jax.random.key(43),
        )
        trap_duration = next_state.stun_durations[5, STUN_CHANNEL_HUNTER_TRAP]
        observed_trap_duration = observation.self_features[
            5, AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION
        ]
        has_nonempty_combat_target = jnp.any(next_mask.select_target_mask[5, 1:])
        return (next_state, next_mask), (
            trap_duration,
            observed_trap_duration,
            has_nonempty_combat_target,
        )

    def rollout(
        state: EnvState,
        action_mask: ActionMask,
    ) -> tuple[tuple[EnvState, ActionMask], tuple[Array, Array, Array]]:
        """Carry paired state and mask through the fixed Trap trajectory."""
        return jax.lax.scan(
            scan_body,
            (state, action_mask),
            xs=None,
            length=5,
        )

    eager_rollout = rollout(initial_state, initial_mask)
    compiled_rollout = cast(
        tuple[tuple[EnvState, ActionMask], tuple[Array, Array, Array]],
        jax.jit(rollout)(initial_state, initial_mask),
    )
    for eager_leaf, compiled_leaf in zip(
        jax.tree_util.tree_leaves(eager_rollout),
        jax.tree_util.tree_leaves(compiled_rollout),
        strict=True,
    ):
        assert bool(jnp.array_equal(eager_leaf, compiled_leaf))

    (
        (final_state, _),
        (
            duration_history,
            observed_duration_history,
            has_nonempty_target_history,
        ),
    ) = compiled_rollout

    assert bool(
        jnp.array_equal(
            duration_history,
            jnp.asarray((4, 3, 2, 1, 0), dtype=jnp.int32),
        )
    )
    assert bool(
        jnp.array_equal(
            observed_duration_history,
            jnp.asarray((4, 3, 2, 1, 0), dtype=jnp.float32),
        )
    )
    assert bool(
        jnp.array_equal(
            has_nonempty_target_history,
            jnp.asarray((False, False, False, False, True)),
        )
    )
    assert final_state.ultimate_cooldowns[0] == 26
