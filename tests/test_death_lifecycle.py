"""Resolve-then-die lifecycle and public death-fact proofs for Milestone 6."""

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
    reset,
    step,
)
from marl_battlegrounds.core.types import (
    AGENT_FEATURE_ACTIVE,
    AGENT_FEATURE_ALIVE,
    AGENT_FEATURE_CURRENT_HEALTH,
    ENVIRONMENT_DIMENSIONS,
    HUNTER_CLASS_ID,
    MAGE_CLASS_ID,
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    MAX_OBSTACLE_SLOTS,
    MOVE_EAST,
    MOVE_NORTH,
    MOVE_STAY,
    NEUTRAL_CLASS_ID,
    NUM_MOVE_ACTIONS,
    NUM_TARGET_ACTIONS,
    NUM_ULTIMATE_ACTIONS,
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
    STUN_CHANNEL_HUNTER_TRAP,
    STUN_CHANNEL_ROGUE_POISON,
    STUN_CHANNEL_WARRIOR_CHARGE,
    WARRIOR_CLASS_ID,
    Action,
    ActionMask,
    DeathTransitionFacts,
    DoneFlags,
    EnvConfig,
    EnvState,
    Info,
    Observation,
    Reward,
    TransitionFacts,
)

_TEAM_A_FIRST_SLOT = 0
_TEAM_A_SECOND_SLOT = 1
_TEAM_A_THIRD_SLOT = 2
_TEAM_B_FIRST_SLOT = MAX_AGENTS_PER_TEAM
_TEAM_B_SECOND_SLOT = MAX_AGENTS_PER_TEAM + 1

_TARGET_NONE = 0
_FIRST_ALLY_TARGET = 1
_SECOND_ALLY_TARGET = 2
_FIRST_ENEMY_TARGET = 1 + MAX_AGENTS_PER_TEAM
_SECOND_ENEMY_TARGET = _FIRST_ENEMY_TARGET + 1


def _empty_obstacles() -> Array:
    """Return an inactive fixed-size obstacle table."""
    return jnp.zeros((MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES), dtype=jnp.float32)


def _clear_pillar_obstacles() -> Array:
    """Return one active pillar clear of the default combat trajectories."""
    obstacles = _empty_obstacles()
    obstacles = obstacles.at[0, OBSTACLE_FEATURE_TYPE].set(OBSTACLE_TYPE_PILLAR)
    obstacles = obstacles.at[0, OBSTACLE_FEATURE_X].set(15.0)
    obstacles = obstacles.at[0, OBSTACLE_FEATURE_Y].set(10.0)
    obstacles = obstacles.at[0, OBSTACLE_FEATURE_RADIUS].set(1.0)
    return obstacles.at[0, OBSTACLE_FEATURE_ACTIVE].set(1.0)


def _requested_roster(
    team_sizes: tuple[int, int],
    *class_rows: tuple[int, int],
) -> Array:
    """Return a padded active roster with selected class assignments."""
    roster = jnp.full((MAX_AGENT_SLOTS,), NEUTRAL_CLASS_ID, dtype=jnp.int32)
    roster = roster.at[: team_sizes[0]].set(HUNTER_CLASS_ID)
    roster = roster.at[MAX_AGENTS_PER_TEAM : MAX_AGENTS_PER_TEAM + team_sizes[1]].set(
        HUNTER_CLASS_ID
    )
    for slot, class_id in class_rows:
        roster = roster.at[slot].set(class_id)
    return roster


def _default_positions(team_sizes: tuple[int, int]) -> Array:
    """Place both active team blocks on clear, non-overlapping vertical lines."""
    positions = jnp.zeros((MAX_AGENT_SLOTS, ENVIRONMENT_DIMENSIONS), dtype=jnp.float32)
    for local_slot in range(team_sizes[0]):
        positions = positions.at[local_slot].set(
            jnp.asarray((3.0, 2.0 + 2.0 * local_slot), dtype=jnp.float32)
        )
    for local_slot in range(team_sizes[1]):
        positions = positions.at[MAX_AGENTS_PER_TEAM + local_slot].set(
            jnp.asarray((10.0, 2.0 + 2.0 * local_slot), dtype=jnp.float32)
        )
    return positions


def _scenario(
    *class_rows: tuple[int, int],
    team_sizes: tuple[int, int] = (1, 1),
    max_steps: int = 100,
    positions: Array | None = None,
    obstacles: Array | None = None,
    ordinary_movement_distance_scale: float = 0.25,
) -> tuple[EnvConfig, EnvState, ActionMask, Info]:
    """Build a deterministic, fully observable public combat snapshot."""
    profile = resolve_agent_profile(
        _requested_roster(team_sizes, *class_rows),
        jnp.asarray(team_sizes, dtype=jnp.int32),
    )
    profile = profile._replace(
        observation_radii=jnp.where(profile.active_mask, 20.0, 0.0).astype(jnp.float32),
        basic_interaction_radii=jnp.where(profile.active_mask, 20.0, 0.0).astype(
            jnp.float32
        ),
        ultimate_interaction_radii=jnp.where(profile.active_mask, 20.0, 0.0).astype(
            jnp.float32
        ),
    )
    initial_positions = (
        _default_positions(team_sizes) if positions is None else positions
    )
    config = EnvConfig(
        max_steps=max_steps,
        map_width=20.0,
        map_height=12.0,
        obstacles=_empty_obstacles() if obstacles is None else obstacles,
        agent_profile=profile,
        ordinary_movement_distance_scale=ordinary_movement_distance_scale,
        team_spawn_pad_positions=initial_positions.reshape(
            (2, MAX_AGENTS_PER_TEAM, ENVIRONMENT_DIMENSIONS)
        ),
        spawn_shield_duration_steps=3,
        spawn_shield_movement_speed=2.0,
        team_respawn_wave_period_step_count=jnp.asarray((5, 5), dtype=jnp.int32),
    )
    state, _, action_mask, info = reset(config, jax.random.key(1))
    return config, state, action_mask, info


def _joint_action(*rows: tuple[int, int, int, int]) -> Action:
    """Return a canonical joint action with selected actor overrides.

    Each row is ``(actor_slot, move, target, use_ultimate)``.
    """
    move = jnp.full((MAX_AGENT_SLOTS,), MOVE_STAY, dtype=jnp.int32)
    select_target = jnp.full((MAX_AGENT_SLOTS,), _TARGET_NONE, dtype=jnp.int32)
    use_ultimate = jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32)
    for actor_slot, move_action, target_action, ultimate_action in rows:
        move = move.at[actor_slot].set(move_action)
        select_target = select_target.at[actor_slot].set(target_action)
        use_ultimate = use_ultimate.at[actor_slot].set(ultimate_action)
    return Action(
        move=move,
        select_target=select_target,
        use_ultimate=use_ultimate,
    )


def _choose_desired_action_from_mask(
    action_mask: ActionMask,
    desired_action: Action,
) -> Action:
    """Choose each desired action when legal and a deterministic fallback otherwise."""
    actor_slots = jnp.arange(MAX_AGENT_SLOTS)
    desired_move_is_legal = action_mask.move_mask[actor_slots, desired_action.move]
    fallback_move = jnp.argmax(action_mask.move_mask, axis=-1)
    chosen_move = jnp.where(
        desired_move_is_legal,
        desired_action.move,
        fallback_move,
    ).astype(jnp.int32)

    desired_pair_is_legal = action_mask.select_target_use_ultimate_joint_mask[
        actor_slots,
        desired_action.select_target,
        desired_action.use_ultimate,
    ]
    flattened_pair_mask = action_mask.select_target_use_ultimate_joint_mask.reshape(
        MAX_AGENT_SLOTS,
        NUM_TARGET_ACTIONS * NUM_ULTIMATE_ACTIONS,
    )
    fallback_pair = jnp.argmax(flattened_pair_mask, axis=-1)
    fallback_target = fallback_pair // NUM_ULTIMATE_ACTIONS
    fallback_ultimate = fallback_pair % NUM_ULTIMATE_ACTIONS

    return Action(
        move=chosen_move,
        select_target=jnp.where(
            desired_pair_is_legal,
            desired_action.select_target,
            fallback_target,
        ).astype(jnp.int32),
        use_ultimate=jnp.where(
            desired_pair_is_legal,
            desired_action.use_ultimate,
            fallback_ultimate,
        ).astype(jnp.int32),
    )


def _current_action_mask(config: EnvConfig, state: EnvState) -> ActionMask:
    """Return the authoritative action mask paired with a direct test state."""
    return _build_observation_and_action_mask(state, config)[1]


def _take_step(
    config: EnvConfig,
    state: EnvState,
    action: Action,
    *,
    action_mask: ActionMask | None = None,
    key: Array | None = None,
) -> tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info]:
    """Advance one deterministic public transition."""
    choosing_mask = (
        _current_action_mask(config, state) if action_mask is None else action_mask
    )
    return step(
        config,
        state,
        choosing_mask,
        action,
        jax.random.key(2) if key is None else key,
    )


def _assert_tree_equal(left: object, right: object) -> None:
    """Assert exact equality for two identically structured JAX PyTrees."""
    assert jax.tree_util.tree_structure(left) == jax.tree_util.tree_structure(right)
    for left_leaf, right_leaf in zip(
        jax.tree_util.tree_leaves(left),
        jax.tree_util.tree_leaves(right),
        strict=True,
    ):
        assert bool(jnp.array_equal(left_leaf, right_leaf))


def _slot_mask(*slots: int) -> Array:
    """Return a fixed-slot boolean mask selecting exactly ``slots``."""
    mask = jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.bool_)
    for slot in slots:
        mask = mask.at[slot].set(True)
    return mask


def _assert_statuses_are_clear(state: EnvState, slot: int) -> None:
    """Assert every transient status family is zero for one successor slot."""
    assert bool(jnp.all(state.slow_durations[slot] == 0))
    assert bool(jnp.all(state.stun_durations[slot] == 0))
    assert int(state.rogue_poison_anti_heal_durations[slot]) == 0
    assert int(state.mage_burst_damage_amplification_durations[slot]) == 0
    assert int(state.priest_blessing_of_freedom_slow_floor_durations[slot]) == 0


def _assert_canonical_dead_action_mask(mask: ActionMask, slot: int) -> None:
    """Assert a dead slot exposes exactly the sampleable neutral action."""
    expected_move = jnp.arange(NUM_MOVE_ACTIONS) == MOVE_STAY
    expected_target = jnp.arange(NUM_TARGET_ACTIONS) == _TARGET_NONE
    expected_ultimate = jnp.arange(NUM_ULTIMATE_ACTIONS) == 0
    expected_joint = jnp.logical_and(
        expected_target[:, None], expected_ultimate[None, :]
    )

    assert bool(jnp.array_equal(mask.move_mask[slot], expected_move))
    assert bool(jnp.array_equal(mask.select_target_mask[slot], expected_target))
    assert bool(jnp.array_equal(mask.use_ultimate_mask[slot], expected_ultimate))
    assert bool(
        jnp.array_equal(
            mask.select_target_use_ultimate_joint_mask[slot],
            expected_joint,
        )
    )


def test_reset_and_neutral_step_publish_canonical_death_fact_schema() -> None:
    """Prove reset and real steps share one fixed death-fact PyTree."""
    config, state, action_mask, reset_info = _scenario()
    reset_facts = reset_info.transition_facts
    death_facts = reset_facts.death_facts

    assert DeathTransitionFacts._fields == (
        "is_newly_dead_by_recipient",
        "contributed_to_new_death_by_source",
        "attributed_death_damage_by_source",
    )
    assert "death_facts" in TransitionFacts._fields
    assert death_facts.is_newly_dead_by_recipient.shape == (MAX_AGENT_SLOTS,)
    assert death_facts.is_newly_dead_by_recipient.dtype == jnp.bool_
    assert death_facts.contributed_to_new_death_by_source.dtype == jnp.bool_
    assert death_facts.attributed_death_damage_by_source.dtype == jnp.float32
    assert not bool(jnp.any(death_facts.is_newly_dead_by_recipient))
    assert not bool(jnp.any(death_facts.contributed_to_new_death_by_source))
    assert bool(jnp.all(death_facts.attributed_death_damage_by_source == 0.0))

    *_, neutral_info = _take_step(
        config,
        state,
        _joint_action(),
        action_mask=action_mask,
    )
    neutral_facts = neutral_info.transition_facts
    assert jax.tree_util.tree_structure(neutral_facts) == jax.tree_util.tree_structure(
        reset_facts
    )
    assert bool(neutral_facts.has_transition)
    assert not bool(jnp.any(neutral_facts.death_facts.is_newly_dead_by_recipient))
    assert not bool(
        jnp.any(neutral_facts.death_facts.contributed_to_new_death_by_source)
    )
    assert bool(
        jnp.all(neutral_facts.death_facts.attributed_death_damage_by_source == 0.0)
    )


@pytest.mark.parametrize(
    "source_class_id",
    (
        pytest.param(MAGE_CLASS_ID, id="mage-basic"),
        pytest.param(WARRIOR_CLASS_ID, id="warrior-basic"),
        pytest.param(HUNTER_CLASS_ID, id="hunter-basic"),
        pytest.param(ROGUE_CLASS_ID, id="rogue-basic"),
    ),
)
def test_every_damage_basic_produces_one_exact_successor_death(
    source_class_id: int,
) -> None:
    """Prove all damaging Basic lanes use the same death and attribution rule."""
    config, state, _, _ = _scenario(
        (_TEAM_A_FIRST_SLOT, source_class_id),
        (_TEAM_B_FIRST_SLOT, HUNTER_CLASS_ID),
    )
    state = state._replace(
        current_health=state.current_health.at[_TEAM_B_FIRST_SLOT].set(1.0)
    )

    next_state, *_, info = _take_step(
        config,
        state,
        _joint_action(
            (
                _TEAM_A_FIRST_SLOT,
                MOVE_STAY,
                _FIRST_ENEMY_TARGET,
                0,
            )
        ),
    )
    combat_facts = info.transition_facts.combat_transition_facts
    death_facts = info.transition_facts.death_facts
    expected_attribution = (
        combat_facts.source_modified_damage_output_by_source[_TEAM_A_FIRST_SLOT]
        * combat_facts.recipient_damage_modifier_by_source[_TEAM_A_FIRST_SLOT]
    )

    assert bool(
        jnp.array_equal(
            death_facts.is_newly_dead_by_recipient,
            _slot_mask(_TEAM_B_FIRST_SLOT),
        )
    )
    assert bool(
        jnp.array_equal(
            death_facts.contributed_to_new_death_by_source,
            _slot_mask(_TEAM_A_FIRST_SLOT),
        )
    )
    assert bool(
        jnp.isclose(
            death_facts.attributed_death_damage_by_source[_TEAM_A_FIRST_SLOT],
            expected_attribution,
        )
    )
    assert not bool(next_state.alive_mask[_TEAM_B_FIRST_SLOT])
    assert float(next_state.current_health[_TEAM_B_FIRST_SLOT]) == 0.0
    _assert_statuses_are_clear(next_state, _TEAM_B_FIRST_SLOT)


def test_exact_zero_health_boundary_is_a_death_not_a_surviving_zero() -> None:
    """Prove clamped exact zero cannot remain officially alive."""
    config, state, _, _ = _scenario(
        (_TEAM_A_FIRST_SLOT, ROGUE_CLASS_ID),
        (_TEAM_B_FIRST_SLOT, HUNTER_CLASS_ID),
    )
    exact_damage = combat.BASIC_DAMAGE_BY_CLASS[ROGUE_CLASS_ID]
    state = state._replace(
        current_health=state.current_health.at[_TEAM_B_FIRST_SLOT].set(exact_damage)
    )

    next_state, *_, info = _take_step(
        config,
        state,
        _joint_action(
            (
                _TEAM_A_FIRST_SLOT,
                MOVE_STAY,
                _FIRST_ENEMY_TARGET,
                0,
            )
        ),
    )

    assert float(next_state.current_health[_TEAM_B_FIRST_SLOT]) == 0.0
    assert not bool(next_state.alive_mask[_TEAM_B_FIRST_SLOT])
    assert bool(
        info.transition_facts.death_facts.is_newly_dead_by_recipient[_TEAM_B_FIRST_SLOT]
    )


@pytest.mark.parametrize(
    "source_class_id",
    (
        pytest.param(WARRIOR_CLASS_ID, id="warrior-charge"),
        pytest.param(HUNTER_CLASS_ID, id="hunter-trap"),
        pytest.param(ROGUE_CLASS_ID, id="rogue-poison"),
    ),
)
def test_every_damage_ultimate_produces_gross_attributed_death_damage(
    source_class_id: int,
) -> None:
    """Prove all damaging Ultimates retain gross contribution and cooldown."""
    config, state, _, _ = _scenario(
        (_TEAM_A_FIRST_SLOT, source_class_id),
        (_TEAM_B_FIRST_SLOT, HUNTER_CLASS_ID),
    )
    state = state._replace(
        current_health=state.current_health.at[_TEAM_B_FIRST_SLOT].set(1.0)
    )

    next_state, *_, info = _take_step(
        config,
        state,
        _joint_action(
            (
                _TEAM_A_FIRST_SLOT,
                MOVE_STAY,
                _FIRST_ENEMY_TARGET,
                1,
            )
        ),
    )
    combat_facts = info.transition_facts.combat_transition_facts
    death_facts = info.transition_facts.death_facts
    expected_attribution = (
        combat_facts.source_modified_damage_output_by_source[_TEAM_A_FIRST_SLOT]
        * combat_facts.recipient_damage_modifier_by_source[_TEAM_A_FIRST_SLOT]
    )

    assert bool(death_facts.is_newly_dead_by_recipient[_TEAM_B_FIRST_SLOT])
    assert bool(death_facts.contributed_to_new_death_by_source[_TEAM_A_FIRST_SLOT])
    assert bool(
        jnp.isclose(
            death_facts.attributed_death_damage_by_source[_TEAM_A_FIRST_SLOT],
            expected_attribution,
        )
    )
    assert float(expected_attribution) > 1.0
    assert (
        next_state.ultimate_cooldowns[_TEAM_A_FIRST_SLOT]
        == combat.ULTIMATE_COOLDOWN_BY_CLASS[source_class_id]
    )
    _assert_statuses_are_clear(next_state, _TEAM_B_FIRST_SLOT)


def test_mutual_lethal_actions_resolve_before_both_successor_deaths() -> None:
    """Prove simultaneous actors can trade without actor-order cancellation."""
    config, state, _, _ = _scenario(
        (_TEAM_A_FIRST_SLOT, ROGUE_CLASS_ID),
        (_TEAM_B_FIRST_SLOT, ROGUE_CLASS_ID),
    )
    state = state._replace(
        current_health=state.current_health.at[_TEAM_A_FIRST_SLOT]
        .set(1.0)
        .at[_TEAM_B_FIRST_SLOT]
        .set(1.0)
    )
    action = _joint_action(
        (
            _TEAM_A_FIRST_SLOT,
            MOVE_EAST,
            _FIRST_ENEMY_TARGET,
            0,
        ),
        (
            _TEAM_B_FIRST_SLOT,
            MOVE_EAST,
            _FIRST_ENEMY_TARGET,
            0,
        ),
    )

    next_state, *_, info = _take_step(config, state, action)
    expected_slots = _slot_mask(_TEAM_A_FIRST_SLOT, _TEAM_B_FIRST_SLOT)

    assert bool(
        jnp.array_equal(
            info.transition_facts.death_facts.is_newly_dead_by_recipient,
            expected_slots,
        )
    )
    assert bool(
        jnp.array_equal(
            info.transition_facts.death_facts.contributed_to_new_death_by_source,
            expected_slots,
        )
    )
    assert bool(jnp.all(next_state.current_health[expected_slots] == 0.0))
    assert not bool(jnp.any(next_state.alive_mask[expected_slots]))
    assert not bool(
        jnp.array_equal(
            next_state.agent_positions[expected_slots],
            state.agent_positions[expected_slots],
        )
    )


@pytest.mark.parametrize("source_count", (2, 3), ids=("two-sources", "many-sources"))
def test_focus_fire_preserves_every_gross_contributor_through_overkill(
    source_count: int,
) -> None:
    """Prove focus fire records all sources without killer or HP-loss apportionment."""
    all_source_slots = (
        _TEAM_A_FIRST_SLOT,
        _TEAM_A_SECOND_SLOT,
        _TEAM_A_THIRD_SLOT,
    )
    source_slots = all_source_slots[:source_count]
    source_classes = (MAGE_CLASS_ID, HUNTER_CLASS_ID, ROGUE_CLASS_ID)
    config, state, _, _ = _scenario(
        *((slot, source_classes[slot]) for slot in source_slots),
        (_TEAM_B_FIRST_SLOT, WARRIOR_CLASS_ID),
        team_sizes=(source_count, 1),
    )
    state = state._replace(
        current_health=state.current_health.at[_TEAM_B_FIRST_SLOT].set(1.0)
    )
    action = _joint_action(
        *(
            (source_slot, MOVE_STAY, _FIRST_ENEMY_TARGET, 0)
            for source_slot in source_slots
        )
    )

    next_state, *_, info = _take_step(config, state, action)
    combat_facts = info.transition_facts.combat_transition_facts
    death_facts = info.transition_facts.death_facts
    source_indices = jnp.asarray(source_slots, dtype=jnp.int32)
    expected_attributions = (
        combat_facts.source_modified_damage_output_by_source[source_indices]
        * combat_facts.recipient_damage_modifier_by_source[source_indices]
    )

    assert not bool(next_state.alive_mask[_TEAM_B_FIRST_SLOT])
    assert bool(
        jnp.array_equal(
            death_facts.contributed_to_new_death_by_source,
            _slot_mask(*source_slots),
        )
    )
    assert bool(
        jnp.allclose(
            death_facts.attributed_death_damage_by_source[source_indices],
            expected_attributions,
        )
    )
    assert bool(
        jnp.isclose(
            jnp.sum(death_facts.attributed_death_damage_by_source[source_indices]),
            combat_facts.total_effective_damage_by_recipient[_TEAM_B_FIRST_SLOT],
        )
    )
    assert float(jnp.sum(expected_attributions)) > 1.0
    assert bool(
        jnp.all(
            combat_facts.recipient_damage_modifier_by_source[source_indices]
            == combat.WARRIOR_DAMAGE_MITIGATION_AURA_MULTIPLIER
        )
    )


def test_status_applying_damage_source_contributes_to_its_dead_recipient() -> None:
    """Prove Hunter Trap's status and damage share one accepted activation."""
    config, state, _, _ = _scenario(
        (_TEAM_A_FIRST_SLOT, HUNTER_CLASS_ID),
        (_TEAM_A_SECOND_SLOT, ROGUE_CLASS_ID),
        (_TEAM_B_FIRST_SLOT, MAGE_CLASS_ID),
        team_sizes=(2, 1),
    )
    state = state._replace(
        current_health=state.current_health.at[_TEAM_B_FIRST_SLOT].set(1.0)
    )
    action = _joint_action(
        (
            _TEAM_A_FIRST_SLOT,
            MOVE_STAY,
            _FIRST_ENEMY_TARGET,
            1,
        ),
        (
            _TEAM_A_SECOND_SLOT,
            MOVE_STAY,
            _FIRST_ENEMY_TARGET,
            0,
        ),
    )

    next_state, *_, info = _take_step(config, state, action)
    combat_facts = info.transition_facts.combat_transition_facts
    death_facts = info.transition_facts.death_facts

    assert bool(death_facts.is_newly_dead_by_recipient[_TEAM_B_FIRST_SLOT])
    assert bool(
        combat_facts.stun_is_applied_by_source_and_channel[
            _TEAM_A_FIRST_SLOT, STUN_CHANNEL_HUNTER_TRAP
        ]
    )
    assert bool(death_facts.contributed_to_new_death_by_source[_TEAM_A_FIRST_SLOT])
    assert (
        float(death_facts.attributed_death_damage_by_source[_TEAM_A_FIRST_SLOT]) > 0.0
    )
    assert bool(death_facts.contributed_to_new_death_by_source[_TEAM_A_SECOND_SLOT])
    _assert_statuses_are_clear(next_state, _TEAM_B_FIRST_SLOT)


@pytest.mark.parametrize(
    ("anti_heal_duration", "expected_death"),
    (
        pytest.param(0, False, id="priest-save"),
        pytest.param(1, True, id="anti-heal-defeats-save"),
    ),
)
def test_simultaneous_healing_and_anti_heal_decide_death_from_final_health(
    anti_heal_duration: int,
    expected_death: bool,
) -> None:
    """Prove death follows post-net health rather than damage receipt."""
    config, state, _, _ = _scenario(
        (_TEAM_A_FIRST_SLOT, HUNTER_CLASS_ID),
        (_TEAM_B_FIRST_SLOT, HUNTER_CLASS_ID),
        (_TEAM_B_SECOND_SLOT, PRIEST_CLASS_ID),
        team_sizes=(1, 2),
    )
    state = state._replace(
        current_health=state.current_health.at[_TEAM_B_FIRST_SLOT].set(1.0),
        rogue_poison_anti_heal_durations=(
            state.rogue_poison_anti_heal_durations.at[_TEAM_B_FIRST_SLOT].set(
                anti_heal_duration
            )
        ),
    )
    action = _joint_action(
        (
            _TEAM_A_FIRST_SLOT,
            MOVE_STAY,
            _FIRST_ENEMY_TARGET,
            0,
        ),
        (
            _TEAM_B_SECOND_SLOT,
            MOVE_STAY,
            _FIRST_ALLY_TARGET,
            0,
        ),
    )

    next_state, *_, info = _take_step(config, state, action)
    combat_facts = info.transition_facts.combat_transition_facts
    death_facts = info.transition_facts.death_facts
    expected_healing_modifier = (
        combat.ROGUE_POISON_ANTI_HEAL_MULTIPLIER if anti_heal_duration > 0 else 1.0
    )
    expected_effective_damage = (
        combat_facts.source_modified_damage_output_by_source[_TEAM_A_FIRST_SLOT]
        * combat_facts.recipient_damage_modifier_by_source[_TEAM_A_FIRST_SLOT]
    )
    expected_health = jnp.clip(
        1.0
        + combat.BASIC_HEALING_BY_CLASS[PRIEST_CLASS_ID] * expected_healing_modifier
        - combat.BASIC_DAMAGE_BY_CLASS[HUNTER_CLASS_ID],
        min=0.0,
        max=config.agent_profile.max_health[_TEAM_B_FIRST_SLOT],
    )

    assert bool(
        combat_facts.priest_blessing_of_freedom_is_applied_by_source[
            _TEAM_B_SECOND_SLOT
        ]
    )
    assert bool(
        jnp.isclose(
            combat_facts.recipient_healing_modifier_by_source[_TEAM_B_SECOND_SLOT],
            expected_healing_modifier,
        )
    )
    assert bool(
        jnp.isclose(
            next_state.current_health[_TEAM_B_FIRST_SLOT],
            expected_health,
        )
    )
    assert (
        bool(death_facts.is_newly_dead_by_recipient[_TEAM_B_FIRST_SLOT])
        is expected_death
    )
    assert (
        bool(death_facts.contributed_to_new_death_by_source[_TEAM_A_FIRST_SLOT])
        is expected_death
    )
    assert bool(
        jnp.isclose(
            death_facts.attributed_death_damage_by_source[_TEAM_A_FIRST_SLOT],
            expected_effective_damage if expected_death else 0.0,
        )
    )
    assert int(
        next_state.priest_blessing_of_freedom_slow_floor_durations[_TEAM_B_FIRST_SLOT]
    ) == (0 if expected_death else combat.PRIEST_HEAL_SPEED_FLOOR_DURATION_TICKS)


@pytest.mark.parametrize(
    (
        "source_class_id",
        "target_action",
        "use_ultimate",
        "move_action",
        "effect_kind",
    ),
    (
        pytest.param(
            MAGE_CLASS_ID,
            _TARGET_NONE,
            1,
            MOVE_EAST,
            "mage-burst",
            id="dying-mage-burst",
        ),
        pytest.param(
            WARRIOR_CLASS_ID,
            _FIRST_ENEMY_TARGET,
            1,
            MOVE_NORTH,
            "warrior-charge",
            id="dying-warrior-charge",
        ),
        pytest.param(
            HUNTER_CLASS_ID,
            _FIRST_ENEMY_TARGET,
            0,
            MOVE_EAST,
            "hunter-basic",
            id="dying-hunter-basic",
        ),
        pytest.param(
            HUNTER_CLASS_ID,
            _FIRST_ENEMY_TARGET,
            1,
            MOVE_EAST,
            "hunter-trap",
            id="dying-hunter-trap",
        ),
        pytest.param(
            ROGUE_CLASS_ID,
            _FIRST_ENEMY_TARGET,
            1,
            MOVE_EAST,
            "rogue-poison",
            id="dying-rogue-poison",
        ),
        pytest.param(
            PRIEST_CLASS_ID,
            _SECOND_ALLY_TARGET,
            0,
            MOVE_EAST,
            "priest-freedom",
            id="dying-priest-heal",
        ),
    ),
)
def test_dying_source_completes_each_class_defining_action(
    source_class_id: int,
    target_action: int,
    use_ultimate: int,
    move_action: int,
    effect_kind: str,
) -> None:
    """Prove a source's accepted action is never cancelled by its own death."""
    config, state, _, _ = _scenario(
        (_TEAM_A_FIRST_SLOT, source_class_id),
        (_TEAM_A_SECOND_SLOT, HUNTER_CLASS_ID),
        (_TEAM_B_FIRST_SLOT, ROGUE_CLASS_ID),
        team_sizes=(2, 1),
    )
    initial_source_cooldown = 0 if use_ultimate else 7
    state = state._replace(
        current_health=state.current_health.at[_TEAM_A_FIRST_SLOT]
        .set(1.0)
        .at[_TEAM_A_SECOND_SLOT]
        .set(50.0),
        ultimate_cooldowns=state.ultimate_cooldowns.at[_TEAM_A_FIRST_SLOT].set(
            initial_source_cooldown
        ),
    )
    source_position = state.agent_positions[_TEAM_A_FIRST_SLOT]
    action = _joint_action(
        (
            _TEAM_A_FIRST_SLOT,
            move_action,
            target_action,
            use_ultimate,
        ),
        (
            _TEAM_B_FIRST_SLOT,
            MOVE_STAY,
            _FIRST_ENEMY_TARGET,
            0,
        ),
    )

    next_state, _, _, _, next_action_mask, info = _take_step(
        config,
        state,
        action,
    )
    acceptance = info.transition_facts.action_acceptance_facts
    combat_facts = info.transition_facts.combat_transition_facts
    death_facts = info.transition_facts.death_facts

    assert bool(death_facts.is_newly_dead_by_recipient[_TEAM_A_FIRST_SLOT])
    assert bool(death_facts.contributed_to_new_death_by_source[_TEAM_B_FIRST_SLOT])
    assert not bool(death_facts.contributed_to_new_death_by_source[_TEAM_A_FIRST_SLOT])
    assert not bool(next_state.alive_mask[_TEAM_A_FIRST_SLOT])
    assert not bool(
        jnp.array_equal(
            next_state.agent_positions[_TEAM_A_FIRST_SLOT],
            source_position,
        )
    )
    assert int(acceptance.accepted_joint_action.move[_TEAM_A_FIRST_SLOT]) == move_action
    assert (
        int(acceptance.accepted_joint_action.select_target[_TEAM_A_FIRST_SLOT])
        == target_action
    )
    assert (
        int(acceptance.accepted_joint_action.use_ultimate[_TEAM_A_FIRST_SLOT])
        == use_ultimate
    )
    assert (
        int(next_state.previous_timestep_move_actions[_TEAM_A_FIRST_SLOT])
        == move_action
    )
    assert (
        int(next_state.previous_timestep_select_target_actions[_TEAM_A_FIRST_SLOT])
        == target_action
    )
    assert (
        int(next_state.previous_timestep_use_ultimate_actions[_TEAM_A_FIRST_SLOT])
        == use_ultimate
    )
    expected_cooldown = (
        int(combat.ULTIMATE_COOLDOWN_BY_CLASS[source_class_id])
        if use_ultimate
        else initial_source_cooldown - 1
    )
    assert int(next_state.ultimate_cooldowns[_TEAM_A_FIRST_SLOT]) == expected_cooldown
    _assert_statuses_are_clear(next_state, _TEAM_A_FIRST_SLOT)
    _assert_canonical_dead_action_mask(next_action_mask, _TEAM_A_FIRST_SLOT)

    if effect_kind in {"warrior-charge", "hunter-basic", "rogue-poison"}:
        expected_source_damage = (
            combat_facts.source_modified_damage_output_by_source[_TEAM_A_FIRST_SLOT]
            * combat_facts.recipient_damage_modifier_by_source[_TEAM_A_FIRST_SLOT]
        )
        resolved_recipient_damage = combat_facts.total_effective_damage_by_recipient[
            _TEAM_B_FIRST_SLOT
        ]
        assert float(expected_source_damage) > 0.0
        assert bool(jnp.isclose(resolved_recipient_damage, expected_source_damage))
        assert bool(
            jnp.isclose(
                next_state.current_health[_TEAM_B_FIRST_SLOT],
                state.current_health[_TEAM_B_FIRST_SLOT] - resolved_recipient_damage,
            )
        )

    if effect_kind == "mage-burst":
        assert bool(
            combat_facts.mage_burst_damage_amplification_is_applied_by_source[
                _TEAM_A_FIRST_SLOT
            ]
        )
    elif effect_kind == "warrior-charge":
        target_position = state.agent_positions[_TEAM_B_FIRST_SLOT]
        expected_charge_x = (
            target_position[0]
            - config.agent_profile.agent_radii[_TEAM_A_FIRST_SLOT]
            - config.agent_profile.agent_radii[_TEAM_B_FIRST_SLOT]
        )
        assert bool(
            jnp.isclose(
                next_state.agent_positions[_TEAM_A_FIRST_SLOT, 0],
                expected_charge_x,
            )
        )
        assert bool(
            jnp.isclose(
                next_state.agent_positions[_TEAM_A_FIRST_SLOT, 1],
                source_position[1]
                + config.agent_profile.base_movement_speeds[_TEAM_A_FIRST_SLOT]
                * config.ordinary_movement_distance_scale,
            )
        )
        assert bool(
            combat_facts.slow_is_applied_by_source_and_channel[
                _TEAM_A_FIRST_SLOT, SLOW_CHANNEL_WARRIOR_CHARGE
            ]
        )
        assert bool(
            combat_facts.stun_is_applied_by_source_and_channel[
                _TEAM_A_FIRST_SLOT, STUN_CHANNEL_WARRIOR_CHARGE
            ]
        )
        assert (
            next_state.stun_durations[_TEAM_B_FIRST_SLOT, STUN_CHANNEL_WARRIOR_CHARGE]
            == combat.WARRIOR_CHARGE_STUN_DURATION_TICKS
        )
    elif effect_kind == "hunter-basic":
        assert bool(
            combat_facts.basic_effect_is_activated_by_source[_TEAM_A_FIRST_SLOT]
        )
        assert bool(
            combat_facts.slow_is_applied_by_source_and_channel[
                _TEAM_A_FIRST_SLOT, SLOW_CHANNEL_HUNTER_BASIC
            ]
        )
    elif effect_kind == "hunter-trap":
        assert bool(
            combat_facts.stun_is_applied_by_source_and_channel[
                _TEAM_A_FIRST_SLOT, STUN_CHANNEL_HUNTER_TRAP
            ]
        )
        assert (
            next_state.stun_durations[_TEAM_B_FIRST_SLOT, STUN_CHANNEL_HUNTER_TRAP]
            == combat.HUNTER_TRAP_STUN_DURATION_TICKS
        )
    elif effect_kind == "rogue-poison":
        assert bool(
            combat_facts.rogue_poison_anti_heal_is_applied_by_source[_TEAM_A_FIRST_SLOT]
        )
        assert (
            next_state.rogue_poison_anti_heal_durations[_TEAM_B_FIRST_SLOT]
            == combat.ROGUE_POISON_ANTI_HEAL_DURATION_TICKS
        )
    else:
        assert effect_kind == "priest-freedom"
        assert bool(
            combat_facts.priest_blessing_of_freedom_is_applied_by_source[
                _TEAM_A_FIRST_SLOT
            ]
        )
        regeneration_facts = info.transition_facts.regeneration_facts
        actual_regeneration = (
            regeneration_facts.actual_health_regenerated_this_step_by_agent[
                _TEAM_A_SECOND_SLOT
            ]
        )
        assert actual_regeneration == 4.0
        assert (
            next_state.current_health[_TEAM_A_SECOND_SLOT]
            == state.current_health[_TEAM_A_SECOND_SLOT]
            + combat.BASIC_HEALING_BY_CLASS[PRIEST_CLASS_ID]
            + actual_regeneration
        )


def test_lethally_hit_mover_remains_a_physical_body_through_collision() -> None:
    """Prove choosing-state liveness governs collision before successor death."""
    positions = (
        jnp.zeros((MAX_AGENT_SLOTS, ENVIRONMENT_DIMENSIONS), dtype=jnp.float32)
        .at[_TEAM_A_FIRST_SLOT]
        .set(jnp.asarray((3.0, 4.0), dtype=jnp.float32))
        .at[_TEAM_A_SECOND_SLOT]
        .set(jnp.asarray((4.0, 4.0), dtype=jnp.float32))
        .at[_TEAM_B_FIRST_SLOT]
        .set(jnp.asarray((10.0, 4.0), dtype=jnp.float32))
    )
    config, state, _, _ = _scenario(
        (_TEAM_A_FIRST_SLOT, HUNTER_CLASS_ID),
        (_TEAM_A_SECOND_SLOT, HUNTER_CLASS_ID),
        (_TEAM_B_FIRST_SLOT, ROGUE_CLASS_ID),
        team_sizes=(2, 1),
        positions=positions,
        ordinary_movement_distance_scale=0.5,
    )
    state = state._replace(
        current_health=state.current_health.at[_TEAM_A_FIRST_SLOT].set(1.0)
    )
    action = _joint_action(
        (
            _TEAM_A_FIRST_SLOT,
            MOVE_EAST,
            _TARGET_NONE,
            0,
        ),
        (
            _TEAM_B_FIRST_SLOT,
            MOVE_STAY,
            _FIRST_ENEMY_TARGET,
            0,
        ),
    )

    next_state, *_, info = _take_step(config, state, action)
    center_distance = cast(
        Array,
        jnp.linalg.norm(
            next_state.agent_positions[_TEAM_A_FIRST_SLOT]
            - next_state.agent_positions[_TEAM_A_SECOND_SLOT]
        ),
    )
    minimum_distance = (
        config.agent_profile.agent_radii[_TEAM_A_FIRST_SLOT]
        + config.agent_profile.agent_radii[_TEAM_A_SECOND_SLOT]
    )

    assert bool(
        info.transition_facts.death_facts.is_newly_dead_by_recipient[_TEAM_A_FIRST_SLOT]
    )
    assert not bool(next_state.alive_mask[_TEAM_A_FIRST_SLOT])
    assert bool(center_distance >= minimum_distance - 1e-5)
    assert bool(
        next_state.agent_positions[_TEAM_A_SECOND_SLOT, 0]
        > state.agent_positions[_TEAM_A_SECOND_SLOT, 0]
    )
    physical_facts = info.transition_facts.physical_facts
    assert bool(jnp.all(physical_facts.charge_phase_displacement_by_agent == 0.0))
    assert bool(
        jnp.allclose(
            physical_facts.ordinary_movement_phase_displacement_by_agent,
            next_state.agent_positions - state.agent_positions,
        )
    )


def test_dying_mage_aura_still_amplifies_an_ally_this_transition() -> None:
    """Prove successor death cannot erase a pre-state outgoing aura."""
    config, state, _, _ = _scenario(
        (_TEAM_A_FIRST_SLOT, MAGE_CLASS_ID),
        (_TEAM_A_SECOND_SLOT, HUNTER_CLASS_ID),
        (_TEAM_B_FIRST_SLOT, ROGUE_CLASS_ID),
        (_TEAM_B_SECOND_SLOT, HUNTER_CLASS_ID),
        team_sizes=(2, 2),
    )
    state = state._replace(
        current_health=state.current_health.at[_TEAM_A_FIRST_SLOT]
        .set(1.0)
        .at[_TEAM_B_SECOND_SLOT]
        .set(1.0)
    )
    action = _joint_action(
        (
            _TEAM_A_SECOND_SLOT,
            MOVE_STAY,
            _SECOND_ENEMY_TARGET,
            0,
        ),
        (
            _TEAM_B_FIRST_SLOT,
            MOVE_STAY,
            _FIRST_ENEMY_TARGET,
            0,
        ),
    )

    next_state, *_, info = _take_step(config, state, action)
    combat_facts = info.transition_facts.combat_transition_facts

    assert not bool(next_state.alive_mask[_TEAM_A_FIRST_SLOT])
    assert not bool(next_state.alive_mask[_TEAM_B_SECOND_SLOT])
    assert bool(
        jnp.isclose(
            combat_facts.source_modified_damage_output_by_source[_TEAM_A_SECOND_SLOT],
            combat.BASIC_DAMAGE_BY_CLASS[HUNTER_CLASS_ID]
            * combat.MAGE_DAMAGE_AMPLIFICATION_AURA_MULTIPLIER,
        )
    )


def test_dying_warrior_aura_still_mitigates_damage_to_an_ally() -> None:
    """Prove successor death cannot erase a pre-state incoming aura."""
    config, state, _, _ = _scenario(
        (_TEAM_A_FIRST_SLOT, WARRIOR_CLASS_ID),
        (_TEAM_A_SECOND_SLOT, HUNTER_CLASS_ID),
        (_TEAM_B_FIRST_SLOT, ROGUE_CLASS_ID),
        (_TEAM_B_SECOND_SLOT, ROGUE_CLASS_ID),
        team_sizes=(2, 2),
    )
    state = state._replace(
        current_health=state.current_health.at[_TEAM_A_FIRST_SLOT]
        .set(1.0)
        .at[_TEAM_A_SECOND_SLOT]
        .set(11.0)
    )
    action = _joint_action(
        (
            _TEAM_B_FIRST_SLOT,
            MOVE_STAY,
            _FIRST_ENEMY_TARGET,
            0,
        ),
        (
            _TEAM_B_SECOND_SLOT,
            MOVE_STAY,
            _SECOND_ENEMY_TARGET,
            0,
        ),
    )

    next_state, *_, info = _take_step(config, state, action)
    combat_facts = info.transition_facts.combat_transition_facts
    expected_damage = (
        combat.BASIC_DAMAGE_BY_CLASS[ROGUE_CLASS_ID]
        * combat.WARRIOR_DAMAGE_MITIGATION_AURA_MULTIPLIER
    )

    assert not bool(next_state.alive_mask[_TEAM_A_FIRST_SLOT])
    assert bool(next_state.alive_mask[_TEAM_A_SECOND_SLOT])
    assert bool(
        jnp.isclose(
            combat_facts.recipient_damage_modifier_by_source[_TEAM_B_SECOND_SLOT],
            combat.WARRIOR_DAMAGE_MITIGATION_AURA_MULTIPLIER,
        )
    )
    assert bool(
        jnp.isclose(
            next_state.current_health[_TEAM_A_SECOND_SLOT],
            11.0 - expected_damage,
        )
    )


def test_new_death_clears_every_existing_and_fresh_transient_status() -> None:
    """Prove status application facts survive while corpse durations are cleared."""
    config, state, _, _ = _scenario(
        (_TEAM_A_FIRST_SLOT, ROGUE_CLASS_ID),
        (_TEAM_B_FIRST_SLOT, MAGE_CLASS_ID),
    )
    slow_durations = state.slow_durations.at[_TEAM_B_FIRST_SLOT].set(
        jnp.asarray(
            (
                combat.WARRIOR_CHARGE_SLOW_DURATION_TICKS,
                combat.HUNTER_BASIC_SLOW_DURATION_TICKS,
                combat.ROGUE_POISON_SLOW_DURATION_TICKS,
            ),
            dtype=jnp.int32,
        )
    )
    stun_durations = state.stun_durations.at[_TEAM_B_FIRST_SLOT].set(
        jnp.asarray(
            (
                combat.WARRIOR_CHARGE_STUN_DURATION_TICKS,
                combat.HUNTER_TRAP_STUN_DURATION_TICKS,
                combat.ROGUE_POISON_STUN_DURATION_TICKS,
            ),
            dtype=jnp.int32,
        )
    )
    state = state._replace(
        current_health=state.current_health.at[_TEAM_B_FIRST_SLOT].set(1.0),
        ultimate_cooldowns=state.ultimate_cooldowns.at[_TEAM_B_FIRST_SLOT].set(7),
        slow_durations=slow_durations,
        stun_durations=stun_durations,
        rogue_poison_anti_heal_durations=(
            state.rogue_poison_anti_heal_durations.at[_TEAM_B_FIRST_SLOT].set(
                combat.ROGUE_POISON_ANTI_HEAL_DURATION_TICKS
            )
        ),
        mage_burst_damage_amplification_durations=(
            state.mage_burst_damage_amplification_durations.at[_TEAM_B_FIRST_SLOT].set(
                combat.MAGE_BURST_DAMAGE_DURATION_TICKS
            )
        ),
        priest_blessing_of_freedom_slow_floor_durations=(
            state.priest_blessing_of_freedom_slow_floor_durations.at[
                _TEAM_B_FIRST_SLOT
            ].set(combat.PRIEST_HEAL_SPEED_FLOOR_DURATION_TICKS)
        ),
    )

    next_state, *_, info = _take_step(
        config,
        state,
        _joint_action(
            (
                _TEAM_A_FIRST_SLOT,
                MOVE_STAY,
                _FIRST_ENEMY_TARGET,
                1,
            )
        ),
    )
    combat_facts = info.transition_facts.combat_transition_facts

    assert bool(
        info.transition_facts.death_facts.is_newly_dead_by_recipient[_TEAM_B_FIRST_SLOT]
    )
    assert bool(
        combat_facts.slow_is_applied_by_source_and_channel[
            _TEAM_A_FIRST_SLOT, SLOW_CHANNEL_ROGUE_POISON
        ]
    )
    assert bool(
        combat_facts.stun_is_applied_by_source_and_channel[
            _TEAM_A_FIRST_SLOT, STUN_CHANNEL_ROGUE_POISON
        ]
    )
    assert bool(
        combat_facts.rogue_poison_anti_heal_is_applied_by_source[_TEAM_A_FIRST_SLOT]
    )
    _assert_statuses_are_clear(next_state, _TEAM_B_FIRST_SLOT)
    assert int(next_state.ultimate_cooldowns[_TEAM_B_FIRST_SLOT]) == 6


def test_dead_successor_exposes_coherent_observation_mask_reward_and_done() -> None:
    """Prove every public successor output agrees on immediate dead semantics."""
    config, state, _, _ = _scenario(
        (_TEAM_A_FIRST_SLOT, ROGUE_CLASS_ID),
        (_TEAM_B_FIRST_SLOT, HUNTER_CLASS_ID),
        obstacles=_clear_pillar_obstacles(),
    )
    state = state._replace(
        current_health=state.current_health.at[_TEAM_B_FIRST_SLOT].set(1.0)
    )
    target_position = state.agent_positions[_TEAM_B_FIRST_SLOT]
    action = _joint_action(
        (
            _TEAM_A_FIRST_SLOT,
            MOVE_STAY,
            _FIRST_ENEMY_TARGET,
            0,
        ),
        (
            _TEAM_B_FIRST_SLOT,
            MOVE_EAST,
            _TARGET_NONE,
            0,
        ),
    )

    (
        next_state,
        observation,
        reward,
        done_flags,
        next_mask,
        _,
    ) = _take_step(config, state, action)

    assert not bool(next_state.alive_mask[_TEAM_B_FIRST_SLOT])
    assert not bool(
        jnp.array_equal(
            next_state.agent_positions[_TEAM_B_FIRST_SLOT],
            target_position,
        )
    )
    assert observation.self_features[_TEAM_B_FIRST_SLOT, AGENT_FEATURE_ACTIVE] == 1.0
    assert observation.self_features[_TEAM_B_FIRST_SLOT, AGENT_FEATURE_ALIVE] == 0.0
    assert (
        observation.self_features[_TEAM_B_FIRST_SLOT, AGENT_FEATURE_CURRENT_HEALTH]
        == 0.0
    )
    assert not bool(jnp.any(observation.ally_visibility_mask[_TEAM_B_FIRST_SLOT]))
    assert not bool(jnp.any(observation.enemy_visibility_mask[_TEAM_B_FIRST_SLOT]))
    assert bool(jnp.all(observation.ally_unit_features[_TEAM_B_FIRST_SLOT] == 0.0))
    assert bool(jnp.all(observation.enemy_unit_features[_TEAM_B_FIRST_SLOT] == 0.0))
    assert not bool(observation.enemy_visibility_mask[_TEAM_A_FIRST_SLOT, 0])
    assert bool(jnp.all(observation.enemy_unit_features[_TEAM_A_FIRST_SLOT, 0] == 0.0))
    assert (
        int(next_state.previous_timestep_move_actions[_TEAM_B_FIRST_SLOT]) == MOVE_EAST
    )
    dead_candidate_previous_action_rows = (
        observation.previous_timestep_actions.enemy_previous_timestep_move_actions_one_hot[
            _TEAM_A_FIRST_SLOT, 0
        ],
        observation.previous_timestep_actions.enemy_previous_timestep_select_target_actions_one_hot[
            _TEAM_A_FIRST_SLOT, 0
        ],
        observation.previous_timestep_actions.enemy_previous_timestep_use_ultimate_actions_one_hot[
            _TEAM_A_FIRST_SLOT, 0
        ],
    )
    for previous_action_row in dead_candidate_previous_action_rows:
        assert bool(jnp.all(previous_action_row == 0.0))
    for previous_action_family in observation.previous_timestep_actions:
        assert bool(jnp.all(previous_action_family[_TEAM_B_FIRST_SLOT] == 0.0))
    assert bool(
        jnp.array_equal(
            observation.map_obstacle_features[_TEAM_B_FIRST_SLOT],
            config.obstacles,
        )
    )
    assert bool(jnp.any(observation.map_obstacle_features[_TEAM_B_FIRST_SLOT] != 0.0))
    assert bool(jnp.any(observation.context_features[_TEAM_B_FIRST_SLOT] != 0.0))
    _assert_canonical_dead_action_mask(next_mask, _TEAM_B_FIRST_SLOT)
    assert bool(jnp.all(reward.rewards == 0.0))
    assert not bool(done_flags.terminated)
    assert not bool(done_flags.truncated)


def test_corpse_transition_ticks_cooldown_replaces_history_and_never_dies_again() -> (
    None
):
    """Prove one death event is followed by canonical inert corpse transitions."""
    config, state, _, _ = _scenario(
        (_TEAM_A_FIRST_SLOT, ROGUE_CLASS_ID),
        (_TEAM_B_FIRST_SLOT, HUNTER_CLASS_ID),
    )
    state = state._replace(
        current_health=state.current_health.at[_TEAM_B_FIRST_SLOT].set(1.0),
        ultimate_cooldowns=state.ultimate_cooldowns.at[_TEAM_B_FIRST_SLOT].set(5),
    )
    lethal_action = _joint_action(
        (
            _TEAM_A_FIRST_SLOT,
            MOVE_STAY,
            _FIRST_ENEMY_TARGET,
            0,
        ),
        (
            _TEAM_B_FIRST_SLOT,
            MOVE_EAST,
            _FIRST_ENEMY_TARGET,
            0,
        ),
    )
    dead_state, _, _, _, dead_mask, lethal_info = _take_step(
        config,
        state,
        lethal_action,
    )
    corpse_position = dead_state.agent_positions[_TEAM_B_FIRST_SLOT]

    assert bool(
        lethal_info.transition_facts.death_facts.is_newly_dead_by_recipient[
            _TEAM_B_FIRST_SLOT
        ]
    )
    assert int(dead_state.ultimate_cooldowns[_TEAM_B_FIRST_SLOT]) == 4
    assert (
        int(dead_state.previous_timestep_move_actions[_TEAM_B_FIRST_SLOT]) == MOVE_EAST
    )

    invalid_corpse_submission = _joint_action(
        (
            _TEAM_B_FIRST_SLOT,
            MOVE_EAST,
            _FIRST_ENEMY_TARGET,
            1,
        )
    )
    (
        later_state,
        _,
        _,
        _,
        later_mask,
        later_info,
    ) = _take_step(
        config,
        dead_state,
        invalid_corpse_submission,
        action_mask=dead_mask,
    )
    acceptance = later_info.transition_facts.action_acceptance_facts
    death_facts = later_info.transition_facts.death_facts

    assert not bool(jnp.any(death_facts.is_newly_dead_by_recipient))
    assert not bool(jnp.any(death_facts.contributed_to_new_death_by_source))
    assert bool(jnp.all(death_facts.attributed_death_damage_by_source == 0.0))
    assert bool(
        acceptance.in_domain_move_action_is_rejected_by_actor[_TEAM_B_FIRST_SLOT]
    )
    assert bool(
        acceptance.in_domain_combat_action_pair_is_rejected_by_actor[_TEAM_B_FIRST_SLOT]
    )
    assert int(acceptance.accepted_joint_action.move[_TEAM_B_FIRST_SLOT]) == MOVE_STAY
    assert (
        int(acceptance.accepted_joint_action.select_target[_TEAM_B_FIRST_SLOT])
        == _TARGET_NONE
    )
    assert int(acceptance.accepted_joint_action.use_ultimate[_TEAM_B_FIRST_SLOT]) == 0
    assert not bool(later_state.alive_mask[_TEAM_B_FIRST_SLOT])
    assert float(later_state.current_health[_TEAM_B_FIRST_SLOT]) == 0.0
    assert int(later_state.ultimate_cooldowns[_TEAM_B_FIRST_SLOT]) == 3
    assert bool(
        jnp.array_equal(
            later_state.agent_positions[_TEAM_B_FIRST_SLOT],
            corpse_position,
        )
    )
    assert (
        int(later_state.previous_timestep_move_actions[_TEAM_B_FIRST_SLOT]) == MOVE_STAY
    )
    assert (
        int(later_state.previous_timestep_select_target_actions[_TEAM_B_FIRST_SLOT])
        == _TARGET_NONE
    )
    assert (
        int(later_state.previous_timestep_use_ultimate_actions[_TEAM_B_FIRST_SLOT]) == 0
    )
    _assert_statuses_are_clear(later_state, _TEAM_B_FIRST_SLOT)
    _assert_canonical_dead_action_mask(later_mask, _TEAM_B_FIRST_SLOT)


def test_compiled_mutual_death_matches_the_complete_eager_public_transition() -> None:
    """Prove eager and compiled mutual kills agree on every public leaf."""
    config, state, action_mask, _ = _scenario(
        (_TEAM_A_FIRST_SLOT, ROGUE_CLASS_ID),
        (_TEAM_B_FIRST_SLOT, ROGUE_CLASS_ID),
    )
    state = state._replace(
        current_health=state.current_health.at[_TEAM_A_FIRST_SLOT]
        .set(1.0)
        .at[_TEAM_B_FIRST_SLOT]
        .set(1.0)
    )
    action_mask = _current_action_mask(config, state)
    action = _joint_action(
        (
            _TEAM_A_FIRST_SLOT,
            MOVE_EAST,
            _FIRST_ENEMY_TARGET,
            0,
        ),
        (
            _TEAM_B_FIRST_SLOT,
            MOVE_EAST,
            _FIRST_ENEMY_TARGET,
            0,
        ),
    )
    key = jax.random.key(31)

    eager = step(config, state, action_mask, action, key)
    compiled = cast(
        tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info],
        jax.jit(step)(config, state, action_mask, action, key),
    )

    _assert_tree_equal(eager, compiled)
    assert bool(
        jnp.array_equal(
            compiled[-1].transition_facts.death_facts.is_newly_dead_by_recipient,
            _slot_mask(_TEAM_A_FIRST_SLOT, _TEAM_B_FIRST_SLOT),
        )
    )


def test_shared_config_vmap_chooses_from_each_successor_mask() -> None:
    """Prove a batched policy consumes living, new-corpse, and corpse masks."""
    config, base_state, _, _ = _scenario(
        (_TEAM_A_FIRST_SLOT, ROGUE_CLASS_ID),
        (_TEAM_B_FIRST_SLOT, HUNTER_CLASS_ID),
    )
    surviving_state = base_state
    lethal_state = base_state._replace(
        current_health=base_state.current_health.at[_TEAM_B_FIRST_SLOT].set(1.0)
    )
    corpse_state = base_state._replace(
        alive_mask=base_state.alive_mask.at[_TEAM_B_FIRST_SLOT].set(False),
        current_health=base_state.current_health.at[_TEAM_B_FIRST_SLOT].set(0.0),
    )
    states = jax.tree.map(
        lambda *leaves: jnp.stack(leaves),
        surviving_state,
        lethal_state,
        corpse_state,
    )
    masks = jax.tree.map(
        lambda *leaves: jnp.stack(leaves),
        _current_action_mask(config, surviving_state),
        _current_action_mask(config, lethal_state),
        _current_action_mask(config, corpse_state),
    )
    attacking_action = _joint_action(
        (
            _TEAM_A_FIRST_SLOT,
            MOVE_STAY,
            _FIRST_ENEMY_TARGET,
            0,
        )
    )
    actions = jax.tree.map(
        lambda *leaves: jnp.stack(leaves),
        attacking_action,
        attacking_action,
        attacking_action,
    )
    desired_second_action = _joint_action(
        (
            _TEAM_A_FIRST_SLOT,
            MOVE_STAY,
            _FIRST_ENEMY_TARGET,
            0,
        ),
        (
            _TEAM_B_FIRST_SLOT,
            MOVE_EAST,
            _FIRST_ENEMY_TARGET,
            1,
        ),
    )
    keys = jax.random.split(jax.random.key(41), 3)
    second_keys = jax.random.split(jax.random.key(42), 3)

    def batched_two_step_trajectory(
        state: EnvState,
        action_mask: ActionMask,
        action: Action,
        key: Array,
        second_key: Array,
    ) -> tuple[
        tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info],
        tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info],
    ]:
        """Choose the second action only after observing the first successor mask."""
        first_outputs = step(
            config,
            state,
            action_mask,
            action,
            key,
        )
        first_state, _, _, _, first_mask, _ = first_outputs
        second_action = _choose_desired_action_from_mask(
            first_mask,
            desired_second_action,
        )
        second_outputs = step(
            config,
            first_state,
            first_mask,
            second_action,
            second_key,
        )
        return first_outputs, second_outputs

    first_outputs, second_outputs = jax.vmap(batched_two_step_trajectory)(
        states,
        masks,
        actions,
        keys,
        second_keys,
    )
    first_states = first_outputs[0]
    first_death_facts = first_outputs[-1].transition_facts.death_facts
    second_states = second_outputs[0]
    second_acceptance = second_outputs[-1].transition_facts.action_acceptance_facts
    second_death_facts = second_outputs[-1].transition_facts.death_facts

    assert bool(
        jnp.array_equal(
            first_states.alive_mask[:, _TEAM_B_FIRST_SLOT],
            jnp.asarray((True, False, False), dtype=jnp.bool_),
        )
    )
    assert bool(
        jnp.array_equal(
            first_death_facts.is_newly_dead_by_recipient[:, _TEAM_B_FIRST_SLOT],
            jnp.asarray((False, True, False), dtype=jnp.bool_),
        )
    )
    assert bool(
        jnp.array_equal(
            first_death_facts.contributed_to_new_death_by_source[:, _TEAM_A_FIRST_SLOT],
            jnp.asarray((False, True, False), dtype=jnp.bool_),
        )
    )
    assert bool(
        jnp.array_equal(
            second_states.alive_mask[:, _TEAM_B_FIRST_SLOT],
            jnp.asarray((True, False, False), dtype=jnp.bool_),
        )
    )
    assert not bool(jnp.any(second_death_facts.is_newly_dead_by_recipient))
    assert bool(
        jnp.array_equal(
            second_acceptance.accepted_joint_action.move[:, _TEAM_B_FIRST_SLOT],
            jnp.asarray((MOVE_EAST, MOVE_STAY, MOVE_STAY), dtype=jnp.int32),
        )
    )
    assert bool(
        jnp.array_equal(
            second_acceptance.accepted_joint_action.select_target[
                :, _TEAM_B_FIRST_SLOT
            ],
            jnp.asarray(
                (_FIRST_ENEMY_TARGET, _TARGET_NONE, _TARGET_NONE),
                dtype=jnp.int32,
            ),
        )
    )
    assert bool(
        jnp.array_equal(
            second_acceptance.accepted_joint_action.use_ultimate[:, _TEAM_B_FIRST_SLOT],
            jnp.asarray((1, 0, 0), dtype=jnp.int32),
        )
    )
    for leaf in jax.tree_util.tree_leaves(first_death_facts):
        assert leaf.shape == (3, MAX_AGENT_SLOTS)


def test_compiled_scan_applies_observes_then_chooses_from_corpse_semantics() -> None:
    """Prove one masked policy spans nonlethal, lethal, and corpse epochs."""
    config, state, action_mask, _ = _scenario(
        (_TEAM_A_FIRST_SLOT, ROGUE_CLASS_ID),
        (_TEAM_B_FIRST_SLOT, HUNTER_CLASS_ID),
    )
    attacker_damage = combat.BASIC_DAMAGE_BY_CLASS[ROGUE_CLASS_ID]
    state = state._replace(
        current_health=state.current_health.at[_TEAM_B_FIRST_SLOT].set(
            attacker_damage + 1.0
        ),
        ultimate_cooldowns=state.ultimate_cooldowns.at[_TEAM_B_FIRST_SLOT].set(5),
    )
    action_mask = _current_action_mask(config, state)
    desired_action = _joint_action(
        (
            _TEAM_A_FIRST_SLOT,
            MOVE_STAY,
            _FIRST_ENEMY_TARGET,
            0,
        ),
        (
            _TEAM_B_FIRST_SLOT,
            MOVE_EAST,
            _FIRST_ENEMY_TARGET,
            0,
        ),
    )
    keys = jax.random.split(jax.random.key(51), 3)

    def rollout(
        initial_state: EnvState,
        initial_mask: ActionMask,
        rollout_keys: Array,
    ) -> tuple[
        tuple[EnvState, ActionMask],
        tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info],
    ]:
        """Run the public step API while carrying each successor action mask."""

        def scan_body(
            carry: tuple[EnvState, ActionMask],
            key: Array,
        ) -> tuple[
            tuple[EnvState, ActionMask],
            tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info],
        ]:
            current_state, current_mask = carry
            chosen_action = _choose_desired_action_from_mask(
                current_mask,
                desired_action,
            )
            outputs = step(
                config,
                current_state,
                current_mask,
                chosen_action,
                key,
            )
            next_state, _, _, _, next_mask, _ = outputs
            return (next_state, next_mask), outputs

        return jax.lax.scan(
            scan_body,
            (initial_state, initial_mask),
            rollout_keys,
        )

    (_, _), scanned_outputs = cast(
        tuple[
            tuple[EnvState, ActionMask],
            tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info],
        ],
        jax.jit(rollout)(state, action_mask, keys),
    )
    (
        scanned_states,
        scanned_observations,
        _,
        _,
        scanned_masks,
        scanned_infos,
    ) = scanned_outputs
    scanned_death_facts = scanned_infos.transition_facts.death_facts

    assert bool(
        jnp.array_equal(
            scanned_states.alive_mask[:, _TEAM_B_FIRST_SLOT],
            jnp.asarray((True, False, False), dtype=jnp.bool_),
        )
    )
    assert bool(
        jnp.array_equal(
            scanned_death_facts.is_newly_dead_by_recipient[:, _TEAM_B_FIRST_SLOT],
            jnp.asarray((False, True, False), dtype=jnp.bool_),
        )
    )
    assert bool(
        jnp.array_equal(
            scanned_death_facts.contributed_to_new_death_by_source[
                :, _TEAM_A_FIRST_SLOT
            ],
            jnp.asarray((False, True, False), dtype=jnp.bool_),
        )
    )
    assert bool(
        jnp.array_equal(
            scanned_states.previous_timestep_move_actions[:, _TEAM_B_FIRST_SLOT],
            jnp.asarray((MOVE_EAST, MOVE_EAST, MOVE_STAY), dtype=jnp.int32),
        )
    )
    assert bool(
        jnp.array_equal(
            scanned_states.ultimate_cooldowns[:, _TEAM_B_FIRST_SLOT],
            jnp.asarray((4, 3, 2), dtype=jnp.int32),
        )
    )
    assert bool(
        jnp.array_equal(
            scanned_observations.self_features[
                :, _TEAM_B_FIRST_SLOT, AGENT_FEATURE_ALIVE
            ],
            jnp.asarray((1.0, 0.0, 0.0), dtype=jnp.float32),
        )
    )
    accepted_actions = scanned_infos.transition_facts.action_acceptance_facts
    assert bool(
        jnp.array_equal(
            accepted_actions.accepted_joint_action.select_target[:, _TEAM_A_FIRST_SLOT],
            jnp.asarray(
                (_FIRST_ENEMY_TARGET, _FIRST_ENEMY_TARGET, _TARGET_NONE),
                dtype=jnp.int32,
            ),
        )
    )
    for time_index in (1, 2):
        time_mask = ActionMask(
            move_mask=scanned_masks.move_mask[time_index],
            select_target_mask=scanned_masks.select_target_mask[time_index],
            use_ultimate_mask=scanned_masks.use_ultimate_mask[time_index],
            select_target_use_ultimate_joint_mask=(
                scanned_masks.select_target_use_ultimate_joint_mask[time_index]
            ),
        )
        _assert_canonical_dead_action_mask(
            time_mask,
            _TEAM_B_FIRST_SLOT,
        )
