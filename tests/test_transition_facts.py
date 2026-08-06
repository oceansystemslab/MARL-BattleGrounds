"""Authoritative action, combat, death, and spawn-shield facts for Milestone 6."""
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
    ENVIRONMENT_DIMENSIONS,
    HUNTER_CLASS_ID,
    MAGE_CLASS_ID,
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    MAX_OBSTACLE_SLOTS,
    MOVE_EAST,
    MOVE_STAY,
    NEUTRAL_CLASS_ID,
    NUM_MOVE_ACTIONS,
    NUM_SLOW_CHANNELS,
    NUM_STUN_CHANNELS,
    NUM_TARGET_ACTIONS,
    NUM_ULTIMATE_ACTIONS,
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
    ActionAcceptanceFacts,
    ActionMask,
    CombatTransitionFacts,
    DeathTransitionFacts,
    DoneFlags,
    EnvConfig,
    EnvState,
    Info,
    Observation,
    RespawnTransitionFacts,
    Reward,
    SpawnShieldTransitionFacts,
    TransitionFacts,
)

_TEAM_A_FIRST_SLOT = 0
_TEAM_B_FIRST_SLOT = MAX_AGENTS_PER_TEAM
_TARGET_NONE = 0
_SELF_TARGET = 1
_FIRST_ENEMY_TARGET = 1 + MAX_AGENTS_PER_TEAM
_TRANSITION_FACT_LEAF_COUNT = 35
_TRANSITION_FACT_RAW_BYTES = 847


def _empty_obstacles() -> Array:
    """Return an inactive fixed-size obstacle table."""
    return jnp.zeros((MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES), dtype=jnp.float32)


def _requested_roster(
    team_sizes: tuple[int, int],
    *class_rows: tuple[int, int],
) -> Array:
    """Return a padded active roster with selected class overrides."""
    roster = jnp.full((MAX_AGENT_SLOTS,), NEUTRAL_CLASS_ID, dtype=jnp.int32)
    roster = roster.at[: team_sizes[0]].set(HUNTER_CLASS_ID)
    roster = roster.at[MAX_AGENTS_PER_TEAM : MAX_AGENTS_PER_TEAM + team_sizes[1]].set(
        HUNTER_CLASS_ID
    )
    for slot, class_id in class_rows:
        roster = roster.at[slot].set(class_id)
    return roster


def _default_positions(team_sizes: tuple[int, int]) -> Array:
    """Place active team blocks on clear, non-overlapping vertical lines."""
    positions = jnp.zeros((MAX_AGENT_SLOTS, ENVIRONMENT_DIMENSIONS), dtype=jnp.float32)
    for local_slot in range(team_sizes[0]):
        positions = positions.at[local_slot].set(
            jnp.asarray((2.0, 2.0 + 1.5 * local_slot), dtype=jnp.float32)
        )
    for local_slot in range(team_sizes[1]):
        positions = positions.at[MAX_AGENTS_PER_TEAM + local_slot].set(
            jnp.asarray((12.0, 2.0 + 1.5 * local_slot), dtype=jnp.float32)
        )
    return positions


def _scenario(
    *class_rows: tuple[int, int],
    team_sizes: tuple[int, int] = (1, 1),
    positions: Array | None = None,
) -> tuple[EnvConfig, EnvState, ActionMask]:
    """Build one deterministic, stationary, fully observable combat scenario."""
    profile = resolve_agent_profile(
        _requested_roster(team_sizes, *class_rows),
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
    initial_positions = (
        _default_positions(team_sizes) if positions is None else positions
    )
    config = EnvConfig(
        max_steps=100,
        map_width=20.0,
        map_height=12.0,
        obstacles=_empty_obstacles(),
        agent_profile=profile,
        ordinary_movement_distance_scale=1.0,
        team_spawn_pad_positions=initial_positions.reshape(
            (2, MAX_AGENTS_PER_TEAM, ENVIRONMENT_DIMENSIONS)
        ),
        spawn_shield_duration_steps=3,
        spawn_shield_movement_speed=2.0,
        team_respawn_wave_period_step_count=jnp.asarray((5, 5), dtype=jnp.int32),
    )
    state, _, action_mask, _ = reset(config, jax.random.key(1))
    return config, state, action_mask


def _joint_action(*rows: tuple[int, int, int, int]) -> Action:
    """Return a canonical joint action with selected actor overrides.

    Each row is ``(actor_slot, move, target, use_ultimate)``.
    """
    move = jnp.full((MAX_AGENT_SLOTS,), MOVE_STAY, dtype=jnp.int32)
    select_target = jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32)
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


def _take_step(
    config: EnvConfig,
    state: EnvState,
    action_mask: ActionMask,
    action: Action,
) -> tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info]:
    """Advance one deterministic transition with the supplied choosing mask."""
    return step(
        config,
        state,
        action_mask,
        action,
        jax.random.key(2),
    )


def _assert_array_equal(left: object, right: object) -> None:
    """Assert equality for two identically structured JAX PyTrees."""
    assert jax.tree_util.tree_structure(left) == jax.tree_util.tree_structure(right)
    for left_leaf, right_leaf in zip(
        jax.tree_util.tree_leaves(left),
        jax.tree_util.tree_leaves(right),
        strict=True,
    ):
        assert bool(jnp.array_equal(left_leaf, right_leaf))


def _single_source_mask(source_slot: int) -> Array:
    """Return a fixed-slot boolean mask with exactly one source selected."""
    return jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.bool_).at[source_slot].set(True)


def _assert_canonical_empty_combat_facts(facts: CombatTransitionFacts) -> None:
    """Assert the reset/neutral canonical values for every combat fact family."""
    boolean_vectors = (
        facts.basic_effect_is_activated_by_source,
        facts.ultimate_effect_is_activated_by_source,
        facts.combat_effect_has_recipient_by_source,
        facts.rogue_poison_anti_heal_is_applied_by_source,
        facts.mage_burst_damage_amplification_is_applied_by_source,
        facts.priest_blessing_of_freedom_is_applied_by_source,
    )
    for vector in boolean_vectors:
        assert vector.shape == (MAX_AGENT_SLOTS,)
        assert vector.dtype == jnp.bool_
        assert not bool(jnp.any(vector))

    assert facts.slow_is_applied_by_source_and_channel.shape == (
        MAX_AGENT_SLOTS,
        NUM_SLOW_CHANNELS,
    )
    assert facts.stun_is_applied_by_source_and_channel.shape == (
        MAX_AGENT_SLOTS,
        NUM_STUN_CHANNELS,
    )
    assert facts.slow_is_applied_by_source_and_channel.dtype == jnp.bool_
    assert facts.stun_is_applied_by_source_and_channel.dtype == jnp.bool_
    assert not bool(jnp.any(facts.slow_is_applied_by_source_and_channel))
    assert not bool(jnp.any(facts.stun_is_applied_by_source_and_channel))

    assert facts.combat_effect_recipient_global_slot_by_source.shape == (
        MAX_AGENT_SLOTS,
    )
    assert facts.combat_effect_recipient_global_slot_by_source.dtype == jnp.int32
    assert bool(jnp.all(facts.combat_effect_recipient_global_slot_by_source == -1))

    float_vectors = (
        facts.raw_damage_output_by_source,
        facts.source_modified_damage_output_by_source,
        facts.recipient_damage_modifier_by_source,
        facts.total_effective_damage_by_recipient,
        facts.raw_healing_output_by_source,
        facts.source_modified_healing_output_by_source,
        facts.recipient_healing_modifier_by_source,
        facts.total_effective_healing_by_recipient,
    )
    for vector in float_vectors:
        assert vector.shape == (MAX_AGENT_SLOTS,)
        assert vector.dtype == jnp.float32
        assert bool(jnp.all(vector == 0.0))


def _assert_canonical_empty_death_facts(facts: DeathTransitionFacts) -> None:
    """Assert the reset/neutral canonical values for every death fact leaf."""
    boolean_vectors = (
        facts.is_newly_dead_by_recipient,
        facts.contributed_to_new_death_by_source,
    )
    for vector in boolean_vectors:
        assert vector.shape == (MAX_AGENT_SLOTS,)
        assert vector.dtype == jnp.bool_
        assert not bool(jnp.any(vector))

    assert facts.attributed_death_damage_by_source.shape == (MAX_AGENT_SLOTS,)
    assert facts.attributed_death_damage_by_source.dtype == jnp.float32
    assert bool(jnp.all(facts.attributed_death_damage_by_source == 0.0))


def _assert_canonical_empty_spawn_shield_facts(
    facts: SpawnShieldTransitionFacts,
) -> None:
    """Assert canonical shape, dtype, and neutrality for spawn-shield facts."""
    for vector in (
        facts.was_active_at_transition_start_by_agent,
        facts.expired_at_transition_end_by_agent,
    ):
        assert vector.shape == (MAX_AGENT_SLOTS,)
        assert vector.dtype == jnp.bool_
        assert not bool(jnp.any(vector))


def test_reset_and_real_neutral_step_share_the_exact_static_fact_schema() -> None:
    """Prove canonical reset facts, real-step identity, and the payload budget."""
    config, state, action_mask = _scenario(
        (_TEAM_A_FIRST_SLOT, HUNTER_CLASS_ID),
        (_TEAM_B_FIRST_SLOT, HUNTER_CLASS_ID),
    )
    _, _, _, reset_info = reset(config, jax.random.key(3))
    reset_facts = reset_info.transition_facts

    assert ActionAcceptanceFacts._fields == (
        "submitted_joint_action",
        "accepted_joint_action",
        "submitted_action_tuple_is_out_of_domain_by_actor",
        "in_domain_move_action_is_rejected_by_actor",
        "in_domain_combat_action_pair_is_rejected_by_actor",
    )
    assert CombatTransitionFacts._fields == (
        "basic_effect_is_activated_by_source",
        "ultimate_effect_is_activated_by_source",
        "combat_effect_has_recipient_by_source",
        "combat_effect_recipient_global_slot_by_source",
        "raw_damage_output_by_source",
        "source_modified_damage_output_by_source",
        "recipient_damage_modifier_by_source",
        "total_effective_damage_by_recipient",
        "raw_healing_output_by_source",
        "source_modified_healing_output_by_source",
        "recipient_healing_modifier_by_source",
        "total_effective_healing_by_recipient",
        "slow_is_applied_by_source_and_channel",
        "stun_is_applied_by_source_and_channel",
        "rogue_poison_anti_heal_is_applied_by_source",
        "mage_burst_damage_amplification_is_applied_by_source",
        "priest_blessing_of_freedom_is_applied_by_source",
    )
    assert DeathTransitionFacts._fields == (
        "is_newly_dead_by_recipient",
        "contributed_to_new_death_by_source",
        "attributed_death_damage_by_source",
    )
    assert SpawnShieldTransitionFacts._fields == (
        "was_active_at_transition_start_by_agent",
        "expired_at_transition_end_by_agent",
    )
    assert RespawnTransitionFacts._fields == (
        "respawn_wave_occurred_this_transition_by_team",
        "was_respawned_this_transition_by_agent",
    )
    assert TransitionFacts._fields == (
        "has_transition",
        "transition_start_step_count",
        "action_acceptance_facts",
        "combat_transition_facts",
        "death_facts",
        "spawn_shield_facts",
        "respawn_facts",
    )
    assert Info._fields == ("transition_facts",)

    assert reset_facts.has_transition.shape == ()
    assert reset_facts.has_transition.dtype == jnp.bool_
    assert not bool(reset_facts.has_transition)
    assert reset_facts.transition_start_step_count.shape == ()
    assert reset_facts.transition_start_step_count.dtype == jnp.int32
    assert int(reset_facts.transition_start_step_count) == -1

    acceptance = reset_facts.action_acceptance_facts
    for action in (acceptance.submitted_joint_action, acceptance.accepted_joint_action):
        for head in action:
            assert head.shape == (MAX_AGENT_SLOTS,)
            assert head.dtype == jnp.int32
            assert bool(jnp.all(head == 0))
    for rejected in (
        acceptance.submitted_action_tuple_is_out_of_domain_by_actor,
        acceptance.in_domain_move_action_is_rejected_by_actor,
        acceptance.in_domain_combat_action_pair_is_rejected_by_actor,
    ):
        assert rejected.shape == (MAX_AGENT_SLOTS,)
        assert rejected.dtype == jnp.bool_
        assert not bool(jnp.any(rejected))
    _assert_canonical_empty_combat_facts(reset_facts.combat_transition_facts)
    _assert_canonical_empty_death_facts(reset_facts.death_facts)
    _assert_canonical_empty_spawn_shield_facts(reset_facts.spawn_shield_facts)

    leaves = jax.tree_util.tree_leaves(reset_facts)
    assert len(leaves) == _TRANSITION_FACT_LEAF_COUNT
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)
    raw_bytes = sum(int(leaf.size) * int(leaf.dtype.itemsize) for leaf in leaves)
    assert raw_bytes == _TRANSITION_FACT_RAW_BYTES

    *_, neutral_info = _take_step(
        config,
        state,
        action_mask,
        _joint_action(),
    )
    neutral_facts = neutral_info.transition_facts
    assert jax.tree_util.tree_structure(neutral_facts) == jax.tree_util.tree_structure(
        reset_facts
    )
    assert bool(neutral_facts.has_transition)
    assert int(neutral_facts.transition_start_step_count) == 0
    _assert_canonical_empty_combat_facts(neutral_facts.combat_transition_facts)
    _assert_canonical_empty_death_facts(neutral_facts.death_facts)
    _assert_canonical_empty_spawn_shield_facts(neutral_facts.spawn_shield_facts)


@pytest.mark.parametrize(
    ("starting_duration", "expected_next_duration", "expected_expiry"),
    (
        pytest.param(1, 0, True, id="surviving-final-shielded-transition"),
        pytest.param(3, 2, False, id="shield-remains-active"),
    ),
)
def test_spawn_shield_facts_distinguish_activity_from_surviving_expiry(
    starting_duration: int,
    expected_next_duration: int,
    expected_expiry: bool,
) -> None:
    """Report transition-start shielding and only a surviving countdown expiry."""
    config, state, _ = _scenario(
        (_TEAM_A_FIRST_SLOT, HUNTER_CLASS_ID),
        (_TEAM_B_FIRST_SLOT, HUNTER_CLASS_ID),
    )
    state = state._replace(
        spawn_shield_durations=state.spawn_shield_durations.at[_TEAM_A_FIRST_SLOT].set(
            starting_duration
        )
    )
    action_mask = _build_observation_and_action_mask(state, config)[1]

    next_state, *_, info = _take_step(
        config,
        state,
        action_mask,
        _joint_action(),
    )
    facts = info.transition_facts.spawn_shield_facts

    assert bool(
        jnp.array_equal(
            facts.was_active_at_transition_start_by_agent,
            _single_source_mask(_TEAM_A_FIRST_SLOT),
        )
    )
    expected_expiry_mask = (
        _single_source_mask(_TEAM_A_FIRST_SLOT)
        if expected_expiry
        else jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.bool_)
    )
    assert bool(
        jnp.array_equal(
            facts.expired_at_transition_end_by_agent,
            expected_expiry_mask,
        )
    )
    assert (
        int(next_state.spawn_shield_durations[_TEAM_A_FIRST_SLOT])
        == expected_next_duration
    )
    assert bool(next_state.alive_mask[_TEAM_A_FIRST_SLOT])


def test_expiring_spawn_shield_rejects_current_target_then_reenables_next_mask() -> (
    None
):
    """Keep a counter-one recipient protected until the next policy action."""
    config, state, _ = _scenario(
        (_TEAM_A_FIRST_SLOT, HUNTER_CLASS_ID),
        (_TEAM_B_FIRST_SLOT, HUNTER_CLASS_ID),
    )
    shielded_recipient_slot = _TEAM_B_FIRST_SLOT
    state = state._replace(
        current_health=state.current_health.at[shielded_recipient_slot].set(1.0),
        spawn_shield_durations=state.spawn_shield_durations.at[
            shielded_recipient_slot
        ].set(1),
    )
    current_action_mask = _build_observation_and_action_mask(state, config)[1]

    next_state, _, _, _, next_action_mask, info = _take_step(
        config,
        state,
        current_action_mask,
        _joint_action(
            (
                _TEAM_A_FIRST_SLOT,
                MOVE_STAY,
                _FIRST_ENEMY_TARGET,
                0,
            )
        ),
    )
    shield_facts = info.transition_facts.spawn_shield_facts
    acceptance_facts = info.transition_facts.action_acceptance_facts

    assert bool(
        shield_facts.was_active_at_transition_start_by_agent[shielded_recipient_slot]
    )
    assert bool(
        acceptance_facts.in_domain_combat_action_pair_is_rejected_by_actor[
            _TEAM_A_FIRST_SLOT
        ]
    )
    assert (
        int(acceptance_facts.accepted_joint_action.select_target[_TEAM_A_FIRST_SLOT])
        == _TARGET_NONE
    )
    _assert_canonical_empty_combat_facts(info.transition_facts.combat_transition_facts)
    _assert_canonical_empty_death_facts(info.transition_facts.death_facts)
    assert bool(
        shield_facts.expired_at_transition_end_by_agent[shielded_recipient_slot]
    )
    assert bool(next_state.alive_mask[shielded_recipient_slot])
    assert next_state.current_health[shielded_recipient_slot] == 1.0
    assert int(next_state.spawn_shield_durations[shielded_recipient_slot]) == 0
    assert bool(
        next_action_mask.select_target_use_ultimate_joint_mask[
            _TEAM_A_FIRST_SLOT,
            _FIRST_ENEMY_TARGET,
            0,
        ]
    )

    final_state, *_, final_info = _take_step(
        config,
        next_state,
        next_action_mask,
        _joint_action(
            (
                _TEAM_A_FIRST_SLOT,
                MOVE_STAY,
                _FIRST_ENEMY_TARGET,
                0,
            )
        ),
    )
    final_acceptance = final_info.transition_facts.action_acceptance_facts
    final_combat = final_info.transition_facts.combat_transition_facts

    assert not bool(
        final_acceptance.in_domain_combat_action_pair_is_rejected_by_actor[
            _TEAM_A_FIRST_SLOT
        ]
    )
    assert (
        int(final_acceptance.accepted_joint_action.select_target[_TEAM_A_FIRST_SLOT])
        == _FIRST_ENEMY_TARGET
    )
    assert bool(final_combat.basic_effect_is_activated_by_source[_TEAM_A_FIRST_SLOT])
    assert bool(final_combat.combat_effect_has_recipient_by_source[_TEAM_A_FIRST_SLOT])
    assert final_state.current_health[shielded_recipient_slot] < 1.0


@pytest.mark.parametrize(
    ("head_name", "invalid_category"),
    (
        pytest.param("move", -1, id="move-below-domain"),
        pytest.param("move", NUM_MOVE_ACTIONS, id="move-above-domain"),
        pytest.param("select_target", -1, id="target-below-domain"),
        pytest.param(
            "select_target",
            NUM_TARGET_ACTIONS,
            id="target-above-domain",
        ),
        pytest.param("use_ultimate", -1, id="ultimate-below-domain"),
        pytest.param(
            "use_ultimate",
            NUM_ULTIMATE_ACTIONS,
            id="ultimate-above-domain",
        ),
    ),
)
def test_out_of_domain_head_has_exclusive_precedence_before_mask_rejection(
    head_name: str,
    invalid_category: int,
) -> None:
    """Prove every malformed head preserves raw intent and canonicalizes the tuple."""
    config, state, action_mask = _scenario(
        (_TEAM_A_FIRST_SLOT, HUNTER_CLASS_ID),
        (_TEAM_B_FIRST_SLOT, HUNTER_CLASS_ID),
    )
    submitted = _joint_action(
        (
            _TEAM_A_FIRST_SLOT,
            MOVE_EAST,
            _FIRST_ENEMY_TARGET,
            0,
        )
    )
    submitted = submitted._replace(
        **{
            head_name: getattr(submitted, head_name)
            .at[_TEAM_A_FIRST_SLOT]
            .set(invalid_category)
        }
    )

    *_, info = _take_step(config, state, action_mask, submitted)
    facts = info.transition_facts.action_acceptance_facts

    _assert_array_equal(facts.submitted_joint_action, submitted)
    assert bool(
        jnp.array_equal(
            facts.submitted_action_tuple_is_out_of_domain_by_actor,
            _single_source_mask(_TEAM_A_FIRST_SLOT),
        )
    )
    assert not bool(jnp.any(facts.in_domain_move_action_is_rejected_by_actor))
    assert not bool(jnp.any(facts.in_domain_combat_action_pair_is_rejected_by_actor))
    assert int(facts.accepted_joint_action.move[_TEAM_A_FIRST_SLOT]) == MOVE_STAY
    assert int(facts.accepted_joint_action.select_target[_TEAM_A_FIRST_SLOT]) == 0
    assert int(facts.accepted_joint_action.use_ultimate[_TEAM_A_FIRST_SLOT]) == 0
    _assert_canonical_empty_combat_facts(info.transition_facts.combat_transition_facts)


@pytest.mark.parametrize(
    (
        "reject_move",
        "submitted_target",
        "expected_move_rejection",
        "expected_combat_rejection",
    ),
    (
        pytest.param(True, _FIRST_ENEMY_TARGET, True, False, id="movement-only"),
        pytest.param(False, _SELF_TARGET, False, True, id="combat-only"),
        pytest.param(True, _SELF_TARGET, True, True, id="both-independent"),
    ),
)
def test_in_domain_rejections_preserve_independent_action_head_acceptance(
    reject_move: bool,
    submitted_target: int,
    expected_move_rejection: bool,
    expected_combat_rejection: bool,
) -> None:
    """Prove movement and combat-pair provenance survives canonicalization."""
    config, state, action_mask = _scenario(
        (_TEAM_A_FIRST_SLOT, HUNTER_CLASS_ID),
        (_TEAM_B_FIRST_SLOT, HUNTER_CLASS_ID),
    )
    if reject_move:
        action_mask = action_mask._replace(
            move_mask=action_mask.move_mask.at[_TEAM_A_FIRST_SLOT, MOVE_EAST].set(False)
        )
    submitted = _joint_action((_TEAM_A_FIRST_SLOT, MOVE_EAST, submitted_target, 0))

    next_state, *_, info = _take_step(config, state, action_mask, submitted)
    acceptance = info.transition_facts.action_acceptance_facts

    assert not bool(
        acceptance.submitted_action_tuple_is_out_of_domain_by_actor[_TEAM_A_FIRST_SLOT]
    )
    assert (
        bool(acceptance.in_domain_move_action_is_rejected_by_actor[_TEAM_A_FIRST_SLOT])
        is expected_move_rejection
    )
    assert (
        bool(
            acceptance.in_domain_combat_action_pair_is_rejected_by_actor[
                _TEAM_A_FIRST_SLOT
            ]
        )
        is expected_combat_rejection
    )
    expected_move = MOVE_STAY if expected_move_rejection else MOVE_EAST
    expected_target = 0 if expected_combat_rejection else _FIRST_ENEMY_TARGET
    assert int(acceptance.accepted_joint_action.move[_TEAM_A_FIRST_SLOT]) == (
        expected_move
    )
    assert (
        int(acceptance.accepted_joint_action.select_target[_TEAM_A_FIRST_SLOT])
        == expected_target
    )
    assert bool(
        info.transition_facts.combat_transition_facts.basic_effect_is_activated_by_source[
            _TEAM_A_FIRST_SLOT
        ]
    ) is (not expected_combat_rejection)
    assert bool(
        jnp.array_equal(
            next_state.previous_timestep_move_actions,
            acceptance.accepted_joint_action.move,
        )
    )
    assert bool(
        jnp.array_equal(
            next_state.previous_timestep_select_target_actions,
            acceptance.accepted_joint_action.select_target,
        )
    )
    assert bool(
        jnp.array_equal(
            next_state.previous_timestep_use_ultimate_actions,
            acceptance.accepted_joint_action.use_ultimate,
        )
    )


@pytest.mark.parametrize(
    "nonacting_kind",
    (
        pytest.param("dead", id="dead-configured-slot"),
        pytest.param("inactive", id="inactive-padding-slot"),
    ),
)
def test_dead_and_inactive_rows_distinguish_canonical_noop_from_rejection(
    nonacting_kind: str,
) -> None:
    """Prove nonactors accept only canonical no-op without ambiguous provenance."""
    config, state, action_mask = _scenario(
        (_TEAM_A_FIRST_SLOT, HUNTER_CLASS_ID),
        (_TEAM_B_FIRST_SLOT, HUNTER_CLASS_ID),
    )
    actor_slot = _TEAM_A_FIRST_SLOT
    if nonacting_kind == "dead":
        state = state._replace(
            alive_mask=state.alive_mask.at[actor_slot].set(False),
            current_health=state.current_health.at[actor_slot].set(0.0),
        )
    else:
        actor_slot = 1
    action_mask = _build_observation_and_action_mask(state, config)[1]

    *_, canonical_info = _take_step(
        config,
        state,
        action_mask,
        _joint_action(),
    )
    canonical = canonical_info.transition_facts.action_acceptance_facts
    assert not bool(canonical.in_domain_move_action_is_rejected_by_actor[actor_slot])
    assert not bool(
        canonical.in_domain_combat_action_pair_is_rejected_by_actor[actor_slot]
    )

    submitted = _joint_action((actor_slot, MOVE_EAST, _FIRST_ENEMY_TARGET, 0))
    *_, rejected_info = _take_step(
        config,
        state,
        action_mask,
        submitted,
    )
    rejected = rejected_info.transition_facts.action_acceptance_facts
    assert not bool(
        rejected.submitted_action_tuple_is_out_of_domain_by_actor[actor_slot]
    )
    assert bool(rejected.in_domain_move_action_is_rejected_by_actor[actor_slot])
    assert bool(rejected.in_domain_combat_action_pair_is_rejected_by_actor[actor_slot])
    assert int(rejected.accepted_joint_action.move[actor_slot]) == MOVE_STAY
    assert int(rejected.accepted_joint_action.select_target[actor_slot]) == 0
    assert int(rejected.accepted_joint_action.use_ultimate[actor_slot]) == 0
    _assert_canonical_empty_combat_facts(
        rejected_info.transition_facts.combat_transition_facts
    )
    _assert_canonical_empty_death_facts(rejected_info.transition_facts.death_facts)


_ROUTING_CASES = (
    *(
        (_TEAM_A_FIRST_SLOT, PRIEST_CLASS_ID, 1 + local_slot, local_slot)
        for local_slot in range(MAX_AGENTS_PER_TEAM)
    ),
    *(
        (
            _TEAM_B_FIRST_SLOT,
            PRIEST_CLASS_ID,
            1 + local_slot,
            MAX_AGENTS_PER_TEAM + local_slot,
        )
        for local_slot in range(MAX_AGENTS_PER_TEAM)
    ),
    *(
        (
            _TEAM_A_FIRST_SLOT,
            HUNTER_CLASS_ID,
            _FIRST_ENEMY_TARGET + local_slot,
            MAX_AGENTS_PER_TEAM + local_slot,
        )
        for local_slot in range(MAX_AGENTS_PER_TEAM)
    ),
    *(
        (
            _TEAM_B_FIRST_SLOT,
            HUNTER_CLASS_ID,
            _FIRST_ENEMY_TARGET + local_slot,
            local_slot,
        )
        for local_slot in range(MAX_AGENTS_PER_TEAM)
    ),
)


@pytest.mark.parametrize(
    ("actor_slot", "actor_class_id", "target_action", "recipient_slot"),
    [
        pytest.param(
            actor_slot,
            actor_class_id,
            target_action,
            recipient_slot,
            id=f"actor-{actor_slot}-target-{target_action}-recipient-{recipient_slot}",
        )
        for actor_slot, actor_class_id, target_action, recipient_slot in (
            _ROUTING_CASES
        )
    ],
)
def test_every_relation_local_target_routes_to_the_stable_global_slot(
    actor_slot: int,
    actor_class_id: int,
    target_action: int,
    recipient_slot: int,
) -> None:
    """Prove ally/enemy routing, team symmetry, and the guarded public index."""
    config, state, action_mask = _scenario(
        (actor_slot, actor_class_id),
        team_sizes=(5, 5),
    )
    submitted = _joint_action((actor_slot, MOVE_STAY, target_action, 0))

    *_, info = _take_step(config, state, action_mask, submitted)
    acceptance = info.transition_facts.action_acceptance_facts
    combat_facts = info.transition_facts.combat_transition_facts

    assert int(acceptance.accepted_joint_action.select_target[actor_slot]) == (
        target_action
    )
    assert bool(
        jnp.array_equal(
            combat_facts.combat_effect_has_recipient_by_source,
            _single_source_mask(actor_slot),
        )
    )
    expected_recipients = (
        jnp.full((MAX_AGENT_SLOTS,), -1, dtype=jnp.int32)
        .at[actor_slot]
        .set(recipient_slot)
    )
    assert bool(
        jnp.array_equal(
            combat_facts.combat_effect_recipient_global_slot_by_source,
            expected_recipients,
        )
    )


def test_inactive_target_and_target_none_never_decode_as_a_real_recipient() -> None:
    """Prove rejected padding and accepted target-none both retain sentinel -1."""
    config, state, action_mask = _scenario(
        (_TEAM_A_FIRST_SLOT, HUNTER_CLASS_ID),
        (_TEAM_B_FIRST_SLOT, HUNTER_CLASS_ID),
    )
    inactive_enemy_target = _FIRST_ENEMY_TARGET + MAX_AGENTS_PER_TEAM - 1
    submitted = _joint_action((_TEAM_A_FIRST_SLOT, MOVE_STAY, inactive_enemy_target, 0))

    *_, rejected_info = _take_step(config, state, action_mask, submitted)
    rejected_acceptance = rejected_info.transition_facts.action_acceptance_facts
    rejected_combat = rejected_info.transition_facts.combat_transition_facts
    assert bool(
        rejected_acceptance.in_domain_combat_action_pair_is_rejected_by_actor[
            _TEAM_A_FIRST_SLOT
        ]
    )
    assert (
        int(rejected_acceptance.accepted_joint_action.select_target[_TEAM_A_FIRST_SLOT])
        == _TARGET_NONE
    )
    assert not bool(
        rejected_combat.combat_effect_has_recipient_by_source[_TEAM_A_FIRST_SLOT]
    )
    assert (
        int(
            rejected_combat.combat_effect_recipient_global_slot_by_source[
                _TEAM_A_FIRST_SLOT
            ]
        )
        == -1
    )

    *_, neutral_info = _take_step(
        config,
        state,
        action_mask,
        _joint_action(),
    )
    neutral_combat = neutral_info.transition_facts.combat_transition_facts
    assert not bool(
        neutral_combat.combat_effect_has_recipient_by_source[_TEAM_A_FIRST_SLOT]
    )
    assert (
        int(
            neutral_combat.combat_effect_recipient_global_slot_by_source[
                _TEAM_A_FIRST_SLOT
            ]
        )
        == -1
    )


def test_mage_burst_is_a_real_ultimate_without_a_slot_zero_recipient() -> None:
    """Prove source-local Mage Burst cannot be decoded as a target event."""
    config, state, action_mask = _scenario(
        (_TEAM_A_FIRST_SLOT, MAGE_CLASS_ID),
        (_TEAM_B_FIRST_SLOT, HUNTER_CLASS_ID),
    )
    submitted = _joint_action((_TEAM_A_FIRST_SLOT, MOVE_STAY, _TARGET_NONE, 1))

    *_, info = _take_step(config, state, action_mask, submitted)
    facts = info.transition_facts.combat_transition_facts

    assert not bool(facts.basic_effect_is_activated_by_source[_TEAM_A_FIRST_SLOT])
    assert bool(facts.ultimate_effect_is_activated_by_source[_TEAM_A_FIRST_SLOT])
    assert not bool(facts.combat_effect_has_recipient_by_source[_TEAM_A_FIRST_SLOT])
    assert (
        int(facts.combat_effect_recipient_global_slot_by_source[_TEAM_A_FIRST_SLOT])
        == -1
    )
    assert bool(
        facts.mage_burst_damage_amplification_is_applied_by_source[_TEAM_A_FIRST_SLOT]
    )
    for amount_or_modifier in (
        facts.raw_damage_output_by_source,
        facts.source_modified_damage_output_by_source,
        facts.recipient_damage_modifier_by_source,
        facts.raw_healing_output_by_source,
        facts.source_modified_healing_output_by_source,
        facts.recipient_healing_modifier_by_source,
    ):
        assert float(amount_or_modifier[_TEAM_A_FIRST_SLOT]) == 0.0
    assert bool(jnp.all(facts.total_effective_damage_by_recipient == 0.0))
    assert bool(jnp.all(facts.total_effective_healing_by_recipient == 0.0))


@pytest.mark.parametrize(
    ("actor_class_id", "target_action", "recipient_slot"),
    (
        pytest.param(
            MAGE_CLASS_ID,
            _FIRST_ENEMY_TARGET,
            _TEAM_B_FIRST_SLOT,
            id="mage-damage",
        ),
        pytest.param(
            WARRIOR_CLASS_ID,
            _FIRST_ENEMY_TARGET,
            _TEAM_B_FIRST_SLOT,
            id="warrior-damage",
        ),
        pytest.param(
            HUNTER_CLASS_ID,
            _FIRST_ENEMY_TARGET,
            _TEAM_B_FIRST_SLOT,
            id="hunter-damage-and-slow",
        ),
        pytest.param(
            ROGUE_CLASS_ID,
            _FIRST_ENEMY_TARGET,
            _TEAM_B_FIRST_SLOT,
            id="rogue-damage",
        ),
        pytest.param(
            PRIEST_CLASS_ID,
            _SELF_TARGET,
            _TEAM_A_FIRST_SLOT,
            id="priest-healing-and-freedom",
        ),
    ),
)
def test_every_class_basic_emits_its_authoritative_effect_lane(
    actor_class_id: int,
    target_action: int,
    recipient_slot: int,
) -> None:
    """Prove every Basic lane exposes catalog payload and application truth."""
    config, state, action_mask = _scenario(
        (_TEAM_A_FIRST_SLOT, actor_class_id),
        (_TEAM_B_FIRST_SLOT, HUNTER_CLASS_ID),
    )
    if actor_class_id == PRIEST_CLASS_ID:
        state = state._replace(
            current_health=state.current_health.at[recipient_slot].add(-20.0)
        )
        action_mask = _build_observation_and_action_mask(state, config)[1]
    submitted = _joint_action((_TEAM_A_FIRST_SLOT, MOVE_STAY, target_action, 0))

    next_state, *_, info = _take_step(
        config,
        state,
        action_mask,
        submitted,
    )
    facts = info.transition_facts.combat_transition_facts

    assert bool(
        jnp.array_equal(
            facts.basic_effect_is_activated_by_source,
            _single_source_mask(_TEAM_A_FIRST_SLOT),
        )
    )
    assert not bool(jnp.any(facts.ultimate_effect_is_activated_by_source))
    assert bool(facts.combat_effect_has_recipient_by_source[_TEAM_A_FIRST_SLOT])
    assert (
        int(facts.combat_effect_recipient_global_slot_by_source[_TEAM_A_FIRST_SLOT])
        == recipient_slot
    )

    raw_damage = combat.BASIC_DAMAGE_BY_CLASS[actor_class_id]
    raw_healing = combat.BASIC_HEALING_BY_CLASS[actor_class_id]
    assert facts.raw_damage_output_by_source[_TEAM_A_FIRST_SLOT] == raw_damage
    assert facts.raw_healing_output_by_source[_TEAM_A_FIRST_SLOT] == raw_healing

    source_damage_multiplier = (
        combat.MAGE_DAMAGE_AURA_MULTIPLIER if actor_class_id == MAGE_CLASS_ID else 1.0
    )
    expected_source_damage = raw_damage * source_damage_multiplier
    assert bool(
        jnp.isclose(
            facts.source_modified_damage_output_by_source[_TEAM_A_FIRST_SLOT],
            expected_source_damage,
        )
    )
    assert (
        facts.source_modified_healing_output_by_source[_TEAM_A_FIRST_SLOT]
        == raw_healing
    )
    expected_damage_modifier = 1.0 if float(raw_damage) > 0.0 else 0.0
    expected_healing_modifier = 1.0 if float(raw_healing) > 0.0 else 0.0
    assert (
        facts.recipient_damage_modifier_by_source[_TEAM_A_FIRST_SLOT]
        == expected_damage_modifier
    )
    assert (
        facts.recipient_healing_modifier_by_source[_TEAM_A_FIRST_SLOT]
        == expected_healing_modifier
    )
    assert bool(
        jnp.isclose(
            facts.total_effective_damage_by_recipient[recipient_slot],
            expected_source_damage,
        )
    )
    assert facts.total_effective_healing_by_recipient[recipient_slot] == raw_healing
    assert bool(
        jnp.isclose(
            next_state.current_health[recipient_slot]
            - state.current_health[recipient_slot],
            raw_healing - expected_source_damage,
        )
    )

    expected_slow = jnp.zeros((MAX_AGENT_SLOTS, NUM_SLOW_CHANNELS), dtype=jnp.bool_)
    if actor_class_id == HUNTER_CLASS_ID:
        expected_slow = expected_slow.at[
            _TEAM_A_FIRST_SLOT, SLOW_CHANNEL_HUNTER_BASIC
        ].set(True)
    assert bool(
        jnp.array_equal(
            facts.slow_is_applied_by_source_and_channel,
            expected_slow,
        )
    )
    assert not bool(jnp.any(facts.stun_is_applied_by_source_and_channel))
    assert bool(
        facts.priest_blessing_of_freedom_is_applied_by_source[_TEAM_A_FIRST_SLOT]
    ) is (actor_class_id == PRIEST_CLASS_ID)
    assert not bool(jnp.any(facts.rogue_poison_anti_heal_is_applied_by_source))
    assert not bool(jnp.any(facts.mage_burst_damage_amplification_is_applied_by_source))


@pytest.mark.parametrize(
    ("actor_class_id", "target_action", "recipient_slot"),
    (
        pytest.param(MAGE_CLASS_ID, _TARGET_NONE, -1, id="mage-burst"),
        pytest.param(
            WARRIOR_CLASS_ID,
            _FIRST_ENEMY_TARGET,
            _TEAM_B_FIRST_SLOT,
            id="warrior-charge",
        ),
        pytest.param(
            HUNTER_CLASS_ID,
            _FIRST_ENEMY_TARGET,
            _TEAM_B_FIRST_SLOT,
            id="hunter-trap",
        ),
        pytest.param(
            ROGUE_CLASS_ID,
            _FIRST_ENEMY_TARGET,
            _TEAM_B_FIRST_SLOT,
            id="rogue-poison",
        ),
        pytest.param(
            PRIEST_CLASS_ID,
            _SELF_TARGET,
            _TEAM_A_FIRST_SLOT,
            id="priest-holy-word",
        ),
    ),
)
def test_every_class_ultimate_emits_health_and_status_application_truth(
    actor_class_id: int,
    target_action: int,
    recipient_slot: int,
) -> None:
    """Prove all Ultimate lanes, including zero-payload lanes, remain explicit."""
    config, state, action_mask = _scenario(
        (_TEAM_A_FIRST_SLOT, actor_class_id),
        (_TEAM_B_FIRST_SLOT, HUNTER_CLASS_ID),
    )
    if actor_class_id == PRIEST_CLASS_ID:
        state = state._replace(
            current_health=state.current_health.at[_TEAM_A_FIRST_SLOT].add(-50.0)
        )
        action_mask = _build_observation_and_action_mask(state, config)[1]
    submitted = _joint_action((_TEAM_A_FIRST_SLOT, MOVE_STAY, target_action, 1))

    next_state, *_, info = _take_step(
        config,
        state,
        action_mask,
        submitted,
    )
    facts = info.transition_facts.combat_transition_facts

    assert not bool(jnp.any(facts.basic_effect_is_activated_by_source))
    assert bool(
        jnp.array_equal(
            facts.ultimate_effect_is_activated_by_source,
            _single_source_mask(_TEAM_A_FIRST_SLOT),
        )
    )
    has_recipient = recipient_slot >= 0
    assert (
        bool(facts.combat_effect_has_recipient_by_source[_TEAM_A_FIRST_SLOT])
        is has_recipient
    )
    assert (
        int(facts.combat_effect_recipient_global_slot_by_source[_TEAM_A_FIRST_SLOT])
        == recipient_slot
    )

    raw_damage = combat.ULTIMATE_DAMAGE_BY_CLASS[actor_class_id]
    raw_healing = combat.ULTIMATE_HEALING_BY_CLASS[actor_class_id]
    assert facts.raw_damage_output_by_source[_TEAM_A_FIRST_SLOT] == raw_damage
    assert facts.raw_healing_output_by_source[_TEAM_A_FIRST_SLOT] == raw_healing
    assert (
        facts.source_modified_damage_output_by_source[_TEAM_A_FIRST_SLOT] == raw_damage
    )
    assert (
        facts.source_modified_healing_output_by_source[_TEAM_A_FIRST_SLOT]
        == raw_healing
    )
    expected_damage_modifier = 1.0 if float(raw_damage) > 0.0 else 0.0
    expected_healing_modifier = 1.0 if float(raw_healing) > 0.0 else 0.0
    assert (
        facts.recipient_damage_modifier_by_source[_TEAM_A_FIRST_SLOT]
        == expected_damage_modifier
    )
    assert (
        facts.recipient_healing_modifier_by_source[_TEAM_A_FIRST_SLOT]
        == expected_healing_modifier
    )

    expected_slow = jnp.zeros((MAX_AGENT_SLOTS, NUM_SLOW_CHANNELS), dtype=jnp.bool_)
    expected_stun = jnp.zeros((MAX_AGENT_SLOTS, NUM_STUN_CHANNELS), dtype=jnp.bool_)
    if actor_class_id == WARRIOR_CLASS_ID:
        expected_slow = expected_slow.at[
            _TEAM_A_FIRST_SLOT, SLOW_CHANNEL_WARRIOR_CHARGE
        ].set(True)
        expected_stun = expected_stun.at[
            _TEAM_A_FIRST_SLOT, STUN_CHANNEL_WARRIOR_CHARGE
        ].set(True)
    elif actor_class_id == HUNTER_CLASS_ID:
        expected_stun = expected_stun.at[
            _TEAM_A_FIRST_SLOT, STUN_CHANNEL_HUNTER_TRAP
        ].set(True)
    elif actor_class_id == ROGUE_CLASS_ID:
        expected_slow = expected_slow.at[
            _TEAM_A_FIRST_SLOT, SLOW_CHANNEL_ROGUE_POISON
        ].set(True)
        expected_stun = expected_stun.at[
            _TEAM_A_FIRST_SLOT, STUN_CHANNEL_ROGUE_POISON
        ].set(True)
    assert bool(
        jnp.array_equal(
            facts.slow_is_applied_by_source_and_channel,
            expected_slow,
        )
    )
    assert bool(
        jnp.array_equal(
            facts.stun_is_applied_by_source_and_channel,
            expected_stun,
        )
    )
    assert bool(
        facts.rogue_poison_anti_heal_is_applied_by_source[_TEAM_A_FIRST_SLOT]
    ) is (actor_class_id == ROGUE_CLASS_ID)
    assert bool(
        facts.mage_burst_damage_amplification_is_applied_by_source[_TEAM_A_FIRST_SLOT]
    ) is (actor_class_id == MAGE_CLASS_ID)
    assert not bool(jnp.any(facts.priest_blessing_of_freedom_is_applied_by_source))

    if recipient_slot >= 0:
        expected_successor_health = jnp.clip(
            state.current_health[recipient_slot] + raw_healing - raw_damage,
            0.0,
            config.agent_profile.max_health[recipient_slot],
        )
        assert next_state.current_health[recipient_slot] == (expected_successor_health)
        assert facts.total_effective_damage_by_recipient[recipient_slot] == raw_damage
        assert facts.total_effective_healing_by_recipient[recipient_slot] == raw_healing
    else:
        assert bool(jnp.all(facts.total_effective_damage_by_recipient == 0.0))
        assert bool(jnp.all(facts.total_effective_healing_by_recipient == 0.0))


def test_status_application_fact_is_not_inferred_from_duration_change() -> None:
    """Prove a refresh remains explicit when the successor duration is unchanged."""
    config, state, action_mask = _scenario(
        (_TEAM_A_FIRST_SLOT, HUNTER_CLASS_ID),
        (_TEAM_B_FIRST_SLOT, HUNTER_CLASS_ID),
    )
    current_trap_duration = combat.HUNTER_TRAP_STUN_DURATION_TICKS + 1
    state = state._replace(
        stun_durations=state.stun_durations.at[
            _TEAM_B_FIRST_SLOT, STUN_CHANNEL_HUNTER_TRAP
        ].set(current_trap_duration)
    )
    action_mask = _build_observation_and_action_mask(state, config)[1]

    next_state, *_, info = _take_step(
        config,
        state,
        action_mask,
        _joint_action((_TEAM_A_FIRST_SLOT, MOVE_STAY, _FIRST_ENEMY_TARGET, 1)),
    )

    assert (
        int(next_state.stun_durations[_TEAM_B_FIRST_SLOT, STUN_CHANNEL_HUNTER_TRAP])
        == current_trap_duration - 1
    )
    assert bool(
        info.transition_facts.combat_transition_facts.stun_is_applied_by_source_and_channel[
            _TEAM_A_FIRST_SLOT, STUN_CHANNEL_HUNTER_TRAP
        ]
    )


def test_focus_fire_reconciles_source_and_recipient_modifier_stages() -> None:
    """Prove amplified sources and mitigated recipients reconcile to core totals."""
    positions = _default_positions((3, 2))
    positions = positions.at[0].set(jnp.asarray((2.0, 2.0), dtype=jnp.float32))
    positions = positions.at[1].set(jnp.asarray((2.0, 3.0), dtype=jnp.float32))
    positions = positions.at[2].set(jnp.asarray((2.0, 4.0), dtype=jnp.float32))
    positions = positions.at[5].set(jnp.asarray((12.0, 2.0), dtype=jnp.float32))
    positions = positions.at[6].set(jnp.asarray((12.0, 3.0), dtype=jnp.float32))
    config, state, action_mask = _scenario(
        (0, MAGE_CLASS_ID),
        (1, HUNTER_CLASS_ID),
        (2, ROGUE_CLASS_ID),
        (5, HUNTER_CLASS_ID),
        (6, WARRIOR_CLASS_ID),
        team_sizes=(3, 2),
        positions=positions,
    )
    state = state._replace(
        mage_burst_damage_amplification_durations=(
            state.mage_burst_damage_amplification_durations.at[0].set(2)
        )
    )
    action_mask = _build_observation_and_action_mask(state, config)[1]
    submitted = _joint_action(
        (0, MOVE_STAY, _FIRST_ENEMY_TARGET, 0),
        (1, MOVE_STAY, _FIRST_ENEMY_TARGET, 0),
        (2, MOVE_STAY, _FIRST_ENEMY_TARGET, 0),
    )

    next_state, *_, info = _take_step(
        config,
        state,
        action_mask,
        submitted,
    )
    facts = info.transition_facts.combat_transition_facts
    sources = jnp.asarray((0, 1, 2), dtype=jnp.int32)
    expected_raw = combat.BASIC_DAMAGE_BY_CLASS[
        jnp.asarray(
            (MAGE_CLASS_ID, HUNTER_CLASS_ID, ROGUE_CLASS_ID),
            dtype=jnp.int32,
        )
    ]
    expected_source_modified = expected_raw * combat.MAGE_DAMAGE_AURA_MULTIPLIER
    expected_source_modified = expected_source_modified.at[0].multiply(
        combat.MAGE_BURST_DAMAGE_MULTIPLIER
    )
    expected_recipient_modifiers = jnp.full(
        (3,),
        combat.WARRIOR_DAMAGE_MITIGATION_AURA_MULTIPLIER,
        dtype=jnp.float32,
    )
    expected_total = jnp.sum(expected_source_modified * expected_recipient_modifiers)

    assert bool(jnp.allclose(facts.raw_damage_output_by_source[sources], expected_raw))
    assert bool(
        jnp.allclose(
            facts.source_modified_damage_output_by_source[sources],
            expected_source_modified,
        )
    )
    assert bool(
        jnp.allclose(
            facts.recipient_damage_modifier_by_source[sources],
            expected_recipient_modifiers,
        )
    )
    reconciled_total = jnp.sum(
        facts.source_modified_damage_output_by_source
        * facts.recipient_damage_modifier_by_source
    )
    assert bool(jnp.isclose(reconciled_total, expected_total))
    assert bool(
        jnp.isclose(
            facts.total_effective_damage_by_recipient[_TEAM_B_FIRST_SLOT],
            expected_total,
        )
    )
    assert bool(
        jnp.isclose(
            state.current_health[_TEAM_B_FIRST_SLOT]
            - next_state.current_health[_TEAM_B_FIRST_SLOT],
            expected_total,
        )
    )
    assert bool(
        jnp.all(
            facts.total_effective_damage_by_recipient.at[_TEAM_B_FIRST_SLOT].set(0.0)
            == 0.0
        )
    )


def test_anti_heal_reconciles_source_healing_to_authoritative_recipient_total() -> None:
    """Prove recipient anti-heal remains distinct from source healing output."""
    config, state, action_mask = _scenario(
        (0, PRIEST_CLASS_ID),
        (1, PRIEST_CLASS_ID),
        (2, HUNTER_CLASS_ID),
        (5, HUNTER_CLASS_ID),
        team_sizes=(3, 1),
    )
    state = state._replace(
        current_health=state.current_health.at[2].add(-30.0),
        rogue_poison_anti_heal_durations=(
            state.rogue_poison_anti_heal_durations.at[2].set(2)
        ),
    )
    action_mask = _build_observation_and_action_mask(state, config)[1]
    submitted = _joint_action(
        (0, MOVE_STAY, 3, 0),
        (1, MOVE_STAY, 3, 0),
    )

    next_state, *_, info = _take_step(
        config,
        state,
        action_mask,
        submitted,
    )
    facts = info.transition_facts.combat_transition_facts
    sources = jnp.asarray((0, 1), dtype=jnp.int32)
    expected_raw = jnp.full(
        (2,),
        combat.BASIC_HEALING_BY_CLASS[PRIEST_CLASS_ID],
        dtype=jnp.float32,
    )
    expected_modifiers = jnp.full(
        (2,),
        combat.ROGUE_POISON_ANTI_HEAL_MULTIPLIER,
        dtype=jnp.float32,
    )
    expected_total = jnp.sum(expected_raw * expected_modifiers)

    assert bool(
        jnp.array_equal(facts.raw_healing_output_by_source[sources], expected_raw)
    )
    assert bool(
        jnp.array_equal(
            facts.source_modified_healing_output_by_source[sources],
            expected_raw,
        )
    )
    assert bool(
        jnp.array_equal(
            facts.recipient_healing_modifier_by_source[sources],
            expected_modifiers,
        )
    )
    assert bool(
        jnp.isclose(
            facts.total_effective_healing_by_recipient[2],
            expected_total,
        )
    )
    assert bool(
        jnp.isclose(
            next_state.current_health[2] - state.current_health[2],
            expected_total,
        )
    )


def test_health_clamps_do_not_rewrite_gross_damage_or_healing_facts() -> None:
    """Prove overkill and overheal retain pre-net, pre-clamp transition amounts."""
    damage_config, damage_state, damage_mask = _scenario(
        (0, ROGUE_CLASS_ID),
        (5, HUNTER_CLASS_ID),
    )
    damage_state = damage_state._replace(
        current_health=damage_state.current_health.at[5].set(1.0)
    )
    damage_mask = _build_observation_and_action_mask(damage_state, damage_config)[1]
    damage_next_state, *_, damage_info = _take_step(
        damage_config,
        damage_state,
        damage_mask,
        _joint_action((0, MOVE_STAY, _FIRST_ENEMY_TARGET, 0)),
    )
    damage_facts = damage_info.transition_facts.combat_transition_facts
    damage_death_facts = damage_info.transition_facts.death_facts
    assert (
        damage_facts.total_effective_damage_by_recipient[5]
        == (combat.BASIC_DAMAGE_BY_CLASS[ROGUE_CLASS_ID])
    )
    assert damage_state.current_health[5] - damage_next_state.current_health[5] == 1.0
    assert damage_facts.total_effective_damage_by_recipient[5] > 1.0
    assert bool(
        jnp.array_equal(
            damage_death_facts.is_newly_dead_by_recipient,
            _single_source_mask(5),
        )
    )
    assert bool(
        jnp.array_equal(
            damage_death_facts.contributed_to_new_death_by_source,
            _single_source_mask(0),
        )
    )
    assert (
        damage_death_facts.attributed_death_damage_by_source[0]
        == damage_facts.source_modified_damage_output_by_source[0]
        * damage_facts.recipient_damage_modifier_by_source[0]
    )
    assert bool(
        jnp.all(
            damage_death_facts.attributed_death_damage_by_source.at[0].set(0.0) == 0.0
        )
    )

    healing_config, healing_state, healing_mask = _scenario(
        (0, PRIEST_CLASS_ID),
        (1, PRIEST_CLASS_ID),
        (2, HUNTER_CLASS_ID),
        (5, HUNTER_CLASS_ID),
        team_sizes=(3, 1),
    )
    healing_state = healing_state._replace(
        current_health=healing_state.current_health.at[2].add(-2.0)
    )
    healing_mask = _build_observation_and_action_mask(healing_state, healing_config)[1]
    healing_next_state, *_, healing_info = _take_step(
        healing_config,
        healing_state,
        healing_mask,
        _joint_action(
            (0, MOVE_STAY, 3, 0),
            (1, MOVE_STAY, 3, 0),
        ),
    )
    healing_facts = healing_info.transition_facts.combat_transition_facts
    expected_gross_healing = 2.0 * combat.BASIC_HEALING_BY_CLASS[PRIEST_CLASS_ID]
    assert healing_facts.total_effective_healing_by_recipient[2] == (
        expected_gross_healing
    )
    assert healing_next_state.current_health[2] - healing_state.current_health[2] == 2.0
    assert healing_facts.total_effective_healing_by_recipient[2] > 2.0


def test_simultaneous_damage_and_healing_remain_separate_gross_totals() -> None:
    """Prove health netting does not collapse authoritative damage/healing facts."""
    config, state, action_mask = _scenario(
        (0, HUNTER_CLASS_ID),
        (5, HUNTER_CLASS_ID),
        (6, PRIEST_CLASS_ID),
        team_sizes=(1, 2),
    )
    state = state._replace(current_health=state.current_health.at[5].set(50.0))
    action_mask = _build_observation_and_action_mask(state, config)[1]
    submitted = _joint_action(
        (0, MOVE_STAY, _FIRST_ENEMY_TARGET, 0),
        (6, MOVE_STAY, _SELF_TARGET, 0),
    )

    next_state, *_, info = _take_step(
        config,
        state,
        action_mask,
        submitted,
    )
    facts = info.transition_facts.combat_transition_facts
    expected_damage = combat.BASIC_DAMAGE_BY_CLASS[HUNTER_CLASS_ID]
    expected_healing = combat.BASIC_HEALING_BY_CLASS[PRIEST_CLASS_ID]

    assert facts.total_effective_damage_by_recipient[5] == expected_damage
    assert facts.total_effective_healing_by_recipient[5] == expected_healing
    assert next_state.current_health[5] == (
        state.current_health[5] + expected_healing - expected_damage
    )


def test_transition_facts_compose_under_eager_jit_vmap_and_scan() -> None:
    """Prove public transition PyTrees remain static across JAX execution modes."""
    config, state, action_mask = _scenario(
        (0, HUNTER_CLASS_ID),
        (5, HUNTER_CLASS_ID),
    )
    neutral_action = _joint_action()
    basic_action = _joint_action((0, MOVE_STAY, _FIRST_ENEMY_TARGET, 0))

    eager = _take_step(config, state, action_mask, basic_action)
    compiled = cast(
        tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info],
        jax.jit(step)(
            config,
            state,
            action_mask,
            basic_action,
            jax.random.key(2),
        ),
    )
    _assert_array_equal(eager, compiled)

    batched_actions = jax.tree.map(
        lambda *heads: jnp.stack(heads),
        neutral_action,
        basic_action,
    )

    def consume_public_transition(
        action: Action,
    ) -> tuple[Observation, ActionMask, TransitionFacts]:
        """Return policy outputs and facts from one shared-config transition."""
        _, observation, _, _, next_mask, info = step(
            config,
            state,
            action_mask,
            action,
            jax.random.key(4),
        )
        return observation, next_mask, info.transition_facts

    batched_observations, batched_masks, batched_facts = jax.vmap(
        consume_public_transition
    )(batched_actions)
    assert (
        batched_observations.spawn_lifecycle.spawn_shield_actual_durations_by_agent_by_team.shape
        == (2, MAX_AGENT_SLOTS, 2, MAX_AGENTS_PER_TEAM)
    )
    assert batched_masks.select_target_use_ultimate_joint_mask.shape == (
        2,
        MAX_AGENT_SLOTS,
        NUM_TARGET_ACTIONS,
        NUM_ULTIMATE_ACTIONS,
    )
    assert batched_facts.has_transition.shape == (2,)
    assert bool(jnp.all(batched_facts.has_transition))
    assert bool(jnp.all(batched_facts.transition_start_step_count == 0))
    assert bool(
        jnp.array_equal(
            batched_facts.combat_transition_facts.basic_effect_is_activated_by_source[
                :, 0
            ],
            jnp.asarray((False, True), dtype=jnp.bool_),
        )
    )
    batched_shield_facts = batched_facts.spawn_shield_facts
    assert batched_shield_facts.was_active_at_transition_start_by_agent.shape == (
        2,
        MAX_AGENT_SLOTS,
    )
    assert batched_shield_facts.expired_at_transition_end_by_agent.shape == (
        2,
        MAX_AGENT_SLOTS,
    )
    assert not bool(
        jnp.any(batched_shield_facts.was_active_at_transition_start_by_agent)
    )
    assert not bool(jnp.any(batched_shield_facts.expired_at_transition_end_by_agent))

    def repeat_action_head(head: Array) -> Array:
        """Repeat one fixed-slot action head along the scan time axis."""
        return jnp.repeat(head[None, :], repeats=3, axis=0)

    scan_actions = jax.tree.map(repeat_action_head, neutral_action)
    scan_keys = jax.random.split(jax.random.key(5), 3)

    def scan_body(
        carry: tuple[EnvState, ActionMask],
        inputs: tuple[Action, Array],
    ) -> tuple[
        tuple[EnvState, ActionMask],
        tuple[Observation, ActionMask, TransitionFacts],
    ]:
        """Carry state/mask and stack every public transition structure."""
        current_state, current_mask = carry
        action, key = inputs
        next_state, observation, _, _, next_mask, info = step(
            config,
            current_state,
            current_mask,
            action,
            key,
        )
        return (next_state, next_mask), (
            observation,
            next_mask,
            info.transition_facts,
        )

    (_, _), (scanned_observations, scanned_masks, scanned_facts) = jax.lax.scan(
        scan_body,
        (state, action_mask),
        (scan_actions, scan_keys),
    )
    assert (
        scanned_observations.spawn_lifecycle.spawn_shield_actual_durations_by_agent_by_team.shape
        == (3, MAX_AGENT_SLOTS, 2, MAX_AGENTS_PER_TEAM)
    )
    assert scanned_masks.select_target_use_ultimate_joint_mask.shape == (
        3,
        MAX_AGENT_SLOTS,
        NUM_TARGET_ACTIONS,
        NUM_ULTIMATE_ACTIONS,
    )
    assert bool(jnp.all(scanned_facts.has_transition))
    assert bool(
        jnp.array_equal(
            scanned_facts.transition_start_step_count,
            jnp.asarray((0, 1, 2), dtype=jnp.int32),
        )
    )
    assert len(jax.tree_util.tree_leaves(scanned_facts)) == (
        _TRANSITION_FACT_LEAF_COUNT
    )
    scanned_shield_facts = scanned_facts.spawn_shield_facts
    assert scanned_shield_facts.was_active_at_transition_start_by_agent.shape == (
        3,
        MAX_AGENT_SLOTS,
    )
    assert scanned_shield_facts.expired_at_transition_end_by_agent.shape == (
        3,
        MAX_AGENT_SLOTS,
    )
    assert not bool(
        jnp.any(scanned_shield_facts.was_active_at_transition_start_by_agent)
    )
    assert not bool(jnp.any(scanned_shield_facts.expired_at_transition_end_by_agent))

    def discard_fact_rollout(
        initial_state: EnvState,
        initial_mask: ActionMask,
    ) -> tuple[EnvState, ActionMask]:
        """Model a training adapter that intentionally retains no Info payload."""

        def discard_body(
            carry: tuple[EnvState, ActionMask],
            inputs: tuple[Action, Array],
        ) -> tuple[tuple[EnvState, ActionMask], None]:
            current_state, current_mask = carry
            action, key = inputs
            next_state, _, _, _, next_mask, _ = step(
                config,
                current_state,
                current_mask,
                action,
                key,
            )
            return (next_state, next_mask), None

        return jax.lax.scan(
            discard_body,
            (initial_state, initial_mask),
            (scan_actions, scan_keys),
        )[0]

    discarded = cast(
        tuple[EnvState, ActionMask],
        jax.jit(discard_fact_rollout)(state, action_mask),
    )
    assert int(discarded[0].step_count) == 3
    assert jax.tree_util.tree_structure(discarded) == jax.tree_util.tree_structure(
        (state, action_mask)
    )
