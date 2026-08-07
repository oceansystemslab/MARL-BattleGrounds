"""Public out-of-combat regeneration proofs for Milestone 6 Checkpoint 2."""

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
    AGENT_FEATURE_CAPABILITY_OUT_OF_COMBAT_DELAY_STEPS,
    AGENT_FEATURE_CAPABILITY_OUT_OF_COMBAT_HEALTH_REGEN_FRACTION_PER_STEP,
    AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER,
    AGENT_FEATURE_STEPS_UNTIL_OUT_OF_COMBAT,
    HUNTER_CLASS_ID,
    MAGE_CLASS_ID,
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    MAX_OBSTACLE_SLOTS,
    MOVE_EAST,
    MOVE_STAY,
    NEUTRAL_CLASS_ID,
    NUM_SLOW_CHANNELS,
    NUM_STUN_CHANNELS,
    OBSTACLE_FEATURES,
    PRIEST_CLASS_ID,
    ROGUE_CLASS_ID,
    WARRIOR_CLASS_ID,
    Action,
    ActionMask,
    DoneFlags,
    EnvConfig,
    EnvState,
    Info,
    Observation,
    RegenerationTransitionFacts,
    Reward,
)

_TEAM_A_FIRST_SLOT = 0
_TEAM_A_SECOND_SLOT = 1
_TEAM_A_THIRD_SLOT = 2
_TEAM_A_FOURTH_SLOT = 3
_TEAM_B_FIRST_SLOT = MAX_AGENTS_PER_TEAM
_TEAM_B_SECOND_SLOT = MAX_AGENTS_PER_TEAM + 1

_TARGET_NONE = 0

_StepResult = tuple[
    EnvState,
    Observation,
    Reward,
    DoneFlags,
    ActionMask,
    Info,
]


def _empty_obstacles() -> Array:
    """Return an inactive fixed-size obstacle table."""
    return jnp.zeros((MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES), dtype=jnp.float32)


def _spawn_pad_positions() -> Array:
    """Return clear immutable pads for every fixed team-local slot."""
    team_a = jnp.asarray(
        ((3.0, 2.0), (3.0, 4.0), (3.0, 6.0), (3.0, 8.0), (3.0, 10.0)),
        dtype=jnp.float32,
    )
    team_b = jnp.asarray(
        ((15.0, 2.0), (15.0, 4.0), (15.0, 6.0), (15.0, 8.0), (15.0, 10.0)),
        dtype=jnp.float32,
    )
    return jnp.stack((team_a, team_b))


def _requested_roster(
    team_sizes: tuple[int, int],
    *class_rows: tuple[int, int],
) -> Array:
    """Return a padded Hunter roster with selected active class overrides."""
    roster = jnp.full((MAX_AGENT_SLOTS,), NEUTRAL_CLASS_ID, dtype=jnp.int32)
    roster = roster.at[: team_sizes[0]].set(HUNTER_CLASS_ID)
    roster = roster.at[MAX_AGENTS_PER_TEAM : MAX_AGENTS_PER_TEAM + team_sizes[1]].set(
        HUNTER_CLASS_ID
    )
    for slot, class_id in class_rows:
        roster = roster.at[slot].set(class_id)
    return roster


def _scenario(
    *class_rows: tuple[int, int],
    team_sizes: tuple[int, int] = (1, 1),
    periods: tuple[int, int] = (20, 20),
    shield_duration: int = 3,
    ordinary_movement_distance_scale: float = 0.25,
) -> tuple[EnvConfig, EnvState, Observation, ActionMask, Info]:
    """Build a deterministic fully visible public-step scenario."""
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
    config = EnvConfig(
        max_steps=100,
        map_width=20.0,
        map_height=12.0,
        obstacles=_empty_obstacles(),
        agent_profile=profile,
        ordinary_movement_distance_scale=ordinary_movement_distance_scale,
        team_spawn_pad_positions=_spawn_pad_positions(),
        spawn_shield_duration_steps=shield_duration,
        spawn_shield_movement_speed=2.0,
        team_respawn_wave_period_step_count=jnp.asarray(periods, dtype=jnp.int32),
    )
    state, observation, action_mask, info = reset(config, jax.random.key(1))
    return config, state, observation, action_mask, info


def _target_action_for_global_slot(actor_slot: int, recipient_slot: int) -> int:
    """Return the actor-relative target category for one global recipient."""
    actor_team = actor_slot // MAX_AGENTS_PER_TEAM
    recipient_team = recipient_slot // MAX_AGENTS_PER_TEAM
    recipient_local_slot = recipient_slot % MAX_AGENTS_PER_TEAM
    relation_offset = 0 if actor_team == recipient_team else MAX_AGENTS_PER_TEAM
    return 1 + relation_offset + recipient_local_slot


def _joint_action(*rows: tuple[int, int, int | None, int]) -> Action:
    """Return a canonical joint action with selected actor overrides.

    Each row is ``(actor_slot, move, recipient_global_slot, use_ultimate)``.
    ``None`` denotes target-none.
    """
    move = jnp.full((MAX_AGENT_SLOTS,), MOVE_STAY, dtype=jnp.int32)
    target = jnp.full((MAX_AGENT_SLOTS,), _TARGET_NONE, dtype=jnp.int32)
    ultimate = jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32)
    for actor_slot, move_action, recipient_slot, ultimate_action in rows:
        move = move.at[actor_slot].set(move_action)
        if recipient_slot is not None:
            target = target.at[actor_slot].set(
                _target_action_for_global_slot(actor_slot, recipient_slot)
            )
        ultimate = ultimate.at[actor_slot].set(ultimate_action)
    return Action(move=move, select_target=target, use_ultimate=ultimate)


def _observation_and_mask(
    config: EnvConfig, state: EnvState
) -> tuple[Observation, ActionMask]:
    """Return the public observation and authoritative mask for an authored state."""
    return _build_observation_and_action_mask(state, config)


def _take_step(
    config: EnvConfig,
    state: EnvState,
    action: Action | None = None,
    *,
    action_mask: ActionMask | None = None,
    key: Array | None = None,
) -> _StepResult:
    """Advance one deterministic public transition."""
    choosing_mask = (
        _observation_and_mask(config, state)[1] if action_mask is None else action_mask
    )
    return step(
        config,
        state,
        choosing_mask,
        _joint_action() if action is None else action,
        jax.random.key(2) if key is None else key,
    )


def _slot_mask(*slots: int) -> Array:
    """Return a fixed-slot boolean mask selecting exactly ``slots``."""
    mask = jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.bool_)
    for slot in slots:
        mask = mask.at[slot].set(True)
    return mask


def _with_dead_slot(state: EnvState, slot: int) -> EnvState:
    """Return one canonical authored corpse while retaining static profile truth."""
    return state._replace(
        alive_mask=state.alive_mask.at[slot].set(False),
        current_health=state.current_health.at[slot].set(0.0),
        slow_durations=state.slow_durations.at[slot].set(
            jnp.zeros((NUM_SLOW_CHANNELS,), dtype=jnp.int32)
        ),
        stun_durations=state.stun_durations.at[slot].set(
            jnp.zeros((NUM_STUN_CHANNELS,), dtype=jnp.int32)
        ),
        rogue_poison_anti_heal_durations=(
            state.rogue_poison_anti_heal_durations.at[slot].set(0)
        ),
        mage_burst_damage_amplification_durations=(
            state.mage_burst_damage_amplification_durations.at[slot].set(0)
        ),
        priest_blessing_of_freedom_slow_floor_durations=(
            state.priest_blessing_of_freedom_slow_floor_durations.at[slot].set(0)
        ),
        spawn_shield_durations=state.spawn_shield_durations.at[slot].set(0),
        steps_until_out_of_combat=state.steps_until_out_of_combat.at[slot].set(0),
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


def _stack_trees(*trees: object) -> object:
    """Stack identically structured PyTrees along a leading batch axis."""
    return jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *trees)


def _assert_close(actual: Array, expected: float) -> None:
    """Assert one scalar JAX value against a readable floating expectation."""
    assert float(actual) == pytest.approx(expected)


def test_catalogs_profile_reset_and_canonical_facts_publish_exact_contract() -> None:
    """Resolve exact class capabilities once and expose canonical reset truth."""
    expected_delays = jnp.asarray((0, 5, 5, 5, 3, 5), dtype=jnp.int32)
    expected_rates = jnp.asarray((0.0, 0.04, 0.04, 0.04, 0.04, 0.04), dtype=jnp.float32)

    assert combat.OUT_OF_COMBAT_DELAY_STEPS_BY_CLASS.shape == (6,)
    assert combat.OUT_OF_COMBAT_DELAY_STEPS_BY_CLASS.dtype == jnp.int32
    assert bool(
        jnp.array_equal(combat.OUT_OF_COMBAT_DELAY_STEPS_BY_CLASS, expected_delays)
    )
    assert (
        combat.OUT_OF_COMBAT_HEALTH_REGENERATION_FRACTION_PER_STEP_BY_CLASS.shape
        == (6,)
    )
    assert (
        combat.OUT_OF_COMBAT_HEALTH_REGENERATION_FRACTION_PER_STEP_BY_CLASS.dtype
        == jnp.float32
    )
    assert bool(
        jnp.array_equal(
            combat.OUT_OF_COMBAT_HEALTH_REGENERATION_FRACTION_PER_STEP_BY_CLASS,
            expected_rates,
        )
    )

    config, state, observation, _, info = _scenario(
        (_TEAM_A_FIRST_SLOT, MAGE_CLASS_ID),
        (_TEAM_A_SECOND_SLOT, WARRIOR_CLASS_ID),
        (_TEAM_A_THIRD_SLOT, HUNTER_CLASS_ID),
        (_TEAM_A_FOURTH_SLOT, ROGUE_CLASS_ID),
        (4, PRIEST_CLASS_ID),
        (_TEAM_B_FIRST_SLOT, MAGE_CLASS_ID),
        team_sizes=(5, 1),
    )
    expected_profile_delays = jnp.asarray(
        (5, 5, 5, 3, 5, 5, 0, 0, 0, 0), dtype=jnp.int32
    )
    expected_profile_rates = jnp.asarray(
        (0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.0, 0.0, 0.0, 0.0),
        dtype=jnp.float32,
    )

    assert bool(
        jnp.array_equal(
            config.agent_profile.out_of_combat_delay_steps,
            expected_profile_delays,
        )
    )
    assert bool(
        jnp.array_equal(
            config.agent_profile.out_of_combat_health_regen_fraction_per_step,
            expected_profile_rates,
        )
    )
    assert state.steps_until_out_of_combat.shape == (MAX_AGENT_SLOTS,)
    assert state.steps_until_out_of_combat.dtype == jnp.int32
    assert not bool(jnp.any(state.steps_until_out_of_combat))
    assert bool(
        jnp.array_equal(
            observation.self_features[
                :, AGENT_FEATURE_CAPABILITY_OUT_OF_COMBAT_DELAY_STEPS
            ],
            expected_profile_delays.astype(jnp.float32),
        )
    )
    assert bool(
        jnp.array_equal(
            observation.self_features[
                :,
                AGENT_FEATURE_CAPABILITY_OUT_OF_COMBAT_HEALTH_REGEN_FRACTION_PER_STEP,
            ],
            expected_profile_rates,
        )
    )

    facts = info.transition_facts.regeneration_facts
    assert isinstance(facts, RegenerationTransitionFacts)
    assert facts.combat_countdown_was_reset_by_agent.shape == (MAX_AGENT_SLOTS,)
    assert facts.combat_countdown_was_reset_by_agent.dtype == jnp.bool_
    assert not bool(jnp.any(facts.combat_countdown_was_reset_by_agent))
    assert facts.actual_health_regenerated_this_step_by_agent.shape == (
        MAX_AGENT_SLOTS,
    )
    assert facts.actual_health_regenerated_this_step_by_agent.dtype == jnp.float32
    assert not bool(jnp.any(facts.actual_health_regenerated_this_step_by_agent))


def test_hunter_trap_resets_class_delays_while_mage_burst_does_not() -> None:
    """Route Hunter damage universally while preserving zero-damage Mage Burst."""
    config, state, *_ = _scenario(
        (_TEAM_A_FIRST_SLOT, ROGUE_CLASS_ID),
        (_TEAM_A_SECOND_SLOT, MAGE_CLASS_ID),
        (_TEAM_B_FIRST_SLOT, HUNTER_CLASS_ID),
        team_sizes=(2, 1),
    )
    state = state._replace(
        current_health=state.current_health.at[_TEAM_A_SECOND_SLOT].set(40.0)
    )
    action = _joint_action(
        (_TEAM_A_SECOND_SLOT, MOVE_STAY, None, 1),
        (_TEAM_B_FIRST_SLOT, MOVE_STAY, _TEAM_A_FIRST_SLOT, 1),
    )

    next_state, observation, _, _, _, info = _take_step(config, state, action)
    facts = info.transition_facts

    assert bool(
        jnp.array_equal(
            facts.regeneration_facts.combat_countdown_was_reset_by_agent,
            _slot_mask(_TEAM_A_FIRST_SLOT, _TEAM_B_FIRST_SLOT),
        )
    )
    assert int(next_state.steps_until_out_of_combat[_TEAM_A_FIRST_SLOT]) == 3
    assert int(next_state.steps_until_out_of_combat[_TEAM_B_FIRST_SLOT]) == 5
    assert int(next_state.steps_until_out_of_combat[_TEAM_A_SECOND_SLOT]) == 0
    _assert_close(next_state.current_health[_TEAM_A_FIRST_SLOT], 90.0)
    _assert_close(next_state.current_health[_TEAM_A_SECOND_SLOT], 43.2)
    _assert_close(
        facts.combat_transition_facts.raw_damage_output_by_source[_TEAM_B_FIRST_SLOT],
        10.0,
    )
    _assert_close(
        facts.combat_transition_facts.raw_damage_output_by_source[_TEAM_A_SECOND_SLOT],
        0.0,
    )
    _assert_close(
        facts.regeneration_facts.actual_health_regenerated_this_step_by_agent[
            _TEAM_A_SECOND_SLOT
        ],
        3.2,
    )
    assert (
        int(
            observation.self_features[
                _TEAM_A_FIRST_SLOT, AGENT_FEATURE_STEPS_UNTIL_OUT_OF_COMBAT
            ]
        )
        == 3
    )


def test_countdown_five_to_zero_blocks_regeneration_until_the_next_selected_step() -> (
    None
):
    """Prove the complete public `5 -> ... -> 1 -> 0 -> regenerate` trajectory."""
    config, state, observation, *_ = _scenario()
    state = state._replace(
        current_health=state.current_health.at[_TEAM_A_FIRST_SLOT].set(50.0),
        steps_until_out_of_combat=state.steps_until_out_of_combat.at[
            _TEAM_A_FIRST_SLOT
        ].set(5),
    )
    observation, _ = _observation_and_mask(config, state)
    assert (
        int(
            observation.self_features[
                _TEAM_A_FIRST_SLOT, AGENT_FEATURE_STEPS_UNTIL_OUT_OF_COMBAT
            ]
        )
        == 5
    )

    expected_countdowns = (4, 3, 2, 1, 0, 0)
    expected_health = (50.0, 50.0, 50.0, 50.0, 50.0, 54.0)
    expected_actual_regeneration = (0.0, 0.0, 0.0, 0.0, 0.0, 4.0)
    quiet = _joint_action()

    for expected_countdown, health, actual_regeneration in zip(
        expected_countdowns,
        expected_health,
        expected_actual_regeneration,
        strict=True,
    ):
        state, observation, _, _, _, info = _take_step(config, state, quiet)
        assert (
            int(state.steps_until_out_of_combat[_TEAM_A_FIRST_SLOT])
            == expected_countdown
        )
        assert (
            int(
                observation.self_features[
                    _TEAM_A_FIRST_SLOT, AGENT_FEATURE_STEPS_UNTIL_OUT_OF_COMBAT
                ]
            )
            == expected_countdown
        )
        _assert_close(state.current_health[_TEAM_A_FIRST_SLOT], health)
        _assert_close(
            info.transition_facts.regeneration_facts.actual_health_regenerated_this_step_by_agent[
                _TEAM_A_FIRST_SLOT
            ],
            actual_regeneration,
        )


def test_delay_zero_damage_still_blocks_regeneration_on_the_interaction_step() -> None:
    """A delay-zero ablation records the reset and regenerates only next step."""
    config, state, *_ = _scenario()
    delay_probe = config.agent_profile.out_of_combat_delay_steps.at[
        _TEAM_A_FIRST_SLOT
    ].set(0)
    config = config._replace(
        agent_profile=config.agent_profile._replace(
            out_of_combat_delay_steps=delay_probe
        )
    )
    state = state._replace(
        current_health=state.current_health.at[_TEAM_A_FIRST_SLOT].set(50.0)
    )
    attack = _joint_action((_TEAM_A_FIRST_SLOT, MOVE_STAY, _TEAM_B_FIRST_SLOT, 0))

    interaction_state, *_, interaction_info = _take_step(config, state, attack)
    assert int(interaction_state.steps_until_out_of_combat[_TEAM_A_FIRST_SLOT]) == 0
    _assert_close(interaction_state.current_health[_TEAM_A_FIRST_SLOT], 50.0)
    assert bool(
        interaction_info.transition_facts.regeneration_facts.combat_countdown_was_reset_by_agent[
            _TEAM_A_FIRST_SLOT
        ]
    )
    _assert_close(
        interaction_info.transition_facts.regeneration_facts.actual_health_regenerated_this_step_by_agent[
            _TEAM_A_FIRST_SLOT
        ],
        0.0,
    )

    recovered_state, *_, recovered_info = _take_step(
        config, interaction_state, _joint_action()
    )
    _assert_close(recovered_state.current_health[_TEAM_A_FIRST_SLOT], 54.0)
    _assert_close(
        recovered_info.transition_facts.regeneration_facts.actual_health_regenerated_this_step_by_agent[
            _TEAM_A_FIRST_SLOT
        ],
        4.0,
    )


def test_delay_one_expires_before_the_first_regeneration_transition() -> None:
    """A delay-one tuning probe preserves the public `1 -> 0 -> regenerate` rule."""
    config, state, *_ = _scenario()
    delay_probe = config.agent_profile.out_of_combat_delay_steps.at[
        _TEAM_A_FIRST_SLOT
    ].set(1)
    config = config._replace(
        agent_profile=config.agent_profile._replace(
            out_of_combat_delay_steps=delay_probe
        )
    )
    state = state._replace(
        current_health=state.current_health.at[_TEAM_A_FIRST_SLOT].set(50.0)
    )
    attack = _joint_action((_TEAM_A_FIRST_SLOT, MOVE_STAY, _TEAM_B_FIRST_SLOT, 0))

    engaged_state, *_, engaged_info = _take_step(config, state, attack)
    assert int(engaged_state.steps_until_out_of_combat[_TEAM_A_FIRST_SLOT]) == 1
    _assert_close(engaged_state.current_health[_TEAM_A_FIRST_SLOT], 50.0)
    assert bool(
        engaged_info.transition_facts.regeneration_facts.combat_countdown_was_reset_by_agent[
            _TEAM_A_FIRST_SLOT
        ]
    )

    expired_state, *_, expired_info = _take_step(config, engaged_state, _joint_action())
    assert int(expired_state.steps_until_out_of_combat[_TEAM_A_FIRST_SLOT]) == 0
    _assert_close(expired_state.current_health[_TEAM_A_FIRST_SLOT], 50.0)
    _assert_close(
        expired_info.transition_facts.regeneration_facts.actual_health_regenerated_this_step_by_agent[
            _TEAM_A_FIRST_SLOT
        ],
        0.0,
    )

    recovered_state, *_, recovered_info = _take_step(
        config, expired_state, _joint_action()
    )
    _assert_close(recovered_state.current_health[_TEAM_A_FIRST_SLOT], 54.0)
    _assert_close(
        recovered_info.transition_facts.regeneration_facts.actual_health_regenerated_this_step_by_agent[
            _TEAM_A_FIRST_SLOT
        ],
        4.0,
    )


def test_ooc_heal_uses_snapshot_then_qualifies_next_transition() -> None:
    """Prove the canonical two-transition heal/damage causal trajectory."""
    config, state, *_ = _scenario(
        (_TEAM_A_FIRST_SLOT, PRIEST_CLASS_ID),
        (_TEAM_A_SECOND_SLOT, HUNTER_CLASS_ID),
        (_TEAM_B_FIRST_SLOT, HUNTER_CLASS_ID),
        team_sizes=(2, 1),
    )
    state = state._replace(
        current_health=(
            state.current_health.at[_TEAM_A_FIRST_SLOT]
            .set(50.0)
            .at[_TEAM_A_SECOND_SLOT]
            .set(50.0)
        )
    )
    opening_action = _joint_action(
        (_TEAM_A_FIRST_SLOT, MOVE_STAY, _TEAM_A_SECOND_SLOT, 0),
        (_TEAM_B_FIRST_SLOT, MOVE_STAY, _TEAM_A_SECOND_SLOT, 0),
    )

    engaged_state, engaged_observation, _, _, _, opening_info = _take_step(
        config, state, opening_action
    )
    opening_regeneration_facts = opening_info.transition_facts.regeneration_facts
    assert bool(
        jnp.array_equal(
            opening_regeneration_facts.combat_countdown_was_reset_by_agent,
            _slot_mask(_TEAM_A_SECOND_SLOT, _TEAM_B_FIRST_SLOT),
        )
    )
    assert int(engaged_state.steps_until_out_of_combat[_TEAM_A_FIRST_SLOT]) == 0
    assert int(engaged_state.steps_until_out_of_combat[_TEAM_A_SECOND_SLOT]) == 5
    _assert_close(engaged_state.current_health[_TEAM_A_FIRST_SLOT], 54.0)
    _assert_close(engaged_state.current_health[_TEAM_A_SECOND_SLOT], 52.0)
    _assert_close(
        opening_regeneration_facts.actual_health_regenerated_this_step_by_agent[
            _TEAM_A_FIRST_SLOT
        ],
        4.0,
    )
    assert (
        int(
            engaged_observation.self_features[
                _TEAM_A_SECOND_SLOT, AGENT_FEATURE_STEPS_UNTIL_OUT_OF_COMBAT
            ]
        )
        == 5
    )

    repeated_heal = _joint_action(
        (_TEAM_A_FIRST_SLOT, MOVE_STAY, _TEAM_A_SECOND_SLOT, 0)
    )
    supported_state, *_, repeated_info = _take_step(
        config, engaged_state, repeated_heal
    )
    repeated_regeneration_facts = repeated_info.transition_facts.regeneration_facts
    assert bool(
        jnp.array_equal(
            repeated_regeneration_facts.combat_countdown_was_reset_by_agent,
            _slot_mask(_TEAM_A_FIRST_SLOT, _TEAM_A_SECOND_SLOT),
        )
    )
    assert int(supported_state.steps_until_out_of_combat[_TEAM_A_FIRST_SLOT]) == 5
    assert int(supported_state.steps_until_out_of_combat[_TEAM_A_SECOND_SLOT]) == 5
    _assert_close(supported_state.current_health[_TEAM_A_FIRST_SLOT], 54.0)
    _assert_close(supported_state.current_health[_TEAM_A_SECOND_SLOT], 60.0)
    _assert_close(
        repeated_regeneration_facts.actual_health_regenerated_this_step_by_agent[
            _TEAM_A_FIRST_SLOT
        ],
        0.0,
    )


def test_ooc_heal_target_dealing_damage_does_not_reset_healer() -> None:
    """Snapshot qualification ignores the recipient's simultaneous attack."""
    config, state, *_ = _scenario(
        (_TEAM_A_FIRST_SLOT, PRIEST_CLASS_ID),
        (_TEAM_A_SECOND_SLOT, HUNTER_CLASS_ID),
        (_TEAM_B_FIRST_SLOT, HUNTER_CLASS_ID),
        team_sizes=(2, 1),
    )
    state = state._replace(
        current_health=(
            state.current_health.at[_TEAM_A_FIRST_SLOT]
            .set(50.0)
            .at[_TEAM_A_SECOND_SLOT]
            .set(50.0)
        )
    )
    action = _joint_action(
        (_TEAM_A_FIRST_SLOT, MOVE_STAY, _TEAM_A_SECOND_SLOT, 0),
        (_TEAM_A_SECOND_SLOT, MOVE_STAY, _TEAM_B_FIRST_SLOT, 0),
    )

    next_state, *_, info = _take_step(config, state, action)
    regeneration_facts = info.transition_facts.regeneration_facts
    assert bool(
        jnp.array_equal(
            regeneration_facts.combat_countdown_was_reset_by_agent,
            _slot_mask(_TEAM_A_SECOND_SLOT, _TEAM_B_FIRST_SLOT),
        )
    )
    assert int(next_state.steps_until_out_of_combat[_TEAM_A_FIRST_SLOT]) == 0
    assert int(next_state.steps_until_out_of_combat[_TEAM_A_SECOND_SLOT]) == 5
    _assert_close(next_state.current_health[_TEAM_A_FIRST_SLOT], 54.0)
    _assert_close(next_state.current_health[_TEAM_A_SECOND_SLOT], 58.0)
    _assert_close(next_state.current_health[_TEAM_B_FIRST_SLOT], 94.0)
    _assert_close(
        regeneration_facts.actual_health_regenerated_this_step_by_agent[
            _TEAM_A_FIRST_SLOT
        ],
        4.0,
    )
    _assert_close(
        regeneration_facts.actual_health_regenerated_this_step_by_agent[
            _TEAM_A_SECOND_SLOT
        ],
        0.0,
    )


def test_lethal_damage_to_ooc_heal_target_does_not_retroactively_reset_healer() -> None:
    """Lethal same-step damage cannot change healing qualification."""
    config, state, *_ = _scenario(
        (_TEAM_A_FIRST_SLOT, PRIEST_CLASS_ID),
        (_TEAM_A_SECOND_SLOT, HUNTER_CLASS_ID),
        (_TEAM_B_FIRST_SLOT, HUNTER_CLASS_ID),
        team_sizes=(2, 1),
    )
    state = state._replace(
        current_health=(
            state.current_health.at[_TEAM_A_FIRST_SLOT]
            .set(50.0)
            .at[_TEAM_A_SECOND_SLOT]
            .set(1.0)
        )
    )
    action = _joint_action(
        (_TEAM_A_FIRST_SLOT, MOVE_STAY, _TEAM_A_SECOND_SLOT, 0),
        (_TEAM_B_FIRST_SLOT, MOVE_STAY, _TEAM_A_SECOND_SLOT, 1),
    )

    next_state, *_, info = _take_step(config, state, action)
    regeneration_facts = info.transition_facts.regeneration_facts
    assert bool(
        jnp.array_equal(
            regeneration_facts.combat_countdown_was_reset_by_agent,
            _slot_mask(_TEAM_A_SECOND_SLOT, _TEAM_B_FIRST_SLOT),
        )
    )
    assert not bool(next_state.alive_mask[_TEAM_A_SECOND_SLOT])
    _assert_close(next_state.current_health[_TEAM_A_SECOND_SLOT], 0.0)
    assert int(next_state.steps_until_out_of_combat[_TEAM_A_SECOND_SLOT]) == 0
    assert bool(
        info.transition_facts.death_facts.is_newly_dead_by_recipient[
            _TEAM_A_SECOND_SLOT
        ]
    )
    _assert_close(next_state.current_health[_TEAM_A_FIRST_SLOT], 54.0)
    _assert_close(
        regeneration_facts.actual_health_regenerated_this_step_by_agent[
            _TEAM_A_FIRST_SLOT
        ],
        4.0,
    )


def test_overheal_and_multiple_healers_reset_all_snapshot_qualified_endpoints() -> None:
    """Positive raw overheal resets every healer and the already-IC recipient."""
    config, state, *_ = _scenario(
        (_TEAM_A_FIRST_SLOT, PRIEST_CLASS_ID),
        (_TEAM_A_SECOND_SLOT, PRIEST_CLASS_ID),
        (_TEAM_A_THIRD_SLOT, HUNTER_CLASS_ID),
        team_sizes=(3, 1),
    )
    state = state._replace(
        steps_until_out_of_combat=state.steps_until_out_of_combat.at[
            _TEAM_A_THIRD_SLOT
        ].set(1)
    )
    action = _joint_action(
        (_TEAM_A_FIRST_SLOT, MOVE_STAY, _TEAM_A_THIRD_SLOT, 0),
        (_TEAM_A_SECOND_SLOT, MOVE_STAY, _TEAM_A_THIRD_SLOT, 0),
    )

    next_state, *_, info = _take_step(config, state, action)
    combat_facts = info.transition_facts.combat_transition_facts
    regeneration_facts = info.transition_facts.regeneration_facts
    assert bool(
        jnp.array_equal(
            regeneration_facts.combat_countdown_was_reset_by_agent,
            _slot_mask(
                _TEAM_A_FIRST_SLOT,
                _TEAM_A_SECOND_SLOT,
                _TEAM_A_THIRD_SLOT,
            ),
        )
    )
    assert bool(
        jnp.all(
            next_state.steps_until_out_of_combat[
                jnp.asarray(
                    (
                        _TEAM_A_FIRST_SLOT,
                        _TEAM_A_SECOND_SLOT,
                        _TEAM_A_THIRD_SLOT,
                    )
                )
            ]
            == 5
        )
    )
    _assert_close(next_state.current_health[_TEAM_A_THIRD_SLOT], 100.0)
    _assert_close(combat_facts.raw_healing_output_by_source[_TEAM_A_FIRST_SLOT], 8.0)
    _assert_close(combat_facts.raw_healing_output_by_source[_TEAM_A_SECOND_SLOT], 8.0)
    _assert_close(
        combat_facts.total_effective_healing_by_recipient[_TEAM_A_THIRD_SLOT],
        16.0,
    )
    assert not bool(
        jnp.any(regeneration_facts.actual_health_regenerated_this_step_by_agent)
    )


@pytest.mark.parametrize(
    ("healer_a", "healer_b", "recipient_c"),
    ((0, 1, 2), (2, 0, 1)),
    ids=("ascending-slots", "permuted-slots"),
)
def test_snapshot_healing_chain_is_invariant_to_slot_permutation(
    healer_a: int,
    healer_b: int,
    recipient_c: int,
) -> None:
    """A newly reset middle healer cannot retroactively qualify an incoming heal."""
    config, state, *_ = _scenario(
        (healer_a, PRIEST_CLASS_ID),
        (healer_b, PRIEST_CLASS_ID),
        (recipient_c, HUNTER_CLASS_ID),
        team_sizes=(3, 1),
    )
    state = state._replace(
        current_health=state.current_health.at[healer_a].set(50.0),
        steps_until_out_of_combat=state.steps_until_out_of_combat.at[recipient_c].set(
            2
        ),
    )
    action = _joint_action(
        (healer_a, MOVE_STAY, healer_b, 0),
        (healer_b, MOVE_STAY, recipient_c, 0),
    )

    next_state, *_, info = _take_step(config, state, action)
    reset_mask = (
        info.transition_facts.regeneration_facts.combat_countdown_was_reset_by_agent
    )
    assert bool(jnp.array_equal(reset_mask, _slot_mask(healer_b, recipient_c)))
    assert int(next_state.steps_until_out_of_combat[healer_a]) == 0
    assert int(next_state.steps_until_out_of_combat[healer_b]) == 5
    assert int(next_state.steps_until_out_of_combat[recipient_c]) == 5
    _assert_close(next_state.current_health[healer_a], 54.0)
    _assert_close(
        info.transition_facts.regeneration_facts.actual_health_regenerated_this_step_by_agent[
            healer_a
        ],
        4.0,
    )


def test_damaged_healer_resets_while_ooc_recipient_regenerates() -> None:
    """Independent damage resets the healer without changing OOC heal qualification."""
    config, state, *_ = _scenario(
        (_TEAM_A_FIRST_SLOT, PRIEST_CLASS_ID),
        (_TEAM_A_SECOND_SLOT, HUNTER_CLASS_ID),
        (_TEAM_B_FIRST_SLOT, HUNTER_CLASS_ID),
        team_sizes=(2, 1),
    )
    state = state._replace(
        current_health=(
            state.current_health.at[_TEAM_A_FIRST_SLOT]
            .set(50.0)
            .at[_TEAM_A_SECOND_SLOT]
            .set(50.0)
        )
    )
    action = _joint_action(
        (_TEAM_A_FIRST_SLOT, MOVE_STAY, _TEAM_A_SECOND_SLOT, 0),
        (_TEAM_B_FIRST_SLOT, MOVE_STAY, _TEAM_A_FIRST_SLOT, 0),
    )

    next_state, *_, info = _take_step(config, state, action)
    regeneration_facts = info.transition_facts.regeneration_facts
    assert bool(
        jnp.array_equal(
            regeneration_facts.combat_countdown_was_reset_by_agent,
            _slot_mask(_TEAM_A_FIRST_SLOT, _TEAM_B_FIRST_SLOT),
        )
    )
    _assert_close(next_state.current_health[_TEAM_A_FIRST_SLOT], 44.0)
    _assert_close(next_state.current_health[_TEAM_A_SECOND_SLOT], 62.0)
    assert int(next_state.steps_until_out_of_combat[_TEAM_A_SECOND_SLOT]) == 0
    _assert_close(
        regeneration_facts.actual_health_regenerated_this_step_by_agent[
            _TEAM_A_SECOND_SLOT
        ],
        4.0,
    )


def test_regeneration_uses_class_maximum_clamp_profile_rate_and_start_anti_heal() -> (
    None
):
    """Cover zero/interior/one rates, class-relative amounts, clamp, and anti-heal."""
    config, state, *_ = _scenario(
        (_TEAM_A_FIRST_SLOT, MAGE_CLASS_ID),
        (_TEAM_A_SECOND_SLOT, WARRIOR_CLASS_ID),
        (_TEAM_A_THIRD_SLOT, HUNTER_CLASS_ID),
        (_TEAM_A_FOURTH_SLOT, ROGUE_CLASS_ID),
        (_TEAM_B_FIRST_SLOT, HUNTER_CLASS_ID),
        (_TEAM_B_SECOND_SLOT, HUNTER_CLASS_ID),
        team_sizes=(4, 2),
    )
    probe_rates = (
        config.agent_profile.out_of_combat_health_regen_fraction_per_step.at[
            _TEAM_B_FIRST_SLOT
        ]
        .set(0.0)
        .at[_TEAM_B_SECOND_SLOT]
        .set(1.0)
    )
    config = config._replace(
        agent_profile=config.agent_profile._replace(
            out_of_combat_health_regen_fraction_per_step=probe_rates
        )
    )
    health = (
        state.current_health.at[_TEAM_A_FIRST_SLOT]
        .set(40.0)
        .at[_TEAM_A_SECOND_SLOT]
        .set(100.0)
        .at[_TEAM_A_THIRD_SLOT]
        .set(99.0)
        .at[_TEAM_A_FOURTH_SLOT]
        .set(50.0)
        .at[_TEAM_B_FIRST_SLOT]
        .set(50.0)
        .at[_TEAM_B_SECOND_SLOT]
        .set(20.0)
    )
    state = state._replace(
        current_health=health,
        rogue_poison_anti_heal_durations=(
            state.rogue_poison_anti_heal_durations.at[_TEAM_A_FOURTH_SLOT].set(1)
        ),
    )

    next_state, observation, _, _, _, info = _take_step(config, state, _joint_action())
    regeneration_facts = info.transition_facts.regeneration_facts
    actual = regeneration_facts.actual_health_regenerated_this_step_by_agent
    expected_health = {
        _TEAM_A_FIRST_SLOT: 43.2,
        _TEAM_A_SECOND_SLOT: 108.0,
        _TEAM_A_THIRD_SLOT: 100.0,
        _TEAM_A_FOURTH_SLOT: 52.0,
        _TEAM_B_FIRST_SLOT: 50.0,
        _TEAM_B_SECOND_SLOT: 100.0,
    }
    expected_actual = {
        _TEAM_A_FIRST_SLOT: 3.2,
        _TEAM_A_SECOND_SLOT: 8.0,
        _TEAM_A_THIRD_SLOT: 1.0,
        _TEAM_A_FOURTH_SLOT: 2.0,
        _TEAM_B_FIRST_SLOT: 0.0,
        _TEAM_B_SECOND_SLOT: 80.0,
    }
    for slot, health_value in expected_health.items():
        _assert_close(next_state.current_health[slot], health_value)
        _assert_close(actual[slot], expected_actual[slot])

    _assert_close(
        observation.self_features[
            _TEAM_B_FIRST_SLOT,
            AGENT_FEATURE_CAPABILITY_OUT_OF_COMBAT_HEALTH_REGEN_FRACTION_PER_STEP,
        ],
        0.0,
    )
    _assert_close(
        observation.self_features[
            _TEAM_B_SECOND_SLOT,
            AGENT_FEATURE_CAPABILITY_OUT_OF_COMBAT_HEALTH_REGEN_FRACTION_PER_STEP,
        ],
        1.0,
    )
    assert not bool(
        jnp.any(
            info.transition_facts.regeneration_facts.combat_countdown_was_reset_by_agent
        )
    )


def test_movement_aura_coverage_and_spawn_shield_do_not_gate_regeneration() -> None:
    """Moving, aura-covered, and shielded OOC agents all use the universal rule."""
    config, state, *_ = _scenario(
        (_TEAM_A_FIRST_SLOT, MAGE_CLASS_ID),
        (_TEAM_A_SECOND_SLOT, HUNTER_CLASS_ID),
        (_TEAM_A_THIRD_SLOT, WARRIOR_CLASS_ID),
        team_sizes=(3, 1),
    )
    positions = state.agent_positions.at[_TEAM_A_SECOND_SLOT].set(
        jnp.asarray((3.0, 3.5), dtype=jnp.float32)
    )
    state = state._replace(
        agent_positions=positions,
        current_health=(
            state.current_health.at[_TEAM_A_SECOND_SLOT]
            .set(50.0)
            .at[_TEAM_A_THIRD_SLOT]
            .set(100.0)
        ),
        spawn_shield_durations=state.spawn_shield_durations.at[_TEAM_A_THIRD_SLOT].set(
            2
        ),
    )
    initial_observation, _ = _observation_and_mask(config, state)
    assert (
        float(
            initial_observation.self_features[
                _TEAM_A_SECOND_SLOT,
                AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER,
            ]
        )
        > 1.0
    )
    action = _joint_action(
        (_TEAM_A_SECOND_SLOT, MOVE_EAST, None, 0),
        (_TEAM_A_THIRD_SLOT, MOVE_EAST, None, 0),
    )

    next_state, *_, info = _take_step(config, state, action)
    assert float(next_state.agent_positions[_TEAM_A_SECOND_SLOT, 0]) > float(
        state.agent_positions[_TEAM_A_SECOND_SLOT, 0]
    )
    assert float(next_state.agent_positions[_TEAM_A_THIRD_SLOT, 0]) > float(
        state.agent_positions[_TEAM_A_THIRD_SLOT, 0]
    )
    _assert_close(next_state.current_health[_TEAM_A_SECOND_SLOT], 54.0)
    _assert_close(next_state.current_health[_TEAM_A_THIRD_SLOT], 108.0)
    assert int(next_state.spawn_shield_durations[_TEAM_A_THIRD_SLOT]) == 1
    assert not bool(
        jnp.any(
            info.transition_facts.regeneration_facts.combat_countdown_was_reset_by_agent
        )
    )
    _assert_close(
        info.transition_facts.regeneration_facts.actual_health_regenerated_this_step_by_agent[
            _TEAM_A_SECOND_SLOT
        ],
        4.0,
    )
    _assert_close(
        info.transition_facts.regeneration_facts.actual_health_regenerated_this_step_by_agent[
            _TEAM_A_THIRD_SLOT
        ],
        8.0,
    )


def test_death_wait_respawn_and_inactive_rows_canonicalize_countdown_and_facts() -> (
    None
):
    """Death and waiting stay at zero; due-wave respawn starts OOC at full health."""
    config, state, *_ = _scenario(
        (_TEAM_A_FIRST_SLOT, ROGUE_CLASS_ID),
        (_TEAM_B_FIRST_SLOT, HUNTER_CLASS_ID),
    )
    state = state._replace(
        current_health=state.current_health.at[_TEAM_A_FIRST_SLOT].set(1.0),
        steps_until_out_of_combat=state.steps_until_out_of_combat.at[
            _TEAM_A_FIRST_SLOT
        ].set(2),
    )
    lethal_action = _joint_action(
        (_TEAM_B_FIRST_SLOT, MOVE_STAY, _TEAM_A_FIRST_SLOT, 1)
    )

    dead_state, dead_observation, _, _, _, lethal_info = _take_step(
        config, state, lethal_action
    )
    assert not bool(dead_state.alive_mask[_TEAM_A_FIRST_SLOT])
    assert int(dead_state.steps_until_out_of_combat[_TEAM_A_FIRST_SLOT]) == 0
    assert bool(
        lethal_info.transition_facts.regeneration_facts.combat_countdown_was_reset_by_agent[
            _TEAM_A_FIRST_SLOT
        ]
    )
    _assert_close(
        dead_observation.self_features[
            _TEAM_A_FIRST_SLOT,
            AGENT_FEATURE_CAPABILITY_OUT_OF_COMBAT_DELAY_STEPS,
        ],
        3.0,
    )
    _assert_close(
        dead_observation.self_features[
            _TEAM_A_FIRST_SLOT,
            AGENT_FEATURE_CAPABILITY_OUT_OF_COMBAT_HEALTH_REGEN_FRACTION_PER_STEP,
        ],
        0.04,
    )
    assert int(dead_state.steps_until_out_of_combat[_TEAM_A_SECOND_SLOT]) == 0
    assert not bool(
        lethal_info.transition_facts.regeneration_facts.combat_countdown_was_reset_by_agent[
            _TEAM_A_SECOND_SLOT
        ]
    )
    _assert_close(
        dead_observation.self_features[
            _TEAM_A_SECOND_SLOT,
            AGENT_FEATURE_CAPABILITY_OUT_OF_COMBAT_HEALTH_REGEN_FRACTION_PER_STEP,
        ],
        0.0,
    )

    waiting_state, *_, waiting_info = _take_step(config, dead_state, _joint_action())
    assert not bool(waiting_state.alive_mask[_TEAM_A_FIRST_SLOT])
    assert int(waiting_state.steps_until_out_of_combat[_TEAM_A_FIRST_SLOT]) == 0
    _assert_close(
        waiting_info.transition_facts.regeneration_facts.actual_health_regenerated_this_step_by_agent[
            _TEAM_A_FIRST_SLOT
        ],
        0.0,
    )

    due_countdowns = waiting_state.team_respawn_wave_countdowns.at[0].set(0)
    due_state = waiting_state._replace(team_respawn_wave_countdowns=due_countdowns)
    respawned_state, respawned_observation, _, _, _, respawn_info = _take_step(
        config, due_state, _joint_action()
    )
    assert bool(respawned_state.alive_mask[_TEAM_A_FIRST_SLOT])
    _assert_close(respawned_state.current_health[_TEAM_A_FIRST_SLOT], 100.0)
    assert int(respawned_state.steps_until_out_of_combat[_TEAM_A_FIRST_SLOT]) == 0
    assert (
        int(respawned_state.spawn_shield_durations[_TEAM_A_FIRST_SLOT])
        == config.spawn_shield_duration_steps
    )
    assert bool(
        respawn_info.transition_facts.respawn_facts.was_respawned_this_transition_by_agent[
            _TEAM_A_FIRST_SLOT
        ]
    )
    _assert_close(
        respawn_info.transition_facts.regeneration_facts.actual_health_regenerated_this_step_by_agent[
            _TEAM_A_FIRST_SLOT
        ],
        0.0,
    )
    assert (
        int(
            respawned_observation.self_features[
                _TEAM_A_FIRST_SLOT, AGENT_FEATURE_STEPS_UNTIL_OUT_OF_COMBAT
            ]
        )
        == 0
    )


def test_observation_exposes_capabilities_and_zeros_hidden_rows() -> None:
    """Preserve public capabilities and ordinary visibility redaction."""
    config, state, *_ = _scenario(
        (_TEAM_A_FIRST_SLOT, ROGUE_CLASS_ID),
        (_TEAM_A_SECOND_SLOT, PRIEST_CLASS_ID),
        (_TEAM_B_FIRST_SLOT, HUNTER_CLASS_ID),
        (_TEAM_B_SECOND_SLOT, MAGE_CLASS_ID),
        team_sizes=(2, 2),
    )
    state = state._replace(
        steps_until_out_of_combat=(
            state.steps_until_out_of_combat.at[_TEAM_A_FIRST_SLOT]
            .set(3)
            .at[_TEAM_A_SECOND_SLOT]
            .set(5)
            .at[_TEAM_B_FIRST_SLOT]
            .set(5)
        )
    )
    visible_observation, _ = _observation_and_mask(config, state)

    assert (
        int(
            visible_observation.self_features[
                _TEAM_A_FIRST_SLOT, AGENT_FEATURE_STEPS_UNTIL_OUT_OF_COMBAT
            ]
        )
        == 3
    )
    assert (
        int(
            visible_observation.ally_unit_features[
                _TEAM_A_FIRST_SLOT,
                1,
                AGENT_FEATURE_STEPS_UNTIL_OUT_OF_COMBAT,
            ]
        )
        == 5
    )
    assert (
        int(
            visible_observation.enemy_unit_features[
                _TEAM_A_FIRST_SLOT,
                0,
                AGENT_FEATURE_STEPS_UNTIL_OUT_OF_COMBAT,
            ]
        )
        == 5
    )
    _assert_close(
        visible_observation.enemy_unit_features[
            _TEAM_A_FIRST_SLOT,
            0,
            AGENT_FEATURE_CAPABILITY_OUT_OF_COMBAT_DELAY_STEPS,
        ],
        5.0,
    )

    dead_and_hidden_state = _with_dead_slot(state, _TEAM_A_SECOND_SLOT)._replace(
        spawn_shield_durations=state.spawn_shield_durations.at[_TEAM_B_FIRST_SLOT].set(
            1
        )
    )
    redacted_observation, _ = _observation_and_mask(config, dead_and_hidden_state)
    _assert_close(
        redacted_observation.self_features[
            _TEAM_A_SECOND_SLOT,
            AGENT_FEATURE_CAPABILITY_OUT_OF_COMBAT_DELAY_STEPS,
        ],
        5.0,
    )
    _assert_close(
        redacted_observation.self_features[
            _TEAM_A_SECOND_SLOT,
            AGENT_FEATURE_CAPABILITY_OUT_OF_COMBAT_HEALTH_REGEN_FRACTION_PER_STEP,
        ],
        0.04,
    )
    assert bool(
        jnp.all(redacted_observation.ally_unit_features[_TEAM_A_FIRST_SLOT, 1] == 0)
    )
    assert bool(
        jnp.all(redacted_observation.enemy_unit_features[_TEAM_A_FIRST_SLOT, 0] == 0)
    )


def test_step_regeneration_outputs_match_eager_jit_and_shared_config_vmap() -> None:
    """Prove complete-output eager/JIT equality and mixed-countdown vmap timing."""
    config, state_zero, *_ = _scenario()
    state_zero = state_zero._replace(
        current_health=state_zero.current_health.at[_TEAM_A_FIRST_SLOT].set(50.0)
    )
    state_one = state_zero._replace(
        steps_until_out_of_combat=state_zero.steps_until_out_of_combat.at[
            _TEAM_A_FIRST_SLOT
        ].set(1)
    )
    quiet = _joint_action()
    mask_zero = _observation_and_mask(config, state_zero)[1]
    mask_one = _observation_and_mask(config, state_one)[1]
    key = jax.random.key(31)

    eager = step(config, state_zero, mask_zero, quiet, key)
    compiled = cast(
        _StepResult,
        jax.jit(step)(config, state_zero, mask_zero, quiet, key),
    )
    _assert_tree_equal(eager, compiled)

    batched_states = cast(EnvState, _stack_trees(state_zero, state_one))
    batched_masks = cast(ActionMask, _stack_trees(mask_zero, mask_one))
    batched_actions = cast(Action, _stack_trees(quiet, quiet))
    batched_keys = jax.random.split(jax.random.key(32), 2)

    def _shared_config_step(
        scenario_state: EnvState,
        scenario_mask: ActionMask,
        scenario_action: Action,
        scenario_key: Array,
    ) -> _StepResult:
        return step(
            config,
            scenario_state,
            scenario_mask,
            scenario_action,
            scenario_key,
        )

    (
        next_states,
        _,
        _,
        _,
        _,
        batched_info,
    ) = jax.vmap(_shared_config_step)(
        batched_states,
        batched_masks,
        batched_actions,
        batched_keys,
    )

    assert bool(
        jnp.array_equal(
            next_states.steps_until_out_of_combat[:, _TEAM_A_FIRST_SLOT],
            jnp.asarray((0, 0), dtype=jnp.int32),
        )
    )
    assert bool(
        jnp.allclose(
            next_states.current_health[:, _TEAM_A_FIRST_SLOT],
            jnp.asarray((54.0, 50.0), dtype=jnp.float32),
        )
    )
    assert bool(
        jnp.allclose(
            batched_info.transition_facts.regeneration_facts.actual_health_regenerated_this_step_by_agent[
                :, _TEAM_A_FIRST_SLOT
            ],
            jnp.asarray((4.0, 0.0), dtype=jnp.float32),
        )
    )


def test_scan_preserves_reset_expiry_and_first_regeneration() -> None:
    """Scan one attack and six quiet actions through the public state/mask carry."""
    config, state, _, action_mask, _ = _scenario()
    state = state._replace(
        current_health=state.current_health.at[_TEAM_B_FIRST_SLOT].set(50.0)
    )
    action_mask = _observation_and_mask(config, state)[1]
    attack = _joint_action((_TEAM_A_FIRST_SLOT, MOVE_STAY, _TEAM_B_FIRST_SLOT, 0))
    quiet = _joint_action()
    actions = cast(
        Action,
        _stack_trees(attack, quiet, quiet, quiet, quiet, quiet, quiet),
    )
    keys = jax.random.split(jax.random.key(41), 7)

    def _scan_step(
        carry: tuple[EnvState, ActionMask],
        scan_inputs: tuple[Action, Array],
    ) -> tuple[
        tuple[EnvState, ActionMask],
        tuple[Array, Array, Array, Array],
    ]:
        current_state, current_mask = carry
        action, key = scan_inputs
        next_state, _, _, _, next_mask, info = step(
            config,
            current_state,
            current_mask,
            action,
            key,
        )
        return (
            (next_state, next_mask),
            (
                next_state.steps_until_out_of_combat,
                next_state.current_health,
                info.transition_facts.regeneration_facts.combat_countdown_was_reset_by_agent,
                info.transition_facts.regeneration_facts.actual_health_regenerated_this_step_by_agent,
            ),
        )

    (
        (_, _),
        (
            countdown_history,
            health_history,
            reset_history,
            actual_regeneration_history,
        ),
    ) = jax.lax.scan(_scan_step, (state, action_mask), (actions, keys))

    assert bool(
        jnp.array_equal(
            countdown_history[:, _TEAM_B_FIRST_SLOT],
            jnp.asarray((5, 4, 3, 2, 1, 0, 0), dtype=jnp.int32),
        )
    )
    assert bool(
        jnp.allclose(
            health_history[:, _TEAM_B_FIRST_SLOT],
            jnp.asarray((44.0, 44.0, 44.0, 44.0, 44.0, 44.0, 48.0)),
        )
    )
    assert bool(
        jnp.array_equal(
            reset_history[:, _TEAM_B_FIRST_SLOT],
            jnp.asarray((True, False, False, False, False, False, False)),
        )
    )
    assert bool(
        jnp.allclose(
            actual_regeneration_history[:, _TEAM_B_FIRST_SLOT],
            jnp.asarray((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0)),
        )
    )
