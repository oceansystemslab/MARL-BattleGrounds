"""Public respawn-wave lifecycle proofs for Milestone 6 CP1."""

# pyright: reportPrivateUsage=false

from typing import cast

import jax
import jax.numpy as jnp
import pytest
from jax import Array

from marl_battlegrounds.core.config import resolve_agent_profile
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
    NEUTRAL_CLASS_ID,
    NUM_SLOW_CHANNELS,
    NUM_STUN_CHANNELS,
    NUM_TEAMS,
    OBSTACLE_FEATURES,
    Action,
    ActionMask,
    DoneFlags,
    EnvConfig,
    EnvState,
    Info,
    Observation,
    RespawnTransitionFacts,
    Reward,
    SpawnLifecycleObservation,
)

_TEAM_A_FIRST_SLOT = 0
_TEAM_A_SECOND_SLOT = 1
_TEAM_B_FIRST_SLOT = MAX_AGENTS_PER_TEAM
_TEAM_B_SECOND_SLOT = MAX_AGENTS_PER_TEAM + 1

_TARGET_NONE = 0
_FIRST_ENEMY_TARGET = 1 + MAX_AGENTS_PER_TEAM

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
    """Return clear immutable pads for every global slot."""
    team_a = jnp.asarray(
        ((3.0, 3.0), (5.0, 3.0), (7.0, 3.0), (9.0, 3.0), (11.0, 3.0)),
        dtype=jnp.float32,
    )
    team_b = jnp.asarray(
        ((3.0, 13.0), (5.0, 13.0), (7.0, 13.0), (9.0, 13.0), (11.0, 13.0)),
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
    periods: tuple[int, int] = (3, 5),
    shield_duration: int = 3,
) -> tuple[EnvConfig, EnvState, Observation, ActionMask, Info]:
    """Build a deterministic fully observable respawn scenario."""
    profile = resolve_agent_profile(
        _requested_roster(team_sizes, *class_rows),
        jnp.asarray(team_sizes, dtype=jnp.int32),
    )
    profile = profile._replace(
        observation_radii=jnp.where(profile.active_mask, 30.0, 0.0).astype(jnp.float32),
        basic_interaction_radii=jnp.where(profile.active_mask, 30.0, 0.0).astype(
            jnp.float32
        ),
        ultimate_interaction_radii=jnp.where(profile.active_mask, 30.0, 0.0).astype(
            jnp.float32
        ),
    )
    config = EnvConfig(
        max_steps=100,
        map_width=24.0,
        map_height=16.0,
        obstacles=_empty_obstacles(),
        agent_profile=profile,
        ordinary_movement_distance_scale=0.25,
        team_spawn_pad_positions=_spawn_pad_positions(),
        spawn_shield_duration_steps=shield_duration,
        spawn_shield_movement_speed=2.0,
        team_respawn_wave_period_step_count=jnp.asarray(periods, dtype=jnp.int32),
    )
    state, observation, action_mask, info = reset(config, jax.random.key(1))
    return config, state, observation, action_mask, info


def _joint_action(*rows: tuple[int, int, int, int]) -> Action:
    """Return a canonical joint action with selected actor overrides."""
    move = jnp.full((MAX_AGENT_SLOTS,), MOVE_STAY, dtype=jnp.int32)
    target = jnp.full((MAX_AGENT_SLOTS,), _TARGET_NONE, dtype=jnp.int32)
    ultimate = jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32)
    for slot, move_action, target_action, ultimate_action in rows:
        move = move.at[slot].set(move_action)
        target = target.at[slot].set(target_action)
        ultimate = ultimate.at[slot].set(ultimate_action)
    return Action(move=move, select_target=target, use_ultimate=ultimate)


def _current_action_mask(config: EnvConfig, state: EnvState) -> ActionMask:
    """Return the authoritative mask paired with one directly authored state."""
    return _build_observation_and_action_mask(state, config)[1]


def _take_step(
    config: EnvConfig,
    state: EnvState,
    action: Action | None = None,
    *,
    action_mask: ActionMask | None = None,
    key: Array | None = None,
) -> _StepResult:
    """Advance one deterministic public transition."""
    return step(
        config,
        state,
        _current_action_mask(config, state) if action_mask is None else action_mask,
        _joint_action() if action is None else action,
        jax.random.key(2) if key is None else key,
    )


def _with_dead_slots(state: EnvState, *slots: int) -> EnvState:
    """Author active corpses with canonical dead health and transient status."""
    alive_mask = state.alive_mask
    health = state.current_health
    slow = state.slow_durations
    stun = state.stun_durations
    anti_heal = state.rogue_poison_anti_heal_durations
    mage_burst = state.mage_burst_damage_amplification_durations
    freedom = state.priest_blessing_of_freedom_slow_floor_durations
    shield = state.spawn_shield_durations
    for slot in slots:
        alive_mask = alive_mask.at[slot].set(False)
        health = health.at[slot].set(0.0)
        slow = slow.at[slot].set(jnp.zeros((NUM_SLOW_CHANNELS,), dtype=jnp.int32))
        stun = stun.at[slot].set(jnp.zeros((NUM_STUN_CHANNELS,), dtype=jnp.int32))
        anti_heal = anti_heal.at[slot].set(0)
        mage_burst = mage_burst.at[slot].set(0)
        freedom = freedom.at[slot].set(0)
        shield = shield.at[slot].set(0)
    return state._replace(
        alive_mask=alive_mask,
        current_health=health,
        slow_durations=slow,
        stun_durations=stun,
        rogue_poison_anti_heal_durations=anti_heal,
        mage_burst_damage_amplification_durations=mage_burst,
        priest_blessing_of_freedom_slow_floor_durations=freedom,
        spawn_shield_durations=shield,
    )


def _slot_mask(*slots: int) -> Array:
    """Return a fixed-slot boolean mask selecting exactly ``slots``."""
    mask = jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.bool_)
    for slot in slots:
        mask = mask.at[slot].set(True)
    return mask


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


def _assert_statuses_are_clear(state: EnvState, slot: int) -> None:
    """Assert every transient status family is zero for one slot."""
    assert bool(jnp.all(state.slow_durations[slot] == 0))
    assert bool(jnp.all(state.stun_durations[slot] == 0))
    assert int(state.rogue_poison_anti_heal_durations[slot]) == 0
    assert int(state.mage_burst_damage_amplification_durations[slot]) == 0
    assert int(state.priest_blessing_of_freedom_slow_floor_durations[slot]) == 0


def _assert_inactive_observer_rows_are_zero(
    lifecycle: SpawnLifecycleObservation,
    active_mask: Array,
) -> None:
    """Assert every lifecycle leaf masks inactive observer rows to zero."""
    inactive_observer_mask = jnp.logical_not(active_mask)
    for leaf in jax.tree_util.tree_leaves(lifecycle):
        assert leaf.shape[0] == MAX_AGENT_SLOTS
        assert bool(jnp.all(leaf[inactive_observer_mask] == 0))


def test_reset_exposes_canonical_clock_observation_and_respawn_facts() -> None:
    """Reset publishes actor-relative clocks and canonical no-transition facts."""
    config, state, observation, _, info = _scenario(
        team_sizes=(2, 1),
        periods=(2, 5),
    )

    assert state.team_respawn_wave_countdowns.shape == (NUM_TEAMS,)
    assert state.team_respawn_wave_countdowns.dtype == jnp.int32
    assert bool(
        jnp.array_equal(
            state.team_respawn_wave_countdowns,
            jnp.asarray((1, 4), dtype=jnp.int32),
        )
    )

    lifecycle = observation.spawn_lifecycle
    assert bool(
        jnp.array_equal(
            lifecycle.respawn_wave_period_step_count_by_agent_by_team[
                _TEAM_A_FIRST_SLOT
            ],
            jnp.asarray((2, 5), dtype=jnp.int32),
        )
    )
    assert bool(
        jnp.array_equal(
            lifecycle.respawn_wave_period_step_count_by_agent_by_team[
                _TEAM_B_FIRST_SLOT
            ],
            jnp.asarray((5, 2), dtype=jnp.int32),
        )
    )
    assert bool(
        jnp.array_equal(
            lifecycle.respawn_wave_countdowns_by_agent_by_team[_TEAM_A_SECOND_SLOT],
            jnp.asarray((1, 4), dtype=jnp.int32),
        )
    )
    assert bool(
        jnp.array_equal(
            lifecycle.respawn_wave_countdowns_by_agent_by_team[_TEAM_B_FIRST_SLOT],
            jnp.asarray((4, 1), dtype=jnp.int32),
        )
    )
    assert bool(
        jnp.array_equal(
            lifecycle.spawn_pad_positions_by_agent_by_team[_TEAM_A_FIRST_SLOT],
            config.team_spawn_pad_positions,
        )
    )
    assert bool(
        jnp.array_equal(
            lifecycle.spawn_pad_positions_by_agent_by_team[_TEAM_B_FIRST_SLOT],
            config.team_spawn_pad_positions[jnp.asarray((1, 0))],
        )
    )
    expected_team_a_view = jnp.asarray(
        (
            (True, True, False, False, False),
            (True, False, False, False, False),
        ),
        dtype=jnp.bool_,
    )
    expected_team_b_view = expected_team_a_view[jnp.asarray((1, 0))]
    assert bool(
        jnp.array_equal(
            lifecycle.active_mask_by_agent_by_team[_TEAM_A_FIRST_SLOT],
            expected_team_a_view,
        )
    )
    assert bool(
        jnp.array_equal(
            lifecycle.active_mask_by_agent_by_team[_TEAM_B_FIRST_SLOT],
            expected_team_b_view,
        )
    )
    assert bool(
        jnp.array_equal(
            lifecycle.alive_mask_by_agent_by_team[_TEAM_A_FIRST_SLOT],
            expected_team_a_view,
        )
    )
    assert bool(
        jnp.array_equal(
            lifecycle.alive_mask_by_agent_by_team[_TEAM_B_FIRST_SLOT],
            expected_team_b_view,
        )
    )
    _assert_inactive_observer_rows_are_zero(
        lifecycle,
        config.agent_profile.active_mask,
    )

    facts = info.transition_facts
    assert RespawnTransitionFacts._fields == (
        "respawn_wave_occurred_this_transition_by_team",
        "was_respawned_this_transition_by_agent",
    )
    assert not bool(facts.has_transition)
    assert int(facts.transition_start_step_count) == -1
    assert facts.respawn_facts.respawn_wave_occurred_this_transition_by_team.shape == (
        NUM_TEAMS,
    )
    assert (
        facts.respawn_facts.respawn_wave_occurred_this_transition_by_team.dtype
        == jnp.bool_
    )
    assert facts.respawn_facts.was_respawned_this_transition_by_agent.shape == (
        MAX_AGENT_SLOTS,
    )
    assert facts.respawn_facts.was_respawned_this_transition_by_agent.dtype == jnp.bool_
    assert not bool(
        jnp.any(facts.respawn_facts.respawn_wave_occurred_this_transition_by_team)
    )
    assert not bool(jnp.any(facts.respawn_facts.was_respawned_this_transition_by_agent))


@pytest.mark.parametrize(
    ("period", "countdown", "expected_successor", "wave_due"),
    (
        pytest.param(1, 0, 0, True, id="period-one-countdown-zero"),
        pytest.param(4, 0, 3, True, id="period-many-countdown-zero"),
        pytest.param(4, 1, 0, False, id="period-many-countdown-one"),
        pytest.param(4, 3, 2, False, id="period-many-countdown-n"),
    ),
)
def test_team_clock_advances_through_zero_one_and_n_on_empty_waves(
    period: int,
    countdown: int,
    expected_successor: int,
    wave_due: bool,
) -> None:
    """A team clock advances independently even when no corpse can respawn."""
    config, state, _, _, _ = _scenario(periods=(period, 5))
    state = state._replace(
        team_respawn_wave_countdowns=jnp.asarray((countdown, 4), dtype=jnp.int32)
    )

    next_state, _, _, _, _, info = _take_step(config, state)

    assert int(next_state.team_respawn_wave_countdowns[0]) == expected_successor
    assert int(next_state.team_respawn_wave_countdowns[1]) == 3
    assert (
        bool(
            info.transition_facts.respawn_facts.respawn_wave_occurred_this_transition_by_team[
                0
            ]
        )
        is wave_due
    )
    assert not bool(
        info.transition_facts.respawn_facts.respawn_wave_occurred_this_transition_by_team[
            1
        ]
    )
    assert not bool(
        jnp.any(
            info.transition_facts.respawn_facts.was_respawned_this_transition_by_agent
        )
    )


def test_period_one_populated_then_empty_wave_is_due_every_transition() -> None:
    """Period one respawns a corpse and stays due when no corpse remains."""
    config, state, _, _, _ = _scenario(
        team_sizes=(1, 1),
        periods=(1, 4),
        shield_duration=0,
    )
    state = _with_dead_slots(state, _TEAM_A_FIRST_SLOT)

    respawned_state, _, _, _, respawned_mask, populated_info = _take_step(
        config,
        state,
    )
    successor_state, _, _, _, _, empty_info = _take_step(
        config,
        respawned_state,
        action_mask=respawned_mask,
    )

    assert bool(
        populated_info.transition_facts.respawn_facts.respawn_wave_occurred_this_transition_by_team[
            0
        ]
    )
    assert bool(
        populated_info.transition_facts.respawn_facts.was_respawned_this_transition_by_agent[
            _TEAM_A_FIRST_SLOT
        ]
    )
    assert bool(
        empty_info.transition_facts.respawn_facts.respawn_wave_occurred_this_transition_by_team[
            0
        ]
    )
    assert not bool(
        jnp.any(
            empty_info.transition_facts.respawn_facts.was_respawned_this_transition_by_agent
        )
    )
    assert int(respawned_state.team_respawn_wave_countdowns[0]) == 0
    assert int(successor_state.team_respawn_wave_countdowns[0]) == 0


def test_simultaneous_empty_waves_publish_real_transition_facts() -> None:
    """Both clocks may be due without manufacturing a realized respawn."""
    config, state, _, _, _ = _scenario(periods=(1, 3))
    state = state._replace(
        team_respawn_wave_countdowns=jnp.zeros((NUM_TEAMS,), dtype=jnp.int32)
    )

    next_state, _, _, _, _, info = _take_step(config, state)
    facts = info.transition_facts

    assert bool(facts.has_transition)
    assert int(facts.transition_start_step_count) == 0
    assert bool(
        jnp.array_equal(
            facts.respawn_facts.respawn_wave_occurred_this_transition_by_team,
            jnp.asarray((True, True)),
        )
    )
    assert not bool(jnp.any(facts.respawn_facts.was_respawned_this_transition_by_agent))
    assert bool(
        jnp.array_equal(
            next_state.team_respawn_wave_countdowns,
            jnp.asarray((0, 2), dtype=jnp.int32),
        )
    )


@pytest.mark.parametrize(
    ("dead_slot", "due_team", "expected_countdowns"),
    (
        pytest.param(_TEAM_A_FIRST_SLOT, 0, (2, 2), id="team-a"),
        pytest.param(_TEAM_B_FIRST_SLOT, 1, (0, 4), id="team-b"),
    ),
)
def test_independent_wave_respawns_only_transition_start_corpses_for_due_team(
    dead_slot: int,
    due_team: int,
    expected_countdowns: tuple[int, int],
) -> None:
    """Eligibility is transition-start death intersected with the due team."""
    config, state, _, _, _ = _scenario(
        team_sizes=(2, 2),
        periods=(3, 5),
    )
    state = _with_dead_slots(state, dead_slot)
    countdowns = jnp.asarray((1, 3), dtype=jnp.int32).at[due_team].set(0)
    state = state._replace(team_respawn_wave_countdowns=countdowns)

    next_state, _, _, _, _, info = _take_step(config, state)
    expected_realized = _slot_mask(dead_slot)

    assert bool(next_state.alive_mask[dead_slot])
    assert float(next_state.current_health[dead_slot]) == float(
        config.agent_profile.max_health[dead_slot]
    )
    assert (
        float(
            info.transition_facts.combat_transition_facts.health_after_combat_resolution_by_recipient[
                dead_slot
            ]
        )
        == 0.0
    )
    physical_facts = info.transition_facts.physical_facts
    assert bool(
        jnp.all(physical_facts.charge_phase_displacement_by_agent[dead_slot] == 0)
    )
    assert bool(
        jnp.all(
            physical_facts.ordinary_movement_phase_displacement_by_agent[dead_slot] == 0
        )
    )
    assert bool(
        jnp.array_equal(
            info.transition_facts.respawn_facts.was_respawned_this_transition_by_agent,
            expected_realized,
        )
    )
    assert bool(
        jnp.array_equal(
            next_state.team_respawn_wave_countdowns,
            jnp.asarray(expected_countdowns, dtype=jnp.int32),
        )
    )
    assert not bool(jnp.any(next_state.alive_mask[2:MAX_AGENTS_PER_TEAM]))
    assert not bool(jnp.any(next_state.alive_mask[_TEAM_B_SECOND_SLOT + 1 :]))
    assert bool(jnp.all(next_state.current_health[2:MAX_AGENTS_PER_TEAM] == 0))
    assert bool(jnp.all(next_state.current_health[_TEAM_B_SECOND_SLOT + 1 :] == 0))


def test_simultaneous_populated_waves_respawn_full_rosters_at_assigned_pads() -> None:
    """A full 5v5 wave preserves fixed slots and team-local pad identity."""
    config, state, _, _, _ = _scenario(
        team_sizes=(MAX_AGENTS_PER_TEAM, MAX_AGENTS_PER_TEAM),
        periods=(3, 4),
    )
    state = _with_dead_slots(state, *range(MAX_AGENT_SLOTS))._replace(
        team_respawn_wave_countdowns=jnp.zeros((NUM_TEAMS,), dtype=jnp.int32)
    )

    next_state, _, _, _, _, info = _take_step(config, state)

    assert bool(jnp.all(next_state.alive_mask))
    assert bool(
        jnp.array_equal(next_state.current_health, config.agent_profile.max_health)
    )
    assert bool(
        jnp.array_equal(
            next_state.agent_positions,
            config.team_spawn_pad_positions.reshape(
                (MAX_AGENT_SLOTS, ENVIRONMENT_DIMENSIONS)
            ),
        )
    )
    assert bool(
        jnp.all(
            info.transition_facts.respawn_facts.was_respawned_this_transition_by_agent
        )
    )


def test_same_transition_death_waits_for_a_later_wave() -> None:
    """A recipient alive at phase start is excluded from that transition's wave."""
    config, state, _, _, _ = _scenario(
        (_TEAM_A_FIRST_SLOT, MAGE_CLASS_ID),
        team_sizes=(1, 1),
        periods=(3, 3),
        shield_duration=0,
    )
    positions = state.agent_positions
    positions = positions.at[_TEAM_A_FIRST_SLOT].set(
        jnp.asarray((7.0, 7.0), dtype=jnp.float32)
    )
    positions = positions.at[_TEAM_B_FIRST_SLOT].set(
        jnp.asarray((8.0, 7.0), dtype=jnp.float32)
    )
    state = state._replace(
        agent_positions=positions,
        current_health=state.current_health.at[_TEAM_B_FIRST_SLOT].set(1.0),
        team_respawn_wave_countdowns=jnp.asarray((2, 0), dtype=jnp.int32),
    )
    lethal_action = _joint_action(
        (
            _TEAM_A_FIRST_SLOT,
            MOVE_STAY,
            _FIRST_ENEMY_TARGET,
            0,
        )
    )

    dead_state, dead_observation, _, _, dead_mask, death_info = _take_step(
        config,
        state,
        lethal_action,
    )

    assert not bool(dead_state.alive_mask[_TEAM_B_FIRST_SLOT])
    assert bool(
        death_info.transition_facts.death_facts.is_newly_dead_by_recipient[
            _TEAM_B_FIRST_SLOT
        ]
    )
    assert bool(
        death_info.transition_facts.respawn_facts.respawn_wave_occurred_this_transition_by_team[
            1
        ]
    )
    assert not bool(
        death_info.transition_facts.respawn_facts.was_respawned_this_transition_by_agent[
            _TEAM_B_FIRST_SLOT
        ]
    )
    assert bool(dead_mask.move_mask[_TEAM_B_FIRST_SLOT, MOVE_STAY])
    assert not bool(dead_mask.move_mask[_TEAM_B_FIRST_SLOT, MOVE_EAST])
    assert bool(
        jnp.array_equal(
            dead_observation.spawn_lifecycle.respawn_wave_countdowns_by_agent_by_team[
                _TEAM_B_FIRST_SLOT
            ],
            jnp.asarray((2, 1), dtype=jnp.int32),
        )
    )

    later_due_state = dead_state._replace(
        team_respawn_wave_countdowns=dead_state.team_respawn_wave_countdowns.at[1].set(
            0
        )
    )
    dead_submitted_action = _joint_action(
        (
            _TEAM_B_FIRST_SLOT,
            MOVE_EAST,
            _FIRST_ENEMY_TARGET,
            1,
        )
    )
    (
        respawned_state,
        respawned_observation,
        _,
        _,
        respawned_mask,
        respawn_info,
    ) = _take_step(
        config,
        later_due_state,
        dead_submitted_action,
        action_mask=_current_action_mask(config, later_due_state),
    )

    assert bool(respawned_state.alive_mask[_TEAM_B_FIRST_SLOT])
    assert bool(
        respawn_info.transition_facts.respawn_facts.was_respawned_this_transition_by_agent[
            _TEAM_B_FIRST_SLOT
        ]
    )
    accepted_action = (
        respawn_info.transition_facts.action_acceptance_facts.accepted_joint_action
    )
    assert int(accepted_action.move[_TEAM_B_FIRST_SLOT]) == MOVE_STAY
    assert int(accepted_action.select_target[_TEAM_B_FIRST_SLOT]) == _TARGET_NONE
    assert int(accepted_action.use_ultimate[_TEAM_B_FIRST_SLOT]) == 0
    assert bool(
        respawn_info.transition_facts.action_acceptance_facts.in_domain_move_action_is_rejected_by_actor[
            _TEAM_B_FIRST_SLOT
        ]
    )
    assert bool(
        respawn_info.transition_facts.action_acceptance_facts.in_domain_combat_action_pair_is_rejected_by_actor[
            _TEAM_B_FIRST_SLOT
        ]
    )
    assert bool(respawned_mask.move_mask[_TEAM_B_FIRST_SLOT, MOVE_EAST])
    assert bool(
        jnp.array_equal(
            respawned_observation.spawn_lifecycle.respawn_wave_countdowns_by_agent_by_team[
                _TEAM_B_FIRST_SLOT
            ],
            jnp.asarray((2, 0), dtype=jnp.int32),
        )
    )

    position_before_agency = respawned_state.agent_positions[_TEAM_B_FIRST_SLOT]
    agency_state, _, _, _, _, agency_info = _take_step(
        config,
        respawned_state,
        _joint_action((_TEAM_B_FIRST_SLOT, MOVE_EAST, _TARGET_NONE, 0)),
        action_mask=respawned_mask,
    )
    assert (
        int(
            agency_info.transition_facts.action_acceptance_facts.accepted_joint_action.move[
                _TEAM_B_FIRST_SLOT
            ]
        )
        == MOVE_EAST
    )
    assert float(agency_state.agent_positions[_TEAM_B_FIRST_SLOT, 0]) > float(
        position_before_agency[0]
    )


def test_assigned_pad_wins_over_live_enemy_occupancy() -> None:
    """Respawn placement is slot identity, not a nearest-free-pad search."""
    config, state, _, _, _ = _scenario(
        team_sizes=(1, 1),
        periods=(3, 4),
        shield_duration=2,
    )
    assigned_pad = config.team_spawn_pad_positions[0, 0]
    state = _with_dead_slots(state, _TEAM_A_FIRST_SLOT)
    state = state._replace(
        agent_positions=state.agent_positions.at[_TEAM_B_FIRST_SLOT].set(assigned_pad),
        team_respawn_wave_countdowns=jnp.asarray((0, 3), dtype=jnp.int32),
    )

    next_state, _, _, _, _, info = _take_step(config, state)

    assert bool(
        jnp.array_equal(next_state.agent_positions[_TEAM_A_FIRST_SLOT], assigned_pad)
    )
    assert bool(
        jnp.array_equal(next_state.agent_positions[_TEAM_B_FIRST_SLOT], assigned_pad)
    )
    assert bool(
        info.transition_facts.respawn_facts.was_respawned_this_transition_by_agent[
            _TEAM_A_FIRST_SLOT
        ]
    )


def test_duration_zero_overlap_uses_ordinary_stay_collision_next_transition() -> None:
    """A zero-shield respawn overlaps now and rejoins body blocking next step."""
    config, state, _, _, _ = _scenario(
        team_sizes=(1, 1),
        periods=(3, 4),
        shield_duration=0,
    )
    assigned_pad = config.team_spawn_pad_positions[0, 0]
    state = _with_dead_slots(state, _TEAM_A_FIRST_SLOT)
    state = state._replace(
        agent_positions=state.agent_positions.at[_TEAM_B_FIRST_SLOT].set(assigned_pad),
        team_respawn_wave_countdowns=jnp.asarray((0, 3), dtype=jnp.int32),
    )

    overlapped_state, _, _, _, overlapped_mask, _ = _take_step(config, state)
    assert bool(
        jnp.array_equal(
            overlapped_state.agent_positions[_TEAM_A_FIRST_SLOT],
            overlapped_state.agent_positions[_TEAM_B_FIRST_SLOT],
        )
    )

    separated_state, _, _, _, _, _ = _take_step(
        config,
        overlapped_state,
        _joint_action(),
        action_mask=overlapped_mask,
    )
    distance = cast(
        Array,
        jnp.linalg.norm(
            separated_state.agent_positions[_TEAM_A_FIRST_SLOT]
            - separated_state.agent_positions[_TEAM_B_FIRST_SLOT]
        ),
    )
    minimum_distance = (
        config.agent_profile.agent_radii[_TEAM_A_FIRST_SLOT]
        + config.agent_profile.agent_radii[_TEAM_B_FIRST_SLOT]
    )
    assert bool(distance >= minimum_distance - GEOMETRY_TOLERANCE)


@pytest.mark.parametrize(
    "shield_duration",
    (
        pytest.param(0, id="zero"),
        pytest.param(1, id="one"),
        pytest.param(4, id="many"),
    ),
)
def test_respawned_shield_is_not_decremented_on_creation_transition(
    shield_duration: int,
) -> None:
    """The end-of-transition shield override starts at its full duration."""
    config, state, _, _, _ = _scenario(
        team_sizes=(1, 1),
        periods=(3, 4),
        shield_duration=shield_duration,
    )
    state = _with_dead_slots(state, _TEAM_A_FIRST_SLOT)._replace(
        team_respawn_wave_countdowns=jnp.asarray((0, 3), dtype=jnp.int32)
    )

    respawned_state, _, _, _, respawned_mask, _ = _take_step(config, state)
    assert (
        int(respawned_state.spawn_shield_durations[_TEAM_A_FIRST_SLOT])
        == shield_duration
    )

    successor_state, _, _, _, _, _ = _take_step(
        config,
        respawned_state,
        action_mask=respawned_mask,
    )
    assert int(successor_state.spawn_shield_durations[_TEAM_A_FIRST_SLOT]) == max(
        0,
        shield_duration - 1,
    )


def test_respawn_restores_health_and_retains_canonical_dead_phase_state() -> None:
    """A genuine death clears statuses while later respawns retain dead-phase state."""
    config, state, _, _, _ = _scenario(
        (_TEAM_B_FIRST_SLOT, MAGE_CLASS_ID),
        team_sizes=(1, 1),
        periods=(3, 4),
        shield_duration=2,
    )
    positions = state.agent_positions
    positions = positions.at[_TEAM_A_FIRST_SLOT].set(
        jnp.asarray((7.0, 7.0), dtype=jnp.float32)
    )
    positions = positions.at[_TEAM_B_FIRST_SLOT].set(
        jnp.asarray((8.0, 7.0), dtype=jnp.float32)
    )
    state = state._replace(
        agent_positions=positions,
        current_health=state.current_health.at[_TEAM_A_FIRST_SLOT].set(1.0),
        slow_durations=state.slow_durations.at[_TEAM_A_FIRST_SLOT].set(
            jnp.ones((NUM_SLOW_CHANNELS,), dtype=jnp.int32) * 2
        ),
        stun_durations=state.stun_durations.at[_TEAM_A_FIRST_SLOT].set(
            jnp.ones((NUM_STUN_CHANNELS,), dtype=jnp.int32) * 2
        ),
        rogue_poison_anti_heal_durations=state.rogue_poison_anti_heal_durations.at[
            _TEAM_A_FIRST_SLOT
        ].set(2),
        mage_burst_damage_amplification_durations=(
            state.mage_burst_damage_amplification_durations.at[_TEAM_A_FIRST_SLOT].set(
                2
            )
        ),
        priest_blessing_of_freedom_slow_floor_durations=(
            state.priest_blessing_of_freedom_slow_floor_durations.at[
                _TEAM_A_FIRST_SLOT
            ].set(2)
        ),
        ultimate_cooldowns=state.ultimate_cooldowns.at[_TEAM_A_FIRST_SLOT].set(5),
        previous_timestep_move_actions=state.previous_timestep_move_actions.at[
            _TEAM_A_FIRST_SLOT
        ].set(MOVE_EAST),
        previous_timestep_select_target_actions=(
            state.previous_timestep_select_target_actions.at[_TEAM_A_FIRST_SLOT].set(
                _FIRST_ENEMY_TARGET
            )
        ),
        previous_timestep_use_ultimate_actions=(
            state.previous_timestep_use_ultimate_actions.at[_TEAM_A_FIRST_SLOT].set(1)
        ),
        has_previous_timestep_joint_action=jnp.asarray(True),
        team_respawn_wave_countdowns=jnp.asarray((1, 3), dtype=jnp.int32),
    )

    dead_state, _, _, _, dead_mask, death_info = _take_step(
        config,
        state,
        _joint_action(
            (
                _TEAM_B_FIRST_SLOT,
                MOVE_STAY,
                _FIRST_ENEMY_TARGET,
                0,
            )
        ),
    )

    assert bool(
        death_info.transition_facts.death_facts.is_newly_dead_by_recipient[
            _TEAM_A_FIRST_SLOT
        ]
    )
    assert not bool(dead_state.alive_mask[_TEAM_A_FIRST_SLOT])
    assert int(dead_state.ultimate_cooldowns[_TEAM_A_FIRST_SLOT]) == 4
    _assert_statuses_are_clear(dead_state, _TEAM_A_FIRST_SLOT)

    first_respawn, first_observation, _, _, _, first_info = _take_step(
        config,
        dead_state,
        action_mask=dead_mask,
    )

    assert bool(first_respawn.alive_mask[_TEAM_A_FIRST_SLOT])
    assert float(first_respawn.current_health[_TEAM_A_FIRST_SLOT]) == float(
        config.agent_profile.max_health[_TEAM_A_FIRST_SLOT]
    )
    assert int(first_respawn.ultimate_cooldowns[_TEAM_A_FIRST_SLOT]) == 3
    _assert_statuses_are_clear(first_respawn, _TEAM_A_FIRST_SLOT)
    assert int(first_respawn.previous_timestep_move_actions[_TEAM_A_FIRST_SLOT]) == (
        MOVE_STAY
    )
    assert (
        int(first_respawn.previous_timestep_select_target_actions[_TEAM_A_FIRST_SLOT])
        == _TARGET_NONE
    )
    assert (
        int(first_respawn.previous_timestep_use_ultimate_actions[_TEAM_A_FIRST_SLOT])
        == 0
    )
    accepted_dead_action = (
        first_info.transition_facts.action_acceptance_facts.accepted_joint_action
    )
    assert bool(
        jnp.array_equal(
            first_respawn.previous_timestep_move_actions,
            accepted_dead_action.move,
        )
    )
    assert bool(
        jnp.array_equal(
            first_respawn.previous_timestep_select_target_actions,
            accepted_dead_action.select_target,
        )
    )
    assert bool(
        jnp.array_equal(
            first_respawn.previous_timestep_use_ultimate_actions,
            accepted_dead_action.use_ultimate,
        )
    )
    assert bool(
        first_info.transition_facts.respawn_facts.was_respawned_this_transition_by_agent[
            _TEAM_A_FIRST_SLOT
        ]
    )
    assert (
        int(
            first_observation.spawn_lifecycle.spawn_shield_actual_durations_by_agent_by_team[
                _TEAM_A_FIRST_SLOT, 0, 0
            ]
        )
        == 2
    )

    second_corpse = _with_dead_slots(first_respawn, _TEAM_A_FIRST_SLOT)._replace(
        team_respawn_wave_countdowns=jnp.asarray((0, 2), dtype=jnp.int32)
    )
    second_respawn, _, _, _, _, second_info = _take_step(config, second_corpse)

    assert bool(second_respawn.alive_mask[_TEAM_A_FIRST_SLOT])
    assert float(second_respawn.current_health[_TEAM_A_FIRST_SLOT]) == float(
        config.agent_profile.max_health[_TEAM_A_FIRST_SLOT]
    )
    assert int(second_respawn.ultimate_cooldowns[_TEAM_A_FIRST_SLOT]) == 2
    _assert_statuses_are_clear(second_respawn, _TEAM_A_FIRST_SLOT)
    assert bool(
        second_info.transition_facts.respawn_facts.was_respawned_this_transition_by_agent[
            _TEAM_A_FIRST_SLOT
        ]
    )


def test_representative_respawn_transition_is_exact_under_jit() -> None:
    """Eager and compiled execution return identical full public PyTrees."""
    config, state, _, _, _ = _scenario(
        team_sizes=(2, 2),
        periods=(3, 4),
        shield_duration=3,
    )
    state = _with_dead_slots(
        state,
        _TEAM_A_SECOND_SLOT,
        _TEAM_B_FIRST_SLOT,
    )._replace(team_respawn_wave_countdowns=jnp.zeros((NUM_TEAMS,), dtype=jnp.int32))
    action_mask = _current_action_mask(config, state)
    action = _joint_action()
    key = jax.random.key(17)

    eager_result = step(config, state, action_mask, action, key)
    compiled_result = cast(
        _StepResult,
        jax.jit(step)(config, state, action_mask, action, key),
    )

    _assert_tree_equal(eager_result, compiled_result)


def test_shared_config_vmap_preserves_fixed_shapes_and_scalar_semantics() -> None:
    """One config vmaps over quiet, empty-wave, and populated-wave states."""
    config, base_state, _, _, _ = _scenario(
        team_sizes=(1, 1),
        periods=(3, 2),
        shield_duration=1,
    )
    populated_a = _with_dead_slots(base_state, _TEAM_A_FIRST_SLOT)._replace(
        team_respawn_wave_countdowns=jnp.asarray((0, 1), dtype=jnp.int32)
    )
    empty_b = base_state._replace(
        team_respawn_wave_countdowns=jnp.asarray((1, 0), dtype=jnp.int32)
    )
    populated_both = _with_dead_slots(
        base_state,
        _TEAM_A_FIRST_SLOT,
        _TEAM_B_FIRST_SLOT,
    )._replace(team_respawn_wave_countdowns=jnp.zeros((NUM_TEAMS,), dtype=jnp.int32))
    states = (populated_a, empty_b, populated_both)
    masks = tuple(_current_action_mask(config, state) for state in states)
    actions = (_joint_action(), _joint_action(), _joint_action())
    keys = tuple(jax.random.split(jax.random.key(23), len(states)))

    batched_states = cast(EnvState, _stack_trees(*states))
    batched_masks = cast(ActionMask, _stack_trees(*masks))
    batched_actions = cast(Action, _stack_trees(*actions))
    batched_keys = jnp.stack(keys)

    def _shared_config_step(
        state: EnvState,
        mask: ActionMask,
        action: Action,
        key: Array,
    ) -> _StepResult:
        return step(config, state, mask, action, key)

    batched_result = jax.vmap(_shared_config_step)(
        batched_states,
        batched_masks,
        batched_actions,
        batched_keys,
    )

    scalar_results = tuple(
        step(config, state, mask, action, key)
        for state, mask, action, key in zip(
            states,
            masks,
            actions,
            keys,
            strict=True,
        )
    )
    expected_batched_result = _stack_trees(*scalar_results)
    _assert_tree_equal(batched_result, expected_batched_result)

    batched_next_states = batched_result[0]
    batched_infos = batched_result[-1]
    assert batched_next_states.agent_positions.shape == (
        len(states),
        MAX_AGENT_SLOTS,
        ENVIRONMENT_DIMENSIONS,
    )
    assert (
        batched_infos.transition_facts.respawn_facts.was_respawned_this_transition_by_agent.shape
        == (len(states), MAX_AGENT_SLOTS)
    )
    assert bool(
        jnp.array_equal(
            batched_infos.transition_facts.respawn_facts.respawn_wave_occurred_this_transition_by_team,
            jnp.asarray(
                ((True, False), (False, True), (True, True)),
                dtype=jnp.bool_,
            ),
        )
    )


def test_lax_scan_crosses_empty_and_populated_waves() -> None:
    """A real scan carries the paired mask across independent public clocks."""
    horizon = 4
    config, state, _, _, _ = _scenario(
        team_sizes=(1, 1),
        periods=(3, 2),
        shield_duration=0,
    )
    state = _with_dead_slots(state, _TEAM_A_FIRST_SLOT)._replace(
        team_respawn_wave_countdowns=jnp.asarray((2, 0), dtype=jnp.int32)
    )
    action_mask = _current_action_mask(config, state)
    action = _joint_action()
    keys = jax.random.split(jax.random.key(31), horizon)

    def _scan_step(
        carry: tuple[EnvState, ActionMask],
        key: Array,
    ) -> tuple[tuple[EnvState, ActionMask], tuple[Array, Array, Array, Array]]:
        current_state, current_mask = carry
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
                next_state.team_respawn_wave_countdowns,
                info.transition_facts.respawn_facts.respawn_wave_occurred_this_transition_by_team,
                info.transition_facts.respawn_facts.was_respawned_this_transition_by_agent,
                next_state.alive_mask,
            ),
        )

    (_, _), (countdown_history, due_history, respawn_history, alive_history) = (
        jax.lax.scan(_scan_step, (state, action_mask), keys)
    )

    assert countdown_history.shape == (horizon, NUM_TEAMS)
    assert due_history.shape == (horizon, NUM_TEAMS)
    assert respawn_history.shape == (horizon, MAX_AGENT_SLOTS)
    assert alive_history.shape == (horizon, MAX_AGENT_SLOTS)
    assert bool(
        jnp.array_equal(
            countdown_history,
            jnp.asarray(
                ((1, 1), (0, 0), (2, 1), (1, 0)),
                dtype=jnp.int32,
            ),
        )
    )
    assert bool(
        jnp.array_equal(
            due_history,
            jnp.asarray(
                (
                    (False, True),
                    (False, False),
                    (True, True),
                    (False, False),
                ),
                dtype=jnp.bool_,
            ),
        )
    )
    assert not bool(jnp.any(respawn_history[0]))
    assert not bool(jnp.any(respawn_history[1]))
    assert bool(respawn_history[2, _TEAM_A_FIRST_SLOT])
    assert int(jnp.sum(respawn_history[2])) == 1
    assert not bool(jnp.any(respawn_history[3]))
    assert not bool(alive_history[1, _TEAM_A_FIRST_SLOT])
    assert bool(alive_history[2, _TEAM_A_FIRST_SLOT])
