"""Fixed scalar metrics against public Core trajectories and lifecycle facts."""

from collections.abc import Callable
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from tests.evaluation_fixtures import evaluation_env_config, neutral_action

from marl_battlegrounds.core.axis_mappings import global_slot_to_target_action
from marl_battlegrounds.core.config import resolve_agent_profile
from marl_battlegrounds.core.env import (
    build_canonical_no_transition_info_object,
    initialize_scenario_state,
    reset,
    step,
)
from marl_battlegrounds.core.types import (
    Action,
    ActionMask,
    DoneFlags,
    EnvConfig,
    EnvState,
    Info,
    Observation,
    Reward,
)
from marl_battlegrounds.evaluation.episode_metrics import (
    MetricValues,
    PriorityTotals,
    initialize_priority,
    priority_values,
    update_priority,
)
from marl_battlegrounds.evaluation.full_metrics import (
    FullTotals,
    full_values,
    initialize_full,
    update_full,
)
from marl_battlegrounds.evaluation.metric_catalog import (
    FULL_METRIC_NAMES,
    METRIC_COLUMNS,
    PRIORITY_METRIC_NAMES,
    STATUS_NAMES,
)

_step = cast(
    Callable[..., tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info]],
    jax.jit(step),
)
_update = cast(
    Callable[[FullTotals, EnvConfig, EnvState, ActionMask, Info], FullTotals],
    jax.jit(update_full),
)
_final = cast(
    Callable[[FullTotals, EnvConfig, MetricValues], MetricValues], jax.jit(full_values)
)
_COLUMN_INDEX = {name: index for index, name in enumerate(FULL_METRIC_NAMES)}


class _Run(NamedTuple):
    config: EnvConfig
    state: EnvState
    mask: ActionMask
    full: FullTotals
    priority: PriorityTotals
    initial_step: Array
    outcome: Array


def _start(
    *,
    team_sizes: tuple[int, int] = (3, 2),
    classes: tuple[tuple[int, int], ...] = (),
    arrange: Callable[[EnvState], EnvState] | None = None,
) -> _Run:
    config = evaluation_env_config(
        team_sizes=team_sizes,
        task_mode=1,
        team_deathmatch_score_threshold=20,
        max_steps=16,
    )
    class_ids = config.agent_profile.class_ids
    for slot, class_id in classes:
        class_ids = class_ids.at[slot].set(class_id)
    config = config._replace(
        agent_profile=resolve_agent_profile(
            class_ids, jnp.asarray(team_sizes, jnp.int32)
        )
    )
    state, _, _, _ = reset(config, jax.random.key(0))
    positions = jnp.asarray(
        [(6.0, 1.5 + 2 * slot) for slot in range(5)]
        + [(9.0, 1.5 + 2 * slot) for slot in range(5)],
        jnp.float32,
    )
    state = state._replace(
        agent_positions=jnp.where(
            config.agent_profile.active_mask[:, None], positions, 0
        )
    )
    if arrange is not None:
        state = arrange(state)
    state, _, mask, _ = initialize_scenario_state(state, config)
    return _Run(
        config,
        state,
        mask,
        initialize_full(config, state),
        initialize_priority(),
        state.step_count,
        jnp.asarray(0, jnp.int32),
    )


def _actions(*choices: tuple[int, int, bool]) -> Action:
    actions = neutral_action()
    for source, recipient, ultimate in choices:
        actions = actions._replace(
            select_target=actions.select_target.at[source].set(
                global_slot_to_target_action(source, recipient)
            ),
            use_ultimate=actions.use_ultimate.at[source].set(int(ultimate)),
        )
    return actions


def _advance(run: _Run, actions: Action) -> tuple[_Run, Info]:
    successor, _, reward, _, mask, info = _step(
        run.config, run.state, run.mask, actions, jax.random.key(1)
    )
    return (
        run._replace(
            state=successor,
            mask=mask,
            full=_update(run.full, run.config, run.state, run.mask, info),
            priority=update_priority(run.priority, reward, info),
            outcome=info.transition_facts.team_deathmatch_facts.outcome,
        ),
        info,
    )


def _priority(run: _Run) -> MetricValues:
    return priority_values(
        run.config, run.state, run.initial_step, run.priority, run.outcome
    )


def _values(run: _Run) -> MetricValues:
    return _final(run.full, run.config, _priority(run))


def _value(values: MetricValues, name: str) -> float:
    index = _COLUMN_INDEX[name]
    assert bool(values.valid[index]), name
    return float(values.values[index])


def _missing(values: MetricValues, name: str) -> None:
    assert not bool(values.valid[_COLUMN_INDEX[name]]), name


def test_fixed_full_schema_preserves_priority_and_missingness_across_rosters() -> None:
    first, _ = _advance(_start(), neutral_action())
    second, _ = _advance(
        _start(team_sizes=(1, 1), classes=((0, 5), (5, 2))), neutral_action()
    )
    compiled = (
        jax.jit(full_values).lower(first.full, first.config, _priority(first)).compile()
    )
    for run in (first, second):
        values = cast(MetricValues, compiled(run.full, run.config, _priority(run)))
        assert len(METRIC_COLUMNS) == 1388
        assert values.values.shape == values.valid.shape == (1388,)
        assert values.values.dtype == jnp.float32 and values.valid.dtype == jnp.bool_
        assert bool(jnp.isfinite(values.values).all())
        np.testing.assert_array_equal(
            values.values[: len(PRIORITY_METRIC_NAMES)], _priority(run).values
        )
        np.testing.assert_array_equal(
            values.valid[: len(PRIORITY_METRIC_NAMES)], _priority(run).valid
        )
        for column in METRIC_COLUMNS:
            if column.scope in ("agent", "source_recipient", "ally_pair") and any(
                not bool(run.config.agent_profile.active_mask[slot])
                for slot in column.subjects
            ):
                _missing(values, column.name)
            if column.required_class_id is not None:
                subjects = (
                    range((column.subjects[0] - 1) * 5, column.subjects[0] * 5)
                    if column.scope == "team"
                    else column.subjects[:1]
                )
                if not any(
                    bool(run.config.agent_profile.active_mask[slot])
                    and int(run.config.agent_profile.class_ids[slot])
                    == column.required_class_id
                    for slot in subjects
                ):
                    _missing(values, column.name)

    values = _values(first)
    assert (
        _value(values, "agent_0_healing_done") == 0
    )  # Active non-healer: measured zero.
    assert _value(values, "agent_2_damage_done") == 0  # Active Priest: measured zero.
    assert _value(values, "agent_0_agent_5_damage_done") == 0
    _missing(values, "agent_0_damage_done_fraction")
    _missing(values, "agent_0_agent_5_damage_done_fraction")
    _missing(values, "agent_0_wasted_healing")  # Class-inapplicable, not zero.
    _missing(values, "team_b_damage_from_mage_aura")
    _missing(values, "team_b_rescue_opportunities")
    _missing(values, "team_a_focus_fire_concentration")
    _missing(_values(second), "team_a_ally_distance_mean")
    assert _value(_values(second), "team_a_ally_distance_observations") == 0


@pytest.mark.parametrize("duration", (0, 1, 3))
def test_all_status_active_time_uses_decision_start_and_natural_expiry(
    duration: int,
) -> None:
    # Hunter's Basic slow, Charge/Poison stuns and Freedom last at most one tick.
    short = min(duration, 1)
    durations = np.asarray(
        (duration, short, duration, short, duration, short, duration, duration, short)
    )

    def arrange(state: EnvState) -> EnvState:
        return state._replace(
            slow_durations=state.slow_durations.at[5].set(durations[:3]),
            stun_durations=state.stun_durations.at[5].set(durations[3:6]),
            rogue_poison_anti_heal_durations=state.rogue_poison_anti_heal_durations.at[
                5
            ].set(duration),
            mage_burst_damage_amplification_durations=state.mage_burst_damage_amplification_durations.at[
                5
            ].set(duration),
            priest_blessing_of_freedom_slow_floor_durations=state.priest_blessing_of_freedom_slow_floor_durations.at[
                5
            ].set(short),
        )

    run = _start(classes=((5, 1),), arrange=arrange)
    assert int(run.full["trap_periods"][0]) == int(duration > 0)
    for transition in range(4):
        run, info = _advance(run, neutral_action())
        lifecycle = info.transition_facts.status_lifecycle_facts
        aged = lifecycle.aged_to_zero_by_recipient_and_status_channel[5, :]
        np.testing.assert_array_equal(aged, durations == transition + 1)
    np.testing.assert_array_equal(run.full["active_steps"][5], durations)
    values = _values(run)
    for status, expected in zip(STATUS_NAMES, durations, strict=True):
        assert _value(values, f"agent_5_{status}_active_steps") == expected
    assert _value(values, "team_a_trap_intervals") == int(duration > 0)
    assert _value(values, "team_a_trap_breaks") == 0
    _missing(values, "team_a_trap_mean_remaining_steps_at_break")


@pytest.mark.parametrize(
    ("source", "recipient", "ultimate", "channels"),
    (
        pytest.param(1, 5, True, (0, 3), id="warrior-charge"),
        pytest.param(5, 0, False, (1,), id="hunter-basic"),
        pytest.param(5, 0, True, (4,), id="hunter-trap"),
        pytest.param(6, 0, True, (2, 5, 6), id="rogue-poison"),
        pytest.param(0, 0, True, (7,), id="mage-burst"),
        pytest.param(2, 0, False, (8,), id="priest-freedom"),
    ),
)
def test_status_applications_bind_every_channel_to_the_accepted_source(
    source: int, recipient: int, ultimate: bool, channels: tuple[int, ...]
) -> None:
    def arrange(state: EnvState) -> EnvState:
        positions = state.agent_positions
        if source == 6:
            positions = positions.at[6].set((7.2, 1.5))
        if source == 2:
            positions = positions.at[2].set((5, 2.5))
        return state._replace(agent_positions=positions)

    run = _start(arrange=arrange)
    actions = (
        neutral_action()._replace(use_ultimate=jnp.zeros(10, jnp.int32).at[0].set(1))
        if source == 0
        else _actions((source, recipient, ultimate))
    )
    run, info = _advance(run, actions)
    combat = info.transition_facts.combat_transition_facts
    activated = (
        combat.ultimate_effect_is_activated_by_source
        if ultimate
        else combat.basic_effect_is_activated_by_source
    )
    assert bool(activated[source])
    expected = np.zeros((10, 9), dtype=np.int32)
    expected[source, list(channels)] = 1
    np.testing.assert_array_equal(run.full["applications"], expected)
    # Fresh applications cannot retroactively increase transition-start exposure.
    np.testing.assert_array_equal(run.full["active_steps"], 0)
    values = _values(run)
    for channel in channels:
        assert (
            _value(values, f"agent_{source}_{STATUS_NAMES[channel]}_applications") == 1
        )


@pytest.mark.parametrize(
    ("duration", "intervals", "aged", "broken"),
    (
        (0, 1, False, False),
        (1, 2, True, False),
        (3, 2, False, True),
    ),
)
def test_trap_reapplication_follows_natural_expiry_or_damage_break(
    duration: int, intervals: int, aged: bool, broken: bool
) -> None:
    run = _start(
        classes=((0, 3),),
        arrange=lambda state: state._replace(
            stun_durations=state.stun_durations.at[5, 1].set(duration)
        ),
    )
    run, info = _advance(run, _actions((0, 5, True)))
    facts = info.transition_facts
    lifecycle = facts.status_lifecycle_facts
    assert bool(
        facts.combat_transition_facts.stun_is_applied_by_source_and_channel[0, 1]
    )
    assert bool(lifecycle.aged_to_zero_by_recipient_and_status_channel[5, 4]) == aged
    # Trap's own accepted raw damage breaks the aged old period before reapplication.
    assert facts.combat_transition_facts.raw_damage_output_by_source[0] > 0
    assert not bool(
        lifecycle.refreshed_or_extended_by_recipient_and_status_channel[5, 4]
    )
    assert (
        bool(lifecycle.broken_by_damage_by_recipient_and_status_channel[5, 4]) == broken
    )
    assert int(run.state.stun_durations[5, 1]) == max(4, duration - 1)
    values = _values(run)
    assert _value(values, "team_a_trap_intervals") == intervals
    assert _value(values, "agent_0_hunter_trap_applications") == 1
    assert _value(values, "agent_5_hunter_trap_active_steps") == int(duration > 0)


@pytest.mark.parametrize("lethal", (False, True))
def test_damage_break_then_fresh_trap_counts_two_periods_even_if_death_clears_new(
    lethal: bool,
) -> None:
    run = _start(
        classes=((1, 3),),
        arrange=lambda state: state._replace(
            agent_positions=state.agent_positions.at[1].set((7, 3)),
            stun_durations=state.stun_durations.at[5, 1].set(3),
            current_health=state.current_health.at[5].set(1 if lethal else 100),
        ),
    )
    run, info = _advance(run, _actions((0, 5, False), (1, 5, True)))
    lifecycle = info.transition_facts.status_lifecycle_facts
    assert bool(lifecycle.broken_by_damage_by_recipient_and_status_channel[5, 4])
    assert not bool(
        lifecycle.refreshed_or_extended_by_recipient_and_status_channel[5, 4]
    )
    assert (
        bool(lifecycle.cleared_by_new_death_by_recipient_and_status_channel[5, 4])
        == lethal
    )
    assert int(run.state.stun_durations[5, 1]) == (0 if lethal else 4)
    values = _values(run)
    assert _value(values, "team_a_trap_intervals") == 2
    assert _value(values, "team_a_trap_breaks") == 1
    assert _value(values, "team_a_trap_break_rate") == 0.5
    assert _value(values, "team_a_trap_mean_remaining_steps_at_break") == 2
    assert _value(values, "agent_0_trap_break_contributions") == 1
    assert _value(values, "agent_1_trap_break_contributions") == 1
    assert _value(values, "team_a_kills_of_hunter_trap_recipient") == int(lethal)
    if lethal:
        assert (
            _value(values, "agent_0_kill_participation_in_hunter_trap_recipient") == 1
        )
        assert (
            _value(values, "agent_1_kill_participation_in_hunter_trap_recipient") == 1
        )
        assert _value(values, "team_a_multi_contributor_kills") == 1


@pytest.mark.parametrize("duration", (0, 1, 3))
def test_burst_application_expiry_and_refresh_use_the_next_decision_epoch(
    duration: int,
) -> None:
    run = _start(
        arrange=lambda state: state._replace(
            mage_burst_damage_amplification_durations=state.mage_burst_damage_amplification_durations.at[
                0
            ].set(duration)
        )
    )
    actions = neutral_action()._replace(
        use_ultimate=jnp.zeros(10, jnp.int32).at[0].set(1)
    )
    run, info = _advance(run, actions)
    lifecycle = info.transition_facts.status_lifecycle_facts
    assert bool(lifecycle.aged_to_zero_by_recipient_and_status_channel[0, 7]) == (
        duration == 1
    )
    assert bool(
        lifecycle.refreshed_or_extended_by_recipient_and_status_channel[0, 7]
    ) == (duration > 1)
    assert int(run.state.mage_burst_damage_amplification_durations[0]) == 5
    assert _value(_values(run), "agent_0_mage_burst_active_steps") == int(duration > 0)
    assert _value(_values(run), "agent_0_burst_damage") == 0
    run, info = _advance(run, _actions((0, 5, False)))
    combat = info.transition_facts.combat_transition_facts
    delivered = (
        combat.source_modified_damage_output_by_source[0]
        * combat.recipient_damage_modifier_by_source[0]
    )
    assert delivered > 0
    values = _values(run)
    assert _value(values, "agent_0_mage_burst_applications") == 1
    assert _value(values, "agent_0_mage_burst_active_steps") == int(duration > 0) + 1
    assert _value(values, "agent_0_burst_damage") == pytest.approx(float(delivered))


def test_overlapping_control_effects_and_new_freedom_keep_their_own_epoch() -> None:
    run = _start(
        arrange=lambda state: state._replace(
            agent_positions=state.agent_positions.at[2].set((5, 2.5)),
            slow_durations=state.slow_durations.at[0].set(jnp.asarray((2, 1, 2))),
            rogue_poison_anti_heal_durations=state.rogue_poison_anti_heal_durations.at[
                0
            ].set(2),
            current_health=state.current_health.at[0].set(40),
        )
    )
    run, info = _advance(run, _actions((2, 0, False), (5, 0, False)))
    combat = info.transition_facts.combat_transition_facts
    damage = float(combat.total_effective_damage_by_recipient[0])
    healing = float(combat.total_effective_healing_by_recipient[0])
    assert damage > 0 and healing > 0
    values = _values(run)
    for status in (*STATUS_NAMES[:3], STATUS_NAMES[6]):
        assert _value(values, f"agent_5_damage_to_{status}_recipient") == pytest.approx(
            damage
        )
        assert _value(
            values, f"agent_2_healing_to_{status}_recipient"
        ) == pytest.approx(healing)
    assert _value(values, "agent_0_freedom_eligible_steps") == 0
    assert int(run.state.priest_blessing_of_freedom_slow_floor_durations[0]) == 1
    before = run.state.agent_positions
    run, _ = _advance(run, neutral_action())
    np.testing.assert_array_equal(run.state.agent_positions, before)
    values = _values(run)
    # Protection concerns the effective restriction even when the agent stays still.
    assert _value(values, "agent_0_freedom_eligible_steps") == 1
    assert _value(values, "agent_0_freedom_protected_steps") == 1
    assert _value(values, "agent_0_freedom_protection_fraction") == 1


def test_duplicate_aura_emitters_overlap_individually_but_team_time_is_unique() -> None:
    def arrange(state: EnvState) -> EnvState:
        return state._replace(
            agent_positions=state.agent_positions.at[0]
            .set((4, 3))
            .at[1]
            .set((6, 3))
            .at[2]
            .set((5, 4))
            .at[3]
            .set((5, 2))
            .at[5]
            .set((7, 3)),
            spawn_shield_durations=state.spawn_shield_durations.at[3].set(2),
        )

    run = _start(team_sizes=(4, 1), classes=((1, 1), (2, 2)), arrange=arrange)
    run, info = _advance(run, _actions((0, 5, False)))
    auras = info.transition_facts.aura_facts
    coverage = auras.is_covered_by_mage_damage_aura_by_emitter_and_beneficiary
    np.testing.assert_array_equal(coverage[:2, :3], True)
    assert not bool(
        coverage[:, 3].any()
    )  # Shielded beneficiary is in radius but excluded.
    values = _values(run)
    assert _value(values, "agent_0_mage_aura_covered_steps") == 3
    assert _value(values, "agent_1_mage_aura_covered_steps") == 3
    assert _value(values, "team_a_mage_aura_covered_steps") == 3
    assert _value(values, "team_a_mage_aura_eligible_steps") == 3
    assert _value(values, "team_a_mage_aura_coverage") == 1
    _missing(values, "agent_2_mage_aura_coverage")
    effects = info.transition_facts.combat_transition_facts
    gain = (
        effects.source_modified_damage_output_by_source[0]
        - effects.raw_damage_output_by_source[0]
    ) * effects.recipient_damage_modifier_by_source[0]
    assert _value(values, "team_a_damage_from_mage_aura") == pytest.approx(
        float(gain), abs=1e-5
    )
    assert gain > 0


def test_whole_action_acceptance_deduplicates_rejections_and_counts_dead_noops() -> (
    None
):
    run = _start(
        team_sizes=(4, 2),
        arrange=lambda state: state._replace(
            stun_durations=state.stun_durations.at[1, 0].set(1),
            alive_mask=state.alive_mask.at[3].set(False),
            current_health=state.current_health.at[3].set(0),
        ),
    )
    actions = neutral_action()._replace(
        move=jnp.zeros(10, jnp.int32).at[0].set(99).at[1].set(1),
        select_target=jnp.zeros(10, jnp.int32).at[1].set(1),
    )
    run, info = _advance(run, actions)
    facts = info.transition_facts.action_acceptance_facts
    assert bool(facts.submitted_action_tuple_is_out_of_domain_by_actor[0])
    assert bool(facts.in_domain_move_action_is_rejected_by_actor[1])
    assert bool(facts.in_domain_combat_action_pair_is_rejected_by_actor[1])
    values = _values(run)
    for name, expected in (
        ("team_a_actions_submitted", 4),
        ("team_a_actions_accepted", 2),
        ("team_a_actions_rejected", 2),
        ("team_a_action_domain_rejections", 1),
        ("team_a_action_movement_rejections", 1),
        ("team_a_action_combat_rejections", 1),
        ("team_a_action_acceptance_rate", 0.5),
        ("agent_3_actions_accepted", 1),
        ("agent_3_dead_steps", 1),
    ):
        assert _value(values, name) == expected


def test_formation_team_mean_weights_real_pair_observations_through_death() -> None:
    def arrange(state: EnvState) -> EnvState:
        return state._replace(
            agent_positions=state.agent_positions.at[0]
            .set((3, 3))
            .at[1]
            .set((6, 3))
            .at[2]
            .set((3, 7))
            .at[5]
            .set((6, 7)),
            current_health=state.current_health.at[2].set(1),
        )

    run = _start(team_sizes=(3, 1), classes=((5, 1),), arrange=arrange)
    run, info = _advance(run, _actions((5, 2, False)))
    assert bool(info.transition_facts.death_facts.is_newly_dead_by_recipient[2])
    run, _ = _advance(run, neutral_action())
    values = _values(run)
    # First tick observes distances 3,4,5; second only the surviving pair at 3.
    assert _value(values, "team_a_ally_distance_observations") == 4
    assert _value(values, "team_a_ally_distance_mean") == 3.75
    assert _value(values, "agent_0_agent_1_ally_distance_observations") == 2
    assert _value(values, "agent_0_agent_2_ally_distance_observations") == 1
    assert _value(values, "agent_1_agent_2_ally_distance_mean") == 5
    assert _value(values, "team_b_ally_distance_observations") == 0
    _missing(values, "team_b_ally_distance_mean")


def test_focus_fire_is_mean_of_eligible_tick_ratios_not_ratio_of_totals() -> None:
    run = _start(
        classes=((1, 1), (2, 1), (5, 2), (6, 2)),
        arrange=lambda state: state._replace(
            agent_positions=state.agent_positions.at[0]
            .set((6, 2))
            .at[1]
            .set((6, 3.5))
            .at[2]
            .set((6, 5))
            .at[5]
            .set((8, 2.5))
            .at[6]
            .set((8, 4.5))
        ),
    )
    run, first = _advance(run, _actions((0, 5, False), (1, 5, False)))
    run, second = _advance(run, _actions((0, 5, False), (1, 6, False), (2, 6, False)))
    assert (
        int(
            first.transition_facts.combat_transition_facts.basic_effect_is_activated_by_source[
                :3
            ].sum()
        )
        == 2
    )
    assert (
        int(
            second.transition_facts.combat_transition_facts.basic_effect_is_activated_by_source[
                :3
            ].sum()
        )
        == 3
    )
    values = _values(run)
    assert _value(values, "team_a_focus_fire_steps") == 2
    assert _value(values, "team_a_focus_fire_concentration") == pytest.approx(5 / 6)
    assert _value(values, "team_a_focus_fire_concentration") != pytest.approx(4 / 5)


def test_no_transition_facts_do_not_count_active_statuses_pairs_or_actions() -> None:
    run = _start(
        arrange=lambda state: state._replace(
            stun_durations=state.stun_durations.at[5, 1].set(3)
        )
    )
    run, _ = _advance(run, neutral_action())
    no_transition = build_canonical_no_transition_info_object(run.state)
    assert not bool(no_transition.transition_facts.has_transition)
    padded = _update(run.full, run.config, run.state, run.mask, no_transition)
    assert set(padded) == set(run.full)
    for name, expected in run.full.items():
        np.testing.assert_array_equal(padded[name], expected, err_msg=name)
    fresh_state, _, _, _ = reset(run.config, jax.random.key(8))
    fresh = initialize_full(run.config, fresh_state)
    for name, value in fresh.items():
        assert value.shape == run.full[name].shape
        np.testing.assert_array_equal(value, 0, err_msg=name)
