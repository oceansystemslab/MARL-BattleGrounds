"""Check scalar full metrics against Core trajectories and transition facts."""

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tests.evaluation_fixtures import neutral_action
from tests.full_metric_fixtures import (
    actions as _actions,
)
from tests.full_metric_fixtures import (
    advance as _advance,
)
from tests.full_metric_fixtures import (
    assert_missing as _missing,
)
from tests.full_metric_fixtures import (
    priority as _priority,
)
from tests.full_metric_fixtures import (
    start_run as _start,
)
from tests.full_metric_fixtures import (
    update_metrics as _update,
)
from tests.full_metric_fixtures import (
    value as _value,
)
from tests.full_metric_fixtures import (
    values as _values,
)

from marl_battlegrounds.core.config import resolve_agent_profile
from marl_battlegrounds.core.env import (
    build_canonical_no_transition_info_object,
    reset,
)
from marl_battlegrounds.core.types import (
    EnvState,
)
from marl_battlegrounds.evaluation.episode_metrics import (
    MetricValues,
)
from marl_battlegrounds.evaluation.full_metrics import (
    full_values,
    initialize_full,
)
from marl_battlegrounds.evaluation.metric_catalog import (
    METRIC_COLUMNS,
    PRIORITY_METRIC_NAMES,
    STATUS_NAMES,
)


def test_fixed_full_schema_preserves_priority_and_missingness_across_rosters() -> None:
    first, _ = _advance(_start(), neutral_action())
    second, _ = _advance(
        _start(team_sizes=(1, 1), classes=((0, 5), (5, 2))), neutral_action()
    )
    compiled = (
        jax.jit(full_values).lower(first.full, first.config, _priority(first)).compile()
    )
    runs = [first, second]
    # Each class occupies every slot once, using the same compiled finalizer.
    for rotation in range(5):
        classes = tuple((slot, (slot + rotation) % 5 + 1) for slot in range(10))
        run, _ = _advance(_start(team_sizes=(5, 5), classes=classes), neutral_action())
        runs.append(run)
    for team_a, team_b in (
        ((1, 5), (1, 5)),
        ((1, 1, 1), (1, 1, 1)),
        ((1, 2, 3, 4), (1, 2, 3, 4)),
        ((5, 5, 5, 5, 5), (5, 5, 5, 5, 5)),
        ((5, 5, 1), (1, 2)),
    ):
        classes = tuple(enumerate(team_a)) + tuple(
            (5 + slot, class_id) for slot, class_id in enumerate(team_b)
        )
        run, _ = _advance(
            _start(team_sizes=(len(team_a), len(team_b)), classes=classes),
            neutral_action(),
        )
        runs.append(run)
    for run in runs:
        values = cast(MetricValues, compiled(run.full, run.config, _priority(run)))
        assert len(METRIC_COLUMNS) == 11192
        assert values.values.shape == values.valid.shape == (11192,)
        assert values.values.dtype == jnp.float32 and values.valid.dtype == jnp.bool_
        assert bool(jnp.isfinite(values.values).all())
        np.testing.assert_array_equal(
            values.values[: len(PRIORITY_METRIC_NAMES)], _priority(run).values
        )
        np.testing.assert_array_equal(
            values.valid[: len(PRIORITY_METRIC_NAMES)], _priority(run).valid
        )
        active = np.asarray(run.config.agent_profile.active_mask)
        class_ids = np.asarray(run.config.agent_profile.class_ids)
        valid = np.asarray(values.valid)
        for index, column in enumerate(METRIC_COLUMNS):
            if column.scope == "agent" and column.stem == "rescue_opportunities":
                recipient = column.subjects[0]
                team_start = recipient // 5 * 5
                allies = slice(team_start, team_start + 5)
                has_priest = bool((active[allies] & (class_ids[allies] == 5)).any())
                assert bool(valid[index]) == (active[recipient] and has_priest)
            if column.scope == "team_recipient" and not active[column.subjects[1]]:
                assert not valid[index], column.name
            if column.scope in ("agent", "source_recipient", "ally_pair") and any(
                not active[slot] for slot in column.subjects
            ):
                assert not valid[index], column.name
            if column.required_class_id is not None:
                subjects = (
                    range((column.subjects[0] - 1) * 5, column.subjects[0] * 5)
                    if column.scope in ("team", "team_recipient")
                    else column.subjects[:1]
                )
                if not any(
                    active[slot] and class_ids[slot] == column.required_class_id
                    for slot in subjects
                ):
                    assert not valid[index], column.name

    values = _values(first)
    assert (
        _value(values, "agent_0_healing_done") == 0
    )  # Active non-healer: measured zero.
    assert _value(values, "agent_2_damage_done") == 0  # Active Priest: measured zero.
    assert _value(values, "agent_0_to_agent_5_damage_done") == 0
    _missing(values, "agent_0_damage_done_fraction")
    _missing(values, "agent_0_to_agent_5_damage_done_fraction")
    _missing(values, "agent_0_excess_healing")  # Class-inapplicable, not zero.
    _missing(values, "team_b_damage_from_mage_aura")
    _missing(values, "team_b_rescue_opportunities")
    _missing(values, "agent_5_rescue_opportunities")
    _missing(values, "agent_4_rescue_opportunities")
    for recipient in (0, 1, 2):
        assert _value(values, f"agent_{recipient}_rescue_opportunities") == 0
    _missing(values, "team_a_focus_fire_concentration")
    _missing(_values(second), "team_a_ally_distance_mean")
    assert _value(_values(second), "team_a_ally_distance_observations") == 0


def test_named_team_applications_count_only_active_sources_of_that_class() -> None:
    run = _start()
    # Distinct source totals expose a wrong team, source or class reduction.
    # These are reducer inputs, not a claim that this matrix is a public event.
    applications = jnp.arange(1, 101, dtype=jnp.int32).reshape(10, 10)
    totals = {**run.full, "ultimate_applications": applications}
    for classes, sizes in (
        ((1, 2, 3, 4, 5) * 2, (5, 5)),
        ((5, 4, 1, 2, 3, 3, 5, 4, 1, 2), (5, 5)),
        ((5, 5, 1, 2, 2, 1, 3, 4, 5, 5), (5, 5)),
        ((4, 4, 2, 2, 5, 5, 5, 4, 4, 2), (5, 5)),
        ((5, 2, 5, 5, 5, 1, 5, 3, 5, 5), (2, 3)),
        ((5, 5, 5, 5, 5, 1, 5, 5, 5, 5), (1, 1)),
        ((1, 2, 3, 4, 1) * 2, (5, 5)),
    ):
        profile = resolve_agent_profile(
            jnp.asarray(classes, jnp.int32), jnp.asarray(sizes, jnp.int32)
        )
        current = run._replace(
            config=run.config._replace(agent_profile=profile), full=totals
        )
        values = _values(current)
        active = np.asarray(profile.active_mask)
        sources = np.asarray(profile.class_ids)
        for team, slots in (("a", range(5)), ("b", range(5, 10))):
            for class_id, stem, status in (
                (2, "warrior_charge_applications", "warrior_charge_slow_applications"),
                (4, "rogue_poison_applications", "rogue_poison_slow_applications"),
                (5, "priest_holy_word_salvation_applications", None),
            ):
                name = f"team_{team}_{stem}"
                matching_sources = [
                    slot for slot in slots if active[slot] and sources[slot] == class_id
                ]
                expected = sum(
                    int(applications[slot].sum()) for slot in matching_sources
                )
                if matching_sources:
                    assert _value(values, name) == expected
                    assert _value(values, name) != _value(
                        values, f"team_{team}_ultimate_activations"
                    )
                    if status is not None:
                        assert _value(values, name) == _value(
                            values, f"team_{team}_{status}"
                        )
                else:
                    _missing(values, name)
                index = next(
                    i for i, column in enumerate(METRIC_COLUMNS) if column.name == name
                )
                assert int(values.values[index]) == expected


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
    assert int(run.full["trap_periods"][5]) == int(duration > 0)
    initial = _values(run)
    assert _value(initial, "agent_5_trap_intervals") == int(duration > 0)
    if duration:
        assert _value(initial, "agent_5_trap_break_rate") == 0
    else:
        _missing(initial, "agent_5_trap_break_rate")
    _missing(initial, "agent_5_trap_mean_remaining_steps_at_break")
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
    assert _value(values, "agent_5_trap_intervals") == int(duration > 0)
    assert _value(values, "agent_5_trap_breaks") == 0
    _missing(values, "team_a_trap_mean_remaining_steps_at_break")
    _missing(values, "agent_5_trap_mean_remaining_steps_at_break")


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
    assert "applications" not in run.full
    # Fresh applications cannot retroactively increase transition-start exposure.
    np.testing.assert_array_equal(run.full["active_steps"], 0)
    values = _values(run)
    for channel in channels:
        assert (
            _value(values, f"agent_{source}_{STATUS_NAMES[channel]}_applications") == 1
        )
        status = STATUS_NAMES[channel]
        if status == "mage_burst":
            # Burst acts on the caster. Its source count retains the whole fact.
            assert _value(values, f"agent_{source}_ultimate_activations") == 1
            continue
        assert (
            _value(values, f"agent_{source}_to_agent_{recipient}_{status}_applications")
            == 1
        )
        team = "a" if source < 5 else "b"
        assert (
            _value(values, f"team_{team}_to_agent_{recipient}_{status}_applications")
            == 1
        )
        assert (
            _value(
                values,
                f"agent_{source}_to_agent_{recipient}_{status}_applications_allocation_fraction",
            )
            == 1
        )
        assert (
            _value(
                values,
                f"agent_{source}_to_agent_{recipient}_{status}_applications_contribution_fraction",
            )
            == 1
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
    assert _value(values, "agent_5_trap_intervals") == intervals
    assert _value(values, "agent_5_trap_breaks") == int(broken)
    assert _value(values, "agent_5_trap_break_rate") == int(broken) / intervals
    if broken:
        assert _value(values, "agent_5_trap_mean_remaining_steps_at_break") == 2
    else:
        _missing(values, "agent_5_trap_mean_remaining_steps_at_break")
    assert _value(values, "agent_0_hunter_trap_applications") == 1
    assert _value(values, "agent_5_hunter_trap_active_steps") == int(duration > 0)


def test_two_hunters_start_one_recipient_trap_period_with_two_applications() -> None:
    run = _start(
        classes=((0, 3), (1, 3)),
        arrange=lambda state: state._replace(
            agent_positions=state.agent_positions.at[0]
            .set((7, 1.5))
            .at[1]
            .set((7, 2.5))
        ),
    )
    run, info = _advance(run, _actions((0, 5, True), (1, 5, True)))
    combat = info.transition_facts.combat_transition_facts
    applied = combat.stun_is_applied_by_source_and_channel
    np.testing.assert_array_equal(applied[:2, 1], True)
    assert int(run.state.stun_durations[5, 1]) == 4
    values = _values(run)
    assert _value(values, "team_a_hunter_trap_applications") == 2
    assert _value(values, "agent_5_trap_intervals") == 1
    assert _value(values, "agent_5_trap_breaks") == 0
    assert _value(values, "agent_5_trap_break_rate") == 0
    assert _value(values, "agent_5_hunter_trap_active_steps") == 0
    _missing(values, "agent_5_trap_mean_remaining_steps_at_break")
    assert _value(values, "agent_6_trap_intervals") == 0
    _missing(values, "agent_6_trap_break_rate")
    _missing(values, "agent_4_trap_intervals")


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
    assert _value(values, "agent_5_trap_intervals") == 2
    assert _value(values, "agent_5_trap_break_rate") == 0.5
    assert _value(values, "agent_5_trap_mean_remaining_steps_at_break") == 2
    assert _value(values, "agent_0_trap_break_contributions") == 1
    assert _value(values, "agent_1_trap_break_contributions") == 1
    assert _value(values, "agent_5_trap_breaks") == 1
    for source in (0, 1):
        assert (
            _value(values, f"agent_{source}_to_agent_5_trap_break_contributions") == 1
        )
        assert (
            _value(
                values,
                f"agent_{source}_to_agent_5_trap_break_contributions_allocation_fraction",
            )
            == 1
        )
        assert (
            _value(values, f"agent_{source}_to_agent_5_trap_break_participation") == 1
        )
        assert (
            _value(values, f"agent_{source}_to_agent_5_damage_to_hunter_trap_recipient")
            > 0
        )
    assert _value(values, "team_a_kills_of_hunter_trap_recipient") == int(lethal)
    if lethal:
        assert _value(values, "agent_5_deaths_while_hunter_trap") == 1
        for source in (0, 1):
            assert (
                _value(
                    values,
                    f"agent_{source}_to_agent_5_kill_contributions_to_hunter_trap_recipient",
                )
                == 1
            )
            assert (
                _value(
                    values,
                    f"agent_{source}_to_agent_5_kill_participation_in_hunter_trap_recipient",
                )
                == 1
            )
        assert (
            _value(values, "agent_0_kill_participation_in_hunter_trap_recipient") == 1
        )
        assert (
            _value(values, "agent_1_kill_participation_in_hunter_trap_recipient") == 1
        )
        assert _value(values, "team_a_multi_contributor_kills") == 1
    else:
        run, info = _advance(run, _actions((0, 5, False)))
        lifecycle = info.transition_facts.status_lifecycle_facts
        assert bool(lifecycle.broken_by_damage_by_recipient_and_status_channel[5, 4])
        values = _values(run)
        assert _value(values, "agent_5_trap_intervals") == 2
        assert _value(values, "agent_5_trap_breaks") == 2
        assert _value(values, "agent_5_trap_break_rate") == 1
        # The first break had two ticks left; the second had three.
        assert _value(values, "agent_5_trap_mean_remaining_steps_at_break") == 2.5
        assert _value(values, "team_a_trap_mean_remaining_steps_at_break") == 2.5


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
        assert _value(
            values, f"agent_5_to_agent_0_damage_to_{status}_recipient"
        ) == pytest.approx(damage)
        assert _value(
            values, f"agent_0_damage_received_while_{status}"
        ) == pytest.approx(damage)
        assert _value(
            values, f"agent_2_to_agent_0_healing_to_{status}_recipient"
        ) == pytest.approx(healing)
        assert _value(
            values, f"agent_0_healing_received_while_{status}"
        ) == pytest.approx(healing)
        assert (
            _value(
                values,
                f"agent_2_to_agent_0_healing_to_{status}_recipient_allocation_fraction",
            )
            == 1
        )
        assert (
            _value(
                values,
                f"agent_2_to_agent_0_healing_to_{status}_recipient_contribution_fraction",
            )
            == 1
        )
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
    for source in (0, 1):
        for recipient in (0, 1, 2):
            prefix = f"agent_{source}_to_agent_{recipient}_mage_aura_"
            assert _value(values, prefix + "covered_steps") == 1
            assert _value(
                values, prefix + "covered_steps_allocation_fraction"
            ) == pytest.approx(1 / 3)
            if source != recipient:
                assert _value(values, prefix + "eligible_steps") == 1
                assert _value(values, prefix + "coverage") == 1
                # Both emitters covered this ally. Their shares can sum above
                # one, while the team's covered time counts the ally once.
                assert (
                    _value(values, prefix + "covered_steps_contribution_fraction") == 1
                )
    assert _value(values, "agent_2_mage_aura_covered_recipient_steps") == 1
    assert _value(values, "agent_3_mage_aura_covered_recipient_steps") == 0
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
    assert _value(values, "agent_0_and_agent_1_ally_distance_observations") == 2
    assert _value(values, "agent_0_and_agent_2_ally_distance_observations") == 1
    assert _value(values, "agent_1_and_agent_2_ally_distance_mean") == 5
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


@pytest.mark.parametrize(
    ("healers", "blocked"),
    ((0, False), (1, False), (2, False), (2, True)),
)
def test_recipient_rescue_opportunities_use_combined_legal_healing(
    healers: int, blocked: bool
) -> None:
    run = _start(
        team_sizes=(4, 2),
        classes=((3, 5), (5, 1)),
        arrange=lambda state: state._replace(
            agent_positions=state.agent_positions.at[2]
            .set((6, 2.5))
            .at[3]
            .set((7, 2.5)),
            current_health=state.current_health.at[0].set(1),
            ultimate_cooldowns=state.ultimate_cooldowns.at[2].set(5).at[3].set(5),
            stun_durations=state.stun_durations.at[3, 0].set(int(blocked)),
        ),
    )
    run = run._replace(config=run.config._replace(max_steps=1))
    initial = _values(run)
    assert _value(initial, "agent_0_rescue_opportunities") == 0
    targets = _actions((2, 0, False), (3, 0, False)).select_target
    mask = run.mask.select_target_use_ultimate_joint_mask
    assert bool(mask[2, targets[2], 0])
    assert bool(mask[3, targets[3], 0]) == (not blocked)
    assert not bool(mask[2, targets[2], 1])
    assert not bool(mask[3, targets[3], 1])
    choices = ((5, 0, False), *((2, 0, False), (3, 0, False))[:healers])
    run, info = _advance(run, _actions(*choices))
    combat = info.transition_facts.combat_transition_facts
    damage = float(combat.total_effective_damage_by_recipient[0])
    healing = float(combat.total_effective_healing_by_recipient[0])
    assert damage > 1
    saved = healers == 2 and not blocked
    assert (1 + healing > damage) == saved
    assert bool(run.state.alive_mask[0]) == saved
    values = _values(run)
    # The chance belongs to the threatened Mage. Two available Priests still
    # create one chance, even if neither Priest actually chooses to heal.
    assert _value(values, "agent_0_rescue_opportunities") == int(not blocked)
    assert _value(values, "team_a_rescue_opportunities") == int(not blocked)
    assert _value(values, "agent_0_rescues") == int(saved)
    for recipient in (1, 2, 3):
        assert _value(values, f"agent_{recipient}_rescue_opportunities") == 0
    _missing(values, "agent_4_rescue_opportunities")
    _missing(values, "agent_5_rescue_opportunities")
    assert int(run.state.step_count) == run.config.max_steps
    no_transition = build_canonical_no_transition_info_object(run.state)
    padded = _update(run.full, run.config, run.state, run.mask, no_transition)
    after_padding = _values(run._replace(full=padded))
    np.testing.assert_array_equal(after_padding.values, values.values)
    np.testing.assert_array_equal(after_padding.valid, values.valid)


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
