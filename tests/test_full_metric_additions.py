"""Public transitions for effective output, Ultimate credit and respawn waves."""

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
    start_run as _start,
)
from tests.full_metric_fixtures import update_metrics
from tests.full_metric_fixtures import (
    value as _value,
)
from tests.full_metric_fixtures import (
    values as _values,
)

from marl_battlegrounds.core.env import build_canonical_no_transition_info_object
from marl_battlegrounds.core.types import EnvState


def test_fully_excess_basic_healing_stays_zero_after_fractional_ultimate_excess() -> (
    None
):
    run = _start(
        classes=((5, 1),),
        arrange=lambda state: state._replace(
            agent_positions=state.agent_positions.at[1]
            .set((1, 3.5))
            .at[2]
            .set((5, 2.5))
        ),
    )
    run, basic = _advance(run, _actions((2, 0, False)))
    combat = basic.transition_facts.combat_transition_facts
    assert combat.basic_effect_is_activated_by_source[2]
    assert float(run.state.current_health[0]) == 80
    for _ in range(4):
        run, damage = _advance(run, _actions((5, 0, False)))
        combat = damage.transition_facts.combat_transition_facts
        assert combat.basic_effect_is_activated_by_source[5]
    run, ultimate = _advance(run, _actions((2, 0, True), (5, 0, False)))
    combat = ultimate.transition_facts.combat_transition_facts
    assert combat.ultimate_effect_is_activated_by_source[2]
    assert float(run.state.current_health[0]) == 80
    # The observed Ultimate excess is fractional, while the earlier Basic8 was
    # entirely excess. Separate episode totals round at different magnitudes.
    assert float(run.full["ultimate_excess"][2, 0]) == pytest.approx(
        125.25, rel=0, abs=2e-5
    )
    result = _values(run)
    for subject in ("agent_2", "team_a", "agent_2_to_agent_0"):
        assert _value(result, f"{subject}_basic_effective_healing_done") == 0
        assert _value(result, f"{subject}_basic_excess_healing") == 8
        assert _value(result, f"{subject}_basic_effective_healing_fraction") == 0
        assert _value(result, f"{subject}_basic_excess_healing_fraction") == 1
    _missing(
        result, "agent_2_to_agent_0_basic_effective_healing_done_allocation_fraction"
    )
    # Every source shares the genuinely zero recipient denominator, including
    # active non-healers whose available general healing amount is zero.
    for source in (0, 1, 2):
        _missing(
            result,
            f"agent_{source}_to_agent_0_basic_effective_healing_done_contribution_fraction",
        )


def test_small_basic_excess_survives_a_later_fully_excess_ultimate() -> None:
    initial_health = np.nextafter(np.float32(72), np.float32(np.inf))
    run = _start(
        arrange=lambda state: state._replace(
            agent_positions=state.agent_positions.at[2].set((5, 2.5)),
            current_health=state.current_health.at[0].set(initial_health),
        )
    )
    run, basic = _advance(run, _actions((2, 0, False)))
    combat = basic.transition_facts.combat_transition_facts
    assert combat.basic_effect_is_activated_by_source[2]
    assert float(run.state.current_health[0]) == 80
    excess = float(initial_health) + 8 - 80
    effective = 8 - excess
    before = _values(run)
    run, ultimate = _advance(run, _actions((2, 0, True)))
    combat = ultimate.transition_facts.combat_transition_facts
    assert combat.ultimate_effect_is_activated_by_source[2]
    assert float(run.state.current_health[0]) == 80
    after = _values(run)
    # The small but real Basic excess must survive an Ultimate 26 million times
    # larger. An epsilon would erase evidence instead of preserving it.
    assert excess == 2**-17
    for values in (before, after):
        for subject in ("agent_2", "team_a", "agent_2_to_agent_0"):
            assert _value(values, f"{subject}_basic_excess_healing") == excess
            assert (
                _value(values, f"{subject}_basic_effective_healing_done") == effective
            )
            assert (
                _value(values, f"{subject}_basic_excess_healing_fraction") == excess / 8
            )
            assert (
                _value(values, f"{subject}_basic_effective_healing_fraction")
                == effective / 8
            )
        for fraction in ("allocation_fraction", "contribution_fraction"):
            assert (
                _value(values, f"agent_2_to_agent_0_basic_excess_healing_{fraction}")
                == 1
            )


@pytest.mark.parametrize("missing_health", (0.0, 4.0, 40.0))
def test_effective_healing_splits_simultaneous_basic_and_ultimate_excess(
    missing_health: float,
) -> None:
    def arrange(state: EnvState) -> EnvState:
        return state._replace(
            agent_positions=state.agent_positions.at[2]
            .set((5, 2.5))
            .at[3]
            .set((7, 2.5)),
            current_health=state.current_health.at[0].set(80 - missing_health),
        )

    run = _start(team_sizes=(4, 2), classes=((3, 5),), arrange=arrange)
    run, info = _advance(run, _actions((2, 0, False), (3, 0, True)))
    combat = info.transition_facts.combat_transition_facts
    assert combat.basic_effect_is_activated_by_source[2]
    assert combat.ultimate_effect_is_activated_by_source[3]
    values = _values(run)
    # 8 Basic + 200 Ultimate HP share the same health-cap excess proportionally.
    assert _value(values, "team_a_healing_done") == 208
    assert _value(values, "team_a_effective_healing_done") == pytest.approx(
        missing_health, abs=2e-5
    )
    assert _value(values, "agent_0_effective_priest_healing_received") == pytest.approx(
        missing_health, abs=2e-5
    )
    assert _value(values, "agent_2_effective_healing_done") == pytest.approx(
        missing_health * 8 / 208, abs=2e-5
    )
    assert _value(values, "agent_3_effective_healing_done") == pytest.approx(
        missing_health * 200 / 208, abs=2e-5
    )
    assert _value(
        values, "agent_3_to_agent_0_ultimate_effective_healing_done"
    ) == pytest.approx(missing_health * 200 / 208, abs=2e-5)
    for prefix in ("agent_2", "agent_3", "team_a"):
        assert _value(values, f"{prefix}_effective_healing_fraction") == pytest.approx(
            missing_health / 208, abs=1e-6
        )
        assert _value(values, f"{prefix}_excess_healing_fraction") == pytest.approx(
            1 - missing_health / 208, abs=1e-6
        )
    assert _value(
        values, "agent_3_ultimate_effective_healing_fraction"
    ) == pytest.approx(missing_health / 208, abs=1e-6)
    assert _value(values, "agent_3_ultimate_excess_healing_fraction") == pytest.approx(
        1 - missing_health / 208, abs=1e-6
    )
    assert _value(
        values, "agent_0_effective_priest_healing_received_fraction"
    ) == pytest.approx(missing_health / 208, abs=1e-6)
    assert _value(
        values, "agent_0_excess_priest_healing_received_fraction"
    ) == pytest.approx(1 - missing_health / 208, abs=1e-6)
    assert _value(values, "agent_0_effective_healing_done") == 0
    _missing(values, "agent_0_effective_healing_fraction")
    _missing(values, "agent_4_effective_healing_done")


@pytest.mark.parametrize("useful", (False, True))
def test_only_useful_ultimate_priest_support_prevents_solo_kills(useful: bool) -> None:
    run = _start(
        arrange=lambda state: state._replace(
            agent_positions=state.agent_positions.at[2].set((5, 2.5)),
            current_health=state.current_health.at[0]
            .set(79 if useful else 80)
            .at[5]
            .set(1),
        )
    )
    run, info = _advance(run, _actions((0, 5, False), (2, 0, True)))
    assert info.transition_facts.death_facts.is_newly_dead_by_recipient[5]
    values = _values(run)
    assert _value(values, "agent_0_solo_kills") == int(not useful)
    assert _value(values, "agent_0_solo_kill_fraction") == int(not useful)
    assert _value(values, "team_a_single_contributor_kills") == int(not useful)
    assert _value(values, "team_a_multi_contributor_kills") == int(useful)
    assert _value(values, "agent_2_ultimate_kill_contributions") == int(useful)
    assert _value(values, "agent_2_ultimate_kill_participation") == int(useful)
    assert _value(values, "team_a_ultimate_kills") == int(useful)
    assert _value(values, "team_a_ultimate_kill_fraction") == int(useful)
    assert _value(values, "agent_0_basic_kill_participation") == 1
    assert _value(values, "agent_2_basic_kill_participation") == 0
    assert _value(values, "team_a_basic_kill_fraction") == 1
    assert _value(values, "agent_2_solo_kills") == 0
    if useful:
        assert _value(values, "agent_2_solo_kill_fraction") == 0
    else:
        _missing(values, "agent_2_solo_kill_fraction")

    # Basic Priest healing earns the same help only when some healing fits.
    basic_run = _start(
        arrange=lambda state: state._replace(
            agent_positions=state.agent_positions.at[2].set((5, 2.5)),
            current_health=state.current_health.at[0]
            .set(79 if useful else 80)
            .at[5]
            .set(1),
        )
    )
    basic_run, basic_info = _advance(basic_run, _actions((0, 5, False), (2, 0, False)))
    assert basic_info.transition_facts.death_facts.is_newly_dead_by_recipient[5]
    basic_values = _values(basic_run)
    assert _value(basic_values, "agent_0_basic_kill_participation") == 1
    assert _value(basic_values, "agent_2_basic_kill_contributions") == int(useful)
    assert _value(basic_values, "agent_2_basic_kill_participation") == int(useful)
    assert _value(basic_values, "team_a_basic_kills") == 1
    assert _value(basic_values, "team_a_basic_kill_fraction") == 1
    assert _value(basic_values, "team_a_mage_basic_kills") == 1
    assert _value(basic_values, "team_a_mage_basic_kill_fraction") == 1
    assert _value(basic_values, "team_a_priest_basic_kills") == int(useful)
    assert _value(basic_values, "team_a_priest_basic_kill_fraction") == int(useful)
    assert _value(basic_values, "team_a_ultimate_kill_fraction") == 0
    assert _value(basic_values, "team_a_single_contributor_kill_fraction") == int(
        not useful
    )
    assert _value(basic_values, "team_a_multi_contributor_kill_fraction") == int(useful)


def test_two_ultimate_healers_participate_in_one_rescue_with_simultaneous_damage() -> (
    None
):
    run = _start(
        team_sizes=(4, 2),
        classes=((3, 5), (5, 1)),
        arrange=lambda state: state._replace(
            agent_positions=state.agent_positions.at[2]
            .set((5, 2.5))
            .at[3]
            .set((7, 2.5)),
            current_health=state.current_health.at[0].set(1),
        ),
    )
    run, info = _advance(run, _actions((2, 0, True), (3, 0, True), (5, 0, False)))
    assert (
        info.transition_facts.combat_transition_facts.total_effective_damage_by_recipient[
            0
        ]
        > 1
    )
    values = _values(run)
    assert _value(values, "team_a_rescues") == 1
    assert _value(values, "team_a_ultimate_rescues") == 1
    assert _value(values, "team_a_ultimate_rescue_fraction") == 1
    for source in (2, 3):
        assert _value(values, f"agent_{source}_ultimate_rescue_contributions") == 1
        assert _value(values, f"agent_{source}_ultimate_rescue_participation") == 1
        assert _value(values, f"agent_{source}_to_agent_0_ultimate_applications") == 1
        assert (
            _value(values, f"agent_{source}_to_agent_0_ultimate_application_fraction")
            == 1
        )
    # Effective healing offsets damage as well as the observed health increase.
    damage = _value(values, "agent_0_damage_received")
    assert _value(values, "agent_0_effective_healing_received") == pytest.approx(
        79 + damage, abs=3e-5
    )
    assert _value(values, "team_a_ultimate_effective_healing_done") == pytest.approx(
        79 + damage, abs=3e-5
    )


def test_burst_self_application_is_separate_from_later_damage_and_kills() -> None:
    run = _start(
        arrange=lambda state: state._replace(
            current_health=state.current_health.at[5].set(1)
        )
    )
    run, _ = _advance(
        run,
        neutral_action()._replace(use_ultimate=jnp.zeros(10, jnp.int32).at[0].set(1)),
    )
    initial = _values(run)
    assert _value(initial, "agent_0_to_agent_0_ultimate_applications") == 1
    assert _value(initial, "agent_0_to_agent_0_ultimate_application_fraction") == 1
    _missing(initial, "agent_0_to_agent_5_ultimate_application_fraction")
    _missing(initial, "agent_0_to_agent_5_ultimate_applications")
    assert _value(initial, "agent_0_ultimate_damage_done") == 0
    assert _value(initial, "agent_0_to_agent_5_burst_damage") == 0
    assert _value(initial, "team_a_mage_basic_kills") == 0
    _missing(initial, "team_a_mage_basic_kill_fraction")
    run, _ = _advance(run, _actions((0, 5, False)))
    values = _values(run)
    assert (
        _value(values, "agent_0_to_agent_5_burst_damage")
        == _value(values, "agent_0_burst_damage")
        > 1
    )
    assert _value(values, "agent_0_to_agent_5_ultimate_damage_done") == 0
    assert _value(values, "team_a_ultimate_kills") == 0
    assert _value(values, "agent_0_solo_kills") == 1
    assert _value(values, "team_a_mage_basic_kills") == 1
    assert _value(values, "team_a_mage_basic_kill_fraction") == 1


def test_respawn_wave_counts_and_sizes_match_each_public_transition() -> None:
    run = _start(
        arrange=lambda state: state._replace(
            alive_mask=state.alive_mask.at[1:3].set(False),
            current_health=state.current_health.at[1:3].set(0).at[5].set(1),
        ),
    )
    initial = _values(run)
    for team in ("a", "b"):
        assert _value(initial, f"team_{team}_respawn_waves") == 0
        _missing(initial, f"team_{team}_mean_agents_per_respawn_wave")
    assert _value(initial, "team_a_mean_observed_respawn_wait_steps") == 0
    _missing(initial, "team_b_mean_observed_respawn_wait_steps")
    wave_sizes: tuple[list[int], list[int]] = ([], [])
    dead_steps = np.zeros(10, dtype=np.int32)
    observed_periods = [2, 0]
    for tick in range(14):
        dead_steps += np.asarray(
            run.config.agent_profile.active_mask & ~run.state.alive_mask,
            dtype=np.int32,
        )
        run, info = _advance(
            run, _actions((0, 5, False)) if tick == 0 else neutral_action()
        )
        facts = info.transition_facts.respawn_facts
        for team, sizes in enumerate(wave_sizes):
            observed_periods[team] += int(
                np.count_nonzero(
                    info.transition_facts.death_facts.is_newly_dead_by_recipient[
                        team * 5 : team * 5 + 5
                    ]
                )
            )
            returned = np.asarray(
                facts.was_respawned_this_transition_by_agent[team * 5 : team * 5 + 5]
            )
            if facts.respawn_wave_occurred_this_transition_by_team[team]:
                sizes.append(int(np.count_nonzero(returned)))
            else:
                assert not returned.any()
        values = _values(run)
        for team, sizes in zip(("a", "b"), wave_sizes, strict=True):
            assert _value(values, f"team_{team}_respawn_waves") == len(sizes)
            if sizes:
                assert _value(
                    values, f"team_{team}_mean_agents_per_respawn_wave"
                ) == pytest.approx(sum(sizes) / len(sizes))
            else:
                _missing(values, f"team_{team}_mean_agents_per_respawn_wave")
        for slot in (1, 2, 5):
            assert _value(values, f"agent_{slot}_dead_steps") == dead_steps[slot]
        assert _value(values, "team_a_dead_steps") == sum(dead_steps[:5])
        assert _value(values, "team_b_dead_steps") == sum(dead_steps[5:])
        for team, label in enumerate(("a", "b")):
            assert _value(
                values, f"team_{label}_mean_observed_respawn_wait_steps"
            ) == pytest.approx(
                sum(dead_steps[team * 5 : team * 5 + 5]) / observed_periods[team]
            )
        assert _value(values, "agent_1_deaths") == 0
        assert _value(values, "agent_2_deaths") == 0
        assert _value(values, "agent_5_deaths") == 1
    # A due wave still counts when everyone on that team is already alive.
    assert wave_sizes == ([2, 0], [1, 0])
    # Padding must not add waves, deaths, activations or time spent dead.
    no_transition = build_canonical_no_transition_info_object(run.state)
    padded = update_metrics(run.full, run.config, run.state, run.mask, no_transition)
    for before, after in zip(
        jax.tree.leaves(run.full), jax.tree.leaves(padded), strict=True
    ):
        np.testing.assert_array_equal(before, after)


def test_terminal_death_keeps_death_counts_without_adding_a_respawn_wave() -> None:
    run = _start(
        arrange=lambda state: state._replace(
            alive_mask=state.alive_mask.at[1].set(False),
            current_health=state.current_health.at[0].set(1).at[1].set(0),
        ),
    )
    run = run._replace(config=run.config._replace(max_steps=2))
    run, _ = _advance(run, neutral_action())
    run, info = _advance(run, _actions((5, 0, False)))
    assert info.transition_facts.death_facts.is_newly_dead_by_recipient[0]
    assert run.state.step_count == run.config.max_steps
    values = _values(run)
    assert _value(values, "agent_0_deaths") == 1
    assert _value(values, "agent_0_dead_steps") == 0
    assert _value(values, "agent_1_deaths") == 0
    assert _value(values, "agent_1_dead_steps") == 2
    assert _value(values, "team_a_deaths") == 1
    assert _value(values, "team_a_dead_steps") == 2
    # The final-tick death adds a period with no observed dead steps yet.
    # Together with the initially dead ally, that gives two steps / two periods.
    assert _value(values, "team_a_mean_observed_respawn_wait_steps") == 1
    _missing(values, "team_b_mean_observed_respawn_wait_steps")
    for team in ("a", "b"):
        assert _value(values, f"team_{team}_respawn_waves") == 0
        _missing(values, f"team_{team}_mean_agents_per_respawn_wave")
    padding = build_canonical_no_transition_info_object(run.state)
    padded = update_metrics(run.full, run.config, run.state, run.mask, padding)
    final = _values(run._replace(full=padded))
    np.testing.assert_array_equal(final.values, values.values)
    np.testing.assert_array_equal(final.valid, values.valid)


def test_basic_recipient_allocation_and_contribution_use_different_denominators() -> (
    None
):
    run = _start(
        classes=((0, 3), (1, 3)),
        arrange=lambda state: state._replace(
            agent_positions=state.agent_positions.at[0]
            .set((5, 1.5))
            .at[1]
            .set((5, 2.5))
            .at[2]
            .set((4, 1.5))
            .at[5]
            .set((7.5, 1.5))
            .at[6]
            .set((7.5, 2.5)),
        ),
    )
    for choices in (
        ((0, 5, False), (1, 5, False), (2, 0, False)),
        ((0, 6, False), (1, 5, False), (2, 1, False)),
    ):
        run, info = _advance(run, _actions(*choices))
        np.testing.assert_array_equal(
            info.transition_facts.combat_transition_facts.basic_effect_is_activated_by_source[
                :3
            ],
            True,
        )
    values = _values(run)
    assert _value(values, "team_a_basic_activations") == 6
    assert _value(values, "agent_0_basic_activations") == 2
    assert _value(values, "agent_0_to_agent_5_basic_applications") == 1
    assert _value(values, "agent_0_to_agent_5_basic_application_fraction") == 0.5
    assert _value(
        values, "agent_0_to_agent_5_basic_application_contribution_fraction"
    ) == pytest.approx(1 / 3)
    assert _value(values, "team_a_to_agent_5_basic_applications") == 3
    assert _value(values, "team_a_to_agent_0_basic_applications") == 1
    assert _value(values, "team_a_to_agent_5_hunter_basic_slow_applications") == 3
    assert _value(values, "agent_0_to_agent_5_hunter_basic_slow_applications") == 1
    assert (
        _value(
            values,
            "agent_0_to_agent_5_hunter_basic_slow_applications_allocation_fraction",
        )
        == 0.5
    )
    assert _value(
        values,
        "agent_0_to_agent_5_hunter_basic_slow_applications_contribution_fraction",
    ) == pytest.approx(1 / 3)
    _missing(values, "agent_0_to_agent_1_basic_applications")
    _missing(values, "agent_2_to_agent_5_basic_applications")
    _missing(values, "team_b_to_agent_5_basic_applications")
    assert "activations" not in run.full
    assert not any("status_application" in name for name in run.full)


def test_mixed_basic_and_repeated_class_ultimate_kills_count_each_event_once() -> None:
    # Move the same repeated-class team across the fixed slot boundary.
    for team, first, enemy, other_enemy in (("a", 0, 5, 6), ("b", 5, 0, 1)):
        second, basic = first + 1, first + 2

        def arrange(
            state: EnvState,
            first: int = first,
            second: int = second,
            basic: int = basic,
            enemy: int = enemy,
            other_enemy: int = other_enemy,
            basic_attacks: bool = False,
        ) -> EnvState:
            return state._replace(
                current_health=state.current_health.at[enemy].set(1),
                agent_positions=state.agent_positions.at[first]
                .set((7 if basic_attacks else 5, 1.5))
                .at[second]
                .set((7 if basic_attacks else 5, 3.5))
                .at[basic]
                .set((6.5, 2.5))
                .at[enemy]
                .set((8, 2.5))
                .at[other_enemy]
                .set((9, 4.5)),
            )

        run = _start(
            team_sizes=(3, 3),
            classes=((first, 2), (second, 2), (basic, 1)),
            arrange=arrange,
        )
        initial = _values(run)
        for source in (first, second, basic, enemy):
            _missing(initial, f"agent_{source}_basic_kill_participation")
        for side in ("a", "b"):
            _missing(initial, f"team_{side}_basic_kill_fraction")
        for class_name in ("mage", "warrior"):
            assert _value(initial, f"team_{team}_{class_name}_basic_kills") == 0
            _missing(initial, f"team_{team}_{class_name}_basic_kill_fraction")
        run, info = _advance(
            run,
            _actions(
                (first, enemy, True), (second, enemy, True), (basic, enemy, False)
            ),
        )
        combat = info.transition_facts.combat_transition_facts
        np.testing.assert_array_equal(
            combat.ultimate_effect_is_activated_by_source[jnp.asarray((first, second))],
            True,
        )
        assert combat.basic_effect_is_activated_by_source[basic]
        assert info.transition_facts.death_facts.is_newly_dead_by_recipient[enemy]
        values = _values(run)
        for stem in (
            "kills",
            "basic_kills",
            "ultimate_kills",
            "warrior_ultimate_kills",
            "basic_kill_fraction",
            "ultimate_kill_fraction",
            "warrior_ultimate_kill_fraction",
        ):
            assert _value(values, f"team_{team}_{stem}") == 1
        assert _value(values, f"agent_{basic}_basic_kill_participation") == 1
        assert _value(values, f"team_{team}_mage_basic_kills") == 1
        assert _value(values, f"team_{team}_mage_basic_kill_fraction") == 1
        assert _value(values, f"team_{team}_warrior_basic_kills") == 0
        assert _value(values, f"team_{team}_warrior_basic_kill_fraction") == 0
        assert _value(values, f"agent_{basic}_ultimate_kill_contributions") == 0
        assert _value(values, f"agent_{basic}_ultimate_kill_participation") == 0
        assert (
            _value(values, f"agent_{basic}_to_agent_{enemy}_basic_kill_contributions")
            == 1
        )
        assert (
            _value(
                values, f"agent_{basic}_to_agent_{enemy}_ultimate_kill_contributions"
            )
            == 0
        )
        for source in (first, second):
            assert _value(values, f"agent_{source}_basic_kill_participation") == 0
            assert _value(values, f"agent_{source}_ultimate_kill_participation") == 1
            assert (
                _value(
                    values,
                    f"agent_{source}_to_agent_{enemy}_ultimate_kill_contributions",
                )
                == 1
            )
            assert (
                _value(
                    values,
                    f"agent_{source}_to_agent_{enemy}_ultimate_kill_participation",
                )
                == 1
            )
            assert (
                _value(
                    values,
                    f"agent_{source}_to_agent_{enemy}_ultimate_kill_contributions_allocation_fraction",
                )
                == 1
            )
            assert _value(values, f"agent_{source}_solo_kills") == 0
        # Opponents have no kills; inactive slots stay missing despite team kills.
        _missing(values, f"agent_{enemy}_basic_kill_participation")
        _missing(values, f"team_{'b' if team == 'a' else 'a'}_basic_kill_fraction")
        _missing(values, f"agent_{first + 3}_basic_kill_participation")

        # Repeated classes share one event, while different classes may overlap.
        for basic_id, basic_name in ((1, "mage"), (4, "rogue")):
            basic_run = _start(
                team_sizes=(3, 3),
                classes=((first, 2), (second, 2), (basic, basic_id)),
                arrange=lambda state, arrange=arrange: arrange(
                    state, basic_attacks=True
                ),
            )
            basic_run, basic_info = _advance(
                basic_run,
                _actions(
                    (first, enemy, False),
                    (second, enemy, False),
                    (basic, enemy, False),
                ),
            )
            assert basic_info.transition_facts.death_facts.is_newly_dead_by_recipient[
                enemy
            ]
            basic_values = _values(basic_run)
            for source in (first, second, basic):
                assert (
                    _value(basic_values, f"agent_{source}_basic_kill_contributions")
                    == 1
                )
            assert _value(basic_values, f"team_{team}_basic_kills") == 1
            for class_name in ("mage", "warrior", "hunter", "rogue", "priest"):
                for stem in ("kills", "kill_fraction"):
                    name = f"team_{team}_{class_name}_basic_{stem}"
                    if class_name in ("warrior", basic_name):
                        assert _value(basic_values, name) == 1
                    else:
                        _missing(basic_values, name)


def test_mixed_class_ultimate_output_fraction_keeps_the_generic_denominator() -> None:
    run = _start(
        classes=((0, 2), (1, 3)),
        arrange=lambda state: state._replace(
            current_health=state.current_health.at[5].set(1).at[6].set(1),
            agent_positions=state.agent_positions.at[1].set((7, 3)),
        ),
    )
    run, info = _advance(run, _actions((0, 5, True), (1, 5, True)))
    combat = info.transition_facts.combat_transition_facts
    damage = np.asarray(combat.source_modified_damage_output_by_source) * np.asarray(
        combat.recipient_damage_modifier_by_source
    )
    assert damage[0] > 0 and damage[1] > 0
    values = _values(run)
    assert _value(values, "team_a_warrior_ultimate_kills") == 1
    assert _value(values, "team_a_hunter_ultimate_kills") == 1
    assert _value(values, "team_a_ultimate_kills") == 1
    assert _value(values, "team_a_basic_kill_fraction") == 0
    assert _value(values, "team_a_hunter_basic_kills") == 0
    assert _value(values, "team_a_hunter_basic_kill_fraction") == 0
    for source in (0, 1):
        assert _value(values, f"agent_{source}_basic_kill_participation") == 0
    assert _value(
        values, "agent_0_to_agent_5_ultimate_damage_done_contribution_fraction"
    ) == pytest.approx(float(damage[0] / damage[:2].sum()))
    assert _value(values, "agent_0_ultimate_damage_done") == pytest.approx(
        float(damage[0])
    )
    assert _value(values, "agent_1_ultimate_damage_done") == pytest.approx(
        float(damage[1])
    )
    # A later Basic-only kill makes each category one of two team kills.
    run, basic_info = _advance(run, _actions((1, 6, False)))
    assert basic_info.transition_facts.death_facts.is_newly_dead_by_recipient[6]
    final = _values(run)
    assert _value(final, "team_a_kills") == 2
    assert _value(final, "team_a_basic_kills") == 1
    assert _value(final, "team_a_basic_kill_fraction") == 0.5
    assert _value(final, "team_a_ultimate_kill_fraction") == 0.5
    assert _value(final, "agent_1_basic_kill_participation") == 0.5
    assert _value(final, "agent_0_basic_kill_participation") == 0
    assert _value(final, "agent_1_ultimate_kill_participation") == 0.5
    assert _value(final, "team_a_hunter_basic_kills") == 1
    assert _value(final, "team_a_hunter_basic_kill_fraction") == 0.5


def test_basic_and_ultimate_healers_share_one_recipient_rescue() -> None:
    run = _start(
        team_sizes=(4, 2),
        classes=((3, 5), (5, 1), (6, 1)),
        arrange=lambda state: state._replace(
            agent_positions=state.agent_positions.at[2]
            .set((5, 2.5))
            .at[3]
            .set((7, 2.5)),
            current_health=state.current_health.at[0].set(1).at[1].set(1),
        ),
    )
    run, info = _advance(run, _actions((2, 0, False), (3, 0, True), (5, 0, False)))
    assert (
        info.transition_facts.combat_transition_facts.total_effective_damage_by_recipient[
            0
        ]
        > 1
    )
    values = _values(run)
    for name in (
        "agent_0_rescues",
        "team_a_rescues",
        "team_a_basic_rescues",
        "team_a_basic_rescue_fraction",
        "team_a_ultimate_rescues",
        "team_a_ultimate_rescue_fraction",
    ):
        assert _value(values, name) == 1
    for source, ability in ((2, "basic"), (3, "ultimate")):
        assert _value(values, f"agent_{source}_{ability}_rescue_participation") == 1
        assert (
            _value(values, f"agent_{source}_to_agent_0_{ability}_rescue_contributions")
            == 1
        )
        assert (
            _value(values, f"agent_{source}_to_agent_0_{ability}_rescue_participation")
            == 1
        )
        assert (
            _value(
                values,
                f"agent_{source}_to_agent_0_{ability}_rescue_contributions_allocation_fraction",
            )
            == 1
        )
    _missing(values, "agent_0_to_agent_1_basic_rescue_contributions")
    assert _value(values, "agent_3_basic_rescue_participation") == 0

    # A later Ultimate-only save belongs in the Basic shares' denominator too.
    run, info = _advance(run, _actions((2, 1, True), (6, 1, False)))
    combat = info.transition_facts.combat_transition_facts
    assert combat.ultimate_effect_is_activated_by_source[2]
    assert combat.basic_effect_is_activated_by_source[6]
    assert run.state.alive_mask[1]
    final = _values(run)
    assert _value(final, "team_a_rescues") == 2
    assert _value(final, "team_a_basic_rescues") == 1
    assert _value(final, "team_a_basic_rescue_fraction") == 0.5
    assert _value(final, "agent_2_basic_rescue_participation") == 0.5
    assert _value(final, "agent_3_basic_rescue_participation") == 0
    assert _value(final, "team_a_ultimate_rescues") == 2
    assert _value(final, "team_a_ultimate_rescue_fraction") == 1
    for source in (2, 3):
        assert _value(final, f"agent_{source}_ultimate_rescue_participation") == 0.5


@pytest.mark.parametrize("team_offset", (0, 5))
def test_basic_rescue_shares_count_one_save_with_repeated_moved_priests(
    team_offset: int,
) -> None:
    recipient = team_offset
    first, second = team_offset + 1, team_offset + 2
    inactive = team_offset + 4
    enemy = 5 - team_offset
    team, other_team = ("a", "b") if team_offset == 0 else ("b", "a")
    run = _start(
        team_sizes=(4, 4),
        classes=((recipient, 1), (first, 5), (second, 5), (inactive, 5), (enemy, 1)),
        arrange=lambda state: state._replace(
            agent_positions=state.agent_positions.at[recipient]
            .set((6, 1.5))
            .at[first]
            .set((5, 2.5))
            .at[second]
            .set((7, 2.5))
            .at[enemy]
            .set((9, 1.5)),
            current_health=state.current_health.at[recipient].set(1),
        ),
    )
    run = run._replace(config=run.config._replace(max_steps=1))
    initial = _values(run)
    for source in range(10):
        _missing(initial, f"agent_{source}_basic_rescue_participation")
    for side in ("a", "b"):
        _missing(initial, f"team_{side}_basic_rescue_fraction")

    run, info = _advance(
        run,
        _actions(
            (first, recipient, False),
            (second, recipient, False),
            (enemy, recipient, False),
        ),
    )
    combat = info.transition_facts.combat_transition_facts
    for source in (first, second, enemy):
        assert combat.basic_effect_is_activated_by_source[source]
    assert combat.total_effective_damage_by_recipient[recipient] > 1
    assert run.state.alive_mask[recipient]
    assert run.state.step_count == run.config.max_steps
    values = _values(run)
    # Both Priests get help credit, but their team saved only one ally.
    assert _value(values, f"team_{team}_rescues") == 1
    assert _value(values, f"team_{team}_basic_rescues") == 1
    assert _value(values, f"team_{team}_basic_rescue_fraction") == 1
    assert _value(values, f"team_{team}_ultimate_rescue_fraction") == 0
    for source in (first, second):
        assert _value(values, f"agent_{source}_basic_rescue_contributions") == 1
        assert _value(values, f"agent_{source}_basic_rescue_participation") == 1
        assert _value(values, f"agent_{source}_ultimate_rescue_participation") == 0
    _missing(values, f"agent_{recipient}_basic_rescue_participation")
    _missing(values, f"agent_{inactive}_basic_rescue_participation")
    _missing(values, f"team_{other_team}_basic_rescue_fraction")

    # Padding after the final tick must keep every saved value and count.
    padding = build_canonical_no_transition_info_object(run.state)
    padded = update_metrics(run.full, run.config, run.state, run.mask, padding)
    final = _values(run._replace(full=padded))
    np.testing.assert_array_equal(final.values, values.values)
    np.testing.assert_array_equal(final.valid, values.valid)
    for before, after in zip(
        jax.tree.leaves(run.full), jax.tree.leaves(padded), strict=True
    ):
        np.testing.assert_array_equal(before, after)


def test_pair_healing_efficiency_preserves_delivery_denominator_and_recipient() -> None:
    run = _start(
        team_sizes=(4, 2),
        classes=((3, 5),),
        arrange=lambda state: state._replace(
            agent_positions=state.agent_positions.at[2]
            .set((5, 2.5))
            .at[3]
            .set((7, 2.5)),
            current_health=state.current_health.at[0].set(76),
        ),
    )
    run, _ = _advance(run, _actions((2, 0, False), (3, 0, True)))
    values = _values(run)
    for source, ability in ((2, "basic"), (3, "ultimate")):
        for prefix in ("", ability + "_"):
            assert _value(
                values, f"agent_{source}_to_agent_0_{prefix}effective_healing_fraction"
            ) == pytest.approx(4 / 208, abs=1e-6)
            assert _value(
                values, f"agent_{source}_to_agent_0_{prefix}excess_healing_fraction"
            ) == pytest.approx(204 / 208, abs=1e-6)
    assert _value(values, "agent_0_basic_priest_healing_received") == 8
    assert _value(values, "agent_0_ultimate_priest_healing_received") == 200
    assert _value(
        values, "agent_0_basic_effective_priest_healing_received"
    ) == pytest.approx(4 * 8 / 208, abs=2e-5)
    assert _value(values, "team_a_basic_effective_healing_fraction") == pytest.approx(
        4 / 208, abs=1e-6
    )
    _missing(values, "agent_2_to_agent_1_basic_effective_healing_fraction")
