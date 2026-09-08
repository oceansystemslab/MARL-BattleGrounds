"""Episode reducers are measured against public apply/observe trajectories."""

import re
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from tests.evaluation_fixtures import (
    CapturedEvaluationTrajectory,
    captured_evaluation_trajectory,
    captured_resumed_team_deathmatch_horizon_trajectory,
    captured_team_deathmatch_threshold_trajectory,
    evaluation_context,
    evaluation_env_config,
    neutral_action,
)

from marl_battlegrounds.core.axis_mappings import (
    GLOBAL_RECIPIENT_SLOT_BY_ACTOR_AND_TARGET_ACTION,
)
from marl_battlegrounds.core.config import resolve_agent_profile
from marl_battlegrounds.core.env import initialize_scenario_state, reset, step
from marl_battlegrounds.core.types import Action, EnvConfig, EnvState
from marl_battlegrounds.evaluation.capture import (
    capture_evaluation_transition_unit_v1,
    capture_initial_evaluation_frame_v1,
)
from marl_battlegrounds.evaluation.metrics import (
    AgentStatisticSubjectV1,
    CountComponentV1,
    DistributionComponentV1,
    DurationComponentV1,
    EvaluationEpisodeObserverV1,
    EvaluationMetricReportV1,
    RatioComponentV1,
    RawSufficientStatisticV1,
    StatisticSubjectV1,
    SufficientStatisticDraftV1,
    SumComponentV1,
    TeamClassStatisticSubjectV1,
    TeamStatisticSubjectV1,
)
from marl_battlegrounds.evaluation.models import (
    EvaluationTransitionV1,
)
from marl_battlegrounds.evaluation.reducers import (
    TDM_BASIC_METRIC_IDS,
    TDM_EPISODE_METRIC_IDS,
    build_tdm_metric_reducers,
)


def _actions(*choices: tuple[int, int | None, bool]) -> Action:
    action = neutral_action()
    for actor, recipient, ultimate in choices:
        target = GLOBAL_RECIPIENT_SLOT_BY_ACTOR_AND_TARGET_ACTION[actor].index(
            recipient
        )
        action = action._replace(
            select_target=action.select_target.at[actor].set(target),
            use_ultimate=action.use_ultimate.at[actor].set(int(ultimate)),
        )
    return action


def _capture(
    actions: tuple[Action, ...],
    *,
    change_state: Callable[[EnvState], EnvState] | None = None,
    config: EnvConfig | None = None,
    horizon: int | None = None,
) -> CapturedEvaluationTrajectory:
    config = config or evaluation_env_config(
        team_sizes=(5, 5),
        task_mode=1,
        team_deathmatch_score_threshold=20,
        max_steps=len(actions) if horizon is None else horizon,
    )
    context = evaluation_context(config=config, expected_horizon=config.max_steps)
    state, _, _, _ = reset(config, jax.random.key(0))
    positions = jnp.asarray(
        [(6.0, 1.5 + 2 * slot) for slot in range(5)]
        + [(9.0, 1.5 + 2 * slot) for slot in range(5)],
        dtype=jnp.float32,
    )
    state = state._replace(
        agent_positions=positions, spawn_shield_durations=jnp.zeros(10, dtype=jnp.int32)
    )
    if change_state is not None:
        state = change_state(state)
    state, observation, mask, _ = initialize_scenario_state(state, config)
    frame = capture_initial_evaluation_frame_v1(context, state, observation, mask)
    frames = [frame]
    transitions: list[EvaluationTransitionV1] = []
    for index, action in enumerate(actions):
        state, observation, reward, done, mask, info = step(
            config,
            state,
            mask,
            action,
            jax.random.key(index + 1),
        )
        transition, frame = capture_evaluation_transition_unit_v1(
            context,
            frame,
            state,
            observation,
            mask,
            info.transition_facts,
            reward,
            done,
        )
        frames.append(frame)
        transitions.append(transition)
    return CapturedEvaluationTrajectory(context, tuple(frames), tuple(transitions))


def _report(
    case: CapturedEvaluationTrajectory, *, full: bool = True
) -> EvaluationMetricReportV1:
    observer = EvaluationEpisodeObserverV1(
        case.context, build_tdm_metric_reducers(full=full)
    )
    observer.start(case.frames[0])
    for index, transition in enumerate(case.transitions):
        observer.append(transition, case.frames[index + 1])
    complete = bool(case.transitions and case.transitions[-1].terminated) or (
        len(case.transitions) == case.context.expected_horizon
    )
    report = observer.finalize(
        completion_state="complete" if complete else "partial",
        end_or_failure_reason=None if complete else "test capture stopped",
    )
    assert report.processing_status.status == "succeeded", report.processing_status
    assert (
        EvaluationMetricReportV1.model_validate(report.model_dump(mode="python"))
        == report
    )
    return report


def _row(
    report: EvaluationMetricReportV1,
    metric: str,
    subject: StatisticSubjectV1,
    *,
    dimensions: tuple[tuple[str, str], ...] = (),
    component: str = "value",
) -> RawSufficientStatisticV1:
    matches = tuple(
        row
        for row in report.statistics
        if row.metric_id == f"marlbg.{metric}.v1"
        and row.subject == subject
        and row.component_name == component
        and tuple((d.name, d.value) for d in row.dimensions)
        == tuple(sorted(dimensions))
    )
    assert len(matches) == 1, (metric, subject, dimensions, matches)
    return matches[0]


def _value(row: RawSufficientStatisticV1) -> float:
    component = row.component
    if isinstance(component, CountComponentV1):
        return float(component.count)
    assert isinstance(component, SumComponentV1), row
    return component.value


@pytest.fixture(scope="module")
def neutral_report() -> EvaluationMetricReportV1:
    return _report(_capture((neutral_action(),)))


def test_suite_covers_the_46_normative_episode_ids(
    neutral_report: EvaluationMetricReportV1,
) -> None:
    specification = (
        Path(__file__).parents[1] / "docs/evaluation/metric_specification.md"
    ).read_text()
    episode_tables = specification.split("## Canonical task-independent metrics")[
        1
    ].split("## Future and pending task-owned metrics")[0]
    expected = set(re.findall(r"^\| `(marlbg\.[^`]+)`", episode_tables, re.MULTILINE))
    assert len(expected) == 46
    assert set(TDM_EPISODE_METRIC_IDS) == expected
    assert {row.metric_id for row in neutral_report.statistics} == expected


def test_default_suite_contains_only_critical_rows_from_the_full_report() -> None:
    case = captured_team_deathmatch_threshold_trajectory()
    basic = _report(case, full=False)
    full = _report(case, full=True)
    assert len(build_tdm_metric_reducers()) == 1
    assert len(build_tdm_metric_reducers(full=True)) == 3
    assert set(TDM_BASIC_METRIC_IDS) == {
        "marlbg.task.outcome_distribution.v1",
        "marlbg.task.terminal_score_differential.v1",
        "marlbg.task.evaluation_return.v1",
        "marlbg.task.episode_length.v1",
        "marlbg.artifact.completion.v1",
    }
    assert {row.metric_id for row in basic.statistics} == set(TDM_BASIC_METRIC_IDS)
    assert basic.statistics == tuple(
        row for row in full.statistics if row.metric_id in TDM_BASIC_METRIC_IDS
    )
    assert basic.processing_status == full.processing_status
    assert basic.completion == full.completion


def test_threshold_outcome_and_horizon_draw_use_task_authority() -> None:
    team = TeamStatisticSubjectV1(team_id=1)
    win = _report(captured_team_deathmatch_threshold_trajectory())
    assert (
        _value(
            _row(
                win, "task.outcome_distribution", team, dimensions=(("outcome", "win"),)
            )
        )
        == 1
    )
    horizon = _report(captured_resumed_team_deathmatch_horizon_trajectory())
    assert (
        _value(
            _row(
                horizon,
                "task.outcome_distribution",
                team,
                dimensions=(("outcome", "draw"),),
            )
        )
        == 1
    )
    assert _value(_row(horizon, "task.terminal_score_differential", team)) != 0


def test_partial_prefix_keeps_counts_but_has_no_outcome_or_terminal_score() -> None:
    report = _report(_capture((neutral_action(),), horizon=100))
    team = TeamStatisticSubjectV1(team_id=1)
    assert (
        _row(
            report, "task.outcome_distribution", team, dimensions=(("outcome", "draw"),)
        ).result_status
        == "insufficient_data"
    )
    assert (
        _row(report, "task.terminal_score_differential", team).result_status
        == "insufficient_data"
    )
    assert _row(report, "combat.death_count", team).result_status == "defined"
    assert _value(_row(report, "combat.death_count", team)) == 0


def test_zero_opportunity_and_undefined_priority_are_explicit(
    neutral_report: EvaluationMetricReportV1,
) -> None:
    team = TeamStatisticSubjectV1(team_id=1)
    assert (
        _row(
            neutral_report, "coordination.focus_fire_concentration", team
        ).result_status
        == "zero_opportunity"
    )
    assert (
        _row(neutral_report, "rogue.priority_target_damage_share", team).result_status
        == "insufficient_data"
    )
    assert (
        _row(
            neutral_report,
            "support.recipient_modified_gross_healing_output",
            AgentStatisticSubjectV1(global_slot=0),
        ).result_status
        == "structurally_inapplicable"
    )


def test_padding_is_excluded_and_absent_classes_are_not_zero() -> None:
    case = captured_evaluation_trajectory(transition_count=1)
    report = _report(case)
    team = TeamStatisticSubjectV1(team_id=1)
    ratio = _row(report, "diagnostic.action_acceptance_rate", team).component
    assert isinstance(ratio, RatioComponentV1)
    assert ratio.denominator == 3
    absent = _row(
        report,
        "rogue.priority_target_damage_share",
        TeamClassStatisticSubjectV1(team_id=1, class_id=4),
    )
    assert absent.result_status == "structurally_inapplicable"
    assert all(
        not isinstance(row.subject, AgentStatisticSubjectV1)
        or row.subject.global_slot in (0, 1, 2, 5, 6)
        for row in report.statistics
    )


def test_gross_lethal_damage_is_not_recipient_net_health_loss() -> None:
    case = _capture(
        (_actions((0, 5, False)),),
        change_state=lambda s: s._replace(
            current_health=s.current_health.at[5].set(1.0)
        ),
    )
    report = _report(case)
    source = AgentStatisticSubjectV1(global_slot=0)
    assert (
        _value(
            _row(
                report,
                "mage.burst_enemy_death_association",
                source,
                component="activations",
            )
        )
        == 0
    )
    assert (
        _value(
            _row(
                report,
                "mage.burst_enemy_death_association",
                source,
                component="team_enemy_deaths",
            )
        )
        == 1
    )
    recipient = AgentStatisticSubjectV1(global_slot=5)
    damage = _value(
        _row(report, "combat.recipient_modified_gross_damage_output", source)
    )
    assert damage > 1
    assert _value(_row(report, "combat.realized_net_health_change", recipient)) == -1
    assert _value(_row(report, "combat.death_count", recipient)) == 1
    assert (
        _value(_row(report, "combat.lethal_transition_damage_contribution", source))
        == 1
    )
    assert (
        _value(
            _row(report, "combat.recipient_modified_gross_damage_received", recipient)
        )
        == damage
    )
    share = _row(
        report,
        "combat.recipient_modified_gross_damage_output",
        source,
        component="team_share",
    ).component
    assert isinstance(share, RatioComponentV1)
    assert share.numerator == share.denominator == damage


def test_multi_contributor_denominator_counts_death_once() -> None:
    def arrange(state: EnvState) -> EnvState:
        return state._replace(
            agent_positions=state.agent_positions.at[3].set((6.0, 2.5)),
            current_health=state.current_health.at[5].set(1.0),
        )

    report = _report(
        _capture((_actions((0, 5, False), (3, 5, False)),), change_state=arrange)
    )
    team = TeamStatisticSubjectV1(team_id=1)
    rate = _row(report, "combat.lethal_transition_contribution_rate", team).component
    assert isinstance(rate, RatioComponentV1)
    assert (rate.numerator, rate.denominator) == (2, 1)
    multiple = _row(
        report, "coordination.multi_contributor_lethal_transition_rate", team
    ).component
    assert isinstance(multiple, RatioComponentV1)
    assert (multiple.numerator, multiple.denominator) == (1, 1)


def test_burst_activation_is_observed_before_its_damage_window() -> None:
    case = _capture((_actions((0, None, True)), _actions((0, 5, False))))
    report = _report(case)
    source = AgentStatisticSubjectV1(global_slot=0)
    assert (
        _value(
            _row(report, "mage.burst_window_damage", source, component="activations")
        )
        == 1
    )
    assert _value(_row(report, "mage.burst_window_damage", source)) == _value(
        _row(report, "combat.recipient_modified_gross_damage_output", source)
    )
    assert _value(_row(report, "ability.ultimate_cooldown_start_count", source)) == 1
    assert (
        _value(
            _row(
                report,
                "ability.activation_count",
                source,
                dimensions=(("ability", "ultimate"),),
            )
        )
        == 1
    )
    assert case.frames[1].snapshot.mage_burst_damage_amplification_durations[0] > 0


@pytest.mark.parametrize("duration", (0, 1, 3))
def test_status_uptime_uses_the_start_of_each_consumed_transition(
    duration: int,
) -> None:
    case = _capture(
        (neutral_action(), neutral_action(), neutral_action()),
        change_state=lambda s: s._replace(
            slow_durations=s.slow_durations.at[0, 0].set(duration)
        ),
    )
    report = _report(case)
    row = _row(
        report,
        "status.active_recipient_steps",
        AgentStatisticSubjectV1(global_slot=0),
        dimensions=(("status", "warrior_charge_slow"),),
    )
    assert isinstance(row.component, DurationComponentV1)
    assert (row.component.qualifying_steps, row.component.eligible_steps) == (
        duration,
        3,
    )


def test_trap_break_and_same_transition_reapplication_keep_distinct_episodes() -> None:
    def arrange(state: EnvState) -> EnvState:
        return state._replace(
            agent_positions=state.agent_positions.at[3].set((8.0, 2.5)),
            stun_durations=state.stun_durations.at[5, 1].set(2),
        )

    case = _capture((_actions((0, 5, False), (3, 5, True)),), change_state=arrange)
    report = _report(case)
    team = TeamStatisticSubjectV1(team_id=1)
    rate = _row(report, "hunter.trap_damage_break_rate", team).component
    assert isinstance(rate, RatioComponentV1)
    assert (rate.numerator, rate.denominator) == (1, 1)
    assert _value(_row(report, "hunter.trap_status_episode_end", team)) == 1
    assert any(
        row.metric_id == "marlbg.hunter.trap_status_episode_end.v1"
        and row.component_name == "open_intervals"
        and _value(row) == 1
        for row in report.statistics
    )
    assert case.frames[1].snapshot.stun_durations[5][1] > 0


def test_freedom_binding_excludes_stun_and_shield_and_keeps_stay_eligible() -> None:
    def arrange(state: EnvState) -> EnvState:
        return state._replace(
            slow_durations=state.slow_durations.at[0:3, 0].set(2),
            priest_blessing_of_freedom_slow_floor_durations=state.priest_blessing_of_freedom_slow_floor_durations.at[
                0:3
            ].set(1),
            stun_durations=state.stun_durations.at[1, 0].set(1),
            spawn_shield_durations=state.spawn_shield_durations.at[2].set(2),
        )

    report = _report(_capture((neutral_action(),), change_state=arrange))
    team = TeamStatisticSubjectV1(team_id=1)
    component = _row(report, "priest.freedom_binding_coverage", team).component
    assert isinstance(component, DurationComponentV1)
    assert (component.qualifying_steps, component.eligible_steps) == (1, 1)


def test_movement_distributions_keep_zero_stay_and_separate_phase_vectors() -> None:
    move = neutral_action()._replace(move=neutral_action().move.at[0].set(4))
    case = _capture((move,))
    report = _report(case)
    actor = AgentStatisticSubjectV1(global_slot=0)
    for phase, expected in (("charge", 0), ("ordinary", -1)):
        component = _row(
            report,
            "movement.phase_displacement",
            actor,
            dimensions=(("phase", phase), ("axis", "x")),
        ).component
        assert isinstance(component, DistributionComponentV1)
        assert len(component.observations) == 1
        assert component.observations[0].value == pytest.approx(expected)


def test_reducer_finalization_is_pure_and_preserves_distribution_ids() -> None:
    case = _capture((neutral_action(), neutral_action()))
    observer = EvaluationEpisodeObserverV1(
        case.context, build_tdm_metric_reducers(full=True)
    )
    observer.start(case.frames[0])
    for index, transition in enumerate(case.transitions):
        observer.append(transition, case.frames[index + 1])
    before = tuple(state.model_dump_json() for state in observer.reducer_states or ())
    report = observer.finalize(completion_state="complete")
    after = tuple(state.model_dump_json() for state in observer.reducer_states or ())
    assert before == after
    for reducer, state in zip(
        build_tdm_metric_reducers(full=True), observer.reducer_states or (), strict=True
    ):
        first = reducer.finalize(state, report.completion, report.processing_status)
        second = reducer.finalize(state, report.completion, report.processing_status)
        assert first == second
        for row in first:
            if isinstance(row.component, DistributionComponentV1):
                ids = [
                    observation.source_observation_id
                    for observation in row.component.observations
                ]
                assert len(ids) == len(set(ids))


def test_simultaneous_healing_retains_gross_amount_clamp_and_unique_rescue() -> None:
    def arrange(state: EnvState) -> EnvState:
        return state._replace(
            agent_positions=state.agent_positions.at[2].set((6.0, 2.5))
        )

    actions = (_actions((2, 0, False), (5, 0, False)),)
    saturated = _report(_capture(actions, change_state=arrange))
    healer = AgentStatisticSubjectV1(global_slot=2)
    recipient = AgentStatisticSubjectV1(global_slot=0)
    assert (
        _value(
            _row(saturated, "support.recipient_modified_gross_healing_output", healer)
        )
        == 8
    )
    assert _value(_row(saturated, "combat.realized_net_health_change", recipient)) == 0
    assert (
        _row(
            saturated,
            "combat.upper_health_clamp_overflow",
            recipient,
            component="gross_healing_exposure",
        ).amount_stage
        == "recipient_modified_gross"
    )
    for component, amount in (("gross_damage", 5.1), ("gross_healing", 8.0)):
        companion = _row(
            saturated,
            "combat.realized_net_health_change",
            recipient,
            component=component,
        )
        assert _value(companion) == pytest.approx(amount)
        assert companion.units == "HP"
        assert companion.amount_stage == "recipient_modified_gross"
    assert _value(
        _row(saturated, "combat.upper_health_clamp_overflow", recipient)
    ) == pytest.approx(2.9)

    def endangered(state: EnvState) -> EnvState:
        return arrange(state)._replace(
            current_health=state.current_health.at[0].set(1.0)
        )

    saved = _report(_capture(actions, change_state=endangered))
    assert (
        _value(
            _row(
                saved,
                "priest.same_transition_lethal_damage_rescue",
                TeamStatisticSubjectV1(team_id=1),
                component="recipient_opportunities",
            )
        )
        == 1
    )
    assert (
        _value(
            _row(
                saved,
                "priest.same_transition_lethal_damage_rescue",
                TeamStatisticSubjectV1(team_id=1),
            )
        )
        == 1
    )
    assert (
        _value(
            _row(
                saved,
                "priest.same_transition_lethal_damage_rescue",
                healer,
                dimensions=(("attribution", "unique_healer"),),
            )
        )
        == 1
    )


def test_duplicate_healers_preserve_recipient_rescue_without_dividing_credit() -> None:
    config = evaluation_env_config(
        team_sizes=(5, 5), task_mode=1, team_deathmatch_score_threshold=20, max_steps=1
    )
    classes = config.agent_profile.class_ids.at[3].set(5)
    config = config._replace(
        agent_profile=resolve_agent_profile(
            classes, jnp.asarray((5, 5), dtype=jnp.int32)
        )
    )

    def arrange(state: EnvState) -> EnvState:
        return state._replace(
            agent_positions=state.agent_positions.at[2]
            .set((6.0, 2.5))
            .at[3]
            .set((7.0, 2.5)),
            current_health=state.current_health.at[0].set(1.0),
        )

    report = _report(
        _capture(
            (_actions((2, 0, False), (3, 0, False), (5, 0, False)),),
            config=config,
            change_state=arrange,
        )
    )
    assert (
        _value(
            _row(
                report,
                "priest.same_transition_lethal_damage_rescue",
                TeamStatisticSubjectV1(team_id=1),
            )
        )
        == 1
    )
    for slot in (2, 3):
        row = _row(
            report,
            "priest.same_transition_lethal_damage_rescue",
            AgentStatisticSubjectV1(global_slot=slot),
            dimensions=(("attribution", "ambiguous_healers"),),
        )
        assert row.result_status == "ambiguous_attribution"
        assert row.component is None


def test_anti_heal_reduction_is_combined_and_regeneration_stays_separate() -> None:
    def arrange(state: EnvState) -> EnvState:
        return state._replace(
            agent_positions=state.agent_positions.at[2].set((6.0, 2.5)),
            current_health=state.current_health.at[0].set(40.0),
            rogue_poison_anti_heal_durations=state.rogue_poison_anti_heal_durations.at[
                0
            ].set(2),
        )

    case = _capture((_actions((2, 0, False)),), change_state=arrange)
    report = _report(case)
    for component, amount, stage in (
        ("source_modified_healing_exposure", 8, "source_modified_gross"),
        ("recipient_modified_healing_exposure", 4, "recipient_modified_gross"),
    ):
        exposure = _row(
            report,
            "rogue.combined_anti_heal_reduction",
            TeamStatisticSubjectV1(team_id=2),
            component=component,
        )
        assert _value(exposure) == amount
        assert exposure.units == "HP"
        assert exposure.amount_stage == stage
    active = _row(
        report,
        "rogue.combined_anti_heal_reduction",
        TeamStatisticSubjectV1(team_id=2),
        component="active_recipient_steps",
    ).component
    assert isinstance(active, DurationComponentV1)
    assert (active.qualifying_steps, active.eligible_steps) == (1, 5)
    assert (
        _value(
            _row(
                report,
                "support.recipient_modified_gross_healing_output",
                AgentStatisticSubjectV1(global_slot=2),
            )
        )
        == 4
    )
    assert (
        _value(
            _row(
                report,
                "rogue.combined_anti_heal_reduction",
                TeamStatisticSubjectV1(team_id=2),
            )
        )
        == 4
    )
    regeneration = _value(
        _row(
            report,
            "recovery.realized_regeneration",
            AgentStatisticSubjectV1(global_slot=0),
        )
    )
    assert (
        regeneration
        == case.transitions[
            0
        ].facts.regeneration_facts.actual_health_regenerated_this_step_by_agent[0]
    )
    assert case.frames[1].snapshot.current_health[0] == pytest.approx(44 + regeneration)


def test_initial_dead_agent_respawns_on_its_wave_without_padding_exposure() -> None:
    def arrange(state: EnvState) -> EnvState:
        return state._replace(
            alive_mask=state.alive_mask.at[0].set(False),
            current_health=state.current_health.at[0].set(0),
            team_respawn_wave_countdowns=state.team_respawn_wave_countdowns.at[0].set(
                0
            ),
        )

    case = _capture((neutral_action(),), change_state=arrange)
    report = _report(case)
    actor = AgentStatisticSubjectV1(global_slot=0)
    dead = _row(report, "lifecycle.dead_agent_steps", actor).component
    assert isinstance(dead, DurationComponentV1)
    assert (dead.qualifying_steps, dead.eligible_steps) == (1, 1)
    assert (
        _value(
            _row(
                report,
                "lifecycle.respawn_wave",
                actor,
                dimensions=(("event", "agent_respawn"),),
            )
        )
        == 1
    )
    assert (
        _value(
            _row(
                report,
                "lifecycle.respawn_wave",
                TeamStatisticSubjectV1(team_id=1),
                dimensions=(("event", "team_wave"),),
            )
        )
        == 1
    )
    assert case.frames[1].snapshot.alive_mask[0]


def test_death_clear_after_break_reapplication_closes_two_trap_episodes() -> None:
    def arrange(state: EnvState) -> EnvState:
        return state._replace(
            agent_positions=state.agent_positions.at[3].set((8.0, 2.5)),
            stun_durations=state.stun_durations.at[5, 1].set(2),
            current_health=state.current_health.at[5].set(1),
        )

    report = _report(
        _capture((_actions((0, 5, False), (3, 5, True)),), change_state=arrange)
    )
    team = TeamStatisticSubjectV1(team_id=1)
    assert (
        _value(
            _row(
                report,
                "hunter.trap_active_steps",
                team,
                component="applications",
            )
        )
        == 1
    )
    assert _value(_row(report, "hunter.trap_status_episode_end", team)) == 2
    rate = _row(report, "hunter.trap_damage_break_rate", team).component
    assert isinstance(rate, RatioComponentV1)
    assert (rate.numerator, rate.denominator) == (1, 2)
    assert not any(
        row.metric_id == "marlbg.hunter.trap_status_episode_end.v1"
        and row.component_name == "open_intervals"
        for row in report.statistics
    )


def test_failed_healing_rescue_preserves_nonzero_recipient_opportunity() -> None:
    def arrange(state: EnvState) -> EnvState:
        return state._replace(
            agent_positions=state.agent_positions.at[2].set((6.0, 2.5)),
            current_health=state.current_health.at[0].set(1.0),
            rogue_poison_anti_heal_durations=state.rogue_poison_anti_heal_durations.at[
                0
            ].set(2),
        )

    report = _report(
        _capture(
            (_actions((2, 0, False), (5, 0, False)),),
            change_state=arrange,
        )
    )
    team = TeamStatisticSubjectV1(team_id=1)
    metric = "priest.same_transition_lethal_damage_rescue"
    assert _value(_row(report, metric, team)) == 0
    for dimensions in ((), (("healer_count", "1"),)):
        assert (
            _value(
                _row(
                    report,
                    metric,
                    team,
                    component="recipient_opportunities",
                    dimensions=dimensions,
                )
            )
            == 1
        )


def test_exact_upper_clamp_boundary_does_not_invent_overflow() -> None:
    def arrange(state: EnvState) -> EnvState:
        return state._replace(
            agent_positions=state.agent_positions.at[2]
            .set((6.0, 2.5))
            .at[8]
            .set((7.2, 1.5)),
            current_health=state.current_health.at[0].set(78.8),
        )

    case = _capture((_actions((2, 0, False), (8, 0, False)),), change_state=arrange)
    assert (
        case.transitions[
            0
        ].facts.combat_transition_facts.health_after_combat_resolution_by_recipient[0]
        == 80
    )

    assert (
        _value(
            _row(
                _report(case),
                "combat.upper_health_clamp_overflow",
                AgentStatisticSubjectV1(global_slot=0),
            )
        )
        == 0
    )


@pytest.mark.parametrize("horizon", (2, 3))
def test_preview_matches_strict_schema_validation_at_every_captured_boundary(
    horizon: int,
) -> None:
    case = _capture(
        (_actions((0, None, True)), _actions((0, 5, False))),
        horizon=horizon,
    )
    observer = EvaluationEpisodeObserverV1(
        case.context, build_tdm_metric_reducers(full=True)
    )
    observer.start(case.frames[0])
    for index in range(len(case.frames)):
        if index:
            observer.append(case.transitions[index - 1], case.frames[index])
        complete = index == horizon
        # Includes class N/A, absent target definitions, and zero opportunities.
        drafts = observer.preview_statistics(
            completion_state="complete" if complete else "partial"
        )
        assert (
            tuple(
                SufficientStatisticDraftV1.model_validate(
                    draft.model_dump(mode="python")
                )
                for draft in drafts
            )
            == drafts
        )
    assert observer.finalized_report is None


def test_external_report_validation_rejects_malformed_computed_rows() -> None:
    report = _report(_capture((neutral_action(),)))
    malformed = report.model_copy(
        update={
            "statistics": (
                report.statistics[0].model_copy(update={"units": ""}),
                *report.statistics[1:],
            )
        }
    )
    with pytest.raises(ValueError):
        EvaluationMetricReportV1.model_validate(malformed.model_dump(mode="python"))


def test_metric_computation_never_serializes_accumulated_reducer_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _capture((neutral_action(), neutral_action()))
    observer = EvaluationEpisodeObserverV1(
        case.context, build_tdm_metric_reducers(full=True)
    )
    observer.start(case.frames[0])
    states = observer.reducer_states
    assert states

    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("metric computation serialized accumulated state")

    for state_type in {type(state) for state in states}:
        monkeypatch.setattr(state_type, "model_dump", forbidden)
    for index, transition in enumerate(case.transitions):
        observer.append(transition, case.frames[index + 1])
    report = observer.finalize(completion_state="complete")
    assert report.processing_status.status == "succeeded"
    assert report.processing_status.processed_transition_count == 2


def test_duplicate_mage_auras_keep_combined_effect_and_per_emitter_coverage() -> None:
    config = evaluation_env_config(
        team_sizes=(5, 5), task_mode=1, team_deathmatch_score_threshold=20, max_steps=1
    )
    classes = config.agent_profile.class_ids.at[2].set(1)
    config = config._replace(
        agent_profile=resolve_agent_profile(
            classes, jnp.asarray((5, 5), dtype=jnp.int32)
        )
    )

    def arrange(state: EnvState) -> EnvState:
        return state._replace(
            agent_positions=state.agent_positions.at[2]
            .set((6.0, 2.5))
            .at[3]
            .set((7.0, 1.5))
        )

    report = _report(
        _capture((_actions((3, 5, False)),), config=config, change_state=arrange)
    )
    team = TeamStatisticSubjectV1(team_id=1)
    assert _value(
        _row(report, "combat.recipient_modified_gross_damage_output", team)
    ) == pytest.approx(7.935)
    assert _value(
        _row(report, "mage.combined_aura_amplification", team)
    ) == pytest.approx(1.935)
    for emitter in (0, 2):
        coverage = _row(
            report, "mage.aura_coverage", AgentStatisticSubjectV1(global_slot=emitter)
        ).component
        assert isinstance(coverage, DurationComponentV1)
        assert coverage.qualifying_steps >= 3
        assert coverage.eligible_steps == 5
    assert not any(
        row.metric_id == "marlbg.mage.combined_aura_amplification.v1"
        and isinstance(row.subject, AgentStatisticSubjectV1)
        for row in report.statistics
    )


def test_identical_opposing_rosters_have_identical_metric_definitions() -> None:
    config = evaluation_env_config(
        team_sizes=(5, 5), task_mode=1, team_deathmatch_score_threshold=20, max_steps=1
    )
    classes = config.agent_profile.class_ids.at[5:].set(
        config.agent_profile.class_ids[:5]
    )
    config = config._replace(
        agent_profile=resolve_agent_profile(
            classes, jnp.asarray((5, 5), dtype=jnp.int32)
        )
    )
    report = _report(_capture((_actions((0, 5, False), (5, 0, False)),), config=config))
    for metric in (
        "combat.recipient_modified_gross_damage_output",
        "combat.recipient_modified_gross_damage_received",
        "combat.death_count",
        "mage.combined_aura_amplification",
        "warrior.combined_aura_mitigation",
    ):
        assert _value(
            _row(report, metric, TeamStatisticSubjectV1(team_id=1))
        ) == _value(_row(report, metric, TeamStatisticSubjectV1(team_id=2)))
