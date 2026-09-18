"""Check historical basic reports and their recorded task and schema rules."""

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
from tests.evaluation_fixtures import (
    historical_initialize_scenario_state as initialize_scenario_state,
)
from tests.evaluation_fixtures import historical_reset as reset
from tests.evaluation_fixtures import historical_step as step

from marl_battlegrounds.core.axis_mappings import (
    GLOBAL_RECIPIENT_SLOT_BY_ACTOR_AND_TARGET_ACTION,
)
from marl_battlegrounds.core.types import Action
from marl_battlegrounds.evaluation.capture import (
    capture_evaluation_transition_unit_v1,
    capture_initial_evaluation_frame_v1,
)
from marl_battlegrounds.evaluation.metrics import (
    AgentStatisticSubjectV1,
    CountComponentV1,
    EpisodeStatisticSubjectV1,
    EvaluationEpisodeObserverV1,
    EvaluationMetricReportV1,
    RawSufficientStatisticV1,
    StatisticSubjectV1,
    SufficientStatisticDraftV1,
    SumComponentV1,
    TeamStatisticSubjectV1,
)
from marl_battlegrounds.evaluation.models import (
    EvaluationTransitionV1,
)
from marl_battlegrounds.evaluation.reducers import (
    TDM_BASIC_METRIC_IDS,
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
    horizon: int | None = None,
) -> CapturedEvaluationTrajectory:
    config = evaluation_env_config(
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


def _report(case: CapturedEvaluationTrajectory) -> EvaluationMetricReportV1:
    observer = EvaluationEpisodeObserverV1(case.context, build_tdm_metric_reducers())
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
) -> RawSufficientStatisticV1:
    matches = tuple(
        row
        for row in report.statistics
        if row.metric_id == f"marlbg.{metric}.v1"
        and row.subject == subject
        and row.component_name == "value"
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


def test_archived_suite_contains_exactly_the_five_basic_metric_ids() -> None:
    report = _report(captured_team_deathmatch_threshold_trajectory())
    assert len(build_tdm_metric_reducers()) == 1
    assert set(TDM_BASIC_METRIC_IDS) == {
        "marlbg.task.outcome_distribution.v1",
        "marlbg.task.terminal_score_differential.v1",
        "marlbg.task.evaluation_return.v1",
        "marlbg.task.episode_length.v1",
        "marlbg.artifact.completion.v1",
    }
    assert {row.metric_id for row in report.statistics} == set(TDM_BASIC_METRIC_IDS)


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
    completion = _row(
        report,
        "artifact.completion",
        EpisodeStatisticSubjectV1(),
        dimensions=(("rollout", "partial"), ("processing", "succeeded")),
    )
    assert completion.result_status == "defined"
    assert _value(completion) == 1


def test_padding_is_excluded_from_archived_agent_returns() -> None:
    case = captured_evaluation_trajectory(transition_count=1)
    report = _report(case)
    assert all(
        not isinstance(row.subject, AgentStatisticSubjectV1)
        or row.subject.global_slot in (0, 1, 2, 5, 6)
        for row in report.statistics
    )


@pytest.mark.parametrize("horizon", (2, 3))
def test_preview_matches_strict_schema_validation_at_every_captured_boundary(
    horizon: int,
) -> None:
    case = _capture(
        (_actions((0, None, True)), _actions((0, 5, False))),
        horizon=horizon,
    )
    observer = EvaluationEpisodeObserverV1(case.context, build_tdm_metric_reducers())
    observer.start(case.frames[0])
    for index in range(len(case.frames)):
        if index:
            observer.append(case.transitions[index - 1], case.frames[index])
        complete = index == horizon
        # Complete and partial prefixes keep strict historical row eligibility.
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
    observer = EvaluationEpisodeObserverV1(case.context, build_tdm_metric_reducers())
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
