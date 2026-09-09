"""Basic V1 outcome reports retained for archived replay generation.

Current metric computation uses the public JAX evaluation workflow. This small
legacy producer preserves the five historical outcome statistics and their bytes.
"""

from dataclasses import dataclass
from typing import Literal, cast

from marl_battlegrounds.evaluation.metrics import (
    AgentStatisticSubjectV1,
    CountComponentV1,
    EpisodeStatisticSubjectV1,
    EvaluationEpisodeCompletionV1,
    EvaluationMetricReducerStateV1,
    EvaluationMetricReducerV1,
    EvaluationProcessingStatusV1,
    EvaluationTransitionViewV1,
    StatisticDimensionV1,
    StatisticSubjectV1,
    SufficientStatisticComponentV1,
    SufficientStatisticDraftV1,
    SumComponentV1,
    TeamStatisticSubjectV1,
)
from marl_battlegrounds.evaluation.models import (
    EvaluationEpisodeContextV1,
    EvaluationFrameV1,
    EvaluationModel,
    TeamDeathmatchCompletedEventV1,
)

type _Dimensions = tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class _Definition:
    kind: Literal["count", "sum"]
    units: str
    complete_only: bool = False


_DEFINITIONS = {
    "task.outcome_distribution": _Definition("count", "games", complete_only=True),
    "task.terminal_score_differential": _Definition(
        "sum", "points", complete_only=True
    ),
    "task.evaluation_return": _Definition("sum", "reward", complete_only=True),
    "task.episode_length": _Definition("sum", "transitions", complete_only=True),
    "artifact.completion": _Definition("count", "episodes"),
}
TDM_BASIC_METRIC_IDS = tuple(sorted(f"marlbg.{key}.v1" for key in _DEFINITIONS))


class _Agent(EvaluationModel):
    slot: int


class _Cell(EvaluationModel):
    metric: str
    subject: StatisticSubjectV1
    dimensions: _Dimensions = ()
    value: float = 0.0
    exposure: float = 0.0


class _State(EvaluationMetricReducerStateV1):
    agents: tuple[_Agent, ...]
    cells: tuple[_Cell, ...] = ()
    task_mode: int
    scores: tuple[int, ...] = (0, 0)
    outcome: str = "pending"
    steps: int = 0


def _cell_key(cell: _Cell) -> tuple[str, str, _Dimensions]:
    return cell.metric, cell.subject.model_dump_json(), cell.dimensions


class _Rows:
    """Build one replacement locally; committed reducer state stays immutable."""

    def __init__(self, cells: tuple[_Cell, ...] = ()) -> None:
        self.cells = {_cell_key(cell): cell for cell in cells}

    def add(
        self,
        metric: str,
        subject: StatisticSubjectV1,
        value: float = 0.0,
        exposure: float = 0.0,
        *,
        dimensions: _Dimensions = (),
    ) -> None:
        dimensions = tuple(sorted(dimensions))
        key = metric, subject.model_dump_json(), dimensions
        previous = self.cells.get(key)
        if previous is None:
            previous = _Cell(metric=metric, subject=subject, dimensions=dimensions)
        elif value == 0 and exposure == 0:
            return
        self.cells[key] = previous.model_copy(
            update={
                "value": previous.value + float(value),
                "exposure": previous.exposure + float(exposure),
            }
        )

    def freeze(self) -> tuple[_Cell, ...]:
        return tuple(self.cells[key] for key in sorted(self.cells))


def _team(team: int) -> TeamStatisticSubjectV1:
    return TeamStatisticSubjectV1(team_id=team)


def _state(
    context: EvaluationEpisodeContextV1, frame: EvaluationFrameV1, reducer_id: str
) -> _State:
    return _State(
        reducer_id=reducer_id,
        reducer_version=1,
        task_mode=context.resolved_env_config.task_mode,
        agents=tuple(
            _Agent(slot=row.global_slot)
            for row in context.roster
            if row.configured_active
        ),
        scores=frame.snapshot.team_deathmatch_scores,
    )


def _model[T: EvaluationModel](model: type[T], **values: object) -> T:
    """Package trusted values; external ingestion validates wire records."""
    return model.model_construct(_fields_set=set(values), **values)


def _drafts(
    state: _State, cells: tuple[_Cell, ...]
) -> tuple[SufficientStatisticDraftV1, ...]:
    result: list[SufficientStatisticDraftV1] = []
    for cell in cells:
        definition = _DEFINITIONS[cell.metric]
        component: SufficientStatisticComponentV1 | None = (
            _model(CountComponentV1, count=int(cell.value), eligible_episode_count=1)
            if definition.kind == "count"
            else _model(
                SumComponentV1,
                value=cell.value,
                observation_count=int(cell.exposure),
                eligible_episode_count=1,
            )
        )
        status: Literal["defined", "structurally_inapplicable", "insufficient_data"] = (
            "defined"
        )
        reason: str | None = None
        if state.task_mode != 1 and cell.metric in (
            "task.outcome_distribution",
            "task.terminal_score_differential",
        ):
            # Preserve the historical report's exact status wording.
            status, reason, component = (
                "structurally_inapplicable",
                "class absent or task has no TDM outcome",
                None,
            )
        elif cell.metric == "task.outcome_distribution" and state.outcome == "pending":
            status, reason, component = (
                "insufficient_data",
                "authoritative task outcome not observed",
                None,
            )
        elif (
            cell.metric == "task.evaluation_return"
            and isinstance(cell.subject, TeamStatisticSubjectV1)
            and cell.exposure == 0
        ):
            status, reason, component = (
                "insufficient_data",
                "no task-authored team reward observed",
                None,
            )
        result.append(
            _model(
                SufficientStatisticDraftV1,
                metric_id=f"marlbg.{cell.metric}.v1",
                metric_version=1,
                component_name="value",
                reducer_id=state.reducer_id,
                reducer_version=1,
                units=definition.units,
                amount_stage=None,
                subject=cell.subject,
                dimensions=tuple(
                    _model(StatisticDimensionV1, name=name, value=value)
                    for name, value in cell.dimensions
                ),
                completion_scope="complete_episode"
                if definition.complete_only
                else "any_gap_free_prefix",
                supports_right_censoring=False,
                result_status=status,
                status_reason=reason,
                endpoint_observation_status="unavailable"
                if status == "insufficient_data"
                else "not_applicable",
                component=component,
            )
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class _OutcomeReducer:
    reducer_id: str = "marlbg.episode.outcome"
    reducer_version: int = 1

    def initialize(
        self, context: EvaluationEpisodeContextV1, initial_frame: EvaluationFrameV1
    ) -> _State:
        state = _state(context, initial_frame, self.reducer_id)
        rows = _Rows()
        for agent in state.agents:
            rows.add(
                "task.evaluation_return",
                AgentStatisticSubjectV1(global_slot=agent.slot),
            )
        for team in (1, 2):
            rows.add("task.evaluation_return", _team(team))
        return state.model_copy(update={"cells": rows.freeze()})

    def advance(
        self,
        previous_state: EvaluationMetricReducerStateV1,
        view: EvaluationTransitionViewV1,
    ) -> _State:
        state = cast(_State, previous_state)
        rows = _Rows(state.cells)
        for agent in state.agents:
            rows.add(
                "task.evaluation_return",
                AgentStatisticSubjectV1(global_slot=agent.slot),
                view.transition.canonical_reward_by_agent[agent.slot],
                1,
            )
        rewards = view.transition.canonical_reward_by_team
        if rewards is not None:
            for index, value in enumerate(rewards):
                rows.add("task.evaluation_return", _team(index + 1), value, 1)
        outcome = state.outcome
        for event in view.transition.events:
            if isinstance(event, TeamDeathmatchCompletedEventV1):
                outcome = event.outcome
        return state.model_copy(
            update={
                "cells": rows.freeze(),
                "steps": state.steps + 1,
                "scores": view.successor_frame.snapshot.team_deathmatch_scores,
                "outcome": outcome,
            }
        )

    def finalize(
        self,
        state: EvaluationMetricReducerStateV1,
        completion: EvaluationEpisodeCompletionV1,
        processing_status: EvaluationProcessingStatusV1,
    ) -> tuple[SufficientStatisticDraftV1, ...]:
        current = cast(_State, state)
        rows = _Rows(current.cells)
        for team in (1, 2):
            for outcome in ("win", "draw", "loss"):
                own_win = current.outcome == (
                    "team_a_win" if team == 1 else "team_b_win"
                )
                actual = (
                    "draw"
                    if current.outcome == "draw"
                    else "win"
                    if own_win
                    else "loss"
                )
                rows.add(
                    "task.outcome_distribution",
                    _team(team),
                    float(actual == outcome and current.outcome != "pending"),
                    dimensions=(("outcome", outcome),),
                )
            rows.add(
                "task.terminal_score_differential",
                _team(team),
                current.scores[team - 1] - current.scores[2 - team],
                1,
            )
        rows.add(
            "task.episode_length",
            EpisodeStatisticSubjectV1(),
            current.steps,
            1,
        )
        rows.add(
            "artifact.completion",
            EpisodeStatisticSubjectV1(),
            1,
            dimensions=(
                ("rollout", completion.completion_state),
                ("processing", processing_status.status),
            ),
        )
        return _drafts(current, rows.freeze())


def build_tdm_metric_reducers() -> tuple[EvaluationMetricReducerV1, ...]:
    """Build only the basic V1 reports used by archived sample generation.

    New computation uses ``evaluate(..., metrics="full")`` or the corresponding
    public environment mode. The former ``full`` factory option is retired.
    """
    return cast(tuple[EvaluationMetricReducerV1, ...], (_OutcomeReducer(),))


__all__ = ["TDM_BASIC_METRIC_IDS", "build_tdm_metric_reducers"]
