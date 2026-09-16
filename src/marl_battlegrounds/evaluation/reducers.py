"""Preserve the five basic V1 outcome reports used by archived replay tools.

build_tdm_metric_reducers returns the legacy outcome reducer only. It consumes
validated host replay views and keeps immutable summary state. Current live
metric computation belongs to the public JAX evaluation workflow; this module
does not recreate the retired full-metric reducer or rewrite historical status
wording and report values.
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
    """A legacy metric's count/sum kind, units and complete-episode requirement."""

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
    """One active global slot whose archived per-agent return is accumulated."""

    slot: int


class _Cell(EvaluationModel):
    """One immutable subject/dimension statistic with value and exposure totals."""

    metric: str
    subject: StatisticSubjectV1
    dimensions: _Dimensions = ()
    value: float = 0.0
    exposure: float = 0.0


class _State(EvaluationMetricReducerStateV1):
    """Immutable legacy reducer state: active slots, accumulated cells and observed
    outcome.
    """

    agents: tuple[_Agent, ...]
    cells: tuple[_Cell, ...] = ()
    task_mode: int
    scores: tuple[int, ...] = (0, 0)
    outcome: str = "pending"
    steps: int = 0


def _cell_key(cell: _Cell) -> tuple[str, str, _Dimensions]:
    """Order cells by metric, serialized subject and sorted dimension identity."""
    return cell.metric, cell.subject.model_dump_json(), cell.dimensions


class _Rows:
    """Build replacement legacy statistic rows without editing committed state.

    The mutable dict is local to one initialize/advance/finalize call. freeze
    returns a stable sorted tuple for the next immutable reducer state.
    """

    def __init__(self, cells: tuple[_Cell, ...] = ()) -> None:
        """Index existing immutable cells by their complete metric/subject/dimension
        key.
        """
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
        """Add value and exposure to one local metric cell.

        Sort dimensions before keying. Create a zero cell when needed, but avoid
        replacing an existing cell for a zero update. Committed reducer state is
        unchanged.
        """
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
        """Return all local replacement cells in deterministic key order."""
        return tuple(self.cells[key] for key in sorted(self.cells))


def _team(team: int) -> TeamStatisticSubjectV1:
    """Construct a validated legacy team subject using configured team ID 1 or 2."""
    return TeamStatisticSubjectV1(team_id=team)


def _state(
    context: EvaluationEpisodeContextV1, frame: EvaluationFrameV1, reducer_id: str
) -> _State:
    """Initialize active-slot identities and starting scores from validated V1
    context/frame.
    """
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
    """Package already trusted internal values without repeating model validation.

    External ingestion remains the validation authority. Callers must supply every
    required invariant; this shortcut is not suitable for untrusted wire data.
    """
    return model.model_construct(_fields_set=set(values), **values)


def _drafts(
    state: _State, cells: tuple[_Cell, ...]
) -> tuple[SufficientStatisticDraftV1, ...]:
    """Convert legacy cells to report drafts with original status semantics.

    Preserve undefined outcomes, absent team reward and non-TDM applicability
    reasons exactly. Count/sum components keep their declared units and exposure;
    this is historical report packaging, not current scalar metric calculation.
    """
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
    """V1 outcome/return reducer with fixed identity and no trajectory history."""

    reducer_id: str = "marlbg.episode.outcome"
    reducer_version: int = 1

    def initialize(
        self, context: EvaluationEpisodeContextV1, initial_frame: EvaluationFrameV1
    ) -> _State:
        """Start legacy returns for configured active agents and both teams.

        context and initial_frame are already validated V1 models from one episode.
        Return immutable _State with zero return cells and the authored starting scores.
        """
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
        """Consume one admitted V1 transition and return new immutable totals.

        previous_state must be this reducer's _State. Add recorded canonical agent/team
        rewards, read authoritative completion events, and keep the successor scores.
        The calling reducer pipeline owns ordering and transition admission.
        """
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
        """Package legacy outcome, score, return, length and completion drafts.

        Read immutable state plus rollout completion and processing status. Pending
        outcomes remain unavailable rather than being guessed. This does not change
        state, write files or convert a partial replay into a completed game.
        """
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
    """Return the basic V1 reducer needed by archived report generation.

    Returns
    -------
    tuple[EvaluationMetricReducerV1, ...]
        One-element tuple implementing EvaluationMetricReducerV1 for the five
        historical outcome/return/length/completion statistics.

    Notes
    -----
    Takes no arguments and creates no game or file. The retired full option is
    not accepted. For new measurements use evaluate(..., metrics="full") or
    the public environment's metric mode. Historical report bytes and status
    semantics remain the responsibility of this compatibility producer.
    """
    return cast(tuple[EvaluationMetricReducerV1, ...], (_OutcomeReducer(),))


__all__ = ["TDM_BASIC_METRIC_IDS", "build_tdm_metric_reducers"]
