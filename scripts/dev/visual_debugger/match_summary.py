"""Current-frame match facts for researcher UI, separate from spatial views."""

from __future__ import annotations

from typing import Annotated, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from marl_battlegrounds.evaluation.models import (
    AssignedPolicySlotV1,
    EvaluationEpisodeContextV1,
    EvaluationFrameV1,
    TeamDeathmatchCompletedEventV1,
)
from marl_battlegrounds.rendering.scene import TeamDeathmatchCompletedEventV2

type MatchOutcome = Literal[
    "in_progress", "team_a_win", "team_b_win", "draw", "not_applicable"
]
type _Count = Annotated[int, Field(ge=0)]
type _Name = Annotated[str, Field(min_length=1)]


class _MatchModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class MatchTeamV1(_MatchModel):
    team_id: Literal[1, 2]
    display_name: _Name
    policy_ids: tuple[_Name, ...]
    checkpoint_digests: tuple[Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")], ...]


class MatchSummaryV1(_MatchModel):
    schema_version: Literal[1]
    episode_id: _Name
    source_frame_index: _Count
    simulator_step_count: _Count
    task_mode: Literal[0, 1]
    score_threshold: _Count
    scores: tuple[_Count, _Count]
    outcome: MatchOutcome
    teams: tuple[MatchTeamV1, MatchTeamV1]

    @model_validator(mode="after")
    def _validate_task(self) -> Self:
        if tuple(team.team_id for team in self.teams) != (1, 2):
            raise ValueError("match teams must retain Team A then Team B order")
        if self.task_mode == 0:
            if self.score_threshold != 0 or self.scores != (0, 0):
                raise ValueError("combat diagnostics cannot claim TDM scores")
            if self.outcome != "not_applicable":
                raise ValueError("combat diagnostics have no TDM result")
        elif self.score_threshold == 0 or self.outcome == "not_applicable":
            raise ValueError("TDM requires its configured threshold and result scope")
        return self


_CONTROLLER_NAMES = {
    "manual": "Manual",
    "scripted": "Scripted scenario",
    "random_valid": "Random (legal actions)",
    "reactive_tdm": "Reactive TDM ALPHA",
    "scenario_5": "Reactive TDM BETA",
}


def build_match_summary_v1(
    context: EvaluationEpisodeContextV1,
    frame: EvaluationFrameV1,
    incoming_events: tuple[object, ...] = (),
) -> MatchSummaryV1:
    """Package recorded scores and task-authored outcomes without new game rules."""
    if frame.episode_id != context.identity.episode_id:
        raise ValueError("match summary frame must join its episode context")
    task_mode = context.resolved_env_config.task_mode
    outcome: MatchOutcome = "in_progress" if task_mode == 1 else "not_applicable"
    completed = tuple(
        event
        for event in incoming_events
        if isinstance(
            event, (TeamDeathmatchCompletedEventV1, TeamDeathmatchCompletedEventV2)
        )
    )
    if len(completed) > 1:
        raise ValueError("match summary cannot have multiple completion events")
    if completed:
        if completed[0].transition_id != (
            f"{frame.episode_id}:transition:{frame.frame_index - 1}"
        ):
            raise ValueError(
                "match result must belong to the frame's incoming transition"
            )
        outcome = completed[0].outcome
    teams: list[MatchTeamV1] = []
    for team_id in (1, 2):
        assignments = tuple(
            assignment
            for roster, assignment in zip(
                context.roster, context.policy_assignments, strict=True
            )
            if roster.configured_team_id == team_id
            and isinstance(assignment, AssignedPolicySlotV1)
        )
        policy_ids = tuple(dict.fromkeys(row.policy_id for row in assignments))
        names = tuple(
            dict.fromkeys(
                _CONTROLLER_NAMES.get(row.policy_kind, row.policy_id)
                for row in assignments
            )
        )
        teams.append(
            MatchTeamV1(
                team_id=team_id,
                display_name=" / ".join(names) if names else "No active policy",
                policy_ids=policy_ids,
                checkpoint_digests=tuple(
                    dict.fromkeys(
                        row.checkpoint_digest
                        for row in assignments
                        if row.checkpoint_digest is not None
                    )
                ),
            )
        )
    return MatchSummaryV1(
        schema_version=1,
        episode_id=frame.episode_id,
        source_frame_index=frame.frame_index,
        simulator_step_count=frame.simulator_step_count,
        task_mode=task_mode,
        score_threshold=context.resolved_env_config.team_deathmatch_score_threshold,
        scores=cast("tuple[int, int]", frame.snapshot.team_deathmatch_scores),
        outcome=outcome,
        teams=(teams[0], teams[1]),
    )
