"""Current-frame match facts for researcher UI, separate from spatial views."""

from __future__ import annotations

import re
from typing import Annotated, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from marl_battlegrounds.evaluation.models import (
    AgentDiedEventV1,
    AssignedPolicySlot,
    AssignedPolicySlotV1,
    AssignedPolicySlotV2,
    ContentAddressedIdentityV1,
    EvaluationEpisodeContext,
    EvaluationEpisodeContextV2,
    EvaluationFrameV1,
    TeamDeathmatchCompletedEventV1,
)
from marl_battlegrounds.rendering.scene import (
    AgentDiedEventV2,
    TeamDeathmatchCompletedEventV2,
)

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


class MatchDeathV1(_MatchModel):
    public_agent_id: _Name
    team_id: Literal[1, 2]
    class_id: Literal[1, 2, 3, 4, 5]


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
    deaths: tuple[MatchDeathV1, ...] = Field(default_factory=tuple, max_length=10)

    @model_validator(mode="after")
    def _validate_task(self) -> Self:
        if tuple(team.team_id for team in self.teams) != (1, 2):
            raise ValueError("match teams must retain Team A then Team B order")
        if len({death.public_agent_id for death in self.deaths}) != len(self.deaths):
            raise ValueError("match deaths must contain each incoming recipient once")
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


def _controller_name(
    context: EvaluationEpisodeContext, row: AssignedPolicySlot, team_id: int
) -> str:
    """Name recognized scenario pressure from its recorded controller identity."""
    scenario = context.identity.scenario
    match = (
        None
        if scenario is None
        else re.fullmatch(
            r"(?:tdm-scenario-|custom-debugger-scenario:scenario_)([1-8])",
            scenario.identifier,
        )
    )
    if match is None and isinstance(context, EvaluationEpisodeContextV2):
        match = re.fullmatch(r"scenario_([1-8])", context.scenario_name or "")
    if match is None:
        names = tuple(
            row.value for row in context.aggregation_keys if row.name == "scenario"
        )
        match = re.fullmatch(r"scenario_([1-8])", names[0]) if len(names) == 1 else None
    variants = {
        "reactive-team-deathmatch-controller": "alpha",
        "scenario-5-pressure-controller": "beta",
    }
    variant = variants.get(row.algorithm_id or "") or variants.get(row.policy_id)
    if variant is None:
        key = f"team_{'a' if team_id == 1 else 'b'}_controller_identity"
        identities = tuple(
            binding.value for binding in context.aggregation_keys if binding.name == key
        )
        if len(identities) == 1:
            try:
                identity = ContentAddressedIdentityV1.model_validate_json(identities[0])
            except ValueError:
                pass
            else:
                variant = variants.get(identity.identifier)
    if team_id == 2 and match is not None and variant is not None:
        return f"tdm-scenario-{match[1]}-controller-{variant}"
    return _CONTROLLER_NAMES.get(row.policy_kind, row.policy_id)


def build_match_summary_v1(
    context: EvaluationEpisodeContext,
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
    deaths: list[MatchDeathV1] = []
    for event in incoming_events:
        if not isinstance(event, (AgentDiedEventV1, AgentDiedEventV2)):
            continue
        if (
            event.transition_id
            != f"{frame.episode_id}:transition:{frame.frame_index - 1}"
        ):
            raise ValueError(
                "match death must belong to the frame's incoming transition"
            )
        roster = context.roster[event.recipient_global_slot]
        deaths.append(
            MatchDeathV1(
                public_agent_id=roster.public_agent_id,
                team_id=cast("Literal[1, 2]", roster.configured_team_id),
                class_id=cast("Literal[1, 2, 3, 4, 5]", roster.class_id),
            )
        )
    teams: list[MatchTeamV1] = []
    for team_id in (1, 2):
        assignments = tuple(
            assignment
            for roster, assignment in zip(
                context.roster, context.policy_assignments, strict=True
            )
            if roster.configured_team_id == team_id
            and isinstance(assignment, (AssignedPolicySlotV1, AssignedPolicySlotV2))
        )
        policy_ids = tuple(dict.fromkeys(row.policy_id for row in assignments))
        names = tuple(
            dict.fromkeys(
                _controller_name(context, row, team_id) for row in assignments
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
        deaths=tuple(deaths),
    )
