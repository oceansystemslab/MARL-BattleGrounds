"""Current-frame match facts for researcher UI, separate from spatial views."""

from __future__ import annotations

import re
from collections.abc import Callable
from functools import cache
from typing import TYPE_CHECKING, Annotated, Any, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from marl_battlegrounds.evaluation.map_identity import RecordedMap, recorded_map
from marl_battlegrounds.evaluation.models import (
    AgentDiedEventV1,
    AssignedPolicySlot,
    AssignedPolicySlotV1,
    AssignedPolicySlotV2,
    ContentAddressedIdentityV1,
    EvaluationEpisodeContext,
    EvaluationEpisodeContextV2,
    EvaluationFrameV1,
    LethalDamageContributionEventV1,
    RecipientHealthResolutionEventV1,
    SourceHealingOutputEventV1,
    TeamDeathmatchCompletedEventV1,
)
from marl_battlegrounds.rendering.scene import (
    AgentDiedEventV2,
    LethalDamageContributionEventV2,
    RecipientHealthResolutionEventV2,
    SourceHealingOutputEventV2,
    TeamDeathmatchCompletedEventV2,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from marl_battlegrounds.evaluation.combat_metrics import CombatCredit

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


class MatchAgentV1(_MatchModel):
    public_agent_id: _Name
    team_id: Literal[1, 2]
    class_id: Literal[1, 2, 3, 4, 5]


class MatchDeathV1(MatchAgentV1):
    killing_team_id: Literal[1, 2] | None = Field(default_factory=lambda: None)
    contributors: tuple[MatchAgentV1, ...] | None = Field(
        default_factory=lambda: None, min_length=1, max_length=5
    )

    @model_validator(mode="after")
    def _validate_credit(self) -> Self:
        if (self.killing_team_id is None) != (self.contributors is None):
            raise ValueError(
                "death attribution must be complete or explicitly unavailable"
            )
        if self.contributors is not None:
            if self.killing_team_id == self.team_id or any(
                row.team_id != self.killing_team_id for row in self.contributors
            ):
                raise ValueError(
                    "kill contributors must belong to the opposing killing team"
                )
            if len({row.public_agent_id for row in self.contributors}) != len(
                self.contributors
            ):
                raise ValueError("kill contributors must be unique for each victim")
        return self


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
    map: RecordedMap | None = Field(default_factory=lambda: None)
    observation_mode: Literal["shared_obs", "no_shared_obs"] | None = Field(
        default_factory=lambda: None
    )
    episode_limit: Annotated[int, Field(gt=0)] | None = Field(
        default_factory=lambda: None
    )
    root_seed: Annotated[int, Field(ge=0, le=2**32 - 1)] | None = Field(
        default_factory=lambda: None
    )
    episode_seed: Annotated[int, Field(ge=0, le=2**32 - 1)] | None = Field(
        default_factory=lambda: None
    )

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


@cache
def _combat_credit() -> Callable[..., CombatCredit]:
    """Load the shared calculation only when a death needs attribution."""
    import jax

    from marl_battlegrounds.evaluation.combat_metrics import combat_credit

    return cast("Callable[..., CombatCredit]", jax.jit(combat_credit))


def _death_contributors(
    context: EvaluationEpisodeContext,
    frame: EvaluationFrameV1,
    incoming_events: tuple[object, ...],
    dead_slots: set[int],
) -> dict[int, tuple[MatchAgentV1, ...]]:
    """Decode only observed credit inputs; sparse historical evidence stays unknown.

    Full global events already carry transition-start health and delivered effects.
    The tiny shared calculation runs on CPU, without reconstructing simulator
    state, computing full metrics or evaluating hypothetical rescue actions.
    """
    # Canonical global bundles start at zero and contain every event in order.
    # A sparse historical subset cannot establish complete support attribution.
    if tuple(getattr(event, "ordinal", None) for event in incoming_events) != tuple(
        range(len(incoming_events))
    ):
        return {}
    import numpy as np

    routing = np.zeros((10, 10), dtype=np.float32)
    healing = np.zeros_like(routing)
    contributed = np.zeros(10, dtype=np.bool_)
    health_before = np.zeros(10, dtype=np.float32)
    health_after = np.zeros(10, dtype=np.float32)
    total_healing = np.zeros(10, dtype=np.float32)
    total_damage = np.zeros(10, dtype=np.float32)
    attributed_damage = np.zeros(10, dtype=np.float32)
    healing_sources: set[int] = set()
    health_recipients: set[int] = set()
    for event in incoming_events:
        if not isinstance(
            event,
            (
                LethalDamageContributionEventV1,
                LethalDamageContributionEventV2,
                SourceHealingOutputEventV1,
                SourceHealingOutputEventV2,
                RecipientHealthResolutionEventV1,
                RecipientHealthResolutionEventV2,
            ),
        ):
            continue
        if (
            event.transition_id
            != f"{frame.episode_id}:transition:{frame.frame_index - 1}"
        ):
            raise ValueError(
                "kill credit must belong to the frame's incoming transition"
            )
        recipient = event.recipient_global_slot
        if recipient is None:
            raise ValueError("kill credit effects require their recorded recipient")
        if isinstance(
            event, (RecipientHealthResolutionEventV1, RecipientHealthResolutionEventV2)
        ):
            if recipient in health_recipients:
                raise ValueError(
                    "kill credit cannot repeat a recipient health resolution"
                )
            health_recipients.add(recipient)
            health_before[recipient] = event.transition_start_health
            health_after[recipient] = event.health_after_combat_resolution
            total_healing[recipient] = event.total_effective_healing
            total_damage[recipient] = event.total_effective_damage
            continue
        source = event.source_global_slot
        roster = context.roster[source]
        if not roster.configured_active:
            raise ValueError("kill credit effects require an active recorded source")
        routing[source, recipient] = 1
        if isinstance(event, (SourceHealingOutputEventV1, SourceHealingOutputEventV2)):
            if source in healing_sources:
                raise ValueError("kill credit cannot repeat a source healing output")
            healing_sources.add(source)
            healing[source, recipient] = np.float32(
                event.source_modified_healing_output
            ) * np.float32(event.recipient_healing_modifier)
        else:
            if contributed[source] or recipient not in dead_slots:
                raise ValueError(
                    "direct contribution must uniquely join an incoming death"
                )
            if (
                roster.configured_team_id
                == context.roster[recipient].configured_team_id
            ):
                raise ValueError("direct contribution must belong to the opposing team")
            if event.attributed_death_damage <= 0:
                raise ValueError(
                    "direct contribution requires positive recorded damage"
                )
            contributed[source] = True
            attributed_damage[recipient] += np.float32(event.attributed_death_damage)
    if not contributed.any():
        return {}
    # Missing source or recipient records can hide useful Priest support. Never
    # present an apparently complete contributor list from partial evidence.
    recorded_healing = healing.sum(axis=0)
    affected = set(np.flatnonzero(recorded_healing > 0)) | dead_slots
    dead = sorted(dead_slots)
    if (
        not affected.issubset(health_recipients)
        or not np.allclose(recorded_healing, total_healing, rtol=1e-6, atol=1e-5)
        # A removed final contribution can leave a contiguous ordinal prefix.
        # Every delivered damage amount on a lethal recipient earns direct credit.
        or not np.allclose(
            attributed_damage[dead], total_damage[dead], rtol=1e-6, atol=1e-5
        )
    ):
        return {}
    inputs = (
        np.asarray([row.class_id for row in context.roster], dtype=np.int32),
        health_before,
        healing,
        routing,
        total_healing,
        total_damage,
        health_after,
        contributed,
    )
    # Explicit CPU placement avoids touching the researcher's training GPU.
    import jax

    cpu = cast(Any, jax.devices("cpu")[0])
    credit = _combat_credit()(*jax.device_put(inputs, cpu))
    contributions = cast("NDArray[np.bool_]", np.asarray(credit.kill_contributions))
    result: dict[int, tuple[MatchAgentV1, ...]] = {}
    for recipient in sorted(dead_slots):
        rows = tuple(
            MatchAgentV1(
                public_agent_id=context.roster[source].public_agent_id,
                team_id=cast(
                    "Literal[1, 2]", context.roster[source].configured_team_id
                ),
                class_id=cast(
                    "Literal[1, 2, 3, 4, 5]", context.roster[source].class_id
                ),
            )
            for source in np.flatnonzero(contributions[:, recipient])
        )
        if rows:
            result[recipient] = rows
    return result


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
    death_events = tuple(
        event
        for event in incoming_events
        if isinstance(event, (AgentDiedEventV1, AgentDiedEventV2))
    )
    contributors = (
        _death_contributors(
            context,
            frame,
            incoming_events,
            {event.recipient_global_slot for event in death_events},
        )
        if death_events
        else {}
    )
    for event in death_events:
        if (
            event.transition_id
            != f"{frame.episode_id}:transition:{frame.frame_index - 1}"
        ):
            raise ValueError(
                "match death must belong to the frame's incoming transition"
            )
        roster = context.roster[event.recipient_global_slot]
        credit = contributors.get(event.recipient_global_slot)
        deaths.append(
            MatchDeathV1(
                public_agent_id=roster.public_agent_id,
                team_id=cast("Literal[1, 2]", roster.configured_team_id),
                class_id=cast("Literal[1, 2, 3, 4, 5]", roster.class_id),
                killing_team_id=None if credit is None else credit[0].team_id,
                contributors=credit,
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
        map=recorded_map(context),
        observation_mode=context.execution_information_mode,
        episode_limit=context.expected_horizon,
        root_seed=context.seed_protocol.root_seed,
        episode_seed=context.seed_protocol.episode_seed,
    )
