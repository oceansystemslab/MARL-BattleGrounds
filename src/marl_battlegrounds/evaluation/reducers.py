"""Episode metrics derived solely from the validated evaluation trajectory.

These three explicit reducer families implement the episode portion of the metric
specification. Mutable dictionaries are short-lived local builders; committed
state and published sufficient components remain immutable. Nothing calls Core.
"""

from dataclasses import dataclass
from math import hypot
from typing import Literal, cast

import numpy as np

from marl_battlegrounds.evaluation.metrics import (
    AgentPairStatisticSubjectV1,
    AgentStatisticSubjectV1,
    CountComponentV1,
    DistributionComponentV1,
    DistributionObservationV1,
    DurationComponentV1,
    EpisodeStatisticSubjectV1,
    EvaluationEpisodeCompletionV1,
    EvaluationMetricReducerStateV1,
    EvaluationMetricReducerV1,
    EvaluationProcessingStatusV1,
    EvaluationTransitionViewV1,
    HealthAmountStage,
    RatioComponentV1,
    StatisticDimensionV1,
    StatisticSubjectV1,
    SufficientStatisticComponentV1,
    SufficientStatisticDraftV1,
    SumComponentV1,
    TeamClassStatisticSubjectV1,
    TeamStatisticSubjectV1,
)
from marl_battlegrounds.evaluation.models import (
    EvaluationEpisodeContextV1,
    EvaluationFrameV1,
    EvaluationModel,
    GlobalAnalysisSnapshotV1,
    StatusAppliedEventV1,
    TeamDeathmatchCompletedEventV1,
)

type _Kind = Literal["count", "sum", "ratio", "duration", "distribution"]
type _Dimensions = tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class _Definition:
    kind: _Kind
    units: str
    stage: HealthAmountStage | None = None
    owner_class: int | None = None
    complete_only: bool = False


# A bounded episode suite, not a dynamic metric/plugin registry.
_DEFINITIONS: dict[str, _Definition] = {
    "task.outcome_distribution": _Definition("count", "games", complete_only=True),
    "task.terminal_score_differential": _Definition(
        "sum", "points", complete_only=True
    ),
    "task.evaluation_return": _Definition("sum", "reward", complete_only=True),
    "task.episode_length": _Definition("sum", "transitions", complete_only=True),
    "artifact.completion": _Definition("count", "episodes"),
    "combat.recipient_modified_gross_damage_output": _Definition(
        "sum", "HP", "recipient_modified_gross"
    ),
    "combat.recipient_modified_gross_damage_received": _Definition(
        "sum", "HP", "recipient_modified_gross"
    ),
    "support.recipient_modified_gross_healing_output": _Definition(
        "sum", "HP", "recipient_modified_gross"
    ),
    "combat.realized_net_health_change": _Definition(
        "sum", "HP", "realized_net_health_change"
    ),
    "combat.upper_health_clamp_overflow": _Definition(
        "sum", "HP", "combat_resolution_health"
    ),
    "combat.death_count": _Definition("count", "deaths"),
    "combat.lethal_transition_damage_contribution": _Definition(
        "count", "contributions"
    ),
    "combat.lethal_transition_contribution_rate": _Definition(
        "ratio", "contributions_per_enemy_death"
    ),
    "coordination.single_contributor_lethal_transition_count": _Definition(
        "count", "enemy_deaths"
    ),
    "coordination.multi_contributor_lethal_transition_rate": _Definition(
        "ratio", "fraction"
    ),
    "coordination.focus_fire_concentration": _Definition("ratio", "fraction"),
    "ability.activation_count": _Definition("count", "activations"),
    "ability.ultimate_cooldown_start_count": _Definition("count", "cooldown_starts"),
    "ability.ultimate_ready_transition_count": _Definition("count", "ready_edges"),
    "movement.phase_displacement": _Definition("distribution", "world_units"),
    "status.application_count": _Definition("count", "applications"),
    "status.active_recipient_steps": _Definition("duration", "recipient_steps"),
    "status.lifecycle_cause_count": _Definition("count", "lifecycle_edges"),
    "control.damage_to_controlled_recipient": _Definition(
        "sum", "HP", "recipient_modified_gross"
    ),
    "control.enemy_death_while_controlled": _Definition("count", "enemy_deaths"),
    "mage.burst_window_damage": _Definition("sum", "HP", "recipient_modified_gross", 1),
    "mage.burst_enemy_death_association": _Definition(
        "count", "contributions", owner_class=1
    ),
    "mage.aura_coverage": _Definition(
        "duration", "emitter_beneficiary_steps", owner_class=1
    ),
    "mage.combined_aura_amplification": _Definition(
        "sum", "HP", "recipient_modified_gross", 1
    ),
    "warrior.aura_coverage": _Definition(
        "duration", "emitter_beneficiary_steps", owner_class=2
    ),
    "warrior.combined_aura_mitigation": _Definition(
        "sum", "HP", "recipient_modified_gross", 2
    ),
    "hunter.trap_active_steps": _Definition(
        "duration", "recipient_steps", owner_class=3
    ),
    "hunter.trap_status_episode_end": _Definition(
        "count", "status_episodes", owner_class=3
    ),
    "hunter.trap_damage_break_rate": _Definition("ratio", "fraction", owner_class=3),
    "rogue.combined_anti_heal_reduction": _Definition(
        "sum", "HP", "recipient_modified_gross", 4
    ),
    "rogue.priority_target_damage_share": _Definition(
        "ratio", "fraction", "recipient_modified_gross", 4
    ),
    "priest.same_transition_lethal_damage_rescue": _Definition(
        "count", "rescues", owner_class=5
    ),
    "priest.freedom_binding_coverage": _Definition(
        "duration", "recipient_steps", owner_class=5
    ),
    "recovery.realized_regeneration": _Definition("sum", "HP", "actual_regeneration"),
    "recovery.combat_countdown_reset_count": _Definition("count", "countdown_resets"),
    "lifecycle.respawn_wave": _Definition("count", "occurrences"),
    "lifecycle.dead_agent_steps": _Definition("duration", "agent_steps"),
    "lifecycle.spawn_shield_expiry": _Definition("count", "expiries"),
    "diagnostic.action_acceptance_rate": _Definition("ratio", "fraction"),
    "diagnostic.action_rejection": _Definition("count", "rejections"),
    "formation.ally_distance_distribution": _Definition("distribution", "world_units"),
}

TDM_EPISODE_METRIC_IDS = tuple(sorted(f"marlbg.{key}.v1" for key in _DEFINITIONS))
TDM_BASIC_METRIC_IDS = tuple(
    metric_id
    for metric_id in TDM_EPISODE_METRIC_IDS
    if metric_id.startswith(("marlbg.task.", "marlbg.artifact."))
)


class _Agent(EvaluationModel):
    slot: int
    team: int
    class_id: int
    maximum_health: float


class _Cell(EvaluationModel):
    metric: str
    subject: StatisticSubjectV1
    dimensions: _Dimensions = ()
    component_name: str = "value"
    kind: _Kind
    units: str | None = None
    amount_stage: HealthAmountStage | None = None
    value: float = 0.0
    exposure: float = 0.0
    observations: tuple[DistributionObservationV1, ...] = ()


class _State(EvaluationMetricReducerStateV1):
    agents: tuple[_Agent, ...]
    cells: tuple[_Cell, ...] = ()
    task_mode: int
    scores: tuple[int, ...] = (0, 0)
    outcome: str = "pending"
    steps: int = 0
    trap_starts: tuple[int | None, ...] = (None,) * 10
    initial_traps: tuple[bool, ...] = (False,) * 10
    life_starts: tuple[int, ...] = (0,) * 10
    initial_lives: tuple[bool, ...] = (True,) * 10
    alive: tuple[bool, ...] = (False,) * 10


def _cell_key(cell: _Cell) -> tuple[str, str, _Dimensions, str]:
    return (
        cell.metric,
        cell.subject.model_dump_json(),
        cell.dimensions,
        cell.component_name,
    )


class _Rows:
    """A private, transition-local builder; never retained by a reducer."""

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
        component: str = "value",
        kind: _Kind | None = None,
        units: str | None = None,
        amount_stage: HealthAmountStage | None = None,
        observation_id: str | None = None,
    ) -> None:
        dimensions = tuple(sorted(dimensions))
        key = (
            metric,
            subject.model_dump_json(),
            dimensions,
            component,
        )
        previous = self.cells.get(key)
        if previous is None:
            previous = _Cell(
                metric=metric,
                subject=subject,
                dimensions=dimensions,
                component_name=component,
                kind=_DEFINITIONS[metric].kind if kind is None else kind,
                units=units,
                amount_stage=amount_stage,
            )
        elif value == 0 and exposure == 0 and observation_id is None:
            return
        observations = previous.observations
        if observation_id is not None:
            observations = (
                *observations,
                DistributionObservationV1(
                    source_observation_id=observation_id,
                    ordinal=len(observations),
                    value=float(value),
                ),
            )
        self.cells[key] = previous.model_copy(
            update={
                "value": previous.value + float(value),
                "exposure": previous.exposure + float(exposure),
                "observations": observations,
            }
        )

    def freeze(self) -> tuple[_Cell, ...]:
        return tuple(self.cells[key] for key in sorted(self.cells))


def _team(team: int) -> TeamStatisticSubjectV1:
    return TeamStatisticSubjectV1(team_id=team)


def _class(team: int, class_id: int) -> TeamClassStatisticSubjectV1:
    return TeamClassStatisticSubjectV1(team_id=team, class_id=class_id)


def _subjects(agent: _Agent) -> tuple[StatisticSubjectV1, ...]:
    return (
        AgentStatisticSubjectV1(global_slot=agent.slot),
        _class(agent.team, agent.class_id),
        _team(agent.team),
    )


def _agent_add(
    rows: _Rows,
    metric: str,
    agent: _Agent,
    value: float = 0.0,
    exposure: float = 0.0,
    *,
    dimensions: _Dimensions = (),
    component: str = "value",
    kind: _Kind | None = None,
    units: str | None = None,
    amount_stage: HealthAmountStage | None = None,
    observation_id: str | None = None,
) -> None:
    for subject in _subjects(agent):
        rows.add(
            metric,
            subject,
            value,
            exposure,
            dimensions=dimensions,
            component=component,
            kind=kind,
            units=units,
            amount_stage=amount_stage,
            observation_id=observation_id,
        )


def _combined(
    rows: _Rows,
    metric: str,
    team: int,
    value: float = 0.0,
    exposure: float = 0.0,
    *,
    dimensions: _Dimensions = (),
    component: str = "value",
    kind: _Kind | None = None,
    units: str | None = None,
    amount_stage: HealthAmountStage | None = None,
) -> None:
    owner = _DEFINITIONS[metric].owner_class
    if owner is None:
        raise ValueError("combined class effect requires its declared class owner")
    for subject in (_class(team, owner), _team(team)):
        rows.add(
            metric,
            subject,
            value,
            exposure,
            dimensions=dimensions,
            component=component,
            kind=kind,
            units=units,
            amount_stage=amount_stage,
        )


def _state(
    context: EvaluationEpisodeContextV1, frame: EvaluationFrameV1, reducer_id: str
) -> _State:
    return _State(
        reducer_id=reducer_id,
        reducer_version=1,
        task_mode=context.resolved_env_config.task_mode,
        agents=tuple(
            _Agent(
                slot=row.global_slot,
                team=row.configured_team_id,
                class_id=row.class_id,
                maximum_health=context.resolved_env_config.slot_mechanics[
                    row.global_slot
                ].maximum_health,
            )
            for row in context.roster
            if row.configured_active
        ),
        scores=frame.snapshot.team_deathmatch_scores,
        alive=frame.snapshot.alive_mask,
        trap_starts=tuple(
            0 if row[1] > 0 else None for row in frame.snapshot.stun_durations
        ),
        initial_traps=tuple(row[1] > 0 for row in frame.snapshot.stun_durations),
    )


def _model[T: EvaluationModel](model: type[T], **values: object) -> T:
    """Package trusted metric values; external ingestion validates wire records."""
    return model.model_construct(_fields_set=set(values), **values)


def _component(cell: _Cell) -> SufficientStatisticComponentV1:
    if cell.kind == "count":
        return _model(CountComponentV1, count=int(cell.value), eligible_episode_count=1)
    if cell.kind == "sum":
        return _model(
            SumComponentV1,
            value=cell.value,
            observation_count=int(cell.exposure),
            eligible_episode_count=1,
        )
    if cell.kind == "ratio":
        return _model(
            RatioComponentV1,
            numerator=cell.value,
            denominator=cell.exposure,
            zero_opportunity_occurrence=int(cell.exposure == 0),
            eligible_episode_count=1,
        )
    if cell.kind == "duration":
        return _model(
            DurationComponentV1,
            qualifying_steps=int(cell.value),
            eligible_steps=int(cell.exposure),
            eligible_episode_count=1,
        )
    return _model(
        DistributionComponentV1,
        observations=cell.observations,
        eligible_episode_count=1,
    )


def _drafts(
    state: _State, cells: tuple[_Cell, ...]
) -> tuple[SufficientStatisticDraftV1, ...]:
    result: list[SufficientStatisticDraftV1] = []
    for cell in cells:
        definition = _DEFINITIONS[cell.metric]
        absent = isinstance(cell.subject, TeamClassStatisticSubjectV1) and not any(
            agent.team == cell.subject.team_id
            and agent.class_id == cell.subject.class_id
            for agent in state.agents
        )
        cannot_heal = (
            cell.metric == "support.recipient_modified_gross_healing_output"
            and (
                (
                    isinstance(cell.subject, TeamClassStatisticSubjectV1)
                    and cell.subject.class_id != 5
                )
                or (
                    isinstance(cell.subject, AgentStatisticSubjectV1)
                    and next(
                        agent.class_id
                        for agent in state.agents
                        if agent.slot == cell.subject.global_slot
                    )
                    != 5
                )
            )
        )
        inapplicable = (
            absent
            or cannot_heal
            or (
                state.task_mode != 1
                and cell.metric
                in ("task.outcome_distribution", "task.terminal_score_differential")
            )
        )
        missing_definition = cell.metric == "rogue.priority_target_damage_share"
        status: Literal[
            "defined",
            "zero_opportunity",
            "structurally_inapplicable",
            "ambiguous_attribution",
            "insufficient_data",
        ] = "defined"
        reason: str | None = None
        component: SufficientStatisticComponentV1 | None = _component(cell)
        if inapplicable:
            status, reason, component = (
                "structurally_inapplicable",
                "class absent or task has no TDM outcome",
                None,
            )
        elif missing_definition:
            status, reason, component = (
                "insufficient_data",
                "no task-declared priority target in episode context",
                None,
            )
        elif ("attribution", "ambiguous_healers") in cell.dimensions:
            status, reason, component = (
                "ambiguous_attribution",
                "multiple positive healers; individual rescue credit is undefined",
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
        elif cell.kind in ("ratio", "duration", "distribution") and cell.exposure == 0:
            status, reason = (
                "zero_opportunity",
                "no genuine opportunities in captured prefix",
            )
        result.append(
            _model(
                SufficientStatisticDraftV1,
                metric_id=f"marlbg.{cell.metric}.v1",
                metric_version=1,
                component_name=cell.component_name,
                reducer_id=state.reducer_id,
                reducer_version=1,
                units=(
                    cell.units
                    if cell.units is not None
                    else "fraction"
                    if cell.component_name == "team_share"
                    else "occurrences"
                    if cell.kind == "count" and definition.kind != "count"
                    else "transitions"
                    if cell.component_name in ("observed_duration", "countdown")
                    else definition.units
                ),
                amount_stage=(cell.amount_stage or definition.stage)
                if cell.kind in ("sum", "ratio")
                else None,
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


def _statuses(snapshot: GlobalAnalysisSnapshotV1, slot: int) -> tuple[int, ...]:
    return (
        *snapshot.slow_durations[slot],
        *snapshot.stun_durations[slot],
        snapshot.rogue_poison_anti_heal_durations[slot],
        snapshot.mage_burst_damage_amplification_durations[slot],
        snapshot.priest_blessing_of_freedom_slow_floor_durations[slot],
    )


def _gross(amount: float, modifier: float) -> float:
    return float(np.float32(amount) * np.float32(modifier))


_EFFECT_METRICS = (
    "combat.recipient_modified_gross_damage_output",
    "combat.recipient_modified_gross_damage_received",
    "support.recipient_modified_gross_healing_output",
    "combat.realized_net_health_change",
    "combat.upper_health_clamp_overflow",
    "combat.death_count",
    "combat.lethal_transition_damage_contribution",
    "combat.lethal_transition_contribution_rate",
)


@dataclass(frozen=True, slots=True)
class _EffectsReducer:
    reducer_id: str = "marlbg.episode.effects"
    reducer_version: int = 1

    def initialize(
        self, context: EvaluationEpisodeContextV1, initial_frame: EvaluationFrameV1
    ) -> _State:
        state = _state(context, initial_frame, self.reducer_id)
        rows = _Rows()
        for agent in state.agents:
            for metric in _EFFECT_METRICS:
                _agent_add(rows, metric, agent)
            for status in context.static_mechanics_catalog.status_channels:
                if status.family in ("slow", "stun", "anti_heal"):
                    _agent_add(
                        rows,
                        "control.damage_to_controlled_recipient",
                        agent,
                        dimensions=(("status", status.status_id),),
                    )
        for team in (1, 2):
            for status in context.static_mechanics_catalog.status_channels:
                if status.family in ("slow", "stun", "anti_heal"):
                    rows.add(
                        "control.enemy_death_while_controlled",
                        _team(team),
                        dimensions=(("status", status.status_id),),
                    )
            for metric in (
                "coordination.single_contributor_lethal_transition_count",
                "coordination.multi_contributor_lethal_transition_rate",
                "coordination.focus_fire_concentration",
            ):
                rows.add(metric, _team(team))
            for metric in (
                "mage.burst_window_damage",
                "mage.burst_enemy_death_association",
                "mage.combined_aura_amplification",
                "warrior.combined_aura_mitigation",
                "rogue.combined_anti_heal_reduction",
                "rogue.priority_target_damage_share",
                "priest.same_transition_lethal_damage_rescue",
            ):
                _combined(rows, metric, team)
        return state.model_copy(update={"cells": rows.freeze()})

    def advance(
        self,
        previous_state: EvaluationMetricReducerStateV1,
        view: EvaluationTransitionViewV1,
    ) -> _State:
        state = cast(_State, previous_state)
        rows = _Rows(state.cells)
        agents = {agent.slot: agent for agent in state.agents}
        facts = view.transition.facts
        combat = facts.combat_transition_facts
        start = view.start_frame.snapshot
        catalog = view.context.static_mechanics_catalog
        damage = {
            slot: _gross(
                combat.source_modified_damage_output_by_source[slot],
                combat.recipient_damage_modifier_by_source[slot],
            )
            for slot in agents
        }
        healing = {
            slot: _gross(
                combat.source_modified_healing_output_by_source[slot],
                combat.recipient_healing_modifier_by_source[slot],
            )
            for slot in agents
        }
        enemy_deaths = {
            team: sum(
                facts.death_facts.is_newly_dead_by_recipient[a.slot]
                for a in state.agents
                if a.team != team
            )
            for team in (1, 2)
        }
        for agent in state.agents:
            slot = agent.slot
            recipient = combat.combat_effect_recipient_global_slot_by_source[slot]
            _agent_add(
                rows,
                "combat.recipient_modified_gross_damage_output",
                agent,
                damage[slot],
                1,
            )
            _agent_add(
                rows,
                "support.recipient_modified_gross_healing_output",
                agent,
                healing[slot],
                1,
            )
            _agent_add(
                rows,
                "combat.recipient_modified_gross_damage_received",
                agent,
                combat.total_effective_damage_by_recipient[slot],
                int(start.alive_mask[slot]),
            )
            if start.alive_mask[slot]:
                _agent_add(
                    rows,
                    "combat.realized_net_health_change",
                    agent,
                    combat.health_after_combat_resolution_by_recipient[slot]
                    - start.current_health[slot],
                    1,
                )
                for component, amount in (
                    ("gross_damage", combat.total_effective_damage_by_recipient[slot]),
                    (
                        "gross_healing",
                        combat.total_effective_healing_by_recipient[slot],
                    ),
                ):
                    _agent_add(
                        rows,
                        "combat.realized_net_health_change",
                        agent,
                        amount,
                        1,
                        component=component,
                        amount_stage="recipient_modified_gross",
                    )
                overflow = max(
                    0.0,
                    float(
                        np.float32(start.current_health[slot])
                        + (
                            np.float32(
                                combat.total_effective_healing_by_recipient[slot]
                            )
                            - np.float32(
                                combat.total_effective_damage_by_recipient[slot]
                            )
                        )
                    )
                    - agent.maximum_health,
                )
                _agent_add(
                    rows, "combat.upper_health_clamp_overflow", agent, overflow, 1
                )
                _agent_add(
                    rows,
                    "combat.upper_health_clamp_overflow",
                    agent,
                    combat.total_effective_healing_by_recipient[slot],
                    1,
                    component="gross_healing_exposure",
                    amount_stage="recipient_modified_gross",
                )
            _agent_add(
                rows,
                "combat.death_count",
                agent,
                int(facts.death_facts.is_newly_dead_by_recipient[slot]),
                1,
            )
            contribution = int(
                facts.death_facts.contributed_to_new_death_by_source[slot]
            )
            _agent_add(
                rows,
                "combat.lethal_transition_damage_contribution",
                agent,
                contribution,
                1,
            )
            if contribution and recipient is not None:
                _agent_add(
                    rows,
                    "combat.lethal_transition_damage_contribution",
                    agent,
                    contribution,
                    dimensions=(("recipient_slot", str(recipient)),),
                )
            # Agent denominators are shared team exposure; team/class denominators
            # are added once below rather than once per contributor.
            rows.add(
                "combat.lethal_transition_contribution_rate",
                AgentStatisticSubjectV1(global_slot=slot),
                contribution,
                enemy_deaths[agent.team],
            )
            if recipient is not None and recipient in agents:
                dims = (
                    (
                        "ability",
                        "ultimate"
                        if combat.ultimate_effect_is_activated_by_source[slot]
                        else "basic",
                    ),
                    ("target_class", str(agents[recipient].class_id)),
                    ("recipient_slot", str(recipient)),
                )
                _agent_add(
                    rows,
                    "combat.recipient_modified_gross_damage_output",
                    agent,
                    damage[slot],
                    1,
                    dimensions=dims,
                )
                _agent_add(
                    rows,
                    "support.recipient_modified_gross_healing_output",
                    agent,
                    healing[slot],
                    1,
                    dimensions=dims,
                )
                for channel, duration in enumerate(_statuses(start, recipient)):
                    if catalog.status_channels[channel].family not in (
                        "slow",
                        "stun",
                        "anti_heal",
                    ):
                        continue
                    status_dims = (
                        ("status", catalog.status_channels[channel].status_id),
                    )
                    _agent_add(
                        rows,
                        "control.damage_to_controlled_recipient",
                        agent,
                        damage[slot] if duration > 0 else 0,
                        int(duration > 0),
                        dimensions=status_dims,
                    )
            if agent.class_id == 1:
                active = start.mage_burst_damage_amplification_durations[slot] > 0
                _agent_add(
                    rows,
                    "mage.burst_window_damage",
                    agent,
                    damage[slot] if active else 0,
                    int(active),
                )
                _agent_add(
                    rows,
                    "mage.burst_window_damage",
                    agent,
                    int(
                        combat.mage_burst_damage_amplification_is_applied_by_source[
                            slot
                        ]
                    ),
                    component="activations",
                    kind="count",
                )
                _agent_add(
                    rows,
                    "mage.burst_enemy_death_association",
                    agent,
                    contribution if active else 0,
                )
                _agent_add(
                    rows,
                    "mage.burst_enemy_death_association",
                    agent,
                    int(
                        combat.mage_burst_damage_amplification_is_applied_by_source[
                            slot
                        ]
                    ),
                    component="activations",
                    units="activations",
                )
                rows.add(
                    "mage.burst_enemy_death_association",
                    AgentStatisticSubjectV1(global_slot=slot),
                    enemy_deaths[agent.team],
                    component="team_enemy_deaths",
                    units="enemy_deaths",
                )
            if damage[slot] > 0:
                burst_magnitude = catalog.status_channels[7].magnitude
                if burst_magnitude is None:
                    raise ValueError("Burst catalog must declare its damage multiplier")
                # Keep Burst and recipient mitigation in the comparison. Subtract
                # at the same gross amount stage; dividing a rounded source amount
                # by a reconstructed aura multiplier would invent rounding credit.
                without_aura = _gross(
                    _gross(
                        combat.raw_damage_output_by_source[slot],
                        burst_magnitude
                        if start.mage_burst_damage_amplification_durations[slot] > 0
                        else 1.0,
                    ),
                    combat.recipient_damage_modifier_by_source[slot],
                )
                _combined(
                    rows,
                    "mage.combined_aura_amplification",
                    agent.team,
                    damage[slot] - without_aura,
                    1,
                )
            if recipient is not None and recipient in agents:
                _combined(
                    rows,
                    "warrior.combined_aura_mitigation",
                    agents[recipient].team,
                    combat.source_modified_damage_output_by_source[slot] - damage[slot],
                    int(damage[slot] > 0),
                )
                _combined(
                    rows,
                    "rogue.combined_anti_heal_reduction",
                    3 - agents[recipient].team,
                    combat.source_modified_healing_output_by_source[slot]
                    - healing[slot],
                    int(healing[slot] > 0),
                )
                if start.rogue_poison_anti_heal_durations[recipient] > 0:
                    for component, amount, stage in (
                        (
                            "source_modified_healing_exposure",
                            combat.source_modified_healing_output_by_source[slot],
                            "source_modified_gross",
                        ),
                        (
                            "recipient_modified_healing_exposure",
                            healing[slot],
                            "recipient_modified_gross",
                        ),
                    ):
                        _combined(
                            rows,
                            "rogue.combined_anti_heal_reduction",
                            3 - agents[recipient].team,
                            amount,
                            int(amount > 0),
                            component=component,
                            amount_stage=cast(HealthAmountStage, stage),
                        )
            _combined(
                rows,
                "rogue.combined_anti_heal_reduction",
                3 - agent.team,
                int(
                    start.alive_mask[slot]
                    and start.rogue_poison_anti_heal_durations[slot] > 0
                ),
                int(start.alive_mask[slot]),
                component="active_recipient_steps",
                kind="duration",
                units="recipient_steps",
            )
        for team in (1, 2):
            _combined(
                rows,
                "mage.burst_enemy_death_association",
                team,
                enemy_deaths[team],
                component="team_enemy_deaths",
                units="enemy_deaths",
            )
            team_agents = tuple(a for a in state.agents if a.team == team)
            for subject in (
                _team(team),
                *(_class(team, class_id) for class_id in range(1, 6)),
            ):
                relevant = tuple(
                    a
                    for a in team_agents
                    if not isinstance(subject, TeamClassStatisticSubjectV1)
                    or a.class_id == subject.class_id
                )
                rows.add(
                    "combat.lethal_transition_contribution_rate",
                    subject,
                    sum(
                        facts.death_facts.contributed_to_new_death_by_source[a.slot]
                        for a in relevant
                    ),
                    enemy_deaths[team],
                )
            target_counts: dict[int, int] = {}
            for agent in team_agents:
                recipient = combat.combat_effect_recipient_global_slot_by_source[
                    agent.slot
                ]
                if (
                    damage[agent.slot] > 0
                    and recipient is not None
                    and agents[recipient].team != team
                ):
                    target_counts[recipient] = target_counts.get(recipient, 0) + 1
            attackers = sum(target_counts.values())
            rows.add(
                "coordination.focus_fire_concentration",
                _team(team),
                max(target_counts.values(), default=0) / attackers
                if attackers >= 2
                else 0,
                int(attackers >= 2),
            )
            singles = multiples = 0
            for victim in state.agents:
                if (
                    victim.team == team
                    or not facts.death_facts.is_newly_dead_by_recipient[victim.slot]
                ):
                    continue
                count = target_counts.get(victim.slot, 0)
                singles += count == 1
                multiples += count >= 2
                for channel, duration in enumerate(_statuses(start, victim.slot)):
                    if catalog.status_channels[channel].family in (
                        "slow",
                        "stun",
                        "anti_heal",
                    ):
                        rows.add(
                            "control.enemy_death_while_controlled",
                            _team(team),
                            int(duration > 0),
                            dimensions=(
                                ("status", catalog.status_channels[channel].status_id),
                            ),
                        )
            rows.add(
                "coordination.single_contributor_lethal_transition_count",
                _team(team),
                singles,
            )
            rows.add(
                "coordination.multi_contributor_lethal_transition_rate",
                _team(team),
                multiples,
                enemy_deaths[team],
            )
        for recipient in state.agents:
            slot = recipient.slot
            healers = tuple(
                a
                for a in state.agents
                if healing[a.slot] > 0
                and combat.combat_effect_recipient_global_slot_by_source[a.slot] == slot
            )
            rescued = (
                start.alive_mask[slot]
                and combat.total_effective_damage_by_recipient[slot]
                >= start.current_health[slot]
                and combat.total_effective_healing_by_recipient[slot] > 0
                and combat.health_after_combat_resolution_by_recipient[slot] > 0
            )
            opportunity = (
                start.alive_mask[slot]
                and combat.total_effective_damage_by_recipient[slot]
                >= start.current_health[slot]
            )
            _combined(
                rows,
                "priest.same_transition_lethal_damage_rescue",
                recipient.team,
                int(rescued),
            )
            _combined(
                rows,
                "priest.same_transition_lethal_damage_rescue",
                recipient.team,
                int(rescued),
                dimensions=(("healer_count", str(len(healers))),),
            )
            for dimensions in ((), (("healer_count", str(len(healers))),)):
                _combined(
                    rows,
                    "priest.same_transition_lethal_damage_rescue",
                    recipient.team,
                    int(opportunity),
                    component="recipient_opportunities",
                    dimensions=dimensions,
                    units="recipient_transitions",
                )
            if opportunity and len(healers) == 1:
                rows.add(
                    "priest.same_transition_lethal_damage_rescue",
                    AgentStatisticSubjectV1(global_slot=healers[0].slot),
                    int(rescued),
                    dimensions=(("attribution", "unique_healer"),),
                )
                rows.add(
                    "priest.same_transition_lethal_damage_rescue",
                    AgentStatisticSubjectV1(global_slot=healers[0].slot),
                    1,
                    component="recipient_opportunities",
                    dimensions=(("attribution", "unique_healer"),),
                    units="recipient_transitions",
                )
            elif rescued:
                for healer in healers:
                    rows.add(
                        "priest.same_transition_lethal_damage_rescue",
                        AgentStatisticSubjectV1(global_slot=healer.slot),
                        dimensions=(("attribution", "ambiguous_healers"),),
                    )
        return state.model_copy(
            update={"cells": rows.freeze(), "steps": state.steps + 1}
        )

    def finalize(
        self,
        state: EvaluationMetricReducerStateV1,
        completion: EvaluationEpisodeCompletionV1,
        processing_status: EvaluationProcessingStatusV1,
    ) -> tuple[SufficientStatisticDraftV1, ...]:
        del completion, processing_status
        current = cast(_State, state)
        rows = _Rows(current.cells)
        share_metrics = {
            "combat.recipient_modified_gross_damage_output",
            "combat.recipient_modified_gross_damage_received",
            "support.recipient_modified_gross_healing_output",
            "combat.death_count",
        }
        for cell in current.cells:
            if (
                cell.metric not in share_metrics
                or cell.dimensions
                or cell.component_name != "value"
                or not isinstance(
                    cell.subject, (AgentStatisticSubjectV1, TeamClassStatisticSubjectV1)
                )
            ):
                continue
            team = (
                cell.subject.team_id
                if isinstance(cell.subject, TeamClassStatisticSubjectV1)
                else next(
                    a.team for a in current.agents if a.slot == cell.subject.global_slot
                )
            )
            total = next(
                row.value
                for row in current.cells
                if row.metric == cell.metric
                and row.subject == _team(team)
                and not row.dimensions
                and row.component_name == "value"
            )
            rows.add(
                cell.metric,
                cell.subject,
                cell.value,
                total,
                component="team_share",
                kind="ratio",
            )
        return _drafts(current, rows.freeze())


_DYNAMIC_METRICS = (
    "ability.ultimate_cooldown_start_count",
    "ability.ultimate_ready_transition_count",
    "recovery.realized_regeneration",
    "recovery.combat_countdown_reset_count",
    "lifecycle.dead_agent_steps",
    "lifecycle.spawn_shield_expiry",
    "diagnostic.action_acceptance_rate",
)


def _close_trap(
    rows: _Rows,
    agent: _Agent,
    start_index: int,
    end_index: int,
    causes: tuple[str, ...],
    *,
    initial: bool,
) -> None:
    dimensions = (
        ("recipient_slot", str(agent.slot)),
        ("causes", "+".join(causes)),
        ("entry", "initial_active" if initial else "recorded_application"),
    )
    _combined(rows, "hunter.trap_status_episode_end", 3 - agent.team, 1)
    _combined(
        rows, "hunter.trap_status_episode_end", 3 - agent.team, 1, dimensions=dimensions
    )
    _combined(
        rows,
        "hunter.trap_damage_break_rate",
        3 - agent.team,
        int("damage_break" in causes),
        1,
    )
    # The interval identity is stable across repeated prefix/final projections.
    for subject in (_class(3 - agent.team, 3), _team(3 - agent.team)):
        rows.add(
            "hunter.trap_status_episode_end",
            subject,
            max(0, end_index - start_index),
            1,
            dimensions=dimensions,
            component="observed_duration",
            kind="distribution",
            observation_id=f"trap:{agent.slot}:{start_index}:{end_index}:{'+'.join(causes)}",
        )


def _close_life(
    rows: _Rows,
    agent: _Agent,
    start: int,
    end: int,
    *,
    alive: bool,
    initial: bool,
) -> None:
    phase = "alive" if alive else "dead"
    _agent_add(
        rows,
        "lifecycle.dead_agent_steps",
        agent,
        end - start,
        1,
        component="observed_duration",
        kind="distribution",
        dimensions=(
            ("life_phase", phase),
            ("entry", "initial_boundary" if initial else "recorded_boundary"),
            ("ending", "death" if alive else "respawn"),
        ),
        observation_id=f"life:{agent.slot}:{phase}:{start}:{end}",
    )


@dataclass(frozen=True, slots=True)
class _DynamicsReducer:
    reducer_id: str = "marlbg.episode.dynamics"
    reducer_version: int = 1

    def initialize(
        self, context: EvaluationEpisodeContextV1, initial_frame: EvaluationFrameV1
    ) -> _State:
        state = _state(context, initial_frame, self.reducer_id)
        rows = _Rows()
        for agent in state.agents:
            for metric in _DYNAMIC_METRICS:
                _agent_add(rows, metric, agent)
            for ability in ("basic", "ultimate"):
                _agent_add(
                    rows,
                    "ability.activation_count",
                    agent,
                    dimensions=(("ability", ability),),
                )
            for component in ("domain", "movement", "combat_pair"):
                _agent_add(
                    rows,
                    "diagnostic.action_rejection",
                    agent,
                    dimensions=(("rejection", component),),
                )
            for status in context.static_mechanics_catalog.status_channels:
                dimensions = (("status", status.status_id),)
                _agent_add(
                    rows, "status.active_recipient_steps", agent, dimensions=dimensions
                )
                _agent_add(
                    rows, "status.application_count", agent, dimensions=dimensions
                )
                _agent_add(
                    rows, "status.lifecycle_cause_count", agent, dimensions=dimensions
                )
            for phase in ("charge", "ordinary"):
                for axis in ("distance", "x", "y"):
                    _agent_add(
                        rows,
                        "movement.phase_displacement",
                        agent,
                        dimensions=(("phase", phase), ("axis", axis)),
                    )
        for team in (1, 2):
            rows.add(
                "lifecycle.respawn_wave",
                _team(team),
                dimensions=(("event", "team_wave"),),
            )
            rows.add("formation.ally_distance_distribution", _team(team))
            for metric in (
                "mage.aura_coverage",
                "warrior.aura_coverage",
                "hunter.trap_active_steps",
                "hunter.trap_status_episode_end",
                "hunter.trap_damage_break_rate",
                "priest.freedom_binding_coverage",
            ):
                _combined(rows, metric, team)
        return state.model_copy(update={"cells": rows.freeze()})

    def advance(
        self,
        previous_state: EvaluationMetricReducerStateV1,
        view: EvaluationTransitionViewV1,
    ) -> _State:
        state = cast(_State, previous_state)
        rows = _Rows(state.cells)
        facts = view.transition.facts
        start = view.start_frame.snapshot
        successor = view.successor_frame.snapshot
        catalog = view.context.static_mechanics_catalog
        combat = facts.combat_transition_facts
        acceptance = facts.action_acceptance_facts
        trap_starts = list(state.trap_starts)
        initial_traps = list(state.initial_traps)
        life_starts = list(state.life_starts)
        initial_lives = list(state.initial_lives)
        applications = tuple(
            event
            for event in view.transition.events
            if isinstance(event, StatusAppliedEventV1)
        )
        for agent in state.agents:
            slot = agent.slot
            alive = start.alive_mask[slot]
            for ability, activated in (
                ("basic", combat.basic_effect_is_activated_by_source[slot]),
                ("ultimate", combat.ultimate_effect_is_activated_by_source[slot]),
            ):
                _agent_add(
                    rows,
                    "ability.activation_count",
                    agent,
                    int(activated),
                    1,
                    dimensions=(("ability", ability),),
                )
            _agent_add(
                rows,
                "ability.ultimate_cooldown_start_count",
                agent,
                acceptance.accepted_joint_action.use_ultimate[slot],
                1,
            )
            _agent_add(
                rows,
                "ability.ultimate_ready_transition_count",
                agent,
                int(
                    start.ultimate_cooldowns[slot] > 0
                    and successor.ultimate_cooldowns[slot] == 0
                ),
                1,
            )
            for phase, displacements in (
                ("charge", facts.physical_facts.charge_phase_displacement_by_agent),
                (
                    "ordinary",
                    facts.physical_facts.ordinary_movement_phase_displacement_by_agent,
                ),
            ):
                if alive:
                    x, y = displacements[slot]
                    for axis, value in (("distance", hypot(x, y)), ("x", x), ("y", y)):
                        _agent_add(
                            rows,
                            "movement.phase_displacement",
                            agent,
                            value,
                            1,
                            dimensions=(("phase", phase), ("axis", axis)),
                            observation_id=f"{view.transition.transition_id}:movement:{slot}:{phase}:{axis}",
                        )
            statuses = _statuses(start, slot)
            for channel, duration in enumerate(statuses):
                dimensions = (("status", catalog.status_channels[channel].status_id),)
                _agent_add(
                    rows,
                    "status.active_recipient_steps",
                    agent,
                    int(alive and duration > 0),
                    int(alive),
                    dimensions=dimensions,
                )
                for cause, matrix in (
                    (
                        "age",
                        facts.status_lifecycle_facts.aged_to_zero_by_recipient_and_status_channel,
                    ),
                    (
                        "refresh",
                        facts.status_lifecycle_facts.refreshed_or_extended_by_recipient_and_status_channel,
                    ),
                    (
                        "damage_break",
                        facts.status_lifecycle_facts.broken_by_damage_by_recipient_and_status_channel,
                    ),
                    (
                        "death_clear",
                        facts.status_lifecycle_facts.cleared_by_new_death_by_recipient_and_status_channel,
                    ),
                ):
                    _agent_add(
                        rows,
                        "status.lifecycle_cause_count",
                        agent,
                        int(matrix[slot][channel]),
                        1,
                        dimensions=(*dimensions, ("cause", cause)),
                    )
                    if matrix[slot][channel]:
                        _agent_add(
                            rows,
                            "status.lifecycle_cause_count",
                            agent,
                            1,
                            dimensions=dimensions,
                        )
            for event in applications:
                if event.source_global_slot == slot:
                    _agent_add(
                        rows,
                        "status.application_count",
                        agent,
                        1,
                        dimensions=(("status", event.status_id),),
                    )
                    _agent_add(
                        rows,
                        "status.application_count",
                        agent,
                        1,
                        dimensions=(
                            ("status", event.status_id),
                            ("recipient_slot", str(event.recipient_global_slot)),
                        ),
                    )
            _agent_add(
                rows,
                "recovery.realized_regeneration",
                agent,
                facts.regeneration_facts.actual_health_regenerated_this_step_by_agent[
                    slot
                ],
                int(alive),
            )
            _agent_add(
                rows,
                "recovery.combat_countdown_reset_count",
                agent,
                int(facts.regeneration_facts.combat_countdown_was_reset_by_agent[slot]),
                1,
            )
            if facts.regeneration_facts.combat_countdown_was_reset_by_agent[slot]:
                for boundary, countdown in (
                    ("start", start.steps_until_out_of_combat[slot]),
                    ("successor", successor.steps_until_out_of_combat[slot]),
                ):
                    _agent_add(
                        rows,
                        "recovery.combat_countdown_reset_count",
                        agent,
                        countdown,
                        1,
                        component="countdown",
                        kind="distribution",
                        dimensions=(("boundary", boundary),),
                        observation_id=f"{view.transition.transition_id}:combat-countdown:{slot}:{boundary}",
                    )
            _agent_add(rows, "lifecycle.dead_agent_steps", agent, int(not alive), 1)
            for boundary, occurred in (
                ("new_death", facts.death_facts.is_newly_dead_by_recipient[slot]),
                (
                    "respawn",
                    facts.respawn_facts.was_respawned_this_transition_by_agent[slot],
                ),
            ):
                _agent_add(
                    rows,
                    "lifecycle.dead_agent_steps",
                    agent,
                    int(occurred),
                    component=boundary,
                    kind="count",
                )
            if facts.death_facts.is_newly_dead_by_recipient[slot]:
                _close_life(
                    rows,
                    agent,
                    life_starts[slot],
                    state.steps + 1,
                    alive=True,
                    initial=initial_lives[slot],
                )
                life_starts[slot], initial_lives[slot] = state.steps + 1, False
            if facts.respawn_facts.was_respawned_this_transition_by_agent[slot]:
                _close_life(
                    rows,
                    agent,
                    life_starts[slot],
                    state.steps + 1,
                    alive=False,
                    initial=initial_lives[slot],
                )
                life_starts[slot], initial_lives[slot] = state.steps + 1, False
            _agent_add(
                rows,
                "lifecycle.respawn_wave",
                agent,
                int(facts.respawn_facts.was_respawned_this_transition_by_agent[slot]),
                dimensions=(("event", "agent_respawn"),),
            )
            _agent_add(
                rows,
                "lifecycle.spawn_shield_expiry",
                agent,
                int(facts.spawn_shield_facts.expired_at_transition_end_by_agent[slot]),
                1,
            )
            _agent_add(
                rows,
                "lifecycle.spawn_shield_expiry",
                agent,
                int(
                    facts.spawn_shield_facts.was_active_at_transition_start_by_agent[
                        slot
                    ]
                ),
                component="active_at_start",
                kind="count",
            )
            rejected = (
                acceptance.submitted_action_tuple_is_out_of_domain_by_actor[slot],
                acceptance.in_domain_move_action_is_rejected_by_actor[slot],
                acceptance.in_domain_combat_action_pair_is_rejected_by_actor[slot],
            )
            _agent_add(
                rows,
                "diagnostic.action_acceptance_rate",
                agent,
                int(not any(rejected)),
                1,
            )
            for name, rejection in zip(
                ("domain", "movement", "combat_pair"), rejected, strict=True
            ):
                _agent_add(
                    rows,
                    "diagnostic.action_rejection",
                    agent,
                    int(rejection),
                    1,
                    dimensions=(("rejection", name),),
                )
                _agent_add(
                    rows,
                    "diagnostic.action_rejection",
                    agent,
                    1,
                    dimensions=(("rejection", name),),
                    component="opportunities",
                    kind="count",
                )
            _combined(
                rows,
                "hunter.trap_active_steps",
                3 - agent.team,
                int(alive and statuses[4] > 0),
                int(alive),
            )
            _combined(
                rows,
                "hunter.trap_active_steps",
                3 - agent.team,
                sum(
                    event.recipient_global_slot == slot and event.status_channel == 4
                    for event in applications
                ),
                component="applications",
                kind="count",
                units="applications",
            )
            causes = tuple(
                name
                for name, occurred in (
                    (
                        "age",
                        facts.status_lifecycle_facts.aged_to_zero_by_recipient_and_status_channel[
                            slot
                        ][4],
                    ),
                    (
                        "damage_break",
                        facts.status_lifecycle_facts.broken_by_damage_by_recipient_and_status_channel[
                            slot
                        ][4],
                    ),
                )
                if occurred
            )
            beginning = trap_starts[slot]
            if beginning is not None and causes:
                _close_trap(
                    rows,
                    agent,
                    beginning,
                    state.steps + 1,
                    causes,
                    initial=initial_traps[slot],
                )
                trap_starts[slot], initial_traps[slot] = None, False
            if trap_starts[slot] is None and any(
                event.recipient_global_slot == slot and event.status_channel == 4
                for event in applications
            ):
                trap_starts[slot] = state.steps + 1
            beginning = trap_starts[slot]
            lifecycle = facts.status_lifecycle_facts
            death_clears = (
                lifecycle.cleared_by_new_death_by_recipient_and_status_channel
            )
            if beginning is not None and death_clears[slot][4]:
                _close_trap(
                    rows,
                    agent,
                    beginning,
                    state.steps + 1,
                    ("death_clear",),
                    initial=initial_traps[slot],
                )
                trap_starts[slot], initial_traps[slot] = None, False
            freedom_eligible = (
                alive
                and start.spawn_shield_durations[slot] == 0
                and not any(start.stun_durations[slot])
                and statuses[8] > 0
            )
            multipliers = tuple(
                np.float32(catalog.status_channels[channel].magnitude)
                if statuses[channel] > 0
                else np.float32(1)
                for channel in range(3)
            )
            slow = float(
                np.prod(np.asarray(multipliers, dtype=np.float32), dtype=np.float32)
            )
            freedom_floor = catalog.status_channels[8].magnitude
            if freedom_floor is None:
                raise ValueError("Freedom catalog must declare its movement floor")
            binding = (
                freedom_eligible
                and max(slow, catalog.global_slow_floor) < freedom_floor
            )
            _combined(
                rows,
                "priest.freedom_binding_coverage",
                agent.team,
                int(binding),
                int(freedom_eligible),
            )
            rows.add(
                "priest.freedom_binding_coverage",
                AgentStatisticSubjectV1(global_slot=slot),
                int(binding),
                int(freedom_eligible),
                dimensions=(("role", "recipient"),),
            )
            if agent.class_id in (1, 2):
                metric = (
                    "mage.aura_coverage"
                    if agent.class_id == 1
                    else "warrior.aura_coverage"
                )
                auras = facts.aura_facts
                matrices = (
                    auras.is_covered_by_mage_damage_aura_by_emitter_and_beneficiary,
                    auras.is_covered_by_warrior_mitigation_aura_by_emitter_and_beneficiary,
                )
                matrix = matrices[agent.class_id - 1]
                for beneficiary in state.agents:
                    if beneficiary.team != agent.team:
                        continue
                    eligible = (
                        alive
                        and start.alive_mask[beneficiary.slot]
                        and start.spawn_shield_durations[slot] == 0
                        and start.spawn_shield_durations[beneficiary.slot] == 0
                    )
                    covered = matrix[slot][beneficiary.slot]
                    _agent_add(rows, metric, agent, int(covered), int(eligible))
                    if beneficiary.slot != slot:
                        rows.add(
                            metric,
                            AgentPairStatisticSubjectV1(
                                primary_global_slot=slot,
                                secondary_global_slot=beneficiary.slot,
                            ),
                            int(covered),
                            int(eligible),
                        )
        for team in (1, 2):
            rows.add(
                "lifecycle.respawn_wave",
                _team(team),
                int(
                    facts.respawn_facts.respawn_wave_occurred_this_transition_by_team[
                        team - 1
                    ]
                ),
                dimensions=(("event", "team_wave"),),
            )
            if facts.respawn_facts.respawn_wave_occurred_this_transition_by_team[
                team - 1
            ]:
                for boundary, countdown in (
                    ("start", start.team_respawn_wave_countdowns[team - 1]),
                    ("successor", successor.team_respawn_wave_countdowns[team - 1]),
                ):
                    rows.add(
                        "lifecycle.respawn_wave",
                        _team(team),
                        countdown,
                        1,
                        component="countdown",
                        kind="distribution",
                        dimensions=(("boundary", boundary),),
                        observation_id=f"{view.transition.transition_id}:wave-countdown:{team}:{boundary}",
                    )
            living = tuple(
                agent
                for agent in state.agents
                if agent.team == team and start.alive_mask[agent.slot]
            )
            for index, first in enumerate(living):
                for second in living[index + 1 :]:
                    distance = hypot(
                        *(
                            a - b
                            for a, b in zip(
                                start.agent_positions[first.slot],
                                start.agent_positions[second.slot],
                                strict=True,
                            )
                        )
                    )
                    identity = (
                        f"{view.start_frame.frame_id}:ally-distance:"
                        f"{first.slot}:{second.slot}"
                    )
                    rows.add(
                        "formation.ally_distance_distribution",
                        _team(team),
                        distance,
                        1,
                        observation_id=identity,
                    )
                    rows.add(
                        "formation.ally_distance_distribution",
                        AgentPairStatisticSubjectV1(
                            primary_global_slot=first.slot,
                            secondary_global_slot=second.slot,
                        ),
                        distance,
                        1,
                        observation_id=identity,
                    )
        return state.model_copy(
            update={
                "cells": rows.freeze(),
                "steps": state.steps + 1,
                "trap_starts": tuple(trap_starts),
                "initial_traps": tuple(initial_traps),
                "life_starts": tuple(life_starts),
                "initial_lives": tuple(initial_lives),
                "alive": successor.alive_mask,
            }
        )

    def finalize(
        self,
        state: EvaluationMetricReducerStateV1,
        completion: EvaluationEpisodeCompletionV1,
        processing_status: EvaluationProcessingStatusV1,
    ) -> tuple[SufficientStatisticDraftV1, ...]:
        del processing_status
        current = cast(_State, state)
        rows = _Rows(current.cells)
        boundary = (
            "horizon_censored"
            if "declared_horizon" in completion.completion_bases
            else "task_terminal_competing"
            if completion.completion_state == "complete"
            else "open_prefix"
        )
        for agent in current.agents:
            life_dimensions = (
                ("boundary", boundary),
                ("life_phase", "alive" if current.alive[agent.slot] else "dead"),
                (
                    "entry",
                    "initial_boundary"
                    if current.initial_lives[agent.slot]
                    else "recorded_boundary",
                ),
            )
            _agent_add(
                rows,
                "lifecycle.dead_agent_steps",
                agent,
                1,
                dimensions=life_dimensions,
                component="open_intervals",
                kind="count",
            )
            _agent_add(
                rows,
                "lifecycle.dead_agent_steps",
                agent,
                current.steps - current.life_starts[agent.slot],
                1,
                dimensions=life_dimensions,
                component="open_interval_elapsed",
                kind="sum",
            )
            if current.trap_starts[agent.slot] is not None:
                _combined(
                    rows,
                    "hunter.trap_status_episode_end",
                    3 - agent.team,
                    1,
                    dimensions=(
                        ("boundary", boundary),
                        ("recipient_slot", str(agent.slot)),
                    ),
                    component="open_intervals",
                    kind="count",
                )
        return _drafts(current, rows.freeze())


def build_tdm_metric_reducers(
    *, full: bool = False
) -> tuple[EvaluationMetricReducerV1, ...]:
    """Return critical episode metrics, or all 46 metrics when explicitly requested.

    Neutral diagnostic captures remain supported with explicit inapplicable TDM
    endpoints. Official benchmark eligibility is enforced by the evaluation
    protocol, separately from the ability to describe a generic recorded episode.
    """
    if type(full) is not bool:
        raise TypeError("full must be a bool")
    # The protocol permits writable identity attributes; these implementations
    # strengthen that contract by freezing their constant identity instead.
    return cast(
        tuple[EvaluationMetricReducerV1, ...],
        (_OutcomeReducer(), _EffectsReducer(), _DynamicsReducer())
        if full
        else (_OutcomeReducer(),),
    )


__all__ = [
    "TDM_BASIC_METRIC_IDS",
    "TDM_EPISODE_METRIC_IDS",
    "build_tdm_metric_reducers",
]
