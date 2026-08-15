"""Opt-in CP3 evaluation streaming and sufficient-statistic tests."""

from __future__ import annotations

import ast
import inspect
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import jax
import pytest
import tests.evaluation_fixtures as evaluation_fixtures_module
from pydantic import ConfigDict, PrivateAttr, ValidationError
from tests.evaluation_fixtures import (
    CapturedEvaluationTrajectory,
    captured_evaluation_trajectory,
    mage_target_none_ultimate_action,
)

import marl_battlegrounds.core.env as core_env_module
import marl_battlegrounds.evaluation as evaluation_package
import marl_battlegrounds.evaluation.capture as capture_module
import marl_battlegrounds.evaluation.metrics as metrics_module
import marl_battlegrounds.evaluation.validation as validation_module
from marl_battlegrounds.evaluation.metrics import (
    AgentPairStatisticSubjectV1,
    AgentStatisticSubjectV1,
    CountComponentV1,
    DistributionComponentV1,
    DistributionObservationV1,
    DurationComponentV1,
    EpisodeStatisticSubjectV1,
    EvaluationEpisodeCompletionV1,
    EvaluationEpisodeObserverV1,
    EvaluationMetricReducerStateV1,
    EvaluationMetricReportV1,
    EvaluationProcessingFailureV1,
    EvaluationProcessingStatusV1,
    EvaluationTransitionViewV1,
    OpportunityComponentV1,
    RatioComponentV1,
    RawSufficientStatisticV1,
    StatisticDimensionV1,
    SufficientStatisticAccumulatorV1,
    SufficientStatisticDraftV1,
    SumComponentV1,
    TeamClassStatisticSubjectV1,
    TeamStatisticSubjectV1,
    build_evaluation_observer_v1,
)
from marl_battlegrounds.evaluation.models import (
    AggregationKeyV1,
    CaptureProfile,
    EvaluationEpisodeContextV1,
    EvaluationEpisodeIdentityV1,
    EvaluationFrameV1,
    EvaluationModel,
    EvaluationTransitionV1,
    GlobalAnalysisSnapshotV1,
    TransitionFactsV1,
)
from marl_battlegrounds.evaluation.validation import (
    validate_initial_evaluation_frame_v1,
)


def _draft(
    component: (
        CountComponentV1
        | SumComponentV1
        | RatioComponentV1
        | DurationComponentV1
        | OpportunityComponentV1
        | DistributionComponentV1
        | None
    ),
    **updates: object,
) -> SufficientStatisticDraftV1:
    payload: dict[str, object] = {
        "metric_id": "marlbg.test.raw_component.v1",
        "metric_version": 1,
        "component_name": "raw_component",
        "reducer_id": "test_reducer",
        "reducer_version": 1,
        "units": "count",
        "amount_stage": None,
        "subject": EpisodeStatisticSubjectV1(),
        "dimensions": (),
        "completion_scope": "any_gap_free_prefix",
        "supports_right_censoring": False,
        "result_status": "defined",
        "status_reason": None,
        "endpoint_observation_status": "not_applicable",
        "component": component,
    }
    payload.update(updates)
    return SufficientStatisticDraftV1.model_validate(payload)


def _component_cases() -> tuple[EvaluationModel, ...]:
    return (
        CountComponentV1(count=2, eligible_episode_count=1),
        SumComponentV1(
            value=-2.5,
            observation_count=3,
            eligible_episode_count=1,
        ),
        RatioComponentV1(
            numerator=2.0,
            denominator=4.0,
            zero_opportunity_occurrence=0,
            eligible_episode_count=1,
        ),
        DurationComponentV1(
            qualifying_steps=2,
            eligible_steps=5,
            eligible_episode_count=1,
        ),
        OpportunityComponentV1(
            opportunity_count=7,
            eligible_episode_count=1,
        ),
        DistributionComponentV1(
            observations=(
                DistributionObservationV1(
                    source_observation_id="event-1",
                    ordinal=0,
                    value=-1.25,
                ),
                DistributionObservationV1(
                    source_observation_id="event-2",
                    ordinal=1,
                    value=3.5,
                ),
            ),
            eligible_episode_count=1,
        ),
    )


class _ReducerState(EvaluationMetricReducerStateV1):
    """Immutable test reducer state exposing exactly processed semantic views."""

    initial_frame_id: str
    transition_ids: tuple[str, ...] = ()
    start_frame_ids: tuple[str, ...] = ()
    successor_frame_ids: tuple[str, ...] = ()
    canonical_reward_sum: float = 0.0


class _AlternateReducerState(EvaluationMetricReducerStateV1):
    """Deliberately incompatible replacement type for negative tests."""

    marker: int = 0


class _MutablePayloadReducerState(EvaluationMetricReducerStateV1):
    """Malicious frozen model whose nested payload is still mutable."""

    values: list[int]


class _PrivateMutableReducerState(EvaluationMetricReducerStateV1):
    """Malicious state hiding mutable data from dumps and equality."""

    _values: list[int] = PrivateAttr(default_factory=lambda: [1])


class _UnfrozenReducerState(EvaluationMetricReducerStateV1):
    """Malicious state weakening the inherited replacement-state contract."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=False,
        strict=True,
    )


class _UnfrozenCountComponent(CountComponentV1):
    """Malicious nested record weakening its inherited frozen contract."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=False,
        strict=True,
    )


class _PrivateCountComponent(CountComponentV1):
    """Malicious nested record hiding mutable data outside public fields."""

    _values: list[int] = PrivateAttr(default_factory=lambda: [1])


class _UnfrozenEvaluationFrame(EvaluationFrameV1):
    """Malicious public-root subtype weakening frame immutability."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=False,
        strict=True,
    )


class _PrivateEvaluationFrame(EvaluationFrameV1):
    """Malicious public-root subtype hiding mutable frame state."""

    _values: list[int] = PrivateAttr(default_factory=lambda: [1])


class _UnfrozenEvaluationTransition(EvaluationTransitionV1):
    """Malicious public-root subtype weakening transition immutability."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=False,
        strict=True,
    )


class _PrivateEvaluationTransition(EvaluationTransitionV1):
    """Malicious public-root subtype hiding mutable transition state."""

    _values: list[int] = PrivateAttr(default_factory=lambda: [1])


class _AdversarialEvaluationContext(EvaluationEpisodeContextV1):
    """Mutable context subtype that lies about equality with the public root."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=False,
        strict=True,
    )
    _values: list[int] = PrivateAttr(default_factory=lambda: [1])

    def __eq__(self, other: object) -> bool:
        del other
        return True

    def __ne__(self, other: object) -> bool:
        del other
        return False


class _AdversarialEpisodeIdentity(EvaluationEpisodeIdentityV1):
    """Mutable nested context identity that lies about equality."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=False,
        strict=True,
    )
    _values: list[int] = PrivateAttr(default_factory=lambda: [1])

    def __eq__(self, other: object) -> bool:
        del other
        return True

    def __ne__(self, other: object) -> bool:
        del other
        return False


class _AdversarialSnapshot(GlobalAnalysisSnapshotV1):
    """Mutable nested frame snapshot that lies about equality."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=False,
        strict=True,
    )
    _values: list[int] = PrivateAttr(default_factory=lambda: [1])

    def __eq__(self, other: object) -> bool:
        del other
        return True

    def __ne__(self, other: object) -> bool:
        del other
        return False


class _AdversarialTransitionFacts(TransitionFactsV1):
    """Mutable nested transition facts that lie about equality."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=False,
        strict=True,
    )
    _values: list[int] = PrivateAttr(default_factory=lambda: [1])

    def __eq__(self, other: object) -> bool:
        del other
        return True

    def __ne__(self, other: object) -> bool:
        del other
        return False


class _UnprintableError(RuntimeError):
    """Hostile reducer exception whose string conversion also raises."""

    def __str__(self) -> str:
        raise RuntimeError("intentional stringification failure")


class _UnreadableTransition:
    """Malformed append input that raises during attempted-index inspection."""

    def __getattribute__(self, name: str) -> object:
        raise RuntimeError(f"cannot inspect {name}")


type _DraftBuilder = Callable[
    [
        _ReducerState,
        EvaluationEpisodeCompletionV1,
        EvaluationProcessingStatusV1,
    ],
    tuple[SufficientStatisticDraftV1, ...],
]


@dataclass(slots=True)
class _Reducer:
    """Trusted pure replacement-state reducer used only to prove CP3 plumbing."""

    reducer_id: str = "test.reducer"
    reducer_version: int = 1
    draft_builder: _DraftBuilder | None = None
    fail_initialize: bool = False
    fail_advance_index: int | None = None
    advance_error: Exception | None = None
    fail_finalize: bool = False
    return_mutable_finalize: bool = False
    mutable_payload_on_initialize: bool = False
    private_payload_on_initialize: bool = False
    unfrozen_state_on_initialize: bool = False
    wrong_identity_on_advance: bool = False
    wrong_type_on_advance: bool = False

    def initialize(
        self,
        context: EvaluationEpisodeContextV1,
        initial_frame: EvaluationFrameV1,
    ) -> EvaluationMetricReducerStateV1:
        del context
        if self.fail_initialize:
            raise RuntimeError("intentional initialize failure")
        if self.mutable_payload_on_initialize:
            return _MutablePayloadReducerState(
                reducer_id=self.reducer_id,
                reducer_version=self.reducer_version,
                values=[1],
            )
        if self.private_payload_on_initialize:
            return _PrivateMutableReducerState(
                reducer_id=self.reducer_id,
                reducer_version=self.reducer_version,
            )
        if self.unfrozen_state_on_initialize:
            return _UnfrozenReducerState(
                reducer_id=self.reducer_id,
                reducer_version=self.reducer_version,
            )
        return _ReducerState(
            reducer_id=self.reducer_id,
            reducer_version=self.reducer_version,
            initial_frame_id=initial_frame.frame_id,
        )

    def advance(
        self,
        previous_state: EvaluationMetricReducerStateV1,
        view: EvaluationTransitionViewV1,
    ) -> EvaluationMetricReducerStateV1:
        if not isinstance(previous_state, _ReducerState):
            raise TypeError("test reducer received an unexpected state type")
        if self.advance_error is not None:
            raise self.advance_error
        if view.transition.transition_index == self.fail_advance_index:
            raise RuntimeError("intentional advance failure")
        if self.wrong_type_on_advance:
            return _AlternateReducerState(
                reducer_id=self.reducer_id,
                reducer_version=self.reducer_version,
            )
        reducer_id = (
            "wrong.reducer" if self.wrong_identity_on_advance else self.reducer_id
        )
        return _ReducerState(
            reducer_id=reducer_id,
            reducer_version=self.reducer_version,
            initial_frame_id=previous_state.initial_frame_id,
            transition_ids=(
                *previous_state.transition_ids,
                view.transition.transition_id,
            ),
            start_frame_ids=(
                *previous_state.start_frame_ids,
                view.start_frame.frame_id,
            ),
            successor_frame_ids=(
                *previous_state.successor_frame_ids,
                view.successor_frame.frame_id,
            ),
            canonical_reward_sum=(
                previous_state.canonical_reward_sum
                + sum(view.transition.canonical_reward_by_agent)
            ),
        )

    def finalize(
        self,
        state: EvaluationMetricReducerStateV1,
        completion: EvaluationEpisodeCompletionV1,
        processing_status: EvaluationProcessingStatusV1,
    ) -> tuple[SufficientStatisticDraftV1, ...]:
        if self.fail_finalize:
            raise RuntimeError("intentional finalize failure")
        if not isinstance(state, _ReducerState):
            raise TypeError("test reducer finalized an unexpected state type")
        drafts = (
            self.draft_builder(state, completion, processing_status)
            if self.draft_builder is not None
            else (
                _draft(
                    CountComponentV1(
                        count=len(state.transition_ids),
                        eligible_episode_count=1,
                    ),
                    component_name=self.reducer_id,
                    reducer_id=self.reducer_id,
                    reducer_version=self.reducer_version,
                ),
            )
        )
        if self.return_mutable_finalize:
            return cast(tuple[SufficientStatisticDraftV1, ...], list(drafts))
        return drafts


def _feed_observer(
    observer: EvaluationEpisodeObserverV1,
    frames: tuple[EvaluationFrameV1, ...],
    transitions: tuple[EvaluationTransitionV1, ...],
) -> None:
    observer.start(frames[0])
    for transition, successor_frame in zip(
        transitions,
        frames[1:],
        strict=True,
    ):
        observer.append(transition, successor_frame)


def _six_family_drafts(
    state: _ReducerState,
    completion: EvaluationEpisodeCompletionV1,
    processing_status: EvaluationProcessingStatusV1,
) -> tuple[SufficientStatisticDraftV1, ...]:
    del completion, processing_status
    transition_count = len(state.transition_ids)
    eligible_steps = transition_count * 10
    return (
        _draft(
            CountComponentV1(
                count=transition_count,
                eligible_episode_count=1,
            ),
            component_name="count",
            reducer_id=state.reducer_id,
            reducer_version=state.reducer_version,
            units="transitions",
        ),
        _draft(
            SumComponentV1(
                value=state.canonical_reward_sum,
                observation_count=transition_count,
                eligible_episode_count=1,
            ),
            component_name="sum",
            reducer_id=state.reducer_id,
            reducer_version=state.reducer_version,
            units="canonical_reward",
        ),
        _draft(
            RatioComponentV1(
                numerator=float(transition_count),
                denominator=float(transition_count * 2),
                zero_opportunity_occurrence=0 if transition_count else 1,
                eligible_episode_count=1,
            ),
            component_name="ratio",
            reducer_id=state.reducer_id,
            reducer_version=state.reducer_version,
            units="ratio_components",
        ),
        _draft(
            DurationComponentV1(
                qualifying_steps=transition_count,
                eligible_steps=eligible_steps,
                eligible_episode_count=1,
            ),
            component_name="duration",
            reducer_id=state.reducer_id,
            reducer_version=state.reducer_version,
            units="transition_ticks",
        ),
        _draft(
            OpportunityComponentV1(
                opportunity_count=transition_count,
                eligible_episode_count=1,
            ),
            component_name="opportunity",
            reducer_id=state.reducer_id,
            reducer_version=state.reducer_version,
            units="opportunities",
        ),
        _draft(
            DistributionComponentV1(
                observations=tuple(
                    DistributionObservationV1(
                        source_observation_id=transition_id,
                        ordinal=ordinal,
                        value=float(ordinal),
                    )
                    for ordinal, transition_id in enumerate(state.transition_ids)
                ),
                eligible_episode_count=1,
            ),
            component_name="distribution",
            reducer_id=state.reducer_id,
            reducer_version=state.reducer_version,
            units="transition_index",
        ),
    )


def _single_draft_builder(
    component: (
        CountComponentV1
        | SumComponentV1
        | RatioComponentV1
        | DurationComponentV1
        | OpportunityComponentV1
        | DistributionComponentV1
        | None
    ),
    **updates: object,
) -> _DraftBuilder:
    """Return a reducer-bound builder for one declarative draft."""

    def build(
        state: _ReducerState,
        completion: EvaluationEpisodeCompletionV1,
        processing_status: EvaluationProcessingStatusV1,
    ) -> tuple[SufficientStatisticDraftV1, ...]:
        del completion, processing_status
        return (
            _draft(
                component,
                reducer_id=state.reducer_id,
                reducer_version=state.reducer_version,
                **updates,
            ),
        )

    return build


@pytest.mark.parametrize(
    "component",
    _component_cases(),
    ids=lambda component: type(component).__name__,
)
def test_each_raw_component_family_round_trips_without_derived_values(
    component: EvaluationModel,
) -> None:
    """Every raw family survives strict JSON without a ratio or mean field."""
    draft = _draft(component)  # type: ignore[arg-type]

    restored = SufficientStatisticDraftV1.model_validate_json(draft.model_dump_json())

    assert restored == draft
    assert "mean" not in restored.model_dump()
    assert "computed_value" not in restored.model_dump()


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    (
        (
            CountComponentV1(count=2, eligible_episode_count=1),
            CountComponentV1(count=3, eligible_episode_count=1),
            CountComponentV1(count=5, eligible_episode_count=1),
        ),
        (
            SumComponentV1(
                value=4.5,
                observation_count=2,
                eligible_episode_count=1,
            ),
            SumComponentV1(
                value=-1.25,
                observation_count=1,
                eligible_episode_count=1,
            ),
            SumComponentV1(
                value=3.25,
                observation_count=3,
                eligible_episode_count=1,
            ),
        ),
        (
            RatioComponentV1(
                numerator=2.0,
                denominator=4.0,
                zero_opportunity_occurrence=0,
                eligible_episode_count=1,
            ),
            RatioComponentV1(
                numerator=-1.0,
                denominator=3.0,
                zero_opportunity_occurrence=0,
                eligible_episode_count=1,
            ),
            RatioComponentV1(
                numerator=1.0,
                denominator=7.0,
                zero_opportunity_occurrence=0,
                eligible_episode_count=1,
            ),
        ),
        (
            DurationComponentV1(
                qualifying_steps=2,
                eligible_steps=5,
                eligible_episode_count=1,
            ),
            DurationComponentV1(
                qualifying_steps=3,
                eligible_steps=7,
                eligible_episode_count=1,
            ),
            DurationComponentV1(
                qualifying_steps=5,
                eligible_steps=12,
                eligible_episode_count=1,
            ),
        ),
        (
            OpportunityComponentV1(
                opportunity_count=2,
                eligible_episode_count=1,
            ),
            OpportunityComponentV1(
                opportunity_count=5,
                eligible_episode_count=1,
            ),
            OpportunityComponentV1(
                opportunity_count=7,
                eligible_episode_count=1,
            ),
        ),
    ),
    ids=("count", "sum", "ratio", "duration", "opportunity"),
)
def test_accumulator_adds_each_scalar_component_family_immutably(
    left: EvaluationModel,
    right: EvaluationModel,
    expected: EvaluationModel,
) -> None:
    """Adding compatible drafts returns a new accumulator with raw totals."""
    first = SufficientStatisticAccumulatorV1().add(_draft(left))  # type: ignore[arg-type]
    first_snapshot = first.model_dump_json()

    merged = first.add(_draft(right))  # type: ignore[arg-type]

    assert first.model_dump_json() == first_snapshot
    assert first.entries[0].component == left
    assert merged.entries[0].component == expected


def test_ratio_merge_recomputes_one_episode_zero_opportunity_incidence() -> None:
    """Local contributions preserve a final per-episode 0/1 incidence value."""
    zero = RatioComponentV1(
        numerator=0.0,
        denominator=0.0,
        zero_opportunity_occurrence=1,
        eligible_episode_count=1,
    )
    positive = RatioComponentV1(
        numerator=1.0,
        denominator=2.0,
        zero_opportunity_occurrence=0,
        eligible_episode_count=1,
    )

    all_zero = SufficientStatisticAccumulatorV1().add(_draft(zero)).add(_draft(zero))
    mixed = SufficientStatisticAccumulatorV1().add(_draft(zero)).add(_draft(positive))

    assert all_zero.entries[0].component == RatioComponentV1(
        numerator=0.0,
        denominator=0.0,
        zero_opportunity_occurrence=1,
        eligible_episode_count=1,
    )
    assert mixed.entries[0].component == RatioComponentV1(
        numerator=1.0,
        denominator=2.0,
        zero_opportunity_occurrence=0,
        eligible_episode_count=1,
    )


def test_accumulator_concatenates_long_form_distribution_with_stable_links() -> None:
    """Distribution merge keeps every observation and assigns gap-free ordinals."""
    left = DistributionComponentV1(
        observations=(
            DistributionObservationV1(
                source_observation_id="transition-0:event-2",
                ordinal=0,
                value=1.5,
            ),
        ),
        eligible_episode_count=1,
    )
    right = DistributionComponentV1(
        observations=(
            DistributionObservationV1(
                source_observation_id="transition-1:event-0",
                ordinal=0,
                value=1.5,
            ),
            DistributionObservationV1(
                source_observation_id="transition-1:event-3",
                ordinal=1,
                value=-2.0,
            ),
        ),
        eligible_episode_count=1,
    )

    merged = SufficientStatisticAccumulatorV1().add(_draft(left)).add(_draft(right))
    component = merged.entries[0].component

    assert isinstance(component, DistributionComponentV1)
    assert tuple(row.ordinal for row in component.observations) == (0, 1, 2)
    assert tuple(row.source_observation_id for row in component.observations) == (
        "transition-0:event-2",
        "transition-1:event-0",
        "transition-1:event-3",
    )
    assert tuple(row.value for row in component.observations) == (1.5, 1.5, -2.0)
    assert component.eligible_episode_count == 1


def test_accumulator_rejects_duplicate_distribution_observation_links() -> None:
    """One linked source observation cannot enter a statistic row twice."""
    component = DistributionComponentV1(
        observations=(
            DistributionObservationV1(
                source_observation_id="event-duplicate",
                ordinal=0,
                value=1.0,
            ),
        ),
        eligible_episode_count=1,
    )
    accumulator = SufficientStatisticAccumulatorV1().add(_draft(component))

    with pytest.raises(ValueError, match="duplicate observation IDs"):
        accumulator.add(_draft(component))


def test_accumulator_canonicalizes_distinct_keys_and_preserves_subjects() -> None:
    """Insertion order never controls output order or collapse distinct subjects."""
    team_draft = _draft(
        CountComponentV1(count=1, eligible_episode_count=1),
        metric_id="marlbg.test.zeta.v1",
        subject=TeamStatisticSubjectV1(team_id=2),
    )
    agent_draft = _draft(
        CountComponentV1(count=2, eligible_episode_count=1),
        metric_id="marlbg.test.alpha.v1",
        subject=AgentStatisticSubjectV1(global_slot=0),
    )

    accumulator = SufficientStatisticAccumulatorV1().add(team_draft).add(agent_draft)

    assert tuple(row.metric_id for row in accumulator.entries) == (
        "marlbg.test.alpha.v1",
        "marlbg.test.zeta.v1",
    )
    assert accumulator.entries[0].subject == agent_draft.subject
    assert accumulator.entries[1].subject == team_draft.subject


def test_accumulator_rejects_metadata_drift_for_one_semantic_row() -> None:
    """The same row key cannot silently change units or reducer authority."""
    initial = _draft(
        CountComponentV1(count=1, eligible_episode_count=1),
        units="transition_ticks",
    )
    conflicting = _draft(
        CountComponentV1(count=1, eligible_episode_count=1),
        units="hit_points",
    )
    accumulator = SufficientStatisticAccumulatorV1().add(initial)

    with pytest.raises(ValueError, match="metadata conflicts"):
        accumulator.add(conflicting)


def test_accumulator_merge_is_copy_on_write_and_matches_repeated_add() -> None:
    """Accumulator merge uses the same strict algebra without mutating either side."""
    left = SufficientStatisticAccumulatorV1().add(
        _draft(
            CountComponentV1(count=2, eligible_episode_count=1),
            metric_id="marlbg.test.left.v1",
        )
    )
    right = SufficientStatisticAccumulatorV1().add(
        _draft(
            OpportunityComponentV1(
                opportunity_count=3,
                eligible_episode_count=1,
            ),
            metric_id="marlbg.test.right.v1",
        )
    )
    left_snapshot = left.model_dump_json()
    right_snapshot = right.model_dump_json()

    merged = left.merge(right)

    assert left.model_dump_json() == left_snapshot
    assert right.model_dump_json() == right_snapshot
    assert merged == left.add(right.entries[0])


@pytest.mark.parametrize(
    "factory",
    (
        lambda: CountComponentV1(count=True, eligible_episode_count=1),
        lambda: CountComponentV1.model_validate(
            {"count": 1, "eligible_episode_count": True}
        ),
        lambda: CountComponentV1.model_validate(
            {"count": 1, "eligible_episode_count": 1.0}
        ),
        lambda: CountComponentV1(count=-1, eligible_episode_count=1),
        lambda: CountComponentV1.model_validate(
            {"count": 1, "eligible_episode_count": 2}
        ),
        lambda: SumComponentV1(
            value=float("nan"),
            observation_count=1,
            eligible_episode_count=1,
        ),
        lambda: SumComponentV1(
            value=1.0,
            observation_count=0,
            eligible_episode_count=1,
        ),
        lambda: RatioComponentV1(
            numerator=1.0,
            denominator=0.0,
            zero_opportunity_occurrence=1,
            eligible_episode_count=1,
        ),
        lambda: RatioComponentV1(
            numerator=1.0,
            denominator=2.0,
            zero_opportunity_occurrence=1,
            eligible_episode_count=1,
        ),
        lambda: RatioComponentV1.model_validate(
            {
                "numerator": 0.0,
                "denominator": 0.0,
                "zero_opportunity_occurrence": True,
                "eligible_episode_count": 1,
            }
        ),
        lambda: RatioComponentV1.model_validate(
            {
                "numerator": 0.0,
                "denominator": 0.0,
                "zero_opportunity_occurrence": 1.0,
                "eligible_episode_count": 1,
            }
        ),
        lambda: DurationComponentV1(
            qualifying_steps=2,
            eligible_steps=1,
            eligible_episode_count=1,
        ),
        lambda: OpportunityComponentV1(
            opportunity_count=-1,
            eligible_episode_count=1,
        ),
        lambda: DistributionObservationV1(
            source_observation_id="event-1",
            ordinal=0,
            value=float("inf"),
        ),
        lambda: DistributionComponentV1(
            observations=(
                DistributionObservationV1(
                    source_observation_id="event-1",
                    ordinal=1,
                    value=1.0,
                ),
            ),
            eligible_episode_count=1,
        ),
        lambda: DistributionComponentV1(
            observations=(
                DistributionObservationV1(
                    source_observation_id="event-1",
                    ordinal=0,
                    value=1.0,
                ),
                DistributionObservationV1(
                    source_observation_id="event-1",
                    ordinal=1,
                    value=2.0,
                ),
            ),
            eligible_episode_count=1,
        ),
    ),
    ids=(
        "bool-as-int",
        "bool-as-episode-eligibility",
        "float-as-episode-eligibility",
        "negative-count",
        "cross-episode-eligibility",
        "nonfinite-sum",
        "nonzero-empty-sum",
        "zero-denominator-positive-numerator",
        "positive-denominator-zero-occurrence",
        "bool-as-zero-opportunity-occurrence",
        "float-as-zero-opportunity-occurrence",
        "duration-overflow",
        "negative-opportunity",
        "nonfinite-distribution",
        "distribution-ordinal-gap",
        "distribution-duplicate-source",
    ),
)
def test_component_models_reject_malformed_raw_values(
    factory: Callable[[], object],
) -> None:
    """Strict raw records reject coercion, nonfinite data, and broken invariants."""
    with pytest.raises((ValidationError, ValueError)):
        factory()


def test_draft_rejects_unknown_fields_lists_and_component_discriminators() -> None:
    """Draft construction remains strict, tuple-backed, and version-aware."""
    valid = _draft(CountComponentV1(count=1, eligible_episode_count=1))
    payload = valid.model_dump(mode="python")

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SufficientStatisticDraftV1.model_validate({**payload, "unknown": "field"})
    with pytest.raises(ValidationError):
        SufficientStatisticDraftV1.model_validate({**payload, "dimensions": []})
    component = dict(payload["component"])  # type: ignore[arg-type]
    component["component_type"] = "mean"
    with pytest.raises(ValidationError):
        SufficientStatisticDraftV1.model_validate({**payload, "component": component})


def test_nested_model_copy_escapes_are_structurally_revalidated() -> None:
    """Frozen outer models cannot bless unchecked invalid nested model copies."""
    valid_component = RatioComponentV1(
        numerator=1.0,
        denominator=1.0,
        zero_opportunity_occurrence=0,
        eligible_episode_count=1,
    )
    unsafe_component = valid_component.model_copy(update={"denominator": 0.0})
    draft_payload = _draft(valid_component).model_dump(mode="python")
    draft_payload["component"] = unsafe_component

    with pytest.raises(ValidationError, match="zero denominator"):
        SufficientStatisticDraftV1.model_validate(draft_payload)

    unsafe_draft = _draft(valid_component).model_copy(
        update={"component": unsafe_component}
    )
    with pytest.raises(ValidationError, match="statistic component"):
        SufficientStatisticAccumulatorV1(entries=(unsafe_draft,))


@pytest.mark.parametrize(
    "component",
    (
        _UnfrozenCountComponent(count=1, eligible_episode_count=1),
        _PrivateCountComponent(count=1, eligible_episode_count=1),
    ),
    ids=("unfrozen-subtype", "private-state-subtype"),
)
def test_draft_rejects_mutable_nested_schema_subtypes(
    component: CountComponentV1,
) -> None:
    """Declared nested unions never retain a mutable schema subtype."""
    payload = _draft(CountComponentV1(count=1, eligible_episode_count=1)).model_dump(
        mode="python"
    )
    payload["component"] = component

    with pytest.raises(ValidationError, match="exact declared schema type"):
        SufficientStatisticDraftV1.model_validate(payload)


@pytest.mark.parametrize(
    "updates",
    (
        {"metric_id": "marlbg.test.metric.v2"},
        {"result_status": "defined", "component": None},
        {"result_status": "defined", "status_reason": "unexpected"},
        {
            "result_status": "insufficient_data",
            "status_reason": None,
        },
        {
            "result_status": "zero_opportunity",
            "status_reason": "no opportunities",
            "component": CountComponentV1(count=1, eligible_episode_count=1),
        },
        {
            "endpoint_observation_status": "right_censored",
            "supports_right_censoring": False,
        },
        {"amount_stage": "effective"},
    ),
    ids=(
        "metric-version-mismatch",
        "defined-without-component",
        "defined-with-reason",
        "undefined-without-reason",
        "false-zero-opportunity",
        "unsupported-right-censoring",
        "unknown-amount-stage",
    ),
)
def test_draft_rejects_inconsistent_semantic_metadata(
    updates: dict[str, object],
) -> None:
    """Eligibility and endpoint metadata cannot contradict raw evidence."""
    payload_updates = dict(updates)
    component = payload_updates.pop(
        "component",
        CountComponentV1(count=1, eligible_episode_count=1),
    )
    with pytest.raises((ValidationError, ValueError)):
        _draft(
            component,  # type: ignore[arg-type]
            **payload_updates,
        )


@pytest.mark.parametrize(
    "component",
    (
        SumComponentV1(
            value=0.0,
            observation_count=0,
            eligible_episode_count=1,
        ),
        RatioComponentV1(
            numerator=0.0,
            denominator=0.0,
            zero_opportunity_occurrence=1,
            eligible_episode_count=1,
        ),
        DurationComponentV1(
            qualifying_steps=0,
            eligible_steps=0,
            eligible_episode_count=1,
        ),
        OpportunityComponentV1(
            opportunity_count=0,
            eligible_episode_count=1,
        ),
        DistributionComponentV1(observations=(), eligible_episode_count=1),
    ),
    ids=("sum", "ratio", "duration", "opportunity", "distribution"),
)
def test_each_component_family_can_represent_zero_opportunity(
    component: EvaluationModel,
) -> None:
    """Zero opportunity is explicit raw evidence, never an implicit zero result."""
    draft = _draft(
        component,  # type: ignore[arg-type]
        result_status="zero_opportunity",
        status_reason="no genuine opportunity",
    )

    assert draft.result_status == "zero_opportunity"
    assert draft.component == component


def test_count_cannot_claim_zero_opportunity_without_exposure_evidence() -> None:
    """A count has no denominator or exposure field proving zero opportunity."""
    with pytest.raises(ValueError, match="zero-opportunity"):
        _draft(
            CountComponentV1(count=0, eligible_episode_count=1),
            result_status="zero_opportunity",
            status_reason="unsupported inference",
        )


def test_zero_opportunity_requires_an_eligible_episode() -> None:
    """An empty neutral aggregate is not evidence of a genuine zero opportunity."""
    with pytest.raises(ValueError, match="zero-opportunity evidence"):
        _draft(
            SumComponentV1(
                value=0.0,
                observation_count=0,
                eligible_episode_count=0,
            ),
            result_status="zero_opportunity",
            status_reason="no eligible episode",
        )


def test_dimensions_are_sorted_unique_and_cannot_shadow_context_truth() -> None:
    """Metric dimensions supplement rather than overwrite episode provenance."""
    alpha = StatisticDimensionV1(name="ability", value="basic")
    zeta = StatisticDimensionV1(name="target_class", value="mage")

    with pytest.raises(ValueError, match="canonically sorted"):
        _draft(
            CountComponentV1(count=1, eligible_episode_count=1),
            dimensions=(zeta, alpha),
        )
    with pytest.raises(ValueError, match="must be unique"):
        _draft(
            CountComponentV1(count=1, eligible_episode_count=1),
            dimensions=(alpha, alpha.model_copy(update={"value": "ultimate"})),
        )
    for reserved_name in (
        "task",
        "algorithm_id",
        "metric_id",
        "schema_version",
        "identity",
    ):
        with pytest.raises(ValueError, match="shadows context truth"):
            StatisticDimensionV1(name=reserved_name, value="forged-truth")


def test_subject_union_preserves_episode_team_agent_class_and_ordered_pair() -> None:
    """Every supported subject kind is discriminated and round-trippable."""
    subjects = (
        EpisodeStatisticSubjectV1(),
        TeamStatisticSubjectV1(team_id=1),
        AgentStatisticSubjectV1(global_slot=0),
        TeamClassStatisticSubjectV1(team_id=2, class_id=5),
        AgentPairStatisticSubjectV1(
            primary_global_slot=0,
            secondary_global_slot=5,
        ),
    )

    for subject in subjects:
        draft = _draft(
            CountComponentV1(count=1, eligible_episode_count=1),
            subject=subject,
        )
        assert (
            SufficientStatisticDraftV1.model_validate_json(
                draft.model_dump_json()
            ).subject
            == subject
        )

    with pytest.raises(ValueError, match="two distinct slots"):
        AgentPairStatisticSubjectV1(
            primary_global_slot=2,
            secondary_global_slot=2,
        )
    with pytest.raises(ValidationError):
        AgentStatisticSubjectV1(global_slot=10)


def test_accumulator_rejects_mutable_or_duplicate_entry_storage() -> None:
    """The aggregate itself remains tuple-backed with one row per semantic key."""
    draft = _draft(CountComponentV1(count=1, eligible_episode_count=1))

    with pytest.raises(ValidationError):
        SufficientStatisticAccumulatorV1.model_validate({"entries": [draft]})
    with pytest.raises(ValueError, match="unique statistic keys"):
        SufficientStatisticAccumulatorV1(entries=(draft, draft))


def test_observer_rejects_adversarial_mutable_context_subtype() -> None:
    """Constructor equality cannot bless a mutable undeclared context root."""
    trajectory = captured_evaluation_trajectory(transition_count=0)
    unsafe_context = _AdversarialEvaluationContext.model_validate(
        trajectory.context.model_dump(mode="python")
    )

    with pytest.raises(ValueError, match="exact declared root type"):
        build_evaluation_observer_v1(unsafe_context)


def test_initial_validator_accepts_public_capture_at_nonzero_simulator_epoch() -> None:
    """Artifact frame zero is independent of the simulator's starting epoch."""
    trajectory = captured_evaluation_trajectory(transition_count=0)
    initial_frame = trajectory.frames[0].model_copy(update={"simulator_step_count": 17})

    validate_initial_evaluation_frame_v1(trajectory.context, initial_frame)


def test_initial_frame_validator_rejects_nonzero_artifact_index() -> None:
    """Every observer starts from artifact frame zero, even for resumed state."""
    trajectory = captured_evaluation_trajectory(transition_count=0)
    episode_id = trajectory.context.identity.episode_id
    noninitial_frame = trajectory.frames[0].model_copy(
        update={
            "frame_index": 1,
            "frame_id": f"{episode_id}:frame:1",
        }
    )

    with pytest.raises(ValueError, match="initial frame index must be zero"):
        validate_initial_evaluation_frame_v1(trajectory.context, noninitial_frame)


def test_initial_frame_validator_rejects_context_episode_mismatch() -> None:
    """A structurally valid frame cannot silently join another episode."""
    trajectory = captured_evaluation_trajectory(transition_count=0)
    other_episode_frame = trajectory.frames[0].model_copy(
        update={
            "episode_id": "other-episode",
            "frame_id": "other-episode:frame:0",
        }
    )

    with pytest.raises(ValueError, match="must join to the context episode"):
        validate_initial_evaluation_frame_v1(
            trajectory.context,
            other_episode_frame,
        )


def test_initial_frame_validator_revalidates_unchecked_nested_payloads() -> None:
    """Pydantic escape hatches cannot insert mutable payloads into a frame."""
    trajectory = captured_evaluation_trajectory(transition_count=0)
    initial_frame = trajectory.frames[0]
    unchecked_snapshot = initial_frame.snapshot.model_copy(
        update={"alive_mask": list(initial_frame.snapshot.alive_mask)}
    )
    unchecked_frame = initial_frame.model_copy(update={"snapshot": unchecked_snapshot})

    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="fails structural revalidation"):
            validate_initial_evaluation_frame_v1(
                trajectory.context,
                unchecked_frame,
            )

    assert emitted == []


@pytest.mark.parametrize(
    "frame_type",
    (_UnfrozenEvaluationFrame, _PrivateEvaluationFrame),
    ids=("unfrozen-subtype", "private-state-subtype"),
)
def test_observer_rejects_undeclared_initial_frame_subtypes(
    frame_type: type[EvaluationFrameV1],
) -> None:
    """Frame-zero validation accepts only the exact frozen public root type."""
    trajectory = captured_evaluation_trajectory(transition_count=0)
    unsafe_frame = frame_type.model_validate(
        trajectory.frames[0].model_dump(mode="python")
    )
    observer = build_evaluation_observer_v1(trajectory.context)

    with pytest.raises(ValueError, match="exact declared root type"):
        observer.start(unsafe_frame)

    assert observer.lifecycle_state == "poisoned"
    assert observer.retained_frames == ()


def test_observer_rejects_undeclared_nested_initial_frame_model() -> None:
    """An exact frame root cannot hide a mutable snapshot subtype."""
    trajectory = captured_evaluation_trajectory(transition_count=0)
    initial_frame = trajectory.frames[0]
    unsafe_snapshot = _AdversarialSnapshot.model_validate(
        initial_frame.snapshot.model_dump(mode="python")
    )
    unsafe_frame = initial_frame.model_copy(update={"snapshot": unsafe_snapshot})
    observer = build_evaluation_observer_v1(trajectory.context)

    with pytest.raises(ValueError, match="undeclared nested model type"):
        observer.start(unsafe_frame)

    assert observer.lifecycle_state == "poisoned"
    assert observer.retained_frames == ()


def test_observer_rejects_hidden_private_state_on_exact_frame_root() -> None:
    """Exact public root types cannot smuggle undeclared Pydantic private data."""
    trajectory = captured_evaluation_trajectory(transition_count=0)
    unsafe_frame = EvaluationFrameV1.model_validate(
        trajectory.frames[0].model_dump(mode="python")
    )
    object.__setattr__(
        unsafe_frame,
        "__pydantic_private__",
        {"_hidden": [1]},
    )
    observer = build_evaluation_observer_v1(trajectory.context)

    with pytest.raises(ValueError, match="undeclared nested model type"):
        observer.start(unsafe_frame)

    assert observer.lifecycle_state == "poisoned"
    assert observer.retained_frames == ()


def test_initial_frame_validator_enforces_inactive_slot_padding() -> None:
    """A context-inactive slot remains neutral in the accepted frame-zero prefix."""
    trajectory = captured_evaluation_trajectory(transition_count=0)
    initial_frame = trajectory.frames[0]
    inactive_slot = next(
        row.global_slot
        for row in trajectory.context.roster
        if not row.configured_active
    )
    alive_mask = list(initial_frame.snapshot.alive_mask)
    alive_mask[inactive_slot] = True
    unchecked_snapshot = initial_frame.snapshot.model_copy(
        update={"alive_mask": tuple(alive_mask)}
    )
    unchecked_frame = initial_frame.model_copy(update={"snapshot": unchecked_snapshot})

    with pytest.raises(
        ValueError,
        match=f"inactive slot {inactive_slot} must be neutral",
    ):
        validate_initial_evaluation_frame_v1(
            trajectory.context,
            unchecked_frame,
        )


@pytest.fixture(scope="module")
def two_transition_trajectory() -> CapturedEvaluationTrajectory:
    """Share one deterministic public trajectory across observer-only tests."""
    return captured_evaluation_trajectory(
        transition_count=2,
        expected_horizon=2,
    )


@pytest.fixture(scope="module")
def one_transition_trajectory() -> CapturedEvaluationTrajectory:
    """Share one exact-horizon transition across adversarial observer tests."""
    return captured_evaluation_trajectory(
        transition_count=1,
        expected_horizon=1,
    )


def test_observer_streams_coherent_views_and_finalizes_all_six_families(
    two_transition_trajectory: CapturedEvaluationTrajectory,
) -> None:
    """One valid stream advances immutable state and yields a joined report."""
    trajectory = two_transition_trajectory
    reducer = _Reducer(draft_builder=_six_family_drafts)
    observer = build_evaluation_observer_v1(
        trajectory.context,
        reducers=(reducer,),
    )

    assert observer.lifecycle_state == "awaiting_initial"
    assert observer.validated_transition_count == 0
    assert observer.processed_transition_count == 0

    observer.start(trajectory.frames[0])
    assert observer.lifecycle_state == "open"

    observer.append(trajectory.transitions[0], trajectory.frames[1])
    assert observer.lifecycle_state == "open"
    assert observer.validated_transition_count == 1
    assert observer.processed_transition_count == 1

    observer.append(trajectory.transitions[1], trajectory.frames[2])
    assert observer.lifecycle_state == "sealed"
    assert observer.validated_transition_count == 2
    assert observer.processed_transition_count == 2

    states = observer.reducer_states
    assert states is not None
    assert len(states) == 1
    state = states[0]
    assert isinstance(state, _ReducerState)
    assert state.initial_frame_id == trajectory.frames[0].frame_id
    assert state.transition_ids == tuple(
        transition.transition_id for transition in trajectory.transitions
    )
    assert state.start_frame_ids == tuple(
        frame.frame_id for frame in trajectory.frames[:-1]
    )
    assert state.successor_frame_ids == tuple(
        frame.frame_id for frame in trajectory.frames[1:]
    )

    report = observer.finalize(completion_state="complete")

    assert observer.lifecycle_state == "finalized"
    assert report.context == trajectory.context
    assert report.completion.completion_state == "complete"
    assert report.completion.completion_bases == ("declared_horizon",)
    assert report.processing_status.status == "succeeded"
    assert report.processing_status.processed_transition_count == 2
    assert tuple(row.component_name for row in report.statistics) == (
        "count",
        "distribution",
        "duration",
        "opportunity",
        "ratio",
        "sum",
    )
    assert all(row.component is not None for row in report.statistics)
    assert {
        row.component.component_type
        for row in report.statistics
        if row.component is not None
    } == {
        "count",
        "sum",
        "ratio",
        "duration",
        "opportunity",
        "distribution",
    }
    for row in report.statistics:
        assert row.episode_id == trajectory.context.identity.episode_id
        assert row.aggregation_keys == trajectory.context.aggregation_keys
        assert row.source_schema_versions == trajectory.context.schema_versions
        assert row.rollout_completion == report.completion
        assert row.processing_status == report.processing_status
        assert row.validated_transition_count == 2
        assert row.processed_transition_count == 2
    assert (
        EvaluationMetricReportV1.model_validate_json(report.model_dump_json()) == report
    )


@pytest.mark.parametrize(
    ("capture_profile", "with_scenario", "retains_trajectory"),
    (
        ("training_light", False, False),
        ("debug", False, False),
        ("evaluation_metric_complete", False, True),
        ("scenario_metric_complete", True, True),
    ),
)
def test_capture_profiles_share_semantic_stream_but_control_retention(
    capture_profile: CaptureProfile,
    with_scenario: bool,
    retains_trajectory: bool,
) -> None:
    """Every enabled profile reduces views; only metric-complete profiles retain."""
    trajectory = captured_evaluation_trajectory(
        transition_count=1,
        capture_profile=capture_profile,
        expected_horizon=1,
        with_scenario=with_scenario,
    )
    observer = build_evaluation_observer_v1(
        trajectory.context,
        reducers=(_Reducer(),),
    )

    _feed_observer(observer, trajectory.frames, trajectory.transitions)
    report = observer.finalize(completion_state="complete")

    assert observer.validated_transition_count == 1
    assert observer.processed_transition_count == 1
    assert report.statistics[0].component == CountComponentV1(
        count=1,
        eligible_episode_count=1,
    )
    if retains_trajectory:
        assert observer.retained_frames == trajectory.frames
        assert observer.retained_transitions == trajectory.transitions
    else:
        assert observer.retained_frames is None
        assert observer.retained_transitions is None


def test_invalid_append_poisons_only_the_last_valid_prefix(
    two_transition_trajectory: CapturedEvaluationTrajectory,
) -> None:
    """A gap is rejected before artifact or reducer progress can advance."""
    trajectory = two_transition_trajectory
    observer = build_evaluation_observer_v1(
        trajectory.context,
        reducers=(_Reducer(),),
    )
    observer.start(trajectory.frames[0])
    state_before = observer.reducer_states

    with pytest.raises(ValueError, match="frame index"):
        observer.append(trajectory.transitions[1], trajectory.frames[2])

    assert observer.lifecycle_state == "poisoned"
    assert observer.validated_transition_count == 0
    assert observer.processed_transition_count == 0
    assert observer.reducer_states == state_before
    assert observer.retained_frames == (trajectory.frames[0],)
    assert observer.retained_transitions == ()

    report = observer.finalize(
        completion_state="partial",
        end_or_failure_reason="invalid transition unit",
    )

    assert report.completion.completion_state == "partial"
    assert report.completion.last_valid_frame_index == 0
    assert report.processing_status.status == "failed"
    assert report.processing_status.failure is not None
    assert report.processing_status.failure.stage == "transition_validation"
    assert report.processing_status.failure.attempted_transition_index == 1
    assert report.statistics[0].result_status == "defined"
    assert report.statistics[0].component == CountComponentV1(
        count=0,
        eligible_episode_count=1,
    )


@pytest.mark.parametrize(
    "frame_type",
    (_UnfrozenEvaluationFrame, _PrivateEvaluationFrame),
    ids=("unfrozen-subtype", "private-state-subtype"),
)
def test_observer_rejects_undeclared_successor_frame_subtypes(
    one_transition_trajectory: CapturedEvaluationTrajectory,
    frame_type: type[EvaluationFrameV1],
) -> None:
    """A mutable successor root cannot enter validated or retained progress."""
    trajectory = one_transition_trajectory
    unsafe_successor = frame_type.model_validate(
        trajectory.frames[1].model_dump(mode="python")
    )
    observer = build_evaluation_observer_v1(
        trajectory.context,
        reducers=(_Reducer(),),
    )
    observer.start(trajectory.frames[0])
    state_before = observer.reducer_states

    with pytest.raises(ValueError, match="exact declared root type"):
        observer.append(trajectory.transitions[0], unsafe_successor)

    assert observer.lifecycle_state == "poisoned"
    assert observer.validated_transition_count == 0
    assert observer.processed_transition_count == 0
    assert observer.reducer_states == state_before
    assert observer.retained_frames == (trajectory.frames[0],)
    assert observer.retained_transitions == ()


@pytest.mark.parametrize(
    "transition_type",
    (_UnfrozenEvaluationTransition, _PrivateEvaluationTransition),
    ids=("unfrozen-subtype", "private-state-subtype"),
)
def test_observer_rejects_undeclared_transition_subtypes(
    one_transition_trajectory: CapturedEvaluationTrajectory,
    transition_type: type[EvaluationTransitionV1],
) -> None:
    """A mutable transition root cannot enter validated or retained progress."""
    trajectory = one_transition_trajectory
    unsafe_transition = transition_type.model_validate(
        trajectory.transitions[0].model_dump(mode="python")
    )
    observer = build_evaluation_observer_v1(
        trajectory.context,
        reducers=(_Reducer(),),
    )
    observer.start(trajectory.frames[0])
    state_before = observer.reducer_states

    with pytest.raises(ValueError, match="exact declared root type"):
        observer.append(unsafe_transition, trajectory.frames[1])

    assert observer.lifecycle_state == "poisoned"
    assert observer.validated_transition_count == 0
    assert observer.processed_transition_count == 0
    assert observer.reducer_states == state_before
    assert observer.retained_frames == (trajectory.frames[0],)
    assert observer.retained_transitions == ()


def test_observer_rejects_undeclared_nested_transition_model(
    one_transition_trajectory: CapturedEvaluationTrajectory,
) -> None:
    """An exact transition root cannot hide a mutable facts subtype."""
    trajectory = one_transition_trajectory
    transition = trajectory.transitions[0]
    unsafe_facts = _AdversarialTransitionFacts.model_validate(
        transition.facts.model_dump(mode="python")
    )
    unsafe_transition = transition.model_copy(update={"facts": unsafe_facts})
    observer = build_evaluation_observer_v1(
        trajectory.context,
        reducers=(_Reducer(),),
    )
    observer.start(trajectory.frames[0])
    state_before = observer.reducer_states

    with pytest.raises(ValueError, match="undeclared nested model type"):
        observer.append(unsafe_transition, trajectory.frames[1])

    assert observer.lifecycle_state == "poisoned"
    assert observer.validated_transition_count == 0
    assert observer.processed_transition_count == 0
    assert observer.reducer_states == state_before
    assert observer.retained_frames == (trajectory.frames[0],)
    assert observer.retained_transitions == ()


def test_observer_retains_canonical_copies_not_caller_owned_roots(
    one_transition_trajectory: CapturedEvaluationTrajectory,
) -> None:
    """Post-validation caller mutation cannot rewrite the observer prefix."""
    trajectory = one_transition_trajectory
    initial_frame = EvaluationFrameV1.model_validate(
        trajectory.frames[0].model_dump(mode="python")
    )
    transition = EvaluationTransitionV1.model_validate(
        trajectory.transitions[0].model_dump(mode="python")
    )
    successor_frame = EvaluationFrameV1.model_validate(
        trajectory.frames[1].model_dump(mode="python")
    )
    observer = build_evaluation_observer_v1(
        trajectory.context,
        reducers=(_Reducer(),),
    )
    observer.start(initial_frame)
    observer.append(transition, successor_frame)

    object.__setattr__(initial_frame, "frame_index", 9)
    object.__setattr__(transition, "transition_index", 9)
    object.__setattr__(successor_frame, "frame_index", 9)

    assert observer.retained_frames is not None
    assert observer.retained_transitions is not None
    assert tuple(row.frame_index for row in observer.retained_frames) == (0, 1)
    assert tuple(row.transition_index for row in observer.retained_transitions) == (0,)


@pytest.mark.parametrize(
    ("error", "expected_detail"),
    (
        (RuntimeError("bad\x00detail\x7f"), r"bad\x00detail\x7f"),
        (_UnprintableError(), "_UnprintableError"),
    ),
    ids=("control-bytes", "broken-string-conversion"),
)
def test_reducer_failure_detail_is_always_printable_and_cannot_mask_poisoning(
    one_transition_trajectory: CapturedEvaluationTrajectory,
    error: Exception,
    expected_detail: str,
) -> None:
    """Failure-envelope construction cannot fail behind a hostile exception."""
    trajectory = one_transition_trajectory
    observer = build_evaluation_observer_v1(
        trajectory.context,
        reducers=(_Reducer(advance_error=error),),
    )
    observer.start(trajectory.frames[0])

    with pytest.raises(RuntimeError, match="advance failed"):
        observer.append(trajectory.transitions[0], trajectory.frames[1])

    assert observer.lifecycle_state == "poisoned"
    assert observer.validated_transition_count == 1
    assert observer.processed_transition_count == 0
    report = observer.finalize(completion_state="complete")
    failure = report.processing_status.failure
    assert failure is not None
    assert failure.stage == "reducer_advance"
    assert expected_detail in failure.detail
    assert all(0x20 <= ord(character) <= 0x7E for character in failure.detail)


def test_malformed_append_attribute_access_is_captured_before_validation(
    one_transition_trajectory: CapturedEvaluationTrajectory,
) -> None:
    """Attempted-index inspection is part of the poisoning transaction."""
    trajectory = one_transition_trajectory
    observer = build_evaluation_observer_v1(
        trajectory.context,
        reducers=(_Reducer(),),
    )
    observer.start(trajectory.frames[0])
    malformed = cast(EvaluationTransitionV1, _UnreadableTransition())

    with pytest.raises(RuntimeError, match="cannot inspect transition_index"):
        observer.append(malformed, trajectory.frames[1])

    assert observer.lifecycle_state == "poisoned"
    assert observer.validated_transition_count == 0
    assert observer.processed_transition_count == 0
    assert observer.retained_frames == (trajectory.frames[0],)
    assert observer.retained_transitions == ()
    report = observer.finalize(
        completion_state="partial",
        end_or_failure_reason="malformed transition",
    )
    failure = report.processing_status.failure
    assert failure is not None
    assert failure.stage == "transition_validation"
    assert failure.attempted_transition_index is None


def test_reducer_failure_commits_artifact_progress_but_no_reducer_state(
    one_transition_trajectory: CapturedEvaluationTrajectory,
) -> None:
    """Reducer replacement is all-or-none while physical completion stays true."""
    trajectory = one_transition_trajectory
    reducers = (
        _Reducer(reducer_id="a.reducer"),
        _Reducer(reducer_id="b.reducer", fail_advance_index=0),
    )
    observer = build_evaluation_observer_v1(
        trajectory.context,
        reducers=reducers,
    )
    observer.start(trajectory.frames[0])
    states_before = observer.reducer_states

    with pytest.raises(RuntimeError, match=r"b\.reducer advance failed"):
        observer.append(trajectory.transitions[0], trajectory.frames[1])

    assert observer.lifecycle_state == "poisoned"
    assert observer.validated_transition_count == 1
    assert observer.processed_transition_count == 0
    assert observer.reducer_states == states_before
    assert observer.retained_frames == trajectory.frames
    assert observer.retained_transitions == trajectory.transitions
    states_after = observer.reducer_states
    assert states_after is not None
    assert all(
        isinstance(state, _ReducerState) and state.transition_ids == ()
        for state in states_after
    )

    report = observer.finalize(completion_state="complete")

    assert report.completion.completion_state == "complete"
    assert report.completion.completion_bases == ("declared_horizon",)
    assert report.completion.validated_transition_count == 1
    assert report.processing_status.status == "failed"
    assert report.processing_status.processed_transition_count == 0
    assert report.processing_status.failure is not None
    assert report.processing_status.failure.stage == "reducer_advance"
    assert report.processing_status.failure.reducer_id == "b.reducer"
    assert report.processing_status.failure.attempted_transition_index == 0


@pytest.mark.parametrize(
    "reducer",
    (
        _Reducer(wrong_identity_on_advance=True),
        _Reducer(wrong_type_on_advance=True),
    ),
    ids=("wrong-identity", "wrong-replacement-type"),
)
def test_reducer_replacement_contract_failure_preserves_previous_state(
    one_transition_trajectory: CapturedEvaluationTrajectory,
    reducer: _Reducer,
) -> None:
    """Replacement state identity and exact type are part of atomic advance."""
    trajectory = one_transition_trajectory
    observer = build_evaluation_observer_v1(
        trajectory.context,
        reducers=(reducer,),
    )
    observer.start(trajectory.frames[0])
    state_before = observer.reducer_states

    with pytest.raises(RuntimeError, match="advance failed"):
        observer.append(trajectory.transitions[0], trajectory.frames[1])

    assert observer.reducer_states == state_before
    assert observer.validated_transition_count == 1
    assert observer.processed_transition_count == 0
    report = observer.finalize(completion_state="complete")
    assert report.completion.completion_state == "complete"
    assert report.processing_status.status == "failed"
    assert report.processing_status.failure is not None
    assert report.processing_status.failure.stage == "reducer_advance"


def test_reducer_initialization_is_atomic_after_valid_frame_zero(
    one_transition_trajectory: CapturedEvaluationTrajectory,
) -> None:
    """Initialization failure preserves a reportable T=0 valid artifact prefix."""
    trajectory = one_transition_trajectory
    observer = build_evaluation_observer_v1(
        trajectory.context,
        reducers=(
            _Reducer(reducer_id="a.reducer"),
            _Reducer(reducer_id="b.reducer", fail_initialize=True),
        ),
    )

    with pytest.raises(RuntimeError, match=r"b\.reducer initialization failed"):
        observer.start(trajectory.frames[0])

    assert observer.lifecycle_state == "poisoned"
    assert observer.reducer_states is None
    assert observer.validated_transition_count == 0
    assert observer.processed_transition_count == 0
    assert observer.retained_frames == (trajectory.frames[0],)

    report = observer.finalize(
        completion_state="failed",
        end_or_failure_reason="metric initialization failed",
        failure_origin="capture",
    )
    assert report.completion.completion_state == "failed"
    assert report.completion.validated_transition_count == 0
    assert report.processing_status.status == "failed"
    assert report.processing_status.failure is not None
    assert report.processing_status.failure.stage == "reducer_initialize"
    assert report.statistics == ()


def test_illegal_lifecycle_call_poisoning_does_not_rewrite_rollout_truth(
    one_transition_trajectory: CapturedEvaluationTrajectory,
) -> None:
    """A post-seal host misuse fails processing without falsifying completion."""
    trajectory = one_transition_trajectory
    observer = build_evaluation_observer_v1(
        trajectory.context,
        reducers=(
            _Reducer(
                draft_builder=_single_draft_builder(
                    CountComponentV1(count=1, eligible_episode_count=1),
                    completion_scope="complete_episode",
                )
            ),
        ),
    )
    _feed_observer(observer, trajectory.frames, trajectory.transitions)
    assert observer.lifecycle_state == "sealed"

    with pytest.raises(RuntimeError, match="append is not allowed"):
        observer.append(trajectory.transitions[0], trajectory.frames[1])

    assert observer.lifecycle_state == "poisoned"
    assert observer.validated_transition_count == 1
    assert observer.processed_transition_count == 1
    report = observer.finalize(completion_state="complete")
    assert report.completion.completion_state == "complete"
    assert report.processing_status.status == "failed"
    assert report.processing_status.failure is not None
    assert report.processing_status.failure.stage == "lifecycle"
    assert report.statistics[0].result_status == "defined"
    assert report.statistics[0].component == CountComponentV1(
        count=1,
        eligible_episode_count=1,
    )


@pytest.mark.parametrize(
    ("completion_state", "failure_origin"),
    (
        ("partial", None),
        ("interrupted", None),
        ("failed", "simulation"),
    ),
)
def test_valid_frame_zero_can_finalize_every_noncomplete_rollout_state(
    completion_state: str,
    failure_origin: str | None,
) -> None:
    """T=0 remains a reportable artifact prefix for every noncomplete outcome."""
    trajectory = captured_evaluation_trajectory(
        transition_count=0,
        expected_horizon=2,
    )
    observer = build_evaluation_observer_v1(
        trajectory.context,
        reducers=(_Reducer(),),
    )
    observer.start(trajectory.frames[0])

    report = observer.finalize(
        completion_state=completion_state,  # type: ignore[arg-type]
        end_or_failure_reason=f"{completion_state} before transition zero",
        failure_origin=failure_origin,  # type: ignore[arg-type]
    )

    assert report.completion.completion_state == completion_state
    assert report.completion.validated_transition_count == 0
    assert report.completion.last_valid_frame_index == 0
    assert report.completion.completion_bases == ()
    assert report.completion.failure_origin == failure_origin
    assert report.processing_status == EvaluationProcessingStatusV1(
        status="succeeded",
        processed_transition_count=0,
    )


def test_complete_rollout_rejects_valid_t_zero_without_terminal_or_horizon() -> None:
    """A valid initial frame alone is not physical evidence of completion."""
    trajectory = captured_evaluation_trajectory(
        transition_count=0,
        expected_horizon=2,
    )
    observer = build_evaluation_observer_v1(trajectory.context)
    observer.start(trajectory.frames[0])

    with pytest.raises(ValueError, match="task termination or declared horizon"):
        observer.finalize(completion_state="complete")

    assert observer.lifecycle_state == "poisoned"
    assert observer.validated_transition_count == 0
    with pytest.raises(RuntimeError, match="finalize is not allowed"):
        observer.finalize(completion_state="complete")


@pytest.mark.parametrize(
    ("completion_state", "reason", "failure_origin", "message"),
    (
        ("partial", None, None, "require a reason"),
        ("interrupted", None, None, "require a reason"),
        ("failed", "runner failed", None, "requires a failure origin"),
        ("partial", "runner stopped", "policy", "only failed"),
    ),
)
def test_noncomplete_rollout_metadata_is_not_invented_or_contradictory(
    completion_state: str,
    reason: str | None,
    failure_origin: str | None,
    message: str,
) -> None:
    """Callers must explicitly and consistently classify incomplete rollouts."""
    trajectory = captured_evaluation_trajectory(
        transition_count=0,
        expected_horizon=2,
    )
    observer = build_evaluation_observer_v1(trajectory.context)
    observer.start(trajectory.frames[0])

    with pytest.raises(ValueError, match=message):
        observer.finalize(
            completion_state=completion_state,  # type: ignore[arg-type]
            end_or_failure_reason=reason,
            failure_origin=failure_origin,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("expected_horizon", "expected_bases"),
    (
        (2, ("task_terminal",)),
        (1, ("task_terminal", "declared_horizon")),
    ),
)
def test_task_terminal_completion_preserves_authoritative_reason_and_bases(
    expected_horizon: int,
    expected_bases: tuple[str, ...],
) -> None:
    """Terminal evidence and horizon evidence remain independent and canonical."""
    trajectory = captured_evaluation_trajectory(
        transition_count=1,
        expected_horizon=expected_horizon,
    )
    terminal_transition = EvaluationTransitionV1.model_validate(
        {
            **trajectory.transitions[0].model_dump(mode="python"),
            "terminated": True,
            "owning_task_end_reason": "objective_complete",
        }
    )
    observer = build_evaluation_observer_v1(
        trajectory.context,
        reducers=(_Reducer(),),
    )
    observer.start(trajectory.frames[0])
    observer.append(terminal_transition, trajectory.frames[1])

    report = observer.finalize(completion_state="complete")

    assert report.completion.terminated is True
    assert report.completion.truncated is False
    assert report.completion.completion_bases == expected_bases
    assert report.completion.end_or_failure_reason == "objective_complete"
    assert report.processing_status.status == "succeeded"


def test_short_truncated_rollout_is_noncomplete_but_still_fully_processed() -> None:
    """Truncation seals input without pretending the declared horizon was met."""
    trajectory = captured_evaluation_trajectory(
        transition_count=1,
        expected_horizon=2,
    )
    truncated_transition = EvaluationTransitionV1.model_validate(
        {
            **trajectory.transitions[0].model_dump(mode="python"),
            "truncated": True,
            "owning_task_end_reason": "external_limit",
        }
    )
    observer = build_evaluation_observer_v1(
        trajectory.context,
        reducers=(_Reducer(),),
    )
    observer.start(trajectory.frames[0])
    observer.append(truncated_transition, trajectory.frames[1])

    report = observer.finalize(
        completion_state="partial",
    )

    assert report.completion.completion_state == "partial"
    assert report.completion.terminated is False
    assert report.completion.truncated is True
    assert report.completion.completion_bases == ()
    assert report.completion.end_or_failure_reason == "external_limit"
    assert report.processing_status.status == "succeeded"
    assert report.processing_status.processed_transition_count == 1


@pytest.mark.parametrize(
    "reducer",
    (
        _Reducer(fail_finalize=True),
        _Reducer(return_mutable_finalize=True),
    ),
    ids=("raised-error", "mutable-result"),
)
def test_final_reducer_failure_does_not_falsify_complete_rollout(
    one_transition_trajectory: CapturedEvaluationTrajectory,
    reducer: _Reducer,
) -> None:
    """Final processing failure is recorded independently from physical truth."""
    trajectory = one_transition_trajectory
    observer = build_evaluation_observer_v1(
        trajectory.context,
        reducers=(reducer,),
    )
    _feed_observer(observer, trajectory.frames, trajectory.transitions)

    report = observer.finalize(completion_state="complete")

    assert report.completion.completion_state == "complete"
    assert report.completion.completion_bases == ("declared_horizon",)
    assert report.completion.validated_transition_count == 1
    assert report.processing_status.status == "failed"
    assert report.processing_status.processed_transition_count == 1
    assert report.processing_status.failure is not None
    assert report.processing_status.failure.stage == "reducer_finalize"
    assert report.statistics == ()


def test_secondary_finalization_failure_preserves_the_first_processing_failure(
    one_transition_trajectory: CapturedEvaluationTrajectory,
) -> None:
    """One failure record keeps the poison cause while disclosing later failure."""
    trajectory = one_transition_trajectory
    observer = build_evaluation_observer_v1(
        trajectory.context,
        reducers=(_Reducer(fail_advance_index=0, fail_finalize=True),),
    )
    observer.start(trajectory.frames[0])
    with pytest.raises(RuntimeError, match="advance failed"):
        observer.append(trajectory.transitions[0], trajectory.frames[1])

    report = observer.finalize(completion_state="complete")

    assert report.processing_status.failure is not None
    assert report.processing_status.failure.stage == "reducer_advance"
    assert "secondary reducer_finalize/reducer_finalize_failed" in (
        report.processing_status.failure.detail
    )
    assert report.statistics == ()


def test_public_roots_reject_rows_from_atomic_finalization_failures(
    one_transition_trajectory: CapturedEvaluationTrajectory,
) -> None:
    """Model revalidation cannot reintroduce rows that were never published."""
    trajectory = one_transition_trajectory
    observer = build_evaluation_observer_v1(
        trajectory.context,
        reducers=(_Reducer(),),
    )
    _feed_observer(observer, trajectory.frames, trajectory.transitions)
    healthy_report = observer.finalize(completion_state="complete")
    failure = EvaluationProcessingFailureV1(
        stage="reducer_finalize",
        code="reducer_finalize_failed",
        reducer_id="test.reducer",
        reducer_version=1,
        attempted_transition_index=None,
        detail="intentional failure",
    )
    failed_processing = EvaluationProcessingStatusV1(
        status="failed",
        processed_transition_count=1,
        failure=failure,
    )
    raw_payload = healthy_report.statistics[0].model_dump(mode="python")
    raw_payload["processing_status"] = failed_processing

    with pytest.raises(ValidationError, match="cannot publish a raw statistic"):
        RawSufficientStatisticV1.model_validate(raw_payload)


def test_public_roots_reject_impossible_failure_stage_progress(
    one_transition_trajectory: CapturedEvaluationTrajectory,
) -> None:
    """Failure stage determines the only valid processed-prefix boundary."""
    trajectory = one_transition_trajectory
    observer = build_evaluation_observer_v1(
        trajectory.context,
        reducers=(_Reducer(),),
    )
    _feed_observer(observer, trajectory.frames, trajectory.transitions)
    report = observer.finalize(completion_state="complete")

    initialization_failure = EvaluationProcessingStatusV1(
        status="failed",
        processed_transition_count=0,
        failure=EvaluationProcessingFailureV1(
            stage="reducer_initialize",
            code="reducer_initialize_failed",
            reducer_id="test.reducer",
            reducer_version=1,
            attempted_transition_index=None,
            detail="impossible late initialization failure",
        ),
    )
    report_payload = report.model_dump(mode="python")
    report_payload["processing_status"] = initialization_failure
    report_payload["statistics"] = ()
    with pytest.raises(ValidationError, match="zero-transition prefix"):
        EvaluationMetricReportV1.model_validate(report_payload)

    advance_failure = EvaluationProcessingStatusV1(
        status="failed",
        processed_transition_count=1,
        failure=EvaluationProcessingFailureV1(
            stage="reducer_advance",
            code="reducer_advance_failed",
            reducer_id="test.reducer",
            reducer_version=1,
            attempted_transition_index=0,
            detail="impossible equal-prefix advance failure",
        ),
    )
    raw_payload = report.statistics[0].model_dump(mode="python")
    raw_payload["processing_status"] = advance_failure
    raw_payload["processed_transition_count"] = 1
    with pytest.raises(ValidationError, match="one unprocessed"):
        RawSufficientStatisticV1.model_validate(raw_payload)


def test_completion_and_processing_records_are_strict_and_round_trip() -> None:
    """Lifecycle records remain immutable, versioned, and independently serializable."""
    completion = EvaluationEpisodeCompletionV1(
        episode_id="episode-001",
        completion_state="partial",
        expected_transition_count=2,
        validated_transition_count=0,
        last_valid_frame_index=0,
        last_valid_frame_id="episode-001:frame:0",
        terminated=False,
        truncated=False,
        completion_bases=(),
        end_or_failure_reason="stopped",
        failure_origin=None,
    )
    failure = EvaluationProcessingFailureV1(
        stage="reducer_advance",
        code="reducer_advance_failed",
        reducer_id="test.reducer",
        reducer_version=1,
        attempted_transition_index=0,
        detail="intentional failure",
    )
    processing = EvaluationProcessingStatusV1(
        status="failed",
        processed_transition_count=0,
        failure=failure,
    )

    assert (
        EvaluationEpisodeCompletionV1.model_validate_json(completion.model_dump_json())
        == completion
    )
    assert (
        EvaluationProcessingStatusV1.model_validate_json(processing.model_dump_json())
        == processing
    )
    with pytest.raises(ValidationError):
        EvaluationProcessingStatusV1.model_validate(
            {
                "status": "succeeded",
                "processed_transition_count": 0,
                "failure": failure,
            }
        )


@pytest.mark.parametrize("done_field", ("terminated", "truncated"))
def test_zero_transition_completion_rejects_unowned_done_flags(
    done_field: str,
) -> None:
    """A public completion root cannot claim done without a transition."""
    payload: dict[str, object] = {
        "episode_id": "episode-001",
        "completion_state": "complete",
        "expected_transition_count": 2,
        "validated_transition_count": 0,
        "last_valid_frame_index": 0,
        "last_valid_frame_id": "episode-001:frame:0",
        "terminated": False,
        "truncated": False,
        "completion_bases": ("task_terminal",),
        "end_or_failure_reason": None,
        "failure_origin": None,
    }
    payload[done_field] = True

    with pytest.raises(ValidationError, match="zero-transition completion"):
        EvaluationEpisodeCompletionV1.model_validate(payload)
    with pytest.raises(ValidationError):
        EvaluationProcessingStatusV1.model_validate(
            {
                "status": "failed",
                "processed_transition_count": 0,
                "failure": None,
            }
        )


@pytest.mark.parametrize(
    "component",
    (
        CountComponentV1(count=0, eligible_episode_count=1),
        SumComponentV1(
            value=0.0,
            observation_count=0,
            eligible_episode_count=1,
        ),
        DurationComponentV1(
            qualifying_steps=0,
            eligible_steps=0,
            eligible_episode_count=1,
        ),
        OpportunityComponentV1(
            opportunity_count=0,
            eligible_episode_count=1,
        ),
        DistributionComponentV1(
            observations=(),
            eligible_episode_count=1,
        ),
    ),
    ids=("count", "sum", "duration", "opportunity", "distribution"),
)
def test_non_ratio_zero_values_do_not_invent_metric_opportunity_semantics(
    one_transition_trajectory: CapturedEvaluationTrajectory,
    component: EvaluationModel,
) -> None:
    """Generic zero values remain defined unless their component owns exposure."""
    trajectory = one_transition_trajectory
    observer = build_evaluation_observer_v1(
        trajectory.context,
        reducers=(
            _Reducer(
                draft_builder=_single_draft_builder(
                    component,  # type: ignore[arg-type]
                    component_name=type(component).__name__.lower(),
                )
            ),
        ),
    )
    _feed_observer(observer, trajectory.frames, trajectory.transitions)

    report = observer.finalize(completion_state="complete")

    assert report.processing_status.status == "succeeded"
    assert report.statistics[0].result_status == "defined"
    assert report.statistics[0].status_reason is None
    assert report.statistics[0].component == component


def test_ratio_zero_opportunity_is_classified_only_at_final_materialization(
    one_transition_trajectory: CapturedEvaluationTrajectory,
) -> None:
    """A provisional ratio draft becomes N/A once episode eligibility is known."""
    component = RatioComponentV1(
        numerator=0.0,
        denominator=0.0,
        zero_opportunity_occurrence=1,
        eligible_episode_count=1,
    )
    assert _draft(component).result_status == "defined"
    trajectory = one_transition_trajectory
    observer = build_evaluation_observer_v1(
        trajectory.context,
        reducers=(
            _Reducer(
                draft_builder=_single_draft_builder(
                    component,
                    component_name="zero_ratio",
                )
            ),
        ),
    )
    _feed_observer(observer, trajectory.frames, trajectory.transitions)

    report = observer.finalize(completion_state="complete")

    assert report.processing_status.status == "succeeded"
    assert report.statistics[0].result_status == "zero_opportunity"
    assert report.statistics[0].status_reason == (
        "no genuine opportunities in eligible episode"
    )
    assert report.statistics[0].component == component


def test_defined_final_statistic_requires_an_eligible_episode(
    one_transition_trajectory: CapturedEvaluationTrajectory,
) -> None:
    """Reducer-local neutral exposure cannot become a defined finalized row."""
    trajectory = one_transition_trajectory
    observer = build_evaluation_observer_v1(
        trajectory.context,
        reducers=(
            _Reducer(
                draft_builder=_single_draft_builder(
                    CountComponentV1(count=0, eligible_episode_count=0),
                )
            ),
        ),
    )
    _feed_observer(observer, trajectory.frames, trajectory.transitions)

    report = observer.finalize(completion_state="complete")

    assert report.processing_status.status == "failed"
    assert report.processing_status.failure is not None
    assert report.processing_status.failure.stage == "statistic_materialization"
    assert "require one eligible episode" in report.processing_status.failure.detail
    assert report.statistics == ()


def test_complete_episode_statistic_downgrades_on_valid_partial_prefix(
    two_transition_trajectory: CapturedEvaluationTrajectory,
) -> None:
    """Scientific eligibility changes the row, not physical rollout completion."""
    trajectory = two_transition_trajectory
    observer = build_evaluation_observer_v1(
        trajectory.context,
        reducers=(
            _Reducer(
                draft_builder=_single_draft_builder(
                    CountComponentV1(count=1, eligible_episode_count=1),
                    completion_scope="complete_episode",
                    endpoint_observation_status="observed",
                )
            ),
        ),
    )
    observer.start(trajectory.frames[0])
    observer.append(trajectory.transitions[0], trajectory.frames[1])

    report = observer.finalize(
        completion_state="partial",
        end_or_failure_reason="runner stopped after a valid prefix",
    )

    row = report.statistics[0]
    assert report.completion.completion_state == "partial"
    assert report.processing_status.status == "succeeded"
    assert row.result_status == "insufficient_data"
    assert row.status_reason == "complete episode and fully processed prefix required"
    assert row.endpoint_observation_status == "unavailable"
    assert row.component == CountComponentV1(
        count=1,
        eligible_episode_count=0,
    )

    tampered_payload = report.model_dump(mode="python")
    tampered_row = dict(tampered_payload["statistics"][0])
    tampered_row.update(
        {
            "result_status": "defined",
            "status_reason": None,
            "endpoint_observation_status": "observed",
        }
    )
    tampered_payload["statistics"] = (tampered_row,)
    with pytest.raises(ValidationError, match="must reflect episode"):
        EvaluationMetricReportV1.model_validate(tampered_payload)


@pytest.mark.parametrize(
    "result_status",
    (
        "invalid_artifact",
        "structurally_inapplicable",
        "ambiguous_attribution",
        "insufficient_data",
    ),
)
def test_complete_episode_downgrade_respects_status_precedence(
    two_transition_trajectory: CapturedEvaluationTrajectory,
    result_status: str,
) -> None:
    """Eligibility cannot overwrite a more authoritative reducer declaration."""
    trajectory = two_transition_trajectory
    observer = build_evaluation_observer_v1(
        trajectory.context,
        reducers=(
            _Reducer(
                draft_builder=_single_draft_builder(
                    None,
                    completion_scope="complete_episode",
                    result_status=result_status,
                    status_reason="reducer-owned status",
                )
            ),
        ),
    )
    observer.start(trajectory.frames[0])
    observer.append(trajectory.transitions[0], trajectory.frames[1])

    report = observer.finalize(
        completion_state="partial",
        end_or_failure_reason="runner stopped after a valid prefix",
    )

    assert report.statistics[0].result_status == result_status
    assert report.statistics[0].status_reason == "reducer-owned status"


@pytest.mark.parametrize(
    ("endpoint_status", "supports_right_censoring"),
    (
        ("not_applicable", False),
        ("observed", False),
        ("right_censored", True),
        ("competing_event", False),
        ("unavailable", False),
    ),
)
def test_all_endpoint_statuses_materialize_when_their_evidence_is_valid(
    one_transition_trajectory: CapturedEvaluationTrajectory,
    endpoint_status: str,
    supports_right_censoring: bool,
) -> None:
    """Endpoint status is reducer evidence constrained by final completion truth."""
    trajectory = one_transition_trajectory
    observer = build_evaluation_observer_v1(
        trajectory.context,
        reducers=(
            _Reducer(
                draft_builder=_single_draft_builder(
                    CountComponentV1(count=1, eligible_episode_count=1),
                    endpoint_observation_status=endpoint_status,
                    supports_right_censoring=supports_right_censoring,
                )
            ),
        ),
    )
    _feed_observer(observer, trajectory.frames, trajectory.transitions)

    report = observer.finalize(completion_state="complete")

    assert report.processing_status.status == "succeeded"
    assert report.statistics[0].endpoint_observation_status == endpoint_status


@pytest.mark.parametrize(
    ("endpoint_status", "supports_right_censoring"),
    (
        ("right_censored", True),
        ("competing_event", False),
    ),
)
def test_endpoint_claim_requiring_completion_fails_on_partial_rollout(
    two_transition_trajectory: CapturedEvaluationTrajectory,
    endpoint_status: str,
    supports_right_censoring: bool,
) -> None:
    """A reducer cannot upgrade a partial artifact into observed endpoint truth."""
    trajectory = two_transition_trajectory
    observer = build_evaluation_observer_v1(
        trajectory.context,
        reducers=(
            _Reducer(
                draft_builder=_single_draft_builder(
                    CountComponentV1(count=1, eligible_episode_count=1),
                    endpoint_observation_status=endpoint_status,
                    supports_right_censoring=supports_right_censoring,
                )
            ),
        ),
    )
    observer.start(trajectory.frames[0])
    observer.append(trajectory.transitions[0], trajectory.frames[1])

    report = observer.finalize(
        completion_state="partial",
        end_or_failure_reason="runner stopped",
    )

    assert report.completion.completion_state == "partial"
    assert report.processing_status.status == "failed"
    assert report.processing_status.failure is not None
    assert report.processing_status.failure.stage == "statistic_materialization"
    assert report.statistics == ()


def test_authoritative_transition_end_reason_rejects_completion_disagreement() -> None:
    """CP3 may preserve but never rewrite a CP2 owning-task end reason."""
    trajectory = captured_evaluation_trajectory(
        transition_count=1,
        expected_horizon=2,
    )
    truncated_transition = EvaluationTransitionV1.model_validate(
        {
            **trajectory.transitions[0].model_dump(mode="python"),
            "truncated": True,
            "owning_task_end_reason": "external_limit",
        }
    )
    observer = build_evaluation_observer_v1(trajectory.context)
    observer.start(trajectory.frames[0])
    observer.append(truncated_transition, trajectory.frames[1])

    with pytest.raises(ValueError, match="authoritative task end reason"):
        observer.finalize(
            completion_state="partial",
            end_or_failure_reason="different_reason",
        )


def test_report_subject_rows_join_active_context_entities_and_sort_canonically(
    one_transition_trajectory: CapturedEvaluationTrajectory,
) -> None:
    """Every supported subject resolves through context.

    Canonical report order is independent of reducer return order.
    """
    trajectory = one_transition_trajectory

    def subject_drafts(
        state: _ReducerState,
        completion: EvaluationEpisodeCompletionV1,
        processing_status: EvaluationProcessingStatusV1,
    ) -> tuple[SufficientStatisticDraftV1, ...]:
        del completion, processing_status
        subjects = (
            (
                "zeta_pair",
                AgentPairStatisticSubjectV1(
                    primary_global_slot=0,
                    secondary_global_slot=5,
                ),
            ),
            ("team", TeamStatisticSubjectV1(team_id=1)),
            ("team_class", TeamClassStatisticSubjectV1(team_id=1, class_id=1)),
            ("episode", EpisodeStatisticSubjectV1()),
            ("agent", AgentStatisticSubjectV1(global_slot=0)),
        )
        return tuple(
            _draft(
                CountComponentV1(count=1, eligible_episode_count=1),
                component_name=component_name,
                reducer_id=state.reducer_id,
                reducer_version=state.reducer_version,
                subject=subject,
            )
            for component_name, subject in subjects
        )

    observer = build_evaluation_observer_v1(
        trajectory.context,
        reducers=(_Reducer(draft_builder=subject_drafts),),
    )
    _feed_observer(observer, trajectory.frames, trajectory.transitions)

    report = observer.finalize(completion_state="complete")

    assert report.processing_status.status == "succeeded"
    assert tuple(row.component_name for row in report.statistics) == (
        "agent",
        "episode",
        "team",
        "team_class",
        "zeta_pair",
    )
    assert tuple(row.subject for row in report.statistics) == (
        AgentStatisticSubjectV1(global_slot=0),
        EpisodeStatisticSubjectV1(),
        TeamStatisticSubjectV1(team_id=1),
        TeamClassStatisticSubjectV1(team_id=1, class_id=1),
        AgentPairStatisticSubjectV1(
            primary_global_slot=0,
            secondary_global_slot=5,
        ),
    )


@pytest.mark.parametrize(
    "subject",
    (
        AgentStatisticSubjectV1(global_slot=3),
        AgentPairStatisticSubjectV1(
            primary_global_slot=0,
            secondary_global_slot=3,
        ),
        TeamClassStatisticSubjectV1(team_id=1, class_id=4),
    ),
    ids=("inactive-agent", "inactive-pair-member", "absent-team-class"),
)
def test_defined_subject_that_does_not_join_context_fails_the_report(
    one_transition_trajectory: CapturedEvaluationTrajectory,
    subject: EvaluationModel,
) -> None:
    """Reducers cannot create defined rows for inactive or absent subjects."""
    trajectory = one_transition_trajectory
    observer = build_evaluation_observer_v1(
        trajectory.context,
        reducers=(
            _Reducer(
                draft_builder=_single_draft_builder(
                    CountComponentV1(count=1, eligible_episode_count=1),
                    subject=subject,
                )
            ),
        ),
    )
    _feed_observer(observer, trajectory.frames, trajectory.transitions)

    report = observer.finalize(completion_state="complete")

    assert report.completion.completion_state == "complete"
    assert report.processing_status.status == "failed"
    assert report.processing_status.failure is not None
    assert report.processing_status.failure.stage == "report_validation"
    assert report.statistics == ()


@pytest.mark.parametrize(
    "result_status",
    ("invalid_artifact", "structurally_inapplicable"),
)
def test_absent_team_class_preserves_stronger_status_precedence(
    one_transition_trajectory: CapturedEvaluationTrajectory,
    result_status: str,
) -> None:
    """Artifact invalidity outranks structural absence for the same stratum."""
    trajectory = one_transition_trajectory
    absent_subject = TeamClassStatisticSubjectV1(team_id=1, class_id=4)
    observer = build_evaluation_observer_v1(
        trajectory.context,
        reducers=(
            _Reducer(
                draft_builder=_single_draft_builder(
                    None,
                    subject=absent_subject,
                    result_status=result_status,
                    status_reason="strongest applicable result status",
                )
            ),
        ),
    )
    _feed_observer(observer, trajectory.frames, trajectory.transitions)

    report = observer.finalize(completion_state="complete")

    assert report.processing_status.status == "succeeded"
    assert report.statistics[0].subject == absent_subject
    assert report.statistics[0].result_status == result_status


def test_duplicate_statistic_key_across_reducers_fails_atomically(
    one_transition_trajectory: CapturedEvaluationTrajectory,
) -> None:
    """Reducer identity does not permit duplicate semantic rows in one report."""
    trajectory = one_transition_trajectory
    observer = build_evaluation_observer_v1(
        trajectory.context,
        reducers=(
            _Reducer(
                reducer_id="a.reducer",
                draft_builder=_single_draft_builder(
                    CountComponentV1(count=1, eligible_episode_count=1),
                    component_name="duplicate",
                ),
            ),
            _Reducer(
                reducer_id="b.reducer",
                draft_builder=_single_draft_builder(
                    CountComponentV1(count=1, eligible_episode_count=1),
                    component_name="duplicate",
                ),
            ),
        ),
    )
    _feed_observer(observer, trajectory.frames, trajectory.transitions)

    report = observer.finalize(completion_state="complete")

    assert report.completion.completion_state == "complete"
    assert report.processing_status.status == "failed"
    assert report.processing_status.failure is not None
    assert report.processing_status.failure.stage == "statistic_materialization"
    assert report.statistics == ()


def test_report_rows_preserve_context_owned_aggregation_keys_exactly() -> None:
    """Aggregation strata are joined from context rather than reducer dimensions."""
    aggregation_keys = (
        AggregationKeyV1(name="fold", value="validation"),
        AggregationKeyV1(name="information_regime", value="no_shared_obs"),
        AggregationKeyV1(name="side", value="team_b"),
    )
    trajectory = captured_evaluation_trajectory(
        transition_count=1,
        expected_horizon=1,
        aggregation_keys=aggregation_keys,
        episode_id="aggregation-episode",
    )
    observer = build_evaluation_observer_v1(
        trajectory.context,
        reducers=(_Reducer(),),
    )
    _feed_observer(observer, trajectory.frames, trajectory.transitions)

    report = observer.finalize(completion_state="complete")

    assert report.context.aggregation_keys == aggregation_keys
    assert report.statistics[0].aggregation_keys == aggregation_keys
    assert report.statistics[0].episode_id == "aggregation-episode"
    assert report.statistics[0].dimensions == ()

    tampered_payload = report.model_dump(mode="python")
    tampered_row = dict(tampered_payload["statistics"][0])
    tampered_row["dimensions"] = (StatisticDimensionV1(name="fold", value="test"),)
    tampered_payload["statistics"] = (tampered_row,)
    with pytest.raises(ValidationError, match="shadow context aggregation keys"):
        EvaluationMetricReportV1.model_validate(tampered_payload)


def test_observer_registration_requires_immutable_unique_valid_reducers(
    one_transition_trajectory: CapturedEvaluationTrajectory,
) -> None:
    """Reducer registration is deterministic before any frame is consumed."""
    context = one_transition_trajectory.context

    with pytest.raises(TypeError, match="immutable tuple"):
        build_evaluation_observer_v1(
            context,
            reducers=cast(tuple[_Reducer, ...], [_Reducer()]),
        )
    with pytest.raises(ValueError, match="IDs must be unique"):
        build_evaluation_observer_v1(
            context,
            reducers=(_Reducer(), _Reducer()),
        )
    with pytest.raises(ValueError, match="ASCII identifier"):
        build_evaluation_observer_v1(
            context,
            reducers=(_Reducer(reducer_id="bad reducer id"),),
        )
    with pytest.raises(ValueError, match="positive strict integer"):
        build_evaluation_observer_v1(
            context,
            reducers=(_Reducer(reducer_version=0),),
        )


def test_observer_rejects_invalid_start_before_initializing_reducers(
    one_transition_trajectory: CapturedEvaluationTrajectory,
) -> None:
    """Invalid frame zero cannot become retained or reducer-visible evidence."""
    trajectory = one_transition_trajectory
    initial_frame = trajectory.frames[0]
    invalid_frame = initial_frame.model_copy(
        update={
            "frame_index": 1,
            "frame_id": f"{trajectory.context.identity.episode_id}:frame:1",
        }
    )
    observer = build_evaluation_observer_v1(
        trajectory.context,
        reducers=(_Reducer(),),
    )

    with pytest.raises(ValueError, match="initial frame index"):
        observer.start(invalid_frame)

    assert observer.lifecycle_state == "poisoned"
    assert observer.reducer_states is None
    assert observer.retained_frames == ()
    assert observer.validated_transition_count == 0
    assert observer.processed_transition_count == 0
    with pytest.raises(RuntimeError, match="valid initial frame"):
        observer.finalize(
            completion_state="failed",
            end_or_failure_reason="invalid initial frame",
            failure_origin="validation",
        )


def test_start_is_single_use_and_report_finalization_is_explicit(
    two_transition_trajectory: CapturedEvaluationTrajectory,
) -> None:
    """The observer never restarts or implicitly finalizes its valid prefix."""
    trajectory = two_transition_trajectory
    observer = build_evaluation_observer_v1(
        trajectory.context,
        reducers=(_Reducer(),),
    )
    observer.start(trajectory.frames[0])

    with pytest.raises(RuntimeError, match="start is not allowed"):
        observer.start(trajectory.frames[0])

    assert observer.lifecycle_state == "poisoned"
    report = observer.finalize(
        completion_state="partial",
        end_or_failure_reason="duplicate start call",
    )
    assert observer.lifecycle_state == "finalized"
    assert report.processing_status.status == "failed"
    assert report.processing_status.failure is not None
    assert report.processing_status.failure.stage == "lifecycle"
    with pytest.raises(RuntimeError, match="finalize is not allowed"):
        observer.finalize(
            completion_state="partial",
            end_or_failure_reason="duplicate finalization",
        )
    with pytest.raises(RuntimeError, match="append is not allowed"):
        observer.append(trajectory.transitions[0], trajectory.frames[1])


def test_live_and_json_rehydrated_trajectories_produce_identical_reports() -> None:
    """Metric reduction depends only on serialized CP2 semantic records."""
    trajectory = captured_evaluation_trajectory(
        transition_count=2,
        expected_horizon=2,
        actions=(
            mage_target_none_ultimate_action(),
            evaluation_fixtures_module.neutral_action(),
        ),
    )
    assert trajectory.transitions[0].events != ()

    def evaluate(
        context: EvaluationEpisodeContextV1,
        frames: tuple[EvaluationFrameV1, ...],
        transitions: tuple[EvaluationTransitionV1, ...],
    ) -> EvaluationMetricReportV1:
        observer = build_evaluation_observer_v1(
            context,
            reducers=(_Reducer(draft_builder=_six_family_drafts),),
        )
        _feed_observer(observer, frames, transitions)
        return observer.finalize(completion_state="complete")

    live_report = evaluate(
        trajectory.context,
        trajectory.frames,
        trajectory.transitions,
    )
    restored_context = EvaluationEpisodeContextV1.model_validate_json(
        trajectory.context.model_dump_json()
    )
    restored_frames = tuple(
        EvaluationFrameV1.model_validate_json(frame.model_dump_json())
        for frame in trajectory.frames
    )
    restored_transitions = tuple(
        EvaluationTransitionV1.model_validate_json(transition.model_dump_json())
        for transition in trajectory.transitions
    )

    restored_report = evaluate(
        restored_context,
        restored_frames,
        restored_transitions,
    )

    assert restored_report == live_report
    assert restored_report.model_dump_json() == live_report.model_dump_json()


def test_metric_report_contains_only_frozen_host_records_and_no_trajectory(
    one_transition_trajectory: CapturedEvaluationTrajectory,
) -> None:
    """The final report is a logging seam, not a replay or device-array container."""
    trajectory = one_transition_trajectory
    observer = build_evaluation_observer_v1(
        trajectory.context,
        reducers=(_Reducer(),),
    )
    _feed_observer(observer, trajectory.frames, trajectory.transitions)
    report = observer.finalize(completion_state="complete")

    assert {"frames", "transitions", "trajectory"}.isdisjoint(
        EvaluationMetricReportV1.model_fields
    )

    def assert_host_only(value: object) -> None:
        assert not type(value).__module__.startswith(("jax", "numpy"))
        if isinstance(value, dict):
            for key, nested in cast(dict[object, object], value).items():
                assert_host_only(key)
                assert_host_only(nested)
        elif isinstance(value, (tuple, list)):
            sequence = cast(tuple[object, ...] | list[object], value)
            for nested in sequence:
                assert_host_only(nested)

    assert_host_only(report.model_dump(mode="python"))
    with pytest.raises(ValidationError, match="frozen"):
        report.statistics = ()


def test_disabled_observer_is_absence_before_any_evaluation_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Library-level caller gating performs no context, capture, or metric work."""
    calls: list[str] = []

    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        calls.append("evaluation work")
        raise AssertionError("disabled evaluation path performed work")

    monkeypatch.setattr(
        evaluation_fixtures_module,
        "captured_evaluation_trajectory",
        forbidden,
    )
    monkeypatch.setattr(evaluation_fixtures_module, "evaluation_context", forbidden)
    monkeypatch.setattr(
        evaluation_fixtures_module,
        "capture_initial_evaluation_frame_v1",
        forbidden,
    )
    monkeypatch.setattr(
        evaluation_fixtures_module,
        "capture_evaluation_transition_unit_v1",
        forbidden,
    )
    monkeypatch.setattr(jax, "device_get", forbidden)
    monkeypatch.setattr(metrics_module, "build_evaluation_observer_v1", forbidden)
    monkeypatch.setattr(
        metrics_module,
        "validate_initial_evaluation_frame_v1",
        forbidden,
    )
    monkeypatch.setattr(
        metrics_module,
        "validate_evaluation_transition_unit_v1",
        forbidden,
        raising=False,
    )
    monkeypatch.setattr(
        validation_module,
        "decode_evaluation_events_v1",
        forbidden,
    )
    monkeypatch.setattr(metrics_module, "EvaluationMetricReportV1", forbidden)

    def run_optional_evaluation(enabled: bool) -> object | None:
        if not enabled:
            return None
        trajectory = evaluation_fixtures_module.captured_evaluation_trajectory(
            transition_count=1,
            expected_horizon=1,
        )
        observer = metrics_module.build_evaluation_observer_v1(trajectory.context)
        _feed_observer(observer, trajectory.frames, trajectory.transitions)
        return observer.finalize(completion_state="complete")

    assert run_optional_evaluation(enabled=False) is None
    assert calls == []
    assert not hasattr(metrics_module, "NoOpEvaluationObserverV1")


def test_metrics_module_dependency_boundary_excludes_execution_and_persistence() -> (
    None
):
    """CP3 remains host plumbing with no core, device, replay, or logging dependency."""
    module_tree = ast.parse(inspect.getsource(metrics_module))
    imported_modules: set[str] = set()
    called_names: set[str] = set()
    for node in ast.walk(module_tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            called_names.add(node.func.id)

    forbidden_roots = {
        "jax",
        "numpy",
        "marl_battlegrounds.core",
        "marl_battlegrounds.debugger",
        "marl_battlegrounds.rendering",
        "marl_battlegrounds.replay",
        "marl_battlegrounds.learner",
    }
    assert not any(
        imported == forbidden or imported.startswith(f"{forbidden}.")
        for imported in imported_modules
        for forbidden in forbidden_roots
    )
    assert {"open", "print", "exec", "eval"}.isdisjoint(called_names)


def test_cp3_public_roots_reject_unknown_versions_and_context_keeps_cp2_bindings(
    one_transition_trajectory: CapturedEvaluationTrajectory,
) -> None:
    """CP3 roots version independently while context retains the eight CP2 roots."""
    trajectory = one_transition_trajectory
    observer = build_evaluation_observer_v1(
        trajectory.context,
        reducers=(_Reducer(),),
    )
    _feed_observer(observer, trajectory.frames, trajectory.transitions)
    report = observer.finalize(completion_state="complete")
    records: tuple[EvaluationModel, ...] = (
        report.completion,
        report.processing_status,
        report.statistics[0],
        report,
    )
    root_types: tuple[type[EvaluationModel], ...] = (
        EvaluationEpisodeCompletionV1,
        EvaluationProcessingStatusV1,
        RawSufficientStatisticV1,
        EvaluationMetricReportV1,
    )
    for root_type, record in zip(root_types, records, strict=True):
        for invalid_version in (2, True, 1.0):
            with pytest.raises(ValidationError):
                root_type.model_validate(
                    {
                        **record.model_dump(mode="python"),
                        "schema_version": invalid_version,
                    }
                )

    unsafe_row = report.statistics[0].model_copy(update={"component": None})
    report_payload = report.model_dump(mode="python")
    report_payload["statistics"] = (unsafe_row,)
    with pytest.raises(ValidationError, match="defined statistics require"):
        EvaluationMetricReportV1.model_validate(report_payload)

    assert len(trajectory.context.schema_versions) == 8
    context_payload = trajectory.context.model_dump(mode="python")
    context_payload["schema_versions"] = trajectory.context.schema_versions[:-1]
    with pytest.raises(ValidationError, match="eight CP2 V1 roots"):
        EvaluationEpisodeContextV1.model_validate(context_payload)

    raw_payload = report.statistics[0].model_dump(mode="python")
    raw_payload["source_schema_versions"] = report.statistics[0].source_schema_versions[
        :-1
    ]
    with pytest.raises(ValidationError, match="exact eight CP2 V1 roots"):
        RawSufficientStatisticV1.model_validate(raw_payload)

    raw_payload = report.statistics[0].model_dump(mode="python")
    raw_payload["processed_transition_count"] = 0
    raw_payload["processing_status"] = EvaluationProcessingStatusV1(
        status="succeeded",
        processed_transition_count=0,
    )
    with pytest.raises(ValidationError, match="equal validated and processed"):
        RawSufficientStatisticV1.model_validate(raw_payload)


@pytest.mark.parametrize(
    "component",
    (
        _UnfrozenCountComponent(count=1, eligible_episode_count=1),
        _PrivateCountComponent(count=1, eligible_episode_count=1),
    ),
    ids=("unfrozen-subtype", "private-state-subtype"),
)
def test_raw_statistic_rejects_mutable_nested_schema_subtypes(
    one_transition_trajectory: CapturedEvaluationTrajectory,
    component: CountComponentV1,
) -> None:
    """Raw-root revalidation cannot retain an undeclared mutable component."""
    trajectory = one_transition_trajectory
    observer = build_evaluation_observer_v1(
        trajectory.context,
        reducers=(_Reducer(),),
    )
    _feed_observer(observer, trajectory.frames, trajectory.transitions)
    report = observer.finalize(completion_state="complete")
    raw_payload = report.statistics[0].model_dump(mode="python")
    raw_payload["component"] = component

    with pytest.raises(ValidationError, match="exact declared schema type"):
        RawSufficientStatisticV1.model_validate(raw_payload)


def test_metric_report_rejects_undeclared_nested_context_model(
    one_transition_trajectory: CapturedEvaluationTrajectory,
) -> None:
    """Direct report revalidation cannot retain a mutable nested identity."""
    trajectory = one_transition_trajectory
    observer = build_evaluation_observer_v1(
        trajectory.context,
        reducers=(_Reducer(),),
    )
    _feed_observer(observer, trajectory.frames, trajectory.transitions)
    report = observer.finalize(completion_state="complete")
    unsafe_identity = _AdversarialEpisodeIdentity.model_validate(
        report.context.identity.model_dump(mode="python")
    )
    unsafe_context = report.context.model_copy(update={"identity": unsafe_identity})
    report_payload = report.model_dump(mode="python")
    report_payload["context"] = unsafe_context

    with pytest.raises(ValidationError, match="undeclared nested model type"):
        EvaluationMetricReportV1.model_validate(report_payload)


def test_package_exports_the_complete_cp3_public_seam() -> None:
    """Consumers can import CP3 contracts from the evaluation package boundary."""
    expected_exports = {
        "CountComponentV1",
        "SumComponentV1",
        "RatioComponentV1",
        "DurationComponentV1",
        "OpportunityComponentV1",
        "DistributionComponentV1",
        "EvaluationEpisodeCompletionV1",
        "EvaluationProcessingStatusV1",
        "RawSufficientStatisticV1",
        "EvaluationMetricReportV1",
        "EvaluationEpisodeObserverV1",
        "build_evaluation_observer_v1",
        "validate_initial_evaluation_frame_v1",
    }

    assert expected_exports.issubset(set(evaluation_package.__all__))
    assert evaluation_package.EvaluationEpisodeObserverV1 is (
        EvaluationEpisodeObserverV1
    )
    assert evaluation_package.build_evaluation_observer_v1 is (
        build_evaluation_observer_v1
    )


def test_simultaneous_terminal_truncation_at_horizon_preserves_all_truth() -> None:
    """Done flags and completion bases are recorded without collapsing evidence."""
    trajectory = captured_evaluation_trajectory(
        transition_count=1,
        expected_horizon=1,
    )
    done_transition = EvaluationTransitionV1.model_validate(
        {
            **trajectory.transitions[0].model_dump(mode="python"),
            "terminated": True,
            "truncated": True,
            "owning_task_end_reason": "simultaneous_done",
        }
    )
    observer = build_evaluation_observer_v1(trajectory.context)
    observer.start(trajectory.frames[0])
    observer.append(done_transition, trajectory.frames[1])

    report = observer.finalize(completion_state="complete")

    assert report.completion.terminated is True
    assert report.completion.truncated is True
    assert report.completion.completion_bases == (
        "task_terminal",
        "declared_horizon",
    )
    assert report.completion.end_or_failure_reason == "simultaneous_done"


def test_processing_gap_downgrades_complete_only_statistic_not_completion(
    one_transition_trajectory: CapturedEvaluationTrajectory,
) -> None:
    """A complete artifact with a reducer gap yields insufficient metric evidence."""
    trajectory = one_transition_trajectory
    observer = build_evaluation_observer_v1(
        trajectory.context,
        reducers=(
            _Reducer(
                fail_advance_index=0,
                draft_builder=_single_draft_builder(
                    CountComponentV1(count=0, eligible_episode_count=1),
                    completion_scope="complete_episode",
                ),
            ),
        ),
    )
    observer.start(trajectory.frames[0])
    with pytest.raises(RuntimeError, match="advance failed"):
        observer.append(trajectory.transitions[0], trajectory.frames[1])

    report = observer.finalize(completion_state="complete")

    assert report.completion.completion_state == "complete"
    assert report.completion.validated_transition_count == 1
    assert report.processing_status.status == "failed"
    assert report.processing_status.processed_transition_count == 0
    assert report.statistics[0].result_status == "insufficient_data"
    assert report.statistics[0].endpoint_observation_status == "unavailable"


def test_context_driven_opportunity_excludes_inactive_padded_slots(
    one_transition_trajectory: CapturedEvaluationTrajectory,
) -> None:
    """Padding is roster truth, not an observer-inferred zero opportunity."""
    trajectory = one_transition_trajectory
    active_slots = tuple(
        row.global_slot for row in trajectory.context.roster if row.configured_active
    )
    inactive_slots = tuple(
        row.global_slot
        for row in trajectory.context.roster
        if not row.configured_active
    )
    assert active_slots == (0, 1, 2, 5, 6)
    assert inactive_slots == (3, 4, 7, 8, 9)
    observer = build_evaluation_observer_v1(
        trajectory.context,
        reducers=(
            _Reducer(
                draft_builder=_single_draft_builder(
                    OpportunityComponentV1(
                        opportunity_count=len(active_slots),
                        eligible_episode_count=1,
                    )
                )
            ),
        ),
    )
    _feed_observer(observer, trajectory.frames, trajectory.transitions)

    report = observer.finalize(completion_state="complete")

    assert report.statistics[0].result_status == "defined"
    assert report.statistics[0].component == OpportunityComponentV1(
        opportunity_count=5,
        eligible_episode_count=1,
    )


def test_reducer_state_rejects_nested_mutable_payload_even_in_frozen_model(
    one_transition_trajectory: CapturedEvaluationTrajectory,
) -> None:
    """Pydantic freezing alone cannot make a nested list valid reducer state."""
    trajectory = one_transition_trajectory
    observer = build_evaluation_observer_v1(
        trajectory.context,
        reducers=(_Reducer(mutable_payload_on_initialize=True),),
    )

    with pytest.raises(RuntimeError, match="initialization failed"):
        observer.start(trajectory.frames[0])

    report = observer.finalize(
        completion_state="failed",
        end_or_failure_reason="invalid reducer state",
        failure_origin="capture",
    )
    assert report.processing_status.status == "failed"
    assert report.processing_status.failure is not None
    assert report.processing_status.failure.stage == "reducer_initialize"
    assert "tuple-backed" in report.processing_status.failure.detail


def test_reducer_state_rejects_hidden_mutable_private_attributes(
    one_transition_trajectory: CapturedEvaluationTrajectory,
) -> None:
    """Private attributes cannot evade replacement-state immutability checks."""
    trajectory = one_transition_trajectory
    observer = build_evaluation_observer_v1(
        trajectory.context,
        reducers=(_Reducer(private_payload_on_initialize=True),),
    )

    with pytest.raises(RuntimeError, match="initialization failed"):
        observer.start(trajectory.frames[0])

    report = observer.finalize(
        completion_state="failed",
        end_or_failure_reason="hidden mutable reducer state",
        failure_origin="capture",
    )
    assert report.processing_status.status == "failed"
    assert report.processing_status.failure is not None
    assert report.processing_status.failure.stage == "reducer_initialize"
    assert "tuple-backed" in report.processing_status.failure.detail


def test_reducer_state_subclass_cannot_weaken_strict_frozen_configuration(
    one_transition_trajectory: CapturedEvaluationTrajectory,
) -> None:
    """Reducer subclasses must preserve the base model's immutable contract."""
    trajectory = one_transition_trajectory
    observer = build_evaluation_observer_v1(
        trajectory.context,
        reducers=(_Reducer(unfrozen_state_on_initialize=True),),
    )

    with pytest.raises(RuntimeError, match="initialization failed"):
        observer.start(trajectory.frames[0])

    report = observer.finalize(
        completion_state="failed",
        end_or_failure_reason="unfrozen reducer state",
        failure_origin="capture",
    )
    assert report.processing_status.failure is not None
    assert report.processing_status.failure.stage == "reducer_initialize"
    assert "strict frozen model config" in report.processing_status.failure.detail


def test_core_reset_and_step_have_no_static_or_runtime_cp3_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ordinary simulator execution remains unaware of opt-in evaluation plumbing."""
    core_directory = Path(core_env_module.__file__).parent
    for source_path in core_directory.glob("*.py"):
        assert "marl_battlegrounds.evaluation" not in source_path.read_text(
            encoding="utf-8"
        )

    config = evaluation_fixtures_module.evaluation_env_config()
    action = evaluation_fixtures_module.neutral_action()
    reset_key = jax.random.PRNGKey(91)
    step_key = jax.random.PRNGKey(92)
    calls: list[str] = []

    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        calls.append("CP3 hook")
        raise AssertionError("ordinary core execution invoked evaluation")

    monkeypatch.setattr(
        capture_module,
        "capture_initial_evaluation_frame_v1",
        forbidden,
    )
    monkeypatch.setattr(
        capture_module,
        "capture_evaluation_transition_unit_v1",
        forbidden,
    )
    monkeypatch.setattr(metrics_module, "build_evaluation_observer_v1", forbidden)
    monkeypatch.setattr(
        validation_module,
        "validate_initial_evaluation_frame_v1",
        forbidden,
    )
    monkeypatch.setattr(
        validation_module,
        "validate_evaluation_transition_unit_v1",
        forbidden,
    )
    monkeypatch.setattr(
        validation_module,
        "decode_evaluation_events_v1",
        forbidden,
    )
    monkeypatch.setattr(jax, "device_get", forbidden)

    state, _observation, action_mask, _info = core_env_module.reset(
        config,
        reset_key,
    )
    core_env_module.step(
        config,
        state,
        action_mask,
        action,
        step_key,
    )

    assert calls == []
