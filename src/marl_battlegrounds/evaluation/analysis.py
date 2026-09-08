"""Offline replay analysis with compact, immutable cursor summaries and CSV.

The captured simulation identity is never rewritten. Analysis has its own source
fingerprint and reducer versions, and the original sidecar remains untouched.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass
from importlib.resources import files
from typing import Literal, cast

from marl_battlegrounds.evaluation.metrics import (
    DistributionComponentV1,
    DistributionObservationV1,
    EvaluationEpisodeObserverV1,
    EvaluationMetricReportV1,
    EvaluationProcessingStatusV1,
    SufficientStatisticDraftV1,
)
from marl_battlegrounds.evaluation.models import AssignedPolicySlotV1
from marl_battlegrounds.evaluation.replay_io import LoadedReplayBundleV1

type MetricScope = Literal["cursor", "final"]

_DYNAMIC_FIELDS = frozenset(
    {
        "component",
        "result_status",
        "status_reason",
        "endpoint_observation_status",
    }
)


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _source_digest() -> str:
    directory = files("marl_battlegrounds.evaluation")
    digest = hashlib.sha256()
    for path in sorted(
        (path for path in directory.iterdir() if path.name.endswith(".py")),
        key=lambda path: path.name,
    ):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class _Value:
    definition: int
    status: str
    reason: str | None
    endpoint: str
    component_json: str
    distribution_count: int | None = None


@dataclass(frozen=True, slots=True)
class ReplayAnalysis:
    """One analysis pass; seeks only project already computed prefix components."""

    source_replay_digest: str
    analysis_source_digest: str
    original_metric_status: Literal["missing", "empty", "available"]
    reducer_versions: tuple[tuple[str, int], ...]
    final_report: EvaluationMetricReportV1
    source_processing_status: EvaluationProcessingStatusV1
    _definitions: tuple[str, ...]
    _prefixes: tuple[tuple[_Value, ...], ...]
    _distributions: tuple[tuple[DistributionObservationV1, ...], ...]

    @property
    def frame_count(self) -> int:
        return len(self._prefixes)

    def _frame(self, frame_index: int, scope: MetricScope) -> int:
        if scope not in ("cursor", "final"):
            raise ValueError("metric scope must be cursor or final")
        if type(frame_index) is not int or not 0 <= frame_index < self.frame_count:
            raise IndexError("metric cursor is outside the captured replay")
        return self.frame_count - 1 if scope == "final" else frame_index

    def summary(
        self, frame_index: int, *, scope: MetricScope = "cursor"
    ) -> dict[str, object]:
        """Return JSON-ready rows; distributions show count/mean/min/max only."""
        selected = self._frame(frame_index, scope)
        rows: list[dict[str, object]] = []
        for value in self._prefixes[selected]:
            row = cast(
                dict[str, object], json.loads(self._definitions[value.definition])
            )
            row.update(
                {
                    "result_status": value.status,
                    "status_reason": value.reason,
                    "endpoint_observation_status": value.endpoint,
                    "component": json.loads(value.component_json),
                }
            )
            component = cast(dict[str, object] | None, row["component"])
            display: object = None
            exposure: object = None
            if component is not None:
                kind = component["component_type"]
                if kind == "ratio":
                    denominator = cast(float, component["denominator"])
                    display = (
                        cast(float, component["numerator"]) / denominator
                        if denominator
                        else None
                    )
                    exposure = denominator
                elif kind == "duration":
                    display = component["qualifying_steps"]
                    exposure = component["eligible_steps"]
                elif kind == "distribution":
                    display = component["mean"]
                    exposure = component["observation_count"]
                else:
                    display = component.get(
                        "value",
                        component.get("count", component.get("opportunity_count")),
                    )
                    exposure = component.get(
                        "observation_count", component.get("eligible_episode_count")
                    )
            row["display_value"] = (
                display if value.status in ("defined", "zero_opportunity") else None
            )
            row["exposure"] = exposure
            rows.append(row)
        revision = self.final_report.context.code_revision
        return {
            "scope": scope,
            "frame_index": selected,
            "captured_transition_count": self.frame_count - 1,
            "episode_id": self.final_report.context.identity.episode_id,
            "source_replay_digest": self.source_replay_digest,
            "analysis_source_digest": self.analysis_source_digest,
            "reducer_versions": self.reducer_versions,
            "original_metric_status": self.original_metric_status,
            "source_processing_status": self.source_processing_status.model_dump(
                mode="json"
            ),
            "simulation_code_revision": revision.model_dump(mode="json"),
            "completion": self.final_report.completion.model_dump(mode="json")
            if selected == self.frame_count - 1
            else None,
            "processing_status": self.final_report.processing_status.model_dump(
                mode="json"
            )
            if selected == self.frame_count - 1
            else None,
            "statistics": rows,
        }

    def csv(self, frame_index: int, *, scope: MetricScope = "cursor") -> str:
        """Export selected-prefix raw values, exposure and long-form samples."""
        selected = self._frame(frame_index, scope)
        output = io.StringIO(newline="")
        columns: tuple[str, ...] = (
            "episode_id",
            "scope",
            "frame_index",
            "metric_id",
            "metric_version",
            "component_name",
            "subject",
            "dimensions",
            "units",
            "amount_stage",
            "result_status",
            "status_reason",
            "endpoint_observation_status",
            "component_type",
            "count",
            "value",
            "observation_count",
            "numerator",
            "denominator",
            "qualifying_steps",
            "eligible_steps",
            "opportunity_count",
            "zero_opportunity_occurrence",
            "eligible_episode_count",
            "source_observation_id",
            "ordinal",
            "reducer_id",
            "reducer_version",
            "source_replay_digest",
            "analysis_source_digest",
            "original_metric_status",
            "simulation_commit_sha",
            "simulation_source_tree_digest",
            "completion_scope",
            "completion_state",
            "completion_reason",
            "completion_bases",
            "processing_status",
            "source_processing_status",
            "simulation_is_dirty",
            "simulation_dirty_patch_digest",
            "team_ids",
            "policy_ids",
            "checkpoint_digests",
        )
        writer: csv.DictWriter[str] = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        context = self.final_report.context
        for value in self._prefixes[selected]:
            definition = cast(
                dict[str, object], json.loads(self._definitions[value.definition])
            )
            subject = cast(dict[str, object], definition["subject"])
            selected_roster = tuple(
                roster
                for roster in context.roster
                if roster.configured_active
                and (
                    "team_id" not in subject
                    or roster.configured_team_id == subject["team_id"]
                )
                and (
                    "class_id" not in subject or roster.class_id == subject["class_id"]
                )
                and (
                    "global_slot" not in subject
                    or roster.global_slot == subject["global_slot"]
                )
                and (
                    subject["subject_type"] != "agent_pair"
                    or roster.global_slot
                    in (
                        subject["primary_global_slot"],
                        subject["secondary_global_slot"],
                    )
                )
            )
            slots = {roster.global_slot for roster in selected_roster}
            assignments = tuple(
                policy
                for policy in context.policy_assignments
                if isinstance(policy, AssignedPolicySlotV1)
                and policy.global_slot in slots
            )
            row = {
                key: definition.get(key)
                for key in (
                    "metric_id",
                    "metric_version",
                    "component_name",
                    "units",
                    "amount_stage",
                    "reducer_id",
                    "reducer_version",
                    "completion_scope",
                )
            }
            row.update(
                {
                    "episode_id": context.identity.episode_id,
                    "scope": scope,
                    "frame_index": selected,
                    "completion_state": (
                        self.final_report.completion.completion_state
                        if selected == self.frame_count - 1
                        else "partial"
                    ),
                    "completion_reason": (
                        self.final_report.completion.end_or_failure_reason
                        if selected == self.frame_count - 1
                        else "cursor_prefix"
                    ),
                    "completion_bases": _json(
                        self.final_report.completion.completion_bases
                        if selected == self.frame_count - 1
                        else ()
                    ),
                    "processing_status": (
                        self.final_report.processing_status.model_dump_json()
                        if selected == self.frame_count - 1
                        else "prefix"
                    ),
                    "source_processing_status": (
                        self.source_processing_status.model_dump_json()
                    ),
                    "simulation_is_dirty": context.code_revision.is_dirty,
                    "simulation_dirty_patch_digest": (
                        context.code_revision.dirty_patch_digest
                    ),
                    "team_ids": _json(
                        tuple(
                            sorted(
                                {
                                    roster.configured_team_id
                                    for roster in selected_roster
                                }
                            )
                        )
                    ),
                    "policy_ids": _json(
                        tuple(dict.fromkeys(policy.policy_id for policy in assignments))
                    ),
                    "checkpoint_digests": _json(
                        tuple(
                            dict.fromkeys(
                                policy.checkpoint_digest
                                for policy in assignments
                                if policy.checkpoint_digest is not None
                            )
                        )
                    ),
                    "subject": _json(definition["subject"]),
                    "dimensions": _json(definition["dimensions"]),
                    "result_status": value.status,
                    "status_reason": value.reason,
                    "endpoint_observation_status": value.endpoint,
                    "source_replay_digest": self.source_replay_digest,
                    "analysis_source_digest": self.analysis_source_digest,
                    "original_metric_status": self.original_metric_status,
                    "simulation_commit_sha": context.code_revision.commit_sha,
                    "simulation_source_tree_digest": (
                        context.code_revision.source_tree_digest
                    ),
                }
            )
            component = cast(dict[str, object] | None, json.loads(value.component_json))
            if component is not None:
                row.update(
                    {key: item for key, item in component.items() if key in columns}
                )
            if value.distribution_count:
                for observation in self._distributions[value.definition][
                    : value.distribution_count
                ]:
                    writer.writerow({**row, **observation.model_dump(mode="python")})
            else:
                writer.writerow(row)
        return output.getvalue()


def analyze_replay(
    bundle: LoadedReplayBundleV1, *, full: bool = False
) -> ReplayAnalysis:
    """Analyze a validated replay once, without simulation or writes to its files.

    Critical outcome/return/length/completion metrics are the default. Pass
    ``full=True`` to opt into the complete tactical diagnostic suite.
    Memory grows with scalar prefixes and newly observed distribution samples,
    never with a copy of every accumulated sample at every cursor.
    """
    from marl_battlegrounds.evaluation.reducers import build_tdm_metric_reducers

    replay = bundle.replay
    analysis_source_digest = _source_digest()
    reducers = build_tdm_metric_reducers(full=full)
    observer = EvaluationEpisodeObserverV1(replay.header.context, reducers)
    definitions: list[str] = []
    definition_indices: dict[str, int] = {}
    distributions: list[list[DistributionObservationV1]] = []
    distribution_summaries: list[tuple[float, float | None, float | None]] = []
    prefixes: list[tuple[_Value, ...]] = []
    previous: dict[int, _Value] = {}

    def snapshot(frame_index: int) -> None:
        final = frame_index == len(replay.transitions)
        completion = replay.completion
        drafts = observer.preview_statistics(
            completion_state=completion.completion_state if final else "partial",
            end_or_failure_reason=completion.end_or_failure_reason
            if final
            else "cursor_prefix",
            failure_origin=completion.failure_origin if final else None,
        )
        values: list[_Value] = []
        for draft in drafts:
            definition = _json(
                draft.model_dump(mode="json", exclude=set(_DYNAMIC_FIELDS))
            )
            index = definition_indices.get(definition)
            if index is None:
                index = len(definitions)
                definitions.append(definition)
                definition_indices[definition] = index
                distributions.append([])
                distribution_summaries.append((0.0, None, None))
            component = draft.component
            count: int | None = None
            if isinstance(component, DistributionComponentV1):
                stream = distributions[index]
                observations = component.observations
                if (
                    len(observations) < len(stream)
                    or (stream and observations[len(stream) - 1] != stream[-1])
                    or (final and tuple(stream) != observations[: len(stream)])
                ):
                    raise ValueError(
                        "cursor distributions must preserve their observed prefix"
                    )
                total, minimum, maximum = distribution_summaries[index]
                for observation in observations[len(stream) :]:
                    if observation.ordinal != len(stream):
                        raise ValueError(
                            "cursor distribution ordinals must be contiguous"
                        )
                    total += observation.value
                    minimum = (
                        observation.value
                        if minimum is None
                        else min(minimum, observation.value)
                    )
                    maximum = (
                        observation.value
                        if maximum is None
                        else max(maximum, observation.value)
                    )
                    stream.append(observation)
                distribution_summaries[index] = (total, minimum, maximum)
                count = len(stream)
                component_json = _json(
                    {
                        "component_type": "distribution",
                        "observation_count": count,
                        "mean": total / count if count else None,
                        "minimum": minimum,
                        "maximum": maximum,
                        "eligible_episode_count": component.eligible_episode_count,
                    }
                )
            else:
                component_json = (
                    "null" if component is None else component.model_dump_json()
                )
            value = _Value(
                index,
                draft.result_status,
                draft.status_reason,
                draft.endpoint_observation_status,
                component_json,
                count,
            )
            if previous.get(index) == value:
                value = previous[index]
            previous[index] = value
            values.append(value)
        prefixes.append(tuple(values))

    observer.start(replay.frames[0])
    snapshot(0)
    for transition, successor in zip(
        replay.transitions, replay.frames[1:], strict=True
    ):
        observer.append(transition, successor)
        snapshot(successor.frame_index)
    completion = replay.completion
    final_report = observer.finalize(
        completion_state=completion.completion_state,
        end_or_failure_reason=completion.end_or_failure_reason,
        failure_origin=completion.failure_origin,
    )
    if final_report.processing_status.status != "succeeded":
        raise ValueError("failed metric processing cannot publish a cursor index")
    final_values = {value.definition: value for value in prefixes[-1]}
    if len(final_values) != len(final_report.statistics):
        raise ValueError("projected final rows differ from the final report")
    for raw in final_report.statistics:
        draft = SufficientStatisticDraftV1.model_construct(
            **{
                name: getattr(raw, name)
                for name in SufficientStatisticDraftV1.model_fields
                if name not in ("schema_id", "schema_version")
            }
        )
        definition = _json(draft.model_dump(mode="json", exclude=set(_DYNAMIC_FIELDS)))
        index = definition_indices.get(definition)
        value = None if index is None else final_values.get(index)
        if value is None or (value.status, value.reason, value.endpoint) != (
            raw.result_status,
            raw.status_reason,
            raw.endpoint_observation_status,
        ):
            raise ValueError("projected final semantics differ from the final report")
        component = raw.component
        if isinstance(component, DistributionComponentV1):
            summary = json.loads(value.component_json)
            if (
                tuple(distributions[value.definition]) != component.observations
                or value.distribution_count != len(component.observations)
                or summary["observation_count"] != len(component.observations)
                or summary["eligible_episode_count"] != component.eligible_episode_count
            ):
                raise ValueError("projected distribution differs from the final report")
        elif json.loads(value.component_json) != (
            None if component is None else component.model_dump(mode="json")
        ):
            raise ValueError("projected component differs from the final report")
    if _source_digest() != analysis_source_digest:
        raise RuntimeError("evaluation source changed during replay analysis")
    original = bundle.metric_report_artifact
    return ReplayAnalysis(
        source_replay_digest=replay.canonical_digest_sha256,
        analysis_source_digest=analysis_source_digest,
        original_metric_status=(
            "missing"
            if original is None
            else "available"
            if original.report.statistics
            else "empty"
        ),
        reducer_versions=tuple(
            (reducer.reducer_id, reducer.reducer_version) for reducer in reducers
        ),
        final_report=final_report,
        source_processing_status=replay.processing_status,
        _definitions=tuple(definitions),
        _prefixes=tuple(prefixes),
        _distributions=tuple(tuple(stream) for stream in distributions),
    )
