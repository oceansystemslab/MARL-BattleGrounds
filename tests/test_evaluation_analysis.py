"""Offline analysis preserves cursor scope, raw exports and artifact identity."""

from __future__ import annotations

import csv
import io
import json

import pytest
from tests.evaluation_fixtures import captured_evaluation_trajectory

from marl_battlegrounds.evaluation.analysis import analyze_replay
from marl_battlegrounds.evaluation.metrics import (
    CompletionState,
    CountComponentV1,
    EvaluationEpisodeObserverV1,
    RolloutFailureOrigin,
    SufficientStatisticDraftV1,
)
from marl_battlegrounds.evaluation.reducers import build_tdm_metric_reducers
from marl_battlegrounds.evaluation.replay import (
    RuntimeProvenanceV1,
    build_replay_bundle_v1,
)
from marl_battlegrounds.evaluation.replay_io import LoadedReplayBundleV1


@pytest.fixture
def runtime_provenance() -> RuntimeProvenanceV1:
    return RuntimeProvenanceV1(
        python_version="3.14.0",
        package_version="0.0.0",
        jax_version="0.10.1",
        jaxlib_version="0.10.1",
        numpy_version="2.3.0",
        pydantic_version="2.11.0",
        platform="linux",
        machine="x86_64",
        backend="cpu",
        device="generic-cpu",
        precision="float32",
        environment_count=1,
        batch_shape=(1,),
        policy_execution_included=False,
    )


def test_replay_analysis_matches_final_report_and_prefix_csv(
    runtime_provenance: RuntimeProvenanceV1,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory = captured_evaluation_trajectory(transition_count=2, expected_horizon=2)
    capture = EvaluationEpisodeObserverV1(trajectory.context)
    direct = EvaluationEpisodeObserverV1(
        trajectory.context, build_tdm_metric_reducers(full=True)
    )
    for observer in (capture, direct):
        observer.start(trajectory.frames[0])
        for transition, frame in zip(
            trajectory.transitions, trajectory.frames[1:], strict=True
        ):
            observer.append(transition, frame)
    original = capture.finalize(completion_state="complete")
    bundle = build_replay_bundle_v1(
        capture, original, runtime_provenance=runtime_provenance
    )
    loaded = LoadedReplayBundleV1(
        bundle.replay, bundle.metric_report_artifact, "complete"
    )
    before = bundle.metric_report_artifact.model_dump_json()
    analysis = analyze_replay(loaded, full=True)
    assert analysis.source_replay_digest == bundle.replay.canonical_digest_sha256
    assert analysis.source_processing_status == bundle.replay.processing_status
    assert analysis.original_metric_status == "empty"
    assert analysis.final_report == direct.finalize(completion_state="complete")
    assert bundle.metric_report_artifact.model_dump_json() == before
    assert analysis.summary(0)["completion"] is None
    assert analysis.summary(1)["completion"] is None
    assert analysis.summary(0, scope="final")["frame_index"] == 2
    first = analysis.summary(0)
    for index in (2, 1, 0, 2, 0):
        assert analysis.summary(index)["frame_index"] == index
    assert analysis.summary(0) == first
    for index in range(3):
        rows = list(csv.DictReader(io.StringIO(analysis.csv(index))))
        assert rows
        assert {row["frame_index"] for row in rows} == {str(index)}
        assert {row["scope"] for row in rows} == {"cursor"}
        assert {row["source_replay_digest"] for row in rows} == {
            analysis.source_replay_digest
        }
        assert all(isinstance(json.loads(row["subject"]), dict) for row in rows)
        assert {row["source_processing_status"] for row in rows} == {
            analysis.source_processing_status.model_dump_json()
        }
        assert {row["completion_state"] for row in rows} == {
            "complete" if index == 2 else "partial"
        }
        if index == 0:
            assert all(row["ordinal"] == "" for row in rows)
    with pytest.raises(IndexError):
        analysis.summary(3)
    with pytest.raises(ValueError):
        analysis.summary(0, scope="invalid")  # pyright: ignore[reportArgumentType]

    basic = analyze_replay(loaded)
    basic_ids = {
        "marlbg.task.outcome_distribution.v1",
        "marlbg.task.terminal_score_differential.v1",
        "marlbg.task.evaluation_return.v1",
        "marlbg.task.episode_length.v1",
        "marlbg.artifact.completion.v1",
    }
    assert {row.metric_id for row in basic.final_report.statistics} == basic_ids
    assert basic.final_report.statistics == tuple(
        row for row in analysis.final_report.statistics if row.metric_id in basic_ids
    )

    original_project = EvaluationEpisodeObserverV1.preview_statistics

    def corrupted_projection(
        self: EvaluationEpisodeObserverV1,
        *,
        completion_state: CompletionState = "partial",
        end_or_failure_reason: str | None = None,
        failure_origin: RolloutFailureOrigin | None = None,
    ) -> tuple[SufficientStatisticDraftV1, ...]:
        rows = list(
            original_project(
                self,
                completion_state=completion_state,
                end_or_failure_reason=end_or_failure_reason,
                failure_origin=failure_origin,
            )
        )
        # A valid count with the wrong value must not bypass the final evidence gate.
        for index, row in enumerate(rows):
            if isinstance(row.component, CountComponentV1):
                component = row.component.model_copy(
                    update={"count": row.component.count + 1}
                )
                rows[index] = row.model_copy(update={"component": component})
                break
        return tuple(rows)

    monkeypatch.setattr(
        EvaluationEpisodeObserverV1, "preview_statistics", corrupted_projection
    )
    with pytest.raises(ValueError, match="projected component differs"):
        analyze_replay(loaded, full=True)
