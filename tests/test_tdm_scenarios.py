"""All-eight package-to-replay joins for the approved TDM scenario suite."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from scripts.dev.qualify_tdm_scenarios import capture_tdm_qualification_episode
from scripts.dev.visual_debugger.revision import discover_debugger_code_revision_v1
from scripts.dev.visual_debugger.runtime_provenance import (
    capture_debugger_runtime_provenance_v1,
)

from marl_battlegrounds.evaluation.metrics import build_evaluation_observer_v1
from marl_battlegrounds.evaluation.models import (
    AssignedPolicySlotV1,
    CodeRevisionV1,
    EvaluationEpisodeContextV1,
    canonical_digest_sha256,
)
from marl_battlegrounds.evaluation.replay import (
    ReplayBundleV1,
    RuntimeProvenanceV1,
    build_replay_bundle_v1,
)
from marl_battlegrounds.evaluation.replay_io import (
    load_replay_bundle_v1,
    load_scenario_evaluation_record_v2,
    save_replay_bundle_v1,
    save_scenario_evaluation_record_v2,
)
from marl_battlegrounds.evaluation.scenario import (
    ResolvedScenarioSpecificationV2,
    ScenarioEvaluationRecordV2,
)
from marl_battlegrounds.evaluation.tdm_scenarios import (
    build_tdm_qualification_seed_schedule,
    build_tdm_scenario_evaluation_record,
    tdm_scenario_pressure_identity,
)
from marl_battlegrounds.tasks import list_tdm_scenarios

type _Evidence = tuple[
    ResolvedScenarioSpecificationV2, ReplayBundleV1, ScenarioEvaluationRecordV2
]


@dataclass(frozen=True)
class _Suite:
    revision: CodeRevisionV1
    runtime: RuntimeProvenanceV1
    evidence: tuple[_Evidence, ...]


@pytest.fixture(scope="module")
def suite() -> _Suite:
    revision = discover_debugger_code_revision_v1(Path(__file__).resolve().parents[1])
    runtime = capture_debugger_runtime_provenance_v1(
        revision, policy_execution_included=True
    )
    return _Suite(
        revision,
        runtime,
        tuple(
            capture_tdm_qualification_episode(
                scenario_id,
                0,
                code_revision=revision,
                runtime_provenance=runtime,
            )
            for scenario_id in range(1, 9)
        ),
    )


@pytest.mark.parametrize("scenario_id", range(1, 9))
def test_all_eight_scenarios_join_packaged_content_and_roundtrip_evidence(
    suite: _Suite, scenario_id: int, tmp_path: Path
) -> None:
    specification, bundle, record = suite.evidence[scenario_id - 1]
    source = list_tdm_scenarios()[scenario_id - 1]
    assert (
        specification.resolved_config_digest_sha256
        == source.resolved_configuration_digest
    )
    assert (
        specification.authored_initial_condition.canonical_digest
        == source.source.semantic_digest
    )
    assert specification.horizon == (10 if scenario_id == 3 else 5)
    assert specification.role_template == ("focal",) * 5 + ("adversarial_opponent",) * 5
    assert specification.pressure_protocol == tdm_scenario_pressure_identity(
        scenario_id
    )
    assert specification.pressure_protocol is not None
    assert specification.pressure_protocol.version == (
        4 if scenario_id in (3, 5, 8) else 2
    )
    assert bundle.replay.completion.completion_state == "complete"
    assert bundle.metric_report_artifact.report.statistics
    assert record.measurement_results[0].result_status == "defined"
    assert record.measurement_results[0].endpoint_observation_status == "observed"
    replay_path = tmp_path / "episode.marlbg-replay.json"
    record_path = tmp_path / "episode.marlbg-scenario.json"
    save_replay_bundle_v1(bundle, replay_path)
    save_scenario_evaluation_record_v2(
        record, bundle.replay, bundle.metric_report_artifact, record_path
    )
    reloaded = load_replay_bundle_v1(replay_path, require_metric_report=True)
    assert reloaded.metric_report_artifact is not None
    assert (
        load_scenario_evaluation_record_v2(
            record_path,
            source_replay=reloaded.replay,
            metric_report_artifact=reloaded.metric_report_artifact,
        )
        == record
    )


def test_schedule_coordinates_are_unique_and_retry_reproduces_exact_evidence(
    suite: _Suite,
) -> None:
    schedule = build_tdm_qualification_seed_schedule()
    digests = tuple(
        canonical_digest_sha256(row) for row in schedule.realized_seed_protocols
    )
    assert len(set(digests)) == 2
    retried = capture_tdm_qualification_episode(
        1, 0, code_revision=suite.revision, runtime_provenance=suite.runtime
    )
    assert retried == suite.evidence[0]
    other_coordinate = capture_tdm_qualification_episode(
        1, 1, code_revision=suite.revision, runtime_provenance=suite.runtime
    )
    assert other_coordinate[0] == retried[0]
    assert other_coordinate[2].schedule_coordinate == 1
    assert (
        other_coordinate[2].canonical_digest_sha256
        != retried[2].canonical_digest_sha256
    )


@pytest.mark.parametrize("stop_after", (0, 1))
def test_interrupted_capture_cannot_become_draw_or_right_censored_success(
    suite: _Suite, stop_after: int
) -> None:
    _, bundle, record = capture_tdm_qualification_episode(
        1,
        0,
        code_revision=suite.revision,
        runtime_provenance=suite.runtime,
        stop_after=stop_after,
    )
    assert bundle.replay.completion.completion_state == "partial"
    assert len(bundle.replay.transitions) == stop_after
    endpoint = record.measurement_results[0]
    assert endpoint.result_status == "unavailable"
    assert endpoint.endpoint_observation_status == "unavailable"
    assert endpoint.value is None
    assert record.predicate_result.status == "unavailable"


@pytest.mark.parametrize(
    "field", ("resolved_config_digest_sha256", "horizon", "hypothesis")
)
def test_resealed_changed_definition_cannot_enter_official_tdm_results(
    suite: _Suite, field: str
) -> None:
    specification, bundle, _ = suite.evidence[0]
    payload = specification.model_dump(
        mode="python", exclude={"canonical_digest_sha256"}
    )
    if field == "horizon":
        payload[field] = 6
    elif field == "hypothesis":
        payload[field] = "A different scientific claim."
    else:
        payload[field] = "0" * 64
    changed = ResolvedScenarioSpecificationV2.model_validate(
        {**payload, "canonical_digest_sha256": canonical_digest_sha256(payload)}
    )
    with pytest.raises(ValueError, match="approved TDM definition"):
        build_tdm_scenario_evaluation_record(
            1,
            changed,
            bundle.replay,
            bundle.metric_report_artifact,
            schedule_coordinate=0,
        )


def _replace_context(
    bundle: ReplayBundleV1, context: EvaluationEpisodeContextV1
) -> ReplayBundleV1:
    observer = build_evaluation_observer_v1(context)
    observer.start(bundle.replay.frames[0])
    for transition, frame in zip(
        bundle.replay.transitions, bundle.replay.frames[1:], strict=True
    ):
        observer.append(transition, frame)
    report = observer.finalize(completion_state="complete")
    return build_replay_bundle_v1(
        observer, report, runtime_provenance=bundle.replay.header.runtime_provenance
    )


def test_changed_opponent_content_is_rejected_even_with_valid_replay_joins(
    suite: _Suite,
) -> None:
    specification, bundle, _ = suite.evidence[0]
    context = bundle.replay.header.context
    assignments = list(context.policy_assignments)
    assignment = assignments[5]
    assert isinstance(assignment, AssignedPolicySlotV1)
    assignments[5] = AssignedPolicySlotV1.model_validate(
        {**assignment.model_dump(mode="python"), "policy_content_digest": "0" * 64}
    )
    context = EvaluationEpisodeContextV1.model_validate(
        {**context.model_dump(mode="python"), "policy_assignments": tuple(assignments)}
    )
    changed = _replace_context(bundle, context)
    with pytest.raises(ValueError, match="frozen pressure rules"):
        build_tdm_scenario_evaluation_record(
            1,
            specification,
            changed.replay,
            changed.metric_report_artifact,
            schedule_coordinate=0,
        )


def test_wrong_seed_coordinate_is_rejected(suite: _Suite) -> None:
    specification, bundle, _ = suite.evidence[0]
    with pytest.raises(ValueError, match="schedule coordinate"):
        build_tdm_scenario_evaluation_record(
            1,
            specification,
            bundle.replay,
            bundle.metric_report_artifact,
            schedule_coordinate=1,
        )
