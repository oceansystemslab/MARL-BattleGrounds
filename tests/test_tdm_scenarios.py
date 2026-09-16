"""Check all eight packaged TDM scenarios through their replay records."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from scripts.dev.qualify_tdm_scenarios import (
    QualificationEpisode,
    capture_tdm_qualification_episode,
)

from marl_battlegrounds.evaluation import recording_context
from marl_battlegrounds.evaluation.metric_catalog import FULL_METRIC_NAMES
from marl_battlegrounds.evaluation.models import (
    AggregationKeyV1,
    AssignedPolicySlotV2,
    ContentAddressedIdentityV1,
    EvaluationEpisodeContextV3,
    canonical_digest_sha256,
    canonical_json_bytes,
)
from marl_battlegrounds.evaluation.replay_io import (
    load_replay,
    load_scenario_evaluation_record_v4,
    save_replay,
    save_scenario_evaluation_record_v4,
)
from marl_battlegrounds.evaluation.replay_v3 import ReplayArtifactV3, build_replay_v3
from marl_battlegrounds.evaluation.run_writer import IDENTITY_COLUMNS
from marl_battlegrounds.evaluation.scenario import ResolvedScenarioSpecificationV3
from marl_battlegrounds.evaluation.tdm_scenarios import (
    TDM_SCENARIO_PUBLIC_AGENT_IDS,
    build_tdm_qualification_seed_schedule,
    build_tdm_scenario_evaluation_record,
    tdm_scenario_pressure_identity,
)
from marl_battlegrounds.tasks import list_tdm_scenarios


@pytest.fixture(scope="module")
def suite() -> Iterator[tuple[QualificationEpisode, ...]]:
    # Freeze actual discovered provenance to represent a fixed candidate even
    # while another developer edits unrelated files during the test process.
    import importlib

    execution = importlib.import_module("marl_battlegrounds.evaluation.evaluate")
    provenance = recording_context.capture_recording_provenance(num_envs=1)

    def fixed_provenance(**_: object) -> dict[str, object]:
        return provenance

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(execution, "capture_recording_provenance", fixed_provenance)
        yield tuple(
            capture_tdm_qualification_episode(scenario_id, 0)
            for scenario_id in range(1, 9)
        )


@pytest.mark.parametrize("scenario_id", range(1, 9))
def test_all_eight_scenarios_join_packaged_content_and_roundtrip_evidence(
    suite: tuple[QualificationEpisode, ...],
    scenario_id: int,
    tmp_path: Path,
) -> None:
    evidence = suite[scenario_id - 1]
    specification, replay, record = (
        evidence.specification,
        evidence.replay,
        evidence.record,
    )
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
    assert replay.completion.completion_state == "complete"
    assert tuple(evidence.full_metrics) == (*IDENTITY_COLUMNS, *FULL_METRIC_NAMES)
    assert all(value.shape == (1,) for value in evidence.full_metrics.values())
    assert record.measurement_results[0].result_status == "defined"
    assert record.measurement_results[0].endpoint_observation_status == "observed"
    assert (
        tuple(row.public_agent_id for row in replay.header.context.roster)
        == TDM_SCENARIO_PUBLIC_AGENT_IDS
    )
    assert all(
        isinstance(row, AssignedPolicySlotV2)
        and row.callable_name is not None
        and row.policy_content_digest is not None
        for row in replay.header.context.policy_assignments
    )
    replay_path = tmp_path / "episode.marlbg-replay.json"
    record_path = tmp_path / "episode.marlbg-scenario.json"
    save_replay(replay, replay_path)
    save_scenario_evaluation_record_v4(record, replay, record_path)
    reloaded = load_replay(replay_path).replay
    assert isinstance(reloaded, ReplayArtifactV3)
    assert reloaded == replay
    assert (
        load_scenario_evaluation_record_v4(record_path, source_replay=reloaded)
        == record
    )
    assert not list(tmp_path.glob("*.marlbg-metrics.json"))
    assert "metric_report_reference" not in record.model_dump()


def test_schedule_coordinates_are_unique_and_retry_reproduces_exact_evidence(
    suite: tuple[QualificationEpisode, ...],
) -> None:
    schedule = build_tdm_qualification_seed_schedule()
    assert (
        len({canonical_digest_sha256(row) for row in schedule.realized_seed_protocols})
        == 2
    )
    retried = capture_tdm_qualification_episode(1, 0)
    original = suite[0]
    assert retried.specification == original.specification
    assert canonical_json_bytes(retried.replay) == canonical_json_bytes(original.replay)
    assert canonical_json_bytes(retried.record) == canonical_json_bytes(original.record)
    for name, values in retried.full_metrics.items():
        np.testing.assert_array_equal(values, original.full_metrics[name])
    other = capture_tdm_qualification_episode(1, 1)
    assert other.specification == retried.specification
    assert other.record.schedule_coordinate == 1
    assert (
        other.record.canonical_digest_sha256 != retried.record.canonical_digest_sha256
    )
    seeds = retried.replay.header.context.seed_protocol
    assert seeds.seed_protocol.identifier == "episode-fold-in-v1"
    assert seeds.root_seed == 0 and seeds.episode_seed == 0
    assert seeds.environment_seed is seeds.focal_policy_seed is None


@pytest.mark.parametrize("stop_after", (0, 1))
def test_interrupted_capture_cannot_become_draw_or_right_censored_success(
    suite: tuple[QualificationEpisode, ...],
    stop_after: int,
) -> None:
    evidence = capture_tdm_qualification_episode(1, 0, stop_after=stop_after)
    assert evidence.replay.completion.completion_state == "partial"
    assert len(evidence.replay.transitions) == stop_after
    assert evidence.replay.frames == suite[0].replay.frames[: stop_after + 1]
    assert evidence.replay.transitions == suite[0].replay.transitions[:stop_after]
    assert evidence.full_metrics == {}
    endpoint = evidence.record.measurement_results[0]
    assert endpoint.result_status == "unavailable"
    assert endpoint.endpoint_observation_status == "unavailable"
    assert endpoint.value is None
    assert evidence.record.predicate_result.status == "unavailable"


@pytest.mark.parametrize(
    "field", ("resolved_config_digest_sha256", "horizon", "hypothesis")
)
def test_resealed_changed_definition_cannot_enter_official_tdm_results(
    suite: tuple[QualificationEpisode, ...],
    field: str,
) -> None:
    evidence = suite[0]
    payload = evidence.specification.model_dump(
        mode="python", exclude={"canonical_digest_sha256"}
    )
    payload[field] = (
        6
        if field == "horizon"
        else "A different scientific claim."
        if field == "hypothesis"
        else "0" * 64
    )
    changed = ResolvedScenarioSpecificationV3.model_validate(
        {**payload, "canonical_digest_sha256": canonical_digest_sha256(payload)}
    )
    with pytest.raises(ValueError, match="approved TDM definition"):
        build_tdm_scenario_evaluation_record(
            1, changed, evidence.replay, schedule_coordinate=0
        )


def _replace_context(
    replay: ReplayArtifactV3, context: EvaluationEpisodeContextV3
) -> ReplayArtifactV3:
    return build_replay_v3(
        context,
        replay.frames,
        replay.transitions,
        runtime_provenance=replay.header.runtime_provenance,
        wrapper_stack=replay.header.wrapper_stack,
        completion_state="complete",
    )


def test_changed_opponent_content_is_rejected_even_with_valid_replay_joins(
    suite: tuple[QualificationEpisode, ...],
) -> None:
    evidence = suite[0]
    context = evidence.replay.header.context
    keys = list(context.aggregation_keys)
    for index, row in enumerate(keys):
        if row.name == "team_b_controller_identity":
            pressure = ContentAddressedIdentityV1.model_validate_json(row.value)
            changed_pressure = pressure.model_copy(
                update={"canonical_digest": "0" * 64}
            )
            keys[index] = AggregationKeyV1(
                name=row.name,
                value=canonical_json_bytes(changed_pressure).decode("ascii"),
            )
    changed_context = EvaluationEpisodeContextV3.model_validate(
        {**context.model_dump(mode="python"), "aggregation_keys": tuple(keys)}
    )
    changed = _replace_context(evidence.replay, changed_context)
    with pytest.raises(ValueError, match="frozen pressure rules"):
        build_tdm_scenario_evaluation_record(
            1, evidence.specification, changed, schedule_coordinate=0
        )


def test_wrong_seed_coordinate_is_rejected(
    suite: tuple[QualificationEpisode, ...],
) -> None:
    evidence = suite[0]
    with pytest.raises(ValueError, match="schedule coordinate"):
        build_tdm_scenario_evaluation_record(
            1, evidence.specification, evidence.replay, schedule_coordinate=1
        )


@pytest.mark.parametrize("envelope", ("schedule", "specification", "record"))
def test_current_evidence_versions_reject_coercion_and_historical_mixing(
    suite: tuple[QualificationEpisode, ...],
    envelope: str,
) -> None:
    evidence = suite[0]
    original = (
        evidence.specification.seed_schedule
        if envelope == "schedule"
        else evidence.specification
        if envelope == "specification"
        else evidence.record
    )
    expected = original.schema_version
    for version in (True, str(expected), float(expected), expected - 1):
        payload = original.model_dump(
            mode="python", exclude={"canonical_digest_sha256"}
        )
        payload["schema_version"] = version
        payload["canonical_digest_sha256"] = canonical_digest_sha256(payload)
        with pytest.raises(ValueError, match=f"exact integer {expected}"):
            type(original).model_validate(payload)


def test_changed_policy_role_cannot_join_frozen_scenario() -> None:
    evidence = capture_tdm_qualification_episode(1, 0)
    context = evidence.replay.header.context
    assignments = list(context.policy_assignments)
    assignment = assignments[1]
    assert isinstance(assignment, AssignedPolicySlotV2)
    assignments[1] = assignment.model_copy(
        update={"evaluation_role": "cooperative_partner"}
    )
    changed_context = EvaluationEpisodeContextV3.model_validate(
        {
            **context.model_dump(mode="python"),
            "policy_assignments": tuple(assignments),
            "seed_protocol": {
                **context.seed_protocol.model_dump(mode="python"),
                "cooperative_partner_seed": None,
            },
        }
    )
    changed = _replace_context(evidence.replay, changed_context)
    with pytest.raises(ValueError, match="assigned policy role"):
        build_tdm_scenario_evaluation_record(
            1, evidence.specification, changed, schedule_coordinate=0
        )
