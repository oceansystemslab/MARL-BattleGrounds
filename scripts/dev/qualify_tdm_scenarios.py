"""Qualify all eight packaged scenarios through existing rollout and V2 evidence APIs.

ALPHA controls Team A solely to exercise the pipeline. These results are neither
winning witnesses nor learned-policy or manuscript evidence. No policy is trained.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import jax
import numpy as np
from jax import Array

from marl_battlegrounds.core.env import initialize_scenario_state
from marl_battlegrounds.evaluation.actor_projection import (
    SHARED_OBS_ACTOR_PROJECTION_V1,
)
from marl_battlegrounds.evaluation.capture import (
    capture_evaluation_transition_unit_v1,
    capture_initial_evaluation_frame_v1,
)
from marl_battlegrounds.evaluation.catalog import build_evaluation_episode_context_v1
from marl_battlegrounds.evaluation.metrics import build_evaluation_observer_v1
from marl_battlegrounds.evaluation.models import (
    AggregationKeyV1,
    AssignedPolicySlotV1,
    CodeRevisionV1,
    ContentAddressedIdentityV1,
    EvaluationEpisodeContextV1,
    EvaluationEpisodeIdentityV1,
    VersionedIdentityV1,
    canonical_digest_sha256,
)
from marl_battlegrounds.evaluation.reducers import (
    TDM_EPISODE_METRIC_IDS,
    build_tdm_metric_reducers,
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
from marl_battlegrounds.evaluation.rollout import (
    ReferenceSuccessorHistory,
    build_rollout_information_availability,
    rollout,
)
from marl_battlegrounds.evaluation.scenario import (
    ResolvedScenarioSpecificationV2,
    ScenarioEvaluationRecordV2,
    ScenarioSeedScheduleV2,
)
from marl_battlegrounds.evaluation.tdm_scenarios import (
    TDM_BETA_SCENARIO_IDS,
    TDM_SCENARIO_PUBLIC_AGENT_IDS,
    build_tdm_qualification_seed_schedule,
    build_tdm_scenario_evaluation_record,
    build_tdm_scenario_specification,
    tdm_scenario_pressure_identity,
)
from marl_battlegrounds.policies.reactive_tdm_alpha import (
    reactive_tdm_alpha_controller_descriptor,
    reactive_tdm_alpha_policy,
)
from marl_battlegrounds.policies.reactive_tdm_beta import reactive_tdm_beta_policy
from marl_battlegrounds.tasks import TDMScenario, list_tdm_scenarios, load_tdm_scenario
from scripts.dev.visual_debugger.revision import discover_debugger_code_revision_v1
from scripts.dev.visual_debugger.runtime_provenance import (
    capture_debugger_runtime_provenance_v1,
)


def _identity(
    identifier: str, content: dict[str, object]
) -> ContentAddressedIdentityV1:
    return ContentAddressedIdentityV1(
        identifier=identifier,
        version=1,
        canonical_digest=canonical_digest_sha256(content),
    )


def _qualification_context(
    scenario: TDMScenario,
    schedule: ScenarioSeedScheduleV2,
    coordinate: int,
    code_revision: CodeRevisionV1,
) -> EvaluationEpisodeContextV1:
    info = scenario.info
    pressure = tdm_scenario_pressure_identity(info.scenario_id)
    alpha = _identity(
        "reactive-team-deathmatch-controller",
        reactive_tdm_alpha_controller_descriptor(),
    )
    experiment = _identity(
        "tdm-closeout-plumbing",
        {"schedule": schedule, "assets": list_tdm_scenarios(), "focal_policy": alpha},
    )
    episode_id = f"tdm-scenario-{info.scenario_id}-coordinate-{coordinate}"
    assignments = tuple(
        AssignedPolicySlotV1(
            global_slot=slot,
            evaluation_role="focal" if slot < 5 else "adversarial_opponent",
            policy_kind="reactive_tdm" if slot < 5 else "scenario_pressure",
            policy_id=alpha.identifier if slot < 5 else pressure.identifier,
            policy_content_digest=(
                alpha.canonical_digest if slot < 5 else pressure.canonical_digest
            ),
            algorithm_id=alpha.identifier if slot < 5 else pressure.identifier,
            training_run_id="not_applicable",
            training_step=0,
            parameter_sharing_group_id="team-a" if slot < 5 else "team-b",
            preprocessing=SHARED_OBS_ACTOR_PROJECTION_V1,
            normalization=VersionedIdentityV1(identifier="none", version=1),
            execution_mode="deterministic",
        )
        for slot in range(10)
    )
    return build_evaluation_episode_context_v1(
        identity=EvaluationEpisodeIdentityV1(
            run_id="tdm-closeout-plumbing",
            evaluation_id=f"tdm-scenario-{info.scenario_id}",
            matchup_id=f"tdm-scenario-{info.scenario_id}-alpha-vs-pressure",
            match_id=episode_id,
            episode_id=episode_id,
            paired_comparison_key=episode_id,
            evaluation_suite=_identity(
                "tdm-eight-scenarios", {"scenarios": list_tdm_scenarios()}
            ),
            experiment_manifest=experiment,
            task=_identity("team_deathmatch", {"task_mode": 1}),
            layout=ContentAddressedIdentityV1(
                identifier=f"tdm-scenario-{info.scenario_id}-map",
                version=1,
                canonical_digest=info.map_semantic_digest,
            ),
            scenario=ContentAddressedIdentityV1(
                identifier=f"tdm-scenario-{info.scenario_id}",
                version=1,
                canonical_digest=info.source.semantic_digest,
            ),
        ),
        aggregation_keys=(AggregationKeyV1(name="purpose", value="pipeline_control"),),
        expected_horizon=info.horizon,
        config=scenario.config,
        public_agent_id_by_global_slot=TDM_SCENARIO_PUBLIC_AGENT_IDS,
        policy_assignments=assignments,
        seed_protocol=schedule.realized_seed_protocols[coordinate],
        capture_profile="scenario_metric_complete",
        execution_information_mode="shared_obs",
        actor_projection=SHARED_OBS_ACTOR_PROJECTION_V1,
        critic_information_regime=VersionedIdentityV1(
            identifier="not_applicable", version=1
        ),
        canonical_reward_mode=VersionedIdentityV1(
            identifier="canonical-task-reward", version=1
        ),
        shaping_configuration=_identity("no-shaping", {"enabled": False}),
        code_revision=code_revision,
    )


def capture_tdm_qualification_episode(
    scenario_id: int,
    coordinate: int,
    *,
    code_revision: CodeRevisionV1,
    runtime_provenance: RuntimeProvenanceV1,
    stop_after: int | None = None,
) -> tuple[ResolvedScenarioSpecificationV2, ReplayBundleV1, ScenarioEvaluationRecordV2]:
    """Compose one diagnostic episode with an optional retained-prefix stop."""
    scenario = load_tdm_scenario(scenario_id)
    schedule = build_tdm_qualification_seed_schedule()
    if type(coordinate) is not int or not 0 <= coordinate < len(
        schedule.realized_seed_protocols
    ):
        raise ValueError("coordinate must select exactly one qualification seed row")
    if stop_after is not None and (type(stop_after) is not int or stop_after < 0):
        raise ValueError("stop_after must be a nonnegative integer or None")
    context = _qualification_context(scenario, schedule, coordinate, code_revision)
    state, observation, mask, _ = initialize_scenario_state(
        scenario.initial_state, scenario.config
    )
    availability = build_rollout_information_availability(scenario.config, "shared_obs")
    first_frame = capture_initial_evaluation_frame_v1(
        context, state, observation, mask, availability
    )
    specification = build_tdm_scenario_specification(scenario_id, first_frame, schedule)
    identity_payload = context.identity.model_dump(mode="python")
    identity_payload["scenario"] = ContentAddressedIdentityV1(
        identifier=specification.scenario_id,
        version=specification.scenario_version,
        canonical_digest=specification.canonical_digest_sha256,
    )
    context = EvaluationEpisodeContextV1.model_validate(
        {**context.model_dump(mode="python"), "identity": identity_payload}
    )
    observer = build_evaluation_observer_v1(
        context, build_tdm_metric_reducers(full=True)
    )
    observer.start(first_frame)
    history = rollout(
        scenario.config,
        state,
        observation,
        mask,
        jax.random.key(context.seed_protocol.environment_seed),
        reactive_tdm_alpha_policy,
        reactive_tdm_beta_policy
        if scenario_id in TDM_BETA_SCENARIO_IDS
        else reactive_tdm_alpha_policy,
        execution_information_mode="shared_obs",
    )
    valid = np.asarray(history.successors[-1].transition_facts.has_transition)
    count = int(np.count_nonzero(valid))
    retained_count = count if stop_after is None else min(count, stop_after)
    frame = first_frame
    for index in range(retained_count):

        def select_epoch(leaf: Array, epoch: int = index) -> Array:
            return leaf[epoch]

        successor = cast(
            ReferenceSuccessorHistory,
            jax.tree_util.tree_map(select_epoch, history.successors),
        )
        next_state, next_observation, reward, done, next_mask, info = successor
        transition, frame = capture_evaluation_transition_unit_v1(
            context,
            frame,
            next_state,
            next_observation,
            next_mask,
            info.transition_facts,
            reward,
            done,
            successor_shared_obs_information_availability_by_recipient_and_sensor_source=availability,
        )
        observer.append(transition, frame)
    report = (
        observer.finalize(completion_state="complete")
        if retained_count == count
        else observer.finalize(
            completion_state="partial",
            end_or_failure_reason="Qualification interruption.",
        )
    )
    bundle = build_replay_bundle_v1(
        observer, report, runtime_provenance=runtime_provenance
    )
    record = build_tdm_scenario_evaluation_record(
        scenario_id,
        specification,
        bundle.replay,
        bundle.metric_report_artifact,
        schedule_coordinate=coordinate,
    )
    return specification, bundle, record


def qualify_tdm_scenarios(destination: Path) -> dict[str, object]:
    """Publish all sixteen coordinate records and a complete or failed suite index."""
    destination.mkdir(parents=True, exist_ok=False)
    repository = Path(__file__).resolve().parents[2]
    revision = discover_debugger_code_revision_v1(repository)
    runtime = capture_debugger_runtime_provenance_v1(
        revision, policy_execution_included=True
    )
    schedule = build_tdm_qualification_seed_schedule()
    rows: list[dict[str, object]] = []
    specifications: list[ResolvedScenarioSpecificationV2] = []
    errors: list[dict[str, object]] = []
    for info in list_tdm_scenarios():
        for coordinate in range(len(schedule.realized_seed_protocols)):
            try:
                specification, bundle, record = capture_tdm_qualification_episode(
                    info.scenario_id,
                    coordinate,
                    code_revision=revision,
                    runtime_provenance=runtime,
                )
                if coordinate == 0:
                    specifications.append(specification)
                stem = f"scenario-{info.scenario_id}-coordinate-{coordinate}"
                replay_path = destination / f"{stem}.marlbg-replay.json"
                record_path = destination / f"{stem}.marlbg-scenario.json"
                save_replay_bundle_v1(bundle, replay_path)
                save_scenario_evaluation_record_v2(
                    record, bundle.replay, bundle.metric_report_artifact, record_path
                )
                loaded = load_replay_bundle_v1(replay_path, require_metric_report=True)
                assert loaded.metric_report_artifact is not None
                reloaded_record = load_scenario_evaluation_record_v2(
                    record_path,
                    source_replay=loaded.replay,
                    metric_report_artifact=loaded.metric_report_artifact,
                )
                if reloaded_record != record:
                    raise ValueError("scenario record changed on semantic roundtrip")
                report = bundle.metric_report_artifact.report
                metric_ids = {row.metric_id for row in report.statistics}
                if report.processing_status.status != "succeeded" or metric_ids != set(
                    TDM_EPISODE_METRIC_IDS
                ):
                    errors.append(
                        {
                            "scenario_id": info.scenario_id,
                            "schedule_coordinate": coordinate,
                            "error": "Incomplete episode metrics or failed processing.",
                        }
                    )
                rows.append(
                    {
                        "scenario_id": info.scenario_id,
                        "schedule_coordinate": coordinate,
                        "specification_digest": specification.canonical_digest_sha256,
                        "record_digest": record.canonical_digest_sha256,
                        "replay": replay_path.name,
                        "scenario_record": record_path.name,
                        "transition_count": len(bundle.replay.transitions),
                        "completion": bundle.replay.completion.completion_state,
                        "metric_processing": report.processing_status.status,
                        "metric_ids": sorted(metric_ids),
                        "metric_rows": len(
                            bundle.metric_report_artifact.report.statistics
                        ),
                        "endpoint": record.measurement_results[0].model_dump(
                            mode="json"
                        ),
                    }
                )
            except Exception as error:
                errors.append(
                    {
                        "scenario_id": info.scenario_id,
                        "schedule_coordinate": coordinate,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
    if discover_debugger_code_revision_v1(repository) != revision:
        errors.append(
            {
                "error": (
                    "Source identity changed during qualification; "
                    "rerun the fixed candidate."
                )
            }
        )
    result: dict[str, object] = {
        "schema_id": "marl_battlegrounds.tdm_scenario_qualification",
        "schema_version": 1,
        "purpose": "ALPHA Team A pipeline control; not winning or manuscript evidence",
        "complete": len(rows) == 16 and not errors,
        "code_revision": revision.model_dump(mode="json"),
        "seed_schedule": schedule.model_dump(mode="json"),
        "specifications": [row.model_dump(mode="json") for row in specifications],
        "records": rows,
        "failures": errors,
    }
    result["canonical_digest_sha256"] = canonical_digest_sha256(result)
    (destination / "suite-index.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="new evidence directory")
    args = parser.parse_args()
    result = qualify_tdm_scenarios(cast(Path, args.destination))
    print(
        json.dumps(
            {"complete": result["complete"], "failures": result["failures"]}, indent=2
        )
    )
    if not result["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
