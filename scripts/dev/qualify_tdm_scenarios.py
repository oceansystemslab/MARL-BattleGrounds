"""Qualify the eight approved scenarios through the shared public evaluator.

ALPHA controls Team A solely to exercise the pipeline. These are neither winning
witnesses nor learned-policy or manuscript results. No policy is trained.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from marl_battlegrounds.evaluation.evaluate import (
    Columns,
    EpisodeSpec,
    evaluate_episodes,
)
from marl_battlegrounds.evaluation.metric_catalog import FULL_METRIC_NAMES
from marl_battlegrounds.evaluation.models import (
    ContentAddressedIdentityV1,
    canonical_digest_sha256,
)
from marl_battlegrounds.evaluation.policy_execution import controller_identity, policy
from marl_battlegrounds.evaluation.replay_io import (
    load_replay,
    load_scenario_evaluation_record_v4,
    save_scenario_evaluation_record_v4,
)
from marl_battlegrounds.evaluation.replay_v3 import ReplayArtifactV3, build_replay_v3
from marl_battlegrounds.evaluation.revision import discover_code_revision_v2
from marl_battlegrounds.evaluation.run_writer import IDENTITY_COLUMNS, RunWriter
from marl_battlegrounds.evaluation.scenario import (
    ResolvedScenarioSpecificationV3,
    ScenarioEvaluationRecordV4,
)
from marl_battlegrounds.evaluation.tdm_scenarios import (
    TDM_BETA_SCENARIO_IDS,
    TDM_SCENARIO_PUBLIC_AGENT_IDS,
    build_tdm_qualification_seed_schedule,
    build_tdm_scenario_evaluation_record,
    build_tdm_scenario_specification,
)
from marl_battlegrounds.tasks import list_tdm_scenarios, load_tdm_scenario


@dataclass(frozen=True)
class QualificationEpisode:
    """Current endpoint evidence and optional complete scalar metric row."""

    specification: ResolvedScenarioSpecificationV3
    replay: ReplayArtifactV3
    record: ScenarioEvaluationRecordV4
    full_metrics: Columns


def _planned_episode(
    scenario_id: int,
    coordinate: int,
) -> tuple[ResolvedScenarioSpecificationV3, EpisodeSpec]:
    scenario = load_tdm_scenario(scenario_id)
    schedule = build_tdm_qualification_seed_schedule()
    if type(coordinate) is not int or not 0 <= coordinate < len(
        schedule.realized_seed_protocols
    ):
        raise ValueError("coordinate must select exactly one qualification seed row")
    specification = build_tdm_scenario_specification(scenario_id, schedule)
    pressure = specification.pressure_protocol
    assert pressure is not None
    metadata: dict[str, object] = {
        "scenario_name": scenario.info.name,
        "scenario_identity": ContentAddressedIdentityV1(
            identifier=specification.scenario_id,
            version=specification.scenario_version,
            canonical_digest=specification.canonical_digest_sha256,
        ).model_dump(mode="json"),
        "layout_identity": specification.layout.model_dump(mode="json"),
        "public_agent_id_by_global_slot": TDM_SCENARIO_PUBLIC_AGENT_IDS,
        "evaluation_role_by_global_slot": specification.role_template,
        "seed_protocol": schedule.realized_seed_protocols[coordinate].model_dump(
            mode="json"
        ),
        "team_a_controller_identity": controller_identity(policy("tdm-alpha")),
        "team_b_controller_identity": pressure.model_dump(mode="json"),
        "paired_comparison_key": f"tdm-scenario-{scenario_id}-coordinate-{coordinate}",
    }
    return specification, EpisodeSpec(
        episode_id=(scenario_id - 1) * 2 + coordinate + 1,
        seed_id=coordinate,
        config=scenario.config,
        initial_state=scenario.initial_state,
        metadata=metadata,
    )


def capture_tdm_qualification_episode(
    scenario_id: int,
    coordinate: int,
    *,
    stop_after: int | None = None,
) -> QualificationEpisode:
    """Capture one real evaluator episode, optionally retain a truthful prefix.

    Prefix witnesses do not expose the completed episode's scalar metrics as if
    they described that prefix. All trajectory execution remains in the evaluator.
    """
    if stop_after is not None and (type(stop_after) is not int or stop_after < 0):
        raise ValueError("stop_after must be a nonnegative integer or None")
    specification, planned = _planned_episode(scenario_id, coordinate)
    result = evaluate_episodes(
        policy("tdm-alpha"),
        policy("tdm-beta" if scenario_id in TDM_BETA_SCENARIO_IDS else "tdm-alpha"),
        (planned,),
        seed=0,
        num_envs=1,
        metrics="full",
        replay_episodes=(planned.episode_id,),
        phase="qualification",
        pass_id="scenarios",
        run_id="tdm-closeout-plumbing",
        chunk_size=5,
    )
    replay = result.replays[0]
    full_metrics = result.full_metrics
    if stop_after is not None and stop_after < len(replay.transitions):
        replay = build_replay_v3(
            replay.header.context,
            replay.frames[: stop_after + 1],
            replay.transitions[:stop_after],
            runtime_provenance=replay.header.runtime_provenance,
            wrapper_stack=replay.header.wrapper_stack,
            completion_state="partial",
            end_or_failure_reason="Qualification interruption.",
        )
        full_metrics = {}
    record = build_tdm_scenario_evaluation_record(
        scenario_id,
        specification,
        replay,
        schedule_coordinate=coordinate,
    )
    return QualificationEpisode(specification, replay, record, full_metrics)


def qualify_tdm_scenarios(destination: Path) -> dict[str, object]:
    """Write one run with sixteen scalar rows and replay-backed scenario records."""
    destination.mkdir(parents=True, exist_ok=False)
    revision = discover_code_revision_v2()
    schedule = build_tdm_qualification_seed_schedule()
    planned = {
        episode.episode_id: (info.scenario_id, coordinate, specification, episode)
        for info in list_tdm_scenarios()
        for coordinate in range(len(schedule.realized_seed_protocols))
        for specification, episode in (_planned_episode(info.scenario_id, coordinate),)
    }
    rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    paths: dict[str, Path] = {}
    try:
        with RunWriter(destination) as writer:
            for pressure_name in ("tdm-alpha", "tdm-beta"):
                episodes = tuple(
                    row[3]
                    for row in planned.values()
                    if (row[0] in TDM_BETA_SCENARIO_IDS)
                    == (pressure_name == "tdm-beta")
                )
                evaluate_episodes(
                    policy("tdm-alpha"),
                    policy(pressure_name),
                    episodes,
                    seed=0,
                    num_envs=1,
                    metrics="full",
                    replay_episodes=tuple(episode.episode_id for episode in episodes),
                    writer=writer,
                    phase="qualification",
                    pass_id=pressure_name,
                    chunk_size=5,
                )
            writer.flush()
            paths = writer.paths
        details = json.loads(paths["run_details"].read_text())
        for entry in details["passes"].values():
            for episode_id, reference in entry["replays"].items():
                scenario_id, coordinate, specification, _ = planned[int(episode_id)]
                replay_path = paths["run_details"].parent / reference["path"]
                replay = load_replay(replay_path).replay
                if not isinstance(replay, ReplayArtifactV3):
                    raise TypeError("current qualification requires replay V3")
                record = build_tdm_scenario_evaluation_record(
                    scenario_id,
                    specification,
                    replay,
                    schedule_coordinate=coordinate,
                )
                stem = f"scenario-{scenario_id}-coordinate-{coordinate}"
                record_path = destination / f"{stem}.marlbg-scenario.json"
                save_scenario_evaluation_record_v4(record, replay, record_path)
                if (
                    load_scenario_evaluation_record_v4(
                        record_path, source_replay=replay
                    )
                    != record
                ):
                    raise ValueError("scenario record changed on semantic roundtrip")
                rows.append(
                    {
                        "scenario_id": scenario_id,
                        "schedule_coordinate": coordinate,
                        "episode_id": int(episode_id),
                        "specification_digest": specification.canonical_digest_sha256,
                        "record_digest": record.canonical_digest_sha256,
                        "replay": str(replay_path.relative_to(destination)),
                        "scenario_record": record_path.name,
                        "transition_count": len(replay.transitions),
                        "completion": replay.completion.completion_state,
                        "endpoint": record.measurement_results[0].model_dump(
                            mode="json"
                        ),
                    }
                )
        with paths["full_metrics"].open(newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != [*IDENTITY_COLUMNS, *FULL_METRIC_NAMES]:
                raise ValueError(
                    "qualification full CSV differs from the current fixed schema"
                )
            metric_rows = list(reader)
            if len(metric_rows) != 16 or {
                int(row["episode_id"]) for row in metric_rows
            } != set(planned):
                raise ValueError(
                    "qualification needs exactly one scalar row per planned episode"
                )
    except Exception as error:
        errors.append({"error": f"{type(error).__name__}: {error}"})
    if discover_code_revision_v2() != revision:
        errors.append(
            {"error": "Source changed during qualification; rerun the fixed candidate."}
        )
    result: dict[str, object] = {
        "schema_id": "marl_battlegrounds.tdm_scenario_qualification",
        "schema_version": 2,
        "purpose": "ALPHA Team A pipeline control; not winning or manuscript evidence",
        "complete": len(rows) == 16 and not errors,
        "code_revision": revision.model_dump(mode="json"),
        "seed_schedule": schedule.model_dump(mode="json"),
        "specifications": [
            row[2].model_dump(mode="json") for row in planned.values() if row[1] == 0
        ],
        "metric_columns": len(FULL_METRIC_NAMES),
        "outputs": {
            name: str(path.relative_to(destination)) for name, path in paths.items()
        },
        "records": sorted(rows, key=lambda row: cast(int, row["episode_id"])),
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
