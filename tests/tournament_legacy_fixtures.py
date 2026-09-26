"""Build old list-tournament folders directly through existing recording owners.

These fixtures preserve the historical coordinator, pass IDs and full source
snapshots. They do not call a fresh legacy production runner. Tests may stop a
writer during a pass to retain an interrupted folder, or finish it through the
public historical-resume adapter and then check read-only recovery.
"""

# Historical fixtures use existing private preparation and recording helpers.
# pyright: reportPrivateUsage=false

from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from marl_battlegrounds.environment import MetricMode
from marl_battlegrounds.evaluation.evaluate import (
    EpisodeSpec,
    _evaluate_tournament_episodes,
    normalize_episode_specs,
    policy_description,
)
from marl_battlegrounds.evaluation.evaluation_conditions import config_record
from marl_battlegrounds.evaluation.policy_execution import Policy
from marl_battlegrounds.evaluation.recording_identity import (
    normalize_system_registration,
)
from marl_battlegrounds.evaluation.run_writer import RunWriter
from marl_battlegrounds.evaluation.sampling_evidence import method_sampling_fact
from marl_battlegrounds.evaluation.tournament_headlines import content_digest
from marl_battlegrounds.evaluation.tournament_schedule import build_tournament_schedule
from marl_battlegrounds.tasks import (
    _swap_spawn_banks,
    canonical_tournament_rosters,
    list_tdm_maps,
)


def create_legacy_tournament(
    output_dir: Path,
    policies: Sequence[Policy],
    *,
    maps: tuple[int, ...] = (12,),
    episodes_per_pair: int = 2,
    seed: int = 0,
    metrics: MetricMode = "priority",
    full_metrics_episodes: tuple[int, ...] = (),
    replay_episodes: tuple[int, ...] = (),
    red_zone_depth: float = 0.0,
    completed_passes: int = 0,
) -> Path:
    rosters = canonical_tournament_rosters()
    configs = {
        int(spec.map_id): spec.env_config  # type: ignore[arg-type]
        for spec in normalize_episode_specs(
            maps, len(maps), *rosters, 20, 1, red_zone_depth=red_zone_depth
        )
    }
    frozen = {item.name: item for item in policies}
    descriptions = [
        policy_description(
            item, item.variables, item.initial_carry, include_digests=True
        )
        for _, item in sorted(frozen.items())
    ]
    participants = {
        str(item["name"]): normalize_system_registration(item, phase="tournament")[0]
        for item in descriptions
    }
    registered = {item.map_id: item.model_dump(mode="json") for item in list_tdm_maps()}
    configurations: dict[str, Any] = {}
    source_ids: dict[int, str] = {}
    resolved: dict[tuple[int, int], Any] = {}
    resolved_ids: dict[tuple[int, int], str] = {}
    for map_id, config in configs.items():
        for choice in (0, 1):
            value = config if choice == 0 else _swap_spawn_banks(config)
            identifier, content = config_record(value)
            configurations[identifier] = content
            resolved[map_id, choice] = value
            resolved_ids[map_id, choice] = identifier
        source_ids[map_id] = resolved_ids[map_id, 0]
    schedule = tuple(
        replace(
            row,
            source_config_id=source_ids[row.map_id],
            resolved_config_id=resolved_ids[row.map_id, int(row.spawn_locations)],  # type: ignore[arg-type]
        )
        for row in build_tournament_schedule(
            tuple(frozen), maps, episodes_per_pair=episodes_per_pair
        )
    )
    details: dict[str, Any] = {
        "seed": seed,
        "rng_protocol": "evaluation-systems-v1",
        "policies": descriptions,
        "method_sampling": {
            name: method_sampling_fact(method) for name, method in frozen.items()
        },
        "map_ids": sorted(maps),
        "episodes_per_pair": episodes_per_pair,
        "num_matches": len(schedule),
        "score_threshold": 20,
        "max_steps": 1,
        "red_zone_depth": red_zone_depth,
        "metrics": metrics,
        "full_metrics_episodes": list(full_metrics_episodes),
        "replay_episodes": list(replay_episodes),
        "opponent_weights": None,
        "configuration_ids_by_map": {
            str(key): value for key, value in source_ids.items()
        },
        "configurations": configurations,
        "registered_maps": {str(key): registered[key] for key in configs},
        "pairing_protocol": "fixed-team-spawn-v1",
        "schedule": [asdict(row) for row in schedule],
        "schedule_digest": content_digest([asdict(row) for row in schedule]),
        "participants": participants,
    }
    with RunWriter(
        output_dir, phase="tournament", pass_id="schedule", details=details
    ) as writer:
        writer.set_tournament_coordinator(details["schedule_digest"])
        pairs = sorted({(row.team_a, row.team_b) for row in schedule})
        for index, (first, second) in enumerate(pairs[:completed_passes], 1):
            rows = [
                row for row in schedule if (row.team_a, row.team_b) == (first, second)
            ]
            ids = {row.episode_id for row in rows}
            specs = tuple(
                EpisodeSpec(
                    row.episode_id,
                    resolved[row.map_id, int(row.spawn_locations)],  # type: ignore[arg-type]
                    row.map_id,
                    row.seed_id,
                    metadata={
                        "block_id": row.block_id,
                        "bootstrap_group": row.bootstrap_group,
                    },
                    source_config=configs[row.map_id],
                    spawn_locations=row.spawn_locations,
                    paired_comparison_key=f"block-{row.block_id}",
                )
                for row in rows
            )
            _evaluate_tournament_episodes(
                frozen[first],
                frozen[second],
                specs,
                seed=seed,
                num_envs=2,
                metrics=metrics,
                full_metrics_episodes=ids.intersection(full_metrics_episodes),
                replay_episodes=ids.intersection(replay_episodes),
                writer=writer,
                phase="tournament",
                pass_id=f"pair-{index}",
                chunk_size=1,
                run_id=writer.run_id,
                registered_maps={key: registered[key] for key in configs},
            )
        return writer.run_dir
