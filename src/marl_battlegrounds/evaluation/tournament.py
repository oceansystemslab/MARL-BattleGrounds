"""Frozen paired cross-play through the same evaluator and durable run writer."""

import csv
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, replace
from numbers import Integral
from pathlib import Path
from typing import cast
from uuid import uuid4

import numpy as np

from marl_battlegrounds._tdm_assets import current_map_id
from marl_battlegrounds.environment import MetricMode, make
from marl_battlegrounds.evaluation.evaluate import (
    Columns,
    EpisodeSpec,
    EvaluationResult,
    evaluate_episodes,
    normalize_episode_specs,
    policy_description,
    positive_int,
)
from marl_battlegrounds.evaluation.metric_catalog import PRIORITY_METRIC_NAMES
from marl_battlegrounds.evaluation.policy_execution import (
    Policy,
    freeze_variables,
    policy,
)
from marl_battlegrounds.evaluation.replay_v3 import ReplayArtifactV3
from marl_battlegrounds.evaluation.run_writer import (
    IDENTITY_COLUMNS,
    MATCH_COLUMNS,
    RunWriter,
)
from marl_battlegrounds.evaluation.tournament_schedule import (
    TournamentMatch,
    build_tournament_schedule,
)
from marl_battlegrounds.evaluation.tournament_statistics import (
    ResultRow,
    summarize_tournament,
    validate_opponent_weights,
)
from marl_battlegrounds.tasks import (
    CANONICAL_TDM_EVALUATION_MAP_IDS,
    AgentClassName,
    TDMMapInfo,
)

_ROSTER: tuple[AgentClassName, ...] = ("mage", "warrior", "hunter", "rogue", "priest")


@dataclass(frozen=True)
class TournamentResult:
    """Complete match evidence, qualified summaries and optional diagnostics."""

    matches: tuple[ResultRow, ...]
    tournament_results: tuple[ResultRow, ...]
    matchup_results: tuple[ResultRow, ...]
    map_results: tuple[ResultRow, ...]
    full_metrics: Columns
    replays: tuple[ReplayArtifactV3, ...]
    metadata: dict[str, object]
    paths: dict[str, Path] | None


def _memory_matches(
    result: EvaluationResult, schedule: Sequence[TournamentMatch]
) -> list[ResultRow]:
    indices = {
        int(value): index
        for index, value in enumerate(result.priority_metrics.get("episode_id", ()))
    }
    completed = {row.episode_id: row for row in result.episodes}
    configurations = cast(
        dict[str, dict[str, object]], result.metadata["configurations"]
    )
    rows: list[ResultRow] = []
    for match in schedule:
        index = indices.get(match.episode_id)
        row: ResultRow = {name: None for name in MATCH_COLUMNS}
        if index is not None:
            for name in (*IDENTITY_COLUMNS, *PRIORITY_METRIC_NAMES):
                value = np.asarray(result.priority_metrics[name][index]).item()
                row[name] = (
                    None
                    if isinstance(value, float) and not np.isfinite(value)
                    else value
                )
        else:
            config_id = completed[match.episode_id].config_id
            config = configurations[config_id]
            profile = cast(dict[str, list[int]], config["agent_profile"])
            row.update(
                run_id=str(result.metadata["run_id"]),
                phase="tournament",
                pass_id=str(result.metadata["pass_id"]),
                episode_id=match.episode_id,
                seed_id=match.seed_id,
                map_id=match.map_id,
                config_id=config_id,
                team_a_policy=match.team_a,
                team_b_policy=match.team_b,
            )
            for slot in range(10):
                row[f"agent_{slot}_class_id"] = int(profile["class_ids"][slot])
                row[f"agent_{slot}_active"] = int(profile["active_mask"][slot])
        row.update(
            block_id=match.block_id,
            bootstrap_group=match.bootstrap_group,
            outcome=completed[match.episode_id].outcome,
        )
        rows.append(row)
    return rows


def _read_matches(path: Path) -> list[ResultRow]:
    integer_columns = {
        "episode_id",
        "seed_id",
        "map_id",
        "block_id",
        "outcome",
        *(
            f"agent_{slot}_{field}"
            for slot in range(10)
            for field in ("class_id", "active")
        ),
    }
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != MATCH_COLUMNS:
            raise ValueError("tournament match table has an incompatible schema")
        return [
            {
                name: None
                if value == ""
                else int(value)
                if name in integer_columns
                else float(value)
                if name in PRIORITY_METRIC_NAMES
                else value
                for name, value in row.items()
            }
            for row in reader
        ]


def _merge_columns(tables: Sequence[Columns]) -> Columns:
    selected = [table for table in tables if table]
    if not selected:
        return {}
    merged = {
        name: np.concatenate([table[name] for table in selected])
        for name in selected[0]
    }
    order = np.argsort(merged["episode_id"])
    return {name: values[order] for name, values in merged.items()}


def run_tournament(
    policies: Sequence[Policy | str],
    *,
    maps: Sequence[int | TDMMapInfo] | None = None,
    episodes_per_pair: int = 100,
    seed: int = 0,
    num_envs: int = 128,
    metrics: MetricMode = "priority",
    full_metrics_episodes: Iterable[int] = (),
    replay_episodes: Iterable[int] = (),
    output_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
    opponent_weights: Mapping[str, float] | None = None,
    score_threshold: int = 20,
    max_steps: int = 300,
    chunk_size: int = 16,
) -> TournamentResult:
    """Run paired, equally weighted maps for every distinct frozen policy pair.

    ``episodes_per_pair`` is the total across maps and both side assignments.
    Full metrics and replay selections use global one-based schedule IDs. Outcome
    evidence is always retained for the requested ratings; ``metrics='none'``
    disables optional episode measurements. No files are created without an
    explicit destination. Persisted tournaments use one match table in place of
    a repeated priority table, plus self-contained selected full rows and replays.
    """
    entrants = tuple(
        policy(item) if isinstance(item, str) else item for item in policies
    )
    map_ids = tuple(
        current_map_id(item) if isinstance(item, TDMMapInfo) else item
        for item in (CANONICAL_TDM_EVALUATION_MAP_IDS if maps is None else maps)
    )
    schedule = build_tournament_schedule(
        [entrant.name for entrant in entrants],
        map_ids,
        episodes_per_pair=episodes_per_pair,
    )
    validate_opponent_weights(
        tuple(sorted(entrant.name for entrant in entrants)), opponent_weights
    )
    num_envs = positive_int(num_envs, "num_envs")
    chunk_size = positive_int(chunk_size, "chunk_size")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, Integral)
        or not 0 <= seed <= 0xFFFFFFFF
    ):
        raise ValueError("seed must be a uint32 integer")
    selected = make(
        "tdm",
        metrics=metrics,
        full_metrics_episodes=full_metrics_episodes,
        replay_episodes=replay_episodes,
    )
    for name, ids in (
        ("full_metrics_episodes", selected.full_metrics_episodes),
        ("replay_episodes", selected.replay_episodes),
    ):
        if any(episode_id > len(schedule) for episode_id in ids):
            raise ValueError(f"{name} contains an episode outside the schedule")
    configs = {
        spec.map_id: spec.config
        for spec in normalize_episode_specs(
            sorted(map_ids), len(map_ids), _ROSTER, _ROSTER, score_threshold, max_steps
        )
    }
    frozen = {
        entrant.name: replace(
            entrant,
            variables=freeze_variables(entrant.variables),
            initial_carry=freeze_variables(entrant.initial_carry),
        )
        for entrant in entrants
    }
    groups: dict[tuple[str, str], list[TournamentMatch]] = defaultdict(list)
    for match in schedule:
        groups[match.team_a, match.team_b].append(match)
    recording = bool(
        output_dir is not None or resume_from is not None or selected.replay_episodes
    )
    metadata: dict[str, object] = {
        "seed": seed,
        "rng_protocol": "episode-fold-in-v1",
        "policies": [
            policy_description(
                team, team.variables, team.initial_carry, include_digests=recording
            )
            for _, team in sorted(frozen.items())
        ],
        "map_ids": sorted(map_ids),
        "episodes_per_pair": episodes_per_pair,
        "num_matches": len(schedule),
        "score_threshold": score_threshold,
        "max_steps": max_steps,
        "metrics": metrics,
        "full_metrics_episodes": list(selected.full_metrics_episodes),
        "replay_episodes": list(selected.replay_episodes),
        "opponent_weights": None
        if opponent_weights is None
        else dict(opponent_weights),
    }
    with ExitStack() as cleanup:
        writer = None
        if output_dir is not None or resume_from is not None:
            writer = cleanup.enter_context(
                RunWriter(
                    output_dir,
                    resume_from=resume_from,
                    phase="tournament",
                    pass_id="schedule",
                    details=metadata,
                )
            )
        run_id = writer.run_id if writer is not None else "in-memory-" + uuid4().hex
        metadata = {**metadata, "run_id": run_id}
        configurations: dict[str, object] = {}
        passes: dict[str, object] = {}
        matches: list[ResultRow] = []
        full_tables: list[Columns] = []
        replays: list[ReplayArtifactV3] = []
        for index, ((first, second), group) in enumerate(sorted(groups.items()), 1):
            group_ids = {match.episode_id for match in group}
            result = evaluate_episodes(
                frozen[first],
                frozen[second],
                [
                    EpisodeSpec(
                        match.episode_id,
                        configs[match.map_id],
                        match.map_id,
                        match.seed_id,
                        metadata={
                            "block_id": match.block_id,
                            "bootstrap_group": match.bootstrap_group,
                            "paired_comparison_key": f"block-{match.block_id}",
                        },
                    )
                    for match in group
                ],
                seed=seed,
                num_envs=num_envs,
                metrics=metrics,
                full_metrics_episodes=group_ids.intersection(
                    selected.full_metrics_episodes
                ),
                replay_episodes=group_ids.intersection(selected.replay_episodes),
                writer=writer,
                phase="tournament",
                pass_id=f"pair-side-{index}",
                chunk_size=chunk_size,
                run_id=run_id,
            )
            configurations.update(
                cast(dict[str, object], result.metadata["configurations"])
            )
            passes[str(result.metadata["pass_id"])] = {
                name: value
                for name, value in result.metadata.items()
                if name not in ("run_id", "configurations")
            }
            if writer is None:
                matches.extend(_memory_matches(result, group))
                full_tables.append(result.full_metrics)
                replays.extend(result.replays)
        if writer is not None:
            matches = _read_matches(writer.paths["match_results"])
        matches.sort(key=lambda row: int(cast(int, row["episode_id"])))
        outcomes = {
            int(cast(int, row["episode_id"])): int(cast(int, row["outcome"]))
            for row in matches
        }
        if len(outcomes) != len(matches):
            raise ValueError("tournament match table repeats an episode identity")
        statistics = summarize_tournament(
            schedule,
            outcomes,
            seed=seed,
            opponent_weights=opponent_weights,
        )
        if writer is not None:
            writer.write_tournament_results(statistics)
        return TournamentResult(
            tuple(matches),
            statistics.tournament_results,
            statistics.matchup_results,
            statistics.map_results,
            _merge_columns(full_tables),
            tuple(replays),
            {
                **metadata,
                "configurations": configurations,
                "passes": passes,
                "statistics": statistics.metadata,
            },
            None if writer is None else writer.paths,
        )
