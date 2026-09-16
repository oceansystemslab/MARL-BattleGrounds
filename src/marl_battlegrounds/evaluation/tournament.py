"""Execute current paired all-pairs tournaments through the shared evaluator.

run_tournament freezes entrants, builds a balanced map/side schedule, executes
each directed matchup and qualifies the complete result population. Optional
RunWriter output owns match/full/replay files. The same in-memory statistics
serve both routes; this module does not define a second rating implementation.
"""

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from numbers import Integral
from pathlib import Path
from typing import cast
from uuid import uuid4

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

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
    _json_bytes,  # pyright: ignore[reportPrivateUsage]
    _json_value,  # pyright: ignore[reportPrivateUsage]
    configuration_identity,
)
from marl_battlegrounds.evaluation.scalar_reports import iter_scalar_rows
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
    TDMMapInfo,
    canonical_tournament_rosters,
)

_ROSTER_A, _ROSTER_B = canonical_tournament_rosters()


@dataclass(frozen=True)
class TournamentResult:
    """Completed tournament evidence, summaries and optional recorded outputs.

    Attributes
    ----------
    matches : tuple[ResultRow, ...]
        Every completed match row in global episode-ID order.
    tournament_results : tuple[ResultRow, ...]
        One aggregate rating/rate row per policy.
    matchup_results : tuple[ResultRow, ...]
        Directed policy/opponent summary rows.
    map_results : tuple[ResultRow, ...]
        Policy/map summary rows.
    full_metrics : Columns
        Selected in-memory full metric columns, or {} with file output.
    replays : tuple[ReplayArtifactV3, ...]
        Selected in-memory replay artifacts, or () with file output.
    metadata : dict[str, object]
        Settings, source/config/pass facts and statistical-method details.
    paths : dict[str, Path] | None
        Produced output paths, or None for a fully in-memory run.

    Notes
    -----
    Summary interval bounds may be None when evidence is insufficient; their
    status fields explain why. Ratings summarize these fixed entrants/maps,
    not learning speed or variation across separately trained seeds. The record
    is frozen, but contained tables/dicts are not deeply immutable.
    """

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
    """Join each scheduled completion to its available scalar and identity columns.

    When optional metrics were disabled, use compact episode/config evidence for
    the required match identity, outcome, length and scores. Other missing metric
    cells remain None. Exact integer summaries replace their float32 views.
    """
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
            episode_length=completed[match.episode_id].episode_length,
            team_a_score=completed[match.episode_id].team_a_score,
            team_b_score=completed[match.episode_id].team_b_score,
        )
        rows.append(row)
    return rows


def _read_matches(path: Path) -> list[ResultRow]:
    """Read the manifest's durable match rows through the shared schema reader.

    Current tables require MATCH_COLUMNS. Historical supported tables retain
    their stored columns and meanings. Ignore an interrupted, uncommitted suffix
    without modifying it. Missing or incompatible metadata raises ValueError.
    """
    manifest = json.loads((path.parent / "run_details.json").read_bytes())
    return [
        row
        for batch in iter_scalar_rows(
            path, manifest=manifest, expected_header=MATCH_COLUMNS
        )
        for row in batch
    ]


def _merge_columns(tables: Sequence[Columns]) -> Columns:
    """Join nonempty metric tables and sort every column by global episode ID."""
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
    """Evaluate every distinct fixed-policy pair with equal maps and both sides.

    Parameters
    ----------
    policies : Sequence[Policy | str]
        At least two Policy objects or built-in names. Resolved policy
        names must be distinct; variables and initial memory are snapshotted.
    maps : Sequence[int | TDMMapInfo] | None
        Optional distinct integer map IDs or TDMMapInfo entries. None uses
        canonical evaluation maps 47..51. Exact EnvConfig inputs are not accepted.
    episodes_per_pair : int
        Positive total game budget across maps and both policy
        side assignments, default 100. Divisible by twice the map count.
    seed : int
        Root uint32 integer, default 0; also seeds summary resampling.
    num_envs : int
        Positive maximum simultaneous games per directed matchup,
        default 128. Smaller pending schedules use smaller batches.
    metrics : MetricMode
        "priority" by default, "full" for all full measurements, or "none"
        to skip optional episode measurements. Match outcomes are always kept.
    full_metrics_episodes : Iterable[int]
        Global schedule IDs selected for full metrics;
        empty by default. Explicit selections still work with metrics="none".
    replay_episodes : Iterable[int]
        Global schedule IDs selected for replay; empty by default.
    output_dir : str | Path | None
        Optional parent folder for a new run with a unique child folder.
    resume_from : str | Path | None
        Optional existing run folder with matching tournament identity.
        Use this or output_dir, not both. Durable matches are not re-executed.
    opponent_weights : Mapping[str, float] | None
        Optional finite positive weight for every entrant name.
        None gives equal opponent weights. Rates/tail scores use these weights;
        the rating fit uses the observed match counts.
    score_threshold : int
        Positive TDM score target for every map, default 20.
    max_steps : int
        Positive horizon in ticks for each game, default 300.
    chunk_size : int
        Positive scheduling chunk length in ticks, default 16.

    Returns
    -------
    TournamentResult
        TournamentResult with complete match evidence, policy/matchup/map summaries
        and method metadata. Without file output, selected full columns and replays
        remain in memory. With output, those fields are empty and paths identifies
        produced files. Persisted priority values share match_results.csv rather
        than a duplicate priority table.

    Raises
    ------
    ValueError
        Names, map choices, budgets, selections, weights, saved identity
        or the complete match population are invalid.
    TypeError
        A config or policy input/output violates its type contract.
    RuntimeError
        A policy, writer or required statistical fit fails.
    OSError
        Run files cannot be accessed or written.

    Notes
    -----
    Host-only orchestration; do not wrap this function in jax.jit. JAX policies
    use the shared compiled evaluator; external policies add their host-boundary
    costs. Ratings and 5,000 paired-block bootstrap replicates run on the host.
    The function owns and closes any writer it creates. A failed run remains
    available for explicit resume. Default ordered rosters are mage, warrior,
    hunter, rogue, priest for both teams. Policy sides swap on fixed map banks;
    no additional bank shuffle occurs. This executes the current all-pairs
    protocol and does not claim future D11 enrollment behavior.

    Examples
    --------
    A small complete in-memory tournament (20 games across two maps):

    >>> import marl_battlegrounds as marl_bgs
    >>> result = marl_bgs.run_tournament(
    ...     ["random", "tdm-alpha"], maps=[47, 48],
    ...     episodes_per_pair=20, num_envs=4, max_steps=16, seed=7,
    ... )
    >>> len(result.matches)
    20
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
        spec.map_id: spec.env_config
        for spec in normalize_episode_specs(
            sorted(map_ids),
            len(map_ids),
            _ROSTER_A,
            _ROSTER_B,
            score_threshold,
            max_steps,
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
    if recording:

        def normalize(value: object) -> Array:
            """Match the numerical config leaves used by environment recording."""
            return jnp.asarray(value)

        metadata["configuration_ids_by_map"] = {
            str(map_id): configuration_identity(jax.tree.map(normalize, config))[0]
            for map_id, config in configs.items()
        }
        metadata["schedule_digest"] = sha256(
            _json_bytes(_json_value([asdict(match) for match in schedule]))
        ).hexdigest()
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
