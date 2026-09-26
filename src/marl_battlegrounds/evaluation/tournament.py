"""Execute current paired all-pairs tournaments through the shared evaluator.

run_tournament freezes entrants, builds a balanced map/spawn schedule, executes
each fixed-participant matchup and qualifies the complete result population. Optional
RunWriter output owns match/full/replay files. The same in-memory statistics
serve both routes; this module does not define a second rating implementation.
"""


# Private setup and population helpers are shared authorities, not parallel rules.
# pyright: reportPrivateUsage=false

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import asdict, replace
from numbers import Integral
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import jax
import numpy as np

from marl_battlegrounds._tdm_assets import current_map_id
from marl_battlegrounds.core.types import EnvConfig
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
from marl_battlegrounds.evaluation.evaluation_conditions import OMITTED, Omitted
from marl_battlegrounds.evaluation.metric_catalog import PRIORITY_METRIC_NAMES
from marl_battlegrounds.evaluation.policy_execution import (
    Policy,
    System,
)
from marl_battlegrounds.evaluation.replay_v4 import ReplayArtifactV4
from marl_battlegrounds.evaluation.results import TournamentResult
from marl_battlegrounds.evaluation.run_writer import (
    IDENTITY_COLUMNS,
    MATCH_COLUMNS,
    RunWriter,
)
from marl_battlegrounds.evaluation.scalar_reports import iter_scalar_rows
from marl_battlegrounds.evaluation.tournament_headlines import (
    content_digest,
    match_digest,
    summarize_headlines,
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
    TDMMapInfo,
    canonical_tournament_rosters,
)

_ROSTER_A, _ROSTER_B = canonical_tournament_rosters()


def _config_matches_identity(
    config: EnvConfig, identifier: str | None, *, historical: bool
) -> bool:
    """Tell whether a config reproduces a recorded configuration identity.

    config is a scalar EnvConfig; identifier is the recorded SHA-256 (None never
    matches); historical selects the identity a config had before the Red Zone
    field existed (only a depth-0.0 config has one). Return False, rather than
    raising, when a historical identity is impossible for this config.
    """
    from marl_battlegrounds.evaluation.evaluation_conditions import config_record

    if identifier is None:
        return False
    try:
        return config_record(config, historical=historical)[0] == identifier
    except ValueError:
        return False


def _prepare_pair_evidence(
    schedule: Sequence[TournamentMatch],
    matches: Sequence[Mapping[str, object]],
    *,
    configurations: Mapping[str, Mapping[str, object]],
    systems: Mapping[str, Mapping[str, object]],
    passes: Mapping[str, Mapping[str, object]],
) -> dict[str, Any]:
    """Verify physical spawn pairs and bind their owners to final game rows.

    Parameters
    ----------
    schedule, matches : sequences
        The complete resolved fixed-team schedule and its exact final raw rows.
    configurations, systems, passes : mappings
        Recorded configuration contents, immutable method registrations and pass
        declarations, keyed by their existing content or phase/pass identities.

    Returns
    -------
    dict
        Version-1 evidence for these rows and this schedule. Physical source
        comparisons run once per distinct source/resolved configuration pair.
        Statistics and headlines consume the same evidence without another fit
        or full metric expansion.

    Raises
    ------
    ValueError
        Coverage, content identities, source banks, declarations or owners differ.

    Notes
    -----
    Host-only. The shared configuration authority restores and validates actual
    contents. IDs alone never certify a pair. Historical reversed-team schedules
    remain valid for their older statistics but do not earn this evidence.
    """
    from marl_battlegrounds.evaluation.evaluation_conditions import (
        restore_recorded_config,
    )
    from marl_battlegrounds.evaluation.models import canonical_digest_sha256
    from marl_battlegrounds.evaluation.tournament_statistics import _population
    from marl_battlegrounds.tasks import spawn_locations_for_source

    for row in matches:
        for field in ("episode_id", "outcome"):
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"tournament {field} must be an exact integer")
    rows = {int(cast(int, row["episode_id"])): row for row in matches}
    if len(rows) != len(matches):
        raise ValueError("tournament match rows repeat an episode ID")
    _population(
        schedule,
        {key: int(cast(int, row["outcome"])) for key, row in rows.items()},
        None,
    )
    if any(match.pairing_protocol != "fixed-team-spawn-v1" for match in schedule):
        raise ValueError(
            "physical pair evidence requires the fixed-team spawn protocol"
        )
    configs: dict[str, Any] = {}
    for identifier in {
        value
        for match in schedule
        for value in (match.source_config_id, match.resolved_config_id)
    }:
        if identifier is None or identifier not in configurations:
            raise ValueError(
                "tournament is missing source or resolved configuration content"
            )
        try:
            config, _ = restore_recorded_config(configurations[identifier], identifier)
        except ValueError as error:
            raise ValueError(
                "tournament configuration content does not match its identity"
            ) from error
        configs[identifier] = config
    choices: dict[tuple[str, str], int] = {}
    games: dict[str, dict[str, Any]] = {}
    participants: dict[str, str] = {}
    used_systems: dict[str, Mapping[str, object]] = {}
    for match in schedule:
        row = rows[match.episode_id]
        if any(
            row.get(field) != getattr(match, field)
            for field in (
                "episode_id",
                "map_id",
                "seed_id",
                "block_id",
                "bootstrap_group",
            )
        ):
            raise ValueError("tournament game differs from its scheduled coordinates")
        if (
            row.get("team_a_policy") != match.team_a
            or row.get("team_b_policy") != match.team_b
            or row.get("config_id") != match.resolved_config_id
        ):
            raise ValueError(
                "tournament game ownership or resolved configuration differs"
            )
        pass_key = json.dumps((row["phase"], row["pass_id"]), separators=(",", ":"))
        if pass_key not in passes:
            raise ValueError("tournament game is missing its original pass declaration")
        entry = cast(dict[str, Any], passes[pass_key])
        if entry.get("phase") != row["phase"] or entry.get("pass_id") != row["pass_id"]:
            raise ValueError("tournament pass identity differs from its game origin")
        declaration = entry.get("episodes", {}).get(str(match.episode_id))
        if declaration is None:
            raise ValueError(
                "tournament game is missing its original episode declaration"
            )
        expected = {
            "configuration_digest": match.resolved_config_id,
            "source_config_id": match.source_config_id,
            "spawn_locations": match.spawn_locations,
            "comparison_kind": "verified_spawn_pair",
            "seed_id": match.seed_id,
            "map_id": match.map_id,
            "initial_state_digest": None,
        }
        if any(declaration.get(key) != value for key, value in expected.items()):
            raise ValueError(
                "tournament pair declaration differs from its physical schedule"
            )
        pair = (cast(str, match.source_config_id), cast(str, match.resolved_config_id))
        if pair not in choices:
            compatible, choice = jax.device_get(
                spawn_locations_for_source(configs[pair[1]], configs[pair[0]])
            )
            if not bool(compatible) or int(choice) not in (0, 1):
                raise ValueError(
                    "tournament source must identify one complete spawn choice"
                )
            choices[pair] = int(choice)
        if choices[pair] != match.spawn_locations:
            raise ValueError("tournament configuration uses the wrong spawn banks")
        owners: dict[str, str] = {}
        for team, name in (("team_a", match.team_a), ("team_b", match.team_b)):
            identifier = declaration.get("system_ids", entry.get("system_ids", {})).get(
                team
            )
            if identifier not in systems or systems[identifier].get("name") != name:
                raise ValueError(
                    "tournament participant ownership has no matching registration"
                )
            if identifier not in used_systems:
                if canonical_digest_sha256(systems[identifier]) != identifier:
                    raise ValueError(
                        "tournament System registration differs from its identity"
                    )
                used_systems[identifier] = systems[identifier]
            if participants.setdefault(name, identifier) != identifier:
                raise ValueError("tournament participant version changes between games")
            owners[team + "_system_id"] = identifier
        games[str(match.episode_id)] = {
            **{
                field: row[field]
                for field in ("run_id", "phase", "pass_id", "episode_id")
            },
            **owners,
            "source_config_id": match.source_config_id,
            "resolved_config_id": match.resolved_config_id,
            "spawn_locations": match.spawn_locations,
            "comparison_kind": "verified_spawn_pair",
            "block_id": match.block_id,
        }
    serialized = [asdict(match) for match in schedule]
    return {
        "version": 1,
        "protocol": "fixed-team-spawn-v1",
        "schedule": serialized,
        "schedule_digest": content_digest(serialized),
        "match_digest": match_digest(matches),
        "configurations": dict(configurations),
        "systems": dict(used_systems),
        "passes": dict(passes),
        "games": games,
        "population_system_ids": sorted(used_systems),
        "participants": participants,
    }


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
        if index is not None:
            episode = completed[match.episode_id]
            for name, exact in (
                ("episode_length", episode.episode_length),
                ("team_a_score", episode.team_a_score),
                ("team_b_score", episode.team_b_score),
            ):
                if row[name] != float(np.float32(exact)):
                    raise ValueError(
                        f"{name} measurement differs from the required outcome"
                    )
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


def _check_saved_matchups(
    manifest: Mapping[str, Any],
    groups: Mapping[tuple[str, str], Sequence[TournamentMatch]],
    descriptions: Mapping[str, Mapping[str, object]],
    *,
    legacy: bool,
    seed: int,
    metrics: str,
    full_ids: Sequence[int],
    replay_ids: Sequence[int],
    verify_execution: bool,
) -> None:
    """Check every existing matchup's scientific identity before writer recovery.

    Partial runs may lack later passes or have a pass with no declarations yet.
    Existing declarations may not change source, seed, ownership or capture mode.
    Full mode records every episode in each matchup; other modes record only
    the explicitly selected full-metric episodes. Check these effective saved
    selections without changing the coordinator's original selection.
    Execution batch/chunk sizes remain changeable. No files are changed here.
    Unknown saved tournament passes or conflicting fields raise ValueError.
    """
    current_code: object = None
    if verify_execution:
        from marl_battlegrounds.evaluation.evaluate import capture_recording_provenance

        current_code = capture_recording_provenance()["code_revision"]
    expected_keys = {'["tournament","schedule"]'}
    for index, ((first, second), group) in enumerate(sorted(groups.items()), 1):
        pass_id = f"pair-side-{index}" if legacy else f"pair-{index}"
        key = json.dumps(("tournament", pass_id), separators=(",", ":"))
        expected_keys.add(key)
        entry = manifest["passes"].get(key)
        if entry is None:
            continue
        if entry.get("policies") != {
            "team_a": descriptions[first],
            "team_b": descriptions[second],
        }:
            raise ValueError("saved tournament pass participant identity differs")
        if verify_execution and entry["details"].get("code_revision") != current_code:
            raise ValueError("saved tournament execution source identity differs")
        ids = {match.episode_id for match in group}
        options = {
            "seed": seed,
            "metrics": metrics,
            "full_metrics_episodes": sorted(
                ids if metrics == "full" else ids.intersection(full_ids)
            ),
            "replay_episodes": sorted(ids.intersection(replay_ids)),
            "episode_ids": sorted(ids),
            "num_episodes": len(group),
        }
        if any(entry["details"].get(name) != value for name, value in options.items()):
            raise ValueError("saved tournament pass scientific conditions differ")
        expected = {str(match.episode_id): match for match in group}
        for identifier, declaration in entry.get("episodes", {}).items():
            if identifier not in expected:
                raise ValueError("saved tournament pass contains an unscheduled game")
            match = expected[identifier]
            fields: dict[str, Any] = {
                "episode_id": match.episode_id,
                "seed_id": match.seed_id,
                "map_id": match.map_id,
                "block_id": match.block_id,
                "bootstrap_group": match.bootstrap_group,
                "initial_state_digest": None,
            }
            if not legacy:
                fields.update(
                    configuration_digest=match.resolved_config_id,
                    source_config_id=match.source_config_id,
                    spawn_locations=match.spawn_locations,
                    comparison_kind="verified_spawn_pair",
                )
            if any(declaration.get(name) != value for name, value in fields.items()):
                raise ValueError("saved tournament episode conditions differ")
    if any(
        key not in expected_keys
        for key, entry in manifest["passes"].items()
        if entry.get("phase") == "tournament"
    ):
        raise ValueError("saved tournament contains an unexpected matchup pass")


def _saved_tournament(
    run_dir: Path,
    manifest: dict[str, Any],
    metadata: dict[str, Any],
) -> TournamentResult:
    """Read completed durable summaries without choosing actions or fitting again.

    The caller first checks supplied scientific options and frozen participants.
    Shared read-only loading validates committed table boundaries. Preserve legacy
    result fields; full reports and replays stay file-backed as on a saved run.
    """
    from marl_battlegrounds.evaluation.results import load_results
    from marl_battlegrounds.evaluation.scalar_reports import iter_summary_rows

    load_results(run_dir, phase="tournament")
    tables: dict[str, tuple[dict[str, Any], ...]] = {}
    paths = {"run_details": run_dir / "run_details.json"}
    if (run_dir / "model_calls").is_dir():
        paths["model_calls"] = run_dir / "model_calls"
    for name in (
        "tournament_results",
        "matchup_results",
        "map_results",
        "tournament_headline_metrics",
    ):
        filename = name + ".csv"
        if filename in manifest.get("tables", {}):
            tables[name] = tuple(
                row
                for batch in iter_summary_rows(run_dir / filename, manifest=manifest)
                for row in batch
            )
            paths[name] = run_dir / filename
    for filename in manifest.get("tables", {}):
        path = run_dir / filename
        if path.is_file():
            paths[path.stem] = path
    return TournamentResult(
        tuple(
            sorted(
                _read_matches(run_dir / "match_results.csv"),
                key=lambda row: int(cast(int, row["episode_id"])),
            )
        ),
        tables["tournament_results"],
        tables["matchup_results"],
        tables["map_results"],
        {},
        (),
        {
            **metadata,
            "run_id": manifest["run_id"],
            "configurations": manifest["configurations"],
            "systems": manifest.get("systems", {}),
            "passes": manifest["passes"],
            "statistics": manifest["tournament_summary"]["metadata"],
        },
        paths,
        headline_metrics=tables.get("tournament_headline_metrics", ()),
    )


def run_tournament(
    policies: Sequence[System | Policy | str] | None = None,
    *,
    config: str | Path | Mapping[str, Any] | None = None,
    save_replays: int | Omitted = OMITTED,
    maps: Sequence[int | TDMMapInfo] | None | Omitted = OMITTED,
    episodes_per_pair: int | Omitted = OMITTED,
    seed: int | Omitted = OMITTED,
    num_envs: int = 128,
    metrics: MetricMode | Omitted = OMITTED,
    full_metrics_episodes: Iterable[int] | Omitted = OMITTED,
    replay_episodes: Iterable[int] | Omitted = OMITTED,
    output_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
    opponent_weights: Mapping[str, float] | None | Omitted = OMITTED,
    score_threshold: int | Omitted = OMITTED,
    max_steps: int | Omitted = OMITTED,
    red_zone_depth: float | Omitted = OMITTED,
    chunk_size: int = 16,
) -> TournamentResult:
    """Evaluate every participant pair with fixed teams and both spawn locations.

    Parameters
    ----------
    policies : Sequence[System | Policy | str] or None
        At least two Systems, Policies or built-in names. Entrant labels must
        be distinct. Numerical variables and initial memory templates are
        snapshotted once; opaque external providers remain caller-owned.
    config : JSON path, mapping or None
        Complete custom tournament population and conditions. Use this or a
        policies sequence, not both. Scientific programmatic options must stay
        omitted on this route. Saved configuration runs inherit their config.
    save_replays : int, default=0
        First N scheduled games to capture. A nonempty explicit selection must
        agree with this shorthand. Configuration runs use logical game IDs.
    maps : Sequence[int | TDMMapInfo] | None
        Optional distinct integer map IDs or TDMMapInfo entries. None uses
        canonical evaluation maps 47..51 for a new run. Omitted maps on resume
        reuse recorded configurations, without consulting current map assets.
        Exact EnvConfig inputs are not accepted by this generic runner.
    episodes_per_pair : int
        Positive total across maps and both spawn choices, normally 100.
        Must be divisible by twice the map count. Omitted scientific settings
        inherit the selected saved tournament on resume. Explicit conflicts fail.
    seed : int
        Root uint32 integer, default 0; also seeds summary resampling.
    num_envs : int
        Positive maximum simultaneous games per fixed-participant matchup,
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
    red_zone_depth : float, default=5.0 (DEFAULT_TDM_RED_ZONE_DEPTH)
        Red Zone depth in map units for every map. When an agent dies inside
        its own team's Red Zone, the enemy team gets 2 points instead of 1; it
        is still one kill and one death. 0.0 keeps one point per death. Must
        be a Python float. New runs record it as "red_zone_depth" in their
        metadata. On resume, omission inherits the saved run; a run saved
        before the Red Zone rule reads as 0.0 and keeps its original
        configuration IDs. A supplied value, even 5.0, must match the saved
        run. The config route owns its rules, so any supplied value is
        refused there.
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
        or the complete match population are invalid, a supplied
        red_zone_depth differs from the saved run ("red_zone_depth differs
        from the saved evaluation conditions") or Core rejects it, or the
        config route receives red_zone_depth. These depth errors are raised
        before any file is written.
    TypeError
        A config or policy input/output violates its type contract, or
        red_zone_depth is not a Python float.
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
    hunter, rogue, priest for both teams. Within each matchup the sorted first
    entrant stays Team A and the other
    stays Team B. Paired games exchange complete spawn banks and have fresh
    episode memory. Priority/full runs produce participant headline summaries;
    none skips them, even with selected full reports. Completed saved summaries
    are loaded without fitting again. Historical reversed-team schedules retain
    their original interpretation and receive no fabricated spawn-pair headlines.
    Configuration runs support verified game reuse through the shared canonical
    machinery. This call does not perform admission or publish official releases.

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
    from marl_battlegrounds.evaluation.evaluation_conditions import (
        config_record,
        option,
        read_saved_pass,
        red_zone_depth_option,
        restore_recorded_config,
    )
    from marl_battlegrounds.evaluation.recording_identity import (
        normalize_system_registration,
    )
    from marl_battlegrounds.evaluation.system_evaluation import (
        freeze_evaluation_method,
        prepare_evaluation_system,
        validate_evaluation_rosters,
    )
    from marl_battlegrounds.tasks import _swap_spawn_banks, _validate_config_choices

    if output_dir is not None and resume_from is not None:
        raise ValueError("output_dir and resume_from are mutually exclusive")
    configured_resume = False
    if resume_from is not None:
        from marl_battlegrounds.evaluation.results import _read_manifest

        configured_resume = "tournament_reuse" in _read_manifest(Path(resume_from))
    if config is not None or configured_resume:
        from marl_battlegrounds.evaluation.canonical import _run_configured_tournament

        if policies is not None:
            raise ValueError("Supply policies or config, not competing populations")
        explicit_rules = {
            "maps": maps,
            "episodes_per_pair": episodes_per_pair,
            "seed": seed,
            "opponent_weights": opponent_weights,
            "score_threshold": score_threshold,
            "max_steps": max_steps,
            "red_zone_depth": red_zone_depth,
        }
        conflicts = [
            name
            for name, value in explicit_rules.items()
            if not isinstance(value, Omitted)
        ]
        if conflicts:
            raise ValueError(
                "Config owns scientific settings; omit " + ", ".join(conflicts)
            )
        return _run_configured_tournament(
            config,
            metrics=metrics,
            full_metrics_episodes=full_metrics_episodes,
            replay_episodes=replay_episodes,
            save_replays=save_replays,
            output_dir=output_dir,
            resume_from=resume_from,
            num_envs=num_envs,
            chunk_size=chunk_size,
        )
    if policies is None:
        raise ValueError("Supply a policies sequence or a tournament config")
    saved = read_saved_pass(resume_from, None, "tournament", "schedule")
    saved_details = None if saved is None else saved[1]["details"]
    seed = cast(int, option(seed, saved_details, "seed", 0))
    metrics = cast(MetricMode, option(metrics, saved_details, "metrics", "priority"))
    episodes_per_pair = cast(
        int, option(episodes_per_pair, saved_details, "episodes_per_pair", 100)
    )
    score_threshold = cast(
        int, option(score_threshold, saved_details, "score_threshold", 20)
    )
    max_steps = cast(int, option(max_steps, saved_details, "max_steps", 300))
    # A run saved before the Red Zone rule has no red_zone_depth: it reads 0.0.
    red_zone_depth = red_zone_depth_option(red_zone_depth, saved_details)
    full_metrics_episodes = cast(
        Iterable[int],
        option(full_metrics_episodes, saved_details, "full_metrics_episodes", ()),
    )
    replay_episodes = cast(
        Iterable[int], option(replay_episodes, saved_details, "replay_episodes", ())
    )
    opponent_weights = cast(
        Mapping[str, float] | None,
        option(opponent_weights, saved_details, "opponent_weights", None),
    )
    supplied_maps = maps
    maps = None if isinstance(maps, Omitted) else maps
    map_ids = tuple(
        current_map_id(item) if isinstance(item, TDMMapInfo) else item
        for item in (CANONICAL_TDM_EVALUATION_MAP_IDS if maps is None else maps)
    )
    if saved_details is not None:
        if (
            not isinstance(supplied_maps, Omitted)
            and list(sorted(map_ids)) != saved_details["map_ids"]
        ):
            raise ValueError("maps differ from the saved tournament conditions")
        map_ids = tuple(saved_details["map_ids"])
    entrants = tuple(freeze_evaluation_method(item) for item in policies)
    frozen = {entrant.name: entrant for entrant in entrants}
    schedule = build_tournament_schedule(
        [entrant.name for entrant in entrants],
        map_ids,
        episodes_per_pair=episodes_per_pair,
    )
    from marl_battlegrounds.evaluation.evaluation_conditions import capture_ids

    if not isinstance(save_replays, Omitted):
        replay_episodes = capture_ids(
            [match.episode_id for match in schedule], replay_episodes, save_replays
        )
    validate_opponent_weights(tuple(sorted(frozen)), opponent_weights)
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
    for name, identifiers in (
        ("full_metrics_episodes", selected.full_metrics_episodes),
        ("replay_episodes", selected.replay_episodes),
    ):
        if any(identifier > len(schedule) for identifier in identifiers):
            raise ValueError(f"{name} contains an episode outside the schedule")
    descriptions = [
        policy_description(
            team, team.variables, team.initial_carry, include_digests=True
        )
        if isinstance(team, Policy)
        else normalize_system_registration(team, phase="tournament", frozen=True)[1]
        for _, team in sorted(frozen.items())
    ]
    if saved_details is not None and descriptions != saved_details.get("policies"):
        raise ValueError(
            "tournament participant identity differs from the saved frozen registration"
        )
    systems: dict[str, dict[str, object]] = {}
    participants: dict[str, str] = {}
    for description in descriptions:
        identifier, registration = normalize_system_registration(
            description, phase="tournament"
        )
        systems[identifier] = registration
        participants[str(description["name"])] = identifier
    legacy = saved_details is not None and saved_details.get("pairing_protocol") is None
    if (
        saved_details is not None
        and not legacy
        and saved_details.get("pairing_protocol") != "fixed-team-spawn-v1"
    ):
        raise ValueError("unknown saved tournament pairing protocol")
    configurations: dict[str, Any] = {}
    # A saved run may predate the Red Zone rule: its 12-key configurations
    # restore at depth 0.0 and keep their original (historical) identities.
    historical_by_map: dict[int, bool] = {}
    if saved is not None:
        assert saved_details is not None
        configurations.update(saved[0]["configurations"])
        restored = {
            int(map_id): restore_recorded_config(configurations[identifier], identifier)
            for map_id, identifier in saved_details["configuration_ids_by_map"].items()
        }
        configs = {map_id: config for map_id, (config, _) in restored.items()}
        historical_by_map = {
            map_id: historical for map_id, (_, historical) in restored.items()
        }
        if not isinstance(supplied_maps, Omitted):
            for current in normalize_episode_specs(
                sorted(map_ids),
                len(map_ids),
                _ROSTER_A,
                _ROSTER_B,
                score_threshold,
                max_steps,
                red_zone_depth=red_zone_depth,
            ):
                recorded = saved_details["configuration_ids_by_map"].get(
                    str(current.map_id)
                )
                if not _config_matches_identity(
                    current.env_config,
                    recorded,
                    historical=historical_by_map.get(cast(int, current.map_id), False),
                ):
                    raise ValueError(
                        "explicit map contents differ from the saved tournament source"
                    )
        if not legacy:
            schedule = tuple(
                TournamentMatch(**record) for record in saved_details["schedule"]
            )
        else:
            schedule = tuple(
                replace(
                    match,
                    team_a=match.team_b if match.spawn_locations == 1 else match.team_a,
                    team_b=match.team_a if match.spawn_locations == 1 else match.team_b,
                    pairing_protocol=None,
                    spawn_locations=None,
                )
                for match in schedule
            )
    else:
        configs = {
            spec.map_id: spec.env_config
            for spec in normalize_episode_specs(
                sorted(map_ids),
                len(map_ids),
                _ROSTER_A,
                _ROSTER_B,
                score_threshold,
                max_steps,
                red_zone_depth=red_zone_depth,
            )
        }
    resolved: dict[tuple[int, int], Any] = {}
    executions = {
        name: prepare_evaluation_system(entrant)[0] for name, entrant in frozen.items()
    }
    for env_config in configs.values():
        for name_a, name_b in {(match.team_a, match.team_b) for match in schedule}:
            validate_evaluation_rosters(
                executions[name_a], executions[name_b], env_config
            )
    source_ids: dict[int, str] = {}
    resolved_ids: dict[tuple[int, int], str] = {}
    for map_id, env_config in configs.items():
        _validate_config_choices(
            env_config, batched=False, both_spawn_choices=not legacy
        )
        historical = historical_by_map.get(cast(int, map_id), False)
        identifier, content = config_record(env_config, historical=historical)
        configurations[identifier] = content
        source_ids[cast(int, map_id)] = identifier
        resolved[cast(int, map_id), 0] = env_config
        resolved_ids[cast(int, map_id), 0] = identifier
        if not legacy:
            exchanged = _swap_spawn_banks(env_config)
            swapped_id, swapped_content = config_record(
                exchanged, historical=historical
            )
            configurations[swapped_id] = swapped_content
            resolved[cast(int, map_id), 1] = exchanged
            resolved_ids[cast(int, map_id), 1] = swapped_id
    if not legacy:
        calculated = tuple(
            replace(
                match,
                source_config_id=source_ids[match.map_id],
                resolved_config_id=resolved_ids[
                    match.map_id, cast(int, match.spawn_locations)
                ],
            )
            for match in schedule
        )
        if saved is not None and calculated != schedule:
            raise ValueError(
                "saved tournament schedule differs from its configuration content"
            )
        schedule = calculated
    groups: dict[tuple[str, str], list[TournamentMatch]] = defaultdict(list)
    for match in schedule:
        groups[match.team_a, match.team_b].append(match)
    metadata: dict[str, Any] = (
        dict(saved_details)
        if saved_details is not None
        else {
            "seed": seed,
            "rng_protocol": "evaluation-systems-v1",
            "policies": descriptions,
            "map_ids": sorted(map_ids),
            "episodes_per_pair": episodes_per_pair,
            "num_matches": len(schedule),
            "score_threshold": score_threshold,
            "max_steps": max_steps,
            "red_zone_depth": red_zone_depth,
            "metrics": metrics,
            "full_metrics_episodes": list(selected.full_metrics_episodes),
            "replay_episodes": list(selected.replay_episodes),
            "opponent_weights": None
            if opponent_weights is None
            else dict(opponent_weights),
            "configuration_ids_by_map": {
                str(key): value for key, value in source_ids.items()
            },
            "pairing_protocol": "fixed-team-spawn-v1",
            "schedule": [asdict(match) for match in schedule],
            "schedule_digest": content_digest([asdict(match) for match in schedule]),
            "participants": participants,
        }
    )
    specs_by_group: dict[tuple[str, str], list[EpisodeSpec]] = {}
    for pair, group in sorted(groups.items()):
        specs = [
            EpisodeSpec(
                match.episode_id,
                configs[match.map_id]
                if legacy
                else resolved[match.map_id, cast(int, match.spawn_locations)],
                match.map_id,
                match.seed_id,
                metadata={
                    "block_id": match.block_id,
                    "bootstrap_group": match.bootstrap_group,
                    **(
                        {"paired_comparison_key": f"block-{match.block_id}"}
                        if legacy
                        else {}
                    ),
                },
                source_config=None if legacy else configs[match.map_id],
                spawn_locations=match.spawn_locations,
                paired_comparison_key=None if legacy else f"block-{match.block_id}",
            )
            for match in group
        ]
        specs_by_group[pair] = specs
    if saved is not None:
        _check_saved_matchups(
            saved[0],
            groups,
            {str(value["name"]): value for value in descriptions},
            legacy=legacy,
            seed=seed,
            metrics=metrics,
            full_ids=selected.full_metrics_episodes,
            replay_ids=selected.replay_episodes,
            verify_execution=saved[0].get("tournament_summary") is None,
        )
    if saved is not None and saved[0].get("tournament_summary") is not None:
        assert resume_from is not None
        manifest = saved[0]
        if not legacy and any(
            entry.get("phase") == "tournament"
            and entry.get("result_state", {}).get("status") != "complete"
            for entry in manifest["passes"].values()
        ):
            with RunWriter(
                resume_from=resume_from,
                phase="tournament",
                pass_id="schedule",
                details=metadata,
            ) as finalizer:
                finalizer.reaffirm_tournament_result(str(metadata["schedule_digest"]))
            manifest = json.loads((Path(resume_from) / "run_details.json").read_bytes())
        return _saved_tournament(Path(resume_from), manifest, metadata)
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
            if not legacy:
                writer.set_tournament_coordinator(str(metadata["schedule_digest"]))
        run_id = writer.run_id if writer is not None else "in-memory-" + uuid4().hex
        metadata = {**metadata, "run_id": run_id}
        passes: dict[str, Any] = {}
        matches: list[ResultRow] = []
        completion_order: list[int] = []
        full_tables: list[Columns] = []
        replays: list[ReplayArtifactV4] = []
        leased: set[int] = set()
        for index, ((first, second), group) in enumerate(sorted(groups.items()), 1):
            group_ids = {match.episode_id for match in group}
            pass_id = f"pair-side-{index}" if legacy else f"pair-{index}"
            pass_key = json.dumps(("tournament", pass_id), separators=(",", ":"))
            completed: set[int] = (
                set(
                    saved[0]["passes"]
                    .get(pass_key, {})
                    .get("completed_episode_ids", ())
                )
                if saved is not None
                else set()
            )
            if group_ids - completed:
                for name in (first, second):
                    method = frozen[name]
                    if (
                        isinstance(method, System)
                        and method.resource_scope is not None
                        and id(method) not in leased
                    ):
                        cleanup.enter_context(method.resource_scope(writer is not None))
                        leased.add(id(method))
            execute = evaluate_episodes
            if legacy:
                from marl_battlegrounds.evaluation.evaluate import _run_evaluation

                execute = _run_evaluation
            result = execute(
                frozen[first],
                frozen[second],
                specs_by_group[first, second],
                seed=seed,
                num_envs=num_envs,
                metrics=metrics,
                full_metrics_episodes=group_ids.intersection(
                    selected.full_metrics_episodes
                ),
                replay_episodes=group_ids.intersection(selected.replay_episodes),
                writer=writer,
                phase="tournament",
                pass_id=pass_id,
                chunk_size=chunk_size,
                run_id=run_id,
            )
            completion_order.extend(
                cast(Sequence[int], result.metadata.get("completion_order", ()))
            )
            configurations.update(
                cast(dict[str, Any], result.metadata["configurations"])
            )
            systems.update(cast(dict[str, Any], result.metadata.get("systems", {})))
            entry = result.metadata
            declared_schedule: Any = entry.get("schedule", {})
            if isinstance(declared_schedule, (tuple, list)):
                declared_schedule = {
                    str(row["episode_id"]): row
                    for row in cast(Sequence[dict[str, Any]], declared_schedule)
                }
            pass_key = json.dumps(
                ("tournament", entry["pass_id"]), separators=(",", ":")
            )
            passes[pass_key] = {
                "phase": "tournament",
                "pass_id": entry["pass_id"],
                "details": entry,
                "system_ids": entry.get("system_ids", {}),
                "episodes": {
                    str(key): value
                    for key, value in cast(dict[int, Any], declared_schedule).items()
                },
                "completed_episode_ids": sorted(group_ids),
                "recorded_metrics_by_episode": {
                    str(identifier): "full"
                    if metrics == "full" or identifier in selected.full_metrics_episodes
                    else metrics
                    for identifier in group_ids
                },
                "result_state": {
                    "version": 1,
                    "status": "complete",
                    "schedule_digest": entry.get("schedule_digest"),
                    "reason": None,
                },
            }
            if writer is None:
                matches.extend(_memory_matches(result, group))
                full_tables.append(result.full_metrics)
                replays.extend(result.replays)
        if writer is not None:
            matches = _read_matches(writer.paths["match_results"])
            manifest = json.loads((writer.run_dir / "run_details.json").read_bytes())
            configurations.update(manifest["configurations"])
            systems.update(manifest["systems"])
            passes = manifest["passes"]
        matches.sort(key=lambda row: int(cast(int, row["episode_id"])))
        outcomes = {
            int(cast(int, row["episode_id"])): int(cast(int, row["outcome"]))
            for row in matches
        }
        if len(outcomes) != len(matches):
            raise ValueError("tournament match table repeats an episode identity")
        evidence = (
            None
            if legacy
            else _prepare_pair_evidence(
                schedule,
                matches,
                configurations=configurations,
                systems=systems,
                passes=passes,
            )
        )
        statistics = summarize_tournament(
            schedule, outcomes, seed=seed, opponent_weights=opponent_weights
        )
        headline = (
            None
            if legacy or metrics == "none"
            else summarize_headlines(matches, cast(dict[str, Any], evidence))
        )
        qualification = (
            None
            if legacy
            else {
                "version": 1,
                "status": "complete",
                "schedule_digest": metadata["schedule_digest"],
                "population_system_ids": sorted(participants.values()),
                "pairing_protocol": "fixed-team-spawn-v1",
                "metrics": metrics,
                "evidence": evidence,
            }
        )
        if writer is not None:
            writer.write_tournament_results(
                statistics, headline=headline, qualification=qualification
            )
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
                "systems": systems,
                "participants": participants,
                "passes": passes,
                "statistics": statistics.metadata,
                "completion_order": completion_order,
                "tournament_completion": qualification,
                "tournament_summary": {
                    "digest": "in-memory-qualified",
                    "qualification": qualification,
                },
            },
            None if writer is None else writer.paths,
            headline_metrics=() if headline is None else headline,
        )
