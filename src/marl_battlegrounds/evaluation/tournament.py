"""Normalize tournament inputs and preserve historical resume contracts.

run_tournament sends fresh entrant lists, short declarations and full configs
through the shared configured runner. Its private historical adapter finishes
old list-tournament folders without migrating their files or random streams.
Statistics, method loading, evaluation and result writing retain shared owners.
"""


# Private setup and population helpers are shared authorities, not parallel rules.
# pyright: reportPrivateUsage=false, reportUnusedFunction=false

import json
from collections import defaultdict
from collections.abc import Generator, Iterable, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, replace
from numbers import Integral
from pathlib import Path
from typing import Any, cast

import jax
import numpy as np

from marl_battlegrounds._tdm_assets import current_map_id
from marl_battlegrounds.core.types import EnvConfig
from marl_battlegrounds.environment import MetricMode, _episode_selection, make
from marl_battlegrounds.evaluation.evaluate import (
    Columns,
    EpisodeSpec,
    EvaluationResult,
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
    list_tdm_maps,
)

_ROSTER_A, _ROSTER_B = canonical_tournament_rosters()


def _saved_configurations(
    manifest: Mapping[str, Any], details: Mapping[str, Any]
) -> dict[str, Any]:
    """Recover pinned map contents from coordinator, pass and recorded snapshots.

    manifest and details are checked saved-run metadata. Return a new mapping;
    neither input is changed. Every duplicate identity must have equal content.
    Missing planned source maps raise an actionable ValueError before recovery.
    Callers restore required content through the shared configuration authority;
    this helper never selects maps from the current catalog.
    """
    contents: dict[str, Any] = {}
    snapshots = [manifest.get("configurations", {}), details.get("configurations", {})]
    snapshots.extend(
        entry.get("details", {}).get("configurations", {})
        for entry in manifest["passes"].values()
        if entry.get("phase") == "tournament"
    )
    for snapshot in snapshots:
        if not isinstance(snapshot, dict):
            raise ValueError("Saved tournament configuration snapshot is invalid")
        for identifier, content in cast(dict[str, Any], snapshot).items():
            if identifier in contents and contents[identifier] != content:
                raise ValueError("Saved tournament configuration snapshots disagree")
            contents[identifier] = content
    for map_id, identifier in details["configuration_ids_by_map"].items():
        if identifier not in contents:
            raise ValueError(
                f"Saved tournament lacks configuration {identifier} for map {map_id}; "
                "restore its original run files or start a new tournament. "
                "Current map content cannot replace missing saved content."
            )
    return contents


def _saved_registered_maps(
    manifest: Mapping[str, Any], details: Mapping[str, Any], configs: Mapping[int, Any]
) -> dict[int, Mapping[str, Any]]:
    """Recover and check the pinned registration of every unfinished source map.

    New coordinators save serialized TDMMapInfo rows. Older runs may retain only
    exact map names and splits in episode declarations; match those identities
    against approved current/history records and check their saved geometry.
    Never choose a replacement by integer ID alone. Missing or conflicting
    evidence raises ValueError before opening the writer. Completed runs do not
    need this recovery helper. Inputs and historical records stay unchanged.
    """
    from marl_battlegrounds._tdm_assets import map_history
    from marl_battlegrounds.evaluation.catalog import build_resolved_env_config_v2
    from marl_battlegrounds.evaluation.map_identity import _snapshot_map_metadata

    saved = details.get("registered_maps")
    result: dict[int, Mapping[str, Any]] = {}
    if saved is not None:
        if not isinstance(saved, dict):
            raise ValueError("Saved tournament map registrations must be a mapping")
        saved = cast(dict[str, Any], saved)
        if set(saved) != {str(key) for key in configs}:
            raise ValueError("Saved tournament map registrations differ from its plan")
        result = {int(key): value for key, value in saved.items()}
    else:
        names: dict[int, tuple[str, str]] = {}
        for entry in manifest["passes"].values():
            if entry.get("phase") != "tournament":
                continue
            for row in entry.get("episodes", {}).values():
                metadata = {
                    item["name"]: item["value"] for item in row.get("map_metadata", [])
                }
                if metadata.get("map_origin") != "registered":
                    continue
                map_id = row["map_id"]
                identity = (metadata["map_name"], metadata["map_split"])
                if map_id in names and names[map_id] != identity:
                    raise ValueError("Saved tournament map registrations disagree")
                names[map_id] = identity
        approved = (*list_tdm_maps(), *(row.info for row in map_history()))
        for map_id, config in configs.items():
            identity = names.get(map_id)
            source = build_resolved_env_config_v2(config)
            for info in approved:
                if info.map_id != map_id or (info.name, info.split) != identity:
                    continue
                declared = info.model_dump(mode="json")
                try:
                    _snapshot_map_metadata(map_id, source, declared)
                except ValueError:
                    # Retained revisions can share a name but have other geometry.
                    continue
                result[map_id] = declared
                break
            else:
                raise ValueError(
                    f"Saved tournament lacks pinned registration for map {map_id}; "
                    "restore its original run files or start a new tournament"
                )
        return result
    for map_id, config in configs.items():
        _snapshot_map_metadata(
            map_id, build_resolved_env_config_v2(config), result[map_id]
        )
    return result


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


def _resume_legacy_tournament(
    policies: Sequence[System | Policy | str] | None = None,
    *,
    save_replays: int | Omitted = OMITTED,
    maps: Sequence[int | TDMMapInfo] | None | Omitted = OMITTED,
    episodes_per_pair: int | Omitted = OMITTED,
    seed: int | Omitted = OMITTED,
    num_envs: int = 128,
    metrics: MetricMode | Omitted = OMITTED,
    full_metrics_episodes: Iterable[int] | Omitted = OMITTED,
    replay_episodes: Iterable[int] | Omitted = OMITTED,
    resume_from: str | Path | None,
    opponent_weights: Mapping[str, float] | None | Omitted = OMITTED,
    score_threshold: int | Omitted = OMITTED,
    max_steps: int | Omitted = OMITTED,
    red_zone_depth: float | Omitted = OMITTED,
    chunk_size: int = 16,
) -> TournamentResult:
    """Resume a historical list tournament without changing its saved format.

    Parameters
    ----------
    policies : sequence of System, Policy or str
        Original frozen entrants, in any order. Their identities must match the
        saved registrations. Each distinct System needed by unfinished jobs keeps
        one resource scope until execution ends, closing before the writer.
        Opaque external providers remain caller-owned.
    resume_from : path
        Exact old-format run folder. Required; this adapter cannot start a run.
    save_replays, maps, episodes_per_pair, seed, metrics : saved options
        Omitted values inherit the saved declaration. Supplied values assert
        those same conditions; conflicting settings fail before writer recovery.
    full_metrics_episodes, replay_episodes : iterable of int or Omitted
        Saved global capture selections, inherited when omitted.
    opponent_weights, score_threshold, max_steps, red_zone_depth : saved options
        Scientific conditions that must remain equal to the saved run. A run
        saved before Red Zone keeps depth 0.0 and historical configuration IDs.
    num_envs : int, default=128
        Positive execution capacity. Changing it does not change planned games.
    chunk_size : int, default=16
        Positive number of ticks per evaluator chunk, allowed to change on resume.

    Returns
    -------
    TournamentResult
        Historical fields and recorded summaries. Completed results are read
        without fitting or execution. New captures remain file-backed.

    Raises
    ------
    ValueError
        The folder, original entrants or saved scientific settings are invalid.
        Missing historical map content is never replaced with current content.
    TypeError
        A supplied method or option violates its existing type contract.
    RuntimeError, OSError
        Evaluation, fitting or recording fails. The failed run stays resumable.

    Notes
    -----
    Saved formats retain their original passes, game IDs, random streams and
    interpretation. Reversed-team games keep the old evaluator contract and
    receive no fabricated spawn-pair headlines. The shared job executor owns
    sequential dispatch; this adapter only reads and finishes the old format.
    It neither migrates folders nor installs canonical configuration records.
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
        prepare_evaluation_system,
        validate_evaluation_rosters,
    )
    from marl_battlegrounds.tasks import _swap_spawn_banks, _validate_config_choices

    if resume_from is None:
        raise ValueError("Historical tournament recovery requires resume_from")
    if policies is None:
        raise ValueError("Supply the original historical tournament policies")
    saved = read_saved_pass(resume_from, None, "tournament", "schedule")
    if saved is None:
        raise ValueError("Historical tournament recovery requires a saved schedule")
    saved_details = cast(dict[str, Any], saved[1]["details"])
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
    if (
        not isinstance(supplied_maps, Omitted)
        and list(sorted(map_ids)) != saved_details["map_ids"]
    ):
        raise ValueError("maps differ from the saved tournament conditions")
    map_ids = tuple(saved_details["map_ids"])
    from marl_battlegrounds.evaluation.sampling_evidence import method_sampling_fact
    from marl_battlegrounds.evaluation.tournament_inputs import (
        _prepare_tournament_method,
    )

    entrants = tuple(_prepare_tournament_method(item) for item in policies)
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
    if descriptions != saved_details.get("policies"):
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
    legacy = saved_details.get("pairing_protocol") is None
    if not legacy and saved_details.get("pairing_protocol") != "fixed-team-spawn-v1":
        raise ValueError("unknown saved tournament pairing protocol")
    configurations: dict[str, Any] = {}
    # A saved run may predate the Red Zone rule: its 12-key configurations
    # restore at depth 0.0 and keep their original (historical) identities.
    historical_by_map: dict[int, bool] = {}
    configurations.update(_saved_configurations(saved[0], saved_details))
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
    registered_maps: dict[int, Mapping[str, Any]] = {}
    if not legacy and saved[0].get("tournament_summary") is None:
        registered_maps = _saved_registered_maps(saved[0], saved_details, configs)
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
        historical = historical_by_map.get(map_id, False)
        identifier, content = config_record(env_config, historical=historical)
        configurations[identifier] = content
        source_ids[map_id] = identifier
        resolved[map_id, 0] = env_config
        resolved_ids[map_id, 0] = identifier
        if not legacy:
            exchanged = _swap_spawn_banks(env_config)
            swapped_id, swapped_content = config_record(
                exchanged, historical=historical
            )
            configurations[swapped_id] = swapped_content
            resolved[map_id, 1] = exchanged
            resolved_ids[map_id, 1] = swapped_id
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
        if calculated != schedule:
            raise ValueError(
                "saved tournament schedule differs from its configuration content"
            )
        schedule = calculated
    groups: dict[tuple[str, str], list[TournamentMatch]] = defaultdict(list)
    for match in schedule:
        groups[match.team_a, match.team_b].append(match)
    metadata: dict[str, Any] = dict(saved_details)
    saved_sampling = metadata.get("method_sampling")
    if saved_sampling is not None and saved_sampling != {
        name: method_sampling_fact(method) for name, method in frozen.items()
    }:
        raise ValueError(
            "Tournament sampling facts differ from the saved frozen methods"
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
    if saved[0].get("tournament_summary") is not None:
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
    with (
        RunWriter(
            resume_from=resume_from,
            phase="tournament",
            pass_id="schedule",
            details=metadata,
        ) as writer,
        ExitStack() as cleanup,
    ):
        if not legacy:
            writer.set_tournament_coordinator(str(metadata["schedule_digest"]))
        run_id = writer.run_id
        metadata = {**metadata, "run_id": run_id}
        completion_order: list[int] = []
        leased: set[int] = set()
        from marl_battlegrounds.evaluation.tournament_execution import (
            TournamentJob,
            execute_tournament_jobs,
        )

        jobs = (
            TournamentJob(
                index,
                first,
                second,
                tuple(specs_by_group[first, second]),
                seed,
                "tournament",
                f"pair-side-{index}" if legacy else f"pair-{index}",
                tuple(
                    sorted(
                        {match.episode_id for match in group}.intersection(
                            selected.full_metrics_episodes
                        )
                    )
                ),
                tuple(
                    sorted(
                        {match.episode_id for match in group}.intersection(
                            selected.replay_episodes
                        )
                    )
                ),
                legacy,
            )
            for index, ((first, second), group) in enumerate(sorted(groups.items()), 1)
        )

        @contextmanager
        def legacy_pair(
            job: TournamentJob,
        ) -> Generator[tuple[System | Policy, System | Policy]]:
            """Lease only unfinished historical games; share clients across pairs."""
            pass_key = json.dumps((job.phase, job.pass_id), separators=(",", ":"))
            completed = set(
                saved[0]["passes"].get(pass_key, {}).get("completed_episode_ids", ())
            )
            if any(spec.episode_id not in completed for spec in job.episodes):
                for name in (job.first, job.second):
                    method = frozen[name]
                    if (
                        isinstance(method, System)
                        and method.resource_scope is not None
                        and id(method) not in leased
                    ):
                        cleanup.enter_context(method.resource_scope(True))
                        leased.add(id(method))
            yield frozen[job.first], frozen[job.second]

        for _job, result in execute_tournament_jobs(
            jobs,
            legacy_pair,
            writer=writer,
            run_id=run_id,
            num_envs=num_envs,
            metrics=metrics,
            chunk_size=chunk_size,
            registered_maps=registered_maps,
        ):
            completion_order.extend(
                cast(Sequence[int], result.metadata.get("completion_order", ()))
            )
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
            schedule,
            outcomes,
            seed=seed,
            opponent_weights=opponent_weights,
            method_sampling=saved_sampling,
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
        writer.write_tournament_results(
            statistics, headline=headline, qualification=qualification
        )
        return TournamentResult(
            tuple(matches),
            statistics.tournament_results,
            statistics.matchup_results,
            statistics.map_results,
            {},
            (),
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
            writer.paths,
            headline_metrics=() if headline is None else headline,
        )


def run_tournament(
    policies: Sequence[System | Policy | str] | None = None,
    *,
    config: str | Path | Mapping[str, Any] | None = None,
    challenger: System | Policy | str | None | Omitted = OMITTED,
    games_per_opponent: int | Omitted = OMITTED,
    episodes_per_pair: int | Omitted = OMITTED,
    save_replays: int | Omitted = OMITTED,
    maps: Sequence[int | TDMMapInfo] | None | Omitted = OMITTED,
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
    rerun_existing: bool | Omitted = OMITTED,
    chunk_size: int = 16,
) -> TournamentResult:
    """Run one round robin from methods, a short config or a saved full field.

    Parameters
    ----------
    policies : sequence of System, Policy or str, optional
        At least two entrants with distinct names. Strings use the shared loader:
        built-in names, actor export/checkpoint folders or trusted module:function
        factories. Live objects stay in this process and are never serialized.
        Use this sequence or config, not both.
    config : path or mapping, optional
        Short field with entrants and optional maps, rosters, rules, budget,
        seed/seeds, output settings and a pre-game selection declaration; or a
        complete saved tournament descriptor. Selection freezes the candidate
        field and budget before validation, and a frozen population before tests.
        Relative folder and output paths resolve beside its JSON file. A mapping
        uses the current directory. Equal explicit values may use different
        sequence types; capture IDs compare as sorted unique selections. Unknown
        fields and conflicting settings fail.
    challenger : System, Policy, str or None, optional
        Add one method as Team A against each field member, exchanging spawn
        ends between paired games. None means no challenger. Omission on resume
        inherits a saved reloadable challenger; a live one must be supplied again.
    games_per_opponent, episodes_per_pair : int, optional
        Total games per unordered pair across all maps and spawn ends. New lists
        default to 100. The total must divide equally over maps and spawn pairs.
        episodes_per_pair is the retained alias; equal simultaneous values pass,
        conflicts fail. Saved and configured budgets are inherited on omission.
    maps : sequence of int or TDMMapInfo, or None, optional
        Distinct map IDs or checked registered map descriptions. None selects
        test maps 47 through 51 for a new list. Resume uses saved map contents.
    seed : int, optional
        Root uint32 random seed and bootstrap seed; new lists default to 0.
        Short configs may instead declare distinct roots in seeds; the total
        budget is shared equally across them, not multiplied.
    num_envs : int, default=128
        Maximum parallel games in the active matchup. Matchups run in sequence.
        Execution settings may change on resume without changing planned games.
    metrics : {'priority', 'full', 'none'}, optional
        Default priority. None-mode is the string 'none'; outcomes and ratings
        remain available while optional metrics are skipped.
    full_metrics_episodes, replay_episodes : iterable of int, optional
        Logical scheduled game IDs to capture; empty by default. Original replay
        execution IDs remain unchanged. Explicit full selections work in none mode.
    save_replays : int, optional
        Capture the first N scheduled games, default 0. An explicit replay list
        must agree with a nonzero shorthand.
    output_dir, resume_from : path or None
        Parent for a new run or the exact saved run to resume; choose at most one.
        With neither, results stay in memory and no run files are created.
    opponent_weights : mapping or None, optional
        Positive weight per entrant name for rate/tail summaries. None gives
        equal weights. It does not change the joint rating fit.
    score_threshold, max_steps : int, optional
        New-list score target and horizon in ticks; defaults 20 and 300.
    red_zone_depth : float, optional
        New-list Red Zone depth in map units, default 5.0. An own-zone death
        gives the other team 2 points. 0.0 keeps one point per death. Historical
        runs keep their saved rules; loading an actor never changes these rules.
    rerun_existing : bool, optional
        Explicit fresh execution of a configured field, default False. Reuse
        requires matching conditions and checked evidence; no hidden fallback.
    chunk_size : int, default=16
        Decisions per evaluator chunk. An explicit value is honored.

    Returns
    -------
    TournamentResult
        Complete match, matchup, map and ranking tables through the shared result
        interface. New runs also expose the configured result's snapshot and reuse
        facts. Selected captures stay in memory only when no output folder is used.

    Raises
    ------
    ValueError, TypeError
        Entrants, rules, budget aliases, saved identities or captures conflict.
        Invalid batch or chunk sizes fail before any entrant loads. Required live
        entrants must be supplied again for unfinished resumed work.
    OSError, RuntimeError
        Loading, evaluation, recording or analysis fails. Durable earlier games
        remain available for explicit resume. Factories are trusted installed code.

    Notes
    -----
    Host orchestration, not a jitted function. All new entry forms share one
    resolved runner, evaluator, writer and statistical owner. Old list folders
    use the same job executor while retaining their original file format. No
    call here publishes results or admits a System to the official ladder.
    Numerical variables and memory templates are frozen at setup; opaque provider
    state remains caller-owned. A needed System keeps its resource scope open
    across unfinished matchups, closing before the writer. This manages clients
    created by the System, not caller-supplied clients or independent servers.
    Backend allocation errors propagate. A smaller num_envs reduces simulation
    memory, not model weights. Scientific settings on resume are assertions,
    not permission to replace saved conditions.

    Examples
    --------
    >>> import marl_battlegrounds as marl_bgs
    >>> result = marl_bgs.run_tournament(
    ...     ["random", "tdm-alpha"], maps=[47], games_per_opponent=2,
    ...     max_steps=16, seed=7,
    ... )
    >>> len(result.matches)
    2
    """
    from marl_battlegrounds.evaluation.canonical import (
        _run_configured_tournament,
        _saved_plan,
    )
    from marl_battlegrounds.evaluation.results import _read_manifest
    from marl_battlegrounds.evaluation.tournament_inputs import (
        _prepare_tournament_method,
        prepare_tournament_inputs,
        read_short_tournament_config,
        resolve_tournament_budget,
    )

    num_envs = positive_int(num_envs, "num_envs")
    chunk_size = positive_int(chunk_size, "chunk_size")
    if output_dir is not None and resume_from is not None:
        raise ValueError("output_dir and resume_from are mutually exclusive")
    if policies is not None and config is not None:
        raise ValueError("Supply policies or config, not competing populations")
    budget = resolve_tournament_budget(games_per_opponent, episodes_per_pair)
    challenger_prepared = False
    if resume_from is not None and "tournament_reuse" not in _read_manifest(
        Path(resume_from)
    ):
        if config is not None or (
            not isinstance(challenger, Omitted) and challenger is not None
        ):
            raise ValueError("Historical list runs resume with their original policies")
        if not isinstance(rerun_existing, Omitted) and rerun_existing:
            raise ValueError("Historical resume cannot replace saved games")
        return _resume_legacy_tournament(
            policies,
            resume_from=resume_from,
            save_replays=save_replays,
            maps=maps,
            episodes_per_pair=budget,
            seed=seed,
            num_envs=num_envs,
            metrics=metrics,
            full_metrics_episodes=full_metrics_episodes,
            replay_episodes=replay_episodes,
            opponent_weights=opponent_weights,
            score_threshold=score_threshold,
            max_steps=max_steps,
            red_zone_depth=red_zone_depth,
            chunk_size=chunk_size,
        )
    short = None if config is None else read_short_tournament_config(config)
    rules: dict[str, Any] = {
        "maps": maps,
        "games_per_opponent": budget,
        "seed": seed,
        "opponent_weights": opponent_weights,
        "score_threshold": score_threshold,
        "max_steps": max_steps,
        "red_zone_depth": red_zone_depth,
    }
    outputs: dict[str, Any] = {
        "metrics": metrics,
        "full_metrics_episodes": full_metrics_episodes,
        "replay_episodes": replay_episodes,
        "save_replays": save_replays,
    }
    raw: dict[str, Any] = {}
    if short is not None:
        raw, _ = short
        policies = raw["entrants"]
        for name in ("full_metrics_episodes", "replay_episodes"):
            if not isinstance(outputs[name], Omitted):
                outputs[name] = _episode_selection(outputs[name], name)
        config_budget = resolve_tournament_budget(
            raw.get("games_per_opponent", OMITTED),
            raw.get("episodes_per_pair", OMITTED),
        )
        if not isinstance(config_budget, Omitted):
            raw = {**raw, "games_per_opponent": config_budget}
        for key, value in [*rules.items(), *outputs.items()]:
            if key not in raw:
                continue
            selected = raw[key]
            if key in {"full_metrics_episodes", "replay_episodes"}:
                selected = _episode_selection(selected, key)
            if not isinstance(value, Omitted):
                actual, expected = value, selected
                if key == "maps":
                    # Schedule order is sorted by ID; keep supplied map objects
                    # below so their complete content still reaches validation.
                    actual, expected = (
                        tuple(
                            sorted(
                                item.map_id if isinstance(item, TDMMapInfo) else item
                                for item in (
                                    CANONICAL_TDM_EVALUATION_MAP_IDS
                                    if choices is None
                                    else choices
                                )
                            )
                        )
                        for choices in (value, selected)
                    )
                if actual != expected:
                    raise ValueError(
                        f"{key} conflicts with the short tournament config"
                    )
                if key == "maps":
                    selected = value
            (rules if key in rules else outputs)[key] = selected
        if raw.get("output_dir") is not None:
            if (
                output_dir is not None
                and Path(output_dir).resolve() != Path(raw["output_dir"]).resolve()
            ):
                raise ValueError(
                    "output_dir conflicts with the short tournament config"
                )
            if resume_from is None:
                output_dir = raw["output_dir"]
    saved = _saved_plan(resume_from)
    bindings: dict[str, System | Policy] | None = None
    input_metadata: dict[str, Any] | None = None
    if saved is not None:
        previous = saved[0]["details"]
        generic = previous.get("input_metadata", {})
        if short is not None and not generic:
            raise ValueError(
                "A full saved field needs its full config, not a new short config"
            )
        for key, value in rules.items():
            if isinstance(value, Omitted):
                continue
            saved_key = (
                "episodes_per_pair"
                if key == "games_per_opponent"
                else "map_ids"
                if key == "maps"
                else key
            )
            expected = generic.get(saved_key)
            actual = value
            if key == "maps":
                actual = sorted(
                    item.map_id if isinstance(item, TDMMapInfo) else item
                    for item in (
                        CANONICAL_TDM_EVALUATION_MAP_IDS if value is None else value
                    )
                )
                for item in value or ():
                    if isinstance(item, TDMMapInfo):
                        known = next(
                            (
                                row["registered_map"]
                                for row in saved[1]["conditions"]["map_sources"]
                                if row["map_id"] == item.map_id
                            ),
                            None,
                        )
                        if item.model_dump(mode="json") != known:
                            raise ValueError(
                                "Supplied map description differs from the "
                                "saved tournament"
                            )
            if not generic:
                if key == "games_per_opponent":
                    expected = previous["budget"]["resolved_games_per_opponent"]
                else:
                    raise ValueError("Config owns scientific settings; omit " + key)
            if actual != expected:
                label = (
                    "red_zone_depth differs from the saved evaluation conditions"
                    if key == "red_zone_depth"
                    else f"{key} differs from the saved tournament conditions"
                )
                raise ValueError(label)
        if generic and not isinstance(rules["maps"], Omitted):
            from marl_battlegrounds.evaluation import tournament_inputs
            from marl_battlegrounds.evaluation.evaluation_conditions import (
                config_record,
            )

            rosters = generic["rosters"]
            for current in tournament_inputs.normalize_episode_specs(
                generic["map_ids"],
                len(generic["map_ids"]),
                rosters["team_a"],
                rosters["team_b"],
                generic["score_threshold"],
                generic["max_steps"],
                red_zone_depth=generic["red_zone_depth"],
            ):
                identifier, _ = config_record(current.env_config)
                if (
                    identifier
                    != generic["configuration_ids_by_map"][str(current.map_id)]
                ):
                    raise ValueError(
                        "explicit map contents differ from the saved tournament source"
                    )
        if "selection" in raw:
            from marl_battlegrounds.evaluation.population_selection import (
                bind_selection,
            )

            previous_bound = saved[1].get("selection")
            if previous_bound is None:
                raise ValueError(
                    "A saved tournament cannot acquire a selection rule after games"
                )
            current_bound = bind_selection(
                raw["selection"],
                saved[1],
                entrant_order=previous_bound["declaration"].get("entrant_order"),
            )
            if current_bound != previous_bound:
                raise ValueError(
                    "selection differs from the saved pre-game declaration"
                )
        for key in ("seeds", "rosters"):
            if key in raw and raw[key] != generic.get(key):
                raise ValueError(f"{key} differs from the saved tournament conditions")
        if policies is not None:
            frozen = tuple(_prepare_tournament_method(value) for value in policies)
            by_name = {method.name: method for method in frozen}
            expected = {row["name"] for row in saved[1]["participants"]}
            if len(by_name) != len(frozen) or set(by_name) != expected:
                raise ValueError(
                    "Supplied participant names differ from the saved field"
                )
            bindings = {
                row["entrant_id"]: by_name[row["name"]]
                for row in saved[1]["participants"]
            }
        selected_config = None if config is None or short is not None else config
    elif policies is not None:
        if isinstance(challenger, str):
            challenger = _prepare_tournament_method(challenger)
            challenger_prepared = True
        defaults: dict[str, Any] = {
            "maps": None,
            "games_per_opponent": 100,
            "seed": 0,
            "opponent_weights": None,
            "score_threshold": 20,
            "max_steps": 300,
            "red_zone_depth": 5.0,
        }
        resolved = {
            key: defaults[key] if isinstance(value, Omitted) else value
            for key, value in rules.items()
        }
        selected_config, bindings, input_metadata = prepare_tournament_inputs(
            policies,
            **resolved,
            seeds=raw.get("seeds"),
            selection=raw.get("selection"),
            rosters=raw.get("rosters"),
            challenger=None if isinstance(challenger, Omitted) else challenger,
        )
    else:
        if config is None:
            raise ValueError("Supply a policies sequence or a tournament config")
        conflicts = [
            key
            for key, value in rules.items()
            if key != "games_per_opponent" and not isinstance(value, Omitted)
        ]
        if conflicts:
            raise ValueError(
                "Config owns scientific settings; omit " + ", ".join(conflicts)
            )
        selected_config = config
    return _run_configured_tournament(
        selected_config,
        challenger=challenger,
        challenger_prepared=challenger_prepared,
        games_per_opponent=cast(int | Omitted, rules["games_per_opponent"]),
        rerun_existing=rerun_existing,
        **outputs,
        output_dir=output_dir,
        resume_from=resume_from,
        num_envs=num_envs,
        chunk_size=chunk_size,
        bindings=bindings,
        input_metadata=input_metadata,
    )
