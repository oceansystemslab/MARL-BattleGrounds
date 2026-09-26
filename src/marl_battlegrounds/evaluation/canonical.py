"""Run immutable tournament plans through the existing evaluator and recorder.

run_canonical_tournament selects a released tournament snapshot, with one optional
challenger. The private resolved route also serves custom configs and maintainer
admission. Original game records keep their identities. Models are loaded only
for unfinished jobs, and complete saved summaries are read without fitting again.
This module neither publishes releases nor makes admission decisions.
"""

# Private helpers below share the established evaluator and writer authorities.
# pyright: reportPrivateUsage=false, reportUnusedFunction=false

from __future__ import annotations

import copy
import json
from collections.abc import Iterable, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from marl_battlegrounds.evaluation.evaluation_conditions import OMITTED, Omitted
from marl_battlegrounds.evaluation.sampling_evidence import (
    combine_sampling_facts,
    method_sampling_fact,
)
from marl_battlegrounds.evaluation.tournament_assets import (
    AssetVerifier,
    asset_location_config,
    controller_content_identity,
    duplicate_controller_id,
    load_tournament_controller,
    loaded_controller_registration,
    validate_loaded_controller,
    verify_tournament_environment,
)
from marl_battlegrounds.evaluation.tournament_config import (
    canonical_json,
    resolve_tournament_config,
)
from marl_battlegrounds.evaluation.tournament_records import (
    TournamentRecords,
    origin_key,
)
from marl_battlegrounds.evaluation.tournament_reuse import (
    ReusePlan,
    analysis_schedule,
    resolve_reuse_plan,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from marl_battlegrounds.core.types import EnvConfig
    from marl_battlegrounds.environment import MetricMode
    from marl_battlegrounds.evaluation.evaluate import EpisodeSpec, EvaluationResult
    from marl_battlegrounds.evaluation.policy_execution import Policy, System
    from marl_battlegrounds.evaluation.results import CanonicalTournamentResult


def _option(
    value: object, previous: Mapping[str, Any] | None, name: str, default: object
) -> object:
    """Return the saved option, new-run default or caller's explicit assertion.

    value is an option or the private Omitted marker. previous is the saved
    settings mapping, or None for a new run. name selects its field; default
    applies only to a new omitted value. A conflicting explicit value raises
    ValueError. The return is borrowed; this helper changes no input or file.
    """
    if isinstance(value, Omitted):
        return default if previous is None else previous[name]
    if previous is not None and value != previous[name]:
        raise ValueError(f"{name} differs from the saved tournament conditions")
    return value


def _saved_plan(
    directory: str | Path | None,
) -> tuple[dict[str, Any], dict[str, Any], tuple[dict[str, Any], ...]] | None:
    """Read a saved manifest, immutable config and ordered logical games.

    directory is an exact run folder; None returns None. Return that three-item
    tuple after shared schema and content checks. Missing, changed or unsupported
    references raise before any write, recovery, model loading or action.
    """
    if directory is None:
        return None
    from marl_battlegrounds.evaluation.canonical_results import read_canonical_files
    from marl_battlegrounds.evaluation.results import _read_manifest

    root = Path(directory)
    manifest = _read_manifest(root)
    config, games = read_canonical_files(root, manifest)
    if manifest.get("tournament_reuse", {}).get("version") != 1:
        raise ValueError("resume_from is not a format-1 configuration tournament")
    return manifest, config, games


def _source_configs(
    config: Mapping[str, Any], verifier: AssetVerifier
) -> tuple[
    dict[int, EnvConfig],
    dict[tuple[int, int], EnvConfig],
    dict[str, Any],
    dict[str, tuple[Mapping[str, Any], EnvConfig]],
    dict[tuple[int, int], str],
]:
    """Prepare each declared map's exact source and two complete spawn choices.

    config is a validated tournament descriptor and verifier owns its checked
    assets. Return source configs by map ID, resolved configs by (map ID, choice),
    serialized contents by config ID, a verified ID-to-(content, config)
    cache for physical evidence, and the frozen configuration ID of each
    (map ID, choice). Choice 0 keeps source banks; choice 1 exchanges both
    complete banks. Other fields stay exact. Inputs are not changed. Content
    saved before the Red Zone rule (12 keys) restores at depth 0.0 and keeps
    its original identity, so its IDs are the historical ones.

    Host setup imports JAX, restores scalar configs and runs Core's existing
    requested-choice validator. Unsupported information/memory rules, wrong
    rosters, thresholds, hashes or registered source geometry raise ValueError.
    This creates no run files, methods or actions.
    """
    import numpy as np

    from marl_battlegrounds.evaluation.evaluation_conditions import (
        config_record,
        restore_recorded_config,
    )
    from marl_battlegrounds.tasks import (
        _roster_ids,
        _swap_spawn_banks,
        _validate_config_choices,
        canonical_tournament_rosters,
    )

    conditions = config["conditions"]
    if (
        config["version"] == 1
        and config["release"] is not None
        and tuple(tuple(conditions["rosters"][team]) for team in ("team_a", "team_b"))
        != canonical_tournament_rosters()
    ):
        raise ValueError("Official rosters must use the canonical ordered team slots")
    if (
        conditions["information_mode"] != "SharedObs"
        or conditions["memory_rule"] != "fresh-per-game"
    ):
        raise ValueError("Unsupported tournament information or episode-memory rule")
    expected_classes = tuple(
        value
        for team in ("team_a", "team_b")
        for value in _roster_ids(conditions["rosters"][team], name=team)
    )
    sources: dict[int, EnvConfig] = {}
    resolved: dict[tuple[int, int], EnvConfig] = {}
    contents: dict[str, Any] = {}
    verified: dict[str, tuple[Mapping[str, Any], EnvConfig]] = {}
    resolved_ids: dict[tuple[int, int], str] = {}
    for declaration in config["conditions"]["map_sources"]:
        content = verifier.read_json(declaration["source_config_asset"])
        if not isinstance(content, dict):
            raise ValueError("Source configuration asset must contain one config")
        try:
            source, historical = restore_recorded_config(
                cast(dict[str, Any], content),
                declaration["source_config_id"],
                validate=False,
            )
        except ValueError as error:
            raise ValueError(
                "Source configuration differs from its declared identity"
            ) from error
        source_id, actual = config_record(source, historical=historical)
        conditions = config["conditions"]
        if (
            int(source.max_steps) != conditions["max_steps"]
            or int(source.team_deathmatch_score_threshold)
            != conditions["score_threshold"]
        ):
            raise ValueError("Source rules differ from tournament conditions")
        _validate_config_choices(source, batched=False, both_spawn_choices=True)
        if (
            tuple(np.asarray(source.agent_profile.class_ids).tolist())
            != expected_classes
        ):
            raise ValueError("Source roster differs from tournament conditions")
        registered = declaration["registered_map"]
        if registered is not None:
            from marl_battlegrounds.evaluation.catalog import (
                build_resolved_env_config_v2,
            )
            from marl_battlegrounds.evaluation.map_identity import (
                _authored_map,
                _geometry_matches,
            )

            identity = registered["source"]
            approved = _authored_map(
                identity["asset_id"], identity["revision"], identity["semantic_digest"]
            )
            if approved is None or not _geometry_matches(
                # Geometry only; V2 accepts any recorded Red Zone depth.
                build_resolved_env_config_v2(source),
                approved[1],
            ):
                raise ValueError(
                    "Source geometry differs from its registered map identity"
                )
            if (
                approved[0].map_id != declaration["map_id"]
                or approved[0].split != declaration["split"]
                or approved[0].technical_name != registered["name"]
            ):
                raise ValueError(
                    "Source map name, number or split differs from its identity"
                )
        map_id = declaration["map_id"]
        sources[map_id] = source
        resolved[map_id, 0] = source
        resolved[map_id, 1] = _swap_spawn_banks(source)
        contents[source_id] = actual
        swapped_id, swapped = config_record(resolved[map_id, 1], historical=historical)
        contents[swapped_id] = swapped
        verified[source_id] = actual, source
        verified[swapped_id] = swapped, resolved[map_id, 1]
        resolved_ids[map_id, 0] = source_id
        resolved_ids[map_id, 1] = swapped_id
    return sources, resolved, contents, verified, resolved_ids


def _local_challenger(method: System | Policy) -> tuple[dict[str, Any], dict[str, Any]]:
    """Describe local frozen evidence honestly without inventing a reusable loader.

    Local methods keep a registration-based entrant ID and unknown executable
    evidence. Their descriptor cannot restore arbitrary Python/provider objects;
    callers must supply the same System again when resuming. Qualified admission
    supplies its separately verified immutable descriptor instead.
    """
    registration = loaded_controller_registration(method)
    identity = sha256(canonical_json(registration)).hexdigest()
    controller = {
        "kind": "factory",
        "factory": "unknown:requires_supplied_system",
        "content": {
            name: None
            for name in (
                "code",
                "parameters",
                "memory_template",
                "input_preparation",
                "decision_settings",
                "adapter_bindings",
                "external_state",
            )
        },
    }
    content_id, _ = controller_content_identity(
        controller, AssetVerifier({"assets": {}})
    )
    return {
        "entrant_id": "challenger-" + identity[:24],
        "name": method.name,
        "controller_id": content_id,
        "controller": controller,
        "registration_asset": "local-registration-" + identity,
        "elo": None,
        "result_ref": None,
    }, registration


def _plan_paths(
    config: Mapping[str, Any], verifier: AssetVerifier, challenger: bool
) -> dict[str, Path]:
    """Return verified local paths for schedule and companion game metadata.

    config and verifier describe one immutable snapshot. challenger says whether
    an explicit companion schedule is required. An existing companion is also
    read for field-only allocation bounds. Missing required references fail;
    this never downloads, loads models or opens optional report payloads.
    """
    schedule_id = config["conditions"]["schedule_asset"]
    schedule = verifier.read_json(schedule_id)
    if not isinstance(schedule, dict):
        raise ValueError("Schedule manifest must contain an object")
    schedule = cast(dict[str, Any], schedule)
    identifiers = [schedule_id, schedule["games_asset"]]
    if challenger or schedule.get("challenger_games_asset") is not None:
        if schedule.get("challenger_games_asset") is None:
            raise ValueError("This snapshot has no declared challenger schedule")
        identifiers.append(schedule["challenger_games_asset"])
    return verifier.require(
        [key for key in identifiers if "inline" not in verifier.assets[key]]
    )


def _specs(
    plan: ReusePlan,
    sources: Mapping[int, EnvConfig],
    resolved: Mapping[tuple[int, int], EnvConfig],
    resolved_ids: Mapping[tuple[int, int], str],
) -> dict[int, tuple[EpisodeSpec, ...]]:
    """Return job-ID to exact EpisodeSpec tuples in declared execution order.

    plan supplies immutable original execution coordinates. sources and resolved
    contain the already validated map configurations; resolved_ids holds their
    frozen configuration IDs from _source_configs (historical IDs for content
    saved before Red Zone), so nothing is re-hashed here. Every resolved config
    ID must match its declared complete bank choice. The specs retain original
    episode/seed IDs; pair block labels are local recording joins, never RNG
    inputs. A mismatch raises before execution. No input changes or actions.
    """
    from marl_battlegrounds.evaluation.evaluate import EpisodeSpec

    by_id = {game["logical_game_id"]: game for game in plan.games}
    pair_ids: dict[str | int, int] = {}
    result: dict[int, tuple[EpisodeSpec, ...]] = {}
    config_ids = resolved_ids
    for game in plan.games:
        if (
            config_ids[game["map_id"], game["spawn_locations"]]
            != game["resolved_config_id"]
        ):
            raise ValueError(
                "Declared resolved configuration is not its complete spawn choice"
            )
    for job in plan.jobs:
        values: list[EpisodeSpec] = []
        for identifier in job["logical_game_ids"]:
            game = by_id[identifier]
            block = pair_ids.setdefault(game["pair_id"], len(pair_ids) + 1)
            values.append(
                EpisodeSpec(
                    game["execution"]["episode_id"],
                    resolved[game["map_id"], game["spawn_locations"]],
                    game["map_id"],
                    game["execution"]["seed_id"],
                    metadata={
                        "block_id": block,
                        "bootstrap_group": game["execution"]["bootstrap_group"],
                    },
                    source_config=sources[game["map_id"]],
                    spawn_locations=game["spawn_locations"],
                    paired_comparison_key=f"pair-{block}",
                )
            )
        result[job["job_id"]] = tuple(values)
    return result


def _check_saved_job_options(
    manifest: Mapping[str, Any],
    plan: ReusePlan,
    *,
    metrics: str,
    full: Mapping[int, Sequence[int]],
    replays: Mapping[int, Sequence[int]],
) -> None:
    """Check saved local pass settings before any recovery or completed return.

    plan contains the checked jobs; full and replays use their execution IDs.
    Missing passes are unstarted jobs. Existing passes must retain their seed,
    episode list and effective captures. Full mode selects every job episode.
    This check does not constrain batch/chunk settings or inspect reused sources;
    TournamentRecords owns source evidence. Mismatches raise without writes.
    """
    games = {row["logical_game_id"]: row for row in plan.games}
    passes = {
        (row["phase"], row["pass_id"]): row for row in manifest["passes"].values()
    }
    for job in plan.jobs:
        entry = passes.get((job["phase"], job["pass_id"]))
        if entry is None:
            continue
        ids = sorted(
            games[key]["execution"]["episode_id"] for key in job["logical_game_ids"]
        )
        expected = {
            "seed": job["root_seed"],
            "metrics": metrics,
            "episode_ids": ids,
            "num_episodes": len(ids),
            "full_metrics_episodes": ids
            if metrics == "full"
            else sorted(full[job["job_id"]]),
            "replay_episodes": sorted(replays[job["job_id"]]),
        }
        if any(entry["details"].get(name) != value for name, value in expected.items()):
            raise ValueError("Saved tournament pass scientific conditions differ")


def _match_rows(records: TournamentRecords, metrics: str) -> list[dict[str, Any]]:
    """Materialize narrow original match rows in declared logical order.

    records owns source verification and row joins. metrics is the selected
    priority/full/none mode; none reads only identities and required outcome
    cells and fills optional return fields with None. Return existing MATCH_COLUMNS
    in their stable order. Missing required assets/rows raise through records.
    Whole wide full reports are never read here.
    """
    from marl_battlegrounds.evaluation.run_writer import IDENTITY_COLUMNS, MATCH_COLUMNS

    columns = (
        (
            *IDENTITY_COLUMNS,
            "block_id",
            "bootstrap_group",
            "outcome",
            "episode_length",
            "team_a_score",
            "team_b_score",
        )
        if metrics == "none"
        else None
    )
    return [
        {name: row.get(name) for name in MATCH_COLUMNS}
        for batch in records.iter_rows("match_results.csv", columns=columns)
        for row in batch
    ]


def _memory_pass(
    result: EvaluationResult, full_ids: Sequence[int], metrics: str
) -> dict[str, Any]:
    """Return one completed in-memory pass without inventing capture coverage.

    result is the just-completed evaluator result. full_ids lists selected original
    execution IDs; metrics is its whole-pass mode. Preserve exact declarations,
    systems, IDs and actual optional coverage. Replay objects remain in the
    result, so no file paths are fabricated. Inputs and files are unchanged.
    """
    details = result.metadata
    schedule = details["schedule"]
    if isinstance(schedule, (tuple, list)):
        schedule = {
            str(row["episode_id"]): row
            for row in cast(Sequence[dict[str, Any]], schedule)
        }
    return {
        "phase": "tournament",
        "pass_id": details["pass_id"],
        "details": details,
        "system_ids": details["system_ids"],
        "episodes": {str(key): value for key, value in schedule.items()},
        "completed_episode_ids": list(result.completed_episode_ids),
        "recorded_metrics_by_episode": {
            str(identifier): "full"
            if metrics == "full" or identifier in full_ids
            else metrics
            for identifier in result.completed_episode_ids
        },
        "replays": {},
        "result_state": {
            "version": 1,
            "status": "complete",
            "reason": None,
            "schedule_digest": details.get("schedule_digest"),
        },
    }


def _saved_result(
    directory: Path,
    manifest: dict[str, Any],
    metadata: dict[str, Any],
    plan: ReusePlan,
    records: TournamentRecords,
    evidence: Mapping[str, Any],
) -> CanonicalTournamentResult:
    """Read committed summaries and build a result without new games or fitting.

    directory and manifest identify the durable run. metadata and plan are its
    resolved scientific settings, records owns the checked original rows, and
    evidence is already prepared complete physical evidence. Read only durable
    summary prefixes and return the shared CanonicalTournamentResult. Missing
    files or malformed cells raise; no recovery or publication occurs here.
    """
    from marl_battlegrounds.evaluation.scalar_reports import iter_summary_rows

    tables: dict[str, tuple[dict[str, Any], ...]] = {}
    paths = {"run_details": directory / "run_details.json"}
    for name in (
        "tournament_results",
        "matchup_results",
        "map_results",
        "tournament_headline_metrics",
    ):
        if name + ".csv" in manifest.get("tables", {}):
            path = directory / (name + ".csv")
            tables[name] = tuple(
                row
                for batch in iter_summary_rows(path, manifest=manifest)
                for row in batch
            )
            paths[name] = path
    for filename in manifest.get("tables", {}):
        path = directory / filename
        if path.is_file():
            paths[path.stem] = path
    matches = _match_rows(records, metadata["metrics"])
    return _result(
        metadata,
        plan,
        records,
        matches,
        tables["tournament_results"],
        tables["matchup_results"],
        tables["map_results"],
        tables.get("tournament_headline_metrics", ()),
        {},
        (),
        paths,
        manifest["tournament_summary"]["metadata"],
        {**manifest["tournament_summary"]["qualification"], "evidence": evidence},
        executed_this_call=0,
    )


def _result(
    metadata: Mapping[str, Any],
    plan: ReusePlan,
    records: TournamentRecords,
    matches: Sequence[Mapping[str, Any]],
    ratings: Sequence[Mapping[str, Any]],
    matchups: Sequence[Mapping[str, Any]],
    maps: Sequence[Mapping[str, Any]],
    headline: Sequence[Mapping[str, Any]],
    full: Mapping[str, object],
    replays: Sequence[object],
    paths: dict[str, Path] | None,
    statistics: Mapping[str, Any],
    qualification: Mapping[str, Any],
    *,
    executed_this_call: int,
) -> CanonicalTournamentResult:
    """Build one shared result from its completed narrow tables and evidence.

    metadata holds immutable settings; plan supplies the origin join. records
    borrows the final manifest snapshot. matches, ratings, matchups, maps and
    headline are complete rows; full and replays hold newly executed optional
    in-memory data. paths is None for a no-file run. statistics and qualification
    identify the accepted calculation and physical evidence. executed_this_call
    counts new completions, separately from the whole plan's fresh/reused counts.

    Return a CanonicalTournamentResult with read-only table access and the same
    ownership aliases on all paths. Keep the large plan only in returned metadata;
    ordinary writer flushes do not serialize a second copy of its game history.
    """
    from marl_battlegrounds.evaluation.results import CanonicalTournamentResult

    config = metadata["canonical_config"]
    manifest = records.manifest
    details = {
        **metadata,
        "canonical_plan": asdict(plan),
        "run_id": manifest["run_id"],
        "configurations": manifest["configurations"],
        "systems": manifest.get("systems", {}),
        "passes": manifest["passes"],
        "statistics": dict(statistics),
        "physical_evidence": qualification["evidence"],
        "tournament_completion": dict(qualification),
        "tournament_summary": manifest["tournament_summary"],
        "tournament_reuse": manifest["tournament_reuse"],
        "executed_this_call": executed_this_call,
        "snapshot_id": config["snapshot_id"],
        "big_12_id": config["snapshot_id"],
        "protocol_id": config["release"]["rules_id"]
        if config["release"]
        else config["conditions"]["pairing_protocol"],
    }
    for key in ("execution_resources", "execution_layouts"):
        if key in manifest:
            details[key] = manifest[key]
    return CanonicalTournamentResult(
        tuple(dict(row) for row in matches),
        tuple(dict(row) for row in ratings),
        tuple(dict(row) for row in matchups),
        tuple(dict(row) for row in maps),
        cast(Any, dict(full)),
        cast(Any, tuple(replays)),
        details,
        paths,
        headline_metrics=tuple(dict(row) for row in headline),
        snapshot_id=config["snapshot_id"],
        challenger_id=metadata["challenger_id"],
        planned_games=len(plan.games),
        reused_games=sum(game["origin"] is not None for game in plan.games),
        executed_games=sum(game["origin"] is None for game in plan.games),
        games_per_opponent=plan.budget["resolved_games_per_opponent"],
        protocol_compliant=plan.budget["protocol_compliant"],
        _record_access=records,
    )


def _resident_request_key(
    identifier: str, participant: Mapping[str, Any]
) -> tuple[str, bytes, str]:
    """Name one exact entrant request without retaining its model or loading it.

    identifier is the participant ID. The checked controller descriptor and
    unchanged-placement marker distinguish requests made to the resident cache.
    """
    return (
        identifier,
        canonical_json(participant["controller"]),
        "unchanged-placement",
    )


@contextmanager
def _active_pair(
    first: str,
    second: str,
    participants: Mapping[str, Mapping[str, Any]],
    verifier: AssetVerifier,
    challenger_id: str | None,
    method: System | Policy | None,
    bindings: Mapping[str, System | Policy] | None = None,
    *,
    residents: dict[tuple[str, bytes, str], System | Policy] | None = None,
) -> Generator[tuple[System | Policy, System | Policy]]:
    """Borrow supplied methods and lazily reuse each exact entrant reference.

    first and second select checked participant IDs. bindings and the optional
    challenger method stay caller-owned. residents is the current run's plain
    dictionary; None keeps loaded references only for this pair. Each key names
    the entrant and its full controller descriptor, so distinct host entrants
    are never merged because their model labels happen to match.

    Loading and calls stay in the caller's context. Errors propagate without
    retry. No provider is closed, moved to another device or sent to a worker.
    Dropping references makes no promise about allocator or provider memory.
    """
    cache = {} if residents is None else residents
    available: dict[str, System | Policy] = {}
    for identifier in dict.fromkeys((first, second)):
        if challenger_id == identifier and method is not None:
            available[identifier] = method
        elif bindings is not None and identifier in bindings:
            available[identifier] = bindings[identifier]
        else:
            participant = participants[identifier]
            request = _resident_request_key(identifier, participant)
            if request not in cache:
                cache[request] = load_tournament_controller(participant, verifier)
            available[identifier] = cache[request]
    yield available[first], available[second]


def _recorded_method_sampling(
    records: TournamentRecords,
    participants: Mapping[str, Mapping[str, Any]],
    observed: Mapping[str, Mapping[str, Any]],
    completed_before: set[int],
) -> dict[str, dict[str, Any]]:
    """Recover sampling facts only from each game's checked original evidence.

    Call after prepare_reuse_evidence has checked every origin, pass and method
    identity. completed_before names games already durable before this call;
    only newly played local games may use observed loaded-method facts. Saved
    pass facts are bound to that pass's team registrations. Older input/summary
    facts need the matching original registration ID. Missing or conflicting
    facts stay unknown. No actor is loaded and no saved record is rewritten.
    """
    facts: dict[str, dict[bytes, dict[str, Any]]] = {
        item["name"]: {} for item in participants.values()
    }
    with records.verifier.verification_scope():
        for game in records.games:
            origin = records.origin(game)
            manifest = records.source_manifest(game)
            entry = records.entry(game)
            assert entry is not None
            declaration = entry["episodes"][str(origin["episode_id"])]
            owners = declaration.get("system_ids", entry.get("system_ids", {}))
            source_input = manifest.get("details", {}).get("input_metadata", {})
            summary = manifest.get("tournament_summary", {})
            summary_systems = (
                summary.get("qualification", {}).get("evidence", {}).get("systems", {})
            )
            for team in ("team_a", "team_b"):
                name = participants[game[team]]["name"]
                registration_id = owners[team]
                original_name = manifest["systems"][registration_id]["name"]
                candidates: list[Mapping[str, Any]] = []
                saved_fact = entry["details"].get("method_sampling", {}).get(team)
                if saved_fact is not None:
                    candidates.append(saved_fact)
                if (
                    source_input.get("participants", {}).get(original_name)
                    == registration_id
                ):
                    previous = source_input.get("method_sampling", {}).get(
                        original_name
                    )
                    if previous is not None:
                        candidates.append(previous)
                if any(
                    item.get("name") == original_name
                    and registration_id in item.get("registration_ids", ())
                    for item in summary_systems.values()
                ):
                    previous = (
                        summary.get("metadata", {})
                        .get("method_sampling", {})
                        .get(original_name)
                    )
                    if previous is not None:
                        candidates.append(previous)
                if (
                    origin.get("source_id") is None
                    and game["logical_game_id"] not in completed_before
                    and name in observed
                ):
                    candidates.append(observed[name])
                try:
                    combine_sampling_facts(candidates)
                    known = [
                        fact for fact in candidates if fact["determinism"] != "unknown"
                    ]
                    # An original pass can explain an older summary's missing
                    # support. Conflicting known facts cannot explain each other.
                    selected = known or candidates
                    value = (
                        {
                            "determinism": "unknown",
                            "basis": "Original sampling facts conflict",
                        }
                        if len({fact["determinism"] for fact in selected}) > 1
                        else combine_sampling_facts(selected)
                    )
                except TypeError, ValueError:
                    value = {
                        "determinism": "unknown",
                        "basis": "Original sampling facts are invalid",
                    }
                facts[name][canonical_json(value)] = value
    result: dict[str, dict[str, Any]] = {}
    for name, unique in facts.items():
        values = tuple(unique.values())
        result[name] = (
            {
                "determinism": "unknown",
                "basis": "Original sampling facts conflict across games",
            }
            if len({fact["determinism"] for fact in values} - {"unknown"}) > 1
            else combine_sampling_facts(values)
        )
    return result


def _run_resolved_tournament(
    config: Mapping[str, Any],
    *,
    system: System | Policy | str | None = None,
    challenger_descriptor: Mapping[str, Any] | None = None,
    challenger_assets: Mapping[str, Any] | None = None,
    plan: ReusePlan | None = None,
    require_reuse: bool = False,
    games_per_opponent: int | None = None,
    rerun_existing: bool = False,
    metrics: MetricMode = "priority",
    full_metrics_episodes: Iterable[int] = (),
    replay_episodes: Iterable[int] = (),
    save_replays: int = 0,
    output_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
    num_envs: int = 128,
    chunk_size: int = 16,
    _bindings: Mapping[str, System | Policy] | None = None,
    _system_prepared: bool = False,
    _input_metadata: Mapping[str, Any] | None = None,
    _explicit_config: bool = True,
    _official_snapshot_verified: bool = False,
) -> CanonicalTournamentResult:
    """Execute one fully resolved custom, official or admission population.

    config is already structurally/release validated. Admission may supply its
    exact challenger descriptor/assets and frozen plan. Public wrappers own
    omission handling. This host helper preflights sources before writer recovery,
    keeps execution IDs unchanged and returns a complete shared result. It never
    downloads, publishes a release or decides promotion.

    system is one optional supplied challenger. challenger_descriptor/assets are
    the maintainer's already qualified loader/evidence declarations, not a public
    qualification bypass. plan, when supplied, must exactly match resolution.
    require_reuse forbids missing incumbent origins; rerun_existing explicitly
    runs the whole selected schedule. games_per_opponent inherits the descriptor
    when None. Metric and capture settings use public canonical meanings.

    output_dir and resume_from are mutually exclusive; neither means no files.
    num_envs is the positive fixed number of parallel game lanes per matchup;
    chunk_size is a positive tick count. Pair scheduling stays sequential.
    _bindings borrows already frozen local entrants. _system_prepared=True
    borrows the challenger frozen by list preparation, without another model copy.
    _input_metadata carries the
    short/list result aliases from their shared normalizer. _explicit_config is
    private: True applies the supplied config's current asset hints; False keeps
    durable hints from an omitted saved config. It never changes scientific
    identity. _official_snapshot_verified records the public canonical resolver
    checking a release pin, or that same durable fact on resume. Custom release
    text alone cannot establish protocol compliance. Invalid settings/assets
    fail before writer mutation; execution or
    output failures propagate with previously durable games preserved. A
    snapshot saved before the Red Zone rule (pins 14, 2, 3) can only reuse its
    recorded games: when any pending game needs execution (a challenger or
    rerun_existing=True), ValueError "Snapshot configurations were saved before
    the Red Zone rule; ..." is raised before any output is created.
    """
    from marl_battlegrounds.evaluation.evaluate import positive_int
    from marl_battlegrounds.evaluation.evaluation_conditions import capture_ids
    from marl_battlegrounds.evaluation.metric_catalog import METRIC_SCHEMA_ID
    from marl_battlegrounds.evaluation.policy_execution import Policy
    from marl_battlegrounds.evaluation.run_writer import (
        RUN_SCHEMA_VERSION,
        RunWriter,
        _json_bytes,
    )
    from marl_battlegrounds.evaluation.tournament import _memory_matches, _merge_columns
    from marl_battlegrounds.evaluation.tournament_evidence import prepare_reuse_evidence
    from marl_battlegrounds.evaluation.tournament_execution import (
        TournamentJob,
        execute_tournament_jobs,
    )
    from marl_battlegrounds.evaluation.tournament_headlines import (
        content_digest,
        summarize_headlines,
    )
    from marl_battlegrounds.evaluation.tournament_inputs import (
        _prepare_tournament_method,
    )
    from marl_battlegrounds.evaluation.tournament_schedule import TournamentMatch
    from marl_battlegrounds.evaluation.tournament_statistics import (
        summarize_tournament,
        validate_opponent_weights,
    )

    if output_dir is not None and resume_from is not None:
        raise ValueError("output_dir and resume_from are mutually exclusive")
    if metrics not in {"priority", "full", "none"}:
        raise ValueError("metrics must be priority, full or none")
    num_envs = positive_int(num_envs, "num_envs")
    chunk_size = positive_int(chunk_size, "chunk_size")
    incoming_config = copy.deepcopy(dict(config))
    incoming_assets = copy.deepcopy(dict(challenger_assets or {}))
    saved = _saved_plan(resume_from)
    saved_manifest = None if saved is None else saved[0]
    config = (
        incoming_config
        if saved is None
        else copy.deepcopy(saved[0]["details"]["canonical_config"])
    )
    if incoming_config["snapshot_id"] != config["snapshot_id"]:
        raise ValueError("Tournament config differs from the saved snapshot")
    record_config = copy.deepcopy(config if saved is None else saved[1])
    asset_locations: dict[str, Any] = (
        {}
        if saved is None
        else copy.deepcopy(saved[0]["tournament_reuse"].get("asset_locations", {}))
    )
    for key, value in incoming_assets.items():
        existing = record_config["assets"].get(key)
        if existing is not None and {
            k: v for k, v in existing.items() if k not in {"path", "url"}
        } != {k: v for k, v in value.items() if k not in {"path", "url"}}:
            raise ValueError("Challenger asset alias conflicts with the snapshot")
        if saved is not None and existing is None:
            raise ValueError("Challenger assets differ from the saved tournament")
        if existing is None:
            record_config["assets"][key] = copy.deepcopy(value)
    if saved is not None:
        challenger_assets = copy.deepcopy(saved[0]["details"]["participant_assets"])
        if _explicit_config:
            for key, value in {**incoming_config["assets"], **incoming_assets}.items():
                if key not in record_config["assets"]:
                    raise ValueError("Asset location names an unknown saved asset")
                asset_locations[key] = {
                    field: value.get(field) for field in ("path", "url")
                }
    access_config = asset_location_config(record_config, asset_locations)
    verifier = AssetVerifier(access_config)
    descriptors = copy.deepcopy(config["participants"])
    registrations: dict[str, Any] = {}
    for descriptor in descriptors:
        registration = verifier.read_json(descriptor["registration_asset"])
        if not isinstance(registration, dict):
            raise ValueError("Participant registration must be a JSON object")
        registrations[descriptor["entrant_id"]] = registration
        identifier, _ = controller_content_identity(descriptor["controller"], verifier)
        if identifier != descriptor["controller_id"]:
            raise ValueError(
                "Participant content differs from its declared controller ID"
            )
    prepared_system = None
    if _system_prepared:
        if isinstance(system, str):
            raise TypeError("A prepared challenger must be a loaded method")
        prepared_system = system
    method = prepared_system
    if system is not None and not _system_prepared:
        method = _prepare_tournament_method(system)
    challenger_id = None
    if method is not None or challenger_descriptor is not None:
        if challenger_descriptor is None:
            assert method is not None
            candidate, registration = _local_challenger(method)
        else:
            candidate = copy.deepcopy(dict(challenger_descriptor))
            registration = verifier.read_json(candidate["registration_asset"])
            if not isinstance(registration, dict):
                raise ValueError("Challenger registration must be a JSON object")
            if method is not None:
                validate_loaded_controller(method, candidate, verifier)
        if (
            method is not None
            and duplicate_controller_id(method, descriptors, registrations, verifier)
            is not None
        ):
            raise ValueError(
                "Exact incumbent duplicate; run the field without a challenger"
            )
        if any(row["name"] == candidate["name"] for row in descriptors):
            raise ValueError(
                "Challenger name collides with an incumbent; supply a distinct label"
            )
        identifier, known = controller_content_identity(
            candidate["controller"], verifier
        )
        if known and any(row["controller_id"] == identifier for row in descriptors):
            raise ValueError(
                "Exact incumbent duplicate; run the field without a challenger"
            )
        challenger_id = candidate["entrant_id"]
        registrations[challenger_id] = registration
        descriptors.append(candidate)
    participants = {row["entrant_id"]: row for row in descriptors}
    if len(participants) != len(descriptors):
        raise ValueError("Challenger entrant identity collides with an incumbent")
    declared_weights = config["analysis"]["opponent_weights"]
    weights = (
        None
        if declared_weights is None
        else {
            participants[key]["name"]: value for key, value in declared_weights.items()
        }
    )
    # Check the complete field before actions; a challenger has no invented weight.
    validate_opponent_weights(tuple(row["name"] for row in descriptors), weights)
    bindings = dict(_bindings or {})
    if set(bindings) - participants.keys():
        raise ValueError("Supplied tournament bindings name an unknown entrant")
    for identifier, bound in bindings.items():
        validate_loaded_controller(bound, participants[identifier], verifier)
    paths = _plan_paths(config, verifier, challenger_id is not None)
    resolved_plan = resolve_reuse_plan(
        config,
        paths,
        asset_reader=verifier.read_json,
        participant_kinds={key: value["kind"] for key, value in registrations.items()},
        challenger_id=challenger_id,
        games_per_opponent=games_per_opponent,
        rerun_existing=rerun_existing,
        require_reuse=require_reuse,
        official_verified=_official_snapshot_verified,
    )
    if plan is not None and canonical_json(asdict(plan)) != canonical_json(
        asdict(resolved_plan)
    ):
        raise ValueError("Supplied frozen plan differs from the resolved tournament")
    plan = resolved_plan
    if "selection" in config:
        from marl_battlegrounds.evaluation.population_selection import (
            check_selection_plan,
        )

        check_selection_plan(
            config, plan.games, challenger_id=challenger_id, budget=plan.budget
        )
    full_ids = tuple(full_metrics_episodes)
    selected_replays = capture_ids(
        [game["logical_game_id"] for game in plan.games],
        tuple(replay_episodes),
        save_replays,
    )
    per_job_full = plan.capture_ids(full_ids)
    per_job_replay = plan.capture_ids(selected_replays)
    sources, resolved, configurations, verified_configs, resolved_ids = _source_configs(
        config, verifier
    )
    specs_by_job = _specs(plan, sources, resolved, resolved_ids)
    schedule = analysis_schedule(
        plan, {key: row["name"] for key, row in participants.items()}
    )
    schedule_digest = content_digest([asdict(row) for row in schedule])
    metadata: dict[str, Any] = {
        "canonical_config": config,
        "official_snapshot_verified": _official_snapshot_verified,
        "participant_descriptors": descriptors,
        "participant_registrations": registrations,
        "participant_assets": copy.deepcopy(dict(challenger_assets or {})),
        "challenger_id": challenger_id,
        "challenger_loadable": challenger_descriptor is not None,
        "metrics": metrics,
        "full_metrics_episodes": list(full_ids),
        "replay_episodes": list(selected_replays),
        "save_replays": save_replays,
        "rerun_existing": rerun_existing,
        "budget": plan.budget,
        "schedule_digest": schedule_digest,
        "pairing_protocol": "fixed-team-spawn-v1",
    }
    input_metadata = (
        _input_metadata if saved is None else saved[0]["details"].get("input_metadata")
    )
    if input_metadata is not None:
        metadata["input_metadata"] = copy.deepcopy(dict(input_metadata))
    if saved is not None:
        old = saved[0]["details"]
        if canonical_json(saved[2]) != canonical_json(plan.games):
            raise ValueError("Resolved games differ from the saved immutable plan")
        for name in metadata:
            if canonical_json(old.get(name)) != canonical_json(metadata[name]):
                raise ValueError(f"Saved tournament {name} differs from this request")
    memory_manifest: dict[str, Any] = {
        "schema_version": RUN_SCHEMA_VERSION,
        "metric_schema_id": METRIC_SCHEMA_ID,
        # The snapshot's pin (14 for a pre-Red-Zone (14, 2, 3) snapshot). This
        # manifest only feeds the record accessors built here. The result a
        # caller gets reports the pin through CanonicalView's metadata.
        "metric_schema_version": config["compatibility"]["scalar_schema"],
        "run_id": "in-memory-" + uuid4().hex,
        "configurations": configurations,
        "systems": {},
        "passes": {},
        "tables": {},
        "details": metadata,
    }
    initial_manifest = memory_manifest if saved_manifest is None else saved_manifest
    _check_saved_job_options(
        initial_manifest,
        plan,
        metrics=metrics,
        full=per_job_full,
        replays=per_job_replay,
    )
    preflight = TournamentRecords(
        record_config,
        plan.games,
        plan.jobs,
        verifier,
        manifest=initial_manifest,
        run_dir=None if resume_from is None else Path(resume_from),
    )
    preflight.require_coverage(
        metrics=metrics, full_ids=full_ids, replay_ids=selected_replays
    )
    if any(game["origin"] is not None for game in plan.games):
        preflight.verify_stored_ratings(descriptors)
    by_id = {game["logical_game_id"]: game for game in plan.games}
    job_games = {
        job["job_id"]: tuple(
            by_id[identifier] for identifier in job["logical_game_ids"]
        )
        for job in plan.jobs
    }
    pending_jobs = [
        job
        for job in plan.jobs
        if not all(preflight.completed(game) for game in job_games[job["job_id"]])
    ]
    if pending_jobs and config["compatibility"]["replay_schema"] == 3:
        raise ValueError(
            "Snapshot configurations were saved before the Red Zone rule; recorded "
            "games can be reused, but new games need a snapshot prepared with "
            "current configurations."
        )
    verify_tournament_environment(config, verifier, execution=bool(pending_jobs))
    completed_games = tuple(game for game in plan.games if preflight.completed(game))
    preflight_evidence = None
    if completed_games:
        completed_ids = {game["logical_game_id"] for game in completed_games}
        completed_jobs = tuple(
            {
                **job,
                "logical_game_ids": [
                    identifier
                    for identifier in job["logical_game_ids"]
                    if identifier in completed_ids
                ],
            }
            for job in plan.jobs
            if any(
                identifier in completed_ids for identifier in job["logical_game_ids"]
            )
        )
        completed_records = TournamentRecords(
            record_config,
            completed_games,
            completed_jobs,
            verifier,
            manifest=initial_manifest,
            run_dir=None if resume_from is None else Path(resume_from),
        )
        preflight_evidence = prepare_reuse_evidence(
            completed_records,
            tuple(row for row in schedule if row.episode_id in completed_ids),
            _match_rows(completed_records, metrics),
            participants=participants,
            registrations=registrations,
            require_complete_population=len(completed_games) == len(plan.games),
            verified_configurations=verified_configs,
        )
    for job in pending_jobs:
        for entrant in (job["team_a"], job["team_b"]):
            if entrant in bindings or (entrant == challenger_id and method is not None):
                continue
            controller = participants[entrant]["controller"]
            if controller.get("factory") == "unknown:requires_supplied_system":
                raise ValueError(
                    "Saved entrant has no reload reference; supply its matching "
                    f"System or Policy: {participants[entrant]['name']}"
                )
            if controller["kind"] == "bundle":
                verifier.require(controller["asset_ids"])
    if (
        saved_manifest is not None
        and saved_manifest.get("tournament_summary", {})
        .get("qualification", {})
        .get("status")
        == "complete"
    ):
        if preflight_evidence is None or pending_jobs:
            raise ValueError("Saved complete summary has unfinished required games")
        if asset_locations != saved_manifest["tournament_reuse"].get(
            "asset_locations", {}
        ) or any(
            entry.get("result_state", {}).get("status") != "complete"
            for entry in saved_manifest["passes"].values()
            if entry["phase"] == "tournament"
        ):
            with RunWriter(
                resume_from=resume_from,
                phase="tournament",
                pass_id="schedule",
                details=metadata,
            ) as finalizer:
                finalizer._set_tournament_asset_locations(asset_locations)
                finalizer.reaffirm_tournament_result(schedule_digest)
                saved_manifest = finalizer._details
            preflight = TournamentRecords(
                record_config,
                plan.games,
                plan.jobs,
                verifier,
                manifest=saved_manifest,
                run_dir=Path(cast(str | Path, resume_from)),
            )
        return _saved_result(
            Path(cast(str | Path, resume_from)),
            saved_manifest,
            metadata,
            plan,
            preflight,
            preflight_evidence,
        )
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
            writer.set_tournament_coordinator(schedule_digest)
            manifest = writer._details
        else:
            manifest = memory_manifest
            manifest["passes"]["coordinator"] = {
                "phase": "tournament",
                "pass_id": "schedule",
                "pass_role": "tournament_coordinator",
                "details": metadata,
                "episodes": {},
                "completed_episode_ids": [],
                "result_state": {
                    "version": 1,
                    "status": "incomplete",
                    "reason": None,
                    "schedule_digest": schedule_digest,
                },
            }
        reuse = {
            "version": 1,
            "source_descriptors": list(plan.source_descriptors),
            "execution_plan": list(plan.jobs),
            "budget": plan.budget,
            "challenger_id": challenger_id,
            "state": "incomplete",
        }
        if writer is not None:
            reuse = writer._install_tournament_plan(record_config, plan.games, reuse)
            writer._set_tournament_asset_locations(asset_locations)
            manifest = writer._details
        else:
            reuse.update(
                config_sha256=sha256(_json_bytes(record_config)).hexdigest(),
                games_sha256=sha256(
                    b"".join(_json_bytes(game) for game in plan.games)
                ).hexdigest(),
            )
            manifest["tournament_reuse"] = reuse
        memory_rows: dict[str, list[dict[str, Any]]] = {
            "match_results.csv": [],
            "full_metrics.csv": [],
        }
        full_tables: list[dict[str, Any]] = []
        replays: list[Any] = []
        executed = 0
        observed_sampling: dict[str, dict[str, Any]] = {}
        residents: dict[tuple[str, bytes, str], System | Policy] = {}
        cleanup.callback(residents.clear)
        by_id = {game["logical_game_id"]: game for game in plan.games}
        jobs_by_id = {job["job_id"]: job for job in plan.jobs}

        def pending() -> Generator[TournamentJob]:
            """Yield exact prepared jobs; completed games remain with the writer."""
            current = TournamentRecords(
                record_config,
                plan.games,
                plan.jobs,
                verifier,
                manifest=manifest,
                run_dir=None if writer is None else writer.run_dir,
                memory=memory_rows,
            )
            for job in plan.jobs:
                if all(
                    current.completed(by_id[identifier])
                    for identifier in job["logical_game_ids"]
                ):
                    continue
                identifier = job["job_id"]
                yield TournamentJob(
                    job_id=identifier,
                    first=job["team_a"],
                    second=job["team_b"],
                    episodes=tuple(specs_by_job[identifier]),
                    seed=job["root_seed"],
                    phase=job["phase"],
                    pass_id=job["pass_id"],
                    full_metrics_episodes=tuple(per_job_full[identifier]),
                    replay_episodes=tuple(per_job_replay[identifier]),
                )

        @contextmanager
        def load_pair(
            task: TournamentJob,
        ) -> Generator[tuple[System | Policy, System | Policy]]:
            """Check actual method sampling and streams before shared dispatch."""
            job = jobs_by_id[task.job_id]
            with _active_pair(
                task.first,
                task.second,
                participants,
                verifier,
                challenger_id,
                method,
                bindings,
                residents=residents,
            ) as (first, second):
                for loaded in (first, second):
                    fact = method_sampling_fact(loaded)
                    prior_fact = observed_sampling.setdefault(loaded.name, fact)
                    if prior_fact != fact:
                        raise ValueError(
                            "Loaded method sampling behavior changed between jobs"
                        )
                expected_stream = (
                    "episode-fold-in-v1"
                    if isinstance(first, Policy) and isinstance(second, Policy)
                    else "evaluation-systems-v1"
                )
                if any(
                    by_id[identifier]["execution"][key] != expected_stream
                    for identifier in job["logical_game_ids"]
                    for key in (
                        "action_stream_version",
                        "initialization_stream_version",
                    )
                ):
                    raise ValueError(
                        "Declared RNG stream is incompatible with "
                        "this job's method execution path"
                    )
                yield first, second

        for task, result in execute_tournament_jobs(
            pending(),
            load_pair,
            writer=writer,
            run_id=manifest["run_id"],
            num_envs=num_envs,
            metrics=metrics,
            chunk_size=chunk_size,
            registered_maps={
                source["map_id"]: source["registered_map"]
                for source in config["conditions"]["map_sources"]
                if source["registered_map"] is not None
            },
        ):
            job = jobs_by_id[task.job_id]
            executed += len(result.episodes)
            first_name = participants[task.first]["name"]
            second_name = participants[task.second]["name"]
            if writer is None:
                entry = _memory_pass(result, per_job_full[job["job_id"]], metrics)
                manifest["passes"][
                    json.dumps((job["phase"], job["pass_id"]), separators=(",", ":"))
                ] = entry
                manifest["configurations"].update(result.metadata["configurations"])
                manifest["systems"].update(result.metadata["systems"])
                raw_schedule = tuple(
                    TournamentMatch(
                        spec.episode_id,
                        cast(int, cast(Mapping[str, Any], spec.metadata)["block_id"]),
                        spec.random_seed_id,
                        cast(int, spec.map_id),
                        first_name,
                        second_name,
                        bootstrap_group=cast(
                            str | None,
                            cast(Mapping[str, Any], spec.metadata)["bootstrap_group"],
                        ),
                    )
                    for spec in specs_by_job[job["job_id"]]
                )
                memory_rows["match_results.csv"].extend(
                    _memory_matches(result, raw_schedule)
                )
                full_tables.append(result.full_metrics)
                if result.full_metrics:
                    for index in range(len(result.full_metrics["episode_id"])):
                        memory_rows["full_metrics.csv"].append(
                            {
                                name: values[index].item()
                                if hasattr(values[index], "item")
                                else values[index]
                                for name, values in result.full_metrics.items()
                            }
                        )
                replays.extend(result.replays)
            else:
                writer.flush()
                manifest = writer._details
        records = TournamentRecords(
            record_config,
            plan.games,
            plan.jobs,
            verifier,
            manifest=manifest,
            run_dir=None if writer is None else writer.run_dir,
            memory=memory_rows,
        )
        matches = _match_rows(records, metrics)
        evidence = (
            {**preflight_evidence, "run_id": records.manifest["run_id"]}
            if preflight_evidence is not None and not pending_jobs
            else prepare_reuse_evidence(
                records,
                schedule,
                matches,
                participants=participants,
                registrations=registrations,
                verified_configurations=verified_configs,
            )
        )
        outcomes_by_origin = {origin_key(row): row["outcome"] for row in matches}
        outcomes = {
            game["logical_game_id"]: outcomes_by_origin[
                origin_key(records.origin(game))
            ]
            for game in plan.games
        }
        statistics = summarize_tournament(
            schedule,
            outcomes,
            seed=config["analysis"]["bootstrap_seed"],
            opponent_weights=weights,
            method_sampling=_recorded_method_sampling(
                records,
                participants,
                observed_sampling,
                {game["logical_game_id"] for game in completed_games},
            ),
        )
        headline = None if metrics == "none" else summarize_headlines(matches, evidence)
        qualification = {
            "version": 1,
            "status": "complete",
            "schedule_digest": schedule_digest,
            "population_system_ids": sorted(participants),
            "pairing_protocol": "fixed-team-spawn-v1",
            "metrics": metrics,
            "evidence": evidence,
        }
        if writer is not None:
            writer._use_tournament_records(records)
            writer.write_tournament_results(
                statistics, headline=headline, qualification=qualification
            )
            writer.reaffirm_tournament_result(schedule_digest)
            records = TournamentRecords(
                record_config,
                plan.games,
                plan.jobs,
                verifier,
                manifest=writer._details,
                run_dir=writer.run_dir,
            )
        else:
            manifest["tournament_summary"] = {
                "digest": "in-memory-qualified",
                "metadata": statistics.metadata,
                "qualification": qualification,
            }
            manifest["tournament_reuse"]["state"] = "complete"
            manifest["passes"]["coordinator"]["result_state"]["status"] = "complete"
        return _result(
            metadata,
            plan,
            records,
            matches,
            statistics.tournament_results,
            statistics.matchup_results,
            statistics.map_results,
            () if headline is None else headline,
            _merge_columns(full_tables),
            replays,
            None if writer is None else writer.paths,
            statistics.metadata,
            qualification,
            executed_this_call=executed,
        )


def _resolve_challenger(
    system: System | Policy | str | None | Omitted,
    previous: Mapping[str, Any] | None,
) -> tuple[
    System | Policy | str | None, Mapping[str, Any] | None, Mapping[str, Any] | None
]:
    """Resolve one challenger while preserving saved loader and omission rules.

    Strings remain references until the resolved runner loads and freezes them.
    An omitted saved live object cannot be reconstructed and raises a useful error.
    Explicit None asserts no challenger. The returned descriptor/assets are
    borrowed saved evidence. The resolved runner checks loaded identities before
    writing results.
    This helper does no loading or copying. The resolved runner prepares methods;
    researchers own their provider processes and memory.
    """
    descriptor = None
    extra_assets = None
    if isinstance(system, Omitted):
        system = None
        if previous is not None and previous["challenger_id"] is not None:
            if not previous["challenger_loadable"]:
                raise ValueError(
                    "Saved challenger has no immutable loader; "
                    "supply the matching System"
                )
            descriptor = next(
                row
                for row in previous["participant_descriptors"]
                if row["entrant_id"] == previous["challenger_id"]
            )
            extra_assets = previous["participant_assets"]
    elif (
        system is None
        and previous is not None
        and previous["challenger_id"] is not None
    ):
        raise ValueError("Explicit system=None conflicts with the saved challenger")
    if (
        previous is not None
        and previous["challenger_id"] is not None
        and previous["challenger_loadable"]
    ):
        descriptor = next(
            row
            for row in previous["participant_descriptors"]
            if row["entrant_id"] == previous["challenger_id"]
        )
        extra_assets = previous["participant_assets"]
    return system, descriptor, extra_assets


def run_canonical_tournament(
    system: System | Policy | str | None | Omitted = OMITTED,
    *,
    config: str | Path | Mapping[str, Any] | None = None,
    games_per_opponent: int | None | Omitted = OMITTED,
    rerun_existing: bool | Omitted = OMITTED,
    metrics: MetricMode | Omitted = OMITTED,
    save_replays: int | Omitted = OMITTED,
    full_metrics_episodes: Iterable[int] | Omitted = OMITTED,
    replay_episodes: Iterable[int] | Omitted = OMITTED,
    output_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
    num_envs: int = 128,
    chunk_size: int = 16,
) -> CanonicalTournamentResult:
    """Compare the released field, optionally adding one challenger.

    Parameters
    ----------
    system : System, Policy, str or None, default=None
        One challenger, or None for the selected field alone. Resume inherits an
        omitted challenger only when its saved loader can reproduce it; otherwise
        supply the matching method. Explicit None asserts field-only execution.
    config : JSON path, mapping or None
        Unchanged pinned official snapshot. None selects the installed release
        once, or the saved snapshot on resume. Version 1 pins twelve entrants;
        version 2 reads its size from the frozen field. Missing releases fail; custom
        populations/rules use run_tournament(config=...).
    games_per_opponent : int or None
        None inherits the snapshot budget. A positive integer divisible by twice
        the map count applies equally to every matchup. A different budget is a
        research override and cannot qualify promotion. Booleans are invalid.
    rerun_existing : bool, default=False
        Reuse verified incumbent games. True explicitly reruns the whole selected
        schedule; missing coverage never triggers that choice automatically.
    metrics : {'priority', 'full', 'none'}, default='priority'
        Requested report coverage. None-mode still retains outcomes and rankings
        but skips headline calculation. Full source reports satisfy priority.
    save_replays : int, default=0
        First N logical scheduled games to capture. Nonzero shorthand must agree
        with a supplied replay_episodes selection.
    full_metrics_episodes, replay_episodes : iterable of int, default=()
        Logical scheduled IDs selected for capture. Original execution IDs and
        stored replay identities remain unchanged. Missing reused assets fail.
    output_dir, resume_from : path or None
        New output parent or exact saved run; supply at most one. Neither means
        no files. Omitted scientific/capture settings inherit saved conditions;
        explicit conflicts fail before recovery or output creation.
    num_envs : int, default=128
        Fixed maximum parallel games within the current matchup. Matchups run
        one at a time. This execution setting may change on resume.
    chunk_size : int, default=16
        Decisions per evaluator chunk. Explicit values are honored; changing it
        does not change the declared scientific schedule.

    Returns
    -------
    CanonicalTournamentResult
        Complete shared result tables and snapshot/budget/reuse metadata. Already
        saved summaries load without fitting. No-file results borrow source files.

    Raises
    ------
    ValueError, TypeError
        Snapshot, method, budget, captures, evidence or resume settings conflict.
        A snapshot saved before the Red Zone rule can only reuse recorded games;
        a call that needs a new game raises ValueError before any output.
    OSError, RuntimeError
        Required assets, execution, fitting or durable output fail. Earlier saved
        games remain resumable. Backend allocation and provider errors propagate;
        no calls are automatically retried. Researchers own provider processes
        and memory. Lowering num_envs does not reduce model weights.

    Notes
    -----
    Host-only orchestration. It owns no learner and never admits, promotes or
    publishes a controller. Fixture approval is not scientific qualification.
    Models load only for active unfinished matchups. None-mode is the string
    "none", not Python None. The signature's private omission marker preserves
    explicit resume assertions; the ordinary defaults above remain unchanged.
    """
    saved = _saved_plan(resume_from)
    previous = None if saved is None else saved[0]["details"]
    if previous is not None and previous.get("official_snapshot_verified") is not True:
        raise ValueError("Saved tournament is custom; resume it through run_tournament")
    selected = resolve_tournament_config(
        config,
        official=True,
        saved=None if previous is None else previous["canonical_config"],
    )
    system, descriptor, extra_assets = _resolve_challenger(system, previous)
    supplied_budget = games_per_opponent
    if isinstance(supplied_budget, Omitted):
        budget = (
            None
            if previous is None
            else previous["budget"]["resolved_games_per_opponent"]
        )
    else:
        budget = supplied_budget
        if (
            previous is not None
            and (
                selected["conditions"]["games_per_opponent"]
                if budget is None
                else budget
            )
            != previous["budget"]["resolved_games_per_opponent"]
        ):
            raise ValueError(
                "games_per_opponent differs from the saved resolved budget"
            )
    full = (
        full_metrics_episodes
        if isinstance(full_metrics_episodes, Omitted)
        else list(full_metrics_episodes)
    )
    replay = (
        replay_episodes
        if isinstance(replay_episodes, Omitted)
        else list(replay_episodes)
    )
    return _run_resolved_tournament(
        selected,
        system=system,
        challenger_descriptor=descriptor,
        challenger_assets=extra_assets,
        require_reuse=True,
        games_per_opponent=budget,
        rerun_existing=cast(
            bool, _option(rerun_existing, previous, "rerun_existing", False)
        ),
        metrics=cast("MetricMode", _option(metrics, previous, "metrics", "priority")),
        save_replays=cast(int, _option(save_replays, previous, "save_replays", 0)),
        full_metrics_episodes=cast(
            Iterable[int], _option(full, previous, "full_metrics_episodes", ())
        ),
        replay_episodes=cast(
            Iterable[int], _option(replay, previous, "replay_episodes", ())
        ),
        output_dir=output_dir,
        resume_from=resume_from,
        num_envs=num_envs,
        chunk_size=chunk_size,
        _explicit_config=config is not None,
        _official_snapshot_verified=True,
    )


def _run_configured_tournament(
    config: str | Path | Mapping[str, Any] | None,
    *,
    challenger: System | Policy | str | None | Omitted = OMITTED,
    games_per_opponent: int | Omitted = OMITTED,
    rerun_existing: bool | Omitted = OMITTED,
    metrics: MetricMode | Omitted,
    full_metrics_episodes: Iterable[int] | Omitted,
    replay_episodes: Iterable[int] | Omitted,
    save_replays: int | Omitted,
    output_dir: str | Path | None,
    resume_from: str | Path | None,
    num_envs: int,
    chunk_size: int,
    bindings: Mapping[str, System | Policy] | None = None,
    input_metadata: Mapping[str, Any] | None = None,
    challenger_prepared: bool = False,
) -> CanonicalTournamentResult:
    """Resolve supplied or saved field conditions through the one tournament engine.

    config is a checked full descriptor or its path. Omitted scientific/output
    values inherit the saved run; a new supplied config owns its conditions.
    A new explicit game budget must equal that config's budget. challenger uses
    the same loading, identity and saved-omission rules as the canonical shortcut.
    bindings contains frozen local entrants; input_metadata supplies generic
    result aliases from short/list normalization. Neither is a new execution path.
    Caller-owned methods keep their existing lifetime. challenger_prepared
    borrows a method already frozen by list input preparation; references and
    other live methods are prepared once by the resolved runner.
    Invalid settings fail before writer recovery. This call does not grant
    official release qualification or publish/admit any method.
    """
    saved = _saved_plan(resume_from)
    previous = None if saved is None else saved[0]["details"]
    selected = resolve_tournament_config(
        config,
        official=False,
        saved=None if previous is None else previous["canonical_config"],
    )
    method, descriptor, assets = _resolve_challenger(challenger, previous)
    expected_budget = (
        selected["conditions"]["games_per_opponent"]
        if previous is None
        else previous["budget"]["resolved_games_per_opponent"]
    )
    if (
        not isinstance(games_per_opponent, Omitted)
        and games_per_opponent != expected_budget
    ):
        raise ValueError("games_per_opponent differs from the configured tournament")
    return _run_resolved_tournament(
        selected,
        system=method,
        challenger_descriptor=descriptor,
        challenger_assets=assets,
        games_per_opponent=expected_budget,
        rerun_existing=cast(
            bool, _option(rerun_existing, previous, "rerun_existing", False)
        ),
        metrics=cast("MetricMode", _option(metrics, previous, "metrics", "priority")),
        full_metrics_episodes=cast(
            Iterable[int],
            _option(
                full_metrics_episodes
                if isinstance(full_metrics_episodes, Omitted)
                else list(full_metrics_episodes),
                previous,
                "full_metrics_episodes",
                (),
            ),
        ),
        replay_episodes=cast(
            Iterable[int],
            _option(
                replay_episodes
                if isinstance(replay_episodes, Omitted)
                else list(replay_episodes),
                previous,
                "replay_episodes",
                (),
            ),
        ),
        save_replays=cast(int, _option(save_replays, previous, "save_replays", 0)),
        output_dir=output_dir,
        resume_from=resume_from,
        num_envs=num_envs,
        chunk_size=chunk_size,
        _bindings=bindings,
        _system_prepared=challenger_prepared,
        _input_metadata=input_metadata,
        _explicit_config=config is not None,
        _official_snapshot_verified=(
            previous is not None and previous.get("official_snapshot_verified") is True
        ),
    )
