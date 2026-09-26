"""Prepare ordinary tournament inputs for the shared configured runner.

Short JSON fields and live entrant lists become checked version-2 descriptions
with inline metadata. The existing loaders, map, registration and schedule
owners supply the facts. No games run and no result or asset file is written.
Opaque live providers remain caller-owned and need the same live binding on
resume; unknown controller content never becomes verified evidence.
"""

from __future__ import annotations

# Preparation uses the same private map and source owners as evaluation.
# pyright: reportPrivateUsage=false
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from hashlib import sha256
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from marl_battlegrounds._method_loading import load_method
from marl_battlegrounds._tdm_assets import current_map_id
from marl_battlegrounds.evaluation.evaluate import normalize_episode_specs
from marl_battlegrounds.evaluation.evaluation_conditions import (
    OMITTED,
    Omitted,
    config_record,
)
from marl_battlegrounds.evaluation.metric_catalog import METRIC_SCHEMA_VERSION
from marl_battlegrounds.evaluation.policy_execution import Policy, System, policy
from marl_battlegrounds.evaluation.recording_identity import (
    normalize_system_registration,
)
from marl_battlegrounds.evaluation.revision import discover_code_revision_v2
from marl_battlegrounds.evaluation.sampling_evidence import method_sampling_fact
from marl_battlegrounds.evaluation.system_evaluation import (
    freeze_evaluation_method,
    prepare_evaluation_system,
    validate_evaluation_rosters,
)
from marl_battlegrounds.evaluation.tournament_assets import (
    AssetVerifier,
    controller_content_identity,
    environment_source_manifest,
    inference_dependency_lock,
    loaded_controller_registration,
    policy_recording_registration,
)
from marl_battlegrounds.evaluation.tournament_config import (
    CONFIG_FORMAT,
    canonical_json,
    load_tournament_config,
    read_config_json,
    snapshot_identity,
)
from marl_battlegrounds.evaluation.tournament_headlines import content_digest
from marl_battlegrounds.evaluation.tournament_reuse import CHALLENGER_PLACEHOLDER
from marl_battlegrounds.evaluation.tournament_schedule import (
    TournamentMatch,
    build_tournament_schedule,
)
from marl_battlegrounds.evaluation.tournament_statistics import (
    validate_opponent_weights,
)
from marl_battlegrounds.tasks import (
    CANONICAL_TDM_EVALUATION_MAP_IDS,
    AgentClassName,
    TDMMapInfo,
    _swap_spawn_banks,
    _validate_config_choices,
    canonical_tournament_rosters,
    list_tdm_maps,
)

if TYPE_CHECKING:
    from marl_battlegrounds.evaluation.policy_execution import _SystemExecution


_SHORT_FIELDS = frozenset(
    {
        "entrants",
        "maps",
        "rosters",
        "games_per_opponent",
        "episodes_per_pair",
        "seed",
        "seeds",
        "score_threshold",
        "max_steps",
        "red_zone_depth",
        "opponent_weights",
        "metrics",
        "save_replays",
        "full_metrics_episodes",
        "replay_episodes",
        "output_dir",
        "selection",
    }
)
_CONTENT_FIELDS = (
    "code",
    "parameters",
    "memory_template",
    "input_preparation",
    "decision_settings",
    "adapter_bindings",
    "external_state",
)


def resolve_tournament_budget(
    games_per_opponent: int | Omitted = OMITTED,
    episodes_per_pair: int | Omitted = OMITTED,
) -> int | Omitted:
    """Resolve the taught budget name and its historical alias without defaults.

    Omission of both returns OMITTED, so resume can inherit saved settings.
    Each supplied value must be a positive exact Python int; equal simultaneous
    values pass and conflicting values raise ValueError. None and bool are not
    budgets. Divisibility across maps and spawn choices belongs to the schedule.
    """
    supplied = [
        (name, value)
        for name, value in (
            ("games_per_opponent", games_per_opponent),
            ("episodes_per_pair", episodes_per_pair),
        )
        if not isinstance(value, Omitted)
    ]
    for name, value in supplied:
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if len(supplied) == 2 and supplied[0][1] != supplied[1][1]:
        raise ValueError("games_per_opponent and episodes_per_pair must agree")
    return OMITTED if not supplied else supplied[0][1]


def _reference(value: str, base: Path) -> str:
    """Resolve folder references while preserving built-in and factory spelling.

    Built-in names keep the shared loader's priority over a same-named folder.
    Existing folders, including names containing a colon, are made absolute.
    Other strings reach the shared loader unchanged; it owns useful failures.
    """
    try:
        policy(value)
        return value
    except ValueError:
        pass
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    if candidate.is_dir() or "/" in value or "\\" in value:
        return str(candidate.resolve())
    return value


def read_short_tournament_config(
    config: str | Path | Mapping[str, Any],
) -> tuple[dict[str, Any], Path] | None:
    """Read a short field declaration, or return None for a full descriptor.

    File inputs use strict JSON and resolve entrant folders, output_dir and
    selection declarations from the file's directory. Mappings are copied as finite
    JSON and use the current directory. Short fields require string entrant
    references. Unknown fields and invalid input shapes raise ValueError before
    a factory is loaded.
    The returned mapping retains supplied options; the runner resolves conflicts
    with explicit call arguments. This function never invokes factories or writes.
    """
    if isinstance(config, (str, Path)):
        path = Path(config).expanduser().resolve()
        raw, base = read_config_json(path), path.parent
    elif isinstance(cast(object, config), Mapping):
        raw, base = json.loads(canonical_json(dict(config))), Path.cwd()
    else:
        raise TypeError("Tournament config must be a JSON path or mapping")
    if raw.get("format") == CONFIG_FORMAT:
        return None
    extra = raw.keys() - _SHORT_FIELDS
    if extra:
        raise ValueError(f"Unknown short tournament fields: {sorted(extra)}")
    entrants = raw.get("entrants")
    if not isinstance(entrants, list) or not all(
        isinstance(item, str) and item.strip() for item in cast(list[object], entrants)
    ):
        raise ValueError("Short tournament entrants must be a list of references")
    raw["entrants"] = [_reference(item, base) for item in cast(list[str], entrants)]
    if "selection" in raw:
        from marl_battlegrounds.evaluation.population_selection import (
            read_selection_request,
        )

        supplied = raw["selection"]
        if isinstance(supplied, str):
            supplied = str((base / Path(supplied).expanduser()).resolve())
        elif isinstance(supplied, dict) and "population" in supplied:
            supplied = dict(cast(dict[str, Any], supplied))
            if isinstance(supplied["population"], str):
                supplied["population"] = str(
                    (base / Path(supplied["population"]).expanduser()).resolve()
                )
        raw["selection"] = read_selection_request(
            cast(str | Path | Mapping[str, Any], supplied)
        )
    if raw.get("output_dir") is not None:
        if not isinstance(raw["output_dir"], str) or not raw["output_dir"].strip():
            raise ValueError("output_dir must be a nonempty folder path or null")
        output = Path(raw["output_dir"]).expanduser()
        raw["output_dir"] = str((base / output).resolve())
    return raw, base


def _inline(assets: dict[str, Any], name: str, value: object, role: str) -> str:
    """Add actual canonical JSON metadata and return its local asset key."""
    encoded = canonical_json(value)
    assets[name] = {
        "sha256": sha256(encoded).hexdigest(),
        "size_bytes": len(encoded),
        "role": role,
        "inline": json.loads(encoded),
    }
    return name


def _participant(
    method: System | Policy, reference: str | None, assets: dict[str, Any]
) -> dict[str, Any]:
    """Record one frozen method without claiming unknown executable evidence."""
    registration = loaded_controller_registration(method)
    identity = sha256(canonical_json(registration)).hexdigest()
    controller: dict[str, Any] = {"content": dict.fromkeys(_CONTENT_FIELDS)}
    controller.update(
        {"kind": "factory", "factory": "unknown:requires_supplied_system"}
        if reference is None
        else {"kind": "reference", "reference": reference}
    )
    controller_id, _ = controller_content_identity(
        controller, AssetVerifier({"assets": {}})
    )
    return {
        "entrant_id": "entrant-" + identity,
        "name": method.name,
        "controller_id": controller_id,
        "controller": controller,
        "registration_asset": _inline(
            assets, "registration-" + identity, registration, "registration"
        ),
        "elo": None,
        "result_ref": None,
    }


def _prepare_tournament_method(value: System | Policy | str) -> System | Policy:
    """Load and freeze one entrant without taking ownership of its provider.

    value is a live method or a shared-loader reference. Reference factories run
    on the calling thread. Return the method with its numerical values frozen;
    opaque state remains caller-owned. This helper does not call init, apply or
    cleanup. Loading and allocation errors propagate without retries.
    """
    return freeze_evaluation_method(
        load_method(value) if isinstance(value, str) else value
    )


def prepare_tournament_inputs(
    entrants: Sequence[System | Policy | str],
    *,
    maps: Sequence[int | TDMMapInfo] | None = None,
    games_per_opponent: int = 100,
    seed: int = 0,
    seeds: Sequence[int] | None = None,
    rosters: Mapping[str, Sequence[AgentClassName]] | None = None,
    score_threshold: int = 20,
    max_steps: int = 300,
    red_zone_depth: float = 5.0,
    opponent_weights: Mapping[str, float] | None = None,
    challenger: System | Policy | None = None,
    selection: str | Path | Mapping[str, Any] | None = None,
    _retain_references: bool = True,
) -> tuple[dict[str, Any], dict[str, System | Policy], dict[str, Any]]:
    """Freeze a fresh field and prepare metadata for the shared tournament runner.

    Parameters
    ----------
    entrants : sequence of System, Policy or reference strings
        At least two distinct method names. Each reference is loaded once and
        each method's numerical values are frozen once. Opaque state is retained.
    maps : sequence of int or TDMMapInfo, or None
        Distinct current registered maps. None uses the canonical test maps.
    games_per_opponent : int, default=100
        Total games per unordered pair, divided equally across maps and spawn
        ends. A seeds list also divides this same total equally across roots.
    seed : int, default=0
        uint32 root for the ordinary route and summary resampling. Nonboolean
        Integral values, including NumPy integer scalars, become Python ints.
    seeds : sequence of int or None
        Optional distinct uint32 roots. Does not multiply the game budget.
    rosters : mapping or None
        Ordered team_a and team_b class lists. None uses canonical team rosters.
    score_threshold, max_steps : int
        Score target and horizon, default 20 points and 300 ticks.
    red_zone_depth : float, default=5.0
        Red Zone depth in map units; 0.0 disables the extra death point.
    opponent_weights : mapping or None
        Positive weights keyed by entrant name; None gives equal weights.
    challenger : System, Policy or None
        Already loaded and frozen optional challenger. Its kind chooses the
        companion stream; it is not an incumbent and is not frozen here again.
        None prepares provisional Policy-compatible companion templates. Their
        explicit participant-method-v1 rule binds streams to checked method
        kinds when a challenger is inserted, without changing RNG coordinates.

    selection : path, mapping or None, default=None
        Optional validation selection rule or test-stage population reference.
        Freeze it against this complete field before any game. Selection fields
        cannot add a separate challenger or replace their declared budget.
    _retain_references : bool, default=True
        Keep loaded reference methods in returned bindings for reuse. False
        retains only live caller methods; reference methods can then be loaded
        from their checked descriptors. This releases Python references only;
        it does not promise that an allocator returns memory to the device.

    Returns
    -------
    tuple of three dicts
        Checked version-2 descriptor, frozen methods keyed by entrant ID, and
        ordinary list metadata. Assets contain actual inline metadata, no model
        payloads or invented outcomes. The caller owns provider lifetime.

    Raises
    ------
    ValueError, TypeError
        Invalid names, budgets, seeds, maps, rosters or shared method contracts.
    OSError
        Required input folders or package source cannot be read.

    Notes
    -----
    No action, initializer, environment step or output write occurs. Pair IDs,
    episode IDs and seed IDs preserve the ordinary sorted schedule. A supplied
    challenger always occupies Team A and starts after incumbent counters.
    Numerical registration hashing may transfer model values to the host.
    """
    if isinstance(entrants, (str, bytes)) or not isinstance(
        cast(object, entrants), Sequence
    ):
        raise TypeError("Tournament entrants must be a sequence of methods")
    resolve_tournament_budget(games_per_opponent)
    if (
        isinstance(seed, bool)
        or not isinstance(cast(object, seed), Integral)
        or not 0 <= seed <= 0xFFFFFFFF
    ):
        raise ValueError("seed must be a uint32 integer")
    seed = int(seed)
    roots = (seed,) if seeds is None else tuple(seeds)
    if (
        not roots
        or any(
            isinstance(value, bool)
            or not isinstance(cast(object, value), Integral)
            or not 0 <= value <= 0xFFFFFFFF
            for value in roots
        )
        or len(set(roots)) != len(roots)
    ):
        raise ValueError("seeds must contain distinct uint32 integers")
    roots = tuple(int(value) for value in roots)
    if rosters is None:
        team_a, team_b = canonical_tournament_rosters()
    else:
        if set(rosters) != {"team_a", "team_b"}:
            raise ValueError("rosters must contain team_a and team_b")
        team_a, team_b = rosters["team_a"], rosters["team_b"]
    map_ids = tuple(
        current_map_id(item) if isinstance(item, TDMMapInfo) else item
        for item in (CANONICAL_TDM_EVALUATION_MAP_IDS if maps is None else maps)
    )
    references = [
        _reference(item, Path.cwd()) if isinstance(item, str) else None
        for item in entrants
    ]
    if type(_retain_references) is not bool:
        raise TypeError("_retain_references must be a Boolean")
    assets: dict[str, Any] = {}
    participants: list[dict[str, Any]] = []
    names: list[str] = []
    scalar_policies: dict[str, bool] = {}
    execution: dict[str, _SystemExecution] = {}
    sampling: dict[str, dict[str, Any]] = {}
    bindings: dict[str, System | Policy] = {}
    for item, reference in zip(entrants, references, strict=True):
        method = _prepare_tournament_method(
            reference if reference is not None else item,
        )
        participant = _participant(method, reference, assets)
        participants.append(participant)
        names.append(method.name)
        scalar_policies[method.name] = isinstance(method, Policy)
        execution[method.name] = prepare_evaluation_system(method)[0]
        sampling[method.name] = method_sampling_fact(method)
        if reference is None or _retain_references:
            bindings[participant["entrant_id"]] = method
        del method
    schedule = build_tournament_schedule(
        names, map_ids, episodes_per_pair=games_per_opponent
    )
    map_ids = tuple(int(value) for value in map_ids)
    if games_per_opponent % (2 * len(map_ids) * len(roots)):
        raise ValueError(
            "games_per_opponent must divide equally across seeds, maps and spawn ends"
        )
    validate_opponent_weights(tuple(names), opponent_weights)
    ids = {row["name"]: row["entrant_id"] for row in participants}
    configurations: dict[str, Any] = {}
    source_ids: dict[int, str] = {}
    resolved_ids: dict[tuple[int, int], str] = {}
    map_sources = []
    registered = {info.map_id: info.model_dump(mode="json") for info in list_tdm_maps()}
    for spec in normalize_episode_specs(
        sorted(map_ids),
        len(map_ids),
        team_a,
        team_b,
        score_threshold,
        max_steps,
        red_zone_depth=red_zone_depth,
    ):
        map_id = cast(int, spec.map_id)
        source = spec.env_config
        _validate_config_choices(source, batched=False, both_spawn_choices=True)
        for first, second in {(row.team_a, row.team_b) for row in schedule}:
            validate_evaluation_rosters(execution[first], execution[second], source)
        source_id, content = config_record(source)
        swapped_id, swapped = config_record(_swap_spawn_banks(source))
        configurations.update({source_id: content, swapped_id: swapped})
        source_ids[map_id] = source_id
        resolved_ids[map_id, 0], resolved_ids[map_id, 1] = source_id, swapped_id
        map_sources.append(
            {
                "map_id": map_id,
                "split": registered[map_id]["split"],
                "source_config_asset": _inline(
                    assets, "config-" + source_id, content, "configuration"
                ),
                "source_config_id": source_id,
                "registered_map": registered[map_id],
            }
        )
    schedule = tuple(
        replace(
            row,
            source_config_id=source_ids[row.map_id],
            resolved_config_id=resolved_ids[row.map_id, cast(int, row.spawn_locations)],
        )
        for row in schedule
    )
    games: list[dict[str, Any]] = []
    companion: list[dict[str, Any]] = []
    groups: dict[str, dict[str, Any]] = {}
    orders: dict[tuple[str, str, int], int] = defaultdict(int)
    name_order = {name: index for index, name in enumerate(sorted(names))}

    def append(row: TournamentMatch, *, template: bool = False) -> None:
        """Translate one ordinary scheduled row without changing its RNG IDs."""
        scalar_first = (
            challenger is None or isinstance(challenger, Policy)
            if template
            else scalar_policies[row.team_a]
        )
        stream = (
            "episode-fold-in-v1"
            if scalar_first and scalar_policies[row.team_b]
            else "evaluation-systems-v1"
        )
        cell = row.team_a, row.team_b, row.map_id
        order = orders[cell]
        root_index = order % len(roots)
        group_id = (
            f"companion-{root_index}-{name_order[row.team_b]}"
            if template
            else f"root-{root_index}-{stream}"
        )
        groups.setdefault(
            group_id,
            {
                "group_id": group_id,
                "root_seed": roots[root_index],
                "action_stream_version": stream,
                "initialization_stream_version": stream,
                "next_episode_id": 1,
                "next_seed_id": 1,
                "extension": "independent-pairs-v1" if len(roots) == 1 else "none",
            },
        )
        (companion if template else games).append(
            {
                "logical_game_id": row.episode_id,
                "matchup_id": "matchup-"
                + sha256(canonical_json([row.team_a, row.team_b])).hexdigest(),
                "pair_id": row.block_id,
                "pair_order": order,
                "team_a": CHALLENGER_PLACEHOLDER if template else ids[row.team_a],
                "team_b": ids[row.team_b],
                "map_id": row.map_id,
                "source_config_id": source_ids[row.map_id],
                "resolved_config_id": resolved_ids[
                    row.map_id, cast(int, row.spawn_locations)
                ],
                "spawn_locations": row.spawn_locations,
                "execution": {
                    "group_id": group_id,
                    "root_seed": roots[root_index],
                    "seed_id": row.seed_id,
                    "episode_id": row.episode_id,
                    "action_stream_version": stream,
                    "initialization_stream_version": stream,
                    "bootstrap_group": None,
                },
                "origin": None,
                "prior_origin": None,
            }
        )
        if row.spawn_locations == 1:
            orders[cell] += 1

    for row in schedule:
        append(row)
    next_episode, next_pair = len(schedule) + 1, len(schedule) // 2 + 1
    for name in sorted(names):
        for map_id in sorted(map_ids):
            for _ in range(games_per_opponent // (2 * len(map_ids))):
                for spawn in (0, 1):
                    append(
                        TournamentMatch(
                            next_episode,
                            next_pair,
                            next_pair,
                            map_id,
                            CHALLENGER_PLACEHOLDER,
                            name,
                            pairing_protocol="fixed-team-spawn-v1",
                            spawn_locations=spawn,
                        ),
                        template=True,
                    )
                    next_episode += 1
                next_pair += 1
    for group in groups.values():
        group.update(next_episode_id=next_episode, next_seed_id=next_pair)
    _inline(assets, "games", games, "schedule")
    _inline(assets, "challenger-games", companion, "schedule")
    _inline(
        assets,
        "schedule",
        {
            "format": "marlbg-tournament-schedule",
            "version": 1,
            "games_asset": "games",
            "challenger_games_asset": "challenger-games",
            "companion_stream_rule": "participant-method-v1",
            "execution_groups": list(groups.values()),
            "selection_blocks": None,
            "allocation_order": [ids[name] for name in sorted(names)],
        },
        "schedule",
    )
    environment = environment_source_manifest()
    _inline(assets, "environment", environment, "qualification")
    _inline(assets, "dependencies", inference_dependency_lock(), "dependencies")
    revision = discover_code_revision_v2()
    descriptor: dict[str, Any] = {
        "format": CONFIG_FORMAT,
        "version": 2,
        "release": None,
        "participants": participants,
        "conditions": {
            "task": "tdm",
            "map_sources": map_sources,
            "rosters": {"team_a": list(team_a), "team_b": list(team_b)},
            "information_mode": "SharedObs",
            "games_per_opponent": games_per_opponent,
            "score_threshold": score_threshold,
            "max_steps": max_steps,
            "memory_rule": "fresh-per-game",
            "pairing_protocol": "fixed-team-spawn-v1",
            "schedule_asset": "schedule",
        },
        "analysis": {
            "method_id": "centered-davidson-v1",
            "bootstrap_replicates": 5000,
            "bootstrap_seed": seed,
            "opponent_weights": None
            if opponent_weights is None
            else {ids[name]: weight for name, weight in opponent_weights.items()},
        },
        "compatibility": {
            "package_version": revision.package_version,
            "code_revision": revision.model_dump(mode="json"),
            "environment_id": sha256(canonical_json(environment)).hexdigest(),
            "source_manifest_asset": "environment",
            "scalar_schema": METRIC_SCHEMA_VERSION,
            "run_schema": 2,
            "replay_schema": 4,
            "action_stream_version": "evaluation-systems-v1",
            "initialization_stream_version": "evaluation-systems-v1",
            "dependency_lock_asset": "dependencies",
        },
        "assets": assets,
        "record_sources": [],
    }
    if selection is not None:
        from marl_battlegrounds.evaluation.population_selection import bind_selection

        if challenger is not None:
            raise ValueError(
                "Declare every selection candidate in entrants before games"
            )
        descriptor["selection"] = bind_selection(
            selection, descriptor, entrant_order=names
        )
    descriptor["snapshot_id"] = snapshot_identity(descriptor)
    registrations = {
        row["name"]: assets[row["registration_asset"]]["inline"] for row in participants
    }
    descriptions: list[dict[str, Any]] = []
    for name in sorted(names):
        description = policy_recording_registration(registrations[name])
        if scalar_policies[name]:
            # Generic list metadata predates the writer's registration fields.
            for field in ("kind", "components", "parameter_status"):
                description.pop(field, None)
        descriptions.append(description)
    metadata: dict[str, Any] = {
        "seed": seed,
        "seeds": list(roots),
        "rosters": {"team_a": list(team_a), "team_b": list(team_b)},
        "input_references": {
            name: reference for name, reference in zip(names, references, strict=True)
        },
        "rng_protocol": "evaluation-systems-v1",
        "policies": descriptions,
        "method_sampling": sampling,
        "map_ids": sorted(map_ids),
        "episodes_per_pair": games_per_opponent,
        "num_matches": len(schedule),
        "score_threshold": score_threshold,
        "max_steps": max_steps,
        "red_zone_depth": red_zone_depth,
        "opponent_weights": None
        if opponent_weights is None
        else dict(opponent_weights),
        "configuration_ids_by_map": {
            str(key): value for key, value in source_ids.items()
        },
        "configurations": configurations,
        "registered_maps": {str(key): registered[key] for key in map_ids},
        "pairing_protocol": "fixed-team-spawn-v1",
        "schedule": [asdict(row) for row in schedule],
        "schedule_digest": content_digest([asdict(row) for row in schedule]),
        "participants": {
            str(row["name"]): normalize_system_registration(row, phase="tournament")[0]
            for row in descriptions
        },
        "source_request": {
            "entrants": [
                reference
                if reference is not None
                else {"entrant_id": ids[name], "name": name}
                for name, reference in zip(names, references, strict=True)
            ],
            "maps": sorted(map_ids),
            "rosters": {"team_a": list(team_a), "team_b": list(team_b)},
            "games_per_opponent": games_per_opponent,
            "seed": seed,
            "seeds": list(roots),
            "score_threshold": score_threshold,
            "max_steps": max_steps,
            "red_zone_depth": red_zone_depth,
            "opponent_weights": None
            if opponent_weights is None
            else dict(opponent_weights),
        },
    }
    if selection is not None:
        metadata["source_request"]["selection"] = descriptor["selection"]["declaration"]
    return load_tournament_config(descriptor), bindings, metadata
