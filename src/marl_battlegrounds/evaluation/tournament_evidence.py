"""Verify original game ownership before canonical statistics or publication.

The canonical runner calls prepare_reuse_evidence once after obtaining complete
rows through TournamentRecords. It reuses the task's spawn-bank comparison and
existing registration/configuration identities. Only preparation imports JAX;
saved result loading and evidence-to-row checks remain host-only elsewhere.
"""

# Shared private validators remain the single owners of these contracts.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from marl_battlegrounds.evaluation.tournament_headlines import (
    content_digest,
    match_digest,
)
from marl_battlegrounds.evaluation.tournament_records import (
    TournamentRecords,
    origin_text,
)

if TYPE_CHECKING:
    from marl_battlegrounds.evaluation.tournament_schedule import TournamentMatch


def prepare_reuse_evidence(
    records: TournamentRecords,
    schedule: Sequence[TournamentMatch],
    matches: Sequence[Mapping[str, Any]],
    *,
    participants: Mapping[str, Mapping[str, Any]],
    registrations: Mapping[str, Mapping[str, Any]],
    require_complete_population: bool = True,
    verified_configurations: Mapping[str, tuple[Mapping[str, Any], Any]] | None = None,
) -> dict[str, Any]:
    """Bind complete original rows to physical spawn pairs and immutable entrants.

    Parameters
    ----------
    records : TournamentRecords
        Shared source accessor for the current logical run. Required sources have
        already passed integrity and capture preflight.
    schedule : sequence of TournamentMatch
        Logical analysis projection in exactly the selected game order. Its IDs
        are never execution IDs or replacements for stored original row fields.
    matches : sequence of mappings
        Complete narrow original match rows. Every selected game occurs once.
    participants : mapping
        Entrant ID to resolved descriptor, including stable controller_id/name.
    registrations : mapping
        Entrant ID to the expected original frozen registration. Reused entries
        come from verified assets; a supplied challenger uses its frozen method.
    require_complete_population : bool
        True also checks the full population through the statistics authority.
        False is only for preflight of already completed records before executing
        missing games. It keeps every row, source, ownership and stream check;
        its evidence cannot qualify a complete summary.
    verified_configurations : mapping or None
        Private setup cache of content ID to (exact content, validated config).
        The caller owns its physical validation. Matching immutable contents need
        no second restoration or Core validation; other contents are checked here.

    Returns
    -------
    dict
        Version-2 evidence covering full origins, original passes/registrations,
        actual configuration contents, logical schedule and participant owners.
        Statistics, headlines and the writer consume this same verified object.

    Raises
    ------
    ValueError
        Rows, versions, owners, random coordinates or physical bank choices differ.

    Notes
    -----
    Host preparation only. Distinct configuration relationships are checked once
    using the existing task authority. No method, initializer or action is called.
    The input rows are not edited; all analysis-only coordinates stay separate.
    """
    with records.verifier.verification_scope():
        import jax

        from marl_battlegrounds.evaluation.evaluation_conditions import (
            restore_recorded_config,
        )
        from marl_battlegrounds.evaluation.models import canonical_digest_sha256
        from marl_battlegrounds.evaluation.tournament_assets import (
            _assert_known_facts,
            policy_recording_registration,
        )
        from marl_battlegrounds.evaluation.tournament_config import canonical_json
        from marl_battlegrounds.evaluation.tournament_statistics import _population
        from marl_battlegrounds.tasks import spawn_locations_for_source

        if len(schedule) != len(records.games) or len(matches) != len(schedule):
            raise ValueError(
                "canonical evidence requires every planned game exactly once"
            )
        rows = {origin_text(row): row for row in matches}
        if len(rows) != len(matches):
            raise ValueError("canonical match rows repeat an original game")
        outcomes: dict[int, int] = {}
        games: dict[str, Any] = {}
        systems: dict[str, Any] = {}
        used_registrations: dict[str, Any] = {}
        passes: dict[str, Any] = {}
        configurations: dict[str, Any] = {}
        configuration_bytes: dict[str, bytes] = {}
        checked_contents: set[tuple[str | None, str]] = set()
        restored: dict[str, Any] = {}
        choices: dict[tuple[str, str], int] = {}
        checked_registrations: set[tuple[str, str]] = set()
        participant_ids: dict[str, str] = {}
        for logical, planned in zip(schedule, records.games, strict=True):
            if logical.episode_id != planned["logical_game_id"]:
                raise ValueError(
                    "analysis schedule order differs from the selected logical games"
                )
            key = origin_text(records.origin(planned))
            row = rows.get(key)
            if row is None:
                raise ValueError("canonical game has no exact original match row")
            entry = records.entry(planned)
            if entry is None or not records.completed(planned):
                raise ValueError("canonical game has no durable original completion")
            manifest = records.source_manifest(planned)
            declaration = entry.get("episodes", {}).get(str(row["episode_id"]))
            if declaration is None:
                raise ValueError(
                    "canonical game is missing its original episode declaration"
                )
            execution = planned["execution"]
            details = entry.get("details", {})
            if details.get("seed") != execution["root_seed"] or any(
                execution[field] != details.get("rng_protocol")
                for field in ("action_stream_version", "initialization_stream_version")
            ):
                raise ValueError(
                    "canonical original random stream differs from its schedule"
                )
            for field, expected in (
                ("episode_id", execution["episode_id"]),
                ("seed_id", execution["seed_id"]),
                ("map_id", planned["map_id"]),
                ("config_id", planned["resolved_config_id"]),
                ("bootstrap_group", execution["bootstrap_group"]),
            ):
                if row.get(field) != expected:
                    raise ValueError(
                        f"canonical original {field} differs from its schedule"
                    )
            expected_declaration = {
                "configuration_digest": planned["resolved_config_id"],
                "source_config_id": planned["source_config_id"],
                "spawn_locations": planned["spawn_locations"],
                "comparison_kind": "verified_spawn_pair",
                "seed_id": execution["seed_id"],
                "map_id": planned["map_id"],
                "initial_state_digest": None,
            }
            if any(
                declaration.get(field) != value
                for field, value in expected_declaration.items()
            ):
                raise ValueError(
                    "canonical original start declaration differs "
                    "from the physical pair"
                )
            if declaration.get("block_id") != row.get("block_id"):
                raise ValueError(
                    "original match block differs from its declared comparison"
                )
            for field in ("outcome", "episode_length", "team_a_score", "team_b_score"):
                value = row.get(field)
                if type(value) is not int or value < 0:
                    raise ValueError(
                        f"canonical {field} requires an exact nonnegative integer"
                    )
            if row["outcome"] not in (1, 2, 3) or row["episode_length"] <= 0:
                raise ValueError("canonical game must have a real completed outcome")
            outcomes[logical.episode_id] = row["outcome"]
            source_id, resolved_id = (
                planned["source_config_id"],
                planned["resolved_config_id"],
            )
            for identifier in (source_id, resolved_id):
                content_owner = records.origin(planned).get("source_id"), identifier
                if content_owner in checked_contents:
                    continue
                content = manifest.get("configurations", {}).get(identifier)
                if content is None:
                    content = details.get("configurations", {}).get(identifier)
                if content is None:
                    raise ValueError(
                        "canonical source lacks actual configuration content"
                    )
                encoded = canonical_json(content)
                if (
                    identifier in configurations
                    and configuration_bytes[identifier] != encoded
                ):
                    raise ValueError(
                        "canonical sources disagree about configuration content"
                    )
                if identifier not in configurations:
                    prepared = (
                        None
                        if verified_configurations is None
                        else verified_configurations.get(identifier)
                    )
                    if prepared is not None:
                        if canonical_json(prepared[0]) != encoded:
                            raise ValueError(
                                "recorded configuration differs from verified content"
                            )
                        config = prepared[1]
                    else:
                        # Content saved before Red Zone keeps its historical ID.
                        try:
                            config, _ = restore_recorded_config(content, identifier)
                        except ValueError as error:
                            raise ValueError(
                                "canonical configuration content differs "
                                "from its identity"
                            ) from error
                    configurations[identifier] = content
                    configuration_bytes[identifier] = encoded
                    restored[identifier] = config
                checked_contents.add(content_owner)
            pair = source_id, resolved_id
            if pair not in choices:
                valid, choice = jax.device_get(
                    spawn_locations_for_source(
                        restored[resolved_id], restored[source_id]
                    )
                )
                if not bool(valid) or int(choice) not in (0, 1):
                    raise ValueError(
                        "canonical configuration is not one unambiguous "
                        "complete spawn choice"
                    )
                choices[pair] = int(choice)
            if choices[pair] != planned["spawn_locations"]:
                raise ValueError("canonical game used different spawn banks")
            owners: dict[str, str] = {}
            registered_owners = declaration.get(
                "system_ids", entry.get("system_ids", {})
            )
            for team in ("team_a", "team_b"):
                entrant_id = planned[team]
                participant = participants[entrant_id]
                registration_id = registered_owners.get(team)
                actual = manifest.get("systems", {}).get(registration_id)
                if actual is None or actual.get("name") != row.get(team + "_policy"):
                    raise ValueError(
                        "canonical team has no matching original registration"
                    )
                if (
                    registration_id in used_registrations
                    and used_registrations[registration_id] != actual
                ):
                    raise ValueError("canonical original registration content changed")
                if (registration_id, entrant_id) not in checked_registrations:
                    if canonical_digest_sha256(actual) != registration_id:
                        raise ValueError(
                            "canonical original registration content changed"
                        )
                    expected = registrations[entrant_id]
                    if actual.get("kind") == "policy" and "hooks" not in actual:
                        expected = policy_recording_registration(expected)
                    _assert_known_facts(expected, actual, "recorded registration")
                    checked_registrations.add((registration_id, entrant_id))
                used_registrations[registration_id] = actual
                identifier = entrant_id
                if (
                    identifier in systems
                    and systems[identifier]["name"] != participant["name"]
                ):
                    raise ValueError(
                        "canonical population repeats a controller identity"
                    )
                system = systems.setdefault(
                    identifier,
                    {
                        "name": participant["name"],
                        "controller_id": participant["controller_id"],
                        "registration_ids": [],
                    },
                )
                if registration_id not in system["registration_ids"]:
                    system["registration_ids"].append(registration_id)
                participant_ids[participant["name"]] = identifier
                owners[team + "_system_id"] = identifier
                owners[team + "_registration_id"] = registration_id
            pass_key = json.dumps(
                (row["run_id"], row["phase"], row["pass_id"]),
                separators=(",", ":"),
                ensure_ascii=False,
            )
            if pass_key in passes and passes[pass_key] != entry:
                raise ValueError("canonical original pass declarations conflict")
            passes[pass_key] = entry
            games[key] = {
                **{
                    field: row[field]
                    for field in ("run_id", "phase", "pass_id", "episode_id")
                },
                **owners,
                "logical_game_id": logical.episode_id,
                "block_id": logical.block_id,
                "source_config_id": source_id,
                "resolved_config_id": resolved_id,
                "spawn_locations": planned["spawn_locations"],
                "comparison_kind": "verified_spawn_pair",
                "execution": dict(execution),
                "source_id": records.origin(planned).get("source_id"),
                "recorded_metrics": records.coverage(planned),
            }
        if require_complete_population:
            _population(schedule, outcomes, None)
        serialized = [asdict(match) for match in schedule]
        return {
            "version": 2,
            "population_complete": require_complete_population,
            "protocol": "fixed-team-spawn-v1",
            "run_id": records.manifest["run_id"],
            "schedule": serialized,
            "schedule_digest": content_digest(serialized),
            "match_digest": match_digest(matches, full_origins=True),
            "games": games,
            "systems": systems,
            "registrations": used_registrations,
            "passes": passes,
            "configurations": configurations,
            "population_system_ids": sorted(systems),
            "participants": participant_ids,
            "source_descriptors": list(records.config["record_sources"]),
        }
