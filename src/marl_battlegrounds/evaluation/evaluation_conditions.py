"""Resolve evaluation conditions before any writer or episode is changed.

The evaluator uses these host helpers for new schedules and saved-first resume.
Exact configurations, source banks, pair claims and capture selections have one
owner here. No helper runs a method, advances a game, opens a writer or changes a
saved file. Numerical execution remains in the existing evaluator.
"""

from __future__ import annotations

# These host helpers share private serialization and setup authorities.
# pyright: reportPrivateUsage=false
import json
from collections.abc import Iterable, Mapping, Sequence
from hashlib import sha256
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp
import numpy as np

from marl_battlegrounds.core.config import validate_env_config
from marl_battlegrounds.core.types import EnvConfig, ResolvedAgentProfile
from marl_battlegrounds.evaluation.run_writer import configuration_identity
from marl_battlegrounds.tasks import (
    CANONICAL_TDM_EVALUATION_MAP_IDS,
    AgentClassName,
    canonical_tournament_rosters,
    list_tdm_maps,
    spawn_locations_for_source,
)

if TYPE_CHECKING:
    from marl_battlegrounds.evaluation.evaluate import EpisodeSpec
    from marl_battlegrounds.evaluation.run_writer import RunWriter


class Omitted:
    """Mark an omitted scientific option without confusing an explicit default.

    This private sentinel is used only at a host call boundary. It never enters
    JAX or a recording. Hover text gives each option's ordinary effective default.
    """

    def __repr__(self) -> str:
        """Display the omission marker clearly when inspecting a signature."""
        return "<omitted>"


OMITTED = Omitted()


def option(
    value: Any,  # noqa: ANN401
    saved: Mapping[str, object] | None,
    name: str,
    default: Any,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """Resolve one omitted option, or check an explicit value against saved facts.

    ``saved`` contains the scientific options of the selected pass, or None for
    a new pass. Explicit values, including zero and empty selections, must agree.
    Sequence containers compare by their values. Return the existing value without
    copying arrays; configuration identities are checked separately.
    """
    if isinstance(value, Omitted):
        return default if saved is None else saved.get(name, default)
    if saved is not None and name in saved:
        left = (
            list(cast(Sequence[object], value))
            if isinstance(value, (tuple, list))
            else value
        )
        right = saved[name]
        right = (
            list(cast(Sequence[object], right))
            if isinstance(right, (tuple, list))
            else right
        )
        if left != right:
            raise ValueError(f"{name} differs from the saved evaluation conditions")
    return cast(Any, value)


def read_saved_pass(
    resume_from: str | Path | None,
    writer: RunWriter | None,
    phase: str,
    pass_id: str,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Read a selected pass without recovery, flushing or pass mutation.

    Return ``(manifest, pass)`` when that exact phase/pass exists, otherwise None.
    A missing resume directory, invalid schema or pending coordinated restore
    raises before the caller opens a writer. A caller-owned writer's recorded
    declarations are inspected in place; this helper does not change them.
    """
    if (
        not isinstance(cast(object, phase), str)
        or not phase
        or not isinstance(cast(object, pass_id), str)
        or not pass_id
    ):
        raise ValueError("phase and pass_id must be nonempty strings")
    if resume_from is not None and writer is not None:
        raise ValueError("writer cannot be combined with resume_from")
    manifest: dict[str, Any]
    if writer is not None:
        manifest = writer._details  # pyright: ignore[reportPrivateUsage]
    elif resume_from is not None:
        path = Path(resume_from) / "run_details.json"
        manifest = json.loads(path.read_text())
    else:
        return None
    if not isinstance(cast(object, manifest), dict) or manifest.get(
        "schema_version"
    ) not in (1, 2):
        raise ValueError("unsupported saved run schema")
    if manifest.get("recording_restore") is not None:
        raise ValueError("finish the pending coordinated recording restore first")
    for entry in manifest.get("passes", {}).values():
        if entry.get("phase") == phase and entry.get("pass_id") == pass_id:
            contract = entry.get("details", {}).get("evaluation_contract")
            if contract is not None and (
                not isinstance(contract, dict)
                or contract.get("version") != 1
                or contract.get("schedule_kind") not in ("generated", "explicit")
            ):
                raise ValueError("unsupported saved evaluation contract")
            return manifest, entry
    return None


def restore_config(content: Mapping[str, Any], *, validate: bool = True) -> EnvConfig:
    """Restore exact recorded EnvConfig content without consulting current assets.

    Content is the existing configuration-identity JSON layout. Array dtypes are
    the established Core config dtypes; scalar rules stay Python values. Reject
    missing/extra fields. validate=True also checks physical validity. Source-only
    evidence may use False when its unused banks need not be valid for the actual
    roster; the requested resolved config must still validate. This helper creates
    numerical arrays but never changes their values or exchanges their banks.
    """
    if set(content) != set(EnvConfig._fields):
        raise ValueError("saved configuration fields do not match EnvConfig")
    profile = content["agent_profile"]
    if not isinstance(profile, Mapping) or set(cast(Mapping[str, Any], profile)) != set(
        ResolvedAgentProfile._fields
    ):
        raise ValueError("saved roster fields do not match ResolvedAgentProfile")
    integer = {"team_ids", "class_ids", "out_of_combat_delay_steps"}
    restored_profile = ResolvedAgentProfile(
        **{
            name: jnp.asarray(
                value,
                dtype=jnp.bool_
                if name == "active_mask"
                else jnp.int32
                if name in integer
                else jnp.float32,
            )
            for name, value in cast(Mapping[str, Any], profile).items()
        }
    )
    data = dict(content)
    data["agent_profile"] = restored_profile
    for name in ("obstacles", "team_spawn_pad_positions"):
        data[name] = jnp.asarray(data[name], dtype=jnp.float32)
    data["team_respawn_wave_period_step_count"] = jnp.asarray(
        data["team_respawn_wave_period_step_count"], dtype=jnp.int32
    )
    config = EnvConfig(**data)
    if validate:
        validate_env_config(config)
    return config


def config_record(config: EnvConfig) -> tuple[str, dict[str, object]]:
    """Identify a scalar config with the same numerical normalization as reset."""

    def numerical(value: object) -> jax.Array:
        """Use the reset boundary's array dtype inference for one config leaf."""
        return jnp.asarray(value)

    return configuration_identity(jax.tree.map(numerical, config))


def capture_ids(
    episode_ids: Sequence[int],
    replay_episodes: Iterable[int],
    save_replays: int,
) -> tuple[int, ...]:
    """Resolve replay shorthand and explicit IDs in scheduled order.

    The count is a nonnegative integer within the schedule. When both nonempty
    forms are supplied, their sets must agree. Invalid IDs, bools and conflicting
    requests raise ValueError before execution or file creation.
    """
    if (
        isinstance(save_replays, bool)
        or not isinstance(save_replays, Integral)
        or not 0 <= save_replays <= len(episode_ids)
    ):
        raise ValueError(
            "save_replays must be an integer within the scheduled game count"
        )
    explicit = tuple(replay_episodes)
    if any(
        isinstance(i, bool) or not isinstance(i, Integral) or i not in episode_ids
        for i in explicit
    ):
        raise ValueError("replay_episodes must identify scheduled episodes")
    shorthand = set(episode_ids[: int(save_replays)])
    if shorthand and explicit and shorthand != set(explicit):
        raise ValueError("save_replays conflicts with replay_episodes")
    selected = shorthand or set(explicit)
    return tuple(i for i in episode_ids if i in selected)


def default_maps(phase: str) -> tuple[int, ...]:
    """Select new-pass maps only for the two supported automatic phase labels."""
    if phase == "validation":
        return tuple(row.map_id for row in list_tdm_maps() if row.split == "validation")
    if phase == "evaluation":
        return CANONICAL_TDM_EVALUATION_MAP_IDS
    raise ValueError(
        'Automatic maps require phase="validation" or phase="evaluation". '
        f"Supply maps explicitly for custom phase {phase!r}."
    )


def roster_default(
    value: Sequence[AgentClassName] | None, team: int
) -> tuple[AgentClassName, ...]:
    """Return supplied ordered classes or this team's canonical setup roster."""
    return tuple(value) if value is not None else canonical_tournament_rosters()[team]


def prepare_schedule(
    specs: Sequence[EpisodeSpec],
    *,
    saved_declarations: Mapping[str, Any] | None = None,
) -> tuple[dict[int, dict[str, object]], dict[str, dict[str, object]], dict[int, str]]:
    """Verify exact source/pair declarations and build shared recording evidence.

    Return ordered declarations, content-addressed source/resolved configurations,
    and resolved IDs keyed by the live configuration object's identity for this
    setup call only. Object IDs never become durable evidence. A single authored
    episode is valid. A comparison needs two exact games and explicit equal seeds;
    only unambiguous source-bank pairs without authored starts earn verified credit.
    Invalid declarations raise ValueError before writer mutation.
    """
    records: dict[str, dict[str, object]] = {}
    cached: dict[int, str] = {}
    comparisons: dict[str, list[int]] = {}
    declarations: dict[int, dict[str, object]] = {}
    relationships: dict[tuple[int, int], tuple[bool, int]] = {}
    map_keys: dict[tuple[int, str], list[dict[str, Any]]] = {}

    def remember(config: EnvConfig) -> str:
        """Hash each live immutable setup config once and retain its content."""
        key = id(config)
        if key not in cached:
            identifier, content = config_record(config)
            cached[key] = identifier
            records[identifier] = content
        return cached[key]

    for spec in specs:
        resolved = remember(spec.env_config)
        source_id = None
        reported = None
        if spec.source_config is not None:
            source_id = remember(spec.source_config)
            pair = (id(spec.env_config), id(spec.source_config))
            if pair not in relationships:
                matched, selected = jax.device_get(
                    spawn_locations_for_source(spec.env_config, spec.source_config)
                )
                relationships[pair] = bool(matched), int(selected)
            matches, choice = relationships[pair]
            if not bool(matches):
                raise ValueError(
                    "resolved configuration differs from its declared source"
                )
            reported = int(choice) if int(choice) >= 0 else None
        if spec.spawn_locations is not None:
            if (
                isinstance(spec.spawn_locations, bool)
                or not isinstance(spec.spawn_locations, Integral)
                or spec.spawn_locations not in (0, 1)
            ):
                raise ValueError("spawn_locations must be 0, 1 or None")
            if source_id is None or reported != spec.spawn_locations:
                raise ValueError(
                    "spawn_locations requires matching unambiguous source evidence"
                )
        key = spec.paired_comparison_key
        if key is not None:
            if not isinstance(cast(object, key), str) or not key.strip():
                raise ValueError("paired_comparison_key must be a nonempty string")
            comparisons.setdefault(key, []).append(spec.episode_id)
        declared: dict[str, object] = {
            "episode_id": spec.episode_id,
            "seed_id": spec.random_seed_id,
            "map_id": spec.map_id,
            "configuration_digest": resolved,
            "source_config_id": source_id,
            "spawn_locations": reported,
            "paired_comparison_key": key,
            "comparison_kind": "unpaired",
            "initial_state_digest": None,
            "expected_horizon": int(spec.env_config.max_steps)
            - (
                int(spec.initial_state.step_count)
                if spec.initial_state is not None
                else 0
            ),
        }
        previous = (
            None
            if saved_declarations is None
            else saved_declarations.get(str(spec.episode_id))
        )
        if previous is not None and previous.get("map_metadata") is not None:
            if (
                previous.get("configuration_digest") != resolved
                or previous.get("source_config_id") != source_id
                or previous.get("map_id") != spec.map_id
            ):
                raise ValueError(
                    "explicit conditions differ from the saved map evidence"
                )
            declared["map_metadata"] = previous["map_metadata"]
        elif (
            spec.map_id is not None and (spec.map_id, source_id or resolved) in map_keys
        ):
            declared["map_metadata"] = map_keys[spec.map_id, source_id or resolved]
        elif spec.map_id is not None:
            from marl_battlegrounds.evaluation.catalog import (
                build_resolved_env_config_v1,
            )
            from marl_battlegrounds.evaluation.map_identity import (
                registered_map_metadata,
            )

            actual = build_resolved_env_config_v1(spec.env_config)
            source_pads = (
                spec.env_config.team_spawn_pad_positions
                if spec.source_config is None
                else spec.source_config.team_spawn_pad_positions
            )
            source_geometry = actual.model_copy(
                update={
                    "team_spawn_pad_positions": tuple(
                        tuple(tuple(float(x) for x in pad) for pad in bank)
                        for bank in np.asarray(source_pads)
                    )
                }
            )
            declared["map_metadata"] = [
                row.model_dump(mode="json")
                for row in registered_map_metadata(spec.map_id, source_geometry)  # pyright: ignore[reportArgumentType]
            ]
            map_keys[spec.map_id, source_id or resolved] = declared["map_metadata"]
        if spec.initial_state is not None:
            from marl_battlegrounds.evaluation.recording_identity import tree_digest

            declared["initial_state_digest"] = tree_digest(spec.initial_state)
        declarations[spec.episode_id] = declared
    by_id = {spec.episode_id: spec for spec in specs}
    for key, identifiers in comparisons.items():
        if len(identifiers) != 2:
            raise ValueError(f"comparison {key!r} requires exactly two episodes")
        first, second = (by_id[i] for i in identifiers)
        if (
            first.seed_id is None
            or second.seed_id is None
            or first.seed_id != second.seed_id
            or first.map_id != second.map_id
        ):
            raise ValueError("comparisons require equal explicit seed IDs and map IDs")
        a, b = (declarations[i] for i in identifiers)
        verified = (
            a["source_config_id"] is not None
            and a["source_config_id"] == b["source_config_id"]
            and {a["spawn_locations"], b["spawn_locations"]} == {0, 1}
            and first.initial_state is None
            and second.initial_state is None
        )
        a["comparison_kind"] = b["comparison_kind"] = (
            "verified_spawn_pair" if verified else "custom"
        )
    for spec in specs:
        declared = declarations[spec.episode_id]
        if spec.metadata:
            for name, value in spec.metadata.items():
                if name in declared and declared[name] != value:
                    raise ValueError(f"metadata conflicts with EpisodeSpec.{name}")
            declarations[spec.episode_id] = {**spec.metadata, **declared}
    return declarations, records, cached


def schedule_digest(schedule: Mapping[int, Mapping[str, object]]) -> str:
    """Hash exact declarations in their scheduled order using shared JSON rules."""
    from marl_battlegrounds.evaluation.run_writer import _json_bytes, _json_value

    return sha256(_json_bytes(_json_value(list(schedule.values())))).hexdigest()


def prepare_source_choices(
    sources: Sequence[EpisodeSpec],
    schedule: Mapping[int, Mapping[str, object]],
    records: dict[str, dict[str, object]],
    cached: dict[int, str],
) -> list[dict[str, object]]:
    """Retain every requested map source, including repeats and unplayed choices.

    sources is the ordered normalized source list for a generated evaluation.
    schedule contains the already prepared played conditions. Reuse its content
    IDs and map evidence; extend the shared records/cache only for unused sources.
    Return ordered map/source references and exact map metadata. This host setup
    helper changes only the caller's temporary dictionaries and opens no files.
    Invalid registered-map evidence raises ValueError before writer construction.
    """
    from marl_battlegrounds.evaluation.catalog import build_resolved_env_config_v1
    from marl_battlegrounds.evaluation.map_identity import registered_map_metadata

    evidence = {
        (row["map_id"], row["source_config_id"]): row.get("map_metadata", [])
        for row in schedule.values()
    }
    choices: list[dict[str, object]] = []
    for source in sources:
        key = id(source.env_config)
        if key not in cached:
            identifier, content = config_record(source.env_config)
            records[identifier] = content
            cached[key] = identifier
        identifier = cached[key]
        pair = source.map_id, identifier
        if pair not in evidence:
            evidence[pair] = (
                []
                if source.map_id is None
                else [
                    row.model_dump(mode="json")
                    for row in registered_map_metadata(
                        source.map_id,
                        build_resolved_env_config_v1(source.env_config),  # pyright: ignore[reportArgumentType]
                    )
                ]
            )
        choices.append(
            {
                "map_id": source.map_id,
                "source_config_id": identifier,
                "map_metadata": evidence[pair],
            }
        )
    return choices


def saved_specs(
    saved: tuple[dict[str, Any], dict[str, Any]],
) -> tuple[EpisodeSpec, ...]:
    """Restore a saved generated schedule from its own config content.

    Authored starts cannot be reconstructed from a digest and must instead be
    supplied through evaluate_episodes. Missing or inconsistent evidence fails;
    no current catalog defaults are used as substitutes.
    """
    from marl_battlegrounds.evaluation.evaluate import EpisodeSpec

    manifest, entry = saved
    details = entry["details"]
    configs = {
        **manifest.get("configurations", {}),
        **details.get("configurations", {}),
    }
    contract = details.get("evaluation_contract", {})
    ids = contract.get("episode_ids", details.get("episode_ids"))
    declarations = entry.get("episodes", {})
    if not ids or len(ids) != len(declarations):
        raise ValueError("saved pass lacks its complete ordered schedule")
    cache: dict[str, EnvConfig] = {}

    def resolve(identifier: object, *, source_only: bool = False) -> EnvConfig:
        """Check saved content identity before constructing its numerical config."""
        if not isinstance(identifier, str) or identifier not in configs:
            raise ValueError("saved pass lacks required configuration content")
        if identifier not in cache:
            config = restore_config(configs[identifier], validate=not source_only)
            if config_record(config)[0] != identifier:
                raise ValueError(
                    "saved configuration content differs from its identity"
                )
            cache[identifier] = config
        return cache[identifier]

    output: list[EpisodeSpec] = []
    for identifier in ids:
        row = declarations.get(str(identifier))
        if row is None or row.get("initial_state_digest") is not None:
            raise ValueError("resupply exact authored starts through evaluate_episodes")
        source = row.get("source_config_id")
        reserved = {
            "episode_id",
            "seed_id",
            "map_id",
            "configuration_digest",
            "source_config_id",
            "spawn_locations",
            "paired_comparison_key",
            "comparison_kind",
            "initial_state_digest",
            "expected_horizon",
        }
        output.append(
            EpisodeSpec(
                episode_id=identifier,
                env_config=resolve(
                    row.get("configuration_digest", row.get("config_id"))
                ),
                map_id=row.get("map_id"),
                seed_id=row.get("seed_id"),
                source_config=resolve(source, source_only=True)
                if source is not None
                else None,
                spawn_locations=row.get("spawn_locations"),
                paired_comparison_key=row.get("paired_comparison_key")
                if contract
                else None,
                metadata={k: v for k, v in row.items() if k not in reserved},
            )
        )
    return tuple(output)
