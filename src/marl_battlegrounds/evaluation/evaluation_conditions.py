"""Resolve evaluation conditions before any writer or episode is changed.

The evaluator uses these host helpers for new schedules and saved-first resume.
Saved-first options (option, and red_zone_depth_option for the Red Zone depth),
exact configurations, source banks, pair claims and capture selections have one
owner here, and so does the refusal to resume a pass saved before the Red Zone
rule (refuse_pass_saved_before_red_zone). No helper runs a method, advances a
game, opens a writer or changes a saved file. Numerical execution remains in the
existing evaluator.
"""

from __future__ import annotations

# These host helpers share private serialization and setup authorities.
# pyright: reportPrivateUsage=false
import json
from array import array
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
from marl_battlegrounds.evaluation.run_writer import (
    configuration_content_identity,
    configuration_identity,
)
from marl_battlegrounds.tasks import (
    CANONICAL_TDM_EVALUATION_MAP_IDS,
    DEFAULT_TDM_RED_ZONE_DEPTH,
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
# The EnvConfig field added with Red Zone scoring. Content saved before it has
# the other twelve fields and restores at depth 0.0.
_RED_ZONE_DEPTH_FIELD = "team_deathmatch_red_zone_depth"
# The one refusal for resuming an evaluation pass saved before the rule.
_SAVED_BEFORE_RED_ZONE = (
    "This pass was saved before the Red Zone rule. Its results stay readable; "
    "resuming it needs the source version that recorded it."
)


def option(
    value: Any,  # noqa: ANN401
    saved: Mapping[str, object] | None,
    name: str,
    default: Any,  # noqa: ANN401
    *,
    missing: Any = OMITTED,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """Resolve one omitted option, or check an explicit value against saved facts.

    Parameters
    ----------
    value : Any
        The caller's value, or OMITTED when the caller did not pass it.
    saved : Mapping[str, object] or None
        The scientific options of the selected saved pass, or None for a new
        pass.
    name : str
        The option's key in saved and in the error message.
    default : Any
        The value a new pass uses when value is omitted.
    missing : Any, default=OMITTED
        The value a saved pass had when it was saved before this option
        existed (saved has no name key). OMITTED keeps the older rule: such a
        pass reads as default and accepts any explicit value. For example,
        red_zone_depth uses missing=0.0, because passes saved before the Red
        Zone rule scored one point per death.

    Returns
    -------
    Any
        Omitted value: default for a new pass, otherwise the saved value (or
        missing, or default, when saved lacks name). Explicit value: returned
        unchanged, without copying arrays.

    Raises
    ------
    ValueError
        An explicit value differs from the saved value (or from missing when
        saved lacks name): "<name> differs from the saved evaluation
        conditions". Explicit values, including zero and empty selections,
        must agree. Sequence containers compare by their values.

    Notes
    -----
    Host-only. Configuration identities are checked separately.
    """
    recorded: Any = OMITTED if saved is None else saved.get(name, missing)
    if isinstance(value, Omitted):
        return default if isinstance(recorded, Omitted) else recorded
    if not isinstance(recorded, Omitted):
        left = (
            list(cast(Sequence[object], value))
            if isinstance(value, (tuple, list))
            else value
        )
        right = (
            list(cast(Sequence[object], recorded))
            if isinstance(recorded, (tuple, list))
            else recorded
        )
        if left != right:
            raise ValueError(f"{name} differs from the saved evaluation conditions")
    return cast(Any, value)


def red_zone_depth_option(
    value: float | Omitted, saved: Mapping[str, object] | None
) -> float:
    """Resolve the red_zone_depth option of evaluate or run_tournament.

    Parameters
    ----------
    value : float or Omitted
        The caller's Red Zone depth in map units, or OMITTED when the caller
        did not pass it. A supplied value must be exactly a Python float, as
        Core requires; Core checks its range when a map config is built.
    saved : Mapping[str, object] or None
        The selected saved pass's scientific options (they hold
        "red_zone_depth" when saved with the rule), or None for a new pass.

    Returns
    -------
    float
        Omitted: DEFAULT_TDM_RED_ZONE_DEPTH (5.0) for a new pass, the saved
        depth on resume, or 0.0 when saved has no "red_zone_depth" (a
        completed tournament run saved before the Red Zone rule; evaluate
        refuses to resume such a pass before it gets here). Supplied: the
        value itself for a new pass; on resume, the saved depth, which has
        the same float32 value (the value configs store and hash), so the
        rebuilt configs are identical.

    Raises
    ------
    TypeError
        A supplied value is not a Python float (bool, int and NumPy scalars
        included), for example red_zone_depth=6.
    ValueError
        A supplied value has a different float32 value from the saved depth
        (0.0 when saved has no "red_zone_depth"): "red_zone_depth differs
        from the saved evaluation conditions". The float32 bytes are
        compared, so -0.0 differs from a saved 0.0.

    Notes
    -----
    Host-only; option owns the saved-pass rule for a saved depth that is not
    a Python float. A supplied value is never ignored: evaluate also checks
    it against every exact source config.
    """
    if not isinstance(value, Omitted):
        if type(value) is not float:
            raise TypeError(
                f"red_zone_depth must be a Python float, not {type(value).__name__}"
            )
        if saved is not None:
            recorded = saved.get("red_zone_depth", 0.0)
            if type(recorded) is float:
                # Python's != treats -0.0 and 0.0 as equal; configs do not.
                if not same_float32(value, recorded):
                    raise ValueError(
                        "red_zone_depth differs from the saved evaluation conditions"
                    )
                value = recorded
    return cast(
        float,
        option(value, saved, "red_zone_depth", DEFAULT_TDM_RED_ZONE_DEPTH, missing=0.0),
    )


def same_float32(left: float, right: float) -> bool:
    """Tell whether two Python floats have the same float32 value and sign.

    Parameters
    ----------
    left, right : float
        The two values to compare, for example a caller's Red Zone depth and a
        saved one, both in map units.

    Returns
    -------
    bool
        True when both values give the same float32 bytes. Configs store and
        hash float rule values as float32, so two depths that pass this check
        give identical configs; for example 5.0000001 matches 5.0.

    Notes
    -----
    The float32 bytes are compared, so -0.0 and 0.0 differ, and values beyond
    the float32 range become infinity of their sign. Host-only; no NumPy or JAX
    work.
    """
    return array("f", [left]).tobytes() == array("f", [right]).tobytes()


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

    Parameters
    ----------
    content : Mapping[str, Any]
        Saved configuration content in the existing configuration-identity
        JSON layout. Its key set must equal EnvConfig's fields exactly (13
        keys), or equal them minus team_deathmatch_red_zone_depth (12 keys).
        The 12-key form is content saved before the Red Zone rule; it is
        restored at depth 0.0, its original one-point scoring. Its
        agent_profile must be a mapping with exactly ResolvedAgentProfile's
        fields.
    validate : bool, default=True
        True also checks physical validity with validate_env_config.
        Source-only evidence may use False when its unused banks need not be
        valid for the actual roster; the requested resolved config must still
        validate.

    Returns
    -------
    EnvConfig
        A new config. Array fields become JAX arrays with the established Core
        config dtypes: active_mask is bool; team_ids, class_ids,
        out_of_combat_delay_steps and team_respawn_wave_period_step_count are
        int32; the other roster arrays, obstacles and team_spawn_pad_positions
        are float32. Scalar rules stay Python values. content is not changed.

    Raises
    ------
    ValueError
        Any other missing or extra field: "saved configuration fields do not
        match EnvConfig", or "saved roster fields do not match
        ResolvedAgentProfile" for the roster. Also, whatever validate is, an
        array field that cannot be converted, such as ragged rows, text or
        None. With validate=True, also any failed Core check: shape, finite
        value, bound, mode, padding, catalog agreement or geometry.
    TypeError
        An array field holds a value that cannot be read as a number, such as
        a JSON object, whatever validate is. With validate=True, also a scalar
        rule of the wrong Python type, for example a float max_steps.

    Notes
    -----
    Host-only. This helper creates numerical arrays on JAX's default device but
    never changes their values or exchanges their banks. It does not check the
    content against its recorded identity; to do that too, use
    restore_recorded_config.
    """
    fields = set(EnvConfig._fields)
    if set(content) == fields - {_RED_ZONE_DEPTH_FIELD}:
        content = {**content, _RED_ZONE_DEPTH_FIELD: 0.0}
    elif set(content) != fields:
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


def config_record(
    config: EnvConfig, *, historical: bool = False
) -> tuple[str, dict[str, object]]:
    """Identify a scalar config with the same numerical normalization as reset.

    Parameters
    ----------
    config : EnvConfig
        One scalar configuration.
    historical : bool, default=False
        True returns the identity the config had before the Red Zone field
        existed (content without team_deathmatch_red_zone_depth). Only a
        config whose depth is exactly +0.0 has one; others raise ValueError.
        Use it only to match identities recorded before Red Zone.

    Returns
    -------
    tuple[str, dict[str, object]]
        (sha256_hex, content), as run_writer.configuration_identity returns.

    Raises
    ------
    ValueError
        historical is True and the depth is not exactly +0.0, or a leaf is NaN
        or infinite.
    TypeError
        A leaf cannot be written as JSON.
    """

    def numerical(value: object) -> jax.Array:
        """Use the reset boundary's array dtype inference for one config leaf."""
        return jnp.asarray(value)

    return configuration_identity(
        jax.tree.map(numerical, config), historical=historical
    )


def restore_recorded_config(
    content: Mapping[str, Any], identifier: str, *, validate: bool = True
) -> tuple[EnvConfig, bool]:
    """Check saved config content against its recorded identity, then restore it.

    Parameters
    ----------
    content : Mapping[str, Any]
        Saved configuration content: current (13 keys) or recorded before the
        Red Zone rule (12 keys).
    identifier : str
        The SHA-256 identity recorded beside the content.
    validate : bool, default=True
        Passed to restore_config.

    Returns
    -------
    tuple[EnvConfig, bool]
        The restored config and whether the content was recorded before the
        Red Zone rule (historical). A historical config has depth 0.0 and its
        identity is config_record(config, historical=True).

    Raises
    ------
    ValueError
        The content's hash differs from identifier (checked before any array
        is built), restore_config rejects it, or the restored config does not
        reproduce identifier: "configuration content differs from its recorded
        identity".

    Notes
    -----
    Host-only. This is the one owner of check-then-restore for content that
    may predate Red Zone: canonical sources, saved tournament maps, reused
    tournament evidence and saved evaluation passes.
    """
    if configuration_content_identity(content) != identifier:
        raise ValueError("configuration content differs from its recorded identity")
    historical = _RED_ZONE_DEPTH_FIELD not in content
    config = restore_config(content, validate=validate)
    if config_record(config, historical=historical)[0] != identifier:
        raise ValueError("configuration content differs from its recorded identity")
    return config, historical


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
    registered_maps: Mapping[int, Mapping[str, Any]] | None = None,
) -> tuple[dict[int, dict[str, object]], dict[str, dict[str, object]], dict[int, str]]:
    """Verify exact source/pair declarations and build shared recording evidence.

    Parameters
    ----------
    specs : Sequence[EpisodeSpec]
        The scheduled games, in order. A single authored episode is valid.
        Games that share a paired_comparison_key form one comparison.
    saved_declarations : Mapping[str, Any] or None, default=None
        The selected saved pass's game declarations, keyed by episode ID as a
        string, or None for a new pass. A saved declaration that holds map
        metadata keeps its earlier map identity; the game must then have the
        same configuration digest, source config ID and map ID.
    registered_maps : Mapping[int, Mapping[str, Any]] or None, default=None
        Private option: exact snapshot TDMMapInfo entries in JSON form, keyed
        by map ID. These still need approved current or history identities
        and matching source geometry. None, or a map ID that is not a key,
        uses today's catalog.

    Returns
    -------
    declarations : dict[int, dict[str, object]]
        One declaration per game, keyed by episode ID in schedule order. Each
        holds episode_id; seed_id (the explicit seed, or the episode ID when
        none was given); map_id; configuration_digest; source_config_id;
        spawn_locations (0 for the source's bank order, 1 for the exchanged
        order, None with no source or identical banks); paired_comparison_key;
        comparison_kind ("unpaired", "custom" or "verified_spawn_pair");
        initial_state_digest (None without an authored start); and
        expected_horizon (max_steps minus the start's step count). A game with
        a map ID also holds map_metadata. The game's own metadata is merged in.
    records : dict[str, dict[str, object]]
        Content-addressed configurations: SHA-256 identity to current content,
        for every resolved and source config.
    cached : dict[int, str]
        Resolved IDs keyed by each live configuration object's identity
        (``id()``), valid for this setup call only. Object IDs never become
        durable evidence.

    Raises
    ------
    ValueError
        Before any writer is changed, when: a resolved config does not match
        its declared source; spawn_locations is not 0, 1 or None, or lacks
        matching unambiguous source evidence; a paired_comparison_key is not a
        nonempty string; a comparison does not have exactly two games, or its
        games lack equal explicit seed IDs and equal map IDs; a game differs
        from its saved map evidence; map evidence is invalid (for example a
        snapshot entry whose identity does not match its approved geometry);
        or a game's metadata conflicts with one of its EpisodeSpec fields.

    Notes
    -----
    A comparison needs two exact games and explicit equal seeds. Only
    unambiguous source-bank pairs without authored starts earn verified credit
    ("verified_spawn_pair"); other comparisons are "custom". Map evidence
    compares map geometry only; the V2 resolved config records any Red Zone
    depth, so every depth passes that check. Host-only: each distinct live
    config is hashed once, and source checks read device values to the host.
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
                build_resolved_env_config_v2,
            )
            from marl_battlegrounds.evaluation.map_identity import (
                _snapshot_map_metadata,
                registered_map_metadata,
            )

            # V2 records any Red Zone depth; only the map geometry is compared.
            actual = build_resolved_env_config_v2(spec.env_config)
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
            map_rows = (
                registered_map_metadata(spec.map_id, source_geometry)  # pyright: ignore[reportArgumentType]
                if registered_maps is None or spec.map_id not in registered_maps
                else _snapshot_map_metadata(
                    spec.map_id, source_geometry, registered_maps[spec.map_id]
                )
            )
            declared["map_metadata"] = [row.model_dump(mode="json") for row in map_rows]
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

    Parameters
    ----------
    sources : Sequence[EpisodeSpec]
        The ordered normalized source list for a generated evaluation: one
        entry per requested map or configuration, repeats included.
    schedule : Mapping[int, Mapping[str, object]]
        The already prepared played conditions (the declarations from
        prepare_schedule). Its content IDs and map evidence are reused.
    records : dict[str, dict[str, object]]
        The shared content-addressed configurations from prepare_schedule.
        Changed in place: content is added only for a source config that was
        not already hashed (a source no played game used).
    cached : dict[int, str]
        The shared identities keyed by live config object ``id()`` from
        prepare_schedule. Changed in place the same way.

    Returns
    -------
    list[dict[str, object]]
        One reference per source, in source order, with map_id,
        source_config_id and the exact map_metadata rows (an empty list when
        map_id is None).

    Raises
    ------
    ValueError
        Invalid registered-map evidence for a source whose map evidence is not
        already in schedule. It is raised before the writer is built.

    Notes
    -----
    This host setup helper changes only the caller's temporary dictionaries
    and opens no files. Map evidence uses the V2 resolved config, which records
    any Red Zone depth.
    """
    from marl_battlegrounds.evaluation.catalog import build_resolved_env_config_v2
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
                        build_resolved_env_config_v2(source.env_config),  # pyright: ignore[reportArgumentType]
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


def refuse_pass_saved_before_red_zone(
    saved: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    """Refuse to resume an evaluation pass saved before the Red Zone rule.

    Parameters
    ----------
    saved : tuple[dict[str, Any], dict[str, Any]]
        (manifest, pass) exactly as read_saved_pass returns them.

    Raises
    ------
    ValueError
        "This pass was saved before the Red Zone rule. Its results stay
        readable; resuming it needs the source version that recorded it."
        A pass counts as saved before the rule when its generated evaluation
        contract has options without "red_zone_depth", or when any
        configuration content one of its games references (resolved config
        or source config) has no team_deathmatch_red_zone_depth.

    Notes
    -----
    Host-only: it reads the two dictionaries and changes nothing. evaluate
    calls it first on resume, before options, maps or sources are resolved,
    so the same message appears whether or not maps and red_zone_depth are
    supplied. Only content this pass references is checked, so an older pass
    in the same run does not affect a newer one. Missing content is left to
    saved_specs and the schedule checks, which fail on it.
    """
    manifest, entry = saved
    details = entry["details"]
    contract = details.get("evaluation_contract")
    if isinstance(contract, dict) and contract.get("schedule_kind") == "generated":
        options = cast(dict[str, Any], contract).get("options")
        if isinstance(options, dict) and "red_zone_depth" not in options:
            raise ValueError(_SAVED_BEFORE_RED_ZONE)
    configs = {
        **manifest.get("configurations", {}),
        **details.get("configurations", {}),
    }
    for row in entry.get("episodes", {}).values():
        for key in ("configuration_digest", "config_id", "source_config_id"):
            identifier = row.get(key)
            content = configs.get(identifier) if isinstance(identifier, str) else None
            if isinstance(content, dict) and _RED_ZONE_DEPTH_FIELD not in content:
                raise ValueError(_SAVED_BEFORE_RED_ZONE)


def saved_specs(
    saved: tuple[dict[str, Any], dict[str, Any]],
) -> tuple[EpisodeSpec, ...]:
    """Restore a saved generated schedule from its own config content.

    Parameters
    ----------
    saved : tuple[dict[str, Any], dict[str, Any]]
        (manifest, pass) exactly as read_saved_pass returns them. Config
        content comes from the manifest's and the pass's "configurations"; the
        pass's copy wins for the same identity.

    Returns
    -------
    tuple[EpisodeSpec, ...]
        One spec per saved game, in the saved episode order. Each has the
        restored resolved config (physically validated), its map ID, seed ID,
        spawn_locations and source config (restored without the physical check
        when present), and the declaration's other fields as metadata.
        paired_comparison_key is kept only when the pass has an evaluation
        contract. Games with the same config identity share one restored
        config object.

    Raises
    ------
    ValueError
        "saved pass lacks its complete ordered schedule" (no episode IDs, or
        their count differs from the declarations); "resupply exact authored
        starts through evaluate_episodes" (a missing declaration or an
        authored start); "saved pass lacks required configuration content";
        "saved configuration content differs from its identity" (the content
        fails its identity check, or restore_config raises ValueError for it,
        including a failed physical check); or the refusal for a pass saved
        before the Red Zone rule (below).
    TypeError
        Not wrapped in the ValueError above: a resolved config with a scalar
        rule of the wrong Python type (for example a float max_steps), or any
        saved config with an array value that cannot be read as a number, such
        as a JSON object. Source configs skip the physical check, so only the
        second case applies to them.

    Notes
    -----
    Authored starts cannot be reconstructed from a digest and must instead be
    supplied through evaluate_episodes. Missing or inconsistent evidence fails;
    no current catalog defaults are used as substitutes. A pass saved before the
    Red Zone rule (configuration content without team_deathmatch_red_zone_depth)
    is refused with the same ValueError as refuse_pass_saved_before_red_zone:
    its results stay readable, but resuming it needs the source version that
    recorded it. Host-only: it reads the two dictionaries, changes neither and
    opens no files; restored arrays live on JAX's default device.
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
            try:
                config, historical = restore_recorded_config(
                    configs[identifier], identifier, validate=not source_only
                )
            except ValueError as error:
                raise ValueError(
                    "saved configuration content differs from its identity"
                ) from error
            if historical:
                raise ValueError(_SAVED_BEFORE_RED_ZONE)
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
