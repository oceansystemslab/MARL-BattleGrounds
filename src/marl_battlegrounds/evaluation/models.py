"""Define strict host records for captured episodes, observations, facts, and events.

Records use immutable Python tuples with frozen versioned schemas. Ten global
slots are Team A 0-4 then Team B 5-9; policy relation rows remain actor-relative.
Context V3 and frame V2 describe current relative actor inputs. Older versions
remain readable with their original meanings.

Models check their local fields and joins. Full cross-record admission lives in
validation and replay readers; constructing one model does not rerun physics.
Snapshot data is privileged analysis truth, not extra actor information. This
module intentionally avoids importing JAX or live Core so artifact readers stay
usable without initializing a device. Docstrings may appear in generated schemas;
canonical content digests cover model values, not their schema descriptions.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from math import hypot, isfinite
from typing import Annotated, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

import marl_battlegrounds.evaluation.wire_shapes as _wire_shapes

CONTEXT_FEATURES = _wire_shapes.CONTEXT_FEATURES_V1
ENVIRONMENT_DIMENSIONS = _wire_shapes.ENVIRONMENT_DIMENSIONS_V1
MAX_AGENT_SLOTS = _wire_shapes.MAX_AGENT_SLOTS_V1
MAX_AGENTS_PER_TEAM = _wire_shapes.MAX_AGENTS_PER_TEAM_V1
MAX_OBJECTIVE_SLOTS = _wire_shapes.MAX_OBJECTIVE_SLOTS_V1
MAX_OBSTACLE_SLOTS = _wire_shapes.MAX_OBSTACLE_SLOTS_V1
NUM_CLASSES = _wire_shapes.NUM_CLASSES_V1
NUM_MOVE_ACTIONS = _wire_shapes.NUM_MOVE_ACTIONS_V1
NUM_SLOW_CHANNELS = _wire_shapes.NUM_SLOW_CHANNELS_V1
NUM_STUN_CHANNELS = _wire_shapes.NUM_STUN_CHANNELS_V1
NUM_TARGET_ACTIONS = _wire_shapes.NUM_TARGET_ACTIONS_V1
NUM_TEAMS = _wire_shapes.NUM_TEAMS_V1
NUM_ULTIMATE_ACTIONS = _wire_shapes.NUM_ULTIMATE_ACTIONS_V1
OBJECTIVE_FEATURES = _wire_shapes.OBJECTIVE_FEATURES_V1
OBSTACLE_FEATURES = _wire_shapes.OBSTACLE_FEATURES_V1
SELF_FEATURES = _wire_shapes.SELF_FEATURES_V1
UNIT_FEATURES = _wire_shapes.UNIT_FEATURES_V1

CATALOG_SCHEMA_ID = "marl_battlegrounds.evaluation.static_mechanics_catalog"
CATALOG_SCHEMA_VERSION = 1
RESOLVED_ENV_CONFIG_SCHEMA_ID = "marl_battlegrounds.evaluation.resolved_env_config"
RESOLVED_ENV_CONFIG_SCHEMA_VERSION = 1
CONTEXT_SCHEMA_ID = "marl_battlegrounds.evaluation.episode_context"
CONTEXT_SCHEMA_VERSION = 1
GLOBAL_ANALYSIS_SNAPSHOT_SCHEMA_ID = (
    "marl_battlegrounds.evaluation.global_analysis_snapshot"
)
GLOBAL_ANALYSIS_SNAPSHOT_SCHEMA_VERSION = 1
FRAME_SCHEMA_ID = "marl_battlegrounds.evaluation.frame"
FRAME_SCHEMA_VERSION = 1
TRANSITION_FACTS_SCHEMA_ID = "marl_battlegrounds.evaluation.transition_facts"
TRANSITION_FACTS_SCHEMA_VERSION = 1
EVENT_SCHEMA_ID = "marl_battlegrounds.evaluation.event"
EVENT_SCHEMA_VERSION = 1
TRANSITION_SCHEMA_ID = "marl_battlegrounds.evaluation.transition"
TRANSITION_SCHEMA_VERSION = 1

type ExecutionInformationMode = Literal["shared_obs", "no_shared_obs"]
type CaptureProfile = Literal[
    "training_light",
    "evaluation_metric_complete",
    "scenario_metric_complete",
    "debug",
]
type EvaluationRole = Literal[
    "focal",
    "cooperative_partner",
    "adversarial_opponent",
]

_AsciiText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        pattern=r"^[\x20-\x7e]+$",
    ),
]
_AsciiIdentifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/+\-]*$",
    ),
]
_Sha256Hex = Annotated[
    str,
    StringConstraints(pattern=r"^[0-9a-f]{64}$"),
]
_GitCommitHex = Annotated[
    str,
    StringConstraints(pattern=r"^[0-9a-f]{40}$"),
]
_NonNegativeInt = Annotated[int, Field(ge=0)]
_PositiveInt = Annotated[int, Field(gt=0)]
_Seed = Annotated[int, Field(ge=0, le=2**32 - 1)]
_FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
_NonNegativeFloat = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
_PositiveFloat = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
_UnitFloat = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
_GlobalSlot = Annotated[int, Field(ge=0, lt=MAX_AGENT_SLOTS)]
_TeamLocalSlot = Annotated[int, Field(ge=0, lt=MAX_AGENTS_PER_TEAM)]
_Int32 = Annotated[int, Field(ge=-(2**31), le=2**31 - 1)]
_StatusChannel = Annotated[int, Field(ge=0, lt=9)]


class EvaluationModel(BaseModel):
    """Provide strict, frozen Pydantic behavior for evaluation records.

    Notes
    -----
    Extra fields and nonfinite floats are forbidden; values are not freely coerced
    across types. Normal attribute assignment is frozen. Concrete models use
    tuple-backed fields for immutable data. Unchecked model_construct/model_copy
    can bypass validation, so artifact admission revalidates exact trees.
    """

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
    )


def _normalize_canonical_json(value: object) -> object:
    """Normalize nested JSON values while preserving numeric and container meaning.

    Change negative float zero to positive zero, convert models to JSON-mode fields,
    and tuples/lists to JSON arrays. Traverse mappings without sorting here.
    Unknown objects pass through for the JSON encoder to reject if unsupported.
    """
    # Replay arrays contain millions of primitive leaves. Avoid model/ABC
    # instance checks for each one while retaining the exact-float zero rule.
    value_type = type(value)
    if value_type is float:
        return 0.0 if value == 0.0 else value
    if value is None or value_type in (bool, int, str):
        return value
    if isinstance(value, BaseModel):
        return _normalize_canonical_json(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, object], value)
        return {key: _normalize_canonical_json(item) for key, item in mapping.items()}
    if isinstance(value, (list, tuple)):
        sequence = cast(list[object] | tuple[object, ...], value)
        return [_normalize_canonical_json(item) for item in sequence]
    return value


def canonical_json_bytes(value: BaseModel | Mapping[str, object]) -> bytes:
    """Encode finite content using the project's stable JSON profile.

    Parameters
    ----------
    value : BaseModel | Mapping[str, object]
        Pydantic model or string-keyed mapping of JSON-compatible values.

    Returns
    -------
    bytes
        UTF-8 bytes with sorted object keys, compact separators, no ASCII escaping,
        and normalized floating zero signs.

    Raises
    ------
    TypeError
        A value or mapping key cannot be encoded by JSON.
    ValueError
        Content contains NaN, infinity, or a circular container.

    Notes
    -----
    Models are dumped in JSON mode. This helper serializes supplied content;
    it does not validate arbitrary mappings or write files.
    """
    payload: object = (
        value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    )
    return json.dumps(
        _normalize_canonical_json(payload),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_digest_sha256(
    value: BaseModel | Mapping[str, object],
    *,
    exclude: set[str] | None = None,
) -> str:
    """Hash a model or mapping through canonical JSON encoding.

    Parameters
    ----------
    value : BaseModel | Mapping[str, object]
        Model or string-keyed JSON-compatible mapping.
    exclude : set[str] | None
        Optional set of top-level fields omitted from the hash, commonly
        the digest field itself. Defaults to no exclusions.

    Returns
    -------
    str
        Lowercase 64-character SHA-256 hexadecimal digest.

    Raises
    ------
    TypeError
        A retained value cannot be encoded as JSON.
    ValueError
        A retained value is nonfinite or a container is circular.

    Notes
    -----
    Does not mutate input. This identifies serialized content, not a file path,
    Python class source, or proof that the supplied facts are scientifically valid.
    """
    excluded = exclude or set()
    if isinstance(value, BaseModel):
        payload = value.model_dump(mode="json", exclude=excluded)
    else:
        payload = {key: item for key, item in value.items() if key not in excluded}
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


class ClassMechanicsV1(EvaluationModel):
    """Record one class's mechanics in the episode's interpretation catalog.

    Attributes
    ----------
    class_id : Annotated[int, Field(ge=0, lt=NUM_CLASSES)]
        Class 0-5; zero is the neutral inactive class.
    class_name : _AsciiText
        Nonempty printable class name.
    maximum_health : _NonNegativeFloat
        Nonnegative maximum health in hit points.
    body_radius : _NonNegativeFloat
        Nonnegative body radius in world units.
    base_movement_speed : _NonNegativeFloat
        Nonnegative world-unit movement per tick before modifiers.
    observation_radius : _NonNegativeFloat
        Nonnegative sight radius in world units.
    basic_target_mode : Literal['unavailable', 'ally', 'enemy']
        unavailable, ally, or enemy.
    basic_interaction_radius : _NonNegativeFloat
        Nonnegative Basic range in world units.
    basic_raw_damage : _NonNegativeFloat
        Nonnegative Basic output in hit points before modifiers.
    basic_raw_healing : _NonNegativeFloat
        Nonnegative Basic healing before modifiers.
    ultimate_target_mode : Literal['unavailable', 'target_none', 'ally', 'enemy']
        unavailable, target_none, ally, or enemy.
    ultimate_interaction_radius : _NonNegativeFloat
        Nonnegative Ultimate range in world units.
    ultimate_cooldown_steps : _NonNegativeInt
        Nonnegative Ultimate cooldown ticks.
    ultimate_raw_damage : _NonNegativeFloat
        Nonnegative Ultimate damage before modifiers.
    ultimate_raw_healing : _NonNegativeFloat
        Nonnegative Ultimate healing before modifiers.
    out_of_combat_delay_steps : _NonNegativeInt
        Nonnegative wait before regeneration, in ticks.
    out_of_combat_health_regeneration_fraction_per_step : _UnitFloat
        Fraction of maximum health regenerated per eligible tick, from 0 to 1.

    Notes
    -----
    Values describe recorded rules; this model does not apply abilities.
    """

    class_id: Annotated[int, Field(ge=0, lt=NUM_CLASSES)]
    class_name: _AsciiText
    maximum_health: _NonNegativeFloat
    body_radius: _NonNegativeFloat
    base_movement_speed: _NonNegativeFloat
    observation_radius: _NonNegativeFloat
    basic_target_mode: Literal["unavailable", "ally", "enemy"]
    basic_interaction_radius: _NonNegativeFloat
    basic_raw_damage: _NonNegativeFloat
    basic_raw_healing: _NonNegativeFloat
    ultimate_target_mode: Literal[
        "unavailable",
        "target_none",
        "ally",
        "enemy",
    ]
    ultimate_interaction_radius: _NonNegativeFloat
    ultimate_cooldown_steps: _NonNegativeInt
    ultimate_raw_damage: _NonNegativeFloat
    ultimate_raw_healing: _NonNegativeFloat
    out_of_combat_delay_steps: _NonNegativeInt
    out_of_combat_health_regeneration_fraction_per_step: _UnitFloat


class StatusMechanicV1(EvaluationModel):
    """Describe one recorded status channel and its duration rule.

    Attributes
    ----------
    status_channel_id : Annotated[int, Field(ge=0, lt=9)]
        Fixed combined channel 0-8.
    status_id : _AsciiIdentifier
        Stable nonempty status identifier.
    family : Literal['slow', 'stun', 'anti_heal', 'damage_amplification',
    'movement_floor']
        slow, stun, anti_heal, damage_amplification, or movement_floor.
    source_class_id : Annotated[int, Field(ge=1, lt=NUM_CLASSES)]
        Applying class ID, 1-5.
    source_action_component : Literal['basic', 'ultimate']
        basic or ultimate.
    duration_steps : _PositiveInt
        Positive configured duration in ticks.
    magnitude_kind : Literal['movement_multiplier', 'none', 'healing_multiplier',
    'damage_multiplier', 'movement_floor']
        movement_multiplier, none, healing_multiplier, damage_multiplier, or
        movement_floor.
    magnitude : float | None
        Finite effect factor, or None only for kind none.
    application_update : Literal['maximum_remaining_duration']
        Fixed maximum_remaining_duration rule; the default.
    breaks_on_positive_damage : bool
        Whether positive damage can remove this channel.

    Notes
    -----
    Recorded channel order is validated by the owning catalog.
    """

    status_channel_id: Annotated[int, Field(ge=0, lt=9)]
    status_id: _AsciiIdentifier
    family: Literal[
        "slow",
        "stun",
        "anti_heal",
        "damage_amplification",
        "movement_floor",
    ]
    source_class_id: Annotated[int, Field(ge=1, lt=NUM_CLASSES)]
    source_action_component: Literal["basic", "ultimate"]
    duration_steps: _PositiveInt
    magnitude_kind: Literal[
        "movement_multiplier",
        "none",
        "healing_multiplier",
        "damage_multiplier",
        "movement_floor",
    ]
    magnitude: float | None
    application_update: Literal["maximum_remaining_duration"] = (
        "maximum_remaining_duration"
    )
    breaks_on_positive_damage: bool

    @model_validator(mode="after")
    def _validate_magnitude(self) -> StatusMechanicV1:
        """Require None for magnitude kind none and a finite number for other kinds.

        Return this row or raise ValueError. This does not impose an extra magnitude
        range beyond the recorded kind and finite-value rule.
        """
        if self.magnitude_kind == "none":
            if self.magnitude is not None:
                raise ValueError("stun-like status channels must omit magnitude")
        elif self.magnitude is None or not isfinite(self.magnitude):
            raise ValueError("non-stun status channels require a finite magnitude")
        return self


class AuraMechanicV1(EvaluationModel):
    """Describe one passive same-team aura from the recorded mechanics catalog.

    Attributes
    ----------
    aura_id : Literal['mage_damage_amplification', 'warrior_damage_mitigation']
        mage_damage_amplification or warrior_damage_mitigation.
    emitter_class_id : Annotated[int, Field(ge=1, lt=NUM_CLASSES)]
        Emitter class ID, 1-5.
    beneficiary_relation : Literal['same_team']
        Fixed same_team relation; the default.
    radius : _NonNegativeFloat
        Nonnegative coverage radius in world units.
    per_emitter_multiplier : _NonNegativeFloat
        Nonnegative multiplicative contribution of one covering emitter.
    stacking_rule : Literal['multiply_then_clamp']
        Fixed multiply_then_clamp rule; the default.
    clamp_kind : Literal['ceiling', 'floor']
        ceiling or floor.
    clamp_value : _NonNegativeFloat
        Nonnegative bound applied to the combined multiplier.
    """

    aura_id: Literal[
        "mage_damage_amplification",
        "warrior_damage_mitigation",
    ]
    emitter_class_id: Annotated[int, Field(ge=1, lt=NUM_CLASSES)]
    beneficiary_relation: Literal["same_team"] = "same_team"
    radius: _NonNegativeFloat
    per_emitter_multiplier: _NonNegativeFloat
    stacking_rule: Literal["multiply_then_clamp"] = "multiply_then_clamp"
    clamp_kind: Literal["ceiling", "floor"]
    clamp_value: _NonNegativeFloat


class StaticMechanicsCatalogV1(EvaluationModel):
    """Store the fixed axes, units, vocabularies, and rules needed to read an episode.

    Attributes
    ----------
    schema_id : Literal['marl_battlegrounds.evaluation.static_mechanics_catalog']
        Fixed static-mechanics-catalog identifier.
    schema_version : Literal[1]
        Version 1; the default.
    canonical_digest_sha256 : _Sha256Hex
        SHA-256 over every other catalog field.
    maximum_agent_slots : Literal[10]
        Fixed 10; the default.
    maximum_agents_per_team : Literal[5]
        Fixed 5; the default.
    number_of_teams : Literal[2]
        Fixed 2; the default.
    number_of_movement_actions : Literal[9]
        Fixed 9; the default.
    number_of_target_actions : Literal[11]
        Fixed 11; the default.
    number_of_ultimate_actions : Literal[2]
        Fixed 2; the default.
    team_global_slot_half_open_ranges : Annotated[tuple[tuple[int, int], ...],
    Field(min_length=NUM_TEAMS, max_length=NUM_TEAMS)]
        Ordered (0, 5) and (5, 10) team blocks.
    class_name_by_id : Annotated[tuple[_AsciiText, ...], Field(min_length=NUM_CLASSES,
    max_length=NUM_CLASSES)]
        Six unique names including neutral class 0.
    team_name_by_id : Annotated[tuple[_AsciiText, ...], Field(min_length=3,
    max_length=3)]
        Three names for no team, Team A, and Team B.
    movement_action_name_by_id : Annotated[tuple[_AsciiText, ...],
    Field(min_length=NUM_MOVE_ACTIONS, max_length=NUM_MOVE_ACTIONS)]
        Nine unique movement labels in action-ID order.
    target_action_name_by_id : Annotated[tuple[_AsciiText, ...],
    Field(min_length=NUM_TARGET_ACTIONS, max_length=NUM_TARGET_ACTIONS)]
        Eleven unique target labels in action-ID order.
    use_ultimate_action_name_by_id : Annotated[tuple[_AsciiText, ...],
    Field(min_length=NUM_ULTIMATE_ACTIONS, max_length=NUM_ULTIMATE_ACTIONS)]
        Two unique Ultimate-use labels.
    ultimate_target_mode_name_by_id : Annotated[tuple[_AsciiText, ...],
    Field(min_length=4, max_length=4)]
        Four unique target-mode labels.
    spawn_lifecycle_team_axis_name_by_id : Annotated[tuple[_AsciiText, ...],
    Field(min_length=NUM_TEAMS, max_length=NUM_TEAMS)]
        Two labels for own and opposing team.
    class_mechanics : Annotated[tuple[ClassMechanicsV1, ...],
    Field(min_length=NUM_CLASSES, max_length=NUM_CLASSES)]
        Six ordered class-ID rows.
    status_channels : Annotated[tuple[StatusMechanicV1, ...], Field(min_length=9,
    max_length=9)]
        Nine ordered channel-ID rows.
    aura_mechanics : Annotated[tuple[AuraMechanicV1, ...], Field(min_length=2,
    max_length=2)]
        Two uniquely identified aura rows.
    global_slow_floor : _UnitFloat
        Minimum combined slow multiplier from 0 to 1.
    global_recipient_slot_by_actor_and_target_action : Annotated[tuple[tuple[int | None,
    ...], ...], Field(min_length=MAX_AGENT_SLOTS, max_length=MAX_AGENT_SLOTS)]
        Shape (10, 11); None for target-none, then global ally/enemy slots.
    global_slot_by_actor_and_ally_observation_row : Annotated[tuple[tuple[int, ...],
    ...], Field(min_length=MAX_AGENT_SLOTS, max_length=MAX_AGENT_SLOTS)]
        Shape (10, 5), actor-relative allied global slots.
    global_slot_by_actor_and_enemy_observation_row : Annotated[tuple[tuple[int, ...],
    ...], Field(min_length=MAX_AGENT_SLOTS, max_length=MAX_AGENT_SLOTS)]
        Shape (10, 5), actor-relative opposing global slots.
    unit_direction_vector_by_movement_action : Annotated[tuple[tuple[float, float],
    ...], Field(min_length=NUM_MOVE_ACTIONS, max_length=NUM_MOVE_ACTIONS)]
        Shape (9, 2), Stay zero then unit direction vectors.
    health_unit : Literal['hit_points']
        Fixed hit_points; the default.
    spatial_unit : Literal['world_units']
        Fixed world_units; the default.
    duration_unit : Literal['transition_ticks']
        Fixed transition_ticks; the default.
    health_effect_stage_name_by_id : Annotated[tuple[_AsciiIdentifier, ...],
    Field(min_length=6, max_length=6)]
        Six frozen labels separating raw, modified, combat, net, and regeneration
        stages.

    Notes
    -----
    Catalog validation checks internal consistency and digest, not equality with the
    currently installed Core.
    """

    schema_id: Literal["marl_battlegrounds.evaluation.static_mechanics_catalog"] = (
        CATALOG_SCHEMA_ID
    )
    schema_version: Literal[1] = CATALOG_SCHEMA_VERSION
    canonical_digest_sha256: _Sha256Hex
    maximum_agent_slots: Literal[10] = MAX_AGENT_SLOTS
    maximum_agents_per_team: Literal[5] = MAX_AGENTS_PER_TEAM
    number_of_teams: Literal[2] = NUM_TEAMS
    number_of_movement_actions: Literal[9] = NUM_MOVE_ACTIONS
    number_of_target_actions: Literal[11] = NUM_TARGET_ACTIONS
    number_of_ultimate_actions: Literal[2] = NUM_ULTIMATE_ACTIONS
    team_global_slot_half_open_ranges: Annotated[
        tuple[tuple[int, int], ...],
        Field(min_length=NUM_TEAMS, max_length=NUM_TEAMS),
    ]
    class_name_by_id: Annotated[
        tuple[_AsciiText, ...],
        Field(min_length=NUM_CLASSES, max_length=NUM_CLASSES),
    ]
    team_name_by_id: Annotated[
        tuple[_AsciiText, ...], Field(min_length=3, max_length=3)
    ]
    movement_action_name_by_id: Annotated[
        tuple[_AsciiText, ...],
        Field(min_length=NUM_MOVE_ACTIONS, max_length=NUM_MOVE_ACTIONS),
    ]
    target_action_name_by_id: Annotated[
        tuple[_AsciiText, ...],
        Field(min_length=NUM_TARGET_ACTIONS, max_length=NUM_TARGET_ACTIONS),
    ]
    use_ultimate_action_name_by_id: Annotated[
        tuple[_AsciiText, ...],
        Field(min_length=NUM_ULTIMATE_ACTIONS, max_length=NUM_ULTIMATE_ACTIONS),
    ]
    ultimate_target_mode_name_by_id: Annotated[
        tuple[_AsciiText, ...], Field(min_length=4, max_length=4)
    ]
    spawn_lifecycle_team_axis_name_by_id: Annotated[
        tuple[_AsciiText, ...],
        Field(min_length=NUM_TEAMS, max_length=NUM_TEAMS),
    ]
    class_mechanics: Annotated[
        tuple[ClassMechanicsV1, ...],
        Field(min_length=NUM_CLASSES, max_length=NUM_CLASSES),
    ]
    status_channels: Annotated[
        tuple[StatusMechanicV1, ...], Field(min_length=9, max_length=9)
    ]
    aura_mechanics: Annotated[
        tuple[AuraMechanicV1, ...], Field(min_length=2, max_length=2)
    ]
    global_slow_floor: _UnitFloat
    global_recipient_slot_by_actor_and_target_action: Annotated[
        tuple[tuple[int | None, ...], ...],
        Field(min_length=MAX_AGENT_SLOTS, max_length=MAX_AGENT_SLOTS),
    ]
    global_slot_by_actor_and_ally_observation_row: Annotated[
        tuple[tuple[int, ...], ...],
        Field(min_length=MAX_AGENT_SLOTS, max_length=MAX_AGENT_SLOTS),
    ]
    global_slot_by_actor_and_enemy_observation_row: Annotated[
        tuple[tuple[int, ...], ...],
        Field(min_length=MAX_AGENT_SLOTS, max_length=MAX_AGENT_SLOTS),
    ]
    unit_direction_vector_by_movement_action: Annotated[
        tuple[tuple[float, float], ...],
        Field(min_length=NUM_MOVE_ACTIONS, max_length=NUM_MOVE_ACTIONS),
    ]
    health_unit: Literal["hit_points"] = "hit_points"
    spatial_unit: Literal["world_units"] = "world_units"
    duration_unit: Literal["transition_ticks"] = "transition_ticks"
    health_effect_stage_name_by_id: Annotated[
        tuple[_AsciiIdentifier, ...], Field(min_length=6, max_length=6)
    ]

    @model_validator(mode="after")
    def _validate_catalog(self) -> StaticMechanicsCatalogV1:
        """Check ordered class/status rows, vocabulary, action mappings, and content
        digest.

        Return this catalog or raise ValueError. Historical values may differ from live
        Core; a separate live-catalog check is needed when current-rule parity matters.
        """
        if tuple(row.class_id for row in self.class_mechanics) != tuple(
            range(NUM_CLASSES)
        ):
            raise ValueError("class mechanics must be ordered by class_id")
        if tuple(row.status_channel_id for row in self.status_channels) != tuple(
            range(9)
        ):
            raise ValueError("status channels must be ordered by status_channel_id")
        _validate_catalog_vocabulary(self)
        _validate_catalog_mappings(self)
        expected_digest = canonical_digest_sha256(
            self,
            exclude={"canonical_digest_sha256"},
        )
        if self.canonical_digest_sha256 != expected_digest:
            raise ValueError("static mechanics catalog canonical digest mismatch")
        return self


def _validate_catalog_vocabulary(catalog: StaticMechanicsCatalogV1) -> None:
    """Require unique names/IDs, aligned class names, and the frozen health-stage order.

    Raise ValueError for an ambiguous vocabulary. Do not substitute current labels.
    """
    vocabularies = (
        ("class_name_by_id", catalog.class_name_by_id),
        ("team_name_by_id", catalog.team_name_by_id),
        ("movement_action_name_by_id", catalog.movement_action_name_by_id),
        ("target_action_name_by_id", catalog.target_action_name_by_id),
        (
            "use_ultimate_action_name_by_id",
            catalog.use_ultimate_action_name_by_id,
        ),
        (
            "ultimate_target_mode_name_by_id",
            catalog.ultimate_target_mode_name_by_id,
        ),
        (
            "spawn_lifecycle_team_axis_name_by_id",
            catalog.spawn_lifecycle_team_axis_name_by_id,
        ),
    )
    for field_name, values in vocabularies:
        if len(values) != len(set(values)):
            raise ValueError(f"{field_name} entries must be unique")
    if tuple(row.class_name for row in catalog.class_mechanics) != (
        catalog.class_name_by_id
    ):
        raise ValueError("class names must align with ordered class mechanics")
    status_ids = tuple(row.status_id for row in catalog.status_channels)
    if len(status_ids) != len(set(status_ids)):
        raise ValueError("status IDs must be unique")
    aura_ids = tuple(row.aura_id for row in catalog.aura_mechanics)
    if len(aura_ids) != len(set(aura_ids)):
        raise ValueError("aura IDs must be unique")
    if catalog.health_effect_stage_name_by_id != (
        "raw_source",
        "source_modified_gross",
        "recipient_modified_gross",
        "combat_resolution_health",
        "realized_net_health_change",
        "actual_regeneration",
    ):
        raise ValueError("health-effect stage vocabulary is not recognized")


def _validate_catalog_mappings(catalog: StaticMechanicsCatalogV1) -> None:
    """Check fixed team partitions, actor-relative target rows, and movement directions.

    Require target-none to map to None, ally/enemy rows to partition slots correctly,
    and non-Stay directions to be unit vectors within the V1 tolerance. Raise
    ValueError for inconsistent mappings; preserve historical accepted tolerances.
    """
    target_rows = catalog.global_recipient_slot_by_actor_and_target_action
    ally_rows = catalog.global_slot_by_actor_and_ally_observation_row
    enemy_rows = catalog.global_slot_by_actor_and_enemy_observation_row
    expected_team_ranges = (
        (0, MAX_AGENTS_PER_TEAM),
        (MAX_AGENTS_PER_TEAM, MAX_AGENT_SLOTS),
    )
    if catalog.team_global_slot_half_open_ranges != expected_team_ranges:
        raise ValueError("team slot ranges must use the ordered V1 team blocks")
    team_slot_sets: list[set[int]] = []
    for start, stop in catalog.team_global_slot_half_open_ranges:
        if not 0 <= start < stop <= MAX_AGENT_SLOTS:
            raise ValueError("team slot ranges must be bounded non-empty ranges")
        team_slots = set(range(start, stop))
        if len(team_slots) != MAX_AGENTS_PER_TEAM:
            raise ValueError("team slot ranges must each contain five slots")
        team_slot_sets.append(team_slots)
    if set().union(*team_slot_sets) != set(range(MAX_AGENT_SLOTS)) or (
        team_slot_sets[0].intersection(team_slot_sets[1])
    ):
        raise ValueError("team slot ranges must partition every global slot")
    for actor in range(MAX_AGENT_SLOTS):
        if target_rows[actor][0] is not None:
            raise ValueError("target-none must map to JSON null")
        if len(target_rows[actor]) != NUM_TARGET_ACTIONS:
            raise ValueError("target mapping rows must have length 11")
        if (
            len(ally_rows[actor]) != MAX_AGENTS_PER_TEAM
            or len(enemy_rows[actor]) != MAX_AGENTS_PER_TEAM
        ):
            raise ValueError("relation mapping rows must have length 5")
        if target_rows[actor][1:] != (*ally_rows[actor], *enemy_rows[actor]):
            raise ValueError("target categories must align with relation rows")
        if len(set(ally_rows[actor])) != MAX_AGENTS_PER_TEAM:
            raise ValueError("ally relation rows must not repeat global slots")
        if len(set(enemy_rows[actor])) != MAX_AGENTS_PER_TEAM:
            raise ValueError("enemy relation rows must not repeat global slots")
        if set(ally_rows[actor]).intersection(enemy_rows[actor]):
            raise ValueError("ally and enemy relation rows must be disjoint")
        if set((*ally_rows[actor], *enemy_rows[actor])) != set(range(MAX_AGENT_SLOTS)):
            raise ValueError("relation rows must partition every global slot")
        actor_team_slots = next(slots for slots in team_slot_sets if actor in slots)
        if set(ally_rows[actor]) != actor_team_slots:
            raise ValueError(
                "ally relation rows must align with serialized team ranges"
            )

    directions = catalog.unit_direction_vector_by_movement_action
    if directions[0] != (0.0, 0.0):
        raise ValueError("movement action row zero must be the Stay zero vector")
    if len(set(directions)) != NUM_MOVE_ACTIONS:
        raise ValueError("movement direction rows must be unique")
    for direction in directions[1:]:
        norm = hypot(*direction)
        # Preserve the V1 ``atol + rtol * abs(reference)`` acceptance boundary.
        if abs(norm - 1.0) > 1e-6 + 1e-6 * abs(1.0):
            raise ValueError("non-Stay movement directions must be unit vectors")


class VersionedIdentityV1(EvaluationModel):
    """Name a contract whose meaning is selected by an explicit version.

    Attributes
    ----------
    identifier : _AsciiIdentifier
        Nonempty ASCII identifier.
    version : _PositiveInt
        Positive integer version.
    """

    identifier: _AsciiIdentifier
    version: _PositiveInt


class ContentAddressedIdentityV1(VersionedIdentityV1):
    """Add immutable-content identity to a named/versioned contract.

    Attributes
    ----------
    canonical_digest : _Sha256Hex
        Lowercase 64-character SHA-256 digest of normalized content.

    Notes
    -----
    Inherits identifier and positive version from VersionedIdentityV1.
    """

    canonical_digest: _Sha256Hex


class AggregationKeyV1(EvaluationModel):
    """Record one named experiment grouping coordinate.

    Attributes
    ----------
    name : _AsciiIdentifier
        Nonempty ASCII key identifier.
    value : _AsciiText
        Nonempty printable ASCII value.

    Notes
    -----
    Owning contexts require unique names in sorted order.
    """

    name: _AsciiIdentifier
    value: _AsciiText


class EvaluationEpisodeIdentityV1(EvaluationModel):
    """Keep the runner's stable episode and experiment identities together.

    Attributes
    ----------
    run_id : _AsciiIdentifier
        Run identifier.
    evaluation_id : _AsciiIdentifier
        Evaluation-pass identifier.
    matchup_id : _AsciiIdentifier
        Policy/task matchup identifier.
    match_id : _AsciiIdentifier
        Match identifier.
    episode_id : _AsciiIdentifier
        Episode identifier used by frames and transitions.
    paired_comparison_key : _AsciiIdentifier | None
        Optional key joining paired comparisons; defaults to None.
    evaluation_suite : ContentAddressedIdentityV1
        Content identity of the declared evaluation suite.
    experiment_manifest : ContentAddressedIdentityV1
        Content identity of the declared experiment.
    task : ContentAddressedIdentityV1
        Content identity of task rules.
    layout : ContentAddressedIdentityV1
        Content identity of map layout.
    curriculum : ContentAddressedIdentityV1 | None
        Optional curriculum identity; defaults to None.
    scenario : ContentAddressedIdentityV1 | None
        Optional scenario identity; defaults to None.

    Notes
    -----
    Identifiers are nonempty ASCII. This record does not discover or invent any
    experiment identity.
    """

    run_id: _AsciiIdentifier
    evaluation_id: _AsciiIdentifier
    matchup_id: _AsciiIdentifier
    match_id: _AsciiIdentifier
    episode_id: _AsciiIdentifier
    paired_comparison_key: _AsciiIdentifier | None = None
    evaluation_suite: ContentAddressedIdentityV1
    experiment_manifest: ContentAddressedIdentityV1
    task: ContentAddressedIdentityV1
    layout: ContentAddressedIdentityV1
    curriculum: ContentAddressedIdentityV1 | None = None
    scenario: ContentAddressedIdentityV1 | None = None


class ResolvedObstacleV1(EvaluationModel):
    """Record one fixed obstacle row with explicit geometry and activity.

    Attributes
    ----------
    obstacle_slot : Annotated[int, Field(ge=0, lt=MAX_OBSTACLE_SLOTS)]
        Slot 0-31.
    obstacle_type_id : _NonNegativeInt
        Nonnegative recorded obstacle category.
    x : _FiniteFloat
        Finite center x coordinate in world units.
    y : _FiniteFloat
        Finite center y coordinate in world units.
    radius : _NonNegativeFloat
        Nonnegative circle radius in world units.
    width : _NonNegativeFloat
        Nonnegative rectangular width in world units.
    height : _NonNegativeFloat
        Nonnegative rectangular height in world units.
    theta : _FiniteFloat
        Finite rectangle angle in radians.
    is_active : bool
        Whether this obstacle participates in the map.

    Notes
    -----
    Geometry validity beyond field bounds belongs to Core admission.
    """

    obstacle_slot: Annotated[int, Field(ge=0, lt=MAX_OBSTACLE_SLOTS)]
    obstacle_type_id: _NonNegativeInt
    x: _FiniteFloat
    y: _FiniteFloat
    radius: _NonNegativeFloat
    width: _NonNegativeFloat
    height: _NonNegativeFloat
    theta: _FiniteFloat
    is_active: bool


class ResolvedSlotMechanicsV1(EvaluationModel):
    """Record mechanics resolved for one global slot.

    Attributes
    ----------
    global_slot : _GlobalSlot
        Slot 0-9.
    body_radius : _NonNegativeFloat
        Nonnegative radius in world units.
    base_movement_speed : _NonNegativeFloat
        Nonnegative world-unit movement per tick.
    observation_radius : _NonNegativeFloat
        Nonnegative sight radius in world units.
    basic_interaction_radius : _NonNegativeFloat
        Nonnegative Basic range in world units.
    ultimate_interaction_radius : _NonNegativeFloat
        Nonnegative Ultimate range in world units.
    maximum_health : _NonNegativeFloat
        Nonnegative health capacity in hit points.
    out_of_combat_delay_steps : _NonNegativeInt
        Nonnegative recovery delay in ticks.
    out_of_combat_health_regeneration_fraction_per_step : _UnitFloat
        Fraction of maximum health regenerated per eligible tick, from 0 to 1.

    Notes
    -----
    Context admission requires each row to match its class catalog and inactive rows to
    be neutral.
    """

    global_slot: _GlobalSlot
    body_radius: _NonNegativeFloat
    base_movement_speed: _NonNegativeFloat
    observation_radius: _NonNegativeFloat
    basic_interaction_radius: _NonNegativeFloat
    ultimate_interaction_radius: _NonNegativeFloat
    maximum_health: _NonNegativeFloat
    out_of_combat_delay_steps: _NonNegativeInt
    out_of_combat_health_regeneration_fraction_per_step: _UnitFloat


class ResolvedEnvConfigV1(EvaluationModel):
    """Store the resolved scalar episode configuration as strict host data.

    Attributes
    ----------
    schema_id : Literal['marl_battlegrounds.evaluation.resolved_env_config']
        Fixed resolved-config identifier.
    schema_version : Literal[1]
        Version 1; the default.
    canonical_digest_sha256 : _Sha256Hex
        SHA-256 of all other fields.
    task_mode : Literal[0, 1]
        0 neutral or 1 Team Deathmatch.
    team_deathmatch_score_threshold : Annotated[int, Field(ge=0, le=2 ** 24 - 4)]
        Zero in neutral mode; positive in TDM, at most 2**24-4.
    maximum_episode_steps : Annotated[int, Field(gt=0, le=2 ** 24)]
        Positive simulator horizon, at most 2**24 ticks.
    map_width : _PositiveFloat
        Positive width in world units.
    map_height : _PositiveFloat
        Positive height in world units.
    obstacle_slots : Annotated[tuple[ResolvedObstacleV1, ...],
    Field(min_length=MAX_OBSTACLE_SLOTS, max_length=MAX_OBSTACLE_SLOTS)]
        All 32 obstacle rows in slot order.
    slot_mechanics : Annotated[tuple[ResolvedSlotMechanicsV1, ...],
    Field(min_length=MAX_AGENT_SLOTS, max_length=MAX_AGENT_SLOTS)]
        All ten resolved profiles in global-slot order.
    ordinary_movement_distance_scale : _NonNegativeFloat
        Nonnegative multiplier applied to ordinary movement distance.
    team_spawn_pad_positions : Annotated[tuple[tuple[tuple[float, float], ...], ...],
    Field(min_length=NUM_TEAMS, max_length=NUM_TEAMS)]
        Shape (2, 5, 2), raw Team A/B pad coordinates in world units.
    spawn_shield_duration_steps : _NonNegativeInt
        Nonnegative configured shield ticks.
    spawn_shield_movement_speed : _NonNegativeFloat
        Nonnegative shielded movement speed in world units per tick.
    team_respawn_wave_period_steps : Annotated[tuple[_PositiveInt, ...],
    Field(min_length=NUM_TEAMS, max_length=NUM_TEAMS)]
        Two positive wave periods, Team A then Team B, in ticks.

    Notes
    -----
    Stores geometry/mechanics separately from roster identity. Full physical validity is
    checked by Core, not this model alone.
    """

    schema_id: Literal["marl_battlegrounds.evaluation.resolved_env_config"] = (
        RESOLVED_ENV_CONFIG_SCHEMA_ID
    )
    schema_version: Literal[1] = RESOLVED_ENV_CONFIG_SCHEMA_VERSION
    canonical_digest_sha256: _Sha256Hex
    task_mode: Literal[0, 1]
    team_deathmatch_score_threshold: Annotated[int, Field(ge=0, le=2**24 - 4)]
    maximum_episode_steps: Annotated[int, Field(gt=0, le=2**24)]
    map_width: _PositiveFloat
    map_height: _PositiveFloat
    obstacle_slots: Annotated[
        tuple[ResolvedObstacleV1, ...],
        Field(min_length=MAX_OBSTACLE_SLOTS, max_length=MAX_OBSTACLE_SLOTS),
    ]
    slot_mechanics: Annotated[
        tuple[ResolvedSlotMechanicsV1, ...],
        Field(min_length=MAX_AGENT_SLOTS, max_length=MAX_AGENT_SLOTS),
    ]
    ordinary_movement_distance_scale: _NonNegativeFloat
    team_spawn_pad_positions: Annotated[
        tuple[tuple[tuple[float, float], ...], ...],
        Field(min_length=NUM_TEAMS, max_length=NUM_TEAMS),
    ]
    spawn_shield_duration_steps: _NonNegativeInt
    spawn_shield_movement_speed: _NonNegativeFloat
    team_respawn_wave_period_steps: Annotated[
        tuple[_PositiveInt, ...],
        Field(min_length=NUM_TEAMS, max_length=NUM_TEAMS),
    ]

    @model_validator(mode="after")
    def _validate_resolved_config(self) -> ResolvedEnvConfigV1:
        """Check task threshold rules, fixed row order, pad count, and config digest.

        Require TDM's positive threshold and neutral mode's zero threshold. This model
        checks recorded structure, not full Core geometry or collision validity.
        """
        if self.task_mode == 1:
            if self.team_deathmatch_score_threshold == 0:
                raise ValueError("Team Deathmatch requires a positive score threshold")
        elif self.team_deathmatch_score_threshold != 0:
            raise ValueError(
                "non-Team-Deathmatch modes require a zero Team Deathmatch threshold"
            )
        if tuple(row.obstacle_slot for row in self.obstacle_slots) != tuple(
            range(MAX_OBSTACLE_SLOTS)
        ):
            raise ValueError("obstacle rows must be ordered by obstacle_slot")
        if tuple(row.global_slot for row in self.slot_mechanics) != tuple(
            range(MAX_AGENT_SLOTS)
        ):
            raise ValueError("slot mechanics must be ordered by global_slot")
        if any(
            len(team_rows) != MAX_AGENTS_PER_TEAM
            for team_rows in self.team_spawn_pad_positions
        ):
            raise ValueError("each team must retain exactly five spawn-pad rows")
        expected_digest = canonical_digest_sha256(
            self,
            exclude={"canonical_digest_sha256"},
        )
        if self.canonical_digest_sha256 != expected_digest:
            raise ValueError("resolved environment config canonical digest mismatch")
        return self


class RosterSlotV1(EvaluationModel):
    """Record public identity and configured membership for one fixed global slot.

    Attributes
    ----------
    global_slot : _GlobalSlot
        Slot 0-9, ordered Team A then Team B.
    team_local_slot : _TeamLocalSlot
        Slot within its five-slot team block, 0-4.
    public_agent_id : _AsciiIdentifier
        Nonempty ASCII public identifier.
    configured_team_id : Annotated[int, Field(ge=0, le=2)]
        0 inactive, 1 Team A, or 2 Team B.
    class_id : Annotated[int, Field(ge=0, le=5)]
        0 inactive or class ID 1-5.
    configured_active : bool
        Whether the slot is part of the configured roster.

    Notes
    -----
    The context checks fixed topology, unique public IDs, and neutral inactive rows.
    """

    global_slot: _GlobalSlot
    team_local_slot: _TeamLocalSlot
    public_agent_id: _AsciiIdentifier
    configured_team_id: Annotated[int, Field(ge=0, le=2)]
    class_id: Annotated[int, Field(ge=0, le=5)]
    configured_active: bool


class AssignedPolicySlotV1(EvaluationModel):
    """Record complete legacy policy and training provenance for an active slot.

    Attributes
    ----------
    assignment_status : Literal['assigned']
        Fixed "assigned" discriminator; the default.
    global_slot : _GlobalSlot
        Assigned global slot, 0-9.
    evaluation_role : EvaluationRole
        focal, cooperative_partner, or adversarial_opponent.
    policy_kind : _AsciiIdentifier
        Stable policy-kind identifier.
    policy_id : _AsciiIdentifier
        Stable policy identifier.
    policy_content_digest : _Sha256Hex
        SHA-256 of policy content.
    checkpoint_digest : _Sha256Hex | None
        Optional checkpoint SHA-256; defaults to None.
    algorithm_id : _AsciiIdentifier
        Training algorithm identifier.
    training_run_id : _AsciiIdentifier
        Training-run identifier.
    training_step : _NonNegativeInt
        Nonnegative declared training coordinate.
    population_member_id : _AsciiIdentifier | None
        Optional population member identifier; defaults to None.
    parameter_sharing_group_id : _AsciiIdentifier
        Identifier grouping slots that share parameters.
    preprocessing : VersionedIdentityV1
        Input preprocessing contract identity/version.
    normalization : VersionedIdentityV1
        Normalization contract identity/version.
    execution_mode : Literal['deterministic', 'stochastic']
        deterministic or stochastic.

    Notes
    -----
    V1 requires historical provenance fields; use V2 to represent facts that were not
    recorded.
    """

    assignment_status: Literal["assigned"] = "assigned"
    global_slot: _GlobalSlot
    evaluation_role: EvaluationRole
    policy_kind: _AsciiIdentifier
    policy_id: _AsciiIdentifier
    policy_content_digest: _Sha256Hex
    checkpoint_digest: _Sha256Hex | None = None
    algorithm_id: _AsciiIdentifier
    training_run_id: _AsciiIdentifier
    training_step: _NonNegativeInt
    population_member_id: _AsciiIdentifier | None = None
    parameter_sharing_group_id: _AsciiIdentifier
    preprocessing: VersionedIdentityV1
    normalization: VersionedIdentityV1
    execution_mode: Literal["deterministic", "stochastic"]


class AssignedPolicySlotV2(EvaluationModel):
    """Record an active policy without inventing unknown training provenance.

    Attributes
    ----------
    assignment_status : Literal['assigned']
        Fixed "assigned" discriminator; the default.
    global_slot : _GlobalSlot
        Assigned global slot, 0-9.
    evaluation_role : EvaluationRole
        focal, cooperative_partner, or adversarial_opponent.
    policy_kind : _AsciiIdentifier
        Stable policy-kind identifier.
    policy_id : _AsciiIdentifier
        Stable policy identifier.
    lifecycle : Literal['frozen', 'evolving']
        frozen or evolving policy lifecycle.
    callable_name : _AsciiIdentifier | None
        Optional callable identifier; defaults to None.
    policy_content_digest : _Sha256Hex | None
        SHA-256 of policy content. None means unrecorded and is the default.
    checkpoint_digest : _Sha256Hex | None
        checkpoint SHA-256. None means unrecorded and is the default.
    algorithm_id : _AsciiIdentifier | None
        Training algorithm identifier. None means unrecorded and is the default.
    training_run_id : _AsciiIdentifier | None
        Training-run identifier. None means unrecorded and is the default.
    training_step : _NonNegativeInt | None
        Nonnegative declared training coordinate. None means unrecorded and is the
        default.
    population_member_id : _AsciiIdentifier | None
        population member identifier. None means unrecorded and is the default.
    parameter_sharing_group_id : _AsciiIdentifier | None
        Identifier grouping slots that share parameters. None means unrecorded and is
        the default.
    preprocessing : VersionedIdentityV1 | None
        Input preprocessing contract identity/version. None means unrecorded and is the
        default.
    normalization : VersionedIdentityV1 | None
        Normalization contract identity/version. None means unrecorded and is the
        default.
    execution_mode : Literal['deterministic', 'stochastic'] | None
        deterministic or stochastic. None means unrecorded and is the default.

    Notes
    -----
    Nullable facts are explicit unknowns, not generated placeholder identities.
    """

    assignment_status: Literal["assigned"] = "assigned"
    global_slot: _GlobalSlot
    evaluation_role: EvaluationRole
    policy_kind: _AsciiIdentifier
    policy_id: _AsciiIdentifier
    lifecycle: Literal["frozen", "evolving"]
    callable_name: _AsciiIdentifier | None = None
    policy_content_digest: _Sha256Hex | None = None
    checkpoint_digest: _Sha256Hex | None = None
    algorithm_id: _AsciiIdentifier | None = None
    training_run_id: _AsciiIdentifier | None = None
    training_step: _NonNegativeInt | None = None
    population_member_id: _AsciiIdentifier | None = None
    parameter_sharing_group_id: _AsciiIdentifier | None = None
    preprocessing: VersionedIdentityV1 | None = None
    normalization: VersionedIdentityV1 | None = None
    execution_mode: Literal["deterministic", "stochastic"] | None = None


class NotApplicablePolicySlotV1(EvaluationModel):
    """Keep the minimal policy row for an inactive slot.

    Attributes
    ----------
    assignment_status : Literal['not_applicable']
        Fixed "not_applicable" discriminator; the default.
    global_slot : _GlobalSlot
        Inactive global slot, 0-9.
    """

    assignment_status: Literal["not_applicable"] = "not_applicable"
    global_slot: _GlobalSlot


type PolicyAssignmentSlotV1 = Annotated[
    AssignedPolicySlotV1 | NotApplicablePolicySlotV1,
    Field(discriminator="assignment_status"),
]

type PolicyAssignmentSlotV2 = Annotated[
    AssignedPolicySlotV2 | NotApplicablePolicySlotV1,
    Field(discriminator="assignment_status"),
]

type AssignedPolicySlot = AssignedPolicySlotV1 | AssignedPolicySlotV2


class EvaluationSeedProtocolV1(EvaluationModel):
    """Record all realized scalar seeds for a historical episode.

    Attributes
    ----------
    seed_protocol : VersionedIdentityV1
        Identifier/version of the runner's seed assignment rule.
    root_seed : _Seed
        Realized unsigned 32-bit seed, from 0 through 2**32-1. Required.
    episode_seed : _Seed
        Realized unsigned 32-bit seed, from 0 through 2**32-1. Required.
    layout_seed : _Seed
        Realized unsigned 32-bit seed, from 0 through 2**32-1. Required.
    environment_seed : _Seed
        Realized unsigned 32-bit seed, from 0 through 2**32-1. Required.
    focal_policy_seed : _Seed
        Realized unsigned 32-bit seed, from 0 through 2**32-1. Required.
    evaluation_seed : _Seed
        Realized unsigned 32-bit seed, from 0 through 2**32-1. Required.
    cooperative_partner_seed : _Seed | Literal['not_applicable']
        Realized unsigned 32-bit seed or not_applicable when its role is absent.
        Required.
    adversarial_opponent_seed : _Seed | Literal['not_applicable']
        Realized unsigned 32-bit seed or not_applicable when its role is absent.
        Required.
    scenario_seed : _Seed | Literal['not_applicable']
        Realized unsigned 32-bit seed or not_applicable when its role is absent.
        Required.

    Notes
    -----
    These are seed coordinates, not JAX keys; construction does not create random
    streams.
    """

    seed_protocol: VersionedIdentityV1
    root_seed: _Seed
    episode_seed: _Seed
    layout_seed: _Seed
    environment_seed: _Seed
    focal_policy_seed: _Seed
    evaluation_seed: _Seed
    cooperative_partner_seed: _Seed | Literal["not_applicable"]
    adversarial_opponent_seed: _Seed | Literal["not_applicable"]
    scenario_seed: _Seed | Literal["not_applicable"]


class CodeRevisionV1(EvaluationModel):
    """Record complete legacy Git/source provenance.

    Attributes
    ----------
    package_version : _AsciiIdentifier
        Nonempty package-version identifier.
    commit_sha : _GitCommitHex
        Full lowercase 40-character Git SHA.
    source_tree_digest : _Sha256Hex
        Lowercase SHA-256 of source bytes.
    is_dirty : bool
        Whether source differs from its commit.
    dirty_patch_digest : _Sha256Hex | None
        SHA-256 required for known dirty source and forbidden otherwise; defaults to
        None.

    Notes
    -----
    No filesystem or Git discovery occurs in this model.
    """

    package_version: _AsciiIdentifier
    commit_sha: _GitCommitHex
    source_tree_digest: _Sha256Hex
    is_dirty: bool
    dirty_patch_digest: _Sha256Hex | None = None

    @model_validator(mode="after")
    def _validate_dirty_revision(self) -> CodeRevisionV1:
        """Require a patch digest exactly when is_dirty is true, or raise ValueError."""
        if self.is_dirty != (self.dirty_patch_digest is not None):
            raise ValueError(
                "dirty revisions require a patch digest and clean revisions forbid one"
            )
        return self


class EvaluationSeedProtocolV2(EvaluationModel):
    """Record known scalar seeds while leaving missing facts explicitly unknown.

    Attributes
    ----------
    schema_version : Literal[2]
        Version 2; the default.
    seed_protocol : VersionedIdentityV1
        Identifier/version of the runner's seed assignment rule.
    root_seed : _Seed | None
        Realized unsigned 32-bit seed, from 0 through 2**32-1. None means unrecorded and
        is the default.
    episode_seed : _Seed | None
        Realized unsigned 32-bit seed, from 0 through 2**32-1. None means unrecorded and
        is the default.
    layout_seed : _Seed | None
        Realized unsigned 32-bit seed, from 0 through 2**32-1. None means unrecorded and
        is the default.
    environment_seed : _Seed | None
        Realized unsigned 32-bit seed, from 0 through 2**32-1. None means unrecorded and
        is the default.
    focal_policy_seed : _Seed | None
        Realized unsigned 32-bit seed, from 0 through 2**32-1. None means unrecorded and
        is the default.
    evaluation_seed : _Seed | None
        Realized unsigned 32-bit seed, from 0 through 2**32-1. None means unrecorded and
        is the default.
    cooperative_partner_seed : _Seed | Literal['not_applicable'] | None
        Realized unsigned 32-bit seed or not_applicable when its role is absent. None
        means unrecorded and is the default.
    adversarial_opponent_seed : _Seed | Literal['not_applicable'] | None
        Realized unsigned 32-bit seed or not_applicable when its role is absent. None
        means unrecorded and is the default.
    scenario_seed : _Seed | Literal['not_applicable'] | None
        Realized unsigned 32-bit seed or not_applicable when its role is absent. None
        means unrecorded and is the default.

    Notes
    -----
    These are seed coordinates, not JAX keys; construction does not create random
    streams.
    """

    schema_version: Literal[2] = 2
    seed_protocol: VersionedIdentityV1
    root_seed: _Seed | None = None
    episode_seed: _Seed | None = None
    layout_seed: _Seed | None = None
    environment_seed: _Seed | None = None
    focal_policy_seed: _Seed | None = None
    evaluation_seed: _Seed | None = None
    cooperative_partner_seed: _Seed | Literal["not_applicable"] | None = None
    adversarial_opponent_seed: _Seed | Literal["not_applicable"] | None = None
    scenario_seed: _Seed | Literal["not_applicable"] | None = None


class CodeRevisionV2(EvaluationModel):
    """Record package provenance with optional discovered Git facts.

    Attributes
    ----------
    schema_version : Literal[2]
        Version 2; the default.
    package_version : _AsciiIdentifier
        Nonempty package-version identifier.
    commit_sha : _GitCommitHex | None
        Full lowercase 40-character Git SHA. None means undiscovered and is the default.
    source_tree_digest : _Sha256Hex | None
        Lowercase SHA-256 of source bytes. None means unrecorded and is the default.
    is_dirty : bool | None
        Whether source differs from its commit. None means unknown and is the default.
    dirty_patch_digest : _Sha256Hex | None
        SHA-256 required for known dirty source and forbidden otherwise; defaults to
        None.

    Notes
    -----
    No filesystem or Git discovery occurs in this model.
    """

    schema_version: Literal[2] = 2
    package_version: _AsciiIdentifier
    commit_sha: _GitCommitHex | None = None
    source_tree_digest: _Sha256Hex | None = None
    is_dirty: bool | None = None
    dirty_patch_digest: _Sha256Hex | None = None

    @model_validator(mode="after")
    def _validate_dirty_revision(self) -> CodeRevisionV2:
        """Require a patch digest for known dirty source and forbid one otherwise.

        Unknown dirty state remains None; do not infer Git facts from a digest.
        """
        if self.is_dirty is True and self.dirty_patch_digest is None:
            raise ValueError("known dirty revisions require their patch digest")
        if self.is_dirty is not True and self.dirty_patch_digest is not None:
            raise ValueError("a patch digest requires a known dirty revision")
        return self


class SchemaVersionEntryV1(EvaluationModel):
    """Name one source schema/version used by context V1.

    Attributes
    ----------
    schema_id : _AsciiIdentifier
        Nonempty schema identifier.
    schema_version : Literal[1]
        Version 1; the default.

    Notes
    -----
    The owning context checks exact ordered bindings; a row alone does not establish a
    supported schema combination.
    """

    schema_id: _AsciiIdentifier
    schema_version: Literal[1] = 1


REQUIRED_SCHEMA_BINDINGS_V1 = (
    ("marl_battlegrounds.evaluation.static_mechanics_catalog", 1),
    ("marl_battlegrounds.evaluation.resolved_env_config", 1),
    ("marl_battlegrounds.evaluation.episode_context", 1),
    ("marl_battlegrounds.evaluation.global_analysis_snapshot", 1),
    ("marl_battlegrounds.evaluation.frame", 1),
    ("marl_battlegrounds.evaluation.transition_facts", 1),
    ("marl_battlegrounds.evaluation.event", 1),
    ("marl_battlegrounds.evaluation.transition", 1),
)


class EvaluationEpisodeContextV1(EvaluationModel):
    """Historical episode metadata stored once for all captured frames.

    Attributes
    ----------
    schema_id : Literal['marl_battlegrounds.evaluation.episode_context']
        Fixed episode-context identifier.
    schema_version : Literal[1]
        Version 1; the default.
    identity : EvaluationEpisodeIdentityV1
        Stable runner-owned episode and experiment identities.
    schema_versions : tuple[SchemaVersionEntryV1, ...]
        Exact ordered REQUIRED_SCHEMA_BINDINGS_V1 entries.
    aggregation_keys : tuple[AggregationKeyV1, ...]
        Unique experiment coordinates sorted by name.
    expected_horizon : _PositiveInt
        Positive expected artifact transition count, no larger than simulator horizon.
    resolved_env_config : ResolvedEnvConfigV1
        Recorded scalar config and ten resolved profiles.
    static_mechanics_catalog : StaticMechanicsCatalogV1
        Recorded mechanics, units, and action mappings.
    roster : Annotated[tuple[RosterSlotV1, ...], Field(min_length=MAX_AGENT_SLOTS,
    max_length=MAX_AGENT_SLOTS)]
        Ten fixed global slots with unique public IDs.
    policy_assignments : Annotated[tuple[PolicyAssignmentSlotV1, ...],
    Field(min_length=MAX_AGENT_SLOTS, max_length=MAX_AGENT_SLOTS)]
        Ten aligned V1 active assignments or inactive not-applicable rows.
    seed_protocol : EvaluationSeedProtocolV1
        Complete V1 seed provenance.
    capture_profile : CaptureProfile
        training_light, evaluation_metric_complete, scenario_metric_complete, or debug.
    execution_information_mode : ExecutionInformationMode
        shared_obs or no_shared_obs.
    actor_projection : VersionedIdentityV1
        Declared historical actor-projection identity.
    critic_information_regime : VersionedIdentityV1
        Declared critic-information contract.
    canonical_reward_mode : VersionedIdentityV1
        Declared canonical task-reward contract.
    shaping_configuration : ContentAddressedIdentityV1
        Content identity for reward-shaping settings.
    code_revision : CodeRevisionV1
        Complete V1 source/Git identity.

    Notes
    -----
    Pairs with frame V1. Context validates slot topology, exact class-catalog profiles,
    active policy roles, and known seed joins. It does not grant actor access to all
    stored provenance.
    """

    schema_id: Literal["marl_battlegrounds.evaluation.episode_context"] = (
        CONTEXT_SCHEMA_ID
    )
    schema_version: Literal[1] = CONTEXT_SCHEMA_VERSION
    identity: EvaluationEpisodeIdentityV1
    schema_versions: tuple[SchemaVersionEntryV1, ...]
    aggregation_keys: tuple[AggregationKeyV1, ...]
    expected_horizon: _PositiveInt
    resolved_env_config: ResolvedEnvConfigV1
    static_mechanics_catalog: StaticMechanicsCatalogV1
    roster: Annotated[
        tuple[RosterSlotV1, ...],
        Field(min_length=MAX_AGENT_SLOTS, max_length=MAX_AGENT_SLOTS),
    ]
    policy_assignments: Annotated[
        tuple[PolicyAssignmentSlotV1, ...],
        Field(min_length=MAX_AGENT_SLOTS, max_length=MAX_AGENT_SLOTS),
    ]
    seed_protocol: EvaluationSeedProtocolV1
    capture_profile: CaptureProfile
    execution_information_mode: ExecutionInformationMode
    actor_projection: VersionedIdentityV1
    critic_information_regime: VersionedIdentityV1
    canonical_reward_mode: VersionedIdentityV1
    shaping_configuration: ContentAddressedIdentityV1
    code_revision: CodeRevisionV1

    @model_validator(mode="after")
    def _validate_context(self) -> EvaluationEpisodeContextV1:
        """Require exact V1 schema bindings and coherent horizon, roster, seed, and
        scenario metadata.

        Historical projection identity remains recorded without imposing current input
        semantics.
        Return this context or raise ValueError.
        """
        schema_bindings = tuple(
            (row.schema_id, row.schema_version) for row in self.schema_versions
        )
        if schema_bindings != REQUIRED_SCHEMA_BINDINGS_V1:
            raise ValueError("schema_versions must equal the eight CP2 V1 roots")
        if self.expected_horizon > self.resolved_env_config.maximum_episode_steps:
            raise ValueError("expected_horizon cannot exceed maximum_episode_steps")
        aggregation_names = tuple(row.name for row in self.aggregation_keys)
        if len(aggregation_names) != len(set(aggregation_names)):
            raise ValueError("aggregation key names must be unique")
        if aggregation_names != tuple(sorted(aggregation_names)):
            raise ValueError("aggregation keys must be sorted by name")
        _validate_context_rows(self)
        _validate_context_seeds(self)
        if (
            self.capture_profile == "scenario_metric_complete"
            and self.identity.scenario is None
        ):
            raise ValueError("scenario metric-complete capture requires a scenario")
        return self


class SchemaVersionEntryV2(EvaluationModel):
    """Name one source schema/version used by context V2.

    Attributes
    ----------
    schema_id : _AsciiIdentifier
        Nonempty schema identifier.
    schema_version : Annotated[int, Field(ge=1, le=2)]
        Integer from 1 through 2.

    Notes
    -----
    The owning context checks exact ordered bindings; a row alone does not establish a
    supported schema combination.
    """

    schema_id: _AsciiIdentifier
    schema_version: Annotated[int, Field(ge=1, le=2)]


REQUIRED_SCHEMA_BINDINGS_V2 = tuple(
    (schema_id, 2 if schema_id == CONTEXT_SCHEMA_ID else version)
    for schema_id, version in REQUIRED_SCHEMA_BINDINGS_V1
)


class EvaluationEpisodeContextV2(EvaluationModel):
    """Historical episode metadata stored once for all captured frames.

    Attributes
    ----------
    schema_id : Literal['marl_battlegrounds.evaluation.episode_context']
        Fixed episode-context identifier.
    schema_version : Literal[2]
        Version 2; the default.
    identity : EvaluationEpisodeIdentityV1
        Stable runner-owned episode and experiment identities.
    schema_versions : tuple[SchemaVersionEntryV2, ...]
        Exact ordered REQUIRED_SCHEMA_BINDINGS_V2 entries.
    aggregation_keys : tuple[AggregationKeyV1, ...]
        Unique experiment coordinates sorted by name.
    expected_horizon : _PositiveInt
        Positive expected artifact transition count, no larger than simulator horizon.
    resolved_env_config : ResolvedEnvConfigV1
        Recorded scalar config and ten resolved profiles.
    static_mechanics_catalog : StaticMechanicsCatalogV1
        Recorded mechanics, units, and action mappings.
    roster : Annotated[tuple[RosterSlotV1, ...], Field(min_length=MAX_AGENT_SLOTS,
    max_length=MAX_AGENT_SLOTS)]
        Ten fixed global slots with unique public IDs.
    policy_assignments : Annotated[tuple[PolicyAssignmentSlotV2, ...],
    Field(min_length=MAX_AGENT_SLOTS, max_length=MAX_AGENT_SLOTS)]
        Ten aligned V2 active assignments or inactive not-applicable rows.
    seed_protocol : EvaluationSeedProtocolV1 | EvaluationSeedProtocolV2
        V1 complete or V2 optional seed provenance.
    capture_profile : CaptureProfile
        training_light, evaluation_metric_complete, scenario_metric_complete, or debug.
    execution_information_mode : ExecutionInformationMode
        shared_obs or no_shared_obs.
    actor_projection : VersionedIdentityV1
        Declared historical actor-projection identity.
    critic_information_regime : VersionedIdentityV1
        Declared critic-information contract.
    canonical_reward_mode : VersionedIdentityV1
        Declared canonical task-reward contract.
    shaping_configuration : ContentAddressedIdentityV1
        Content identity for reward-shaping settings.
    code_revision : CodeRevisionV1 | CodeRevisionV2
        V1 complete or V2 optional source/Git identity.
    scenario_name : _AsciiText | None
        Optional printable display name requiring scenario identity; defaults to None.

    Notes
    -----
    Pairs with frame V1. Context validates slot topology, exact class-catalog profiles,
    active policy roles, and known seed joins. It does not grant actor access to all
    stored provenance.
    """

    schema_id: Literal["marl_battlegrounds.evaluation.episode_context"] = (
        CONTEXT_SCHEMA_ID
    )
    schema_version: Literal[2] = 2
    identity: EvaluationEpisodeIdentityV1
    schema_versions: tuple[SchemaVersionEntryV2, ...]
    aggregation_keys: tuple[AggregationKeyV1, ...]
    expected_horizon: _PositiveInt
    resolved_env_config: ResolvedEnvConfigV1
    static_mechanics_catalog: StaticMechanicsCatalogV1
    roster: Annotated[
        tuple[RosterSlotV1, ...],
        Field(min_length=MAX_AGENT_SLOTS, max_length=MAX_AGENT_SLOTS),
    ]
    policy_assignments: Annotated[
        tuple[PolicyAssignmentSlotV2, ...],
        Field(min_length=MAX_AGENT_SLOTS, max_length=MAX_AGENT_SLOTS),
    ]
    seed_protocol: EvaluationSeedProtocolV1 | EvaluationSeedProtocolV2
    capture_profile: CaptureProfile
    execution_information_mode: ExecutionInformationMode
    actor_projection: VersionedIdentityV1
    critic_information_regime: VersionedIdentityV1
    canonical_reward_mode: VersionedIdentityV1
    shaping_configuration: ContentAddressedIdentityV1
    code_revision: CodeRevisionV1 | CodeRevisionV2
    scenario_name: _AsciiText | None = None

    @model_validator(mode="after")
    def _validate_context(self) -> EvaluationEpisodeContextV2:
        """Require exact V2 schema bindings and coherent horizon, roster, seed, and
        scenario metadata.

        Historical projection identity remains recorded without imposing current input
        semantics.
        Return this context or raise ValueError.
        """
        if (
            tuple((row.schema_id, row.schema_version) for row in self.schema_versions)
            != REQUIRED_SCHEMA_BINDINGS_V2
        ):
            raise ValueError("schema_versions must equal the V2 context bindings")
        _validate_context_contents(self)
        return self


class SchemaVersionEntryV3(EvaluationModel):
    """Name one source schema/version used by context V3.

    Attributes
    ----------
    schema_id : _AsciiIdentifier
        Nonempty schema identifier.
    schema_version : Annotated[int, Field(ge=1, le=3)]
        Integer from 1 through 3.

    Notes
    -----
    The owning context checks exact ordered bindings; a row alone does not establish a
    supported schema combination.
    """

    schema_id: _AsciiIdentifier
    schema_version: Annotated[int, Field(ge=1, le=3)]


REQUIRED_SCHEMA_BINDINGS_V3 = tuple(
    (
        schema_id,
        3
        if schema_id == CONTEXT_SCHEMA_ID
        else 2
        if schema_id == FRAME_SCHEMA_ID
        else version,
    )
    for schema_id, version in REQUIRED_SCHEMA_BINDINGS_V1
)


class EvaluationEpisodeContextV3(EvaluationModel):
    """Current episode metadata stored once for all captured frames.

    Attributes
    ----------
    schema_id : Literal['marl_battlegrounds.evaluation.episode_context']
        Fixed episode-context identifier.
    schema_version : Literal[3]
        Version 3; the default.
    identity : EvaluationEpisodeIdentityV1
        Stable runner-owned episode and experiment identities.
    schema_versions : tuple[SchemaVersionEntryV3, ...]
        Exact ordered REQUIRED_SCHEMA_BINDINGS_V3 entries.
    aggregation_keys : tuple[AggregationKeyV1, ...]
        Unique experiment coordinates sorted by name.
    expected_horizon : _PositiveInt
        Positive expected artifact transition count, no larger than simulator horizon.
    resolved_env_config : ResolvedEnvConfigV1
        Recorded scalar config and ten resolved profiles.
    static_mechanics_catalog : StaticMechanicsCatalogV1
        Recorded mechanics, units, and action mappings.
    roster : Annotated[tuple[RosterSlotV1, ...], Field(min_length=MAX_AGENT_SLOTS,
    max_length=MAX_AGENT_SLOTS)]
        Ten fixed global slots with unique public IDs.
    policy_assignments : Annotated[tuple[PolicyAssignmentSlotV2, ...],
    Field(min_length=MAX_AGENT_SLOTS, max_length=MAX_AGENT_SLOTS)]
        Ten aligned V2 active assignments or inactive not-applicable rows.
    seed_protocol : EvaluationSeedProtocolV1 | EvaluationSeedProtocolV2
        V1 complete or V2 optional seed provenance.
    capture_profile : CaptureProfile
        training_light, evaluation_metric_complete, scenario_metric_complete, or debug.
    execution_information_mode : ExecutionInformationMode
        shared_obs or no_shared_obs.
    actor_projection : VersionedIdentityV1
        SharedObs V2 or NoSharedObs V3 matching the information mode.
    critic_information_regime : VersionedIdentityV1
        Declared critic-information contract.
    canonical_reward_mode : VersionedIdentityV1
        Declared canonical task-reward contract.
    shaping_configuration : ContentAddressedIdentityV1
        Content identity for reward-shaping settings.
    code_revision : CodeRevisionV1 | CodeRevisionV2
        V1 complete or V2 optional source/Git identity.
    scenario_name : _AsciiText | None
        Optional printable display name requiring scenario identity; defaults to None.

    Notes
    -----
    Pairs with frame V2. Context validates slot topology, exact class-catalog profiles,
    active policy roles, and known seed joins. It does not grant actor access to all
    stored provenance.
    """

    schema_id: Literal["marl_battlegrounds.evaluation.episode_context"] = (
        CONTEXT_SCHEMA_ID
    )
    schema_version: Literal[3] = 3
    identity: EvaluationEpisodeIdentityV1
    schema_versions: tuple[SchemaVersionEntryV3, ...]
    aggregation_keys: tuple[AggregationKeyV1, ...]
    expected_horizon: _PositiveInt
    resolved_env_config: ResolvedEnvConfigV1
    static_mechanics_catalog: StaticMechanicsCatalogV1
    roster: Annotated[
        tuple[RosterSlotV1, ...],
        Field(min_length=MAX_AGENT_SLOTS, max_length=MAX_AGENT_SLOTS),
    ]
    policy_assignments: Annotated[
        tuple[PolicyAssignmentSlotV2, ...],
        Field(min_length=MAX_AGENT_SLOTS, max_length=MAX_AGENT_SLOTS),
    ]
    seed_protocol: EvaluationSeedProtocolV1 | EvaluationSeedProtocolV2
    capture_profile: CaptureProfile
    execution_information_mode: ExecutionInformationMode
    actor_projection: VersionedIdentityV1
    critic_information_regime: VersionedIdentityV1
    canonical_reward_mode: VersionedIdentityV1
    shaping_configuration: ContentAddressedIdentityV1
    code_revision: CodeRevisionV1 | CodeRevisionV2
    scenario_name: _AsciiText | None = None

    @model_validator(mode="after")
    def _validate_context(self) -> EvaluationEpisodeContextV3:
        """Require exact V3 schema bindings and coherent horizon, roster, seed, and
        scenario metadata.

        Also require the current projection identity matching SharedObs or NoSharedObs.
        Return this context or raise ValueError.
        """
        if (
            tuple((row.schema_id, row.schema_version) for row in self.schema_versions)
            != REQUIRED_SCHEMA_BINDINGS_V3
        ):
            raise ValueError("schema_versions must equal the V3 context bindings")
        from marl_battlegrounds.evaluation.actor_projection import (
            NO_SHARED_OBS_ACTOR_PROJECTION_V3,
            SHARED_OBS_ACTOR_PROJECTION_V2,
        )

        expected_projection = (
            SHARED_OBS_ACTOR_PROJECTION_V2
            if self.execution_information_mode == "shared_obs"
            else NO_SHARED_OBS_ACTOR_PROJECTION_V3
        )
        if self.actor_projection != expected_projection:
            raise ValueError(
                "current context requires the matching relative-input projection"
            )
        _validate_context_contents(self)
        return self


type EvaluationEpisodeContext = (
    EvaluationEpisodeContextV1 | EvaluationEpisodeContextV2 | EvaluationEpisodeContextV3
)


def _validate_context_contents(
    context: EvaluationEpisodeContextV2 | EvaluationEpisodeContextV3,
) -> None:
    """Check shared V2/V3 horizon, sorted strata, roster, seeds, and scenario metadata.

    Scenario capture/name needs a scenario identity. Raise ValueError on disagreement;
    version-specific schema and actor-projection checks stay with each context model.
    """
    if context.expected_horizon > context.resolved_env_config.maximum_episode_steps:
        raise ValueError("expected_horizon cannot exceed maximum_episode_steps")
    names = tuple(row.name for row in context.aggregation_keys)
    if names != tuple(sorted(set(names))):
        raise ValueError("aggregation keys must be unique and sorted by name")
    _validate_context_rows(context)
    _validate_context_seeds(context)
    if (
        context.capture_profile == "scenario_metric_complete"
        and context.identity.scenario is None
    ):
        raise ValueError("scenario capture requires a scenario identity")
    if context.scenario_name is not None and context.identity.scenario is None:
        raise ValueError("scenario display name requires its recorded identity")


def evaluation_context_type(
    context: EvaluationEpisodeContext,
) -> (
    type[EvaluationEpisodeContextV1]
    | type[EvaluationEpisodeContextV2]
    | type[EvaluationEpisodeContextV3]
):
    """Return the exact supported context class after checking its root type.

    Parameters
    ----------
    context : EvaluationEpisodeContext
        Episode context V1, V2, or V3.

    Returns
    -------
    type[EvaluationEpisodeContextV1] | type[EvaluationEpisodeContextV2] |
    type[EvaluationEpisodeContextV3]
        Its exact concrete model class.

    Raises
    ------
    TypeError
        The root is unsupported or a subclass of a declared context.

    Notes
    -----
    This is a root-type check only, not deep content revalidation.
    """
    if type(context) not in (
        EvaluationEpisodeContextV1,
        EvaluationEpisodeContextV2,
        EvaluationEpisodeContextV3,
    ):
        raise TypeError("context must be an exact supported episode-context root")
    return type(context)


def _validate_context_rows(context: EvaluationEpisodeContext) -> None:
    """Check fixed roster/policy order and exact class-catalog profile membership.

    Require unique public IDs, fixed team blocks, active assigned policies, and
    neutral inactive profile/policy rows. Resolved slot mechanics must exactly match
    their recorded class catalog; TDM needs active members on both teams.
    Raise ValueError without repairing or substituting roster values.
    """
    slots = tuple(row.global_slot for row in context.roster)
    if slots != tuple(range(MAX_AGENT_SLOTS)):
        raise ValueError("roster must contain ordered fixed global slots")
    public_ids = tuple(row.public_agent_id for row in context.roster)
    if len(public_ids) != len(set(public_ids)):
        raise ValueError("public agent IDs must be unique")
    if tuple(row.global_slot for row in context.policy_assignments) != slots:
        raise ValueError("policy assignments must align with roster slots")
    for roster_row, mechanics_row, policy_row in zip(
        context.roster,
        context.resolved_env_config.slot_mechanics,
        context.policy_assignments,
        strict=True,
    ):
        if roster_row.team_local_slot != roster_row.global_slot % MAX_AGENTS_PER_TEAM:
            raise ValueError("team-local slots must follow fixed team blocks")
        expected_team_id = 1 if roster_row.global_slot < MAX_AGENTS_PER_TEAM else 2
        if roster_row.configured_active:
            if (
                roster_row.configured_team_id != expected_team_id
                or roster_row.class_id == 0
            ):
                raise ValueError("active roster rows require fixed team and class")
            if not isinstance(policy_row, (AssignedPolicySlotV1, AssignedPolicySlotV2)):
                raise ValueError("active roster rows require assigned policies")
        else:
            if roster_row.configured_team_id != 0 or roster_row.class_id != 0:
                raise ValueError("inactive roster rows require neutral team and class")
            if any(
                value != 0
                for value in (
                    mechanics_row.body_radius,
                    mechanics_row.base_movement_speed,
                    mechanics_row.observation_radius,
                    mechanics_row.basic_interaction_radius,
                    mechanics_row.ultimate_interaction_radius,
                    mechanics_row.maximum_health,
                    mechanics_row.out_of_combat_delay_steps,
                    mechanics_row.out_of_combat_health_regeneration_fraction_per_step,
                )
            ):
                raise ValueError("inactive roster rows require neutral profile values")
            if not isinstance(policy_row, NotApplicablePolicySlotV1):
                raise ValueError("inactive roster rows require not-applicable policies")
        class_row = context.static_mechanics_catalog.class_mechanics[
            roster_row.class_id
        ]
        expected_profile = (
            class_row.body_radius,
            class_row.base_movement_speed,
            class_row.observation_radius,
            class_row.basic_interaction_radius,
            class_row.ultimate_interaction_radius,
            class_row.maximum_health,
            class_row.out_of_combat_delay_steps,
            class_row.out_of_combat_health_regeneration_fraction_per_step,
        )
        actual_profile = (
            mechanics_row.body_radius,
            mechanics_row.base_movement_speed,
            mechanics_row.observation_radius,
            mechanics_row.basic_interaction_radius,
            mechanics_row.ultimate_interaction_radius,
            mechanics_row.maximum_health,
            mechanics_row.out_of_combat_delay_steps,
            mechanics_row.out_of_combat_health_regeneration_fraction_per_step,
        )
        if actual_profile != expected_profile:
            raise ValueError("roster profile disagrees with mechanics catalog")

    if context.resolved_env_config.task_mode == 1:
        active_team_ids = {
            row.configured_team_id for row in context.roster if row.configured_active
        }
        if active_team_ids != {1, 2}:
            raise ValueError(
                "Team Deathmatch context requires at least one configured active "
                "member on each team"
            )


def _validate_context_seeds(context: EvaluationEpisodeContext) -> None:
    """Require a focal policy and consistency between known role seeds and assignments.

    Known partner/opponent/scenario seeds must be present exactly when those roles
    exist. V2 None values mean unknown and are left unknown. Raise ValueError on a
    contradiction; do not generate missing seeds.
    """
    active_roles = {
        row.evaluation_role
        for row in context.policy_assignments
        if isinstance(row, (AssignedPolicySlotV1, AssignedPolicySlotV2))
    }
    if "focal" not in active_roles:
        raise ValueError("evaluation context requires at least one focal policy")
    seeds = context.seed_protocol
    role_seed_pairs = (
        ("cooperative_partner", seeds.cooperative_partner_seed),
        ("adversarial_opponent", seeds.adversarial_opponent_seed),
    )
    for role, seed in role_seed_pairs:
        if seed is None:
            continue
        if (role in active_roles) != (seed != "not_applicable"):
            raise ValueError(f"{role} seed presence must match policy assignments")
    if seeds.scenario_seed is not None and (context.identity.scenario is not None) != (
        seeds.scenario_seed != "not_applicable"
    ):
        raise ValueError("scenario seed presence must match scenario identity")


type _FloatVector = tuple[_FiniteFloat, ...]
type _NonNegativeFloatVector = tuple[_NonNegativeFloat, ...]
type _IntegerVector = tuple[int, ...]
type _NonNegativeIntegerVector = tuple[_NonNegativeInt, ...]
type _BooleanVector = tuple[bool, ...]
type _FloatMatrix = tuple[_FloatVector, ...]
type _NonNegativeIntegerMatrix = tuple[_NonNegativeIntegerVector, ...]
type _BooleanMatrix = tuple[_BooleanVector, ...]
type _FloatTensor3 = tuple[_FloatMatrix, ...]
type _NonNegativeIntegerTensor3 = tuple[_NonNegativeIntegerMatrix, ...]
type _BooleanTensor3 = tuple[_BooleanMatrix, ...]
type _FloatTensor4 = tuple[_FloatTensor3, ...]


def _require_tuple_shape(
    value: object,
    expected_shape: tuple[int, ...],
    *,
    field_name: str,
) -> None:
    """Check every tuple axis against a fixed shape without coercing container types.

    A scalar shape ends recursion; typed model fields own leaf-value checks.
    Raise ValueError naming the field on a missing tuple or wrong axis length.
    """
    if not expected_shape:
        return
    if not isinstance(value, tuple):
        raise ValueError(f"{field_name} must have shape {expected_shape}")
    sequence = cast(tuple[object, ...], value)
    if len(sequence) != expected_shape[0]:
        raise ValueError(f"{field_name} must have shape {expected_shape}")
    for item in sequence:
        _require_tuple_shape(
            item,
            expected_shape[1:],
            field_name=field_name,
        )


class GlobalAnalysisSnapshotV1(EvaluationModel):
    """Keep one frame's dynamic privileged simulator state as host tuples.

    Attributes
    ----------
    schema_id : Literal['marl_battlegrounds.evaluation.global_analysis_snapshot']
        Fixed global-analysis-snapshot identifier.
    schema_version : Literal[1]
        Version 1; the default.
    team_deathmatch_scores : _NonNegativeIntegerVector
        Two nonnegative scores, Team A then Team B.
    alive_mask : _BooleanVector
        Ten bool values in global-slot order.
    agent_positions : _FloatMatrix
        Shape (10, 2), finite world-unit coordinates.
    current_health : _NonNegativeFloatVector
        Ten nonnegative hit-point values.
    ultimate_cooldowns : _NonNegativeIntegerVector
        Ten nonnegative remaining tick counts.
    slow_durations : _NonNegativeIntegerMatrix
        Shape (10, 3), nonnegative slow-channel ticks.
    stun_durations : _NonNegativeIntegerMatrix
        Shape (10, 3), nonnegative stun-channel ticks.
    rogue_poison_anti_heal_durations : _NonNegativeIntegerVector
        Ten nonnegative remaining Poison anti-heal ticks.
    mage_burst_damage_amplification_durations : _NonNegativeIntegerVector
        Ten nonnegative remaining Burst amplification ticks.
    priest_blessing_of_freedom_slow_floor_durations : _NonNegativeIntegerVector
        Ten nonnegative remaining Freedom floor ticks.
    team_respawn_wave_countdowns : _NonNegativeIntegerVector
        Two nonnegative wave countdowns, Team A then Team B.
    spawn_shield_durations : _NonNegativeIntegerVector
        Ten nonnegative remaining shield ticks.
    steps_until_out_of_combat : _NonNegativeIntegerVector
        Ten nonnegative recovery countdowns.
    previous_timestep_move_actions : tuple[Annotated[int, Field(ge=0,
    lt=NUM_MOVE_ACTIONS)], ...]
        Ten accepted movement IDs, 0-8.
    previous_timestep_select_target_actions : tuple[Annotated[int, Field(ge=0,
    lt=NUM_TARGET_ACTIONS)], ...]
        Ten accepted target IDs, 0-10.
    previous_timestep_use_ultimate_actions : tuple[Annotated[int, Field(ge=0,
    lt=NUM_ULTIMATE_ACTIONS)], ...]
        Ten accepted Ultimate-use IDs, 0-1.
    has_previous_timestep_joint_action : bool
        Whether previous-action rows represent an actual prior transition.

    Notes
    -----
    Simulator tick is stored on the owning frame. This full snapshot is analysis data,
    not automatically permitted policy input.
    """

    schema_id: Literal["marl_battlegrounds.evaluation.global_analysis_snapshot"] = (
        GLOBAL_ANALYSIS_SNAPSHOT_SCHEMA_ID
    )
    schema_version: Literal[1] = GLOBAL_ANALYSIS_SNAPSHOT_SCHEMA_VERSION
    team_deathmatch_scores: _NonNegativeIntegerVector
    alive_mask: _BooleanVector
    agent_positions: _FloatMatrix
    current_health: _NonNegativeFloatVector
    ultimate_cooldowns: _NonNegativeIntegerVector
    slow_durations: _NonNegativeIntegerMatrix
    stun_durations: _NonNegativeIntegerMatrix
    rogue_poison_anti_heal_durations: _NonNegativeIntegerVector
    mage_burst_damage_amplification_durations: _NonNegativeIntegerVector
    priest_blessing_of_freedom_slow_floor_durations: _NonNegativeIntegerVector
    team_respawn_wave_countdowns: _NonNegativeIntegerVector
    spawn_shield_durations: _NonNegativeIntegerVector
    steps_until_out_of_combat: _NonNegativeIntegerVector
    previous_timestep_move_actions: tuple[
        Annotated[int, Field(ge=0, lt=NUM_MOVE_ACTIONS)], ...
    ]
    previous_timestep_select_target_actions: tuple[
        Annotated[int, Field(ge=0, lt=NUM_TARGET_ACTIONS)], ...
    ]
    previous_timestep_use_ultimate_actions: tuple[
        Annotated[int, Field(ge=0, lt=NUM_ULTIMATE_ACTIONS)], ...
    ]
    has_previous_timestep_joint_action: bool

    @model_validator(mode="after")
    def _validate_snapshot_shapes(self) -> GlobalAnalysisSnapshotV1:
        """Require ten-slot, two-team, three-status-channel, and two-coordinate tuple
        axes.

        Typed fields check value ranges; this validator raises ValueError on shape
        mismatch.
        """
        vector_fields = (
            "alive_mask",
            "current_health",
            "ultimate_cooldowns",
            "rogue_poison_anti_heal_durations",
            "mage_burst_damage_amplification_durations",
            "priest_blessing_of_freedom_slow_floor_durations",
            "spawn_shield_durations",
            "steps_until_out_of_combat",
            "previous_timestep_move_actions",
            "previous_timestep_select_target_actions",
            "previous_timestep_use_ultimate_actions",
        )
        for field_name in vector_fields:
            _require_tuple_shape(
                getattr(self, field_name),
                (MAX_AGENT_SLOTS,),
                field_name=field_name,
            )
        _require_tuple_shape(
            self.team_deathmatch_scores,
            (NUM_TEAMS,),
            field_name="team_deathmatch_scores",
        )
        _require_tuple_shape(
            self.agent_positions,
            (MAX_AGENT_SLOTS, ENVIRONMENT_DIMENSIONS),
            field_name="agent_positions",
        )
        _require_tuple_shape(
            self.slow_durations,
            (MAX_AGENT_SLOTS, NUM_SLOW_CHANNELS),
            field_name="slow_durations",
        )
        _require_tuple_shape(
            self.stun_durations,
            (MAX_AGENT_SLOTS, NUM_STUN_CHANNELS),
            field_name="stun_durations",
        )
        _require_tuple_shape(
            self.team_respawn_wave_countdowns,
            (NUM_TEAMS,),
            field_name="team_respawn_wave_countdowns",
        )
        return self


class PreviousTimestepActionObservationV1(EvaluationModel):
    """Keep each observer's recorded ally/enemy accepted-action history.

    Attributes
    ----------
    ally_previous_timestep_move_actions_one_hot : _FloatTensor3
        Shape (10, 5, 9), finite recorded ally move features in observer-relative row
        order.
    ally_previous_timestep_select_target_actions_one_hot : _FloatTensor3
        Shape (10, 5, 11), finite recorded ally select_target features in
        observer-relative row order.
    ally_previous_timestep_use_ultimate_actions_one_hot : _FloatTensor3
        Shape (10, 5, 2), finite recorded ally use_ultimate features in
        observer-relative row order.
    enemy_previous_timestep_move_actions_one_hot : _FloatTensor3
        Shape (10, 5, 9), finite recorded enemy move features in observer-relative row
        order.
    enemy_previous_timestep_select_target_actions_one_hot : _FloatTensor3
        Shape (10, 5, 11), finite recorded enemy select_target features in
        observer-relative row order.
    enemy_previous_timestep_use_ultimate_actions_one_hot : _FloatTensor3
        Shape (10, 5, 2), finite recorded enemy use_ultimate features in
        observer-relative row order.

    Notes
    -----
    The producer defines one-hot or unavailable neutral rows. This model checks finite
    values and shapes, not whether rows are truly one-hot.
    """

    ally_previous_timestep_move_actions_one_hot: _FloatTensor3
    enemy_previous_timestep_move_actions_one_hot: _FloatTensor3
    ally_previous_timestep_select_target_actions_one_hot: _FloatTensor3
    enemy_previous_timestep_select_target_actions_one_hot: _FloatTensor3
    ally_previous_timestep_use_ultimate_actions_one_hot: _FloatTensor3
    enemy_previous_timestep_use_ultimate_actions_one_hot: _FloatTensor3

    @model_validator(mode="after")
    def _validate_previous_action_shapes(
        self,
    ) -> PreviousTimestepActionObservationV1:
        """Require (10, 5, head_categories) for all six previous-action tensors.

        Raise ValueError on shape mismatch; do not infer visibility or enforce one-hot
        sums.
        """
        action_fields_and_sizes = (
            ("ally_previous_timestep_move_actions_one_hot", NUM_MOVE_ACTIONS),
            ("enemy_previous_timestep_move_actions_one_hot", NUM_MOVE_ACTIONS),
            (
                "ally_previous_timestep_select_target_actions_one_hot",
                NUM_TARGET_ACTIONS,
            ),
            (
                "enemy_previous_timestep_select_target_actions_one_hot",
                NUM_TARGET_ACTIONS,
            ),
            (
                "ally_previous_timestep_use_ultimate_actions_one_hot",
                NUM_ULTIMATE_ACTIONS,
            ),
            (
                "enemy_previous_timestep_use_ultimate_actions_one_hot",
                NUM_ULTIMATE_ACTIONS,
            ),
        )
        for field_name, category_count in action_fields_and_sizes:
            _require_tuple_shape(
                getattr(self, field_name),
                (MAX_AGENT_SLOTS, MAX_AGENTS_PER_TEAM, category_count),
                field_name=field_name,
            )
        return self


class SpawnLifecycleObservationV1(EvaluationModel):
    """Keep public spawn, shield, roster, and wave-clock input in actor-relative order.

    Attributes
    ----------
    spawn_pad_positions_by_agent_by_team : _FloatTensor4
        Shape (10, 2, 5, 2), finite world-unit coordinates.
    spawn_shield_actual_durations_by_agent_by_team : _NonNegativeIntegerTensor3
        Shape (10, 2, 5), nonnegative remaining shield ticks.
    spawn_shield_configured_duration_by_agent : _NonNegativeIntegerVector
        Ten nonnegative configured shield durations in ticks.
    spawn_shield_speed_by_agent : _NonNegativeFloatVector
        Ten nonnegative world-unit speeds per tick.
    respawn_wave_period_step_count_by_agent_by_team : _NonNegativeIntegerMatrix
        Shape (10, 2), nonnegative recorded wave periods.
    respawn_wave_countdowns_by_agent_by_team : _NonNegativeIntegerMatrix
        Shape (10, 2), nonnegative wave countdown ticks.
    active_mask_by_agent_by_team : _BooleanTensor3
        Shape (10, 2, 5), bool configured membership.
    alive_mask_by_agent_by_team : _BooleanTensor3
        Shape (10, 2, 5), bool life-state values.

    Notes
    -----
    First axis is observer; team axis is own team then opponent. Class IDs are
    reconstructed from context rather than added to this historical wire subtree.
    """

    spawn_pad_positions_by_agent_by_team: _FloatTensor4
    spawn_shield_actual_durations_by_agent_by_team: _NonNegativeIntegerTensor3
    spawn_shield_configured_duration_by_agent: _NonNegativeIntegerVector
    spawn_shield_speed_by_agent: _NonNegativeFloatVector
    respawn_wave_period_step_count_by_agent_by_team: _NonNegativeIntegerMatrix
    respawn_wave_countdowns_by_agent_by_team: _NonNegativeIntegerMatrix
    active_mask_by_agent_by_team: _BooleanTensor3
    alive_mask_by_agent_by_team: _BooleanTensor3

    @model_validator(mode="after")
    def _validate_spawn_lifecycle_shapes(self) -> SpawnLifecycleObservationV1:
        """Check fixed observer/team/slot/coordinate shapes for all spawn input fields.

        Raise ValueError on mismatch; context and capture own semantic roster joins.
        """
        _require_tuple_shape(
            self.spawn_pad_positions_by_agent_by_team,
            (
                MAX_AGENT_SLOTS,
                NUM_TEAMS,
                MAX_AGENTS_PER_TEAM,
                ENVIRONMENT_DIMENSIONS,
            ),
            field_name="spawn_pad_positions_by_agent_by_team",
        )
        for field_name in (
            "spawn_shield_actual_durations_by_agent_by_team",
            "active_mask_by_agent_by_team",
            "alive_mask_by_agent_by_team",
        ):
            _require_tuple_shape(
                getattr(self, field_name),
                (MAX_AGENT_SLOTS, NUM_TEAMS, MAX_AGENTS_PER_TEAM),
                field_name=field_name,
            )
        for field_name in (
            "spawn_shield_configured_duration_by_agent",
            "spawn_shield_speed_by_agent",
        ):
            _require_tuple_shape(
                getattr(self, field_name),
                (MAX_AGENT_SLOTS,),
                field_name=field_name,
            )
        for field_name in (
            "respawn_wave_period_step_count_by_agent_by_team",
            "respawn_wave_countdowns_by_agent_by_team",
        ):
            _require_tuple_shape(
                getattr(self, field_name),
                (MAX_AGENT_SLOTS, NUM_TEAMS),
                field_name=field_name,
            )
        return self


class BaseObservationV1(EvaluationModel):
    """Store the ten actors' historical base observations once per frame.

    Attributes
    ----------
    self_features : _FloatMatrix
        Shape (10, 58), finite self features with historical Team ID column.
    ally_unit_features : _FloatTensor3
        Shape (10, 5, 58), finite actor-relative ally rows.
    enemy_unit_features : _FloatTensor3
        Shape (10, 5, 58), finite actor-relative enemy rows.
    map_obstacle_features : _FloatTensor3
        Shape (10, 32, 8), finite obstacle features.
    objective_features : _FloatTensor3
        Shape (10, 8, 12), finite objective features.
    context_features : _FloatMatrix
        Shape (10, 19), finite public task/context features.
    ally_visibility_mask : _BooleanMatrix
        Shape (10, 5), bool ally visibility.
    enemy_visibility_mask : _BooleanMatrix
        Shape (10, 5), bool enemy visibility.
    previous_timestep_actions : PreviousTimestepActionObservationV1
        Recorded ally/enemy previous-action tensors.
    spawn_lifecycle : SpawnLifecycleObservationV1
        Actor-relative spawn and wave-clock data.

    Notes
    -----
    These are all actors' source observations. Actor projection applies the selected
    actor's information rights; the whole record is not one actor's input.
    """

    self_features: _FloatMatrix
    ally_unit_features: _FloatTensor3
    enemy_unit_features: _FloatTensor3
    map_obstacle_features: _FloatTensor3
    objective_features: _FloatTensor3
    context_features: _FloatMatrix
    ally_visibility_mask: _BooleanMatrix
    enemy_visibility_mask: _BooleanMatrix
    previous_timestep_actions: PreviousTimestepActionObservationV1
    spawn_lifecycle: SpawnLifecycleObservationV1

    @model_validator(mode="after")
    def _validate_observation_shapes(self) -> BaseObservationV1:
        """Require the frozen observer/slot/feature shapes for base observation arrays.

        Raise ValueError on mismatch; version-specific identity checks live in V2.
        """
        shapes = (
            ("self_features", (MAX_AGENT_SLOTS, SELF_FEATURES)),
            (
                "ally_unit_features",
                (MAX_AGENT_SLOTS, MAX_AGENTS_PER_TEAM, UNIT_FEATURES),
            ),
            (
                "enemy_unit_features",
                (MAX_AGENT_SLOTS, MAX_AGENTS_PER_TEAM, UNIT_FEATURES),
            ),
            (
                "map_obstacle_features",
                (MAX_AGENT_SLOTS, MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES),
            ),
            (
                "objective_features",
                (MAX_AGENT_SLOTS, MAX_OBJECTIVE_SLOTS, OBJECTIVE_FEATURES),
            ),
            ("context_features", (MAX_AGENT_SLOTS, CONTEXT_FEATURES)),
            ("ally_visibility_mask", (MAX_AGENT_SLOTS, MAX_AGENTS_PER_TEAM)),
            ("enemy_visibility_mask", (MAX_AGENT_SLOTS, MAX_AGENTS_PER_TEAM)),
        )
        for field_name, shape in shapes:
            _require_tuple_shape(
                getattr(self, field_name),
                shape,
                field_name=field_name,
            )
        return self


class BaseObservationV2(BaseObservationV1):
    """Store current relative policy features and each actor's own ally-row index.

    Attributes
    ----------
    self_ally_index : tuple[_TeamLocalSlot, ...]
        Ten integers 0-4; active actors use their team-local slot and inactive actors
        use zero.

    Notes
    -----
    Inherits all BaseObservationV1 field shapes. Column 3 now means is_enemy: zero for
    self/allies and equal to visibility for enemies. This replaces historical Team ID
    meaning; context V3 names this contract.
    """

    self_ally_index: tuple[_TeamLocalSlot, ...]

    @model_validator(mode="after")
    def _validate_relative_identity(self) -> BaseObservationV2:
        """Check zero self/ally is_enemy flags, visible enemy flags, and active self
        indices.

        Use self feature column 4 for active state. Return this model or raise
        ValueError.
        """
        _require_tuple_shape(
            self.self_ally_index, (MAX_AGENT_SLOTS,), field_name="self_ally_index"
        )
        for slot, row in enumerate(self.self_features):
            if row[3] != 0.0:
                raise ValueError("current self feature is_enemy must be zero")
            expected_index = slot % MAX_AGENTS_PER_TEAM if row[4] == 1.0 else 0
            if self.self_ally_index[slot] != expected_index:
                raise ValueError(
                    "self_ally_index must match the active actor's own-team row"
                )
            if any(candidate[3] != 0.0 for candidate in self.ally_unit_features[slot]):
                raise ValueError("current ally feature is_enemy must be zero")
            for visible, candidate in zip(
                self.enemy_visibility_mask[slot],
                self.enemy_unit_features[slot],
                strict=True,
            ):
                if candidate[3] != float(visible):
                    raise ValueError(
                        "current enemy feature is_enemy must match visibility"
                    )
        return self


class ActionMaskV1(EvaluationModel):
    """Record fixed action masks with joint target/Ultimate legality as authority.

    Attributes
    ----------
    move_mask : _BooleanMatrix
        Shape (10, 9), bool movement legality.
    select_target_mask : _BooleanMatrix
        Shape (10, 11), bool marginal over legal Ultimate choices.
    use_ultimate_mask : _BooleanMatrix
        Shape (10, 2), bool marginal over legal target choices.
    select_target_use_ultimate_joint_mask : _BooleanTensor3
        Shape (10, 11, 2), bool legal target/Ultimate pairs.

    Notes
    -----
    Independent combat-head marginals do not guarantee a legal pair; sample from or
    check the joint mask.
    """

    move_mask: _BooleanMatrix
    select_target_mask: _BooleanMatrix
    use_ultimate_mask: _BooleanMatrix
    select_target_use_ultimate_joint_mask: _BooleanTensor3

    @model_validator(mode="after")
    def _validate_action_mask(self) -> ActionMaskV1:
        """Require fixed shapes and target/Ultimate marginals equal to the joint mask.

        Raise ValueError on mismatch. This does not recompute simulator legality.
        """
        shapes = (
            ("move_mask", (MAX_AGENT_SLOTS, NUM_MOVE_ACTIONS)),
            ("select_target_mask", (MAX_AGENT_SLOTS, NUM_TARGET_ACTIONS)),
            ("use_ultimate_mask", (MAX_AGENT_SLOTS, NUM_ULTIMATE_ACTIONS)),
            (
                "select_target_use_ultimate_joint_mask",
                (MAX_AGENT_SLOTS, NUM_TARGET_ACTIONS, NUM_ULTIMATE_ACTIONS),
            ),
        )
        for field_name, shape in shapes:
            _require_tuple_shape(
                getattr(self, field_name),
                shape,
                field_name=field_name,
            )
        select_target_marginal = tuple(
            tuple(any(ultimate_row) for ultimate_row in actor_rows)
            for actor_rows in self.select_target_use_ultimate_joint_mask
        )
        use_ultimate_marginal = tuple(
            tuple(
                any(
                    actor_rows[target][ultimate] for target in range(NUM_TARGET_ACTIONS)
                )
                for ultimate in range(NUM_ULTIMATE_ACTIONS)
            )
            for actor_rows in self.select_target_use_ultimate_joint_mask
        )
        if self.select_target_mask != select_target_marginal:
            raise ValueError("select_target_mask must equal the joint-mask marginal")
        if self.use_ultimate_mask != use_ultimate_marginal:
            raise ValueError("use_ultimate_mask must equal the joint-mask marginal")
        return self


class EvaluationFrameV1(EvaluationModel):
    """Store one historical decision epoch and its same-epoch input material.

    Attributes
    ----------
    schema_id : Literal['marl_battlegrounds.evaluation.frame']
        Fixed frame identifier.
    schema_version : Literal[1]
        Version 1; the default.
    episode_id : _AsciiIdentifier
        Owning episode identifier.
    frame_index : _NonNegativeInt
        Nonnegative artifact coordinate starting at zero.
    frame_id : _AsciiIdentifier
        Canonical episode ID followed by :frame: and frame_index.
    simulator_step_count : _NonNegativeInt
        Nonnegative simulator tick, independent of artifact index.
    snapshot : GlobalAnalysisSnapshotV1
        Full privileged dynamic snapshot.
    base_observation : BaseObservationV1
        BaseObservationV1 for all ten actors.
    action_mask : ActionMaskV1
        Recorded same-epoch action masks.
    shared_obs_information_availability_by_recipient_and_sensor_source : _BooleanMatrix
    | None
        Optional (10, 10) bool recipient/source matrix; defaults to None.

    Notes
    -----
    Context V1/V2 owns version and information-mode admission. Model construction checks
    local ID and shape; cross-team, inactive, and self-sharing restrictions need the
    context join.
    """

    schema_id: Literal["marl_battlegrounds.evaluation.frame"] = FRAME_SCHEMA_ID
    schema_version: Literal[1] = FRAME_SCHEMA_VERSION
    episode_id: _AsciiIdentifier
    frame_index: _NonNegativeInt
    frame_id: _AsciiIdentifier
    simulator_step_count: _NonNegativeInt
    snapshot: GlobalAnalysisSnapshotV1
    base_observation: BaseObservationV1
    action_mask: ActionMaskV1
    shared_obs_information_availability_by_recipient_and_sensor_source: (
        _BooleanMatrix | None
    ) = None

    @model_validator(mode="after")
    def _validate_frame(self) -> EvaluationFrameV1:
        """Check shared canonical ID and availability shape, then return this frame.

        Raise ValueError on a local mismatch; context-specific admission is separate.
        """
        _validate_frame_fields(self)
        return self


class EvaluationFrameV2(EvaluationModel):
    """Store one current decision epoch and its same-epoch input material.

    Attributes
    ----------
    schema_id : Literal['marl_battlegrounds.evaluation.frame']
        Fixed frame identifier.
    schema_version : Literal[2]
        Version 2; the default.
    episode_id : _AsciiIdentifier
        Owning episode identifier.
    frame_index : _NonNegativeInt
        Nonnegative artifact coordinate starting at zero.
    frame_id : _AsciiIdentifier
        Canonical episode ID followed by :frame: and frame_index.
    simulator_step_count : _NonNegativeInt
        Nonnegative simulator tick, independent of artifact index.
    snapshot : GlobalAnalysisSnapshotV1
        Full privileged dynamic snapshot.
    base_observation : BaseObservationV2
        BaseObservationV2 for all ten actors.
    action_mask : ActionMaskV1
        Recorded same-epoch action masks.
    shared_obs_information_availability_by_recipient_and_sensor_source : _BooleanMatrix
    | None
        Optional (10, 10) bool recipient/source matrix; defaults to None.

    Notes
    -----
    Context V3 owns version and information-mode admission. Model construction checks
    local ID and shape; cross-team, inactive, and self-sharing restrictions need the
    context join.
    """

    schema_id: Literal["marl_battlegrounds.evaluation.frame"] = FRAME_SCHEMA_ID
    schema_version: Literal[2] = 2
    episode_id: _AsciiIdentifier
    frame_index: _NonNegativeInt
    frame_id: _AsciiIdentifier
    simulator_step_count: _NonNegativeInt
    snapshot: GlobalAnalysisSnapshotV1
    base_observation: BaseObservationV2
    action_mask: ActionMaskV1
    shared_obs_information_availability_by_recipient_and_sensor_source: (
        _BooleanMatrix | None
    ) = None

    @model_validator(mode="after")
    def _validate_frame(self) -> EvaluationFrameV2:
        """Check shared canonical ID and availability shape, then return this frame.

        Raise ValueError on a local mismatch; context-specific admission is separate.
        """
        _validate_frame_fields(self)
        return self


type EvaluationFrame = EvaluationFrameV1 | EvaluationFrameV2


def evaluation_frame_type(
    frame: EvaluationFrame,
) -> type[EvaluationFrameV1] | type[EvaluationFrameV2]:
    """Return the exact supported frame class after checking its root type.

    Parameters
    ----------
    frame : EvaluationFrame
        Historical frame V1 or current frame V2.

    Returns
    -------
    type[EvaluationFrameV1] | type[EvaluationFrameV2]
        Its exact concrete model class.

    Raises
    ------
    TypeError
        The root is unsupported or an undeclared subclass.

    Notes
    -----
    This helper does not deeply revalidate frame content or its context join.
    """
    if type(frame) not in (EvaluationFrameV1, EvaluationFrameV2):
        raise TypeError("frame must be an exact supported evaluation frame")
    return type(frame)


def _validate_frame_fields(frame: EvaluationFrame) -> None:
    """Check canonical frame identity and optional (10, 10) availability shape.

    Context-specific permission rules are checked when the frame joins its context.
    Raise ValueError on an ID or shape mismatch.
    """
    expected_id = f"{frame.episode_id}:frame:{frame.frame_index}"
    if frame.frame_id != expected_id:
        raise ValueError("frame_id must be derived from episode_id and frame_index")
    availability = (
        frame.shared_obs_information_availability_by_recipient_and_sensor_source
    )
    if availability is not None:
        _require_tuple_shape(
            availability,
            (MAX_AGENT_SLOTS, MAX_AGENT_SLOTS),
            field_name=(
                "shared_obs_information_availability_by_recipient_and_sensor_source"
            ),
        )


class JointActionV1(EvaluationModel):
    """Keep submitted or accepted action categories for all ten global slots.

    Attributes
    ----------
    move : tuple[_Int32, ...]
        Ten exact int32-range movement integers.
    select_target : tuple[_Int32, ...]
        Ten exact int32-range target integers.
    use_ultimate : tuple[_Int32, ...]
        Ten exact int32-range Ultimate-use integers.

    Notes
    -----
    Submitted intent may be outside valid head categories. ActionAcceptanceFactsV1
    enforces category bounds on the accepted action only.
    """

    move: tuple[_Int32, ...]
    select_target: tuple[_Int32, ...]
    use_ultimate: tuple[_Int32, ...]

    @model_validator(mode="after")
    def _validate_action_shapes(self) -> JointActionV1:
        """Require length ten for each action head; raise ValueError on mismatch."""
        for field_name in ("move", "select_target", "use_ultimate"):
            _require_tuple_shape(
                getattr(self, field_name),
                (MAX_AGENT_SLOTS,),
                field_name=field_name,
            )
        return self


class ActionAcceptanceFactsV1(EvaluationModel):
    """Keep submitted intent, accepted actions, and independent rejection reasons.

    Attributes
    ----------
    submitted_joint_action : JointActionV1
        Original three-head int32-range action, including invalid categories.
    accepted_joint_action : JointActionV1
        Accepted action with movement 0-8, target 0-10, Ultimate 0-1.
    submitted_action_tuple_is_out_of_domain_by_actor : _BooleanVector
        Ten bool flags for out-of-domain submitted tuples.
    in_domain_move_action_is_rejected_by_actor : _BooleanVector
        Ten bool flags for rejected in-domain movement.
    in_domain_combat_action_pair_is_rejected_by_actor : _BooleanVector
        Ten bool flags for rejected in-domain target/Ultimate pairs.

    Notes
    -----
    Facts are Core-authored. This model checks shapes and accepted category domains;
    full action acceptance is not recalculated.
    """

    submitted_joint_action: JointActionV1
    accepted_joint_action: JointActionV1
    submitted_action_tuple_is_out_of_domain_by_actor: _BooleanVector
    in_domain_move_action_is_rejected_by_actor: _BooleanVector
    in_domain_combat_action_pair_is_rejected_by_actor: _BooleanVector

    @model_validator(mode="after")
    def _validate_action_acceptance(self) -> ActionAcceptanceFactsV1:
        """Check ten rejection flags per family and valid categories for accepted heads.

        Preserve submitted invalid categories for analysis; raise ValueError on
        mismatch.
        """
        for field_name in (
            "submitted_action_tuple_is_out_of_domain_by_actor",
            "in_domain_move_action_is_rejected_by_actor",
            "in_domain_combat_action_pair_is_rejected_by_actor",
        ):
            _require_tuple_shape(
                getattr(self, field_name),
                (MAX_AGENT_SLOTS,),
                field_name=field_name,
            )
        accepted = self.accepted_joint_action
        accepted_domains = (
            (accepted.move, NUM_MOVE_ACTIONS, "accepted move"),
            (accepted.select_target, NUM_TARGET_ACTIONS, "accepted target"),
            (accepted.use_ultimate, NUM_ULTIMATE_ACTIONS, "accepted ultimate"),
        )
        for values, category_count, field_name in accepted_domains:
            if any(value < 0 or value >= category_count for value in values):
                raise ValueError(f"{field_name} actions must be category-bounded")
        return self


class CombatTransitionFactsV1(EvaluationModel):
    """Keep source outputs and recipient totals from one combat resolution.

    Attributes
    ----------
    basic_effect_is_activated_by_source : _BooleanVector
        Ten bool Basic activation flags.
    ultimate_effect_is_activated_by_source : _BooleanVector
        Ten bool Ultimate activation flags.
    combat_effect_has_recipient_by_source : _BooleanVector
        Ten bool flags identifying routed sources.
    combat_effect_recipient_global_slot_by_source : tuple[_GlobalSlot | None, ...]
        Ten recipient slots 0-9 or None, matching has-recipient flags.
    raw_damage_output_by_source : _NonNegativeFloatVector
        Ten nonnegative raw damage amounts in hit points.
    source_modified_damage_output_by_source : _NonNegativeFloatVector
        Ten nonnegative source-modified gross damage amounts.
    recipient_damage_modifier_by_source : _NonNegativeFloatVector
        Ten nonnegative routed recipient multipliers.
    total_effective_damage_by_recipient : _NonNegativeFloatVector
        Ten nonnegative combined recipient damage amounts.
    raw_healing_output_by_source : _NonNegativeFloatVector
        Ten nonnegative raw healing amounts in hit points.
    source_modified_healing_output_by_source : _NonNegativeFloatVector
        Ten nonnegative source-modified gross healing amounts.
    recipient_healing_modifier_by_source : _NonNegativeFloatVector
        Ten nonnegative routed recipient healing multipliers.
    total_effective_healing_by_recipient : _NonNegativeFloatVector
        Ten nonnegative combined recipient healing amounts.
    health_after_combat_resolution_by_recipient : _NonNegativeFloatVector
        Ten nonnegative hit-point values before later regeneration.
    slow_is_applied_by_source_and_channel : _BooleanMatrix
        Shape (10, 3), bool slow applications.
    stun_is_applied_by_source_and_channel : _BooleanMatrix
        Shape (10, 3), bool stun applications.
    rogue_poison_anti_heal_is_applied_by_source : _BooleanVector
        Ten bool Poison anti-heal application flags.
    mage_burst_damage_amplification_is_applied_by_source : _BooleanVector
        Ten bool Burst amplification application flags.
    priest_blessing_of_freedom_is_applied_by_source : _BooleanVector
        Ten bool Freedom application flags.

    Notes
    -----
    Global-slot axes stay raw simulator order. Gross source output is distinct from net
    recipient health change.
    """

    basic_effect_is_activated_by_source: _BooleanVector
    ultimate_effect_is_activated_by_source: _BooleanVector
    combat_effect_has_recipient_by_source: _BooleanVector
    combat_effect_recipient_global_slot_by_source: tuple[_GlobalSlot | None, ...]
    raw_damage_output_by_source: _NonNegativeFloatVector
    source_modified_damage_output_by_source: _NonNegativeFloatVector
    recipient_damage_modifier_by_source: _NonNegativeFloatVector
    total_effective_damage_by_recipient: _NonNegativeFloatVector
    raw_healing_output_by_source: _NonNegativeFloatVector
    source_modified_healing_output_by_source: _NonNegativeFloatVector
    recipient_healing_modifier_by_source: _NonNegativeFloatVector
    total_effective_healing_by_recipient: _NonNegativeFloatVector
    health_after_combat_resolution_by_recipient: _NonNegativeFloatVector
    slow_is_applied_by_source_and_channel: _BooleanMatrix
    stun_is_applied_by_source_and_channel: _BooleanMatrix
    rogue_poison_anti_heal_is_applied_by_source: _BooleanVector
    mage_burst_damage_amplification_is_applied_by_source: _BooleanVector
    priest_blessing_of_freedom_is_applied_by_source: _BooleanVector

    @model_validator(mode="after")
    def _validate_combat_facts(self) -> CombatTransitionFactsV1:
        """Check ten-slot and three-channel shapes plus nullable recipient-route flags.

        Raise ValueError on mismatch; preserve Core's separate source and recipient
        stages.
        """
        matrix_shapes = (
            ("slow_is_applied_by_source_and_channel", NUM_SLOW_CHANNELS),
            ("stun_is_applied_by_source_and_channel", NUM_STUN_CHANNELS),
        )
        matrix_names = {name for name, _width in matrix_shapes}
        for field_name in self.__class__.model_fields:
            if field_name in matrix_names:
                continue
            _require_tuple_shape(
                getattr(self, field_name),
                (MAX_AGENT_SLOTS,),
                field_name=field_name,
            )
        for field_name, width in matrix_shapes:
            _require_tuple_shape(
                getattr(self, field_name),
                (MAX_AGENT_SLOTS, width),
                field_name=field_name,
            )
        for has_recipient, recipient in zip(
            self.combat_effect_has_recipient_by_source,
            self.combat_effect_recipient_global_slot_by_source,
            strict=True,
        ):
            if has_recipient != (recipient is not None):
                raise ValueError(
                    "has-recipient facts must agree with nullable recipient slots"
                )
        return self


class DeathTransitionFactsV1(EvaluationModel):
    """Keep newly dead recipients and their direct damage contributors.

    Attributes
    ----------
    is_newly_dead_by_recipient : _BooleanVector
        Ten bool flags for new deaths.
    contributed_to_new_death_by_source : _BooleanVector
        Ten bool direct-contributor flags.
    attributed_death_damage_by_source : _NonNegativeFloatVector
        Ten nonnegative gross damage amounts attributed to new deaths.

    Notes
    -----
    These are recorded facts. Multiple flags can coexist; model construction checks
    shape, not fresh simulator causality.
    """

    is_newly_dead_by_recipient: _BooleanVector
    contributed_to_new_death_by_source: _BooleanVector
    attributed_death_damage_by_source: _NonNegativeFloatVector

    @model_validator(mode="after")
    def _validate_death_shapes(self) -> DeathTransitionFactsV1:
        """Require fixed ten-slot death tuple axes.

        Return this fact record or raise ValueError; typed fields own primitive bounds.
        """
        for field_name in self.__class__.model_fields:
            _require_tuple_shape(
                getattr(self, field_name),
                (MAX_AGENT_SLOTS,),
                field_name=field_name,
            )
        return self


class SpawnShieldTransitionFactsV1(EvaluationModel):
    """Keep transition-start shield activity and ordinary expiry.

    Attributes
    ----------
    was_active_at_transition_start_by_agent : _BooleanVector
        Ten bool flags for shield activity at the decision epoch.
    expired_at_transition_end_by_agent : _BooleanVector
        Ten bool flags for ordinary end-of-transition expiry.

    Notes
    -----
    These are recorded facts. Multiple flags can coexist; model construction checks
    shape, not fresh simulator causality.
    """

    was_active_at_transition_start_by_agent: _BooleanVector
    expired_at_transition_end_by_agent: _BooleanVector

    @model_validator(mode="after")
    def _validate_spawn_shield_shapes(self) -> SpawnShieldTransitionFactsV1:
        """Require fixed ten-slot shield tuple axes.

        Return this fact record or raise ValueError; typed fields own primitive bounds.
        """
        for field_name in self.__class__.model_fields:
            _require_tuple_shape(
                getattr(self, field_name),
                (MAX_AGENT_SLOTS,),
                field_name=field_name,
            )
        return self


class RespawnTransitionFactsV1(EvaluationModel):
    """Keep team waves and realized agent respawns separately.

    Attributes
    ----------
    respawn_wave_occurred_this_transition_by_team : _BooleanVector
        Two bool wave flags, Team A then Team B.
    was_respawned_this_transition_by_agent : _BooleanVector
        Ten bool realized respawn flags.

    Notes
    -----
    These are recorded facts. Multiple flags can coexist; model construction checks
    shape, not fresh simulator causality.
    """

    respawn_wave_occurred_this_transition_by_team: _BooleanVector
    was_respawned_this_transition_by_agent: _BooleanVector

    @model_validator(mode="after")
    def _validate_respawn_shapes(self) -> RespawnTransitionFactsV1:
        """Require fixed two-team and ten-slot respawn tuple axes.

        Return this fact record or raise ValueError; typed fields own primitive bounds.
        """
        _require_tuple_shape(
            self.respawn_wave_occurred_this_transition_by_team,
            (NUM_TEAMS,),
            field_name="respawn_wave_occurred_this_transition_by_team",
        )
        _require_tuple_shape(
            self.was_respawned_this_transition_by_agent,
            (MAX_AGENT_SLOTS,),
            field_name="was_respawned_this_transition_by_agent",
        )
        return self


class RegenerationTransitionFactsV1(EvaluationModel):
    """Keep recovery-countdown resets and actual health regeneration.

    Attributes
    ----------
    combat_countdown_was_reset_by_agent : _BooleanVector
        Ten bool countdown-reset flags.
    actual_health_regenerated_this_step_by_agent : _NonNegativeFloatVector
        Ten nonnegative actual regenerated hit-point amounts.

    Notes
    -----
    These are recorded facts. Multiple flags can coexist; model construction checks
    shape, not fresh simulator causality.
    """

    combat_countdown_was_reset_by_agent: _BooleanVector
    actual_health_regenerated_this_step_by_agent: _NonNegativeFloatVector

    @model_validator(mode="after")
    def _validate_regeneration_shapes(self) -> RegenerationTransitionFactsV1:
        """Require fixed ten-slot regeneration tuple axes.

        Return this fact record or raise ValueError; typed fields own primitive bounds.
        """
        for field_name in self.__class__.model_fields:
            _require_tuple_shape(
                getattr(self, field_name),
                (MAX_AGENT_SLOTS,),
                field_name=field_name,
            )
        return self


class PhysicalTransitionFactsV1(EvaluationModel):
    """Keep separate realized movement vectors for each physical phase.

    Attributes
    ----------
    charge_phase_displacement_by_agent : _FloatMatrix
        Shape (10, 2), finite Charge displacement in world units.
    ordinary_movement_phase_displacement_by_agent : _FloatMatrix
        Shape (10, 2), finite ordinary-movement displacement in world units.

    Notes
    -----
    These are recorded facts. Multiple flags can coexist; model construction checks
    shape, not fresh simulator causality.
    """

    charge_phase_displacement_by_agent: _FloatMatrix
    ordinary_movement_phase_displacement_by_agent: _FloatMatrix

    @model_validator(mode="after")
    def _validate_displacement_shapes(self) -> PhysicalTransitionFactsV1:
        """Require fixed (10, 2) displacement tuple axes.

        Return this fact record or raise ValueError; typed fields own primitive bounds.
        """
        for field_name in (
            "charge_phase_displacement_by_agent",
            "ordinary_movement_phase_displacement_by_agent",
        ):
            _require_tuple_shape(
                getattr(self, field_name),
                (MAX_AGENT_SLOTS, ENVIRONMENT_DIMENSIONS),
                field_name=field_name,
            )
        return self


class AuraTransitionFactsV1(EvaluationModel):
    """Keep transition-start aura coverage by emitter and beneficiary.

    Attributes
    ----------
    is_covered_by_mage_damage_aura_by_emitter_and_beneficiary : _BooleanMatrix
        Shape (10, 10), bool Mage damage-aura coverage.
    is_covered_by_warrior_mitigation_aura_by_emitter_and_beneficiary : _BooleanMatrix
        Shape (10, 10), bool Warrior mitigation-aura coverage.

    Notes
    -----
    These are recorded facts. Multiple flags can coexist; model construction checks
    shape, not fresh simulator causality.
    """

    is_covered_by_mage_damage_aura_by_emitter_and_beneficiary: _BooleanMatrix
    is_covered_by_warrior_mitigation_aura_by_emitter_and_beneficiary: _BooleanMatrix

    @model_validator(mode="after")
    def _validate_aura_shapes(self) -> AuraTransitionFactsV1:
        """Require fixed (10, 10) emitter/beneficiary tuple axes.

        Return this fact record or raise ValueError; typed fields own primitive bounds.
        """
        for field_name in self.__class__.model_fields:
            _require_tuple_shape(
                getattr(self, field_name),
                (MAX_AGENT_SLOTS, MAX_AGENT_SLOTS),
                field_name=field_name,
            )
        return self


class StatusLifecycleTransitionFactsV1(EvaluationModel):
    """Keep independent status changes by recipient and combined channel.

    Attributes
    ----------
    aged_to_zero_by_recipient_and_status_channel : _BooleanMatrix
        Shape (10, 9), bool ordinary-aging flags.
    refreshed_or_extended_by_recipient_and_status_channel : _BooleanMatrix
        Shape (10, 9), bool accepted-duration increase flags.
    broken_by_damage_by_recipient_and_status_channel : _BooleanMatrix
        Shape (10, 9), bool positive-damage break flags.
    cleared_by_new_death_by_recipient_and_status_channel : _BooleanMatrix
        Shape (10, 9), bool death-clearing flags.

    Notes
    -----
    These are recorded facts. Multiple flags can coexist; model construction checks
    shape, not fresh simulator causality.
    """

    aged_to_zero_by_recipient_and_status_channel: _BooleanMatrix
    refreshed_or_extended_by_recipient_and_status_channel: _BooleanMatrix
    broken_by_damage_by_recipient_and_status_channel: _BooleanMatrix
    cleared_by_new_death_by_recipient_and_status_channel: _BooleanMatrix

    @model_validator(mode="after")
    def _validate_status_lifecycle_shapes(self) -> StatusLifecycleTransitionFactsV1:
        """Require fixed (10, 9) recipient/channel tuple axes.

        Return this fact record or raise ValueError; typed fields own primitive bounds.
        """
        for field_name in self.__class__.model_fields:
            _require_tuple_shape(
                getattr(self, field_name),
                (MAX_AGENT_SLOTS, 9),
                field_name=field_name,
            )
        return self


class TeamDeathmatchTransitionFactsV1(EvaluationModel):
    """Keep the task-authored outcome after one transition.

    Attributes
    ----------
    outcome : Annotated[int, Field(ge=0, le=3)]
        0 ongoing, 1 Team A win, 2 Team B win, or 3 draw.

    Notes
    -----
    Context/frame joins check the outcome against recorded scores, deaths, and horizon.
    """

    outcome: Annotated[int, Field(ge=0, le=3)]


class TransitionFactsV1(EvaluationModel):
    """Keep normalized Core facts for either initialization or a real transition.

    Attributes
    ----------
    schema_id : Literal['marl_battlegrounds.evaluation.transition_facts']
        Fixed transition-facts identifier.
    schema_version : Literal[1]
        Version 1; the default.
    has_transition : bool
        False for initialization; true for actual transitions.
    transition_start_step_count : Annotated[int, Field(ge=-1, le=2 ** 31 - 1)]
        -1 exactly for initialization, otherwise nonnegative int32 simulator tick.
    action_acceptance_facts : ActionAcceptanceFactsV1
        Submitted/accepted actions and rejection causes.
    combat_transition_facts : CombatTransitionFactsV1
        Combat output stages, recipients, and status applications.
    death_facts : DeathTransitionFactsV1
        New deaths and direct damage attribution.
    spawn_shield_facts : SpawnShieldTransitionFactsV1
        Start shield activity and expiry.
    respawn_facts : RespawnTransitionFactsV1
        Team waves and realized respawns.
    regeneration_facts : RegenerationTransitionFactsV1
        Recovery reset and actual regenerated health.
    physical_facts : PhysicalTransitionFactsV1
        Separate Charge and ordinary displacement.
    aura_facts : AuraTransitionFactsV1
        Transition-start emitter/beneficiary coverage.
    status_lifecycle_facts : StatusLifecycleTransitionFactsV1
        Independent changes for nine status channels.
    team_deathmatch_facts : TeamDeathmatchTransitionFactsV1
        Task outcome category.

    Notes
    -----
    Core sentinel -1 for absent recipients is represented by None inside the wire combat
    record.
    """

    schema_id: Literal["marl_battlegrounds.evaluation.transition_facts"] = (
        TRANSITION_FACTS_SCHEMA_ID
    )
    schema_version: Literal[1] = TRANSITION_FACTS_SCHEMA_VERSION
    has_transition: bool
    transition_start_step_count: Annotated[int, Field(ge=-1, le=2**31 - 1)]
    action_acceptance_facts: ActionAcceptanceFactsV1
    combat_transition_facts: CombatTransitionFactsV1
    death_facts: DeathTransitionFactsV1
    spawn_shield_facts: SpawnShieldTransitionFactsV1
    respawn_facts: RespawnTransitionFactsV1
    regeneration_facts: RegenerationTransitionFactsV1
    physical_facts: PhysicalTransitionFactsV1
    aura_facts: AuraTransitionFactsV1
    status_lifecycle_facts: StatusLifecycleTransitionFactsV1
    team_deathmatch_facts: TeamDeathmatchTransitionFactsV1

    @model_validator(mode="after")
    def _validate_transition_step_sentinel(self) -> TransitionFactsV1:
        """Require start tick -1 exactly when has_transition is false.

        Return this record or raise ValueError; the owning transition rejects
        initialization.
        """
        if self.has_transition != (self.transition_start_step_count >= 0):
            raise ValueError(
                "transition facts require -1 exactly when has_transition is false"
            )
        return self


class EvaluationEventBaseV1(EvaluationModel):
    """Provide common identity for one atomic recorded event.

    Attributes
    ----------
    schema_id : Literal['marl_battlegrounds.evaluation.event']
        Fixed event identifier.
    schema_version : Literal[1]
        Version 1; the default.
    transition_id : _AsciiIdentifier
        Canonical owning transition ID.
    ordinal : _NonNegativeInt
        Nonnegative event position in that transition.
    event_id : _AsciiIdentifier
        transition_id followed by :event: and ordinal padded to at least four digits.

    Notes
    -----
    Concrete event classes supply fixed event_type and phase_rank. Ranks order
    serialized facts; they do not create policy decision points.
    """

    schema_id: Literal["marl_battlegrounds.evaluation.event"] = EVENT_SCHEMA_ID
    schema_version: Literal[1] = EVENT_SCHEMA_VERSION
    transition_id: _AsciiIdentifier
    ordinal: _NonNegativeInt
    event_id: _AsciiIdentifier

    @model_validator(mode="after")
    def _validate_event_identity(self) -> EvaluationEventBaseV1:
        """Require event_id to match transition_id and zero-padded ordinal.

        Return this event or raise ValueError. Containing transitions check ordinal
        order.
        """
        expected_id = f"{self.transition_id}:event:{self.ordinal:04d}"
        if self.event_id != expected_id:
            raise ValueError("event_id must be derived from transition_id and ordinal")
        return self


class ActionRejectedEventV1(EvaluationEventBaseV1):
    """Record one independently rejected action component.

    Attributes
    ----------
    event_type : Literal['action_rejected']
        Fixed "action_rejected"; the default.
    phase_rank : Literal[10]
        Fixed ordering rank 10; the default.
    actor_global_slot : _GlobalSlot
        Actor slot 0-9.
    rejection_component : Literal['domain', 'movement', 'combat_pair']
        domain, movement, or combat_pair.
    submitted_move_action : _Int32
        Original int32-range movement submission.
    submitted_select_target_action : _Int32
        Original int32-range target submission.
    submitted_use_ultimate_action : _Int32
        Original int32-range Ultimate-use submission.

    Notes
    -----
    Inherits event identity from EvaluationEventBaseV1. Local model validity does not
    replace canonical decoding from joined facts.
    """

    event_type: Literal["action_rejected"] = "action_rejected"
    phase_rank: Literal[10] = 10
    actor_global_slot: _GlobalSlot
    rejection_component: Literal["domain", "movement", "combat_pair"]
    submitted_move_action: _Int32
    submitted_select_target_action: _Int32
    submitted_use_ultimate_action: _Int32


class AbilityActivatedEventV1(EvaluationEventBaseV1):
    """Record one accepted Basic or Ultimate activation.

    Attributes
    ----------
    event_type : Literal['ability_activated']
        Fixed "ability_activated"; the default.
    phase_rank : Literal[20]
        Fixed ordering rank 20; the default.
    source_global_slot : _GlobalSlot
        Source slot 0-9.
    recipient_global_slot : _GlobalSlot | None
        Recipient slot 0-9 or None when no target applies.
    ability_component : Literal['basic', 'ultimate']
        basic or ultimate.

    Notes
    -----
    Inherits event identity from EvaluationEventBaseV1. Local model validity does not
    replace canonical decoding from joined facts.
    """

    event_type: Literal["ability_activated"] = "ability_activated"
    phase_rank: Literal[20] = 20
    source_global_slot: _GlobalSlot
    ability_component: Literal["basic", "ultimate"]
    recipient_global_slot: _GlobalSlot | None


class SourceDamageOutputEventV1(EvaluationEventBaseV1):
    """Record one source's damage stages and direct aura coverage.

    Attributes
    ----------
    event_type : Literal['source_damage_output']
        Fixed "source_damage_output"; the default.
    phase_rank : Literal[30]
        Fixed ordering rank 30; the default.
    source_global_slot : _GlobalSlot
        Source slot 0-9.
    recipient_global_slot : _GlobalSlot | None
        Recipient slot 0-9 or None when no target applies.
    raw_damage_output : _NonNegativeFloat
        Nonnegative raw damage in hit points.
    source_modified_damage_output : _NonNegativeFloat
        Nonnegative gross damage after source modifiers.
    recipient_damage_modifier : _NonNegativeFloat
        Nonnegative routed recipient multiplier.
    mage_damage_aura_covering_emitter_global_slots : tuple[_GlobalSlot, ...]
        Sorted unique global Mage emitter slots covering the source.
    warrior_mitigation_aura_covering_emitter_global_slots : tuple[_GlobalSlot, ...]
        Sorted unique global Warrior emitter slots covering the recipient.

    Notes
    -----
    Inherits event identity from EvaluationEventBaseV1. Local model validity does not
    replace canonical decoding from joined facts.
    """

    event_type: Literal["source_damage_output"] = "source_damage_output"
    phase_rank: Literal[30] = 30
    source_global_slot: _GlobalSlot
    recipient_global_slot: _GlobalSlot | None
    raw_damage_output: _NonNegativeFloat
    source_modified_damage_output: _NonNegativeFloat
    recipient_damage_modifier: _NonNegativeFloat
    mage_damage_aura_covering_emitter_global_slots: tuple[_GlobalSlot, ...]
    warrior_mitigation_aura_covering_emitter_global_slots: tuple[_GlobalSlot, ...]

    @model_validator(mode="after")
    def _validate_covering_emitters(self) -> SourceDamageOutputEventV1:
        """Require each emitter tuple sorted, unique, and no longer than ten slots.

        Return this event or raise ValueError; coverage itself comes from recorded
        facts.
        """
        for field_name in (
            "mage_damage_aura_covering_emitter_global_slots",
            "warrior_mitigation_aura_covering_emitter_global_slots",
        ):
            emitters = getattr(self, field_name)
            if len(emitters) > MAX_AGENT_SLOTS:
                raise ValueError(f"{field_name} cannot exceed ten fixed slots")
            if emitters != tuple(sorted(set(emitters))):
                raise ValueError(f"{field_name} must be sorted and unique")
        return self


class SourceHealingOutputEventV1(EvaluationEventBaseV1):
    """Record one source's healing stages.

    Attributes
    ----------
    event_type : Literal['source_healing_output']
        Fixed "source_healing_output"; the default.
    phase_rank : Literal[30]
        Fixed ordering rank 30; the default.
    source_global_slot : _GlobalSlot
        Source slot 0-9.
    recipient_global_slot : _GlobalSlot | None
        Recipient slot 0-9 or None when no target applies.
    raw_healing_output : _NonNegativeFloat
        Nonnegative raw healing in hit points.
    source_modified_healing_output : _NonNegativeFloat
        Nonnegative gross healing after source modifiers.
    recipient_healing_modifier : _NonNegativeFloat
        Nonnegative routed recipient healing multiplier.

    Notes
    -----
    Inherits event identity from EvaluationEventBaseV1. Local model validity does not
    replace canonical decoding from joined facts.
    """

    event_type: Literal["source_healing_output"] = "source_healing_output"
    phase_rank: Literal[30] = 30
    source_global_slot: _GlobalSlot
    recipient_global_slot: _GlobalSlot | None
    raw_healing_output: _NonNegativeFloat
    source_modified_healing_output: _NonNegativeFloat
    recipient_healing_modifier: _NonNegativeFloat


class RecipientHealthResolutionEventV1(EvaluationEventBaseV1):
    """Record one recipient's simultaneous combat health result.

    Attributes
    ----------
    event_type : Literal['recipient_health_resolution']
        Fixed "recipient_health_resolution"; the default.
    phase_rank : Literal[40]
        Fixed ordering rank 40; the default.
    recipient_global_slot : _GlobalSlot
        Recipient slot 0-9.
    transition_start_health : _NonNegativeFloat
        Nonnegative initial hit points.
    total_effective_damage : _NonNegativeFloat
        Nonnegative combined damage in hit points.
    total_effective_healing : _NonNegativeFloat
        Nonnegative combined healing in hit points.
    health_after_combat_resolution : _NonNegativeFloat
        Nonnegative hit points before later regeneration.
    realized_net_health_change : _FiniteFloat
        Finite post-combat minus start health; may be negative.

    Notes
    -----
    Inherits event identity from EvaluationEventBaseV1. Local model validity does not
    replace canonical decoding from joined facts.
    """

    event_type: Literal["recipient_health_resolution"] = "recipient_health_resolution"
    phase_rank: Literal[40] = 40
    recipient_global_slot: _GlobalSlot
    transition_start_health: _NonNegativeFloat
    total_effective_damage: _NonNegativeFloat
    total_effective_healing: _NonNegativeFloat
    health_after_combat_resolution: _NonNegativeFloat
    realized_net_health_change: _FiniteFloat


class CombatCountdownResetEventV1(EvaluationEventBaseV1):
    """Record an out-of-combat countdown reset.

    Attributes
    ----------
    event_type : Literal['combat_countdown_reset']
        Fixed "combat_countdown_reset"; the default.
    phase_rank : Literal[50]
        Fixed ordering rank 50; the default.
    agent_global_slot : _GlobalSlot
        Affected slot 0-9.

    Notes
    -----
    Inherits event identity from EvaluationEventBaseV1. Local model validity does not
    replace canonical decoding from joined facts.
    """

    event_type: Literal["combat_countdown_reset"] = "combat_countdown_reset"
    phase_rank: Literal[50] = 50
    agent_global_slot: _GlobalSlot


class AgentLeftCombatEventV1(EvaluationEventBaseV1):
    """Record a living agent's countdown change from one to zero.

    Attributes
    ----------
    event_type : Literal['agent_left_combat']
        Fixed "agent_left_combat"; the default.
    phase_rank : Literal[50]
        Fixed ordering rank 50; the default.
    agent_global_slot : _GlobalSlot
        Affected slot 0-9.

    Notes
    -----
    Inherits event identity from EvaluationEventBaseV1. Local model validity does not
    replace canonical decoding from joined facts.
    """

    event_type: Literal["agent_left_combat"] = "agent_left_combat"
    phase_rank: Literal[50] = 50
    agent_global_slot: _GlobalSlot


class HealthRegeneratedEventV1(EvaluationEventBaseV1):
    """Record actual out-of-combat health restoration.

    Attributes
    ----------
    event_type : Literal['health_regenerated']
        Fixed "health_regenerated"; the default.
    phase_rank : Literal[50]
        Fixed ordering rank 50; the default.
    agent_global_slot : _GlobalSlot
        Affected slot 0-9.
    actual_health_regenerated : _NonNegativeFloat
        Nonnegative restored hit points; canonical decoding emits positive amounts.

    Notes
    -----
    Inherits event identity from EvaluationEventBaseV1. Local model validity does not
    replace canonical decoding from joined facts.
    """

    event_type: Literal["health_regenerated"] = "health_regenerated"
    phase_rank: Literal[50] = 50
    agent_global_slot: _GlobalSlot
    actual_health_regenerated: _NonNegativeFloat


class CooldownStartedEventV1(EvaluationEventBaseV1):
    """Record an accepted Ultimate cooldown start.

    Attributes
    ----------
    event_type : Literal['cooldown_started']
        Fixed "cooldown_started"; the default.
    phase_rank : Literal[60]
        Fixed ordering rank 60; the default.
    agent_global_slot : _GlobalSlot
        Acting slot 0-9.

    Notes
    -----
    Inherits event identity from EvaluationEventBaseV1. Local model validity does not
    replace canonical decoding from joined facts.
    """

    event_type: Literal["cooldown_started"] = "cooldown_started"
    phase_rank: Literal[60] = 60
    agent_global_slot: _GlobalSlot


class CooldownReadyEventV1(EvaluationEventBaseV1):
    """Record a positive-to-zero Ultimate cooldown change.

    Attributes
    ----------
    event_type : Literal['cooldown_ready']
        Fixed "cooldown_ready"; the default.
    phase_rank : Literal[60]
        Fixed ordering rank 60; the default.
    agent_global_slot : _GlobalSlot
        Affected slot 0-9.

    Notes
    -----
    Inherits event identity from EvaluationEventBaseV1. Local model validity does not
    replace canonical decoding from joined facts.
    """

    event_type: Literal["cooldown_ready"] = "cooldown_ready"
    phase_rank: Literal[60] = 60
    agent_global_slot: _GlobalSlot


class ChargePhaseDisplacementEventV1(EvaluationEventBaseV1):
    """Record a realized Charge displacement.

    Attributes
    ----------
    event_type : Literal['charge_phase_displacement']
        Fixed "charge_phase_displacement"; the default.
    phase_rank : Literal[70]
        Fixed ordering rank 70; the default.
    agent_global_slot : _GlobalSlot
        Moved slot 0-9.
    realized_displacement : tuple[_FiniteFloat, _FiniteFloat]
        Finite (dx, dy) in world units; canonical decoding omits zero vectors.

    Notes
    -----
    Inherits event identity from EvaluationEventBaseV1. Local model validity does not
    replace canonical decoding from joined facts.
    """

    event_type: Literal["charge_phase_displacement"] = "charge_phase_displacement"
    phase_rank: Literal[70] = 70
    agent_global_slot: _GlobalSlot
    realized_displacement: tuple[_FiniteFloat, _FiniteFloat]


class OrdinaryMovementPhaseDisplacementEventV1(EvaluationEventBaseV1):
    """Record a realized ordinary-movement displacement.

    Attributes
    ----------
    event_type : Literal['ordinary_movement_phase_displacement']
        Fixed "ordinary_movement_phase_displacement"; the default.
    phase_rank : Literal[80]
        Fixed ordering rank 80; the default.
    agent_global_slot : _GlobalSlot
        Moved slot 0-9.
    realized_displacement : tuple[_FiniteFloat, _FiniteFloat]
        Finite (dx, dy) in world units; canonical decoding omits zero vectors.

    Notes
    -----
    Inherits event identity from EvaluationEventBaseV1. Local model validity does not
    replace canonical decoding from joined facts.
    """

    event_type: Literal["ordinary_movement_phase_displacement"] = (
        "ordinary_movement_phase_displacement"
    )
    phase_rank: Literal[80] = 80
    agent_global_slot: _GlobalSlot
    realized_displacement: tuple[_FiniteFloat, _FiniteFloat]


class AgentDiedEventV1(EvaluationEventBaseV1):
    """Record a newly dead recipient.

    Attributes
    ----------
    event_type : Literal['agent_died']
        Fixed "agent_died"; the default.
    phase_rank : Literal[90]
        Fixed ordering rank 90; the default.
    recipient_global_slot : _GlobalSlot
        Newly dead slot 0-9.

    Notes
    -----
    Inherits event identity from EvaluationEventBaseV1. Local model validity does not
    replace canonical decoding from joined facts.
    """

    event_type: Literal["agent_died"] = "agent_died"
    phase_rank: Literal[90] = 90
    recipient_global_slot: _GlobalSlot


class LethalDamageContributionEventV1(EvaluationEventBaseV1):
    """Record one direct damage contribution to a new death.

    Attributes
    ----------
    event_type : Literal['lethal_damage_contribution']
        Fixed "lethal_damage_contribution"; the default.
    phase_rank : Literal[90]
        Fixed ordering rank 90; the default.
    source_global_slot : _GlobalSlot
        Contributing slot 0-9.
    recipient_global_slot : _GlobalSlot
        Newly dead slot 0-9.
    attributed_death_damage : _NonNegativeFloat
        Nonnegative gross hit-point contribution; canonical decoding emits positive
        values.

    Notes
    -----
    Inherits event identity from EvaluationEventBaseV1. Local model validity does not
    replace canonical decoding from joined facts.
    """

    event_type: Literal["lethal_damage_contribution"] = "lethal_damage_contribution"
    phase_rank: Literal[90] = 90
    source_global_slot: _GlobalSlot
    recipient_global_slot: _GlobalSlot
    attributed_death_damage: _NonNegativeFloat


class StatusLifecycleEventBaseV1(EvaluationEventBaseV1):
    """Add recipient and channel identity to a status lifecycle event.

    Attributes
    ----------
    recipient_global_slot : _GlobalSlot
        Affected global slot 0-9.
    status_channel : _StatusChannel
        Combined channel 0-8.
    status_id : _AsciiIdentifier
        Recorded catalog status identifier for that channel.

    Notes
    -----
    Inherits transition/ordinal/event identity. Events without source_global_slot do not
    invent a caster for persistent status.
    """

    recipient_global_slot: _GlobalSlot
    status_channel: _StatusChannel
    status_id: _AsciiIdentifier


class StatusAgedToZeroEventV1(StatusLifecycleEventBaseV1):
    """Record ordinary duration aging to zero.

    Attributes
    ----------
    event_type : Literal['status_aged_to_zero']
        Fixed "status_aged_to_zero"; the default.
    phase_rank : Literal[100]
        Fixed ordering rank 100; the default.

    Notes
    -----
    Inherits recipient/channel fields from StatusLifecycleEventBaseV1 and event identity
    from EvaluationEventBaseV1. Local model validity does not replace canonical decoding
    from joined facts.
    """

    event_type: Literal["status_aged_to_zero"] = "status_aged_to_zero"
    phase_rank: Literal[100] = 100


class StatusBrokenByDamageEventV1(StatusLifecycleEventBaseV1):
    """Record removal of a damage-breakable status.

    Attributes
    ----------
    event_type : Literal['status_broken_by_damage']
        Fixed "status_broken_by_damage"; the default.
    phase_rank : Literal[100]
        Fixed ordering rank 100; the default.

    Notes
    -----
    Inherits recipient/channel fields from StatusLifecycleEventBaseV1 and event identity
    from EvaluationEventBaseV1. Local model validity does not replace canonical decoding
    from joined facts.
    """

    event_type: Literal["status_broken_by_damage"] = "status_broken_by_damage"
    phase_rank: Literal[100] = 100


class StatusAppliedEventV1(StatusLifecycleEventBaseV1):
    """Record one attributed status application.

    Attributes
    ----------
    event_type : Literal['status_applied']
        Fixed "status_applied"; the default.
    phase_rank : Literal[100]
        Fixed ordering rank 100; the default.
    source_global_slot : _GlobalSlot
        Accepted applying source slot 0-9.

    Notes
    -----
    Inherits recipient/channel fields from StatusLifecycleEventBaseV1 and event identity
    from EvaluationEventBaseV1. Local model validity does not replace canonical decoding
    from joined facts.
    """

    event_type: Literal["status_applied"] = "status_applied"
    phase_rank: Literal[100] = 100
    source_global_slot: _GlobalSlot


class StatusRefreshedOrExtendedEventV1(StatusLifecycleEventBaseV1):
    """Record an accepted application that increased remaining duration.

    Attributes
    ----------
    event_type : Literal['status_refreshed_or_extended']
        Fixed "status_refreshed_or_extended"; the default.
    phase_rank : Literal[100]
        Fixed ordering rank 100; the default.

    Notes
    -----
    Inherits recipient/channel fields from StatusLifecycleEventBaseV1 and event identity
    from EvaluationEventBaseV1. Local model validity does not replace canonical decoding
    from joined facts.
    """

    event_type: Literal["status_refreshed_or_extended"] = "status_refreshed_or_extended"
    phase_rank: Literal[100] = 100


class StatusClearedByNewDeathEventV1(StatusLifecycleEventBaseV1):
    """Record status clearing when a recipient newly dies.

    Attributes
    ----------
    event_type : Literal['status_cleared_by_new_death']
        Fixed "status_cleared_by_new_death"; the default.
    phase_rank : Literal[100]
        Fixed ordering rank 100; the default.

    Notes
    -----
    Inherits recipient/channel fields from StatusLifecycleEventBaseV1 and event identity
    from EvaluationEventBaseV1. Local model validity does not replace canonical decoding
    from joined facts.
    """

    event_type: Literal["status_cleared_by_new_death"] = "status_cleared_by_new_death"
    phase_rank: Literal[100] = 100


class SpawnShieldExpiredEventV1(EvaluationEventBaseV1):
    """Record ordinary spawn-shield expiry.

    Attributes
    ----------
    event_type : Literal['spawn_shield_expired']
        Fixed "spawn_shield_expired"; the default.
    phase_rank : Literal[110]
        Fixed ordering rank 110; the default.
    agent_global_slot : _GlobalSlot
        Affected slot 0-9.

    Notes
    -----
    Inherits event identity from EvaluationEventBaseV1. Local model validity does not
    replace canonical decoding from joined facts.
    """

    event_type: Literal["spawn_shield_expired"] = "spawn_shield_expired"
    phase_rank: Literal[110] = 110
    agent_global_slot: _GlobalSlot


class RespawnWaveOccurredEventV1(EvaluationEventBaseV1):
    """Record a due team wave, even if nobody respawns.

    Attributes
    ----------
    event_type : Literal['respawn_wave_occurred']
        Fixed "respawn_wave_occurred"; the default.
    phase_rank : Literal[120]
        Fixed ordering rank 120; the default.
    team_index : Literal[0, 1]
        0 for Team A or 1 for Team B.
    team_id : Literal[1, 2]
        team_index plus one.

    Notes
    -----
    Inherits event identity from EvaluationEventBaseV1. Local model validity does not
    replace canonical decoding from joined facts.
    """

    event_type: Literal["respawn_wave_occurred"] = "respawn_wave_occurred"
    phase_rank: Literal[120] = 120
    team_index: Literal[0, 1]
    team_id: Literal[1, 2]

    @model_validator(mode="after")
    def _validate_team_join(self) -> RespawnWaveOccurredEventV1:
        """Require one-based team_id to equal zero-based team_index plus one."""
        if self.team_id != self.team_index + 1:
            raise ValueError("team_id must match the zero-based team_index")
        return self


class AgentRespawnedEventV1(EvaluationEventBaseV1):
    """Record a realized agent respawn and successor position.

    Attributes
    ----------
    event_type : Literal['agent_respawned']
        Fixed "agent_respawned"; the default.
    phase_rank : Literal[120]
        Fixed ordering rank 120; the default.
    agent_global_slot : _GlobalSlot
        Respawned slot 0-9.
    team_id : Literal[1, 2]
        1 for Team A or 2 for Team B.
    realized_successor_position : tuple[_FiniteFloat, _FiniteFloat]
        Finite (x, y) world-unit position after respawn.

    Notes
    -----
    Inherits event identity from EvaluationEventBaseV1. Local model validity does not
    replace canonical decoding from joined facts.
    """

    event_type: Literal["agent_respawned"] = "agent_respawned"
    phase_rank: Literal[120] = 120
    agent_global_slot: _GlobalSlot
    team_id: Literal[1, 2]
    realized_successor_position: tuple[_FiniteFloat, _FiniteFloat]


class TeamDeathmatchScoreChangedEventV1(EvaluationEventBaseV1):
    """Record one team's positive score change.

    Attributes
    ----------
    event_type : Literal['team_deathmatch_score_changed']
        Fixed "team_deathmatch_score_changed"; the default.
    phase_rank : Literal[130]
        Fixed ordering rank 130; the default.
    team_index : Literal[0, 1]
        0 for Team A or 1 for Team B.
    team_id : Literal[1, 2]
        team_index plus one.
    score_increment : _PositiveInt
        Positive point gain during the transition.
    previous_score : _NonNegativeInt
        Nonnegative pre-transition score.
    successor_score : _NonNegativeInt
        previous_score plus score_increment.

    Notes
    -----
    Inherits event identity from EvaluationEventBaseV1. Local model validity does not
    replace canonical decoding from joined facts.
    """

    event_type: Literal["team_deathmatch_score_changed"] = (
        "team_deathmatch_score_changed"
    )
    phase_rank: Literal[130] = 130
    team_index: Literal[0, 1]
    team_id: Literal[1, 2]
    score_increment: _PositiveInt
    previous_score: _NonNegativeInt
    successor_score: _NonNegativeInt

    @model_validator(mode="after")
    def _validate_score_edge(self) -> TeamDeathmatchScoreChangedEventV1:
        """Check team-index mapping and exact positive score increment.

        Return this event or raise ValueError on a mismatched successor score.
        """
        if self.team_id != self.team_index + 1:
            raise ValueError("team_id must match the zero-based team_index")
        if self.successor_score != self.previous_score + self.score_increment:
            raise ValueError(
                "successor_score must equal previous_score plus score_increment"
            )
        return self


class TeamDeathmatchCompletedEventV1(EvaluationEventBaseV1):
    """Record a TDM result and its exact completion basis.

    Attributes
    ----------
    event_type : Literal['team_deathmatch_completed']
        Fixed "team_deathmatch_completed"; the default.
    phase_rank : Literal[140]
        Fixed ordering rank 140; the default.
    outcome : Literal['team_a_win', 'team_b_win', 'draw']
        team_a_win, team_b_win, or draw.
    completion_basis : Literal['score_threshold', 'horizon',
    'score_threshold_at_horizon']
        score_threshold, horizon, or score_threshold_at_horizon.

    Notes
    -----
    Inherits event identity from EvaluationEventBaseV1. Local model validity does not
    replace canonical decoding from joined facts.
    """

    event_type: Literal["team_deathmatch_completed"] = "team_deathmatch_completed"
    phase_rank: Literal[140] = 140
    outcome: Literal["team_a_win", "team_b_win", "draw"]
    completion_basis: Literal[
        "score_threshold",
        "horizon",
        "score_threshold_at_horizon",
    ]


type EvaluationEventV1 = Annotated[
    ActionRejectedEventV1
    | AbilityActivatedEventV1
    | SourceDamageOutputEventV1
    | SourceHealingOutputEventV1
    | RecipientHealthResolutionEventV1
    | CombatCountdownResetEventV1
    | AgentLeftCombatEventV1
    | HealthRegeneratedEventV1
    | CooldownStartedEventV1
    | CooldownReadyEventV1
    | ChargePhaseDisplacementEventV1
    | OrdinaryMovementPhaseDisplacementEventV1
    | AgentDiedEventV1
    | LethalDamageContributionEventV1
    | StatusAgedToZeroEventV1
    | StatusBrokenByDamageEventV1
    | StatusAppliedEventV1
    | StatusRefreshedOrExtendedEventV1
    | StatusClearedByNewDeathEventV1
    | SpawnShieldExpiredEventV1
    | RespawnWaveOccurredEventV1
    | AgentRespawnedEventV1
    | TeamDeathmatchScoreChangedEventV1
    | TeamDeathmatchCompletedEventV1,
    Field(discriminator="event_type"),
]


class EvaluationTransitionV1(EvaluationModel):
    """Join adjacent captured frames with normalized facts, events, rewards, and done
    flags.

    Attributes
    ----------
    schema_id : Literal['marl_battlegrounds.evaluation.transition']
        Fixed transition identifier.
    schema_version : Literal[1]
        Version 1; the default.
    episode_id : _AsciiIdentifier
        Owning episode ID.
    transition_index : _NonNegativeInt
        Nonnegative artifact coordinate equal to the start-frame index.
    transition_id : _AsciiIdentifier
        Canonical episode ID followed by :transition: and transition_index.
    start_frame_id : _AsciiIdentifier
        Canonical frame ID at transition_index.
    successor_frame_id : _AsciiIdentifier
        Canonical frame ID at transition_index plus one.
    facts : TransitionFactsV1
        Actual transition facts; initialization is forbidden.
    events : tuple[EvaluationEventV1, ...]
        Ordered atomic events with gap-free ordinals joined to this transition.
    canonical_reward_by_agent : _FloatVector
        Ten finite rewards in raw global-slot order.
    canonical_reward_by_team : _FloatVector | None
        Optional two-team rewards, Team A then Team B; defaults to None.
    terminated : bool
        Task-termination flag from this transition.
    truncated : bool
        Truncation flag from this transition, independent of terminated.
    owning_task_end_reason : _AsciiText | None
        Optional printable task end reason, allowed only when done; defaults to None.

    Notes
    -----
    Local validation checks IDs, shapes, actual-transition marker, and event joins.
    Cross-record validation checks simulator epochs, task reward/result authority, and
    the complete canonical event sequence.
    """

    schema_id: Literal["marl_battlegrounds.evaluation.transition"] = (
        TRANSITION_SCHEMA_ID
    )
    schema_version: Literal[1] = TRANSITION_SCHEMA_VERSION
    episode_id: _AsciiIdentifier
    transition_index: _NonNegativeInt
    transition_id: _AsciiIdentifier
    start_frame_id: _AsciiIdentifier
    successor_frame_id: _AsciiIdentifier
    facts: TransitionFactsV1
    events: tuple[EvaluationEventV1, ...]
    canonical_reward_by_agent: _FloatVector
    canonical_reward_by_team: _FloatVector | None = None
    terminated: bool
    truncated: bool
    owning_task_end_reason: _AsciiText | None = None

    @model_validator(mode="after")
    def _validate_transition(self) -> EvaluationTransitionV1:
        """Check canonical adjacent IDs, actual facts, reward axes, done reason, and
        event order.

        Return this transition or raise ValueError; joined frame/task semantics are
        checked separately.
        """
        expected_transition_id = f"{self.episode_id}:transition:{self.transition_index}"
        if self.transition_id != expected_transition_id:
            raise ValueError(
                "transition_id must be derived from episode_id and transition_index"
            )
        expected_start_frame_id = f"{self.episode_id}:frame:{self.transition_index}"
        expected_successor_frame_id = (
            f"{self.episode_id}:frame:{self.transition_index + 1}"
        )
        if self.start_frame_id != expected_start_frame_id:
            raise ValueError("start_frame_id must identify the transition-start frame")
        if self.successor_frame_id != expected_successor_frame_id:
            raise ValueError("successor_frame_id must identify the adjacent frame")
        if not self.facts.has_transition:
            raise ValueError("evaluation transitions require has_transition facts")
        _require_tuple_shape(
            self.canonical_reward_by_agent,
            (MAX_AGENT_SLOTS,),
            field_name="canonical_reward_by_agent",
        )
        if self.canonical_reward_by_team is not None:
            _require_tuple_shape(
                self.canonical_reward_by_team,
                (NUM_TEAMS,),
                field_name="canonical_reward_by_team",
            )
        if self.owning_task_end_reason is not None and not (
            self.terminated or self.truncated
        ):
            raise ValueError("owning_task_end_reason is allowed only when done")
        for expected_ordinal, event in enumerate(self.events):
            if event.transition_id != self.transition_id:
                raise ValueError("every event must join its containing transition")
            if event.ordinal != expected_ordinal:
                raise ValueError("event ordinals must be gap-free and ordered")
        return self


__all__ = [
    "CATALOG_SCHEMA_ID",
    "CATALOG_SCHEMA_VERSION",
    "CONTEXT_SCHEMA_ID",
    "CONTEXT_SCHEMA_VERSION",
    "EVENT_SCHEMA_ID",
    "EVENT_SCHEMA_VERSION",
    "FRAME_SCHEMA_ID",
    "FRAME_SCHEMA_VERSION",
    "GLOBAL_ANALYSIS_SNAPSHOT_SCHEMA_ID",
    "GLOBAL_ANALYSIS_SNAPSHOT_SCHEMA_VERSION",
    "REQUIRED_SCHEMA_BINDINGS_V1",
    "REQUIRED_SCHEMA_BINDINGS_V2",
    "REQUIRED_SCHEMA_BINDINGS_V3",
    "RESOLVED_ENV_CONFIG_SCHEMA_ID",
    "RESOLVED_ENV_CONFIG_SCHEMA_VERSION",
    "TRANSITION_FACTS_SCHEMA_ID",
    "TRANSITION_FACTS_SCHEMA_VERSION",
    "TRANSITION_SCHEMA_ID",
    "TRANSITION_SCHEMA_VERSION",
    "AbilityActivatedEventV1",
    "ActionAcceptanceFactsV1",
    "ActionMaskV1",
    "ActionRejectedEventV1",
    "AgentDiedEventV1",
    "AgentLeftCombatEventV1",
    "AgentRespawnedEventV1",
    "AggregationKeyV1",
    "AssignedPolicySlot",
    "AssignedPolicySlotV1",
    "AssignedPolicySlotV2",
    "AuraMechanicV1",
    "AuraTransitionFactsV1",
    "BaseObservationV1",
    "BaseObservationV2",
    "CaptureProfile",
    "ChargePhaseDisplacementEventV1",
    "ClassMechanicsV1",
    "CodeRevisionV1",
    "CodeRevisionV2",
    "CombatCountdownResetEventV1",
    "CombatTransitionFactsV1",
    "ContentAddressedIdentityV1",
    "CooldownReadyEventV1",
    "CooldownStartedEventV1",
    "DeathTransitionFactsV1",
    "EvaluationEpisodeContext",
    "EvaluationEpisodeContextV1",
    "EvaluationEpisodeContextV2",
    "EvaluationEpisodeContextV3",
    "EvaluationEpisodeIdentityV1",
    "EvaluationEventBaseV1",
    "EvaluationEventV1",
    "EvaluationFrame",
    "EvaluationFrameV1",
    "EvaluationFrameV2",
    "EvaluationModel",
    "EvaluationRole",
    "EvaluationSeedProtocolV1",
    "EvaluationSeedProtocolV2",
    "EvaluationTransitionV1",
    "ExecutionInformationMode",
    "GlobalAnalysisSnapshotV1",
    "HealthRegeneratedEventV1",
    "JointActionV1",
    "LethalDamageContributionEventV1",
    "NotApplicablePolicySlotV1",
    "OrdinaryMovementPhaseDisplacementEventV1",
    "PhysicalTransitionFactsV1",
    "PolicyAssignmentSlotV1",
    "PolicyAssignmentSlotV2",
    "PreviousTimestepActionObservationV1",
    "RecipientHealthResolutionEventV1",
    "RegenerationTransitionFactsV1",
    "ResolvedEnvConfigV1",
    "ResolvedObstacleV1",
    "ResolvedSlotMechanicsV1",
    "RespawnTransitionFactsV1",
    "RespawnWaveOccurredEventV1",
    "RosterSlotV1",
    "SchemaVersionEntryV1",
    "SchemaVersionEntryV2",
    "SchemaVersionEntryV3",
    "SourceDamageOutputEventV1",
    "SourceHealingOutputEventV1",
    "SpawnLifecycleObservationV1",
    "SpawnShieldExpiredEventV1",
    "SpawnShieldTransitionFactsV1",
    "StaticMechanicsCatalogV1",
    "StatusAgedToZeroEventV1",
    "StatusAppliedEventV1",
    "StatusBrokenByDamageEventV1",
    "StatusClearedByNewDeathEventV1",
    "StatusLifecycleEventBaseV1",
    "StatusLifecycleTransitionFactsV1",
    "StatusMechanicV1",
    "StatusRefreshedOrExtendedEventV1",
    "TeamDeathmatchCompletedEventV1",
    "TeamDeathmatchScoreChangedEventV1",
    "TeamDeathmatchTransitionFactsV1",
    "TransitionFactsV1",
    "VersionedIdentityV1",
    "canonical_digest_sha256",
    "canonical_json_bytes",
    "evaluation_context_type",
    "evaluation_frame_type",
]
