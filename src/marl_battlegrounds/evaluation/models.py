"""Strict, immutable, versioned host models for evaluation records."""

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
    """Strict immutable base for every evaluation record."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
    )


def _normalize_canonical_json(value: object) -> object:
    """Recursively normalize JSON values before content addressing."""
    if isinstance(value, BaseModel):
        return _normalize_canonical_json(value.model_dump(mode="json"))
    if type(value) is float and value == 0.0:
        return 0.0
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, object], value)
        return {key: _normalize_canonical_json(item) for key, item in mapping.items()}
    if isinstance(value, (list, tuple)):
        sequence = cast(list[object] | tuple[object, ...], value)
        return [_normalize_canonical_json(item) for item in sequence]
    return value


def canonical_json_bytes(value: BaseModel | Mapping[str, object]) -> bytes:
    """Return finite compact sorted-key UTF-8 JSON with normalized zeroes."""
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
    """Return SHA-256 over the canonical JSON projection of one record."""
    excluded = exclude or set()
    if isinstance(value, BaseModel):
        payload = value.model_dump(mode="json", exclude=excluded)
    else:
        payload = {key: item for key, item in value.items() if key not in excluded}
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


class ClassMechanicsV1(EvaluationModel):
    """One stable class row in the supported mechanics catalog."""

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
    """One channel in the canonical combined status axis."""

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
        if self.magnitude_kind == "none":
            if self.magnitude is not None:
                raise ValueError("stun-like status channels must omit magnitude")
        elif self.magnitude is None or not isfinite(self.magnitude):
            raise ValueError("non-stun status channels require a finite magnitude")
        return self


class AuraMechanicV1(EvaluationModel):
    """One supported many-to-many passive aura mechanic."""

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
    """Self-describing interpretation catalog stored once per episode."""

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
    """Stable named identity whose semantics are selected by a version."""

    identifier: _AsciiIdentifier
    version: _PositiveInt


class ContentAddressedIdentityV1(VersionedIdentityV1):
    """Versioned identity backed by immutable normalized content."""

    canonical_digest: _Sha256Hex


class AggregationKeyV1(EvaluationModel):
    """One immutable evaluation stratum coordinate."""

    name: _AsciiIdentifier
    value: _AsciiText


class EvaluationEpisodeIdentityV1(EvaluationModel):
    """Runner-owned stable identity for one evaluation episode."""

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
    """One semantic row from the fixed resolved obstacle matrix."""

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
    """One fixed slot's resolved mechanics, independent of public identity."""

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
    """JSON-compatible episode configuration stored once in context."""

    schema_id: Literal["marl_battlegrounds.evaluation.resolved_env_config"] = (
        RESOLVED_ENV_CONFIG_SCHEMA_ID
    )
    schema_version: Literal[1] = RESOLVED_ENV_CONFIG_SCHEMA_VERSION
    canonical_digest_sha256: _Sha256Hex
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
    """One fixed-slot identity and topology row stored in context."""

    global_slot: _GlobalSlot
    team_local_slot: _TeamLocalSlot
    public_agent_id: _AsciiIdentifier
    configured_team_id: Annotated[int, Field(ge=0, le=2)]
    class_id: Annotated[int, Field(ge=0, le=5)]
    configured_active: bool


class AssignedPolicySlotV1(EvaluationModel):
    """Durable policy provenance for one configured active slot."""

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


class NotApplicablePolicySlotV1(EvaluationModel):
    """Minimal canonical policy row for one inactive fixed slot."""

    assignment_status: Literal["not_applicable"] = "not_applicable"
    global_slot: _GlobalSlot


type PolicyAssignmentSlotV1 = Annotated[
    AssignedPolicySlotV1 | NotApplicablePolicySlotV1,
    Field(discriminator="assignment_status"),
]


class EvaluationSeedProtocolV1(EvaluationModel):
    """Named realized seeds needed to reproduce evaluation assignment."""

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
    """Exact code/source identity, including dirty-worktree provenance."""

    package_version: _AsciiIdentifier
    commit_sha: _GitCommitHex
    source_tree_digest: _Sha256Hex
    is_dirty: bool
    dirty_patch_digest: _Sha256Hex | None = None

    @model_validator(mode="after")
    def _validate_dirty_revision(self) -> CodeRevisionV1:
        if self.is_dirty != (self.dirty_patch_digest is not None):
            raise ValueError(
                "dirty revisions require a patch digest and clean revisions forbid one"
            )
        return self


class SchemaVersionEntryV1(EvaluationModel):
    """One recognized member of the context schema-version map."""

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
    """Immutable self-describing provenance stored once per episode."""

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


def _validate_context_rows(context: EvaluationEpisodeContextV1) -> None:
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
            if not isinstance(policy_row, AssignedPolicySlotV1):
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


def _validate_context_seeds(context: EvaluationEpisodeContextV1) -> None:
    active_roles = {
        row.evaluation_role
        for row in context.policy_assignments
        if isinstance(row, AssignedPolicySlotV1)
    }
    if "focal" not in active_roles:
        raise ValueError("evaluation context requires at least one focal policy")
    seeds = context.seed_protocol
    role_seed_pairs = (
        ("cooperative_partner", seeds.cooperative_partner_seed),
        ("adversarial_opponent", seeds.adversarial_opponent_seed),
    )
    for role, seed in role_seed_pairs:
        if (role in active_roles) != (seed != "not_applicable"):
            raise ValueError(f"{role} seed presence must match policy assignments")
    if (context.identity.scenario is not None) != (
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
    """Require one already typed tuple payload to have an exact fixed shape."""
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
    """Dynamic-only privileged simulator truth for one frame."""

    schema_id: Literal["marl_battlegrounds.evaluation.global_analysis_snapshot"] = (
        GLOBAL_ANALYSIS_SNAPSHOT_SCHEMA_ID
    )
    schema_version: Literal[1] = GLOBAL_ANALYSIS_SNAPSHOT_SCHEMA_VERSION
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
    """Observer-relative one-hot accepted-action history."""

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
    """Actor-relative public spawn, shield, roster, and wave-clock truth."""

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
    """All ten fixed slots' base policy observations stored once."""

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


class ActionMaskV1(EvaluationModel):
    """Exact fixed-slot action masks with authoritative joint combat legality."""

    move_mask: _BooleanMatrix
    select_target_mask: _BooleanMatrix
    use_ultimate_mask: _BooleanMatrix
    select_target_use_ultimate_joint_mask: _BooleanTensor3

    @model_validator(mode="after")
    def _validate_action_mask(self) -> ActionMaskV1:
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
    """One stable evaluation epoch and its same-epoch policy input material."""

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
        expected_id = f"{self.episode_id}:frame:{self.frame_index}"
        if self.frame_id != expected_id:
            raise ValueError("frame_id must be derived from episode_id and frame_index")
        availability = (
            self.shared_obs_information_availability_by_recipient_and_sensor_source
        )
        if availability is not None:
            _require_tuple_shape(
                availability,
                (MAX_AGENT_SLOTS, MAX_AGENT_SLOTS),
                field_name=(
                    "shared_obs_information_availability_by_recipient_and_sensor_source"
                ),
            )
        return self


class JointActionV1(EvaluationModel):
    """One submitted or accepted factored action for all fixed actor slots."""

    move: tuple[_Int32, ...]
    select_target: tuple[_Int32, ...]
    use_ultimate: tuple[_Int32, ...]

    @model_validator(mode="after")
    def _validate_action_shapes(self) -> JointActionV1:
        for field_name in ("move", "select_target", "use_ultimate"):
            _require_tuple_shape(
                getattr(self, field_name),
                (MAX_AGENT_SLOTS,),
                field_name=field_name,
            )
        return self


class ActionAcceptanceFactsV1(EvaluationModel):
    """Submitted intent, accepted action, and rejection provenance."""

    submitted_joint_action: JointActionV1
    accepted_joint_action: JointActionV1
    submitted_action_tuple_is_out_of_domain_by_actor: _BooleanVector
    in_domain_move_action_is_rejected_by_actor: _BooleanVector
    in_domain_combat_action_pair_is_rejected_by_actor: _BooleanVector

    @model_validator(mode="after")
    def _validate_action_acceptance(self) -> ActionAcceptanceFactsV1:
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
    """Source-aligned combat outputs and authoritative recipient totals."""

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
    """New deaths and their positive-damage source attribution facts."""

    is_newly_dead_by_recipient: _BooleanVector
    contributed_to_new_death_by_source: _BooleanVector
    attributed_death_damage_by_source: _NonNegativeFloatVector

    @model_validator(mode="after")
    def _validate_death_shapes(self) -> DeathTransitionFactsV1:
        for field_name in self.__class__.model_fields:
            _require_tuple_shape(
                getattr(self, field_name),
                (MAX_AGENT_SLOTS,),
                field_name=field_name,
            )
        return self


class SpawnShieldTransitionFactsV1(EvaluationModel):
    """Spawn-shield activity and ordinary expiry facts."""

    was_active_at_transition_start_by_agent: _BooleanVector
    expired_at_transition_end_by_agent: _BooleanVector

    @model_validator(mode="after")
    def _validate_spawn_shield_shapes(self) -> SpawnShieldTransitionFactsV1:
        for field_name in self.__class__.model_fields:
            _require_tuple_shape(
                getattr(self, field_name),
                (MAX_AGENT_SLOTS,),
                field_name=field_name,
            )
        return self


class RespawnTransitionFactsV1(EvaluationModel):
    """Team-wave and realized agent-respawn facts."""

    respawn_wave_occurred_this_transition_by_team: _BooleanVector
    was_respawned_this_transition_by_agent: _BooleanVector

    @model_validator(mode="after")
    def _validate_respawn_shapes(self) -> RespawnTransitionFactsV1:
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
    """Combat-countdown reset and actual regeneration facts."""

    combat_countdown_was_reset_by_agent: _BooleanVector
    actual_health_regenerated_this_step_by_agent: _NonNegativeFloatVector

    @model_validator(mode="after")
    def _validate_regeneration_shapes(self) -> RegenerationTransitionFactsV1:
        for field_name in self.__class__.model_fields:
            _require_tuple_shape(
                getattr(self, field_name),
                (MAX_AGENT_SLOTS,),
                field_name=field_name,
            )
        return self


class PhysicalTransitionFactsV1(EvaluationModel):
    """Realized displacement authored separately by each movement phase."""

    charge_phase_displacement_by_agent: _FloatMatrix
    ordinary_movement_phase_displacement_by_agent: _FloatMatrix

    @model_validator(mode="after")
    def _validate_displacement_shapes(self) -> PhysicalTransitionFactsV1:
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
    """Transition-start emitter-by-beneficiary aura coverage facts."""

    is_covered_by_mage_damage_aura_by_emitter_and_beneficiary: _BooleanMatrix
    is_covered_by_warrior_mitigation_aura_by_emitter_and_beneficiary: _BooleanMatrix

    @model_validator(mode="after")
    def _validate_aura_shapes(self) -> AuraTransitionFactsV1:
        for field_name in self.__class__.model_fields:
            _require_tuple_shape(
                getattr(self, field_name),
                (MAX_AGENT_SLOTS, MAX_AGENT_SLOTS),
                field_name=field_name,
            )
        return self


class StatusLifecycleTransitionFactsV1(EvaluationModel):
    """Independent recipient-aligned lifecycle facts for nine status channels."""

    aged_to_zero_by_recipient_and_status_channel: _BooleanMatrix
    refreshed_or_extended_by_recipient_and_status_channel: _BooleanMatrix
    broken_by_damage_by_recipient_and_status_channel: _BooleanMatrix
    cleared_by_new_death_by_recipient_and_status_channel: _BooleanMatrix

    @model_validator(mode="after")
    def _validate_status_lifecycle_shapes(self) -> StatusLifecycleTransitionFactsV1:
        for field_name in self.__class__.model_fields:
            _require_tuple_shape(
                getattr(self, field_name),
                (MAX_AGENT_SLOTS, 9),
                field_name=field_name,
            )
        return self


class TransitionFactsV1(EvaluationModel):
    """Lossless normalized facts for a real transition or initialization."""

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

    @model_validator(mode="after")
    def _validate_transition_step_sentinel(self) -> TransitionFactsV1:
        if self.has_transition != (self.transition_start_step_count >= 0):
            raise ValueError(
                "transition facts require -1 exactly when has_transition is false"
            )
        return self


class EvaluationEventBaseV1(EvaluationModel):
    """Common identity, order, and phase fields for one atomic event."""

    schema_id: Literal["marl_battlegrounds.evaluation.event"] = EVENT_SCHEMA_ID
    schema_version: Literal[1] = EVENT_SCHEMA_VERSION
    transition_id: _AsciiIdentifier
    ordinal: _NonNegativeInt
    event_id: _AsciiIdentifier

    @model_validator(mode="after")
    def _validate_event_identity(self) -> EvaluationEventBaseV1:
        expected_id = f"{self.transition_id}:event:{self.ordinal:04d}"
        if self.event_id != expected_id:
            raise ValueError("event_id must be derived from transition_id and ordinal")
        return self


class ActionRejectedEventV1(EvaluationEventBaseV1):
    """One independently rejected action component."""

    event_type: Literal["action_rejected"] = "action_rejected"
    phase_rank: Literal[10] = 10
    actor_global_slot: _GlobalSlot
    rejection_component: Literal["domain", "movement", "combat_pair"]
    submitted_move_action: _Int32
    submitted_select_target_action: _Int32
    submitted_use_ultimate_action: _Int32


class AbilityActivatedEventV1(EvaluationEventBaseV1):
    """One accepted Basic or Ultimate activation."""

    event_type: Literal["ability_activated"] = "ability_activated"
    phase_rank: Literal[20] = 20
    source_global_slot: _GlobalSlot
    ability_component: Literal["basic", "ultimate"]
    recipient_global_slot: _GlobalSlot | None


class SourceDamageOutputEventV1(EvaluationEventBaseV1):
    """One source-authored damage-output fact with direct aura coverage."""

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
    """One source-authored healing-output fact."""

    event_type: Literal["source_healing_output"] = "source_healing_output"
    phase_rank: Literal[30] = 30
    source_global_slot: _GlobalSlot
    recipient_global_slot: _GlobalSlot | None
    raw_healing_output: _NonNegativeFloat
    source_modified_healing_output: _NonNegativeFloat
    recipient_healing_modifier: _NonNegativeFloat


class RecipientHealthResolutionEventV1(EvaluationEventBaseV1):
    """One authoritative recipient-level simultaneous health resolution."""

    event_type: Literal["recipient_health_resolution"] = "recipient_health_resolution"
    phase_rank: Literal[40] = 40
    recipient_global_slot: _GlobalSlot
    transition_start_health: _NonNegativeFloat
    total_effective_damage: _NonNegativeFloat
    total_effective_healing: _NonNegativeFloat
    health_after_combat_resolution: _NonNegativeFloat
    realized_net_health_change: _FiniteFloat


class CombatCountdownResetEventV1(EvaluationEventBaseV1):
    """One authoritative out-of-combat countdown reset."""

    event_type: Literal["combat_countdown_reset"] = "combat_countdown_reset"
    phase_rank: Literal[50] = 50
    agent_global_slot: _GlobalSlot


class AgentLeftCombatEventV1(EvaluationEventBaseV1):
    """One alive agent whose out-of-combat countdown reached zero."""

    event_type: Literal["agent_left_combat"] = "agent_left_combat"
    phase_rank: Literal[50] = 50
    agent_global_slot: _GlobalSlot


class HealthRegeneratedEventV1(EvaluationEventBaseV1):
    """One realized positive out-of-combat health regeneration."""

    event_type: Literal["health_regenerated"] = "health_regenerated"
    phase_rank: Literal[50] = 50
    agent_global_slot: _GlobalSlot
    actual_health_regenerated: _NonNegativeFloat


class CooldownStartedEventV1(EvaluationEventBaseV1):
    """One accepted Ultimate cooldown start."""

    event_type: Literal["cooldown_started"] = "cooldown_started"
    phase_rank: Literal[60] = 60
    agent_global_slot: _GlobalSlot


class CooldownReadyEventV1(EvaluationEventBaseV1):
    """One adjacent-frame positive-to-zero Ultimate cooldown edge."""

    event_type: Literal["cooldown_ready"] = "cooldown_ready"
    phase_rank: Literal[60] = 60
    agent_global_slot: _GlobalSlot


class ChargePhaseDisplacementEventV1(EvaluationEventBaseV1):
    """One nonzero displacement authored by the Charge phase."""

    event_type: Literal["charge_phase_displacement"] = "charge_phase_displacement"
    phase_rank: Literal[70] = 70
    agent_global_slot: _GlobalSlot
    realized_displacement: tuple[_FiniteFloat, _FiniteFloat]


class OrdinaryMovementPhaseDisplacementEventV1(EvaluationEventBaseV1):
    """One nonzero displacement authored by ordinary movement."""

    event_type: Literal["ordinary_movement_phase_displacement"] = (
        "ordinary_movement_phase_displacement"
    )
    phase_rank: Literal[80] = 80
    agent_global_slot: _GlobalSlot
    realized_displacement: tuple[_FiniteFloat, _FiniteFloat]


class AgentDiedEventV1(EvaluationEventBaseV1):
    """One newly dead recipient."""

    event_type: Literal["agent_died"] = "agent_died"
    phase_rank: Literal[90] = 90
    recipient_global_slot: _GlobalSlot


class LethalDamageContributionEventV1(EvaluationEventBaseV1):
    """One positive gross-damage contribution to a new death."""

    event_type: Literal["lethal_damage_contribution"] = "lethal_damage_contribution"
    phase_rank: Literal[90] = 90
    source_global_slot: _GlobalSlot
    recipient_global_slot: _GlobalSlot
    attributed_death_damage: _NonNegativeFloat


class StatusLifecycleEventBaseV1(EvaluationEventBaseV1):
    """Common recipient/status identity for one lifecycle edge."""

    recipient_global_slot: _GlobalSlot
    status_channel: _StatusChannel
    status_id: _AsciiIdentifier


class StatusAgedToZeroEventV1(StatusLifecycleEventBaseV1):
    """One status duration that aged to zero."""

    event_type: Literal["status_aged_to_zero"] = "status_aged_to_zero"
    phase_rank: Literal[100] = 100


class StatusBrokenByDamageEventV1(StatusLifecycleEventBaseV1):
    """One damage-breakable status removed by positive damage."""

    event_type: Literal["status_broken_by_damage"] = "status_broken_by_damage"
    phase_rank: Literal[100] = 100


class StatusAppliedEventV1(StatusLifecycleEventBaseV1):
    """One authoritative status application and its accepted source."""

    event_type: Literal["status_applied"] = "status_applied"
    phase_rank: Literal[100] = 100
    source_global_slot: _GlobalSlot


class StatusRefreshedOrExtendedEventV1(StatusLifecycleEventBaseV1):
    """One status whose accepted application increased remaining duration."""

    event_type: Literal["status_refreshed_or_extended"] = "status_refreshed_or_extended"
    phase_rank: Literal[100] = 100


class StatusClearedByNewDeathEventV1(StatusLifecycleEventBaseV1):
    """One status cleared during new-death canonicalization."""

    event_type: Literal["status_cleared_by_new_death"] = "status_cleared_by_new_death"
    phase_rank: Literal[100] = 100


class SpawnShieldExpiredEventV1(EvaluationEventBaseV1):
    """One ordinary spawn-shield expiry."""

    event_type: Literal["spawn_shield_expired"] = "spawn_shield_expired"
    phase_rank: Literal[110] = 110
    agent_global_slot: _GlobalSlot


class RespawnWaveOccurredEventV1(EvaluationEventBaseV1):
    """One due team respawn wave."""

    event_type: Literal["respawn_wave_occurred"] = "respawn_wave_occurred"
    phase_rank: Literal[120] = 120
    team_index: Literal[0, 1]
    team_id: Literal[1, 2]

    @model_validator(mode="after")
    def _validate_team_join(self) -> RespawnWaveOccurredEventV1:
        if self.team_id != self.team_index + 1:
            raise ValueError("team_id must match the zero-based team_index")
        return self


class AgentRespawnedEventV1(EvaluationEventBaseV1):
    """One realized agent respawn at its assigned pad."""

    event_type: Literal["agent_respawned"] = "agent_respawned"
    phase_rank: Literal[120] = 120
    agent_global_slot: _GlobalSlot
    team_id: Literal[1, 2]
    realized_successor_position: tuple[_FiniteFloat, _FiniteFloat]


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
    | AgentRespawnedEventV1,
    Field(discriminator="event_type"),
]


class EvaluationTransitionV1(EvaluationModel):
    """One transition joining adjacent frames, normalized facts, and events."""

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
    "AssignedPolicySlotV1",
    "AuraMechanicV1",
    "AuraTransitionFactsV1",
    "BaseObservationV1",
    "CaptureProfile",
    "ChargePhaseDisplacementEventV1",
    "ClassMechanicsV1",
    "CodeRevisionV1",
    "CombatCountdownResetEventV1",
    "CombatTransitionFactsV1",
    "ContentAddressedIdentityV1",
    "CooldownReadyEventV1",
    "CooldownStartedEventV1",
    "DeathTransitionFactsV1",
    "EvaluationEpisodeContextV1",
    "EvaluationEpisodeIdentityV1",
    "EvaluationEventBaseV1",
    "EvaluationEventV1",
    "EvaluationFrameV1",
    "EvaluationModel",
    "EvaluationRole",
    "EvaluationSeedProtocolV1",
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
    "TransitionFactsV1",
    "VersionedIdentityV1",
    "canonical_digest_sha256",
    "canonical_json_bytes",
]
