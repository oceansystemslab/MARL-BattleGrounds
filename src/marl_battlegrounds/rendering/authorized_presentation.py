"""Authority-neutral presentation facts derived from validated evaluation rows.

This module deliberately owns no HTTP, replay-artifact, persistence, simulator,
JAX, or NumPy dependency.  Its Oracle builder consumes one already-authorized
durable scene, the same episode context, the optional incoming visual inventory,
and (only when inspection is active) the recorded outgoing transition.  A
successor frame is not an input, so outgoing anchors can only come from the
displayed state.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from hashlib import sha256
from math import isclose, isfinite
from struct import pack, unpack
from typing import Annotated, ClassVar, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from pydantic_core import PydanticSerializationError

from marl_battlegrounds.evaluation.models import (
    EvaluationEpisodeContextV1,
    EvaluationTransitionV1,
)
from marl_battlegrounds.evaluation.wire_shapes import (
    NUM_MOVE_ACTIONS_V1,
    NUM_TARGET_ACTIONS_V1,
    NUM_ULTIMATE_ACTIONS_V1,
)
from marl_battlegrounds.rendering.evaluation_adapter import (
    validate_oracle_scene_static_authority_v1,
)
from marl_battlegrounds.rendering.scene import (
    AbilityActivatedEventV2,
    ActionRejectedEventV2,
    AgentDiedEventV2,
    AgentRespawnedEventV2,
    AgentSceneV2,
    AuraRecipientModifierSceneV2,
    BattlefieldSceneV2,
    ChargePhaseDisplacementEventV2,
    ClassAuraMechanicSceneV2,
    ClassMechanicsSceneV2,
    ClassStatusMechanicSceneV2,
    CombatCountdownResetEventV2,
    CooldownReadyEventV2,
    CooldownStartedEventV2,
    HealthRegeneratedEventV2,
    LethalDamageContributionEventV2,
    MapSceneV1,
    ObstacleSceneV1,
    OrdinaryMovementPhaseDisplacementEventV2,
    RecipientHealthResolutionEventV2,
    RespawnWaveOccurredEventV2,
    RespawnWaveSceneV2,
    SourceDamageOutputEventV2,
    SourceHealingOutputEventV2,
    SpawnShieldExpiredEventV2,
    StatusAgedToZeroEventV2,
    StatusAppliedEventV2,
    StatusBrokenByDamageEventV2,
    StatusClearedByNewDeathEventV2,
    StatusRefreshedOrExtendedEventV2,
    StatusSceneV2,
    VisualAgentAnchorV2,
    VisualAgentPhaseTrajectoryV2,
    VisualAnchorPhaseV2,
    VisualEventBatchV2,
    VisualEventV2,
)
from marl_battlegrounds.rendering.vocabulary import CATALOG_STATUS_ID_BY_CHANNEL

AUTHORIZED_PRESENTATION_SCHEMA_VERSION = 1
_STRICT_WIRE_DATACLASS_CONFIG = ConfigDict(
    allow_inf_nan=False,
    extra="forbid",
    strict=True,
)

type AuthorizedRelationV1 = Literal["oracle", "self", "ally", "opponent"]
type AgentLifeStateV1 = Literal["alive", "corpse"]
type AcceptedLaneV1 = Literal["basic", "ultimate"]
type Point2D = tuple[float, float]
type ReplayIncomingAnchorPhaseV1 = Literal[
    "transition_start",
    "post_charge",
    "successor",
]
type AuthorizedAuraIdV1 = Literal[
    "mage_damage_amplification",
    "warrior_damage_mitigation",
]
type ReplayIncomingEventKindV1 = Literal[
    "action_rejected",
    "ability_activated",
    "source_damage_output",
    "source_healing_output",
    "recipient_health_resolution",
    "combat_countdown_reset",
    "health_regenerated",
    "cooldown_started",
    "cooldown_ready",
    "charge_phase_displacement",
    "ordinary_movement_phase_displacement",
    "agent_died",
    "lethal_damage_contribution",
    "status_aged_to_zero",
    "status_broken_by_damage",
    "status_applied",
    "status_refreshed_or_extended",
    "status_cleared_by_new_death",
    "spawn_shield_expired",
    "respawn_wave_occurred",
    "agent_respawned",
]

_REPLAY_INCOMING_EVENT_KINDS_V1 = frozenset(
    (
        "action_rejected",
        "ability_activated",
        "source_damage_output",
        "source_healing_output",
        "recipient_health_resolution",
        "combat_countdown_reset",
        "health_regenerated",
        "cooldown_started",
        "cooldown_ready",
        "charge_phase_displacement",
        "ordinary_movement_phase_displacement",
        "agent_died",
        "lethal_damage_contribution",
        "status_aged_to_zero",
        "status_broken_by_damage",
        "status_applied",
        "status_refreshed_or_extended",
        "status_cleared_by_new_death",
        "spawn_shield_expired",
        "respawn_wave_occurred",
        "agent_respawned",
    )
)
_CANONICAL_CLASS_NAME_BY_ID_V1 = {
    1: "Mage",
    2: "Warrior",
    3: "Hunter",
    4: "Rogue",
    5: "Priest",
}
_STATUS_SOURCE_CLASS_BY_CHANNEL_V1 = (2, 3, 4, 2, 3, 4, 4, 1, 5)
_AURA_SOURCE_CLASS_BY_ID_V1 = {
    "mage_damage_amplification": 1,
    "warrior_damage_mitigation": 2,
}


def _require_python_int(value: int, *, name: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be a Python int >= {minimum}.")


def _require_text(value: str, *, name: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a non-empty Python string.")


def _require_finite(value: float, *, name: str, minimum: float | None = None) -> None:
    if type(value) is not float or not isfinite(value):
        raise ValueError(f"{name} must be a finite Python float.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")


def _require_point(value: Point2D, *, name: str) -> None:
    if type(value) is not tuple or len(value) != 2:
        raise ValueError(f"{name} must be a two-coordinate Python tuple.")
    for coordinate in value:
        _require_finite(coordinate, name=f"{name} coordinate")


def _points_close(left: Point2D, right: Point2D) -> bool:
    return all(
        isclose(left_value, right_value, rel_tol=1e-6, abs_tol=1e-5)
        for left_value, right_value in zip(left, right, strict=True)
    )


def _equals_catalog_or_exact_f32_encoding(
    recorded_value: float,
    catalog_value: float,
) -> bool:
    """Join one recorded float32 fact to its public catalog authority exactly.

    POV feature rows preserve the binary32 value recorded on the observation
    wire, while a validated public catalog may retain a wider Python float.
    This is not a tolerance: the values join only when they are already equal
    or the recorded value is the exact IEEE-754 binary32 encoding of the
    catalog value.
    """
    if recorded_value == catalog_value:
        return True
    try:
        catalog_as_f32 = unpack(">f", pack(">f", catalog_value))[0]
    except OverflowError:
        return False
    return recorded_value == catalog_as_f32


def _optional_catalog_float_joins(
    recorded_value: float | None,
    catalog_value: float | None,
) -> bool:
    if recorded_value is None or catalog_value is None:
        return recorded_value is catalog_value
    return _equals_catalog_or_exact_f32_encoding(recorded_value, catalog_value)


def _require_exact_tuple(
    value: object,
    *,
    name: str,
    item_type: type[object],
) -> None:
    if type(value) is not tuple:
        raise ValueError(f"{name} must be a Python tuple.")
    items = cast(tuple[object, ...], value)
    if any(type(item) is not item_type for item in items):
        raise ValueError(f"{name} must contain exact {item_type.__name__} rows.")


def oracle_presentation_key_v1(
    *,
    authority_session_id: str,
    public_agent_id: str,
) -> str:
    """Return a stable, authority-namespaced key with no embedded slot meaning."""
    payload = f"oracle\x00{authority_session_id}\x00{public_agent_id}".encode()
    return f"oracle_{sha256(payload).hexdigest()}"


@dataclass(frozen=True, slots=True, kw_only=True)
class AuthorizedObstacleV1:
    """One strict authority-neutral static obstacle."""

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG

    obstacle_id: str
    kind: Literal["pillar", "wall"]
    center: Point2D
    radius: float | None
    width: float | None
    height: float | None
    theta: float

    def __post_init__(self) -> None:
        _require_text(self.obstacle_id, name="obstacle_id")
        _require_point(self.center, name="center")
        _require_finite(self.theta, name="theta")
        if self.kind == "pillar":
            if self.radius is None:
                raise ValueError("pillar obstacles require a radius.")
            _require_finite(self.radius, name="radius", minimum=0.0)
            if self.radius <= 0.0 or self.width is not None or self.height is not None:
                raise ValueError("pillar obstacle dimensions are inconsistent.")
        elif self.kind == "wall":
            if self.width is None or self.height is None:
                raise ValueError("wall obstacles require width and height.")
            _require_finite(self.width, name="width", minimum=0.0)
            _require_finite(self.height, name="height", minimum=0.0)
            if self.width <= 0.0 or self.height <= 0.0 or self.radius is not None:
                raise ValueError("wall obstacle dimensions are inconsistent.")
        else:
            raise ValueError("unknown authorized obstacle kind.")


@dataclass(frozen=True, slots=True, kw_only=True)
class AuthorizedMapV1:
    """Strict finite map bounds and ordered static obstacles."""

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG

    width: float
    height: float
    obstacles: tuple[AuthorizedObstacleV1, ...]

    def __post_init__(self) -> None:
        _require_finite(self.width, name="width", minimum=0.0)
        _require_finite(self.height, name="height", minimum=0.0)
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("map width and height must be positive.")
        _require_exact_tuple(
            self.obstacles,
            name="obstacles",
            item_type=AuthorizedObstacleV1,
        )
        obstacle_ids = tuple(row.obstacle_id for row in self.obstacles)
        if len(obstacle_ids) != len(set(obstacle_ids)):
            raise ValueError("authorized obstacle IDs must be unique.")


@dataclass(frozen=True, slots=True, kw_only=True)
class AuthorizedAuraModifierV1:
    """One exact non-neutral recipient-local aggregate aura multiplier."""

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG

    aura_id: AuthorizedAuraIdV1
    multiplier: float

    def __post_init__(self) -> None:
        if self.aura_id not in (
            "mage_damage_amplification",
            "warrior_damage_mitigation",
        ):
            raise ValueError("unknown authorized aura modifier identity.")
        _require_finite(self.multiplier, name="multiplier", minimum=0.0)
        if self.multiplier == 1.0:
            raise ValueError("neutral aura modifiers must be omitted.")


@dataclass(frozen=True, slots=True, kw_only=True)
class AuthorizedClassStatusMechanicV1:
    """One strict public-catalog status mechanic."""

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG

    status_channel: int
    status_id: str
    family: Literal[
        "slow",
        "stun",
        "anti_heal",
        "damage_amplification",
        "movement_floor",
    ]
    source_action_component: Literal["basic", "ultimate"]
    duration_steps: int
    magnitude_kind: Literal[
        "movement_multiplier",
        "none",
        "healing_multiplier",
        "damage_multiplier",
        "movement_floor",
    ]
    magnitude: float | None
    breaks_on_positive_damage: bool

    def __post_init__(self) -> None:
        _require_python_int(self.status_channel, name="status_channel")
        if self.status_channel >= 9:
            raise ValueError("status channel is outside the V1 catalog axis.")
        _require_text(self.status_id, name="status_id")
        if CATALOG_STATUS_ID_BY_CHANNEL[self.status_channel] != self.status_id:
            raise ValueError("status channel and ID must retain V1 identity.")
        if self.family not in (
            "slow",
            "stun",
            "anti_heal",
            "damage_amplification",
            "movement_floor",
        ):
            raise ValueError("unknown class status family.")
        if self.source_action_component not in ("basic", "ultimate"):
            raise ValueError("unknown class status action component.")
        _require_python_int(self.duration_steps, name="duration_steps", minimum=1)
        if self.magnitude_kind not in (
            "movement_multiplier",
            "none",
            "healing_multiplier",
            "damage_multiplier",
            "movement_floor",
        ):
            raise ValueError("unknown class status magnitude kind.")
        if self.magnitude_kind == "none":
            if self.magnitude is not None:
                raise ValueError("none class status magnitude must omit its value.")
        elif self.magnitude is None:
            raise ValueError("non-none class status magnitude requires its value.")
        else:
            _require_finite(self.magnitude, name="magnitude")
        if type(self.breaks_on_positive_damage) is not bool:
            raise ValueError("breaks_on_positive_damage must be a Python bool.")


@dataclass(frozen=True, slots=True, kw_only=True)
class AuthorizedClassAuraMechanicV1:
    """One strict public-catalog aura mechanic."""

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG

    aura_id: AuthorizedAuraIdV1
    radius: float
    per_emitter_multiplier: float
    stacking_rule: Literal["multiply_then_clamp"]
    clamp_kind: Literal["ceiling", "floor"]
    clamp_value: float

    def __post_init__(self) -> None:
        if self.aura_id not in _AURA_SOURCE_CLASS_BY_ID_V1:
            raise ValueError("unknown class aura identity.")
        for name in ("radius", "per_emitter_multiplier", "clamp_value"):
            _require_finite(cast(float, getattr(self, name)), name=name, minimum=0.0)
        if self.stacking_rule != "multiply_then_clamp":
            raise ValueError("unknown class aura stacking rule.")
        if self.clamp_kind not in ("ceiling", "floor"):
            raise ValueError("unknown class aura clamp kind.")


@dataclass(frozen=True, slots=True, kw_only=True)
class AuthorizedClassMechanicsV1:
    """Strict serialized class vocabulary and exact catalog mechanics."""

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG

    class_id: int
    class_name: str
    maximum_health: float
    body_radius: float
    base_movement_speed: float
    observation_radius: float
    basic_target_mode: Literal["unavailable", "ally", "enemy"]
    basic_interaction_radius: float
    basic_raw_damage: float
    basic_raw_healing: float
    ultimate_target_mode: Literal["unavailable", "target_none", "ally", "enemy"]
    ultimate_interaction_radius: float
    ultimate_cooldown_steps: int
    ultimate_raw_damage: float
    ultimate_raw_healing: float
    out_of_combat_delay_steps: int
    out_of_combat_health_regeneration_fraction_per_step: float
    status_mechanics: tuple[AuthorizedClassStatusMechanicV1, ...]
    aura_mechanics: tuple[AuthorizedClassAuraMechanicV1, ...]

    def __post_init__(self) -> None:
        _require_python_int(self.class_id, name="class_id", minimum=1)
        if self.class_id > 5:
            raise ValueError("class_id must identify a real V1 class.")
        _require_text(self.class_name, name="class_name")
        if _CANONICAL_CLASS_NAME_BY_ID_V1[self.class_id] != self.class_name:
            raise ValueError("class name must retain canonical V1 identity.")
        for name in (
            "maximum_health",
            "body_radius",
            "base_movement_speed",
            "observation_radius",
            "basic_interaction_radius",
            "basic_raw_damage",
            "basic_raw_healing",
            "ultimate_interaction_radius",
            "ultimate_raw_damage",
            "ultimate_raw_healing",
            "out_of_combat_health_regeneration_fraction_per_step",
        ):
            _require_finite(cast(float, getattr(self, name)), name=name, minimum=0.0)
        if self.maximum_health <= 0.0 or self.body_radius <= 0.0:
            raise ValueError("class health and body radius must be positive.")
        if self.basic_target_mode not in ("unavailable", "ally", "enemy"):
            raise ValueError("unknown basic target mode.")
        if self.ultimate_target_mode not in (
            "unavailable",
            "target_none",
            "ally",
            "enemy",
        ):
            raise ValueError("unknown ultimate target mode.")
        _require_python_int(
            self.ultimate_cooldown_steps,
            name="ultimate_cooldown_steps",
        )
        _require_python_int(
            self.out_of_combat_delay_steps,
            name="out_of_combat_delay_steps",
        )
        if self.out_of_combat_health_regeneration_fraction_per_step > 1.0:
            raise ValueError("class regeneration fraction cannot exceed one.")
        _require_exact_tuple(
            self.status_mechanics,
            name="status_mechanics",
            item_type=AuthorizedClassStatusMechanicV1,
        )
        _require_exact_tuple(
            self.aura_mechanics,
            name="aura_mechanics",
            item_type=AuthorizedClassAuraMechanicV1,
        )
        status_channels = tuple(row.status_channel for row in self.status_mechanics)
        if status_channels != tuple(sorted(status_channels)) or len(
            status_channels
        ) != len(set(status_channels)):
            raise ValueError("class status mechanics require unique ordered channels.")
        aura_ids = tuple(row.aura_id for row in self.aura_mechanics)
        if len(aura_ids) != len(set(aura_ids)):
            raise ValueError("class aura mechanics require unique identities.")


@dataclass(frozen=True, slots=True, kw_only=True)
class AuthorizedRespawnWaveV1:
    """One strict current team lifecycle countdown."""

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG

    team_index: int
    team_id: int
    period_steps: int
    countdown_steps: int

    def __post_init__(self) -> None:
        _require_python_int(self.team_index, name="team_index")
        _require_python_int(self.team_id, name="team_id", minimum=1)
        if self.team_index not in (0, 1) or self.team_id != self.team_index + 1:
            raise ValueError("respawn wave team identity is inconsistent.")
        _require_python_int(self.period_steps, name="period_steps", minimum=1)
        _require_python_int(self.countdown_steps, name="countdown_steps")
        if self.countdown_steps >= self.period_steps:
            raise ValueError("respawn countdown must be less than its period.")


@dataclass(frozen=True, slots=True, kw_only=True)
class AuthorizedSpawnShieldMechanicsAvailableV1:
    """Exact configured spawn-shield mechanics available to presentation."""

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG

    availability_kind: Literal["available"]
    configured_duration_steps: int
    movement_speed: float

    def __post_init__(self) -> None:
        if self.availability_kind != "available":
            raise ValueError("unknown available spawn-shield discriminator.")
        _require_python_int(
            self.configured_duration_steps,
            name="configured_duration_steps",
        )
        _require_finite(self.movement_speed, name="movement_speed", minimum=0.0)
        if self.movement_speed <= 0.0:
            raise ValueError("spawn-shield movement speed must be positive.")


@dataclass(frozen=True, slots=True, kw_only=True)
class AuthorizedSpawnShieldMechanicsUnavailableV1:
    """Explicit absence of recorded spawn-shield configuration facts."""

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG

    availability_kind: Literal["unavailable"]

    def __post_init__(self) -> None:
        if self.availability_kind != "unavailable":
            raise ValueError("unknown unavailable spawn-shield discriminator.")


type AuthorizedSpawnShieldMechanicsV1 = Annotated[
    AuthorizedSpawnShieldMechanicsAvailableV1
    | AuthorizedSpawnShieldMechanicsUnavailableV1,
    Field(discriminator="availability_kind"),
]


@dataclass(frozen=True, slots=True, kw_only=True)
class AuthorizedStatusSourceV1:
    """One authorized direct source identity without a raw slot or event ID."""

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG

    source_presentation_key: str
    source_public_agent_id: str

    def __post_init__(self) -> None:
        _require_text(self.source_presentation_key, name="source_presentation_key")
        _require_text(self.source_public_agent_id, name="source_public_agent_id")


@dataclass(frozen=True, slots=True, kw_only=True)
class AuthorizedStatusV1:
    """One durable status row and only its authorized source identities."""

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG

    status_channel: int
    status_id: str
    family: Literal[
        "slow",
        "stun",
        "anti_heal",
        "damage_amplification",
        "movement_floor",
    ]
    configured_duration_steps: int
    remaining_duration: int
    source_class_id: int
    source_class_name: str
    source_action_component: Literal["basic", "ultimate"]
    magnitude_kind: Literal[
        "movement_multiplier",
        "none",
        "healing_multiplier",
        "damage_multiplier",
        "movement_floor",
    ]
    magnitude: float | None
    breaks_on_positive_damage: bool
    direct_sources: tuple[AuthorizedStatusSourceV1, ...]

    def __post_init__(self) -> None:
        _require_python_int(self.status_channel, name="status_channel")
        if self.status_channel >= 9:
            raise ValueError("status channel is outside the V1 catalog axis.")
        _require_text(self.status_id, name="status_id")
        if CATALOG_STATUS_ID_BY_CHANNEL[self.status_channel] != self.status_id:
            raise ValueError("status channel and ID must retain V1 identity.")
        if self.family not in (
            "slow",
            "stun",
            "anti_heal",
            "damage_amplification",
            "movement_floor",
        ):
            raise ValueError("unknown authorized status family.")
        _require_python_int(
            self.configured_duration_steps,
            name="configured_duration_steps",
            minimum=1,
        )
        _require_python_int(
            self.remaining_duration,
            name="remaining_duration",
            minimum=1,
        )
        if self.remaining_duration > self.configured_duration_steps:
            raise ValueError("remaining status duration exceeds configured duration.")
        _require_python_int(self.source_class_id, name="source_class_id", minimum=1)
        if self.source_class_id > 5:
            raise ValueError("source_class_id must identify a real V1 class.")
        _require_text(self.source_class_name, name="source_class_name")
        if self.source_action_component not in ("basic", "ultimate"):
            raise ValueError("unknown authorized status action component.")
        if self.magnitude_kind not in (
            "movement_multiplier",
            "none",
            "healing_multiplier",
            "damage_multiplier",
            "movement_floor",
        ):
            raise ValueError("unknown authorized status magnitude kind.")
        if self.magnitude is None:
            if self.magnitude_kind != "none":
                raise ValueError("non-none status magnitude kinds require a value.")
        else:
            _require_finite(self.magnitude, name="magnitude")
            if self.magnitude_kind == "none":
                raise ValueError("none status magnitude kind must omit its value.")
        if type(self.breaks_on_positive_damage) is not bool:
            raise ValueError("breaks_on_positive_damage must be a Python bool.")
        _require_exact_tuple(
            self.direct_sources,
            name="direct_sources",
            item_type=AuthorizedStatusSourceV1,
        )
        keys = tuple(row.source_presentation_key for row in self.direct_sources)
        if len(keys) != len(set(keys)):
            raise ValueError(
                "direct status sources must have unique presentation keys."
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class AuthorizedAgentV1:
    """One authority-neutral durable agent row keyed for presentation only."""

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG

    presentation_key: str
    public_agent_id: str
    relation: AuthorizedRelationV1
    team_id: int
    class_id: int
    class_name: str
    position: Point2D
    radius: float
    life_state: AgentLifeStateV1
    current_health: float
    maximum_health: float
    base_movement_speed: float
    effective_movement_speed: float
    observation_radius: float
    basic_interaction_radius: float
    ultimate_interaction_radius: float
    ultimate_cooldown_remaining: int
    spawn_shield_remaining: int
    steps_until_out_of_combat: int
    out_of_combat_delay_steps: int
    out_of_combat_health_regeneration_fraction_per_step: float
    statuses: tuple[AuthorizedStatusV1, ...]
    aura_modifiers: tuple[AuthorizedAuraModifierV1, ...]

    def __post_init__(self) -> None:
        _require_text(self.presentation_key, name="presentation_key")
        _require_text(self.public_agent_id, name="public_agent_id")
        if self.relation not in ("oracle", "self", "ally", "opponent"):
            raise ValueError("unknown authorized relation.")
        _require_python_int(self.team_id, name="team_id", minimum=1)
        if self.team_id not in (1, 2):
            raise ValueError("team_id must be one or two.")
        _require_python_int(self.class_id, name="class_id", minimum=1)
        if self.class_id > 5:
            raise ValueError("class_id must identify a real V1 class.")
        _require_text(self.class_name, name="class_name")
        if _CANONICAL_CLASS_NAME_BY_ID_V1[self.class_id] != self.class_name:
            raise ValueError("agent class identity must be canonical.")
        _require_point(self.position, name="position")
        for name in (
            "radius",
            "maximum_health",
            "base_movement_speed",
            "effective_movement_speed",
            "observation_radius",
            "basic_interaction_radius",
            "ultimate_interaction_radius",
            "out_of_combat_health_regeneration_fraction_per_step",
        ):
            _require_finite(cast(float, getattr(self, name)), name=name, minimum=0.0)
        _require_finite(self.current_health, name="current_health", minimum=0.0)
        if self.radius <= 0.0 or self.maximum_health <= 0.0:
            raise ValueError("agent radius and maximum health must be positive.")
        if self.current_health > self.maximum_health:
            raise ValueError("current health cannot exceed maximum health.")
        if self.life_state not in ("alive", "corpse"):
            raise ValueError("unknown agent life state.")
        for name in (
            "ultimate_cooldown_remaining",
            "spawn_shield_remaining",
            "steps_until_out_of_combat",
            "out_of_combat_delay_steps",
        ):
            _require_python_int(cast(int, getattr(self, name)), name=name)
        if self.steps_until_out_of_combat > self.out_of_combat_delay_steps:
            raise ValueError("out-of-combat countdown exceeds its configured delay.")
        if self.out_of_combat_health_regeneration_fraction_per_step > 1.0:
            raise ValueError("agent regeneration fraction cannot exceed one.")
        _require_exact_tuple(
            self.statuses,
            name="statuses",
            item_type=AuthorizedStatusV1,
        )
        channels = tuple(row.status_channel for row in self.statuses)
        if len(channels) != len(set(channels)):
            raise ValueError("agent statuses must have unique channels.")
        _require_exact_tuple(
            self.aura_modifiers,
            name="aura_modifiers",
            item_type=AuthorizedAuraModifierV1,
        )
        aura_ids = tuple(row.aura_id for row in self.aura_modifiers)
        if len(aura_ids) != len(set(aura_ids)):
            raise ValueError("agent aura modifiers must have unique identities.")
        if any(row.multiplier == 1.0 for row in self.aura_modifiers):
            raise ValueError("neutral aura modifiers must not enter presentation rows.")


@dataclass(frozen=True, slots=True, kw_only=True)
class AuthorizedAuraFieldV1:
    """One authorized emitter field keyed without exposing a raw actor slot."""

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG

    aura_id: AuthorizedAuraIdV1
    source_presentation_key: str
    source_public_agent_id: str
    source_class_id: int
    source_class_name: str
    source_alive: bool
    center: Point2D
    radius: float
    beneficiary_relation: Literal["same_team"]
    per_emitter_multiplier: float
    stacking_rule: Literal["multiply_then_clamp"]
    clamp_kind: Literal["ceiling", "floor"]
    clamp_value: float

    def __post_init__(self) -> None:
        if self.aura_id not in (
            "mage_damage_amplification",
            "warrior_damage_mitigation",
        ):
            raise ValueError("unknown authorized aura field identity.")
        _require_text(self.source_presentation_key, name="source_presentation_key")
        _require_text(self.source_public_agent_id, name="source_public_agent_id")
        _require_python_int(self.source_class_id, name="source_class_id", minimum=1)
        _require_text(self.source_class_name, name="source_class_name")
        if type(self.source_alive) is not bool:
            raise ValueError("source_alive must be a Python bool.")
        _require_point(self.center, name="center")
        for name in ("radius", "per_emitter_multiplier", "clamp_value"):
            _require_finite(cast(float, getattr(self, name)), name=name, minimum=0.0)
        if self.radius <= 0.0:
            raise ValueError("aura radius must be positive.")
        if self.beneficiary_relation != "same_team":
            raise ValueError("authorized aura fields require same-team beneficiaries.")
        if self.stacking_rule != "multiply_then_clamp":
            raise ValueError("unknown authorized aura stacking rule.")
        if self.clamp_kind not in ("ceiling", "floor"):
            raise ValueError("unknown authorized aura clamp kind.")


@dataclass(frozen=True, slots=True, kw_only=True)
class AuthorizedSpawnPadV1:
    """One authorized lifecycle pad with an optional visible-body assignment."""

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG

    team_id: int
    team_local_slot: int
    assigned_presentation_key: str | None
    assigned_public_agent_id: str | None
    position: Point2D
    configured_active: bool
    currently_alive: bool
    spawn_shield_remaining: int

    def __post_init__(self) -> None:
        _require_python_int(self.team_id, name="team_id", minimum=1)
        if self.team_id not in (1, 2):
            raise ValueError("team_id must be one or two.")
        _require_python_int(self.team_local_slot, name="team_local_slot")
        if self.team_local_slot >= 5:
            raise ValueError("team_local_slot must be less than five.")
        if (self.assigned_presentation_key is None) != (
            self.assigned_public_agent_id is None
        ):
            raise ValueError("spawn-pad assignee identity must be present as a pair.")
        if self.assigned_presentation_key is not None:
            _require_text(
                self.assigned_presentation_key,
                name="assigned_presentation_key",
            )
            if self.assigned_public_agent_id is None:  # pragma: no cover - paired.
                raise AssertionError("spawn-pad public assignee disappeared")
            _require_text(
                self.assigned_public_agent_id,
                name="assigned_public_agent_id",
            )
        _require_point(self.position, name="position")
        if (
            type(self.configured_active) is not bool
            or type(self.currently_alive) is not bool
        ):
            raise ValueError("spawn-pad lifecycle flags must be Python bools.")
        if self.currently_alive and not self.configured_active:
            raise ValueError("an inactive spawn-pad occupant cannot be alive.")
        if not self.configured_active and self.assigned_presentation_key is not None:
            raise ValueError("an inactive spawn pad cannot assign a visible body.")
        _require_python_int(
            self.spawn_shield_remaining,
            name="spawn_shield_remaining",
        )
        if not self.configured_active and self.spawn_shield_remaining != 0:
            raise ValueError("inactive spawn-pad rows cannot retain spawn shield.")


@dataclass(frozen=True, slots=True, kw_only=True)
class AuthorizedBattlefieldSceneV1:
    """Neutral durable battlefield facts with authority and epochs outside it."""

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG

    schema_version: Literal[1]
    map: AuthorizedMapV1
    agents: tuple[AuthorizedAgentV1, ...]
    aura_fields: tuple[AuthorizedAuraFieldV1, ...]
    class_mechanics: tuple[AuthorizedClassMechanicsV1, ...]
    spawn_shield_mechanics: AuthorizedSpawnShieldMechanicsV1
    spawn_pads: tuple[AuthorizedSpawnPadV1, ...]
    respawn_waves: tuple[AuthorizedRespawnWaveV1, ...]

    def __post_init__(self) -> None:
        if self.schema_version != AUTHORIZED_PRESENTATION_SCHEMA_VERSION:
            raise ValueError("unknown authorized battlefield schema version.")
        if type(self.map) is not AuthorizedMapV1:
            raise ValueError("map must be the exact authorized map root.")
        _require_exact_tuple(self.agents, name="agents", item_type=AuthorizedAgentV1)
        _require_exact_tuple(
            self.aura_fields,
            name="aura_fields",
            item_type=AuthorizedAuraFieldV1,
        )
        _require_exact_tuple(
            self.class_mechanics,
            name="class_mechanics",
            item_type=AuthorizedClassMechanicsV1,
        )
        if type(self.spawn_shield_mechanics) not in (
            AuthorizedSpawnShieldMechanicsAvailableV1,
            AuthorizedSpawnShieldMechanicsUnavailableV1,
        ):
            raise ValueError("spawn_shield_mechanics must use an exact variant.")
        _require_exact_tuple(
            self.spawn_pads,
            name="spawn_pads",
            item_type=AuthorizedSpawnPadV1,
        )
        _require_exact_tuple(
            self.respawn_waves,
            name="respawn_waves",
            item_type=AuthorizedRespawnWaveV1,
        )
        keys = tuple(row.presentation_key for row in self.agents)
        public_ids = tuple(row.public_agent_id for row in self.agents)
        if len(keys) != len(set(keys)) or len(public_ids) != len(set(public_ids)):
            raise ValueError("authorized agents require unique keys and public IDs.")
        agent_by_key = {row.presentation_key: row for row in self.agents}
        mechanics_ids = tuple(row.class_id for row in self.class_mechanics)
        represented_class_ids = tuple(sorted({row.class_id for row in self.agents}))
        if mechanics_ids != represented_class_ids:
            raise ValueError(
                "class mechanics must equal the represented authorized classes."
            )
        mechanics_by_id = {row.class_id: row for row in self.class_mechanics}
        projected_status_channels = tuple(
            status.status_channel
            for mechanics in self.class_mechanics
            for status in mechanics.status_mechanics
        )
        expected_status_channels = tuple(
            status_channel
            for class_id in represented_class_ids
            for status_channel, source_class_id in enumerate(
                _STATUS_SOURCE_CLASS_BY_CHANNEL_V1
            )
            if source_class_id == class_id
        )
        if projected_status_channels != expected_status_channels:
            raise ValueError("class mechanics changed the represented V1 status axis.")
        for mechanics in self.class_mechanics:
            if any(
                _STATUS_SOURCE_CLASS_BY_CHANNEL_V1[status.status_channel]
                != mechanics.class_id
                for status in mechanics.status_mechanics
            ):
                raise ValueError("class status mechanics changed source class.")
        projected_aura_ids = tuple(
            aura.aura_id
            for mechanics in self.class_mechanics
            for aura in mechanics.aura_mechanics
        )
        expected_aura_ids = tuple(
            aura_id
            for aura_id, source_class_id in _AURA_SOURCE_CLASS_BY_ID_V1.items()
            if source_class_id in represented_class_ids
        )
        if projected_aura_ids != expected_aura_ids:
            raise ValueError("class mechanics changed the represented V1 aura axis.")
        for mechanics in self.class_mechanics:
            if any(
                _AURA_SOURCE_CLASS_BY_ID_V1[aura.aura_id] != mechanics.class_id
                for aura in mechanics.aura_mechanics
            ):
                raise ValueError("class aura mechanics changed emitter class.")
        status_by_channel = {
            status.status_channel: status
            for class_mechanics in self.class_mechanics
            for status in class_mechanics.status_mechanics
        }
        for agent in self.agents:
            mechanics = mechanics_by_id.get(agent.class_id)
            # Agent geometry, health, speed, ranges, and OOC facts may be exact
            # per-slot profile overrides.  Class mechanics remain public class
            # documentation, so only categorical class identity joins here.
            if mechanics is None or mechanics.class_name != agent.class_name:
                raise ValueError("agent class identity must join class mechanics.")
            if agent.ultimate_cooldown_remaining > mechanics.ultimate_cooldown_steps:
                raise ValueError("agent cooldown remaining exceeds its class duration.")
            if agent.relation == "oracle" and (
                not _equals_catalog_or_exact_f32_encoding(
                    agent.maximum_health,
                    mechanics.maximum_health,
                )
                or not _equals_catalog_or_exact_f32_encoding(
                    agent.radius,
                    mechanics.body_radius,
                )
                or not _equals_catalog_or_exact_f32_encoding(
                    agent.base_movement_speed,
                    mechanics.base_movement_speed,
                )
                or not _equals_catalog_or_exact_f32_encoding(
                    agent.observation_radius,
                    mechanics.observation_radius,
                )
                or not _equals_catalog_or_exact_f32_encoding(
                    agent.basic_interaction_radius,
                    mechanics.basic_interaction_radius,
                )
                or not _equals_catalog_or_exact_f32_encoding(
                    agent.ultimate_interaction_radius,
                    mechanics.ultimate_interaction_radius,
                )
                or agent.out_of_combat_delay_steps
                != mechanics.out_of_combat_delay_steps
                or not _equals_catalog_or_exact_f32_encoding(
                    agent.out_of_combat_health_regeneration_fraction_per_step,
                    mechanics.out_of_combat_health_regeneration_fraction_per_step,
                )
            ):
                raise ValueError("Oracle agent static facts must join class mechanics.")
            for status in agent.statuses:
                status_mechanic = status_by_channel.get(status.status_channel)
                if (
                    _STATUS_SOURCE_CLASS_BY_CHANNEL_V1[status.status_channel]
                    != status.source_class_id
                    or _CANONICAL_CLASS_NAME_BY_ID_V1[status.source_class_id]
                    != status.source_class_name
                ):
                    raise ValueError("durable status changed its V1 source identity.")
                if status_mechanic is not None and (
                    status_mechanic.status_id != status.status_id
                    or status_mechanic.duration_steps
                    != status.configured_duration_steps
                    or status_mechanic.family != status.family
                    or status_mechanic.source_action_component
                    != status.source_action_component
                    or status_mechanic.magnitude_kind != status.magnitude_kind
                    or not _optional_catalog_float_joins(
                        status.magnitude,
                        status_mechanic.magnitude,
                    )
                    or status_mechanic.breaks_on_positive_damage
                    != status.breaks_on_positive_damage
                ):
                    raise ValueError("durable status must join its catalog mechanic.")
                for source in status.direct_sources:
                    source_agent = agent_by_key.get(source.source_presentation_key)
                    if source_agent is None or (
                        source_agent.public_agent_id != source.source_public_agent_id
                        or source_agent.class_id != status.source_class_id
                        or source_agent.class_name != status.source_class_name
                    ):
                        raise ValueError(
                            "status source must join an authorized source-class agent."
                        )
            if any(
                modifier.aura_id not in _AURA_SOURCE_CLASS_BY_ID_V1
                for modifier in agent.aura_modifiers
            ):
                raise ValueError("agent aura modifier is outside the V1 catalog.")
        for field in self.aura_fields:
            source = agent_by_key.get(field.source_presentation_key)
            if source is None or (
                source.public_agent_id != field.source_public_agent_id
                or source.class_id != field.source_class_id
                or source.class_name != field.source_class_name
                or source.position != field.center
                or (source.life_state == "alive") != field.source_alive
            ):
                raise ValueError("aura field must join its authorized source agent.")
            source_mechanics = mechanics_by_id[field.source_class_id]
            matching_mechanics = tuple(
                row
                for row in source_mechanics.aura_mechanics
                if row.aura_id == field.aura_id
            )
            if len(matching_mechanics) != 1:
                raise ValueError("aura field must join one catalog aura mechanic.")
            aura_mechanic = matching_mechanics[0]
            if (
                not _equals_catalog_or_exact_f32_encoding(
                    field.radius,
                    aura_mechanic.radius,
                )
                or not _equals_catalog_or_exact_f32_encoding(
                    field.per_emitter_multiplier,
                    aura_mechanic.per_emitter_multiplier,
                )
                or aura_mechanic.stacking_rule != field.stacking_rule
                or aura_mechanic.clamp_kind != field.clamp_kind
                or aura_mechanic.clamp_value != field.clamp_value
            ):
                raise ValueError("aura field facts must equal its catalog mechanic.")
        pad_keys = tuple((row.team_id, row.team_local_slot) for row in self.spawn_pads)
        if pad_keys != tuple(sorted(pad_keys)) or len(pad_keys) != len(set(pad_keys)):
            raise ValueError("spawn pads must retain unique source ordering.")
        for pad in self.spawn_pads:
            if pad.assigned_presentation_key is None:
                continue
            assigned = agent_by_key.get(pad.assigned_presentation_key)
            if assigned is None or (
                assigned.public_agent_id != pad.assigned_public_agent_id
                or assigned.team_id != pad.team_id
                or pad.currently_alive != (assigned.life_state == "alive")
                or pad.spawn_shield_remaining != assigned.spawn_shield_remaining
            ):
                raise ValueError("spawn pad must join its authorized assignee.")
        if (
            type(self.spawn_shield_mechanics)
            is AuthorizedSpawnShieldMechanicsAvailableV1
        ):
            configured_duration = self.spawn_shield_mechanics.configured_duration_steps
            if any(
                row.spawn_shield_remaining > configured_duration for row in self.agents
            ) or any(
                row.spawn_shield_remaining > configured_duration
                for row in self.spawn_pads
            ):
                raise ValueError(
                    "spawn-shield remaining duration exceeds configured duration."
                )
        if tuple(row.team_index for row in self.respawn_waves) != (0, 1):
            raise ValueError("respawn waves must retain ordered team indices.")


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayIncomingAuthorizedAgentIdentityV1:
    """One configured-active Oracle identity, detached from internal slot axes."""

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG

    identity_kind: Literal["authorized_agent"]
    presentation_key: str
    public_agent_id: str

    def __post_init__(self) -> None:
        if self.identity_kind != "authorized_agent":
            raise ValueError("unknown authorized incoming agent identity.")
        _require_text(self.presentation_key, name="presentation_key")
        _require_text(self.public_agent_id, name="public_agent_id")


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayIncomingFeedOnlyAgentIdentityV1:
    """One configured-inactive rejection identity with no invented scene body."""

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG

    identity_kind: Literal["inactive_feed_only"]
    public_agent_id: str

    def __post_init__(self) -> None:
        if self.identity_kind != "inactive_feed_only":
            raise ValueError("unknown feed-only incoming agent identity.")
        _require_text(self.public_agent_id, name="public_agent_id")


type ReplayIncomingAgentIdentityV1 = Annotated[
    ReplayIncomingAuthorizedAgentIdentityV1 | ReplayIncomingFeedOnlyAgentIdentityV1,
    Field(discriminator="identity_kind"),
]


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayIncomingAgentAnchorV1:
    """One authority-neutral actor anchor at an exact scientific phase."""

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG

    phase: ReplayIncomingAnchorPhaseV1
    presentation_key: str
    public_agent_id: str
    position: Point2D

    def __post_init__(self) -> None:
        if self.phase not in ("transition_start", "post_charge", "successor"):
            raise ValueError("unknown replay incoming anchor phase.")
        _require_text(self.presentation_key, name="presentation_key")
        _require_text(self.public_agent_id, name="public_agent_id")
        _require_point(self.position, name="position")


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayIncomingTeamAnchorV1:
    """One axis-free team cue at an exact scientific phase."""

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG

    phase: Literal["successor"]
    team_index: int
    team_id: int

    def __post_init__(self) -> None:
        if self.phase != "successor":
            raise ValueError("replay incoming team anchors must use successor phase.")
        _require_python_int(self.team_index, name="team_index")
        if self.team_index not in (0, 1):
            raise ValueError("team_index must be zero or one.")
        _require_python_int(self.team_id, name="team_id", minimum=1)
        if self.team_id != self.team_index + 1:
            raise ValueError("team_id must equal team_index plus one.")


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayIncomingAgentPhaseTrajectoryV1:
    """One ordered, slot-free start/post-Charge/successor trajectory."""

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG

    agent_presentation_key: str
    agent_public_agent_id: str
    transition_start: ReplayIncomingAgentAnchorV1
    post_charge: ReplayIncomingAgentAnchorV1
    successor: ReplayIncomingAgentAnchorV1

    def __post_init__(self) -> None:
        _require_text(
            self.agent_presentation_key,
            name="agent_presentation_key",
        )
        _require_text(self.agent_public_agent_id, name="agent_public_agent_id")
        for name, phase in (
            ("transition_start", "transition_start"),
            ("post_charge", "post_charge"),
            ("successor", "successor"),
        ):
            anchor = cast(ReplayIncomingAgentAnchorV1, getattr(self, name))
            if type(anchor) is not ReplayIncomingAgentAnchorV1 or (
                anchor.phase != phase
                or anchor.presentation_key != self.agent_presentation_key
                or anchor.public_agent_id != self.agent_public_agent_id
            ):
                raise ValueError(
                    "incoming trajectory anchors must retain one identity and phase."
                )


@dataclass(frozen=True, slots=True, kw_only=True)
class _ReplayIncomingEventBaseV1:
    """Shared strict identity fields for one neutral atomic event."""

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG

    event_id: str
    ordinal: int
    phase_rank: int

    def _validate_base(
        self,
        *,
        event_kind: ReplayIncomingEventKindV1,
        expected_kind: ReplayIncomingEventKindV1,
        expected_phase_rank: int,
    ) -> None:
        _require_text(self.event_id, name="event_id")
        _require_python_int(self.ordinal, name="ordinal")
        _require_python_int(self.phase_rank, name="phase_rank")
        if event_kind != expected_kind or self.phase_rank != expected_phase_rank:
            raise ValueError(
                "incoming event kind and phase rank must remain canonical."
            )


def _require_incoming_anchor(
    value: ReplayIncomingAgentAnchorV1,
    *,
    name: str,
    phase: ReplayIncomingAnchorPhaseV1,
) -> None:
    if type(value) is not ReplayIncomingAgentAnchorV1 or value.phase != phase:
        raise ValueError(f"{name} must be an exact {phase} incoming anchor.")


def _require_optional_incoming_anchor(
    value: ReplayIncomingAgentAnchorV1 | None,
    *,
    name: str,
    phase: ReplayIncomingAnchorPhaseV1,
) -> None:
    if value is not None:
        _require_incoming_anchor(value, name=name, phase=phase)


def _require_incoming_anchor_tuple(
    value: tuple[ReplayIncomingAgentAnchorV1, ...],
    *,
    name: str,
    phase: ReplayIncomingAnchorPhaseV1,
) -> None:
    _require_exact_tuple(value, name=name, item_type=ReplayIncomingAgentAnchorV1)
    for anchor in value:
        _require_incoming_anchor(anchor, name=f"{name} item", phase=phase)
    keys = tuple(anchor.presentation_key for anchor in value)
    if len(keys) != len(set(keys)):
        raise ValueError(f"{name} must contain unique authorized emitters.")


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayIncomingActionRejectedEventV1(_ReplayIncomingEventBaseV1):
    event_kind: Literal["action_rejected"]
    actor_identity: ReplayIncomingAgentIdentityV1
    actor_configured_active: bool
    rejection_component: Literal["domain", "movement", "combat_pair"]
    submitted_action: SubmittedActionTupleV1
    actor_anchor: ReplayIncomingAgentAnchorV1 | None

    def __post_init__(self) -> None:
        self._validate_base(
            event_kind=self.event_kind,
            expected_kind="action_rejected",
            expected_phase_rank=10,
        )
        if type(self.actor_configured_active) is not bool:
            raise ValueError("actor_configured_active must be a Python bool.")
        if self.rejection_component not in ("domain", "movement", "combat_pair"):
            raise ValueError("unknown action rejection component.")
        if type(self.submitted_action) is not SubmittedActionTupleV1:
            raise ValueError("rejected action must retain its exact submitted tuple.")
        if self.actor_configured_active:
            if type(self.actor_identity) is not ReplayIncomingAuthorizedAgentIdentityV1:
                raise ValueError("active rejection requires an authorized identity.")
            _require_incoming_anchor(
                cast(ReplayIncomingAgentAnchorV1, self.actor_anchor),
                name="actor_anchor",
                phase="transition_start",
            )
            identity = self.actor_identity
            anchor = cast(ReplayIncomingAgentAnchorV1, self.actor_anchor)
            if (
                anchor.presentation_key != identity.presentation_key
                or anchor.public_agent_id != identity.public_agent_id
            ):
                raise ValueError("rejected actor identity must join its start anchor.")
        elif (
            type(self.actor_identity) is not ReplayIncomingFeedOnlyAgentIdentityV1
            or self.actor_anchor is not None
        ):
            raise ValueError("inactive rejection must remain feed-only and unanchored.")


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayIncomingAbilityActivatedEventV1(_ReplayIncomingEventBaseV1):
    event_kind: Literal["ability_activated"]
    ability_component: Literal["basic", "ultimate"]
    source_anchor: ReplayIncomingAgentAnchorV1
    recipient_anchor: ReplayIncomingAgentAnchorV1 | None

    def __post_init__(self) -> None:
        self._validate_base(
            event_kind=self.event_kind,
            expected_kind="ability_activated",
            expected_phase_rank=20,
        )
        if self.ability_component not in ("basic", "ultimate"):
            raise ValueError("ability component must be basic or ultimate.")
        _require_incoming_anchor(
            self.source_anchor,
            name="source_anchor",
            phase="transition_start",
        )
        _require_optional_incoming_anchor(
            self.recipient_anchor,
            name="recipient_anchor",
            phase="transition_start",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayIncomingSourceDamageOutputEventV1(_ReplayIncomingEventBaseV1):
    event_kind: Literal["source_damage_output"]
    source_anchor: ReplayIncomingAgentAnchorV1
    recipient_anchor: ReplayIncomingAgentAnchorV1 | None
    raw_damage_output: float
    source_modified_damage_output: float
    recipient_damage_modifier: float
    mage_damage_aura_covering_emitters: tuple[ReplayIncomingAgentAnchorV1, ...]
    warrior_mitigation_aura_covering_emitters: tuple[ReplayIncomingAgentAnchorV1, ...]

    def __post_init__(self) -> None:
        self._validate_base(
            event_kind=self.event_kind,
            expected_kind="source_damage_output",
            expected_phase_rank=30,
        )
        _require_incoming_anchor(
            self.source_anchor,
            name="source_anchor",
            phase="transition_start",
        )
        _require_optional_incoming_anchor(
            self.recipient_anchor,
            name="recipient_anchor",
            phase="transition_start",
        )
        for name in (
            "raw_damage_output",
            "source_modified_damage_output",
            "recipient_damage_modifier",
        ):
            _require_finite(cast(float, getattr(self, name)), name=name, minimum=0.0)
        _require_incoming_anchor_tuple(
            self.mage_damage_aura_covering_emitters,
            name="mage_damage_aura_covering_emitters",
            phase="transition_start",
        )
        _require_incoming_anchor_tuple(
            self.warrior_mitigation_aura_covering_emitters,
            name="warrior_mitigation_aura_covering_emitters",
            phase="transition_start",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayIncomingSourceHealingOutputEventV1(_ReplayIncomingEventBaseV1):
    event_kind: Literal["source_healing_output"]
    source_anchor: ReplayIncomingAgentAnchorV1
    recipient_anchor: ReplayIncomingAgentAnchorV1 | None
    raw_healing_output: float
    source_modified_healing_output: float
    recipient_healing_modifier: float

    def __post_init__(self) -> None:
        self._validate_base(
            event_kind=self.event_kind,
            expected_kind="source_healing_output",
            expected_phase_rank=30,
        )
        _require_incoming_anchor(
            self.source_anchor,
            name="source_anchor",
            phase="transition_start",
        )
        _require_optional_incoming_anchor(
            self.recipient_anchor,
            name="recipient_anchor",
            phase="transition_start",
        )
        for name in (
            "raw_healing_output",
            "source_modified_healing_output",
            "recipient_healing_modifier",
        ):
            _require_finite(cast(float, getattr(self, name)), name=name, minimum=0.0)


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayIncomingRecipientHealthResolutionEventV1(_ReplayIncomingEventBaseV1):
    event_kind: Literal["recipient_health_resolution"]
    recipient_anchor: ReplayIncomingAgentAnchorV1
    transition_start_health: float
    total_effective_damage: float
    total_effective_healing: float
    health_after_combat_resolution: float
    realized_net_health_change: float

    def __post_init__(self) -> None:
        self._validate_base(
            event_kind=self.event_kind,
            expected_kind="recipient_health_resolution",
            expected_phase_rank=40,
        )
        _require_incoming_anchor(
            self.recipient_anchor,
            name="recipient_anchor",
            phase="transition_start",
        )
        for name in (
            "transition_start_health",
            "total_effective_damage",
            "total_effective_healing",
            "health_after_combat_resolution",
        ):
            _require_finite(cast(float, getattr(self, name)), name=name, minimum=0.0)
        _require_finite(
            self.realized_net_health_change,
            name="realized_net_health_change",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayIncomingCombatCountdownResetEventV1(_ReplayIncomingEventBaseV1):
    event_kind: Literal["combat_countdown_reset"]
    agent_anchor: ReplayIncomingAgentAnchorV1

    def __post_init__(self) -> None:
        self._validate_base(
            event_kind=self.event_kind,
            expected_kind="combat_countdown_reset",
            expected_phase_rank=50,
        )
        _require_incoming_anchor(
            self.agent_anchor,
            name="agent_anchor",
            phase="transition_start",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayIncomingHealthRegeneratedEventV1(_ReplayIncomingEventBaseV1):
    event_kind: Literal["health_regenerated"]
    agent_anchor: ReplayIncomingAgentAnchorV1
    actual_health_regenerated: float

    def __post_init__(self) -> None:
        self._validate_base(
            event_kind=self.event_kind,
            expected_kind="health_regenerated",
            expected_phase_rank=50,
        )
        _require_incoming_anchor(
            self.agent_anchor,
            name="agent_anchor",
            phase="transition_start",
        )
        _require_finite(
            self.actual_health_regenerated,
            name="actual_health_regenerated",
            minimum=0.0,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayIncomingCooldownStartedEventV1(_ReplayIncomingEventBaseV1):
    event_kind: Literal["cooldown_started"]
    agent_anchor: ReplayIncomingAgentAnchorV1

    def __post_init__(self) -> None:
        self._validate_base(
            event_kind=self.event_kind,
            expected_kind="cooldown_started",
            expected_phase_rank=60,
        )
        _require_incoming_anchor(
            self.agent_anchor,
            name="agent_anchor",
            phase="transition_start",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayIncomingCooldownReadyEventV1(_ReplayIncomingEventBaseV1):
    event_kind: Literal["cooldown_ready"]
    agent_anchor: ReplayIncomingAgentAnchorV1

    def __post_init__(self) -> None:
        self._validate_base(
            event_kind=self.event_kind,
            expected_kind="cooldown_ready",
            expected_phase_rank=60,
        )
        _require_incoming_anchor(
            self.agent_anchor,
            name="agent_anchor",
            phase="transition_start",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayIncomingChargePhaseDisplacementEventV1(_ReplayIncomingEventBaseV1):
    event_kind: Literal["charge_phase_displacement"]
    realized_displacement: Point2D
    start_anchor: ReplayIncomingAgentAnchorV1
    end_anchor: ReplayIncomingAgentAnchorV1

    def __post_init__(self) -> None:
        self._validate_base(
            event_kind=self.event_kind,
            expected_kind="charge_phase_displacement",
            expected_phase_rank=70,
        )
        _require_point(self.realized_displacement, name="realized_displacement")
        _require_incoming_anchor(
            self.start_anchor,
            name="start_anchor",
            phase="transition_start",
        )
        _require_incoming_anchor(
            self.end_anchor,
            name="end_anchor",
            phase="post_charge",
        )
        if (
            self.start_anchor.presentation_key != self.end_anchor.presentation_key
            or self.start_anchor.public_agent_id != self.end_anchor.public_agent_id
        ):
            raise ValueError("Charge anchors must retain one authorized identity.")
        expected_end = (
            self.start_anchor.position[0] + self.realized_displacement[0],
            self.start_anchor.position[1] + self.realized_displacement[1],
        )
        if not _points_close(self.end_anchor.position, expected_end):
            raise ValueError("Charge end anchor must apply its displacement.")


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayIncomingOrdinaryMovementPhaseDisplacementEventV1(
    _ReplayIncomingEventBaseV1
):
    event_kind: Literal["ordinary_movement_phase_displacement"]
    realized_displacement: Point2D
    start_anchor: ReplayIncomingAgentAnchorV1
    end_anchor: ReplayIncomingAgentAnchorV1

    def __post_init__(self) -> None:
        self._validate_base(
            event_kind=self.event_kind,
            expected_kind="ordinary_movement_phase_displacement",
            expected_phase_rank=80,
        )
        _require_point(self.realized_displacement, name="realized_displacement")
        _require_incoming_anchor(
            self.start_anchor,
            name="start_anchor",
            phase="post_charge",
        )
        _require_incoming_anchor(
            self.end_anchor,
            name="end_anchor",
            phase="successor",
        )
        if (
            self.start_anchor.presentation_key != self.end_anchor.presentation_key
            or self.start_anchor.public_agent_id != self.end_anchor.public_agent_id
        ):
            raise ValueError("movement anchors must retain one authorized identity.")
        expected_end = (
            self.start_anchor.position[0] + self.realized_displacement[0],
            self.start_anchor.position[1] + self.realized_displacement[1],
        )
        if not _points_close(self.end_anchor.position, expected_end):
            raise ValueError("movement end anchor must apply its displacement.")


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayIncomingAgentDiedEventV1(_ReplayIncomingEventBaseV1):
    event_kind: Literal["agent_died"]
    recipient_anchor: ReplayIncomingAgentAnchorV1

    def __post_init__(self) -> None:
        self._validate_base(
            event_kind=self.event_kind,
            expected_kind="agent_died",
            expected_phase_rank=90,
        )
        _require_incoming_anchor(
            self.recipient_anchor,
            name="recipient_anchor",
            phase="successor",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayIncomingLethalDamageContributionEventV1(_ReplayIncomingEventBaseV1):
    event_kind: Literal["lethal_damage_contribution"]
    source_anchor: ReplayIncomingAgentAnchorV1
    recipient_anchor: ReplayIncomingAgentAnchorV1
    attributed_death_damage: float

    def __post_init__(self) -> None:
        self._validate_base(
            event_kind=self.event_kind,
            expected_kind="lethal_damage_contribution",
            expected_phase_rank=90,
        )
        _require_incoming_anchor(
            self.source_anchor,
            name="source_anchor",
            phase="successor",
        )
        _require_incoming_anchor(
            self.recipient_anchor,
            name="recipient_anchor",
            phase="successor",
        )
        _require_finite(
            self.attributed_death_damage,
            name="attributed_death_damage",
            minimum=0.0,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class _ReplayIncomingStatusEventBaseV1(_ReplayIncomingEventBaseV1):
    recipient_anchor: ReplayIncomingAgentAnchorV1
    status_channel: int
    status_id: str

    def _validate_status(
        self,
        *,
        event_kind: ReplayIncomingEventKindV1,
        expected_kind: ReplayIncomingEventKindV1,
    ) -> None:
        self._validate_base(
            event_kind=event_kind,
            expected_kind=expected_kind,
            expected_phase_rank=100,
        )
        _require_incoming_anchor(
            self.recipient_anchor,
            name="recipient_anchor",
            phase="successor",
        )
        _require_python_int(self.status_channel, name="status_channel")
        _require_text(self.status_id, name="status_id")
        if (
            self.status_channel >= len(CATALOG_STATUS_ID_BY_CHANNEL)
            or CATALOG_STATUS_ID_BY_CHANNEL[self.status_channel] != self.status_id
        ):
            raise ValueError("incoming status channel and ID must retain V1 identity.")


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayIncomingStatusAgedToZeroEventV1(_ReplayIncomingStatusEventBaseV1):
    event_kind: Literal["status_aged_to_zero"]

    def __post_init__(self) -> None:
        self._validate_status(
            event_kind=self.event_kind,
            expected_kind="status_aged_to_zero",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayIncomingStatusBrokenByDamageEventV1(_ReplayIncomingStatusEventBaseV1):
    event_kind: Literal["status_broken_by_damage"]

    def __post_init__(self) -> None:
        self._validate_status(
            event_kind=self.event_kind,
            expected_kind="status_broken_by_damage",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayIncomingStatusAppliedEventV1(_ReplayIncomingStatusEventBaseV1):
    event_kind: Literal["status_applied"]
    source_anchor: ReplayIncomingAgentAnchorV1

    def __post_init__(self) -> None:
        self._validate_status(
            event_kind=self.event_kind,
            expected_kind="status_applied",
        )
        _require_incoming_anchor(
            self.source_anchor,
            name="source_anchor",
            phase="successor",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayIncomingStatusRefreshedOrExtendedEventV1(_ReplayIncomingStatusEventBaseV1):
    event_kind: Literal["status_refreshed_or_extended"]

    def __post_init__(self) -> None:
        self._validate_status(
            event_kind=self.event_kind,
            expected_kind="status_refreshed_or_extended",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayIncomingStatusClearedByNewDeathEventV1(_ReplayIncomingStatusEventBaseV1):
    event_kind: Literal["status_cleared_by_new_death"]

    def __post_init__(self) -> None:
        self._validate_status(
            event_kind=self.event_kind,
            expected_kind="status_cleared_by_new_death",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayIncomingSpawnShieldExpiredEventV1(_ReplayIncomingEventBaseV1):
    event_kind: Literal["spawn_shield_expired"]
    agent_anchor: ReplayIncomingAgentAnchorV1

    def __post_init__(self) -> None:
        self._validate_base(
            event_kind=self.event_kind,
            expected_kind="spawn_shield_expired",
            expected_phase_rank=110,
        )
        _require_incoming_anchor(
            self.agent_anchor,
            name="agent_anchor",
            phase="successor",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayIncomingRespawnWaveOccurredEventV1(_ReplayIncomingEventBaseV1):
    event_kind: Literal["respawn_wave_occurred"]
    team_anchor: ReplayIncomingTeamAnchorV1

    def __post_init__(self) -> None:
        self._validate_base(
            event_kind=self.event_kind,
            expected_kind="respawn_wave_occurred",
            expected_phase_rank=120,
        )
        if type(self.team_anchor) is not ReplayIncomingTeamAnchorV1:
            raise ValueError("respawn wave must retain an exact neutral team anchor.")


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayIncomingAgentRespawnedEventV1(_ReplayIncomingEventBaseV1):
    event_kind: Literal["agent_respawned"]
    agent_anchor: ReplayIncomingAgentAnchorV1
    team_id: int
    realized_successor_position: Point2D

    def __post_init__(self) -> None:
        self._validate_base(
            event_kind=self.event_kind,
            expected_kind="agent_respawned",
            expected_phase_rank=120,
        )
        _require_incoming_anchor(
            self.agent_anchor,
            name="agent_anchor",
            phase="successor",
        )
        _require_python_int(self.team_id, name="team_id", minimum=1)
        if self.team_id not in (1, 2):
            raise ValueError("team_id must be one or two.")
        _require_point(
            self.realized_successor_position,
            name="realized_successor_position",
        )
        if self.agent_anchor.position != self.realized_successor_position:
            raise ValueError("respawn position must equal its successor anchor.")


type ReplayIncomingEventV1 = Annotated[
    ReplayIncomingActionRejectedEventV1
    | ReplayIncomingAbilityActivatedEventV1
    | ReplayIncomingSourceDamageOutputEventV1
    | ReplayIncomingSourceHealingOutputEventV1
    | ReplayIncomingRecipientHealthResolutionEventV1
    | ReplayIncomingCombatCountdownResetEventV1
    | ReplayIncomingHealthRegeneratedEventV1
    | ReplayIncomingCooldownStartedEventV1
    | ReplayIncomingCooldownReadyEventV1
    | ReplayIncomingChargePhaseDisplacementEventV1
    | ReplayIncomingOrdinaryMovementPhaseDisplacementEventV1
    | ReplayIncomingAgentDiedEventV1
    | ReplayIncomingLethalDamageContributionEventV1
    | ReplayIncomingStatusAgedToZeroEventV1
    | ReplayIncomingStatusBrokenByDamageEventV1
    | ReplayIncomingStatusAppliedEventV1
    | ReplayIncomingStatusRefreshedOrExtendedEventV1
    | ReplayIncomingStatusClearedByNewDeathEventV1
    | ReplayIncomingSpawnShieldExpiredEventV1
    | ReplayIncomingRespawnWaveOccurredEventV1
    | ReplayIncomingAgentRespawnedEventV1,
    Field(discriminator="event_kind"),
]


_REPLAY_INCOMING_EVENT_TYPES_V1: tuple[type[object], ...] = (
    ReplayIncomingActionRejectedEventV1,
    ReplayIncomingAbilityActivatedEventV1,
    ReplayIncomingSourceDamageOutputEventV1,
    ReplayIncomingSourceHealingOutputEventV1,
    ReplayIncomingRecipientHealthResolutionEventV1,
    ReplayIncomingCombatCountdownResetEventV1,
    ReplayIncomingHealthRegeneratedEventV1,
    ReplayIncomingCooldownStartedEventV1,
    ReplayIncomingCooldownReadyEventV1,
    ReplayIncomingChargePhaseDisplacementEventV1,
    ReplayIncomingOrdinaryMovementPhaseDisplacementEventV1,
    ReplayIncomingAgentDiedEventV1,
    ReplayIncomingLethalDamageContributionEventV1,
    ReplayIncomingStatusAgedToZeroEventV1,
    ReplayIncomingStatusBrokenByDamageEventV1,
    ReplayIncomingStatusAppliedEventV1,
    ReplayIncomingStatusRefreshedOrExtendedEventV1,
    ReplayIncomingStatusClearedByNewDeathEventV1,
    ReplayIncomingSpawnShieldExpiredEventV1,
    ReplayIncomingRespawnWaveOccurredEventV1,
    ReplayIncomingAgentRespawnedEventV1,
)


def _incoming_event_agent_anchors(
    event: ReplayIncomingEventV1,
) -> tuple[ReplayIncomingAgentAnchorV1, ...]:
    if type(event) is ReplayIncomingActionRejectedEventV1:
        return () if event.actor_anchor is None else (event.actor_anchor,)
    if type(event) in (
        ReplayIncomingAbilityActivatedEventV1,
        ReplayIncomingSourceHealingOutputEventV1,
    ):
        source_recipient_event = cast(
            ReplayIncomingAbilityActivatedEventV1
            | ReplayIncomingSourceHealingOutputEventV1,
            event,
        )
        source_anchor = source_recipient_event.source_anchor
        recipient_anchor = source_recipient_event.recipient_anchor
        return (
            (source_anchor,)
            if recipient_anchor is None
            else (source_anchor, recipient_anchor)
        )
    if type(event) is ReplayIncomingSourceDamageOutputEventV1:
        recipient = () if event.recipient_anchor is None else (event.recipient_anchor,)
        return (
            event.source_anchor,
            *recipient,
            *event.mage_damage_aura_covering_emitters,
            *event.warrior_mitigation_aura_covering_emitters,
        )
    if type(event) is ReplayIncomingRecipientHealthResolutionEventV1:
        return (event.recipient_anchor,)
    if type(event) in (
        ReplayIncomingCombatCountdownResetEventV1,
        ReplayIncomingHealthRegeneratedEventV1,
        ReplayIncomingCooldownStartedEventV1,
        ReplayIncomingCooldownReadyEventV1,
        ReplayIncomingSpawnShieldExpiredEventV1,
        ReplayIncomingAgentRespawnedEventV1,
    ):
        agent_event = cast(
            ReplayIncomingCombatCountdownResetEventV1
            | ReplayIncomingHealthRegeneratedEventV1
            | ReplayIncomingCooldownStartedEventV1
            | ReplayIncomingCooldownReadyEventV1
            | ReplayIncomingSpawnShieldExpiredEventV1
            | ReplayIncomingAgentRespawnedEventV1,
            event,
        )
        return (agent_event.agent_anchor,)
    if type(event) in (
        ReplayIncomingChargePhaseDisplacementEventV1,
        ReplayIncomingOrdinaryMovementPhaseDisplacementEventV1,
    ):
        displacement_event = cast(
            ReplayIncomingChargePhaseDisplacementEventV1
            | ReplayIncomingOrdinaryMovementPhaseDisplacementEventV1,
            event,
        )
        return (displacement_event.start_anchor, displacement_event.end_anchor)
    if type(event) is ReplayIncomingAgentDiedEventV1:
        return (event.recipient_anchor,)
    if type(event) is ReplayIncomingLethalDamageContributionEventV1:
        return (event.source_anchor, event.recipient_anchor)
    if type(event) is ReplayIncomingStatusAppliedEventV1:
        return (event.recipient_anchor, event.source_anchor)
    if type(event) in (
        ReplayIncomingStatusAgedToZeroEventV1,
        ReplayIncomingStatusBrokenByDamageEventV1,
        ReplayIncomingStatusRefreshedOrExtendedEventV1,
        ReplayIncomingStatusClearedByNewDeathEventV1,
    ):
        status_event = cast(
            ReplayIncomingStatusAgedToZeroEventV1
            | ReplayIncomingStatusBrokenByDamageEventV1
            | ReplayIncomingStatusRefreshedOrExtendedEventV1
            | ReplayIncomingStatusClearedByNewDeathEventV1,
            event,
        )
        return (status_event.recipient_anchor,)
    if type(event) is ReplayIncomingRespawnWaveOccurredEventV1:
        return ()
    raise TypeError("unknown replay incoming neutral event variant.")


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayIncomingSummaryV1:
    """Exact incoming identity, neutral trajectories, and atomic payloads."""

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG

    summary_kind: Literal["replay_incoming_inventory"]
    incoming_transition_index: int
    incoming_transition_id: str
    incoming_start_frame_id: str
    incoming_successor_frame_id: str
    incoming_start_simulator_step_count: int
    incoming_successor_simulator_step_count: int
    agent_phase_trajectories: tuple[ReplayIncomingAgentPhaseTrajectoryV1, ...]
    ordered_event_ids: tuple[str, ...]
    ordered_event_kinds: tuple[ReplayIncomingEventKindV1, ...]
    events: tuple[ReplayIncomingEventV1, ...]
    event_count: int

    def __post_init__(self) -> None:
        if self.summary_kind != "replay_incoming_inventory":
            raise ValueError("unknown replay incoming summary kind.")
        _require_python_int(
            self.incoming_transition_index,
            name="incoming_transition_index",
        )
        for name in (
            "incoming_transition_id",
            "incoming_start_frame_id",
            "incoming_successor_frame_id",
        ):
            _require_text(cast(str, getattr(self, name)), name=name)
        transition_suffix = f":transition:{self.incoming_transition_index}"
        if not self.incoming_transition_id.endswith(transition_suffix):
            raise ValueError("incoming transition ID is not canonical for its index.")
        episode_id = self.incoming_transition_id.removesuffix(transition_suffix)
        if not episode_id or (
            self.incoming_start_frame_id
            != f"{episode_id}:frame:{self.incoming_transition_index}"
            or self.incoming_successor_frame_id
            != f"{episode_id}:frame:{self.incoming_transition_index + 1}"
        ):
            raise ValueError("incoming frame IDs must join the transition epoch.")
        for name in (
            "incoming_start_simulator_step_count",
            "incoming_successor_simulator_step_count",
            "event_count",
        ):
            _require_python_int(cast(int, getattr(self, name)), name=name)
        if self.incoming_successor_simulator_step_count != (
            self.incoming_start_simulator_step_count + 1
        ):
            raise ValueError("incoming inventory simulator ticks must be adjacent.")
        if (
            type(self.agent_phase_trajectories) is not tuple
            or type(self.events) is not tuple
            or any(
                type(value) is not ReplayIncomingAgentPhaseTrajectoryV1
                for value in self.agent_phase_trajectories
            )
            or any(
                type(value) not in _REPLAY_INCOMING_EVENT_TYPES_V1
                for value in self.events
            )
        ):
            raise ValueError(
                "incoming trajectories and events must retain exact neutral rows."
            )
        if (
            type(self.ordered_event_ids) is not tuple
            or type(self.ordered_event_kinds) is not tuple
        ):
            raise ValueError("incoming event inventory rows must be Python tuples.")
        if any(type(value) is not str or not value for value in self.ordered_event_ids):
            raise ValueError("incoming event IDs must be non-empty strings.")
        if any(
            type(value) is not str or value not in _REPLAY_INCOMING_EVENT_KINDS_V1
            for value in self.ordered_event_kinds
        ):
            raise ValueError("incoming event kinds must use the canonical V1 union.")
        if (
            self.event_count != len(self.ordered_event_ids)
            or self.event_count != len(self.ordered_event_kinds)
            or self.event_count != len(self.events)
        ):
            raise ValueError("incoming count must equal its inventory and payloads.")
        expected_ids = tuple(
            f"{self.incoming_transition_id}:event:{ordinal:04d}"
            for ordinal in range(self.event_count)
        )
        if self.ordered_event_ids != expected_ids:
            raise ValueError("incoming inventory event IDs must be canonical.")
        if (
            tuple(event.ordinal for event in self.events)
            != tuple(range(self.event_count))
            or tuple(event.event_id for event in self.events) != self.ordered_event_ids
            or tuple(event.event_kind for event in self.events)
            != self.ordered_event_kinds
            or tuple(event.phase_rank for event in self.events)
            != tuple(sorted(event.phase_rank for event in self.events))
        ):
            raise ValueError(
                "incoming payloads must equal the ordered identity/kind inventory."
            )
        trajectory_by_key = {
            trajectory.agent_presentation_key: trajectory
            for trajectory in self.agent_phase_trajectories
        }
        if len(trajectory_by_key) != len(self.agent_phase_trajectories) or len(
            {row.agent_public_agent_id for row in self.agent_phase_trajectories}
        ) != len(self.agent_phase_trajectories):
            raise ValueError("incoming trajectories must retain unique identities.")
        trajectory_order_by_key = {
            trajectory.agent_presentation_key: index
            for index, trajectory in enumerate(self.agent_phase_trajectories)
        }
        for event in self.events:
            for anchor in _incoming_event_agent_anchors(event):
                trajectory = trajectory_by_key.get(anchor.presentation_key)
                if trajectory is None or (
                    trajectory.agent_public_agent_id != anchor.public_agent_id
                    or getattr(trajectory, anchor.phase) != anchor
                ):
                    raise ValueError(
                        "incoming event anchors must join the ordered trajectories."
                    )
            if type(event) is ReplayIncomingSourceDamageOutputEventV1:
                for emitters in (
                    event.mage_damage_aura_covering_emitters,
                    event.warrior_mitigation_aura_covering_emitters,
                ):
                    emitter_order = tuple(
                        trajectory_order_by_key[row.presentation_key]
                        for row in emitters
                    )
                    if emitter_order != tuple(sorted(emitter_order)):
                        raise ValueError(
                            "incoming aura emitters must preserve trajectory order."
                        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SubmittedActionTupleV1:
    """One recorded submitted tuple, including possible out-of-domain integers."""

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG

    move_action: int
    target_action: int
    use_ultimate_action: int

    def __post_init__(self) -> None:
        for name in ("move_action", "target_action", "use_ultimate_action"):
            value = cast(int, getattr(self, name))
            if type(value) is not int or not -(2**31) <= value <= 2**31 - 1:
                raise ValueError(f"{name} must be a signed 32-bit Python int.")


@dataclass(frozen=True, slots=True, kw_only=True)
class AcceptedActionTupleV1:
    """One canonical category-bounded accepted tuple."""

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG

    move_action: int
    target_action: int
    use_ultimate_action: int

    def __post_init__(self) -> None:
        domains = (
            ("move_action", self.move_action, NUM_MOVE_ACTIONS_V1),
            ("target_action", self.target_action, NUM_TARGET_ACTIONS_V1),
            (
                "use_ultimate_action",
                self.use_ultimate_action,
                NUM_ULTIMATE_ACTIONS_V1,
            ),
        )
        for name, value, count in domains:
            if type(value) is not int or not 0 <= value < count:
                raise ValueError(f"{name} must be in [0, {count}).")


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayAcceptedNoTargetV1:
    """Canonical target-none accepted action disclosure."""

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG

    target_kind: Literal["none"]

    def __post_init__(self) -> None:
        if self.target_kind != "none":
            raise ValueError("unknown target-none discriminator.")


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayAcceptedAuthorizedTargetV1:
    """One accepted target joined to the displayed current scene.

    The pure builder owns the actor-relative target-axis join because the HTTP
    envelope intentionally does not duplicate the replay context catalog.
    """

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG

    target_kind: Literal["authorized_agent"]
    target_presentation_key: str
    target_public_agent_id: str
    target_anchor: Point2D

    def __post_init__(self) -> None:
        if self.target_kind != "authorized_agent":
            raise ValueError("unknown authorized-target discriminator.")
        _require_text(self.target_presentation_key, name="target_presentation_key")
        _require_text(self.target_public_agent_id, name="target_public_agent_id")
        _require_point(self.target_anchor, name="target_anchor")


type ReplayAcceptedTargetV1 = Annotated[
    ReplayAcceptedNoTargetV1 | ReplayAcceptedAuthorizedTargetV1,
    Field(discriminator="target_kind"),
]


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayOutgoingInspectionV1:
    """Recorded outgoing intent at s_n, structurally separate from history."""

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG

    inspection_kind: Literal["replay_recorded_outgoing_action"]
    outgoing_transition_index: int
    outgoing_transition_id: str
    outgoing_start_frame_id: str
    outgoing_successor_frame_id: str
    actor_presentation_key: str
    actor_public_agent_id: str
    actor_anchor: Point2D
    submitted_action: SubmittedActionTupleV1
    accepted_action: AcceptedActionTupleV1
    accepted_lane: AcceptedLaneV1
    accepted_target: ReplayAcceptedTargetV1

    def __post_init__(self) -> None:
        if self.inspection_kind != "replay_recorded_outgoing_action":
            raise ValueError("unknown replay outgoing inspection kind.")
        _require_python_int(
            self.outgoing_transition_index,
            name="outgoing_transition_index",
        )
        for name in (
            "outgoing_transition_id",
            "outgoing_start_frame_id",
            "outgoing_successor_frame_id",
            "actor_presentation_key",
            "actor_public_agent_id",
        ):
            _require_text(cast(str, getattr(self, name)), name=name)
        _require_point(self.actor_anchor, name="actor_anchor")
        if type(self.submitted_action) is not SubmittedActionTupleV1:
            raise ValueError("submitted_action must be its exact tuple root.")
        if type(self.accepted_action) is not AcceptedActionTupleV1:
            raise ValueError("accepted_action must be its exact tuple root.")
        expected_lane: AcceptedLaneV1 = (
            "ultimate" if self.accepted_action.use_ultimate_action == 1 else "basic"
        )
        if self.accepted_lane != expected_lane:
            raise ValueError("accepted lane must equal the accepted action head.")
        if self.accepted_action.target_action == 0:
            if type(self.accepted_target) is not ReplayAcceptedNoTargetV1:
                raise ValueError("target action zero requires target-none disclosure.")
        elif type(self.accepted_target) is not ReplayAcceptedAuthorizedTargetV1:
            raise ValueError("positive target action requires an authorized target.")


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayOraclePresentationPartsV1:
    """Authority-neutral siblings packaged by the replay HTTP protocol layer."""

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG

    current_scene: AuthorizedBattlefieldSceneV1
    incoming_summary: ReplayIncomingSummaryV1 | None
    outgoing_inspection: ReplayOutgoingInspectionV1 | None

    def __post_init__(self) -> None:
        if type(self.current_scene) is not AuthorizedBattlefieldSceneV1:
            raise ValueError("current_scene must be its exact neutral root.")
        if (
            self.incoming_summary is not None
            and type(self.incoming_summary) is not ReplayIncomingSummaryV1
        ):
            raise ValueError("incoming_summary must be its exact replay root or None.")
        if self.incoming_summary is not None:
            incoming_identity_and_successor = tuple(
                (
                    row.agent_presentation_key,
                    row.agent_public_agent_id,
                    row.successor.position,
                )
                for row in self.incoming_summary.agent_phase_trajectories
            )
            scene_identity_and_position = tuple(
                (row.presentation_key, row.public_agent_id, row.position)
                for row in self.current_scene.agents
            )
            if incoming_identity_and_successor != scene_identity_and_position:
                raise ValueError(
                    "incoming successor trajectories must join the current scene."
                )
            scene_public_ids = {
                row.public_agent_id for row in self.current_scene.agents
            }
            if any(
                type(event) is ReplayIncomingActionRejectedEventV1
                and type(event.actor_identity) is ReplayIncomingFeedOnlyAgentIdentityV1
                and event.actor_identity.public_agent_id in scene_public_ids
                for event in self.incoming_summary.events
            ):
                raise ValueError(
                    "feed-only rejection identities cannot invent a scene body."
                )
        if (
            self.outgoing_inspection is not None
            and type(self.outgoing_inspection) is not ReplayOutgoingInspectionV1
        ):
            raise ValueError(
                "outgoing_inspection must be its exact replay root or None."
            )


def _authorized_obstacle(obstacle: ObstacleSceneV1) -> AuthorizedObstacleV1:
    return AuthorizedObstacleV1(
        obstacle_id=obstacle.obstacle_id,
        kind=obstacle.kind,
        center=obstacle.center,
        radius=obstacle.radius,
        width=obstacle.width,
        height=obstacle.height,
        theta=obstacle.theta,
    )


def _authorized_map(scene_map: MapSceneV1) -> AuthorizedMapV1:
    return AuthorizedMapV1(
        width=scene_map.width,
        height=scene_map.height,
        obstacles=tuple(_authorized_obstacle(row) for row in scene_map.obstacles),
    )


def _authorized_aura_modifier(
    modifier: AuraRecipientModifierSceneV2,
) -> AuthorizedAuraModifierV1:
    return AuthorizedAuraModifierV1(
        aura_id=modifier.aura_id,
        multiplier=modifier.multiplier,
    )


def _authorized_class_status_mechanic(
    mechanic: ClassStatusMechanicSceneV2,
) -> AuthorizedClassStatusMechanicV1:
    return AuthorizedClassStatusMechanicV1(
        status_channel=mechanic.status_channel,
        status_id=mechanic.status_id,
        family=mechanic.family,
        source_action_component=mechanic.source_action_component,
        duration_steps=mechanic.duration_steps,
        magnitude_kind=mechanic.magnitude_kind,
        magnitude=mechanic.magnitude,
        breaks_on_positive_damage=mechanic.breaks_on_positive_damage,
    )


def _authorized_class_aura_mechanic(
    mechanic: ClassAuraMechanicSceneV2,
) -> AuthorizedClassAuraMechanicV1:
    return AuthorizedClassAuraMechanicV1(
        aura_id=cast(AuthorizedAuraIdV1, mechanic.aura_id),
        radius=mechanic.radius,
        per_emitter_multiplier=mechanic.per_emitter_multiplier,
        stacking_rule=mechanic.stacking_rule,
        clamp_kind=mechanic.clamp_kind,
        clamp_value=mechanic.clamp_value,
    )


def _authorized_class_mechanics(
    mechanics: ClassMechanicsSceneV2,
) -> AuthorizedClassMechanicsV1:
    return AuthorizedClassMechanicsV1(
        class_id=mechanics.class_id,
        class_name=mechanics.class_name,
        maximum_health=mechanics.maximum_health,
        body_radius=mechanics.body_radius,
        base_movement_speed=mechanics.base_movement_speed,
        observation_radius=mechanics.observation_radius,
        basic_target_mode=mechanics.basic_target_mode,
        basic_interaction_radius=mechanics.basic_interaction_radius,
        basic_raw_damage=mechanics.basic_raw_damage,
        basic_raw_healing=mechanics.basic_raw_healing,
        ultimate_target_mode=mechanics.ultimate_target_mode,
        ultimate_interaction_radius=mechanics.ultimate_interaction_radius,
        ultimate_cooldown_steps=mechanics.ultimate_cooldown_steps,
        ultimate_raw_damage=mechanics.ultimate_raw_damage,
        ultimate_raw_healing=mechanics.ultimate_raw_healing,
        out_of_combat_delay_steps=mechanics.out_of_combat_delay_steps,
        out_of_combat_health_regeneration_fraction_per_step=(
            mechanics.out_of_combat_health_regeneration_fraction_per_step
        ),
        status_mechanics=tuple(
            _authorized_class_status_mechanic(row) for row in mechanics.status_mechanics
        ),
        aura_mechanics=tuple(
            _authorized_class_aura_mechanic(row) for row in mechanics.aura_mechanics
        ),
    )


def _authorized_respawn_wave(
    wave: RespawnWaveSceneV2,
) -> AuthorizedRespawnWaveV1:
    return AuthorizedRespawnWaveV1(
        team_index=wave.team_index,
        team_id=wave.team_id,
        period_steps=wave.period_steps,
        countdown_steps=wave.countdown_steps,
    )


def _status_row(
    status: StatusSceneV2,
    *,
    configured_duration_steps: int,
    key_by_internal_slot: dict[int, str],
    public_id_by_internal_slot: dict[int, str],
) -> AuthorizedStatusV1:
    direct_sources: list[AuthorizedStatusSourceV1] = []
    seen_keys: set[str] = set()
    for evidence in status.direct_source_evidence:
        key = key_by_internal_slot.get(evidence.source_global_slot)
        public_id = public_id_by_internal_slot.get(evidence.source_global_slot)
        if key is None or public_id != evidence.source_public_agent_id:
            raise ValueError("status evidence source is absent from authorized scene.")
        if key in seen_keys:
            continue
        seen_keys.add(key)
        if public_id is None:  # pragma: no cover - narrowed by the join above.
            raise AssertionError("authorized status source identity disappeared")
        direct_sources.append(
            AuthorizedStatusSourceV1(
                source_presentation_key=key,
                source_public_agent_id=public_id,
            )
        )
    return AuthorizedStatusV1(
        status_channel=status.status_channel,
        status_id=status.status_id,
        family=status.family,
        configured_duration_steps=configured_duration_steps,
        remaining_duration=status.remaining_duration,
        source_class_id=status.source_class_id,
        source_class_name=status.source_class_name,
        source_action_component=status.source_action_component,
        magnitude_kind=status.magnitude_kind,
        magnitude=status.magnitude,
        breaks_on_positive_damage=status.breaks_on_positive_damage,
        direct_sources=tuple(direct_sources),
    )


def _agent_row(
    agent: AgentSceneV2,
    *,
    mechanics: ClassMechanicsSceneV2,
    status_mechanics_by_channel: dict[int, ClassStatusMechanicSceneV2],
    key_by_internal_slot: dict[int, str],
    public_id_by_internal_slot: dict[int, str],
) -> AuthorizedAgentV1:
    if (
        agent.class_id != mechanics.class_id
        or agent.max_health != mechanics.maximum_health
        or agent.radius != mechanics.body_radius
    ):
        raise ValueError("agent durable facts must join its class mechanics.")
    return AuthorizedAgentV1(
        presentation_key=key_by_internal_slot[agent.global_slot],
        public_agent_id=agent.public_agent_id,
        relation="oracle",
        team_id=agent.team_id,
        class_id=agent.class_id,
        class_name=mechanics.class_name,
        position=agent.position,
        radius=agent.radius,
        life_state=agent.life_state,
        current_health=agent.current_health,
        maximum_health=agent.max_health,
        base_movement_speed=mechanics.base_movement_speed,
        effective_movement_speed=agent.effective_movement_speed,
        observation_radius=mechanics.observation_radius,
        basic_interaction_radius=mechanics.basic_interaction_radius,
        ultimate_interaction_radius=mechanics.ultimate_interaction_radius,
        ultimate_cooldown_remaining=agent.ultimate_cooldown_remaining,
        spawn_shield_remaining=agent.spawn_shield_remaining,
        steps_until_out_of_combat=agent.steps_until_out_of_combat,
        out_of_combat_delay_steps=mechanics.out_of_combat_delay_steps,
        out_of_combat_health_regeneration_fraction_per_step=(
            mechanics.out_of_combat_health_regeneration_fraction_per_step
        ),
        statuses=tuple(
            _status_row(
                status,
                configured_duration_steps=status_mechanics_by_channel[
                    status.status_channel
                ].duration_steps,
                key_by_internal_slot=key_by_internal_slot,
                public_id_by_internal_slot=public_id_by_internal_slot,
            )
            for status in agent.statuses
        ),
        aura_modifiers=tuple(
            _authorized_aura_modifier(modifier)
            for modifier in agent.aura_modifiers
            if modifier.multiplier != 1.0
        ),
    )


def _authorized_scene(
    context: EvaluationEpisodeContextV1,
    scene: BattlefieldSceneV2,
    *,
    authority_session_id: str,
) -> tuple[AuthorizedBattlefieldSceneV1, dict[int, str]]:
    active_roster = tuple(row for row in context.roster if row.configured_active)
    internal_slots = tuple(row.global_slot for row in active_roster)
    if tuple(agent.global_slot for agent in scene.agents) != internal_slots:
        raise ValueError("Oracle scene agents must equal the configured-active roster.")
    for roster, agent in zip(active_roster, scene.agents, strict=True):
        if (
            roster.public_agent_id != agent.public_agent_id
            or roster.configured_team_id != agent.team_id
            or roster.team_local_slot != agent.team_local_slot
            or roster.class_id != agent.class_id
        ):
            raise ValueError("Oracle scene agents must join context roster identity.")
    if tuple(row.class_id for row in scene.class_mechanics) != (1, 2, 3, 4, 5):
        raise ValueError("Oracle source scene must retain the complete V1 class axis.")
    if tuple(
        sorted(
            status.status_channel
            for mechanics in scene.class_mechanics
            for status in mechanics.status_mechanics
        )
    ) != tuple(range(9)):
        raise ValueError("Oracle source scene must retain the complete status axis.")
    if tuple(
        aura.aura_id
        for mechanics in scene.class_mechanics
        for aura in mechanics.aura_mechanics
    ) != tuple(_AURA_SOURCE_CLASS_BY_ID_V1):
        raise ValueError("Oracle source scene must retain the complete aura axis.")
    key_by_internal_slot = {
        row.global_slot: oracle_presentation_key_v1(
            authority_session_id=authority_session_id,
            public_agent_id=row.public_agent_id,
        )
        for row in active_roster
    }
    public_id_by_internal_slot = {
        row.global_slot: row.public_agent_id for row in active_roster
    }
    mechanics_by_class = {row.class_id: row for row in scene.class_mechanics}
    if any(agent.class_id not in mechanics_by_class for agent in scene.agents):
        raise ValueError("each Oracle agent must join ordered class mechanics.")
    represented_class_ids = {row.class_id for row in scene.agents}
    status_mechanics_by_channel = {
        status.status_channel: status
        for mechanics in scene.class_mechanics
        for status in mechanics.status_mechanics
    }
    agents = tuple(
        _agent_row(
            agent,
            mechanics=mechanics_by_class[agent.class_id],
            status_mechanics_by_channel=status_mechanics_by_channel,
            key_by_internal_slot=key_by_internal_slot,
            public_id_by_internal_slot=public_id_by_internal_slot,
        )
        for agent in scene.agents
    )
    aura_fields: list[AuthorizedAuraFieldV1] = []
    for field in scene.aura_fields:
        source_key = key_by_internal_slot.get(field.source_global_slot)
        if source_key is None:
            raise ValueError("each Oracle aura field must join an authorized source.")
        aura_fields.append(
            AuthorizedAuraFieldV1(
                aura_id=field.aura_id,
                source_presentation_key=source_key,
                source_public_agent_id=field.source_public_agent_id,
                source_class_id=field.source_class_id,
                source_class_name=field.source_class_name,
                source_alive=field.source_alive,
                center=field.center,
                radius=field.radius,
                beneficiary_relation=field.beneficiary_relation,
                per_emitter_multiplier=field.per_emitter_multiplier,
                stacking_rule=field.stacking_rule,
                clamp_kind=field.clamp_kind,
                clamp_value=field.clamp_value,
            )
        )
    spawn_pads: list[AuthorizedSpawnPadV1] = []
    raw_agent_by_internal_slot = {row.global_slot: row for row in scene.agents}
    for pad in scene.spawn_pads:
        assigned_key = key_by_internal_slot.get(pad.assigned_global_slot)
        assigned_agent = raw_agent_by_internal_slot.get(pad.assigned_global_slot)
        if assigned_key is None or assigned_agent is None:
            raise ValueError("each Oracle spawn pad must join an assignee.")
        spawn_pads.append(
            AuthorizedSpawnPadV1(
                team_id=pad.team_id,
                team_local_slot=pad.team_local_slot,
                assigned_presentation_key=assigned_key,
                assigned_public_agent_id=pad.assigned_public_agent_id,
                position=pad.position,
                configured_active=True,
                currently_alive=assigned_agent.life_state == "alive",
                spawn_shield_remaining=assigned_agent.spawn_shield_remaining,
            )
        )
    return (
        AuthorizedBattlefieldSceneV1(
            schema_version=AUTHORIZED_PRESENTATION_SCHEMA_VERSION,
            map=_authorized_map(scene.map),
            agents=agents,
            aura_fields=tuple(aura_fields),
            class_mechanics=tuple(
                _authorized_class_mechanics(row)
                for row in scene.class_mechanics
                if row.class_id in represented_class_ids
            ),
            spawn_shield_mechanics=AuthorizedSpawnShieldMechanicsAvailableV1(
                availability_kind="available",
                configured_duration_steps=(
                    context.resolved_env_config.spawn_shield_duration_steps
                ),
                movement_speed=(
                    context.resolved_env_config.spawn_shield_movement_speed
                ),
            ),
            spawn_pads=tuple(spawn_pads),
            respawn_waves=tuple(
                _authorized_respawn_wave(row) for row in scene.respawn_waves
            ),
        ),
        key_by_internal_slot,
    )


def build_oracle_authorized_scene_v1(
    context: EvaluationEpisodeContextV1,
    scene: BattlefieldSceneV2,
    *,
    authority_session_id: str,
) -> AuthorizedBattlefieldSceneV1:
    """Project one exact epoch-bearing Oracle scene without adjacent branches."""
    if type(context) is not EvaluationEpisodeContextV1:
        raise TypeError("context must be the exact EvaluationEpisodeContextV1 root.")
    if type(scene) is not BattlefieldSceneV2:
        raise TypeError("scene must be the exact BattlefieldSceneV2 root.")
    _require_text(authority_session_id, name="authority_session_id")
    scene_adapter = TypeAdapter(BattlefieldSceneV2)
    try:
        context_json = context.model_dump_json(warnings="error")
        scene_json = scene_adapter.dump_json(scene, warnings="error")
    except PydanticSerializationError as error:
        raise ValueError(
            "Oracle authority inputs must retain exact runtime wire types."
        ) from error
    validated_context = EvaluationEpisodeContextV1.model_validate_json(context_json)
    validated_scene = scene_adapter.validate_json(scene_json)

    def exact_tree_matches(candidate: object, canonical: object) -> bool:
        if isinstance(canonical, BaseModel):
            if type(candidate) is not type(canonical):
                return False
            candidate_model = candidate
            field_names = set(type(canonical).model_fields)
            return (
                set(candidate_model.__dict__) == field_names
                and not getattr(candidate_model, "__pydantic_extra__", None)
                and not getattr(candidate_model, "__pydantic_private__", None)
                and all(
                    exact_tree_matches(
                        getattr(candidate_model, name),
                        getattr(canonical, name),
                    )
                    for name in type(canonical).model_fields
                )
            )
        if is_dataclass(canonical) and not isinstance(canonical, type):
            return type(candidate) is type(canonical) and all(
                exact_tree_matches(
                    getattr(candidate, field.name),
                    getattr(canonical, field.name),
                )
                for field in fields(canonical)
            )
        if type(canonical) is tuple:
            candidate_tuple = cast(tuple[object, ...], candidate)
            canonical_tuple = cast(tuple[object, ...], canonical)
            return (
                type(candidate) is tuple
                and len(candidate_tuple) == len(canonical_tuple)
                and all(
                    exact_tree_matches(left, right)
                    for left, right in zip(
                        candidate_tuple,
                        canonical_tuple,
                        strict=True,
                    )
                )
            )
        if type(canonical) is dict:
            candidate_dict = cast(dict[object, object], candidate)
            canonical_dict = cast(dict[object, object], canonical)
            return (
                type(candidate) is dict
                and tuple(candidate_dict) == tuple(canonical_dict)
                and all(
                    exact_tree_matches(candidate_dict[key], value)
                    for key, value in canonical_dict.items()
                )
            )
        return type(candidate) is type(canonical) and candidate == canonical

    if not exact_tree_matches(context, validated_context):
        raise ValueError("Oracle context must retain exact runtime wire types.")
    if not exact_tree_matches(scene, validated_scene):
        raise ValueError("Oracle source scene must retain exact runtime wire types.")
    if validated_scene.episode_id != validated_context.identity.episode_id:
        raise ValueError("Oracle scene and context must join one episode.")
    validate_oracle_scene_static_authority_v1(validated_context, validated_scene)
    current_scene, _ = _authorized_scene(
        validated_context,
        validated_scene,
        authority_session_id=authority_session_id,
    )
    return current_scene


def _replay_incoming_anchor(
    anchor: VisualAgentAnchorV2,
    *,
    key_by_internal_slot: dict[int, str],
) -> ReplayIncomingAgentAnchorV1:
    key = key_by_internal_slot.get(anchor.global_slot)
    if key is None:
        raise ValueError("incoming anchor must join an authorized Oracle agent.")
    return ReplayIncomingAgentAnchorV1(
        phase=anchor.phase,
        presentation_key=key,
        public_agent_id=anchor.public_agent_id,
        position=anchor.position,
    )


def _replay_incoming_trajectory(
    trajectory: VisualAgentPhaseTrajectoryV2,
    *,
    key_by_internal_slot: dict[int, str],
) -> ReplayIncomingAgentPhaseTrajectoryV1:
    key = key_by_internal_slot.get(trajectory.global_slot)
    if key is None:
        raise ValueError("incoming trajectory must join an authorized Oracle agent.")
    return ReplayIncomingAgentPhaseTrajectoryV1(
        agent_presentation_key=key,
        agent_public_agent_id=trajectory.public_agent_id,
        transition_start=_replay_incoming_anchor(
            trajectory.transition_start,
            key_by_internal_slot=key_by_internal_slot,
        ),
        post_charge=_replay_incoming_anchor(
            trajectory.post_charge,
            key_by_internal_slot=key_by_internal_slot,
        ),
        successor=_replay_incoming_anchor(
            trajectory.successor,
            key_by_internal_slot=key_by_internal_slot,
        ),
    )


def _replay_incoming_trajectory_anchor(
    internal_slot: int,
    phase: VisualAnchorPhaseV2,
    *,
    trajectory_by_internal_slot: dict[int, VisualAgentPhaseTrajectoryV2],
    key_by_internal_slot: dict[int, str],
) -> ReplayIncomingAgentAnchorV1:
    trajectory = trajectory_by_internal_slot.get(internal_slot)
    if trajectory is None:
        raise ValueError("incoming event identity has no authorized trajectory.")
    return _replay_incoming_anchor(
        cast(VisualAgentAnchorV2, getattr(trajectory, phase)),
        key_by_internal_slot=key_by_internal_slot,
    )


def _replay_incoming_event(
    event: VisualEventV2,
    *,
    trajectory_by_internal_slot: dict[int, VisualAgentPhaseTrajectoryV2],
    key_by_internal_slot: dict[int, str],
) -> ReplayIncomingEventV1:
    if type(event) is ActionRejectedEventV2:
        actor_identity: ReplayIncomingAgentIdentityV1
        if event.actor_configured_active:
            key = key_by_internal_slot.get(event.actor_global_slot)
            if key is None:
                raise ValueError("active rejected actor must join the Oracle scene.")
            actor_identity = ReplayIncomingAuthorizedAgentIdentityV1(
                identity_kind="authorized_agent",
                presentation_key=key,
                public_agent_id=event.actor_public_agent_id,
            )
        else:
            actor_identity = ReplayIncomingFeedOnlyAgentIdentityV1(
                identity_kind="inactive_feed_only",
                public_agent_id=event.actor_public_agent_id,
            )
        return ReplayIncomingActionRejectedEventV1(
            event_id=event.event_id,
            ordinal=event.ordinal,
            phase_rank=event.phase_rank,
            event_kind=event.event_type,
            actor_identity=actor_identity,
            actor_configured_active=event.actor_configured_active,
            rejection_component=event.rejection_component,
            submitted_action=SubmittedActionTupleV1(
                move_action=event.submitted_move_action,
                target_action=event.submitted_select_target_action,
                use_ultimate_action=event.submitted_use_ultimate_action,
            ),
            actor_anchor=(
                None
                if event.actor_anchor is None
                else _replay_incoming_anchor(
                    event.actor_anchor,
                    key_by_internal_slot=key_by_internal_slot,
                )
            ),
        )
    if type(event) is AbilityActivatedEventV2:
        return ReplayIncomingAbilityActivatedEventV1(
            event_id=event.event_id,
            ordinal=event.ordinal,
            phase_rank=event.phase_rank,
            event_kind=event.event_type,
            ability_component=event.ability_component,
            source_anchor=_replay_incoming_anchor(
                event.source_anchor,
                key_by_internal_slot=key_by_internal_slot,
            ),
            recipient_anchor=(
                None
                if event.recipient_anchor is None
                else _replay_incoming_anchor(
                    event.recipient_anchor,
                    key_by_internal_slot=key_by_internal_slot,
                )
            ),
        )
    if type(event) is SourceDamageOutputEventV2:
        return ReplayIncomingSourceDamageOutputEventV1(
            event_id=event.event_id,
            ordinal=event.ordinal,
            phase_rank=event.phase_rank,
            event_kind=event.event_type,
            source_anchor=_replay_incoming_anchor(
                event.source_anchor,
                key_by_internal_slot=key_by_internal_slot,
            ),
            recipient_anchor=(
                None
                if event.recipient_anchor is None
                else _replay_incoming_anchor(
                    event.recipient_anchor,
                    key_by_internal_slot=key_by_internal_slot,
                )
            ),
            raw_damage_output=event.raw_damage_output,
            source_modified_damage_output=event.source_modified_damage_output,
            recipient_damage_modifier=event.recipient_damage_modifier,
            mage_damage_aura_covering_emitters=tuple(
                _replay_incoming_trajectory_anchor(
                    slot,
                    "transition_start",
                    trajectory_by_internal_slot=trajectory_by_internal_slot,
                    key_by_internal_slot=key_by_internal_slot,
                )
                for slot in event.mage_damage_aura_covering_emitter_global_slots
            ),
            warrior_mitigation_aura_covering_emitters=tuple(
                _replay_incoming_trajectory_anchor(
                    slot,
                    "transition_start",
                    trajectory_by_internal_slot=trajectory_by_internal_slot,
                    key_by_internal_slot=key_by_internal_slot,
                )
                for slot in event.warrior_mitigation_aura_covering_emitter_global_slots
            ),
        )
    if type(event) is SourceHealingOutputEventV2:
        return ReplayIncomingSourceHealingOutputEventV1(
            event_id=event.event_id,
            ordinal=event.ordinal,
            phase_rank=event.phase_rank,
            event_kind=event.event_type,
            source_anchor=_replay_incoming_anchor(
                event.source_anchor,
                key_by_internal_slot=key_by_internal_slot,
            ),
            recipient_anchor=(
                None
                if event.recipient_anchor is None
                else _replay_incoming_anchor(
                    event.recipient_anchor,
                    key_by_internal_slot=key_by_internal_slot,
                )
            ),
            raw_healing_output=event.raw_healing_output,
            source_modified_healing_output=event.source_modified_healing_output,
            recipient_healing_modifier=event.recipient_healing_modifier,
        )
    if type(event) is RecipientHealthResolutionEventV2:
        return ReplayIncomingRecipientHealthResolutionEventV1(
            event_id=event.event_id,
            ordinal=event.ordinal,
            phase_rank=event.phase_rank,
            event_kind=event.event_type,
            recipient_anchor=_replay_incoming_anchor(
                event.recipient_anchor,
                key_by_internal_slot=key_by_internal_slot,
            ),
            transition_start_health=event.transition_start_health,
            total_effective_damage=event.total_effective_damage,
            total_effective_healing=event.total_effective_healing,
            health_after_combat_resolution=event.health_after_combat_resolution,
            realized_net_health_change=event.realized_net_health_change,
        )
    if type(event) is CombatCountdownResetEventV2:
        return ReplayIncomingCombatCountdownResetEventV1(
            event_id=event.event_id,
            ordinal=event.ordinal,
            phase_rank=event.phase_rank,
            event_kind=event.event_type,
            agent_anchor=_replay_incoming_anchor(
                event.agent_anchor,
                key_by_internal_slot=key_by_internal_slot,
            ),
        )
    if type(event) is HealthRegeneratedEventV2:
        return ReplayIncomingHealthRegeneratedEventV1(
            event_id=event.event_id,
            ordinal=event.ordinal,
            phase_rank=event.phase_rank,
            event_kind=event.event_type,
            agent_anchor=_replay_incoming_anchor(
                event.agent_anchor,
                key_by_internal_slot=key_by_internal_slot,
            ),
            actual_health_regenerated=event.actual_health_regenerated,
        )
    if type(event) is CooldownStartedEventV2:
        return ReplayIncomingCooldownStartedEventV1(
            event_id=event.event_id,
            ordinal=event.ordinal,
            phase_rank=event.phase_rank,
            event_kind=event.event_type,
            agent_anchor=_replay_incoming_anchor(
                event.agent_anchor,
                key_by_internal_slot=key_by_internal_slot,
            ),
        )
    if type(event) is CooldownReadyEventV2:
        return ReplayIncomingCooldownReadyEventV1(
            event_id=event.event_id,
            ordinal=event.ordinal,
            phase_rank=event.phase_rank,
            event_kind=event.event_type,
            agent_anchor=_replay_incoming_anchor(
                event.agent_anchor,
                key_by_internal_slot=key_by_internal_slot,
            ),
        )
    if type(event) is ChargePhaseDisplacementEventV2:
        return ReplayIncomingChargePhaseDisplacementEventV1(
            event_id=event.event_id,
            ordinal=event.ordinal,
            phase_rank=event.phase_rank,
            event_kind=event.event_type,
            realized_displacement=event.realized_displacement,
            start_anchor=_replay_incoming_anchor(
                event.start_anchor,
                key_by_internal_slot=key_by_internal_slot,
            ),
            end_anchor=_replay_incoming_anchor(
                event.end_anchor,
                key_by_internal_slot=key_by_internal_slot,
            ),
        )
    if type(event) is OrdinaryMovementPhaseDisplacementEventV2:
        return ReplayIncomingOrdinaryMovementPhaseDisplacementEventV1(
            event_id=event.event_id,
            ordinal=event.ordinal,
            phase_rank=event.phase_rank,
            event_kind=event.event_type,
            realized_displacement=event.realized_displacement,
            start_anchor=_replay_incoming_anchor(
                event.start_anchor,
                key_by_internal_slot=key_by_internal_slot,
            ),
            end_anchor=_replay_incoming_anchor(
                event.end_anchor,
                key_by_internal_slot=key_by_internal_slot,
            ),
        )
    if type(event) is AgentDiedEventV2:
        return ReplayIncomingAgentDiedEventV1(
            event_id=event.event_id,
            ordinal=event.ordinal,
            phase_rank=event.phase_rank,
            event_kind=event.event_type,
            recipient_anchor=_replay_incoming_anchor(
                event.recipient_anchor,
                key_by_internal_slot=key_by_internal_slot,
            ),
        )
    if type(event) is LethalDamageContributionEventV2:
        return ReplayIncomingLethalDamageContributionEventV1(
            event_id=event.event_id,
            ordinal=event.ordinal,
            phase_rank=event.phase_rank,
            event_kind=event.event_type,
            source_anchor=_replay_incoming_anchor(
                event.source_anchor,
                key_by_internal_slot=key_by_internal_slot,
            ),
            recipient_anchor=_replay_incoming_anchor(
                event.recipient_anchor,
                key_by_internal_slot=key_by_internal_slot,
            ),
            attributed_death_damage=event.attributed_death_damage,
        )
    if type(event) is StatusAgedToZeroEventV2:
        return ReplayIncomingStatusAgedToZeroEventV1(
            event_id=event.event_id,
            ordinal=event.ordinal,
            phase_rank=event.phase_rank,
            event_kind=event.event_type,
            recipient_anchor=_replay_incoming_anchor(
                event.recipient_anchor,
                key_by_internal_slot=key_by_internal_slot,
            ),
            status_channel=event.status_channel,
            status_id=event.status_id,
        )
    if type(event) is StatusBrokenByDamageEventV2:
        return ReplayIncomingStatusBrokenByDamageEventV1(
            event_id=event.event_id,
            ordinal=event.ordinal,
            phase_rank=event.phase_rank,
            event_kind=event.event_type,
            recipient_anchor=_replay_incoming_anchor(
                event.recipient_anchor,
                key_by_internal_slot=key_by_internal_slot,
            ),
            status_channel=event.status_channel,
            status_id=event.status_id,
        )
    if type(event) is StatusAppliedEventV2:
        return ReplayIncomingStatusAppliedEventV1(
            event_id=event.event_id,
            ordinal=event.ordinal,
            phase_rank=event.phase_rank,
            event_kind=event.event_type,
            recipient_anchor=_replay_incoming_anchor(
                event.recipient_anchor,
                key_by_internal_slot=key_by_internal_slot,
            ),
            status_channel=event.status_channel,
            status_id=event.status_id,
            source_anchor=_replay_incoming_anchor(
                event.source_anchor,
                key_by_internal_slot=key_by_internal_slot,
            ),
        )
    if type(event) is StatusRefreshedOrExtendedEventV2:
        return ReplayIncomingStatusRefreshedOrExtendedEventV1(
            event_id=event.event_id,
            ordinal=event.ordinal,
            phase_rank=event.phase_rank,
            event_kind=event.event_type,
            recipient_anchor=_replay_incoming_anchor(
                event.recipient_anchor,
                key_by_internal_slot=key_by_internal_slot,
            ),
            status_channel=event.status_channel,
            status_id=event.status_id,
        )
    if type(event) is StatusClearedByNewDeathEventV2:
        return ReplayIncomingStatusClearedByNewDeathEventV1(
            event_id=event.event_id,
            ordinal=event.ordinal,
            phase_rank=event.phase_rank,
            event_kind=event.event_type,
            recipient_anchor=_replay_incoming_anchor(
                event.recipient_anchor,
                key_by_internal_slot=key_by_internal_slot,
            ),
            status_channel=event.status_channel,
            status_id=event.status_id,
        )
    if type(event) is SpawnShieldExpiredEventV2:
        return ReplayIncomingSpawnShieldExpiredEventV1(
            event_id=event.event_id,
            ordinal=event.ordinal,
            phase_rank=event.phase_rank,
            event_kind=event.event_type,
            agent_anchor=_replay_incoming_anchor(
                event.agent_anchor,
                key_by_internal_slot=key_by_internal_slot,
            ),
        )
    if type(event) is RespawnWaveOccurredEventV2:
        return ReplayIncomingRespawnWaveOccurredEventV1(
            event_id=event.event_id,
            ordinal=event.ordinal,
            phase_rank=event.phase_rank,
            event_kind=event.event_type,
            team_anchor=ReplayIncomingTeamAnchorV1(
                phase="successor",
                team_index=event.team_index,
                team_id=event.team_id,
            ),
        )
    if type(event) is AgentRespawnedEventV2:
        return ReplayIncomingAgentRespawnedEventV1(
            event_id=event.event_id,
            ordinal=event.ordinal,
            phase_rank=event.phase_rank,
            event_kind=event.event_type,
            agent_anchor=_replay_incoming_anchor(
                event.agent_anchor,
                key_by_internal_slot=key_by_internal_slot,
            ),
            team_id=event.team_id,
            realized_successor_position=event.realized_successor_position,
        )
    raise TypeError("unsupported canonical V2 event kind.")


def _project_replay_incoming_summary_v1(
    incoming_events: VisualEventBatchV2,
    *,
    key_by_internal_slot: dict[int, str],
) -> ReplayIncomingSummaryV1:
    """Project one validated Oracle batch without retaining internal axes."""
    VisualEventBatchV2.__post_init__(incoming_events)
    expected_slots = tuple(
        slot
        for slot, configured_active in enumerate(
            incoming_events.configured_active_by_global_slot
        )
        if configured_active
    )
    if (
        type(key_by_internal_slot) is not dict
        or tuple(sorted(key_by_internal_slot)) != expected_slots
        or any(type(key) is not int for key in key_by_internal_slot)
        or any(
            type(value) is not str or not value
            for value in key_by_internal_slot.values()
        )
        or len(set(key_by_internal_slot.values())) != len(key_by_internal_slot)
    ):
        raise ValueError(
            "incoming Oracle key map must equal the active trajectory axis."
        )
    trajectory_by_internal_slot = {
        row.global_slot: row for row in incoming_events.agent_phase_trajectories
    }
    trajectories = tuple(
        _replay_incoming_trajectory(
            row,
            key_by_internal_slot=key_by_internal_slot,
        )
        for row in incoming_events.agent_phase_trajectories
    )
    events = tuple(
        _replay_incoming_event(
            event,
            trajectory_by_internal_slot=trajectory_by_internal_slot,
            key_by_internal_slot=key_by_internal_slot,
        )
        for event in incoming_events.events
    )
    return ReplayIncomingSummaryV1(
        summary_kind="replay_incoming_inventory",
        incoming_transition_index=incoming_events.transition_index,
        incoming_transition_id=incoming_events.transition_id,
        incoming_start_frame_id=incoming_events.start_frame_id,
        incoming_successor_frame_id=incoming_events.successor_frame_id,
        incoming_start_simulator_step_count=(
            incoming_events.start_simulator_step_count
        ),
        incoming_successor_simulator_step_count=(
            incoming_events.successor_simulator_step_count
        ),
        agent_phase_trajectories=trajectories,
        ordered_event_ids=tuple(event.event_id for event in incoming_events.events),
        ordered_event_kinds=tuple(event.event_type for event in incoming_events.events),
        events=events,
        event_count=len(incoming_events.events),
    )


def _incoming_summary(
    scene: BattlefieldSceneV2,
    incoming_events: VisualEventBatchV2 | None,
    *,
    key_by_internal_slot: dict[int, str],
    expected_public_agent_id_by_global_slot: tuple[str, ...],
    expected_configured_active_by_global_slot: tuple[bool, ...],
) -> ReplayIncomingSummaryV1 | None:
    if scene.frame_index == 0:
        if incoming_events is not None:
            raise ValueError("frame zero cannot carry incoming presentation events.")
        return None
    if type(incoming_events) is not VisualEventBatchV2:
        raise ValueError("non-initial Oracle scenes require exact incoming events.")
    if (
        incoming_events.public_agent_id_by_global_slot
        != expected_public_agent_id_by_global_slot
        or incoming_events.configured_active_by_global_slot
        != expected_configured_active_by_global_slot
    ):
        raise ValueError(
            "incoming event roster identity must equal the context roster."
        )
    if (
        incoming_events.episode_id != scene.episode_id
        or incoming_events.transition_index != scene.frame_index - 1
        or incoming_events.transition_id != scene.incoming_transition_id
        or incoming_events.successor_frame_id != scene.frame_id
        or incoming_events.successor_simulator_step_count != scene.simulator_step_count
        or tuple(event.event_id for event in incoming_events.events)
        != scene.incoming_event_ids
    ):
        raise ValueError("incoming event inventory must join the current scene.")
    return _project_replay_incoming_summary_v1(
        incoming_events,
        key_by_internal_slot=key_by_internal_slot,
    )


def _outgoing_inspection(
    context: EvaluationEpisodeContextV1,
    scene: BattlefieldSceneV2,
    *,
    key_by_internal_slot: dict[int, str],
    selected_internal_slot: int | None,
    outgoing_transition: EvaluationTransitionV1 | None,
    final_frame_index: int,
) -> ReplayOutgoingInspectionV1 | None:
    should_have_outgoing = (
        selected_internal_slot is not None and scene.frame_index < final_frame_index
    )
    if not should_have_outgoing:
        if outgoing_transition is not None:
            raise ValueError("this current scene must not receive an outgoing row.")
        return None
    if type(outgoing_transition) is not EvaluationTransitionV1:
        raise ValueError("selected non-final scenes require an exact outgoing row.")
    transition = outgoing_transition
    if (
        transition.episode_id != scene.episode_id
        or transition.transition_index != scene.frame_index
        or transition.transition_id
        != f"{scene.episode_id}:transition:{scene.frame_index}"
        or transition.start_frame_id != scene.frame_id
        or transition.facts.transition_start_step_count != scene.simulator_step_count
        or transition.successor_frame_id
        != f"{scene.episode_id}:frame:{scene.frame_index + 1}"
    ):
        raise ValueError("outgoing transition must start at the displayed scene.")
    if selected_internal_slot is None:  # pragma: no cover - narrowed above.
        raise AssertionError("selected slot disappeared during outgoing projection")
    roster = context.roster[selected_internal_slot]
    if not roster.configured_active or roster.global_slot != selected_internal_slot:
        raise ValueError("inspection actor must be configured active.")
    agent_by_internal_slot = {agent.global_slot: agent for agent in scene.agents}
    actor = agent_by_internal_slot.get(selected_internal_slot)
    if actor is None or actor.public_agent_id != roster.public_agent_id:
        raise ValueError("inspection actor must occur in the current authorized scene.")
    acceptance = transition.facts.action_acceptance_facts
    submitted = acceptance.submitted_joint_action
    accepted = acceptance.accepted_joint_action
    accepted_target_action = accepted.select_target[selected_internal_slot]
    catalog = context.static_mechanics_catalog
    target_axis = catalog.global_recipient_slot_by_actor_and_target_action[
        selected_internal_slot
    ]
    target_internal_slot = target_axis[accepted_target_action]
    if target_internal_slot is None:
        accepted_target: ReplayAcceptedTargetV1 = ReplayAcceptedNoTargetV1(
            target_kind="none"
        )
    else:
        target_roster = context.roster[target_internal_slot]
        target_agent = agent_by_internal_slot.get(target_internal_slot)
        if (
            not target_roster.configured_active
            or target_agent is None
            or target_agent.public_agent_id != target_roster.public_agent_id
        ):
            raise ValueError("accepted target must occur in the current scene.")
        accepted_target = ReplayAcceptedAuthorizedTargetV1(
            target_kind="authorized_agent",
            target_presentation_key=key_by_internal_slot[target_internal_slot],
            target_public_agent_id=target_agent.public_agent_id,
            target_anchor=target_agent.position,
        )
    accepted_use_ultimate = accepted.use_ultimate[selected_internal_slot]
    return ReplayOutgoingInspectionV1(
        inspection_kind="replay_recorded_outgoing_action",
        outgoing_transition_index=transition.transition_index,
        outgoing_transition_id=transition.transition_id,
        outgoing_start_frame_id=transition.start_frame_id,
        outgoing_successor_frame_id=transition.successor_frame_id,
        actor_presentation_key=key_by_internal_slot[selected_internal_slot],
        actor_public_agent_id=actor.public_agent_id,
        actor_anchor=actor.position,
        submitted_action=SubmittedActionTupleV1(
            move_action=submitted.move[selected_internal_slot],
            target_action=submitted.select_target[selected_internal_slot],
            use_ultimate_action=submitted.use_ultimate[selected_internal_slot],
        ),
        accepted_action=AcceptedActionTupleV1(
            move_action=accepted.move[selected_internal_slot],
            target_action=accepted_target_action,
            use_ultimate_action=accepted_use_ultimate,
        ),
        accepted_lane="ultimate" if accepted_use_ultimate == 1 else "basic",
        accepted_target=accepted_target,
    )


def build_replay_oracle_presentation_parts_v1(
    context: EvaluationEpisodeContextV1,
    scene: BattlefieldSceneV2,
    incoming_events: VisualEventBatchV2 | None,
    *,
    authority_session_id: str,
    final_frame_index: int,
    selected_internal_slot: int | None,
    outgoing_transition: EvaluationTransitionV1 | None,
) -> ReplayOraclePresentationPartsV1:
    """Build neutral Oracle scene/history/inspection siblings from recorded facts.

    The incoming branch losslessly remaps canonical visual events and phase
    trajectories to authorized identities without choosing presentation
    anchors or composing atomic status motifs.  No successor frame is accepted
    by this API; the incoming batch's successor anchors must join ``scene``.
    """
    if type(context) is not EvaluationEpisodeContextV1:
        raise TypeError("context must be the exact EvaluationEpisodeContextV1 root.")
    if type(scene) is not BattlefieldSceneV2:
        raise TypeError("scene must be the exact BattlefieldSceneV2 root.")
    if incoming_events is not None and type(incoming_events) is not VisualEventBatchV2:
        raise TypeError("incoming_events must be VisualEventBatchV2 or None.")
    if (
        outgoing_transition is not None
        and type(outgoing_transition) is not EvaluationTransitionV1
    ):
        raise TypeError("outgoing_transition must be EvaluationTransitionV1 or None.")
    _require_text(authority_session_id, name="authority_session_id")
    _require_python_int(final_frame_index, name="final_frame_index")
    if selected_internal_slot is not None:
        _require_python_int(selected_internal_slot, name="selected_internal_slot")
        if selected_internal_slot >= len(context.roster):
            raise ValueError("selected_internal_slot is outside the context roster.")
    if scene.episode_id != context.identity.episode_id:
        raise ValueError("Oracle scene and context must join one episode.")
    if scene.frame_index > final_frame_index:
        raise ValueError("current scene cannot exceed the retained replay prefix.")
    current_scene, key_by_internal_slot = _authorized_scene(
        context,
        scene,
        authority_session_id=authority_session_id,
    )
    return ReplayOraclePresentationPartsV1(
        current_scene=current_scene,
        incoming_summary=_incoming_summary(
            scene,
            incoming_events,
            key_by_internal_slot=key_by_internal_slot,
            expected_public_agent_id_by_global_slot=tuple(
                row.public_agent_id for row in context.roster
            ),
            expected_configured_active_by_global_slot=tuple(
                row.configured_active for row in context.roster
            ),
        ),
        outgoing_inspection=_outgoing_inspection(
            context,
            scene,
            key_by_internal_slot=key_by_internal_slot,
            selected_internal_slot=selected_internal_slot,
            outgoing_transition=outgoing_transition,
            final_frame_index=final_frame_index,
        ),
    )


__all__ = [
    "AUTHORIZED_PRESENTATION_SCHEMA_VERSION",
    "AcceptedActionTupleV1",
    "AuthorizedAgentV1",
    "AuthorizedAuraFieldV1",
    "AuthorizedAuraIdV1",
    "AuthorizedAuraModifierV1",
    "AuthorizedBattlefieldSceneV1",
    "AuthorizedClassAuraMechanicV1",
    "AuthorizedClassMechanicsV1",
    "AuthorizedClassStatusMechanicV1",
    "AuthorizedMapV1",
    "AuthorizedObstacleV1",
    "AuthorizedRespawnWaveV1",
    "AuthorizedSpawnPadV1",
    "AuthorizedSpawnShieldMechanicsAvailableV1",
    "AuthorizedSpawnShieldMechanicsUnavailableV1",
    "AuthorizedSpawnShieldMechanicsV1",
    "AuthorizedStatusSourceV1",
    "AuthorizedStatusV1",
    "ReplayAcceptedAuthorizedTargetV1",
    "ReplayAcceptedNoTargetV1",
    "ReplayAcceptedTargetV1",
    "ReplayIncomingAbilityActivatedEventV1",
    "ReplayIncomingActionRejectedEventV1",
    "ReplayIncomingAgentAnchorV1",
    "ReplayIncomingAgentDiedEventV1",
    "ReplayIncomingAgentIdentityV1",
    "ReplayIncomingAgentPhaseTrajectoryV1",
    "ReplayIncomingAgentRespawnedEventV1",
    "ReplayIncomingAnchorPhaseV1",
    "ReplayIncomingAuthorizedAgentIdentityV1",
    "ReplayIncomingChargePhaseDisplacementEventV1",
    "ReplayIncomingCombatCountdownResetEventV1",
    "ReplayIncomingCooldownReadyEventV1",
    "ReplayIncomingCooldownStartedEventV1",
    "ReplayIncomingEventKindV1",
    "ReplayIncomingEventV1",
    "ReplayIncomingFeedOnlyAgentIdentityV1",
    "ReplayIncomingHealthRegeneratedEventV1",
    "ReplayIncomingLethalDamageContributionEventV1",
    "ReplayIncomingOrdinaryMovementPhaseDisplacementEventV1",
    "ReplayIncomingRecipientHealthResolutionEventV1",
    "ReplayIncomingRespawnWaveOccurredEventV1",
    "ReplayIncomingSourceDamageOutputEventV1",
    "ReplayIncomingSourceHealingOutputEventV1",
    "ReplayIncomingSpawnShieldExpiredEventV1",
    "ReplayIncomingStatusAgedToZeroEventV1",
    "ReplayIncomingStatusAppliedEventV1",
    "ReplayIncomingStatusBrokenByDamageEventV1",
    "ReplayIncomingStatusClearedByNewDeathEventV1",
    "ReplayIncomingStatusRefreshedOrExtendedEventV1",
    "ReplayIncomingSummaryV1",
    "ReplayIncomingTeamAnchorV1",
    "ReplayOraclePresentationPartsV1",
    "ReplayOutgoingInspectionV1",
    "SubmittedActionTupleV1",
    "build_oracle_authorized_scene_v1",
    "build_replay_oracle_presentation_parts_v1",
    "oracle_presentation_key_v1",
]
