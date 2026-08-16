"""Versioned, renderer-neutral battlefield and semantic-event contracts.

The records in this module are deliberately scalar-only.  They may be produced
from JAX-backed simulator artifacts, but they never retain JAX/NumPy arrays,
callbacks, renderer objects, or mutable presentation state.
"""

from dataclasses import dataclass, field
from math import isclose, isfinite
from typing import Literal, cast

from marl_battlegrounds.evaluation.wire_shapes import MAX_AGENT_SLOTS_V1
from marl_battlegrounds.rendering.vocabulary import (
    CATALOG_STATUS_ID_BY_CHANNEL,
    ActivationTokenId,
    StatusLifecycleKind,
    status_sort_key,
    status_token_id_from_catalog_status_id,
)

SCENE_SCHEMA_VERSION = 1
SCENE_V2_SCHEMA_VERSION = 2
EVENT_SCHEMA_VERSION = 1
EVENT_V2_SCHEMA_VERSION = 2
RESEARCHER_ANALYZER_PROJECTION_SCHEMA_VERSION = 2
STATUS_SOURCE_EVIDENCE_SCHEMA_VERSION = 2

MAX_AGENT_SLOTS = MAX_AGENT_SLOTS_V1

_CANONICAL_CLASS_NAME_BY_ID_V1 = {
    1: "Mage",
    2: "Warrior",
    3: "Hunter",
    4: "Rogue",
    5: "Priest",
}

type Point2D = tuple[float, float]
type SceneAudience = Literal["researcher", "agent_pov"]
type ObstacleKind = Literal["pillar", "wall"]
type RangeKind = Literal["observation", "basic", "ultimate"]
type Lane = Literal[0, 1]
type TargetDisclosure = Literal["public", "target_none", "redacted", "invalid"]
type HealthOutcome = Literal["damage", "healing", "unchanged"]
type ChargePathKind = Literal["charge_only", "combined_charge_and_movement"]
type RejectionComponent = Literal["movement", "combat", "complete_tuple_domain"]
type AgentLifeStateV2 = Literal["alive", "corpse"]
type StatusFamilyV2 = Literal[
    "slow",
    "stun",
    "anti_heal",
    "damage_amplification",
    "movement_floor",
]
type StatusMagnitudeKindV2 = Literal[
    "movement_multiplier",
    "none",
    "healing_multiplier",
    "damage_multiplier",
    "movement_floor",
]
type BasicTargetModeV2 = Literal[
    "unavailable",
    "ally",
    "enemy",
]
type UltimateTargetModeV2 = Literal[
    "unavailable",
    "target_none",
    "ally",
    "enemy",
]
type StatusActionComponentV2 = Literal["basic", "ultimate"]
type AuraStackingRuleV2 = Literal["multiply_then_clamp"]
type AuraClampKindV2 = Literal["ceiling", "floor"]
type VisualAnchorPhaseV2 = Literal[
    "transition_start",
    "post_charge",
    "successor",
]


def _require_python_int(value: int, *, name: str, minimum: int | None = None) -> None:
    if type(value) is not int:
        raise ValueError(f"{name} must be a Python int; got {type(value).__name__}.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}; got {value}.")


def _require_python_bool(value: bool, *, name: str) -> None:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a Python bool; got {type(value).__name__}.")


def _require_lane(
    value: Lane | None,
    *,
    name: str,
    allow_none: bool,
) -> None:
    if value is None:
        if allow_none:
            return
        raise ValueError(f"{name} must be the Python int 0 or 1.")
    if type(value) is not int or value not in (0, 1):
        suffix = ", or None" if allow_none else ""
        raise ValueError(f"{name} must be the Python int 0 or 1{suffix}.")


def _require_optional_record(
    value: object,
    *,
    name: str,
    record_type: type[object],
) -> None:
    if value is not None and type(value) is not record_type:
        raise ValueError(
            f"{name} must be {record_type.__name__} or None; "
            f"got {type(value).__name__}."
        )


def _require_slot(value: int, *, name: str) -> None:
    _require_python_int(value, name=name)
    if not 0 <= value < MAX_AGENT_SLOTS:
        raise ValueError(f"{name} must be in [0, {MAX_AGENT_SLOTS}); got {value}.")


def _require_finite(value: float, *, name: str) -> None:
    if type(value) is not float or not isfinite(value):
        raise ValueError(f"{name} must be a finite Python float; got {value!r}.")


def _require_nonnegative_finite(value: float, *, name: str) -> None:
    _require_finite(value, name=name)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative; got {value!r}.")


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


def _require_text(value: str, *, name: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a non-empty Python string.")


def _require_tuple_items(
    value: object,
    *,
    name: str,
    item_types: tuple[type[object], ...],
) -> None:
    if type(value) is not tuple:
        raise ValueError(f"{name} must be a Python tuple.")
    allowed_names = " or ".join(item_type.__name__ for item_type in item_types)
    for index, item in enumerate(cast(tuple[object, ...], value)):
        if not any(type(item) is item_type for item_type in item_types):
            raise ValueError(
                f"{name}[{index}] must be {allowed_names}; got {type(item).__name__}."
            )


def _require_unique(values: tuple[str, ...], *, name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{name} values must be unique.")


def _is_canonical_event_before_frame(
    event_id: str,
    *,
    episode_id: str,
    frame_index: int,
) -> bool:
    prefix = f"{episode_id}:transition:"
    if not event_id.startswith(prefix):
        return False
    suffix = event_id[len(prefix) :]
    transition_text, separator, ordinal_text = suffix.partition(":event:")
    if separator != ":event:" or not transition_text.isdigit():
        return False
    if str(int(transition_text)) != transition_text:
        return False
    if len(ordinal_text) != 4 or not ordinal_text.isdigit():
        return False
    return int(transition_text) < frame_index


@dataclass(frozen=True, slots=True, kw_only=True)
class ObstacleSceneV1:
    """One allowlisted static map obstacle."""

    obstacle_id: str
    kind: ObstacleKind
    center: Point2D
    radius: float | None = None
    width: float | None = None
    height: float | None = None
    theta: float = 0.0

    def __post_init__(self) -> None:
        _require_text(self.obstacle_id, name="obstacle_id")
        _require_point(self.center, name="center")
        _require_finite(self.theta, name="theta")
        if self.kind == "pillar":
            if self.radius is None:
                raise ValueError("pillar obstacles require radius.")
            _require_finite(self.radius, name="radius")
            if self.radius <= 0:
                raise ValueError("pillar radius must be positive.")
            if self.width is not None or self.height is not None:
                raise ValueError("pillar obstacles must not define width or height.")
        elif self.kind == "wall":
            if self.width is None or self.height is None:
                raise ValueError("wall obstacles require width and height.")
            _require_finite(self.width, name="width")
            _require_finite(self.height, name="height")
            if self.width <= 0 or self.height <= 0:
                raise ValueError("wall width and height must be positive.")
            if self.radius is not None:
                raise ValueError("wall obstacles must not define radius.")
        else:
            raise ValueError(f"unknown obstacle kind: {self.kind!r}.")


@dataclass(frozen=True, slots=True, kw_only=True)
class MapSceneV1:
    """Finite map bounds and ordered static obstacles."""

    width: float
    height: float
    obstacles: tuple[ObstacleSceneV1, ...] = ()

    def __post_init__(self) -> None:
        _require_finite(self.width, name="width")
        _require_finite(self.height, name="height")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("map width and height must be positive.")
        _require_tuple_items(
            self.obstacles,
            name="obstacles",
            item_types=(ObstacleSceneV1,),
        )
        _require_unique(
            tuple(obstacle.obstacle_id for obstacle in self.obstacles),
            name="obstacle_id",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class StatusSceneV1:
    """One durable successor-state status token."""

    token_id: str
    duration: int
    source_class_id: int
    label: str
    short_label: str
    accessible_name: str
    priority: int

    def __post_init__(self) -> None:
        _require_text(self.token_id, name="token_id")
        _require_python_int(self.duration, name="duration", minimum=1)
        _require_python_int(self.source_class_id, name="source_class_id")
        _require_text(self.label, name="label")
        _require_text(self.short_label, name="short_label")
        _require_text(self.accessible_name, name="accessible_name")
        _require_python_int(self.priority, name="priority", minimum=0)


@dataclass(frozen=True, slots=True, kw_only=True)
class ModifierSceneV1:
    """One exact recipient-local public multiplier."""

    token_id: str
    multiplier: float
    label: str
    accessible_name: str

    def __post_init__(self) -> None:
        _require_text(self.token_id, name="token_id")
        _require_finite(self.multiplier, name="multiplier")
        _require_text(self.label, name="label")
        _require_text(self.accessible_name, name="accessible_name")


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentSceneV1:
    """Allowlisted durable facts for one authorized active agent."""

    global_slot: int
    team_id: int
    class_id: int
    position: Point2D
    radius: float
    active: bool
    alive: bool
    current_health: float
    max_health: float
    ultimate_cooldown: int
    effective_speed: float
    statuses: tuple[StatusSceneV1, ...] = ()
    modifiers: tuple[ModifierSceneV1, ...] = ()

    def __post_init__(self) -> None:
        _require_slot(self.global_slot, name="global_slot")
        _require_python_int(self.team_id, name="team_id")
        _require_python_int(self.class_id, name="class_id")
        _require_point(self.position, name="position")
        _require_finite(self.radius, name="radius")
        _require_python_bool(self.active, name="active")
        _require_python_bool(self.alive, name="alive")
        _require_finite(self.current_health, name="current_health")
        _require_finite(self.max_health, name="max_health")
        _require_python_int(
            self.ultimate_cooldown,
            name="ultimate_cooldown",
            minimum=0,
        )
        _require_finite(self.effective_speed, name="effective_speed")
        if self.radius <= 0:
            raise ValueError("agent radius must be positive.")
        if self.current_health < 0 or self.max_health <= 0:
            raise ValueError("health values must be non-negative with max_health > 0.")
        if self.current_health > self.max_health:
            raise ValueError("current_health must not exceed max_health.")
        _require_tuple_items(
            self.statuses,
            name="statuses",
            item_types=(StatusSceneV1,),
        )
        _require_tuple_items(
            self.modifiers,
            name="modifiers",
            item_types=(ModifierSceneV1,),
        )
        priorities = tuple(status.priority for status in self.statuses)
        if priorities != tuple(sorted(priorities)):
            raise ValueError("statuses must use canonical non-decreasing priority.")
        _require_unique(
            tuple(status.token_id for status in self.statuses),
            name="status token_id",
        )
        _require_unique(
            tuple(modifier.token_id for modifier in self.modifiers),
            name="modifier token_id",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class AuraFieldSceneV1:
    """One public world-space aura source field."""

    source_global_slot: int
    token_id: Literal["mage_amplification", "warrior_mitigation"]
    center: Point2D
    radius: float

    def __post_init__(self) -> None:
        _require_slot(self.source_global_slot, name="source_global_slot")
        _require_point(self.center, name="center")
        _require_finite(self.radius, name="radius")
        if self.radius <= 0:
            raise ValueError("aura radius must be positive.")
        if self.token_id not in ("mage_amplification", "warrior_mitigation"):
            raise ValueError(f"unknown aura token: {self.token_id!r}.")


@dataclass(frozen=True, slots=True, kw_only=True)
class RangeSceneV1:
    """One already-derived world-space range circle."""

    global_slot: int
    center: Point2D
    radius: float
    kind: RangeKind

    def __post_init__(self) -> None:
        _require_slot(self.global_slot, name="global_slot")
        _require_point(self.center, name="center")
        _require_finite(self.radius, name="radius")
        if self.radius < 0:
            raise ValueError("range radius must be non-negative.")
        if self.kind not in ("observation", "basic", "ultimate"):
            raise ValueError(f"unknown range kind: {self.kind!r}.")


@dataclass(frozen=True, slots=True, kw_only=True)
class SelectionSceneV1:
    """Current Python-owned controlled actor and selected target."""

    controlled_global_slot: int
    selected_global_slot: int | None

    def __post_init__(self) -> None:
        _require_slot(self.controlled_global_slot, name="controlled_global_slot")
        if self.selected_global_slot is not None:
            _require_slot(self.selected_global_slot, name="selected_global_slot")


@dataclass(frozen=True, slots=True, kw_only=True)
class SelectedLegalitySceneV1:
    """Exact selected target pair values copied from the authoritative mask."""

    controlled_global_slot: int
    target_global_slot: int
    target_action: int
    lane_0_available: bool
    lane_1_available: bool
    armed_lane: Lane | None
    armed_pair_legal: bool

    def __post_init__(self) -> None:
        _require_slot(self.controlled_global_slot, name="controlled_global_slot")
        _require_slot(self.target_global_slot, name="target_global_slot")
        _require_python_int(self.target_action, name="target_action", minimum=1)
        _require_python_bool(self.lane_0_available, name="lane_0_available")
        _require_python_bool(self.lane_1_available, name="lane_1_available")
        _require_python_bool(self.armed_pair_legal, name="armed_pair_legal")
        _require_lane(self.armed_lane, name="armed_lane", allow_none=True)
        expected = (
            False
            if self.armed_lane is None
            else self.lane_0_available
            if self.armed_lane == 0
            else self.lane_1_available
        )
        if self.armed_pair_legal is not expected:
            raise ValueError("armed_pair_legal must match the selected exact lane.")


@dataclass(frozen=True, slots=True, kw_only=True)
class PendingRouteSceneV1:
    """Current pending source-target request, never an accepted action."""

    source_global_slot: int
    target_global_slot: int
    source_anchor: Point2D
    target_anchor: Point2D
    lane: Lane
    legal: bool

    def __post_init__(self) -> None:
        _require_slot(self.source_global_slot, name="source_global_slot")
        _require_slot(self.target_global_slot, name="target_global_slot")
        _require_point(self.source_anchor, name="source_anchor")
        _require_point(self.target_anchor, name="target_anchor")
        _require_lane(self.lane, name="lane", allow_none=False)
        _require_python_bool(self.legal, name="legal")


@dataclass(frozen=True, slots=True, kw_only=True)
class ObserverVisibilitySceneV1:
    """One exact Python-provided observer/candidate visibility fact."""

    observer_global_slot: int
    candidate_global_slot: int
    visible: bool

    def __post_init__(self) -> None:
        _require_slot(self.observer_global_slot, name="observer_global_slot")
        _require_slot(self.candidate_global_slot, name="candidate_global_slot")
        _require_python_bool(self.visible, name="visible")


@dataclass(frozen=True, slots=True, kw_only=True)
class BattlefieldSceneV1:
    """One durable battlefield snapshot for a specific authorized audience."""

    schema_version: int
    audience: SceneAudience
    audience_badge: str
    map: MapSceneV1
    agents: tuple[AgentSceneV1, ...]
    aura_fields: tuple[AuraFieldSceneV1, ...] = ()
    ranges: tuple[RangeSceneV1, ...] = ()
    selection: SelectionSceneV1 | None = None
    selected_legality: SelectedLegalitySceneV1 | None = None
    pending_route: PendingRouteSceneV1 | None = None
    observer_visibility: tuple[ObserverVisibilitySceneV1, ...] = ()

    def __post_init__(self) -> None:
        _require_python_int(self.schema_version, name="schema_version")
        if self.schema_version != SCENE_SCHEMA_VERSION:
            raise ValueError(
                f"scene schema_version must be {SCENE_SCHEMA_VERSION}; "
                f"got {self.schema_version}."
            )
        if self.audience not in ("researcher", "agent_pov"):
            raise ValueError(f"unknown scene audience: {self.audience!r}.")
        _require_text(self.audience_badge, name="audience_badge")
        if type(self.map) is not MapSceneV1:
            raise ValueError(f"map must be MapSceneV1; got {type(self.map).__name__}.")
        _require_optional_record(
            self.selection,
            name="selection",
            record_type=SelectionSceneV1,
        )
        _require_optional_record(
            self.selected_legality,
            name="selected_legality",
            record_type=SelectedLegalitySceneV1,
        )
        _require_optional_record(
            self.pending_route,
            name="pending_route",
            record_type=PendingRouteSceneV1,
        )
        _require_tuple_items(
            self.agents,
            name="agents",
            item_types=(AgentSceneV1,),
        )
        _require_tuple_items(
            self.aura_fields,
            name="aura_fields",
            item_types=(AuraFieldSceneV1,),
        )
        _require_tuple_items(
            self.ranges,
            name="ranges",
            item_types=(RangeSceneV1,),
        )
        _require_tuple_items(
            self.observer_visibility,
            name="observer_visibility",
            item_types=(ObserverVisibilitySceneV1,),
        )
        slots = tuple(agent.global_slot for agent in self.agents)
        if slots != tuple(sorted(slots)):
            raise ValueError("agents must be ordered by global slot.")
        _require_unique(tuple(str(slot) for slot in slots), name="agent global_slot")
        if self.audience == "researcher" and "PRIVILEGED" not in self.audience_badge:
            raise ValueError("researcher scenes require an explicit PRIVILEGED badge.")
        if self.audience == "agent_pov" and self.observer_visibility:
            raise ValueError(
                "agent-POV scenes must not disclose privileged visibility."
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class ClassStatusMechanicSceneV2:
    """One catalog-authored status mechanic owned by a researcher class card."""

    status_channel: int
    status_id: str
    family: StatusFamilyV2
    source_action_component: StatusActionComponentV2
    duration_steps: int
    magnitude_kind: StatusMagnitudeKindV2
    magnitude: float | None
    breaks_on_positive_damage: bool

    def __post_init__(self) -> None:
        _require_python_int(
            self.status_channel,
            name="status_channel",
            minimum=0,
        )
        if self.status_channel >= 9:
            raise ValueError("status_channel must identify the exact V1 status axis.")
        _require_text(self.status_id, name="status_id")
        if CATALOG_STATUS_ID_BY_CHANNEL[self.status_channel] != self.status_id:
            raise ValueError(
                "class status channel and catalog status ID must retain V1 identity."
            )
        if self.family not in (
            "slow",
            "stun",
            "anti_heal",
            "damage_amplification",
            "movement_floor",
        ):
            raise ValueError(f"unknown status family: {self.family!r}.")
        if self.source_action_component not in ("basic", "ultimate"):
            raise ValueError("status source action must be basic or ultimate.")
        _require_python_int(self.duration_steps, name="duration_steps", minimum=1)
        if self.magnitude_kind not in (
            "movement_multiplier",
            "none",
            "healing_multiplier",
            "damage_multiplier",
            "movement_floor",
        ):
            raise ValueError(f"unknown status magnitude kind: {self.magnitude_kind!r}.")
        if self.magnitude_kind == "none":
            if self.magnitude is not None:
                raise ValueError(
                    "stun-like class status mechanics must omit magnitude."
                )
        elif self.magnitude is None:
            raise ValueError("non-stun class status mechanics require magnitude.")
        else:
            _require_finite(self.magnitude, name="magnitude")
        _require_python_bool(
            self.breaks_on_positive_damage,
            name="breaks_on_positive_damage",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ClassAuraMechanicSceneV2:
    """One catalog-authored passive aura owned by a researcher class card."""

    aura_id: str
    radius: float
    per_emitter_multiplier: float
    stacking_rule: AuraStackingRuleV2
    clamp_kind: AuraClampKindV2
    clamp_value: float

    def __post_init__(self) -> None:
        _require_text(self.aura_id, name="aura_id")
        for name in ("radius", "per_emitter_multiplier", "clamp_value"):
            _require_nonnegative_finite(cast(float, getattr(self, name)), name=name)
        if self.stacking_rule != "multiply_then_clamp":
            raise ValueError("class aura stacking rule must be multiply_then_clamp.")
        if self.clamp_kind not in ("ceiling", "floor"):
            raise ValueError("class aura clamp kind must be ceiling or floor.")


@dataclass(frozen=True, slots=True, kw_only=True)
class ClassMechanicsSceneV2:
    """Serialized class vocabulary and exact context-owned mechanics."""

    class_id: int
    class_name: str
    maximum_health: float
    body_radius: float
    base_movement_speed: float
    observation_radius: float
    basic_target_mode: BasicTargetModeV2
    basic_interaction_radius: float
    basic_raw_damage: float
    basic_raw_healing: float
    ultimate_target_mode: UltimateTargetModeV2
    ultimate_interaction_radius: float
    ultimate_cooldown_steps: int
    ultimate_raw_damage: float
    ultimate_raw_healing: float
    out_of_combat_delay_steps: int
    out_of_combat_health_regeneration_fraction_per_step: float
    status_mechanics: tuple[ClassStatusMechanicSceneV2, ...]
    aura_mechanics: tuple[ClassAuraMechanicSceneV2, ...]

    def __post_init__(self) -> None:
        _require_python_int(self.class_id, name="class_id", minimum=1)
        if self.class_id > 5:
            raise ValueError("class_id must identify a real V1 class.")
        _require_text(self.class_name, name="class_name")
        if self.class_name != _CANONICAL_CLASS_NAME_BY_ID_V1[self.class_id]:
            raise ValueError(
                "class_name must match the canonical V1 identity for class_id."
            )
        for name in (
            "body_radius",
            "maximum_health",
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
            value = cast(float, getattr(self, name))
            _require_finite(value, name=name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative.")
        _require_python_int(
            self.ultimate_cooldown_steps,
            name="ultimate_cooldown_steps",
            minimum=0,
        )
        if self.maximum_health <= 0.0 or self.body_radius <= 0.0:
            raise ValueError("real class health and body radius must be positive.")
        _require_python_int(
            self.out_of_combat_delay_steps,
            name="out_of_combat_delay_steps",
            minimum=0,
        )
        if self.basic_target_mode not in (
            "unavailable",
            "ally",
            "enemy",
        ):
            raise ValueError(f"unknown basic_target_mode: {self.basic_target_mode!r}.")
        if self.ultimate_target_mode not in (
            "unavailable",
            "target_none",
            "ally",
            "enemy",
        ):
            raise ValueError(
                f"unknown ultimate_target_mode: {self.ultimate_target_mode!r}."
            )
        if self.out_of_combat_health_regeneration_fraction_per_step > 1.0:
            raise ValueError(
                "out_of_combat_health_regeneration_fraction_per_step must not "
                "exceed one."
            )
        _require_tuple_items(
            self.status_mechanics,
            name="status_mechanics",
            item_types=(ClassStatusMechanicSceneV2,),
        )
        _require_tuple_items(
            self.aura_mechanics,
            name="aura_mechanics",
            item_types=(ClassAuraMechanicSceneV2,),
        )
        status_channels = tuple(row.status_channel for row in self.status_mechanics)
        if status_channels != tuple(sorted(status_channels)) or len(
            status_channels
        ) != len(set(status_channels)):
            raise ValueError(
                "class status mechanics must have unique increasing channels."
            )
        _require_unique(
            tuple(row.status_id for row in self.status_mechanics),
            name="class status mechanic ID",
        )
        _require_unique(
            tuple(row.aura_id for row in self.aura_mechanics),
            name="class aura mechanic ID",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class StatusSourceEvidenceSceneV2:
    """One direct incoming-event source for a durable status row."""

    source_global_slot: int
    source_public_agent_id: str
    event_id: str

    def __post_init__(self) -> None:
        _require_slot(self.source_global_slot, name="source_global_slot")
        _require_text(self.source_public_agent_id, name="source_public_agent_id")
        _require_text(self.event_id, name="event_id")


@dataclass(frozen=True, slots=True, kw_only=True)
class StatusSourceChannelEvidenceV2:
    """Durable direct-source evidence for one active recipient/status channel."""

    recipient_global_slot: int
    recipient_public_agent_id: str
    status_channel: int
    status_id: str
    direct_source_evidence: tuple[StatusSourceEvidenceSceneV2, ...] = ()

    def __post_init__(self) -> None:
        _require_slot(self.recipient_global_slot, name="recipient_global_slot")
        _require_text(
            self.recipient_public_agent_id,
            name="recipient_public_agent_id",
        )
        _require_python_int(self.status_channel, name="status_channel", minimum=0)
        _require_text(self.status_id, name="status_id")
        if (
            self.status_channel >= len(CATALOG_STATUS_ID_BY_CHANNEL)
            or CATALOG_STATUS_ID_BY_CHANNEL[self.status_channel] != self.status_id
        ):
            raise ValueError(
                "status-source channel and catalog status ID must retain V1 identity."
            )
        _require_tuple_items(
            self.direct_source_evidence,
            name="direct_source_evidence",
            item_types=(StatusSourceEvidenceSceneV2,),
        )
        evidence_keys = tuple(
            (row.source_global_slot, row.event_id)
            for row in self.direct_source_evidence
        )
        if evidence_keys != tuple(sorted(evidence_keys)) or len(evidence_keys) != len(
            set(evidence_keys)
        ):
            raise ValueError("status source evidence must have canonical unique keys.")


@dataclass(frozen=True, slots=True, kw_only=True)
class StatusSourceEvidenceStateV2:
    """Immutable status-source evidence bound to one canonical frame."""

    schema_version: int
    episode_id: str
    frame_index: int
    frame_id: str
    active_statuses: tuple[StatusSourceChannelEvidenceV2, ...]

    def __post_init__(self) -> None:
        _require_python_int(self.schema_version, name="schema_version")
        if self.schema_version != STATUS_SOURCE_EVIDENCE_SCHEMA_VERSION:
            raise ValueError("unknown status-source evidence state version.")
        _require_text(self.episode_id, name="episode_id")
        _require_python_int(self.frame_index, name="frame_index", minimum=0)
        if self.frame_id != f"{self.episode_id}:frame:{self.frame_index}":
            raise ValueError("status-source state frame ID is not canonical.")
        _require_tuple_items(
            self.active_statuses,
            name="active_statuses",
            item_types=(StatusSourceChannelEvidenceV2,),
        )
        keys = tuple(
            (row.recipient_global_slot, row.status_channel)
            for row in self.active_statuses
        )
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("active status-source rows require canonical unique keys.")
        recipient_public_id_by_slot: dict[int, str] = {}
        direct_event_ids: list[str] = []
        for row in self.active_statuses:
            existing_public_id = recipient_public_id_by_slot.setdefault(
                row.recipient_global_slot,
                row.recipient_public_agent_id,
            )
            if existing_public_id != row.recipient_public_agent_id:
                raise ValueError(
                    "one recipient slot cannot carry conflicting public IDs."
                )
            for evidence in row.direct_source_evidence:
                if not _is_canonical_event_before_frame(
                    evidence.event_id,
                    episode_id=self.episode_id,
                    frame_index=self.frame_index,
                ):
                    raise ValueError(
                        "direct status-source evidence must identify a canonical "
                        "event before the bound frame."
                    )
                direct_event_ids.append(evidence.event_id)
        if len(direct_event_ids) != len(set(direct_event_ids)):
            raise ValueError(
                "one direct status-application event cannot support multiple rows."
            )

    def evidence_by_recipient_and_channel(
        self,
    ) -> dict[tuple[int, int], tuple[StatusSourceEvidenceSceneV2, ...]]:
        """Return a fresh lookup without exposing mutable stored state."""
        return {
            (row.recipient_global_slot, row.status_channel): (
                row.direct_source_evidence
            )
            for row in self.active_statuses
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class StatusSourceEvidenceIndexV2:
    """Gap-free immutable replay index of per-frame source-evidence states."""

    schema_version: int
    episode_id: str
    frame_states: tuple[StatusSourceEvidenceStateV2, ...]

    def __post_init__(self) -> None:
        _require_python_int(self.schema_version, name="schema_version")
        if self.schema_version != STATUS_SOURCE_EVIDENCE_SCHEMA_VERSION:
            raise ValueError("unknown status-source evidence index version.")
        _require_text(self.episode_id, name="episode_id")
        if type(self.frame_states) is not tuple or not self.frame_states:
            raise ValueError("status-source index requires at least frame zero.")
        if any(
            type(row) is not StatusSourceEvidenceStateV2 for row in self.frame_states
        ):
            raise ValueError("frame_states must contain exact evidence states.")
        if tuple(row.frame_index for row in self.frame_states) != tuple(
            range(len(self.frame_states))
        ):
            raise ValueError("status-source index frame states must be gap-free.")
        if any(row.episode_id != self.episode_id for row in self.frame_states):
            raise ValueError("status-source index states must join its episode.")

    def state_for_frame(self, frame_index: int) -> StatusSourceEvidenceStateV2:
        """Return one O(1) frame-bound immutable state."""
        _require_python_int(frame_index, name="frame_index", minimum=0)
        if frame_index >= len(self.frame_states):
            raise IndexError("frame_index is outside the status-source index.")
        return self.frame_states[frame_index]


@dataclass(frozen=True, slots=True, kw_only=True)
class StatusSceneV2:
    """One recorded status channel with conservative source evidence."""

    status_channel: int
    status_id: str
    family: StatusFamilyV2
    remaining_duration: int
    source_class_id: int
    source_class_name: str
    source_action_component: Literal["basic", "ultimate"]
    magnitude_kind: StatusMagnitudeKindV2
    magnitude: float | None
    breaks_on_positive_damage: bool
    direct_source_evidence: tuple[StatusSourceEvidenceSceneV2, ...] = ()

    def __post_init__(self) -> None:
        _require_python_int(self.status_channel, name="status_channel", minimum=0)
        _require_text(self.status_id, name="status_id")
        if (
            self.status_channel >= len(CATALOG_STATUS_ID_BY_CHANNEL)
            or CATALOG_STATUS_ID_BY_CHANNEL[self.status_channel] != self.status_id
        ):
            raise ValueError(
                "status channel and catalog status ID must retain V1 identity."
            )
        _require_python_int(
            self.remaining_duration,
            name="remaining_duration",
            minimum=1,
        )
        _require_python_int(self.source_class_id, name="source_class_id", minimum=1)
        if self.source_class_id > 5:
            raise ValueError("source_class_id must identify a real V1 class.")
        _require_text(self.source_class_name, name="source_class_name")
        if self.family not in (
            "slow",
            "stun",
            "anti_heal",
            "damage_amplification",
            "movement_floor",
        ):
            raise ValueError(f"unknown status family: {self.family!r}.")
        if self.source_action_component not in ("basic", "ultimate"):
            raise ValueError("source_action_component must be 'basic' or 'ultimate'.")
        if self.magnitude_kind not in (
            "movement_multiplier",
            "none",
            "healing_multiplier",
            "damage_multiplier",
            "movement_floor",
        ):
            raise ValueError(f"unknown magnitude_kind: {self.magnitude_kind!r}.")
        if self.magnitude is None:
            if self.magnitude_kind != "none":
                raise ValueError("non-none magnitude kinds require a magnitude.")
        else:
            _require_finite(self.magnitude, name="magnitude")
            if self.magnitude_kind == "none":
                raise ValueError("the none magnitude kind must omit magnitude.")
        _require_python_bool(
            self.breaks_on_positive_damage,
            name="breaks_on_positive_damage",
        )
        _require_tuple_items(
            self.direct_source_evidence,
            name="direct_source_evidence",
            item_types=(StatusSourceEvidenceSceneV2,),
        )
        evidence_slots = tuple(
            row.source_global_slot for row in self.direct_source_evidence
        )
        if evidence_slots != tuple(sorted(evidence_slots)):
            raise ValueError("direct source evidence must be ordered by source slot.")
        _require_unique(
            tuple(row.event_id for row in self.direct_source_evidence),
            name="direct source event_id",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class AuraRecipientModifierSceneV2:
    """One exact recipient-local aura multiplier from the selected frame."""

    aura_id: Literal[
        "mage_damage_amplification",
        "warrior_damage_mitigation",
    ]
    multiplier: float

    def __post_init__(self) -> None:
        if self.aura_id not in (
            "mage_damage_amplification",
            "warrior_damage_mitigation",
        ):
            raise ValueError(f"unknown aura_id: {self.aura_id!r}.")
        _require_finite(self.multiplier, name="multiplier")
        if self.multiplier < 0.0:
            raise ValueError("aura multiplier must be non-negative.")


@dataclass(frozen=True, slots=True, kw_only=True)
class AuraFieldSceneV2:
    """One catalog-declared aura capability anchored to its recorded emitter."""

    aura_id: Literal[
        "mage_damage_amplification",
        "warrior_damage_mitigation",
    ]
    source_global_slot: int
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
            raise ValueError(f"unknown aura_id: {self.aura_id!r}.")
        _require_slot(self.source_global_slot, name="source_global_slot")
        _require_text(self.source_public_agent_id, name="source_public_agent_id")
        _require_python_int(self.source_class_id, name="source_class_id", minimum=1)
        if self.source_class_id > 5:
            raise ValueError("source_class_id must identify a real V1 class.")
        _require_text(self.source_class_name, name="source_class_name")
        _require_python_bool(self.source_alive, name="source_alive")
        _require_point(self.center, name="center")
        for name in ("radius", "per_emitter_multiplier", "clamp_value"):
            value = cast(float, getattr(self, name))
            _require_finite(value, name=name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative.")
        if self.radius <= 0.0:
            raise ValueError("aura radius must be positive.")
        if self.beneficiary_relation != "same_team":
            raise ValueError("V2 aura fields require same-team beneficiaries.")
        if self.stacking_rule != "multiply_then_clamp":
            raise ValueError("V2 aura fields require multiply-then-clamp stacking.")
        if self.clamp_kind not in ("ceiling", "floor"):
            raise ValueError("clamp_kind must be ceiling or floor.")


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentSceneV2:
    """One configured-active actor copied from a selected evaluation frame."""

    global_slot: int
    public_agent_id: str
    team_id: int
    team_local_slot: int
    class_id: int
    position: Point2D
    radius: float
    life_state: AgentLifeStateV2
    current_health: float
    max_health: float
    effective_movement_speed: float
    ultimate_cooldown_remaining: int
    spawn_shield_remaining: int
    steps_until_out_of_combat: int
    respawned_on_incoming_transition: bool
    respawn_event_id: str | None
    statuses: tuple[StatusSceneV2, ...] = ()
    aura_modifiers: tuple[AuraRecipientModifierSceneV2, ...] = ()

    def __post_init__(self) -> None:
        _require_slot(self.global_slot, name="global_slot")
        _require_text(self.public_agent_id, name="public_agent_id")
        _require_python_int(self.team_id, name="team_id", minimum=1)
        if self.team_id not in (1, 2):
            raise ValueError("team_id must be one or two.")
        _require_python_int(self.team_local_slot, name="team_local_slot", minimum=0)
        if self.team_local_slot >= 5:
            raise ValueError("team_local_slot must be less than five.")
        _require_python_int(self.class_id, name="class_id", minimum=1)
        if self.class_id > 5:
            raise ValueError("class_id must identify a real V1 class.")
        _require_point(self.position, name="position")
        _require_finite(self.radius, name="radius")
        _require_finite(self.current_health, name="current_health")
        _require_finite(self.max_health, name="max_health")
        _require_finite(
            self.effective_movement_speed,
            name="effective_movement_speed",
        )
        if self.radius <= 0.0 or self.max_health <= 0.0:
            raise ValueError("active agents require positive radius and max_health.")
        if not 0.0 <= self.current_health <= self.max_health:
            raise ValueError("current_health must lie within [0, max_health].")
        if self.effective_movement_speed < 0.0:
            raise ValueError("effective_movement_speed must be non-negative.")
        if self.life_state not in ("alive", "corpse"):
            raise ValueError(f"unknown life_state: {self.life_state!r}.")
        for name in (
            "ultimate_cooldown_remaining",
            "spawn_shield_remaining",
            "steps_until_out_of_combat",
        ):
            _require_python_int(cast(int, getattr(self, name)), name=name, minimum=0)
        _require_python_bool(
            self.respawned_on_incoming_transition,
            name="respawned_on_incoming_transition",
        )
        if self.respawned_on_incoming_transition != (self.respawn_event_id is not None):
            raise ValueError(
                "respawn_event_id presence must match incoming-transition respawn."
            )
        if self.respawn_event_id is not None:
            _require_text(self.respawn_event_id, name="respawn_event_id")
        _require_tuple_items(
            self.statuses, name="statuses", item_types=(StatusSceneV2,)
        )
        channels = tuple(status.status_channel for status in self.statuses)
        if len(channels) != len(set(channels)):
            raise ValueError("statuses must have unique scientific channel IDs.")
        for status in self.statuses:
            if (
                status.status_channel >= len(CATALOG_STATUS_ID_BY_CHANNEL)
                or CATALOG_STATUS_ID_BY_CHANNEL[status.status_channel]
                != status.status_id
            ):
                raise ValueError(
                    "status channel and catalog status ID must retain V1 identity."
                )
        presentation_keys = tuple(
            status_sort_key(status_token_id_from_catalog_status_id(status.status_id))
            for status in self.statuses
        )
        if presentation_keys != tuple(sorted(presentation_keys)):
            raise ValueError("statuses must use canonical presentation order.")
        _require_tuple_items(
            self.aura_modifiers,
            name="aura_modifiers",
            item_types=(AuraRecipientModifierSceneV2,),
        )
        modifier_ids = tuple(row.aura_id for row in self.aura_modifiers)
        if modifier_ids != tuple(sorted(modifier_ids)) or len(modifier_ids) != len(
            set(modifier_ids)
        ):
            raise ValueError("aura modifiers must have unique sorted aura IDs.")


@dataclass(frozen=True, slots=True, kw_only=True)
class SpawnPadSceneV2:
    """One configured-active actor's recorded team spawn pad."""

    team_id: int
    team_local_slot: int
    assigned_global_slot: int
    assigned_public_agent_id: str
    position: Point2D

    def __post_init__(self) -> None:
        _require_python_int(self.team_id, name="team_id", minimum=1)
        if self.team_id not in (1, 2):
            raise ValueError("team_id must be one or two.")
        _require_python_int(self.team_local_slot, name="team_local_slot", minimum=0)
        if self.team_local_slot >= 5:
            raise ValueError("team_local_slot must be less than five.")
        _require_slot(self.assigned_global_slot, name="assigned_global_slot")
        _require_text(self.assigned_public_agent_id, name="assigned_public_agent_id")
        _require_point(self.position, name="position")


@dataclass(frozen=True, slots=True, kw_only=True)
class RespawnWaveSceneV2:
    """One team's exact configured wave period and selected-frame countdown."""

    team_index: int
    team_id: int
    period_steps: int
    countdown_steps: int

    def __post_init__(self) -> None:
        _require_python_int(self.team_index, name="team_index", minimum=0)
        if self.team_index not in (0, 1):
            raise ValueError("team_index must be zero or one.")
        _require_python_int(self.team_id, name="team_id", minimum=1)
        if self.team_id != self.team_index + 1:
            raise ValueError("team_id must match team_index + 1.")
        _require_python_int(self.period_steps, name="period_steps", minimum=1)
        _require_python_int(self.countdown_steps, name="countdown_steps", minimum=0)
        if self.countdown_steps >= self.period_steps:
            raise ValueError("countdown_steps must be less than period_steps.")


@dataclass(frozen=True, slots=True, kw_only=True)
class BattlefieldSceneV2:
    """Researcher-authorized durable state from one canonical evaluation frame."""

    schema_version: int
    audience: Literal["researcher"]
    audience_badge: str
    episode_id: str
    frame_index: int
    frame_id: str
    simulator_step_count: int
    incoming_transition_id: str | None
    incoming_event_ids: tuple[str, ...]
    map: MapSceneV1
    agents: tuple[AgentSceneV2, ...]
    aura_fields: tuple[AuraFieldSceneV2, ...]
    class_mechanics: tuple[ClassMechanicsSceneV2, ...]
    spawn_pads: tuple[SpawnPadSceneV2, ...]
    respawn_waves: tuple[RespawnWaveSceneV2, ...]
    ranges: tuple[RangeSceneV1, ...] = ()
    selection: SelectionSceneV1 | None = None
    next_decision_selected_legality: SelectedLegalitySceneV1 | None = None
    observer_visibility: tuple[ObserverVisibilitySceneV1, ...] = ()

    def __post_init__(self) -> None:
        _require_python_int(self.schema_version, name="schema_version")
        if self.schema_version != SCENE_V2_SCHEMA_VERSION:
            raise ValueError(
                f"scene schema_version must be {SCENE_V2_SCHEMA_VERSION}; "
                f"got {self.schema_version}."
            )
        if self.audience != "researcher":
            raise ValueError("BattlefieldSceneV2 is researcher-authorized only.")
        if "PRIVILEGED" not in self.audience_badge:
            raise ValueError("researcher scenes require an explicit PRIVILEGED badge.")
        _require_text(self.episode_id, name="episode_id")
        _require_python_int(self.frame_index, name="frame_index", minimum=0)
        _require_text(self.frame_id, name="frame_id")
        if self.frame_id != f"{self.episode_id}:frame:{self.frame_index}":
            raise ValueError("frame_id must match episode_id and frame_index.")
        _require_python_int(
            self.simulator_step_count,
            name="simulator_step_count",
            minimum=0,
        )
        expected_transition_id = (
            None
            if self.frame_index == 0
            else f"{self.episode_id}:transition:{self.frame_index - 1}"
        )
        if self.incoming_transition_id != expected_transition_id:
            raise ValueError(
                "incoming_transition_id must identify the transition entering frame."
            )
        _require_tuple_items(
            self.incoming_event_ids,
            name="incoming_event_ids",
            item_types=(str,),
        )
        expected_event_ids = tuple(
            f"{self.incoming_transition_id}:event:{ordinal:04d}"
            for ordinal in range(len(self.incoming_event_ids))
        )
        if self.incoming_event_ids != expected_event_ids:
            raise ValueError("incoming_event_ids must be canonical and gap-free.")
        if type(self.map) is not MapSceneV1:
            raise ValueError(f"map must be MapSceneV1; got {type(self.map).__name__}.")
        _require_tuple_items(self.agents, name="agents", item_types=(AgentSceneV2,))
        _require_tuple_items(
            self.aura_fields,
            name="aura_fields",
            item_types=(AuraFieldSceneV2,),
        )
        _require_tuple_items(
            self.class_mechanics,
            name="class_mechanics",
            item_types=(ClassMechanicsSceneV2,),
        )
        _require_tuple_items(
            self.spawn_pads,
            name="spawn_pads",
            item_types=(SpawnPadSceneV2,),
        )
        _require_tuple_items(
            self.respawn_waves,
            name="respawn_waves",
            item_types=(RespawnWaveSceneV2,),
        )
        _require_tuple_items(self.ranges, name="ranges", item_types=(RangeSceneV1,))
        _require_optional_record(
            self.selection,
            name="selection",
            record_type=SelectionSceneV1,
        )
        _require_optional_record(
            self.next_decision_selected_legality,
            name="next_decision_selected_legality",
            record_type=SelectedLegalitySceneV1,
        )
        _require_tuple_items(
            self.observer_visibility,
            name="observer_visibility",
            item_types=(ObserverVisibilitySceneV1,),
        )
        slots = tuple(agent.global_slot for agent in self.agents)
        if slots != tuple(sorted(slots)) or len(slots) != len(set(slots)):
            raise ValueError("agents must have unique increasing global slots.")
        _require_unique(
            tuple(agent.public_agent_id for agent in self.agents),
            name="public_agent_id",
        )
        agent_by_slot = {agent.global_slot: agent for agent in self.agents}
        if self.selection is None:
            if self.observer_visibility:
                raise ValueError(
                    "observer visibility requires a selected researcher observer."
                )
        else:
            if self.selection.controlled_global_slot not in agent_by_slot:
                raise ValueError(
                    "observer visibility requires an active controlled researcher."
                )
            visibility_observers = tuple(
                row.observer_global_slot for row in self.observer_visibility
            )
            visibility_candidates = tuple(
                row.candidate_global_slot for row in self.observer_visibility
            )
            if visibility_candidates != slots:
                raise ValueError(
                    "observer visibility must cover the ordered scene roster exactly."
                )
            if visibility_observers != (self.selection.controlled_global_slot,) * len(
                slots
            ):
                raise ValueError(
                    "observer visibility must belong to the controlled researcher."
                )
        for agent in self.agents:
            expected_team_id = 1 if agent.global_slot < 5 else 2
            if agent.team_id != expected_team_id:
                raise ValueError("agent team must match the V1 global-slot topology.")
            if agent.team_local_slot != agent.global_slot % 5:
                raise ValueError(
                    "agent team-local slot must match the V1 global-slot topology."
                )
            if agent.respawn_event_id is not None and (
                agent.respawn_event_id not in self.incoming_event_ids
            ):
                raise ValueError("respawn evidence must join an incoming event ID.")
            expected_modifier_ids = (
                "mage_damage_amplification",
                "warrior_damage_mitigation",
            )
            if tuple(row.aura_id for row in agent.aura_modifiers) != (
                expected_modifier_ids
            ):
                raise ValueError(
                    "each researcher agent requires both ordered aura modifiers."
                )
            for status in agent.statuses:
                for evidence in status.direct_source_evidence:
                    source_agent = agent_by_slot.get(evidence.source_global_slot)
                    if source_agent is None or (
                        source_agent.public_agent_id != evidence.source_public_agent_id
                    ):
                        raise ValueError(
                            "status source evidence must join a scene agent identity."
                        )
                    if not _is_canonical_event_before_frame(
                        evidence.event_id,
                        episode_id=self.episode_id,
                        frame_index=self.frame_index,
                    ):
                        raise ValueError(
                            "status source evidence must identify a canonical "
                            "event before the selected frame."
                        )
        aura_keys = tuple(
            (field.source_global_slot, field.aura_id) for field in self.aura_fields
        )
        if aura_keys != tuple(sorted(aura_keys)) or len(aura_keys) != len(
            set(aura_keys)
        ):
            raise ValueError("aura fields must have unique canonical source/aura keys.")
        for aura_field in self.aura_fields:
            source_agent = agent_by_slot.get(aura_field.source_global_slot)
            if source_agent is None or (
                source_agent.public_agent_id != aura_field.source_public_agent_id
                or source_agent.class_id != aura_field.source_class_id
                or source_agent.position != aura_field.center
                or (source_agent.life_state == "alive") != aura_field.source_alive
            ):
                raise ValueError("aura fields must join their scene source agent.")
        class_ids = tuple(row.class_id for row in self.class_mechanics)
        if class_ids != tuple(sorted(class_ids)) or len(class_ids) != len(
            set(class_ids)
        ):
            raise ValueError("class mechanics must have unique increasing class IDs.")
        if any(
            _CANONICAL_CLASS_NAME_BY_ID_V1.get(row.class_id) != row.class_name
            for row in self.class_mechanics
        ):
            raise ValueError(
                "class mechanics must retain canonical V1 class identities."
            )
        if not set(agent.class_id for agent in self.agents).issubset(class_ids):
            raise ValueError("every scene agent requires its class mechanics row.")
        projected_status_channels = tuple(
            status.status_channel
            for mechanics in self.class_mechanics
            for status in mechanics.status_mechanics
        )
        if tuple(sorted(projected_status_channels)) != tuple(range(9)):
            raise ValueError(
                "class mechanics must partition the exact nine status channels."
            )
        expected_status_class_by_channel = (2, 3, 4, 2, 3, 4, 4, 1, 5)
        for mechanics in self.class_mechanics:
            if any(
                expected_status_class_by_channel[status.status_channel]
                != mechanics.class_id
                for status in mechanics.status_mechanics
            ):
                raise ValueError(
                    "class status mechanics must retain their catalog source class."
                )
        projected_aura_ids = tuple(
            aura.aura_id
            for mechanics in self.class_mechanics
            for aura in mechanics.aura_mechanics
        )
        if projected_aura_ids != (
            "mage_damage_amplification",
            "warrior_damage_mitigation",
        ):
            raise ValueError(
                "class mechanics must partition the exact ordered aura catalog."
            )
        expected_aura_class = {
            "mage_damage_amplification": 1,
            "warrior_damage_mitigation": 2,
        }
        for mechanics in self.class_mechanics:
            if any(
                expected_aura_class.get(aura.aura_id) != mechanics.class_id
                for aura in mechanics.aura_mechanics
            ):
                raise ValueError(
                    "class aura mechanics must retain their catalog emitter class."
                )
        mechanics_by_class = {row.class_id: row for row in self.class_mechanics}
        for agent in self.agents:
            for status in agent.statuses:
                source_mechanics = mechanics_by_class.get(status.source_class_id)
                if source_mechanics is None or (
                    source_mechanics.class_name != status.source_class_name
                ):
                    raise ValueError(
                        "status source class must join scene class mechanics."
                    )
        pad_keys = tuple((pad.team_id, pad.team_local_slot) for pad in self.spawn_pads)
        if pad_keys != tuple(sorted(pad_keys)) or len(pad_keys) != len(set(pad_keys)):
            raise ValueError("spawn pads must have unique canonical team/slot keys.")
        if len(self.spawn_pads) != len(self.agents):
            raise ValueError("each scene agent requires exactly one spawn pad.")
        for pad in self.spawn_pads:
            agent = agent_by_slot.get(pad.assigned_global_slot)
            if agent is None or (
                agent.public_agent_id != pad.assigned_public_agent_id
                or agent.team_id != pad.team_id
                or agent.team_local_slot != pad.team_local_slot
            ):
                raise ValueError("spawn pads must join their assigned scene agent.")
        if tuple(wave.team_index for wave in self.respawn_waves) != (0, 1):
            raise ValueError("respawn waves must contain ordered team indices 0 and 1.")
        for range_row in self.ranges:
            if range_row.global_slot not in agent_by_slot:
                raise ValueError("ranges must join a scene agent.")
        if self.selection is not None and (
            self.selection.controlled_global_slot not in agent_by_slot
            or (
                self.selection.selected_global_slot is not None
                and self.selection.selected_global_slot not in agent_by_slot
            )
        ):
            raise ValueError("selection must join configured-active scene agents.")
        if self.next_decision_selected_legality is not None:
            legality = self.next_decision_selected_legality
            if self.selection is None or (
                legality.controlled_global_slot != self.selection.controlled_global_slot
                or legality.target_global_slot != self.selection.selected_global_slot
            ):
                raise ValueError(
                    "next-decision legality must join the current scene selection."
                )


@dataclass(frozen=True, slots=True, kw_only=True)
class AcceptedActivationEventV1:
    """One exact accepted activation, intentionally without a health amount."""

    event_type: Literal["accepted_activation"] = field(
        default="accepted_activation",
        init=False,
    )
    event_id: str
    transition_id: int
    token_id: ActivationTokenId
    source_global_slot: int
    target_global_slot: int | None
    source_anchor: Point2D | None
    target_anchor: Point2D | None
    target_disclosure: TargetDisclosure
    lane: Lane
    source_class_id: int

    def __post_init__(self) -> None:
        _validate_event_header(self.event_id, self.transition_id)
        _require_slot(self.source_global_slot, name="source_global_slot")
        _require_python_int(self.source_class_id, name="source_class_id")
        if self.source_anchor is not None:
            _require_point(self.source_anchor, name="source_anchor")
        if self.target_global_slot is not None:
            _require_slot(self.target_global_slot, name="target_global_slot")
        if self.target_anchor is not None:
            _require_point(self.target_anchor, name="target_anchor")
        _require_lane(self.lane, name="lane", allow_none=False)
        if self.target_disclosure == "public":
            if self.target_global_slot is None or self.target_anchor is None:
                raise ValueError("public targets require identity and anchor.")
        elif self.target_disclosure == "target_none":
            if self.target_global_slot is not None or self.target_anchor is not None:
                raise ValueError("target-none activations must not define a target.")
        elif self.target_disclosure in ("redacted", "invalid"):
            if self.target_global_slot is not None or self.target_anchor is not None:
                raise ValueError(
                    f"{self.target_disclosure} targets must omit identity and anchor."
                )
        else:
            raise ValueError(f"unknown target disclosure: {self.target_disclosure!r}.")
        if self.token_id == "mage_burst" and self.target_disclosure != "target_none":
            raise ValueError("Mage Burst must use target_none disclosure.")
        if self.token_id != "mage_burst" and self.target_disclosure == "target_none":
            raise ValueError("only Mage Burst is target-none.")
        if self.target_disclosure == "invalid":
            raise ValueError("accepted activations cannot have an invalid target.")


@dataclass(frozen=True, slots=True, kw_only=True)
class NetHealthEventV1:
    """One exact recipient-level before/after health outcome."""

    event_type: Literal["net_health"] = field(default="net_health", init=False)
    event_id: str
    transition_id: int
    recipient_global_slot: int
    recipient_anchor: Point2D | None
    health_before: float
    health_after: float
    net_delta: float
    outcome: HealthOutcome

    def __post_init__(self) -> None:
        _validate_event_header(self.event_id, self.transition_id)
        _require_slot(self.recipient_global_slot, name="recipient_global_slot")
        if self.recipient_anchor is not None:
            _require_point(self.recipient_anchor, name="recipient_anchor")
        for name in ("health_before", "health_after", "net_delta"):
            _require_finite(getattr(self, name), name=name)
        if not isclose(
            self.health_after - self.health_before,
            self.net_delta,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError("net_delta must equal health_after - health_before.")
        expected: HealthOutcome = (
            "damage"
            if self.net_delta < 0
            else "healing"
            if self.net_delta > 0
            else "unchanged"
        )
        if self.outcome != expected:
            raise ValueError(f"outcome must be {expected!r} for this net_delta.")


@dataclass(frozen=True, slots=True, kw_only=True)
class ChargeDisplacementEventV1:
    """Exact public before/after endpoints, not a continuous Charge path."""

    event_type: Literal["charge_displacement"] = field(
        default="charge_displacement",
        init=False,
    )
    event_id: str
    transition_id: int
    source_global_slot: int
    target_global_slot: int
    start: Point2D
    end: Point2D
    path_kind: ChargePathKind

    def __post_init__(self) -> None:
        _validate_event_header(self.event_id, self.transition_id)
        _require_slot(self.source_global_slot, name="source_global_slot")
        _require_slot(self.target_global_slot, name="target_global_slot")
        _require_point(self.start, name="start")
        _require_point(self.end, name="end")
        if self.path_kind not in ("charge_only", "combined_charge_and_movement"):
            raise ValueError(f"unknown Charge path kind: {self.path_kind!r}.")


@dataclass(frozen=True, slots=True, kw_only=True)
class StatusLifecycleEventV1:
    """One conservative durable-status lifecycle classification."""

    event_type: Literal["status_lifecycle"] = field(
        default="status_lifecycle",
        init=False,
    )
    event_id: str
    transition_id: int
    recipient_global_slot: int
    recipient_anchor: Point2D | None
    token_id: str
    change: StatusLifecycleKind
    duration_before: int
    duration_after: int
    source_class_id: int
    application_event_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_event_header(self.event_id, self.transition_id)
        _require_slot(self.recipient_global_slot, name="recipient_global_slot")
        if self.recipient_anchor is not None:
            _require_point(self.recipient_anchor, name="recipient_anchor")
        _require_text(self.token_id, name="token_id")
        _require_python_int(self.duration_before, name="duration_before", minimum=0)
        _require_python_int(self.duration_after, name="duration_after", minimum=0)
        _require_python_int(self.source_class_id, name="source_class_id")
        _require_tuple_items(
            self.application_event_ids,
            name="application_event_ids",
            item_types=(str,),
        )
        for application_event_id in self.application_event_ids:
            _require_text(application_event_id, name="application_event_id")
        _require_unique(self.application_event_ids, name="application_event_id")


@dataclass(frozen=True, slots=True, kw_only=True)
class RejectedActionEventV1:
    """One exact rejection fact without an inferred causal reason."""

    event_type: Literal["rejected_action"] = field(
        default="rejected_action",
        init=False,
    )
    event_id: str
    transition_id: int
    actor_global_slot: int
    component: RejectionComponent
    actor_anchor: Point2D | None
    target_global_slot: int | None
    target_anchor: Point2D | None
    target_disclosure: TargetDisclosure
    lane: Lane | None
    movement_mask_value: bool
    pair_mask_value: bool

    def __post_init__(self) -> None:
        _validate_event_header(self.event_id, self.transition_id)
        _require_slot(self.actor_global_slot, name="actor_global_slot")
        if self.actor_anchor is not None:
            _require_point(self.actor_anchor, name="actor_anchor")
        if self.target_global_slot is not None:
            _require_slot(self.target_global_slot, name="target_global_slot")
        if self.target_anchor is not None:
            _require_point(self.target_anchor, name="target_anchor")
        if self.component not in ("movement", "combat", "complete_tuple_domain"):
            raise ValueError(f"unknown rejection component: {self.component!r}.")
        _require_lane(self.lane, name="lane", allow_none=True)
        if self.target_disclosure == "public":
            if self.target_global_slot is None or self.target_anchor is None:
                raise ValueError("public rejected targets require identity and anchor.")
        elif self.target_disclosure in ("target_none", "redacted", "invalid"):
            if self.target_global_slot is not None or self.target_anchor is not None:
                raise ValueError(
                    f"{self.target_disclosure} rejected targets must omit target data."
                )
        else:
            raise ValueError(f"unknown target disclosure: {self.target_disclosure!r}.")
        _require_python_bool(self.movement_mask_value, name="movement_mask_value")
        _require_python_bool(self.pair_mask_value, name="pair_mask_value")


type VisualEventV1 = (
    AcceptedActivationEventV1
    | NetHealthEventV1
    | ChargeDisplacementEventV1
    | StatusLifecycleEventV1
    | RejectedActionEventV1
)


@dataclass(frozen=True, slots=True, kw_only=True)
class VisualEventBatchV1:
    """The one latest authoritative transition's ordered presentation events."""

    schema_version: int
    transition_id: int
    simulator_step: int
    events: tuple[VisualEventV1, ...]

    def __post_init__(self) -> None:
        _require_python_int(self.schema_version, name="schema_version")
        if self.schema_version != EVENT_SCHEMA_VERSION:
            raise ValueError(
                f"event schema_version must be {EVENT_SCHEMA_VERSION}; "
                f"got {self.schema_version}."
            )
        _require_python_int(self.transition_id, name="transition_id", minimum=0)
        _require_python_int(self.simulator_step, name="simulator_step", minimum=0)
        _require_tuple_items(
            self.events,
            name="events",
            item_types=(
                AcceptedActivationEventV1,
                NetHealthEventV1,
                ChargeDisplacementEventV1,
                StatusLifecycleEventV1,
                RejectedActionEventV1,
            ),
        )
        event_ids = tuple(event.event_id for event in self.events)
        _require_unique(event_ids, name="event_id")
        if any(event.transition_id != self.transition_id for event in self.events):
            raise ValueError("every event must belong to the batch transition_id.")
        accepted_activation_ids = {
            event.event_id
            for event in self.events
            if type(event) is AcceptedActivationEventV1
        }
        for event in self.events:
            if type(event) is not StatusLifecycleEventV1:
                continue
            missing_application_ids = tuple(
                application_event_id
                for application_event_id in event.application_event_ids
                if application_event_id not in accepted_activation_ids
            )
            if missing_application_ids:
                raise ValueError(
                    "status lifecycle application_event_ids must reference "
                    "AcceptedActivationEventV1 events in the same batch; "
                    f"missing {missing_application_ids!r}."
                )


@dataclass(frozen=True, slots=True, kw_only=True)
class VisualAgentAnchorV2:
    """One public actor position at an explicit canonical transition phase."""

    phase: VisualAnchorPhaseV2
    global_slot: int
    public_agent_id: str
    position: Point2D

    def __post_init__(self) -> None:
        if self.phase not in ("transition_start", "post_charge", "successor"):
            raise ValueError(f"unknown visual anchor phase: {self.phase!r}.")
        _require_slot(self.global_slot, name="global_slot")
        _require_text(self.public_agent_id, name="public_agent_id")
        _require_point(self.position, name="position")


@dataclass(frozen=True, slots=True, kw_only=True)
class VisualTeamAnchorV2:
    """One non-spatial team cue at an explicit canonical transition phase."""

    phase: VisualAnchorPhaseV2
    team_index: int
    team_id: int

    def __post_init__(self) -> None:
        if self.phase not in ("transition_start", "post_charge", "successor"):
            raise ValueError(f"unknown visual anchor phase: {self.phase!r}.")
        _require_python_int(self.team_index, name="team_index", minimum=0)
        if self.team_index not in (0, 1):
            raise ValueError("team_index must be zero or one.")
        _require_python_int(self.team_id, name="team_id", minimum=1)
        if self.team_id != self.team_index + 1:
            raise ValueError("team_id must equal team_index + 1.")


def _require_agent_anchor(
    anchor: VisualAgentAnchorV2,
    *,
    name: str,
    phase: VisualAnchorPhaseV2,
    global_slot: int,
) -> None:
    if type(anchor) is not VisualAgentAnchorV2:
        raise ValueError(f"{name} must be the exact VisualAgentAnchorV2.")
    if anchor.phase != phase or anchor.global_slot != global_slot:
        raise ValueError(f"{name} must join slot {global_slot} at phase {phase}.")


def _require_optional_agent_anchor(
    anchor: VisualAgentAnchorV2 | None,
    *,
    name: str,
    phase: VisualAnchorPhaseV2,
    global_slot: int | None,
) -> None:
    if (anchor is None) != (global_slot is None):
        raise ValueError(f"{name} presence must match its nullable global slot.")
    if anchor is not None and global_slot is not None:
        _require_agent_anchor(
            anchor,
            name=name,
            phase=phase,
            global_slot=global_slot,
        )


def _require_int32(value: int, *, name: str) -> None:
    _require_python_int(value, name=name)
    if not -(2**31) <= value < 2**31:
        raise ValueError(f"{name} must fit signed int32.")


def _require_slot_tuple(value: tuple[int, ...], *, name: str) -> None:
    if type(value) is not tuple:
        raise ValueError(f"{name} must be a Python tuple.")
    for slot in value:
        _require_slot(slot, name=f"{name} item")
    if value != tuple(sorted(set(value))):
        raise ValueError(f"{name} must be sorted and unique.")


@dataclass(frozen=True, slots=True, kw_only=True)
class CanonicalVisualEventBaseV2:
    """Common string identity and gap-free ordinal for one CP2 event."""

    event_id: str
    transition_id: str
    ordinal: int

    def __post_init__(self) -> None:
        _require_text(self.transition_id, name="transition_id")
        _require_python_int(self.ordinal, name="ordinal", minimum=0)
        if self.event_id != f"{self.transition_id}:event:{self.ordinal:04d}":
            raise ValueError("visual event ID must remain canonical and gap-free.")


@dataclass(frozen=True, slots=True, kw_only=True)
class ActionRejectedEventV2(CanonicalVisualEventBaseV2):
    event_type: Literal["action_rejected"] = field(
        default="action_rejected", init=False
    )
    phase_rank: Literal[10] = field(default=10, init=False)
    actor_global_slot: int
    actor_public_agent_id: str
    actor_configured_active: bool
    rejection_component: Literal["domain", "movement", "combat_pair"]
    submitted_move_action: int
    submitted_select_target_action: int
    submitted_use_ultimate_action: int
    actor_anchor: VisualAgentAnchorV2 | None

    def __post_init__(self) -> None:
        CanonicalVisualEventBaseV2.__post_init__(self)
        _require_slot(self.actor_global_slot, name="actor_global_slot")
        _require_text(self.actor_public_agent_id, name="actor_public_agent_id")
        _require_python_bool(
            self.actor_configured_active,
            name="actor_configured_active",
        )
        if self.rejection_component not in ("domain", "movement", "combat_pair"):
            raise ValueError("unknown action rejection component.")
        for name in (
            "submitted_move_action",
            "submitted_select_target_action",
            "submitted_use_ultimate_action",
        ):
            _require_int32(cast(int, getattr(self, name)), name=name)
        if self.actor_configured_active:
            if self.actor_anchor is None:
                raise ValueError("active rejected actors require a start anchor.")
            _require_agent_anchor(
                self.actor_anchor,
                name="actor_anchor",
                phase="transition_start",
                global_slot=self.actor_global_slot,
            )
            if self.actor_anchor.public_agent_id != self.actor_public_agent_id:
                raise ValueError("rejected actor anchor must join its public ID.")
        elif self.actor_anchor is not None:
            raise ValueError("inactive rejected actors must remain feed-only.")


@dataclass(frozen=True, slots=True, kw_only=True)
class AbilityActivatedEventV2(CanonicalVisualEventBaseV2):
    event_type: Literal["ability_activated"] = field(
        default="ability_activated", init=False
    )
    phase_rank: Literal[20] = field(default=20, init=False)
    source_global_slot: int
    ability_component: Literal["basic", "ultimate"]
    recipient_global_slot: int | None
    source_anchor: VisualAgentAnchorV2
    recipient_anchor: VisualAgentAnchorV2 | None

    def __post_init__(self) -> None:
        CanonicalVisualEventBaseV2.__post_init__(self)
        _require_slot(self.source_global_slot, name="source_global_slot")
        if self.ability_component not in ("basic", "ultimate"):
            raise ValueError("ability_component must be basic or ultimate.")
        if self.recipient_global_slot is not None:
            _require_slot(self.recipient_global_slot, name="recipient_global_slot")
        _require_agent_anchor(
            self.source_anchor,
            name="source_anchor",
            phase="transition_start",
            global_slot=self.source_global_slot,
        )
        _require_optional_agent_anchor(
            self.recipient_anchor,
            name="recipient_anchor",
            phase="transition_start",
            global_slot=self.recipient_global_slot,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceDamageOutputEventV2(CanonicalVisualEventBaseV2):
    event_type: Literal["source_damage_output"] = field(
        default="source_damage_output", init=False
    )
    phase_rank: Literal[30] = field(default=30, init=False)
    source_global_slot: int
    recipient_global_slot: int | None
    raw_damage_output: float
    source_modified_damage_output: float
    recipient_damage_modifier: float
    mage_damage_aura_covering_emitter_global_slots: tuple[int, ...]
    warrior_mitigation_aura_covering_emitter_global_slots: tuple[int, ...]
    source_anchor: VisualAgentAnchorV2
    recipient_anchor: VisualAgentAnchorV2 | None

    def __post_init__(self) -> None:
        CanonicalVisualEventBaseV2.__post_init__(self)
        _require_slot(self.source_global_slot, name="source_global_slot")
        if self.recipient_global_slot is not None:
            _require_slot(self.recipient_global_slot, name="recipient_global_slot")
        for name in (
            "raw_damage_output",
            "source_modified_damage_output",
            "recipient_damage_modifier",
        ):
            _require_nonnegative_finite(cast(float, getattr(self, name)), name=name)
        _require_slot_tuple(
            self.mage_damage_aura_covering_emitter_global_slots,
            name="mage_damage_aura_covering_emitter_global_slots",
        )
        _require_slot_tuple(
            self.warrior_mitigation_aura_covering_emitter_global_slots,
            name="warrior_mitigation_aura_covering_emitter_global_slots",
        )
        _require_agent_anchor(
            self.source_anchor,
            name="source_anchor",
            phase="transition_start",
            global_slot=self.source_global_slot,
        )
        _require_optional_agent_anchor(
            self.recipient_anchor,
            name="recipient_anchor",
            phase="transition_start",
            global_slot=self.recipient_global_slot,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceHealingOutputEventV2(CanonicalVisualEventBaseV2):
    event_type: Literal["source_healing_output"] = field(
        default="source_healing_output", init=False
    )
    phase_rank: Literal[30] = field(default=30, init=False)
    source_global_slot: int
    recipient_global_slot: int | None
    raw_healing_output: float
    source_modified_healing_output: float
    recipient_healing_modifier: float
    source_anchor: VisualAgentAnchorV2
    recipient_anchor: VisualAgentAnchorV2 | None

    def __post_init__(self) -> None:
        CanonicalVisualEventBaseV2.__post_init__(self)
        _require_slot(self.source_global_slot, name="source_global_slot")
        if self.recipient_global_slot is not None:
            _require_slot(self.recipient_global_slot, name="recipient_global_slot")
        for name in (
            "raw_healing_output",
            "source_modified_healing_output",
            "recipient_healing_modifier",
        ):
            _require_nonnegative_finite(cast(float, getattr(self, name)), name=name)
        _require_agent_anchor(
            self.source_anchor,
            name="source_anchor",
            phase="transition_start",
            global_slot=self.source_global_slot,
        )
        _require_optional_agent_anchor(
            self.recipient_anchor,
            name="recipient_anchor",
            phase="transition_start",
            global_slot=self.recipient_global_slot,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class RecipientHealthResolutionEventV2(CanonicalVisualEventBaseV2):
    event_type: Literal["recipient_health_resolution"] = field(
        default="recipient_health_resolution", init=False
    )
    phase_rank: Literal[40] = field(default=40, init=False)
    recipient_global_slot: int
    transition_start_health: float
    total_effective_damage: float
    total_effective_healing: float
    health_after_combat_resolution: float
    realized_net_health_change: float
    recipient_anchor: VisualAgentAnchorV2

    def __post_init__(self) -> None:
        CanonicalVisualEventBaseV2.__post_init__(self)
        _require_slot(self.recipient_global_slot, name="recipient_global_slot")
        for name in (
            "transition_start_health",
            "total_effective_damage",
            "total_effective_healing",
            "health_after_combat_resolution",
        ):
            _require_nonnegative_finite(cast(float, getattr(self, name)), name=name)
        _require_finite(
            self.realized_net_health_change,
            name="realized_net_health_change",
        )
        _require_agent_anchor(
            self.recipient_anchor,
            name="recipient_anchor",
            phase="transition_start",
            global_slot=self.recipient_global_slot,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class CombatCountdownResetEventV2(CanonicalVisualEventBaseV2):
    event_type: Literal["combat_countdown_reset"] = field(
        default="combat_countdown_reset", init=False
    )
    phase_rank: Literal[50] = field(default=50, init=False)
    agent_global_slot: int
    agent_anchor: VisualAgentAnchorV2

    def __post_init__(self) -> None:
        CanonicalVisualEventBaseV2.__post_init__(self)
        _require_slot(self.agent_global_slot, name="agent_global_slot")
        _require_agent_anchor(
            self.agent_anchor,
            name="agent_anchor",
            phase="transition_start",
            global_slot=self.agent_global_slot,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class HealthRegeneratedEventV2(CanonicalVisualEventBaseV2):
    event_type: Literal["health_regenerated"] = field(
        default="health_regenerated", init=False
    )
    phase_rank: Literal[50] = field(default=50, init=False)
    agent_global_slot: int
    actual_health_regenerated: float
    agent_anchor: VisualAgentAnchorV2

    def __post_init__(self) -> None:
        CanonicalVisualEventBaseV2.__post_init__(self)
        _require_slot(self.agent_global_slot, name="agent_global_slot")
        _require_nonnegative_finite(
            self.actual_health_regenerated,
            name="actual_health_regenerated",
        )
        _require_agent_anchor(
            self.agent_anchor,
            name="agent_anchor",
            phase="transition_start",
            global_slot=self.agent_global_slot,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class CooldownStartedEventV2(CanonicalVisualEventBaseV2):
    event_type: Literal["cooldown_started"] = field(
        default="cooldown_started", init=False
    )
    phase_rank: Literal[60] = field(default=60, init=False)
    agent_global_slot: int
    agent_anchor: VisualAgentAnchorV2

    def __post_init__(self) -> None:
        CanonicalVisualEventBaseV2.__post_init__(self)
        _require_slot(self.agent_global_slot, name="agent_global_slot")
        _require_agent_anchor(
            self.agent_anchor,
            name="agent_anchor",
            phase="transition_start",
            global_slot=self.agent_global_slot,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class CooldownReadyEventV2(CanonicalVisualEventBaseV2):
    event_type: Literal["cooldown_ready"] = field(default="cooldown_ready", init=False)
    phase_rank: Literal[60] = field(default=60, init=False)
    agent_global_slot: int
    agent_anchor: VisualAgentAnchorV2

    def __post_init__(self) -> None:
        CanonicalVisualEventBaseV2.__post_init__(self)
        _require_slot(self.agent_global_slot, name="agent_global_slot")
        _require_agent_anchor(
            self.agent_anchor,
            name="agent_anchor",
            phase="transition_start",
            global_slot=self.agent_global_slot,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ChargePhaseDisplacementEventV2(CanonicalVisualEventBaseV2):
    event_type: Literal["charge_phase_displacement"] = field(
        default="charge_phase_displacement", init=False
    )
    phase_rank: Literal[70] = field(default=70, init=False)
    agent_global_slot: int
    realized_displacement: Point2D
    start_anchor: VisualAgentAnchorV2
    end_anchor: VisualAgentAnchorV2

    def __post_init__(self) -> None:
        CanonicalVisualEventBaseV2.__post_init__(self)
        _require_slot(self.agent_global_slot, name="agent_global_slot")
        _require_point(self.realized_displacement, name="realized_displacement")
        _require_agent_anchor(
            self.start_anchor,
            name="start_anchor",
            phase="transition_start",
            global_slot=self.agent_global_slot,
        )
        _require_agent_anchor(
            self.end_anchor,
            name="end_anchor",
            phase="post_charge",
            global_slot=self.agent_global_slot,
        )
        expected_end = (
            self.start_anchor.position[0] + self.realized_displacement[0],
            self.start_anchor.position[1] + self.realized_displacement[1],
        )
        if not _points_close(self.end_anchor.position, expected_end):
            raise ValueError("Charge end anchor must apply its recorded displacement.")


@dataclass(frozen=True, slots=True, kw_only=True)
class OrdinaryMovementPhaseDisplacementEventV2(CanonicalVisualEventBaseV2):
    event_type: Literal["ordinary_movement_phase_displacement"] = field(
        default="ordinary_movement_phase_displacement", init=False
    )
    phase_rank: Literal[80] = field(default=80, init=False)
    agent_global_slot: int
    realized_displacement: Point2D
    start_anchor: VisualAgentAnchorV2
    end_anchor: VisualAgentAnchorV2

    def __post_init__(self) -> None:
        CanonicalVisualEventBaseV2.__post_init__(self)
        _require_slot(self.agent_global_slot, name="agent_global_slot")
        _require_point(self.realized_displacement, name="realized_displacement")
        _require_agent_anchor(
            self.start_anchor,
            name="start_anchor",
            phase="post_charge",
            global_slot=self.agent_global_slot,
        )
        _require_agent_anchor(
            self.end_anchor,
            name="end_anchor",
            phase="successor",
            global_slot=self.agent_global_slot,
        )
        expected_end = (
            self.start_anchor.position[0] + self.realized_displacement[0],
            self.start_anchor.position[1] + self.realized_displacement[1],
        )
        if not _points_close(self.end_anchor.position, expected_end):
            raise ValueError(
                "ordinary-movement end anchor must apply its recorded displacement."
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentDiedEventV2(CanonicalVisualEventBaseV2):
    event_type: Literal["agent_died"] = field(default="agent_died", init=False)
    phase_rank: Literal[90] = field(default=90, init=False)
    recipient_global_slot: int
    recipient_anchor: VisualAgentAnchorV2

    def __post_init__(self) -> None:
        CanonicalVisualEventBaseV2.__post_init__(self)
        _require_slot(self.recipient_global_slot, name="recipient_global_slot")
        _require_agent_anchor(
            self.recipient_anchor,
            name="recipient_anchor",
            phase="successor",
            global_slot=self.recipient_global_slot,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class LethalDamageContributionEventV2(CanonicalVisualEventBaseV2):
    event_type: Literal["lethal_damage_contribution"] = field(
        default="lethal_damage_contribution", init=False
    )
    phase_rank: Literal[90] = field(default=90, init=False)
    source_global_slot: int
    recipient_global_slot: int
    attributed_death_damage: float
    source_anchor: VisualAgentAnchorV2
    recipient_anchor: VisualAgentAnchorV2

    def __post_init__(self) -> None:
        CanonicalVisualEventBaseV2.__post_init__(self)
        _require_slot(self.source_global_slot, name="source_global_slot")
        _require_slot(self.recipient_global_slot, name="recipient_global_slot")
        _require_nonnegative_finite(
            self.attributed_death_damage,
            name="attributed_death_damage",
        )
        _require_agent_anchor(
            self.source_anchor,
            name="source_anchor",
            phase="successor",
            global_slot=self.source_global_slot,
        )
        _require_agent_anchor(
            self.recipient_anchor,
            name="recipient_anchor",
            phase="successor",
            global_slot=self.recipient_global_slot,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class StatusVisualEventBaseV2(CanonicalVisualEventBaseV2):
    """Common direct recipient/status payload for one lifecycle event."""

    recipient_global_slot: int
    status_channel: int
    status_id: str
    recipient_anchor: VisualAgentAnchorV2

    def __post_init__(self) -> None:
        CanonicalVisualEventBaseV2.__post_init__(self)
        _require_slot(self.recipient_global_slot, name="recipient_global_slot")
        _require_python_int(self.status_channel, name="status_channel", minimum=0)
        _require_text(self.status_id, name="status_id")
        if (
            self.status_channel >= len(CATALOG_STATUS_ID_BY_CHANNEL)
            or CATALOG_STATUS_ID_BY_CHANNEL[self.status_channel] != self.status_id
        ):
            raise ValueError(
                "status event channel and catalog status ID must retain V1 identity."
            )
        _require_agent_anchor(
            self.recipient_anchor,
            name="recipient_anchor",
            phase="successor",
            global_slot=self.recipient_global_slot,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class StatusAgedToZeroEventV2(StatusVisualEventBaseV2):
    event_type: Literal["status_aged_to_zero"] = field(
        default="status_aged_to_zero", init=False
    )
    phase_rank: Literal[100] = field(default=100, init=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class StatusBrokenByDamageEventV2(StatusVisualEventBaseV2):
    event_type: Literal["status_broken_by_damage"] = field(
        default="status_broken_by_damage", init=False
    )
    phase_rank: Literal[100] = field(default=100, init=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class StatusAppliedEventV2(StatusVisualEventBaseV2):
    event_type: Literal["status_applied"] = field(default="status_applied", init=False)
    phase_rank: Literal[100] = field(default=100, init=False)
    source_global_slot: int
    source_anchor: VisualAgentAnchorV2

    def __post_init__(self) -> None:
        StatusVisualEventBaseV2.__post_init__(self)
        _require_slot(self.source_global_slot, name="source_global_slot")
        _require_agent_anchor(
            self.source_anchor,
            name="source_anchor",
            phase="successor",
            global_slot=self.source_global_slot,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class StatusRefreshedOrExtendedEventV2(StatusVisualEventBaseV2):
    event_type: Literal["status_refreshed_or_extended"] = field(
        default="status_refreshed_or_extended", init=False
    )
    phase_rank: Literal[100] = field(default=100, init=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class StatusClearedByNewDeathEventV2(StatusVisualEventBaseV2):
    event_type: Literal["status_cleared_by_new_death"] = field(
        default="status_cleared_by_new_death", init=False
    )
    phase_rank: Literal[100] = field(default=100, init=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class SpawnShieldExpiredEventV2(CanonicalVisualEventBaseV2):
    event_type: Literal["spawn_shield_expired"] = field(
        default="spawn_shield_expired", init=False
    )
    phase_rank: Literal[110] = field(default=110, init=False)
    agent_global_slot: int
    agent_anchor: VisualAgentAnchorV2

    def __post_init__(self) -> None:
        CanonicalVisualEventBaseV2.__post_init__(self)
        _require_slot(self.agent_global_slot, name="agent_global_slot")
        _require_agent_anchor(
            self.agent_anchor,
            name="agent_anchor",
            phase="successor",
            global_slot=self.agent_global_slot,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class RespawnWaveOccurredEventV2(CanonicalVisualEventBaseV2):
    event_type: Literal["respawn_wave_occurred"] = field(
        default="respawn_wave_occurred", init=False
    )
    phase_rank: Literal[120] = field(default=120, init=False)
    team_index: int
    team_id: int
    team_anchor: VisualTeamAnchorV2

    def __post_init__(self) -> None:
        CanonicalVisualEventBaseV2.__post_init__(self)
        _require_python_int(self.team_index, name="team_index", minimum=0)
        if self.team_index not in (0, 1):
            raise ValueError("team_index must be zero or one.")
        _require_python_int(self.team_id, name="team_id", minimum=1)
        if self.team_id != self.team_index + 1:
            raise ValueError("team_id must equal team_index + 1.")
        if type(self.team_anchor) is not VisualTeamAnchorV2 or (
            self.team_anchor.phase != "successor"
            or self.team_anchor.team_index != self.team_index
            or self.team_anchor.team_id != self.team_id
        ):
            raise ValueError("team_anchor must join the successor team wave.")


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentRespawnedEventV2(CanonicalVisualEventBaseV2):
    event_type: Literal["agent_respawned"] = field(
        default="agent_respawned", init=False
    )
    phase_rank: Literal[120] = field(default=120, init=False)
    agent_global_slot: int
    team_id: int
    realized_successor_position: Point2D
    agent_anchor: VisualAgentAnchorV2

    def __post_init__(self) -> None:
        CanonicalVisualEventBaseV2.__post_init__(self)
        _require_slot(self.agent_global_slot, name="agent_global_slot")
        _require_python_int(self.team_id, name="team_id", minimum=1)
        if self.team_id not in (1, 2):
            raise ValueError("team_id must be one or two.")
        _require_point(
            self.realized_successor_position,
            name="realized_successor_position",
        )
        _require_agent_anchor(
            self.agent_anchor,
            name="agent_anchor",
            phase="successor",
            global_slot=self.agent_global_slot,
        )
        if self.agent_anchor.position != self.realized_successor_position:
            raise ValueError(
                "respawn anchor must equal the recorded successor position."
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class TeamDeathmatchScoreChangedEventV2(CanonicalVisualEventBaseV2):
    """One authoritative team-score change after lifecycle resolution."""

    event_type: Literal["team_deathmatch_score_changed"] = field(
        default="team_deathmatch_score_changed", init=False
    )
    phase_rank: Literal[130] = field(default=130, init=False)
    team_index: Literal[0, 1]
    team_id: Literal[1, 2]
    score_increment: int
    previous_score: int
    successor_score: int
    team_anchor: VisualTeamAnchorV2

    def __post_init__(self) -> None:
        CanonicalVisualEventBaseV2.__post_init__(self)
        _require_python_int(self.team_index, name="team_index", minimum=0)
        _require_python_int(self.team_id, name="team_id", minimum=1)
        if self.team_index not in (0, 1) or self.team_id != self.team_index + 1:
            raise ValueError("Team Deathmatch score event has an invalid team join.")
        for name in ("score_increment", "previous_score", "successor_score"):
            _require_int32(cast(int, getattr(self, name)), name=name)
        if self.score_increment <= 0 or self.previous_score < 0:
            raise ValueError(
                "Team Deathmatch score events require a positive increment and "
                "nonnegative previous score."
            )
        if self.successor_score != self.previous_score + self.score_increment:
            raise ValueError(
                "Team Deathmatch successor score must equal previous score plus "
                "the recorded increment."
            )
        if (
            type(self.team_anchor) is not VisualTeamAnchorV2
            or self.team_anchor.phase != "successor"
            or self.team_anchor.team_index != self.team_index
            or self.team_anchor.team_id != self.team_id
        ):
            raise ValueError("team_anchor must join the successor scoring team.")


@dataclass(frozen=True, slots=True, kw_only=True)
class TeamDeathmatchCompletedEventV2(CanonicalVisualEventBaseV2):
    """One authoritative Team Deathmatch result and its completion basis."""

    event_type: Literal["team_deathmatch_completed"] = field(
        default="team_deathmatch_completed", init=False
    )
    phase_rank: Literal[140] = field(default=140, init=False)
    outcome: Literal["team_a_win", "team_b_win", "draw"]
    completion_basis: Literal[
        "score_threshold", "horizon", "score_threshold_at_horizon"
    ]

    def __post_init__(self) -> None:
        CanonicalVisualEventBaseV2.__post_init__(self)
        if self.outcome not in ("team_a_win", "team_b_win", "draw"):
            raise ValueError("Team Deathmatch completion outcome is invalid.")
        if self.completion_basis not in (
            "score_threshold",
            "horizon",
            "score_threshold_at_horizon",
        ):
            raise ValueError("Team Deathmatch completion basis is invalid.")


@dataclass(frozen=True, slots=True, kw_only=True)
class VisualAgentPhaseTrajectoryV2:
    """One actor's explicit start, post-Charge, and successor anchors."""

    global_slot: int
    public_agent_id: str
    transition_start: VisualAgentAnchorV2
    post_charge: VisualAgentAnchorV2
    successor: VisualAgentAnchorV2

    def __post_init__(self) -> None:
        _require_slot(self.global_slot, name="global_slot")
        _require_text(self.public_agent_id, name="public_agent_id")
        for name, phase in (
            ("transition_start", "transition_start"),
            ("post_charge", "post_charge"),
            ("successor", "successor"),
        ):
            anchor = cast(VisualAgentAnchorV2, getattr(self, name))
            _require_agent_anchor(
                anchor,
                name=name,
                phase=cast(VisualAnchorPhaseV2, phase),
                global_slot=self.global_slot,
            )
            if anchor.public_agent_id != self.public_agent_id:
                raise ValueError("phase anchors must retain one public agent ID.")


type VisualEventV2 = (
    ActionRejectedEventV2
    | AbilityActivatedEventV2
    | SourceDamageOutputEventV2
    | SourceHealingOutputEventV2
    | RecipientHealthResolutionEventV2
    | CombatCountdownResetEventV2
    | HealthRegeneratedEventV2
    | CooldownStartedEventV2
    | CooldownReadyEventV2
    | ChargePhaseDisplacementEventV2
    | OrdinaryMovementPhaseDisplacementEventV2
    | AgentDiedEventV2
    | LethalDamageContributionEventV2
    | StatusAgedToZeroEventV2
    | StatusBrokenByDamageEventV2
    | StatusAppliedEventV2
    | StatusRefreshedOrExtendedEventV2
    | StatusClearedByNewDeathEventV2
    | SpawnShieldExpiredEventV2
    | RespawnWaveOccurredEventV2
    | AgentRespawnedEventV2
    | TeamDeathmatchScoreChangedEventV2
    | TeamDeathmatchCompletedEventV2
)

_VISUAL_EVENT_V2_TYPES: tuple[type[object], ...] = (
    ActionRejectedEventV2,
    AbilityActivatedEventV2,
    SourceDamageOutputEventV2,
    SourceHealingOutputEventV2,
    RecipientHealthResolutionEventV2,
    CombatCountdownResetEventV2,
    HealthRegeneratedEventV2,
    CooldownStartedEventV2,
    CooldownReadyEventV2,
    ChargePhaseDisplacementEventV2,
    OrdinaryMovementPhaseDisplacementEventV2,
    AgentDiedEventV2,
    LethalDamageContributionEventV2,
    StatusAgedToZeroEventV2,
    StatusBrokenByDamageEventV2,
    StatusAppliedEventV2,
    StatusRefreshedOrExtendedEventV2,
    StatusClearedByNewDeathEventV2,
    SpawnShieldExpiredEventV2,
    RespawnWaveOccurredEventV2,
    AgentRespawnedEventV2,
    TeamDeathmatchScoreChangedEventV2,
    TeamDeathmatchCompletedEventV2,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class VisualEventBatchV2:
    """All canonical CP2 events for one transition, unchanged in order."""

    schema_version: int
    episode_id: str
    transition_index: int
    transition_id: str
    start_frame_id: str
    successor_frame_id: str
    start_simulator_step_count: int
    successor_simulator_step_count: int
    public_agent_id_by_global_slot: tuple[str, ...]
    configured_active_by_global_slot: tuple[bool, ...]
    agent_phase_trajectories: tuple[VisualAgentPhaseTrajectoryV2, ...]
    events: tuple[VisualEventV2, ...]

    def __post_init__(self) -> None:
        _require_python_int(self.schema_version, name="schema_version")
        if self.schema_version != EVENT_V2_SCHEMA_VERSION:
            raise ValueError("unknown VisualEventBatchV2 schema version.")
        _require_text(self.episode_id, name="episode_id")
        _require_python_int(
            self.transition_index,
            name="transition_index",
            minimum=0,
        )
        expected_transition_id = f"{self.episode_id}:transition:{self.transition_index}"
        if self.transition_id != expected_transition_id:
            raise ValueError("visual batch transition ID is not canonical.")
        if self.start_frame_id != f"{self.episode_id}:frame:{self.transition_index}":
            raise ValueError("visual batch start frame ID is not canonical.")
        if self.successor_frame_id != (
            f"{self.episode_id}:frame:{self.transition_index + 1}"
        ):
            raise ValueError("visual batch successor frame ID is not canonical.")
        for name in (
            "start_simulator_step_count",
            "successor_simulator_step_count",
        ):
            _require_python_int(cast(int, getattr(self, name)), name=name, minimum=0)
        if self.successor_simulator_step_count != (self.start_simulator_step_count + 1):
            raise ValueError("V2 event batches require adjacent simulator step counts.")
        _require_tuple_items(
            self.public_agent_id_by_global_slot,
            name="public_agent_id_by_global_slot",
            item_types=(str,),
        )
        if len(self.public_agent_id_by_global_slot) != MAX_AGENT_SLOTS:
            raise ValueError("V2 event batches require all ten public agent IDs.")
        for public_agent_id in self.public_agent_id_by_global_slot:
            _require_text(public_agent_id, name="public_agent_id")
        _require_unique(
            self.public_agent_id_by_global_slot,
            name="public_agent_id_by_global_slot",
        )
        if (
            type(self.configured_active_by_global_slot) is not tuple
            or len(self.configured_active_by_global_slot) != MAX_AGENT_SLOTS
        ):
            raise ValueError("V2 event batches require all ten active flags.")
        for active in self.configured_active_by_global_slot:
            _require_python_bool(active, name="configured_active")
        _require_tuple_items(
            self.agent_phase_trajectories,
            name="agent_phase_trajectories",
            item_types=(VisualAgentPhaseTrajectoryV2,),
        )
        trajectory_slots = tuple(
            row.global_slot for row in self.agent_phase_trajectories
        )
        if trajectory_slots != tuple(sorted(trajectory_slots)) or len(
            trajectory_slots
        ) != len(set(trajectory_slots)):
            raise ValueError("agent phase trajectories require canonical unique slots.")
        expected_trajectory_slots = tuple(
            slot
            for slot, active in enumerate(self.configured_active_by_global_slot)
            if active
        )
        if trajectory_slots != expected_trajectory_slots:
            raise ValueError("phase trajectories must cover configured-active slots.")
        for trajectory in self.agent_phase_trajectories:
            if (
                trajectory.public_agent_id
                != self.public_agent_id_by_global_slot[trajectory.global_slot]
            ):
                raise ValueError("phase trajectory must join public roster identity.")
        _require_tuple_items(
            self.events,
            name="events",
            item_types=_VISUAL_EVENT_V2_TYPES,
        )
        if tuple(event.ordinal for event in self.events) != tuple(
            range(len(self.events))
        ):
            raise ValueError("V2 event ordinals must be gap-free and ordered.")
        if any(event.transition_id != self.transition_id for event in self.events):
            raise ValueError("every V2 event must join its containing transition.")
        if tuple(event.phase_rank for event in self.events) != tuple(
            sorted(event.phase_rank for event in self.events)
        ):
            raise ValueError("V2 event phase ranks must retain canonical order.")
        trajectory_by_slot = {
            row.global_slot: row for row in self.agent_phase_trajectories
        }
        for event in self.events:
            if type(event) is ActionRejectedEventV2 and (
                event.actor_public_agent_id
                != self.public_agent_id_by_global_slot[event.actor_global_slot]
                or event.actor_configured_active
                != self.configured_active_by_global_slot[event.actor_global_slot]
            ):
                raise ValueError("rejected action must join batch roster identity.")
            for field_name in (
                "actor_anchor",
                "source_anchor",
                "recipient_anchor",
                "agent_anchor",
                "start_anchor",
                "end_anchor",
            ):
                anchor = getattr(event, field_name, None)
                if anchor is None:
                    continue
                if type(anchor) is not VisualAgentAnchorV2:
                    raise ValueError("V2 event agent anchors must use exact roots.")
                trajectory = trajectory_by_slot.get(anchor.global_slot)
                if trajectory is None or (getattr(trajectory, anchor.phase) != anchor):
                    raise ValueError(
                        "V2 event anchors must join their canonical phase trajectory."
                    )


@dataclass(frozen=True, slots=True, kw_only=True)
class ResearcherAnalyzerProjectionV2:
    """Stable researcher scene plus its exact incoming event batch."""

    schema_version: int
    scene: BattlefieldSceneV2
    incoming_events: VisualEventBatchV2 | None
    status_source_evidence: StatusSourceEvidenceStateV2

    def __post_init__(self) -> None:
        _require_python_int(self.schema_version, name="schema_version")
        if self.schema_version != RESEARCHER_ANALYZER_PROJECTION_SCHEMA_VERSION:
            raise ValueError("unknown researcher analyzer projection version.")
        if type(self.scene) is not BattlefieldSceneV2:
            raise ValueError("scene must be the exact BattlefieldSceneV2 root.")
        if type(self.status_source_evidence) is not StatusSourceEvidenceStateV2:
            raise ValueError("status_source_evidence must use its exact state root.")
        state = self.status_source_evidence
        if (
            state.episode_id != self.scene.episode_id
            or state.frame_index != self.scene.frame_index
            or state.frame_id != self.scene.frame_id
        ):
            raise ValueError("status-source evidence must join the selected scene.")
        if self.scene.frame_index == 0:
            if self.incoming_events is not None:
                raise ValueError("frame zero cannot have incoming visual events.")
        elif type(self.incoming_events) is not VisualEventBatchV2:
            raise ValueError("non-initial scenes require exact incoming V2 events.")
        if self.incoming_events is not None:
            batch = self.incoming_events
            if (
                batch.episode_id != self.scene.episode_id
                or batch.transition_id != self.scene.incoming_transition_id
                or batch.successor_frame_id != self.scene.frame_id
                or tuple(event.event_id for event in batch.events)
                != self.scene.incoming_event_ids
            ):
                raise ValueError("incoming V2 events must join the selected scene.")
            agents = {row.global_slot: row for row in self.scene.agents}
            if tuple(
                row.global_slot for row in batch.agent_phase_trajectories
            ) != tuple(agents):
                raise ValueError("V2 phase trajectories must join all scene agents.")
            for trajectory in batch.agent_phase_trajectories:
                agent = agents[trajectory.global_slot]
                if (
                    trajectory.public_agent_id != agent.public_agent_id
                    or trajectory.successor.position != agent.position
                ):
                    raise ValueError(
                        "V2 phase trajectory must join successor scene identity."
                    )
            event_by_id = {event.event_id: event for event in batch.events}
            for agent in self.scene.agents:
                if agent.respawn_event_id is None:
                    continue
                respawn_event = event_by_id.get(agent.respawn_event_id)
                if (
                    type(respawn_event) is not AgentRespawnedEventV2
                    or respawn_event.agent_global_slot != agent.global_slot
                ):
                    raise ValueError(
                        "agent respawn evidence must identify the same agent's "
                        "incoming respawn event."
                    )
        state_by_key = {
            (row.recipient_global_slot, row.status_channel): row
            for row in state.active_statuses
        }
        scene_status_keys: set[tuple[int, int]] = set()
        for agent in self.scene.agents:
            for status in agent.statuses:
                key = (agent.global_slot, status.status_channel)
                row = state_by_key.get(key)
                if row is None or (
                    row.recipient_public_agent_id != agent.public_agent_id
                    or row.status_id != status.status_id
                    or row.direct_source_evidence != status.direct_source_evidence
                ):
                    raise ValueError(
                        "scene status evidence must equal its frame-bound state."
                    )
                scene_status_keys.add(key)
        if scene_status_keys != set(state_by_key):
            raise ValueError("status-source state must cover exactly active statuses.")


def _validate_event_header(event_id: str, transition_id: int) -> None:
    _require_text(event_id, name="event_id")
    _require_python_int(transition_id, name="transition_id", minimum=0)


def to_jsonable(value: object) -> object:
    """Recursively convert a scene/event record into JSON-compatible values."""
    raw_fields = cast(
        object,
        getattr(value, "__dataclass_fields__", None),
    )
    if isinstance(raw_fields, dict) and not isinstance(value, type):
        dataclass_fields = cast(dict[str, object], raw_fields)
        return {
            name: to_jsonable(cast(object, getattr(value, name)))
            for name in dataclass_fields
        }
    if isinstance(value, tuple):
        items = cast(tuple[object, ...], value)
        return [to_jsonable(item) for item in items]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError(
        "scene payloads may contain only dataclasses, tuples, and Python JSON "
        f"scalars; got {type(value).__name__}."
    )
