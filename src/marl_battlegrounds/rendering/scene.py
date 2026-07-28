"""Versioned, renderer-neutral battlefield and semantic-event contracts.

The records in this module are deliberately scalar-only.  They may be produced
from JAX-backed simulator artifacts, but they never retain JAX/NumPy arrays,
callbacks, renderer objects, or mutable presentation state.
"""

from dataclasses import dataclass, field
from math import isclose, isfinite
from typing import Literal, cast

from marl_battlegrounds.core.types import MAX_AGENT_SLOTS
from marl_battlegrounds.rendering.vocabulary import (
    ActivationTokenId,
    StatusLifecycleKind,
)

SCENE_SCHEMA_VERSION = 1
EVENT_SCHEMA_VERSION = 1

type Point2D = tuple[float, float]
type SceneAudience = Literal["researcher", "agent_pov"]
type ObstacleKind = Literal["pillar", "wall"]
type RangeKind = Literal["observation", "basic", "ultimate"]
type Lane = Literal[0, 1]
type TargetDisclosure = Literal["public", "target_none", "redacted", "invalid"]
type HealthOutcome = Literal["damage", "healing", "unchanged"]
type ChargePathKind = Literal["charge_only", "combined_charge_and_movement"]
type RejectionComponent = Literal["movement", "combat", "complete_tuple_domain"]


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


def _require_point(value: Point2D, *, name: str) -> None:
    if type(value) is not tuple or len(value) != 2:
        raise ValueError(f"{name} must be a two-coordinate Python tuple.")
    for coordinate in value:
        _require_finite(coordinate, name=f"{name} coordinate")


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
