"""Define immutable host records for battlefield scenes and displayed events.

Scene adapters build these renderer-neutral contracts from authorized data.
V1 supports legacy live/debugger views; V2 joins canonical evaluation frames,
phase anchors and events for researcher analysis. Source-evidence records
track only directly recorded status applications, without inventing sources.

All records are frozen, slotted and keyword-only. Constructors validate
Python scalar/tuple shapes, declared bounds and the joins described on each
class. They do not run JAX, recompute simulator rules or grant information
rights. Global slots are Team A 0..4 then Team B 5..9; world points are finite
Python (x, y) float tuples. Records do not retain array or renderer state.

Use to_jsonable to build JSON-compatible output; it does not write files or
redact privileged fields. Audience filtering belongs to the producing adapter.
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
    """Require value to be an exact Python int, with an optional lower bound.

    name labels ValueError messages. minimum=None allows any integer; otherwise
    the inclusive minimum is checked. Bool and NumPy/JAX scalars are rejected.
    Return None on success; this is a host check, not a conversion.
    """
    if type(value) is not int:
        raise ValueError(f"{name} must be a Python int; got {type(value).__name__}.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}; got {value}.")


def _require_python_bool(value: bool, *, name: str) -> None:
    """Require value to be an exact Python bool.

    Raise ValueError labelled by name for any other type; return None otherwise.
    Do not coerce integers or array scalars.
    """
    if type(value) is not bool:
        raise ValueError(f"{name} must be a Python bool; got {type(value).__name__}.")


def _require_lane(
    value: Lane | None,
    *,
    name: str,
    allow_none: bool,
) -> None:
    """Check a Basic/Ultimate lane supplied as a Python value.

    Accept integer 0 for Basic and 1 for Ultimate. None is accepted only when
    allow_none is True. Raise ValueError labelled by name for anything else;
    return None on success.
    """
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
    """Require value to be None or an exact instance of record_type.

    Subclasses are rejected. name labels the ValueError; valid inputs return
    None unchanged. This checks the record root, not its provenance.
    """
    if value is not None and type(value) is not record_type:
        raise ValueError(
            f"{name} must be {record_type.__name__} or None; "
            f"got {type(value).__name__}."
        )


def _require_slot(value: int, *, name: str) -> None:
    """Check value is a Python global slot in 0..9.

    Slots 0..4 belong to Team A and 5..9 to Team B. name labels ValueError for
    bad type/range. Return None; this does not check configured membership.
    """
    _require_python_int(value, name=name)
    if not 0 <= value < MAX_AGENT_SLOTS:
        raise ValueError(f"{name} must be in [0, {MAX_AGENT_SLOTS}); got {value}.")


def _require_finite(value: float, *, name: str) -> None:
    """Require value to be an exact finite Python float.

    Reject integers, array scalars, NaN and infinity with a name-labelled
    ValueError. Return None without coercing the value.
    """
    if type(value) is not float or not isfinite(value):
        raise ValueError(f"{name} must be a finite Python float; got {value!r}.")


def _require_nonnegative_finite(value: float, *, name: str) -> None:
    """Require value to be a finite Python float greater than or equal to zero.

    name labels ValueError from the type, finiteness or sign check. Return None
    on success; positive infinity and integer zero are not accepted.
    """
    _require_finite(value, name=name)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative; got {value!r}.")


def _require_point(value: Point2D, *, name: str) -> None:
    """Check a world point is an exact pair of finite Python floats.

    value must be a tuple of length two, ordered (x, y). name labels ValueError.
    Return None without checking map bounds, occupancy or collision clearance.
    """
    if type(value) is not tuple or len(value) != 2:
        raise ValueError(f"{name} must be a two-coordinate Python tuple.")
    for coordinate in value:
        _require_finite(coordinate, name=f"{name} coordinate")


def _points_close(left: Point2D, right: Point2D) -> bool:
    """Compare both coordinates of left and right using display tolerances.

    Inputs are two-coordinate world tuples. Return True when each coordinate
    is close with relative tolerance 1e-6 and absolute tolerance 1e-5. A length
    mismatch raises ValueError through strict zip; no geometry is projected.
    """
    return all(
        isclose(left_value, right_value, rel_tol=1e-6, abs_tol=1e-5)
        for left_value, right_value in zip(left, right, strict=True)
    )


def _require_text(value: str, *, name: str) -> None:
    """Require value to be a nonblank exact Python string.

    Whitespace around nonblank text is retained. name labels ValueError;
    return None without normalizing or escaping the string.
    """
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a non-empty Python string.")


def _require_tuple_items(
    value: object,
    *,
    name: str,
    item_types: tuple[type[object], ...],
) -> None:
    """Check value is a tuple containing only the exact item_types.

    An empty tuple is allowed. name labels ValueError for a wrong container or
    item type; subclasses are rejected. Return None without copying items.
    """
    if type(value) is not tuple:
        raise ValueError(f"{name} must be a Python tuple.")
    allowed_names = " or ".join(item_type.__name__ for item_type in item_types)
    for index, item in enumerate(cast(tuple[object, ...], value)):
        if not any(type(item) is item_type for item_type in item_types):
            raise ValueError(
                f"{name}[{index}] must be {allowed_names}; got {type(item).__name__}."
            )


def _require_unique(values: tuple[str, ...], *, name: str) -> None:
    """Reject repeated entries in values with a name-labelled ValueError.

    values is a tuple of hashable strings supplied by the caller. Preserve its
    order and contents; return None when all entries differ.
    """
    if len(values) != len(set(values)):
        raise ValueError(f"{name} values must be unique.")


def _is_canonical_event_before_frame(
    event_id: str,
    *,
    episode_id: str,
    frame_index: int,
) -> bool:
    """Check event_id names an earlier event in episode_id.

    Accept the canonical transition prefix, an unpadded nonnegative transition
    index, and exactly four decimal event digits. Its transition index must be
    less than frame_index. Return False on a format or ordering mismatch. This
    string check does not prove that an event exists in a recorded episode.
    """
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
    """Describe one static pillar or rotated wall.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    obstacle_id : str
        Unique nonblank ID within the containing map.
    kind : ObstacleKind
        The literal pillar or wall.
    center : Point2D
        World (x, y) center as a tuple of two finite Python floats.
    radius : float | None, default None
        Positive finite pillar radius in world units; None for walls.
    width : float | None, default None
        Positive finite wall width along its local x-axis; None for pillars.
    height : float | None, default None
        Positive finite wall height along its local y-axis; None for pillars.
    theta : float, default 0.0
        Finite wall rotation in radians; zero by default.

    Raises
    ------
    ValueError
        A required value has the wrong Python type or is not finite, kind is
        unknown, or kind-specific dimensions are absent, nonpositive or mixed.

    Notes
    -----
    Coordinates are not checked against map bounds here. Pillars retain theta
    but their circular shape does not depend on it.
    """

    obstacle_id: str
    """Unique nonblank ID within the containing map."""
    kind: ObstacleKind
    """The literal pillar or wall."""
    center: Point2D
    """World (x, y) center as a tuple of two finite Python floats."""
    radius: float | None = None
    """Positive finite pillar radius in world units; None for walls."""
    width: float | None = None
    """Positive finite wall width along its local x-axis; None for pillars."""
    height: float | None = None
    """Positive finite wall height along its local y-axis; None for pillars."""
    theta: float = 0.0
    """Finite wall rotation in radians; zero by default."""

    def __post_init__(self) -> None:
        """Validate ObstacleSceneV1 during host construction.

        Raise ValueError if a required value has the wrong Python type or is not
        finite, kind is unknown, or kind-specific dimensions are absent,
        nonpositive or mixed.
        Otherwise return None without changing values.
        """
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
    """Describe positive map bounds and ordered static obstacles.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    width : float
        Positive finite Python map width in world units, starting at x = 0.
    height : float
        Positive finite Python map height in world units, starting at y = 0.
    obstacles : tuple[ObstacleSceneV1, ...], default ()
        Tuple of exact ObstacleSceneV1 records; empty means no obstacles.

    Raises
    ------
    ValueError
        Bounds are invalid, obstacles is not an exact tuple of obstacle records,
        or obstacle IDs repeat.

    Notes
    -----
    This presentation record does not check obstacle containment or overlap.
    """

    width: float
    """Positive finite Python map width in world units, starting at x = 0."""
    height: float
    """Positive finite Python map height in world units, starting at y = 0."""
    obstacles: tuple[ObstacleSceneV1, ...] = ()
    """Tuple of exact ObstacleSceneV1 records; empty means no obstacles."""

    def __post_init__(self) -> None:
        """Validate MapSceneV1 during host construction.

        Raise ValueError if bounds are invalid, obstacles is not an exact tuple
        of obstacle records, or obstacle IDs repeat.
        Otherwise return None without changing values.
        """
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
    """Describe one active status in a legacy successor scene.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    token_id : str
        Nonblank presentation token ID; no catalog lookup is enforced here.
    duration : int
        Positive Python count of remaining decision steps.
    source_class_id : int
        Python source-class ID; this legacy record does not enforce its range.
    label : str
        Nonblank full display label.
    short_label : str
        Nonblank compact label.
    accessible_name : str
        Nonblank label for assistive presentation.
    priority : int
        Nonnegative Python display-order priority.

    Raises
    ------
    ValueError
        Text is blank or has the wrong type, duration is below one, or priority
        is negative.
    """

    token_id: str
    """Nonblank presentation token ID; no catalog lookup is enforced here."""
    duration: int
    """Positive Python count of remaining decision steps."""
    source_class_id: int
    """Python source-class ID; this legacy record does not enforce its range."""
    label: str
    """Nonblank full display label."""
    short_label: str
    """Nonblank compact label."""
    accessible_name: str
    """Nonblank label for assistive presentation."""
    priority: int
    """Nonnegative Python display-order priority."""

    def __post_init__(self) -> None:
        """Validate StatusSceneV1 during host construction.

        Raise ValueError if text is blank or has the wrong type, duration is
        below one, or priority is negative.
        Otherwise return None without changing values.
        """
        _require_text(self.token_id, name="token_id")
        _require_python_int(self.duration, name="duration", minimum=1)
        _require_python_int(self.source_class_id, name="source_class_id")
        _require_text(self.label, name="label")
        _require_text(self.short_label, name="short_label")
        _require_text(self.accessible_name, name="accessible_name")
        _require_python_int(self.priority, name="priority", minimum=0)


@dataclass(frozen=True, slots=True, kw_only=True)
class ModifierSceneV1:
    """Describe one exact recipient-local public multiplier.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    token_id : str
        Nonblank presentation token ID.
    multiplier : float
        Finite Python multiplier copied from public facts; this record imposes
        no sign bound.
    label : str
        Nonblank display label.
    accessible_name : str
        Nonblank label for assistive presentation.

    Raises
    ------
    ValueError
        Text is blank or has the wrong type, or multiplier is not a finite Python float.
    """

    token_id: str
    """Nonblank presentation token ID."""
    multiplier: float
    """Finite Python multiplier copied from public facts; this record imposes no
    sign bound.
    """
    label: str
    """Nonblank display label."""
    accessible_name: str
    """Nonblank label for assistive presentation."""

    def __post_init__(self) -> None:
        """Validate ModifierSceneV1 during host construction.

        Raise ValueError if text is blank or has the wrong type, or multiplier
        is not a finite Python float.
        Otherwise return None without changing values.
        """
        _require_text(self.token_id, name="token_id")
        _require_finite(self.multiplier, name="multiplier")
        _require_text(self.label, name="label")
        _require_text(self.accessible_name, name="accessible_name")


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentSceneV1:
    """Describe the durable facts shown for one legacy scene agent.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    global_slot : int
        Global simulator slot, a Python int in 0..9; Team A precedes Team B.
    team_id : int
        Python physical team ID; its range is not checked by this legacy record.
    class_id : int
        Python catalog class ID; its range is not checked by this legacy record.
    position : Point2D
        World (x, y) tuple of two finite Python floats.
    radius : float
        Positive finite Python body radius in world units.
    active : bool
        Python bool copied from configured membership.
    alive : bool
        Python bool copied from current life status.
    current_health : float
        Finite Python health in [0, max_health].
    max_health : float
        Positive finite Python health capacity.
    ultimate_cooldown : int
        Nonnegative Python steps before Ultimate is ready.
    effective_speed : float
        Finite Python current movement speed in world units per step; no sign
        check here.
    statuses : tuple[StatusSceneV1, ...], default ()
        Tuple of exact StatusSceneV1 records in nondecreasing priority order;
        token IDs are unique.
    modifiers : tuple[ModifierSceneV1, ...], default ()
        Tuple of exact ModifierSceneV1 records with unique token IDs.

    Raises
    ------
    ValueError
        Field types, slot, radius or health bounds are invalid, or
        status/modifier types, order or unique IDs fail.

    Notes
    -----
    Construction does not prove audience permission, active membership or
    consistency between alive and health. The producer owns those joins.
    """

    global_slot: int
    """Global simulator slot, a Python int in 0..9; Team A precedes Team B."""
    team_id: int
    """Python physical team ID; its range is not checked by this legacy record."""
    class_id: int
    """Python catalog class ID; its range is not checked by this legacy record."""
    position: Point2D
    """World (x, y) tuple of two finite Python floats."""
    radius: float
    """Positive finite Python body radius in world units."""
    active: bool
    """Python bool copied from configured membership."""
    alive: bool
    """Python bool copied from current life status."""
    current_health: float
    """Finite Python health in [0, max_health]."""
    max_health: float
    """Positive finite Python health capacity."""
    ultimate_cooldown: int
    """Nonnegative Python steps before Ultimate is ready."""
    effective_speed: float
    """Finite Python current movement speed in world units per step; no sign check
    here.
    """
    statuses: tuple[StatusSceneV1, ...] = ()
    """Tuple of exact StatusSceneV1 records in nondecreasing priority order; token
    IDs are unique.
    """
    modifiers: tuple[ModifierSceneV1, ...] = ()
    """Tuple of exact ModifierSceneV1 records with unique token IDs."""

    def __post_init__(self) -> None:
        """Validate AgentSceneV1 during host construction.

        Raise ValueError if field types, slot, radius or health bounds are
        invalid, or status/modifier types, order or unique IDs fail.
        Otherwise return None without changing values.
        """
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
    """Describe a supplied legacy aura circle.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    source_global_slot : int
        Source global slot, a Python int in 0..9.
    token_id : Literal['mage_amplification', 'warrior_mitigation']
        The literal mage_amplification or warrior_mitigation.
    center : Point2D
        World (x, y) center as a tuple of two finite Python floats.
    radius : float
        Positive finite Python aura radius in world units.

    Raises
    ------
    ValueError
        Slot, center, token or radius is invalid.

    Notes
    -----
    This shows a supplied field; it does not calculate beneficiaries or multipliers.
    """

    source_global_slot: int
    """Source global slot, a Python int in 0..9."""
    token_id: Literal["mage_amplification", "warrior_mitigation"]
    """The literal mage_amplification or warrior_mitigation."""
    center: Point2D
    """World (x, y) center as a tuple of two finite Python floats."""
    radius: float
    """Positive finite Python aura radius in world units."""

    def __post_init__(self) -> None:
        """Validate AuraFieldSceneV1 during host construction.

        Raise ValueError if slot, center, token or radius is invalid.
        Otherwise return None without changing values.
        """
        _require_slot(self.source_global_slot, name="source_global_slot")
        _require_point(self.center, name="center")
        _require_finite(self.radius, name="radius")
        if self.radius <= 0:
            raise ValueError("aura radius must be positive.")
        if self.token_id not in ("mage_amplification", "warrior_mitigation"):
            raise ValueError(f"unknown aura token: {self.token_id!r}.")


@dataclass(frozen=True, slots=True, kw_only=True)
class RangeSceneV1:
    """Describe a supplied observation or ability range circle.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    global_slot : int
        Global simulator slot, a Python int in 0..9; Team A precedes Team B.
    center : Point2D
        World (x, y) center as a tuple of two finite Python floats.
    radius : float
        Nonnegative finite Python radius in world units; zero is allowed.
    kind : RangeKind
        The literal observation, basic or ultimate.

    Raises
    ------
    ValueError
        Slot, center, radius or range kind is invalid.

    Notes
    -----
    A range circle alone does not establish line of sight or target legality.
    """

    global_slot: int
    """Global simulator slot, a Python int in 0..9; Team A precedes Team B."""
    center: Point2D
    """World (x, y) center as a tuple of two finite Python floats."""
    radius: float
    """Nonnegative finite Python radius in world units; zero is allowed."""
    kind: RangeKind
    """The literal observation, basic or ultimate."""

    def __post_init__(self) -> None:
        """Validate RangeSceneV1 during host construction.

        Raise ValueError if slot, center, radius or range kind is invalid.
        Otherwise return None without changing values.
        """
        _require_slot(self.global_slot, name="global_slot")
        _require_point(self.center, name="center")
        _require_finite(self.radius, name="radius")
        if self.radius < 0:
            raise ValueError("range radius must be non-negative.")
        if self.kind not in ("observation", "basic", "ultimate"):
            raise ValueError(f"unknown range kind: {self.kind!r}.")


@dataclass(frozen=True, slots=True, kw_only=True)
class SelectionSceneV1:
    """Keep the current controlled agent and optional selected target.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    controlled_global_slot : int
        Controlled or inspected global slot, a Python int in 0..9.
    selected_global_slot : int | None
        Selected global target slot in 0..9, or None for no selection.

    Raises
    ------
    ValueError
        Either supplied slot is not a Python int in 0..9.

    Notes
    -----
    This stores presentation choice only. The containing scene may check that
    both agents exist.
    """

    controlled_global_slot: int
    """Controlled or inspected global slot, a Python int in 0..9."""
    selected_global_slot: int | None
    """Selected global target slot in 0..9, or None for no selection."""

    def __post_init__(self) -> None:
        """Validate SelectionSceneV1 during host construction.

        Raise ValueError if either supplied slot is not a Python int in 0..9.
        Otherwise return None without changing values.
        """
        _require_slot(self.controlled_global_slot, name="controlled_global_slot")
        if self.selected_global_slot is not None:
            _require_slot(self.selected_global_slot, name="selected_global_slot")


@dataclass(frozen=True, slots=True, kw_only=True)
class SelectedLegalitySceneV1:
    """Keep exact combat-mask values for one selected target.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    controlled_global_slot : int
        Controlled or inspected global slot, a Python int in 0..9.
    target_global_slot : int
        Selected recipient global slot in 0..9.
    target_action : int
        Positive Python actor-relative target category; producers use 1..10, but
        this record checks only the lower bound.
    lane_0_available : bool
        Python bool copied from the selected target/Basic joint-mask entry.
    lane_1_available : bool
        Python bool copied from the selected target/Ultimate joint-mask entry.
    armed_lane : Lane | None
        Python int 0 for Basic, 1 for Ultimate, or None when no lane is armed.
    armed_pair_legal : bool
        Availability of the armed lane; must be False when armed_lane is None.

    Raises
    ------
    ValueError
        Slots, category, booleans or lane are invalid, or armed_pair_legal
        disagrees with the selected lane.

    Notes
    -----
    This record validates internal agreement. It does not compute a mask or
    check target-action/global-slot identity.
    """

    controlled_global_slot: int
    """Controlled or inspected global slot, a Python int in 0..9."""
    target_global_slot: int
    """Selected recipient global slot in 0..9."""
    target_action: int
    """Positive Python actor-relative target category; producers use 1..10, but
    this record checks only the lower bound.
    """
    lane_0_available: bool
    """Python bool copied from the selected target/Basic joint-mask entry."""
    lane_1_available: bool
    """Python bool copied from the selected target/Ultimate joint-mask entry."""
    armed_lane: Lane | None
    """Python int 0 for Basic, 1 for Ultimate, or None when no lane is armed."""
    armed_pair_legal: bool
    """Availability of the armed lane; must be False when armed_lane is None."""

    def __post_init__(self) -> None:
        """Validate SelectedLegalitySceneV1 during host construction.

        Raise ValueError if slots, category, booleans or lane are invalid, or
        armed_pair_legal disagrees with the selected lane.
        Otherwise return None without changing values.
        """
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
    """Describe a pending source-target request before action acceptance.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    source_global_slot : int
        Source global slot, a Python int in 0..9.
    target_global_slot : int
        Requested target global slot in 0..9.
    source_anchor : Point2D
        Supplied source world (x, y) tuple of finite Python floats.
    target_anchor : Point2D
        Supplied target world (x, y) tuple of finite Python floats.
    lane : Lane
        Python int 0 for Basic or 1 for Ultimate.
    legal : bool
        Python bool copied from the current exact pair mask.

    Raises
    ------
    ValueError
        Slots, anchors, lane or legal flag is invalid.

    Notes
    -----
    The route is pending intent, not evidence that an action was accepted or applied.
    """

    source_global_slot: int
    """Source global slot, a Python int in 0..9."""
    target_global_slot: int
    """Requested target global slot in 0..9."""
    source_anchor: Point2D
    """Supplied source world (x, y) tuple of finite Python floats."""
    target_anchor: Point2D
    """Supplied target world (x, y) tuple of finite Python floats."""
    lane: Lane
    """Python int 0 for Basic or 1 for Ultimate."""
    legal: bool
    """Python bool copied from the current exact pair mask."""

    def __post_init__(self) -> None:
        """Validate PendingRouteSceneV1 during host construction.

        Raise ValueError if slots, anchors, lane or legal flag is invalid.
        Otherwise return None without changing values.
        """
        _require_slot(self.source_global_slot, name="source_global_slot")
        _require_slot(self.target_global_slot, name="target_global_slot")
        _require_point(self.source_anchor, name="source_anchor")
        _require_point(self.target_anchor, name="target_anchor")
        _require_lane(self.lane, name="lane", allow_none=False)
        _require_python_bool(self.legal, name="legal")


@dataclass(frozen=True, slots=True, kw_only=True)
class ObserverVisibilitySceneV1:
    """Keep one supplied observer-to-candidate visibility fact.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    observer_global_slot : int
        Observer global slot in 0..9.
    candidate_global_slot : int
        Candidate global slot in 0..9.
    visible : bool
        Exact Python bool from that observer's base sensor visibility.

    Raises
    ------
    ValueError
        Either slot or the boolean has an invalid type or value.

    Notes
    -----
    A researcher may inspect this fact; the record does not grant it to another actor.
    """

    observer_global_slot: int
    """Observer global slot in 0..9."""
    candidate_global_slot: int
    """Candidate global slot in 0..9."""
    visible: bool
    """Exact Python bool from that observer's base sensor visibility."""

    def __post_init__(self) -> None:
        """Validate ObserverVisibilitySceneV1 during host construction.

        Raise ValueError if either slot or the boolean has an invalid type or value.
        Otherwise return None without changing values.
        """
        _require_slot(self.observer_global_slot, name="observer_global_slot")
        _require_slot(self.candidate_global_slot, name="candidate_global_slot")
        _require_python_bool(self.visible, name="visible")


@dataclass(frozen=True, slots=True, kw_only=True)
class BattlefieldSceneV1:
    """Collect a legacy battlefield view for one declared audience.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    schema_version : int
        Exact Python int 1.
    audience : SceneAudience
        researcher or agent_pov.
    audience_badge : str
        Nonblank visible authority label; researcher badges must contain PRIVILEGED.
    map : MapSceneV1
        Exact MapSceneV1 containing public static geometry.
    agents : tuple[AgentSceneV1, ...]
        Tuple of exact AgentSceneV1 records with unique increasing global slots.
    aura_fields : tuple[AuraFieldSceneV1, ...], default ()
        Tuple of exact AuraFieldSceneV1 records; default empty.
    ranges : tuple[RangeSceneV1, ...], default ()
        Tuple of exact RangeSceneV1 records; default empty.
    selection : SelectionSceneV1 | None, default None
        Optional exact SelectionSceneV1; None means no selection.
    selected_legality : SelectedLegalitySceneV1 | None, default None
        Optional exact SelectedLegalitySceneV1 for the selected pair.
    pending_route : PendingRouteSceneV1 | None, default None
        Optional exact PendingRouteSceneV1 for unsubmitted intent.
    observer_visibility : tuple[ObserverVisibilitySceneV1, ...], default ()
        Tuple of exact ObserverVisibilitySceneV1 records; must be empty for agent_pov.

    Raises
    ------
    ValueError
        Version, audience, badge, nested record types or agent ordering is
        invalid, or an actor view contains privileged observer visibility.

    Notes
    -----
    The producer must already remove unauthorized data. This record does not
    sanitize arbitrary researcher data into an actor view.
    """

    schema_version: int
    """Exact Python int 1."""
    audience: SceneAudience
    """researcher or agent_pov."""
    audience_badge: str
    """Nonblank visible authority label; researcher badges must contain PRIVILEGED."""
    map: MapSceneV1
    """Exact MapSceneV1 containing public static geometry."""
    agents: tuple[AgentSceneV1, ...]
    """Tuple of exact AgentSceneV1 records with unique increasing global slots."""
    aura_fields: tuple[AuraFieldSceneV1, ...] = ()
    """Tuple of exact AuraFieldSceneV1 records; default empty."""
    ranges: tuple[RangeSceneV1, ...] = ()
    """Tuple of exact RangeSceneV1 records; default empty."""
    selection: SelectionSceneV1 | None = None
    """Optional exact SelectionSceneV1; None means no selection."""
    selected_legality: SelectedLegalitySceneV1 | None = None
    """Optional exact SelectedLegalitySceneV1 for the selected pair."""
    pending_route: PendingRouteSceneV1 | None = None
    """Optional exact PendingRouteSceneV1 for unsubmitted intent."""
    observer_visibility: tuple[ObserverVisibilitySceneV1, ...] = ()
    """Tuple of exact ObserverVisibilitySceneV1 records; must be empty for agent_pov."""

    def __post_init__(self) -> None:
        """Validate BattlefieldSceneV1 during host construction.

        Raise ValueError if version, audience, badge, nested record types or
        agent ordering is invalid, or an actor view contains privileged observer
        visibility.
        Otherwise return None without changing values.
        """
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
    """Describe one catalog status mechanic on a researcher class card.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    status_channel : int
        Scientific status channel, a Python int in 0..8.
    status_id : str
        Catalog status ID matching status_channel.
    family : StatusFamilyV2
        Status family: slow, stun, anti_heal, damage_amplification or movement_floor.
    source_action_component : StatusActionComponentV2
        The status-producing action component, basic or ultimate.
    duration_steps : int
        Positive Python application duration in decision steps.
    magnitude_kind : StatusMagnitudeKindV2
        Strength interpretation: movement_multiplier, none, healing_multiplier,
        damage_multiplier or movement_floor.
    magnitude : float | None
        Finite Python strength float; None exactly when magnitude_kind is none.
    breaks_on_positive_damage : bool
        Python bool declaring whether positive raw damage breaks this status.

    Raises
    ------
    ValueError
        Status channel/ID identity, enum values, positive duration, magnitude
        presence/type or the break flag is invalid.

    Notes
    -----
    The containing scene checks which class owns each channel; this record does
    not apply the mechanic.
    """

    status_channel: int
    """Scientific status channel, a Python int in 0..8."""
    status_id: str
    """Catalog status ID matching status_channel."""
    family: StatusFamilyV2
    """Status family: slow, stun, anti_heal, damage_amplification or movement_floor."""
    source_action_component: StatusActionComponentV2
    """The status-producing action component, basic or ultimate."""
    duration_steps: int
    """Positive Python application duration in decision steps."""
    magnitude_kind: StatusMagnitudeKindV2
    """Strength interpretation: movement_multiplier, none, healing_multiplier,
    damage_multiplier or movement_floor.
    """
    magnitude: float | None
    """Finite Python strength float; None exactly when magnitude_kind is none."""
    breaks_on_positive_damage: bool
    """Python bool declaring whether positive raw damage breaks this status."""

    def __post_init__(self) -> None:
        """Validate ClassStatusMechanicSceneV2 during host construction.

        Raise ValueError if status channel/ID identity, enum values, positive
        duration, magnitude presence/type or the break flag is invalid.
        Otherwise return None without changing values.
        """
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
    """Describe one catalog passive aura on a researcher class card.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    aura_id : str
        Nonblank catalog aura ID; the containing scene checks catalog membership.
    radius : float
        Nonnegative finite Python aura radius in world units.
    per_emitter_multiplier : float
        Nonnegative finite Python multiplier contributed by one eligible emitter.
    stacking_rule : AuraStackingRuleV2
        The literal multiply_then_clamp; the record does not apply the rule.
    clamp_kind : AuraClampKindV2
        Ceiling or floor applied after multiplying eligible emitter strengths.
    clamp_value : float
        Nonnegative finite Python bound used by the declared clamp rule.

    Raises
    ------
    ValueError
        Aura text, finite nonnegative quantities, stacking rule or clamp kind is
        invalid.
    """

    aura_id: str
    """Nonblank catalog aura ID; the containing scene checks catalog membership."""
    radius: float
    """Nonnegative finite Python aura radius in world units."""
    per_emitter_multiplier: float
    """Nonnegative finite Python multiplier contributed by one eligible emitter."""
    stacking_rule: AuraStackingRuleV2
    """The literal multiply_then_clamp; the record does not apply the rule."""
    clamp_kind: AuraClampKindV2
    """Ceiling or floor applied after multiplying eligible emitter strengths."""
    clamp_value: float
    """Nonnegative finite Python bound used by the declared clamp rule."""

    def __post_init__(self) -> None:
        """Validate ClassAuraMechanicSceneV2 during host construction.

        Raise ValueError if aura text, finite nonnegative quantities, stacking
        rule or clamp kind is invalid.
        Otherwise return None without changing values.
        """
        _require_text(self.aura_id, name="aura_id")
        for name in ("radius", "per_emitter_multiplier", "clamp_value"):
            _require_nonnegative_finite(cast(float, getattr(self, name)), name=name)
        if self.stacking_rule != "multiply_then_clamp":
            raise ValueError("class aura stacking rule must be multiply_then_clamp.")
        if self.clamp_kind not in ("ceiling", "floor"):
            raise ValueError("class aura clamp kind must be ceiling or floor.")


@dataclass(frozen=True, slots=True, kw_only=True)
class ClassMechanicsSceneV2:
    """Describe the recorded mechanics of one real class.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    class_id : int
        Python catalog class ID in 1..5.
    class_name : str
        Canonical name matching class_id: Mage, Warrior, Hunter, Rogue or Priest.
    maximum_health : float
        Positive finite Python health capacity.
    body_radius : float
        Positive finite Python body radius in world units.
    base_movement_speed : float
        Nonnegative finite Python unmodified speed in world units per step.
    observation_radius : float
        Nonnegative finite Python sensor radius in world units.
    basic_target_mode : BasicTargetModeV2
        unavailable, ally or enemy.
    basic_interaction_radius : float
        Nonnegative finite Python Basic radius in world units.
    basic_raw_damage : float
        Nonnegative finite Python Basic health damage before modifiers.
    basic_raw_healing : float
        Nonnegative finite Python Basic health healing before modifiers.
    ultimate_target_mode : UltimateTargetModeV2
        unavailable, target_none, ally or enemy.
    ultimate_interaction_radius : float
        Nonnegative finite Python Ultimate radius in world units.
    ultimate_cooldown_steps : int
        Nonnegative Python full Ultimate cooldown in steps.
    ultimate_raw_damage : float
        Nonnegative finite Python Ultimate health damage before modifiers.
    ultimate_raw_healing : float
        Nonnegative finite Python Ultimate health healing before modifiers.
    out_of_combat_delay_steps : int
        Nonnegative Python delay before out-of-combat recovery.
    out_of_combat_health_regeneration_fraction_per_step : float
        Finite Python fraction of maximum health recovered per eligible step, in [0, 1].
    status_mechanics : tuple[ClassStatusMechanicSceneV2, ...]
        Tuple of exact status mechanics with unique increasing channels and unique IDs.
    aura_mechanics : tuple[ClassAuraMechanicSceneV2, ...]
        Tuple of exact aura mechanics with unique aura IDs.

    Raises
    ------
    ValueError
        Class identity, quantity bounds, target modes or nested mechanics
        types/order/uniqueness are invalid.

    Notes
    -----
    Values come from the recorded context. Validation checks structure and
    bounds, not equality with the current runtime catalog.
    """

    class_id: int
    """Python catalog class ID in 1..5."""
    class_name: str
    """Canonical name matching class_id: Mage, Warrior, Hunter, Rogue or Priest."""
    maximum_health: float
    """Positive finite Python health capacity."""
    body_radius: float
    """Positive finite Python body radius in world units."""
    base_movement_speed: float
    """Nonnegative finite Python unmodified speed in world units per step."""
    observation_radius: float
    """Nonnegative finite Python sensor radius in world units."""
    basic_target_mode: BasicTargetModeV2
    """unavailable, ally or enemy."""
    basic_interaction_radius: float
    """Nonnegative finite Python Basic radius in world units."""
    basic_raw_damage: float
    """Nonnegative finite Python Basic health damage before modifiers."""
    basic_raw_healing: float
    """Nonnegative finite Python Basic health healing before modifiers."""
    ultimate_target_mode: UltimateTargetModeV2
    """unavailable, target_none, ally or enemy."""
    ultimate_interaction_radius: float
    """Nonnegative finite Python Ultimate radius in world units."""
    ultimate_cooldown_steps: int
    """Nonnegative Python full Ultimate cooldown in steps."""
    ultimate_raw_damage: float
    """Nonnegative finite Python Ultimate health damage before modifiers."""
    ultimate_raw_healing: float
    """Nonnegative finite Python Ultimate health healing before modifiers."""
    out_of_combat_delay_steps: int
    """Nonnegative Python delay before out-of-combat recovery."""
    out_of_combat_health_regeneration_fraction_per_step: float
    """Finite Python fraction of maximum health recovered per eligible step, in [0,
    1].
    """
    status_mechanics: tuple[ClassStatusMechanicSceneV2, ...]
    """Tuple of exact status mechanics with unique increasing channels and unique
    IDs.
    """
    aura_mechanics: tuple[ClassAuraMechanicSceneV2, ...]
    """Tuple of exact aura mechanics with unique aura IDs."""

    def __post_init__(self) -> None:
        """Validate ClassMechanicsSceneV2 during host construction.

        Raise ValueError if class identity, quantity bounds, target modes or
        nested mechanics types/order/uniqueness are invalid.
        Otherwise return None without changing values.
        """
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
    """Identify one directly recorded application source for a status.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    source_global_slot : int
        Source global slot, a Python int in 0..9.
    source_public_agent_id : str
        Nonblank recorded public ID of the source agent.
    event_id : str
        Nonblank direct application event ID; the containing frame state checks
        its canonical epoch.

    Raises
    ------
    ValueError
        Source slot or nonblank public/event ID is invalid.

    Notes
    -----
    A source-class label alone is not direct source evidence.
    """

    source_global_slot: int
    """Source global slot, a Python int in 0..9."""
    source_public_agent_id: str
    """Nonblank recorded public ID of the source agent."""
    event_id: str
    """Nonblank direct application event ID; the containing frame state checks its
    canonical epoch.
    """

    def __post_init__(self) -> None:
        """Validate StatusSourceEvidenceSceneV2 during host construction.

        Raise ValueError if source slot or nonblank public/event ID is invalid.
        Otherwise return None without changing values.
        """
        _require_slot(self.source_global_slot, name="source_global_slot")
        _require_text(self.source_public_agent_id, name="source_public_agent_id")
        _require_text(self.event_id, name="event_id")


@dataclass(frozen=True, slots=True, kw_only=True)
class StatusSourceChannelEvidenceV2:
    """Keep known direct sources for one active recipient/status pair.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    recipient_global_slot : int
        Recipient global slot, a Python int in 0..9.
    recipient_public_agent_id : str
        Nonblank recorded public ID of the recipient agent.
    status_channel : int
        Scientific status channel, a Python int in 0..8.
    status_id : str
        Catalog status ID matching status_channel.
    direct_source_evidence : tuple[StatusSourceEvidenceSceneV2, ...], default ()
        Tuple of direct application-event evidence; empty means the source is unknown.

    Raises
    ------
    ValueError
        Recipient identity, status channel/ID or evidence type is invalid, or
        (source slot, event ID) keys are not unique and sorted.

    Notes
    -----
    An empty evidence tuple preserves an active status with unknown source. It
    does not mean the status is absent.
    """

    recipient_global_slot: int
    """Recipient global slot, a Python int in 0..9."""
    recipient_public_agent_id: str
    """Nonblank recorded public ID of the recipient agent."""
    status_channel: int
    """Scientific status channel, a Python int in 0..8."""
    status_id: str
    """Catalog status ID matching status_channel."""
    direct_source_evidence: tuple[StatusSourceEvidenceSceneV2, ...] = ()
    """Tuple of direct application-event evidence; empty means the source is unknown."""

    def __post_init__(self) -> None:
        """Validate StatusSourceChannelEvidenceV2 during host construction.

        Raise ValueError if recipient identity, status channel/ID or evidence
        type is invalid, or (source slot, event ID) keys are not unique and
        sorted.
        Otherwise return None without changing values.
        """
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
    """Bind immutable direct-source evidence to one recorded frame.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    schema_version : int
        Exact Python int 2.
    episode_id : str
        Nonblank recorded episode ID used to join frames and transitions.
    frame_index : int
        Nonnegative zero-based recorded frame index, separate from simulator step count.
    frame_id : str
        Canonical episode_id:frame:frame_index string.
    active_statuses : tuple[StatusSourceChannelEvidenceV2, ...]
        Tuple of active recipient/channel rows, with unique sorted (recipient
        slot, channel) keys.

    Raises
    ------
    ValueError
        Version/frame identity, row types/order, recipient ID agreement or
        canonical earlier-event references fail, or one event supports multiple
        rows.

    Notes
    -----
    Frame zero has no earlier application events. The state does not invent
    sources for authored initial statuses.
    """

    schema_version: int
    """Exact Python int 2."""
    episode_id: str
    """Nonblank recorded episode ID used to join frames and transitions."""
    frame_index: int
    """Nonnegative zero-based recorded frame index, separate from simulator step
    count.
    """
    frame_id: str
    """Canonical episode_id:frame:frame_index string."""
    active_statuses: tuple[StatusSourceChannelEvidenceV2, ...]
    """Tuple of active recipient/channel rows, with unique sorted (recipient slot,
    channel) keys.
    """

    def __post_init__(self) -> None:
        """Validate StatusSourceEvidenceStateV2 during host construction.

        Raise ValueError if version/frame identity, row types/order, recipient
        ID agreement or canonical earlier-event references fail, or one event
        supports multiple rows.
        Otherwise return None without changing values.
        """
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
        """Build a fresh lookup of direct evidence by recipient and status channel.

        Returns
        -------
        dict
            Keys are (global slot in 0..9, status channel in 0..8). Values are the
            stored immutable evidence tuples. An empty value means no known direct
            source; an absent key means no active row is stored for that pair.

        Notes
        -----
        Changing the returned dictionary does not change this frozen state.
        """
        return {
            (row.recipient_global_slot, row.status_channel): (
                row.direct_source_evidence
            )
            for row in self.active_statuses
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class StatusSourceEvidenceIndexV2:
    """Keep a gap-free sequence of immutable frame evidence states.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    schema_version : int
        Exact Python int 2.
    episode_id : str
        Nonblank recorded episode ID used to join frames and transitions.
    frame_states : tuple[StatusSourceEvidenceStateV2, ...]
        Nonempty exact tuple of StatusSourceEvidenceStateV2 for frames 0..N of
        this episode.

    Raises
    ------
    ValueError
        Version, episode text, state types or gap-free frame/episode joins are invalid.

    Notes
    -----
    Indexing returns stored states. Building this index is a separate host replay pass.
    """

    schema_version: int
    """Exact Python int 2."""
    episode_id: str
    """Nonblank recorded episode ID used to join frames and transitions."""
    frame_states: tuple[StatusSourceEvidenceStateV2, ...]
    """Nonempty exact tuple of StatusSourceEvidenceStateV2 for frames 0..N of this
    episode.
    """

    def __post_init__(self) -> None:
        """Validate StatusSourceEvidenceIndexV2 during host construction.

        Raise ValueError if version, episode text, state types or gap-free
        frame/episode joins are invalid.
        Otherwise return None without changing values.
        """
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
        """Return the stored evidence state for one frame index.

        Parameters
        ----------
        frame_index : int
            Exact Python int from zero through len(frame_states) - 1.

        Returns
        -------
        StatusSourceEvidenceStateV2
            The existing immutable state, retrieved by direct tuple indexing.

        Raises
        ------
        ValueError
            frame_index is not a Python int or is negative.
        IndexError
            frame_index is beyond the recorded frame range.
        """
        _require_python_int(frame_index, name="frame_index", minimum=0)
        if frame_index >= len(self.frame_states):
            raise IndexError("frame_index is outside the status-source index.")
        return self.frame_states[frame_index]


@dataclass(frozen=True, slots=True, kw_only=True)
class StatusSceneV2:
    """Describe one active recorded status and its known direct sources.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    status_channel : int
        Scientific status channel, a Python int in 0..8.
    status_id : str
        Catalog status ID matching status_channel.
    family : StatusFamilyV2
        Status family: slow, stun, anti_heal, damage_amplification or movement_floor.
    remaining_duration : int
        Positive Python count of remaining decision steps.
    source_class_id : int
        Python catalog source-class ID in 1..5.
    source_class_name : str
        Nonblank source-class display name; scene assembly checks its catalog join.
    source_action_component : Literal['basic', 'ultimate']
        The status-producing action component, basic or ultimate.
    magnitude_kind : StatusMagnitudeKindV2
        Strength interpretation: movement_multiplier, none, healing_multiplier,
        damage_multiplier or movement_floor.
    magnitude : float | None
        Finite Python strength float; None exactly when magnitude_kind is none.
    breaks_on_positive_damage : bool
        Python bool declaring whether positive raw damage breaks this status.
    direct_source_evidence : tuple[StatusSourceEvidenceSceneV2, ...], default ()
        Tuple of evidence rows sorted by source slot with unique event IDs;
        empty means unknown source.

    Raises
    ------
    ValueError
        Channel/catalog identity, duration, class range, enum values, magnitude
        presence, break flag or evidence ordering/type/uniqueness is invalid.

    Notes
    -----
    Source class describes the mechanic; direct event evidence identifies
    particular agents. The containing scene joins those agents and event epochs.
    """

    status_channel: int
    """Scientific status channel, a Python int in 0..8."""
    status_id: str
    """Catalog status ID matching status_channel."""
    family: StatusFamilyV2
    """Status family: slow, stun, anti_heal, damage_amplification or movement_floor."""
    remaining_duration: int
    """Positive Python count of remaining decision steps."""
    source_class_id: int
    """Python catalog source-class ID in 1..5."""
    source_class_name: str
    """Nonblank source-class display name; scene assembly checks its catalog join."""
    source_action_component: Literal["basic", "ultimate"]
    """The status-producing action component, basic or ultimate."""
    magnitude_kind: StatusMagnitudeKindV2
    """Strength interpretation: movement_multiplier, none, healing_multiplier,
    damage_multiplier or movement_floor.
    """
    magnitude: float | None
    """Finite Python strength float; None exactly when magnitude_kind is none."""
    breaks_on_positive_damage: bool
    """Python bool declaring whether positive raw damage breaks this status."""
    direct_source_evidence: tuple[StatusSourceEvidenceSceneV2, ...] = ()
    """Tuple of evidence rows sorted by source slot with unique event IDs; empty
    means unknown source.
    """

    def __post_init__(self) -> None:
        """Validate StatusSceneV2 during host construction.

        Raise ValueError if channel/catalog identity, duration, class range,
        enum values, magnitude presence, break flag or evidence
        ordering/type/uniqueness is invalid.
        Otherwise return None without changing values.
        """
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
    """Keep a selected frame's exact recipient aura multiplier.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    aura_id : Literal['mage_damage_amplification', 'warrior_damage_mitigation']
        mage_damage_amplification or warrior_damage_mitigation.
    multiplier : float
        Nonnegative finite Python multiplier; 1.0 is neutral.

    Raises
    ------
    ValueError
        Aura ID is unknown or multiplier is not a finite nonnegative Python float.
    """

    aura_id: Literal[
        "mage_damage_amplification",
        "warrior_damage_mitigation",
    ]
    """mage_damage_amplification or warrior_damage_mitigation."""
    multiplier: float
    """Nonnegative finite Python multiplier; 1.0 is neutral."""

    def __post_init__(self) -> None:
        """Validate AuraRecipientModifierSceneV2 during host construction.

        Raise ValueError if aura ID is unknown or multiplier is not a finite
        nonnegative Python float.
        Otherwise return None without changing values.
        """
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
    """Describe recorded aura capability at its emitter's position.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    aura_id : Literal['mage_damage_amplification', 'warrior_damage_mitigation']
        mage_damage_amplification or warrior_damage_mitigation.
    source_global_slot : int
        Source global slot, a Python int in 0..9.
    source_public_agent_id : str
        Nonblank recorded public ID of the source agent.
    source_class_id : int
        Python source-class ID in 1..5.
    source_class_name : str
        Nonblank source-class display name; scene assembly checks its catalog join.
    source_alive : bool
        Python bool from the selected frame; renderers can omit dead-source fields.
    center : Point2D
        World (x, y) center as a tuple of two finite Python floats.
    radius : float
        Positive finite Python aura radius in world units.
    beneficiary_relation : Literal['same_team']
        The literal same_team, including the emitter when eligible.
    per_emitter_multiplier : float
        Nonnegative finite Python multiplier contributed by one eligible emitter.
    stacking_rule : Literal['multiply_then_clamp']
        The literal multiply_then_clamp; the record does not apply the rule.
    clamp_kind : Literal['ceiling', 'floor']
        Ceiling or floor applied after multiplying eligible emitter strengths.
    clamp_value : float
        Nonnegative finite Python bound used by the declared clamp rule.

    Raises
    ------
    ValueError
        Aura identity, source fields, position, positive radius, nonnegative
        strengths or declared stacking/clamp values are invalid.

    Notes
    -----
    A field describes capability and position, not a list of current affected
    recipients. Source shield eligibility remains a simulator concern.
    """

    aura_id: Literal[
        "mage_damage_amplification",
        "warrior_damage_mitigation",
    ]
    """mage_damage_amplification or warrior_damage_mitigation."""
    source_global_slot: int
    """Source global slot, a Python int in 0..9."""
    source_public_agent_id: str
    """Nonblank recorded public ID of the source agent."""
    source_class_id: int
    """Python source-class ID in 1..5."""
    source_class_name: str
    """Nonblank source-class display name; scene assembly checks its catalog join."""
    source_alive: bool
    """Python bool from the selected frame; renderers can omit dead-source fields."""
    center: Point2D
    """World (x, y) center as a tuple of two finite Python floats."""
    radius: float
    """Positive finite Python aura radius in world units."""
    beneficiary_relation: Literal["same_team"]
    """The literal same_team, including the emitter when eligible."""
    per_emitter_multiplier: float
    """Nonnegative finite Python multiplier contributed by one eligible emitter."""
    stacking_rule: Literal["multiply_then_clamp"]
    """The literal multiply_then_clamp; the record does not apply the rule."""
    clamp_kind: Literal["ceiling", "floor"]
    """Ceiling or floor applied after multiplying eligible emitter strengths."""
    clamp_value: float
    """Nonnegative finite Python bound used by the declared clamp rule."""

    def __post_init__(self) -> None:
        """Validate AuraFieldSceneV2 during host construction.

        Raise ValueError if aura identity, source fields, position, positive
        radius, nonnegative strengths or declared stacking/clamp values are
        invalid.
        Otherwise return None without changing values.
        """
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
    """Describe one configured agent in a selected researcher frame.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    global_slot : int
        Global simulator slot, a Python int in 0..9; Team A precedes Team B.
    public_agent_id : str
        Nonblank recorded public agent ID; distinct from the numerical slot.
    team_id : int
        Python physical team ID 1 or 2.
    team_local_slot : int
        Python roster position in 0..4; the containing scene joins it to global_slot.
    class_id : int
        Python real catalog class ID in 1..5.
    position : Point2D
        World (x, y) tuple of two finite Python floats.
    radius : float
        Positive finite Python body radius in world units.
    life_state : AgentLifeStateV2
        alive or corpse, copied from selected-frame life status.
    current_health : float
        Finite Python health in [0, max_health].
    max_health : float
        Positive finite Python health capacity.
    effective_movement_speed : float
        Nonnegative finite Python current speed in world units per step.
    ultimate_cooldown_remaining : int
        Nonnegative Python steps until Ultimate is ready.
    spawn_shield_remaining : int
        Nonnegative Python remaining shield steps.
    steps_until_out_of_combat : int
        Nonnegative Python recovery countdown in steps.
    respawned_on_incoming_transition : bool
        Python bool saying this agent respawned in the transition entering the frame.
    respawn_event_id : str | None
        Nonblank incoming respawn event ID when the respawn flag is True;
        otherwise None.
    statuses : tuple[StatusSceneV2, ...], default ()
        Tuple of exact StatusSceneV2 rows with unique channels in canonical
        display order.
    aura_modifiers : tuple[AuraRecipientModifierSceneV2, ...], default ()
        Tuple of exact modifier rows with unique sorted aura IDs; the containing
        researcher scene requires both aura kinds.

    Raises
    ------
    ValueError
        Identity/ranges, life state, finite quantities, counter values, respawn
        evidence presence or nested row types/order/uniqueness are invalid.

    Notes
    -----
    This record checks value shape and bounds. Scene assembly owns roster and
    event joins; current health alone is not used here to infer life state.
    """

    global_slot: int
    """Global simulator slot, a Python int in 0..9; Team A precedes Team B."""
    public_agent_id: str
    """Nonblank recorded public agent ID; distinct from the numerical slot."""
    team_id: int
    """Python physical team ID 1 or 2."""
    team_local_slot: int
    """Python roster position in 0..4; the containing scene joins it to global_slot."""
    class_id: int
    """Python real catalog class ID in 1..5."""
    position: Point2D
    """World (x, y) tuple of two finite Python floats."""
    radius: float
    """Positive finite Python body radius in world units."""
    life_state: AgentLifeStateV2
    """alive or corpse, copied from selected-frame life status."""
    current_health: float
    """Finite Python health in [0, max_health]."""
    max_health: float
    """Positive finite Python health capacity."""
    effective_movement_speed: float
    """Nonnegative finite Python current speed in world units per step."""
    ultimate_cooldown_remaining: int
    """Nonnegative Python steps until Ultimate is ready."""
    spawn_shield_remaining: int
    """Nonnegative Python remaining shield steps."""
    steps_until_out_of_combat: int
    """Nonnegative Python recovery countdown in steps."""
    respawned_on_incoming_transition: bool
    """Python bool saying this agent respawned in the transition entering the frame."""
    respawn_event_id: str | None
    """Nonblank incoming respawn event ID when the respawn flag is True; otherwise
    None.
    """
    statuses: tuple[StatusSceneV2, ...] = ()
    """Tuple of exact StatusSceneV2 rows with unique channels in canonical display
    order.
    """
    aura_modifiers: tuple[AuraRecipientModifierSceneV2, ...] = ()
    """Tuple of exact modifier rows with unique sorted aura IDs; the containing
    researcher scene requires both aura kinds.
    """

    def __post_init__(self) -> None:
        """Validate AgentSceneV2 during host construction.

        Raise ValueError if identity/ranges, life state, finite quantities,
        counter values, respawn evidence presence or nested row
        types/order/uniqueness are invalid.
        Otherwise return None without changing values.
        """
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
    """Assign one recorded team spawn pad to one configured agent.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    team_id : int
        Python physical team ID 1 or 2.
    team_local_slot : int
        Python roster position in 0..4.
    assigned_global_slot : int
        Assigned global slot in 0..9.
    assigned_public_agent_id : str
        Nonblank public ID assigned to this pad.
    position : Point2D
        World (x, y) tuple of two finite Python floats.

    Raises
    ------
    ValueError
        Team/slot values, public ID or world position is invalid.

    Notes
    -----
    The containing scene checks the pad's roster join. This record does not
    choose a new position.
    """

    team_id: int
    """Python physical team ID 1 or 2."""
    team_local_slot: int
    """Python roster position in 0..4."""
    assigned_global_slot: int
    """Assigned global slot in 0..9."""
    assigned_public_agent_id: str
    """Nonblank public ID assigned to this pad."""
    position: Point2D
    """World (x, y) tuple of two finite Python floats."""

    def __post_init__(self) -> None:
        """Validate SpawnPadSceneV2 during host construction.

        Raise ValueError if team/slot values, public ID or world position is invalid.
        Otherwise return None without changing values.
        """
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
    """Describe one team's current respawn clock.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    team_index : int
        Python team-axis index 0 for Team A or 1 for Team B.
    team_id : int
        Python physical team ID equal to team_index + 1.
    period_steps : int
        Positive Python configured wave period in steps.
    countdown_steps : int
        Python selected-frame countdown in [0, period_steps).

    Raises
    ------
    ValueError
        Team IDs disagree, the period is not positive, or the countdown is
        outside its range.
    """

    team_index: int
    """Python team-axis index 0 for Team A or 1 for Team B."""
    team_id: int
    """Python physical team ID equal to team_index + 1."""
    period_steps: int
    """Positive Python configured wave period in steps."""
    countdown_steps: int
    """Python selected-frame countdown in [0, period_steps)."""

    def __post_init__(self) -> None:
        """Validate RespawnWaveSceneV2 during host construction.

        Raise ValueError if team IDs disagree, the period is not positive, or
        the countdown is outside its range.
        Otherwise return None without changing values.
        """
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
    """Collect one canonical researcher frame and its presentation choices.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    schema_version : int
        Exact Python int 2.
    audience : Literal['researcher']
        The literal researcher; V2 is not an actor-POV record.
    audience_badge : str
        Visible label containing PRIVILEGED.
    episode_id : str
        Nonblank recorded episode ID used to join frames and transitions.
    frame_index : int
        Nonnegative zero-based recorded frame index, separate from simulator step count.
    frame_id : str
        Canonical episode_id:frame:frame_index string.
    simulator_step_count : int
        Nonnegative Python simulator step count; authored starts may differ from
        frame_index.
    incoming_transition_id : str | None
        None at frame zero; otherwise episode_id:transition:(frame_index - 1).
    incoming_event_ids : tuple[str, ...]
        Exact tuple of canonical incoming event IDs in gap-free ordinal order.
    map : MapSceneV1
        Exact MapSceneV1 of recorded static geometry.
    agents : tuple[AgentSceneV2, ...]
        Exact tuple of configured AgentSceneV2 rows with unique increasing slots
        and public IDs.
    aura_fields : tuple[AuraFieldSceneV2, ...]
        Exact tuple of AuraFieldSceneV2 with unique sorted (source slot, aura
        ID) keys joined to scene agents.
    class_mechanics : tuple[ClassMechanicsSceneV2, ...]
        Exact tuple of class cards ordered by unique class IDs, covering all
        nine status channels and both catalog auras.
    spawn_pads : tuple[SpawnPadSceneV2, ...]
        Exact tuple with one pad per agent in sorted team/local-slot order.
    respawn_waves : tuple[RespawnWaveSceneV2, ...]
        Exact tuple of wave rows ordered by team indices (0, 1).
    ranges : tuple[RangeSceneV1, ...], default ()
        Exact tuple of supplied range circles joined to scene agents; default empty.
    selection : SelectionSceneV1 | None, default None
        Optional selection of configured scene agents; None means no observer selection.
    next_decision_selected_legality : SelectedLegalitySceneV1 | None, default None
        Optional exact pair-mask display joined to selection; describes the next
        decision at this frame.
    observer_visibility : tuple[ObserverVisibilitySceneV1, ...], default ()
        Exact tuple covering the ordered roster for the controlled observer;
        empty when selection is None.

    Raises
    ------
    ValueError
        Version/audience, canonical IDs, roster topology, nested types/order,
        class/status/aura partitions, source evidence, pads, clocks or selection
        joins fail.

    Notes
    -----
    The adapter owns authorization and static-context truth. Constructor checks
    enforce joins but do not replay simulator rules or infer hidden sources.
    Incoming events describe the completed transition; selected legality
    describes the next action.
    """

    schema_version: int
    """Exact Python int 2."""
    audience: Literal["researcher"]
    """The literal researcher; V2 is not an actor-POV record."""
    audience_badge: str
    """Visible label containing PRIVILEGED."""
    episode_id: str
    """Nonblank recorded episode ID used to join frames and transitions."""
    frame_index: int
    """Nonnegative zero-based recorded frame index, separate from simulator step
    count.
    """
    frame_id: str
    """Canonical episode_id:frame:frame_index string."""
    simulator_step_count: int
    """Nonnegative Python simulator step count; authored starts may differ from
    frame_index.
    """
    incoming_transition_id: str | None
    """None at frame zero; otherwise episode_id:transition:(frame_index - 1)."""
    incoming_event_ids: tuple[str, ...]
    """Exact tuple of canonical incoming event IDs in gap-free ordinal order."""
    map: MapSceneV1
    """Exact MapSceneV1 of recorded static geometry."""
    agents: tuple[AgentSceneV2, ...]
    """Exact tuple of configured AgentSceneV2 rows with unique increasing slots and
    public IDs.
    """
    aura_fields: tuple[AuraFieldSceneV2, ...]
    """Exact tuple of AuraFieldSceneV2 with unique sorted (source slot, aura ID)
    keys joined to scene agents.
    """
    class_mechanics: tuple[ClassMechanicsSceneV2, ...]
    """Exact tuple of class cards ordered by unique class IDs, covering all nine
    status channels and both catalog auras.
    """
    spawn_pads: tuple[SpawnPadSceneV2, ...]
    """Exact tuple with one pad per agent in sorted team/local-slot order."""
    respawn_waves: tuple[RespawnWaveSceneV2, ...]
    """Exact tuple of wave rows ordered by team indices (0, 1)."""
    ranges: tuple[RangeSceneV1, ...] = ()
    """Exact tuple of supplied range circles joined to scene agents; default empty."""
    selection: SelectionSceneV1 | None = None
    """Optional selection of configured scene agents; None means no observer
    selection.
    """
    next_decision_selected_legality: SelectedLegalitySceneV1 | None = None
    """Optional exact pair-mask display joined to selection; describes the next
    decision at this frame.
    """
    observer_visibility: tuple[ObserverVisibilitySceneV1, ...] = ()
    """Exact tuple covering the ordered roster for the controlled observer; empty
    when selection is None.
    """

    def __post_init__(self) -> None:
        """Validate BattlefieldSceneV2 during host construction.

        Raise ValueError if version/audience, canonical IDs, roster topology,
        nested types/order, class/status/aura partitions, source evidence, pads,
        clocks or selection joins fail.
        Otherwise return None without changing values.
        """
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
    """Describe an accepted legacy activation without a health amount.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    event_type : Literal['accepted_activation']
        Fixed serialized event kind, supplied by the class and excluded from
        construction.
    event_id : str
        Nonblank event ID, unique within its containing legacy batch.
    transition_id : int
        Nonnegative Python legacy transition ID.
    token_id : ActivationTokenId
        Activation token; Mage Burst alone uses target_none disclosure.
    source_global_slot : int
        Source global slot, a Python int in 0..9.
    target_global_slot : int | None
        Disclosed target global slot in 0..9, or None when absent/redacted.
    source_anchor : Point2D | None
        Finite world (x, y) tuple when the source position is disclosed; otherwise None.
    target_anchor : Point2D | None
        Disclosed finite world (x, y) tuple, or None when absent/redacted.
    target_disclosure : TargetDisclosure
        public requires target ID and anchor; target_none and redacted omit
        both. invalid is forbidden for accepted actions.
    lane : Lane
        Python int 0 for Basic or 1 for Ultimate.
    source_class_id : int
        Python source-class ID; no range check is applied here.

    Raises
    ------
    ValueError
        Header/slots/anchors/lane are invalid, target disclosure fields
        disagree, or Mage Burst target-none rules fail.

    Notes
    -----
    Acceptance does not imply a particular damage/healing amount or visible
    target position.
    """

    event_type: Literal["accepted_activation"] = field(
        default="accepted_activation",
        init=False,
    )
    """Fixed serialized event kind, supplied by the class and excluded from
    construction.
    """
    event_id: str
    """Nonblank event ID, unique within its containing legacy batch."""
    transition_id: int
    """Nonnegative Python legacy transition ID."""
    token_id: ActivationTokenId
    """Activation token; Mage Burst alone uses target_none disclosure."""
    source_global_slot: int
    """Source global slot, a Python int in 0..9."""
    target_global_slot: int | None
    """Disclosed target global slot in 0..9, or None when absent/redacted."""
    source_anchor: Point2D | None
    """Finite world (x, y) tuple when the source position is disclosed; otherwise
    None.
    """
    target_anchor: Point2D | None
    """Disclosed finite world (x, y) tuple, or None when absent/redacted."""
    target_disclosure: TargetDisclosure
    """public requires target ID and anchor; target_none and redacted omit both.
    invalid is forbidden for accepted actions.
    """
    lane: Lane
    """Python int 0 for Basic or 1 for Ultimate."""
    source_class_id: int
    """Python source-class ID; no range check is applied here."""

    def __post_init__(self) -> None:
        """Validate AcceptedActivationEventV1 during host construction.

        Raise ValueError if header/slots/anchors/lane are invalid, target
        disclosure fields disagree, or Mage Burst target-none rules fail.
        Otherwise return None without changing values.
        """
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
    """Describe a recipient's exact net before/after health change.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    event_type : Literal['net_health']
        Fixed serialized event kind, supplied by the class and excluded from
        construction.
    event_id : str
        Nonblank event ID, unique within its containing legacy batch.
    transition_id : int
        Nonnegative Python legacy transition ID.
    recipient_global_slot : int
        Recipient global slot, a Python int in 0..9.
    recipient_anchor : Point2D | None
        Finite world (x, y) tuple when the recipient position is disclosed;
        otherwise None.
    health_before : float
        Finite Python health before the recorded transition.
    health_after : float
        Finite Python health after the recorded transition.
    net_delta : float
        Finite Python health_after minus health_before, checked with absolute
        tolerance 1e-6.
    outcome : HealthOutcome
        damage for a negative delta, healing for positive, or unchanged for zero.

    Raises
    ------
    ValueError
        Header/recipient/anchor is invalid, health values are nonfinite, delta
        disagrees, or outcome has the wrong sign.

    Notes
    -----
    Net change is not a source-attributed damage amount. This legacy record does
    not separately enforce nonnegative health bounds.
    """

    event_type: Literal["net_health"] = field(default="net_health", init=False)
    """Fixed serialized event kind, supplied by the class and excluded from
    construction.
    """
    event_id: str
    """Nonblank event ID, unique within its containing legacy batch."""
    transition_id: int
    """Nonnegative Python legacy transition ID."""
    recipient_global_slot: int
    """Recipient global slot, a Python int in 0..9."""
    recipient_anchor: Point2D | None
    """Finite world (x, y) tuple when the recipient position is disclosed;
    otherwise None.
    """
    health_before: float
    """Finite Python health before the recorded transition."""
    health_after: float
    """Finite Python health after the recorded transition."""
    net_delta: float
    """Finite Python health_after minus health_before, checked with absolute
    tolerance 1e-6.
    """
    outcome: HealthOutcome
    """damage for a negative delta, healing for positive, or unchanged for zero."""

    def __post_init__(self) -> None:
        """Validate NetHealthEventV1 during host construction.

        Raise ValueError if header/recipient/anchor is invalid, health values
        are nonfinite, delta disagrees, or outcome has the wrong sign.
        Otherwise return None without changing values.
        """
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
    """Describe recorded Charge endpoints without a continuous path.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    event_type : Literal['charge_displacement']
        Fixed serialized event kind, supplied by the class and excluded from
        construction.
    event_id : str
        Nonblank event ID, unique within its containing legacy batch.
    transition_id : int
        Nonnegative Python legacy transition ID.
    source_global_slot : int
        Source global slot, a Python int in 0..9.
    target_global_slot : int
        Target global slot, a Python int in 0..9.
    start : Point2D
        Finite world (x, y) position before the displayed relocation.
    end : Point2D
        Finite world (x, y) position after the displayed relocation.
    path_kind : ChargePathKind
        charge_only or combined_charge_and_movement; names which recorded phases
        the endpoints span.

    Raises
    ------
    ValueError
        Header, slots, endpoint tuples or path kind is invalid.

    Notes
    -----
    The line between endpoints is a display cue, not a tested travel segment.
    """

    event_type: Literal["charge_displacement"] = field(
        default="charge_displacement",
        init=False,
    )
    """Fixed serialized event kind, supplied by the class and excluded from
    construction.
    """
    event_id: str
    """Nonblank event ID, unique within its containing legacy batch."""
    transition_id: int
    """Nonnegative Python legacy transition ID."""
    source_global_slot: int
    """Source global slot, a Python int in 0..9."""
    target_global_slot: int
    """Target global slot, a Python int in 0..9."""
    start: Point2D
    """Finite world (x, y) position before the displayed relocation."""
    end: Point2D
    """Finite world (x, y) position after the displayed relocation."""
    path_kind: ChargePathKind
    """charge_only or combined_charge_and_movement; names which recorded phases the
    endpoints span.
    """

    def __post_init__(self) -> None:
        """Validate ChargeDisplacementEventV1 during host construction.

        Raise ValueError if header, slots, endpoint tuples or path kind is invalid.
        Otherwise return None without changing values.
        """
        _validate_event_header(self.event_id, self.transition_id)
        _require_slot(self.source_global_slot, name="source_global_slot")
        _require_slot(self.target_global_slot, name="target_global_slot")
        _require_point(self.start, name="start")
        _require_point(self.end, name="end")
        if self.path_kind not in ("charge_only", "combined_charge_and_movement"):
            raise ValueError(f"unknown Charge path kind: {self.path_kind!r}.")


@dataclass(frozen=True, slots=True, kw_only=True)
class StatusLifecycleEventV1:
    """Describe a conservative legacy status change and linked activations.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    event_type : Literal['status_lifecycle']
        Fixed serialized event kind, supplied by the class and excluded from
        construction.
    event_id : str
        Nonblank event ID, unique within its containing legacy batch.
    transition_id : int
        Nonnegative Python legacy transition ID.
    recipient_global_slot : int
        Recipient global slot, a Python int in 0..9.
    recipient_anchor : Point2D | None
        Finite world (x, y) tuple when the recipient position is disclosed;
        otherwise None.
    token_id : str
        Nonblank status presentation token ID.
    change : StatusLifecycleKind
        Lifecycle presentation kind supplied by the producer; this constructor
        does not validate enum membership.
    duration_before : int
        Nonnegative Python duration in steps before the transition.
    duration_after : int
        Nonnegative Python duration in steps after the transition.
    source_class_id : int
        Python source-class ID; no range check is applied here.
    application_event_ids : tuple[str, ...], default ()
        Unique tuple of nonblank activation IDs; the containing batch checks each link.

    Raises
    ------
    ValueError
        Header, recipient, anchor, text/counter types or unique application IDs
        are invalid.

    Notes
    -----
    This record does not infer exact application causes from duration changes.
    """

    event_type: Literal["status_lifecycle"] = field(
        default="status_lifecycle",
        init=False,
    )
    """Fixed serialized event kind, supplied by the class and excluded from
    construction.
    """
    event_id: str
    """Nonblank event ID, unique within its containing legacy batch."""
    transition_id: int
    """Nonnegative Python legacy transition ID."""
    recipient_global_slot: int
    """Recipient global slot, a Python int in 0..9."""
    recipient_anchor: Point2D | None
    """Finite world (x, y) tuple when the recipient position is disclosed;
    otherwise None.
    """
    token_id: str
    """Nonblank status presentation token ID."""
    change: StatusLifecycleKind
    """Lifecycle presentation kind supplied by the producer; this constructor does
    not validate enum membership.
    """
    duration_before: int
    """Nonnegative Python duration in steps before the transition."""
    duration_after: int
    """Nonnegative Python duration in steps after the transition."""
    source_class_id: int
    """Python source-class ID; no range check is applied here."""
    application_event_ids: tuple[str, ...] = ()
    """Unique tuple of nonblank activation IDs; the containing batch checks each
    link.
    """

    def __post_init__(self) -> None:
        """Validate StatusLifecycleEventV1 during host construction.

        Raise ValueError if header, recipient, anchor, text/counter types or
        unique application IDs are invalid.
        Otherwise return None without changing values.
        """
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
    """Describe a legacy rejection fact without guessing a causal reason.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    event_type : Literal['rejected_action']
        Fixed serialized event kind, supplied by the class and excluded from
        construction.
    event_id : str
        Nonblank event ID, unique within its containing legacy batch.
    transition_id : int
        Nonnegative Python legacy transition ID.
    actor_global_slot : int
        Rejected actor global slot in 0..9.
    component : RejectionComponent
        movement, combat or complete_tuple_domain.
    actor_anchor : Point2D | None
        Disclosed finite world (x, y) tuple for the actor, or None.
    target_global_slot : int | None
        Disclosed target global slot in 0..9, or None when absent/redacted.
    target_anchor : Point2D | None
        Disclosed finite world (x, y) tuple, or None when absent/redacted.
    target_disclosure : TargetDisclosure
        public requires ID/anchor; target_none, redacted and invalid omit both.
    lane : Lane | None
        Python int 0/1 when meaningful, or None.
    movement_mask_value : bool
        Python bool copied from the submitted movement mask entry.
    pair_mask_value : bool
        Python bool copied from the submitted target/Ultimate pair entry.

    Raises
    ------
    ValueError
        Header, actor/target fields, component, lane, disclosure pairing or mask
        booleans are invalid.
    """

    event_type: Literal["rejected_action"] = field(
        default="rejected_action",
        init=False,
    )
    """Fixed serialized event kind, supplied by the class and excluded from
    construction.
    """
    event_id: str
    """Nonblank event ID, unique within its containing legacy batch."""
    transition_id: int
    """Nonnegative Python legacy transition ID."""
    actor_global_slot: int
    """Rejected actor global slot in 0..9."""
    component: RejectionComponent
    """movement, combat or complete_tuple_domain."""
    actor_anchor: Point2D | None
    """Disclosed finite world (x, y) tuple for the actor, or None."""
    target_global_slot: int | None
    """Disclosed target global slot in 0..9, or None when absent/redacted."""
    target_anchor: Point2D | None
    """Disclosed finite world (x, y) tuple, or None when absent/redacted."""
    target_disclosure: TargetDisclosure
    """public requires ID/anchor; target_none, redacted and invalid omit both."""
    lane: Lane | None
    """Python int 0/1 when meaningful, or None."""
    movement_mask_value: bool
    """Python bool copied from the submitted movement mask entry."""
    pair_mask_value: bool
    """Python bool copied from the submitted target/Ultimate pair entry."""

    def __post_init__(self) -> None:
        """Validate RejectedActionEventV1 during host construction.

        Raise ValueError if header, actor/target fields, component, lane,
        disclosure pairing or mask booleans are invalid.
        Otherwise return None without changing values.
        """
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
    """Collect one legacy transition's ordered presentation events.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    schema_version : int
        Exact Python int 1.
    transition_id : int
        Nonnegative Python legacy transition ID.
    simulator_step : int
        Nonnegative Python simulator step recorded for this batch.
    events : tuple[VisualEventV1, ...]
        Tuple of supported exact V1 event records with unique IDs and this
        transition ID.

    Raises
    ------
    ValueError
        Version/header/types/IDs fail, an event belongs to another transition,
        or a status application link does not name an accepted activation in
        this batch.

    Notes
    -----
    Input tuple order is retained. Legacy batches do not require canonical
    string IDs or gap-free event ordinals.
    """

    schema_version: int
    """Exact Python int 1."""
    transition_id: int
    """Nonnegative Python legacy transition ID."""
    simulator_step: int
    """Nonnegative Python simulator step recorded for this batch."""
    events: tuple[VisualEventV1, ...]
    """Tuple of supported exact V1 event records with unique IDs and this
    transition ID.
    """

    def __post_init__(self) -> None:
        """Validate VisualEventBatchV1 during host construction.

        Raise ValueError if version/header/types/IDs fail, an event belongs to
        another transition, or a status application link does not name an
        accepted activation in this batch.
        Otherwise return None without changing values.
        """
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
    """Locate one agent at an explicit recorded transition phase.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    phase : VisualAnchorPhaseV2
        Anchor epoch: transition_start, post_charge or successor.
    global_slot : int
        Global simulator slot, a Python int in 0..9; Team A precedes Team B.
    public_agent_id : str
        Nonblank recorded public agent ID; distinct from the numerical slot.
    position : Point2D
        World (x, y) tuple of two finite Python floats.

    Raises
    ------
    ValueError
        Phase, global slot, public ID or finite world position is invalid.

    Notes
    -----
    An anchor is a displayed fact. It does not authorize data access or imply a
    continuous path.
    """

    phase: VisualAnchorPhaseV2
    """Anchor epoch: transition_start, post_charge or successor."""
    global_slot: int
    """Global simulator slot, a Python int in 0..9; Team A precedes Team B."""
    public_agent_id: str
    """Nonblank recorded public agent ID; distinct from the numerical slot."""
    position: Point2D
    """World (x, y) tuple of two finite Python floats."""

    def __post_init__(self) -> None:
        """Validate VisualAgentAnchorV2 during host construction.

        Raise ValueError if phase, global slot, public ID or finite world
        position is invalid.
        Otherwise return None without changing values.
        """
        if self.phase not in ("transition_start", "post_charge", "successor"):
            raise ValueError(f"unknown visual anchor phase: {self.phase!r}.")
        _require_slot(self.global_slot, name="global_slot")
        _require_text(self.public_agent_id, name="public_agent_id")
        _require_point(self.position, name="position")


@dataclass(frozen=True, slots=True, kw_only=True)
class VisualTeamAnchorV2:
    """Identify one non-spatial team cue at a recorded transition phase.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    phase : VisualAnchorPhaseV2
        Anchor epoch: transition_start, post_charge or successor.
    team_index : int
        Python index 0 for Team A or 1 for Team B.
    team_id : int
        Python physical team ID equal to team_index + 1.

    Raises
    ------
    ValueError
        Phase or team identity is invalid.
    """

    phase: VisualAnchorPhaseV2
    """Anchor epoch: transition_start, post_charge or successor."""
    team_index: int
    """Python index 0 for Team A or 1 for Team B."""
    team_id: int
    """Python physical team ID equal to team_index + 1."""

    def __post_init__(self) -> None:
        """Validate VisualTeamAnchorV2 during host construction.

        Raise ValueError if phase or team identity is invalid.
        Otherwise return None without changing values.
        """
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
    """Check anchor has the exact record type, phase and global_slot.

    name labels ValueError on mismatch. Return None without checking the anchor
    against a frame or validating its public ID against another record.
    """
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
    """Require anchor and global_slot to be absent or present together.

    When present, use phase and global_slot to check the exact anchor. name
    labels ValueError; return None otherwise. None does not mean slot zero.
    """
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
    """Require value to be a Python int within signed int32 bounds.

    Accept -2**31 through 2**31 - 1, including submitted invalid action values.
    name labels ValueError for a bad type/range; return None on success.
    """
    _require_python_int(value, name=name)
    if not -(2**31) <= value < 2**31:
        raise ValueError(f"{name} must fit signed int32.")


def _require_slot_tuple(value: tuple[int, ...], *, name: str) -> None:
    """Check value contains unique global slots in increasing order.

    Require an exact tuple of Python ints in 0..9; empty is allowed. name labels
    ValueError. Return None without sorting or changing the supplied tuple.
    """
    if type(value) is not tuple:
        raise ValueError(f"{name} must be a Python tuple.")
    for slot in value:
        _require_slot(slot, name=f"{name} item")
    if value != tuple(sorted(set(value))):
        raise ValueError(f"{name} must be sorted and unique.")


@dataclass(frozen=True, slots=True, kw_only=True)
class CanonicalVisualEventBaseV2:
    """Give an event its canonical transition ID and ordinal identity.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    event_id : str
        Canonical string formed from transition_id and a zero-padded event ordinal.
    transition_id : str
        Nonblank recorded transition string; the containing batch checks its
        episode/index format.
    ordinal : int
        Nonnegative Python event ordinal; the containing batch enforces gap-free order.

    Raises
    ------
    ValueError
        Transition text or ordinal is invalid, or event_id does not equal
        transition_id:event:ordinal with at least four digits.

    Notes
    -----
    The containing batch enforces gap-free ordinals. A single event cannot
    establish sequence completeness.
    """

    event_id: str
    """Canonical string formed from transition_id and a zero-padded event ordinal."""
    transition_id: str
    """Nonblank recorded transition string; the containing batch checks its
    episode/index format.
    """
    ordinal: int
    """Nonnegative Python event ordinal; the containing batch enforces gap-free
    order.
    """

    def __post_init__(self) -> None:
        """Validate CanonicalVisualEventBaseV2 during host construction.

        Raise ValueError if transition text or ordinal is invalid, or event_id
        does not equal transition_id:event:ordinal with at least four digits.
        Otherwise return None without changing values.
        """
        _require_text(self.transition_id, name="transition_id")
        _require_python_int(self.ordinal, name="ordinal", minimum=0)
        if self.event_id != f"{self.transition_id}:event:{self.ordinal:04d}":
            raise ValueError("visual event ID must remain canonical and gap-free.")


@dataclass(frozen=True, slots=True, kw_only=True)
class ActionRejectedEventV2(CanonicalVisualEventBaseV2):
    """Describe one recorded rejected action component.

    Inherited fields follow CanonicalVisualEventBaseV2. This is a frozen,
    slotted, keyword-only host record.

    Attributes
    ----------
    event_type : Literal['action_rejected']
        Fixed serialized event kind, supplied by the class and excluded from
        construction.
    phase_rank : Literal[10]
        Fixed event ordering rank, supplied by the class and excluded from construction.
    actor_global_slot : int
        Rejected global slot in 0..9, including unused slots.
    actor_public_agent_id : str
        Nonblank recorded public ID for that slot.
    actor_configured_active : bool
        Python bool distinguishing a configured actor from an unused slot.
    rejection_component : Literal['domain', 'movement', 'combat_pair']
        domain, movement or combat_pair.
    submitted_move_action : int
        Original submitted Python int within signed int32 bounds; it may be
        outside legal movement categories.
    submitted_select_target_action : int
        Original submitted signed-int32 Python target category; it may be
        outside the legal domain.
    submitted_use_ultimate_action : int
        Original submitted signed-int32 Python Ultimate category; it may be
        outside the legal domain.
    actor_anchor : VisualAgentAnchorV2 | None
        Matching transition_start anchor when configured active; None for unused
        actors, which remain feed-only.

    Raises
    ------
    ValueError
        Canonical header, slot/ID/flag, rejection kind, submitted int32 values
        or active/anchor agreement is invalid.
    """

    event_type: Literal["action_rejected"] = field(
        default="action_rejected", init=False
    )
    """Fixed serialized event kind, supplied by the class and excluded from
    construction.
    """
    phase_rank: Literal[10] = field(default=10, init=False)
    """Fixed event ordering rank, supplied by the class and excluded from
    construction.
    """
    actor_global_slot: int
    """Rejected global slot in 0..9, including unused slots."""
    actor_public_agent_id: str
    """Nonblank recorded public ID for that slot."""
    actor_configured_active: bool
    """Python bool distinguishing a configured actor from an unused slot."""
    rejection_component: Literal["domain", "movement", "combat_pair"]
    """domain, movement or combat_pair."""
    submitted_move_action: int
    """Original submitted Python int within signed int32 bounds; it may be outside
    legal movement categories.
    """
    submitted_select_target_action: int
    """Original submitted signed-int32 Python target category; it may be outside
    the legal domain.
    """
    submitted_use_ultimate_action: int
    """Original submitted signed-int32 Python Ultimate category; it may be outside
    the legal domain.
    """
    actor_anchor: VisualAgentAnchorV2 | None
    """Matching transition_start anchor when configured active; None for unused
    actors, which remain feed-only.
    """

    def __post_init__(self) -> None:
        """Validate ActionRejectedEventV2 during host construction.

        Raise ValueError if canonical header, slot/ID/flag, rejection kind,
        submitted int32 values or active/anchor agreement is invalid.
        Otherwise return None without changing values.
        """
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
    """Describe one canonical accepted Basic or Ultimate activation.

    Inherited fields follow CanonicalVisualEventBaseV2. This is a frozen,
    slotted, keyword-only host record.

    Attributes
    ----------
    event_type : Literal['ability_activated']
        Fixed serialized event kind, supplied by the class and excluded from
        construction.
    phase_rank : Literal[20]
        Fixed event ordering rank, supplied by the class and excluded from construction.
    source_global_slot : int
        Source global slot, a Python int in 0..9.
    ability_component : Literal['basic', 'ultimate']
        basic or ultimate.
    recipient_global_slot : int | None
        Recipient global slot in 0..9, or None for a recipient-free activation.
    source_anchor : VisualAgentAnchorV2
        Matching source anchor at transition_start.
    recipient_anchor : VisualAgentAnchorV2 | None
        Matching recipient anchor at transition_start, or None exactly when
        recipient_global_slot is None.

    Raises
    ------
    ValueError
        Canonical header, source/recipient slots, ability kind or phase-anchor
        joins are invalid.

    Notes
    -----
    This records activation only; separate output and resolution events carry amounts.
    """

    event_type: Literal["ability_activated"] = field(
        default="ability_activated", init=False
    )
    """Fixed serialized event kind, supplied by the class and excluded from
    construction.
    """
    phase_rank: Literal[20] = field(default=20, init=False)
    """Fixed event ordering rank, supplied by the class and excluded from
    construction.
    """
    source_global_slot: int
    """Source global slot, a Python int in 0..9."""
    ability_component: Literal["basic", "ultimate"]
    """basic or ultimate."""
    recipient_global_slot: int | None
    """Recipient global slot in 0..9, or None for a recipient-free activation."""
    source_anchor: VisualAgentAnchorV2
    """Matching source anchor at transition_start."""
    recipient_anchor: VisualAgentAnchorV2 | None
    """Matching recipient anchor at transition_start, or None exactly when
    recipient_global_slot is None.
    """

    def __post_init__(self) -> None:
        """Validate AbilityActivatedEventV2 during host construction.

        Raise ValueError if canonical header, source/recipient slots, ability
        kind or phase-anchor joins are invalid.
        Otherwise return None without changing values.
        """
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
    """Describe one source's recorded damage output and aura context.

    Inherited fields follow CanonicalVisualEventBaseV2. This is a frozen,
    slotted, keyword-only host record.

    Attributes
    ----------
    event_type : Literal['source_damage_output']
        Fixed serialized event kind, supplied by the class and excluded from
        construction.
    phase_rank : Literal[30]
        Fixed event ordering rank, supplied by the class and excluded from construction.
    source_global_slot : int
        Source global slot, a Python int in 0..9.
    recipient_global_slot : int | None
        Recipient slot in 0..9, or None when the recorded output has no recipient.
    raw_damage_output : float
        Nonnegative finite Python raw damage in health units before modifiers.
    source_modified_damage_output : float
        Nonnegative finite Python damage after source modifiers, before the
        recipient multiplier.
    recipient_damage_modifier : float
        Nonnegative finite Python recipient multiplier copied from the event.
    mage_damage_aura_covering_emitter_global_slots : tuple[int, ...]
        Sorted unique tuple of Mage aura emitter slots covering the source.
    warrior_mitigation_aura_covering_emitter_global_slots : tuple[int, ...]
        Sorted unique tuple of Warrior aura emitter slots covering the recipient.
    source_anchor : VisualAgentAnchorV2
        Matching source anchor at transition_start.
    recipient_anchor : VisualAgentAnchorV2 | None
        Matching transition_start recipient anchor, or None exactly when
        recipient_global_slot is None.

    Raises
    ------
    ValueError
        Canonical header, slots, finite nonnegative amounts, ordered emitter
        slots or anchor joins are invalid.

    Notes
    -----
    Gross source output is not realized health loss or kill credit. This record
    does not recalculate damage.
    """

    event_type: Literal["source_damage_output"] = field(
        default="source_damage_output", init=False
    )
    """Fixed serialized event kind, supplied by the class and excluded from
    construction.
    """
    phase_rank: Literal[30] = field(default=30, init=False)
    """Fixed event ordering rank, supplied by the class and excluded from
    construction.
    """
    source_global_slot: int
    """Source global slot, a Python int in 0..9."""
    recipient_global_slot: int | None
    """Recipient slot in 0..9, or None when the recorded output has no recipient."""
    raw_damage_output: float
    """Nonnegative finite Python raw damage in health units before modifiers."""
    source_modified_damage_output: float
    """Nonnegative finite Python damage after source modifiers, before the
    recipient multiplier.
    """
    recipient_damage_modifier: float
    """Nonnegative finite Python recipient multiplier copied from the event."""
    mage_damage_aura_covering_emitter_global_slots: tuple[int, ...]
    """Sorted unique tuple of Mage aura emitter slots covering the source."""
    warrior_mitigation_aura_covering_emitter_global_slots: tuple[int, ...]
    """Sorted unique tuple of Warrior aura emitter slots covering the recipient."""
    source_anchor: VisualAgentAnchorV2
    """Matching source anchor at transition_start."""
    recipient_anchor: VisualAgentAnchorV2 | None
    """Matching transition_start recipient anchor, or None exactly when
    recipient_global_slot is None.
    """

    def __post_init__(self) -> None:
        """Validate SourceDamageOutputEventV2 during host construction.

        Raise ValueError if canonical header, slots, finite nonnegative amounts,
        ordered emitter slots or anchor joins are invalid.
        Otherwise return None without changing values.
        """
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
    """Describe one source's recorded healing output before health clipping.

    Inherited fields follow CanonicalVisualEventBaseV2. This is a frozen,
    slotted, keyword-only host record.

    Attributes
    ----------
    event_type : Literal['source_healing_output']
        Fixed serialized event kind, supplied by the class and excluded from
        construction.
    phase_rank : Literal[30]
        Fixed event ordering rank, supplied by the class and excluded from construction.
    source_global_slot : int
        Source global slot, a Python int in 0..9.
    recipient_global_slot : int | None
        Recipient slot in 0..9, or None when the recorded output has no recipient.
    raw_healing_output : float
        Nonnegative finite Python raw healing in health units.
    source_modified_healing_output : float
        Nonnegative finite Python healing after source modifiers and before
        recipient effects.
    recipient_healing_modifier : float
        Nonnegative finite Python recipient healing multiplier.
    source_anchor : VisualAgentAnchorV2
        Matching source anchor at transition_start.
    recipient_anchor : VisualAgentAnchorV2 | None
        Matching transition_start recipient anchor, or None exactly when
        recipient_global_slot is None.

    Raises
    ------
    ValueError
        Canonical header, slots, finite nonnegative values or anchor joins are invalid.

    Notes
    -----
    Output may exceed health actually restored. This record does not recompute
    healing or clipping.
    """

    event_type: Literal["source_healing_output"] = field(
        default="source_healing_output", init=False
    )
    """Fixed serialized event kind, supplied by the class and excluded from
    construction.
    """
    phase_rank: Literal[30] = field(default=30, init=False)
    """Fixed event ordering rank, supplied by the class and excluded from
    construction.
    """
    source_global_slot: int
    """Source global slot, a Python int in 0..9."""
    recipient_global_slot: int | None
    """Recipient slot in 0..9, or None when the recorded output has no recipient."""
    raw_healing_output: float
    """Nonnegative finite Python raw healing in health units."""
    source_modified_healing_output: float
    """Nonnegative finite Python healing after source modifiers and before
    recipient effects.
    """
    recipient_healing_modifier: float
    """Nonnegative finite Python recipient healing multiplier."""
    source_anchor: VisualAgentAnchorV2
    """Matching source anchor at transition_start."""
    recipient_anchor: VisualAgentAnchorV2 | None
    """Matching transition_start recipient anchor, or None exactly when
    recipient_global_slot is None.
    """

    def __post_init__(self) -> None:
        """Validate SourceHealingOutputEventV2 during host construction.

        Raise ValueError if canonical header, slots, finite nonnegative values
        or anchor joins are invalid.
        Otherwise return None without changing values.
        """
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
    """Describe one recipient's simultaneous combat health resolution.

    Inherited fields follow CanonicalVisualEventBaseV2. This is a frozen,
    slotted, keyword-only host record.

    Attributes
    ----------
    event_type : Literal['recipient_health_resolution']
        Fixed serialized event kind, supplied by the class and excluded from
        construction.
    phase_rank : Literal[40]
        Fixed event ordering rank, supplied by the class and excluded from construction.
    recipient_global_slot : int
        Recipient global slot, a Python int in 0..9.
    transition_start_health : float
        Nonnegative finite Python health before combat resolution.
    total_effective_damage : float
        Nonnegative finite Python summed effective damage in health units.
    total_effective_healing : float
        Nonnegative finite Python summed effective healing in health units.
    health_after_combat_resolution : float
        Nonnegative finite Python health after simultaneous combat and clipping,
        before regeneration.
    realized_net_health_change : float
        Finite signed Python net change in health units.
    recipient_anchor : VisualAgentAnchorV2
        Matching recipient anchor at transition_start.

    Raises
    ------
    ValueError
        Canonical header, recipient, finite values or anchor join is invalid.

    Notes
    -----
    The constructor checks bounds, not the arithmetic relationship between these
    recorded amounts.
    """

    event_type: Literal["recipient_health_resolution"] = field(
        default="recipient_health_resolution", init=False
    )
    """Fixed serialized event kind, supplied by the class and excluded from
    construction.
    """
    phase_rank: Literal[40] = field(default=40, init=False)
    """Fixed event ordering rank, supplied by the class and excluded from
    construction.
    """
    recipient_global_slot: int
    """Recipient global slot, a Python int in 0..9."""
    transition_start_health: float
    """Nonnegative finite Python health before combat resolution."""
    total_effective_damage: float
    """Nonnegative finite Python summed effective damage in health units."""
    total_effective_healing: float
    """Nonnegative finite Python summed effective healing in health units."""
    health_after_combat_resolution: float
    """Nonnegative finite Python health after simultaneous combat and clipping,
    before regeneration.
    """
    realized_net_health_change: float
    """Finite signed Python net change in health units."""
    recipient_anchor: VisualAgentAnchorV2
    """Matching recipient anchor at transition_start."""

    def __post_init__(self) -> None:
        """Validate RecipientHealthResolutionEventV2 during host construction.

        Raise ValueError if canonical header, recipient, finite values or anchor
        join is invalid.
        Otherwise return None without changing values.
        """
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
    """Record that combat participation restarted the recovery countdown.

    Inherited fields follow CanonicalVisualEventBaseV2. This is a frozen,
    slotted, keyword-only host record.

    Attributes
    ----------
    event_type : Literal['combat_countdown_reset']
        Fixed serialized event kind, supplied by the class and excluded from
        construction.
    phase_rank : Literal[50]
        Fixed event ordering rank, supplied by the class and excluded from construction.
    agent_global_slot : int
        Affected agent global slot, a Python int in 0..9.
    agent_anchor : VisualAgentAnchorV2
        Matching affected-agent anchor at transition_start.

    Raises
    ------
    ValueError
        Canonical header, affected global slot or required phase-anchor join is invalid.

    Notes
    -----
    The event producer owns the occurrence condition; constructing this record
    does not verify a simulator transition.
    """

    event_type: Literal["combat_countdown_reset"] = field(
        default="combat_countdown_reset", init=False
    )
    """Fixed serialized event kind, supplied by the class and excluded from
    construction.
    """
    phase_rank: Literal[50] = field(default=50, init=False)
    """Fixed event ordering rank, supplied by the class and excluded from
    construction.
    """
    agent_global_slot: int
    """Affected agent global slot, a Python int in 0..9."""
    agent_anchor: VisualAgentAnchorV2
    """Matching affected-agent anchor at transition_start."""

    def __post_init__(self) -> None:
        """Validate CombatCountdownResetEventV2 during host construction.

        Raise ValueError if canonical header, affected global slot or required
        phase-anchor join is invalid.
        Otherwise return None without changing values.
        """
        CanonicalVisualEventBaseV2.__post_init__(self)
        _require_slot(self.agent_global_slot, name="agent_global_slot")
        _require_agent_anchor(
            self.agent_anchor,
            name="agent_anchor",
            phase="transition_start",
            global_slot=self.agent_global_slot,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentLeftCombatEventV2(CanonicalVisualEventBaseV2):
    """Record that an agent reached out-of-combat status.

    Inherited fields follow CanonicalVisualEventBaseV2. This is a frozen,
    slotted, keyword-only host record.

    Attributes
    ----------
    event_type : Literal['agent_left_combat']
        Fixed serialized event kind, supplied by the class and excluded from
        construction.
    phase_rank : Literal[50]
        Fixed event ordering rank, supplied by the class and excluded from construction.
    agent_global_slot : int
        Affected agent global slot, a Python int in 0..9.
    agent_anchor : VisualAgentAnchorV2
        Matching affected-agent anchor at successor.

    Raises
    ------
    ValueError
        Canonical header, affected global slot or required phase-anchor join is invalid.

    Notes
    -----
    The event producer owns the occurrence condition; constructing this record
    does not verify a simulator transition.
    """

    event_type: Literal["agent_left_combat"] = field(
        default="agent_left_combat", init=False
    )
    """Fixed serialized event kind, supplied by the class and excluded from
    construction.
    """
    phase_rank: Literal[50] = field(default=50, init=False)
    """Fixed event ordering rank, supplied by the class and excluded from
    construction.
    """
    agent_global_slot: int
    """Affected agent global slot, a Python int in 0..9."""
    agent_anchor: VisualAgentAnchorV2
    """Matching affected-agent anchor at successor."""

    def __post_init__(self) -> None:
        """Validate AgentLeftCombatEventV2 during host construction.

        Raise ValueError if canonical header, affected global slot or required
        phase-anchor join is invalid.
        Otherwise return None without changing values.
        """
        CanonicalVisualEventBaseV2.__post_init__(self)
        _require_slot(self.agent_global_slot, name="agent_global_slot")
        _require_agent_anchor(
            self.agent_anchor,
            name="agent_anchor",
            phase="successor",
            global_slot=self.agent_global_slot,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class HealthRegeneratedEventV2(CanonicalVisualEventBaseV2):
    """Record the health actually restored by out-of-combat recovery.

    Inherited fields follow CanonicalVisualEventBaseV2. This is a frozen,
    slotted, keyword-only host record.

    Attributes
    ----------
    event_type : Literal['health_regenerated']
        Fixed serialized event kind, supplied by the class and excluded from
        construction.
    phase_rank : Literal[50]
        Fixed event ordering rank, supplied by the class and excluded from construction.
    agent_global_slot : int
        Affected agent global slot, a Python int in 0..9.
    actual_health_regenerated : float
        Nonnegative finite Python realized health gain after clipping.
    agent_anchor : VisualAgentAnchorV2
        Matching agent anchor at transition_start.

    Raises
    ------
    ValueError
        Canonical header, agent slot, nonnegative finite amount or anchor join
        is invalid.
    """

    event_type: Literal["health_regenerated"] = field(
        default="health_regenerated", init=False
    )
    """Fixed serialized event kind, supplied by the class and excluded from
    construction.
    """
    phase_rank: Literal[50] = field(default=50, init=False)
    """Fixed event ordering rank, supplied by the class and excluded from
    construction.
    """
    agent_global_slot: int
    """Affected agent global slot, a Python int in 0..9."""
    actual_health_regenerated: float
    """Nonnegative finite Python realized health gain after clipping."""
    agent_anchor: VisualAgentAnchorV2
    """Matching agent anchor at transition_start."""

    def __post_init__(self) -> None:
        """Validate HealthRegeneratedEventV2 during host construction.

        Raise ValueError if canonical header, agent slot, nonnegative finite
        amount or anchor join is invalid.
        Otherwise return None without changing values.
        """
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
    """Record that accepted Ultimate use started its cooldown.

    Inherited fields follow CanonicalVisualEventBaseV2. This is a frozen,
    slotted, keyword-only host record.

    Attributes
    ----------
    event_type : Literal['cooldown_started']
        Fixed serialized event kind, supplied by the class and excluded from
        construction.
    phase_rank : Literal[60]
        Fixed event ordering rank, supplied by the class and excluded from construction.
    agent_global_slot : int
        Affected agent global slot, a Python int in 0..9.
    agent_anchor : VisualAgentAnchorV2
        Matching affected-agent anchor at transition_start.

    Raises
    ------
    ValueError
        Canonical header, affected global slot or required phase-anchor join is invalid.

    Notes
    -----
    The event producer owns the occurrence condition; constructing this record
    does not verify a simulator transition.
    """

    event_type: Literal["cooldown_started"] = field(
        default="cooldown_started", init=False
    )
    """Fixed serialized event kind, supplied by the class and excluded from
    construction.
    """
    phase_rank: Literal[60] = field(default=60, init=False)
    """Fixed event ordering rank, supplied by the class and excluded from
    construction.
    """
    agent_global_slot: int
    """Affected agent global slot, a Python int in 0..9."""
    agent_anchor: VisualAgentAnchorV2
    """Matching affected-agent anchor at transition_start."""

    def __post_init__(self) -> None:
        """Validate CooldownStartedEventV2 during host construction.

        Raise ValueError if canonical header, affected global slot or required
        phase-anchor join is invalid.
        Otherwise return None without changing values.
        """
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
    """Record that the Ultimate cooldown became ready.

    Inherited fields follow CanonicalVisualEventBaseV2. This is a frozen,
    slotted, keyword-only host record.

    Attributes
    ----------
    event_type : Literal['cooldown_ready']
        Fixed serialized event kind, supplied by the class and excluded from
        construction.
    phase_rank : Literal[60]
        Fixed event ordering rank, supplied by the class and excluded from construction.
    agent_global_slot : int
        Affected agent global slot, a Python int in 0..9.
    agent_anchor : VisualAgentAnchorV2
        Matching affected-agent anchor at transition_start.

    Raises
    ------
    ValueError
        Canonical header, affected global slot or required phase-anchor join is invalid.

    Notes
    -----
    The event producer owns the occurrence condition; constructing this record
    does not verify a simulator transition.
    """

    event_type: Literal["cooldown_ready"] = field(default="cooldown_ready", init=False)
    """Fixed serialized event kind, supplied by the class and excluded from
    construction.
    """
    phase_rank: Literal[60] = field(default=60, init=False)
    """Fixed event ordering rank, supplied by the class and excluded from
    construction.
    """
    agent_global_slot: int
    """Affected agent global slot, a Python int in 0..9."""
    agent_anchor: VisualAgentAnchorV2
    """Matching affected-agent anchor at transition_start."""

    def __post_init__(self) -> None:
        """Validate CooldownReadyEventV2 during host construction.

        Raise ValueError if canonical header, affected global slot or required
        phase-anchor join is invalid.
        Otherwise return None without changing values.
        """
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
    """Describe realized displacement during the Charge phase.

    Inherited fields follow CanonicalVisualEventBaseV2. This is a frozen,
    slotted, keyword-only host record.

    Attributes
    ----------
    event_type : Literal['charge_phase_displacement']
        Fixed serialized event kind, supplied by the class and excluded from
        construction.
    phase_rank : Literal[70]
        Fixed event ordering rank, supplied by the class and excluded from construction.
    agent_global_slot : int
        Affected agent global slot, a Python int in 0..9.
    realized_displacement : Point2D
        Finite Python (delta x, delta y) tuple in world units.
    start_anchor : VisualAgentAnchorV2
        Matching agent anchor at transition_start.
    end_anchor : VisualAgentAnchorV2
        Matching agent anchor at post_charge; position must match start plus
        displacement within display tolerances.

    Raises
    ------
    ValueError
        Canonical header, slot, displacement or phase joins are invalid, or
        endpoint coordinates disagree at relative tolerance 1e-6 and absolute
        tolerance 1e-5.

    Notes
    -----
    These endpoints describe the named phase. They do not prove that every point
    on a drawn line was traversed.
    """

    event_type: Literal["charge_phase_displacement"] = field(
        default="charge_phase_displacement", init=False
    )
    """Fixed serialized event kind, supplied by the class and excluded from
    construction.
    """
    phase_rank: Literal[70] = field(default=70, init=False)
    """Fixed event ordering rank, supplied by the class and excluded from
    construction.
    """
    agent_global_slot: int
    """Affected agent global slot, a Python int in 0..9."""
    realized_displacement: Point2D
    """Finite Python (delta x, delta y) tuple in world units."""
    start_anchor: VisualAgentAnchorV2
    """Matching agent anchor at transition_start."""
    end_anchor: VisualAgentAnchorV2
    """Matching agent anchor at post_charge; position must match start plus
    displacement within display tolerances.
    """

    def __post_init__(self) -> None:
        """Validate ChargePhaseDisplacementEventV2 during host construction.

        Raise ValueError if canonical header, slot, displacement or phase joins
        are invalid, or endpoint coordinates disagree at relative tolerance 1e-6
        and absolute tolerance 1e-5.
        Otherwise return None without changing values.
        """
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
    """Describe realized displacement during ordinary movement.

    Inherited fields follow CanonicalVisualEventBaseV2. This is a frozen,
    slotted, keyword-only host record.

    Attributes
    ----------
    event_type : Literal['ordinary_movement_phase_displacement']
        Fixed serialized event kind, supplied by the class and excluded from
        construction.
    phase_rank : Literal[80]
        Fixed event ordering rank, supplied by the class and excluded from construction.
    agent_global_slot : int
        Affected agent global slot, a Python int in 0..9.
    realized_displacement : Point2D
        Finite Python (delta x, delta y) tuple in world units.
    start_anchor : VisualAgentAnchorV2
        Matching agent anchor at post_charge.
    end_anchor : VisualAgentAnchorV2
        Matching agent anchor at successor; position must match start plus
        displacement within display tolerances.

    Raises
    ------
    ValueError
        Canonical header, slot, displacement or phase joins are invalid, or
        endpoint coordinates disagree at relative tolerance 1e-6 and absolute
        tolerance 1e-5.

    Notes
    -----
    These endpoints describe the named phase. They do not prove that every point
    on a drawn line was traversed.
    """

    event_type: Literal["ordinary_movement_phase_displacement"] = field(
        default="ordinary_movement_phase_displacement", init=False
    )
    """Fixed serialized event kind, supplied by the class and excluded from
    construction.
    """
    phase_rank: Literal[80] = field(default=80, init=False)
    """Fixed event ordering rank, supplied by the class and excluded from
    construction.
    """
    agent_global_slot: int
    """Affected agent global slot, a Python int in 0..9."""
    realized_displacement: Point2D
    """Finite Python (delta x, delta y) tuple in world units."""
    start_anchor: VisualAgentAnchorV2
    """Matching agent anchor at post_charge."""
    end_anchor: VisualAgentAnchorV2
    """Matching agent anchor at successor; position must match start plus
    displacement within display tolerances.
    """

    def __post_init__(self) -> None:
        """Validate OrdinaryMovementPhaseDisplacementEventV2 during host
        construction.

        Raise ValueError if canonical header, slot, displacement or phase joins
        are invalid, or endpoint coordinates disagree at relative tolerance 1e-6
        and absolute tolerance 1e-5.
        Otherwise return None without changing values.
        """
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
    """Record a new death at the successor position.

    Inherited fields follow CanonicalVisualEventBaseV2. This is a frozen,
    slotted, keyword-only host record.

    Attributes
    ----------
    event_type : Literal['agent_died']
        Fixed serialized event kind, supplied by the class and excluded from
        construction.
    phase_rank : Literal[90]
        Fixed event ordering rank, supplied by the class and excluded from construction.
    recipient_global_slot : int
        Recipient global slot, a Python int in 0..9.
    recipient_anchor : VisualAgentAnchorV2
        Matching affected-agent anchor at successor.

    Raises
    ------
    ValueError
        Canonical header, affected global slot or required phase-anchor join is invalid.

    Notes
    -----
    The event producer owns the occurrence condition; constructing this record
    does not verify a simulator transition.
    """

    event_type: Literal["agent_died"] = field(default="agent_died", init=False)
    """Fixed serialized event kind, supplied by the class and excluded from
    construction.
    """
    phase_rank: Literal[90] = field(default=90, init=False)
    """Fixed event ordering rank, supplied by the class and excluded from
    construction.
    """
    recipient_global_slot: int
    """Recipient global slot, a Python int in 0..9."""
    recipient_anchor: VisualAgentAnchorV2
    """Matching affected-agent anchor at successor."""

    def __post_init__(self) -> None:
        """Validate AgentDiedEventV2 during host construction.

        Raise ValueError if canonical header, affected global slot or required
        phase-anchor join is invalid.
        Otherwise return None without changing values.
        """
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
    """Record a source's attributed contribution to one recorded death.

    Inherited fields follow CanonicalVisualEventBaseV2. This is a frozen,
    slotted, keyword-only host record.

    Attributes
    ----------
    event_type : Literal['lethal_damage_contribution']
        Fixed serialized event kind, supplied by the class and excluded from
        construction.
    phase_rank : Literal[90]
        Fixed event ordering rank, supplied by the class and excluded from construction.
    source_global_slot : int
        Source global slot, a Python int in 0..9.
    recipient_global_slot : int
        Recipient global slot, a Python int in 0..9.
    attributed_death_damage : float
        Nonnegative finite Python death-attributed damage in health units.
    source_anchor : VisualAgentAnchorV2
        Matching source anchor at successor.
    recipient_anchor : VisualAgentAnchorV2
        Matching dead recipient anchor at successor.

    Raises
    ------
    ValueError
        Canonical header, slots, finite nonnegative amount or successor-anchor
        joins are invalid.

    Notes
    -----
    This is contribution evidence, not a unique last-hit or killer assignment.
    """

    event_type: Literal["lethal_damage_contribution"] = field(
        default="lethal_damage_contribution", init=False
    )
    """Fixed serialized event kind, supplied by the class and excluded from
    construction.
    """
    phase_rank: Literal[90] = field(default=90, init=False)
    """Fixed event ordering rank, supplied by the class and excluded from
    construction.
    """
    source_global_slot: int
    """Source global slot, a Python int in 0..9."""
    recipient_global_slot: int
    """Recipient global slot, a Python int in 0..9."""
    attributed_death_damage: float
    """Nonnegative finite Python death-attributed damage in health units."""
    source_anchor: VisualAgentAnchorV2
    """Matching source anchor at successor."""
    recipient_anchor: VisualAgentAnchorV2
    """Matching dead recipient anchor at successor."""

    def __post_init__(self) -> None:
        """Validate LethalDamageContributionEventV2 during host construction.

        Raise ValueError if canonical header, slots, finite nonnegative amount
        or successor-anchor joins are invalid.
        Otherwise return None without changing values.
        """
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
    """Identify a recipient and status for a canonical lifecycle event.

    Inherited fields follow CanonicalVisualEventBaseV2. This is a frozen,
    slotted, keyword-only host record.

    Attributes
    ----------
    recipient_global_slot : int
        Recipient global slot, a Python int in 0..9.
    status_channel : int
        Scientific status channel, a Python int in 0..8.
    status_id : str
        Catalog status ID matching status_channel.
    recipient_anchor : VisualAgentAnchorV2
        Matching recipient anchor at successor.

    Raises
    ------
    ValueError
        Canonical header, recipient slot, scientific channel/catalog ID or
        successor anchor is invalid.
    """

    recipient_global_slot: int
    """Recipient global slot, a Python int in 0..9."""
    status_channel: int
    """Scientific status channel, a Python int in 0..8."""
    status_id: str
    """Catalog status ID matching status_channel."""
    recipient_anchor: VisualAgentAnchorV2
    """Matching recipient anchor at successor."""

    def __post_init__(self) -> None:
        """Validate StatusVisualEventBaseV2 during host construction.

        Raise ValueError if canonical header, recipient slot, scientific
        channel/catalog ID or successor anchor is invalid.
        Otherwise return None without changing values.
        """
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
    """Record that aging reduced an old status duration to zero.

    Inherited fields follow StatusVisualEventBaseV2. This is a frozen, slotted,
    keyword-only host record.

    Attributes
    ----------
    event_type : Literal['status_aged_to_zero']
        Fixed serialized event kind, supplied by the class and excluded from
        construction.
    phase_rank : Literal[100]
        Fixed event ordering rank, supplied by the class and excluded from construction.

    Raises
    ------
    ValueError
        Inherited canonical identity and recipient/status anchor checks fail.

    Notes
    -----
    Later application in the same transition may create an active successor status.
    """

    event_type: Literal["status_aged_to_zero"] = field(
        default="status_aged_to_zero", init=False
    )
    """Fixed serialized event kind, supplied by the class and excluded from
    construction.
    """
    phase_rank: Literal[100] = field(default=100, init=False)
    """Fixed event ordering rank, supplied by the class and excluded from
    construction.
    """


@dataclass(frozen=True, slots=True, kw_only=True)
class StatusBrokenByDamageEventV2(StatusVisualEventBaseV2):
    """Record that positive raw damage broke an old breakable status.

    Inherited fields follow StatusVisualEventBaseV2. This is a frozen, slotted,
    keyword-only host record.

    Attributes
    ----------
    event_type : Literal['status_broken_by_damage']
        Fixed serialized event kind, supplied by the class and excluded from
        construction.
    phase_rank : Literal[100]
        Fixed event ordering rank, supplied by the class and excluded from construction.

    Raises
    ------
    ValueError
        Inherited canonical identity and recipient/status anchor checks fail.

    Notes
    -----
    A later fresh application can coexist with this break event.
    """

    event_type: Literal["status_broken_by_damage"] = field(
        default="status_broken_by_damage", init=False
    )
    """Fixed serialized event kind, supplied by the class and excluded from
    construction.
    """
    phase_rank: Literal[100] = field(default=100, init=False)
    """Fixed event ordering rank, supplied by the class and excluded from
    construction.
    """


@dataclass(frozen=True, slots=True, kw_only=True)
class StatusAppliedEventV2(StatusVisualEventBaseV2):
    """Identify a direct source that applied a status this transition.

    Inherited fields follow StatusVisualEventBaseV2. This is a frozen, slotted,
    keyword-only host record.

    Attributes
    ----------
    event_type : Literal['status_applied']
        Fixed serialized event kind, supplied by the class and excluded from
        construction.
    phase_rank : Literal[100]
        Fixed event ordering rank, supplied by the class and excluded from construction.
    source_global_slot : int
        Source global slot, a Python int in 0..9.
    source_anchor : VisualAgentAnchorV2
        Matching source anchor at successor.

    Raises
    ------
    ValueError
        Inherited event/status checks, source slot or source successor anchor is
        invalid.

    Notes
    -----
    This direct application record can support durable source evidence; it does
    not by itself prove the status survives death clearing.
    """

    event_type: Literal["status_applied"] = field(default="status_applied", init=False)
    """Fixed serialized event kind, supplied by the class and excluded from
    construction.
    """
    phase_rank: Literal[100] = field(default=100, init=False)
    """Fixed event ordering rank, supplied by the class and excluded from
    construction.
    """
    source_global_slot: int
    """Source global slot, a Python int in 0..9."""
    source_anchor: VisualAgentAnchorV2
    """Matching source anchor at successor."""

    def __post_init__(self) -> None:
        """Validate StatusAppliedEventV2 during host construction.

        Raise ValueError if inherited event/status checks, source slot or source
        successor anchor is invalid.
        Otherwise return None without changing values.
        """
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
    """Record that a status was refreshed or extended on this transition.

    Inherited fields follow StatusVisualEventBaseV2. This is a frozen, slotted,
    keyword-only host record.

    Attributes
    ----------
    event_type : Literal['status_refreshed_or_extended']
        Fixed serialized event kind, supplied by the class and excluded from
        construction.
    phase_rank : Literal[100]
        Fixed event ordering rank, supplied by the class and excluded from construction.

    Raises
    ------
    ValueError
        Inherited canonical identity and recipient/status anchor checks fail.

    Notes
    -----
    This summary event does not identify a particular applying source.
    """

    event_type: Literal["status_refreshed_or_extended"] = field(
        default="status_refreshed_or_extended", init=False
    )
    """Fixed serialized event kind, supplied by the class and excluded from
    construction.
    """
    phase_rank: Literal[100] = field(default=100, init=False)
    """Fixed event ordering rank, supplied by the class and excluded from
    construction.
    """


@dataclass(frozen=True, slots=True, kw_only=True)
class StatusClearedByNewDeathEventV2(StatusVisualEventBaseV2):
    """Record that a new death cleared a status channel.

    Inherited fields follow StatusVisualEventBaseV2. This is a frozen, slotted,
    keyword-only host record.

    Attributes
    ----------
    event_type : Literal['status_cleared_by_new_death']
        Fixed serialized event kind, supplied by the class and excluded from
        construction.
    phase_rank : Literal[100]
        Fixed event ordering rank, supplied by the class and excluded from construction.

    Raises
    ------
    ValueError
        Inherited canonical identity and recipient/status anchor checks fail.

    Notes
    -----
    The recipient is anchored at its successor position.
    """

    event_type: Literal["status_cleared_by_new_death"] = field(
        default="status_cleared_by_new_death", init=False
    )
    """Fixed serialized event kind, supplied by the class and excluded from
    construction.
    """
    phase_rank: Literal[100] = field(default=100, init=False)
    """Fixed event ordering rank, supplied by the class and excluded from
    construction.
    """


@dataclass(frozen=True, slots=True, kw_only=True)
class SpawnShieldExpiredEventV2(CanonicalVisualEventBaseV2):
    """Record that a spawn shield expired on this transition.

    Inherited fields follow CanonicalVisualEventBaseV2. This is a frozen,
    slotted, keyword-only host record.

    Attributes
    ----------
    event_type : Literal['spawn_shield_expired']
        Fixed serialized event kind, supplied by the class and excluded from
        construction.
    phase_rank : Literal[110]
        Fixed event ordering rank, supplied by the class and excluded from construction.
    agent_global_slot : int
        Affected agent global slot, a Python int in 0..9.
    agent_anchor : VisualAgentAnchorV2
        Matching affected-agent anchor at successor.

    Raises
    ------
    ValueError
        Canonical header, affected global slot or required phase-anchor join is invalid.

    Notes
    -----
    The event producer owns the occurrence condition; constructing this record
    does not verify a simulator transition.
    """

    event_type: Literal["spawn_shield_expired"] = field(
        default="spawn_shield_expired", init=False
    )
    """Fixed serialized event kind, supplied by the class and excluded from
    construction.
    """
    phase_rank: Literal[110] = field(default=110, init=False)
    """Fixed event ordering rank, supplied by the class and excluded from
    construction.
    """
    agent_global_slot: int
    """Affected agent global slot, a Python int in 0..9."""
    agent_anchor: VisualAgentAnchorV2
    """Matching affected-agent anchor at successor."""

    def __post_init__(self) -> None:
        """Validate SpawnShieldExpiredEventV2 during host construction.

        Raise ValueError if canonical header, affected global slot or required
        phase-anchor join is invalid.
        Otherwise return None without changing values.
        """
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
    """Record a team wave boundary even if no actor respawned.

    Inherited fields follow CanonicalVisualEventBaseV2. This is a frozen,
    slotted, keyword-only host record.

    Attributes
    ----------
    event_type : Literal['respawn_wave_occurred']
        Fixed serialized event kind, supplied by the class and excluded from
        construction.
    phase_rank : Literal[120]
        Fixed event ordering rank, supplied by the class and excluded from construction.
    team_index : int
        Python team-axis index 0 or 1.
    team_id : int
        Python physical team ID equal to team_index + 1.
    team_anchor : VisualTeamAnchorV2
        Exact VisualTeamAnchorV2 for the same team at successor.

    Raises
    ------
    ValueError
        Canonical header or successor team-anchor identity is invalid.
    """

    event_type: Literal["respawn_wave_occurred"] = field(
        default="respawn_wave_occurred", init=False
    )
    """Fixed serialized event kind, supplied by the class and excluded from
    construction.
    """
    phase_rank: Literal[120] = field(default=120, init=False)
    """Fixed event ordering rank, supplied by the class and excluded from
    construction.
    """
    team_index: int
    """Python team-axis index 0 or 1."""
    team_id: int
    """Python physical team ID equal to team_index + 1."""
    team_anchor: VisualTeamAnchorV2
    """Exact VisualTeamAnchorV2 for the same team at successor."""

    def __post_init__(self) -> None:
        """Validate RespawnWaveOccurredEventV2 during host construction.

        Raise ValueError if canonical header or successor team-anchor identity
        is invalid.
        Otherwise return None without changing values.
        """
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
    """Record one agent's realized respawn position.

    Inherited fields follow CanonicalVisualEventBaseV2. This is a frozen,
    slotted, keyword-only host record.

    Attributes
    ----------
    event_type : Literal['agent_respawned']
        Fixed serialized event kind, supplied by the class and excluded from
        construction.
    phase_rank : Literal[120]
        Fixed event ordering rank, supplied by the class and excluded from construction.
    agent_global_slot : int
        Affected agent global slot, a Python int in 0..9.
    team_id : int
        Python physical team ID 1 or 2.
    realized_successor_position : Point2D
        Finite world (x, y) tuple at the recorded respawn endpoint.
    agent_anchor : VisualAgentAnchorV2
        Matching successor anchor whose position equals
        realized_successor_position exactly.

    Raises
    ------
    ValueError
        Canonical header, slot/team, world position or successor anchor
        agreement is invalid.
    """

    event_type: Literal["agent_respawned"] = field(
        default="agent_respawned", init=False
    )
    """Fixed serialized event kind, supplied by the class and excluded from
    construction.
    """
    phase_rank: Literal[120] = field(default=120, init=False)
    """Fixed event ordering rank, supplied by the class and excluded from
    construction.
    """
    agent_global_slot: int
    """Affected agent global slot, a Python int in 0..9."""
    team_id: int
    """Python physical team ID 1 or 2."""
    realized_successor_position: Point2D
    """Finite world (x, y) tuple at the recorded respawn endpoint."""
    agent_anchor: VisualAgentAnchorV2
    """Matching successor anchor whose position equals realized_successor_position
    exactly.
    """

    def __post_init__(self) -> None:
        """Validate AgentRespawnedEventV2 during host construction.

        Raise ValueError if canonical header, slot/team, world position or
        successor anchor agreement is invalid.
        Otherwise return None without changing values.
        """
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
    """Record an authoritative team-score increase after lifecycle resolution.

    Inherited fields follow CanonicalVisualEventBaseV2. This is a frozen,
    slotted, keyword-only host record.

    Attributes
    ----------
    event_type : Literal['team_deathmatch_score_changed']
        Fixed serialized event kind, supplied by the class and excluded from
        construction.
    phase_rank : Literal[130]
        Fixed event ordering rank, supplied by the class and excluded from construction.
    team_index : Literal[0, 1]
        Python team-axis index 0 or 1.
    team_id : Literal[1, 2]
        Python physical team ID equal to team_index + 1.
    score_increment : int
        Positive Python signed-int32 score increase.
    previous_score : int
        Nonnegative Python signed-int32 score before this transition.
    successor_score : int
        Python signed-int32 score equal to previous_score plus score_increment.
    team_anchor : VisualTeamAnchorV2
        Exact successor team anchor matching team_index and team_id.

    Raises
    ------
    ValueError
        Canonical header, team join, int32 bounds, positive increment, score
        arithmetic or anchor is invalid.
    """

    event_type: Literal["team_deathmatch_score_changed"] = field(
        default="team_deathmatch_score_changed", init=False
    )
    """Fixed serialized event kind, supplied by the class and excluded from
    construction.
    """
    phase_rank: Literal[130] = field(default=130, init=False)
    """Fixed event ordering rank, supplied by the class and excluded from
    construction.
    """
    team_index: Literal[0, 1]
    """Python team-axis index 0 or 1."""
    team_id: Literal[1, 2]
    """Python physical team ID equal to team_index + 1."""
    score_increment: int
    """Positive Python signed-int32 score increase."""
    previous_score: int
    """Nonnegative Python signed-int32 score before this transition."""
    successor_score: int
    """Python signed-int32 score equal to previous_score plus score_increment."""
    team_anchor: VisualTeamAnchorV2
    """Exact successor team anchor matching team_index and team_id."""

    def __post_init__(self) -> None:
        """Validate TeamDeathmatchScoreChangedEventV2 during host construction.

        Raise ValueError if canonical header, team join, int32 bounds, positive
        increment, score arithmetic or anchor is invalid.
        Otherwise return None without changing values.
        """
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
    """Record the declared Team Deathmatch outcome and completion basis.

    Inherited fields follow CanonicalVisualEventBaseV2. This is a frozen,
    slotted, keyword-only host record.

    Attributes
    ----------
    event_type : Literal['team_deathmatch_completed']
        Fixed serialized event kind, supplied by the class and excluded from
        construction.
    phase_rank : Literal[140]
        Fixed event ordering rank, supplied by the class and excluded from construction.
    outcome : Literal['team_a_win', 'team_b_win', 'draw']
        team_a_win, team_b_win or draw.
    completion_basis : Literal['score_threshold', 'horizon',
    'score_threshold_at_horizon']
        score_threshold, horizon or score_threshold_at_horizon.

    Raises
    ------
    ValueError
        Canonical header, outcome or completion basis is invalid.

    Notes
    -----
    The record preserves the producer's result. It does not recompute scores or
    decide the winner.
    """

    event_type: Literal["team_deathmatch_completed"] = field(
        default="team_deathmatch_completed", init=False
    )
    """Fixed serialized event kind, supplied by the class and excluded from
    construction.
    """
    phase_rank: Literal[140] = field(default=140, init=False)
    """Fixed event ordering rank, supplied by the class and excluded from
    construction.
    """
    outcome: Literal["team_a_win", "team_b_win", "draw"]
    """team_a_win, team_b_win or draw."""
    completion_basis: Literal[
        "score_threshold", "horizon", "score_threshold_at_horizon"
    ]
    """score_threshold, horizon or score_threshold_at_horizon."""

    def __post_init__(self) -> None:
        """Validate TeamDeathmatchCompletedEventV2 during host construction.

        Raise ValueError if canonical header, outcome or completion basis is invalid.
        Otherwise return None without changing values.
        """
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
    """Keep one agent's three explicit recorded phase anchors.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    global_slot : int
        Global simulator slot, a Python int in 0..9; Team A precedes Team B.
    public_agent_id : str
        Nonblank recorded public agent ID; distinct from the numerical slot.
    transition_start : VisualAgentAnchorV2
        Exact anchor before the transition.
    post_charge : VisualAgentAnchorV2
        Exact anchor after Charge and before ordinary movement.
    successor : VisualAgentAnchorV2
        Exact anchor in the successor frame, including any respawn.

    Raises
    ------
    ValueError
        Slot/public ID or any anchor's phase, slot or public ID disagrees.

    Notes
    -----
    The tuple of phase facts does not encode a continuous path between anchors.
    """

    global_slot: int
    """Global simulator slot, a Python int in 0..9; Team A precedes Team B."""
    public_agent_id: str
    """Nonblank recorded public agent ID; distinct from the numerical slot."""
    transition_start: VisualAgentAnchorV2
    """Exact anchor before the transition."""
    post_charge: VisualAgentAnchorV2
    """Exact anchor after Charge and before ordinary movement."""
    successor: VisualAgentAnchorV2
    """Exact anchor in the successor frame, including any respawn."""

    def __post_init__(self) -> None:
        """Validate VisualAgentPhaseTrajectoryV2 during host construction.

        Raise ValueError if slot/public ID or any anchor's phase, slot or public
        ID disagrees.
        Otherwise return None without changing values.
        """
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
    | AgentLeftCombatEventV2
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
    AgentLeftCombatEventV2,
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
    """Collect one transition's canonical events and phase anchors.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    schema_version : int
        Exact Python int 2.
    episode_id : str
        Nonblank recorded episode ID used to join frames and transitions.
    transition_index : int
        Nonnegative Python recorded transition index.
    transition_id : str
        Canonical episode_id:transition:transition_index string.
    start_frame_id : str
        Canonical frame ID at transition_index.
    successor_frame_id : str
        Canonical frame ID at transition_index + 1.
    start_simulator_step_count : int
        Nonnegative Python simulator step before the transition.
    successor_simulator_step_count : int
        Python simulator step exactly one greater than the start count.
    public_agent_id_by_global_slot : tuple[str, ...]
        Exact tuple of ten unique nonblank IDs, including unused slots.
    configured_active_by_global_slot : tuple[bool, ...]
        Exact tuple of ten Python bool membership flags.
    agent_phase_trajectories : tuple[VisualAgentPhaseTrajectoryV2, ...]
        Exact tuple covering configured slots once in increasing global order.
    events : tuple[VisualEventV2, ...]
        Tuple of supported exact V2 event types with ordinals 0..N-1, this
        transition ID and nondecreasing phase ranks.

    Raises
    ------
    ValueError
        Canonical identity/step joins, ten-slot roster, trajectory coverage,
        event ordering/types or phase-anchor joins are invalid.

    Notes
    -----
    Events remain in recorded order. Validation checks structure and joins; the
    canonical evaluation record remains the scientific event authority.
    """

    schema_version: int
    """Exact Python int 2."""
    episode_id: str
    """Nonblank recorded episode ID used to join frames and transitions."""
    transition_index: int
    """Nonnegative Python recorded transition index."""
    transition_id: str
    """Canonical episode_id:transition:transition_index string."""
    start_frame_id: str
    """Canonical frame ID at transition_index."""
    successor_frame_id: str
    """Canonical frame ID at transition_index + 1."""
    start_simulator_step_count: int
    """Nonnegative Python simulator step before the transition."""
    successor_simulator_step_count: int
    """Python simulator step exactly one greater than the start count."""
    public_agent_id_by_global_slot: tuple[str, ...]
    """Exact tuple of ten unique nonblank IDs, including unused slots."""
    configured_active_by_global_slot: tuple[bool, ...]
    """Exact tuple of ten Python bool membership flags."""
    agent_phase_trajectories: tuple[VisualAgentPhaseTrajectoryV2, ...]
    """Exact tuple covering configured slots once in increasing global order."""
    events: tuple[VisualEventV2, ...]
    """Tuple of supported exact V2 event types with ordinals 0..N-1, this
    transition ID and nondecreasing phase ranks.
    """

    def __post_init__(self) -> None:
        """Validate VisualEventBatchV2 during host construction.

        Raise ValueError if canonical identity/step joins, ten-slot roster,
        trajectory coverage, event ordering/types or phase-anchor joins are
        invalid.
        Otherwise return None without changing values.
        """
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
    """Bind a researcher scene to its incoming events and status evidence.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    schema_version : int
        Exact Python int 2.
    scene : BattlefieldSceneV2
        Exact BattlefieldSceneV2 for the selected recorded frame.
    incoming_events : VisualEventBatchV2 | None
        None for frame zero; otherwise the exact VisualEventBatchV2 entering the scene.
    status_source_evidence : StatusSourceEvidenceStateV2
        Exact StatusSourceEvidenceStateV2 matching the scene frame and every
        active status.

    Raises
    ------
    ValueError
        Version/root types, frame/event joins, trajectory endpoints, respawn
        evidence or active-status evidence coverage disagrees.

    Notes
    -----
    This is privileged researcher output. It must not be passed off as an
    actor's authorized policy input.
    """

    schema_version: int
    """Exact Python int 2."""
    scene: BattlefieldSceneV2
    """Exact BattlefieldSceneV2 for the selected recorded frame."""
    incoming_events: VisualEventBatchV2 | None
    """None for frame zero; otherwise the exact VisualEventBatchV2 entering the
    scene.
    """
    status_source_evidence: StatusSourceEvidenceStateV2
    """Exact StatusSourceEvidenceStateV2 matching the scene frame and every active
    status.
    """

    def __post_init__(self) -> None:
        """Validate ResearcherAnalyzerProjectionV2 during host construction.

        Raise ValueError if version/root types, frame/event joins, trajectory
        endpoints, respawn evidence or active-status evidence coverage
        disagrees.
        Otherwise return None without changing values.
        """
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
    """Check a legacy event's nonblank ID and nonnegative transition ID.

    event_id must be a Python string; transition_id must be a Python int.
    Raise ValueError for invalid values. No canonical string pattern or
    relationship between the two fields is enforced here. Return None.
    """
    _require_text(event_id, name="event_id")
    _require_python_int(transition_id, name="transition_id", minimum=0)


def to_jsonable(value: object) -> object:
    """Convert a scene record into a fresh tree of JSON-compatible values.

    Parameters
    ----------
    value : object
        Dataclass instance, tuple, None, or exact Python str/int/float/bool.
        Nested values must satisfy the same contract. Lists, dictionaries,
        arrays and renderer objects are not accepted as input containers.

    Returns
    -------
    object
        Dataclasses become dictionaries in field order, tuples become lists,
        and scalar values are retained. Input records are unchanged.

    Raises
    ------
    TypeError
        A value is outside the accepted record/container/scalar types.

    Notes
    -----
    This host helper does not write a file, check audience authorization or
    filter fields. It trusts the record's validation and does not separately
    reject nonfinite floats or detect cycles in arbitrary dataclass inputs.
    """
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
