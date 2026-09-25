"""Build typed presentation facts from validated recorded evaluation data.
Use ``build_oracle_authorized_scene_v1`` for the displayed Oracle scene,
``build_replay_oracle_presentation_parts_v1`` for replay scene/history/action
parts, and ``build_agent_pov_visual_incoming_summary_v1`` for events allowed
by adjacent Agent POV scenes. Frozen dataclasses define the wire fields and
reject inconsistent identities, phases, counts, and values at construction.
These helpers do not run the simulator or perform HTTP, file, JAX, or NumPy
work. Callers own authorization and the outer session/epoch envelope. Oracle
outgoing action anchors come from the displayed scene, never a future frame.
Spatial values use world units, health uses hit points, and durations use
transition ticks unless a field states otherwise. Scenes from recordings that
record the Team Deathmatch Red Zone rule carry AuthorizedMapV2, whose
AuthorizedRedZoneV1 row holds each team's exact float32 strip;
build_authorized_map_v2 builds it for the Oracle and Agent POV scenes alike.
Class docstrings also feed Pydantic schema descriptions; the compact browser
schema strips that prose.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from hashlib import sha256
from math import copysign, isclose, isfinite
from struct import pack, unpack
from typing import Annotated, ClassVar, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from pydantic_core import PydanticSerializationError

from marl_battlegrounds.evaluation.models import (
    FLOAT32_SMALLEST_NORMAL,
    EvaluationEpisodeContext,
    EvaluationTransitionV1,
    ResolvedEnvConfigV2,
    StaticMechanicsCatalogV1,
    evaluation_context_type,
    float32_value,
    red_zone_team_on_right,
    red_zone_x_range,
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
    AgentLeftCombatEventV2,
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
    TeamDeathmatchCompletedEventV2,
    TeamDeathmatchScoreChangedEventV2,
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
    "agent_left_combat",
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
    "team_deathmatch_score_changed",
    "team_deathmatch_completed",
]

_REPLAY_INCOMING_EVENT_KINDS_V1 = frozenset(
    (
        "action_rejected",
        "ability_activated",
        "source_damage_output",
        "source_healing_output",
        "recipient_health_resolution",
        "combat_countdown_reset",
        "agent_left_combat",
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
        "team_deathmatch_score_changed",
        "team_deathmatch_completed",
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
AUTHORIZED_CLASS_DOCUMENTATION_CATALOG_FINGERPRINT_V1 = (
    "7e5306209ecc91f34b0dee0b23d0e0049deb142e8a3c8877aff168597c6462a6"
)
AUTHORIZED_CLASS_DOCUMENTATION_PROFILE_ID_V1 = (
    "marl_battlegrounds.class_documentation.canonical_v1"
)

_CANONICAL_CLASS_DOCUMENTATION_SHAPE_V1: tuple[tuple[object, ...], ...] = (
    (1, "Mage", "enemy", "target_none"),
    (2, "Warrior", "enemy", "enemy"),
    (3, "Hunter", "enemy", "enemy"),
    (4, "Rogue", "enemy", "enemy"),
    (5, "Priest", "ally", "ally"),
)

_CANONICAL_STATUS_DOCUMENTATION_SHAPE_V1: tuple[tuple[object, ...], ...] = (
    (
        0,
        "warrior_charge_slow",
        "slow",
        2,
        "ultimate",
        "movement_multiplier",
        "maximum_remaining_duration",
        False,
    ),
    (
        1,
        "hunter_basic_slow",
        "slow",
        3,
        "basic",
        "movement_multiplier",
        "maximum_remaining_duration",
        False,
    ),
    (
        2,
        "rogue_poison_slow",
        "slow",
        4,
        "ultimate",
        "movement_multiplier",
        "maximum_remaining_duration",
        False,
    ),
    (
        3,
        "warrior_charge_stun",
        "stun",
        2,
        "ultimate",
        "none",
        "maximum_remaining_duration",
        False,
    ),
    (
        4,
        "hunter_trap_stun",
        "stun",
        3,
        "ultimate",
        "none",
        "maximum_remaining_duration",
        True,
    ),
    (
        5,
        "rogue_poison_stun",
        "stun",
        4,
        "ultimate",
        "none",
        "maximum_remaining_duration",
        False,
    ),
    (
        6,
        "rogue_poison_anti_heal",
        "anti_heal",
        4,
        "ultimate",
        "healing_multiplier",
        "maximum_remaining_duration",
        False,
    ),
    (
        7,
        "mage_burst_damage_amplification",
        "damage_amplification",
        1,
        "ultimate",
        "damage_multiplier",
        "maximum_remaining_duration",
        False,
    ),
    (
        8,
        "priest_blessing_of_freedom_movement_floor",
        "movement_floor",
        5,
        "basic",
        "movement_floor",
        "maximum_remaining_duration",
        False,
    ),
)

_CANONICAL_AURA_DOCUMENTATION_SHAPE_V1: tuple[tuple[object, ...], ...] = (
    ("mage_damage_amplification", 1, "same_team", "multiply_then_clamp", "ceiling"),
    ("warrior_damage_mitigation", 2, "same_team", "multiply_then_clamp", "floor"),
)


def _require_python_int(value: int, *, name: str, minimum: int = 0) -> None:
    """Require an exact Python int at or above a bound.
    Reject bools and numeric scalar substitutes. minimum defaults to 0.
    Parameters
    ----------
    value : int
        Value to check; this helper does not coerce it.
    name : str
        Field label used in validation errors.
    minimum : int
        Inclusive lower bound; default follows the function signature.
    Returns
    -------
    None
        The value satisfies the integer contract.
    Raises
    ------
    ValueError
        The value has the wrong type or is below minimum.
    """
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be a Python int >= {minimum}.")


def _require_text(value: str, *, name: str) -> None:
    """Require a nonempty non-whitespace Python string.
    Keep the original string; whitespace is checked, not stripped.
    Parameters
    ----------
    value : str
        Value to check; this helper does not coerce it.
    name : str
        Field label used in validation errors.
    Returns
    -------
    None
        The text contract is satisfied.
    Raises
    ------
    ValueError
        The value is not an exact str or contains only whitespace.
    """
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a non-empty Python string.")


def _require_finite(value: float, *, name: str, minimum: float | None = None) -> None:
    """Require an exact finite Python float and optional lower bound.
    minimum defaults to None, which adds no lower bound. Integers are not
    coerced.
    Parameters
    ----------
    value : float
        Value to check; this helper does not coerce it.
    name : str
        Field label used in validation errors.
    minimum : float | None
        Inclusive lower bound; default follows the function signature.
    Returns
    -------
    None
        The numeric contract is satisfied.
    Raises
    ------
    ValueError
        The type, finiteness, or requested bound is invalid.
    """
    if type(value) is not float or not isfinite(value):
        raise ValueError(f"{name} must be a finite Python float.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")


def _require_point(value: Point2D, *, name: str) -> None:
    """Require an (x, y) tuple of finite Python floats.
    The tuple has exactly two coordinates; no array/list conversion is
    performed.
    Parameters
    ----------
    value : Point2D
        Value to check; this helper does not coerce it.
    name : str
        Field label used in validation errors.
    Returns
    -------
    None
        The point contract is satisfied.
    Raises
    ------
    ValueError
        The tuple shape or either coordinate is invalid.
    """
    if type(value) is not tuple or len(value) != 2:
        raise ValueError(f"{name} must be a two-coordinate Python tuple.")
    for coordinate in value:
        _require_finite(coordinate, name=f"{name} coordinate")


def _points_close(left: Point2D, right: Point2D) -> bool:
    """Compare two points using the recorded-geometry tolerance.
    Use relative tolerance 1e-6 and absolute tolerance 1e-5 on both coordinates.
    Parameters
    ----------
    left : Point2D
        First value to compare.
    right : Point2D
        Second value to compare.
    Returns
    -------
    bool
        Whether every paired coordinate is close.
    Raises
    ------
    ValueError
        Input lengths differ; callers normally supply two-coordinate points.
    """
    return all(
        isclose(left_value, right_value, rel_tol=1e-6, abs_tol=1e-5)
        for left_value, right_value in zip(left, right, strict=True)
    )


def _equals_catalog_or_exact_f32_encoding(
    recorded_value: float,
    catalog_value: float,
) -> bool:
    """Join a recorded number to its catalog value without a tolerance.
    Accept exact equality or the exact IEEE-754 float32 encoding of the catalog
    float. A catalog value outside float32 range returns False.
    Parameters
    ----------
    recorded_value : float
        Value preserved by the recorded observation wire.
    catalog_value : float
        Corresponding public catalog value.
    Returns
    -------
    bool
        Whether the two representations describe the same permitted value.
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
    """Compare optional recorded/catalog values with the exact float32 join rule.
    If either value is None, both must be None. Otherwise use the exact
    recorded/catalog comparison.
    Parameters
    ----------
    recorded_value : float | None
        Value preserved by the recorded observation wire.
    catalog_value : float | None
        Corresponding public catalog value.
    Returns
    -------
    bool
        Whether optional values agree.
    """
    if recorded_value is None or catalog_value is None:
        return recorded_value is catalog_value
    return _equals_catalog_or_exact_f32_encoding(recorded_value, catalog_value)


def _require_exact_tuple(
    value: object,
    *,
    name: str,
    item_type: type[object],
) -> None:
    """Require a Python tuple whose rows have one exact class.
    Subclasses and lists are rejected; an empty tuple is permitted.
    Parameters
    ----------
    value : object
        Value to check; this helper does not coerce it.
    name : str
        Field label used in validation errors.
    item_type : type[object]
        Exact permitted row class; subclasses do not qualify.
    Returns
    -------
    None
        The container/type contract is satisfied.
    Raises
    ------
    ValueError
        The container or any row has the wrong exact type.
    """
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
    """Create a stable opaque key for an Oracle agent in one authority namespace.
    Hash the namespace and public ID with a fixed Oracle prefix. This helper
    does not validate its strings or encode an internal slot.
    Parameters
    ----------
    authority_session_id : str
        Authority/session namespace to hash; callers validate it before this
        helper.
    public_agent_id : str
        Public identity used in this authority namespace.
    Returns
    -------
    str
        oracle_ followed by the SHA-256 hex digest.
    """
    payload = f"oracle\x00{authority_session_id}\x00{public_agent_id}".encode()
    return f"oracle_{sha256(payload).hexdigest()}"


@dataclass(frozen=True, slots=True, kw_only=True)
class AuthorizedObstacleV1:
    """Store one static obstacle with a consistent shape.
    Pillars require positive radius and no wall dimensions. Walls require
    positive width and height and no radius. Coordinates and rotation are
    finite.
    All constructor fields are required keyword arguments; instances are frozen.
    Attributes
    ----------
    obstacle_id : str
        Nonempty identifier, unique within the map.
    kind : Literal['pillar', 'wall']
        Obstacle shape: pillar uses radius; wall uses width and height.
    center : Point2D
        Center coordinates (x, y), in world units.
    radius : float | None
        Positive pillar radius in world units, or None for a wall.
    width : float | None
        Positive wall width in world units, or None for a pillar.
    height : float | None
        Positive wall height in world units, or None for a pillar.
    theta : float
        Obstacle rotation in radians; must be a finite Python float.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG
    """Strict Pydantic wire settings: forbid extra fields, nonfinite numbers, and
    coercion."""

    obstacle_id: str
    """Nonempty identifier, unique within the map."""
    kind: Literal["pillar", "wall"]
    """Obstacle shape: pillar uses radius; wall uses width and height."""
    center: Point2D
    """Center coordinates (x, y), in world units."""
    radius: float | None
    """Positive pillar radius in world units, or None for a wall."""
    width: float | None
    """Positive wall width in world units, or None for a pillar."""
    height: float | None
    """Positive wall height in world units, or None for a pillar."""
    theta: float
    """Obstacle rotation in radians; must be a finite Python float."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Pillars require positive radius and no wall dimensions. Walls require
        positive width and height and no radius. Coordinates and rotation are
        finite.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
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
    """Store positive map bounds and ordered static obstacles.
    Widths and heights are positive finite floats. Obstacle rows have exact
    types and unique IDs.
    All constructor fields are required keyword arguments; instances are frozen.
    Attributes
    ----------
    width : float
        Positive finite map width in world units.
    height : float
        Positive finite map height in world units.
    obstacles : tuple[AuthorizedObstacleV1, ...]
        Ordered obstacle rows with unique IDs.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG
    """Strict Pydantic wire settings: forbid extra fields, nonfinite numbers, and
    coercion."""

    width: float
    """Positive finite map width in world units."""
    height: float
    """Positive finite map height in world units."""
    obstacles: tuple[AuthorizedObstacleV1, ...]
    """Ordered obstacle rows with unique IDs."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Widths and heights are positive finite floats. Obstacle rows have exact
        types and unique IDs.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
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
class AuthorizedRedZoneV1:
    """Store both teams' recorded Red Zone strips on one map.
    Each team's Red Zone is a full-height strip at its own spawn side. When an
    agent dies with its centre inside its own team's strip, the enemy team gets
    2 points instead of 1. Values are float32 numbers exactly as Core scores
    with them. This row checks only types and shapes; AuthorizedMapV2 checks
    the numbers against its map width.
    All constructor fields are required keyword arguments; instances are frozen.
    Attributes
    ----------
    depth : float
        Strip depth in map units: the float32 value of the configured depth.
    team_a_x_range : tuple[float, float]
        Team A strip as inclusive (x_min, x_max) world x bounds: (0.0, depth)
        on the left, or (float32(width - depth), float32 width) on the right.
        The strip covers the full map height. A collapsed range (x_min equal to
        x_max) is legal.
    team_b_x_range : tuple[float, float]
        Team B strip, with the same meaning and rules as team_a_x_range.
    Raises
    ------
    ValueError
        The depth is not a finite Python float, or a range is not a tuple of
        two finite Python floats.
    """

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG
    """Strict Pydantic wire settings: forbid extra fields, nonfinite numbers, and
    coercion."""

    depth: float
    """Strip depth in map units: the float32 value of the configured depth."""
    team_a_x_range: tuple[float, float]
    """Team A strip as inclusive (x_min, x_max) world x bounds, full map height."""
    team_b_x_range: tuple[float, float]
    """Team B strip as inclusive (x_min, x_max) world x bounds, full map height."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        The depth is a finite Python float and each range is a tuple of two
        finite Python floats. Width-dependent rules live in AuthorizedMapV2.
        Raises
        ------
        ValueError
            A required type or value is invalid.
        """
        _require_finite(self.depth, name="red zone depth")
        _require_point(self.team_a_x_range, name="team_a_x_range")
        _require_point(self.team_b_x_range, name="team_b_x_range")


def _validate_authorized_red_zone(
    map_width: float,
    red_zone: AuthorizedRedZoneV1,
) -> None:
    """Check that recorded strips are exactly the strips Core scores with.
    Use the same float32 numbers as Core and the browser check
    (validateAuthorizedRedZone). With w32 the float32 map width, the depth must
    be exactly a float32 value, at least the smallest normal float32 (2**-126)
    and at most w32. The only permitted ranges are the left strip (0.0, depth)
    and the right strip (float32(w32 - depth), w32), built by
    models.red_zone_x_range. Each team's range must equal one of them exactly;
    both teams on one side is legal, and a collapsed range is legal. Display
    clipping belongs to the renderer and never changes this record.
    Parameters
    ----------
    map_width : float
        Recorded map width in world units; only its float32 value is used.
    red_zone : AuthorizedRedZoneV1
        Row whose types were already checked by its constructor.
    Returns
    -------
    None
        The strips are the exact scoring strips for this width and depth.
    Raises
    ------
    ValueError
        The depth is not a float32 value, is below the smallest normal float32
        or above w32, or a range is not one of the two permitted strips (for
        example (0, 1) and (19, 20) at width 20 and depth 5).
    """
    depth = red_zone.depth
    if float32_value(depth) != depth:
        raise ValueError("red zone depth must be an exact float32 value.")
    if depth < FLOAT32_SMALLEST_NORMAL or depth > float32_value(map_width):
        raise ValueError(
            "red zone depth must be a normal float32 value no larger than the map "
            "width."
        )
    permitted = (
        red_zone_x_range(map_width, depth, False),
        red_zone_x_range(map_width, depth, True),
    )
    for name, x_range in (
        ("team_a_x_range", red_zone.team_a_x_range),
        ("team_b_x_range", red_zone.team_b_x_range),
    ):
        if x_range not in permitted:
            raise ValueError(f"{name} must be one team's exact Red Zone strip.")


@dataclass(frozen=True, slots=True, kw_only=True)
class AuthorizedMapV2(AuthorizedMapV1):
    """Store map bounds, obstacles and the recorded Red Zone strips.
    Used for recordings that record the Red Zone rule (resolved config V2,
    20-column context). Inherits AuthorizedMapV1's width, height and obstacle
    rules. A separate version, so older scenes keep AuthorizedMapV1 unchanged.
    All constructor fields are required keyword arguments; instances are frozen.
    Attributes
    ----------
    map_version : Literal[2]
        Exact integer 2; tells this record apart from AuthorizedMapV1.
    red_zone : AuthorizedRedZoneV1 | None
        Both teams' strips, or None when the rule is recorded with depth 0
        (one point per death). Required; there is no default.
    Raises
    ------
    ValueError
        An inherited map rule fails, map_version is not the integer 2, red_zone
        is not an exact AuthorizedRedZoneV1 or None, or its numbers break the
        rules in _validate_authorized_red_zone.
    Notes
    -----
    The strips use float32 numbers exactly as Core scores, so an Oracle scene
    (raw config width) and an Agent POV scene (float32 context width) record
    the same strips.
    """

    map_version: Literal[2]
    """Exact integer 2; tells this record apart from AuthorizedMapV1."""
    red_zone: AuthorizedRedZoneV1 | None
    """Both teams' strips, or None when the rule is recorded with depth 0."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Apply the inherited map checks, then the version and Red Zone checks.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
        super(AuthorizedMapV2, self).__post_init__()
        if type(self.map_version) is not int or self.map_version != 2:
            raise ValueError("map_version must be the exact integer 2.")
        if self.red_zone is None:
            return
        if type(self.red_zone) is not AuthorizedRedZoneV1:
            raise ValueError("red_zone must be the exact AuthorizedRedZoneV1 row.")
        _validate_authorized_red_zone(self.width, self.red_zone)


type AuthorizedMap = AuthorizedMapV1 | AuthorizedMapV2


def build_authorized_map_v2(
    base: AuthorizedMapV1,
    *,
    red_zone_depth: float,
    team_spawn_pad_x: Sequence[Sequence[float]],
) -> AuthorizedMapV2:
    """Add the recorded Red Zone strips to an authorized V1 map.
    Keep the base map's width, height and obstacles. Decide each team's side
    with models.red_zone_team_on_right (the exact host copy of Core's side
    rule: right when the mean pad x is past the centre) and build each strip
    with models.red_zone_x_range, both on the float32 width.
    Parameters
    ----------
    base : AuthorizedMapV1
        Exact V1 map from the calling module's own map conversion.
    red_zone_depth : float
        Recorded depth in map units: a resolved config's depth or a
        recipient's context column 19. 0.0 gives red_zone=None; a positive
        value is stored as its float32 value.
    team_spawn_pad_x : Sequence[Sequence[float]]
        Pad x values shaped (2, 5): Team A's five pads, then Team B's five, in
        any order within a team and including unused pads.
    Returns
    -------
    AuthorizedMapV2
        New map record with map_version 2.
    Raises
    ------
    TypeError
        base is not an exact AuthorizedMapV1.
    ValueError
        The depth is not a finite nonnegative float, a positive depth is not a
        normal float32 value at most the float32 width, or the pads do not have
        shape (2, 5).
    """
    if type(base) is not AuthorizedMapV1:
        raise TypeError("base must be the exact AuthorizedMapV1 row.")
    if (
        type(red_zone_depth) is not float
        or not isfinite(red_zone_depth)
        or copysign(1.0, red_zone_depth) < 0.0
    ):
        raise ValueError("red_zone_depth must be a finite nonnegative float.")
    if len(team_spawn_pad_x) != 2 or any(len(team) != 5 for team in team_spawn_pad_x):
        raise ValueError("team_spawn_pad_x must have shape (2, 5).")
    red_zone: AuthorizedRedZoneV1 | None = None
    if red_zone_depth != 0.0:
        depth = float32_value(red_zone_depth)
        team_a_range, team_b_range = (
            red_zone_x_range(
                base.width,
                depth,
                red_zone_team_on_right(base.width, team_pads),
            )
            for team_pads in team_spawn_pad_x
        )
        red_zone = AuthorizedRedZoneV1(
            depth=depth,
            team_a_x_range=team_a_range,
            team_b_x_range=team_b_range,
        )
    return AuthorizedMapV2(
        width=base.width,
        height=base.height,
        obstacles=base.obstacles,
        map_version=2,
        red_zone=red_zone,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class AuthorizedAuraModifierV1:
    """Store one non-neutral aggregate aura effect on a recipient.
    Only the two canonical aura IDs are accepted; the multiplier is finite,
    nonnegative, and different from 1.
    All constructor fields are required keyword arguments; instances are frozen.
    Attributes
    ----------
    aura_id : AuthorizedAuraIdV1
        Canonical Mage damage or Warrior mitigation aura identifier.
    multiplier : float
        Finite nonnegative aggregate multiplier; neutral 1.0 is omitted.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG
    """Strict Pydantic wire settings: forbid extra fields, nonfinite numbers, and
    coercion."""

    aura_id: AuthorizedAuraIdV1
    """Canonical Mage damage or Warrior mitigation aura identifier."""
    multiplier: float
    """Finite nonnegative aggregate multiplier; neutral 1.0 is omitted."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Only the two canonical aura IDs are accepted; the multiplier is finite,
        nonnegative, and different from 1.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
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
    """Describe one public class status mechanic.
    The channel and ID retain the nine-status catalog identity. Duration is
    positive; magnitude presence agrees with its kind.
    All constructor fields are required keyword arguments; instances are frozen.
    Attributes
    ----------
    status_channel : int
        Zero-based channel on the fixed nine-status axis.
    status_id : str
        Canonical status identifier matching status_channel.
    family : str
        Allowed values: 'slow', 'stun', 'anti_heal', 'damage_amplification',
        'movement_floor'.
        Effect family named by the status catalog.
    source_action_component : Literal['basic', 'ultimate']
        Basic or Ultimate action component that creates the effect.
    duration_steps : int
        Configured positive duration in transition ticks.
    magnitude_kind : str
        Allowed values: 'movement_multiplier', 'none', 'healing_multiplier',
        'damage_multiplier', 'movement_floor'.
        Meaning of magnitude; none requires magnitude to be None.
    magnitude : float | None
        Finite configured effect value, or None for magnitude kind none.
    breaks_on_positive_damage : bool
        Whether positive damage removes this status; an exact bool.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG
    """Strict Pydantic wire settings: forbid extra fields, nonfinite numbers, and
    coercion."""

    status_channel: int
    """Zero-based channel on the fixed nine-status axis."""
    status_id: str
    """Canonical status identifier matching status_channel."""
    family: Literal[
        "slow",
        "stun",
        "anti_heal",
        "damage_amplification",
        "movement_floor",
    ]
    """Allowed values: 'slow', 'stun', 'anti_heal', 'damage_amplification',
    'movement_floor'. Effect family named by the status catalog."""
    source_action_component: Literal["basic", "ultimate"]
    """Basic or Ultimate action component that creates the effect."""
    duration_steps: int
    """Configured positive duration in transition ticks."""
    magnitude_kind: Literal[
        "movement_multiplier",
        "none",
        "healing_multiplier",
        "damage_multiplier",
        "movement_floor",
    ]
    """Allowed values: 'movement_multiplier', 'none', 'healing_multiplier',
    'damage_multiplier', 'movement_floor'. Meaning of magnitude; none requires
    magnitude to be None."""
    magnitude: float | None
    """Finite configured effect value, or None for magnitude kind none."""
    breaks_on_positive_damage: bool
    """Whether positive damage removes this status; an exact bool."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        The channel and ID retain the nine-status catalog identity. Duration is
        positive; magnitude presence agrees with its kind.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
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
    """Describe one public class aura mechanic.
    The aura ID is known, numeric values are finite and nonnegative, and
    stacking uses multiply_then_clamp with a ceiling or floor.
    All constructor fields are required keyword arguments; instances are frozen.
    Attributes
    ----------
    aura_id : AuthorizedAuraIdV1
        Canonical Mage damage or Warrior mitigation aura identifier.
    radius : float
        Nonnegative configured aura radius in world units.
    per_emitter_multiplier : float
        Finite nonnegative multiplier contributed by one emitter.
    stacking_rule : Literal['multiply_then_clamp']
        multiply_then_clamp: combine emitter factors, then apply the bound.
    clamp_kind : Literal['ceiling', 'floor']
        Whether clamp_value is the upper ceiling or lower floor.
    clamp_value : float
        Finite nonnegative bound for the combined multiplier.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG
    """Strict Pydantic wire settings: forbid extra fields, nonfinite numbers, and
    coercion."""

    aura_id: AuthorizedAuraIdV1
    """Canonical Mage damage or Warrior mitigation aura identifier."""
    radius: float
    """Nonnegative configured aura radius in world units."""
    per_emitter_multiplier: float
    """Finite nonnegative multiplier contributed by one emitter."""
    stacking_rule: Literal["multiply_then_clamp"]
    """multiply_then_clamp: combine emitter factors, then apply the bound."""
    clamp_kind: Literal["ceiling", "floor"]
    """Whether clamp_value is the upper ceiling or lower floor."""
    clamp_value: float
    """Finite nonnegative bound for the combined multiplier."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        The aura ID is known, numeric values are finite and nonnegative, and
        stacking uses multiply_then_clamp with a ceiling or floor.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
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
    """Store public class identity and configured mechanics.
    Require a canonical class, positive health/body radius, valid target modes,
    bounded regeneration, and unique ordered status channels and aura IDs.
    All constructor fields are required keyword arguments; instances are frozen.
    Attributes
    ----------
    class_id : int
        Canonical real-class ID from 1 through 5.
    class_name : str
        Canonical class name paired with class_id.
    maximum_health : float
        Positive maximum health in hit points.
    body_radius : float
        Positive body radius in world units.
    base_movement_speed : float
        Nonnegative movement distance per tick before current effects.
    observation_radius : float
        Nonnegative observation radius in world units.
    basic_target_mode : Literal['unavailable', 'ally', 'enemy']
        Whether Basic targets allies, enemies, or is unavailable.
    basic_interaction_radius : float
        Nonnegative Basic range in world units.
    basic_raw_damage : float
        Nonnegative configured Basic damage before modifiers, in hit points.
    basic_raw_healing : float
        Nonnegative configured Basic healing before modifiers, in hit points.
    ultimate_target_mode : Literal['unavailable', 'target_none', 'ally', 'enemy']
        Ultimate target relation, target_none, or unavailable.
    ultimate_interaction_radius : float
        Nonnegative Ultimate range in world units.
    ultimate_cooldown_steps : int
        Nonnegative configured Ultimate cooldown in ticks.
    ultimate_raw_damage : float
        Nonnegative configured Ultimate damage before modifiers, in hit points.
    ultimate_raw_healing : float
        Nonnegative configured Ultimate healing before modifiers, in hit points.
    out_of_combat_delay_steps : int
        Nonnegative configured delay before leaving combat, in ticks.
    out_of_combat_health_regeneration_fraction_per_step : float
        Fraction of maximum health restored per eligible tick, between 0 and 1.
    status_mechanics : tuple[AuthorizedClassStatusMechanicV1, ...]
        Unique status mechanics in channel order for this class.
    aura_mechanics : tuple[AuthorizedClassAuraMechanicV1, ...]
        Unique aura mechanics belonging to this class.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG
    """Strict Pydantic wire settings: forbid extra fields, nonfinite numbers, and
    coercion."""

    class_id: int
    """Canonical real-class ID from 1 through 5."""
    class_name: str
    """Canonical class name paired with class_id."""
    maximum_health: float
    """Positive maximum health in hit points."""
    body_radius: float
    """Positive body radius in world units."""
    base_movement_speed: float
    """Nonnegative movement distance per tick before current effects."""
    observation_radius: float
    """Nonnegative observation radius in world units."""
    basic_target_mode: Literal["unavailable", "ally", "enemy"]
    """Whether Basic targets allies, enemies, or is unavailable."""
    basic_interaction_radius: float
    """Nonnegative Basic range in world units."""
    basic_raw_damage: float
    """Nonnegative configured Basic damage before modifiers, in hit points."""
    basic_raw_healing: float
    """Nonnegative configured Basic healing before modifiers, in hit points."""
    ultimate_target_mode: Literal["unavailable", "target_none", "ally", "enemy"]
    """Ultimate target relation, target_none, or unavailable."""
    ultimate_interaction_radius: float
    """Nonnegative Ultimate range in world units."""
    ultimate_cooldown_steps: int
    """Nonnegative configured Ultimate cooldown in ticks."""
    ultimate_raw_damage: float
    """Nonnegative configured Ultimate damage before modifiers, in hit points."""
    ultimate_raw_healing: float
    """Nonnegative configured Ultimate healing before modifiers, in hit points."""
    out_of_combat_delay_steps: int
    """Nonnegative configured delay before leaving combat, in ticks."""
    out_of_combat_health_regeneration_fraction_per_step: float
    """Fraction of maximum health restored per eligible tick, between 0 and 1."""
    status_mechanics: tuple[AuthorizedClassStatusMechanicV1, ...]
    """Unique status mechanics in channel order for this class."""
    aura_mechanics: tuple[AuthorizedClassAuraMechanicV1, ...]
    """Unique aura mechanics belonging to this class."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Require a canonical class, positive health/body radius, valid target modes,
        bounded regeneration, and unique ordered status channels and aura IDs.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
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
class AuthorizedClassDocumentationProfileAvailableV1:
    """Certify that the authored canonical class guide applies.
    Require the available discriminator and the exact versioned profile ID. This
    record carries a prior certification result; it does not inspect a catalog
    itself.
    All constructor fields are required keyword arguments; instances are frozen.
    Attributes
    ----------
    availability_kind : Literal['available']
        Exact discriminator identifying the available or unavailable variant.
    profile_id : Literal['marl_battlegrounds.class_documentation.canonical_v1']
        Versioned identifier for the canonical authored class guide.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG
    """Strict Pydantic wire settings: forbid extra fields, nonfinite numbers, and
    coercion."""

    availability_kind: Literal["available"]
    """Exact discriminator identifying the available or unavailable variant."""
    profile_id: Literal["marl_battlegrounds.class_documentation.canonical_v1"]
    """Versioned identifier for the canonical authored class guide."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Require the available discriminator and the exact versioned profile ID. This
        record carries a prior certification result; it does not inspect a catalog
        itself.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
        if self.availability_kind != "available":
            raise ValueError("unknown available class-documentation discriminator.")
        if self.profile_id != AUTHORIZED_CLASS_DOCUMENTATION_PROFILE_ID_V1:
            raise ValueError("unknown class-documentation profile identity.")


@dataclass(frozen=True, slots=True, kw_only=True)
class AuthorizedClassDocumentationProfileUnavailableV1:
    """State that the canonical authored class guide is unavailable.
    Require the unavailable discriminator. Numeric class facts can still be
    shown without this categorical prose.
    All constructor fields are required keyword arguments; instances are frozen.
    Attributes
    ----------
    availability_kind : Literal['unavailable']
        Exact discriminator identifying the available or unavailable variant.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG
    """Strict Pydantic wire settings: forbid extra fields, nonfinite numbers, and
    coercion."""

    availability_kind: Literal["unavailable"]
    """Exact discriminator identifying the available or unavailable variant."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Require the unavailable discriminator. Numeric class facts can still be
        shown without this categorical prose.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
        if self.availability_kind != "unavailable":
            raise ValueError("unknown unavailable class-documentation discriminator.")


type AuthorizedClassDocumentationProfileV1 = Annotated[
    AuthorizedClassDocumentationProfileAvailableV1
    | AuthorizedClassDocumentationProfileUnavailableV1,
    Field(discriminator="availability_kind"),
]


@dataclass(frozen=True, slots=True, kw_only=True)
class AuthorizedClassMechanicsV2(AuthorizedClassMechanicsV1):
    """Add a catalog-checked guide profile to class mechanics.
    Validate inherited V1 mechanics, mechanics_version 2, and one exact
    available/unavailable profile variant. All inherited constructor fields
    remain required.
    All constructor fields are required keyword arguments; instances are frozen.
    Inherited fields are documented on AuthorizedClassMechanicsV1.
    Attributes
    ----------
    mechanics_version : Literal[2]
        Version discriminator; this row requires 2.
    documentation_profile : AuthorizedClassDocumentationProfileV1
        Certification or explicit denial of canonical class-guide prose.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    mechanics_version: Literal[2]
    """Version discriminator; this row requires 2."""
    documentation_profile: AuthorizedClassDocumentationProfileV1
    """Certification or explicit denial of canonical class-guide prose."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Validate inherited V1 mechanics, mechanics_version 2, and one exact
        available/unavailable profile variant. All inherited constructor fields
        remain required.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
        AuthorizedClassMechanicsV1.__post_init__(self)
        if self.mechanics_version != 2:
            raise ValueError("unknown authorized class mechanics version.")
        if type(self.documentation_profile) not in (
            AuthorizedClassDocumentationProfileAvailableV1,
            AuthorizedClassDocumentationProfileUnavailableV1,
        ):
            raise ValueError("documentation_profile must use an exact variant.")


type AuthorizedClassMechanics = AuthorizedClassMechanicsV1 | AuthorizedClassMechanicsV2


def _catalog_has_complete_class_documentation_profile_v1(
    catalog: StaticMechanicsCatalogV1,
) -> bool:
    """Check whether canonical authored class explanations match a validated catalog.
    Check class/status/aura meanings, units, and only the numeric relationships
    used by the prose. A tuned catalog may qualify if those claims remain true;
    this is not whole-catalog hash equality.
    Parameters
    ----------
    catalog : StaticMechanicsCatalogV1
        Exact static mechanics catalog to check.
    Returns
    -------
    bool
        Whether the complete categorical and qualitative guide profile is
        justified.
    """
    class_shape = tuple(
        (
            row.class_id,
            row.class_name,
            row.basic_target_mode,
            row.ultimate_target_mode,
        )
        for row in catalog.class_mechanics[1:]
    )
    status_shape = tuple(
        (
            row.status_channel_id,
            row.status_id,
            row.family,
            row.source_class_id,
            row.source_action_component,
            row.magnitude_kind,
            row.application_update,
            row.breaks_on_positive_damage,
        )
        for row in catalog.status_channels
    )
    aura_shape = tuple(
        (
            row.aura_id,
            row.emitter_class_id,
            row.beneficiary_relation,
            row.stacking_rule,
            row.clamp_kind,
        )
        for row in catalog.aura_mechanics
    )
    if (
        catalog.class_name_by_id
        != ("Neutral", "Mage", "Warrior", "Hunter", "Rogue", "Priest")
        or catalog.health_unit != "hit_points"
        or catalog.spatial_unit != "world_units"
        or catalog.duration_unit != "transition_ticks"
        or class_shape != _CANONICAL_CLASS_DOCUMENTATION_SHAPE_V1
        or status_shape != _CANONICAL_STATUS_DOCUMENTATION_SHAPE_V1
        or aura_shape != _CANONICAL_AURA_DOCUMENTATION_SHAPE_V1
    ):
        return False

    by_id = {row.class_id: row for row in catalog.class_mechanics[1:]}
    status_by_id = {row.status_id: row for row in catalog.status_channels}
    aura_by_id = {row.aura_id: row for row in catalog.aura_mechanics}
    damage_classes = tuple(by_id[class_id] for class_id in (1, 2, 3, 4))
    priest = by_id[5]
    maximum_body_radius = max(row.body_radius for row in by_id.values())
    melee_basic_radius = max(
        by_id[class_id].basic_interaction_radius for class_id in (2, 4)
    )
    ranged_basic_radius = min(
        by_id[class_id].basic_interaction_radius for class_id in (1, 3)
    )
    if not (
        all(row.maximum_health > 0.0 for row in by_id.values())
        and all(
            row.basic_interaction_radius >= row.body_radius + maximum_body_radius
            and row.observation_radius >= row.body_radius + maximum_body_radius
            for row in by_id.values()
        )
        and all(
            row.ultimate_interaction_radius >= row.body_radius + maximum_body_radius
            for row in (by_id[2], by_id[3], by_id[4], by_id[5])
        )
        and ranged_basic_radius > melee_basic_radius
        and all(row.basic_raw_damage > 0.0 for row in damage_classes)
        and all(row.basic_raw_healing == 0.0 for row in damage_classes)
        and by_id[1].ultimate_raw_damage == 0.0
        and by_id[1].ultimate_raw_healing == 0.0
        and all(row.ultimate_raw_damage > 0.0 for row in damage_classes[1:])
        and all(row.ultimate_raw_healing == 0.0 for row in damage_classes[1:])
        and priest.basic_raw_damage == 0.0
        and priest.ultimate_raw_damage == 0.0
        and priest.basic_raw_healing > 0.0
        and priest.ultimate_raw_healing > 0.0
        and by_id[4].out_of_combat_health_regeneration_fraction_per_step > 0.0
        and all(
            row.duration_steps > 0
            and (
                (row.magnitude_kind == "none" and row.magnitude is None)
                or (row.magnitude_kind != "none" and row.magnitude is not None)
            )
            for row in status_by_id.values()
        )
        and all(
            row.magnitude is not None and 0.0 < row.magnitude < 1.0
            for row in status_by_id.values()
            if row.magnitude_kind in ("movement_multiplier", "healing_multiplier")
        )
        and status_by_id["mage_burst_damage_amplification"].magnitude is not None
        and status_by_id["mage_burst_damage_amplification"].magnitude > 1.0
        and status_by_id["priest_blessing_of_freedom_movement_floor"].magnitude
        is not None
        and catalog.global_slow_floor
        < status_by_id["priest_blessing_of_freedom_movement_floor"].magnitude
        <= 1.0
        and aura_by_id["mage_damage_amplification"].per_emitter_multiplier > 1.0
        and aura_by_id["mage_damage_amplification"].clamp_value
        >= aura_by_id["mage_damage_amplification"].per_emitter_multiplier
        and 0.0 < aura_by_id["warrior_damage_mitigation"].per_emitter_multiplier < 1.0
        and 0.0
        < aura_by_id["warrior_damage_mitigation"].clamp_value
        <= aura_by_id["warrior_damage_mitigation"].per_emitter_multiplier
    ):
        return False

    positive_basic_damage = sorted(
        (row.basic_raw_damage, row.class_id)
        for row in by_id.values()
        if row.basic_raw_damage > 0.0
    )
    stun_duration_by_class = {
        row.source_class_id: row.duration_steps
        for row in catalog.status_channels
        if row.family == "stun"
    }
    return (
        max(by_id.values(), key=lambda row: row.basic_raw_damage).class_id == 1
        and sum(
            row.basic_raw_damage == by_id[1].basic_raw_damage for row in by_id.values()
        )
        == 1
        and min(by_id.values(), key=lambda row: row.maximum_health).class_id == 1
        and sum(row.maximum_health == by_id[1].maximum_health for row in by_id.values())
        == 1
        and max(by_id.values(), key=lambda row: row.maximum_health).class_id == 2
        and sum(row.maximum_health == by_id[2].maximum_health for row in by_id.values())
        == 1
        and by_id[4].maximum_health < by_id[2].maximum_health
        and positive_basic_damage[0][1] == 3
        and sum(
            row.basic_raw_damage == by_id[3].basic_raw_damage for row in damage_classes
        )
        == 1
        and positive_basic_damage[1][1] == 2
        and sum(
            row.basic_raw_damage == by_id[2].basic_raw_damage for row in damage_classes
        )
        == 1
        and max(by_id.values(), key=lambda row: row.base_movement_speed).class_id == 4
        and sum(
            row.base_movement_speed == by_id[4].base_movement_speed
            for row in by_id.values()
        )
        == 1
        and stun_duration_by_class[3] == max(stun_duration_by_class.values())
        and sum(
            duration == stun_duration_by_class[3]
            for duration in stun_duration_by_class.values()
        )
        == 1
        and by_id[5].basic_raw_healing > 0.0
        and by_id[5].ultimate_raw_healing > 0.0
    )


def authorized_class_documentation_profile_v1(
    catalog: StaticMechanicsCatalogV1,
) -> AuthorizedClassDocumentationProfileV1:
    """Certify or deny the canonical class-guide profile for a catalog.
    Revalidate a Python dump before checking all prose-dependent identities and
    tuning relationships. Do not infer that a changed numeric catalog always
    invalidates the guide.
    Parameters
    ----------
    catalog : StaticMechanicsCatalogV1
        Exact static mechanics catalog to check.
    Returns
    -------
    AuthorizedClassDocumentationProfileV1
        Frozen available profile or explicit unavailable variant.
    Raises
    ------
    TypeError, ValueError
        The root is not the exact catalog type, or strict catalog validation
        fails.
    """
    if type(catalog) is not StaticMechanicsCatalogV1:
        raise TypeError("catalog must be the exact StaticMechanicsCatalogV1 root.")
    validated = StaticMechanicsCatalogV1.model_validate(
        catalog.model_dump(mode="python")
    )
    if _catalog_has_complete_class_documentation_profile_v1(validated):
        return AuthorizedClassDocumentationProfileAvailableV1(
            availability_kind="available",
            profile_id=AUTHORIZED_CLASS_DOCUMENTATION_PROFILE_ID_V1,
        )
    return AuthorizedClassDocumentationProfileUnavailableV1(
        availability_kind="unavailable"
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class AuthorizedRespawnWaveV1:
    """Store the current respawn countdown for one team.
    Team index and ID agree; the period is positive and the countdown is
    nonnegative and smaller than the period.
    All constructor fields are required keyword arguments; instances are frozen.
    Attributes
    ----------
    team_index : int
        Zero-based team index: 0 for Team A, 1 for Team B.
    team_id : int
        Team ID: 1 for Team A, 2 for Team B.
    period_steps : int
        Positive respawn-wave period in transition ticks.
    countdown_steps : int
        Ticks until the wave, from 0 up to but excluding period_steps.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG
    """Strict Pydantic wire settings: forbid extra fields, nonfinite numbers, and
    coercion."""

    team_index: int
    """Zero-based team index: 0 for Team A, 1 for Team B."""
    team_id: int
    """Team ID: 1 for Team A, 2 for Team B."""
    period_steps: int
    """Positive respawn-wave period in transition ticks."""
    countdown_steps: int
    """Ticks until the wave, from 0 up to but excluding period_steps."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Team index and ID agree; the period is positive and the countdown is
        nonnegative and smaller than the period.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
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
    """Store recorded Spawn Shield duration and movement speed.
    The duration is a nonnegative Python int and speed is a positive finite
    float.
    All constructor fields are required keyword arguments; instances are frozen.
    Attributes
    ----------
    availability_kind : Literal['available']
        Exact discriminator identifying the available or unavailable variant.
    configured_duration_steps : int
        Configured effect duration in transition ticks.
    movement_speed : float
        Positive Spawn Shield movement distance per tick.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG
    """Strict Pydantic wire settings: forbid extra fields, nonfinite numbers, and
    coercion."""

    availability_kind: Literal["available"]
    """Exact discriminator identifying the available or unavailable variant."""
    configured_duration_steps: int
    """Configured effect duration in transition ticks."""
    movement_speed: float
    """Positive Spawn Shield movement distance per tick."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        The duration is a nonnegative Python int and speed is a positive finite
        float.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
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
class AuthorizedSpawnShieldMechanicsAvailableV2:
    """Store configured Spawn Shield facts and their canonical meanings.
    Require the V2 discriminator, valid duration/speed, and all seven exact
    semantic literals. These facts support the authored Spawn Shield
    explanation.
    All constructor fields are required keyword arguments; instances are frozen.
    Attributes
    ----------
    availability_kind : Literal['available_v2']
        Exact discriminator identifying the available or unavailable variant.
    configured_duration_steps : int
        Configured effect duration in transition ticks.
    movement_speed : float
        Positive Spawn Shield movement distance per tick.
    protection_effect : Literal['invulnerable']
        invulnerable: protected agents take no damage.
    visibility_effect : Literal['concealed_from_opponents']
        concealed_from_opponents: opponents cannot observe the shielded agent.
    targetability_effect : Literal['untargetable']
        untargetable: combat targeting cannot select the shielded agent.
    action_scope : Literal['movement_only']
        movement_only: only movement is available while protected.
    aura_effect : Literal['excluded_as_emitter_and_beneficiary']
        excluded_as_emitter_and_beneficiary: shielded agents neither give nor
        receive aura effects.
    agent_collision_effect : Literal['phased_until_expiring_endpoint_rejoin']
        phased_until_expiring_endpoint_rejoin: body collision rejoins at shield
        expiry.
    ordinary_application_mechanism : Literal['end_of_transition_respawn_lifecycle']
        end_of_transition_respawn_lifecycle: ordinary application occurs at
        respawn.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG
    """Strict Pydantic wire settings: forbid extra fields, nonfinite numbers, and
    coercion."""

    availability_kind: Literal["available_v2"]
    """Exact discriminator identifying the available or unavailable variant."""
    configured_duration_steps: int
    """Configured effect duration in transition ticks."""
    movement_speed: float
    """Positive Spawn Shield movement distance per tick."""
    protection_effect: Literal["invulnerable"]
    """invulnerable: protected agents take no damage."""
    visibility_effect: Literal["concealed_from_opponents"]
    """concealed_from_opponents: opponents cannot observe the shielded agent."""
    targetability_effect: Literal["untargetable"]
    """untargetable: combat targeting cannot select the shielded agent."""
    action_scope: Literal["movement_only"]
    """movement_only: only movement is available while protected."""
    aura_effect: Literal["excluded_as_emitter_and_beneficiary"]
    """excluded_as_emitter_and_beneficiary: shielded agents neither give nor
    receive aura effects."""
    agent_collision_effect: Literal["phased_until_expiring_endpoint_rejoin"]
    """phased_until_expiring_endpoint_rejoin: body collision rejoins at shield
    expiry."""
    ordinary_application_mechanism: Literal["end_of_transition_respawn_lifecycle"]
    """end_of_transition_respawn_lifecycle: ordinary application occurs at respawn."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Require the V2 discriminator, valid duration/speed, and all seven exact
        semantic literals. These facts support the authored Spawn Shield
        explanation.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
        if self.availability_kind != "available_v2":
            raise ValueError("unknown V2 spawn-shield discriminator.")
        _require_python_int(
            self.configured_duration_steps,
            name="configured_duration_steps",
        )
        _require_finite(self.movement_speed, name="movement_speed", minimum=0.0)
        if self.movement_speed <= 0.0:
            raise ValueError("spawn-shield movement speed must be positive.")
        expected = (
            "invulnerable",
            "concealed_from_opponents",
            "untargetable",
            "movement_only",
            "excluded_as_emitter_and_beneficiary",
            "phased_until_expiring_endpoint_rejoin",
            "end_of_transition_respawn_lifecycle",
        )
        actual = (
            self.protection_effect,
            self.visibility_effect,
            self.targetability_effect,
            self.action_scope,
            self.aura_effect,
            self.agent_collision_effect,
            self.ordinary_application_mechanism,
        )
        if actual != expected:
            raise ValueError("spawn-shield documentation semantics are not canonical.")


@dataclass(frozen=True, slots=True, kw_only=True)
class AuthorizedSpawnShieldMechanicsUnavailableV1:
    """State that recorded Spawn Shield configuration is unavailable.
    Require the unavailable discriminator; no duration or speed is invented.
    All constructor fields are required keyword arguments; instances are frozen.
    Attributes
    ----------
    availability_kind : Literal['unavailable']
        Exact discriminator identifying the available or unavailable variant.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG
    """Strict Pydantic wire settings: forbid extra fields, nonfinite numbers, and
    coercion."""

    availability_kind: Literal["unavailable"]
    """Exact discriminator identifying the available or unavailable variant."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Require the unavailable discriminator; no duration or speed is invented.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
        if self.availability_kind != "unavailable":
            raise ValueError("unknown unavailable spawn-shield discriminator.")


type AuthorizedSpawnShieldMechanicsV1 = Annotated[
    AuthorizedSpawnShieldMechanicsAvailableV1
    | AuthorizedSpawnShieldMechanicsUnavailableV1,
    Field(discriminator="availability_kind"),
]

type AuthorizedSpawnShieldMechanics = Annotated[
    AuthorizedSpawnShieldMechanicsAvailableV1
    | AuthorizedSpawnShieldMechanicsAvailableV2
    | AuthorizedSpawnShieldMechanicsUnavailableV1,
    Field(discriminator="availability_kind"),
]


@dataclass(frozen=True, slots=True, kw_only=True)
class AuthorizedStatusSourceV1:
    """Identify one authorized direct source of a durable status.
    Both nonempty identifiers describe the same source; the enclosing scene
    checks that join. No raw slot or event ID is stored.
    All constructor fields are required keyword arguments; instances are frozen.
    Attributes
    ----------
    source_presentation_key : str
        Nonempty opaque key for an authorized source agent.
    source_public_agent_id : str
        Nonempty public ID for the same source agent.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG
    """Strict Pydantic wire settings: forbid extra fields, nonfinite numbers, and
    coercion."""

    source_presentation_key: str
    """Nonempty opaque key for an authorized source agent."""
    source_public_agent_id: str
    """Nonempty public ID for the same source agent."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Both nonempty identifiers describe the same source; the enclosing scene
        checks that join. No raw slot or event ID is stored.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
        _require_text(self.source_presentation_key, name="source_presentation_key")
        _require_text(self.source_public_agent_id, name="source_public_agent_id")


@dataclass(frozen=True, slots=True, kw_only=True)
class AuthorizedStatusV1:
    """Store a durable status and the sources allowed in this view.
    Validate catalog channel/ID, positive bounded duration, source class domain,
    magnitude presence, and unique direct source keys. Scene validation checks
    catalog and source-agent joins.
    All constructor fields are required keyword arguments; instances are frozen.
    Attributes
    ----------
    status_channel : int
        Zero-based channel on the fixed nine-status axis.
    status_id : str
        Canonical status identifier matching status_channel.
    family : str
        Allowed values: 'slow', 'stun', 'anti_heal', 'damage_amplification',
        'movement_floor'.
        Effect family named by the status catalog.
    configured_duration_steps : int
        Positive configured status duration in ticks.
    remaining_duration : int
        Positive remaining duration, no greater than configured duration, in
        ticks.
    source_class_id : int
        Canonical source-class ID from 1 through 5.
    source_class_name : str
        Public source-class name; scene validation joins it to the class ID.
    source_action_component : Literal['basic', 'ultimate']
        Basic or Ultimate action component that creates the effect.
    magnitude_kind : str
        Allowed values: 'movement_multiplier', 'none', 'healing_multiplier',
        'damage_multiplier', 'movement_floor'.
        Meaning of magnitude; none requires magnitude to be None.
    magnitude : float | None
        Finite configured effect value, or None for magnitude kind none.
    breaks_on_positive_damage : bool
        Whether positive damage removes this status; an exact bool.
    direct_sources : tuple[AuthorizedStatusSourceV1, ...]
        Unique authorized source identities; no internal slot or event IDs.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG
    """Strict Pydantic wire settings: forbid extra fields, nonfinite numbers, and
    coercion."""

    status_channel: int
    """Zero-based channel on the fixed nine-status axis."""
    status_id: str
    """Canonical status identifier matching status_channel."""
    family: Literal[
        "slow",
        "stun",
        "anti_heal",
        "damage_amplification",
        "movement_floor",
    ]
    """Allowed values: 'slow', 'stun', 'anti_heal', 'damage_amplification',
    'movement_floor'. Effect family named by the status catalog."""
    configured_duration_steps: int
    """Positive configured status duration in ticks."""
    remaining_duration: int
    """Positive remaining duration, no greater than configured duration, in ticks."""
    source_class_id: int
    """Canonical source-class ID from 1 through 5."""
    source_class_name: str
    """Public source-class name; scene validation joins it to the class ID."""
    source_action_component: Literal["basic", "ultimate"]
    """Basic or Ultimate action component that creates the effect."""
    magnitude_kind: Literal[
        "movement_multiplier",
        "none",
        "healing_multiplier",
        "damage_multiplier",
        "movement_floor",
    ]
    """Allowed values: 'movement_multiplier', 'none', 'healing_multiplier',
    'damage_multiplier', 'movement_floor'. Meaning of magnitude; none requires
    magnitude to be None."""
    magnitude: float | None
    """Finite configured effect value, or None for magnitude kind none."""
    breaks_on_positive_damage: bool
    """Whether positive damage removes this status; an exact bool."""
    direct_sources: tuple[AuthorizedStatusSourceV1, ...]
    """Unique authorized source identities; no internal slot or event IDs."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Validate catalog channel/ID, positive bounded duration, source class domain,
        magnitude presence, and unique direct source keys. Scene validation checks
        catalog and source-agent joins.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
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
    """Store one authorized agent body and its durable public facts.
    Require canonical identity, finite geometry/health/speeds, bounded
    countdowns and regeneration, and unique status/aura rows. This model does
    not decide whether the caller may see the agent.
    All constructor fields are required keyword arguments; instances are frozen.
    Attributes
    ----------
    presentation_key : str
        Nonempty opaque key scoped to the presentation authority.
    public_agent_id : str
        Nonempty public identity of the same agent.
    relation : AuthorizedRelationV1
        oracle, self, ally, or opponent, as allowed by the enclosing view.
    team_id : int
        Team ID: 1 for Team A, 2 for Team B.
    class_id : int
        Canonical real-class ID from 1 through 5.
    class_name : str
        Canonical class name paired with class_id.
    position : Point2D
        Agent or pad center (x, y), in world units.
    radius : float
        Positive agent body radius in world units.
    life_state : AgentLifeStateV1
        alive or corpse; inactive roster slots have no body row.
    current_health : float
        Nonnegative health in hit points, at most maximum_health.
    maximum_health : float
        Positive maximum health in hit points.
    base_movement_speed : float
        Nonnegative movement distance per tick before current effects.
    effective_movement_speed : float
        Nonnegative movement distance per tick after current movement effects.
    observation_radius : float
        Nonnegative observation radius in world units.
    basic_interaction_radius : float
        Nonnegative Basic range in world units.
    ultimate_interaction_radius : float
        Nonnegative Ultimate range in world units.
    ultimate_cooldown_remaining : int
        Nonnegative remaining Ultimate cooldown in ticks.
    spawn_shield_remaining : int
        Nonnegative remaining Spawn Shield duration in ticks.
    steps_until_out_of_combat : int
        Nonnegative combat countdown, at most its configured delay, in ticks.
    out_of_combat_delay_steps : int
        Nonnegative configured delay before leaving combat, in ticks.
    out_of_combat_health_regeneration_fraction_per_step : float
        Fraction of maximum health restored per eligible tick, between 0 and 1.
    statuses : tuple[AuthorizedStatusV1, ...]
        Durable status rows with unique status channels.
    aura_modifiers : tuple[AuthorizedAuraModifierV1, ...]
        Unique recipient aura multipliers; omit neutral factors.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG
    """Strict Pydantic wire settings: forbid extra fields, nonfinite numbers, and
    coercion."""

    presentation_key: str
    """Nonempty opaque key scoped to the presentation authority."""
    public_agent_id: str
    """Nonempty public identity of the same agent."""
    relation: AuthorizedRelationV1
    """oracle, self, ally, or opponent, as allowed by the enclosing view."""
    team_id: int
    """Team ID: 1 for Team A, 2 for Team B."""
    class_id: int
    """Canonical real-class ID from 1 through 5."""
    class_name: str
    """Canonical class name paired with class_id."""
    position: Point2D
    """Agent or pad center (x, y), in world units."""
    radius: float
    """Positive agent body radius in world units."""
    life_state: AgentLifeStateV1
    """alive or corpse; inactive roster slots have no body row."""
    current_health: float
    """Nonnegative health in hit points, at most maximum_health."""
    maximum_health: float
    """Positive maximum health in hit points."""
    base_movement_speed: float
    """Nonnegative movement distance per tick before current effects."""
    effective_movement_speed: float
    """Nonnegative movement distance per tick after current movement effects."""
    observation_radius: float
    """Nonnegative observation radius in world units."""
    basic_interaction_radius: float
    """Nonnegative Basic range in world units."""
    ultimate_interaction_radius: float
    """Nonnegative Ultimate range in world units."""
    ultimate_cooldown_remaining: int
    """Nonnegative remaining Ultimate cooldown in ticks."""
    spawn_shield_remaining: int
    """Nonnegative remaining Spawn Shield duration in ticks."""
    steps_until_out_of_combat: int
    """Nonnegative combat countdown, at most its configured delay, in ticks."""
    out_of_combat_delay_steps: int
    """Nonnegative configured delay before leaving combat, in ticks."""
    out_of_combat_health_regeneration_fraction_per_step: float
    """Fraction of maximum health restored per eligible tick, between 0 and 1."""
    statuses: tuple[AuthorizedStatusV1, ...]
    """Durable status rows with unique status channels."""
    aura_modifiers: tuple[AuthorizedAuraModifierV1, ...]
    """Unique recipient aura multipliers; omit neutral factors."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Require canonical identity, finite geometry/health/speeds, bounded
        countdowns and regeneration, and unique status/aura rows. This model does
        not decide whether the caller may see the agent.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
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
    """Store an authorized aura emitter and its field mechanics.
    Require a known aura, finite positive radius, nonnegative multipliers, and
    exact lifecycle/stacking values. The enclosing scene checks source identity
    and catalog equality.
    All constructor fields are required keyword arguments; instances are frozen.
    Attributes
    ----------
    aura_id : AuthorizedAuraIdV1
        Canonical Mage damage or Warrior mitigation aura identifier.
    source_presentation_key : str
        Nonempty opaque key for an authorized source agent.
    source_public_agent_id : str
        Nonempty public ID for the same source agent.
    source_class_id : int
        Canonical source-class ID from 1 through 5.
    source_class_name : str
        Public source-class name; scene validation joins it to the class ID.
    source_alive : bool
        Whether the authorized emitter is alive; an exact bool.
    center : Point2D
        Center coordinates (x, y), in world units.
    radius : float
        Positive emitter aura radius in world units.
    beneficiary_relation : Literal['same_team']
        same_team: this aura affects agents on the emitter team.
    per_emitter_multiplier : float
        Finite nonnegative multiplier contributed by one emitter.
    stacking_rule : Literal['multiply_then_clamp']
        multiply_then_clamp: combine emitter factors, then apply the bound.
    clamp_kind : Literal['ceiling', 'floor']
        Whether clamp_value is the upper ceiling or lower floor.
    clamp_value : float
        Finite nonnegative bound for the combined multiplier.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG
    """Strict Pydantic wire settings: forbid extra fields, nonfinite numbers, and
    coercion."""

    aura_id: AuthorizedAuraIdV1
    """Canonical Mage damage or Warrior mitigation aura identifier."""
    source_presentation_key: str
    """Nonempty opaque key for an authorized source agent."""
    source_public_agent_id: str
    """Nonempty public ID for the same source agent."""
    source_class_id: int
    """Canonical source-class ID from 1 through 5."""
    source_class_name: str
    """Public source-class name; scene validation joins it to the class ID."""
    source_alive: bool
    """Whether the authorized emitter is alive; an exact bool."""
    center: Point2D
    """Center coordinates (x, y), in world units."""
    radius: float
    """Positive emitter aura radius in world units."""
    beneficiary_relation: Literal["same_team"]
    """same_team: this aura affects agents on the emitter team."""
    per_emitter_multiplier: float
    """Finite nonnegative multiplier contributed by one emitter."""
    stacking_rule: Literal["multiply_then_clamp"]
    """multiply_then_clamp: combine emitter factors, then apply the bound."""
    clamp_kind: Literal["ceiling", "floor"]
    """Whether clamp_value is the upper ceiling or lower floor."""
    clamp_value: float
    """Finite nonnegative bound for the combined multiplier."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Require a known aura, finite positive radius, nonnegative multipliers, and
        exact lifecycle/stacking values. The enclosing scene checks source identity
        and catalog equality.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
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
    """Store a lifecycle pad with an optional authorized body assignment.
    Assignee key and public ID appear together. Team/slot IDs are bounded;
    inactive pads cannot be alive, assigned, or shielded.
    All constructor fields are required keyword arguments; instances are frozen.
    Attributes
    ----------
    team_id : int
        Team ID: 1 for Team A, 2 for Team B.
    team_local_slot : int
        Zero-based roster slot within the team, from 0 through 4.
    assigned_presentation_key : str | None
        Authorized body key, or None when no visible body is assigned.
    assigned_public_agent_id : str | None
        Matching public ID, present exactly when the assigned key is present.
    position : Point2D
        Agent or pad center (x, y), in world units.
    configured_active : bool
        Whether this roster slot participates in the episode.
    currently_alive : bool
        Whether its occupant is alive; inactive slots cannot be alive.
    spawn_shield_remaining : int
        Nonnegative remaining Spawn Shield duration in ticks.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG
    """Strict Pydantic wire settings: forbid extra fields, nonfinite numbers, and
    coercion."""

    team_id: int
    """Team ID: 1 for Team A, 2 for Team B."""
    team_local_slot: int
    """Zero-based roster slot within the team, from 0 through 4."""
    assigned_presentation_key: str | None
    """Authorized body key, or None when no visible body is assigned."""
    assigned_public_agent_id: str | None
    """Matching public ID, present exactly when the assigned key is present."""
    position: Point2D
    """Agent or pad center (x, y), in world units."""
    configured_active: bool
    """Whether this roster slot participates in the episode."""
    currently_alive: bool
    """Whether its occupant is alive; inactive slots cannot be alive."""
    spawn_shield_remaining: int
    """Nonnegative remaining Spawn Shield duration in ticks."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Assignee key and public ID appear together. Team/slot IDs are bounded;
        inactive pads cannot be alive, assigned, or shielded.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
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
    """Join authorized bodies, mechanics, and lifecycle facts into one scene.
    Check exact row types, unique identities, represented class/status/aura
    axes, source/assignee joins, shield bounds, and ordered team waves. Oracle
    static values match catalog values exactly or their exact float32 encoding;
    non-Oracle per-slot static overrides remain allowed.
    All constructor fields are required keyword arguments; instances are frozen.
    Attributes
    ----------
    schema_version : Literal[1]
        Wire version discriminator; this model requires 1.
    map : AuthorizedMap
        Exact AuthorizedMapV1 (no Red Zone record) or AuthorizedMapV2 (with the
        recorded Red Zone strips): finite map bounds and ordered static
        obstacles.
    agents : tuple[AuthorizedAgentV1, ...]
        Ordered authorized durable agents with unique keys and public IDs.
    aura_fields : tuple[AuthorizedAuraFieldV1, ...]
        Emitter fields joined to their authorized source agents and catalog
        mechanics.
    class_mechanics : tuple[AuthorizedClassMechanics, ...]
        Mechanics for exactly the represented classes, in ascending class-ID
        order.
    spawn_shield_mechanics : AuthorizedSpawnShieldMechanics
        Available configured shield facts or an explicit unavailable variant.
    spawn_pads : tuple[AuthorizedSpawnPadV1, ...]
        Unique lifecycle pads ordered by team ID and team-local slot.
    respawn_waves : tuple[AuthorizedRespawnWaveV1, ...]
        Team A then Team B lifecycle countdowns.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG
    """Strict Pydantic wire settings: forbid extra fields, nonfinite numbers, and
    coercion."""

    schema_version: Literal[1]
    """Wire version discriminator; this model requires 1."""
    map: AuthorizedMap
    """Exact AuthorizedMapV1 or AuthorizedMapV2 (with Red Zone strips)."""
    agents: tuple[AuthorizedAgentV1, ...]
    """Ordered authorized durable agents with unique keys and public IDs."""
    aura_fields: tuple[AuthorizedAuraFieldV1, ...]
    """Emitter fields joined to their authorized source agents and catalog
    mechanics."""
    class_mechanics: tuple[AuthorizedClassMechanics, ...]
    """Mechanics for exactly the represented classes, in ascending class-ID order."""
    spawn_shield_mechanics: AuthorizedSpawnShieldMechanics
    """Available configured shield facts or an explicit unavailable variant."""
    spawn_pads: tuple[AuthorizedSpawnPadV1, ...]
    """Unique lifecycle pads ordered by team ID and team-local slot."""
    respawn_waves: tuple[AuthorizedRespawnWaveV1, ...]
    """Team A then Team B lifecycle countdowns."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Check exact row types, unique identities, represented class/status/aura
        axes, source/assignee joins, shield bounds, and ordered team waves. Oracle
        static values match catalog values exactly or their exact float32 encoding;
        non-Oracle per-slot static overrides remain allowed.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
        if self.schema_version != AUTHORIZED_PRESENTATION_SCHEMA_VERSION:
            raise ValueError("unknown authorized battlefield schema version.")
        if type(self.map) not in (AuthorizedMapV1, AuthorizedMapV2):
            raise ValueError("map must be the exact authorized map root.")
        _require_exact_tuple(self.agents, name="agents", item_type=AuthorizedAgentV1)
        _require_exact_tuple(
            self.aura_fields,
            name="aura_fields",
            item_type=AuthorizedAuraFieldV1,
        )
        if type(self.class_mechanics) is not tuple or any(
            type(row) not in (AuthorizedClassMechanicsV1, AuthorizedClassMechanicsV2)
            for row in self.class_mechanics
        ):
            raise ValueError("class_mechanics must contain exact V1 or V2 rows.")
        mechanics_versions = {type(row) for row in self.class_mechanics}
        if len(mechanics_versions) > 1:
            raise ValueError("class_mechanics cannot mix V1 and V2 rows.")
        if mechanics_versions == {AuthorizedClassMechanicsV2}:
            profiles = {
                cast(AuthorizedClassMechanicsV2, row).documentation_profile
                for row in self.class_mechanics
            }
            if len(profiles) > 1:
                raise ValueError(
                    "V2 class mechanics must share one documentation profile."
                )
        if type(self.spawn_shield_mechanics) not in (
            AuthorizedSpawnShieldMechanicsAvailableV1,
            AuthorizedSpawnShieldMechanicsAvailableV2,
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
        if type(self.spawn_shield_mechanics) in (
            AuthorizedSpawnShieldMechanicsAvailableV1,
            AuthorizedSpawnShieldMechanicsAvailableV2,
        ):
            available_shield = cast(
                AuthorizedSpawnShieldMechanicsAvailableV1
                | AuthorizedSpawnShieldMechanicsAvailableV2,
                self.spawn_shield_mechanics,
            )
            configured_duration = available_shield.configured_duration_steps
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
    """Identify an active incoming-event agent without exposing its internal slot.
    Require the authorized_agent discriminator and nonempty key/public ID.
    All constructor fields are required keyword arguments; instances are frozen.
    Attributes
    ----------
    identity_kind : Literal['authorized_agent']
        Exact discriminator for a scene agent or inactive feed-only identity.
    presentation_key : str
        Nonempty opaque key scoped to the presentation authority.
    public_agent_id : str
        Nonempty public identity of the same agent.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG
    """Strict Pydantic wire settings: forbid extra fields, nonfinite numbers, and
    coercion."""

    identity_kind: Literal["authorized_agent"]
    """Exact discriminator for a scene agent or inactive feed-only identity."""
    presentation_key: str
    """Nonempty opaque key scoped to the presentation authority."""
    public_agent_id: str
    """Nonempty public identity of the same agent."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Require the authorized_agent discriminator and nonempty key/public ID.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
        if self.identity_kind != "authorized_agent":
            raise ValueError("unknown authorized incoming agent identity.")
        _require_text(self.presentation_key, name="presentation_key")
        _require_text(self.public_agent_id, name="public_agent_id")


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayIncomingFeedOnlyAgentIdentityV1:
    """Identify an inactive rejected actor only in the event feed.
    Require inactive_feed_only and a public ID. There is no presentation key or
    invented scene body.
    All constructor fields are required keyword arguments; instances are frozen.
    Attributes
    ----------
    identity_kind : Literal['inactive_feed_only']
        Exact discriminator for a scene agent or inactive feed-only identity.
    public_agent_id : str
        Nonempty public identity of the same agent.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG
    """Strict Pydantic wire settings: forbid extra fields, nonfinite numbers, and
    coercion."""

    identity_kind: Literal["inactive_feed_only"]
    """Exact discriminator for a scene agent or inactive feed-only identity."""
    public_agent_id: str
    """Nonempty public identity of the same agent."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Require inactive_feed_only and a public ID. There is no presentation key or
        invented scene body.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
        if self.identity_kind != "inactive_feed_only":
            raise ValueError("unknown feed-only incoming agent identity.")
        _require_text(self.public_agent_id, name="public_agent_id")


type ReplayIncomingAgentIdentityV1 = Annotated[
    ReplayIncomingAuthorizedAgentIdentityV1 | ReplayIncomingFeedOnlyAgentIdentityV1,
    Field(discriminator="identity_kind"),
]


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayIncomingAgentAnchorV1:
    """Store an authorized agent position at one scientific phase.
    The phase is transition_start, post_charge, or successor; identity strings
    and both float coordinates are valid.
    All constructor fields are required keyword arguments; instances are frozen.
    Attributes
    ----------
    phase : ReplayIncomingAnchorPhaseV1
        Scientific endpoint named by the row; it is not an animation timing
        choice.
    presentation_key : str
        Nonempty opaque key scoped to the presentation authority.
    public_agent_id : str
        Nonempty public identity of the same agent.
    position : Point2D
        Agent or pad center (x, y), in world units.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG
    """Strict Pydantic wire settings: forbid extra fields, nonfinite numbers, and
    coercion."""

    phase: ReplayIncomingAnchorPhaseV1
    """Scientific endpoint named by the row; it is not an animation timing choice."""
    presentation_key: str
    """Nonempty opaque key scoped to the presentation authority."""
    public_agent_id: str
    """Nonempty public identity of the same agent."""
    position: Point2D
    """Agent or pad center (x, y), in world units."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        The phase is transition_start, post_charge, or successor; identity strings
        and both float coordinates are valid.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
        if self.phase not in ("transition_start", "post_charge", "successor"):
            raise ValueError("unknown replay incoming anchor phase.")
        _require_text(self.presentation_key, name="presentation_key")
        _require_text(self.public_agent_id, name="public_agent_id")
        _require_point(self.position, name="position")


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayIncomingTeamAnchorV1:
    """Store a non-spatial team cue at the successor phase.
    Require successor and a matching zero-based index and one-based team ID.
    All constructor fields are required keyword arguments; instances are frozen.
    Attributes
    ----------
    phase : Literal['successor']
        Scientific endpoint named by the row; it is not an animation timing
        choice.
    team_index : int
        Zero-based team index: 0 for Team A, 1 for Team B.
    team_id : int
        Team ID: 1 for Team A, 2 for Team B.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG
    """Strict Pydantic wire settings: forbid extra fields, nonfinite numbers, and
    coercion."""

    phase: Literal["successor"]
    """Scientific endpoint named by the row; it is not an animation timing choice."""
    team_index: int
    """Zero-based team index: 0 for Team A, 1 for Team B."""
    team_id: int
    """Team ID: 1 for Team A, 2 for Team B."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Require successor and a matching zero-based index and one-based team ID.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
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
    """Store one agent at start, after Charge, and at the successor.
    All three exact anchors retain the same public and presentation identities
    and their named phases.
    All constructor fields are required keyword arguments; instances are frozen.
    Attributes
    ----------
    agent_presentation_key : str
        Opaque identity shared by this agent trajectory and its anchors.
    agent_public_agent_id : str
        Public identity shared by this agent trajectory and its anchors.
    transition_start : ReplayIncomingAgentAnchorV1
        Authorized position before the incoming transition.
    post_charge : ReplayIncomingAgentAnchorV1
        Authorized position after Charge and before ordinary movement.
    successor : ReplayIncomingAgentAnchorV1
        Authorized position after the incoming transition.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG
    """Strict Pydantic wire settings: forbid extra fields, nonfinite numbers, and
    coercion."""

    agent_presentation_key: str
    """Opaque identity shared by this agent trajectory and its anchors."""
    agent_public_agent_id: str
    """Public identity shared by this agent trajectory and its anchors."""
    transition_start: ReplayIncomingAgentAnchorV1
    """Authorized position before the incoming transition."""
    post_charge: ReplayIncomingAgentAnchorV1
    """Authorized position after Charge and before ordinary movement."""
    successor: ReplayIncomingAgentAnchorV1
    """Authorized position after the incoming transition."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        All three exact anchors retain the same public and presentation identities
        and their named phases.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
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
    """Share identity and ordering fields across incoming atomic events.
    Concrete subclasses call _validate_base with their exact kind and scientific
    phase rank. This base is not a complete event variant.
    All constructor fields are required keyword arguments; instances are frozen.
    Attributes
    ----------
    event_id : str
        Nonempty event identity within the enclosing transition inventory.
    ordinal : int
        Zero-based event position; the enclosing inventory requires dense order.
    phase_rank : int
        Canonical scientific phase rank used to order events.
    """

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG
    """Strict Pydantic wire settings: forbid extra fields, nonfinite numbers, and
    coercion."""

    event_id: str
    """Nonempty event identity within the enclosing transition inventory."""
    ordinal: int
    """Zero-based event position; the enclosing inventory requires dense order."""
    phase_rank: int
    """Canonical scientific phase rank used to order events."""

    def _validate_base(
        self,
        *,
        event_kind: ReplayIncomingEventKindV1,
        expected_kind: ReplayIncomingEventKindV1,
        expected_phase_rank: int,
    ) -> None:
        """Validate shared event identity and canonical ordering fields.
        Require nonempty event ID, nonnegative Python ordinal/rank, and the concrete
        expected kind/rank.
        Parameters
        ----------
        event_kind : ReplayIncomingEventKindV1
            Actual concrete event discriminator.
        expected_kind : ReplayIncomingEventKindV1
            Required discriminator for the concrete event type.
        expected_phase_rank : int
            Canonical scientific ordering rank required by the event type.
        Returns
        -------
        None
            Shared event fields satisfy the concrete variant contract.
        Raises
        ------
        ValueError
            Identity, integer domains, kind, or rank are invalid.
        """
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
    """Require an exact incoming agent anchor at the named phase.
    This helper checks its row type and phase; the enclosing summary joins
    identity and position to a trajectory.
    Parameters
    ----------
    value : ReplayIncomingAgentAnchorV1
        Value to check; this helper does not coerce it.
    name : str
        Field label used in validation errors.
    phase : ReplayIncomingAnchorPhaseV1
        Required scientific anchor phase.
    Returns
    -------
    None
        The phase/type check passed.
    Raises
    ------
    ValueError
        The anchor type or phase differs.
    """
    if type(value) is not ReplayIncomingAgentAnchorV1 or value.phase != phase:
        raise ValueError(f"{name} must be an exact {phase} incoming anchor.")


def _require_optional_incoming_anchor(
    value: ReplayIncomingAgentAnchorV1 | None,
    *,
    name: str,
    phase: ReplayIncomingAnchorPhaseV1,
) -> None:
    """Validate an incoming anchor only when one is present.
    None is a deliberate absence; a present anchor follows the exact phase/type
    rule.
    Parameters
    ----------
    value : ReplayIncomingAgentAnchorV1 | None
        Value to check; this helper does not coerce it.
    name : str
        Field label used in validation errors.
    phase : ReplayIncomingAnchorPhaseV1
        Required scientific anchor phase.
    Returns
    -------
    None
        The optional anchor is absent or valid.
    Raises
    ------
    ValueError
        A present anchor has the wrong type or phase.
    """
    if value is not None:
        _require_incoming_anchor(value, name=name, phase=phase)


def _require_incoming_anchor_tuple(
    value: tuple[ReplayIncomingAgentAnchorV1, ...],
    *,
    name: str,
    phase: ReplayIncomingAnchorPhaseV1,
) -> None:
    """Require unique authorized anchors at one scientific phase.
    Validate exact tuple/row types, the required phase, and unique presentation
    keys.
    Parameters
    ----------
    value : tuple[ReplayIncomingAgentAnchorV1, ...]
        Value to check; this helper does not coerce it.
    name : str
        Field label used in validation errors.
    phase : ReplayIncomingAnchorPhaseV1
        Required scientific anchor phase.
    Returns
    -------
    None
        Every tuple member satisfies the anchor contract.
    Raises
    ------
    ValueError
        A type, phase, or uniqueness check fails.
    """
    _require_exact_tuple(value, name=name, item_type=ReplayIncomingAgentAnchorV1)
    for anchor in value:
        _require_incoming_anchor(anchor, name=f"{name} item", phase=phase)
    keys = tuple(anchor.presentation_key for anchor in value)
    if len(keys) != len(set(keys)):
        raise ValueError(f"{name} must contain unique authorized emitters.")


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayIncomingActionRejectedEventV1(_ReplayIncomingEventBaseV1):
    """Record an action component rejected at transition start.
    Require the canonical event kind and phase rank 10. Active actors require
    matching authorized identity and start anchor. Inactive actors remain feed-
    only with no anchor; preserve the submitted tuple.
    All constructor fields are required keyword arguments; instances are frozen.
    Inherited fields are documented on _ReplayIncomingEventBaseV1.
    Attributes
    ----------
    event_kind : Literal['action_rejected']
        Exact event discriminator for this concrete row type.
    actor_identity : ReplayIncomingAgentIdentityV1
        Scene identity for an active actor, or a feed-only identity for an
        inactive actor.
    actor_configured_active : bool
        Whether the rejected actor belongs to the active roster.
    rejection_component : Literal['domain', 'movement', 'combat_pair']
        Rejected domain, movement component, or target/Ultimate pair.
    submitted_action : SubmittedActionTupleV1
        Recorded submitted integer tuple, including rejected out-of-domain
        values.
    actor_anchor : ReplayIncomingAgentAnchorV1 | None
        Rejected actor position at transition start; None for inactive feed-only
        rows.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    event_kind: Literal["action_rejected"]
    """Exact event discriminator for this concrete row type."""
    actor_identity: ReplayIncomingAgentIdentityV1
    """Scene identity for an active actor, or a feed-only identity for an inactive
    actor."""
    actor_configured_active: bool
    """Whether the rejected actor belongs to the active roster."""
    rejection_component: Literal["domain", "movement", "combat_pair"]
    """Rejected domain, movement component, or target/Ultimate pair."""
    submitted_action: SubmittedActionTupleV1
    """Recorded submitted integer tuple, including rejected out-of-domain values."""
    actor_anchor: ReplayIncomingAgentAnchorV1 | None
    """Rejected actor position at transition start; None for inactive feed-only
    rows."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Require the canonical event kind and phase rank 10. Active actors require
        matching authorized identity and start anchor. Inactive actors remain feed-
        only with no anchor; preserve the submitted tuple.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
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
    """Record a Basic or Ultimate activation at transition start.
    Require the canonical event kind and phase rank 20. Require a start source
    anchor and an optional start recipient anchor; the ability component is
    basic or ultimate.
    All constructor fields are required keyword arguments; instances are frozen.
    Inherited fields are documented on _ReplayIncomingEventBaseV1.
    Attributes
    ----------
    event_kind : Literal['ability_activated']
        Exact event discriminator for this concrete row type.
    ability_component : Literal['basic', 'ultimate']
        basic or ultimate, identifying the activated ability.
    source_anchor : ReplayIncomingAgentAnchorV1
        Authorized source identity and position at the event required phase.
    recipient_anchor : ReplayIncomingAgentAnchorV1 | None
        Authorized recipient identity and position at the event required phase.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    event_kind: Literal["ability_activated"]
    """Exact event discriminator for this concrete row type."""
    ability_component: Literal["basic", "ultimate"]
    """basic or ultimate, identifying the activated ability."""
    source_anchor: ReplayIncomingAgentAnchorV1
    """Authorized source identity and position at the event required phase."""
    recipient_anchor: ReplayIncomingAgentAnchorV1 | None
    """Authorized recipient identity and position at the event required phase."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Require the canonical event kind and phase rank 20. Require a start source
        anchor and an optional start recipient anchor; the ability component is
        basic or ultimate.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
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
    """Record source damage and the covering aura evidence.
    Require the canonical event kind and phase rank 30. Source and optional
    recipient use start anchors. Damage values and recipient multiplier are
    finite and nonnegative; emitter tuples contain unique start anchors.
    All constructor fields are required keyword arguments; instances are frozen.
    Inherited fields are documented on _ReplayIncomingEventBaseV1.
    Attributes
    ----------
    event_kind : Literal['source_damage_output']
        Exact event discriminator for this concrete row type.
    source_anchor : ReplayIncomingAgentAnchorV1
        Authorized source identity and position at the event required phase.
    recipient_anchor : ReplayIncomingAgentAnchorV1 | None
        Authorized recipient identity and position at the event required phase.
    raw_damage_output : float
        Nonnegative source damage before modifiers, in hit points.
    source_modified_damage_output : float
        Nonnegative damage after source modifiers, before recipient modifiers.
    recipient_damage_modifier : float
        Nonnegative recipient damage multiplier.
    mage_damage_aura_covering_emitters : tuple[ReplayIncomingAgentAnchorV1, ...]
        Unique covering Mage emitters at transition start, in trajectory order.
    warrior_mitigation_aura_covering_emitters : tuple[ReplayIncomingAgentAnchorV1,
    ...]
        Unique covering Warrior emitters at transition start, in trajectory
        order.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    event_kind: Literal["source_damage_output"]
    """Exact event discriminator for this concrete row type."""
    source_anchor: ReplayIncomingAgentAnchorV1
    """Authorized source identity and position at the event required phase."""
    recipient_anchor: ReplayIncomingAgentAnchorV1 | None
    """Authorized recipient identity and position at the event required phase."""
    raw_damage_output: float
    """Nonnegative source damage before modifiers, in hit points."""
    source_modified_damage_output: float
    """Nonnegative damage after source modifiers, before recipient modifiers."""
    recipient_damage_modifier: float
    """Nonnegative recipient damage multiplier."""
    mage_damage_aura_covering_emitters: tuple[ReplayIncomingAgentAnchorV1, ...]
    """Unique covering Mage emitters at transition start, in trajectory order."""
    warrior_mitigation_aura_covering_emitters: tuple[ReplayIncomingAgentAnchorV1, ...]
    """"""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Require the canonical event kind and phase rank 30. Source and optional
        recipient use start anchors. Damage values and recipient multiplier are
        finite and nonnegative; emitter tuples contain unique start anchors.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
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
    """Record source healing before recipient health resolution.
    Require the canonical event kind and phase rank 30. Source and optional
    recipient use start anchors; healing values and recipient multiplier are
    finite and nonnegative.
    All constructor fields are required keyword arguments; instances are frozen.
    Inherited fields are documented on _ReplayIncomingEventBaseV1.
    Attributes
    ----------
    event_kind : Literal['source_healing_output']
        Exact event discriminator for this concrete row type.
    source_anchor : ReplayIncomingAgentAnchorV1
        Authorized source identity and position at the event required phase.
    recipient_anchor : ReplayIncomingAgentAnchorV1 | None
        Authorized recipient identity and position at the event required phase.
    raw_healing_output : float
        Nonnegative source healing before modifiers, in hit points.
    source_modified_healing_output : float
        Nonnegative healing after source modifiers, before recipient modifiers.
    recipient_healing_modifier : float
        Nonnegative recipient healing multiplier.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    event_kind: Literal["source_healing_output"]
    """Exact event discriminator for this concrete row type."""
    source_anchor: ReplayIncomingAgentAnchorV1
    """Authorized source identity and position at the event required phase."""
    recipient_anchor: ReplayIncomingAgentAnchorV1 | None
    """Authorized recipient identity and position at the event required phase."""
    raw_healing_output: float
    """Nonnegative source healing before modifiers, in hit points."""
    source_modified_healing_output: float
    """Nonnegative healing after source modifiers, before recipient modifiers."""
    recipient_healing_modifier: float
    """Nonnegative recipient healing multiplier."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Require the canonical event kind and phase rank 30. Source and optional
        recipient use start anchors; healing values and recipient multiplier are
        finite and nonnegative.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
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
    """Record combined combat health resolution for one recipient.
    Require the canonical event kind and phase rank 40. Use a start recipient
    anchor, nonnegative health/damage/healing values, and finite signed net
    change. This row stores recorded totals rather than recomputing combat.
    All constructor fields are required keyword arguments; instances are frozen.
    Inherited fields are documented on _ReplayIncomingEventBaseV1.
    Attributes
    ----------
    event_kind : Literal['recipient_health_resolution']
        Exact event discriminator for this concrete row type.
    recipient_anchor : ReplayIncomingAgentAnchorV1
        Authorized recipient identity and position at the event required phase.
    transition_start_health : float
        Recipient health before combat resolution, in hit points.
    total_effective_damage : float
        Nonnegative combined effective damage in hit points.
    total_effective_healing : float
        Nonnegative combined effective healing in hit points.
    health_after_combat_resolution : float
        Nonnegative health after combined combat resolution, in hit points.
    realized_net_health_change : float
        Signed health change in hit points: positive for gain, negative for
        loss.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    event_kind: Literal["recipient_health_resolution"]
    """Exact event discriminator for this concrete row type."""
    recipient_anchor: ReplayIncomingAgentAnchorV1
    """Authorized recipient identity and position at the event required phase."""
    transition_start_health: float
    """Recipient health before combat resolution, in hit points."""
    total_effective_damage: float
    """Nonnegative combined effective damage in hit points."""
    total_effective_healing: float
    """Nonnegative combined effective healing in hit points."""
    health_after_combat_resolution: float
    """Nonnegative health after combined combat resolution, in hit points."""
    realized_net_health_change: float
    """Signed health change in hit points: positive for gain, negative for loss."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Require the canonical event kind and phase rank 40. Use a start recipient
        anchor, nonnegative health/damage/healing values, and finite signed net
        change. This row stores recorded totals rather than recomputing combat.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
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
    """Record a reset of an agent combat countdown.
    Require the canonical event kind and phase rank 50. Use the agent
    transition-start anchor.
    All constructor fields are required keyword arguments; instances are frozen.
    Inherited fields are documented on _ReplayIncomingEventBaseV1.
    Attributes
    ----------
    event_kind : Literal['combat_countdown_reset']
        Exact event discriminator for this concrete row type.
    agent_anchor : ReplayIncomingAgentAnchorV1
        Authorized agent identity and position at the event required phase.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    event_kind: Literal["combat_countdown_reset"]
    """Exact event discriminator for this concrete row type."""
    agent_anchor: ReplayIncomingAgentAnchorV1
    """Authorized agent identity and position at the event required phase."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Require the canonical event kind and phase rank 50. Use the agent
        transition-start anchor.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
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
class ReplayIncomingAgentLeftCombatEventV1(_ReplayIncomingEventBaseV1):
    """Record an agent leaving combat at the successor.
    Require the canonical event kind and phase rank 50. Use the successor agent
    anchor.
    All constructor fields are required keyword arguments; instances are frozen.
    Inherited fields are documented on _ReplayIncomingEventBaseV1.
    Attributes
    ----------
    event_kind : Literal['agent_left_combat']
        Exact event discriminator for this concrete row type.
    agent_anchor : ReplayIncomingAgentAnchorV1
        Authorized agent identity and position at the event required phase.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    event_kind: Literal["agent_left_combat"]
    """Exact event discriminator for this concrete row type."""
    agent_anchor: ReplayIncomingAgentAnchorV1
    """Authorized agent identity and position at the event required phase."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Require the canonical event kind and phase rank 50. Use the successor agent
        anchor.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
        self._validate_base(
            event_kind=self.event_kind,
            expected_kind="agent_left_combat",
            expected_phase_rank=50,
        )
        _require_incoming_anchor(
            self.agent_anchor,
            name="agent_anchor",
            phase="successor",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayIncomingHealthRegeneratedEventV1(_ReplayIncomingEventBaseV1):
    """Record health actually restored by regeneration.
    Require the canonical event kind and phase rank 50. Use a transition-start
    anchor and a nonnegative finite restored-health value.
    All constructor fields are required keyword arguments; instances are frozen.
    Inherited fields are documented on _ReplayIncomingEventBaseV1.
    Attributes
    ----------
    event_kind : Literal['health_regenerated']
        Exact event discriminator for this concrete row type.
    agent_anchor : ReplayIncomingAgentAnchorV1
        Authorized agent identity and position at the event required phase.
    actual_health_regenerated : float
        Nonnegative health actually restored by regeneration, in hit points.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    event_kind: Literal["health_regenerated"]
    """Exact event discriminator for this concrete row type."""
    agent_anchor: ReplayIncomingAgentAnchorV1
    """Authorized agent identity and position at the event required phase."""
    actual_health_regenerated: float
    """Nonnegative health actually restored by regeneration, in hit points."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Require the canonical event kind and phase rank 50. Use a transition-start
        anchor and a nonnegative finite restored-health value.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
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
    """Record an Ultimate cooldown starting.
    Require the canonical event kind and phase rank 60. Use the transition-start
    agent anchor.
    All constructor fields are required keyword arguments; instances are frozen.
    Inherited fields are documented on _ReplayIncomingEventBaseV1.
    Attributes
    ----------
    event_kind : Literal['cooldown_started']
        Exact event discriminator for this concrete row type.
    agent_anchor : ReplayIncomingAgentAnchorV1
        Authorized agent identity and position at the event required phase.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    event_kind: Literal["cooldown_started"]
    """Exact event discriminator for this concrete row type."""
    agent_anchor: ReplayIncomingAgentAnchorV1
    """Authorized agent identity and position at the event required phase."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Require the canonical event kind and phase rank 60. Use the transition-start
        agent anchor.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
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
    """Record an Ultimate cooldown becoming ready.
    Require the canonical event kind and phase rank 60. Use the transition-start
    agent anchor.
    All constructor fields are required keyword arguments; instances are frozen.
    Inherited fields are documented on _ReplayIncomingEventBaseV1.
    Attributes
    ----------
    event_kind : Literal['cooldown_ready']
        Exact event discriminator for this concrete row type.
    agent_anchor : ReplayIncomingAgentAnchorV1
        Authorized agent identity and position at the event required phase.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    event_kind: Literal["cooldown_ready"]
    """Exact event discriminator for this concrete row type."""
    agent_anchor: ReplayIncomingAgentAnchorV1
    """Authorized agent identity and position at the event required phase."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Require the canonical event kind and phase rank 60. Use the transition-start
        agent anchor.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
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
    """Record actual movement during the Charge phase.
    Require the canonical event kind and phase rank 70. Join transition_start to
    post_charge for one identity. End position equals start plus displacement
    within relative 1e-6 and absolute 1e-5 tolerance.
    All constructor fields are required keyword arguments; instances are frozen.
    Inherited fields are documented on _ReplayIncomingEventBaseV1.
    Attributes
    ----------
    event_kind : Literal['charge_phase_displacement']
        Exact event discriminator for this concrete row type.
    realized_displacement : Point2D
        Actual (dx, dy) displacement in world units.
    start_anchor : ReplayIncomingAgentAnchorV1
        Authorized identity and position at this movement phase start.
    end_anchor : ReplayIncomingAgentAnchorV1
        Same agent at this movement phase end, after realized displacement.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    event_kind: Literal["charge_phase_displacement"]
    """Exact event discriminator for this concrete row type."""
    realized_displacement: Point2D
    """Actual (dx, dy) displacement in world units."""
    start_anchor: ReplayIncomingAgentAnchorV1
    """Authorized identity and position at this movement phase start."""
    end_anchor: ReplayIncomingAgentAnchorV1
    """Same agent at this movement phase end, after realized displacement."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Require the canonical event kind and phase rank 70. Join transition_start to
        post_charge for one identity. End position equals start plus displacement
        within relative 1e-6 and absolute 1e-5 tolerance.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
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
    """Record actual movement after Charge.
    Require the canonical event kind and phase rank 80. Join post_charge to
    successor for one identity. End position equals start plus displacement
    within relative 1e-6 and absolute 1e-5 tolerance.
    All constructor fields are required keyword arguments; instances are frozen.
    Inherited fields are documented on _ReplayIncomingEventBaseV1.
    Attributes
    ----------
    event_kind : Literal['ordinary_movement_phase_displacement']
        Exact event discriminator for this concrete row type.
    realized_displacement : Point2D
        Actual (dx, dy) displacement in world units.
    start_anchor : ReplayIncomingAgentAnchorV1
        Authorized identity and position at this movement phase start.
    end_anchor : ReplayIncomingAgentAnchorV1
        Same agent at this movement phase end, after realized displacement.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    event_kind: Literal["ordinary_movement_phase_displacement"]
    """Exact event discriminator for this concrete row type."""
    realized_displacement: Point2D
    """Actual (dx, dy) displacement in world units."""
    start_anchor: ReplayIncomingAgentAnchorV1
    """Authorized identity and position at this movement phase start."""
    end_anchor: ReplayIncomingAgentAnchorV1
    """Same agent at this movement phase end, after realized displacement."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Require the canonical event kind and phase rank 80. Join post_charge to
        successor for one identity. End position equals start plus displacement
        within relative 1e-6 and absolute 1e-5 tolerance.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
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
    """Record a death at the successor position.
    Require the canonical event kind and phase rank 90. Use the successor
    recipient anchor.
    All constructor fields are required keyword arguments; instances are frozen.
    Inherited fields are documented on _ReplayIncomingEventBaseV1.
    Attributes
    ----------
    event_kind : Literal['agent_died']
        Exact event discriminator for this concrete row type.
    recipient_anchor : ReplayIncomingAgentAnchorV1
        Authorized recipient identity and position at the event required phase.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    event_kind: Literal["agent_died"]
    """Exact event discriminator for this concrete row type."""
    recipient_anchor: ReplayIncomingAgentAnchorV1
    """Authorized recipient identity and position at the event required phase."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Require the canonical event kind and phase rank 90. Use the successor
        recipient anchor.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
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
    """Record a source contribution to a recipient death.
    Require the canonical event kind and phase rank 90. Use successor source and
    recipient anchors and nonnegative finite attributed damage.
    All constructor fields are required keyword arguments; instances are frozen.
    Inherited fields are documented on _ReplayIncomingEventBaseV1.
    Attributes
    ----------
    event_kind : Literal['lethal_damage_contribution']
        Exact event discriminator for this concrete row type.
    source_anchor : ReplayIncomingAgentAnchorV1
        Authorized source identity and position at the event required phase.
    recipient_anchor : ReplayIncomingAgentAnchorV1
        Authorized recipient identity and position at the event required phase.
    attributed_death_damage : float
        Nonnegative damage contribution attributed to this death, in hit points.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    event_kind: Literal["lethal_damage_contribution"]
    """Exact event discriminator for this concrete row type."""
    source_anchor: ReplayIncomingAgentAnchorV1
    """Authorized source identity and position at the event required phase."""
    recipient_anchor: ReplayIncomingAgentAnchorV1
    """Authorized recipient identity and position at the event required phase."""
    attributed_death_damage: float
    """Nonnegative damage contribution attributed to this death, in hit points."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Require the canonical event kind and phase rank 90. Use successor source and
        recipient anchors and nonnegative finite attributed damage.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
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
    """Share recipient and status identity across status lifecycle events.
    Subclasses validate successor anchors, the canonical status channel/ID pair,
    and phase rank 100 through _validate_status.
    All constructor fields are required keyword arguments; instances are frozen.
    Inherited fields are documented on _ReplayIncomingEventBaseV1.
    Attributes
    ----------
    recipient_anchor : ReplayIncomingAgentAnchorV1
        Authorized recipient identity and position at the event required phase.
    status_channel : int
        Zero-based channel on the fixed nine-status axis.
    status_id : str
        Canonical status identifier matching status_channel.
    """

    recipient_anchor: ReplayIncomingAgentAnchorV1
    """Authorized recipient identity and position at the event required phase."""
    status_channel: int
    """Zero-based channel on the fixed nine-status axis."""
    status_id: str
    """Canonical status identifier matching status_channel."""

    def _validate_status(
        self,
        *,
        event_kind: ReplayIncomingEventKindV1,
        expected_kind: ReplayIncomingEventKindV1,
    ) -> None:
        """Validate a status lifecycle event at the successor phase.
        Check base event fields at rank 100, the exact successor anchor, and the
        nine-channel status ID mapping.
        Parameters
        ----------
        event_kind : ReplayIncomingEventKindV1
            Actual concrete event discriminator.
        expected_kind : ReplayIncomingEventKindV1
            Required discriminator for the concrete event type.
        Returns
        -------
        None
            Shared status-event fields are valid.
        Raises
        ------
        ValueError
            The event, anchor, or catalog identity is inconsistent.
        """
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
    """Record a status expiring through normal ageing.
    Require the canonical event kind and phase rank 100. Validate the successor
    recipient and canonical status channel/ID.
    All constructor fields are required keyword arguments; instances are frozen.
    Inherited fields are documented on _ReplayIncomingStatusEventBaseV1.
    Attributes
    ----------
    event_kind : Literal['status_aged_to_zero']
        Exact event discriminator for this concrete row type.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    event_kind: Literal["status_aged_to_zero"]
    """Exact event discriminator for this concrete row type."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Require the canonical event kind and phase rank 100. Validate the successor
        recipient and canonical status channel/ID.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
        self._validate_status(
            event_kind=self.event_kind,
            expected_kind="status_aged_to_zero",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayIncomingStatusBrokenByDamageEventV1(_ReplayIncomingStatusEventBaseV1):
    """Record a status removed by positive damage.
    Require the canonical event kind and phase rank 100. Validate the successor
    recipient and canonical status channel/ID.
    All constructor fields are required keyword arguments; instances are frozen.
    Inherited fields are documented on _ReplayIncomingStatusEventBaseV1.
    Attributes
    ----------
    event_kind : Literal['status_broken_by_damage']
        Exact event discriminator for this concrete row type.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    event_kind: Literal["status_broken_by_damage"]
    """Exact event discriminator for this concrete row type."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Require the canonical event kind and phase rank 100. Validate the successor
        recipient and canonical status channel/ID.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
        self._validate_status(
            event_kind=self.event_kind,
            expected_kind="status_broken_by_damage",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayIncomingStatusAppliedEventV1(_ReplayIncomingStatusEventBaseV1):
    """Record a new status application and its source.
    Require the canonical event kind and phase rank 100. Validate successor
    source/recipient anchors and canonical status channel/ID.
    All constructor fields are required keyword arguments; instances are frozen.
    Inherited fields are documented on _ReplayIncomingStatusEventBaseV1.
    Attributes
    ----------
    event_kind : Literal['status_applied']
        Exact event discriminator for this concrete row type.
    source_anchor : ReplayIncomingAgentAnchorV1
        Authorized source identity and position at the event required phase.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    event_kind: Literal["status_applied"]
    """Exact event discriminator for this concrete row type."""
    source_anchor: ReplayIncomingAgentAnchorV1
    """Authorized source identity and position at the event required phase."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Require the canonical event kind and phase rank 100. Validate successor
        source/recipient anchors and canonical status channel/ID.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
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
    """Record a status duration being refreshed or extended.
    Require the canonical event kind and phase rank 100. Validate the successor
    recipient and canonical status channel/ID.
    All constructor fields are required keyword arguments; instances are frozen.
    Inherited fields are documented on _ReplayIncomingStatusEventBaseV1.
    Attributes
    ----------
    event_kind : Literal['status_refreshed_or_extended']
        Exact event discriminator for this concrete row type.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    event_kind: Literal["status_refreshed_or_extended"]
    """Exact event discriminator for this concrete row type."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Require the canonical event kind and phase rank 100. Validate the successor
        recipient and canonical status channel/ID.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
        self._validate_status(
            event_kind=self.event_kind,
            expected_kind="status_refreshed_or_extended",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayIncomingStatusClearedByNewDeathEventV1(_ReplayIncomingStatusEventBaseV1):
    """Record a status removed because its recipient just died.
    Require the canonical event kind and phase rank 100. Validate the successor
    recipient and canonical status channel/ID.
    All constructor fields are required keyword arguments; instances are frozen.
    Inherited fields are documented on _ReplayIncomingStatusEventBaseV1.
    Attributes
    ----------
    event_kind : Literal['status_cleared_by_new_death']
        Exact event discriminator for this concrete row type.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    event_kind: Literal["status_cleared_by_new_death"]
    """Exact event discriminator for this concrete row type."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Require the canonical event kind and phase rank 100. Validate the successor
        recipient and canonical status channel/ID.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
        self._validate_status(
            event_kind=self.event_kind,
            expected_kind="status_cleared_by_new_death",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayIncomingSpawnShieldExpiredEventV1(_ReplayIncomingEventBaseV1):
    """Record Spawn Shield expiry at the successor.
    Require the canonical event kind and phase rank 110. Use the successor agent
    anchor.
    All constructor fields are required keyword arguments; instances are frozen.
    Inherited fields are documented on _ReplayIncomingEventBaseV1.
    Attributes
    ----------
    event_kind : Literal['spawn_shield_expired']
        Exact event discriminator for this concrete row type.
    agent_anchor : ReplayIncomingAgentAnchorV1
        Authorized agent identity and position at the event required phase.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    event_kind: Literal["spawn_shield_expired"]
    """Exact event discriminator for this concrete row type."""
    agent_anchor: ReplayIncomingAgentAnchorV1
    """Authorized agent identity and position at the event required phase."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Require the canonical event kind and phase rank 110. Use the successor agent
        anchor.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
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
    """Record a team respawn wave without a spatial anchor.
    Require the canonical event kind and phase rank 120. Require an exact
    successor team anchor.
    All constructor fields are required keyword arguments; instances are frozen.
    Inherited fields are documented on _ReplayIncomingEventBaseV1.
    Attributes
    ----------
    event_kind : Literal['respawn_wave_occurred']
        Exact event discriminator for this concrete row type.
    team_anchor : ReplayIncomingTeamAnchorV1
        Matching team identity at the successor phase, without a spatial
        position.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    event_kind: Literal["respawn_wave_occurred"]
    """Exact event discriminator for this concrete row type."""
    team_anchor: ReplayIncomingTeamAnchorV1
    """Matching team identity at the successor phase, without a spatial position."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Require the canonical event kind and phase rank 120. Require an exact
        successor team anchor.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
        self._validate_base(
            event_kind=self.event_kind,
            expected_kind="respawn_wave_occurred",
            expected_phase_rank=120,
        )
        if type(self.team_anchor) is not ReplayIncomingTeamAnchorV1:
            raise ValueError("respawn wave must retain an exact neutral team anchor.")


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayIncomingAgentRespawnedEventV1(_ReplayIncomingEventBaseV1):
    """Record an agent respawn at its actual successor position.
    Require the canonical event kind and phase rank 120. Require a valid team ID
    and a realized position exactly equal to the successor anchor.
    All constructor fields are required keyword arguments; instances are frozen.
    Inherited fields are documented on _ReplayIncomingEventBaseV1.
    Attributes
    ----------
    event_kind : Literal['agent_respawned']
        Exact event discriminator for this concrete row type.
    agent_anchor : ReplayIncomingAgentAnchorV1
        Authorized agent identity and position at the event required phase.
    team_id : int
        Team ID: 1 for Team A, 2 for Team B.
    realized_successor_position : Point2D
        Respawn position (x, y), exactly equal to the successor anchor.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    event_kind: Literal["agent_respawned"]
    """Exact event discriminator for this concrete row type."""
    agent_anchor: ReplayIncomingAgentAnchorV1
    """Authorized agent identity and position at the event required phase."""
    team_id: int
    """Team ID: 1 for Team A, 2 for Team B."""
    realized_successor_position: Point2D
    """Respawn position (x, y), exactly equal to the successor anchor."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Require the canonical event kind and phase rank 120. Require a valid team ID
        and a realized position exactly equal to the successor anchor.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
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


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayIncomingTeamDeathmatchScoreChangedEventV1(_ReplayIncomingEventBaseV1):
    """Record a global Team Deathmatch score increase.
    Require the canonical event kind and phase rank 130. Require matching team
    identity/anchor, positive increment, and successor score equal to previous
    score plus increment.
    All constructor fields are required keyword arguments; instances are frozen.
    Inherited fields are documented on _ReplayIncomingEventBaseV1.
    Attributes
    ----------
    event_kind : Literal['team_deathmatch_score_changed']
        Exact event discriminator for this concrete row type.
    team_index : int
        Zero-based team index: 0 for Team A, 1 for Team B.
    team_id : int
        Team ID: 1 for Team A, 2 for Team B.
    score_increment : int
        Positive Team Deathmatch score increase.
    previous_score : int
        Nonnegative score before this transition.
    successor_score : int
        Previous score plus score_increment.
    team_anchor : ReplayIncomingTeamAnchorV1
        Matching team identity at the successor phase, without a spatial
        position.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    event_kind: Literal["team_deathmatch_score_changed"]
    """Exact event discriminator for this concrete row type."""
    team_index: int
    """Zero-based team index: 0 for Team A, 1 for Team B."""
    team_id: int
    """Team ID: 1 for Team A, 2 for Team B."""
    score_increment: int
    """Positive Team Deathmatch score increase."""
    previous_score: int
    """Nonnegative score before this transition."""
    successor_score: int
    """Previous score plus score_increment."""
    team_anchor: ReplayIncomingTeamAnchorV1
    """Matching team identity at the successor phase, without a spatial position."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Require the canonical event kind and phase rank 130. Require matching team
        identity/anchor, positive increment, and successor score equal to previous
        score plus increment.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
        self._validate_base(
            event_kind=self.event_kind,
            expected_kind="team_deathmatch_score_changed",
            expected_phase_rank=130,
        )
        _require_python_int(self.team_index, name="team_index")
        if self.team_index not in (0, 1):
            raise ValueError("team_index must be zero or one.")
        _require_python_int(self.team_id, name="team_id", minimum=1)
        if self.team_id != self.team_index + 1:
            raise ValueError("team_id must equal team_index plus one.")
        _require_python_int(self.score_increment, name="score_increment", minimum=1)
        _require_python_int(self.previous_score, name="previous_score", minimum=0)
        _require_python_int(self.successor_score, name="successor_score", minimum=0)
        if self.successor_score != self.previous_score + self.score_increment:
            raise ValueError("successor_score must apply the exact score increment.")
        if type(self.team_anchor) is not ReplayIncomingTeamAnchorV1:
            raise ValueError("score change must retain an exact neutral team anchor.")
        if (
            self.team_anchor.team_index != self.team_index
            or self.team_anchor.team_id != self.team_id
        ):
            raise ValueError("score change team facts must join its neutral anchor.")


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayIncomingTeamDeathmatchCompletedEventV1(_ReplayIncomingEventBaseV1):
    """Record the global Team Deathmatch result and completion reason.
    Require the canonical event kind and phase rank 140. Require a known
    win/draw outcome and score-threshold/horizon completion basis.
    All constructor fields are required keyword arguments; instances are frozen.
    Inherited fields are documented on _ReplayIncomingEventBaseV1.
    Attributes
    ----------
    event_kind : Literal['team_deathmatch_completed']
        Exact event discriminator for this concrete row type.
    outcome : Literal['team_a_win', 'team_b_win', 'draw']
        team_a_win, team_b_win, or draw.
    completion_basis : str
        Allowed values: 'score_threshold', 'horizon', 'score_threshold_at_horizon'.
        Score threshold, horizon, or both on the same transition.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    event_kind: Literal["team_deathmatch_completed"]
    """Exact event discriminator for this concrete row type."""
    outcome: Literal["team_a_win", "team_b_win", "draw"]
    """team_a_win, team_b_win, or draw."""
    completion_basis: Literal[
        "score_threshold",
        "horizon",
        "score_threshold_at_horizon",
    ]
    """Allowed values: 'score_threshold', 'horizon', 'score_threshold_at_horizon'.
    Score threshold, horizon, or both on the same transition."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Require the canonical event kind and phase rank 140. Require a known
        win/draw outcome and score-threshold/horizon completion basis.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
        self._validate_base(
            event_kind=self.event_kind,
            expected_kind="team_deathmatch_completed",
            expected_phase_rank=140,
        )
        if self.outcome not in ("team_a_win", "team_b_win", "draw"):
            raise ValueError("unknown Team Deathmatch outcome.")
        if self.completion_basis not in (
            "score_threshold",
            "horizon",
            "score_threshold_at_horizon",
        ):
            raise ValueError("unknown Team Deathmatch completion basis.")


type ReplayIncomingEventV1 = Annotated[
    ReplayIncomingActionRejectedEventV1
    | ReplayIncomingAbilityActivatedEventV1
    | ReplayIncomingSourceDamageOutputEventV1
    | ReplayIncomingSourceHealingOutputEventV1
    | ReplayIncomingRecipientHealthResolutionEventV1
    | ReplayIncomingCombatCountdownResetEventV1
    | ReplayIncomingAgentLeftCombatEventV1
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
    | ReplayIncomingAgentRespawnedEventV1
    | ReplayIncomingTeamDeathmatchScoreChangedEventV1
    | ReplayIncomingTeamDeathmatchCompletedEventV1,
    Field(discriminator="event_kind"),
]


_REPLAY_INCOMING_EVENT_TYPES_V1: tuple[type[object], ...] = (
    ReplayIncomingActionRejectedEventV1,
    ReplayIncomingAbilityActivatedEventV1,
    ReplayIncomingSourceDamageOutputEventV1,
    ReplayIncomingSourceHealingOutputEventV1,
    ReplayIncomingRecipientHealthResolutionEventV1,
    ReplayIncomingCombatCountdownResetEventV1,
    ReplayIncomingAgentLeftCombatEventV1,
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
    ReplayIncomingTeamDeathmatchScoreChangedEventV1,
    ReplayIncomingTeamDeathmatchCompletedEventV1,
)


def _incoming_event_agent_anchors(
    event: ReplayIncomingEventV1,
) -> tuple[ReplayIncomingAgentAnchorV1, ...]:
    """Collect all agent anchors carried by one concrete incoming event.
    Include optional recipients and covering aura emitters in payload order.
    Non-spatial team events and inactive rejections return an empty tuple; no
    deduplication occurs.
    Parameters
    ----------
    event : ReplayIncomingEventV1
        Typed event whose anchors or authorized payload are needed.
    Returns
    -------
    tuple[ReplayIncomingAgentAnchorV1, ...]
        Anchors whose trajectory joins the summary must check.
    Raises
    ------
    TypeError
        The event is not a supported exact concrete variant.
    """
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
        ReplayIncomingAgentLeftCombatEventV1,
        ReplayIncomingHealthRegeneratedEventV1,
        ReplayIncomingCooldownStartedEventV1,
        ReplayIncomingCooldownReadyEventV1,
        ReplayIncomingSpawnShieldExpiredEventV1,
        ReplayIncomingAgentRespawnedEventV1,
    ):
        agent_event = cast(
            ReplayIncomingCombatCountdownResetEventV1
            | ReplayIncomingAgentLeftCombatEventV1
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
    if type(event) in (
        ReplayIncomingRespawnWaveOccurredEventV1,
        ReplayIncomingTeamDeathmatchScoreChangedEventV1,
        ReplayIncomingTeamDeathmatchCompletedEventV1,
    ):
        return ()
    raise TypeError("unknown replay incoming neutral event variant.")


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayIncomingSummaryV1:
    """Store one complete Oracle incoming event inventory.
    Join canonical episode/frame/transition IDs, adjacent ticks, dense event
    order, unique trajectories, and every anchor. Covering aura emitters retain
    trajectory order.
    All constructor fields are required keyword arguments; instances are frozen.
    Attributes
    ----------
    summary_kind : Literal['replay_incoming_inventory']
        Exact discriminator for the incoming inventory variant.
    incoming_transition_index : int
        Zero-based index of the transition ending at the displayed frame.
    incoming_transition_id : str
        Canonical episode-scoped ID for the incoming transition.
    incoming_start_frame_id : str
        Canonical frame ID immediately before the incoming transition.
    incoming_successor_frame_id : str
        Canonical frame ID immediately after the incoming transition.
    incoming_start_simulator_step_count : int
        Nonnegative simulator tick before the transition.
    incoming_successor_simulator_step_count : int
        Simulator tick after the transition; exactly one above the start tick.
    agent_phase_trajectories : tuple[ReplayIncomingAgentPhaseTrajectoryV1, ...]
        Unique ordered authorized trajectories used by every event anchor.
    ordered_event_ids : tuple[str, ...]
        Event IDs in payload order, with dense zero-based ordinals.
    ordered_event_kinds : tuple[ReplayIncomingEventKindV1, ...]
        Event discriminators in the same order as IDs and payloads.
    events : tuple[ReplayIncomingEventV1, ...]
        Concrete typed event rows in nondecreasing scientific phase order.
    event_count : int
        Nonnegative count equal to the ID, kind, and payload tuple lengths.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG
    """Strict Pydantic wire settings: forbid extra fields, nonfinite numbers, and
    coercion."""

    summary_kind: Literal["replay_incoming_inventory"]
    """Exact discriminator for the incoming inventory variant."""
    incoming_transition_index: int
    """Zero-based index of the transition ending at the displayed frame."""
    incoming_transition_id: str
    """Canonical episode-scoped ID for the incoming transition."""
    incoming_start_frame_id: str
    """Canonical frame ID immediately before the incoming transition."""
    incoming_successor_frame_id: str
    """Canonical frame ID immediately after the incoming transition."""
    incoming_start_simulator_step_count: int
    """Nonnegative simulator tick before the transition."""
    incoming_successor_simulator_step_count: int
    """Simulator tick after the transition; exactly one above the start tick."""
    agent_phase_trajectories: tuple[ReplayIncomingAgentPhaseTrajectoryV1, ...]
    """Unique ordered authorized trajectories used by every event anchor."""
    ordered_event_ids: tuple[str, ...]
    """Event IDs in payload order, with dense zero-based ordinals."""
    ordered_event_kinds: tuple[ReplayIncomingEventKindV1, ...]
    """Event discriminators in the same order as IDs and payloads."""
    events: tuple[ReplayIncomingEventV1, ...]
    """Concrete typed event rows in nondecreasing scientific phase order."""
    event_count: int
    """Nonnegative count equal to the ID, kind, and payload tuple lengths."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Join canonical episode/frame/transition IDs, adjacent ticks, dense event
        order, unique trajectories, and every anchor. Covering aura emitters retain
        trajectory order.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
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


type AgentPovVisualIncomingEventKindV1 = Literal[
    "action_rejected",
    "ability_activated",
    "recipient_health_resolution",
    "agent_left_combat",
    "health_regenerated",
    "cooldown_started",
    "cooldown_ready",
    "agent_died",
    "status_aged_to_zero",
    "status_broken_by_damage",
    "status_applied",
    "status_refreshed_or_extended",
    "status_cleared_by_new_death",
    "spawn_shield_expired",
    "agent_respawned",
]

_AGENT_POV_VISUAL_INCOMING_EVENT_KINDS_V1 = frozenset(
    (
        "action_rejected",
        "ability_activated",
        "recipient_health_resolution",
        "agent_left_combat",
        "health_regenerated",
        "cooldown_started",
        "cooldown_ready",
        "agent_died",
        "status_aged_to_zero",
        "status_broken_by_damage",
        "status_applied",
        "status_refreshed_or_extended",
        "status_cleared_by_new_death",
        "spawn_shield_expired",
        "agent_respawned",
    )
)


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentPovVisualIncomingAgentPhaseTrajectoryV1:
    """Store only authorized start and successor positions for an Agent POV agent.
    At least one endpoint exists; each present endpoint retains the same
    identity and its correct phase. No post-Charge position is carried.
    All constructor fields are required keyword arguments; instances are frozen.
    Attributes
    ----------
    agent_presentation_key : str
        Opaque identity shared by this agent trajectory and its anchors.
    agent_public_agent_id : str
        Public identity shared by this agent trajectory and its anchors.
    agent_class_id : int
        Canonical trajectory class ID from 1 through 5.
    transition_start : ReplayIncomingAgentAnchorV1 | None
        Authorized start anchor, or None when the agent was absent from that
        allowed endpoint.
    successor : ReplayIncomingAgentAnchorV1 | None
        Authorized successor anchor, or None when the agent is absent from that
        allowed endpoint.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG
    """Strict Pydantic wire settings: forbid extra fields, nonfinite numbers, and
    coercion."""

    agent_presentation_key: str
    """Opaque identity shared by this agent trajectory and its anchors."""
    agent_public_agent_id: str
    """Public identity shared by this agent trajectory and its anchors."""
    agent_class_id: int
    """Canonical trajectory class ID from 1 through 5."""
    transition_start: ReplayIncomingAgentAnchorV1 | None
    """Authorized start anchor, or None when the agent was absent from that allowed
    endpoint."""
    successor: ReplayIncomingAgentAnchorV1 | None
    """Authorized successor anchor, or None when the agent is absent from that
    allowed endpoint."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        At least one endpoint exists; each present endpoint retains the same
        identity and its correct phase. No post-Charge position is carried.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
        _require_text(self.agent_presentation_key, name="agent_presentation_key")
        _require_text(self.agent_public_agent_id, name="agent_public_agent_id")
        _require_python_int(self.agent_class_id, name="agent_class_id", minimum=1)
        if self.agent_class_id > 5:
            raise ValueError("agent_class_id must identify a real V1 class.")
        _require_optional_incoming_anchor(
            self.transition_start,
            name="transition_start",
            phase="transition_start",
        )
        _require_optional_incoming_anchor(
            self.successor,
            name="successor",
            phase="successor",
        )
        if self.transition_start is None and self.successor is None:
            raise ValueError(
                "Agent POV visual trajectories require at least one authorized anchor."
            )
        for anchor in (self.transition_start, self.successor):
            if anchor is not None and (
                anchor.presentation_key != self.agent_presentation_key
                or anchor.public_agent_id != self.agent_public_agent_id
            ):
                raise ValueError(
                    "Agent POV visual trajectory anchors must retain one identity."
                )


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentPovVisualIncomingRecipientHealthResolutionEventV1(
    _ReplayIncomingEventBaseV1
):
    """Show a visible recipient health result without hidden gross causes.
    Use phase rank 40 and a transition-start recipient anchor. Health values are
    nonnegative; signed net change equals their difference within the declared
    float tolerance.
    All constructor fields are required keyword arguments; instances are frozen.
    Inherited fields are documented on _ReplayIncomingEventBaseV1.
    Attributes
    ----------
    event_kind : Literal['recipient_health_resolution']
        Exact event discriminator for this concrete row type.
    recipient_anchor : ReplayIncomingAgentAnchorV1
        Authorized recipient identity and position at the event required phase.
    transition_start_health : float
        Recipient health before combat resolution, in hit points.
    health_after_combat_resolution : float
        Nonnegative health after combined combat resolution, in hit points.
    realized_net_health_change : float
        Signed health change in hit points: positive for gain, negative for
        loss.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    event_kind: Literal["recipient_health_resolution"]
    """Exact event discriminator for this concrete row type."""
    recipient_anchor: ReplayIncomingAgentAnchorV1
    """Authorized recipient identity and position at the event required phase."""
    transition_start_health: float
    """Recipient health before combat resolution, in hit points."""
    health_after_combat_resolution: float
    """Nonnegative health after combined combat resolution, in hit points."""
    realized_net_health_change: float
    """Signed health change in hit points: positive for gain, negative for loss."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Use phase rank 40 and a transition-start recipient anchor. Health values are
        nonnegative; signed net change equals their difference within the declared
        float tolerance.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
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
            "health_after_combat_resolution",
        ):
            _require_finite(cast(float, getattr(self, name)), name=name, minimum=0.0)
        _require_finite(
            self.realized_net_health_change,
            name="realized_net_health_change",
        )
        if not isclose(
            self.health_after_combat_resolution - self.transition_start_health,
            self.realized_net_health_change,
            rel_tol=1e-6,
            abs_tol=1e-5,
        ):
            raise ValueError(
                "Agent POV health result must equal its visible realized net change."
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentPovVisualIncomingAgentRespawnedEventV1(_ReplayIncomingEventBaseV1):
    """Show an authorized respawn cue without unjoined team metadata.
    Require event kind agent_respawned, phase rank 120, and a successor agent
    anchor.
    All constructor fields are required keyword arguments; instances are frozen.
    Inherited fields are documented on _ReplayIncomingEventBaseV1.
    Attributes
    ----------
    event_kind : Literal['agent_respawned']
        Exact event discriminator for this concrete row type.
    agent_anchor : ReplayIncomingAgentAnchorV1
        Authorized agent identity and position at the event required phase.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    event_kind: Literal["agent_respawned"]
    """Exact event discriminator for this concrete row type."""
    agent_anchor: ReplayIncomingAgentAnchorV1
    """Authorized agent identity and position at the event required phase."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Require event kind agent_respawned, phase rank 120, and a successor agent
        anchor.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
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


type AgentPovVisualIncomingEventV1 = Annotated[
    ReplayIncomingActionRejectedEventV1
    | ReplayIncomingAbilityActivatedEventV1
    | AgentPovVisualIncomingRecipientHealthResolutionEventV1
    | ReplayIncomingAgentLeftCombatEventV1
    | ReplayIncomingHealthRegeneratedEventV1
    | ReplayIncomingCooldownStartedEventV1
    | ReplayIncomingCooldownReadyEventV1
    | ReplayIncomingAgentDiedEventV1
    | ReplayIncomingStatusAgedToZeroEventV1
    | ReplayIncomingStatusBrokenByDamageEventV1
    | ReplayIncomingStatusAppliedEventV1
    | ReplayIncomingStatusRefreshedOrExtendedEventV1
    | ReplayIncomingStatusClearedByNewDeathEventV1
    | ReplayIncomingSpawnShieldExpiredEventV1
    | AgentPovVisualIncomingAgentRespawnedEventV1,
    Field(discriminator="event_kind"),
]

_AGENT_POV_VISUAL_INCOMING_EVENT_TYPES_V1: tuple[type[object], ...] = (
    ReplayIncomingActionRejectedEventV1,
    ReplayIncomingAbilityActivatedEventV1,
    AgentPovVisualIncomingRecipientHealthResolutionEventV1,
    ReplayIncomingAgentLeftCombatEventV1,
    ReplayIncomingHealthRegeneratedEventV1,
    ReplayIncomingCooldownStartedEventV1,
    ReplayIncomingCooldownReadyEventV1,
    ReplayIncomingAgentDiedEventV1,
    ReplayIncomingStatusAgedToZeroEventV1,
    ReplayIncomingStatusBrokenByDamageEventV1,
    ReplayIncomingStatusAppliedEventV1,
    ReplayIncomingStatusRefreshedOrExtendedEventV1,
    ReplayIncomingStatusClearedByNewDeathEventV1,
    ReplayIncomingSpawnShieldExpiredEventV1,
    AgentPovVisualIncomingAgentRespawnedEventV1,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentPovVisualIncomingSummaryV1:
    """Store fog-filtered events in a recipient-local namespace.
    Require adjacent ticks, dense local IDs/order, unique endpoint trajectories,
    and the exact recipient at both endpoints. Event anchors must join
    authorized endpoints; after-state facts need an authorized successor and
    rejections belong only to the recipient.
    All constructor fields are required keyword arguments; instances are frozen.
    Attributes
    ----------
    schema_version : Literal[1]
        Wire version discriminator; this model requires 1.
    summary_kind : Literal['agent_pov_fog_filtered_visual_events']
        Exact discriminator for the incoming inventory variant.
    source_episode_id : str
        Nonempty episode ID supplying the recorded event facts.
    recipient_public_agent_id : str
        Public ID of the Agent POV recipient authorized at both endpoints.
    recipient_presentation_key : str
        Stable recipient key shared by both adjacent authorized scenes.
    incoming_transition_index : int
        Zero-based index of the transition ending at the displayed frame.
    incoming_recipient_transition_id : str
        Transition ID in the actor-pov or shared-obs-visual-union recipient
        namespace.
    incoming_start_recipient_frame_id : str
        Recipient-local frame ID immediately before this transition.
    incoming_successor_recipient_frame_id : str
        Recipient-local frame ID immediately after this transition.
    incoming_start_simulator_step_count : int
        Nonnegative simulator tick before the transition.
    incoming_successor_simulator_step_count : int
        Simulator tick after the transition; exactly one above the start tick.
    agent_phase_trajectories : tuple[AgentPovVisualIncomingAgentPhaseTrajectoryV1,
    ...]
        Unique ordered authorized trajectories used by every event anchor.
    ordered_event_ids : tuple[str, ...]
        Event IDs in payload order, with dense zero-based ordinals.
    ordered_event_kinds : tuple[AgentPovVisualIncomingEventKindV1, ...]
        Event discriminators in the same order as IDs and payloads.
    events : tuple[AgentPovVisualIncomingEventV1, ...]
        Concrete typed event rows in nondecreasing scientific phase order.
    event_count : int
        Nonnegative count equal to the ID, kind, and payload tuple lengths.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG
    """Strict Pydantic wire settings: forbid extra fields, nonfinite numbers, and
    coercion."""

    schema_version: Literal[1]
    """Wire version discriminator; this model requires 1."""
    summary_kind: Literal["agent_pov_fog_filtered_visual_events"]
    """Exact discriminator for the incoming inventory variant."""
    source_episode_id: str
    """Nonempty episode ID supplying the recorded event facts."""
    recipient_public_agent_id: str
    """Public ID of the Agent POV recipient authorized at both endpoints."""
    recipient_presentation_key: str
    """Stable recipient key shared by both adjacent authorized scenes."""
    incoming_transition_index: int
    """Zero-based index of the transition ending at the displayed frame."""
    incoming_recipient_transition_id: str
    """Transition ID in the actor-pov or shared-obs-visual-union recipient
    namespace."""
    incoming_start_recipient_frame_id: str
    """Recipient-local frame ID immediately before this transition."""
    incoming_successor_recipient_frame_id: str
    """Recipient-local frame ID immediately after this transition."""
    incoming_start_simulator_step_count: int
    """Nonnegative simulator tick before the transition."""
    incoming_successor_simulator_step_count: int
    """Simulator tick after the transition; exactly one above the start tick."""
    agent_phase_trajectories: tuple[AgentPovVisualIncomingAgentPhaseTrajectoryV1, ...]
    """"""
    ordered_event_ids: tuple[str, ...]
    """Event IDs in payload order, with dense zero-based ordinals."""
    ordered_event_kinds: tuple[AgentPovVisualIncomingEventKindV1, ...]
    """Event discriminators in the same order as IDs and payloads."""
    events: tuple[AgentPovVisualIncomingEventV1, ...]
    """Concrete typed event rows in nondecreasing scientific phase order."""
    event_count: int
    """Nonnegative count equal to the ID, kind, and payload tuple lengths."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Require adjacent ticks, dense local IDs/order, unique endpoint trajectories,
        and the exact recipient at both endpoints. Event anchors must join
        authorized endpoints; after-state facts need an authorized successor and
        rejections belong only to the recipient.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
        if self.schema_version != AUTHORIZED_PRESENTATION_SCHEMA_VERSION:
            raise ValueError("unknown Agent POV visual incoming schema version.")
        if self.summary_kind != "agent_pov_fog_filtered_visual_events":
            raise ValueError("unknown Agent POV visual incoming summary kind.")
        for name in (
            "source_episode_id",
            "recipient_public_agent_id",
            "recipient_presentation_key",
            "incoming_recipient_transition_id",
            "incoming_start_recipient_frame_id",
            "incoming_successor_recipient_frame_id",
        ):
            _require_text(cast(str, getattr(self, name)), name=name)
        _require_python_int(
            self.incoming_transition_index,
            name="incoming_transition_index",
            minimum=0,
        )
        transition_suffix = f":transition:{self.incoming_transition_index}"
        if not self.incoming_recipient_transition_id.endswith(transition_suffix):
            raise ValueError(
                "Agent POV visual transition ID is not canonical for its index."
            )
        local_prefix = self.incoming_recipient_transition_id.removesuffix(
            transition_suffix
        )
        allowed_prefixes = (
            f"{self.source_episode_id}:actor-pov:{self.recipient_public_agent_id}",
            f"{self.source_episode_id}:shared-obs-visual-union:"
            f"{self.recipient_public_agent_id}",
        )
        if local_prefix not in allowed_prefixes:
            raise ValueError(
                "Agent POV visual epochs must use an exact recipient-local namespace."
            )
        if (
            self.incoming_start_recipient_frame_id
            != f"{local_prefix}:frame:{self.incoming_transition_index}"
            or self.incoming_successor_recipient_frame_id
            != f"{local_prefix}:frame:{self.incoming_transition_index + 1}"
        ):
            raise ValueError(
                "Agent POV visual frame IDs must join the local transition epoch."
            )
        for name in (
            "incoming_start_simulator_step_count",
            "incoming_successor_simulator_step_count",
            "event_count",
        ):
            _require_python_int(
                cast(int, getattr(self, name)),
                name=name,
                minimum=0,
            )
        if self.incoming_successor_simulator_step_count != (
            self.incoming_start_simulator_step_count + 1
        ):
            raise ValueError("Agent POV visual simulator ticks must be adjacent.")
        if (
            type(self.agent_phase_trajectories) is not tuple
            or not self.agent_phase_trajectories
            or any(
                type(value) is not AgentPovVisualIncomingAgentPhaseTrajectoryV1
                for value in self.agent_phase_trajectories
            )
            or type(self.events) is not tuple
            or any(
                type(value) not in _AGENT_POV_VISUAL_INCOMING_EVENT_TYPES_V1
                for value in self.events
            )
        ):
            raise ValueError(
                "Agent POV visual trajectories and events require exact rows."
            )
        if (
            type(self.ordered_event_ids) is not tuple
            or type(self.ordered_event_kinds) is not tuple
            or any(
                type(value) is not str or not value for value in self.ordered_event_ids
            )
            or any(
                type(value) is not str
                or value not in _AGENT_POV_VISUAL_INCOMING_EVENT_KINDS_V1
                for value in self.ordered_event_kinds
            )
        ):
            raise ValueError("Agent POV visual inventories require exact tuples.")
        if (
            self.event_count != len(self.ordered_event_ids)
            or self.event_count != len(self.ordered_event_kinds)
            or self.event_count != len(self.events)
        ):
            raise ValueError("Agent POV visual count must equal every event inventory.")
        expected_ids = tuple(
            f"{self.incoming_recipient_transition_id}:visual-event:{ordinal:04d}"
            for ordinal in range(self.event_count)
        )
        if self.ordered_event_ids != expected_ids or (
            tuple(event.ordinal for event in self.events)
            != tuple(range(self.event_count))
            or tuple(event.event_id for event in self.events) != self.ordered_event_ids
            or tuple(event.event_kind for event in self.events)
            != self.ordered_event_kinds
            or tuple(event.phase_rank for event in self.events)
            != tuple(sorted(event.phase_rank for event in self.events))
        ):
            raise ValueError(
                "Agent POV visual events must use dense local identity and order."
            )
        trajectory_by_key = {
            row.agent_presentation_key: row for row in self.agent_phase_trajectories
        }
        if len(trajectory_by_key) != len(self.agent_phase_trajectories) or len(
            {row.agent_public_agent_id for row in self.agent_phase_trajectories}
        ) != len(self.agent_phase_trajectories):
            raise ValueError("Agent POV visual trajectories require unique identities.")
        recipient_rows = tuple(
            row
            for row in self.agent_phase_trajectories
            if row.agent_public_agent_id == self.recipient_public_agent_id
            and row.agent_presentation_key == self.recipient_presentation_key
        )
        if len(recipient_rows) != 1:
            raise ValueError(
                "Agent POV visual trajectories must contain the exact recipient."
            )
        if (
            recipient_rows[0].transition_start is None
            or recipient_rows[0].successor is None
        ):
            raise ValueError(
                "Agent POV visual recipient must remain authorized at both endpoints."
            )
        for event in self.events:
            if type(event) is ReplayIncomingActionRejectedEventV1 and not (
                event.actor_configured_active
                and type(event.actor_identity)
                is ReplayIncomingAuthorizedAgentIdentityV1
                and event.actor_anchor is not None
                and event.actor_identity.public_agent_id
                == self.recipient_public_agent_id
                and event.actor_identity.presentation_key
                == self.recipient_presentation_key
                and event.actor_anchor.public_agent_id == self.recipient_public_agent_id
                and event.actor_anchor.presentation_key
                == self.recipient_presentation_key
            ):
                raise ValueError(
                    "Agent POV visual rejection must belong to its active recipient."
                )
            if type(event) is AgentPovVisualIncomingRecipientHealthResolutionEventV1:
                anchors = (event.recipient_anchor,)
            elif type(event) is AgentPovVisualIncomingAgentRespawnedEventV1:
                anchors = (event.agent_anchor,)
            else:
                anchors = _incoming_event_agent_anchors(
                    cast(ReplayIncomingEventV1, event)
                )
            for anchor in anchors:
                if anchor.phase == "post_charge":
                    raise ValueError(
                        "Agent POV visual events cannot retain post-Charge anchors."
                    )
                trajectory = trajectory_by_key.get(anchor.presentation_key)
                expected_anchor = (
                    None if trajectory is None else getattr(trajectory, anchor.phase)
                )
                if trajectory is None or (
                    trajectory.agent_public_agent_id != anchor.public_agent_id
                    or expected_anchor != anchor
                ):
                    raise ValueError(
                        "Agent POV visual anchors must join authorized trajectories."
                    )
            successor_required_anchors: tuple[ReplayIncomingAgentAnchorV1, ...] = ()
            if type(event) is AgentPovVisualIncomingRecipientHealthResolutionEventV1:
                successor_required_anchors = (event.recipient_anchor,)
            elif type(event) in (
                ReplayIncomingHealthRegeneratedEventV1,
                ReplayIncomingCooldownStartedEventV1,
                ReplayIncomingCooldownReadyEventV1,
            ):
                after_state_event = cast(
                    ReplayIncomingHealthRegeneratedEventV1
                    | ReplayIncomingCooldownStartedEventV1
                    | ReplayIncomingCooldownReadyEventV1,
                    event,
                )
                successor_required_anchors = (after_state_event.agent_anchor,)
            if any(
                trajectory_by_key[anchor.presentation_key].successor is None
                for anchor in successor_required_anchors
            ):
                raise ValueError(
                    "Agent POV visual after-state event requires an authorized "
                    "successor."
                )


@dataclass(frozen=True, slots=True, kw_only=True)
class SubmittedActionTupleV1:
    """Preserve a submitted action tuple, including invalid categories.
    Each component is an exact Python int in the signed 32-bit range. Domain
    rejection can therefore be reported without changing what was submitted.
    All constructor fields are required keyword arguments; instances are frozen.
    Attributes
    ----------
    move_action : int
        Recorded movement category; the enclosing tuple defines its integer
        domain.
    target_action : int
        Recorded actor-relative target category; zero means no target when
        accepted.
    use_ultimate_action : int
        Recorded Basic/Ultimate category; the enclosing tuple defines its
        integer domain.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG
    """Strict Pydantic wire settings: forbid extra fields, nonfinite numbers, and
    coercion."""

    move_action: int
    """Recorded movement category; the enclosing tuple defines its integer domain."""
    target_action: int
    """Recorded actor-relative target category; zero means no target when accepted."""
    use_ultimate_action: int
    """Recorded Basic/Ultimate category; the enclosing tuple defines its integer
    domain."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Each component is an exact Python int in the signed 32-bit range. Domain
        rejection can therefore be reported without changing what was submitted.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
        for name in ("move_action", "target_action", "use_ultimate_action"):
            value = cast(int, getattr(self, name))
            if type(value) is not int or not -(2**31) <= value <= 2**31 - 1:
                raise ValueError(f"{name} must be a signed 32-bit Python int.")


@dataclass(frozen=True, slots=True, kw_only=True)
class AcceptedActionTupleV1:
    """Store the recorded accepted action categories.
    Each exact Python int lies within its canonical movement, target, or
    Ultimate category count. The tuple does not recompute action acceptance.
    All constructor fields are required keyword arguments; instances are frozen.
    Attributes
    ----------
    move_action : int
        Recorded movement category; the enclosing tuple defines its integer
        domain.
    target_action : int
        Recorded actor-relative target category; zero means no target when
        accepted.
    use_ultimate_action : int
        Recorded Basic/Ultimate category; the enclosing tuple defines its
        integer domain.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG
    """Strict Pydantic wire settings: forbid extra fields, nonfinite numbers, and
    coercion."""

    move_action: int
    """Recorded movement category; the enclosing tuple defines its integer domain."""
    target_action: int
    """Recorded actor-relative target category; zero means no target when accepted."""
    use_ultimate_action: int
    """Recorded Basic/Ultimate category; the enclosing tuple defines its integer
    domain."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Each exact Python int lies within its canonical movement, target, or
        Ultimate category count. The tuple does not recompute action acceptance.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
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
    """Represent an accepted action with no selected target.
    Require the none discriminator. The enclosing inspection joins this variant
    to accepted target category zero.
    All constructor fields are required keyword arguments; instances are frozen.
    Attributes
    ----------
    target_kind : Literal['none']
        Exact discriminator for no target or an authorized target.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG
    """Strict Pydantic wire settings: forbid extra fields, nonfinite numbers, and
    coercion."""

    target_kind: Literal["none"]
    """Exact discriminator for no target or an authorized target."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Require the none discriminator. The enclosing inspection joins this variant
        to accepted target category zero.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
        if self.target_kind != "none":
            raise ValueError("unknown target-none discriminator.")


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayAcceptedAuthorizedTargetV1:
    """Join an accepted target to the displayed current scene.
    Require nonempty identity and a finite point. The builder owns actor-
    relative target-axis resolution because the HTTP envelope does not repeat
    the context catalog.
    All constructor fields are required keyword arguments; instances are frozen.
    Attributes
    ----------
    target_kind : Literal['authorized_agent']
        Exact discriminator for no target or an authorized target.
    target_presentation_key : str
        Opaque key for the accepted target in the displayed current scene.
    target_public_agent_id : str
        Public ID of that same accepted target.
    target_anchor : Point2D
        Accepted target center (x, y) in the displayed scene, in world units.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG
    """Strict Pydantic wire settings: forbid extra fields, nonfinite numbers, and
    coercion."""

    target_kind: Literal["authorized_agent"]
    """Exact discriminator for no target or an authorized target."""
    target_presentation_key: str
    """Opaque key for the accepted target in the displayed current scene."""
    target_public_agent_id: str
    """Public ID of that same accepted target."""
    target_anchor: Point2D
    """Accepted target center (x, y) in the displayed scene, in world units."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Require nonempty identity and a finite point. The builder owns actor-
        relative target-axis resolution because the HTTP envelope does not repeat
        the context catalog.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
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
    """Store recorded outgoing intent at the displayed frame.
    Keep submitted and accepted actions separate. Accepted lane follows the
    Ultimate head, and accepted target variant follows zero versus nonzero
    target category. The builder checks the frame/actor/target joins.
    All constructor fields are required keyword arguments; instances are frozen.
    Attributes
    ----------
    inspection_kind : Literal['replay_recorded_outgoing_action']
        replay_recorded_outgoing_action discriminator.
    outgoing_transition_index : int
        Zero-based transition index starting at the displayed frame.
    outgoing_transition_id : str
        Canonical ID of the recorded outgoing transition.
    outgoing_start_frame_id : str
        Displayed frame ID at which the recorded action was submitted.
    outgoing_successor_frame_id : str
        Recorded ID of the next frame; its scene is not used for current
        anchors.
    actor_presentation_key : str
        Opaque key for the selected actor in the displayed scene.
    actor_public_agent_id : str
        Public ID of the selected actor.
    actor_anchor : Point2D
        Selected actor center (x, y) in the displayed current scene, in world
        units.
    submitted_action : SubmittedActionTupleV1
        Recorded submitted integer tuple, including rejected out-of-domain
        values.
    accepted_action : AcceptedActionTupleV1
        Recorded action after simulator acceptance, bounded to canonical
        categories.
    accepted_lane : AcceptedLaneV1
        basic or ultimate, matching the accepted Ultimate action head.
    accepted_target : ReplayAcceptedTargetV1
        No-target variant for category zero, otherwise the joined authorized
        target.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG
    """Strict Pydantic wire settings: forbid extra fields, nonfinite numbers, and
    coercion."""

    inspection_kind: Literal["replay_recorded_outgoing_action"]
    """replay_recorded_outgoing_action discriminator."""
    outgoing_transition_index: int
    """Zero-based transition index starting at the displayed frame."""
    outgoing_transition_id: str
    """Canonical ID of the recorded outgoing transition."""
    outgoing_start_frame_id: str
    """Displayed frame ID at which the recorded action was submitted."""
    outgoing_successor_frame_id: str
    """Recorded ID of the next frame; its scene is not used for current anchors."""
    actor_presentation_key: str
    """Opaque key for the selected actor in the displayed scene."""
    actor_public_agent_id: str
    """Public ID of the selected actor."""
    actor_anchor: Point2D
    """Selected actor center (x, y) in the displayed current scene, in world units."""
    submitted_action: SubmittedActionTupleV1
    """Recorded submitted integer tuple, including rejected out-of-domain values."""
    accepted_action: AcceptedActionTupleV1
    """Recorded action after simulator acceptance, bounded to canonical categories."""
    accepted_lane: AcceptedLaneV1
    """basic or ultimate, matching the accepted Ultimate action head."""
    accepted_target: ReplayAcceptedTargetV1
    """No-target variant for category zero, otherwise the joined authorized target."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Keep submitted and accepted actions separate. Accepted lane follows the
        Ultimate head, and accepted target variant follows zero versus nonzero
        target category. The builder checks the frame/actor/target joins.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
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
    """Package current Oracle scene, incoming history, and optional outgoing action.
    Incoming successor identities and positions equal the current scene in
    order. Feed-only rejected actors cannot invent a scene body. The outer
    protocol adds session and cursor authority.
    All constructor fields are required keyword arguments; instances are frozen.
    Attributes
    ----------
    current_scene : AuthorizedBattlefieldSceneV1
        Authorized durable scene at the displayed replay frame.
    incoming_summary : ReplayIncomingSummaryV1 | None
        History ending at this scene, or None only at frame zero.
    outgoing_inspection : ReplayOutgoingInspectionV1 | None
        Selected recorded action starting here, or None without eligible
        inspection.
    Raises
    ------
    ValueError
        A field or cross-field invariant described above is invalid.
    """

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_DATACLASS_CONFIG
    """Strict Pydantic wire settings: forbid extra fields, nonfinite numbers, and
    coercion."""

    current_scene: AuthorizedBattlefieldSceneV1
    """Authorized durable scene at the displayed replay frame."""
    incoming_summary: ReplayIncomingSummaryV1 | None
    """History ending at this scene, or None only at frame zero."""
    outgoing_inspection: ReplayOutgoingInspectionV1 | None
    """Selected recorded action starting here, or None without eligible inspection."""

    def __post_init__(self) -> None:
        """Validate this row after construction.
        Incoming successor identities and positions equal the current scene in
        order. Feed-only rejected actors cannot invent a scene body. The outer
        protocol adds session and cursor authority.
        Raises
        ------
        ValueError
            A required type, value, identity, or relationship is invalid.
        """
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
    """Copy a validated obstacle into the strict presentation row.
    Keep shape, dimensions, position, and rotation unchanged.
    Parameters
    ----------
    obstacle : ObstacleSceneV1
        Validated source obstacle row.
    Returns
    -------
    AuthorizedObstacleV1
        New frozen obstacle row.
    Raises
    ------
    ValueError
        Copied values violate the destination row contract.
    """
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
    """Copy map bounds and ordered obstacles into presentation rows.
    Preserve source order and validate destination dimensions and unique
    obstacle IDs.
    Parameters
    ----------
    scene_map : MapSceneV1
        Validated source map bounds and obstacle rows.
    Returns
    -------
    AuthorizedMapV1
        New frozen map row.
    Raises
    ------
    ValueError
        A copied map or obstacle invariant is invalid.
    """
    return AuthorizedMapV1(
        width=scene_map.width,
        height=scene_map.height,
        obstacles=tuple(_authorized_obstacle(row) for row in scene_map.obstacles),
    )


def _authorized_aura_modifier(
    modifier: AuraRecipientModifierSceneV2,
) -> AuthorizedAuraModifierV1:
    """Copy one non-neutral recipient aura multiplier into presentation.
    Callers omit neutral values; the destination validator rejects 1.0.
    Parameters
    ----------
    modifier : AuraRecipientModifierSceneV2
        Source recipient aura multiplier row.
    Returns
    -------
    AuthorizedAuraModifierV1
        New frozen aggregate modifier.
    Raises
    ------
    ValueError
        The aura ID or multiplier is invalid or neutral.
    """
    return AuthorizedAuraModifierV1(
        aura_id=modifier.aura_id,
        multiplier=modifier.multiplier,
    )


def _authorized_class_status_mechanic(
    mechanic: ClassStatusMechanicSceneV2,
) -> AuthorizedClassStatusMechanicV1:
    """Copy one class status mechanic into its strict presentation form.
    Preserve catalog channel/ID, duration, magnitude, and damage-break meaning.
    Parameters
    ----------
    mechanic : ClassStatusMechanicSceneV2
        Validated source class status or aura mechanic.
    Returns
    -------
    AuthorizedClassStatusMechanicV1
        New frozen status mechanic.
    Raises
    ------
    ValueError
        Copied catalog fields violate the destination contract.
    """
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
    """Copy one class aura mechanic into its strict presentation form.
    Preserve the configured radius, factor, stacking rule, and clamp.
    Parameters
    ----------
    mechanic : ClassAuraMechanicSceneV2
        Validated source class status or aura mechanic.
    Returns
    -------
    AuthorizedClassAuraMechanicV1
        New frozen aura mechanic.
    Raises
    ------
    ValueError
        Copied aura fields violate the destination contract.
    """
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
    *,
    documentation_profile: AuthorizedClassDocumentationProfileV1,
) -> AuthorizedClassMechanicsV2:
    """Copy class mechanics and attach the checked guide profile.
    Build the V2 row and preserve status/aura order. This helper does not
    independently certify the supplied documentation profile.
    Parameters
    ----------
    mechanics : ClassMechanicsSceneV2
        Source class mechanics for the same agent class.
    documentation_profile : AuthorizedClassDocumentationProfileV1
        Catalog-checked availability result for authored class-guide prose.
    Returns
    -------
    AuthorizedClassMechanicsV2
        New frozen mechanics row with the supplied profile.
    Raises
    ------
    ValueError
        Copied fields or profile variant violate the row contract.
    """
    return AuthorizedClassMechanicsV2(
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
        mechanics_version=2,
        documentation_profile=documentation_profile,
    )


def _authorized_respawn_wave(
    wave: RespawnWaveSceneV2,
) -> AuthorizedRespawnWaveV1:
    """Copy one team wave countdown into its strict presentation form.
    Preserve team identity, period, and remaining ticks.
    Parameters
    ----------
    wave : RespawnWaveSceneV2
        Validated source team respawn countdown.
    Returns
    -------
    AuthorizedRespawnWaveV1
        New frozen wave row.
    Raises
    ------
    ValueError
        The team or countdown values are inconsistent.
    """
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
    """Project a status and replace direct-source slots with authorized identities.
    Require each evidence source to join the supplied key/public-ID maps, and
    keep its first occurrence only. Do not carry event IDs or raw source slots
    into the result.
    Parameters
    ----------
    status : StatusSceneV2
        Source durable status, including recorded direct-source evidence.
    configured_duration_steps : int
        Positive configured duration from the matching status catalog, in ticks.
    key_by_internal_slot : dict[int, str]
        Lookup from configured-active internal slots to unique authorized opaque
        keys.
    public_id_by_internal_slot : dict[int, str]
        Lookup from those same internal slots to public agent IDs.
    Returns
    -------
    AuthorizedStatusV1
        New durable status with deduplicated authorized source rows.
    Raises
    ------
    ValueError
        A source identity is absent/mismatched or a destination field is
        invalid.
    """
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
    """Project one Oracle agent with its matching class/status mechanics.
    Require exact class ID, maximum health, and body radius agreement. Resolve
    source identities, preserve durable state, and omit neutral aura modifiers.
    Parameters
    ----------
    agent : AgentSceneV2
        Source durable agent row from the Oracle scene.
    mechanics : ClassMechanicsSceneV2
        Source class mechanics for the same agent class.
    status_mechanics_by_channel : dict[int, ClassStatusMechanicSceneV2]
        Status catalog indexed by the fixed channel axis.
    key_by_internal_slot : dict[int, str]
        Lookup from configured-active internal slots to unique authorized opaque
        keys.
    public_id_by_internal_slot : dict[int, str]
        Lookup from those same internal slots to public agent IDs.
    Returns
    -------
    AuthorizedAgentV1
        New Oracle agent row with no internal slot field.
    Raises
    ------
    ValueError, KeyError
        Static facts disagree, a destination invariant fails, or required lookup
        entries are absent.
    """
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
    context: EvaluationEpisodeContext,
    scene: BattlefieldSceneV2,
    *,
    authority_session_id: str,
) -> tuple[AuthorizedBattlefieldSceneV1, dict[int, str]]:
    """Build an Oracle presentation scene and its internal-to-public key lookup.
    Require configured-active roster order and complete source class/status/aura
    axes. Project only represented classes, join aura emitters and pads, and
    attach configured Spawn Shield facts and checked guide availability. A
    context whose resolved config is V2 (it records the Red Zone depth) gives an
    AuthorizedMapV2 built from the config's depth and spawn pads; a V1 config
    gives the unchanged AuthorizedMapV1.
    Parameters
    ----------
    context : EvaluationEpisodeContext
        Validated episode context with roster, resolved configuration, and
        public catalog.
    scene : BattlefieldSceneV2
        Displayed durable Oracle scene for the same episode.
    authority_session_id : str
        Nonempty authority/session namespace for opaque presentation keys.
    Returns
    -------
    tuple[AuthorizedBattlefieldSceneV1, dict[int, str]]
        Frozen scene plus the temporary slot-to-key map used for adjacent
        branches.
    Raises
    ------
    ValueError
        Roster, source axes, row identity, or a destination invariant is
        inconsistent.
    """
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
    documentation_profile = authorized_class_documentation_profile_v1(
        context.static_mechanics_catalog
    )
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
    scene_map: AuthorizedMap = _authorized_map(scene.map)
    config = context.resolved_env_config
    if type(config) is ResolvedEnvConfigV2:
        scene_map = build_authorized_map_v2(
            scene_map,
            red_zone_depth=config.team_deathmatch_red_zone_depth,
            team_spawn_pad_x=tuple(
                tuple(pad[0] for pad in team)
                for team in config.team_spawn_pad_positions
            ),
        )
    return (
        AuthorizedBattlefieldSceneV1(
            schema_version=AUTHORIZED_PRESENTATION_SCHEMA_VERSION,
            map=scene_map,
            agents=agents,
            aura_fields=tuple(aura_fields),
            class_mechanics=tuple(
                _authorized_class_mechanics(
                    row,
                    documentation_profile=documentation_profile,
                )
                for row in scene.class_mechanics
                if row.class_id in represented_class_ids
            ),
            spawn_shield_mechanics=AuthorizedSpawnShieldMechanicsAvailableV2(
                availability_kind="available_v2",
                configured_duration_steps=(
                    context.resolved_env_config.spawn_shield_duration_steps
                ),
                movement_speed=(
                    context.resolved_env_config.spawn_shield_movement_speed
                ),
                protection_effect="invulnerable",
                visibility_effect="concealed_from_opponents",
                targetability_effect="untargetable",
                action_scope="movement_only",
                aura_effect="excluded_as_emitter_and_beneficiary",
                agent_collision_effect=("phased_until_expiring_endpoint_rejoin"),
                ordinary_application_mechanism=("end_of_transition_respawn_lifecycle"),
            ),
            spawn_pads=tuple(spawn_pads),
            respawn_waves=tuple(
                _authorized_respawn_wave(row) for row in scene.respawn_waves
            ),
        ),
        key_by_internal_slot,
    )


def build_oracle_authorized_scene_v1(
    context: EvaluationEpisodeContext,
    scene: BattlefieldSceneV2,
    *,
    authority_session_id: str,
) -> AuthorizedBattlefieldSceneV1:
    """Validate and project one current Oracle scene without adjacent replay branches.
    Round-trip context and scene through strict wire validation, reject changed
    runtime types or hidden model fields, and join episode/static authority.
    Perform host-side validation only; inputs are unchanged and no I/O or
    simulator step runs.
    Parameters
    ----------
    context : EvaluationEpisodeContext
        Validated episode context with roster, resolved configuration, and
        public catalog.
    scene : BattlefieldSceneV2
        Displayed durable Oracle scene for the same episode.
    authority_session_id : str
        Nonempty authority/session namespace for opaque presentation keys.
    Returns
    -------
    AuthorizedBattlefieldSceneV1
        Frozen durable scene whose keys belong to authority_session_id.
    Raises
    ------
    TypeError, ValueError
        Input roots, runtime wire types, episode identity, or static/durable
        joins are invalid.
    """
    evaluation_context_type(context)
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
    validated_context = evaluation_context_type(context).model_validate_json(
        context_json
    )
    validated_scene = scene_adapter.validate_json(scene_json)

    def exact_tree_matches(candidate: object, canonical: object) -> bool:
        """Check that strict wire validation retained exact runtime types and values.
        Recurse through models, dataclasses, tuples, and ordered dictionaries.
        Reject extra/private model state and scalar type coercion.
        Parameters
        ----------
        candidate : object
            Original runtime tree whose exact types and values must be retained.
        canonical : object
            Tree reconstructed by strict wire validation.
        Returns
        -------
        bool
            Whether candidate exactly matches the canonical runtime tree.
        """
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
    """Replace one recorded anchor slot with its authorized presentation key.
    Preserve phase, public ID, and position. The enclosing trajectory/summary
    checks the complete identity join.
    Parameters
    ----------
    anchor : VisualAgentAnchorV2
        Recorded agent anchor at one scientific phase.
    key_by_internal_slot : dict[int, str]
        Lookup from configured-active internal slots to unique authorized opaque
        keys.
    Returns
    -------
    ReplayIncomingAgentAnchorV1
        New slot-free anchor.
    Raises
    ------
    ValueError
        The internal slot lacks an authorized key or the row fields are invalid.
    """
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
    """Project all three recorded trajectory anchors to authorized keys.
    Preserve start, post-Charge, and successor positions and their order.
    Parameters
    ----------
    trajectory : VisualAgentPhaseTrajectoryV2
        Recorded start/post-Charge/successor trajectory for one agent.
    key_by_internal_slot : dict[int, str]
        Lookup from configured-active internal slots to unique authorized opaque
        keys.
    Returns
    -------
    ReplayIncomingAgentPhaseTrajectoryV1
        New slot-free trajectory.
    Raises
    ------
    ValueError
        The agent has no key or its anchor identities/phases disagree.
    """
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
    """Resolve one internal slot and scientific phase to an authorized anchor.
    Use the recorded trajectory; never invent a coordinate from the displayed
    scene.
    Parameters
    ----------
    internal_slot : int
        Internal roster slot used only to resolve the authorized anchor.
    phase : VisualAnchorPhaseV2
        Required scientific anchor phase.
    trajectory_by_internal_slot : dict[int, VisualAgentPhaseTrajectoryV2]
        Recorded canonical trajectories indexed by internal roster slot.
    key_by_internal_slot : dict[int, str]
        Lookup from configured-active internal slots to unique authorized opaque
        keys.
    Returns
    -------
    ReplayIncomingAgentAnchorV1
        Projected anchor for the requested phase.
    Raises
    ------
    ValueError
        The trajectory or authorized key is missing.
    """
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
    """Project one canonical visual event without changing its recorded meaning.
    Replace slot references with authorized anchors and preserve event identity,
    order, numeric facts, and atomic event kind. Inactive rejected actors stay
    feed-only; source and recipient phases come from recorded evidence.
    Parameters
    ----------
    event : VisualEventV2
        Typed event whose anchors or authorized payload are needed.
    trajectory_by_internal_slot : dict[int, VisualAgentPhaseTrajectoryV2]
        Recorded canonical trajectories indexed by internal roster slot.
    key_by_internal_slot : dict[int, str]
        Lookup from configured-active internal slots to unique authorized opaque
        keys.
    Returns
    -------
    ReplayIncomingEventV1
        Exact matching incoming-event variant.
    Raises
    ------
    TypeError, ValueError
        The event kind is unsupported, an identity lookup fails, or projected
        fields are invalid.
    """
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
    if type(event) is AgentLeftCombatEventV2:
        return ReplayIncomingAgentLeftCombatEventV1(
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
    if type(event) is TeamDeathmatchScoreChangedEventV2:
        return ReplayIncomingTeamDeathmatchScoreChangedEventV1(
            event_id=event.event_id,
            ordinal=event.ordinal,
            phase_rank=event.phase_rank,
            event_kind=event.event_type,
            team_index=event.team_index,
            team_id=event.team_id,
            score_increment=event.score_increment,
            previous_score=event.previous_score,
            successor_score=event.successor_score,
            team_anchor=ReplayIncomingTeamAnchorV1(
                phase="successor",
                team_index=event.team_index,
                team_id=event.team_id,
            ),
        )
    if type(event) is TeamDeathmatchCompletedEventV2:
        return ReplayIncomingTeamDeathmatchCompletedEventV1(
            event_id=event.event_id,
            ordinal=event.ordinal,
            phase_rank=event.phase_rank,
            event_kind=event.event_type,
            outcome=event.outcome,
            completion_basis=event.completion_basis,
        )
    raise TypeError("unsupported canonical V2 event kind.")


def _visual_event_agent_anchors(
    event: VisualEventV2,
    *,
    trajectory_by_internal_slot: dict[int, VisualAgentPhaseTrajectoryV2],
) -> tuple[VisualAgentAnchorV2, ...]:
    """Collect the source visual anchors needed to authorize an event.
    Read explicit anchor fields and add transition-start anchors for recorded
    damage-aura emitters. Preserve order and duplicates.
    Parameters
    ----------
    event : VisualEventV2
        Typed event whose anchors or authorized payload are needed.
    trajectory_by_internal_slot : dict[int, VisualAgentPhaseTrajectoryV2]
        Recorded canonical trajectories indexed by internal roster slot.
    Returns
    -------
    tuple[VisualAgentAnchorV2, ...]
        All referenced agent anchors to check against visible endpoint scenes.
    Raises
    ------
    ValueError
        An explicit anchor has the wrong type or an aura emitter has no
        trajectory.
    """
    anchors: list[VisualAgentAnchorV2] = []
    for field_name in (
        "actor_anchor",
        "source_anchor",
        "recipient_anchor",
        "agent_anchor",
        "start_anchor",
        "end_anchor",
    ):
        anchor = getattr(event, field_name, None)
        if anchor is not None:
            if type(anchor) is not VisualAgentAnchorV2:
                raise ValueError("visual events must retain exact agent anchors.")
            anchors.append(anchor)
    if type(event) is SourceDamageOutputEventV2:
        for slot in (
            *event.mage_damage_aura_covering_emitter_global_slots,
            *event.warrior_mitigation_aura_covering_emitter_global_slots,
        ):
            trajectory = trajectory_by_internal_slot.get(slot)
            if trajectory is None:
                raise ValueError(
                    "visual damage aura emitters must join an active trajectory."
                )
            anchors.append(trajectory.transition_start)
    return tuple(anchors)


def _agent_pov_visual_scene_agents_by_slot(
    scene: AuthorizedBattlefieldSceneV1,
    *,
    scene_name: str,
    phase: Literal["transition_start", "successor"],
    recipient_public_agent_id: str,
    slot_by_public_agent_id: dict[str, int],
    trajectory_by_internal_slot: dict[int, VisualAgentPhaseTrajectoryV2],
    configured_active_by_global_slot: tuple[bool, ...],
) -> dict[int, AuthorizedAgentV1]:
    """Join an Agent POV endpoint scene to canonical recorded trajectories.
    Recheck scene invariants, forbid Oracle relations, require the exact self
    recipient, and join configured-active public IDs and positions at the
    requested endpoint.
    Parameters
    ----------
    scene : AuthorizedBattlefieldSceneV1
        Displayed durable Oracle scene for the same episode.
    scene_name : str
        Input label included in validation errors.
    phase : Literal['transition_start', 'successor']
        Required scientific anchor phase.
    recipient_public_agent_id : str
        Public ID of the single self recipient at both authorized endpoints.
    slot_by_public_agent_id : dict[str, int]
        Canonical roster lookup from public IDs to internal slots.
    trajectory_by_internal_slot : dict[int, VisualAgentPhaseTrajectoryV2]
        Recorded canonical trajectories indexed by internal roster slot.
    configured_active_by_global_slot : tuple[bool, ...]
        Fixed roster-axis bool tuple identifying configured-active slots.
    Returns
    -------
    dict[int, AuthorizedAgentV1]
        Authorized scene bodies indexed temporarily by internal slot.
    Raises
    ------
    ValueError
        A scene, recipient, identity, activity, position, or uniqueness check
        fails.
    """
    if type(scene) is not AuthorizedBattlefieldSceneV1:
        raise ValueError(f"{scene_name} must be an exact authorized scene.")
    AuthorizedBattlefieldSceneV1.__post_init__(scene)
    if any(agent.relation == "oracle" for agent in scene.agents):
        raise ValueError(f"{scene_name} must use Agent POV authority.")
    self_rows = tuple(agent for agent in scene.agents if agent.relation == "self")
    if len(self_rows) != 1 or self_rows[0].public_agent_id != recipient_public_agent_id:
        raise ValueError(f"{scene_name} must contain the exact POV recipient.")
    agents_by_slot: dict[int, AuthorizedAgentV1] = {}
    for agent in scene.agents:
        slot = slot_by_public_agent_id.get(agent.public_agent_id)
        trajectory = None if slot is None else trajectory_by_internal_slot.get(slot)
        if (
            slot is None
            or not configured_active_by_global_slot[slot]
            or trajectory is None
        ):
            raise ValueError(
                f"{scene_name} agents must join configured-active visual trajectories."
            )
        canonical_anchor = cast(
            VisualAgentAnchorV2,
            getattr(trajectory, phase),
        )
        if (
            canonical_anchor.public_agent_id != agent.public_agent_id
            or canonical_anchor.position != agent.position
        ):
            raise ValueError(
                f"{scene_name} positions must join canonical visual trajectories."
            )
        if slot in agents_by_slot:
            raise ValueError(f"{scene_name} cannot duplicate a visual trajectory.")
        agents_by_slot[slot] = agent
    return agents_by_slot


def _agent_pov_visual_event_is_authorized(
    event: VisualEventV2,
    *,
    recipient_global_slot: int,
    transition_start_agents_by_slot: dict[int, AuthorizedAgentV1],
    successor_agents_by_slot: dict[int, AuthorizedAgentV1],
    trajectory_by_internal_slot: dict[int, VisualAgentPhaseTrajectoryV2],
) -> bool:
    """Check whether an event can be shown from the two allowed Agent POV endpoints.
    Omit hidden gross causes, global team events, movement-phase details, and
    other actors rejection facts. Require every anchor at its authorized
    endpoint and a successor for after-state facts; check visible start health
    for health resolution.
    Parameters
    ----------
    event : VisualEventV2
        Typed event whose anchors or authorized payload are needed.
    recipient_global_slot : int
        Internal slot of the authorized POV recipient.
    transition_start_agents_by_slot : dict[int, AuthorizedAgentV1]
        Agent-input-authorized bodies at the start endpoint, indexed by slot.
    successor_agents_by_slot : dict[int, AuthorizedAgentV1]
        Agent-input-authorized bodies at the successor endpoint, indexed by
        slot.
    trajectory_by_internal_slot : dict[int, VisualAgentPhaseTrajectoryV2]
        Recorded canonical trajectories indexed by internal roster slot.
    Returns
    -------
    bool
        Whether the event is permitted for projection.
    Raises
    ------
    ValueError
        A retained health fact disagrees with visible start health or an anchor
        lookup is invalid.
    """
    if type(event) in (
        ChargePhaseDisplacementEventV2,
        OrdinaryMovementPhaseDisplacementEventV2,
        RespawnWaveOccurredEventV2,
        SourceDamageOutputEventV2,
        SourceHealingOutputEventV2,
        CombatCountdownResetEventV2,
        LethalDamageContributionEventV2,
        TeamDeathmatchScoreChangedEventV2,
        TeamDeathmatchCompletedEventV2,
    ):
        return False
    if type(event) is ActionRejectedEventV2 and (
        not event.actor_configured_active
        or event.actor_global_slot != recipient_global_slot
    ):
        return False
    for anchor in _visual_event_agent_anchors(
        event,
        trajectory_by_internal_slot=trajectory_by_internal_slot,
    ):
        if anchor.phase == "transition_start":
            if anchor.global_slot not in transition_start_agents_by_slot:
                return False
        elif anchor.phase == "successor":
            if anchor.global_slot not in successor_agents_by_slot:
                return False
        else:
            return False
    successor_required_slots: tuple[int, ...] = ()
    if type(event) is RecipientHealthResolutionEventV2:
        successor_required_slots = (event.recipient_global_slot,)
    elif type(event) in (
        HealthRegeneratedEventV2,
        CooldownStartedEventV2,
        CooldownReadyEventV2,
    ):
        after_state_event = cast(
            HealthRegeneratedEventV2 | CooldownStartedEventV2 | CooldownReadyEventV2,
            event,
        )
        successor_required_slots = (after_state_event.agent_global_slot,)
    if any(slot not in successor_agents_by_slot for slot in successor_required_slots):
        return False
    if type(event) is RecipientHealthResolutionEventV2:
        recipient = transition_start_agents_by_slot.get(event.recipient_global_slot)
        if recipient is not None and not isclose(
            recipient.current_health,
            event.transition_start_health,
            rel_tol=1e-6,
            abs_tol=1e-5,
        ):
            raise ValueError(
                "visible health result must join transition-start scene health."
            )
    return True


def _agent_pov_corpse_choreography_agents_by_slot(
    scene: AuthorizedBattlefieldSceneV1,
    *,
    base_agents_by_slot: dict[int, AuthorizedAgentV1],
    scene_name: str,
    phase: Literal["transition_start", "successor"],
    recipient_public_agent_id: str,
    slot_by_public_agent_id: dict[str, int],
    trajectory_by_internal_slot: dict[int, VisualAgentPhaseTrajectoryV2],
    configured_active_by_global_slot: tuple[bool, ...],
) -> dict[int, AuthorizedAgentV1]:
    """Validate an endpoint scene extended only with permitted corpse rows.
    Preserve every base actor-input row exactly. Additional rows must be non-
    self corpses with zero health within absolute tolerance 1e-8 and must join
    canonical endpoint trajectories.
    Parameters
    ----------
    scene : AuthorizedBattlefieldSceneV1
        Displayed durable Oracle scene for the same episode.
    base_agents_by_slot : dict[int, AuthorizedAgentV1]
        Base actor-input scene rows that a corpse extension must preserve
        exactly.
    scene_name : str
        Input label included in validation errors.
    phase : Literal['transition_start', 'successor']
        Required scientific anchor phase.
    recipient_public_agent_id : str
        Public ID of the single self recipient at both authorized endpoints.
    slot_by_public_agent_id : dict[str, int]
        Canonical roster lookup from public IDs to internal slots.
    trajectory_by_internal_slot : dict[int, VisualAgentPhaseTrajectoryV2]
        Recorded canonical trajectories indexed by internal roster slot.
    configured_active_by_global_slot : tuple[bool, ...]
        Fixed roster-axis bool tuple identifying configured-active slots.
    Returns
    -------
    dict[int, AuthorizedAgentV1]
        Base and corpse rows indexed by internal slot.
    Raises
    ------
    ValueError
        A base row changed or an added row/identity is not a valid corpse
        extension.
    """
    agents_by_slot = _agent_pov_visual_scene_agents_by_slot(
        scene,
        scene_name=scene_name,
        phase=phase,
        recipient_public_agent_id=recipient_public_agent_id,
        slot_by_public_agent_id=slot_by_public_agent_id,
        trajectory_by_internal_slot=trajectory_by_internal_slot,
        configured_active_by_global_slot=configured_active_by_global_slot,
    )
    for slot, base_agent in base_agents_by_slot.items():
        if agents_by_slot.get(slot) != base_agent:
            raise ValueError(f"{scene_name} changed an actor-input scene agent.")
    for slot, agent in agents_by_slot.items():
        if slot in base_agents_by_slot:
            continue
        if (
            agent.relation == "self"
            or agent.life_state != "corpse"
            or not isclose(
                agent.current_health,
                0.0,
                rel_tol=0.0,
                abs_tol=1e-8,
            )
        ):
            raise ValueError(f"{scene_name} may add only zero-health corpse rows.")
    return agents_by_slot


def build_agent_pov_visual_incoming_summary_v1(
    incoming_events: VisualEventBatchV2,
    *,
    transition_start_scene: AuthorizedBattlefieldSceneV1,
    successor_scene: AuthorizedBattlefieldSceneV1,
    transition_start_corpse_choreography_scene: (
        AuthorizedBattlefieldSceneV1 | None
    ) = None,
    successor_corpse_choreography_scene: AuthorizedBattlefieldSceneV1 | None = None,
    recipient_public_agent_id: str,
    incoming_recipient_transition_id: str,
    incoming_start_recipient_frame_id: str,
    incoming_successor_recipient_frame_id: str,
) -> AgentPovVisualIncomingSummaryV1:
    """Project only incoming visual facts allowed by adjacent Agent POV scenes.
    Validate recorded events and both recipient endpoints. Optional corpse
    scenes must be supplied together and can extend death/respawn cues only.
    Omit hidden gross damage/healing causes and post-Charge anchors, preserve
    allowed endpoints, and assign dense recipient-local event IDs. Inputs are
    not mutated; no simulator or I/O work runs.
    Parameters
    ----------
    incoming_events : VisualEventBatchV2
        Canonical recorded visual batch for one incoming transition.
    transition_start_scene : AuthorizedBattlefieldSceneV1
        Authorized actor-input scene before the transition, including its self
        recipient.
    successor_scene : AuthorizedBattlefieldSceneV1
        Authorized actor-input scene after the same transition and for the same
        recipient.
    transition_start_corpse_choreography_scene : AuthorizedBattlefieldSceneV1 | None
        Optional start scene that may add only zero-health non-self corpses;
        defaults to None.
    successor_corpse_choreography_scene : AuthorizedBattlefieldSceneV1 | None
        Matching optional successor corpse scene; defaults to None and must be
        supplied with the start scene.
    recipient_public_agent_id : str
        Public ID of the single self recipient at both authorized endpoints.
    incoming_recipient_transition_id : str
        Canonical recipient-local transition ID, using actor-pov or shared-obs-
        visual-union.
    incoming_start_recipient_frame_id : str
        Recipient-local start frame ID at the incoming transition index.
    incoming_successor_recipient_frame_id : str
        Recipient-local successor frame ID at incoming index plus one.
    Returns
    -------
    AgentPovVisualIncomingSummaryV1
        Frozen recipient-local trajectories and allowed event payloads.
    Raises
    ------
    ValueError
        Input types, episode/recipient identities, endpoint evidence, corpse
        pairing, or summary invariants are inconsistent.
    """
    if type(incoming_events) is not VisualEventBatchV2:
        raise ValueError("incoming_events must be an exact VisualEventBatchV2.")
    VisualEventBatchV2.__post_init__(incoming_events)
    _require_text(
        recipient_public_agent_id,
        name="recipient_public_agent_id",
    )
    for name, value in (
        ("incoming_recipient_transition_id", incoming_recipient_transition_id),
        ("incoming_start_recipient_frame_id", incoming_start_recipient_frame_id),
        (
            "incoming_successor_recipient_frame_id",
            incoming_successor_recipient_frame_id,
        ),
    ):
        _require_text(value, name=name)
    slot_by_public_agent_id = {
        public_agent_id: slot
        for slot, public_agent_id in enumerate(
            incoming_events.public_agent_id_by_global_slot
        )
    }
    trajectory_by_internal_slot = {
        row.global_slot: row for row in incoming_events.agent_phase_trajectories
    }
    transition_start_agents_by_slot = _agent_pov_visual_scene_agents_by_slot(
        transition_start_scene,
        scene_name="transition_start_scene",
        phase="transition_start",
        recipient_public_agent_id=recipient_public_agent_id,
        slot_by_public_agent_id=slot_by_public_agent_id,
        trajectory_by_internal_slot=trajectory_by_internal_slot,
        configured_active_by_global_slot=(
            incoming_events.configured_active_by_global_slot
        ),
    )
    successor_agents_by_slot = _agent_pov_visual_scene_agents_by_slot(
        successor_scene,
        scene_name="successor_scene",
        phase="successor",
        recipient_public_agent_id=recipient_public_agent_id,
        slot_by_public_agent_id=slot_by_public_agent_id,
        trajectory_by_internal_slot=trajectory_by_internal_slot,
        configured_active_by_global_slot=(
            incoming_events.configured_active_by_global_slot
        ),
    )
    if (transition_start_corpse_choreography_scene is None) != (
        successor_corpse_choreography_scene is None
    ):
        raise ValueError(
            "Agent POV corpse choreography scenes must be supplied as a pair."
        )
    if transition_start_corpse_choreography_scene is None:
        corpse_start_agents_by_slot = transition_start_agents_by_slot
        corpse_successor_agents_by_slot = successor_agents_by_slot
    else:
        if successor_corpse_choreography_scene is None:  # pragma: no cover - paired.
            raise AssertionError("corpse choreography successor scene disappeared")
        corpse_start_agents_by_slot = _agent_pov_corpse_choreography_agents_by_slot(
            transition_start_corpse_choreography_scene,
            base_agents_by_slot=transition_start_agents_by_slot,
            scene_name="transition_start_corpse_choreography_scene",
            phase="transition_start",
            recipient_public_agent_id=recipient_public_agent_id,
            slot_by_public_agent_id=slot_by_public_agent_id,
            trajectory_by_internal_slot=trajectory_by_internal_slot,
            configured_active_by_global_slot=(
                incoming_events.configured_active_by_global_slot
            ),
        )
        corpse_successor_agents_by_slot = _agent_pov_corpse_choreography_agents_by_slot(
            successor_corpse_choreography_scene,
            base_agents_by_slot=successor_agents_by_slot,
            scene_name="successor_corpse_choreography_scene",
            phase="successor",
            recipient_public_agent_id=recipient_public_agent_id,
            slot_by_public_agent_id=slot_by_public_agent_id,
            trajectory_by_internal_slot=trajectory_by_internal_slot,
            configured_active_by_global_slot=(
                incoming_events.configured_active_by_global_slot
            ),
        )
    start_recipient = next(
        row for row in transition_start_scene.agents if row.relation == "self"
    )
    successor_recipient = next(
        row for row in successor_scene.agents if row.relation == "self"
    )
    recipient_global_slot = slot_by_public_agent_id[recipient_public_agent_id]
    if start_recipient.presentation_key != successor_recipient.presentation_key:
        raise ValueError("Agent POV recipient authority changed within one transition.")
    common_slots = (
        transition_start_agents_by_slot.keys() & successor_agents_by_slot.keys()
    )
    for slot in common_slots:
        if (
            transition_start_agents_by_slot[slot].presentation_key
            != successor_agents_by_slot[slot].presentation_key
        ):
            raise ValueError(
                "Agent POV visual identities changed within one transition."
            )
        if (
            transition_start_agents_by_slot[slot].class_id
            != successor_agents_by_slot[slot].class_id
        ):
            raise ValueError(
                "Agent POV visual class identity changed within one transition."
            )
    corpse_common_slots = (
        corpse_start_agents_by_slot.keys() & corpse_successor_agents_by_slot.keys()
    )
    for slot in corpse_common_slots:
        if (
            corpse_start_agents_by_slot[slot].presentation_key
            != corpse_successor_agents_by_slot[slot].presentation_key
            or corpse_start_agents_by_slot[slot].class_id
            != corpse_successor_agents_by_slot[slot].class_id
        ):
            raise ValueError(
                "Agent POV corpse choreography identity changed within one transition."
            )
    authorized_source_events: list[VisualEventV2] = []
    death_slots: set[int] = set()
    respawn_slots: set[int] = set()
    for source_event in incoming_events.events:
        corpse_lifecycle = type(source_event) in (
            AgentDiedEventV2,
            AgentRespawnedEventV2,
        )
        if not _agent_pov_visual_event_is_authorized(
            source_event,
            recipient_global_slot=recipient_global_slot,
            transition_start_agents_by_slot=(
                corpse_start_agents_by_slot
                if corpse_lifecycle
                else transition_start_agents_by_slot
            ),
            successor_agents_by_slot=(
                corpse_successor_agents_by_slot
                if corpse_lifecycle
                else successor_agents_by_slot
            ),
            trajectory_by_internal_slot=trajectory_by_internal_slot,
        ):
            continue
        authorized_source_events.append(source_event)
        if type(source_event) is AgentDiedEventV2:
            death_slots.add(source_event.recipient_global_slot)
        elif type(source_event) is AgentRespawnedEventV2:
            respawn_slots.add(source_event.agent_global_slot)
    corpse_lifecycle_slots = death_slots | respawn_slots
    base_union_slots = (
        *transition_start_agents_by_slot,
        *(
            slot
            for slot in successor_agents_by_slot
            if slot not in transition_start_agents_by_slot
        ),
    )
    union_slots = (
        *base_union_slots,
        *(
            trajectory.global_slot
            for trajectory in incoming_events.agent_phase_trajectories
            if trajectory.global_slot in corpse_lifecycle_slots
            and trajectory.global_slot not in base_union_slots
        ),
    )
    trajectory_start_agents_by_slot = {
        slot: (
            transition_start_agents_by_slot.get(slot)
            or (
                corpse_start_agents_by_slot.get(slot) if slot in respawn_slots else None
            )
        )
        for slot in union_slots
    }
    trajectory_successor_agents_by_slot = {
        slot: (
            successor_agents_by_slot.get(slot)
            or (
                corpse_successor_agents_by_slot.get(slot)
                if slot in death_slots
                else None
            )
        )
        for slot in union_slots
    }
    if any(
        trajectory_start_agents_by_slot[slot] is None
        and trajectory_successor_agents_by_slot[slot] is None
        for slot in union_slots
    ):
        raise AssertionError("authorized Agent trajectory lost both endpoints")
    trajectory_agents_by_slot = {
        slot: cast(
            AuthorizedAgentV1,
            (
                trajectory_start_agents_by_slot[slot]
                if trajectory_start_agents_by_slot[slot] is not None
                else trajectory_successor_agents_by_slot[slot]
            ),
        )
        for slot in union_slots
    }
    key_by_internal_slot = {
        slot: trajectory_agents_by_slot[slot].presentation_key for slot in union_slots
    }
    trajectories = tuple(
        AgentPovVisualIncomingAgentPhaseTrajectoryV1(
            agent_presentation_key=key_by_internal_slot[slot],
            agent_public_agent_id=trajectory_agents_by_slot[slot].public_agent_id,
            agent_class_id=trajectory_agents_by_slot[slot].class_id,
            transition_start=(
                None
                if trajectory_start_agents_by_slot[slot] is None
                else _replay_incoming_anchor(
                    trajectory_by_internal_slot[slot].transition_start,
                    key_by_internal_slot=key_by_internal_slot,
                )
            ),
            successor=(
                None
                if trajectory_successor_agents_by_slot[slot] is None
                else _replay_incoming_anchor(
                    trajectory_by_internal_slot[slot].successor,
                    key_by_internal_slot=key_by_internal_slot,
                )
            ),
        )
        for slot in union_slots
    )
    events: list[AgentPovVisualIncomingEventV1] = []
    for source_event in authorized_source_events:
        projected_event: (
            ReplayIncomingEventV1
            | AgentPovVisualIncomingRecipientHealthResolutionEventV1
            | AgentPovVisualIncomingAgentRespawnedEventV1
        )
        if type(source_event) is RecipientHealthResolutionEventV2:
            projected_event = AgentPovVisualIncomingRecipientHealthResolutionEventV1(
                event_id=source_event.event_id,
                ordinal=source_event.ordinal,
                phase_rank=source_event.phase_rank,
                event_kind=source_event.event_type,
                recipient_anchor=_replay_incoming_anchor(
                    source_event.recipient_anchor,
                    key_by_internal_slot=key_by_internal_slot,
                ),
                transition_start_health=(source_event.transition_start_health),
                health_after_combat_resolution=(
                    source_event.health_after_combat_resolution
                ),
                realized_net_health_change=(source_event.realized_net_health_change),
            )
        elif type(source_event) is AgentRespawnedEventV2:
            projected_event = AgentPovVisualIncomingAgentRespawnedEventV1(
                event_id=source_event.event_id,
                ordinal=source_event.ordinal,
                phase_rank=source_event.phase_rank,
                event_kind=source_event.event_type,
                agent_anchor=_replay_incoming_anchor(
                    source_event.agent_anchor,
                    key_by_internal_slot=key_by_internal_slot,
                ),
            )
        else:
            projected_event = _replay_incoming_event(
                source_event,
                trajectory_by_internal_slot=trajectory_by_internal_slot,
                key_by_internal_slot=key_by_internal_slot,
            )
        if type(projected_event) not in _AGENT_POV_VISUAL_INCOMING_EVENT_TYPES_V1:
            raise AssertionError("filtered Agent POV event used an excluded variant")
        local_ordinal = len(events)
        projected_event = replace(
            projected_event,
            event_id=(
                f"{incoming_recipient_transition_id}:visual-event:{local_ordinal:04d}"
            ),
            ordinal=local_ordinal,
        )
        events.append(cast(AgentPovVisualIncomingEventV1, projected_event))
    event_rows = tuple(events)
    return AgentPovVisualIncomingSummaryV1(
        schema_version=AUTHORIZED_PRESENTATION_SCHEMA_VERSION,
        summary_kind="agent_pov_fog_filtered_visual_events",
        source_episode_id=incoming_events.episode_id,
        recipient_public_agent_id=recipient_public_agent_id,
        recipient_presentation_key=successor_recipient.presentation_key,
        incoming_transition_index=incoming_events.transition_index,
        incoming_recipient_transition_id=incoming_recipient_transition_id,
        incoming_start_recipient_frame_id=incoming_start_recipient_frame_id,
        incoming_successor_recipient_frame_id=(incoming_successor_recipient_frame_id),
        incoming_start_simulator_step_count=(
            incoming_events.start_simulator_step_count
        ),
        incoming_successor_simulator_step_count=(
            incoming_events.successor_simulator_step_count
        ),
        agent_phase_trajectories=trajectories,
        ordered_event_ids=tuple(event.event_id for event in event_rows),
        ordered_event_kinds=tuple(event.event_kind for event in event_rows),
        events=event_rows,
        event_count=len(event_rows),
    )


def _project_replay_incoming_summary_v1(
    incoming_events: VisualEventBatchV2,
    *,
    key_by_internal_slot: dict[int, str],
) -> ReplayIncomingSummaryV1:
    """Project a complete validated Oracle event batch without raw slot fields.
    Require the key dictionary to cover exactly the configured-active axis with
    unique nonempty strings. Preserve canonical trajectory/event ordering and
    recorded transition IDs.
    Parameters
    ----------
    incoming_events : VisualEventBatchV2
        Canonical recorded visual batch for one incoming transition.
    key_by_internal_slot : dict[int, str]
        Lookup from configured-active internal slots to unique authorized opaque
        keys.
    Returns
    -------
    ReplayIncomingSummaryV1
        Frozen full incoming inventory.
    Raises
    ------
    ValueError
        The batch, exact key-map coverage, or a projected join is invalid.
    """
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
    """Join incoming history to the displayed Oracle scene.
    Frame zero requires no incoming batch and returns None. Later frames require
    the exact previous transition, roster axes, successor epoch/tick, and event
    ID inventory.
    Parameters
    ----------
    scene : BattlefieldSceneV2
        Displayed durable Oracle scene for the same episode.
    incoming_events : VisualEventBatchV2 | None
        Recorded incoming visual batch; None at frame zero, required for every
        later frame.
    key_by_internal_slot : dict[int, str]
        Lookup from configured-active internal slots to unique authorized opaque
        keys.
    expected_public_agent_id_by_global_slot : tuple[str, ...]
        Complete context roster public IDs, including inactive slots.
    expected_configured_active_by_global_slot : tuple[bool, ...]
        Complete context roster activity flags on the same slot axis.
    Returns
    -------
    ReplayIncomingSummaryV1 or None
        Projected history, or None at frame zero.
    Raises
    ------
    ValueError
        History is missing, unexpectedly present, or does not join this
        scene/context.
    """
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
    context: EvaluationEpisodeContext,
    scene: BattlefieldSceneV2,
    *,
    key_by_internal_slot: dict[int, str],
    selected_internal_slot: int | None,
    outgoing_transition: EvaluationTransitionV1 | None,
    final_frame_index: int,
) -> ReplayOutgoingInspectionV1 | None:
    """Project the selected recorded action starting at the displayed scene.
    Require an outgoing row only when a configured-active actor is selected
    before the final retained frame. Resolve accepted target categories through
    the actor-relative catalog axis and take actor/target positions only from
    the current scene.
    Parameters
    ----------
    context : EvaluationEpisodeContext
        Validated episode context with roster, resolved configuration, and
        public catalog.
    scene : BattlefieldSceneV2
        Displayed durable Oracle scene for the same episode.
    key_by_internal_slot : dict[int, str]
        Lookup from configured-active internal slots to unique authorized opaque
        keys.
    selected_internal_slot : int | None
        Selected configured-active actor slot, or None to omit outgoing
        inspection.
    outgoing_transition : EvaluationTransitionV1 | None
        Exact recorded row starting at the displayed scene, or None when no
        outgoing inspection is allowed.
    final_frame_index : int
        Nonnegative final retained replay frame index, including a partial
        prefix.
    Returns
    -------
    ReplayOutgoingInspectionV1 or None
        Submitted/accepted action disclosure, or None when inspection is
        unavailable.
    Raises
    ------
    ValueError
        Outgoing-row presence, transition epoch, actor identity, or accepted
        target does not join.
    """
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
    context: EvaluationEpisodeContext,
    scene: BattlefieldSceneV2,
    incoming_events: VisualEventBatchV2 | None,
    *,
    authority_session_id: str,
    final_frame_index: int,
    selected_internal_slot: int | None,
    outgoing_transition: EvaluationTransitionV1 | None,
) -> ReplayOraclePresentationPartsV1:
    """Build current Oracle scene, incoming history, and selected outgoing action.
    Consume already validated recorded roots from one episode. Remap canonical
    incoming events without changing atomic meanings; their successor positions
    must join scene. No successor scene is accepted, so outgoing anchors cannot
    use future positions. This pure host projection does not perform I/O or
    simulator work.
    Parameters
    ----------
    context : EvaluationEpisodeContext
        Validated episode context with roster, resolved configuration, and
        public catalog.
    scene : BattlefieldSceneV2
        Displayed durable Oracle scene for the same episode.
    incoming_events : VisualEventBatchV2 | None
        Recorded incoming visual batch; None at frame zero, required for every
        later frame.
    authority_session_id : str
        Nonempty authority/session namespace for opaque presentation keys.
    final_frame_index : int
        Nonnegative final retained replay frame index, including a partial
        prefix.
    selected_internal_slot : int | None
        Selected configured-active actor slot, or None to omit outgoing
        inspection.
    outgoing_transition : EvaluationTransitionV1 | None
        Exact recorded row starting at the displayed scene, or None when no
        outgoing inspection is allowed.
    Returns
    -------
    ReplayOraclePresentationPartsV1
        Frozen sibling branches ready for the outer replay authority envelope.
    Raises
    ------
    TypeError, ValueError
        A root type, index, episode, roster, history, action, or scene join is
        invalid.
    """
    evaluation_context_type(context)
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
    "AUTHORIZED_CLASS_DOCUMENTATION_CATALOG_FINGERPRINT_V1",
    "AUTHORIZED_CLASS_DOCUMENTATION_PROFILE_ID_V1",
    "AUTHORIZED_PRESENTATION_SCHEMA_VERSION",
    "AcceptedActionTupleV1",
    "AgentPovVisualIncomingAgentPhaseTrajectoryV1",
    "AgentPovVisualIncomingAgentRespawnedEventV1",
    "AgentPovVisualIncomingEventKindV1",
    "AgentPovVisualIncomingEventV1",
    "AgentPovVisualIncomingRecipientHealthResolutionEventV1",
    "AgentPovVisualIncomingSummaryV1",
    "AuthorizedAgentV1",
    "AuthorizedAuraFieldV1",
    "AuthorizedAuraIdV1",
    "AuthorizedAuraModifierV1",
    "AuthorizedBattlefieldSceneV1",
    "AuthorizedClassAuraMechanicV1",
    "AuthorizedClassDocumentationProfileAvailableV1",
    "AuthorizedClassDocumentationProfileUnavailableV1",
    "AuthorizedClassDocumentationProfileV1",
    "AuthorizedClassMechanics",
    "AuthorizedClassMechanicsV1",
    "AuthorizedClassMechanicsV2",
    "AuthorizedClassStatusMechanicV1",
    "AuthorizedMap",
    "AuthorizedMapV1",
    "AuthorizedMapV2",
    "AuthorizedObstacleV1",
    "AuthorizedRedZoneV1",
    "AuthorizedRespawnWaveV1",
    "AuthorizedSpawnPadV1",
    "AuthorizedSpawnShieldMechanics",
    "AuthorizedSpawnShieldMechanicsAvailableV1",
    "AuthorizedSpawnShieldMechanicsAvailableV2",
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
    "ReplayIncomingAgentLeftCombatEventV1",
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
    "ReplayIncomingTeamDeathmatchCompletedEventV1",
    "ReplayIncomingTeamDeathmatchScoreChangedEventV1",
    "ReplayOraclePresentationPartsV1",
    "ReplayOutgoingInspectionV1",
    "SubmittedActionTupleV1",
    "authorized_class_documentation_profile_v1",
    "build_agent_pov_visual_incoming_summary_v1",
    "build_authorized_map_v2",
    "build_oracle_authorized_scene_v1",
    "build_replay_oracle_presentation_parts_v1",
    "oracle_presentation_key_v1",
]
