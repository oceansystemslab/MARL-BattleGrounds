"""Build renderer-neutral scenes from already recipient-sliced POV records.

build_actor_pov_analyzer_projection_v1 reads a validated replay index, raw
POV replay or current slice. It copies self, visible bodies, public map and
lifecycle data into a scene, then attaches the exact next-decision mask and
incoming POV-local cues. build_actor_pov_projection_index_v1 validates a
captured prefix once for repeated frame selection.

No researcher snapshot, simulator state or hidden body row is accepted.
V1 preserves historical physical-team feature columns; V2 decodes current
actor-relative relation flags using explicit public identity mappings.
All work is on host records, with no JAX execution, file I/O or input mutation.
"""

from dataclasses import dataclass
from math import isfinite
from typing import Literal, cast

from marl_battlegrounds.evaluation.pov import (
    ActorPovActionMaskV1,
    ActorPovAxisMapping,
    ActorPovAxisMappingV1,
    ActorPovAxisMappingV2,
    ActorPovCurrentSlice,
    ActorPovCurrentSliceV1,
    ActorPovCurrentSliceV2,
    ActorPovFrame,
    ActorPovFrameV1,
    ActorPovFrameV2,
    ActorPovPresentationCueV1,
    ActorPovReplayContent,
    ActorPovReplayContentV1,
    ActorPovReplayContentV2,
    ActorPovTransitionV1,
    validate_actor_pov_replay_content,
)
from marl_battlegrounds.rendering.evaluation_wire_features import (
    AGENT_FEATURE_ACTIVE_V1,
    AGENT_FEATURE_ALIVE_V1,
    AGENT_FEATURE_CLASS_ID_V1,
    AGENT_FEATURE_CURRENT_HEALTH_V1,
    AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED_V1,
    AGENT_FEATURE_MAX_HEALTH_V1,
    AGENT_FEATURE_RADIUS_V1,
    AGENT_FEATURE_STEPS_UNTIL_OUT_OF_COMBAT_V1,
    AGENT_FEATURE_ULTIMATE_COOLDOWN_REMAINING_V1,
    AGENT_FEATURE_X_V1,
    AGENT_FEATURE_Y_V1,
    AGENT_STATUS_FEATURE_START_V1,
    AGENT_STATUS_FEATURE_STOP_V1,
    CONTEXT_FEATURE_MAP_HEIGHT_V1,
    CONTEXT_FEATURE_MAP_WIDTH_V1,
    OBSTACLE_FEATURE_ACTIVE_V1,
    OBSTACLE_FEATURE_HEIGHT_V1,
    OBSTACLE_FEATURE_RADIUS_V1,
    OBSTACLE_FEATURE_THETA_V1,
    OBSTACLE_FEATURE_TYPE_V1,
    OBSTACLE_FEATURE_WIDTH_V1,
    OBSTACLE_FEATURE_X_V1,
    OBSTACLE_FEATURE_Y_V1,
    decode_agent_feature_row,
)
from marl_battlegrounds.rendering.scene import MapSceneV1, ObstacleSceneV1, Point2D

ACTOR_POV_SCENE_SCHEMA_VERSION = 1

_PILLAR_OBSTACLE_TYPE_ID_V1 = 1
_WALL_OBSTACLE_TYPE_ID_V1 = 2


def _require_text(value: str, *, name: str) -> None:
    """Require value to be a nonblank exact Python string.

    name labels ValueError. Return None without trimming, copying or coercing
    valid text.
    """
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a non-empty Python string.")


def _require_int(value: int, *, name: str, minimum: int = 0) -> None:
    """Require value to be a Python int at least minimum, default zero.

    Bool and array scalars are rejected. name labels ValueError; no upper bound
    is imposed. Return None without converting valid values.
    """
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be a Python int at least {minimum}.")


def _require_float(value: float, *, name: str, minimum: float = 0.0) -> None:
    """Require value to be a finite Python float at least minimum.

    minimum defaults to 0.0. Integer and array scalar inputs are rejected.
    name labels ValueError; return None without conversion.
    """
    if type(value) is not float or not isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be a finite Python float at least {minimum}.")


def _require_point(value: Point2D, *, name: str) -> None:
    """Check a finite Python (x, y) tuple in world coordinates.

    value must have exactly two float entries; negative coordinates are allowed.
    name labels ValueError. Return None without checking geometry or map bounds.
    """
    if type(value) is not tuple or len(value) != 2:
        raise ValueError(f"{name} must be a two-coordinate Python tuple.")
    for coordinate in value:
        _require_float(coordinate, name=f"{name} coordinate", minimum=-float("inf"))


def _decode_wire_bool(value: float, *, name: str) -> bool:
    """Decode exact Python float 0.0/1.0 into False/True.

    name labels ValueError for another type/value. No truthiness coercion is
    performed.
    """
    if type(value) is not float or value not in (0.0, 1.0):
        raise ValueError(f"{name} must be the exact wire float 0.0 or 1.0.")
    return value == 1.0


def _decode_wire_int(
    value: float,
    *,
    name: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    """Decode a finite integral Python float within inclusive bounds.

    value must have no fractional part. minimum defaults to zero; maximum=None
    leaves the upper end unbounded. Return a Python int, or raise ValueError
    labelled by name for an invalid type/value.
    """
    if type(value) is not float or not isfinite(value) or not value.is_integer():
        raise ValueError(f"{name} must be an integer-valued finite wire float.")
    decoded = int(value)
    if decoded < minimum or (maximum is not None and decoded > maximum):
        raise ValueError(f"{name} is outside its V1 wire domain.")
    return decoded


@dataclass(frozen=True, slots=True, kw_only=True)
class ActorPovSelfSceneV1:
    """Describe the selected actor's exact self row and export identity.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    global_slot : int
        Nonnegative Python global slot supplied by the export; this record does
        not impose the upper bound of 9.
    public_agent_id : str
        Nonblank public ID from the authorized export mapping.
    team_local_slot : int
        Python own-team roster index in 0..4.
    team_id : int
        Python physical team ID 1 or 2, decoded according to the wire version.
    class_id : int
        Python real class ID in 1..5.
    position : Point2D
        Finite Python world (x, y) float tuple; no map-containment check here.
    radius : float
        Positive finite Python body radius in world units.
    alive : bool
        Exact Python current-life boolean.
    current_health : float
        Nonnegative finite Python health in health units.
    max_health : float
        Positive finite Python maximum health; current_health must not exceed it.
    effective_movement_speed : float
        Nonnegative finite Python current speed in world units per step.
    ultimate_cooldown_remaining : int
        Nonnegative Python remaining Ultimate cooldown in steps.
    steps_until_out_of_combat : int
        Nonnegative Python remaining recovery countdown in steps.
    spawn_shield_remaining : int
        Nonnegative Python remaining shield duration from the own-team lifecycle row.
    status_feature_values : tuple[float, ...]
        Exact tuple of 14 nonnegative finite Python floats from wire columns
        15..28, retaining status duration/strength order.

    Raises
    ------
    ValueError
        Identity types/ranges, scalar/tuple values, positive radius/max health,
        current-health upper bound or the life boolean is invalid.

    Notes
    -----
    Self remains in its roster position. These are already authorized facts;
    constructing the record does not grant access to an export.
    """

    global_slot: int
    """Nonnegative Python global slot supplied by the export; this record does not
    impose the upper bound of 9.
    """
    public_agent_id: str
    """Nonblank public ID from the authorized export mapping."""
    team_local_slot: int
    """Python own-team roster index in 0..4."""
    team_id: int
    """Python physical team ID 1 or 2, decoded according to the wire version."""
    class_id: int
    """Python real class ID in 1..5."""
    position: Point2D
    """Finite Python world (x, y) float tuple; no map-containment check here."""
    radius: float
    """Positive finite Python body radius in world units."""
    alive: bool
    """Exact Python current-life boolean."""
    current_health: float
    """Nonnegative finite Python health in health units."""
    max_health: float
    """Positive finite Python maximum health; current_health must not exceed it."""
    effective_movement_speed: float
    """Nonnegative finite Python current speed in world units per step."""
    ultimate_cooldown_remaining: int
    """Nonnegative Python remaining Ultimate cooldown in steps."""
    steps_until_out_of_combat: int
    """Nonnegative Python remaining recovery countdown in steps."""
    spawn_shield_remaining: int
    """Nonnegative Python remaining shield duration from the own-team lifecycle row."""
    status_feature_values: tuple[float, ...]
    """Exact tuple of 14 nonnegative finite Python floats from wire columns 15..28,
    retaining status duration/strength order.
    """

    def __post_init__(self) -> None:
        """Check ActorPovSelfSceneV1 during host construction.

        Raise ValueError if identity types/ranges, scalar/tuple values, positive
        radius/max health, current-health upper bound or the life boolean is
        invalid.
        Return None without changing valid fields.
        """
        _require_int(self.global_slot, name="global_slot")
        _require_text(self.public_agent_id, name="public_agent_id")
        _require_int(self.team_local_slot, name="team_local_slot")
        if self.team_local_slot >= 5:
            raise ValueError("team_local_slot must be less than five.")
        _require_int(self.team_id, name="team_id", minimum=1)
        if self.team_id not in (1, 2):
            raise ValueError("team_id must be one or two.")
        _require_int(self.class_id, name="class_id", minimum=1)
        if self.class_id > 5:
            raise ValueError("class_id must identify a real V1 class.")
        _require_point(self.position, name="position")
        for name in (
            "radius",
            "current_health",
            "max_health",
            "effective_movement_speed",
        ):
            _require_float(cast(float, getattr(self, name)), name=name)
        if self.radius <= 0.0 or self.max_health <= 0.0:
            raise ValueError("self body radius and max health must be positive.")
        if self.current_health > self.max_health:
            raise ValueError("current_health must not exceed max_health.")
        if type(self.alive) is not bool:
            raise ValueError("alive must be a Python bool.")
        for name in (
            "ultimate_cooldown_remaining",
            "steps_until_out_of_combat",
            "spawn_shield_remaining",
        ):
            _require_int(cast(int, getattr(self, name)), name=name)
        if type(self.status_feature_values) is not tuple or len(
            self.status_feature_values
        ) != (AGENT_STATUS_FEATURE_STOP_V1 - AGENT_STATUS_FEATURE_START_V1):
            raise ValueError("status_feature_values must retain exact V1 columns.")
        for value in self.status_feature_values:
            _require_float(value, name="status feature value")


@dataclass(frozen=True, slots=True, kw_only=True)
class ActorPovVisibleBodySceneV1:
    """Describe one visible observation row without guessing a global slot.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    relation : Literal['ally', 'enemy']
        ally or enemy in this recipient's base sensor axes.
    observation_row : int
        Python row index in 0..4 on the named relation axis.
    public_agent_id : str
        Nonblank public ID from the authorized export mapping.
    position : Point2D
        Finite Python world (x, y) float tuple; no map-containment check here.
    radius : float
        Nonnegative finite Python body radius in world units.
    team_id : int
        Python physical team ID 1 or 2, decoded according to the wire version.
    class_id : int
        Python real class ID in 1..5.
    alive : bool
        Exact Python current-life boolean.
    current_health : float
        Nonnegative finite Python health in health units.
    max_health : float
        Nonnegative finite Python maximum health in health units.
    effective_movement_speed : float
        Nonnegative finite Python current speed in world units per step.
    ultimate_cooldown_remaining : int
        Nonnegative Python remaining Ultimate cooldown in steps.
    steps_until_out_of_combat : int
        Nonnegative Python remaining recovery countdown in steps.
    status_feature_values : tuple[float, ...]
        Exact tuple of 14 nonnegative finite Python floats from wire columns
        15..28, retaining status duration/strength order.

    Raises
    ------
    ValueError
        Relation/row, public identity, team/class IDs, nonnegative
        scalar/counter values or 14-column status tuple is invalid.

    Notes
    -----
    This record has no global-slot field. Unlike the self-row constructor, it
    allows zero radius/max health and does not separately check current_health
    <= max_health. The validated source and decoder own those stronger joins.
    """

    relation: Literal["ally", "enemy"]
    """ally or enemy in this recipient's base sensor axes."""
    observation_row: int
    """Python row index in 0..4 on the named relation axis."""
    public_agent_id: str
    """Nonblank public ID from the authorized export mapping."""
    position: Point2D
    """Finite Python world (x, y) float tuple; no map-containment check here."""
    radius: float
    """Nonnegative finite Python body radius in world units."""
    team_id: int
    """Python physical team ID 1 or 2, decoded according to the wire version."""
    class_id: int
    """Python real class ID in 1..5."""
    alive: bool
    """Exact Python current-life boolean."""
    current_health: float
    """Nonnegative finite Python health in health units."""
    max_health: float
    """Nonnegative finite Python maximum health in health units."""
    effective_movement_speed: float
    """Nonnegative finite Python current speed in world units per step."""
    ultimate_cooldown_remaining: int
    """Nonnegative Python remaining Ultimate cooldown in steps."""
    steps_until_out_of_combat: int
    """Nonnegative Python remaining recovery countdown in steps."""
    status_feature_values: tuple[float, ...]
    """Exact tuple of 14 nonnegative finite Python floats from wire columns 15..28,
    retaining status duration/strength order.
    """

    def __post_init__(self) -> None:
        """Check ActorPovVisibleBodySceneV1 during host construction.

        Raise ValueError if relation/row, public identity, team/class IDs,
        nonnegative scalar/counter values or 14-column status tuple is invalid.
        Return None without changing valid fields.
        """
        if self.relation not in ("ally", "enemy"):
            raise ValueError("relation must be ally or enemy.")
        _require_int(self.observation_row, name="observation_row")
        if self.observation_row >= 5:
            raise ValueError("observation_row must be less than five.")
        _require_text(self.public_agent_id, name="public_agent_id")
        _require_point(self.position, name="position")
        for name in (
            "radius",
            "current_health",
            "max_health",
            "effective_movement_speed",
        ):
            _require_float(cast(float, getattr(self, name)), name=name)
        _require_int(self.team_id, name="team_id", minimum=1)
        _require_int(self.class_id, name="class_id", minimum=1)
        if self.team_id not in (1, 2) or self.class_id > 5:
            raise ValueError("visible body team/class IDs are outside V1 vocabulary.")
        if type(self.alive) is not bool:
            raise ValueError("alive must be a Python bool.")
        _require_int(
            self.ultimate_cooldown_remaining,
            name="ultimate_cooldown_remaining",
        )
        _require_int(
            self.steps_until_out_of_combat,
            name="steps_until_out_of_combat",
        )
        if type(self.status_feature_values) is not tuple or len(
            self.status_feature_values
        ) != (AGENT_STATUS_FEATURE_STOP_V1 - AGENT_STATUS_FEATURE_START_V1):
            raise ValueError("status_feature_values must retain exact V1 columns.")
        for value in self.status_feature_values:
            _require_float(value, name="status feature value")


@dataclass(frozen=True, slots=True, kw_only=True)
class ActorPovSpawnPadSceneV1:
    """Describe one spawn pad explicitly present in the recipient input.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    actor_relative_team_index : int
        Python team index 0 for own team or 1 for opponent team.
    team_relation : Literal['own', 'opponent']
        own or opponent, matching actor_relative_team_index.
    team_label : str
        Exact serialized label Own Team or Opponent Team, matching the index.
    team_local_slot : int
        Python team-local roster index in 0..4.
    position : Point2D
        Finite Python world (x, y) float tuple.
    configured_active : bool
        Exact Python configured-membership flag for this team/local slot.
    currently_alive : bool
        Exact Python current-life flag from the recipient lifecycle input.
    spawn_shield_remaining : int
        Nonnegative Python shield counter in steps.

    Raises
    ------
    ValueError
        Team/slot indices, exact relation/label, world point, membership/life
        booleans or shield counter is invalid.

    Notes
    -----
    Padding remains on the public lifecycle axes. This record does not infer a
    hidden body position from its spawn pad.
    """

    actor_relative_team_index: int
    """Python team index 0 for own team or 1 for opponent team."""
    team_relation: Literal["own", "opponent"]
    """own or opponent, matching actor_relative_team_index."""
    team_label: str
    """Exact serialized label Own Team or Opponent Team, matching the index."""
    team_local_slot: int
    """Python team-local roster index in 0..4."""
    position: Point2D
    """Finite Python world (x, y) float tuple."""
    configured_active: bool
    """Exact Python configured-membership flag for this team/local slot."""
    currently_alive: bool
    """Exact Python current-life flag from the recipient lifecycle input."""
    spawn_shield_remaining: int
    """Nonnegative Python shield counter in steps."""

    def __post_init__(self) -> None:
        """Check ActorPovSpawnPadSceneV1 during host construction.

        Raise ValueError if team/slot indices, exact relation/label, world
        point, membership/life booleans or shield counter is invalid.
        Return None without changing valid fields.
        """
        _require_int(
            self.actor_relative_team_index,
            name="actor_relative_team_index",
        )
        _require_int(self.team_local_slot, name="team_local_slot")
        if self.actor_relative_team_index not in (0, 1) or self.team_local_slot >= 5:
            raise ValueError("spawn pad team/slot coordinates are outside V1 axes.")
        expected_relation = "own" if self.actor_relative_team_index == 0 else "opponent"
        if self.team_relation != expected_relation:
            raise ValueError("team_relation must match the actor-relative team axis.")
        expected_label = (
            "Own Team" if self.actor_relative_team_index == 0 else "Opponent Team"
        )
        if self.team_label != expected_label:
            raise ValueError("team_label must preserve the serialized POV axis name.")
        _require_point(self.position, name="position")
        if (
            type(self.configured_active) is not bool
            or type(self.currently_alive) is not bool
        ):
            raise ValueError("spawn lifecycle flags must be Python bools.")
        _require_int(self.spawn_shield_remaining, name="spawn_shield_remaining")


@dataclass(frozen=True, slots=True, kw_only=True)
class ActorPovRespawnWaveSceneV1:
    """Describe a team wave clock copied from the actor's input.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    actor_relative_team_index : int
        Python team index 0 for own team or 1 for opponent team.
    team_relation : Literal['own', 'opponent']
        own or opponent, matching actor_relative_team_index.
    team_label : str
        Exact serialized label Own Team or Opponent Team, matching the index.
    period_steps : int
        Positive Python configured period in steps.
    countdown_steps : int
        Nonnegative Python current countdown in steps; this record does not
        check it against period_steps.

    Raises
    ------
    ValueError
        Team index, exact relation/label, period or countdown type/lower bound
        is invalid.
    """

    actor_relative_team_index: int
    """Python team index 0 for own team or 1 for opponent team."""
    team_relation: Literal["own", "opponent"]
    """own or opponent, matching actor_relative_team_index."""
    team_label: str
    """Exact serialized label Own Team or Opponent Team, matching the index."""
    period_steps: int
    """Positive Python configured period in steps."""
    countdown_steps: int
    """Nonnegative Python current countdown in steps; this record does not check it
    against period_steps.
    """

    def __post_init__(self) -> None:
        """Check ActorPovRespawnWaveSceneV1 during host construction.

        Raise ValueError if team index, exact relation/label, period or
        countdown type/lower bound is invalid.
        Return None without changing valid fields.
        """
        _require_int(
            self.actor_relative_team_index,
            name="actor_relative_team_index",
        )
        if self.actor_relative_team_index not in (0, 1):
            raise ValueError("actor_relative_team_index must be zero or one.")
        expected_relation = "own" if self.actor_relative_team_index == 0 else "opponent"
        if self.team_relation != expected_relation:
            raise ValueError("team_relation must match the actor-relative team axis.")
        expected_label = (
            "Own Team" if self.actor_relative_team_index == 0 else "Opponent Team"
        )
        if self.team_label != expected_label:
            raise ValueError("team_label must preserve the serialized POV axis name.")
        _require_int(self.period_steps, name="period_steps", minimum=1)
        _require_int(self.countdown_steps, name="countdown_steps")


@dataclass(frozen=True, slots=True, kw_only=True)
class ActorPovBattlefieldSceneV1:
    """Collect a recipient-authorized battlefield without researcher state.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    schema_version : int
        Exact Python int 1; this scene version accepts both supported POV source
        versions.
    audience_badge : str
        Nonblank text containing AGENT POV.
    observation_materialization : Literal['exact_no_shared_obs_actor_input']
        The literal exact_no_shared_obs_actor_input.
    episode_id : str
        Nonblank source episode ID.
    frame_index : int
        Nonnegative Python recorded frame index.
    pov_frame_id : str
        Nonblank source POV-frame ID; this record does not check its canonical pattern.
    source_frame_id : str
        Nonblank underlying evaluation-frame ID.
    simulator_step_count : int
        Nonnegative Python simulator count, separate from recorded frame index.
    map : MapSceneV1
        Exact MapSceneV1 copied from the authorized observation.
    self_actor : ActorPovSelfSceneV1
        Exact ActorPovSelfSceneV1 for the selected recipient.
    visible_bodies : tuple[ActorPovVisibleBodySceneV1, ...]
        Exact tuple of body rows with unique sorted (relation, row) keys; hidden
        rows are absent.
    spawn_pads : tuple[ActorPovSpawnPadSceneV1, ...]
        Exact tuple of public lifecycle pad rows with unique sorted (team index,
        local slot) keys.
    respawn_waves : tuple[ActorPovRespawnWaveSceneV1, ...]
        Tuple of own/opponent wave rows ordered by actor-relative indices (0, 1).

    Raises
    ------
    ValueError
        Version/materialization/badge, identity/counter types, nested roots or
        body/pad/wave ordering is invalid.

    Notes
    -----
    The producer must already validate and restrict source content. The scene
    constructor does not sanitize an arbitrary researcher snapshot.
    """

    schema_version: int
    """Exact Python int 1; this scene version accepts both supported POV source
    versions.
    """
    audience_badge: str
    """Nonblank text containing AGENT POV."""
    observation_materialization: Literal["exact_no_shared_obs_actor_input"]
    """The literal exact_no_shared_obs_actor_input."""
    episode_id: str
    """Nonblank source episode ID."""
    frame_index: int
    """Nonnegative Python recorded frame index."""
    pov_frame_id: str
    """Nonblank source POV-frame ID; this record does not check its canonical
    pattern.
    """
    source_frame_id: str
    """Nonblank underlying evaluation-frame ID."""
    simulator_step_count: int
    """Nonnegative Python simulator count, separate from recorded frame index."""
    map: MapSceneV1
    """Exact MapSceneV1 copied from the authorized observation."""
    self_actor: ActorPovSelfSceneV1
    """Exact ActorPovSelfSceneV1 for the selected recipient."""
    visible_bodies: tuple[ActorPovVisibleBodySceneV1, ...]
    """Exact tuple of body rows with unique sorted (relation, row) keys; hidden
    rows are absent.
    """
    spawn_pads: tuple[ActorPovSpawnPadSceneV1, ...]
    """Exact tuple of public lifecycle pad rows with unique sorted (team index,
    local slot) keys.
    """
    respawn_waves: tuple[ActorPovRespawnWaveSceneV1, ...]
    """Tuple of own/opponent wave rows ordered by actor-relative indices (0, 1)."""

    def __post_init__(self) -> None:
        """Check ActorPovBattlefieldSceneV1 during host construction.

        Raise ValueError if version/materialization/badge, identity/counter
        types, nested roots or body/pad/wave ordering is invalid.
        Return None without changing valid fields.
        """
        if type(self.schema_version) is not int or (
            self.schema_version != ACTOR_POV_SCENE_SCHEMA_VERSION
        ):
            raise ValueError("unknown actor POV scene version.")
        _require_text(self.audience_badge, name="audience_badge")
        if "AGENT POV" not in self.audience_badge:
            raise ValueError("actor POV scenes require an explicit audience badge.")
        if self.observation_materialization != "exact_no_shared_obs_actor_input":
            raise ValueError(
                "actor POV scene must disclose exact materialization mode."
            )
        _require_text(self.episode_id, name="episode_id")
        _require_int(self.frame_index, name="frame_index")
        _require_text(self.pov_frame_id, name="pov_frame_id")
        _require_text(self.source_frame_id, name="source_frame_id")
        _require_int(self.simulator_step_count, name="simulator_step_count")
        if type(self.map) is not MapSceneV1:
            raise ValueError("map must be the exact scalar MapSceneV1.")
        if type(self.self_actor) is not ActorPovSelfSceneV1:
            raise ValueError("self_actor must be ActorPovSelfSceneV1.")
        if type(self.visible_bodies) is not tuple or any(
            type(row) is not ActorPovVisibleBodySceneV1 for row in self.visible_bodies
        ):
            raise ValueError("visible_bodies must contain only POV body rows.")
        body_keys = tuple(
            (row.relation, row.observation_row) for row in self.visible_bodies
        )
        if body_keys != tuple(sorted(body_keys)) or len(body_keys) != len(
            set(body_keys)
        ):
            raise ValueError("visible body rows must have unique canonical keys.")
        if type(self.spawn_pads) is not tuple or any(
            type(row) is not ActorPovSpawnPadSceneV1 for row in self.spawn_pads
        ):
            raise ValueError("spawn_pads must contain only authorized POV rows.")
        pad_keys = tuple(
            (row.actor_relative_team_index, row.team_local_slot)
            for row in self.spawn_pads
        )
        if pad_keys != tuple(sorted(pad_keys)) or len(pad_keys) != len(set(pad_keys)):
            raise ValueError("spawn pad rows must have unique canonical keys.")
        if type(self.respawn_waves) is not tuple or tuple(
            row.actor_relative_team_index for row in self.respawn_waves
        ) != (
            0,
            1,
        ):
            raise ValueError("POV respawn waves must contain ordered team rows.")


@dataclass(frozen=True, slots=True, kw_only=True)
class ActorPovAnalyzerProjectionV1:
    """Join a POV scene, next-decision mask and incoming local cues.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    scene : ActorPovBattlefieldSceneV1
        Exact ActorPovBattlefieldSceneV1 for the selected frame.
    next_decision_action_mask : ActorPovActionMaskV1
        Exact ActorPovActionMaskV1 from the same current frame.
    incoming_transition_id : str | None
        None at frame zero; otherwise the canonical actor-POV transition entering scene.
    incoming_cues : tuple[ActorPovPresentationCueV1, ...]
        Python tuple of recipient-local cues with matching transition ID and
        ordinals 0..N-1; empty at frame zero.

    Raises
    ------
    ValueError
        Scene/mask root types, incoming transition identity or cue
        ordering/joins are invalid.

    Notes
    -----
    Incoming cues describe the completed transition. The mask describes the next
    decision. This envelope does not compare the mask with an independent frame.
    """

    scene: ActorPovBattlefieldSceneV1
    """Exact ActorPovBattlefieldSceneV1 for the selected frame."""
    next_decision_action_mask: ActorPovActionMaskV1
    """Exact ActorPovActionMaskV1 from the same current frame."""
    incoming_transition_id: str | None
    """None at frame zero; otherwise the canonical actor-POV transition entering
    scene.
    """
    incoming_cues: tuple[ActorPovPresentationCueV1, ...]
    """Python tuple of recipient-local cues with matching transition ID and
    ordinals 0..N-1; empty at frame zero.
    """

    def __post_init__(self) -> None:
        """Check ActorPovAnalyzerProjectionV1 during host construction.

        Raise ValueError if scene/mask root types, incoming transition identity
        or cue ordering/joins are invalid.
        Return None without changing valid fields.
        """
        if type(self.scene) is not ActorPovBattlefieldSceneV1:
            raise ValueError("scene must be ActorPovBattlefieldSceneV1.")
        if type(self.next_decision_action_mask) is not ActorPovActionMaskV1:
            raise ValueError("next_decision_action_mask must be the exact POV root.")
        expected_transition_id = (
            None
            if self.scene.frame_index == 0
            else (
                f"{self.scene.episode_id}:actor-pov:"
                f"{self.scene.self_actor.public_agent_id}:transition:"
                f"{self.scene.frame_index - 1}"
            )
        )
        if self.incoming_transition_id != expected_transition_id:
            raise ValueError(
                "incoming POV transition ID must enter the selected frame."
            )
        if type(self.incoming_cues) is not tuple:
            raise ValueError("incoming_cues must be a Python tuple.")
        if any(
            cue.pov_transition_id != self.incoming_transition_id
            for cue in self.incoming_cues
        ):
            raise ValueError("incoming POV cues must join their transition.")
        if tuple(cue.ordinal for cue in self.incoming_cues) != tuple(
            range(len(self.incoming_cues))
        ):
            raise ValueError("incoming POV cues must be gap-free and ordered.")


def _point(row: tuple[float, ...]) -> Point2D:
    """Read world (x, y) from one authorized 58-column feature row.

    row is already validated host data. Return the two stored Python floats
    without inferring hidden coordinates or changing their frame of reference.
    """
    return (row[AGENT_FEATURE_X_V1], row[AGENT_FEATURE_Y_V1])


def _status_values(row: tuple[float, ...]) -> tuple[float, ...]:
    """Slice the 14 status duration/strength columns from row.

    row is a validated 58-column Python float tuple. Return columns 15..28 in
    wire order without assigning source identity or interpreting a new effect.
    """
    return row[AGENT_STATUS_FEATURE_START_V1:AGENT_STATUS_FEATURE_STOP_V1]


def _map_scene(
    frame_rows: tuple[tuple[float, ...], ...], context: tuple[float, ...]
) -> MapSceneV1:
    """Decode map geometry from recipient-visible obstacle and context rows.

    frame_rows is the (32, 8) obstacle tuple and context is the 19-column context
    tuple. Ignore inactive obstacle padding; build circles/walls from supplied
    world values and return MapSceneV1. Raise ValueError for malformed wire
    values or unsupported active obstacle types. No privileged map is read.
    """
    obstacles: list[ObstacleSceneV1] = []
    for obstacle_slot, row in enumerate(frame_rows):
        if not _decode_wire_bool(
            row[OBSTACLE_FEATURE_ACTIVE_V1],
            name=f"obstacle row {obstacle_slot} active",
        ):
            continue
        obstacle_type = _decode_wire_int(
            row[OBSTACLE_FEATURE_TYPE_V1],
            name=f"obstacle row {obstacle_slot} type",
            maximum=2,
        )
        center = (row[OBSTACLE_FEATURE_X_V1], row[OBSTACLE_FEATURE_Y_V1])
        if obstacle_type == _PILLAR_OBSTACLE_TYPE_ID_V1:
            obstacles.append(
                ObstacleSceneV1(
                    obstacle_id=f"pov-obstacle-{obstacle_slot}",
                    kind="pillar",
                    center=center,
                    radius=row[OBSTACLE_FEATURE_RADIUS_V1],
                )
            )
        elif obstacle_type == _WALL_OBSTACLE_TYPE_ID_V1:
            obstacles.append(
                ObstacleSceneV1(
                    obstacle_id=f"pov-obstacle-{obstacle_slot}",
                    kind="wall",
                    center=center,
                    width=row[OBSTACLE_FEATURE_WIDTH_V1],
                    height=row[OBSTACLE_FEATURE_HEIGHT_V1],
                    theta=row[OBSTACLE_FEATURE_THETA_V1],
                )
            )
        else:
            raise ValueError("visible obstacle has no V1 POV presentation vocabulary.")
    return MapSceneV1(
        width=context[CONTEXT_FEATURE_MAP_WIDTH_V1],
        height=context[CONTEXT_FEATURE_MAP_HEIGHT_V1],
        obstacles=tuple(obstacles),
    )


def _visible_bodies(
    relation: Literal["ally", "enemy"],
    rows: tuple[tuple[float, ...], ...],
    visibility: tuple[bool, ...],
    public_agent_ids: tuple[str, ...],
    *,
    schema_version: int,
    configured_team_id: int,
) -> tuple[ActorPovVisibleBodySceneV1, ...]:
    """Decode only relation rows marked visible for the recipient.

    relation is ally/enemy; rows is a (5, 58) float tuple, visibility is bool
    (5,), and public_agent_ids maps its five rows. schema_version selects wire
    team/relation decoding; configured_team_id is recipient team 1/2. Return
    body rows in observation order. Visible inactive rows and invalid wire
    values raise ValueError. Hidden rows yield no position or body record.
    """
    bodies: list[ActorPovVisibleBodySceneV1] = []
    for observation_row, (row, visible) in enumerate(
        zip(rows, visibility, strict=True)
    ):
        if not visible:
            continue
        if not _decode_wire_bool(
            row[AGENT_FEATURE_ACTIVE_V1],
            name=f"{relation} row {observation_row} active",
        ):
            raise ValueError("visible POV body rows must be recorded active.")
        bodies.append(
            ActorPovVisibleBodySceneV1(
                relation=relation,
                observation_row=observation_row,
                public_agent_id=public_agent_ids[observation_row],
                position=_point(row),
                radius=row[AGENT_FEATURE_RADIUS_V1],
                team_id=decode_agent_feature_row(
                    row,
                    schema_version=schema_version,
                    team_id=configured_team_id
                    if relation == "ally"
                    else 3 - configured_team_id,
                    is_enemy=relation == "enemy",
                ).team_id,
                class_id=_decode_wire_int(
                    row[AGENT_FEATURE_CLASS_ID_V1],
                    name=f"{relation} row {observation_row} class",
                    minimum=1,
                    maximum=5,
                ),
                alive=_decode_wire_bool(
                    row[AGENT_FEATURE_ALIVE_V1],
                    name=f"{relation} row {observation_row} alive",
                ),
                current_health=row[AGENT_FEATURE_CURRENT_HEALTH_V1],
                max_health=row[AGENT_FEATURE_MAX_HEALTH_V1],
                effective_movement_speed=row[AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED_V1],
                ultimate_cooldown_remaining=_decode_wire_int(
                    row[AGENT_FEATURE_ULTIMATE_COOLDOWN_REMAINING_V1],
                    name=f"{relation} row {observation_row} cooldown",
                ),
                steps_until_out_of_combat=_decode_wire_int(
                    row[AGENT_FEATURE_STEPS_UNTIL_OUT_OF_COMBAT_V1],
                    name=f"{relation} row {observation_row} combat countdown",
                ),
                status_feature_values=_status_values(row),
            )
        )
    return tuple(bodies)


@dataclass(frozen=True, slots=True, kw_only=True)
class ActorPovProjectionIndexV1:
    """Validate POV replay content once for repeated frame selection.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    content : ActorPovReplayContent
        Exact ActorPovReplayContentV1 or V2 containing a coherent recorded prefix.

    Raises
    ------
    TypeError
        content is not an exact supported replay-content root.
    ValueError
        Replay content fails its owned structural or temporal validation.

    Notes
    -----
    The immutable index retains the existing content object. Frame lookup uses
    direct tuple indexing; scene decoding still performs fixed-size work for the
    selected frame.
    """

    content: ActorPovReplayContent
    """Exact ActorPovReplayContentV1 or V2 containing a coherent recorded prefix."""

    def __post_init__(self) -> None:
        """Validate the complete recipient replay before index use.

        Raise TypeError for a nonexact replay-content root and ValueError for a
        malformed trajectory through validate_actor_pov_replay_content. Retain the
        same immutable content and return None; no index file is written.
        """
        if (
            type(self.content) is not ActorPovReplayContentV1
            and type(self.content) is not ActorPovReplayContentV2
        ):
            raise TypeError("content must be the exact ActorPovReplayContentV1 root.")
        validate_actor_pov_replay_content(self.content)


def build_actor_pov_projection_index_v1(
    content: ActorPovReplayContent,
) -> ActorPovProjectionIndexV1:
    """Validate a captured POV prefix for repeated interactive frame selection.

    Parameters
    ----------
    content : ActorPovReplayContentV1 or ActorPovReplayContentV2
        Exact recipient-sliced replay content to validate and retain.

    Returns
    -------
    ActorPovProjectionIndexV1
        Frozen index retaining the same content object. Pass it to the analyzer
        builder with a concrete frame_index to avoid revalidating the full prefix.

    Raises
    ------
    TypeError
        content is not an exact supported replay-content root.
    ValueError
        The captured POV content fails its structural or trajectory checks.

    Notes
    -----
    This host helper reads no file and never uses full researcher frames.
    """
    return ActorPovProjectionIndexV1(content=content)


def _build_actor_pov_battlefield_scene_v1(
    frame: ActorPovFrame,
    *,
    episode_id: str,
    selected_global_slot: int,
    selected_team_local_slot: int,
    public_agent_id: str,
    configured_team_id: int,
    class_id: int,
    observation_materialization: Literal["exact_no_shared_obs_actor_input"],
    axis_mapping: ActorPovAxisMapping,
) -> ActorPovBattlefieldSceneV1:
    """Decode an already authorized recipient frame into battlefield facts.

    frame and axis_mapping must be exact supported versioned POV roots.
    episode_id/public_agent_id and selected_global_slot/selected_team_local_slot
    identify the recipient; configured_team_id/class_id must match its self row.
    observation_materialization is exact_no_shared_obs_actor_input. Return a
    new scene using only self, visible rows, public map/lifecycle and supplied
    axis labels. Raise TypeError for wrong roots and ValueError for an inactive
    self row, identity mismatch or invalid decoded record. No Oracle data is
    accepted and no new information is inferred.
    """
    if type(frame) is not ActorPovFrameV1 and type(frame) is not ActorPovFrameV2:
        raise TypeError("selected POV frame must be the exact V1 root.")
    if (
        type(axis_mapping) is not ActorPovAxisMappingV1
        and type(axis_mapping) is not ActorPovAxisMappingV2
    ):
        raise TypeError("POV axis mapping must be the exact V1 root.")
    self_row = frame.self_features
    if not _decode_wire_bool(
        self_row[AGENT_FEATURE_ACTIVE_V1],
        name="selected actor active",
    ):
        raise ValueError("selected POV self row must remain configured active.")
    if (
        decode_agent_feature_row(
            self_row,
            schema_version=frame.schema_version,
            team_id=configured_team_id,
            is_enemy=False,
        ).team_id
        != configured_team_id
        or _decode_wire_int(
            self_row[AGENT_FEATURE_CLASS_ID_V1],
            name="selected actor class",
            minimum=1,
            maximum=5,
        )
        != class_id
    ):
        raise ValueError("POV self row team/class must join content identity.")
    lifecycle = frame.spawn_lifecycle
    own_team_index = 0
    own_spawn_shield = lifecycle.spawn_shield_actual_durations_by_team[own_team_index][
        selected_team_local_slot
    ]
    self_actor = ActorPovSelfSceneV1(
        global_slot=selected_global_slot,
        public_agent_id=public_agent_id,
        team_local_slot=selected_team_local_slot,
        team_id=configured_team_id,
        class_id=class_id,
        position=_point(self_row),
        radius=self_row[AGENT_FEATURE_RADIUS_V1],
        alive=_decode_wire_bool(
            self_row[AGENT_FEATURE_ALIVE_V1],
            name="selected actor alive",
        ),
        current_health=self_row[AGENT_FEATURE_CURRENT_HEALTH_V1],
        max_health=self_row[AGENT_FEATURE_MAX_HEALTH_V1],
        effective_movement_speed=self_row[AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED_V1],
        ultimate_cooldown_remaining=_decode_wire_int(
            self_row[AGENT_FEATURE_ULTIMATE_COOLDOWN_REMAINING_V1],
            name="selected actor cooldown",
        ),
        steps_until_out_of_combat=_decode_wire_int(
            self_row[AGENT_FEATURE_STEPS_UNTIL_OUT_OF_COMBAT_V1],
            name="selected actor combat countdown",
        ),
        spawn_shield_remaining=own_spawn_shield,
        status_feature_values=_status_values(self_row),
    )
    spawn_pads = tuple(
        ActorPovSpawnPadSceneV1(
            actor_relative_team_index=team_index,
            team_relation="own" if team_index == 0 else "opponent",
            team_label=axis_mapping.spawn_lifecycle_team_axis_name_by_id[team_index],
            team_local_slot=team_local_slot,
            position=(position[0], position[1]),
            configured_active=lifecycle.active_mask_by_team[team_index][
                team_local_slot
            ],
            currently_alive=lifecycle.alive_mask_by_team[team_index][team_local_slot],
            spawn_shield_remaining=(
                lifecycle.spawn_shield_actual_durations_by_team[team_index][
                    team_local_slot
                ]
            ),
        )
        for team_index, team_positions in enumerate(
            lifecycle.spawn_pad_positions_by_team
        )
        for team_local_slot, position in enumerate(team_positions)
    )
    respawn_waves = tuple(
        ActorPovRespawnWaveSceneV1(
            actor_relative_team_index=team_index,
            team_relation="own" if team_index == 0 else "opponent",
            team_label=axis_mapping.spawn_lifecycle_team_axis_name_by_id[team_index],
            period_steps=lifecycle.respawn_wave_period_step_count_by_team[team_index],
            countdown_steps=lifecycle.respawn_wave_countdowns_by_team[team_index],
        )
        for team_index in range(2)
    )
    return ActorPovBattlefieldSceneV1(
        schema_version=ACTOR_POV_SCENE_SCHEMA_VERSION,
        audience_badge=f"AGENT POV · {public_agent_id}",
        observation_materialization=observation_materialization,
        episode_id=episode_id,
        frame_index=frame.frame_index,
        pov_frame_id=frame.pov_frame_id,
        source_frame_id=frame.source_frame_id,
        simulator_step_count=frame.simulator_step_count,
        map=_map_scene(frame.map_obstacle_features, frame.context_features),
        self_actor=self_actor,
        visible_bodies=(
            *_visible_bodies(
                "ally",
                frame.ally_unit_features,
                frame.ally_visibility_mask,
                axis_mapping.ally_observation_row_public_agent_id_by_id,
                schema_version=frame.schema_version,
                configured_team_id=configured_team_id,
            ),
            *_visible_bodies(
                "enemy",
                frame.enemy_unit_features,
                frame.enemy_visibility_mask,
                axis_mapping.enemy_observation_row_public_agent_id_by_id,
                schema_version=frame.schema_version,
                configured_team_id=configured_team_id,
            ),
        ),
        spawn_pads=spawn_pads,
        respawn_waves=respawn_waves,
    )


def build_actor_pov_analyzer_projection_v1(
    source: (ActorPovProjectionIndexV1 | ActorPovReplayContent | ActorPovCurrentSlice),
    *,
    frame_index: int | None = None,
) -> ActorPovAnalyzerProjectionV1:
    """Build one scene and cue envelope from recipient-authorized POV data.

    Parameters
    ----------
    source : ActorPovProjectionIndexV1, ActorPovReplayContent or ActorPovCurrentSlice
        Exact supported root. An index reuses prior whole-prefix validation;
        raw replay content creates an index for this call. A live slice supplies
        its own current frame and incoming cues. V1 and V2 sources are supported.
    frame_index : int or None, optional
        Required Python frame index for an index or replay, in its captured range.
        For a current slice, None selects its own frame; a supplied value must
        equal that frame's index.

    Returns
    -------
    ActorPovAnalyzerProjectionV1
        New scene, the frame's exact next-decision mask, and its incoming POV
        cues. Frame zero has no incoming transition and an empty cue tuple.

    Raises
    ------
    TypeError
        Source or a required nested root is not an exact supported record type.
    IndexError
        Replay/index frame_index is missing, not a Python int or outside range.
    ValueError
        A current-slice index differs, or source validation/decoding/joins fail.

    Notes
    -----
    The builder uses only recipient-authorized content and public axis labels.
    It neither reads an episode snapshot nor reconstructs hidden bodies. Input
    records are unchanged. Use the reusable index for repeated replay selection;
    passing raw content repeats whole-prefix validation each call.
    """
    if type(source) is ActorPovProjectionIndexV1:
        content = source.content
        if type(frame_index) is not int or not 0 <= frame_index < len(content.frames):
            raise IndexError("frame_index is outside the captured POV prefix.")
        frame = content.frames[frame_index]
        incoming = None if frame_index == 0 else content.transitions[frame_index - 1]
        episode_id = content.episode_id
        selected_global_slot = content.selected_global_slot
        selected_team_local_slot = content.selected_team_local_slot
        public_agent_id = content.public_agent_id
        configured_team_id = content.configured_team_id
        class_id = content.class_id
        observation_materialization = content.observation_materialization
        axis_mapping = content.axis_mapping
    elif (
        type(source) is ActorPovReplayContentV1
        or type(source) is ActorPovReplayContentV2
    ):
        return build_actor_pov_analyzer_projection_v1(
            build_actor_pov_projection_index_v1(source),
            frame_index=frame_index,
        )
    elif (
        type(source) is ActorPovCurrentSliceV1 or type(source) is ActorPovCurrentSliceV2
    ):
        if frame_index is not None and frame_index != source.frame.frame_index:
            raise ValueError(
                "a live current slice accepts only its own canonical frame index."
            )
        frame = source.frame
        incoming = source.incoming_transition
        episode_id = source.episode_id
        selected_global_slot = source.selected_global_slot
        selected_team_local_slot = source.selected_team_local_slot
        public_agent_id = source.public_agent_id
        configured_team_id = source.configured_team_id
        class_id = source.class_id
        observation_materialization = source.observation_materialization
        axis_mapping = source.axis_mapping
    else:
        raise TypeError(
            "source must be an exact POV projection index, replay content, or "
            "current slice."
        )
    if incoming is not None and type(incoming) is not ActorPovTransitionV1:
        raise TypeError("incoming POV transition must be the exact V1 root.")
    scene = _build_actor_pov_battlefield_scene_v1(
        frame,
        episode_id=episode_id,
        selected_global_slot=selected_global_slot,
        selected_team_local_slot=selected_team_local_slot,
        public_agent_id=public_agent_id,
        configured_team_id=configured_team_id,
        class_id=class_id,
        observation_materialization=observation_materialization,
        axis_mapping=axis_mapping,
    )
    return ActorPovAnalyzerProjectionV1(
        scene=scene,
        next_decision_action_mask=frame.action_mask,
        incoming_transition_id=(
            None if incoming is None else incoming.pov_transition_id
        ),
        incoming_cues=() if incoming is None else incoming.cues,
    )


__all__ = [
    "ACTOR_POV_SCENE_SCHEMA_VERSION",
    "ActorPovAnalyzerProjectionV1",
    "ActorPovBattlefieldSceneV1",
    "ActorPovProjectionIndexV1",
    "ActorPovRespawnWaveSceneV1",
    "ActorPovSelfSceneV1",
    "ActorPovSpawnPadSceneV1",
    "ActorPovVisibleBodySceneV1",
    "build_actor_pov_analyzer_projection_v1",
    "build_actor_pov_projection_index_v1",
]
