"""Independent renderer-neutral projection of recipient-sliced POV content."""

from dataclasses import dataclass
from math import isfinite
from typing import Literal, cast

from marl_battlegrounds.evaluation.pov import (
    ActorPovActionMaskV1,
    ActorPovPresentationCueV1,
    ActorPovReplayContentV1,
    validate_actor_pov_replay_content_v1,
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
    AGENT_FEATURE_TEAM_ID_V1,
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
)
from marl_battlegrounds.rendering.scene import MapSceneV1, ObstacleSceneV1, Point2D

ACTOR_POV_SCENE_SCHEMA_VERSION = 1

_PILLAR_OBSTACLE_TYPE_ID_V1 = 1
_WALL_OBSTACLE_TYPE_ID_V1 = 2


def _require_text(value: str, *, name: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a non-empty Python string.")


def _require_int(value: int, *, name: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be a Python int at least {minimum}.")


def _require_float(value: float, *, name: str, minimum: float = 0.0) -> None:
    if type(value) is not float or not isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be a finite Python float at least {minimum}.")


def _require_point(value: Point2D, *, name: str) -> None:
    if type(value) is not tuple or len(value) != 2:
        raise ValueError(f"{name} must be a two-coordinate Python tuple.")
    for coordinate in value:
        _require_float(coordinate, name=f"{name} coordinate", minimum=-float("inf"))


def _decode_wire_bool(value: float, *, name: str) -> bool:
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
    if type(value) is not float or not isfinite(value) or not value.is_integer():
        raise ValueError(f"{name} must be an integer-valued finite wire float.")
    decoded = int(value)
    if decoded < minimum or (maximum is not None and decoded > maximum):
        raise ValueError(f"{name} is outside its V1 wire domain.")
    return decoded


@dataclass(frozen=True, slots=True, kw_only=True)
class ActorPovSelfSceneV1:
    """The selected actor's exact self row and public export identity."""

    global_slot: int
    public_agent_id: str
    team_local_slot: int
    team_id: int
    class_id: int
    position: Point2D
    radius: float
    alive: bool
    current_health: float
    max_health: float
    effective_movement_speed: float
    ultimate_cooldown_remaining: int
    steps_until_out_of_combat: int
    spawn_shield_remaining: int
    status_feature_values: tuple[float, ...]

    def __post_init__(self) -> None:
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
    """One visible ally/enemy observation row without a guessed global identity."""

    relation: Literal["ally", "enemy"]
    observation_row: int
    public_agent_id: str
    position: Point2D
    radius: float
    team_id: int
    class_id: int
    alive: bool
    current_health: float
    max_health: float
    effective_movement_speed: float
    ultimate_cooldown_remaining: int
    steps_until_out_of_combat: int
    status_feature_values: tuple[float, ...]

    def __post_init__(self) -> None:
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
    """One pad row explicitly present in the recipient's lifecycle input."""

    actor_relative_team_index: int
    team_relation: Literal["own", "opponent"]
    team_label: str
    team_local_slot: int
    position: Point2D
    configured_active: bool
    currently_alive: bool
    spawn_shield_remaining: int

    def __post_init__(self) -> None:
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
    """One team wave row copied from the selected actor's authorized input."""

    actor_relative_team_index: int
    team_relation: Literal["own", "opponent"]
    team_label: str
    period_steps: int
    countdown_steps: int

    def __post_init__(self) -> None:
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
    """One recipient-authorized POV battlefield with no researcher snapshot."""

    schema_version: int
    audience_badge: str
    observation_materialization: Literal["exact_no_shared_obs_actor_input"]
    episode_id: str
    frame_index: int
    pov_frame_id: str
    source_frame_id: str
    simulator_step_count: int
    map: MapSceneV1
    self_actor: ActorPovSelfSceneV1
    visible_bodies: tuple[ActorPovVisibleBodySceneV1, ...]
    spawn_pads: tuple[ActorPovSpawnPadSceneV1, ...]
    respawn_waves: tuple[ActorPovRespawnWaveSceneV1, ...]

    def __post_init__(self) -> None:
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
    """POV scene, next-decision mask, and incoming POV-local cue batch."""

    scene: ActorPovBattlefieldSceneV1
    next_decision_action_mask: ActorPovActionMaskV1
    incoming_transition_id: str | None
    incoming_cues: tuple[ActorPovPresentationCueV1, ...]

    def __post_init__(self) -> None:
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
    return (row[AGENT_FEATURE_X_V1], row[AGENT_FEATURE_Y_V1])


def _status_values(row: tuple[float, ...]) -> tuple[float, ...]:
    return row[AGENT_STATUS_FEATURE_START_V1:AGENT_STATUS_FEATURE_STOP_V1]


def _map_scene(
    frame_rows: tuple[tuple[float, ...], ...], context: tuple[float, ...]
) -> MapSceneV1:
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
) -> tuple[ActorPovVisibleBodySceneV1, ...]:
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
                team_id=_decode_wire_int(
                    row[AGENT_FEATURE_TEAM_ID_V1],
                    name=f"{relation} row {observation_row} team",
                    minimum=1,
                    maximum=2,
                ),
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
    """Once-validated POV content supporting O(1) frame projection."""

    content: ActorPovReplayContentV1

    def __post_init__(self) -> None:
        if type(self.content) is not ActorPovReplayContentV1:
            raise TypeError("content must be the exact ActorPovReplayContentV1 root.")
        validate_actor_pov_replay_content_v1(self.content)


def build_actor_pov_projection_index_v1(
    content: ActorPovReplayContentV1,
) -> ActorPovProjectionIndexV1:
    """Validate recipient content once before interactive frame selection."""
    return ActorPovProjectionIndexV1(content=content)


def build_actor_pov_analyzer_projection_v1(
    source: ActorPovProjectionIndexV1 | ActorPovReplayContentV1,
    *,
    frame_index: int,
) -> ActorPovAnalyzerProjectionV1:
    """Build one projection exclusively from recipient-authorized POV content."""
    if type(source) is ActorPovProjectionIndexV1:
        index = source
    elif type(source) is ActorPovReplayContentV1:
        index = build_actor_pov_projection_index_v1(source)
    else:
        raise TypeError(
            "source must be ActorPovProjectionIndexV1 or ActorPovReplayContentV1."
        )
    content = index.content
    if type(frame_index) is not int or not 0 <= frame_index < len(content.frames):
        raise IndexError("frame_index is outside the captured POV prefix.")
    frame = content.frames[frame_index]
    self_row = frame.self_features
    if not _decode_wire_bool(
        self_row[AGENT_FEATURE_ACTIVE_V1],
        name="selected actor active",
    ):
        raise ValueError("selected POV self row must remain configured active.")
    if (
        _decode_wire_int(
            self_row[AGENT_FEATURE_TEAM_ID_V1],
            name="selected actor team",
            minimum=1,
            maximum=2,
        )
        != content.configured_team_id
        or _decode_wire_int(
            self_row[AGENT_FEATURE_CLASS_ID_V1],
            name="selected actor class",
            minimum=1,
            maximum=5,
        )
        != content.class_id
    ):
        raise ValueError("POV self row team/class must join content identity.")
    lifecycle = frame.spawn_lifecycle
    own_team_index = 0
    own_spawn_shield = lifecycle.spawn_shield_actual_durations_by_team[own_team_index][
        content.selected_team_local_slot
    ]
    self_actor = ActorPovSelfSceneV1(
        global_slot=content.selected_global_slot,
        public_agent_id=content.public_agent_id,
        team_local_slot=content.selected_team_local_slot,
        team_id=content.configured_team_id,
        class_id=content.class_id,
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
            team_label=content.axis_mapping.spawn_lifecycle_team_axis_name_by_id[
                team_index
            ],
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
            team_label=content.axis_mapping.spawn_lifecycle_team_axis_name_by_id[
                team_index
            ],
            period_steps=lifecycle.respawn_wave_period_step_count_by_team[team_index],
            countdown_steps=lifecycle.respawn_wave_countdowns_by_team[team_index],
        )
        for team_index in range(2)
    )
    incoming = None if frame_index == 0 else content.transitions[frame_index - 1]
    return ActorPovAnalyzerProjectionV1(
        scene=ActorPovBattlefieldSceneV1(
            schema_version=ACTOR_POV_SCENE_SCHEMA_VERSION,
            audience_badge=f"AGENT POV · {content.public_agent_id}",
            observation_materialization=content.observation_materialization,
            episode_id=content.episode_id,
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
                    content.axis_mapping.ally_observation_row_public_agent_id_by_id,
                ),
                *_visible_bodies(
                    "enemy",
                    frame.enemy_unit_features,
                    frame.enemy_visibility_mask,
                    content.axis_mapping.enemy_observation_row_public_agent_id_by_id,
                ),
            ),
            spawn_pads=spawn_pads,
            respawn_waves=respawn_waves,
        ),
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
