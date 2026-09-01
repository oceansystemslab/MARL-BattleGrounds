"""Authoritative host compilation and validation for DevClient drafts.

The browser edits JSON-shaped product drafts.  This module is the only bridge
from those drafts to existing MARL-BGs runtime objects: it normalizes content,
constructs an ``EnvConfig``, calls ordinary reset, overlays the explicitly
authorable ``EnvState`` leaves, forces neutral previous-action history, and
then invokes the unchanged core validators and authored-state initializer.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from marl_battlegrounds.core import combat
from marl_battlegrounds.core.config import (
    CANONICAL_PRODUCT_MOVEMENT_SCALE,
    resolve_agent_profile,
    validate_product_env_config,
    validate_scenario_initial_state,
)
from marl_battlegrounds.core.env import initialize_scenario_state, reset
from marl_battlegrounds.core.geometry import disc_overlaps_obstacle
from marl_battlegrounds.core.types import (
    HUNTER_CLASS_ID,
    MAGE_CLASS_ID,
    MAX_AGENT_SLOTS,
    MAX_OBSTACLE_SLOTS,
    NEUTRAL_CLASS_ID,
    NUM_SLOW_CHANNELS,
    NUM_STUN_CHANNELS,
    OBSTACLE_FEATURE_ACTIVE,
    OBSTACLE_FEATURE_HEIGHT,
    OBSTACLE_FEATURE_RADIUS,
    OBSTACLE_FEATURE_THETA,
    OBSTACLE_FEATURE_TYPE,
    OBSTACLE_FEATURE_WIDTH,
    OBSTACLE_FEATURE_X,
    OBSTACLE_FEATURE_Y,
    OBSTACLE_FEATURES,
    OBSTACLE_TYPE_PILLAR,
    OBSTACLE_TYPE_WALL,
    PRIEST_CLASS_ID,
    ROGUE_CLASS_ID,
    TASK_MODE_TDM,
    WARRIOR_CLASS_ID,
    ActionMask,
    EnvConfig,
    EnvState,
    Info,
    Observation,
)
from marl_battlegrounds.evaluation.catalog import build_resolved_env_config_v1
from marl_battlegrounds.evaluation.models import canonical_digest_sha256
from scripts.dev.visual_debugger.authoring_models import (
    DevAgentStateV1,
    DevAuthoringProblemV1,
    DevMapContentV1,
    DevMapDraftV1,
    DevObstacleV1,
    DevPillarV1,
    DevPointV1,
    DevRosterSlotV1,
    DevScenarioContentV1,
    DevScenarioDraftV1,
    DevSpawnPadV1,
    DevWallV1,
)

_INT32_MIN = int(np.iinfo(np.int32).min)
_INT32_MAX = int(np.iinfo(np.int32).max)
_CLASS_ID_BY_NAME = {
    "mage": MAGE_CLASS_ID,
    "warrior": WARRIOR_CLASS_ID,
    "hunter": HUNTER_CLASS_ID,
    "rogue": ROGUE_CLASS_ID,
    "priest": PRIEST_CLASS_ID,
}


class DevAuthoringValidationError(ValueError):
    """An authoring operation failed with stable linked validation problems."""

    def __init__(self, problems: tuple[DevAuthoringProblemV1, ...]) -> None:
        if not problems or not any(problem.severity == "error" for problem in problems):
            raise ValueError("DevAuthoringValidationError requires at least one error")
        self.problems = problems
        super().__init__("; ".join(problem.message for problem in problems))


class _NumericNormalizationError(ValueError):
    """One authoring number cannot be represented by the float32 runtime."""

    def __init__(
        self,
        field_path: str,
        *,
        object_id: str | None = None,
    ) -> None:
        self.field_path = field_path
        self.object_id = object_id
        super().__init__("Value is not representable as finite float32.")


@dataclass(frozen=True, slots=True)
class CompiledDevMapV1:
    content: DevMapContentV1
    obstacles: Array
    team_spawn_pad_positions: Array
    semantic_digest: str
    problems: tuple[DevAuthoringProblemV1, ...]


@dataclass(frozen=True, slots=True)
class CompiledDevScenarioV1:
    content: DevScenarioContentV1
    config: EnvConfig
    initial_state: EnvState
    observation: Observation
    action_mask: ActionMask
    info: Info
    map_semantic_digest: str
    semantic_digest: str
    resolved_configuration_digest: str
    resolved_initial_state_digest: str
    problems: tuple[DevAuthoringProblemV1, ...]


def _problem(
    severity: Literal["error", "warning"],
    code: str,
    message: str,
    field_path: str,
    *,
    object_id: str | None = None,
) -> DevAuthoringProblemV1:
    return DevAuthoringProblemV1(
        severity=severity,
        stable_code=code,
        message=message,
        object_id=object_id,
        field_path=field_path,
    )


def _normalization_problem(
    code: str,
    error: ValueError,
    *,
    fallback_field_path: str,
) -> DevAuthoringProblemV1:
    if isinstance(error, _NumericNormalizationError):
        return _problem(
            "error",
            code,
            str(error),
            error.field_path,
            object_id=error.object_id,
        )
    return _problem("error", code, str(error), fallback_field_path)


def _float32(value: float) -> float:
    try:
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            normalized = np.float32(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("value is not representable as finite float32") from error
    if not bool(np.isfinite(normalized)):
        raise ValueError("value is not representable as finite float32")
    return float(normalized)


def _normalized_float32(
    value: float,
    field_path: str,
    *,
    object_id: str | None = None,
) -> float:
    try:
        return _float32(value)
    except ValueError as error:
        raise _NumericNormalizationError(
            field_path,
            object_id=object_id,
        ) from error


def _normalized_radians(rotation_degrees: float) -> float:
    radians = math.radians(rotation_degrees)
    normalized = (radians + math.pi) % (2.0 * math.pi) - math.pi
    return _float32(normalized)


def _normalized_degrees_from_radians(radians: float) -> float:
    return _float32(math.degrees(radians))


def _normalized_rotation_degrees(
    value: float,
    field_path: str,
    *,
    object_id: str,
) -> float:
    try:
        # Reject an unrepresentable authored value before periodic normalization.
        _float32(value)
        return _normalized_degrees_from_radians(_normalized_radians(value))
    except (OverflowError, ValueError) as error:
        raise _NumericNormalizationError(
            field_path,
            object_id=object_id,
        ) from error


def _field_path(prefix: str, field_path: str) -> str:
    return field_path if not prefix else f"{prefix}.{field_path}"


def normalize_map_content(
    content: DevMapContentV1,
    *,
    field_prefix: str = "",
) -> DevMapContentV1:
    """Return deterministic float32 map content without changing object order."""
    normalized_obstacles: list[DevObstacleV1] = []
    for obstacle_index, obstacle in enumerate(content.obstacles):
        obstacle_prefix = _field_path(
            field_prefix,
            f"obstacles.{obstacle_index}",
        )
        if isinstance(obstacle, DevWallV1):
            normalized_obstacles.append(
                DevWallV1(
                    object_id=obstacle.object_id,
                    center_x=_normalized_float32(
                        obstacle.center_x,
                        f"{obstacle_prefix}.center_x",
                        object_id=obstacle.object_id,
                    ),
                    center_y=_normalized_float32(
                        obstacle.center_y,
                        f"{obstacle_prefix}.center_y",
                        object_id=obstacle.object_id,
                    ),
                    width=_normalized_float32(
                        obstacle.width,
                        f"{obstacle_prefix}.width",
                        object_id=obstacle.object_id,
                    ),
                    height=_normalized_float32(
                        obstacle.height,
                        f"{obstacle_prefix}.height",
                        object_id=obstacle.object_id,
                    ),
                    rotation_degrees=_normalized_rotation_degrees(
                        obstacle.rotation_degrees,
                        f"{obstacle_prefix}.rotation_degrees",
                        object_id=obstacle.object_id,
                    ),
                )
            )
        else:
            normalized_obstacles.append(
                DevPillarV1(
                    object_id=obstacle.object_id,
                    center_x=_normalized_float32(
                        obstacle.center_x,
                        f"{obstacle_prefix}.center_x",
                        object_id=obstacle.object_id,
                    ),
                    center_y=_normalized_float32(
                        obstacle.center_y,
                        f"{obstacle_prefix}.center_y",
                        object_id=obstacle.object_id,
                    ),
                    radius=_normalized_float32(
                        obstacle.radius,
                        f"{obstacle_prefix}.radius",
                        object_id=obstacle.object_id,
                    ),
                )
            )
    normalized_pads = tuple(
        DevSpawnPadV1(
            object_id=pad.object_id,
            team=pad.team,
            team_local_slot=pad.team_local_slot,
            position=DevPointV1(
                x=_normalized_float32(
                    pad.position.x,
                    _field_path(
                        field_prefix,
                        f"spawn_pads.{pad_index}.position.x",
                    ),
                    object_id=pad.object_id,
                ),
                y=_normalized_float32(
                    pad.position.y,
                    _field_path(
                        field_prefix,
                        f"spawn_pads.{pad_index}.position.y",
                    ),
                    object_id=pad.object_id,
                ),
            ),
        )
        for pad_index, pad in enumerate(content.spawn_pads)
    )
    return DevMapContentV1(
        name=content.name,
        description=content.description,
        width=_normalized_float32(
            content.width,
            _field_path(field_prefix, "width"),
        ),
        height=_normalized_float32(
            content.height,
            _field_path(field_prefix, "height"),
        ),
        obstacles=tuple(normalized_obstacles),
        spawn_pads=normalized_pads,
    )


def normalize_scenario_content(
    content: DevScenarioContentV1,
) -> DevScenarioContentV1:
    """Normalize every persisted floating scenario input to runtime float32."""
    normalized_states = tuple(
        state.model_copy(
            update={
                "position": DevPointV1(
                    x=_normalized_float32(
                        state.position.x,
                        f"agent_states.{global_slot}.position.x",
                        object_id=state.object_id,
                    ),
                    y=_normalized_float32(
                        state.position.y,
                        f"agent_states.{global_slot}.position.y",
                        object_id=state.object_id,
                    ),
                ),
                "current_health": _normalized_float32(
                    state.current_health,
                    f"agent_states.{global_slot}.current_health",
                    object_id=state.object_id,
                ),
            }
        )
        for global_slot, state in enumerate(content.agent_states)
    )
    normalized_episode = content.episode.model_copy(
        update={
            "spawn_shield_movement_speed": _normalized_float32(
                content.episode.spawn_shield_movement_speed,
                "episode.spawn_shield_movement_speed",
            )
        }
    )
    return content.model_copy(
        update={
            "embedded_map": normalize_map_content(
                content.embedded_map,
                field_prefix="embedded_map",
            ),
            "episode": normalized_episode,
            "agent_states": normalized_states,
        },
        deep=True,
    )


def _compile_obstacles(content: DevMapContentV1) -> Array:
    rows = np.zeros((MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES), dtype=np.float32)
    for obstacle_index, obstacle in enumerate(content.obstacles):
        if obstacle_index >= MAX_OBSTACLE_SLOTS:
            break
        row = rows[obstacle_index]
        row[OBSTACLE_FEATURE_ACTIVE] = 1.0
        row[OBSTACLE_FEATURE_X] = obstacle.center_x
        row[OBSTACLE_FEATURE_Y] = obstacle.center_y
        if isinstance(obstacle, DevWallV1):
            row[OBSTACLE_FEATURE_TYPE] = OBSTACLE_TYPE_WALL
            row[OBSTACLE_FEATURE_WIDTH] = obstacle.width
            row[OBSTACLE_FEATURE_HEIGHT] = obstacle.height
            row[OBSTACLE_FEATURE_THETA] = _normalized_radians(obstacle.rotation_degrees)
        else:
            row[OBSTACLE_FEATURE_TYPE] = OBSTACLE_TYPE_PILLAR
            row[OBSTACLE_FEATURE_RADIUS] = obstacle.radius
    return jnp.asarray(rows, dtype=jnp.float32)


def _compile_spawn_pads(content: DevMapContentV1) -> Array:
    rows = np.asarray(
        tuple((pad.position.x, pad.position.y) for pad in content.spawn_pads),
        dtype=np.float32,
    )
    return jnp.asarray(rows.reshape(2, 5, 2), dtype=jnp.float32)


def _obstacle_semantic_row(obstacle: DevObstacleV1) -> Mapping[str, object]:
    if isinstance(obstacle, DevWallV1):
        return {
            "kind": "wall",
            "center_x": _float32(obstacle.center_x),
            "center_y": _float32(obstacle.center_y),
            "width": _float32(obstacle.width),
            "height": _float32(obstacle.height),
            "rotation_radians": _normalized_radians(obstacle.rotation_degrees),
        }
    return {
        "kind": "pillar",
        "center_x": _float32(obstacle.center_x),
        "center_y": _float32(obstacle.center_y),
        "radius": _float32(obstacle.radius),
    }


def map_semantic_payload(content: DevMapContentV1) -> dict[str, object]:
    """Project map semantics while excluding names and browser object IDs."""
    normalized = normalize_map_content(content)
    return {
        "schema": "dev-map-semantics@1",
        "width": normalized.width,
        "height": normalized.height,
        "obstacles": tuple(
            _obstacle_semantic_row(obstacle) for obstacle in normalized.obstacles
        ),
        "spawn_pads": tuple(
            {
                "team": pad.team,
                "team_local_slot": pad.team_local_slot,
                "x": pad.position.x,
                "y": pad.position.y,
            }
            for pad in normalized.spawn_pads
        ),
    }


def map_semantic_digest(content: DevMapContentV1) -> str:
    return canonical_digest_sha256(map_semantic_payload(content))


def _obstacle_extents(obstacle: DevObstacleV1) -> tuple[float, float, float, float]:
    if isinstance(obstacle, DevPillarV1):
        return (
            obstacle.center_x - obstacle.radius,
            obstacle.center_x + obstacle.radius,
            obstacle.center_y - obstacle.radius,
            obstacle.center_y + obstacle.radius,
        )
    radians = _normalized_radians(obstacle.rotation_degrees)
    extent_x = (
        abs(math.cos(radians)) * obstacle.width / 2.0
        + abs(math.sin(radians)) * obstacle.height / 2.0
    )
    extent_y = (
        abs(math.sin(radians)) * obstacle.width / 2.0
        + abs(math.cos(radians)) * obstacle.height / 2.0
    )
    return (
        obstacle.center_x - extent_x,
        obstacle.center_x + extent_x,
        obstacle.center_y - extent_y,
        obstacle.center_y + extent_y,
    )


def validate_map_content(
    content: DevMapContentV1,
) -> tuple[DevAuthoringProblemV1, ...]:
    """Validate reusable-map rules and return stable linked problems."""
    problems: list[DevAuthoringProblemV1] = []
    try:
        normalized = normalize_map_content(content)
    except ValueError as error:
        return (
            _normalization_problem(
                "map-float32-normalization-failed",
                error,
                fallback_field_path="map",
            ),
        )

    if normalized.width <= 0.0:
        problems.append(
            _problem(
                "error",
                "map-width-not-positive",
                "Map width must be strictly positive.",
                "width",
            )
        )
    if normalized.height <= 0.0:
        problems.append(
            _problem(
                "error",
                "map-height-not-positive",
                "Map height must be strictly positive.",
                "height",
            )
        )
    for obstacle_index, obstacle in enumerate(normalized.obstacles):
        if isinstance(obstacle, DevWallV1):
            for field_name, value in (
                ("width", obstacle.width),
                ("height", obstacle.height),
            ):
                if value <= 0.0:
                    problems.append(
                        _problem(
                            "error",
                            f"map-wall-{field_name}-not-positive",
                            f"Wall {field_name} must be strictly positive.",
                            f"obstacles.{obstacle_index}.{field_name}",
                            object_id=obstacle.object_id,
                        )
                    )
        elif obstacle.radius <= 0.0:
            problems.append(
                _problem(
                    "error",
                    "map-pillar-radius-not-positive",
                    "Pillar radius must be strictly positive.",
                    f"obstacles.{obstacle_index}.radius",
                    object_id=obstacle.object_id,
                )
            )

    if len(normalized.obstacles) > MAX_OBSTACLE_SLOTS:
        problems.append(
            _problem(
                "error",
                "map-obstacle-capacity-exceeded",
                f"Map has {len(normalized.obstacles)} obstacles; capacity is "
                f"{MAX_OBSTACLE_SLOTS}.",
                "obstacles",
            )
        )
    if any(problem.severity == "error" for problem in problems):
        return tuple(problems)

    obstacles = _compile_obstacles(normalized)
    maximum_body_radius = float(np.max(np.asarray(combat.BODY_RADIUS_BY_CLASS)))
    pad_positions = np.asarray(_compile_spawn_pads(normalized)).reshape(10, 2)
    for pad_index, pad in enumerate(normalized.spawn_pads):
        x, y = pad_positions[pad_index]
        if not (
            maximum_body_radius <= float(x) <= normalized.width - maximum_body_radius
            and maximum_body_radius
            <= float(y)
            <= normalized.height - maximum_body_radius
        ):
            problems.append(
                _problem(
                    "error",
                    "map-spawn-pad-out-of-bounds",
                    f"{pad.team}{pad.team_local_slot} cannot fit the largest "
                    "supported body inside the map.",
                    f"spawn_pads.{pad_index}.position",
                    object_id=pad.object_id,
                )
            )
        for obstacle_index in range(len(normalized.obstacles)):
            if bool(
                disc_overlaps_obstacle(
                    jnp.asarray((x, y), dtype=jnp.float32),
                    jnp.asarray(maximum_body_radius, dtype=jnp.float32),
                    obstacles[obstacle_index],
                )
            ):
                problems.append(
                    _problem(
                        "error",
                        "map-spawn-pad-overlaps-obstacle",
                        f"{pad.team}{pad.team_local_slot} overlaps obstacle "
                        f"{obstacle_index} for the largest supported body.",
                        f"spawn_pads.{pad_index}.position",
                        object_id=pad.object_id,
                    )
                )

    for first_index in range(len(normalized.spawn_pads)):
        for second_index in range(first_index + 1, len(normalized.spawn_pads)):
            distance = math.dist(
                tuple(float(value) for value in pad_positions[first_index]),
                tuple(float(value) for value in pad_positions[second_index]),
            )
            if distance < 2.0 * maximum_body_radius:
                first = normalized.spawn_pads[first_index]
                second = normalized.spawn_pads[second_index]
                problems.append(
                    _problem(
                        "error",
                        "map-spawn-pad-overlap",
                        f"Pads {first.team}{first.team_local_slot} and "
                        f"{second.team}{second.team_local_slot} overlap for the "
                        "largest supported body.",
                        f"spawn_pads.{second_index}.position",
                        object_id=second.object_id,
                    )
                )

    extents = tuple(_obstacle_extents(obstacle) for obstacle in normalized.obstacles)
    for obstacle_index, (obstacle, (left, right, bottom, top)) in enumerate(
        zip(normalized.obstacles, extents, strict=True)
    ):
        if (
            left < 0.0
            or right > normalized.width
            or bottom < 0.0
            or top > normalized.height
        ):
            problems.append(
                _problem(
                    "warning",
                    "map-obstacle-outside-bounds",
                    "Obstacle lies partly or wholly outside the map; the simulator "
                    "permits this geometry.",
                    f"obstacles.{obstacle_index}",
                    object_id=obstacle.object_id,
                )
            )
    for first_index, first_extent in enumerate(extents):
        for second_index in range(first_index + 1, len(extents)):
            second_extent = extents[second_index]
            overlaps = not (
                first_extent[1] <= second_extent[0]
                or second_extent[1] <= first_extent[0]
                or first_extent[3] <= second_extent[2]
                or second_extent[3] <= first_extent[2]
            )
            if overlaps:
                second = normalized.obstacles[second_index]
                problems.append(
                    _problem(
                        "warning",
                        "map-obstacle-overlap",
                        "Obstacle bounds overlap another obstacle. This is legal, "
                        "but verify that the overlap is intentional.",
                        f"obstacles.{second_index}",
                        object_id=second.object_id,
                    )
                )
    return tuple(problems)


def compile_dev_map(
    source: DevMapDraftV1 | DevMapContentV1,
    *,
    require_valid: bool = True,
) -> CompiledDevMapV1:
    content = source.content if isinstance(source, DevMapDraftV1) else source
    try:
        normalized = normalize_map_content(content)
    except ValueError as error:
        raise DevAuthoringValidationError(
            (
                _normalization_problem(
                    "map-float32-normalization-failed",
                    error,
                    fallback_field_path="map",
                ),
            )
        ) from error
    problems = validate_map_content(normalized)
    if require_valid and any(problem.severity == "error" for problem in problems):
        raise DevAuthoringValidationError(problems)
    return CompiledDevMapV1(
        content=normalized,
        obstacles=_compile_obstacles(normalized),
        team_spawn_pad_positions=_compile_spawn_pads(normalized),
        semantic_digest=map_semantic_digest(normalized),
        problems=problems,
    )


def _active(global_slot: int, content: DevScenarioContentV1) -> bool:
    return global_slot % 5 < (
        content.team_a_size if global_slot < 5 else content.team_b_size
    )


def _scenario_custom_problems(
    content: DevScenarioContentV1,
) -> tuple[DevAuthoringProblemV1, ...]:
    problems = [
        problem.model_copy(
            update={
                "field_path": (
                    "embedded_map"
                    if problem.field_path == "map"
                    else f"embedded_map.{problem.field_path}"
                )
            }
        )
        for problem in validate_map_content(content.embedded_map)
    ]
    episode = content.episode
    state = content.global_state

    def require_int32(
        value: int,
        *,
        field_path: str,
        object_id: str | None = None,
    ) -> bool:
        if _INT32_MIN <= value <= _INT32_MAX:
            return True
        problems.append(
            _problem(
                "error",
                "scenario-integer-not-int32",
                "Value must fit the signed 32-bit runtime integer contract.",
                field_path,
                object_id=object_id,
            )
        )
        return False

    def require_minimum(
        value: int | float,
        minimum: int | float,
        *,
        code: str,
        message: str,
        field_path: str,
        object_id: str | None = None,
    ) -> bool:
        if value >= minimum:
            return True
        problems.append(
            _problem(
                "error",
                code,
                message,
                field_path,
                object_id=object_id,
            )
        )
        return False

    for value, field_path in (
        (content.task.score_threshold, "task.score_threshold"),
        (episode.max_steps, "episode.max_steps"),
        (
            episode.spawn_shield_duration_steps,
            "episode.spawn_shield_duration_steps",
        ),
        (
            episode.team_a_respawn_wave_period_steps,
            "episode.team_a_respawn_wave_period_steps",
        ),
        (
            episode.team_b_respawn_wave_period_steps,
            "episode.team_b_respawn_wave_period_steps",
        ),
        (state.step_count, "global_state.step_count"),
        (state.team_a_score, "global_state.team_a_score"),
        (state.team_b_score, "global_state.team_b_score"),
        (
            state.team_a_respawn_countdown,
            "global_state.team_a_respawn_countdown",
        ),
        (
            state.team_b_respawn_countdown,
            "global_state.team_b_respawn_countdown",
        ),
    ):
        require_int32(value, field_path=field_path)

    threshold_valid = require_minimum(
        content.task.score_threshold,
        1,
        code="scenario-score-threshold-not-positive",
        message="TDM score threshold must be at least one.",
        field_path="task.score_threshold",
    )
    max_steps_valid = require_minimum(
        episode.max_steps,
        1,
        code="scenario-max-steps-not-positive",
        message="max_steps must be at least one.",
        field_path="episode.max_steps",
    )
    require_minimum(
        episode.spawn_shield_duration_steps,
        0,
        code="scenario-spawn-shield-duration-negative",
        message="Spawn-shield duration must be nonnegative.",
        field_path="episode.spawn_shield_duration_steps",
    )
    if episode.spawn_shield_movement_speed <= 0.0:
        problems.append(
            _problem(
                "error",
                "scenario-spawn-shield-speed-not-positive",
                "Spawn-shield movement speed must be strictly positive.",
                "episode.spawn_shield_movement_speed",
            )
        )
    period_valid: dict[str, bool] = {}
    for team, period in (
        ("a", episode.team_a_respawn_wave_period_steps),
        ("b", episode.team_b_respawn_wave_period_steps),
    ):
        period_valid[team] = require_minimum(
            period,
            1,
            code="scenario-respawn-period-not-positive",
            message="Respawn-wave period must be at least one.",
            field_path=f"episode.team_{team}_respawn_wave_period_steps",
        )
    team_sizes_valid = True
    for team, team_size in (("a", content.team_a_size), ("b", content.team_b_size)):
        if not 1 <= team_size <= 5:
            team_sizes_valid = False
            problems.append(
                _problem(
                    "error",
                    "scenario-team-size-out-of-range",
                    "Team size must be between one and five.",
                    f"team_{team}_size",
                )
            )

    step_nonnegative = require_minimum(
        state.step_count,
        0,
        code="scenario-step-count-negative",
        message="step_count must be nonnegative.",
        field_path="global_state.step_count",
    )
    if step_nonnegative and max_steps_valid and state.step_count >= episode.max_steps:
        problems.append(
            _problem(
                "error",
                "scenario-step-count-out-of-range",
                "step_count must be strictly less than max_steps.",
                "global_state.step_count",
            )
        )
    for team, score in (("a", state.team_a_score), ("b", state.team_b_score)):
        score_nonnegative = require_minimum(
            score,
            0,
            code="scenario-score-negative",
            message="Current TDM score must be nonnegative.",
            field_path=f"global_state.team_{team}_score",
        )
        if (
            score_nonnegative
            and threshold_valid
            and score >= content.task.score_threshold
        ):
            problems.append(
                _problem(
                    "error",
                    "scenario-score-out-of-range",
                    "Current score must be strictly less than the TDM threshold.",
                    f"global_state.team_{team}_score",
                )
            )
    for team, countdown, period in (
        (
            "a",
            state.team_a_respawn_countdown,
            episode.team_a_respawn_wave_period_steps,
        ),
        (
            "b",
            state.team_b_respawn_countdown,
            episode.team_b_respawn_wave_period_steps,
        ),
    ):
        countdown_nonnegative = require_minimum(
            countdown,
            0,
            code="scenario-respawn-countdown-negative",
            message="Respawn countdown must be nonnegative.",
            field_path=f"global_state.team_{team}_respawn_countdown",
        )
        if countdown_nonnegative and period_valid[team] and countdown >= period:
            problems.append(
                _problem(
                    "error",
                    "scenario-respawn-countdown-out-of-range",
                    "Respawn countdown must be strictly less than its period.",
                    f"global_state.team_{team}_respawn_countdown",
                )
            )

    timer_fields = (
        "ultimate_cooldown_remaining",
        "spawn_shield_duration_remaining",
        "steps_until_out_of_combat",
        "warrior_charge_slow_duration",
        "hunter_basic_slow_duration",
        "rogue_poison_slow_duration",
        "warrior_charge_stun_duration",
        "hunter_trap_stun_duration",
        "rogue_poison_stun_duration",
        "rogue_poison_anti_heal_duration",
        "mage_burst_duration",
        "priest_blessing_of_freedom_duration",
    )
    for global_slot, (roster, agent) in enumerate(
        zip(content.roster, content.agent_states, strict=True)
    ):
        for field_name in timer_fields:
            require_int32(
                getattr(agent, field_name),
                field_path=f"agent_states.{global_slot}.{field_name}",
                object_id=agent.object_id,
            )
        if agent.current_health < 0.0:
            problems.append(
                _problem(
                    "error",
                    "scenario-agent-health-negative",
                    "Current health must be nonnegative.",
                    f"agent_states.{global_slot}.current_health",
                    object_id=agent.object_id,
                )
            )
        for field_name in timer_fields:
            if getattr(agent, field_name) < 0:
                problems.append(
                    _problem(
                        "error",
                        "scenario-agent-duration-negative",
                        "Cooldown and status durations must be nonnegative.",
                        f"agent_states.{global_slot}.{field_name}",
                        object_id=agent.object_id,
                    )
                )

        if not team_sizes_valid:
            continue
        active = _active(global_slot, content)
        field_prefix = f"agent_states.{global_slot}"
        if active:
            if roster.class_name == "not_applicable":
                problems.append(
                    _problem(
                        "error",
                        "scenario-active-class-missing",
                        "Active roster rows require one supported class.",
                        f"roster.{global_slot}.class_name",
                        object_id=roster.object_id,
                    )
                )
        else:
            if roster.class_name != "not_applicable":
                problems.append(
                    _problem(
                        "error",
                        "scenario-inactive-class-noncanonical",
                        "Inactive roster rows must use not_applicable class padding.",
                        f"roster.{global_slot}.class_name",
                        object_id=roster.object_id,
                    )
                )
            noncanonical_state = agent != DevAgentStateV1(
                object_id=agent.object_id,
                position=DevPointV1(x=0.0, y=0.0),
                alive=False,
                current_health=0.0,
            )
            if noncanonical_state:
                problems.append(
                    _problem(
                        "error",
                        "scenario-inactive-state-noncanonical",
                        "Inactive agent rows must be canonical zero padding.",
                        field_prefix,
                        object_id=agent.object_id,
                    )
                )
    return tuple(problems)


def _requested_class_ids(content: DevScenarioContentV1) -> Array:
    def class_id(global_slot: int, slot: DevRosterSlotV1) -> int:
        if not _active(global_slot, content):
            return NEUTRAL_CLASS_ID
        if slot.class_name == "not_applicable":
            raise ValueError("active roster slot has no supported class")
        return _CLASS_ID_BY_NAME[slot.class_name]

    return jnp.asarray(
        tuple(
            class_id(global_slot, slot)
            for global_slot, slot in enumerate(content.roster)
        ),
        dtype=jnp.int32,
    )


def _build_config(
    content: DevScenarioContentV1, compiled_map: CompiledDevMapV1
) -> EnvConfig:
    profile = resolve_agent_profile(
        _requested_class_ids(content),
        jnp.asarray((content.team_a_size, content.team_b_size), dtype=jnp.int32),
    )
    return EnvConfig(
        task_mode=TASK_MODE_TDM,
        team_deathmatch_score_threshold=content.task.score_threshold,
        max_steps=content.episode.max_steps,
        map_width=float(compiled_map.content.width),
        map_height=float(compiled_map.content.height),
        obstacles=compiled_map.obstacles,
        agent_profile=profile,
        ordinary_movement_distance_scale=float(CANONICAL_PRODUCT_MOVEMENT_SCALE),
        team_spawn_pad_positions=compiled_map.team_spawn_pad_positions,
        spawn_shield_duration_steps=content.episode.spawn_shield_duration_steps,
        spawn_shield_movement_speed=float(
            _float32(content.episode.spawn_shield_movement_speed)
        ),
        team_respawn_wave_period_step_count=jnp.asarray(
            (
                content.episode.team_a_respawn_wave_period_steps,
                content.episode.team_b_respawn_wave_period_steps,
            ),
            dtype=jnp.int32,
        ),
    )


def _overlay_authored_state(
    reset_state: EnvState,
    content: DevScenarioContentV1,
) -> EnvState:
    states = content.agent_states
    positions = jnp.asarray(
        tuple((state.position.x, state.position.y) for state in states),
        dtype=jnp.float32,
    )
    alive = jnp.asarray(tuple(state.alive for state in states), dtype=jnp.bool_)
    health = jnp.asarray(
        tuple(state.current_health for state in states), dtype=jnp.float32
    )
    cooldowns = jnp.asarray(
        tuple(state.ultimate_cooldown_remaining for state in states), dtype=jnp.int32
    )
    slow_durations = jnp.asarray(
        tuple(
            (
                state.warrior_charge_slow_duration,
                state.hunter_basic_slow_duration,
                state.rogue_poison_slow_duration,
            )
            for state in states
        ),
        dtype=jnp.int32,
    ).reshape(MAX_AGENT_SLOTS, NUM_SLOW_CHANNELS)
    stun_durations = jnp.asarray(
        tuple(
            (
                state.warrior_charge_stun_duration,
                state.hunter_trap_stun_duration,
                state.rogue_poison_stun_duration,
            )
            for state in states
        ),
        dtype=jnp.int32,
    ).reshape(MAX_AGENT_SLOTS, NUM_STUN_CHANNELS)
    return reset_state._replace(
        team_deathmatch_scores=jnp.asarray(
            (
                content.global_state.team_a_score,
                content.global_state.team_b_score,
            ),
            dtype=jnp.int32,
        ),
        step_count=jnp.asarray(content.global_state.step_count, dtype=jnp.int32),
        agent_positions=positions,
        alive_mask=alive,
        current_health=health,
        ultimate_cooldowns=cooldowns,
        slow_durations=slow_durations,
        stun_durations=stun_durations,
        rogue_poison_anti_heal_durations=jnp.asarray(
            tuple(state.rogue_poison_anti_heal_duration for state in states),
            dtype=jnp.int32,
        ),
        mage_burst_damage_amplification_durations=jnp.asarray(
            tuple(state.mage_burst_duration for state in states), dtype=jnp.int32
        ),
        priest_blessing_of_freedom_slow_floor_durations=jnp.asarray(
            tuple(state.priest_blessing_of_freedom_duration for state in states),
            dtype=jnp.int32,
        ),
        team_respawn_wave_countdowns=jnp.asarray(
            (
                content.global_state.team_a_respawn_countdown,
                content.global_state.team_b_respawn_countdown,
            ),
            dtype=jnp.int32,
        ),
        spawn_shield_durations=jnp.asarray(
            tuple(state.spawn_shield_duration_remaining for state in states),
            dtype=jnp.int32,
        ),
        steps_until_out_of_combat=jnp.asarray(
            tuple(state.steps_until_out_of_combat for state in states),
            dtype=jnp.int32,
        ),
        # Previous-action history deliberately remains exactly reset-neutral.
        previous_timestep_move_actions=reset_state.previous_timestep_move_actions,
        previous_timestep_select_target_actions=(
            reset_state.previous_timestep_select_target_actions
        ),
        previous_timestep_use_ultimate_actions=(
            reset_state.previous_timestep_use_ultimate_actions
        ),
        has_previous_timestep_joint_action=(
            reset_state.has_previous_timestep_joint_action
        ),
    )


def _array_payload(value: Array) -> dict[str, object]:
    host = np.asarray(value)
    return {
        "dtype": str(host.dtype),
        "shape": tuple(int(dimension) for dimension in host.shape),
        "values": host.tolist(),
    }


def _state_digest(state: EnvState) -> str:
    return canonical_digest_sha256(
        {
            "schema": "dev-resolved-initial-state@1",
            **{
                field_name: _array_payload(cast(Array, getattr(state, field_name)))
                for field_name in state._fields
            },
        }
    )


def scenario_semantic_payload(content: DevScenarioContentV1) -> dict[str, object]:
    """Project physical semantics, excluding display prose and provenance IDs."""
    content = normalize_scenario_content(content)
    return {
        "schema": "dev-scenario-semantics@1",
        "embedded_map": map_semantic_payload(content.embedded_map),
        "task": content.task.model_dump(mode="json"),
        "episode": content.episode.model_dump(mode="json"),
        "global_state": content.global_state.model_dump(mode="json"),
        "team_sizes": (content.team_a_size, content.team_b_size),
        "roster": tuple(
            {
                "team": slot.team,
                "team_local_slot": slot.team_local_slot,
                "global_slot": slot.global_slot,
                "class_name": slot.class_name,
            }
            for slot in content.roster
        ),
        "agent_states": tuple(
            {
                key: value
                for key, value in state.model_dump(mode="json").items()
                if key != "object_id"
            }
            for state in content.agent_states
        ),
    }


def scenario_semantic_digest(content: DevScenarioContentV1) -> str:
    return canonical_digest_sha256(scenario_semantic_payload(content))


_SLOW_AUTHORING_FIELDS = (
    "warrior_charge_slow_duration",
    "hunter_basic_slow_duration",
    "rogue_poison_slow_duration",
)
_STUN_AUTHORING_FIELDS = (
    "warrior_charge_stun_duration",
    "hunter_trap_stun_duration",
    "rogue_poison_stun_duration",
)


def _first_invalid_index(mask: np.ndarray) -> tuple[int, int | None] | None:
    indices = np.argwhere(mask)
    if indices.size == 0:
        return None
    first = indices[0]
    return int(first[0]), int(first[1]) if first.size > 1 else None


def _core_agent_target(
    message: str,
    config: EnvConfig,
    state: EnvState,
) -> tuple[int, str] | None:
    indexed_position = re.search(r"agent_positions\[(\d+)\]", message)
    if indexed_position is not None:
        return int(indexed_position.group(1)), "position"
    overlap = re.search(r"slots (\d+) and (\d+)", message)
    if overlap is not None:
        return int(overlap.group(2)), "position"

    active = np.asarray(config.agent_profile.active_mask, dtype=np.bool_)
    alive = np.asarray(state.alive_mask, dtype=np.bool_)
    if "alive_mask" in message:
        found = _first_invalid_index(alive & ~active)
        return None if found is None else (found[0], "alive")

    if "current_health" in message:
        health = np.asarray(state.current_health)
        maximum = np.asarray(config.agent_profile.max_health)
        if "strictly positive" in message:
            mask = active & alive & (health <= 0.0)
        elif "exceed max_health" in message:
            mask = active & alive & (health > maximum)
        elif "active dead" in message:
            mask = active & ~alive & (health != 0.0)
        else:
            mask = ~active & (health != 0.0)
        found = _first_invalid_index(mask)
        return None if found is None else (found[0], "current_health")

    if "ultimate_cooldowns" in message:
        values = np.asarray(state.ultimate_cooldowns)
        maximum = np.asarray(
            combat.get_ultimate_cooldown_by_class_ids(config.agent_profile.class_ids)
        )
        mask = (
            (~active & (values != 0))
            if "inactive" in message
            else ((values < 0) | (values > maximum))
        )
        found = _first_invalid_index(mask)
        return None if found is None else (found[0], "ultimate_cooldown_remaining")

    duration_families: tuple[
        tuple[str, np.ndarray, tuple[str, ...], np.ndarray | int], ...
    ] = (
        (
            "slow_durations",
            np.asarray(state.slow_durations),
            _SLOW_AUTHORING_FIELDS,
            np.asarray(
                (
                    combat.WARRIOR_CHARGE_SLOW_DURATION_TICKS,
                    combat.HUNTER_BASIC_SLOW_DURATION_TICKS,
                    combat.ROGUE_POISON_SLOW_DURATION_TICKS,
                )
            )[None, :],
        ),
        (
            "stun_durations",
            np.asarray(state.stun_durations),
            _STUN_AUTHORING_FIELDS,
            np.asarray(
                (
                    combat.WARRIOR_CHARGE_STUN_DURATION_TICKS,
                    combat.HUNTER_TRAP_STUN_DURATION_TICKS,
                    combat.ROGUE_POISON_STUN_DURATION_TICKS,
                )
            )[None, :],
        ),
        (
            "rogue_poison_anti_heal_durations",
            np.asarray(state.rogue_poison_anti_heal_durations)[:, None],
            ("rogue_poison_anti_heal_duration",),
            combat.ROGUE_POISON_ANTI_HEAL_DURATION_TICKS,
        ),
        (
            "mage_burst_damage_amplification_durations",
            np.asarray(state.mage_burst_damage_amplification_durations)[:, None],
            ("mage_burst_duration",),
            combat.MAGE_BURST_DAMAGE_DURATION_TICKS,
        ),
        (
            "priest_blessing_of_freedom_slow_floor_durations",
            np.asarray(state.priest_blessing_of_freedom_slow_floor_durations)[:, None],
            ("priest_blessing_of_freedom_duration",),
            combat.PRIEST_HEAL_SPEED_FLOOR_DURATION_TICKS,
        ),
    )
    for core_name, values, authoring_fields, maximum in duration_families:
        if core_name not in message:
            continue
        if "active dead" in message:
            mask = (active & ~alive)[:, None] & (values != 0)
        elif "inactive" in message:
            mask = (~active)[:, None] & (values != 0)
        elif "nonnegative" in message:
            mask = values < 0
        elif "Mage slots" in message:
            mask = (np.asarray(config.agent_profile.class_ids) != MAGE_CLASS_ID)[
                :, None
            ] & (values != 0)
        else:
            mask = values > maximum
        found = _first_invalid_index(mask)
        if found is not None:
            column = 0 if found[1] is None else found[1]
            return found[0], authoring_fields[column]

    if "spawn_shield_durations" in message:
        values = np.asarray(state.spawn_shield_durations)
        if "active dead" in message:
            mask = active & ~alive & (values != 0)
        elif "inactive" in message:
            mask = ~active & (values != 0)
        elif "nonnegative" in message:
            mask = values < 0
        elif "stun_durations" in message:
            mask = (values > 0) & np.any(np.asarray(state.stun_durations) > 0, axis=-1)
        else:
            mask = values > config.spawn_shield_duration_steps
        found = _first_invalid_index(mask)
        return None if found is None else (found[0], "spawn_shield_duration_remaining")

    if "steps_until_out_of_combat" in message:
        values = np.asarray(state.steps_until_out_of_combat)
        if "active dead" in message:
            mask = active & ~alive & (values != 0)
        elif "inactive" in message:
            mask = ~active & (values != 0)
        elif "nonnegative" in message:
            mask = values < 0
        else:
            mask = values > np.asarray(config.agent_profile.out_of_combat_delay_steps)
        found = _first_invalid_index(mask)
        return None if found is None else (found[0], "steps_until_out_of_combat")
    return None


def _core_problem(
    error: Exception,
    *,
    phase: Literal["config", "state"],
    content: DevScenarioContentV1,
    config: EnvConfig | None = None,
    state: EnvState | None = None,
) -> DevAuthoringProblemV1:
    message = str(error)
    field_path = "scenario"
    object_id: str | None = None
    if config is not None and state is not None:
        target = _core_agent_target(message, config, state)
        if target is not None:
            global_slot, field_name = target
            object_id = content.agent_states[global_slot].object_id
            field_path = f"agent_states.{global_slot}.{field_name}"
    if field_path == "scenario":
        if "step_count" in message:
            field_path = "global_state.step_count"
        elif "team_deathmatch_scores" in message:
            team = "a" if content.global_state.team_a_score < 0 else "b"
            field_path = f"global_state.team_{team}_score"
        elif "team_respawn_wave_countdowns" in message:
            team = (
                "a"
                if content.global_state.team_a_respawn_countdown < 0
                or content.global_state.team_a_respawn_countdown
                >= content.episode.team_a_respawn_wave_period_steps
                else "b"
            )
            field_path = f"global_state.team_{team}_respawn_countdown"
        elif "team_spawn_pad_positions" in message:
            field_path = "embedded_map.spawn_pads"
    return _problem(
        "error",
        f"scenario-core-{phase}-invalid",
        message,
        field_path,
        object_id=object_id,
    )


def compile_dev_scenario(
    source: DevScenarioDraftV1 | DevScenarioContentV1,
) -> CompiledDevScenarioV1:
    """Compile, revalidate, and expose one immutable authored scenario snapshot."""
    raw_content = (
        source.content if not isinstance(source, DevScenarioContentV1) else source
    )
    try:
        content = normalize_scenario_content(raw_content)
    except ValueError as error:
        raise DevAuthoringValidationError(
            (
                _normalization_problem(
                    "scenario-float32-normalization-failed",
                    error,
                    fallback_field_path="scenario",
                ),
            )
        ) from error
    custom_problems = list(_scenario_custom_problems(content))
    if any(problem.severity == "error" for problem in custom_problems):
        raise DevAuthoringValidationError(tuple(custom_problems))

    compiled_map = compile_dev_map(content.embedded_map)
    config = _build_config(content, compiled_map)
    try:
        validate_product_env_config(config)
    except (TypeError, ValueError) as error:
        custom_problems.append(
            _core_problem(error, phase="config", content=content, config=config)
        )
        raise DevAuthoringValidationError(tuple(custom_problems)) from error

    reset_state, _, _, _ = reset(config, jax.random.key(0))
    authored_state = _overlay_authored_state(reset_state, content)
    try:
        validate_scenario_initial_state(config, authored_state)
        initialized_state, observation, action_mask, info = initialize_scenario_state(
            authored_state,
            config,
        )
    except (TypeError, ValueError) as error:
        custom_problems.append(
            _core_problem(
                error,
                phase="state",
                content=content,
                config=config,
                state=authored_state,
            )
        )
        raise DevAuthoringValidationError(tuple(custom_problems)) from error

    resolved_config = build_resolved_env_config_v1(config)
    return CompiledDevScenarioV1(
        content=content.model_copy(deep=True),
        config=config,
        initial_state=initialized_state,
        observation=observation,
        action_mask=action_mask,
        info=info,
        map_semantic_digest=compiled_map.semantic_digest,
        semantic_digest=scenario_semantic_digest(content),
        resolved_configuration_digest=resolved_config.canonical_digest_sha256,
        resolved_initial_state_digest=_state_digest(initialized_state),
        problems=tuple(custom_problems),
    )


def validate_dev_scenario(
    source: DevScenarioDraftV1 | DevScenarioContentV1,
) -> tuple[DevAuthoringProblemV1, ...]:
    """Return execution problems without leaking partially compiled state."""
    try:
        return compile_dev_scenario(source).problems
    except DevAuthoringValidationError as error:
        return error.problems


def apply_alive_edit(
    content: DevScenarioContentV1,
    *,
    global_slot: int,
    alive: bool,
) -> DevScenarioContentV1:
    """Apply the explicit Alive/Dead authoring transaction for one active row."""
    if not 0 <= global_slot < MAX_AGENT_SLOTS:
        raise ValueError(f"global_slot must be in [0, {MAX_AGENT_SLOTS})")
    if not _active(global_slot, content):
        raise ValueError("inactive roster rows cannot receive Alive/Dead edits")
    current = content.agent_states[global_slot]
    if alive or not current.alive:
        replacement = current.model_copy(update={"alive": alive})
    else:
        replacement = DevAgentStateV1(
            object_id=current.object_id,
            position=current.position,
            alive=False,
            current_health=0.0,
            ultimate_cooldown_remaining=current.ultimate_cooldown_remaining,
        )
    rows = list(content.agent_states)
    rows[global_slot] = replacement
    return content.model_copy(update={"agent_states": tuple(rows)})


def canonicalize_inactive_rows(content: DevScenarioContentV1) -> DevScenarioContentV1:
    """Canonicalize only rows made inactive by an explicit team-size edit."""
    roster: list[DevRosterSlotV1] = []
    states: list[DevAgentStateV1] = []
    for global_slot, (roster_slot, agent_state) in enumerate(
        zip(content.roster, content.agent_states, strict=True)
    ):
        if _active(global_slot, content):
            roster.append(roster_slot)
            states.append(agent_state)
            continue
        roster.append(
            roster_slot.model_copy(
                update={
                    "class_name": "not_applicable",
                }
            )
        )
        states.append(
            DevAgentStateV1(
                object_id=agent_state.object_id,
                position=DevPointV1(x=0.0, y=0.0),
                alive=False,
                current_health=0.0,
            )
        )
    return content.model_copy(
        update={"roster": tuple(roster), "agent_states": tuple(states)}
    )


__all__ = [
    "CompiledDevMapV1",
    "CompiledDevScenarioV1",
    "DevAuthoringValidationError",
    "apply_alive_edit",
    "canonicalize_inactive_rows",
    "compile_dev_map",
    "compile_dev_scenario",
    "map_semantic_digest",
    "map_semantic_payload",
    "normalize_map_content",
    "normalize_scenario_content",
    "scenario_semantic_digest",
    "scenario_semantic_payload",
    "validate_dev_scenario",
    "validate_map_content",
]
