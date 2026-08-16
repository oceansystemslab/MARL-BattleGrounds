"""Pure canonical-evaluation to researcher-presentation projection.

The adapter consumes only strict host records.  It does not import simulator
state, JAX/NumPy arrays, mechanic helpers, policy code, replay persistence, or a
renderer.  Durable state comes from the selected frame and context; an optional
coherent transition view contributes only direct incoming-event identity and
source evidence.
"""

from dataclasses import dataclass
from math import isfinite
from typing import Literal, TypedDict, cast

from marl_battlegrounds.evaluation.metrics import EvaluationTransitionViewV1
from marl_battlegrounds.evaluation.models import (
    AbilityActivatedEventV1 as EvaluationAbilityActivatedEventV1,
)
from marl_battlegrounds.evaluation.models import (
    ActionRejectedEventV1 as EvaluationActionRejectedEventV1,
)
from marl_battlegrounds.evaluation.models import (
    AgentDiedEventV1 as EvaluationAgentDiedEventV1,
)
from marl_battlegrounds.evaluation.models import (
    AgentRespawnedEventV1 as EvaluationAgentRespawnedEventV1,
)
from marl_battlegrounds.evaluation.models import (
    ChargePhaseDisplacementEventV1 as EvaluationChargePhaseDisplacementEventV1,
)
from marl_battlegrounds.evaluation.models import (
    CombatCountdownResetEventV1 as EvaluationCombatCountdownResetEventV1,
)
from marl_battlegrounds.evaluation.models import (
    CooldownReadyEventV1 as EvaluationCooldownReadyEventV1,
)
from marl_battlegrounds.evaluation.models import (
    CooldownStartedEventV1 as EvaluationCooldownStartedEventV1,
)
from marl_battlegrounds.evaluation.models import (
    EvaluationEpisodeContextV1,
    EvaluationEventV1,
    EvaluationFrameV1,
    EvaluationTransitionV1,
)
from marl_battlegrounds.evaluation.models import (
    HealthRegeneratedEventV1 as EvaluationHealthRegeneratedEventV1,
)
from marl_battlegrounds.evaluation.models import (
    LethalDamageContributionEventV1 as EvaluationLethalDamageContributionEventV1,
)
from marl_battlegrounds.evaluation.models import (
    OrdinaryMovementPhaseDisplacementEventV1 as Cp2OrdinaryMovementEventV1,
)
from marl_battlegrounds.evaluation.models import (
    RecipientHealthResolutionEventV1 as Cp2RecipientHealthEventV1,
)
from marl_battlegrounds.evaluation.models import (
    RespawnWaveOccurredEventV1 as EvaluationRespawnWaveOccurredEventV1,
)
from marl_battlegrounds.evaluation.models import (
    SourceDamageOutputEventV1 as EvaluationSourceDamageOutputEventV1,
)
from marl_battlegrounds.evaluation.models import (
    SourceHealingOutputEventV1 as EvaluationSourceHealingOutputEventV1,
)
from marl_battlegrounds.evaluation.models import (
    SpawnShieldExpiredEventV1 as EvaluationSpawnShieldExpiredEventV1,
)
from marl_battlegrounds.evaluation.models import (
    StatusAgedToZeroEventV1 as EvaluationStatusAgedToZeroEventV1,
)
from marl_battlegrounds.evaluation.models import (
    StatusAppliedEventV1 as EvaluationStatusAppliedEventV1,
)
from marl_battlegrounds.evaluation.models import (
    StatusBrokenByDamageEventV1 as EvaluationStatusBrokenByDamageEventV1,
)
from marl_battlegrounds.evaluation.models import (
    StatusClearedByNewDeathEventV1 as EvaluationStatusClearedByNewDeathEventV1,
)
from marl_battlegrounds.evaluation.models import (
    StatusLifecycleEventBaseV1 as EvaluationStatusLifecycleEventBaseV1,
)
from marl_battlegrounds.evaluation.models import (
    StatusRefreshedOrExtendedEventV1 as Cp2StatusRefreshedEventV1,
)
from marl_battlegrounds.evaluation.models import (
    TeamDeathmatchCompletedEventV1 as EvaluationTeamDeathmatchCompletedEventV1,
)
from marl_battlegrounds.evaluation.models import (
    TeamDeathmatchScoreChangedEventV1 as EvaluationTeamDeathmatchScoreChangedEventV1,
)
from marl_battlegrounds.evaluation.pov import (
    ActorPovActionMaskV1,
    ActorPovAxisMappingV1,
    ActorPovPreviousTimestepActionsV1,
    ActorPovSpawnLifecycleV1,
)
from marl_battlegrounds.evaluation.validation import (
    validate_initial_evaluation_frame_v1,
)
from marl_battlegrounds.evaluation.wire_shapes import (
    CONTEXT_FEATURES_V1,
    MAX_AGENT_SLOTS_V1,
    MAX_AGENTS_PER_TEAM_V1,
    MAX_OBJECTIVE_SLOTS_V1,
    MAX_OBSTACLE_SLOTS_V1,
    OBJECTIVE_FEATURES_V1,
    OBSTACLE_FEATURES_V1,
    SELF_FEATURES_V1,
    UNIT_FEATURES_V1,
)
from marl_battlegrounds.rendering.evaluation_wire_features import (
    AGENT_FEATURE_ACTIVE_V1,
    AGENT_FEATURE_ALIVE_V1,
    AGENT_FEATURE_CLASS_ID_V1,
    AGENT_FEATURE_CURRENT_HEALTH_V1,
    AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER_V1,
    AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER_V1,
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
from marl_battlegrounds.rendering.pov_scene import (
    ActorPovRespawnWaveSceneV1,
    ActorPovSelfSceneV1,
    ActorPovSpawnPadSceneV1,
    ActorPovVisibleBodySceneV1,
)
from marl_battlegrounds.rendering.scene import (
    EVENT_V2_SCHEMA_VERSION,
    RESEARCHER_ANALYZER_PROJECTION_SCHEMA_VERSION,
    SCENE_V2_SCHEMA_VERSION,
    STATUS_SOURCE_EVIDENCE_SCHEMA_VERSION,
    AbilityActivatedEventV2,
    ActionRejectedEventV2,
    AgentDiedEventV2,
    AgentRespawnedEventV2,
    AgentSceneV2,
    AuraFieldSceneV2,
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
    ObserverVisibilitySceneV1,
    ObstacleSceneV1,
    OrdinaryMovementPhaseDisplacementEventV2,
    RangeKind,
    RangeSceneV1,
    RecipientHealthResolutionEventV2,
    ResearcherAnalyzerProjectionV2,
    RespawnWaveOccurredEventV2,
    RespawnWaveSceneV2,
    SelectedLegalitySceneV1,
    SelectionSceneV1,
    SourceDamageOutputEventV2,
    SourceHealingOutputEventV2,
    SpawnPadSceneV2,
    SpawnShieldExpiredEventV2,
    StatusAgedToZeroEventV2,
    StatusAppliedEventV2,
    StatusBrokenByDamageEventV2,
    StatusClearedByNewDeathEventV2,
    StatusRefreshedOrExtendedEventV2,
    StatusSceneV2,
    StatusSourceChannelEvidenceV2,
    StatusSourceEvidenceIndexV2,
    StatusSourceEvidenceSceneV2,
    StatusSourceEvidenceStateV2,
    TeamDeathmatchCompletedEventV2,
    TeamDeathmatchScoreChangedEventV2,
    VisualAgentAnchorV2,
    VisualAgentPhaseTrajectoryV2,
    VisualEventBatchV2,
    VisualEventV2,
    VisualTeamAnchorV2,
)
from marl_battlegrounds.rendering.vocabulary import (
    status_sort_key,
    status_token_id_from_catalog_status_id,
)

type PresentationLaneV1 = Literal[0, 1]

_PILLAR_OBSTACLE_TYPE_ID_V1 = 1
_WALL_OBSTACLE_TYPE_ID_V1 = 2
SHARED_OBS_SOURCE_MATERIAL_PROJECTION_SCHEMA_VERSION = 1

_SHARED_OBS_SOURCE_MATERIAL_DISCLOSURE = (
    "SOURCE MATERIAL ONLY · NOT MATERIALIZED SHAREDOBS ACTOR INPUT"
)


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluationScenePresentationStateV1:
    """Presentation-only selection for one researcher scene projection."""

    controlled_global_slot: int | None = None
    selected_global_slot: int | None = None
    armed_lane: PresentationLaneV1 | None = None
    show_ranges: bool = True

    def __post_init__(self) -> None:
        for name in ("controlled_global_slot", "selected_global_slot"):
            value = cast(int | None, getattr(self, name))
            if value is not None and (
                type(value) is not int or not 0 <= value < MAX_AGENT_SLOTS_V1
            ):
                raise ValueError(
                    f"{name} must be a Python int in [0, {MAX_AGENT_SLOTS_V1}) or None."
                )
        if (
            self.selected_global_slot is not None
            and self.controlled_global_slot is None
        ):
            raise ValueError("selected_global_slot requires a controlled actor.")
        if self.armed_lane is not None and (
            type(self.armed_lane) is not int or self.armed_lane not in (0, 1)
        ):
            raise ValueError("armed_lane must be the Python int zero or one, or None.")
        if self.armed_lane is not None and self.selected_global_slot is None:
            raise ValueError("armed_lane requires a selected actor.")
        if type(self.show_ranges) is not bool:
            raise ValueError("show_ranges must be a Python bool.")


@dataclass(frozen=True, slots=True, kw_only=True)
class SharedObsSensorSourceAvailabilityV1:
    """One recorded source-availability cell for a selected recipient."""

    sensor_source_global_slot: int
    sensor_source_public_agent_id: str
    sensor_source_team_local_slot: int
    sensor_source_configured_team_id: int
    sensor_source_configured_active: bool
    relation_to_recipient: Literal["self", "ally", "opponent", "inactive"]
    base_sensor_relation_axis: Literal["ally", "enemy"]
    base_sensor_observation_row: int
    recorded_available: bool

    def __post_init__(self) -> None:
        if type(self.sensor_source_global_slot) is not int or not (
            0 <= self.sensor_source_global_slot < MAX_AGENT_SLOTS_V1
        ):
            raise ValueError("sensor source global slot is outside the V1 axis.")
        if (
            type(self.sensor_source_public_agent_id) is not str
            or not self.sensor_source_public_agent_id.strip()
        ):
            raise ValueError("sensor source public agent ID must be non-empty.")
        if type(self.sensor_source_team_local_slot) is not int or not (
            0 <= self.sensor_source_team_local_slot < 5
        ):
            raise ValueError("sensor source team-local slot is outside the V1 axis.")
        if type(self.sensor_source_configured_team_id) is not int or not (
            0 <= self.sensor_source_configured_team_id <= 2
        ):
            raise ValueError("sensor source configured team ID is outside V1.")
        if type(self.sensor_source_configured_active) is not bool:
            raise ValueError("sensor source configured-active must be a Python bool.")
        if self.sensor_source_configured_active != (
            self.sensor_source_configured_team_id in (1, 2)
        ):
            raise ValueError("sensor source active/team metadata is contradictory.")
        if self.relation_to_recipient not in (
            "self",
            "ally",
            "opponent",
            "inactive",
        ):
            raise ValueError("sensor source relation is outside the V1 vocabulary.")
        if self.base_sensor_relation_axis not in ("ally", "enemy"):
            raise ValueError("base-sensor relation axis must be ally or enemy.")
        if type(self.base_sensor_observation_row) is not int or not (
            0 <= self.base_sensor_observation_row < MAX_AGENTS_PER_TEAM_V1
        ):
            raise ValueError("base-sensor observation row is outside the V1 axis.")
        if type(self.recorded_available) is not bool:
            raise ValueError("recorded availability must be a Python bool.")
        if self.recorded_available and (
            not self.sensor_source_configured_active
            or self.relation_to_recipient != "ally"
        ):
            raise ValueError(
                "available SharedObs sources must be configured-active allies."
            )


def _require_tuple_shape(
    value: object,
    shape: tuple[int, ...],
    *,
    name: str,
    leaf_type: type[float] | type[bool],
) -> None:
    if not shape:
        if type(value) is not leaf_type:
            raise ValueError(f"{name} must contain exact {leaf_type.__name__} values.")
        if leaf_type is float and not isfinite(cast(float, value)):
            raise ValueError(f"{name} must contain finite floats.")
        return
    if type(value) is not tuple or len(cast(tuple[object, ...], value)) != shape[0]:
        raise ValueError(f"{name} must have exact tuple shape {shape}.")
    for item in cast(tuple[object, ...], value):
        _require_tuple_shape(
            item,
            shape[1:],
            name=name,
            leaf_type=leaf_type,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SharedObsBaseSensorFrameV1:
    """One selected recipient's source-only base-sensor frame."""

    schema_version: int
    observation_materialization: Literal["source_material_only"]
    episode_id: str
    public_agent_id: str
    frame_index: int
    source_material_frame_id: str
    source_frame_id: str
    simulator_step_count: int
    self_features: tuple[float, ...]
    ally_unit_features: tuple[tuple[float, ...], ...]
    enemy_unit_features: tuple[tuple[float, ...], ...]
    map_obstacle_features: tuple[tuple[float, ...], ...]
    objective_features: tuple[tuple[float, ...], ...]
    context_features: tuple[float, ...]
    ally_visibility_mask: tuple[bool, ...]
    enemy_visibility_mask: tuple[bool, ...]
    previous_timestep_actions: ActorPovPreviousTimestepActionsV1
    spawn_lifecycle: ActorPovSpawnLifecycleV1
    action_mask: ActorPovActionMaskV1

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or (
            self.schema_version != SHARED_OBS_SOURCE_MATERIAL_PROJECTION_SCHEMA_VERSION
        ):
            raise ValueError("unknown SharedObs base-sensor frame version.")
        if self.observation_materialization != "source_material_only":
            raise ValueError("SharedObs base-sensor frames are source material only.")
        for name in ("episode_id", "public_agent_id"):
            value = cast(str, getattr(self, name))
            if type(value) is not str or not value.strip():
                raise ValueError(f"{name} must be a non-empty Python string.")
        if type(self.frame_index) is not int or self.frame_index < 0:
            raise ValueError("frame_index must be a nonnegative Python int.")
        expected_id = (
            f"{self.episode_id}:shared-obs-source-material:"
            f"{self.public_agent_id}:frame:{self.frame_index}"
        )
        if self.source_material_frame_id != expected_id:
            raise ValueError("SharedObs source-material frame ID is not canonical.")
        if self.source_frame_id != f"{self.episode_id}:frame:{self.frame_index}":
            raise ValueError("SharedObs source frame ID is not canonical.")
        if type(self.simulator_step_count) is not int or self.simulator_step_count < 0:
            raise ValueError("simulator_step_count must be a nonnegative Python int.")
        for name, shape in (
            ("self_features", (SELF_FEATURES_V1,)),
            (
                "ally_unit_features",
                (MAX_AGENTS_PER_TEAM_V1, UNIT_FEATURES_V1),
            ),
            (
                "enemy_unit_features",
                (MAX_AGENTS_PER_TEAM_V1, UNIT_FEATURES_V1),
            ),
            (
                "map_obstacle_features",
                (MAX_OBSTACLE_SLOTS_V1, OBSTACLE_FEATURES_V1),
            ),
            (
                "objective_features",
                (MAX_OBJECTIVE_SLOTS_V1, OBJECTIVE_FEATURES_V1),
            ),
            ("context_features", (CONTEXT_FEATURES_V1,)),
        ):
            _require_tuple_shape(
                getattr(self, name),
                shape,
                name=name,
                leaf_type=float,
            )
        for name in ("ally_visibility_mask", "enemy_visibility_mask"):
            _require_tuple_shape(
                getattr(self, name),
                (MAX_AGENTS_PER_TEAM_V1,),
                name=name,
                leaf_type=bool,
            )
        if (
            type(self.previous_timestep_actions)
            is not ActorPovPreviousTimestepActionsV1
        ):
            raise ValueError("previous actions must use the exact recipient-safe root.")
        if type(self.spawn_lifecycle) is not ActorPovSpawnLifecycleV1:
            raise ValueError("spawn lifecycle must use the exact recipient-safe root.")
        if type(self.action_mask) is not ActorPovActionMaskV1:
            raise ValueError("action mask must use the exact recipient-safe root.")
        for value, expected_type, name in (
            (
                self.previous_timestep_actions,
                ActorPovPreviousTimestepActionsV1,
                "previous_timestep_actions",
            ),
            (
                self.spawn_lifecycle,
                ActorPovSpawnLifecycleV1,
                "spawn_lifecycle",
            ),
            (self.action_mask, ActorPovActionMaskV1, "action_mask"),
        ):
            reconstructed = expected_type.model_validate(
                value.model_dump(mode="python")
            )
            if reconstructed != value:
                raise ValueError(f"{name} changes under structural revalidation.")


@dataclass(frozen=True, slots=True, kw_only=True)
class SharedObsBaseSensorSceneV1:
    """Selected actor's base-sensor scene, explicitly not composed SharedObs."""

    schema_version: int
    audience_badge: str
    observation_materialization: Literal["source_material_only"]
    episode_id: str
    frame_index: int
    source_frame_id: str
    simulator_step_count: int
    map: MapSceneV1
    self_actor: ActorPovSelfSceneV1
    visible_bodies: tuple[ActorPovVisibleBodySceneV1, ...]
    spawn_pads: tuple[ActorPovSpawnPadSceneV1, ...]
    respawn_waves: tuple[ActorPovRespawnWaveSceneV1, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or (
            self.schema_version != SHARED_OBS_SOURCE_MATERIAL_PROJECTION_SCHEMA_VERSION
        ):
            raise ValueError("unknown SharedObs source-material scene version.")
        if self.audience_badge != _SHARED_OBS_SOURCE_MATERIAL_DISCLOSURE:
            raise ValueError("SharedObs source material requires its exact badge.")
        if self.observation_materialization != "source_material_only":
            raise ValueError("SharedObs base-sensor scenes are source material only.")
        if type(self.episode_id) is not str or not self.episode_id.strip():
            raise ValueError("episode_id must be a non-empty Python string.")
        if type(self.frame_index) is not int or self.frame_index < 0:
            raise ValueError("frame_index must be a nonnegative Python int.")
        if self.source_frame_id != f"{self.episode_id}:frame:{self.frame_index}":
            raise ValueError("SharedObs source frame ID is not canonical.")
        if type(self.simulator_step_count) is not int or self.simulator_step_count < 0:
            raise ValueError("simulator_step_count must be a nonnegative Python int.")
        if type(self.map) is not MapSceneV1:
            raise ValueError("map must be the exact scalar MapSceneV1.")
        if type(self.self_actor) is not ActorPovSelfSceneV1:
            raise ValueError("self_actor must be the exact recipient-safe scene row.")
        if type(self.visible_bodies) is not tuple or any(
            type(row) is not ActorPovVisibleBodySceneV1 for row in self.visible_bodies
        ):
            raise ValueError("visible bodies must be recipient-safe scene rows.")
        visible_keys = tuple(
            (row.relation, row.observation_row) for row in self.visible_bodies
        )
        if visible_keys != tuple(sorted(visible_keys)) or len(visible_keys) != len(
            set(visible_keys)
        ):
            raise ValueError("visible body rows must have canonical unique keys.")
        if type(self.spawn_pads) is not tuple or any(
            type(row) is not ActorPovSpawnPadSceneV1 for row in self.spawn_pads
        ):
            raise ValueError("spawn pads must be recipient-safe scene rows.")
        if tuple(
            (row.actor_relative_team_index, row.team_local_slot)
            for row in self.spawn_pads
        ) != tuple((team, slot) for team in range(2) for slot in range(5)):
            raise ValueError("SharedObs base-sensor pads must retain both team axes.")
        if type(self.respawn_waves) is not tuple or tuple(
            row.actor_relative_team_index for row in self.respawn_waves
        ) != (0, 1):
            raise ValueError("SharedObs base-sensor waves must retain both team axes.")


@dataclass(frozen=True, slots=True, kw_only=True)
class SharedObsSourceMaterialProjectionV1:
    """Non-exportable base-sensor and availability evidence for SharedObs."""

    schema_version: int
    disclosure_label: str
    observation_materialization: Literal["source_material_only"]
    exact_actor_input_export_available: Literal[False]
    axis_mapping: ActorPovAxisMappingV1
    ally_observation_row_global_slot_by_id: tuple[int, ...]
    enemy_observation_row_global_slot_by_id: tuple[int, ...]
    base_sensor_frame: SharedObsBaseSensorFrameV1
    base_sensor_scene: SharedObsBaseSensorSceneV1
    incoming_transition_id: str | None
    sensor_source_availability: tuple[SharedObsSensorSourceAvailabilityV1, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or (
            self.schema_version != SHARED_OBS_SOURCE_MATERIAL_PROJECTION_SCHEMA_VERSION
        ):
            raise ValueError("unknown SharedObs source-material projection version.")
        if self.disclosure_label != _SHARED_OBS_SOURCE_MATERIAL_DISCLOSURE:
            raise ValueError("SharedObs projection requires its exact disclosure.")
        if self.observation_materialization != "source_material_only":
            raise ValueError("SharedObs projection must remain source material only.")
        if self.exact_actor_input_export_available is not False:
            raise ValueError("SharedObs exact actor-input export is unavailable.")
        if type(self.axis_mapping) is not ActorPovAxisMappingV1:
            raise ValueError("axis_mapping must be the exact actor POV mapping root.")
        reconstructed_axis = ActorPovAxisMappingV1.model_validate(
            self.axis_mapping.model_dump(mode="python")
        )
        if reconstructed_axis != self.axis_mapping:
            raise ValueError("axis_mapping changes under structural revalidation.")
        for name in (
            "ally_observation_row_global_slot_by_id",
            "enemy_observation_row_global_slot_by_id",
        ):
            values = cast(tuple[int, ...], getattr(self, name))
            if type(values) is not tuple or len(values) != MAX_AGENTS_PER_TEAM_V1:
                raise ValueError(f"{name} must retain the five-row V1 axis.")
            if any(
                type(value) is not int or not 0 <= value < MAX_AGENT_SLOTS_V1
                for value in values
            ):
                raise ValueError(f"{name} contains an invalid global slot.")
        if set(
            (
                *self.ally_observation_row_global_slot_by_id,
                *self.enemy_observation_row_global_slot_by_id,
            )
        ) != set(range(MAX_AGENT_SLOTS_V1)):
            raise ValueError("SharedObs relation axes must partition global slots.")
        if type(self.base_sensor_frame) is not SharedObsBaseSensorFrameV1:
            raise ValueError(
                "base_sensor_frame must use its exact source-material root."
            )
        if type(self.base_sensor_scene) is not SharedObsBaseSensorSceneV1:
            raise ValueError("base_sensor_scene must use its exact source-only root.")
        if (
            self.base_sensor_scene.episode_id != self.base_sensor_frame.episode_id
            or self.base_sensor_scene.frame_index != self.base_sensor_frame.frame_index
            or self.base_sensor_scene.source_frame_id
            != self.base_sensor_frame.source_frame_id
            or self.base_sensor_scene.simulator_step_count
            != self.base_sensor_frame.simulator_step_count
            or self.base_sensor_scene.self_actor.public_agent_id
            != self.base_sensor_frame.public_agent_id
        ):
            raise ValueError("SharedObs base-sensor frame and scene do not join.")
        expected_transition_id = (
            None
            if self.base_sensor_frame.frame_index == 0
            else (
                f"{self.base_sensor_frame.episode_id}:transition:"
                f"{self.base_sensor_frame.frame_index - 1}"
            )
        )
        if self.incoming_transition_id != expected_transition_id:
            raise ValueError("incoming transition must enter the selected frame.")
        if type(self.sensor_source_availability) is not tuple or any(
            type(row) is not SharedObsSensorSourceAvailabilityV1
            for row in self.sensor_source_availability
        ):
            raise ValueError("SharedObs availability must be exact source rows.")
        if tuple(
            row.sensor_source_global_slot for row in self.sensor_source_availability
        ) != tuple(range(MAX_AGENT_SLOTS_V1)):
            raise ValueError("SharedObs availability must retain the full source axis.")
        self_slot = self.base_sensor_scene.self_actor.global_slot
        recipient_team = self.base_sensor_scene.self_actor.team_id
        axis_keys: list[tuple[str, int]] = []
        for row in self.sensor_source_availability:
            if (row.sensor_source_global_slot == self_slot) != (
                row.relation_to_recipient == "self"
            ):
                raise ValueError("SharedObs self-source relation is contradictory.")
            expected_team_local_slot = row.sensor_source_global_slot % 5
            if row.sensor_source_team_local_slot != expected_team_local_slot:
                raise ValueError("SharedObs source team-local topology is invalid.")
            source_block_team = 1 if row.sensor_source_global_slot < 5 else 2
            expected_configured_team = (
                source_block_team if row.sensor_source_configured_active else 0
            )
            if row.sensor_source_configured_team_id != expected_configured_team:
                raise ValueError("SharedObs source team metadata is contradictory.")
            if not row.sensor_source_configured_active:
                expected_relation = "inactive"
            elif row.sensor_source_global_slot == self_slot:
                expected_relation = "self"
            elif source_block_team == recipient_team:
                expected_relation = "ally"
            else:
                expected_relation = "opponent"
            if row.relation_to_recipient != expected_relation:
                raise ValueError("SharedObs recipient-relative relation is invalid.")
            expected_axis = "ally" if source_block_team == recipient_team else "enemy"
            if row.base_sensor_relation_axis != expected_axis:
                raise ValueError("SharedObs source relation axis is invalid.")
            axis_global_slots = (
                self.ally_observation_row_global_slot_by_id
                if expected_axis == "ally"
                else self.enemy_observation_row_global_slot_by_id
            )
            if (
                row.sensor_source_global_slot
                != axis_global_slots[row.base_sensor_observation_row]
            ):
                raise ValueError("SharedObs source global slot does not join its axis.")
            axis_public_ids = (
                self.axis_mapping.ally_observation_row_public_agent_id_by_id
                if expected_axis == "ally"
                else self.axis_mapping.enemy_observation_row_public_agent_id_by_id
            )
            if (
                row.sensor_source_public_agent_id
                != axis_public_ids[row.base_sensor_observation_row]
            ):
                raise ValueError("SharedObs source public ID does not join its axis.")
            axis_keys.append(
                (row.base_sensor_relation_axis, row.base_sensor_observation_row)
            )
            if row.relation_to_recipient == "self" and (
                row.sensor_source_public_agent_id
                != self.base_sensor_frame.public_agent_id
            ):
                raise ValueError("SharedObs self-source public identity is invalid.")
        if len(set(axis_keys)) != MAX_AGENT_SLOTS_V1:
            raise ValueError("SharedObs source rows must partition relation axes.")
        expected_scene = _shared_obs_base_sensor_scene(
            self.base_sensor_frame,
            selected_global_slot=self.base_sensor_scene.self_actor.global_slot,
            selected_team_local_slot=(
                self.base_sensor_scene.self_actor.team_local_slot
            ),
            configured_team_id=self.base_sensor_scene.self_actor.team_id,
            class_id=self.base_sensor_scene.self_actor.class_id,
            axis_mapping=self.axis_mapping,
        )
        if expected_scene != self.base_sensor_scene:
            raise ValueError("SharedObs base-sensor scene must derive from its frame.")


def _validate_projection_inputs(
    context: EvaluationEpisodeContextV1,
    frame: EvaluationFrameV1,
    transition_view: EvaluationTransitionViewV1 | None,
) -> EvaluationTransitionViewV1 | None:
    if type(context) is not EvaluationEpisodeContextV1:
        raise TypeError("context must be the exact EvaluationEpisodeContextV1 root.")
    if type(frame) is not EvaluationFrameV1:
        raise TypeError("frame must be the exact EvaluationFrameV1 root.")
    if frame.episode_id != context.identity.episode_id:
        raise ValueError("selected frame must join the context episode.")
    if transition_view is None:
        if frame.frame_index != 0:
            raise ValueError(
                "non-initial selected frames require their coherent incoming "
                "transition view."
            )
        validate_initial_evaluation_frame_v1(context, frame)
        return None
    if type(transition_view) is not EvaluationTransitionViewV1:
        raise TypeError(
            "transition_view must be the exact EvaluationTransitionViewV1 or None."
        )
    canonical_view = EvaluationTransitionViewV1(
        context=transition_view.context,
        start_frame=transition_view.start_frame,
        transition=transition_view.transition,
        successor_frame=transition_view.successor_frame,
    )
    if canonical_view.context != context:
        raise ValueError("incoming transition view must use the selected context.")
    if canonical_view.successor_frame != frame:
        raise ValueError(
            "incoming transition view successor must equal the selected frame."
        )
    return canonical_view


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


def _base_sensor_axis_mapping(
    context: EvaluationEpisodeContextV1,
    *,
    selected_global_slot: int,
) -> ActorPovAxisMappingV1:
    catalog = context.static_mechanics_catalog

    def public_id(global_slot: int) -> str:
        return context.roster[global_slot].public_agent_id

    ally_slots = catalog.global_slot_by_actor_and_ally_observation_row[
        selected_global_slot
    ]
    enemy_slots = catalog.global_slot_by_actor_and_enemy_observation_row[
        selected_global_slot
    ]
    target_slots = catalog.global_recipient_slot_by_actor_and_target_action[
        selected_global_slot
    ]
    return ActorPovAxisMappingV1(
        actor_projection_identifier=context.actor_projection.identifier,
        actor_projection_version=context.actor_projection.version,
        target_action_recipient_public_agent_id_by_id=tuple(
            None if slot is None else public_id(slot) for slot in target_slots
        ),
        ally_observation_row_public_agent_id_by_id=tuple(
            public_id(slot) for slot in ally_slots
        ),
        enemy_observation_row_public_agent_id_by_id=tuple(
            public_id(slot) for slot in enemy_slots
        ),
        movement_action_name_by_id=catalog.movement_action_name_by_id,
        unit_direction_vector_by_movement_action=(
            catalog.unit_direction_vector_by_movement_action
        ),
        target_action_name_by_id=catalog.target_action_name_by_id,
        use_ultimate_action_name_by_id=catalog.use_ultimate_action_name_by_id,
        spawn_lifecycle_team_axis_name_by_id=(
            catalog.spawn_lifecycle_team_axis_name_by_id
        ),
    )


def _base_sensor_frame(
    frame: EvaluationFrameV1,
    *,
    selected_global_slot: int,
    public_agent_id: str,
) -> SharedObsBaseSensorFrameV1:
    observation = frame.base_observation
    previous = observation.previous_timestep_actions
    lifecycle = observation.spawn_lifecycle
    mask = frame.action_mask
    return SharedObsBaseSensorFrameV1(
        schema_version=SHARED_OBS_SOURCE_MATERIAL_PROJECTION_SCHEMA_VERSION,
        observation_materialization="source_material_only",
        episode_id=frame.episode_id,
        public_agent_id=public_agent_id,
        frame_index=frame.frame_index,
        source_material_frame_id=(
            f"{frame.episode_id}:shared-obs-source-material:"
            f"{public_agent_id}:frame:"
            f"{frame.frame_index}"
        ),
        source_frame_id=frame.frame_id,
        simulator_step_count=frame.simulator_step_count,
        self_features=observation.self_features[selected_global_slot],
        ally_unit_features=observation.ally_unit_features[selected_global_slot],
        enemy_unit_features=observation.enemy_unit_features[selected_global_slot],
        map_obstacle_features=observation.map_obstacle_features[selected_global_slot],
        objective_features=observation.objective_features[selected_global_slot],
        context_features=observation.context_features[selected_global_slot],
        ally_visibility_mask=observation.ally_visibility_mask[selected_global_slot],
        enemy_visibility_mask=observation.enemy_visibility_mask[selected_global_slot],
        previous_timestep_actions=ActorPovPreviousTimestepActionsV1(
            ally_move_actions_one_hot=(
                previous.ally_previous_timestep_move_actions_one_hot[
                    selected_global_slot
                ]
            ),
            enemy_move_actions_one_hot=(
                previous.enemy_previous_timestep_move_actions_one_hot[
                    selected_global_slot
                ]
            ),
            ally_select_target_actions_one_hot=(
                previous.ally_previous_timestep_select_target_actions_one_hot[
                    selected_global_slot
                ]
            ),
            enemy_select_target_actions_one_hot=(
                previous.enemy_previous_timestep_select_target_actions_one_hot[
                    selected_global_slot
                ]
            ),
            ally_use_ultimate_actions_one_hot=(
                previous.ally_previous_timestep_use_ultimate_actions_one_hot[
                    selected_global_slot
                ]
            ),
            enemy_use_ultimate_actions_one_hot=(
                previous.enemy_previous_timestep_use_ultimate_actions_one_hot[
                    selected_global_slot
                ]
            ),
        ),
        spawn_lifecycle=ActorPovSpawnLifecycleV1(
            spawn_pad_positions_by_team=(
                lifecycle.spawn_pad_positions_by_agent_by_team[selected_global_slot]
            ),
            spawn_shield_actual_durations_by_team=(
                lifecycle.spawn_shield_actual_durations_by_agent_by_team[
                    selected_global_slot
                ]
            ),
            spawn_shield_configured_duration=(
                lifecycle.spawn_shield_configured_duration_by_agent[
                    selected_global_slot
                ]
            ),
            spawn_shield_speed=(
                lifecycle.spawn_shield_speed_by_agent[selected_global_slot]
            ),
            respawn_wave_period_step_count_by_team=(
                lifecycle.respawn_wave_period_step_count_by_agent_by_team[
                    selected_global_slot
                ]
            ),
            respawn_wave_countdowns_by_team=(
                lifecycle.respawn_wave_countdowns_by_agent_by_team[selected_global_slot]
            ),
            active_mask_by_team=(
                lifecycle.active_mask_by_agent_by_team[selected_global_slot]
            ),
            alive_mask_by_team=(
                lifecycle.alive_mask_by_agent_by_team[selected_global_slot]
            ),
        ),
        action_mask=ActorPovActionMaskV1(
            move=mask.move_mask[selected_global_slot],
            select_target=mask.select_target_mask[selected_global_slot],
            use_ultimate=mask.use_ultimate_mask[selected_global_slot],
            select_target_use_ultimate_joint=(
                mask.select_target_use_ultimate_joint_mask[selected_global_slot]
            ),
        ),
    )


def _base_sensor_map_scene(frame: SharedObsBaseSensorFrameV1) -> MapSceneV1:
    obstacles: list[ObstacleSceneV1] = []
    for obstacle_slot, row in enumerate(frame.map_obstacle_features):
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
                    obstacle_id=f"base-sensor-obstacle-{obstacle_slot}",
                    kind="pillar",
                    center=center,
                    radius=row[OBSTACLE_FEATURE_RADIUS_V1],
                )
            )
        elif obstacle_type == _WALL_OBSTACLE_TYPE_ID_V1:
            obstacles.append(
                ObstacleSceneV1(
                    obstacle_id=f"base-sensor-obstacle-{obstacle_slot}",
                    kind="wall",
                    center=center,
                    width=row[OBSTACLE_FEATURE_WIDTH_V1],
                    height=row[OBSTACLE_FEATURE_HEIGHT_V1],
                    theta=row[OBSTACLE_FEATURE_THETA_V1],
                )
            )
        else:
            raise ValueError(
                "visible obstacle has no V1 base-sensor presentation vocabulary."
            )
    return MapSceneV1(
        width=frame.context_features[CONTEXT_FEATURE_MAP_WIDTH_V1],
        height=frame.context_features[CONTEXT_FEATURE_MAP_HEIGHT_V1],
        obstacles=tuple(obstacles),
    )


def _base_sensor_visible_bodies(
    *,
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
            raise ValueError("visible base-sensor body rows must be recorded active.")
        bodies.append(
            ActorPovVisibleBodySceneV1(
                relation=relation,
                observation_row=observation_row,
                public_agent_id=public_agent_ids[observation_row],
                position=(row[AGENT_FEATURE_X_V1], row[AGENT_FEATURE_Y_V1]),
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
                status_feature_values=row[
                    AGENT_STATUS_FEATURE_START_V1:AGENT_STATUS_FEATURE_STOP_V1
                ],
            )
        )
    return tuple(bodies)


def _shared_obs_base_sensor_scene(
    base_sensor_frame: SharedObsBaseSensorFrameV1,
    *,
    selected_global_slot: int,
    selected_team_local_slot: int,
    configured_team_id: int,
    class_id: int,
    axis_mapping: ActorPovAxisMappingV1,
) -> SharedObsBaseSensorSceneV1:
    self_row = base_sensor_frame.self_features
    if not _decode_wire_bool(
        self_row[AGENT_FEATURE_ACTIVE_V1],
        name="selected actor active",
    ):
        raise ValueError("selected SharedObs actor must remain configured active.")
    if (
        _decode_wire_int(
            self_row[AGENT_FEATURE_TEAM_ID_V1],
            name="selected actor team",
            minimum=1,
            maximum=2,
        )
        != configured_team_id
        or _decode_wire_int(
            self_row[AGENT_FEATURE_CLASS_ID_V1],
            name="selected actor class",
            minimum=1,
            maximum=5,
        )
        != class_id
    ):
        raise ValueError("SharedObs self row team/class must join context identity.")
    lifecycle = base_sensor_frame.spawn_lifecycle
    self_actor = ActorPovSelfSceneV1(
        global_slot=selected_global_slot,
        public_agent_id=base_sensor_frame.public_agent_id,
        team_local_slot=selected_team_local_slot,
        team_id=configured_team_id,
        class_id=class_id,
        position=(self_row[AGENT_FEATURE_X_V1], self_row[AGENT_FEATURE_Y_V1]),
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
        spawn_shield_remaining=(
            lifecycle.spawn_shield_actual_durations_by_team[0][selected_team_local_slot]
        ),
        status_feature_values=self_row[
            AGENT_STATUS_FEATURE_START_V1:AGENT_STATUS_FEATURE_STOP_V1
        ],
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
        for team_index, positions in enumerate(lifecycle.spawn_pad_positions_by_team)
        for team_local_slot, position in enumerate(positions)
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
    return SharedObsBaseSensorSceneV1(
        schema_version=SHARED_OBS_SOURCE_MATERIAL_PROJECTION_SCHEMA_VERSION,
        audience_badge=_SHARED_OBS_SOURCE_MATERIAL_DISCLOSURE,
        observation_materialization="source_material_only",
        episode_id=base_sensor_frame.episode_id,
        frame_index=base_sensor_frame.frame_index,
        source_frame_id=base_sensor_frame.source_frame_id,
        simulator_step_count=base_sensor_frame.simulator_step_count,
        map=_base_sensor_map_scene(base_sensor_frame),
        self_actor=self_actor,
        visible_bodies=(
            *_base_sensor_visible_bodies(
                relation="ally",
                rows=base_sensor_frame.ally_unit_features,
                visibility=base_sensor_frame.ally_visibility_mask,
                public_agent_ids=(
                    axis_mapping.ally_observation_row_public_agent_id_by_id
                ),
            ),
            *_base_sensor_visible_bodies(
                relation="enemy",
                rows=base_sensor_frame.enemy_unit_features,
                visibility=base_sensor_frame.enemy_visibility_mask,
                public_agent_ids=(
                    axis_mapping.enemy_observation_row_public_agent_id_by_id
                ),
            ),
        ),
        spawn_pads=spawn_pads,
        respawn_waves=respawn_waves,
    )


def _map_scene(context: EvaluationEpisodeContextV1) -> MapSceneV1:
    config = context.resolved_env_config
    obstacles: list[ObstacleSceneV1] = []
    for row in config.obstacle_slots:
        if not row.is_active:
            continue
        center = (row.x, row.y)
        if row.obstacle_type_id == _PILLAR_OBSTACLE_TYPE_ID_V1:
            obstacles.append(
                ObstacleSceneV1(
                    obstacle_id=f"obstacle-{row.obstacle_slot}",
                    kind="pillar",
                    center=center,
                    radius=row.radius,
                )
            )
        elif row.obstacle_type_id == _WALL_OBSTACLE_TYPE_ID_V1:
            obstacles.append(
                ObstacleSceneV1(
                    obstacle_id=f"obstacle-{row.obstacle_slot}",
                    kind="wall",
                    center=center,
                    width=row.width,
                    height=row.height,
                    theta=row.theta,
                )
            )
        else:
            raise ValueError(
                "active resolved obstacle has no V1 presentation vocabulary: "
                f"type {row.obstacle_type_id}."
            )
    return MapSceneV1(
        width=config.map_width,
        height=config.map_height,
        obstacles=tuple(obstacles),
    )


def _class_mechanics(
    context: EvaluationEpisodeContextV1,
) -> tuple[ClassMechanicsSceneV2, ...]:
    real_class_ids = range(1, len(context.static_mechanics_catalog.class_mechanics))
    rows: list[ClassMechanicsSceneV2] = []
    for class_id in real_class_ids:
        row = context.static_mechanics_catalog.class_mechanics[class_id]
        status_mechanics = tuple(
            ClassStatusMechanicSceneV2(
                status_channel=status.status_channel_id,
                status_id=status.status_id,
                family=status.family,
                source_action_component=status.source_action_component,
                duration_steps=status.duration_steps,
                magnitude_kind=status.magnitude_kind,
                magnitude=status.magnitude,
                breaks_on_positive_damage=status.breaks_on_positive_damage,
            )
            for status in context.static_mechanics_catalog.status_channels
            if status.source_class_id == class_id
        )
        aura_mechanics = tuple(
            ClassAuraMechanicSceneV2(
                aura_id=aura.aura_id,
                radius=aura.radius,
                per_emitter_multiplier=aura.per_emitter_multiplier,
                stacking_rule=aura.stacking_rule,
                clamp_kind=aura.clamp_kind,
                clamp_value=aura.clamp_value,
            )
            for aura in context.static_mechanics_catalog.aura_mechanics
            if aura.emitter_class_id == class_id
        )
        rows.append(
            ClassMechanicsSceneV2(
                class_id=row.class_id,
                class_name=row.class_name,
                maximum_health=row.maximum_health,
                body_radius=row.body_radius,
                base_movement_speed=row.base_movement_speed,
                observation_radius=row.observation_radius,
                basic_target_mode=row.basic_target_mode,
                basic_interaction_radius=row.basic_interaction_radius,
                basic_raw_damage=row.basic_raw_damage,
                basic_raw_healing=row.basic_raw_healing,
                ultimate_target_mode=row.ultimate_target_mode,
                ultimate_interaction_radius=row.ultimate_interaction_radius,
                ultimate_cooldown_steps=row.ultimate_cooldown_steps,
                ultimate_raw_damage=row.ultimate_raw_damage,
                ultimate_raw_healing=row.ultimate_raw_healing,
                out_of_combat_delay_steps=row.out_of_combat_delay_steps,
                out_of_combat_health_regeneration_fraction_per_step=(
                    row.out_of_combat_health_regeneration_fraction_per_step
                ),
                status_mechanics=status_mechanics,
                aura_mechanics=aura_mechanics,
            )
        )
    return tuple(rows)


def _incoming_status_sources(
    context: EvaluationEpisodeContextV1,
    transition_view: EvaluationTransitionViewV1 | None,
) -> dict[tuple[int, int], tuple[StatusSourceEvidenceSceneV2, ...]]:
    if transition_view is None:
        return {}
    sources: dict[tuple[int, int], list[StatusSourceEvidenceSceneV2]] = {}
    refreshed_keys: set[tuple[int, int]] = set()
    for event in transition_view.transition.events:
        if type(event) is Cp2StatusRefreshedEventV1:
            refreshed = event
            refreshed_keys.add(
                (refreshed.recipient_global_slot, refreshed.status_channel)
            )
        elif type(event) is EvaluationStatusAppliedEventV1:
            applied = event
            roster = context.roster[applied.source_global_slot]
            key = (applied.recipient_global_slot, applied.status_channel)
            sources.setdefault(key, []).append(
                StatusSourceEvidenceSceneV2(
                    source_global_slot=applied.source_global_slot,
                    source_public_agent_id=roster.public_agent_id,
                    event_id=applied.event_id,
                )
            )
    return {
        key: ()
        if key in refreshed_keys
        else tuple(sorted(rows, key=lambda row: row.source_global_slot))
        for key, rows in sources.items()
    }


def _status_durations(frame: EvaluationFrameV1, global_slot: int) -> tuple[int, ...]:
    snapshot = frame.snapshot
    return (
        *snapshot.slow_durations[global_slot],
        *snapshot.stun_durations[global_slot],
        snapshot.rogue_poison_anti_heal_durations[global_slot],
        snapshot.mage_burst_damage_amplification_durations[global_slot],
        snapshot.priest_blessing_of_freedom_slow_floor_durations[global_slot],
    )


def _status_source_state_from_frame(
    context: EvaluationEpisodeContextV1,
    frame: EvaluationFrameV1,
    evidence_by_key: dict[tuple[int, int], tuple[StatusSourceEvidenceSceneV2, ...]],
) -> StatusSourceEvidenceStateV2:
    catalog = context.static_mechanics_catalog
    rows: list[StatusSourceChannelEvidenceV2] = []
    for roster in context.roster:
        if not roster.configured_active:
            continue
        durations = _status_durations(frame, roster.global_slot)
        for status_channel, (duration, mechanic) in enumerate(
            zip(durations, catalog.status_channels, strict=True)
        ):
            if duration <= 0:
                continue
            rows.append(
                StatusSourceChannelEvidenceV2(
                    recipient_global_slot=roster.global_slot,
                    recipient_public_agent_id=roster.public_agent_id,
                    status_channel=status_channel,
                    status_id=mechanic.status_id,
                    direct_source_evidence=evidence_by_key.get(
                        (roster.global_slot, status_channel),
                        (),
                    ),
                )
            )
    return StatusSourceEvidenceStateV2(
        schema_version=STATUS_SOURCE_EVIDENCE_SCHEMA_VERSION,
        episode_id=context.identity.episode_id,
        frame_index=frame.frame_index,
        frame_id=frame.frame_id,
        active_statuses=tuple(rows),
    )


def initialize_status_source_evidence_v2(
    context: EvaluationEpisodeContextV1,
    initial_frame: EvaluationFrameV1,
) -> StatusSourceEvidenceStateV2:
    """Initialize frame-zero status evidence without inventing source agents."""
    _validate_projection_inputs(context, initial_frame, None)
    return _status_source_state_from_frame(context, initial_frame, {})


def advance_status_source_evidence_v2(
    previous_state: StatusSourceEvidenceStateV2,
    coherent_view: EvaluationTransitionViewV1,
) -> StatusSourceEvidenceStateV2:
    """Return the next immutable evidence state from one validated CP2 view."""
    if type(previous_state) is not StatusSourceEvidenceStateV2:
        raise TypeError("previous_state must be the exact StatusSourceEvidenceStateV2.")
    if type(coherent_view) is not EvaluationTransitionViewV1:
        raise TypeError("coherent_view must be EvaluationTransitionViewV1.")
    view = EvaluationTransitionViewV1(
        context=coherent_view.context,
        start_frame=coherent_view.start_frame,
        transition=coherent_view.transition,
        successor_frame=coherent_view.successor_frame,
    )
    if (
        previous_state.episode_id != view.context.identity.episode_id
        or previous_state.frame_index != view.start_frame.frame_index
        or previous_state.frame_id != view.start_frame.frame_id
    ):
        raise ValueError("previous status-source state must join the view start frame.")
    evidence_by_key = previous_state.evidence_by_recipient_and_channel()
    for event in view.transition.events:
        if type(event) is EvaluationStatusAppliedEventV1:
            key = (event.recipient_global_slot, event.status_channel)
            source = view.context.roster[event.source_global_slot]
            candidate = StatusSourceEvidenceSceneV2(
                source_global_slot=event.source_global_slot,
                source_public_agent_id=source.public_agent_id,
                event_id=event.event_id,
            )
            evidence_by_key[key] = tuple(
                sorted(
                    (*evidence_by_key.get(key, ()), candidate),
                    key=lambda row: (row.source_global_slot, row.event_id),
                )
            )
        elif type(event) is Cp2StatusRefreshedEventV1:
            evidence_by_key[(event.recipient_global_slot, event.status_channel)] = ()
        elif type(event) in (
            EvaluationStatusAgedToZeroEventV1,
            EvaluationStatusBrokenByDamageEventV1,
            EvaluationStatusClearedByNewDeathEventV1,
        ):
            lifecycle_event = cast(EvaluationStatusLifecycleEventBaseV1, event)
            evidence_by_key.pop(
                (
                    lifecycle_event.recipient_global_slot,
                    lifecycle_event.status_channel,
                ),
                None,
            )
    return _status_source_state_from_frame(
        view.context,
        view.successor_frame,
        evidence_by_key,
    )


def build_status_source_evidence_index_v2(
    context: EvaluationEpisodeContextV1,
    frames: tuple[EvaluationFrameV1, ...],
    transitions: tuple[EvaluationTransitionV1, ...],
) -> StatusSourceEvidenceIndexV2:
    """Build one O(T) replay index through the live replacement-state reducer."""
    if type(frames) is not tuple or type(transitions) is not tuple:
        raise TypeError("frames and transitions must be Python tuples.")
    if not frames or len(frames) != len(transitions) + 1:
        raise ValueError("status-source index requires exact T+1/T trajectory shape.")
    state = initialize_status_source_evidence_v2(context, frames[0])
    states = [state]
    for transition_index, transition in enumerate(transitions):
        if type(transition) is not EvaluationTransitionV1:
            raise TypeError("transitions must contain exact EvaluationTransitionV1.")
        view = EvaluationTransitionViewV1(
            context=context,
            start_frame=frames[transition_index],
            transition=transition,
            successor_frame=frames[transition_index + 1],
        )
        state = advance_status_source_evidence_v2(state, view)
        states.append(state)
    return StatusSourceEvidenceIndexV2(
        schema_version=STATUS_SOURCE_EVIDENCE_SCHEMA_VERSION,
        episode_id=context.identity.episode_id,
        frame_states=tuple(states),
    )


def _status_scenes(
    context: EvaluationEpisodeContextV1,
    frame: EvaluationFrameV1,
    global_slot: int,
    source_evidence: dict[tuple[int, int], tuple[StatusSourceEvidenceSceneV2, ...]],
) -> tuple[StatusSceneV2, ...]:
    durations = _status_durations(frame, global_slot)
    catalog = context.static_mechanics_catalog
    if len(durations) != len(catalog.status_channels):
        raise ValueError("selected-frame status axes do not match the context catalog.")
    rows: list[StatusSceneV2] = []
    for channel, (duration, mechanic) in enumerate(
        zip(durations, catalog.status_channels, strict=True)
    ):
        if duration <= 0:
            continue
        source_class = catalog.class_mechanics[mechanic.source_class_id]
        rows.append(
            StatusSceneV2(
                status_channel=channel,
                status_id=mechanic.status_id,
                family=mechanic.family,
                remaining_duration=duration,
                source_class_id=mechanic.source_class_id,
                source_class_name=source_class.class_name,
                source_action_component=mechanic.source_action_component,
                magnitude_kind=mechanic.magnitude_kind,
                magnitude=mechanic.magnitude,
                breaks_on_positive_damage=mechanic.breaks_on_positive_damage,
                direct_source_evidence=source_evidence.get((global_slot, channel), ()),
            )
        )
    return tuple(
        sorted(
            rows,
            key=lambda row: status_sort_key(
                status_token_id_from_catalog_status_id(row.status_id)
            ),
        )
    )


def _incoming_respawn_event_ids(
    transition_view: EvaluationTransitionViewV1 | None,
) -> dict[int, str]:
    if transition_view is None:
        return {}
    return {
        event.agent_global_slot: event.event_id
        for event in transition_view.transition.events
        if type(event) is EvaluationAgentRespawnedEventV1
    }


def _agent_scenes(
    context: EvaluationEpisodeContextV1,
    frame: EvaluationFrameV1,
    transition_view: EvaluationTransitionViewV1 | None,
    status_source_evidence_state: StatusSourceEvidenceStateV2 | None = None,
) -> tuple[AgentSceneV2, ...]:
    if status_source_evidence_state is None:
        source_evidence = _incoming_status_sources(context, transition_view)
    else:
        if (
            type(status_source_evidence_state) is not StatusSourceEvidenceStateV2
            or status_source_evidence_state.episode_id != context.identity.episode_id
            or status_source_evidence_state.frame_index != frame.frame_index
            or status_source_evidence_state.frame_id != frame.frame_id
        ):
            raise ValueError(
                "status_source_evidence_state must join the selected frame."
            )
        source_evidence = (
            status_source_evidence_state.evidence_by_recipient_and_channel()
        )
    respawn_event_ids = _incoming_respawn_event_ids(transition_view)
    snapshot = frame.snapshot
    rows: list[AgentSceneV2] = []
    for roster, mechanics in zip(
        context.roster,
        context.resolved_env_config.slot_mechanics,
        strict=True,
    ):
        if not roster.configured_active:
            continue
        respawn_event_id = respawn_event_ids.get(roster.global_slot)
        self_features = frame.base_observation.self_features[roster.global_slot]
        rows.append(
            AgentSceneV2(
                global_slot=roster.global_slot,
                public_agent_id=roster.public_agent_id,
                team_id=roster.configured_team_id,
                team_local_slot=roster.team_local_slot,
                class_id=roster.class_id,
                position=(
                    snapshot.agent_positions[roster.global_slot][0],
                    snapshot.agent_positions[roster.global_slot][1],
                ),
                radius=mechanics.body_radius,
                life_state=(
                    "alive" if snapshot.alive_mask[roster.global_slot] else "corpse"
                ),
                current_health=snapshot.current_health[roster.global_slot],
                max_health=mechanics.maximum_health,
                effective_movement_speed=self_features[
                    AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED_V1
                ],
                ultimate_cooldown_remaining=(
                    snapshot.ultimate_cooldowns[roster.global_slot]
                ),
                spawn_shield_remaining=(
                    snapshot.spawn_shield_durations[roster.global_slot]
                ),
                steps_until_out_of_combat=(
                    snapshot.steps_until_out_of_combat[roster.global_slot]
                ),
                respawned_on_incoming_transition=respawn_event_id is not None,
                respawn_event_id=respawn_event_id,
                statuses=_status_scenes(
                    context,
                    frame,
                    roster.global_slot,
                    source_evidence,
                ),
                aura_modifiers=(
                    AuraRecipientModifierSceneV2(
                        aura_id="mage_damage_amplification",
                        multiplier=self_features[
                            AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER_V1
                        ],
                    ),
                    AuraRecipientModifierSceneV2(
                        aura_id="warrior_damage_mitigation",
                        multiplier=self_features[
                            AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER_V1
                        ],
                    ),
                ),
            )
        )
    return tuple(rows)


def _aura_fields(
    context: EvaluationEpisodeContextV1,
    frame: EvaluationFrameV1,
) -> tuple[AuraFieldSceneV2, ...]:
    catalog = context.static_mechanics_catalog
    fields: list[AuraFieldSceneV2] = []
    for roster in context.roster:
        if not roster.configured_active:
            continue
        class_row = catalog.class_mechanics[roster.class_id]
        for aura in catalog.aura_mechanics:
            if aura.emitter_class_id != roster.class_id:
                continue
            fields.append(
                AuraFieldSceneV2(
                    aura_id=aura.aura_id,
                    source_global_slot=roster.global_slot,
                    source_public_agent_id=roster.public_agent_id,
                    source_class_id=roster.class_id,
                    source_class_name=class_row.class_name,
                    source_alive=frame.snapshot.alive_mask[roster.global_slot],
                    center=(
                        frame.snapshot.agent_positions[roster.global_slot][0],
                        frame.snapshot.agent_positions[roster.global_slot][1],
                    ),
                    radius=aura.radius,
                    beneficiary_relation=aura.beneficiary_relation,
                    per_emitter_multiplier=aura.per_emitter_multiplier,
                    stacking_rule=aura.stacking_rule,
                    clamp_kind=aura.clamp_kind,
                    clamp_value=aura.clamp_value,
                )
            )
    return tuple(
        sorted(fields, key=lambda field: (field.source_global_slot, field.aura_id))
    )


def _spawn_pads(
    context: EvaluationEpisodeContextV1,
) -> tuple[SpawnPadSceneV2, ...]:
    positions = context.resolved_env_config.team_spawn_pad_positions
    rows = [
        SpawnPadSceneV2(
            team_id=roster.configured_team_id,
            team_local_slot=roster.team_local_slot,
            assigned_global_slot=roster.global_slot,
            assigned_public_agent_id=roster.public_agent_id,
            position=positions[roster.configured_team_id - 1][roster.team_local_slot],
        )
        for roster in context.roster
        if roster.configured_active
    ]
    return tuple(sorted(rows, key=lambda row: (row.team_id, row.team_local_slot)))


def _respawn_waves(
    context: EvaluationEpisodeContextV1,
    frame: EvaluationFrameV1,
) -> tuple[RespawnWaveSceneV2, ...]:
    config = context.resolved_env_config
    return tuple(
        RespawnWaveSceneV2(
            team_index=team_index,
            team_id=team_index + 1,
            period_steps=config.team_respawn_wave_period_steps[team_index],
            countdown_steps=frame.snapshot.team_respawn_wave_countdowns[team_index],
        )
        for team_index in range(2)
    )


def _selection_projection(
    context: EvaluationEpisodeContextV1,
    frame: EvaluationFrameV1,
    presentation: EvaluationScenePresentationStateV1,
) -> tuple[
    tuple[RangeSceneV1, ...],
    SelectionSceneV1 | None,
    SelectedLegalitySceneV1 | None,
]:
    controlled = presentation.controlled_global_slot
    selected = presentation.selected_global_slot
    active_slots = {
        roster.global_slot for roster in context.roster if roster.configured_active
    }
    if controlled is None:
        return (), None, None
    if controlled not in active_slots:
        raise ValueError("controlled_global_slot must be configured active.")
    if selected is not None and selected not in active_slots:
        raise ValueError("selected_global_slot must be configured active.")
    selection = SelectionSceneV1(
        controlled_global_slot=controlled,
        selected_global_slot=selected,
    )
    mechanics = context.resolved_env_config.slot_mechanics[controlled]
    controlled_position = frame.snapshot.agent_positions[controlled]
    center = (controlled_position[0], controlled_position[1])
    range_specs: tuple[tuple[RangeKind, float], ...] = (
        ("observation", mechanics.observation_radius),
        ("basic", mechanics.basic_interaction_radius),
        ("ultimate", mechanics.ultimate_interaction_radius),
    )
    ranges = (
        tuple(
            RangeSceneV1(
                global_slot=controlled,
                center=center,
                radius=radius,
                kind=kind,
            )
            for kind, radius in range_specs
        )
        if presentation.show_ranges
        else ()
    )
    if selected is None:
        return ranges, selection, None
    catalog = context.static_mechanics_catalog
    target_mapping = catalog.global_recipient_slot_by_actor_and_target_action[
        controlled
    ]
    try:
        target_action = target_mapping.index(selected)
    except ValueError as exc:
        raise ValueError(
            "selected actor is absent from the context target-axis mapping."
        ) from exc
    lane_values = frame.action_mask.select_target_use_ultimate_joint_mask[controlled][
        target_action
    ]
    armed_lane = presentation.armed_lane
    legality = SelectedLegalitySceneV1(
        controlled_global_slot=controlled,
        target_global_slot=selected,
        target_action=target_action,
        lane_0_available=lane_values[0],
        lane_1_available=lane_values[1],
        armed_lane=armed_lane,
        armed_pair_legal=False if armed_lane is None else lane_values[armed_lane],
    )
    return ranges, selection, legality


def _observer_visibility_projection(
    context: EvaluationEpisodeContextV1,
    frame: EvaluationFrameV1,
    presentation: EvaluationScenePresentationStateV1,
) -> tuple[ObserverVisibilitySceneV1, ...]:
    """Project one researcher's exact recorded base-sensor visibility row."""
    observer = presentation.controlled_global_slot
    if observer is None:
        return ()
    catalog = context.static_mechanics_catalog
    ally_slots = catalog.global_slot_by_actor_and_ally_observation_row[observer]
    enemy_slots = catalog.global_slot_by_actor_and_enemy_observation_row[observer]
    observation = frame.base_observation
    visible_by_global_slot = dict(
        zip(
            (*ally_slots, *enemy_slots),
            (
                *observation.ally_visibility_mask[observer],
                *observation.enemy_visibility_mask[observer],
            ),
            strict=True,
        )
    )
    return tuple(
        ObserverVisibilitySceneV1(
            observer_global_slot=observer,
            candidate_global_slot=roster.global_slot,
            visible=visible_by_global_slot[roster.global_slot],
        )
        for roster in context.roster
        if roster.configured_active
    )


def build_evaluation_battlefield_scene_v2(
    context: EvaluationEpisodeContextV1,
    frame: EvaluationFrameV1,
    *,
    transition_view: EvaluationTransitionViewV1 | None = None,
    audience: Literal["researcher"] = "researcher",
    presentation: EvaluationScenePresentationStateV1 | None = None,
    status_source_evidence_state: StatusSourceEvidenceStateV2 | None = None,
) -> BattlefieldSceneV2:
    """Project one canonical selected frame into the V2 researcher scene."""
    if audience != "researcher":
        raise ValueError(
            "researcher projection cannot be reused as an actor-POV projection."
        )
    canonical_view = _validate_projection_inputs(context, frame, transition_view)
    if (
        presentation is not None
        and type(presentation) is not EvaluationScenePresentationStateV1
    ):
        raise TypeError(
            "presentation must be EvaluationScenePresentationStateV1 or None."
        )
    state = presentation or EvaluationScenePresentationStateV1()
    ranges, selection, legality = _selection_projection(context, frame, state)
    transition = None if canonical_view is None else canonical_view.transition
    return BattlefieldSceneV2(
        schema_version=SCENE_V2_SCHEMA_VERSION,
        audience="researcher",
        audience_badge="PRIVILEGED RESEARCHER VIEW · CANONICAL EVALUATION",
        episode_id=context.identity.episode_id,
        frame_index=frame.frame_index,
        frame_id=frame.frame_id,
        simulator_step_count=frame.simulator_step_count,
        incoming_transition_id=(
            None if transition is None else transition.transition_id
        ),
        incoming_event_ids=(
            ()
            if transition is None
            else tuple(event.event_id for event in transition.events)
        ),
        map=_map_scene(context),
        agents=_agent_scenes(
            context,
            frame,
            canonical_view,
            status_source_evidence_state,
        ),
        aura_fields=_aura_fields(context, frame),
        class_mechanics=_class_mechanics(context),
        spawn_pads=_spawn_pads(context),
        respawn_waves=_respawn_waves(context, frame),
        ranges=ranges,
        selection=selection,
        next_decision_selected_legality=legality,
        observer_visibility=_observer_visibility_projection(context, frame, state),
    )


def build_shared_obs_source_material_projection_v1(
    context: EvaluationEpisodeContextV1,
    frame: EvaluationFrameV1,
    *,
    selected_global_slot: int,
    transition_view: EvaluationTransitionViewV1 | None = None,
) -> SharedObsSourceMaterialProjectionV1:
    """Project labelled base-sensor and availability evidence for SharedObs.

    This full-record researcher aid deliberately does not export or claim the
    composed actor input.  Exact SharedObs materialization remains unavailable.
    """
    canonical_view = _validate_projection_inputs(context, frame, transition_view)
    if context.execution_information_mode != "shared_obs":
        raise ValueError(
            "SharedObs source-material projection requires a shared_obs episode."
        )
    if type(selected_global_slot) is not int or not (
        0 <= selected_global_slot < MAX_AGENT_SLOTS_V1
    ):
        raise ValueError("selected_global_slot is outside the V1 actor axis.")
    recipient = context.roster[selected_global_slot]
    if not recipient.configured_active:
        raise ValueError(
            "SharedObs source material requires a configured-active actor."
        )
    availability_matrix = (
        frame.shared_obs_information_availability_by_recipient_and_sensor_source
    )
    if availability_matrix is None:
        raise ValueError("SharedObs source material requires recorded availability.")
    axis_mapping = _base_sensor_axis_mapping(
        context,
        selected_global_slot=selected_global_slot,
    )
    base_sensor_frame = _base_sensor_frame(
        frame,
        selected_global_slot=selected_global_slot,
        public_agent_id=recipient.public_agent_id,
    )
    base_sensor_scene = _shared_obs_base_sensor_scene(
        base_sensor_frame,
        selected_global_slot=selected_global_slot,
        selected_team_local_slot=recipient.team_local_slot,
        configured_team_id=recipient.configured_team_id,
        class_id=recipient.class_id,
        axis_mapping=axis_mapping,
    )
    catalog = context.static_mechanics_catalog
    ally_slots = catalog.global_slot_by_actor_and_ally_observation_row[
        selected_global_slot
    ]
    enemy_slots = catalog.global_slot_by_actor_and_enemy_observation_row[
        selected_global_slot
    ]
    source_rows: list[SharedObsSensorSourceAvailabilityV1] = []
    for source in context.roster:
        source_slot = source.global_slot
        if source_slot in ally_slots:
            relation_axis: Literal["ally", "enemy"] = "ally"
            observation_row = ally_slots.index(source_slot)
        else:
            relation_axis = "enemy"
            observation_row = enemy_slots.index(source_slot)
        if not source.configured_active:
            recipient_relation: Literal["self", "ally", "opponent", "inactive"] = (
                "inactive"
            )
        elif source_slot == selected_global_slot:
            recipient_relation = "self"
        elif source.configured_team_id == recipient.configured_team_id:
            recipient_relation = "ally"
        else:
            recipient_relation = "opponent"
        source_rows.append(
            SharedObsSensorSourceAvailabilityV1(
                sensor_source_global_slot=source_slot,
                sensor_source_public_agent_id=source.public_agent_id,
                sensor_source_team_local_slot=source.team_local_slot,
                sensor_source_configured_team_id=source.configured_team_id,
                sensor_source_configured_active=source.configured_active,
                relation_to_recipient=recipient_relation,
                base_sensor_relation_axis=relation_axis,
                base_sensor_observation_row=observation_row,
                recorded_available=availability_matrix[selected_global_slot][
                    source_slot
                ],
            )
        )
    return SharedObsSourceMaterialProjectionV1(
        schema_version=SHARED_OBS_SOURCE_MATERIAL_PROJECTION_SCHEMA_VERSION,
        disclosure_label=_SHARED_OBS_SOURCE_MATERIAL_DISCLOSURE,
        observation_materialization="source_material_only",
        exact_actor_input_export_available=False,
        axis_mapping=axis_mapping,
        ally_observation_row_global_slot_by_id=ally_slots,
        enemy_observation_row_global_slot_by_id=enemy_slots,
        base_sensor_frame=base_sensor_frame,
        base_sensor_scene=base_sensor_scene,
        incoming_transition_id=(
            None if canonical_view is None else canonical_view.transition.transition_id
        ),
        sensor_source_availability=tuple(source_rows),
    )


def _visual_phase_trajectories_v2(
    view: EvaluationTransitionViewV1,
) -> tuple[VisualAgentPhaseTrajectoryV2, ...]:
    context = view.context
    start_positions = view.start_frame.snapshot.agent_positions
    successor_positions = view.successor_frame.snapshot.agent_positions
    charge_displacements = (
        view.transition.facts.physical_facts.charge_phase_displacement_by_agent
    )
    rows: list[VisualAgentPhaseTrajectoryV2] = []
    for roster in context.roster:
        if not roster.configured_active:
            continue
        global_slot = roster.global_slot
        start = start_positions[global_slot]
        charge = charge_displacements[global_slot]
        post_charge = (start[0] + charge[0], start[1] + charge[1])
        successor = successor_positions[global_slot]
        rows.append(
            VisualAgentPhaseTrajectoryV2(
                global_slot=global_slot,
                public_agent_id=roster.public_agent_id,
                transition_start=VisualAgentAnchorV2(
                    phase="transition_start",
                    global_slot=global_slot,
                    public_agent_id=roster.public_agent_id,
                    position=(start[0], start[1]),
                ),
                post_charge=VisualAgentAnchorV2(
                    phase="post_charge",
                    global_slot=global_slot,
                    public_agent_id=roster.public_agent_id,
                    position=post_charge,
                ),
                successor=VisualAgentAnchorV2(
                    phase="successor",
                    global_slot=global_slot,
                    public_agent_id=roster.public_agent_id,
                    position=(successor[0], successor[1]),
                ),
            )
        )
    return tuple(rows)


class _VisualEventIdentityV2(TypedDict):
    event_id: str
    transition_id: str
    ordinal: int


def _project_visual_event_v2(
    event: EvaluationEventV1,
    *,
    trajectory_by_slot: dict[int, VisualAgentPhaseTrajectoryV2],
    public_agent_id_by_global_slot: tuple[str, ...],
    configured_active_by_global_slot: tuple[bool, ...],
) -> VisualEventV2:
    def anchor(
        global_slot: int,
        phase: Literal["transition_start", "post_charge", "successor"],
    ) -> VisualAgentAnchorV2:
        trajectory = trajectory_by_slot.get(global_slot)
        if trajectory is None:
            raise ValueError(
                "canonical event references an inactive presentation slot."
            )
        return cast(VisualAgentAnchorV2, getattr(trajectory, phase))

    identity: _VisualEventIdentityV2 = {
        "event_id": event.event_id,
        "transition_id": event.transition_id,
        "ordinal": event.ordinal,
    }
    if type(event) is EvaluationActionRejectedEventV1:
        return ActionRejectedEventV2(
            **identity,
            actor_global_slot=event.actor_global_slot,
            actor_public_agent_id=public_agent_id_by_global_slot[
                event.actor_global_slot
            ],
            actor_configured_active=configured_active_by_global_slot[
                event.actor_global_slot
            ],
            rejection_component=event.rejection_component,
            submitted_move_action=event.submitted_move_action,
            submitted_select_target_action=event.submitted_select_target_action,
            submitted_use_ultimate_action=event.submitted_use_ultimate_action,
            actor_anchor=(
                anchor(event.actor_global_slot, "transition_start")
                if configured_active_by_global_slot[event.actor_global_slot]
                else None
            ),
        )
    if type(event) is EvaluationAbilityActivatedEventV1:
        return AbilityActivatedEventV2(
            **identity,
            source_global_slot=event.source_global_slot,
            ability_component=event.ability_component,
            recipient_global_slot=event.recipient_global_slot,
            source_anchor=anchor(event.source_global_slot, "transition_start"),
            recipient_anchor=(
                None
                if event.recipient_global_slot is None
                else anchor(event.recipient_global_slot, "transition_start")
            ),
        )
    if type(event) is EvaluationSourceDamageOutputEventV1:
        return SourceDamageOutputEventV2(
            **identity,
            source_global_slot=event.source_global_slot,
            recipient_global_slot=event.recipient_global_slot,
            raw_damage_output=event.raw_damage_output,
            source_modified_damage_output=event.source_modified_damage_output,
            recipient_damage_modifier=event.recipient_damage_modifier,
            mage_damage_aura_covering_emitter_global_slots=(
                event.mage_damage_aura_covering_emitter_global_slots
            ),
            warrior_mitigation_aura_covering_emitter_global_slots=(
                event.warrior_mitigation_aura_covering_emitter_global_slots
            ),
            source_anchor=anchor(event.source_global_slot, "transition_start"),
            recipient_anchor=(
                None
                if event.recipient_global_slot is None
                else anchor(event.recipient_global_slot, "transition_start")
            ),
        )
    if type(event) is EvaluationSourceHealingOutputEventV1:
        return SourceHealingOutputEventV2(
            **identity,
            source_global_slot=event.source_global_slot,
            recipient_global_slot=event.recipient_global_slot,
            raw_healing_output=event.raw_healing_output,
            source_modified_healing_output=event.source_modified_healing_output,
            recipient_healing_modifier=event.recipient_healing_modifier,
            source_anchor=anchor(event.source_global_slot, "transition_start"),
            recipient_anchor=(
                None
                if event.recipient_global_slot is None
                else anchor(event.recipient_global_slot, "transition_start")
            ),
        )
    if type(event) is Cp2RecipientHealthEventV1:
        return RecipientHealthResolutionEventV2(
            **identity,
            recipient_global_slot=event.recipient_global_slot,
            transition_start_health=event.transition_start_health,
            total_effective_damage=event.total_effective_damage,
            total_effective_healing=event.total_effective_healing,
            health_after_combat_resolution=event.health_after_combat_resolution,
            realized_net_health_change=event.realized_net_health_change,
            recipient_anchor=anchor(
                event.recipient_global_slot,
                "transition_start",
            ),
        )
    if type(event) is EvaluationCombatCountdownResetEventV1:
        return CombatCountdownResetEventV2(
            **identity,
            agent_global_slot=event.agent_global_slot,
            agent_anchor=anchor(event.agent_global_slot, "transition_start"),
        )
    if type(event) is EvaluationHealthRegeneratedEventV1:
        return HealthRegeneratedEventV2(
            **identity,
            agent_global_slot=event.agent_global_slot,
            actual_health_regenerated=event.actual_health_regenerated,
            agent_anchor=anchor(event.agent_global_slot, "transition_start"),
        )
    if type(event) is EvaluationCooldownStartedEventV1:
        return CooldownStartedEventV2(
            **identity,
            agent_global_slot=event.agent_global_slot,
            agent_anchor=anchor(event.agent_global_slot, "transition_start"),
        )
    if type(event) is EvaluationCooldownReadyEventV1:
        return CooldownReadyEventV2(
            **identity,
            agent_global_slot=event.agent_global_slot,
            agent_anchor=anchor(event.agent_global_slot, "transition_start"),
        )
    if type(event) is EvaluationChargePhaseDisplacementEventV1:
        return ChargePhaseDisplacementEventV2(
            **identity,
            agent_global_slot=event.agent_global_slot,
            realized_displacement=event.realized_displacement,
            start_anchor=anchor(event.agent_global_slot, "transition_start"),
            end_anchor=anchor(event.agent_global_slot, "post_charge"),
        )
    if type(event) is Cp2OrdinaryMovementEventV1:
        return OrdinaryMovementPhaseDisplacementEventV2(
            **identity,
            agent_global_slot=event.agent_global_slot,
            realized_displacement=event.realized_displacement,
            start_anchor=anchor(event.agent_global_slot, "post_charge"),
            end_anchor=anchor(event.agent_global_slot, "successor"),
        )
    if type(event) is EvaluationAgentDiedEventV1:
        return AgentDiedEventV2(
            **identity,
            recipient_global_slot=event.recipient_global_slot,
            recipient_anchor=anchor(event.recipient_global_slot, "successor"),
        )
    if type(event) is EvaluationLethalDamageContributionEventV1:
        return LethalDamageContributionEventV2(
            **identity,
            source_global_slot=event.source_global_slot,
            recipient_global_slot=event.recipient_global_slot,
            attributed_death_damage=event.attributed_death_damage,
            source_anchor=anchor(event.source_global_slot, "successor"),
            recipient_anchor=anchor(event.recipient_global_slot, "successor"),
        )
    if type(event) is EvaluationStatusAgedToZeroEventV1:
        return StatusAgedToZeroEventV2(
            **identity,
            recipient_global_slot=event.recipient_global_slot,
            status_channel=event.status_channel,
            status_id=event.status_id,
            recipient_anchor=anchor(event.recipient_global_slot, "successor"),
        )
    if type(event) is EvaluationStatusBrokenByDamageEventV1:
        return StatusBrokenByDamageEventV2(
            **identity,
            recipient_global_slot=event.recipient_global_slot,
            status_channel=event.status_channel,
            status_id=event.status_id,
            recipient_anchor=anchor(event.recipient_global_slot, "successor"),
        )
    if type(event) is EvaluationStatusAppliedEventV1:
        return StatusAppliedEventV2(
            **identity,
            recipient_global_slot=event.recipient_global_slot,
            status_channel=event.status_channel,
            status_id=event.status_id,
            recipient_anchor=anchor(event.recipient_global_slot, "successor"),
            source_global_slot=event.source_global_slot,
            source_anchor=anchor(event.source_global_slot, "successor"),
        )
    if type(event) is Cp2StatusRefreshedEventV1:
        return StatusRefreshedOrExtendedEventV2(
            **identity,
            recipient_global_slot=event.recipient_global_slot,
            status_channel=event.status_channel,
            status_id=event.status_id,
            recipient_anchor=anchor(event.recipient_global_slot, "successor"),
        )
    if type(event) is EvaluationStatusClearedByNewDeathEventV1:
        return StatusClearedByNewDeathEventV2(
            **identity,
            recipient_global_slot=event.recipient_global_slot,
            status_channel=event.status_channel,
            status_id=event.status_id,
            recipient_anchor=anchor(event.recipient_global_slot, "successor"),
        )
    if type(event) is EvaluationSpawnShieldExpiredEventV1:
        return SpawnShieldExpiredEventV2(
            **identity,
            agent_global_slot=event.agent_global_slot,
            agent_anchor=anchor(event.agent_global_slot, "successor"),
        )
    if type(event) is EvaluationRespawnWaveOccurredEventV1:
        return RespawnWaveOccurredEventV2(
            **identity,
            team_index=event.team_index,
            team_id=event.team_id,
            team_anchor=VisualTeamAnchorV2(
                phase="successor",
                team_index=event.team_index,
                team_id=event.team_id,
            ),
        )
    if type(event) is EvaluationAgentRespawnedEventV1:
        return AgentRespawnedEventV2(
            **identity,
            agent_global_slot=event.agent_global_slot,
            team_id=event.team_id,
            realized_successor_position=event.realized_successor_position,
            agent_anchor=anchor(event.agent_global_slot, "successor"),
        )
    if type(event) is EvaluationTeamDeathmatchScoreChangedEventV1:
        return TeamDeathmatchScoreChangedEventV2(
            **identity,
            team_index=event.team_index,
            team_id=event.team_id,
            score_increment=event.score_increment,
            previous_score=event.previous_score,
            successor_score=event.successor_score,
            team_anchor=VisualTeamAnchorV2(
                phase="successor",
                team_index=event.team_index,
                team_id=event.team_id,
            ),
        )
    if type(event) is EvaluationTeamDeathmatchCompletedEventV1:
        return TeamDeathmatchCompletedEventV2(
            **identity,
            outcome=event.outcome,
            completion_basis=event.completion_basis,
        )
    raise TypeError(f"unsupported canonical event type: {type(event).__name__}.")


def build_visual_event_batch_v2(
    coherent_view: EvaluationTransitionViewV1,
) -> VisualEventBatchV2:
    """Project every canonical event independently without cross-event joins."""
    if type(coherent_view) is not EvaluationTransitionViewV1:
        raise TypeError("coherent_view must be EvaluationTransitionViewV1.")
    view = EvaluationTransitionViewV1(
        context=coherent_view.context,
        start_frame=coherent_view.start_frame,
        transition=coherent_view.transition,
        successor_frame=coherent_view.successor_frame,
    )
    trajectories = _visual_phase_trajectories_v2(view)
    by_slot = {row.global_slot: row for row in trajectories}
    public_agent_ids = tuple(row.public_agent_id for row in view.context.roster)
    configured_active = tuple(row.configured_active for row in view.context.roster)
    return VisualEventBatchV2(
        schema_version=EVENT_V2_SCHEMA_VERSION,
        episode_id=view.context.identity.episode_id,
        transition_index=view.transition.transition_index,
        transition_id=view.transition.transition_id,
        start_frame_id=view.start_frame.frame_id,
        successor_frame_id=view.successor_frame.frame_id,
        start_simulator_step_count=view.start_frame.simulator_step_count,
        successor_simulator_step_count=view.successor_frame.simulator_step_count,
        public_agent_id_by_global_slot=public_agent_ids,
        configured_active_by_global_slot=configured_active,
        agent_phase_trajectories=trajectories,
        events=tuple(
            _project_visual_event_v2(
                event,
                trajectory_by_slot=by_slot,
                public_agent_id_by_global_slot=public_agent_ids,
                configured_active_by_global_slot=configured_active,
            )
            for event in view.transition.events
        ),
    )


def build_researcher_analyzer_projection_v2(
    context: EvaluationEpisodeContextV1,
    frame: EvaluationFrameV1,
    *,
    transition_view: EvaluationTransitionViewV1 | None = None,
    presentation: EvaluationScenePresentationStateV1 | None = None,
    status_source_evidence_state: StatusSourceEvidenceStateV2 | None = None,
) -> ResearcherAnalyzerProjectionV2:
    """Build the stable researcher scene/event envelope for one frame."""
    canonical_view = _validate_projection_inputs(context, frame, transition_view)
    if frame.frame_index == 0:
        evidence_state = (
            initialize_status_source_evidence_v2(context, frame)
            if status_source_evidence_state is None
            else status_source_evidence_state
        )
    else:
        if status_source_evidence_state is None:
            raise ValueError(
                "non-initial researcher projections require frame-bound "
                "status-source evidence state."
            )
        evidence_state = status_source_evidence_state
    scene = build_evaluation_battlefield_scene_v2(
        context,
        frame,
        transition_view=canonical_view,
        presentation=presentation,
        status_source_evidence_state=evidence_state,
    )
    return ResearcherAnalyzerProjectionV2(
        schema_version=RESEARCHER_ANALYZER_PROJECTION_SCHEMA_VERSION,
        scene=scene,
        incoming_events=(
            None
            if canonical_view is None
            else build_visual_event_batch_v2(canonical_view)
        ),
        status_source_evidence=evidence_state,
    )


__all__ = [
    "SHARED_OBS_SOURCE_MATERIAL_PROJECTION_SCHEMA_VERSION",
    "EvaluationScenePresentationStateV1",
    "SharedObsBaseSensorFrameV1",
    "SharedObsBaseSensorSceneV1",
    "SharedObsSensorSourceAvailabilityV1",
    "SharedObsSourceMaterialProjectionV1",
    "advance_status_source_evidence_v2",
    "build_evaluation_battlefield_scene_v2",
    "build_researcher_analyzer_projection_v2",
    "build_shared_obs_source_material_projection_v1",
    "build_status_source_evidence_index_v2",
    "build_visual_event_batch_v2",
    "initialize_status_source_evidence_v2",
]
