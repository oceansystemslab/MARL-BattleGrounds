"""Recipient-sliced actor-POV replay artifacts.

The export boundary in this module is deliberately narrower than the standard
researcher replay.  Every frame is copied from one actor's recorded policy
input row, and every transition contains only that actor's submitted/accepted
action, rejection flags, reward, public done truth, and locally derived cues.
Privileged snapshots and CP2 events are never consulted while building cues.

The artifact separates recipient-authorized content from source provenance.
This preserves a truthful one-way reference to the full replay while allowing
privacy noninterference to be checked over canonical authorized-content bytes.
"""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict, cast

from pydantic import BeforeValidator, Field, StringConstraints, model_validator

from marl_battlegrounds.evaluation.metrics import EvaluationTransitionViewV1
from marl_battlegrounds.evaluation.models import (
    CONTEXT_FEATURES,
    CONTEXT_SCHEMA_ID,
    ENVIRONMENT_DIMENSIONS,
    FRAME_SCHEMA_ID,
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    MAX_OBJECTIVE_SLOTS,
    MAX_OBSTACLE_SLOTS,
    NUM_MOVE_ACTIONS,
    NUM_TARGET_ACTIONS,
    NUM_TEAMS,
    NUM_ULTIMATE_ACTIONS,
    OBJECTIVE_FEATURES,
    OBSTACLE_FEATURES,
    SELF_FEATURES,
    TRANSITION_SCHEMA_ID,
    UNIT_FEATURES,
    EvaluationEpisodeContextV1,
    EvaluationFrameV1,
    EvaluationModel,
    EvaluationTransitionV1,
    RosterSlotV1,
    canonical_digest_sha256,
    canonical_json_bytes,
)
from marl_battlegrounds.evaluation.replay import (
    ReplayArtifactReferenceV1,
    ReplayArtifactV1,
    validate_replay_artifact_v1,
)
from marl_battlegrounds.evaluation.validation import (
    validate_declared_model_tree,
    validate_initial_evaluation_frame_v1,
)

ACTOR_POV_SCHEMA_VERSION: Literal[1] = 1
ACTOR_POV_COMPLETION_SCHEMA_ID = "marl_battlegrounds.evaluation.actor_pov_completion"
ACTOR_POV_PREVIOUS_ACTIONS_SCHEMA_ID = (
    "marl_battlegrounds.evaluation.actor_pov_previous_actions"
)
ACTOR_POV_SPAWN_LIFECYCLE_SCHEMA_ID = (
    "marl_battlegrounds.evaluation.actor_pov_spawn_lifecycle"
)
ACTOR_POV_ACTION_MASK_SCHEMA_ID = "marl_battlegrounds.evaluation.actor_pov_action_mask"
ACTOR_POV_AXIS_MAPPING_SCHEMA_ID = (
    "marl_battlegrounds.evaluation.actor_pov_axis_mapping"
)
ACTOR_POV_FRAME_SCHEMA_ID = "marl_battlegrounds.evaluation.actor_pov_frame"
ACTOR_POV_SUBMITTED_ACTION_SCHEMA_ID = (
    "marl_battlegrounds.evaluation.actor_pov_submitted_action"
)
ACTOR_POV_ACCEPTED_ACTION_SCHEMA_ID = (
    "marl_battlegrounds.evaluation.actor_pov_accepted_action"
)
ACTOR_POV_CUE_SCHEMA_ID = "marl_battlegrounds.evaluation.actor_pov_cue"
ACTOR_POV_TRANSITION_SCHEMA_ID = "marl_battlegrounds.evaluation.actor_pov_transition"
ACTOR_POV_CURRENT_SLICE_SCHEMA_ID = (
    "marl_battlegrounds.evaluation.actor_pov_current_slice"
)
ACTOR_POV_ADJACENT_TRANSITION_SLICE_SCHEMA_ID = (
    "marl_battlegrounds.evaluation.actor_pov_adjacent_transition_slice"
)
ACTOR_POV_CONTENT_SCHEMA_ID = "marl_battlegrounds.evaluation.actor_pov_content"
ACTOR_POV_ARTIFACT_SCHEMA_ID = "marl_battlegrounds.evaluation.actor_pov_artifact"

# V1 coordinates in the serialized 58-value actor/unit feature contract.  They
# are artifact-schema constants, not imports from the simulator.  Changing the
# policy-input axis requires a new POV schema version.
_FEATURE_X = 0
_FEATURE_Y = 1
_FEATURE_TEAM_ID = 3
_FEATURE_ACTIVE = 4
_FEATURE_ALIVE = 5
_FEATURE_CLASS_ID = 6
_FEATURE_CURRENT_HEALTH = 12
_FEATURE_ULTIMATE_COOLDOWN = 14
_STATUS_FEATURE_START = 15
_STATUS_FEATURE_STOP = 29


def _require_schema_version_one(value: object) -> object:
    if type(value) is not int or value != 1:
        raise ValueError("schema_version must be the exact integer 1")
    return value


_SchemaVersionV1 = Annotated[
    Literal[1],
    BeforeValidator(_require_schema_version_one),
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
_Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
_NonNegativeInt = Annotated[int, Field(ge=0)]
_GlobalSlot = Annotated[int, Field(ge=0, lt=MAX_AGENT_SLOTS)]
_TeamLocalSlot = Annotated[int, Field(ge=0, lt=MAX_AGENTS_PER_TEAM)]
_TeamId = Annotated[int, Field(ge=1, le=NUM_TEAMS)]
_ClassId = Annotated[int, Field(ge=1, le=5)]
_MoveAction = Annotated[int, Field(ge=0, lt=NUM_MOVE_ACTIONS)]
_TargetAction = Annotated[int, Field(ge=0, lt=NUM_TARGET_ACTIONS)]
_UltimateAction = Annotated[int, Field(ge=0, lt=NUM_ULTIMATE_ACTIONS)]
_Int32 = Annotated[int, Field(ge=-(2**31), le=2**31 - 1)]

type _FloatVector = tuple[_FiniteFloat, ...]
type _FloatMatrix = tuple[_FloatVector, ...]
type _FloatTensor3 = tuple[_FloatMatrix, ...]
type _BooleanVector = tuple[bool, ...]
type _BooleanMatrix = tuple[_BooleanVector, ...]
type _IntegerVector = tuple[_NonNegativeInt, ...]
type _IntegerMatrix = tuple[_IntegerVector, ...]


def _require_tuple_shape(
    value: object,
    expected_shape: tuple[int, ...],
    *,
    field_name: str,
) -> None:
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


class ActorPovEpisodeCompletionV1(EvaluationModel):
    """Minimal rollout-prefix truth safe for a standalone POV consumer."""

    schema_id: Literal["marl_battlegrounds.evaluation.actor_pov_completion"] = (
        ACTOR_POV_COMPLETION_SCHEMA_ID
    )
    schema_version: _SchemaVersionV1 = ACTOR_POV_SCHEMA_VERSION
    completion_state: Literal["complete", "partial", "interrupted", "failed"]
    expected_transition_count: Annotated[int, Field(gt=0)]
    captured_transition_count: _NonNegativeInt
    terminated: bool
    truncated: bool
    completion_bases: tuple[Literal["task_terminal", "declared_horizon"], ...]
    public_end_or_failure_reason: _AsciiText | None = None

    @model_validator(mode="after")
    def _validate_completion(self) -> ActorPovEpisodeCompletionV1:
        if self.captured_transition_count > self.expected_transition_count:
            raise ValueError("captured transitions cannot exceed the horizon")
        expected_bases: list[Literal["task_terminal", "declared_horizon"]] = []
        if self.terminated:
            expected_bases.append("task_terminal")
        if self.captured_transition_count == self.expected_transition_count:
            expected_bases.append("declared_horizon")
        if self.completion_state == "complete":
            if not expected_bases or self.completion_bases != tuple(expected_bases):
                raise ValueError(
                    "complete POV content must preserve terminal/horizon evidence"
                )
        else:
            if expected_bases or self.completion_bases:
                raise ValueError(
                    "incomplete POV content cannot carry completion evidence"
                )
            if self.public_end_or_failure_reason is None:
                raise ValueError("incomplete POV content requires a public reason")
        if self.captured_transition_count == 0 and (self.terminated or self.truncated):
            raise ValueError("a zero-transition POV prefix cannot carry done flags")
        return self


class ActorPovPreviousTimestepActionsV1(EvaluationModel):
    """The selected recipient row of all six previous-action tensors."""

    schema_id: Literal["marl_battlegrounds.evaluation.actor_pov_previous_actions"] = (
        ACTOR_POV_PREVIOUS_ACTIONS_SCHEMA_ID
    )
    schema_version: _SchemaVersionV1 = ACTOR_POV_SCHEMA_VERSION
    ally_move_actions_one_hot: _FloatMatrix
    enemy_move_actions_one_hot: _FloatMatrix
    ally_select_target_actions_one_hot: _FloatMatrix
    enemy_select_target_actions_one_hot: _FloatMatrix
    ally_use_ultimate_actions_one_hot: _FloatMatrix
    enemy_use_ultimate_actions_one_hot: _FloatMatrix

    @model_validator(mode="after")
    def _validate_shapes(self) -> ActorPovPreviousTimestepActionsV1:
        for field_name, category_count in (
            ("ally_move_actions_one_hot", NUM_MOVE_ACTIONS),
            ("enemy_move_actions_one_hot", NUM_MOVE_ACTIONS),
            ("ally_select_target_actions_one_hot", NUM_TARGET_ACTIONS),
            ("enemy_select_target_actions_one_hot", NUM_TARGET_ACTIONS),
            ("ally_use_ultimate_actions_one_hot", NUM_ULTIMATE_ACTIONS),
            ("enemy_use_ultimate_actions_one_hot", NUM_ULTIMATE_ACTIONS),
        ):
            _require_tuple_shape(
                getattr(self, field_name),
                (MAX_AGENTS_PER_TEAM, category_count),
                field_name=field_name,
            )
        return self


class ActorPovSpawnLifecycleV1(EvaluationModel):
    """The selected actor's actor-relative spawn-lifecycle observation row."""

    schema_id: Literal["marl_battlegrounds.evaluation.actor_pov_spawn_lifecycle"] = (
        ACTOR_POV_SPAWN_LIFECYCLE_SCHEMA_ID
    )
    schema_version: _SchemaVersionV1 = ACTOR_POV_SCHEMA_VERSION
    spawn_pad_positions_by_team: _FloatTensor3
    spawn_shield_actual_durations_by_team: _IntegerMatrix
    spawn_shield_configured_duration: _NonNegativeInt
    spawn_shield_speed: Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
    respawn_wave_period_step_count_by_team: _IntegerVector
    respawn_wave_countdowns_by_team: _IntegerVector
    active_mask_by_team: _BooleanMatrix
    alive_mask_by_team: _BooleanMatrix

    @model_validator(mode="after")
    def _validate_shapes(self) -> ActorPovSpawnLifecycleV1:
        _require_tuple_shape(
            self.spawn_pad_positions_by_team,
            (NUM_TEAMS, MAX_AGENTS_PER_TEAM, ENVIRONMENT_DIMENSIONS),
            field_name="spawn_pad_positions_by_team",
        )
        for field_name in (
            "spawn_shield_actual_durations_by_team",
            "active_mask_by_team",
            "alive_mask_by_team",
        ):
            _require_tuple_shape(
                getattr(self, field_name),
                (NUM_TEAMS, MAX_AGENTS_PER_TEAM),
                field_name=field_name,
            )
        for field_name in (
            "respawn_wave_period_step_count_by_team",
            "respawn_wave_countdowns_by_team",
        ):
            _require_tuple_shape(
                getattr(self, field_name),
                (NUM_TEAMS,),
                field_name=field_name,
            )
        return self


class ActorPovActionMaskV1(EvaluationModel):
    """The exact action-mask row paired with one actor decision."""

    schema_id: Literal["marl_battlegrounds.evaluation.actor_pov_action_mask"] = (
        ACTOR_POV_ACTION_MASK_SCHEMA_ID
    )
    schema_version: _SchemaVersionV1 = ACTOR_POV_SCHEMA_VERSION
    move: _BooleanVector
    select_target: _BooleanVector
    use_ultimate: _BooleanVector
    select_target_use_ultimate_joint: _BooleanMatrix

    @model_validator(mode="after")
    def _validate_shapes_and_marginals(self) -> ActorPovActionMaskV1:
        _require_tuple_shape(self.move, (NUM_MOVE_ACTIONS,), field_name="move")
        _require_tuple_shape(
            self.select_target,
            (NUM_TARGET_ACTIONS,),
            field_name="select_target",
        )
        _require_tuple_shape(
            self.use_ultimate,
            (NUM_ULTIMATE_ACTIONS,),
            field_name="use_ultimate",
        )
        _require_tuple_shape(
            self.select_target_use_ultimate_joint,
            (NUM_TARGET_ACTIONS, NUM_ULTIMATE_ACTIONS),
            field_name="select_target_use_ultimate_joint",
        )
        target_marginal = tuple(
            any(row) for row in self.select_target_use_ultimate_joint
        )
        ultimate_marginal = tuple(
            any(
                self.select_target_use_ultimate_joint[target][ultimate]
                for target in range(NUM_TARGET_ACTIONS)
            )
            for ultimate in range(NUM_ULTIMATE_ACTIONS)
        )
        if self.select_target != target_marginal:
            raise ValueError("POV select-target mask must equal its joint marginal")
        if self.use_ultimate != ultimate_marginal:
            raise ValueError("POV Ultimate mask must equal its joint marginal")
        return self


class ActorPovAxisMappingV1(EvaluationModel):
    """Recipient-local categorical-axis vocabulary needed offline."""

    schema_id: Literal["marl_battlegrounds.evaluation.actor_pov_axis_mapping"] = (
        ACTOR_POV_AXIS_MAPPING_SCHEMA_ID
    )
    schema_version: _SchemaVersionV1 = ACTOR_POV_SCHEMA_VERSION
    actor_projection_identifier: _AsciiIdentifier
    actor_projection_version: Annotated[int, Field(gt=0)]
    source_context_schema_id: Literal[
        "marl_battlegrounds.evaluation.episode_context"
    ] = CONTEXT_SCHEMA_ID
    source_context_schema_version: _SchemaVersionV1 = ACTOR_POV_SCHEMA_VERSION
    source_frame_schema_id: Literal["marl_battlegrounds.evaluation.frame"] = (
        FRAME_SCHEMA_ID
    )
    source_frame_schema_version: _SchemaVersionV1 = ACTOR_POV_SCHEMA_VERSION
    source_transition_schema_id: Literal["marl_battlegrounds.evaluation.transition"] = (
        TRANSITION_SCHEMA_ID
    )
    source_transition_schema_version: _SchemaVersionV1 = ACTOR_POV_SCHEMA_VERSION
    target_action_recipient_public_agent_id_by_id: tuple[_AsciiIdentifier | None, ...]
    ally_observation_row_public_agent_id_by_id: tuple[_AsciiIdentifier, ...]
    enemy_observation_row_public_agent_id_by_id: tuple[_AsciiIdentifier, ...]
    movement_action_name_by_id: tuple[_AsciiText, ...]
    unit_direction_vector_by_movement_action: _FloatMatrix
    target_action_name_by_id: tuple[_AsciiText, ...]
    use_ultimate_action_name_by_id: tuple[_AsciiText, ...]
    spawn_lifecycle_team_axis_name_by_id: tuple[_AsciiText, ...]

    @model_validator(mode="after")
    def _validate_axes(self) -> ActorPovAxisMappingV1:
        for field_name, length in (
            (
                "target_action_recipient_public_agent_id_by_id",
                NUM_TARGET_ACTIONS,
            ),
            (
                "ally_observation_row_public_agent_id_by_id",
                MAX_AGENTS_PER_TEAM,
            ),
            (
                "enemy_observation_row_public_agent_id_by_id",
                MAX_AGENTS_PER_TEAM,
            ),
            ("movement_action_name_by_id", NUM_MOVE_ACTIONS),
            ("target_action_name_by_id", NUM_TARGET_ACTIONS),
            ("use_ultimate_action_name_by_id", NUM_ULTIMATE_ACTIONS),
            ("spawn_lifecycle_team_axis_name_by_id", NUM_TEAMS),
        ):
            _require_tuple_shape(
                getattr(self, field_name),
                (length,),
                field_name=field_name,
            )
        _require_tuple_shape(
            self.unit_direction_vector_by_movement_action,
            (NUM_MOVE_ACTIONS, ENVIRONMENT_DIMENSIONS),
            field_name="unit_direction_vector_by_movement_action",
        )
        expected_targets = (
            None,
            *self.ally_observation_row_public_agent_id_by_id,
            *self.enemy_observation_row_public_agent_id_by_id,
        )
        if self.target_action_recipient_public_agent_id_by_id != expected_targets:
            raise ValueError(
                "POV target actions must align with ally/enemy observation rows"
            )
        all_relation_ids = (
            *self.ally_observation_row_public_agent_id_by_id,
            *self.enemy_observation_row_public_agent_id_by_id,
        )
        if len(set(all_relation_ids)) != MAX_AGENT_SLOTS:
            raise ValueError("POV relation axes must partition public agent IDs")
        for field_name in (
            "movement_action_name_by_id",
            "target_action_name_by_id",
            "use_ultimate_action_name_by_id",
        ):
            values = getattr(self, field_name)
            if len(set(values)) != len(values):
                raise ValueError(f"{field_name} entries must be unique")
        if self.spawn_lifecycle_team_axis_name_by_id != (
            "Own Team",
            "Opponent Team",
        ):
            raise ValueError("POV spawn-lifecycle axes must remain actor-relative")
        return self


class ActorPovFrameV1(EvaluationModel):
    """One recipient-sliced decision frame with no privileged snapshot."""

    schema_id: Literal["marl_battlegrounds.evaluation.actor_pov_frame"] = (
        ACTOR_POV_FRAME_SCHEMA_ID
    )
    schema_version: _SchemaVersionV1 = ACTOR_POV_SCHEMA_VERSION
    episode_id: _AsciiIdentifier
    public_agent_id: _AsciiIdentifier
    frame_index: _NonNegativeInt
    pov_frame_id: _AsciiIdentifier
    source_frame_id: _AsciiIdentifier
    simulator_step_count: _NonNegativeInt
    self_features: _FloatVector
    ally_unit_features: _FloatMatrix
    enemy_unit_features: _FloatMatrix
    map_obstacle_features: _FloatMatrix
    objective_features: _FloatMatrix
    context_features: _FloatVector
    ally_visibility_mask: _BooleanVector
    enemy_visibility_mask: _BooleanVector
    previous_timestep_actions: ActorPovPreviousTimestepActionsV1
    spawn_lifecycle: ActorPovSpawnLifecycleV1
    action_mask: ActorPovActionMaskV1

    @model_validator(mode="after")
    def _validate_frame(self) -> ActorPovFrameV1:
        expected_pov_id = (
            f"{self.episode_id}:actor-pov:{self.public_agent_id}:frame:"
            f"{self.frame_index}"
        )
        if self.pov_frame_id != expected_pov_id:
            raise ValueError("POV frame ID is not canonical")
        if self.source_frame_id != f"{self.episode_id}:frame:{self.frame_index}":
            raise ValueError("POV source frame ID is not canonical")
        for field_name, shape in (
            ("self_features", (SELF_FEATURES,)),
            ("ally_unit_features", (MAX_AGENTS_PER_TEAM, UNIT_FEATURES)),
            ("enemy_unit_features", (MAX_AGENTS_PER_TEAM, UNIT_FEATURES)),
            (
                "map_obstacle_features",
                (MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES),
            ),
            ("objective_features", (MAX_OBJECTIVE_SLOTS, OBJECTIVE_FEATURES)),
            ("context_features", (CONTEXT_FEATURES,)),
            ("ally_visibility_mask", (MAX_AGENTS_PER_TEAM,)),
            ("enemy_visibility_mask", (MAX_AGENTS_PER_TEAM,)),
        ):
            _require_tuple_shape(
                getattr(self, field_name),
                shape,
                field_name=field_name,
            )
        return self


class ActorPovSubmittedActionV1(EvaluationModel):
    """One selected actor's exact submitted int32 action heads."""

    schema_id: Literal["marl_battlegrounds.evaluation.actor_pov_submitted_action"] = (
        ACTOR_POV_SUBMITTED_ACTION_SCHEMA_ID
    )
    schema_version: _SchemaVersionV1 = ACTOR_POV_SCHEMA_VERSION
    move: _Int32
    select_target: _Int32
    use_ultimate: _Int32


class ActorPovAcceptedActionV1(EvaluationModel):
    """One selected actor's canonical category-bounded accepted action."""

    schema_id: Literal["marl_battlegrounds.evaluation.actor_pov_accepted_action"] = (
        ACTOR_POV_ACCEPTED_ACTION_SCHEMA_ID
    )
    schema_version: _SchemaVersionV1 = ACTOR_POV_SCHEMA_VERSION
    move: _MoveAction
    select_target: _TargetAction
    use_ultimate: _UltimateAction


class ActorPovCueBaseV1(EvaluationModel):
    """Common canonical identity carried by every local presentation cue."""

    schema_id: Literal["marl_battlegrounds.evaluation.actor_pov_cue"] = (
        ACTOR_POV_CUE_SCHEMA_ID
    )
    schema_version: _SchemaVersionV1 = ACTOR_POV_SCHEMA_VERSION
    cue_id: _AsciiIdentifier
    pov_transition_id: _AsciiIdentifier
    ordinal: _NonNegativeInt


class ActorPovOwnActionOutcomeCueV1(ActorPovCueBaseV1):
    cue_type: Literal["own_action_outcome"] = "own_action_outcome"
    outcome: Literal["accepted", "rejected"]


class ActorPovOwnPositionChangedCueV1(ActorPovCueBaseV1):
    cue_type: Literal["own_position_changed"] = "own_position_changed"
    start_position: tuple[_FiniteFloat, _FiniteFloat]
    successor_position: tuple[_FiniteFloat, _FiniteFloat]

    @model_validator(mode="after")
    def _validate_change(self) -> ActorPovOwnPositionChangedCueV1:
        if self.start_position == self.successor_position:
            raise ValueError("position-change cues require a changed position")
        return self


class ActorPovOwnHealthChangedCueV1(ActorPovCueBaseV1):
    cue_type: Literal["own_health_changed"] = "own_health_changed"
    start_health: _FiniteFloat
    successor_health: _FiniteFloat

    @model_validator(mode="after")
    def _validate_change(self) -> ActorPovOwnHealthChangedCueV1:
        if self.start_health == self.successor_health:
            raise ValueError("health-change cues require changed health")
        return self


class ActorPovOwnStatusChangedCueV1(ActorPovCueBaseV1):
    cue_type: Literal["own_status_changed"] = "own_status_changed"
    changed_feature_indices: tuple[
        Annotated[int, Field(ge=_STATUS_FEATURE_START, lt=_STATUS_FEATURE_STOP)],
        ...,
    ]
    start_values: _FloatVector
    successor_values: _FloatVector

    @model_validator(mode="after")
    def _validate_changes(self) -> ActorPovOwnStatusChangedCueV1:
        if not self.changed_feature_indices:
            raise ValueError("status-change cues require at least one feature")
        if self.changed_feature_indices != tuple(sorted(self.changed_feature_indices)):
            raise ValueError("status feature indices must be sorted")
        if len(set(self.changed_feature_indices)) != len(self.changed_feature_indices):
            raise ValueError("status feature indices must be unique")
        if not (
            len(self.changed_feature_indices)
            == len(self.start_values)
            == len(self.successor_values)
        ):
            raise ValueError("status feature/value tuples must align")
        if any(
            start == successor
            for start, successor in zip(
                self.start_values,
                self.successor_values,
                strict=True,
            )
        ):
            raise ValueError("status cues may contain only changed features")
        return self


class ActorPovOwnCooldownChangedCueV1(ActorPovCueBaseV1):
    cue_type: Literal["own_cooldown_changed"] = "own_cooldown_changed"
    start_remaining_ticks: _FiniteFloat
    successor_remaining_ticks: _FiniteFloat

    @model_validator(mode="after")
    def _validate_change(self) -> ActorPovOwnCooldownChangedCueV1:
        if self.start_remaining_ticks == self.successor_remaining_ticks:
            raise ValueError("cooldown-change cues require a changed countdown")
        return self


class ActorPovOwnLifecycleChangedCueV1(ActorPovCueBaseV1):
    cue_type: Literal["own_lifecycle_changed"] = "own_lifecycle_changed"
    start_active: bool
    successor_active: bool
    start_alive: bool
    successor_alive: bool
    start_spawn_shield_remaining_ticks: _NonNegativeInt
    successor_spawn_shield_remaining_ticks: _NonNegativeInt

    @model_validator(mode="after")
    def _validate_change(self) -> ActorPovOwnLifecycleChangedCueV1:
        if (
            self.start_active,
            self.start_alive,
            self.start_spawn_shield_remaining_ticks,
        ) == (
            self.successor_active,
            self.successor_alive,
            self.successor_spawn_shield_remaining_ticks,
        ):
            raise ValueError("lifecycle-change cues require changed own lifecycle")
        return self


class ActorPovVisibleBodyObservationChangedCueV1(ActorPovCueBaseV1):
    cue_type: Literal["visible_body_observation_changed"] = (
        "visible_body_observation_changed"
    )
    relation: Literal["ally", "enemy"]
    observation_row: Annotated[int, Field(ge=0, lt=MAX_AGENTS_PER_TEAM)]
    start_visible: bool
    successor_visible: bool
    observed_payload_changed: bool

    @model_validator(mode="after")
    def _validate_change(self) -> ActorPovVisibleBodyObservationChangedCueV1:
        if not (self.start_visible or self.successor_visible):
            raise ValueError("visible-body cues require visibility at one endpoint")
        if (
            self.start_visible == self.successor_visible
            and not self.observed_payload_changed
        ):
            raise ValueError("visible-body cues require visibility or payload change")
        return self


class ActorPovEpisodeEndedCueV1(ActorPovCueBaseV1):
    cue_type: Literal["episode_ended"] = "episode_ended"
    terminated: bool
    truncated: bool
    public_end_reason: _AsciiText | None = None

    @model_validator(mode="after")
    def _validate_done(self) -> ActorPovEpisodeEndedCueV1:
        if not (self.terminated or self.truncated):
            raise ValueError("episode-ended cues require a recorded done flag")
        return self


type ActorPovPresentationCueV1 = Annotated[
    ActorPovOwnActionOutcomeCueV1
    | ActorPovOwnPositionChangedCueV1
    | ActorPovOwnHealthChangedCueV1
    | ActorPovOwnStatusChangedCueV1
    | ActorPovOwnCooldownChangedCueV1
    | ActorPovOwnLifecycleChangedCueV1
    | ActorPovVisibleBodyObservationChangedCueV1
    | ActorPovEpisodeEndedCueV1,
    Field(discriminator="cue_type"),
]


class ActorPovTransitionV1(EvaluationModel):
    """One recipient-sliced transition with no privileged event feed."""

    schema_id: Literal["marl_battlegrounds.evaluation.actor_pov_transition"] = (
        ACTOR_POV_TRANSITION_SCHEMA_ID
    )
    schema_version: _SchemaVersionV1 = ACTOR_POV_SCHEMA_VERSION
    episode_id: _AsciiIdentifier
    public_agent_id: _AsciiIdentifier
    transition_index: _NonNegativeInt
    pov_transition_id: _AsciiIdentifier
    start_pov_frame_id: _AsciiIdentifier
    successor_pov_frame_id: _AsciiIdentifier
    submitted_action: ActorPovSubmittedActionV1
    accepted_action: ActorPovAcceptedActionV1
    submitted_action_tuple_is_out_of_domain: bool
    in_domain_move_action_is_rejected: bool
    in_domain_combat_action_pair_is_rejected: bool
    canonical_reward: _FiniteFloat
    terminated: bool
    truncated: bool
    public_end_reason: _AsciiText | None = None
    cues: tuple[ActorPovPresentationCueV1, ...]

    @model_validator(mode="after")
    def _validate_transition(self) -> ActorPovTransitionV1:
        prefix = (
            f"{self.episode_id}:actor-pov:{self.public_agent_id}:transition:"
            f"{self.transition_index}"
        )
        if self.pov_transition_id != prefix:
            raise ValueError("POV transition ID is not canonical")
        expected_start = (
            f"{self.episode_id}:actor-pov:{self.public_agent_id}:frame:"
            f"{self.transition_index}"
        )
        expected_successor = (
            f"{self.episode_id}:actor-pov:{self.public_agent_id}:frame:"
            f"{self.transition_index + 1}"
        )
        if self.start_pov_frame_id != expected_start:
            raise ValueError("POV transition start frame is not canonical")
        if self.successor_pov_frame_id != expected_successor:
            raise ValueError("POV transition successor frame is not canonical")
        if self.public_end_reason is not None and not (
            self.terminated or self.truncated
        ):
            raise ValueError("POV end reason is allowed only when done")
        for ordinal, cue in enumerate(self.cues):
            if cue.ordinal != ordinal:
                raise ValueError("POV cue ordinals must be gap-free and ordered")
            if cue.pov_transition_id != self.pov_transition_id:
                raise ValueError("POV cues must join their transition")
            if cue.cue_id != f"{self.pov_transition_id}:cue:{ordinal}":
                raise ValueError("POV cue ID is not canonical")
        return self


class ActorPovCurrentSliceV1(EvaluationModel):
    """One live recipient slice without a fabricated retained prefix.

    The selected frame is sufficient at artifact frame zero.  Every later
    frame carries exactly its incoming recipient-local transition, but never
    earlier frames, privileged events, completion claims, or replay provenance.

    This is a trusted in-memory projection, not a standalone persisted artifact:
    only :func:`build_actor_pov_current_slice_v1` can revalidate the authoritative
    coherent source records and rederive recipient-local cues.
    """

    schema_id: Literal["marl_battlegrounds.evaluation.actor_pov_current_slice"] = (
        ACTOR_POV_CURRENT_SLICE_SCHEMA_ID
    )
    schema_version: _SchemaVersionV1 = ACTOR_POV_SCHEMA_VERSION
    episode_id: _AsciiIdentifier
    selected_global_slot: _GlobalSlot
    selected_team_local_slot: _TeamLocalSlot
    public_agent_id: _AsciiIdentifier
    configured_team_id: _TeamId
    class_id: _ClassId
    observation_materialization: Literal["exact_no_shared_obs_actor_input"] = (
        "exact_no_shared_obs_actor_input"
    )
    axis_mapping: ActorPovAxisMappingV1
    frame: ActorPovFrameV1
    incoming_transition: ActorPovTransitionV1 | None = None

    @model_validator(mode="after")
    def _validate_current_slice(self) -> ActorPovCurrentSliceV1:
        if self.selected_team_local_slot != (
            self.selected_global_slot % MAX_AGENTS_PER_TEAM
        ):
            raise ValueError("POV team-local slot must follow the fixed team block")
        expected_team_id = 1 if self.selected_global_slot < MAX_AGENTS_PER_TEAM else 2
        if self.configured_team_id != expected_team_id:
            raise ValueError("POV configured team must follow the fixed slot block")
        if (
            self.axis_mapping.ally_observation_row_public_agent_id_by_id[
                self.selected_team_local_slot
            ]
            != self.public_agent_id
        ):
            raise ValueError("POV ally axis must place the selected public agent")
        frame = self.frame
        if (
            frame.episode_id != self.episode_id
            or frame.public_agent_id != self.public_agent_id
        ):
            raise ValueError("POV current frame must join the selected identity")
        if frame.self_features[_FEATURE_ACTIVE] != 1.0:
            raise ValueError("configured-active POV self rows require ACTIVE=1")
        if frame.self_features[_FEATURE_ALIVE] not in (0.0, 1.0):
            raise ValueError("POV self ALIVE must be exactly zero or one")
        if frame.self_features[_FEATURE_TEAM_ID] != float(self.configured_team_id):
            raise ValueError("POV self team feature must match current metadata")
        if frame.self_features[_FEATURE_CLASS_ID] != float(self.class_id):
            raise ValueError("POV self class feature must match current metadata")
        incoming = self.incoming_transition
        if frame.frame_index == 0:
            if incoming is not None:
                raise ValueError("POV frame zero cannot have an incoming transition")
            return self
        if incoming is None:
            raise ValueError("non-initial POV frames require their incoming transition")
        if (
            incoming.episode_id != self.episode_id
            or incoming.public_agent_id != self.public_agent_id
            or incoming.transition_index != frame.frame_index - 1
            or incoming.successor_pov_frame_id != frame.pov_frame_id
        ):
            raise ValueError("incoming POV transition must enter the current frame")
        return self


def _adjacent_cue_endpoint(frame: ActorPovFrameV1) -> ActorPovFrameV1:
    """Zero masked relation rows before carrier-local cue derivation."""
    zero_row = (0.0,) * UNIT_FEATURES
    ally_rows = tuple(
        row if visible else zero_row
        for row, visible in zip(
            frame.ally_unit_features,
            frame.ally_visibility_mask,
            strict=True,
        )
    )
    enemy_rows = tuple(
        row if visible else zero_row
        for row, visible in zip(
            frame.enemy_unit_features,
            frame.enemy_visibility_mask,
            strict=True,
        )
    )
    return frame.model_copy(
        update={
            "ally_unit_features": ally_rows,
            "enemy_unit_features": enemy_rows,
        }
    )


class ActorPovAdjacentTransitionSliceV1(EvaluationModel):
    """One live recipient transition with both exact authorized endpoints.

    The carrier is an in-memory evaluation-to-presentation seam, not a
    persisted artifact.  It retains only the chosen recipient's recorded
    actor-input frames, recipient-local transition, and actor-relative axes;
    Oracle events and source-evidence roots are deliberately absent.
    """

    schema_id: Literal[
        "marl_battlegrounds.evaluation.actor_pov_adjacent_transition_slice"
    ] = ACTOR_POV_ADJACENT_TRANSITION_SLICE_SCHEMA_ID
    schema_version: _SchemaVersionV1 = ACTOR_POV_SCHEMA_VERSION
    episode_id: _AsciiIdentifier
    selected_global_slot: _GlobalSlot
    selected_team_local_slot: _TeamLocalSlot
    public_agent_id: _AsciiIdentifier
    configured_team_id: _TeamId
    class_id: _ClassId
    observation_materialization: Literal["exact_no_shared_obs_actor_input"] = (
        "exact_no_shared_obs_actor_input"
    )
    axis_mapping: ActorPovAxisMappingV1
    start_frame: ActorPovFrameV1
    transition: ActorPovTransitionV1
    successor_frame: ActorPovFrameV1

    @model_validator(mode="after")
    def _validate_adjacent_slice(self) -> ActorPovAdjacentTransitionSliceV1:
        expected_local_slot = self.selected_global_slot % MAX_AGENTS_PER_TEAM
        expected_team_id = 1 if self.selected_global_slot < MAX_AGENTS_PER_TEAM else 2
        if self.selected_team_local_slot != expected_local_slot:
            raise ValueError("POV team-local slot must follow the fixed team block")
        if self.configured_team_id != expected_team_id:
            raise ValueError("POV configured team must follow the fixed slot block")
        if (
            self.axis_mapping.ally_observation_row_public_agent_id_by_id[
                self.selected_team_local_slot
            ]
            != self.public_agent_id
        ):
            raise ValueError("POV ally axis must place the selected public agent")

        start = self.start_frame
        transition = self.transition
        successor = self.successor_frame
        for endpoint_name, endpoint in (
            ("start", start),
            ("successor", successor),
        ):
            if (
                endpoint.episode_id != self.episode_id
                or endpoint.public_agent_id != self.public_agent_id
            ):
                raise ValueError(
                    f"POV {endpoint_name} frame must join the selected identity"
                )
            _require_selected_self_topology(
                endpoint,
                configured_team_id=self.configured_team_id,
                class_id=self.class_id,
            )
            self_diagonal_is_visible = endpoint.ally_visibility_mask[
                self.selected_team_local_slot
            ]
            if (
                self_diagonal_is_visible
                and endpoint.ally_unit_features[self.selected_team_local_slot]
                != endpoint.self_features
            ):
                raise ValueError(
                    f"POV {endpoint_name} visible self row must join its ally axis slot"
                )
            lifecycle = endpoint.spawn_lifecycle
            if not lifecycle.active_mask_by_team[0][
                self.selected_team_local_slot
            ] or lifecycle.alive_mask_by_team[0][self.selected_team_local_slot] != (
                endpoint.self_features[_FEATURE_ALIVE] == 1.0
            ):
                raise ValueError(
                    f"POV {endpoint_name} self row must join its lifecycle slot"
                )

        if (
            transition.episode_id != self.episode_id
            or transition.public_agent_id != self.public_agent_id
        ):
            raise ValueError("POV adjacent transition must join selected identity")
        if (
            transition.transition_index != start.frame_index
            or successor.frame_index != start.frame_index + 1
            or transition.start_pov_frame_id != start.pov_frame_id
            or transition.successor_pov_frame_id != successor.pov_frame_id
        ):
            raise ValueError("POV adjacent transition epochs do not join")
        if successor.simulator_step_count != start.simulator_step_count + 1:
            raise ValueError("POV adjacent endpoint simulator ticks are not adjacent")
        cue_start = _adjacent_cue_endpoint(start)
        cue_successor = _adjacent_cue_endpoint(successor)
        expected_cues = _derive_cues(
            episode_id=self.episode_id,
            public_agent_id=self.public_agent_id,
            transition_index=transition.transition_index,
            team_local_slot=self.selected_team_local_slot,
            start_frame=cue_start,
            successor_frame=cue_successor,
            has_any_rejection=(
                transition.submitted_action_tuple_is_out_of_domain
                or transition.in_domain_move_action_is_rejected
                or transition.in_domain_combat_action_pair_is_rejected
            ),
            terminated=transition.terminated,
            truncated=transition.truncated,
            public_end_reason=transition.public_end_reason,
        )
        if transition.cues != expected_cues:
            raise ValueError(
                "POV adjacent cues must be derived exactly from their endpoints"
            )
        return self


class ActorPovReplayContentV1(EvaluationModel):
    """Canonical recipient-authorized content, independent of source provenance."""

    schema_id: Literal["marl_battlegrounds.evaluation.actor_pov_content"] = (
        ACTOR_POV_CONTENT_SCHEMA_ID
    )
    schema_version: _SchemaVersionV1 = ACTOR_POV_SCHEMA_VERSION
    content_id: _AsciiIdentifier
    canonical_digest_sha256: _Sha256Hex
    episode_id: _AsciiIdentifier
    selected_global_slot: _GlobalSlot
    selected_team_local_slot: _TeamLocalSlot
    public_agent_id: _AsciiIdentifier
    configured_team_id: _TeamId
    class_id: _ClassId
    observation_materialization: Literal["exact_no_shared_obs_actor_input"] = (
        "exact_no_shared_obs_actor_input"
    )
    axis_mapping: ActorPovAxisMappingV1
    completion: ActorPovEpisodeCompletionV1
    frames: tuple[ActorPovFrameV1, ...]
    transitions: tuple[ActorPovTransitionV1, ...]

    @model_validator(mode="after")
    def _validate_content(self) -> ActorPovReplayContentV1:
        if self.content_id != (
            f"{self.episode_id}:actor-pov:{self.public_agent_id}:content"
        ):
            raise ValueError("POV content ID is not canonical")
        if not self.frames or len(self.frames) != len(self.transitions) + 1:
            raise ValueError("POV content requires exact T+1/T frame structure")
        expected_team_local_slot = self.selected_global_slot % MAX_AGENTS_PER_TEAM
        expected_team_id = 1 if self.selected_global_slot < MAX_AGENTS_PER_TEAM else 2
        if self.selected_team_local_slot != expected_team_local_slot:
            raise ValueError("POV team-local slot must follow the fixed team block")
        if self.configured_team_id != expected_team_id:
            raise ValueError("POV configured team must follow the fixed slot block")
        if (
            self.axis_mapping.ally_observation_row_public_agent_id_by_id[
                self.selected_team_local_slot
            ]
            != self.public_agent_id
        ):
            raise ValueError("POV ally axis must place the selected public agent")
        if self.completion.captured_transition_count != len(self.transitions):
            raise ValueError("POV completion count must equal transition count")
        for frame_index, frame in enumerate(self.frames):
            if frame.frame_index != frame_index:
                raise ValueError("POV frame positions must equal frame indices")
            if (
                frame.episode_id != self.episode_id
                or frame.public_agent_id != self.public_agent_id
            ):
                raise ValueError("POV frames must join content identity")
            if frame.self_features[_FEATURE_ACTIVE] != 1.0:
                raise ValueError("configured-active POV self rows require ACTIVE=1")
            if frame.self_features[_FEATURE_ALIVE] not in (0.0, 1.0):
                raise ValueError("POV self ALIVE must be exactly zero or one")
            if frame.self_features[_FEATURE_TEAM_ID] != float(self.configured_team_id):
                raise ValueError("POV self team feature must match content metadata")
            if frame.self_features[_FEATURE_CLASS_ID] != float(self.class_id):
                raise ValueError("POV self class feature must match content metadata")
            if frame_index > 0 and frame.simulator_step_count != (
                self.frames[frame_index - 1].simulator_step_count + 1
            ):
                raise ValueError("POV simulator epochs must be adjacent")
        for transition_index, transition in enumerate(self.transitions):
            if transition.transition_index != transition_index:
                raise ValueError(
                    "POV transition positions must equal transition indices"
                )
            if (
                transition.episode_id != self.episode_id
                or transition.public_agent_id != self.public_agent_id
            ):
                raise ValueError("POV transitions must join content identity")
            if (
                transition.start_pov_frame_id
                != self.frames[transition_index].pov_frame_id
            ):
                raise ValueError("POV transition must join its stored start frame")
            if (
                transition.successor_pov_frame_id
                != self.frames[transition_index + 1].pov_frame_id
            ):
                raise ValueError("POV transition must join its stored successor frame")
            if transition_index > 0:
                previous = self.transitions[transition_index - 1]
                if previous.terminated or previous.truncated:
                    raise ValueError("POV content cannot continue after done")
        tail = self.transitions[-1] if self.transitions else None
        terminated = False if tail is None else tail.terminated
        truncated = False if tail is None else tail.truncated
        if (
            self.completion.terminated != terminated
            or self.completion.truncated != truncated
        ):
            raise ValueError("POV completion done flags must equal its tail")
        if (
            tail is not None
            and tail.public_end_reason is not None
            and self.completion.public_end_or_failure_reason != tail.public_end_reason
        ):
            raise ValueError("POV tail end reason must agree with completion truth")
        if self.canonical_digest_sha256 != canonical_digest_sha256(
            self,
            exclude={"canonical_digest_sha256"},
        ):
            raise ValueError("POV content digest is not canonical")
        return self


class ActorPovReplayArtifactV1(EvaluationModel):
    """Recipient content wrapped in truthful full-replay provenance."""

    schema_id: Literal["marl_battlegrounds.evaluation.actor_pov_artifact"] = (
        ACTOR_POV_ARTIFACT_SCHEMA_ID
    )
    schema_version: _SchemaVersionV1 = ACTOR_POV_SCHEMA_VERSION
    artifact_id: _AsciiIdentifier
    canonical_digest_sha256: _Sha256Hex
    source_replay: ReplayArtifactReferenceV1
    content: ActorPovReplayContentV1

    @model_validator(mode="after")
    def _validate_artifact(self) -> ActorPovReplayArtifactV1:
        if self.artifact_id != (
            f"{self.content.episode_id}:actor-pov:{self.content.public_agent_id}"
        ):
            raise ValueError("POV artifact ID is not canonical")
        if self.source_replay.episode_id != self.content.episode_id:
            raise ValueError("POV content must join source replay episode")
        if self.canonical_digest_sha256 != canonical_digest_sha256(
            self,
            exclude={"canonical_digest_sha256"},
        ):
            raise ValueError("POV artifact digest is not canonical")
        return self


def _build_replay_reference_from_validated(
    replay: ReplayArtifactV1,
) -> ReplayArtifactReferenceV1:
    return ReplayArtifactReferenceV1(
        artifact_id=replay.artifact_id,
        episode_id=replay.header.context.identity.episode_id,
        context_digest_sha256=replay.header.context_digest_sha256,
        trajectory_content_digest_sha256=replay.trajectory_content_digest_sha256,
        canonical_digest_sha256=replay.canonical_digest_sha256,
        canonical_byte_length=len(canonical_json_bytes(replay)),
    )


def _slice_frame_from_source(
    source: EvaluationFrameV1,
    *,
    global_slot: int,
    public_agent_id: str,
) -> ActorPovFrameV1:
    observation = source.base_observation
    previous = observation.previous_timestep_actions
    lifecycle = observation.spawn_lifecycle
    mask = source.action_mask
    return ActorPovFrameV1(
        episode_id=source.episode_id,
        public_agent_id=public_agent_id,
        frame_index=source.frame_index,
        pov_frame_id=(
            f"{source.episode_id}:actor-pov:{public_agent_id}:frame:"
            f"{source.frame_index}"
        ),
        source_frame_id=source.frame_id,
        simulator_step_count=source.simulator_step_count,
        self_features=observation.self_features[global_slot],
        ally_unit_features=observation.ally_unit_features[global_slot],
        enemy_unit_features=observation.enemy_unit_features[global_slot],
        map_obstacle_features=observation.map_obstacle_features[global_slot],
        objective_features=observation.objective_features[global_slot],
        context_features=observation.context_features[global_slot],
        ally_visibility_mask=observation.ally_visibility_mask[global_slot],
        enemy_visibility_mask=observation.enemy_visibility_mask[global_slot],
        previous_timestep_actions=ActorPovPreviousTimestepActionsV1(
            ally_move_actions_one_hot=(
                previous.ally_previous_timestep_move_actions_one_hot[global_slot]
            ),
            enemy_move_actions_one_hot=(
                previous.enemy_previous_timestep_move_actions_one_hot[global_slot]
            ),
            ally_select_target_actions_one_hot=(
                previous.ally_previous_timestep_select_target_actions_one_hot[
                    global_slot
                ]
            ),
            enemy_select_target_actions_one_hot=(
                previous.enemy_previous_timestep_select_target_actions_one_hot[
                    global_slot
                ]
            ),
            ally_use_ultimate_actions_one_hot=(
                previous.ally_previous_timestep_use_ultimate_actions_one_hot[
                    global_slot
                ]
            ),
            enemy_use_ultimate_actions_one_hot=(
                previous.enemy_previous_timestep_use_ultimate_actions_one_hot[
                    global_slot
                ]
            ),
        ),
        spawn_lifecycle=ActorPovSpawnLifecycleV1(
            spawn_pad_positions_by_team=(
                lifecycle.spawn_pad_positions_by_agent_by_team[global_slot]
            ),
            spawn_shield_actual_durations_by_team=(
                lifecycle.spawn_shield_actual_durations_by_agent_by_team[global_slot]
            ),
            spawn_shield_configured_duration=(
                lifecycle.spawn_shield_configured_duration_by_agent[global_slot]
            ),
            spawn_shield_speed=(lifecycle.spawn_shield_speed_by_agent[global_slot]),
            respawn_wave_period_step_count_by_team=(
                lifecycle.respawn_wave_period_step_count_by_agent_by_team[global_slot]
            ),
            respawn_wave_countdowns_by_team=(
                lifecycle.respawn_wave_countdowns_by_agent_by_team[global_slot]
            ),
            active_mask_by_team=(lifecycle.active_mask_by_agent_by_team[global_slot]),
            alive_mask_by_team=(lifecycle.alive_mask_by_agent_by_team[global_slot]),
        ),
        action_mask=ActorPovActionMaskV1(
            move=mask.move_mask[global_slot],
            select_target=mask.select_target_mask[global_slot],
            use_ultimate=mask.use_ultimate_mask[global_slot],
            select_target_use_ultimate_joint=(
                mask.select_target_use_ultimate_joint_mask[global_slot]
            ),
        ),
    )


def _slice_frame(
    replay: ReplayArtifactV1,
    *,
    global_slot: int,
    public_agent_id: str,
    frame_index: int,
) -> ActorPovFrameV1:
    return _slice_frame_from_source(
        replay.frames[frame_index],
        global_slot=global_slot,
        public_agent_id=public_agent_id,
    )


class _CueIdentity(TypedDict):
    cue_id: str
    pov_transition_id: str
    ordinal: int


def _cue_identity(
    transition: ActorPovTransitionV1 | None,
    *,
    episode_id: str,
    public_agent_id: str,
    transition_index: int,
    ordinal: int,
) -> _CueIdentity:
    transition_id = (
        transition.pov_transition_id
        if transition is not None
        else (f"{episode_id}:actor-pov:{public_agent_id}:transition:{transition_index}")
    )
    return {
        "cue_id": f"{transition_id}:cue:{ordinal}",
        "pov_transition_id": transition_id,
        "ordinal": ordinal,
    }


def _derive_cues(
    *,
    episode_id: str,
    public_agent_id: str,
    transition_index: int,
    team_local_slot: int,
    start_frame: ActorPovFrameV1,
    successor_frame: ActorPovFrameV1,
    has_any_rejection: bool,
    terminated: bool,
    truncated: bool,
    public_end_reason: str | None,
) -> tuple[ActorPovPresentationCueV1, ...]:
    cues: list[ActorPovPresentationCueV1] = []

    def identity() -> _CueIdentity:
        return _cue_identity(
            None,
            episode_id=episode_id,
            public_agent_id=public_agent_id,
            transition_index=transition_index,
            ordinal=len(cues),
        )

    cues.append(
        ActorPovOwnActionOutcomeCueV1(
            **identity(),
            outcome="rejected" if has_any_rejection else "accepted",
        )
    )
    start_self = start_frame.self_features
    successor_self = successor_frame.self_features
    start_position = (start_self[_FEATURE_X], start_self[_FEATURE_Y])
    successor_position = (
        successor_self[_FEATURE_X],
        successor_self[_FEATURE_Y],
    )
    if start_position != successor_position:
        cues.append(
            ActorPovOwnPositionChangedCueV1(
                **identity(),
                start_position=start_position,
                successor_position=successor_position,
            )
        )
    if start_self[_FEATURE_CURRENT_HEALTH] != successor_self[_FEATURE_CURRENT_HEALTH]:
        cues.append(
            ActorPovOwnHealthChangedCueV1(
                **identity(),
                start_health=start_self[_FEATURE_CURRENT_HEALTH],
                successor_health=successor_self[_FEATURE_CURRENT_HEALTH],
            )
        )
    changed_status_indices = tuple(
        index
        for index in range(_STATUS_FEATURE_START, _STATUS_FEATURE_STOP)
        if start_self[index] != successor_self[index]
    )
    if changed_status_indices:
        cues.append(
            ActorPovOwnStatusChangedCueV1(
                **identity(),
                changed_feature_indices=changed_status_indices,
                start_values=tuple(
                    start_self[index] for index in changed_status_indices
                ),
                successor_values=tuple(
                    successor_self[index] for index in changed_status_indices
                ),
            )
        )
    if (
        start_self[_FEATURE_ULTIMATE_COOLDOWN]
        != successor_self[_FEATURE_ULTIMATE_COOLDOWN]
    ):
        cues.append(
            ActorPovOwnCooldownChangedCueV1(
                **identity(),
                start_remaining_ticks=start_self[_FEATURE_ULTIMATE_COOLDOWN],
                successor_remaining_ticks=(successor_self[_FEATURE_ULTIMATE_COOLDOWN]),
            )
        )
    start_shield = start_frame.spawn_lifecycle.spawn_shield_actual_durations_by_team[0][
        team_local_slot
    ]
    successor_shield = (
        successor_frame.spawn_lifecycle.spawn_shield_actual_durations_by_team[0][
            team_local_slot
        ]
    )
    start_lifecycle = (
        start_self[_FEATURE_ACTIVE] == 1.0,
        start_self[_FEATURE_ALIVE] == 1.0,
        start_shield,
    )
    successor_lifecycle = (
        successor_self[_FEATURE_ACTIVE] == 1.0,
        successor_self[_FEATURE_ALIVE] == 1.0,
        successor_shield,
    )
    if start_lifecycle != successor_lifecycle:
        cues.append(
            ActorPovOwnLifecycleChangedCueV1(
                **identity(),
                start_active=start_lifecycle[0],
                successor_active=successor_lifecycle[0],
                start_alive=start_lifecycle[1],
                successor_alive=successor_lifecycle[1],
                start_spawn_shield_remaining_ticks=start_lifecycle[2],
                successor_spawn_shield_remaining_ticks=successor_lifecycle[2],
            )
        )
    for relation in ("ally", "enemy"):
        start_rows = getattr(start_frame, f"{relation}_unit_features")
        successor_rows = getattr(successor_frame, f"{relation}_unit_features")
        start_visibility = getattr(start_frame, f"{relation}_visibility_mask")
        successor_visibility = getattr(
            successor_frame,
            f"{relation}_visibility_mask",
        )
        for row in range(MAX_AGENTS_PER_TEAM):
            was_visible = start_visibility[row]
            is_visible = successor_visibility[row]
            payload_changed = start_rows[row] != successor_rows[row]
            if (was_visible or is_visible) and (
                was_visible != is_visible or payload_changed
            ):
                cues.append(
                    ActorPovVisibleBodyObservationChangedCueV1(
                        **identity(),
                        relation=relation,
                        observation_row=row,
                        start_visible=was_visible,
                        successor_visible=is_visible,
                        observed_payload_changed=payload_changed,
                    )
                )
    if terminated or truncated:
        cues.append(
            ActorPovEpisodeEndedCueV1(
                **identity(),
                terminated=terminated,
                truncated=truncated,
                public_end_reason=public_end_reason,
            )
        )
    return tuple(cues)


def _slice_transition_from_source(
    source: EvaluationTransitionV1,
    *,
    global_slot: int,
    team_local_slot: int,
    public_agent_id: str,
    start_frame: ActorPovFrameV1,
    successor_frame: ActorPovFrameV1,
) -> ActorPovTransitionV1:
    acceptance = source.facts.action_acceptance_facts
    submitted = acceptance.submitted_joint_action
    accepted = acceptance.accepted_joint_action
    out_of_domain = acceptance.submitted_action_tuple_is_out_of_domain_by_actor[
        global_slot
    ]
    move_rejected = acceptance.in_domain_move_action_is_rejected_by_actor[global_slot]
    combat_rejected = acceptance.in_domain_combat_action_pair_is_rejected_by_actor[
        global_slot
    ]
    transition_id = (
        f"{source.episode_id}:actor-pov:{public_agent_id}:transition:"
        f"{source.transition_index}"
    )
    return ActorPovTransitionV1(
        episode_id=source.episode_id,
        public_agent_id=public_agent_id,
        transition_index=source.transition_index,
        pov_transition_id=transition_id,
        start_pov_frame_id=start_frame.pov_frame_id,
        successor_pov_frame_id=successor_frame.pov_frame_id,
        submitted_action=ActorPovSubmittedActionV1(
            move=submitted.move[global_slot],
            select_target=submitted.select_target[global_slot],
            use_ultimate=submitted.use_ultimate[global_slot],
        ),
        accepted_action=ActorPovAcceptedActionV1(
            move=accepted.move[global_slot],
            select_target=accepted.select_target[global_slot],
            use_ultimate=accepted.use_ultimate[global_slot],
        ),
        submitted_action_tuple_is_out_of_domain=out_of_domain,
        in_domain_move_action_is_rejected=move_rejected,
        in_domain_combat_action_pair_is_rejected=combat_rejected,
        canonical_reward=source.canonical_reward_by_agent[global_slot],
        terminated=source.terminated,
        truncated=source.truncated,
        public_end_reason=source.owning_task_end_reason,
        cues=_derive_cues(
            episode_id=source.episode_id,
            public_agent_id=public_agent_id,
            transition_index=source.transition_index,
            team_local_slot=team_local_slot,
            start_frame=start_frame,
            successor_frame=successor_frame,
            has_any_rejection=(out_of_domain or move_rejected or combat_rejected),
            terminated=source.terminated,
            truncated=source.truncated,
            public_end_reason=source.owning_task_end_reason,
        ),
    )


def _slice_transition(
    replay: ReplayArtifactV1,
    *,
    global_slot: int,
    team_local_slot: int,
    public_agent_id: str,
    transition_index: int,
    frames: tuple[ActorPovFrameV1, ...],
) -> ActorPovTransitionV1:
    return _slice_transition_from_source(
        replay.transitions[transition_index],
        global_slot=global_slot,
        team_local_slot=team_local_slot,
        public_agent_id=public_agent_id,
        start_frame=frames[transition_index],
        successor_frame=frames[transition_index + 1],
    )


def _axis_mapping_from_context(
    context: EvaluationEpisodeContextV1,
    *,
    global_slot: int,
) -> ActorPovAxisMappingV1:
    catalog = context.static_mechanics_catalog
    ally_slots = catalog.global_slot_by_actor_and_ally_observation_row[global_slot]
    enemy_slots = catalog.global_slot_by_actor_and_enemy_observation_row[global_slot]
    target_slots = catalog.global_recipient_slot_by_actor_and_target_action[global_slot]

    def public_id(slot: int) -> str:
        return context.roster[slot].public_agent_id

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


def _axis_mapping_from_replay(
    replay: ReplayArtifactV1,
    *,
    global_slot: int,
) -> ActorPovAxisMappingV1:
    return _axis_mapping_from_context(
        replay.header.context,
        global_slot=global_slot,
    )


def _require_selected_self_topology(
    frame: ActorPovFrameV1,
    *,
    configured_team_id: int,
    class_id: int,
) -> None:
    row = frame.self_features
    if row[_FEATURE_ACTIVE] != 1.0:
        raise ValueError("configured-active POV self rows require ACTIVE=1")
    if row[_FEATURE_ALIVE] not in (0.0, 1.0):
        raise ValueError("POV self ALIVE must be exactly zero or one")
    if row[_FEATURE_TEAM_ID] != float(configured_team_id):
        raise ValueError("POV self team feature must match selected roster metadata")
    if row[_FEATURE_CLASS_ID] != float(class_id):
        raise ValueError("POV self class feature must match selected roster metadata")


def _selected_pov_roster_row(
    context: EvaluationEpisodeContextV1,
    *,
    global_slot: int,
) -> RosterSlotV1:
    if type(global_slot) is not int or not 0 <= global_slot < MAX_AGENT_SLOTS:
        raise ValueError("actor POV global_slot must be an exact bounded integer")
    if context.execution_information_mode != "no_shared_obs":
        raise ValueError(
            "exact actor POV slicing is unavailable for shared_obs source material"
        )
    roster = context.roster[global_slot]
    if not roster.configured_active:
        raise ValueError("actor POV slicing requires a configured-active actor")
    return roster


def build_actor_pov_current_slice_v1(
    context: EvaluationEpisodeContextV1,
    frame: EvaluationFrameV1,
    *,
    global_slot: int,
    incoming_transition_view: EvaluationTransitionViewV1 | None = None,
) -> ActorPovCurrentSliceV1:
    """Build one exact live POV slice without retaining earlier trajectory units."""
    canonical_context = cast(
        EvaluationEpisodeContextV1,
        validate_declared_model_tree(
            context,
            record_name="actor POV current context",
            expected_type=EvaluationEpisodeContextV1,
        ),
    )
    roster = _selected_pov_roster_row(
        canonical_context,
        global_slot=global_slot,
    )
    public_agent_id = roster.public_agent_id
    incoming: ActorPovTransitionV1 | None = None
    if incoming_transition_view is None:
        validate_initial_evaluation_frame_v1(canonical_context, frame)
        canonical_frame = frame
    else:
        if type(incoming_transition_view) is not EvaluationTransitionViewV1:
            raise TypeError(
                "incoming_transition_view must be the exact "
                "EvaluationTransitionViewV1 or None"
            )
        canonical_view = EvaluationTransitionViewV1(
            context=incoming_transition_view.context,
            start_frame=incoming_transition_view.start_frame,
            transition=incoming_transition_view.transition,
            successor_frame=incoming_transition_view.successor_frame,
        )
        if canonical_view.context != canonical_context:
            raise ValueError("incoming transition view must use the selected context")
        if canonical_view.successor_frame != frame:
            raise ValueError("incoming transition view must enter the selected frame")
        start_slice = _slice_frame_from_source(
            canonical_view.start_frame,
            global_slot=global_slot,
            public_agent_id=public_agent_id,
        )
        _require_selected_self_topology(
            start_slice,
            configured_team_id=roster.configured_team_id,
            class_id=roster.class_id,
        )
        canonical_frame = canonical_view.successor_frame
        successor_slice = _slice_frame_from_source(
            canonical_frame,
            global_slot=global_slot,
            public_agent_id=public_agent_id,
        )
        _require_selected_self_topology(
            successor_slice,
            configured_team_id=roster.configured_team_id,
            class_id=roster.class_id,
        )
        incoming = _slice_transition_from_source(
            canonical_view.transition,
            global_slot=global_slot,
            team_local_slot=roster.team_local_slot,
            public_agent_id=public_agent_id,
            start_frame=start_slice,
            successor_frame=successor_slice,
        )

    current_frame = _slice_frame_from_source(
        canonical_frame,
        global_slot=global_slot,
        public_agent_id=public_agent_id,
    )
    _require_selected_self_topology(
        current_frame,
        configured_team_id=roster.configured_team_id,
        class_id=roster.class_id,
    )
    return ActorPovCurrentSliceV1(
        episode_id=canonical_context.identity.episode_id,
        selected_global_slot=global_slot,
        selected_team_local_slot=roster.team_local_slot,
        public_agent_id=public_agent_id,
        configured_team_id=roster.configured_team_id,
        class_id=roster.class_id,
        axis_mapping=_axis_mapping_from_context(
            canonical_context,
            global_slot=global_slot,
        ),
        frame=current_frame,
        incoming_transition=incoming,
    )


def build_actor_pov_adjacent_transition_slice_v1(
    transition_view: EvaluationTransitionViewV1,
    *,
    global_slot: int,
) -> ActorPovAdjacentTransitionSliceV1:
    """Build one exact live recipient transition from a coherent CP2 view."""
    if type(transition_view) is not EvaluationTransitionViewV1:
        raise TypeError("transition_view must be the exact EvaluationTransitionViewV1")
    canonical_view = EvaluationTransitionViewV1(
        context=transition_view.context,
        start_frame=transition_view.start_frame,
        transition=transition_view.transition,
        successor_frame=transition_view.successor_frame,
    )
    roster = _selected_pov_roster_row(
        canonical_view.context,
        global_slot=global_slot,
    )
    public_agent_id = roster.public_agent_id
    start = _slice_frame_from_source(
        canonical_view.start_frame,
        global_slot=global_slot,
        public_agent_id=public_agent_id,
    )
    successor = _slice_frame_from_source(
        canonical_view.successor_frame,
        global_slot=global_slot,
        public_agent_id=public_agent_id,
    )
    _require_selected_self_topology(
        start,
        configured_team_id=roster.configured_team_id,
        class_id=roster.class_id,
    )
    _require_selected_self_topology(
        successor,
        configured_team_id=roster.configured_team_id,
        class_id=roster.class_id,
    )
    transition = _slice_transition_from_source(
        canonical_view.transition,
        global_slot=global_slot,
        team_local_slot=roster.team_local_slot,
        public_agent_id=public_agent_id,
        start_frame=_adjacent_cue_endpoint(start),
        successor_frame=_adjacent_cue_endpoint(successor),
    )
    return ActorPovAdjacentTransitionSliceV1(
        episode_id=canonical_view.context.identity.episode_id,
        selected_global_slot=global_slot,
        selected_team_local_slot=roster.team_local_slot,
        public_agent_id=public_agent_id,
        configured_team_id=roster.configured_team_id,
        class_id=roster.class_id,
        axis_mapping=_axis_mapping_from_context(
            canonical_view.context,
            global_slot=global_slot,
        ),
        start_frame=start,
        transition=transition,
        successor_frame=successor,
    )


def slice_actor_pov_current_frame_v1(
    context: EvaluationEpisodeContextV1,
    frame: EvaluationFrameV1,
    *,
    global_slot: int,
    incoming_transition_view: EvaluationTransitionViewV1 | None = None,
) -> ActorPovFrameV1:
    """Return only the recipient-authorized current frame from the live seam."""
    return build_actor_pov_current_slice_v1(
        context,
        frame,
        global_slot=global_slot,
        incoming_transition_view=incoming_transition_view,
    ).frame


def slice_actor_pov_current_transition_v1(
    transition_view: EvaluationTransitionViewV1,
    *,
    global_slot: int,
) -> ActorPovTransitionV1:
    """Return one recipient-local transition from an exact coherent CP2 view."""
    current = build_actor_pov_current_slice_v1(
        transition_view.context,
        transition_view.successor_frame,
        global_slot=global_slot,
        incoming_transition_view=transition_view,
    )
    if current.incoming_transition is None:
        raise AssertionError("coherent transition slicing must produce an incoming row")
    return current.incoming_transition


def _export_from_validated_replay(
    replay: ReplayArtifactV1,
    *,
    global_slot: int,
) -> ActorPovReplayArtifactV1:
    context = replay.header.context
    if context.execution_information_mode != "no_shared_obs":
        raise ValueError(
            "exact actor POV export is unavailable for shared_obs source material"
        )
    if type(global_slot) is not int or not 0 <= global_slot < MAX_AGENT_SLOTS:
        raise ValueError("actor POV global_slot must be an exact bounded integer")
    roster = context.roster[global_slot]
    if not roster.configured_active:
        raise ValueError("actor POV export requires a configured-active actor")
    episode_id = context.identity.episode_id
    public_agent_id = roster.public_agent_id
    frames = tuple(
        _slice_frame(
            replay,
            global_slot=global_slot,
            public_agent_id=public_agent_id,
            frame_index=frame_index,
        )
        for frame_index in range(len(replay.frames))
    )
    for frame in frames:
        _require_selected_self_topology(
            frame,
            configured_team_id=roster.configured_team_id,
            class_id=roster.class_id,
        )
    transitions = tuple(
        _slice_transition(
            replay,
            global_slot=global_slot,
            team_local_slot=roster.team_local_slot,
            public_agent_id=public_agent_id,
            transition_index=transition_index,
            frames=frames,
        )
        for transition_index in range(len(replay.transitions))
    )
    source_completion = replay.completion
    completion = ActorPovEpisodeCompletionV1(
        completion_state=source_completion.completion_state,
        expected_transition_count=source_completion.expected_transition_count,
        captured_transition_count=source_completion.validated_transition_count,
        terminated=source_completion.terminated,
        truncated=source_completion.truncated,
        completion_bases=source_completion.completion_bases,
        public_end_or_failure_reason=source_completion.end_or_failure_reason,
    )
    content_payload: dict[str, object] = {
        "schema_id": ACTOR_POV_CONTENT_SCHEMA_ID,
        "schema_version": ACTOR_POV_SCHEMA_VERSION,
        "content_id": f"{episode_id}:actor-pov:{public_agent_id}:content",
        "episode_id": episode_id,
        "selected_global_slot": global_slot,
        "selected_team_local_slot": roster.team_local_slot,
        "public_agent_id": public_agent_id,
        "configured_team_id": roster.configured_team_id,
        "class_id": roster.class_id,
        "observation_materialization": "exact_no_shared_obs_actor_input",
        "axis_mapping": _axis_mapping_from_replay(replay, global_slot=global_slot),
        "completion": completion,
        "frames": frames,
        "transitions": transitions,
    }
    content = ActorPovReplayContentV1.model_validate(
        {
            **content_payload,
            "canonical_digest_sha256": canonical_digest_sha256(content_payload),
        }
    )
    artifact_payload: dict[str, object] = {
        "schema_id": ACTOR_POV_ARTIFACT_SCHEMA_ID,
        "schema_version": ACTOR_POV_SCHEMA_VERSION,
        "artifact_id": f"{episode_id}:actor-pov:{public_agent_id}",
        "source_replay": _build_replay_reference_from_validated(replay),
        "content": content,
    }
    return ActorPovReplayArtifactV1.model_validate(
        {
            **artifact_payload,
            "canonical_digest_sha256": canonical_digest_sha256(artifact_payload),
        }
    )


def validate_actor_pov_replay_content_v1(
    content: ActorPovReplayContentV1,
) -> None:
    """Validate one exact recipient-content tree and rederive all local cues."""
    canonical = cast(
        ActorPovReplayContentV1,
        validate_declared_model_tree(
            content,
            record_name="actor POV replay content",
            expected_type=ActorPovReplayContentV1,
        ),
    )
    for transition_index, transition in enumerate(canonical.transitions):
        expected_cues = _derive_cues(
            episode_id=canonical.episode_id,
            public_agent_id=canonical.public_agent_id,
            transition_index=transition_index,
            team_local_slot=canonical.selected_team_local_slot,
            start_frame=canonical.frames[transition_index],
            successor_frame=canonical.frames[transition_index + 1],
            has_any_rejection=(
                transition.submitted_action_tuple_is_out_of_domain
                or transition.in_domain_move_action_is_rejected
                or transition.in_domain_combat_action_pair_is_rejected
            ),
            terminated=transition.terminated,
            truncated=transition.truncated,
            public_end_reason=transition.public_end_reason,
        )
        if transition.cues != expected_cues:
            raise ValueError("POV cues must equal authorized local rederivation")


def validate_actor_pov_replay_artifact_v1(
    artifact: ActorPovReplayArtifactV1,
) -> None:
    """Validate one standalone POV envelope and its authorized content."""
    canonical = cast(
        ActorPovReplayArtifactV1,
        validate_declared_model_tree(
            artifact,
            record_name="actor POV replay artifact",
            expected_type=ActorPovReplayArtifactV1,
        ),
    )
    validate_actor_pov_replay_content_v1(canonical.content)


def export_actor_pov_replay_v1(
    replay: ReplayArtifactV1,
    *,
    global_slot: int,
) -> ActorPovReplayArtifactV1:
    """Export one exact NoSharedObs recipient slice from a validated replay."""
    if type(replay) is not ReplayArtifactV1:
        raise TypeError("actor POV export requires ReplayArtifactV1")
    validate_replay_artifact_v1(replay)
    artifact = _export_from_validated_replay(replay, global_slot=global_slot)
    validate_actor_pov_replay_artifact_v1(artifact)
    return artifact


def validate_actor_pov_replay_against_replay_v1(
    artifact: ActorPovReplayArtifactV1,
    replay: ReplayArtifactV1,
) -> None:
    """Cross-validate an actor POV artifact against its authoritative replay."""
    validate_actor_pov_replay_artifact_v1(artifact)
    if type(replay) is not ReplayArtifactV1:
        raise TypeError("actor POV source must be ReplayArtifactV1")
    validate_replay_artifact_v1(replay)
    expected = _export_from_validated_replay(
        replay,
        global_slot=artifact.content.selected_global_slot,
    )
    if artifact != expected:
        raise ValueError("actor POV artifact does not match its source replay")


def canonical_actor_pov_content_json_bytes_v1(
    content: ActorPovReplayContentV1,
) -> bytes:
    """Return canonical bytes for recipient-authorized content only."""
    validate_actor_pov_replay_content_v1(content)
    return canonical_json_bytes(content)


def canonical_actor_pov_replay_json_bytes_v1(
    artifact: ActorPovReplayArtifactV1,
) -> bytes:
    """Return canonical bytes for the full provenance-bearing POV artifact."""
    validate_actor_pov_replay_artifact_v1(artifact)
    return canonical_json_bytes(artifact)


__all__ = [
    "ACTOR_POV_ADJACENT_TRANSITION_SLICE_SCHEMA_ID",
    "ACTOR_POV_ARTIFACT_SCHEMA_ID",
    "ACTOR_POV_AXIS_MAPPING_SCHEMA_ID",
    "ACTOR_POV_CONTENT_SCHEMA_ID",
    "ACTOR_POV_CURRENT_SLICE_SCHEMA_ID",
    "ACTOR_POV_SCHEMA_VERSION",
    "ActorPovAcceptedActionV1",
    "ActorPovActionMaskV1",
    "ActorPovAdjacentTransitionSliceV1",
    "ActorPovAxisMappingV1",
    "ActorPovCurrentSliceV1",
    "ActorPovEpisodeCompletionV1",
    "ActorPovEpisodeEndedCueV1",
    "ActorPovFrameV1",
    "ActorPovOwnActionOutcomeCueV1",
    "ActorPovOwnCooldownChangedCueV1",
    "ActorPovOwnHealthChangedCueV1",
    "ActorPovOwnLifecycleChangedCueV1",
    "ActorPovOwnPositionChangedCueV1",
    "ActorPovOwnStatusChangedCueV1",
    "ActorPovPresentationCueV1",
    "ActorPovPreviousTimestepActionsV1",
    "ActorPovReplayArtifactV1",
    "ActorPovReplayContentV1",
    "ActorPovSpawnLifecycleV1",
    "ActorPovSubmittedActionV1",
    "ActorPovTransitionV1",
    "ActorPovVisibleBodyObservationChangedCueV1",
    "build_actor_pov_adjacent_transition_slice_v1",
    "build_actor_pov_current_slice_v1",
    "canonical_actor_pov_content_json_bytes_v1",
    "canonical_actor_pov_replay_json_bytes_v1",
    "export_actor_pov_replay_v1",
    "slice_actor_pov_current_frame_v1",
    "slice_actor_pov_current_transition_v1",
    "validate_actor_pov_replay_against_replay_v1",
    "validate_actor_pov_replay_artifact_v1",
    "validate_actor_pov_replay_content_v1",
]
