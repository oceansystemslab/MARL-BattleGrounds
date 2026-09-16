"""Build and validate one actor's recorded view for live and saved playback.

The live builders copy only the selected actor's base observation, action masks,
own submitted and accepted actions, reward, and public end flags. Presentation cues
describe changes in those allowed rows; they do not read privileged state or the
simulator event feed. These host-side Pydantic records are not JAX rollout state.

POV V1 reads historical replay V1. POV V2 reads replay V3 and preserves current
actor-relative relation flags and public roster classes. Export requires NoSharedObs
and a configured-active actor; dead actors remain selectable. No files are written
here. replay_io owns storage.

Content hashes cover the selected actor's content. Artifact hashes also cover the
full source replay reference. Standalone checks prove internal consistency;
validate_actor_pov_replay_against_replay_v1/v2 additionally prove the exact source
projection. EvaluationModel owns the shared strict, frozen record rules.
"""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict, cast, overload

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
    EvaluationEpisodeContext,
    EvaluationFrame,
    EvaluationFrameV1,
    EvaluationFrameV2,
    EvaluationModel,
    EvaluationTransitionV1,
    RosterSlotV1,
    canonical_digest_sha256,
    canonical_json_bytes,
    evaluation_context_type,
)
from marl_battlegrounds.evaluation.replay import (
    ReplayArtifactReferenceV1,
    ReplayArtifactV1,
    validate_replay_artifact_v1,
)
from marl_battlegrounds.evaluation.replay_v3 import (
    ReplayArtifactReferenceV3,
    ReplayArtifactV3,
    replay_reference_v3,
)
from marl_battlegrounds.evaluation.validation import (
    validate_declared_model_tree,
    validate_initial_evaluation_frame_v1,
)

ACTOR_POV_SCHEMA_VERSION: Literal[1] = 1
ACTOR_POV_SCHEMA_VERSION_V2: Literal[2] = 2
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


def _require_schema_version_two(value: object) -> object:
    """Return value only when it is the exact Python integer 2; otherwise raise
    ValueError.
    """
    if type(value) is not int or value != 2:
        raise ValueError("schema_version must be the exact integer 2")
    return value


_SchemaVersionV2 = Annotated[Literal[2], BeforeValidator(_require_schema_version_two)]


def _require_schema_version_one(value: object) -> object:
    """Return value only when it is the exact Python integer 1; otherwise raise
    ValueError.
    """
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
    """Check nested tuple dimensions without checking scalar values.

    Parameters
    ----------
    value : object
        Nested tuple to inspect.
    expected_shape : tuple[int, ...]
        Required lengths from outermost to innermost axis. Empty means the scalar leaf
        is accepted.
    field_name : str
        Field name included in ValueError.

    Returns
    -------
    None
        None.

    Raises
    ------
    ValueError
        A required axis is not a tuple or has the wrong length.
    """
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
    """Store the public completion status of a recorded actor prefix.

    Attributes
    ----------
    schema_id : Literal['marl_battlegrounds.evaluation.actor_pov_completion']
        Fixed actor POV completion schema identifier.
    schema_version : _SchemaVersionV1
        Exact integer 1, the default.
    completion_state : Literal['complete', 'partial', 'interrupted', 'failed']
        complete, partial, interrupted, or failed.
    expected_transition_count : Annotated[int, Field(gt=0)]
        Declared positive episode horizon in transitions.
    captured_transition_count : _NonNegativeInt
        Saved transition count from zero through the declared horizon.
    terminated : bool
        Whether the final transition ended the task.
    truncated : bool
        Whether the final transition has the separate truncation flag.
    completion_bases : tuple[Literal['task_terminal', 'declared_horizon'], ...]
        Ordered task_terminal then declared_horizon entries when their evidence exists.
        Must be empty for an incomplete prefix.
    public_end_or_failure_reason : _AsciiText | None
        Optional public reason, default None. Required for an incomplete prefix.

    Notes
    -----
    A complete prefix needs termination or the full declared horizon; truncation
    alone is not completion evidence. A zero-transition prefix cannot be done.
    """

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
        """Return this record after checking count bounds, ordered completion evidence,
        and the required incomplete reason; raise ValueError for contradictions.
        """
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
    """Store the prior-action rows visible to one actor at this decision.

    Attributes
    ----------
    schema_id : Literal['marl_battlegrounds.evaluation.actor_pov_previous_actions']
        Fixed actor POV previous-action schema identifier.
    schema_version : _SchemaVersionV1
        Exact integer 1, the default.
    ally_move_actions_one_hot : _FloatMatrix
        Finite float tuple matrix, shape (5, 9), in ally row order.
    enemy_move_actions_one_hot : _FloatMatrix
        Finite float tuple matrix, shape (5, 9), in enemy row order.
    ally_select_target_actions_one_hot : _FloatMatrix
        Finite float tuple matrix, shape (5, 11), for ally targets.
    enemy_select_target_actions_one_hot : _FloatMatrix
        Finite float tuple matrix, shape (5, 11), for enemy targets.
    ally_use_ultimate_actions_one_hot : _FloatMatrix
        Finite float tuple matrix, shape (5, 2), for ally Ultimate choices.
    enemy_use_ultimate_actions_one_hot : _FloatMatrix
        Finite float tuple matrix, shape (5, 2), for enemy Ultimate choices.

    Notes
    -----
    Producers supply one-hot or neutral rows under the observation contract.
    This model checks finite values and shapes, not one-hot semantics.
    """

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
        """Return this record when all six prior-action matrices have five rows and
        their declared category widths; otherwise raise ValueError.
        """
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
    """Store the selected actor's public spawn and respawn observations.

    Attributes
    ----------
    schema_id : Literal['marl_battlegrounds.evaluation.actor_pov_spawn_lifecycle']
        Fixed actor POV spawn-lifecycle schema identifier.
    schema_version : _SchemaVersionV1
        Exact integer 1, the default.
    spawn_pad_positions_by_team : _FloatTensor3
        Finite float tuples, shape (2, 5, 2), with x/y coordinates in map units.
    spawn_shield_actual_durations_by_team : _IntegerMatrix
        Nonnegative integer tuples, shape (2, 5), of remaining shield ticks.
    spawn_shield_configured_duration : _NonNegativeInt
        Configured nonnegative shield duration in ticks.
    spawn_shield_speed : Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
        Nonnegative finite speed while shielded, in map units per tick.
    respawn_wave_period_step_count_by_team : _IntegerVector
        Nonnegative integer tuple, shape (2,), of wave periods in ticks.
    respawn_wave_countdowns_by_team : _IntegerVector
        Nonnegative integer tuple, shape (2,), of ticks to each next wave.
    active_mask_by_team : _BooleanMatrix
        Boolean tuple matrix, shape (2, 5), marking configured roster slots.
    alive_mask_by_team : _BooleanMatrix
        Boolean tuple matrix, shape (2, 5), marking living slots.

    Notes
    -----
    Team axis zero means Own Team; one means Opponent Team. Slot order remains
    team-local. These are copied observation values, not a privileged state view.
    """

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
        """Return this record after checking the fixed team, slot, and position axes;
        raise ValueError for a wrong shape.
        """
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
    """Store the legal categories for one actor's next action.

    Attributes
    ----------
    schema_id : Literal['marl_battlegrounds.evaluation.actor_pov_action_mask']
        Fixed actor POV action-mask schema identifier.
    schema_version : _SchemaVersionV1
        Exact integer 1, the default.
    move : _BooleanVector
        Boolean tuple of length 9.
    select_target : _BooleanVector
        Boolean tuple of length 11, equal to the joint mask's target marginal.
    use_ultimate : _BooleanVector
        Boolean tuple of length 2, equal to the joint mask's Ultimate marginal.
    select_target_use_ultimate_joint : _BooleanMatrix
        Boolean tuples, shape (11, 2), for legal target/Ultimate pairs.

    Notes
    -----
    True marks a permitted category or pair. Sampling the two combat marginals
    independently does not ensure that the resulting pair is permitted.
    """

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
        """Return this mask after checking its fixed shapes and that both combat
        marginals equal any-true reductions of the joint mask; otherwise raise
        ValueError.
        """
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


def _validate_axismapping[T: ActorPovAxisMappingV1 | ActorPovAxisMappingV2](
    self: T,
) -> T:
    """Return the axis record after checking supported projection identity, fixed
    lengths, unique action names, disjoint public agent IDs, target order, and Own
    Team/Opponent Team order; raise ValueError for a mismatch.
    """
    from marl_battlegrounds.evaluation.actor_projection import (
        NO_SHARED_OBS_ACTOR_PROJECTION_V3,
        SHARED_OBS_ACTOR_PROJECTION_V2,
    )

    if self.schema_version == 1:
        if self.actor_projection_version != 1:
            raise ValueError("POV V1 requires actor projection version 1")
    elif (self.actor_projection_identifier, self.actor_projection_version) not in (
        (
            NO_SHARED_OBS_ACTOR_PROJECTION_V3.identifier,
            NO_SHARED_OBS_ACTOR_PROJECTION_V3.version,
        ),
        (
            SHARED_OBS_ACTOR_PROJECTION_V2.identifier,
            SHARED_OBS_ACTOR_PROJECTION_V2.version,
        ),
    ):
        raise ValueError("POV V2 requires the current actor projection")
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


class ActorPovAxisMappingV1(EvaluationModel):
    """Name the public categories for POV version 1.

    Reads context/frame/transition version 1 and requires projection version 1.

    Attributes
    ----------
    schema_id : Literal['marl_battlegrounds.evaluation.actor_pov_axis_mapping']
        Fixed actor POV axis-mapping schema identifier.
    schema_version : _SchemaVersionV1
        This record's exact default version.
    actor_projection_identifier : _AsciiIdentifier
        Source actor-input projection identifier.
    actor_projection_version : Annotated[int, Field(gt=0)]
        Source projection version.
    source_context_schema_id : Literal['marl_battlegrounds.evaluation.episode_context']
        Fixed evaluation context schema identifier.
    source_context_schema_version : _SchemaVersionV1
        Version of the compatible source context.
    source_frame_schema_id : Literal['marl_battlegrounds.evaluation.frame']
        Fixed evaluation frame schema identifier.
    source_frame_schema_version : _SchemaVersionV1
        Version of the compatible source frames.
    source_transition_schema_id : Literal['marl_battlegrounds.evaluation.transition']
        Fixed evaluation transition schema identifier.
    source_transition_schema_version : _SchemaVersionV1
        Exact integer 1, the default.
    target_action_recipient_public_agent_id_by_id : tuple[_AsciiIdentifier | None, ...]
        Length-11 tuple: None, then five ally IDs, then five enemy IDs.
    ally_observation_row_public_agent_id_by_id : tuple[_AsciiIdentifier, ...]
        Five distinct public IDs in ally observation order.
    enemy_observation_row_public_agent_id_by_id : tuple[_AsciiIdentifier, ...]
        Five distinct public IDs in enemy observation order, disjoint from allies.
    movement_action_name_by_id : tuple[_AsciiText, ...]
        Nine unique movement labels.
    unit_direction_vector_by_movement_action : _FloatMatrix
        Finite float tuples, shape (9, 2), indexed by movement category.
    target_action_name_by_id : tuple[_AsciiText, ...]
        Eleven unique target labels.
    use_ultimate_action_name_by_id : tuple[_AsciiText, ...]
        Two unique Ultimate labels.
    spawn_lifecycle_team_axis_name_by_id : tuple[_AsciiText, ...]
        Exactly Own Team then Opponent Team.
    """

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
        """Return this record after the shared projection, shape, identity, and
        category-order checks; raise ValueError for a mismatch.
        """
        return _validate_axismapping(self)


class ActorPovAxisMappingV2(EvaluationModel):
    """Name the public categories for POV version 2.

    Reads context V3, frame V2, and transition V1. The record accepts the current
    NoSharedObs V3 or SharedObs V2 projection identity; export separately requires
    NoSharedObs.

    Attributes
    ----------
    schema_id : Literal['marl_battlegrounds.evaluation.actor_pov_axis_mapping']
        Fixed actor POV axis-mapping schema identifier.
    schema_version : _SchemaVersionV2
        This record's exact default version.
    actor_projection_identifier : _AsciiIdentifier
        Source actor-input projection identifier.
    actor_projection_version : Annotated[int, Field(gt=0)]
        Source projection version.
    source_context_schema_id : Literal['marl_battlegrounds.evaluation.episode_context']
        Fixed evaluation context schema identifier.
    source_context_schema_version : Literal[3]
        Version of the compatible source context.
    source_frame_schema_id : Literal['marl_battlegrounds.evaluation.frame']
        Fixed evaluation frame schema identifier.
    source_frame_schema_version : _SchemaVersionV2
        Version of the compatible source frames.
    source_transition_schema_id : Literal['marl_battlegrounds.evaluation.transition']
        Fixed evaluation transition schema identifier.
    source_transition_schema_version : _SchemaVersionV1
        Exact integer 1, the default.
    target_action_recipient_public_agent_id_by_id : tuple[_AsciiIdentifier | None, ...]
        Length-11 tuple: None, then five ally IDs, then five enemy IDs.
    ally_observation_row_public_agent_id_by_id : tuple[_AsciiIdentifier, ...]
        Five distinct public IDs in ally observation order.
    enemy_observation_row_public_agent_id_by_id : tuple[_AsciiIdentifier, ...]
        Five distinct public IDs in enemy observation order, disjoint from allies.
    movement_action_name_by_id : tuple[_AsciiText, ...]
        Nine unique movement labels.
    unit_direction_vector_by_movement_action : _FloatMatrix
        Finite float tuples, shape (9, 2), indexed by movement category.
    target_action_name_by_id : tuple[_AsciiText, ...]
        Eleven unique target labels.
    use_ultimate_action_name_by_id : tuple[_AsciiText, ...]
        Two unique Ultimate labels.
    spawn_lifecycle_team_axis_name_by_id : tuple[_AsciiText, ...]
        Exactly Own Team then Opponent Team.
    """

    schema_id: Literal["marl_battlegrounds.evaluation.actor_pov_axis_mapping"] = (
        ACTOR_POV_AXIS_MAPPING_SCHEMA_ID
    )
    schema_version: _SchemaVersionV2 = 2
    actor_projection_identifier: _AsciiIdentifier
    actor_projection_version: Annotated[int, Field(gt=0)]
    source_context_schema_id: Literal[
        "marl_battlegrounds.evaluation.episode_context"
    ] = CONTEXT_SCHEMA_ID
    source_context_schema_version: Literal[3] = 3
    source_frame_schema_id: Literal["marl_battlegrounds.evaluation.frame"] = (
        FRAME_SCHEMA_ID
    )
    source_frame_schema_version: _SchemaVersionV2 = 2
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
    def _validate_axes(self) -> ActorPovAxisMappingV2:
        """Return this record after the shared projection, shape, identity, and
        category-order checks; raise ValueError for a mismatch.
        """
        return _validate_axismapping(self)


def _validate_frame[T: ActorPovFrameV1 | ActorPovFrameV2](self: T) -> T:
    """Return the frame after checking canonical IDs and feature shapes. For V2 also
    check public roster classes, self slot, and visible ally/enemy relation flags.
    Raise ValueError for mismatches; this does not compare the frame to its source
    replay.
    """
    expected_pov_id = (
        f"{self.episode_id}:actor-pov:{self.public_agent_id}:frame:{self.frame_index}"
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
    if isinstance(self, ActorPovFrameV2):
        _require_tuple_shape(
            self.class_ids_by_team,
            (NUM_TEAMS, MAX_AGENTS_PER_TEAM),
            field_name="class_ids_by_team",
        )
        if any(
            not 0 <= value <= 5 for team in self.class_ids_by_team for value in team
        ):
            raise ValueError("POV class IDs must be between zero and five")
        for classes, active in zip(
            self.class_ids_by_team,
            self.spawn_lifecycle.active_mask_by_team,
            strict=True,
        ):
            if any(
                (class_id != 0) != enabled
                for class_id, enabled in zip(classes, active, strict=True)
            ):
                raise ValueError("POV class IDs must match the active roster")
        if self.class_ids_by_team[0][self.self_ally_index] != int(
            self.self_features[_FEATURE_CLASS_ID]
        ):
            raise ValueError("POV self class must join self_ally_index")
        if self.self_features[_FEATURE_TEAM_ID] != 0.0:
            raise ValueError("POV self is_enemy must be zero")
        for relation, rows, visibility in (
            (0.0, self.ally_unit_features, self.ally_visibility_mask),
            (1.0, self.enemy_unit_features, self.enemy_visibility_mask),
        ):
            if any(
                visible and row[_FEATURE_TEAM_ID] != relation
                for row, visible in zip(rows, visibility, strict=True)
            ):
                raise ValueError("POV visible relation flags must match their axes")
    return self


class ActorPovFrameV1(EvaluationModel):
    """Store one actor's exact version-1 decision frame without privileged state.

    Attributes
    ----------
    schema_id : Literal['marl_battlegrounds.evaluation.actor_pov_frame']
        Fixed actor POV frame schema identifier.
    schema_version : _SchemaVersionV1
        This record's exact default version.
    episode_id : _AsciiIdentifier
        Public episode identity.
    public_agent_id : _AsciiIdentifier
        Selected actor's public identity.
    frame_index : _NonNegativeInt
        Nonnegative position in the retained prefix.
    pov_frame_id : _AsciiIdentifier
        Canonical episode/actor/frame identity.
    source_frame_id : _AsciiIdentifier
        Canonical episode/frame identity in the full source.
    simulator_step_count : _NonNegativeInt
        Nonnegative simulator tick represented by this decision frame.
    self_features : _FloatVector
        Finite float tuple of length 58 for the selected actor.
    ally_unit_features : _FloatMatrix
        Finite float tuples, shape (5, 58), in ally observation order.
    enemy_unit_features : _FloatMatrix
        Finite float tuples, shape (5, 58), in enemy observation order.
    map_obstacle_features : _FloatMatrix
        Finite float tuples, shape (32, 8), in fixed obstacle-slot order.
    objective_features : _FloatMatrix
        Finite float tuples, shape (8, 12), in fixed objective-slot order.
    context_features : _FloatVector
        Finite float tuple of length 19 for public task context.
    ally_visibility_mask : _BooleanVector
        Five booleans governing ally rows.
    enemy_visibility_mask : _BooleanVector
        Five booleans governing enemy rows.
    previous_timestep_actions : ActorPovPreviousTimestepActionsV1
        Prior-action rows visible to this actor.
    spawn_lifecycle : ActorPovSpawnLifecycleV1
        Public Own Team/Opponent Team lifecycle observations.
    action_mask : ActorPovActionMaskV1
        Legal categories for the action chosen from this frame.

    Notes
    -----
    Feature column 3 preserves the historical simulator team ID.
    Array data is held as frozen host tuples. Visibility and source topology are
    checked more fully by the live/content validators and source projection.
    """

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
        """Return this frame after the shared canonical-ID, shape, and version-specific
        roster/relation checks; otherwise raise ValueError.
        """
        return _validate_frame(self)


class ActorPovFrameV2(EvaluationModel):
    """Store one actor's exact version-2 decision frame without privileged state.

    Attributes
    ----------
    schema_id : Literal['marl_battlegrounds.evaluation.actor_pov_frame']
        Fixed actor POV frame schema identifier.
    schema_version : _SchemaVersionV2
        This record's exact default version.
    episode_id : _AsciiIdentifier
        Public episode identity.
    public_agent_id : _AsciiIdentifier
        Selected actor's public identity.
    frame_index : _NonNegativeInt
        Nonnegative position in the retained prefix.
    pov_frame_id : _AsciiIdentifier
        Canonical episode/actor/frame identity.
    source_frame_id : _AsciiIdentifier
        Canonical episode/frame identity in the full source.
    simulator_step_count : _NonNegativeInt
        Nonnegative simulator tick represented by this decision frame.
    self_features : _FloatVector
        Finite float tuple of length 58 for the selected actor.
    ally_unit_features : _FloatMatrix
        Finite float tuples, shape (5, 58), in ally observation order.
    enemy_unit_features : _FloatMatrix
        Finite float tuples, shape (5, 58), in enemy observation order.
    map_obstacle_features : _FloatMatrix
        Finite float tuples, shape (32, 8), in fixed obstacle-slot order.
    objective_features : _FloatMatrix
        Finite float tuples, shape (8, 12), in fixed objective-slot order.
    context_features : _FloatVector
        Finite float tuple of length 19 for public task context.
    ally_visibility_mask : _BooleanVector
        Five booleans governing ally rows.
    enemy_visibility_mask : _BooleanVector
        Five booleans governing enemy rows.
    previous_timestep_actions : ActorPovPreviousTimestepActionsV1
        Prior-action rows visible to this actor.
    spawn_lifecycle : ActorPovSpawnLifecycleV1
        Public Own Team/Opponent Team lifecycle observations.
    action_mask : ActorPovActionMaskV1
        Legal categories for the action chosen from this frame.
    self_ally_index : _TeamLocalSlot
        Team-local index from 0 through 4 locating self in ally rows.
    class_ids_by_team : _IntegerMatrix
        Integer tuples, shape (2, 5), Own Team first; zero marks inactive slots and 1
        through 5 are public classes.

    Notes
    -----
    Feature column 3 is actor-relative: self/allies zero, visible enemies one. Classes
    must agree with configured-active masks and the self row.
    Array data is held as frozen host tuples. Visibility and source topology are
    checked more fully by the live/content validators and source projection.
    """

    schema_id: Literal["marl_battlegrounds.evaluation.actor_pov_frame"] = (
        ACTOR_POV_FRAME_SCHEMA_ID
    )
    schema_version: _SchemaVersionV2 = 2
    episode_id: _AsciiIdentifier
    public_agent_id: _AsciiIdentifier
    frame_index: _NonNegativeInt
    pov_frame_id: _AsciiIdentifier
    source_frame_id: _AsciiIdentifier
    simulator_step_count: _NonNegativeInt
    self_ally_index: _TeamLocalSlot
    class_ids_by_team: _IntegerMatrix
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
    def _validate_frame(self) -> ActorPovFrameV2:
        """Return this frame after the shared canonical-ID, shape, and version-specific
        roster/relation checks; otherwise raise ValueError.
        """
        return _validate_frame(self)


class ActorPovSubmittedActionV1(EvaluationModel):
    """Store one actor's exact submitted action, including invalid categories.

    Attributes
    ----------
    schema_id : Literal['marl_battlegrounds.evaluation.actor_pov_submitted_action']
        Fixed actor POV submitted-action schema identifier.
    schema_version : _SchemaVersionV1
        Exact integer 1, the default.
    move : _Int32
        Signed int32-range integer submitted for movement.
    select_target : _Int32
        Signed int32-range integer submitted for an actor-relative target.
    use_ultimate : _Int32
        Signed int32-range integer submitted for the Ultimate choice.

    Notes
    -----
    Category bounds are deliberately not imposed on submitted actions. The
    transition records which parts were rejected and the accepted replacement.
    """

    schema_id: Literal["marl_battlegrounds.evaluation.actor_pov_submitted_action"] = (
        ACTOR_POV_SUBMITTED_ACTION_SCHEMA_ID
    )
    schema_version: _SchemaVersionV1 = ACTOR_POV_SCHEMA_VERSION
    move: _Int32
    select_target: _Int32
    use_ultimate: _Int32


class ActorPovAcceptedActionV1(EvaluationModel):
    """Store the category-bounded action actually accepted for one actor.

    Attributes
    ----------
    schema_id : Literal['marl_battlegrounds.evaluation.actor_pov_accepted_action']
        Fixed actor POV accepted-action schema identifier.
    schema_version : _SchemaVersionV1
        Exact integer 1, the default.
    move : _MoveAction
        Movement category from 0 through 8.
    select_target : _TargetAction
        Actor-relative target category from 0 through 10.
    use_ultimate : _UltimateAction
        Ultimate category 0 or 1.

    Notes
    -----
    The record checks category bounds. Its source transition owns legality and
    rejection truth; this record alone does not check an action mask.
    """

    schema_id: Literal["marl_battlegrounds.evaluation.actor_pov_accepted_action"] = (
        ACTOR_POV_ACCEPTED_ACTION_SCHEMA_ID
    )
    schema_version: _SchemaVersionV1 = ACTOR_POV_SCHEMA_VERSION
    move: _MoveAction
    select_target: _TargetAction
    use_ultimate: _UltimateAction


class ActorPovCueBaseV1(EvaluationModel):
    """Give a local presentation cue a stable transition-relative identity.

    Attributes
    ----------
    schema_id : Literal['marl_battlegrounds.evaluation.actor_pov_cue']
        Fixed actor POV cue schema identifier.
    schema_version : _SchemaVersionV1
        Exact integer 1, the default.
    cue_id : _AsciiIdentifier
        Canonical transition ID followed by :cue: and the zero-based ordinal.
    pov_transition_id : _AsciiIdentifier
        Selected actor's POV transition ID.
    ordinal : _NonNegativeInt
        Nonnegative position in the transition's cue tuple.

    Notes
    -----
    The containing transition checks canonical IDs and gap-free ordering. Each
    subclass adds its cue_type and the specific actor-visible change.
    """

    schema_id: Literal["marl_battlegrounds.evaluation.actor_pov_cue"] = (
        ACTOR_POV_CUE_SCHEMA_ID
    )
    schema_version: _SchemaVersionV1 = ACTOR_POV_SCHEMA_VERSION
    cue_id: _AsciiIdentifier
    pov_transition_id: _AsciiIdentifier
    ordinal: _NonNegativeInt


class ActorPovOwnActionOutcomeCueV1(ActorPovCueBaseV1):
    """Report whether any part of this actor's submitted action was rejected.

    Attributes
    ----------
    cue_type : Literal['own_action_outcome']
        Fixed own_action_outcome label.
    outcome : Literal['accepted', 'rejected']
        accepted when no rejection flag is set; rejected otherwise.

    Notes
    -----
    Identity fields are defined by ActorPovCueBaseV1. Accepted does not mean the
    action achieved its intended physical effect.
    """

    cue_type: Literal["own_action_outcome"] = "own_action_outcome"
    outcome: Literal["accepted", "rejected"]


class ActorPovOwnPositionChangedCueV1(ActorPovCueBaseV1):
    """Describe a change in the actor's own observed position.

    Attributes
    ----------
    cue_type : Literal['own_position_changed']
        Fixed own_position_changed label.
    start_position : tuple[_FiniteFloat, _FiniteFloat]
        Finite x/y pair in map units at the decision frame.
    successor_position : tuple[_FiniteFloat, _FiniteFloat]
        Different finite x/y pair at the next frame.

    Notes
    -----
    Inherits the cue identity fields. This is an observed change, not its cause.
    """

    cue_type: Literal["own_position_changed"] = "own_position_changed"
    start_position: tuple[_FiniteFloat, _FiniteFloat]
    successor_position: tuple[_FiniteFloat, _FiniteFloat]

    @model_validator(mode="after")
    def _validate_change(self) -> ActorPovOwnPositionChangedCueV1:
        """Return this cue when the paired position values differ; otherwise raise
        ValueError.
        """
        if self.start_position == self.successor_position:
            raise ValueError("position-change cues require a changed position")
        return self


class ActorPovOwnHealthChangedCueV1(ActorPovCueBaseV1):
    """Describe a change in the actor's own observed health.

    Attributes
    ----------
    cue_type : Literal['own_health_changed']
        Fixed own_health_changed label.
    start_health : _FiniteFloat
        Finite health value before the transition.
    successor_health : _FiniteFloat
        Different finite health value after the transition.

    Notes
    -----
    Inherits the cue identity fields. No attacker, healer, or hidden cause is inferred.
    """

    cue_type: Literal["own_health_changed"] = "own_health_changed"
    start_health: _FiniteFloat
    successor_health: _FiniteFloat

    @model_validator(mode="after")
    def _validate_change(self) -> ActorPovOwnHealthChangedCueV1:
        """Return this cue when the paired health values differ; otherwise raise
        ValueError.
        """
        if self.start_health == self.successor_health:
            raise ValueError("health-change cues require changed health")
        return self


class ActorPovOwnStatusChangedCueV1(ActorPovCueBaseV1):
    """Store only the own-status feature entries that changed.

    Attributes
    ----------
    cue_type : Literal['own_status_changed']
        Fixed own_status_changed label.
    changed_feature_indices : tuple[Annotated[int, Field(ge=_STATUS_FEATURE_START,
    lt=_STATUS_FEATURE_STOP)], ...]
        Nonempty sorted unique feature indices from 15 through 28.
    start_values : _FloatVector
        Finite float tuple aligned with the indices at the start frame.
    successor_values : _FloatVector
        Same-length tuple of changed finite values at the successor frame.

    Notes
    -----
    Identity comes from ActorPovCueBaseV1. Every paired value must differ; the
    cue does not infer the source of a status change.
    """

    cue_type: Literal["own_status_changed"] = "own_status_changed"
    changed_feature_indices: tuple[
        Annotated[int, Field(ge=_STATUS_FEATURE_START, lt=_STATUS_FEATURE_STOP)],
        ...,
    ]
    start_values: _FloatVector
    successor_values: _FloatVector

    @model_validator(mode="after")
    def _validate_changes(self) -> ActorPovOwnStatusChangedCueV1:
        """Return this cue after checking nonempty, sorted, unique indices and
        equal-length value tuples whose paired values all differ; otherwise raise
        ValueError.
        """
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
    """Describe a change in the actor's observed Ultimate cooldown.

    Attributes
    ----------
    cue_type : Literal['own_cooldown_changed']
        Fixed own_cooldown_changed label.
    start_remaining_ticks : _FiniteFloat
        Finite observed cooldown value at the start frame.
    successor_remaining_ticks : _FiniteFloat
        Different finite observed cooldown value at the successor.

    Notes
    -----
    Inherits cue identity. The fields preserve the float feature values; this
    record does not independently require nonnegative or integer values.
    """

    cue_type: Literal["own_cooldown_changed"] = "own_cooldown_changed"
    start_remaining_ticks: _FiniteFloat
    successor_remaining_ticks: _FiniteFloat

    @model_validator(mode="after")
    def _validate_change(self) -> ActorPovOwnCooldownChangedCueV1:
        """Return this cue when the paired cooldown values differ; otherwise raise
        ValueError.
        """
        if self.start_remaining_ticks == self.successor_remaining_ticks:
            raise ValueError("cooldown-change cues require a changed countdown")
        return self


class ActorPovOwnLifecycleChangedCueV1(ActorPovCueBaseV1):
    """Describe a change in the actor's activity, life, or spawn shield.

    Attributes
    ----------
    cue_type : Literal['own_lifecycle_changed']
        Fixed own_lifecycle_changed label.
    start_active : bool
        Configured-active flag at the start frame.
    successor_active : bool
        Configured-active flag at the successor frame.
    start_alive : bool
        Living flag at the start frame.
    successor_alive : bool
        Living flag at the successor frame.
    start_spawn_shield_remaining_ticks : _NonNegativeInt
        Nonnegative integer shield ticks before the transition.
    successor_spawn_shield_remaining_ticks : _NonNegativeInt
        Nonnegative integer shield ticks afterward.

    Notes
    -----
    Inherits cue identity. At least one paired field must change.
    """

    cue_type: Literal["own_lifecycle_changed"] = "own_lifecycle_changed"
    start_active: bool
    successor_active: bool
    start_alive: bool
    successor_alive: bool
    start_spawn_shield_remaining_ticks: _NonNegativeInt
    successor_spawn_shield_remaining_ticks: _NonNegativeInt

    @model_validator(mode="after")
    def _validate_change(self) -> ActorPovOwnLifecycleChangedCueV1:
        """Return this cue when the paired activity, life, or shield duration values
        differ; otherwise raise ValueError.
        """
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
    """Report a changed ally/enemy row that was visible at an endpoint.

    Attributes
    ----------
    cue_type : Literal['visible_body_observation_changed']
        Fixed visible_body_observation_changed label.
    relation : Literal['ally', 'enemy']
        ally or enemy, relative to the selected actor.
    observation_row : Annotated[int, Field(ge=0, lt=MAX_AGENTS_PER_TEAM)]
        Team-local observation row from 0 through 4.
    start_visible : bool
        Whether this row was visible before the transition.
    successor_visible : bool
        Whether this row is visible afterward.
    observed_payload_changed : bool
        Whether the compared feature rows differ.

    Notes
    -----
    Inherits cue identity. At least one endpoint must be visible and either
    visibility or payload must change. The cue carries no hidden body snapshot.
    """

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
        """Return this cue when a row is visible at an endpoint and either visibility or
        payload changed; otherwise raise ValueError.
        """
        if not (self.start_visible or self.successor_visible):
            raise ValueError("visible-body cues require visibility at one endpoint")
        if (
            self.start_visible == self.successor_visible
            and not self.observed_payload_changed
        ):
            raise ValueError("visible-body cues require visibility or payload change")
        return self


class ActorPovEpisodeEndedCueV1(ActorPovCueBaseV1):
    """Report the public done flags attached to this transition.

    Attributes
    ----------
    cue_type : Literal['episode_ended']
        Fixed episode_ended label.
    terminated : bool
        Whether the task ended.
    truncated : bool
        Whether the separate truncation flag was set.
    public_end_reason : _AsciiText | None
        Optional recorded public reason, default None.

    Notes
    -----
    Inherits cue identity. At least one done flag must be true.
    """

    cue_type: Literal["episode_ended"] = "episode_ended"
    terminated: bool
    truncated: bool
    public_end_reason: _AsciiText | None = None

    @model_validator(mode="after")
    def _validate_done(self) -> ActorPovEpisodeEndedCueV1:
        """Return this cue if terminated or truncated is true; otherwise raise
        ValueError.
        """
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
    """Store one actor's action, reward, and visible cues for one transition.

    Attributes
    ----------
    schema_id : Literal['marl_battlegrounds.evaluation.actor_pov_transition']
        Fixed actor POV transition schema identifier.
    schema_version : _SchemaVersionV1
        Exact integer 1, the default.
    episode_id : _AsciiIdentifier
        Public episode identity shared by both endpoints.
    public_agent_id : _AsciiIdentifier
        Public identity of the selected actor.
    transition_index : _NonNegativeInt
        Nonnegative index, equal to the start frame index.
    pov_transition_id : _AsciiIdentifier
        Canonical episode/actor/transition identity.
    start_pov_frame_id : _AsciiIdentifier
        Canonical selected-actor frame ID at transition_index.
    successor_pov_frame_id : _AsciiIdentifier
        Canonical selected-actor frame ID at transition_index + 1.
    submitted_action : ActorPovSubmittedActionV1
        Exact signed-int32 submitted categories, even when invalid.
    accepted_action : ActorPovAcceptedActionV1
        Category-bounded action recorded as accepted.
    submitted_action_tuple_is_out_of_domain : bool
        Whether any submitted category was outside its domain.
    in_domain_move_action_is_rejected : bool
        Whether the in-domain movement choice was rejected.
    in_domain_combat_action_pair_is_rejected : bool
        Whether the in-domain target/Ultimate pair was rejected.
    canonical_reward : _FiniteFloat
        Finite recorded reward for this actor for this transition.
    terminated : bool
        Recorded task termination flag.
    truncated : bool
        Recorded truncation flag.
    public_end_reason : _AsciiText | None
        Optional public task reason, default None; allowed only when done.
    cues : tuple[ActorPovPresentationCueV1, ...]
        Ordered tuple of recipient-local presentation cues with canonical, gap-free IDs.

    Notes
    -----
    Contains no privileged event feed. Model construction checks identity and
    cue joins; content/live-carrier validation separately rederives cue payloads.
    """

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
        """Return this transition after checking canonical endpoint/transition IDs,
        done-only end reasons, and ordered cue IDs; raise ValueError for a mismatch.
        """
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


def _validate_currentslice[T: ActorPovCurrentSliceV1 | ActorPovCurrentSliceV2](
    self: T,
) -> T:
    """Return the current slice after checking selected slot/team/class identity and its
    exact incoming transition reference; frame zero must have no incoming transition.
    Raise ValueError for mismatches. Full source validation belongs to the builder.
    """
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
        isinstance(frame, ActorPovFrameV2)
        and frame.self_ally_index != self.selected_team_local_slot
    ):
        raise ValueError("POV self index must match its selected local slot")
    if (
        frame.episode_id != self.episode_id
        or frame.public_agent_id != self.public_agent_id
    ):
        raise ValueError("POV current frame must join the selected identity")
    if frame.self_features[_FEATURE_ACTIVE] != 1.0:
        raise ValueError("configured-active POV self rows require ACTIVE=1")
    if frame.self_features[_FEATURE_ALIVE] not in (0.0, 1.0):
        raise ValueError("POV self ALIVE must be exactly zero or one")
    if frame.self_features[_FEATURE_TEAM_ID] != (
        float(self.configured_team_id) if frame.schema_version == 1 else 0.0
    ):
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


class ActorPovCurrentSliceV1(EvaluationModel):
    """Carry the selected actor's current frame and its incoming transition.

    Attributes
    ----------
    schema_id : Literal['marl_battlegrounds.evaluation.actor_pov_current_slice']
        Fixed actor POV current-slice schema identifier.
    schema_version : _SchemaVersionV1
        Exact integer 1, the default.
    episode_id : _AsciiIdentifier
        Public episode identity.
    selected_global_slot : _GlobalSlot
        Simulator slot from 0 through 9; Team A occupies 0 through 4.
    selected_team_local_slot : _TeamLocalSlot
        Slot within the selected actor's own team, from 0 through 4.
    public_agent_id : _AsciiIdentifier
        Public identity at the selected roster slot.
    configured_team_id : _TeamId
        Simulator team ID 1 or 2, preserved as metadata.
    class_id : _ClassId
        Selected actor's public class ID from 1 through 5.
    observation_materialization : Literal['exact_no_shared_obs_actor_input']
        Fixed exact_no_shared_obs_actor_input label.
    axis_mapping : ActorPovAxisMappingV1
        Version-matched public action and observation axes.
    frame : ActorPovFrameV1
        Version-1 selected-actor frame.
    incoming_transition : ActorPovTransitionV1 | None
        None at frame zero; exactly the transition entering every later frame.

    Notes
    -----
    An in-memory live carrier, not a saved replay artifact. It holds no earlier
    prefix, privileged event stream, completion claim, or source replay reference.
    Use build_actor_pov_current_slice_v1 to validate the source records and derive cues.
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
        """Return this carrier after checking selected identity and the current/incoming
        frame join; otherwise raise ValueError.
        """
        return _validate_currentslice(self)


class ActorPovCurrentSliceV2(EvaluationModel):
    """Carry the selected actor's current frame and its incoming transition.

    Attributes
    ----------
    schema_id : Literal['marl_battlegrounds.evaluation.actor_pov_current_slice']
        Fixed actor POV current-slice schema identifier.
    schema_version : _SchemaVersionV2
        Exact integer 2, the default.
    episode_id : _AsciiIdentifier
        Public episode identity.
    selected_global_slot : _GlobalSlot
        Simulator slot from 0 through 9; Team A occupies 0 through 4.
    selected_team_local_slot : _TeamLocalSlot
        Slot within the selected actor's own team, from 0 through 4.
    public_agent_id : _AsciiIdentifier
        Public identity at the selected roster slot.
    configured_team_id : _TeamId
        Simulator team ID 1 or 2, preserved as metadata.
    class_id : _ClassId
        Selected actor's public class ID from 1 through 5.
    observation_materialization : Literal['exact_no_shared_obs_actor_input']
        Fixed exact_no_shared_obs_actor_input label.
    axis_mapping : ActorPovAxisMappingV2
        Version-matched public action and observation axes.
    frame : ActorPovFrameV2
        Version-2 selected-actor frame.
    incoming_transition : ActorPovTransitionV1 | None
        None at frame zero; exactly the transition entering every later frame.

    Notes
    -----
    An in-memory live carrier, not a saved replay artifact. It holds no earlier
    prefix, privileged event stream, completion claim, or source replay reference.
    Use build_actor_pov_current_slice_v1 to validate the source records and derive cues.
    """

    schema_id: Literal["marl_battlegrounds.evaluation.actor_pov_current_slice"] = (
        ACTOR_POV_CURRENT_SLICE_SCHEMA_ID
    )
    schema_version: _SchemaVersionV2 = 2
    episode_id: _AsciiIdentifier
    selected_global_slot: _GlobalSlot
    selected_team_local_slot: _TeamLocalSlot
    public_agent_id: _AsciiIdentifier
    configured_team_id: _TeamId
    class_id: _ClassId
    observation_materialization: Literal["exact_no_shared_obs_actor_input"] = (
        "exact_no_shared_obs_actor_input"
    )
    axis_mapping: ActorPovAxisMappingV2
    frame: ActorPovFrameV2
    incoming_transition: ActorPovTransitionV1 | None = None

    @model_validator(mode="after")
    def _validate_current_slice(self) -> ActorPovCurrentSliceV2:
        """Return this carrier after checking selected identity and the current/incoming
        frame join; otherwise raise ValueError.
        """
        return _validate_currentslice(self)


def _adjacent_cue_endpoint(frame: ActorPovFrame) -> ActorPovFrame:
    """Copy a frame with every hidden ally/enemy row replaced by zeros.

    Parameters
    ----------
    frame : ActorPovFrame
        Valid selected-actor endpoint with matching feature and visibility axes.

    Returns
    -------
    ActorPovFrame
        A model copy used only for cue derivation; the original exact endpoint is
        unchanged.

    Notes
    -----
    This prevents hidden row payload changes from producing a visible-body cue.
    Other fields retain their existing immutable values.
    """
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


def _validate_adjacenttransitionslice[
    T: ActorPovAdjacentTransitionSliceV1 | ActorPovAdjacentTransitionSliceV2
](self: T) -> T:
    """Return this live carrier after checking selected actor identity, visible
    self/lifecycle agreement, adjacent frame indices and simulator ticks, and exact
    cue rederivation from masked endpoints. Raise ValueError for any mismatch.
    """
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
        if (
            isinstance(endpoint, ActorPovFrameV2)
            and endpoint.self_ally_index != self.selected_team_local_slot
        ):
            raise ValueError("POV self index must match its selected local slot")
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


class ActorPovAdjacentTransitionSliceV1(EvaluationModel):
    """Carry one actor's transition together with both allowed observation endpoints.

    Attributes
    ----------
    schema_id :
    Literal['marl_battlegrounds.evaluation.actor_pov_adjacent_transition_slice']
        Fixed actor POV adjacent-transition-slice schema identifier.
    schema_version : _SchemaVersionV1
        Exact integer 1, the default.
    episode_id : _AsciiIdentifier
        Public episode identity.
    selected_global_slot : _GlobalSlot
        Simulator slot from 0 through 9; Team A occupies 0 through 4.
    selected_team_local_slot : _TeamLocalSlot
        Slot within the selected actor's own team, from 0 through 4.
    public_agent_id : _AsciiIdentifier
        Public identity at the selected roster slot.
    configured_team_id : _TeamId
        Simulator team ID 1 or 2, preserved as metadata.
    class_id : _ClassId
        Selected actor's public class ID from 1 through 5.
    observation_materialization : Literal['exact_no_shared_obs_actor_input']
        Fixed exact_no_shared_obs_actor_input label.
    axis_mapping : ActorPovAxisMappingV1
        Version-matched public action and observation axes.
    start_frame : ActorPovFrameV1
        Version-1 observation used to choose the action.
    transition : ActorPovTransitionV1
        Selected actor's action, reward, and local cues.
    successor_frame : ActorPovFrameV1
        Version-1 observation exactly one simulator tick later.

    Notes
    -----
    In-memory live carrier; contains no privileged events or source-evidence root.
    Validation checks actor/topology/time joins and derives cues after masking
    hidden relation rows. Stored endpoints themselves remain exact source rows.
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
        """Return this carrier after the shared endpoint, actor, lifecycle, and
        cue-consistency checks; otherwise raise ValueError.
        """
        return _validate_adjacenttransitionslice(self)


class ActorPovAdjacentTransitionSliceV2(EvaluationModel):
    """Carry one actor's transition together with both allowed observation endpoints.

    Attributes
    ----------
    schema_id :
    Literal['marl_battlegrounds.evaluation.actor_pov_adjacent_transition_slice']
        Fixed actor POV adjacent-transition-slice schema identifier.
    schema_version : _SchemaVersionV2
        Exact integer 2, the default.
    episode_id : _AsciiIdentifier
        Public episode identity.
    selected_global_slot : _GlobalSlot
        Simulator slot from 0 through 9; Team A occupies 0 through 4.
    selected_team_local_slot : _TeamLocalSlot
        Slot within the selected actor's own team, from 0 through 4.
    public_agent_id : _AsciiIdentifier
        Public identity at the selected roster slot.
    configured_team_id : _TeamId
        Simulator team ID 1 or 2, preserved as metadata.
    class_id : _ClassId
        Selected actor's public class ID from 1 through 5.
    observation_materialization : Literal['exact_no_shared_obs_actor_input']
        Fixed exact_no_shared_obs_actor_input label.
    axis_mapping : ActorPovAxisMappingV2
        Version-matched public action and observation axes.
    start_frame : ActorPovFrameV2
        Version-2 observation used to choose the action.
    transition : ActorPovTransitionV1
        Selected actor's action, reward, and local cues.
    successor_frame : ActorPovFrameV2
        Version-2 observation exactly one simulator tick later.

    Notes
    -----
    In-memory live carrier; contains no privileged events or source-evidence root.
    Validation checks actor/topology/time joins and derives cues after masking
    hidden relation rows. Stored endpoints themselves remain exact source rows.
    """

    schema_id: Literal[
        "marl_battlegrounds.evaluation.actor_pov_adjacent_transition_slice"
    ] = ACTOR_POV_ADJACENT_TRANSITION_SLICE_SCHEMA_ID
    schema_version: _SchemaVersionV2 = 2
    episode_id: _AsciiIdentifier
    selected_global_slot: _GlobalSlot
    selected_team_local_slot: _TeamLocalSlot
    public_agent_id: _AsciiIdentifier
    configured_team_id: _TeamId
    class_id: _ClassId
    observation_materialization: Literal["exact_no_shared_obs_actor_input"] = (
        "exact_no_shared_obs_actor_input"
    )
    axis_mapping: ActorPovAxisMappingV2
    start_frame: ActorPovFrameV2
    transition: ActorPovTransitionV1
    successor_frame: ActorPovFrameV2

    @model_validator(mode="after")
    def _validate_adjacent_slice(self) -> ActorPovAdjacentTransitionSliceV2:
        """Return this carrier after the shared endpoint, actor, lifecycle, and
        cue-consistency checks; otherwise raise ValueError.
        """
        return _validate_adjacenttransitionslice(self)


def _validate_replaycontent[T: ActorPovReplayContentV1 | ActorPovReplayContentV2](
    self: T,
) -> T:
    """Return actor content after checking canonical identity/digest, a gap-free T+1/T
    frame/transition prefix, selected roster topology, adjacent ticks, and
    completion/tail agreement. Raise ValueError for mismatches. Public validators
    additionally rederive cues.
    """
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
        if (
            isinstance(frame, ActorPovFrameV2)
            and frame.self_ally_index != self.selected_team_local_slot
        ):
            raise ValueError("POV self index must match its selected local slot")
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
        if frame.self_features[_FEATURE_TEAM_ID] != (
            float(self.configured_team_id) if frame.schema_version == 1 else 0.0
        ):
            raise ValueError("POV self team feature must match content metadata")
        if frame.self_features[_FEATURE_CLASS_ID] != float(self.class_id):
            raise ValueError("POV self class feature must match content metadata")
        if frame_index > 0 and frame.simulator_step_count != (
            self.frames[frame_index - 1].simulator_step_count + 1
        ):
            raise ValueError("POV simulator epochs must be adjacent")
    for transition_index, transition in enumerate(self.transitions):
        if transition.transition_index != transition_index:
            raise ValueError("POV transition positions must equal transition indices")
        if (
            transition.episode_id != self.episode_id
            or transition.public_agent_id != self.public_agent_id
        ):
            raise ValueError("POV transitions must join content identity")
        if transition.start_pov_frame_id != self.frames[transition_index].pov_frame_id:
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


class ActorPovReplayContentV1(EvaluationModel):
    """Store version-1 actor content independently of full-source provenance.

    Attributes
    ----------
    schema_id : Literal['marl_battlegrounds.evaluation.actor_pov_content']
        Fixed actor POV content schema identifier.
    schema_version : _SchemaVersionV1
        Exact integer 1, the default.
    content_id : _AsciiIdentifier
        Canonical episode/actor/content identity.
    canonical_digest_sha256 : _Sha256Hex
        SHA-256 of canonical content excluding this digest field.
    episode_id : _AsciiIdentifier
        Public episode identity.
    selected_global_slot : _GlobalSlot
        Simulator slot from 0 through 9; Team A occupies 0 through 4.
    selected_team_local_slot : _TeamLocalSlot
        Slot within the selected actor's own team, from 0 through 4.
    public_agent_id : _AsciiIdentifier
        Public identity at the selected roster slot.
    configured_team_id : _TeamId
        Simulator team ID 1 or 2, preserved as metadata.
    class_id : _ClassId
        Selected actor's public class ID from 1 through 5.
    observation_materialization : Literal['exact_no_shared_obs_actor_input']
        Fixed exact_no_shared_obs_actor_input label.
    axis_mapping : ActorPovAxisMappingV1
        Version-matched public action and observation axes.
    completion : ActorPovEpisodeCompletionV1
        Public completion state matching the saved transition tail.
    frames : tuple[ActorPovFrameV1, ...]
        Ordered version-1 frame tuple, length T+1, beginning at frame zero.
    transitions : tuple[ActorPovTransitionV1, ...]
        Ordered actor transition tuple of length T; no transition follows a done row.

    Notes
    -----
    Content construction checks IDs, topology, adjacent ticks, completion, and
    digest. Public content validators also rederive cues. To establish that this
    content is the exact source projection, validate its artifact against a replay.
    """

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
        """Return this content after the shared prefix, identity, completion, topology,
        and digest checks; otherwise raise ValueError.
        """
        return _validate_replaycontent(self)


class ActorPovReplayContentV2(EvaluationModel):
    """Store version-2 actor content independently of full-source provenance.

    Attributes
    ----------
    schema_id : Literal['marl_battlegrounds.evaluation.actor_pov_content']
        Fixed actor POV content schema identifier.
    schema_version : _SchemaVersionV2
        Exact integer 2, the default.
    content_id : _AsciiIdentifier
        Canonical episode/actor/content identity.
    canonical_digest_sha256 : _Sha256Hex
        SHA-256 of canonical content excluding this digest field.
    episode_id : _AsciiIdentifier
        Public episode identity.
    selected_global_slot : _GlobalSlot
        Simulator slot from 0 through 9; Team A occupies 0 through 4.
    selected_team_local_slot : _TeamLocalSlot
        Slot within the selected actor's own team, from 0 through 4.
    public_agent_id : _AsciiIdentifier
        Public identity at the selected roster slot.
    configured_team_id : _TeamId
        Simulator team ID 1 or 2, preserved as metadata.
    class_id : _ClassId
        Selected actor's public class ID from 1 through 5.
    observation_materialization : Literal['exact_no_shared_obs_actor_input']
        Fixed exact_no_shared_obs_actor_input label.
    axis_mapping : ActorPovAxisMappingV2
        Version-matched public action and observation axes.
    completion : ActorPovEpisodeCompletionV1
        Public completion state matching the saved transition tail.
    frames : tuple[ActorPovFrameV2, ...]
        Ordered version-2 frame tuple, length T+1, beginning at frame zero.
    transitions : tuple[ActorPovTransitionV1, ...]
        Ordered actor transition tuple of length T; no transition follows a done row.

    Notes
    -----
    Content construction checks IDs, topology, adjacent ticks, completion, and
    digest. Public content validators also rederive cues. To establish that this
    content is the exact source projection, validate its artifact against a replay.
    """

    schema_id: Literal["marl_battlegrounds.evaluation.actor_pov_content"] = (
        ACTOR_POV_CONTENT_SCHEMA_ID
    )
    schema_version: _SchemaVersionV2 = 2
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
    axis_mapping: ActorPovAxisMappingV2
    completion: ActorPovEpisodeCompletionV1
    frames: tuple[ActorPovFrameV2, ...]
    transitions: tuple[ActorPovTransitionV1, ...]

    @model_validator(mode="after")
    def _validate_content(self) -> ActorPovReplayContentV2:
        """Return this content after the shared prefix, identity, completion, topology,
        and digest checks; otherwise raise ValueError.
        """
        return _validate_replaycontent(self)


def _validate_replayartifact[T: ActorPovReplayArtifactV1 | ActorPovReplayArtifactV2](
    self: T,
) -> T:
    """Return the envelope after checking canonical artifact ID/digest and the
    source/content episode join. Raise ValueError for mismatches; matching source
    bytes require the separate against-replay validator.
    """
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


class ActorPovReplayArtifactV1(EvaluationModel):
    """Wrap version-1 actor content with its full source replay reference.

    Attributes
    ----------
    schema_id : Literal['marl_battlegrounds.evaluation.actor_pov_artifact']
        Fixed actor POV artifact schema identifier.
    schema_version : _SchemaVersionV1
        Exact integer 1, the default.
    artifact_id : _AsciiIdentifier
        Canonical episode/actor artifact identity.
    canonical_digest_sha256 : _Sha256Hex
        SHA-256 of the full canonical envelope excluding this field.
    source_replay : ReplayArtifactReferenceV1
        Reference to replay V1, including its identities, digests, and byte length.
    content : ActorPovReplayContentV1
        Version-1 actor-authorized replay content.

    Notes
    -----
    Source provenance can change while actor content remains equal. The envelope
    checks its own digest and episode join; against-replay validation is required
    to prove that the referenced source produced this exact content.
    """

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
        """Return this envelope after checking canonical artifact identity/digest and
        the source/content episode join; otherwise raise ValueError.
        """
        return _validate_replayartifact(self)


class ActorPovReplayArtifactV2(EvaluationModel):
    """Wrap version-2 actor content with its full source replay reference.

    Attributes
    ----------
    schema_id : Literal['marl_battlegrounds.evaluation.actor_pov_artifact']
        Fixed actor POV artifact schema identifier.
    schema_version : _SchemaVersionV2
        Exact integer 2, the default.
    artifact_id : _AsciiIdentifier
        Canonical episode/actor artifact identity.
    canonical_digest_sha256 : _Sha256Hex
        SHA-256 of the full canonical envelope excluding this field.
    source_replay : ReplayArtifactReferenceV3
        Reference to replay V3, including its identities, digests, and byte length.
    content : ActorPovReplayContentV2
        Version-2 actor-authorized replay content.

    Notes
    -----
    Source provenance can change while actor content remains equal. The envelope
    checks its own digest and episode join; against-replay validation is required
    to prove that the referenced source produced this exact content.
    """

    schema_id: Literal["marl_battlegrounds.evaluation.actor_pov_artifact"] = (
        ACTOR_POV_ARTIFACT_SCHEMA_ID
    )
    schema_version: _SchemaVersionV2 = 2
    artifact_id: _AsciiIdentifier
    canonical_digest_sha256: _Sha256Hex
    source_replay: ReplayArtifactReferenceV3
    content: ActorPovReplayContentV2

    @model_validator(mode="after")
    def _validate_artifact(self) -> ActorPovReplayArtifactV2:
        """Return this envelope after checking canonical artifact identity/digest and
        the source/content episode join; otherwise raise ValueError.
        """
        return _validate_replayartifact(self)


type ActorPovAxisMapping = ActorPovAxisMappingV1 | ActorPovAxisMappingV2

type ActorPovFrame = ActorPovFrameV1 | ActorPovFrameV2

type ActorPovCurrentSlice = ActorPovCurrentSliceV1 | ActorPovCurrentSliceV2

type ActorPovAdjacentTransitionSlice = (
    ActorPovAdjacentTransitionSliceV1 | ActorPovAdjacentTransitionSliceV2
)

type ActorPovReplayContent = ActorPovReplayContentV1 | ActorPovReplayContentV2

type ActorPovReplayArtifact = ActorPovReplayArtifactV1 | ActorPovReplayArtifactV2


def _actor_class_ids(
    context: EvaluationEpisodeContext, global_slot: int
) -> tuple[tuple[int, ...], ...] | None:
    """Read public roster classes in the selected actor's team order.

    Parameters
    ----------
    context : EvaluationEpisodeContext
        Valid evaluation context. Historical versions return None.
    global_slot : int
        Selected simulator slot, from 0 through 9.

    Returns
    -------
    tuple[tuple[int, ...], ...] | None
        For context V3, immutable integer tuples of shape (2, 5), Own Team first;
        zero marks inactive slots. Otherwise None.
    """
    if context.schema_version != 3:
        return None
    from marl_battlegrounds.evaluation.actor_projection import (
        reconstruct_actor_class_ids_by_team_v3,
    )

    return tuple(
        tuple(int(value) for value in team)
        for team in reconstruct_actor_class_ids_by_team_v3(context, global_slot)
    )


def _build_replay_reference_from_validated(
    replay: ReplayArtifactV1 | ReplayArtifactV3,
) -> ReplayArtifactReferenceV1 | ReplayArtifactReferenceV3:
    """Build a source reference without repeating full replay validation.

    Parameters
    ----------
    replay : ReplayArtifactV1 | ReplayArtifactV3
        Already validated exact replay V1 or V3.

    Returns
    -------
    ReplayArtifactReferenceV1 | ReplayArtifactReferenceV3
        Version-matched reference with source identities, digests, and canonical byte
        length.

    Notes
    -----
    Serializes canonical bytes to measure their length; writes no file.
    """
    if type(replay) is ReplayArtifactV3:
        return replay_reference_v3(replay)
    return ReplayArtifactReferenceV1(
        artifact_id=replay.artifact_id,
        episode_id=replay.header.context.identity.episode_id,
        context_digest_sha256=replay.header.context_digest_sha256,
        trajectory_content_digest_sha256=replay.trajectory_content_digest_sha256,
        canonical_digest_sha256=replay.canonical_digest_sha256,
        canonical_byte_length=len(canonical_json_bytes(replay)),
    )


def _slice_frame_from_source(
    source: EvaluationFrame,
    *,
    global_slot: int,
    public_agent_id: str,
    class_ids_by_team: tuple[tuple[int, ...], ...] | None = None,
) -> ActorPovFrame:
    """Copy one actor row from an already validated evaluation frame.

    Parameters
    ----------
    source : EvaluationFrame
        Evaluation frame V1 or V2 containing simulator-global actor rows.
    global_slot : int
        Selected simulator slot from 0 through 9.
    public_agent_id : str
        Matching public roster identity used in POV IDs.
    class_ids_by_team : tuple[tuple[int, ...], ...] | None
        Public class tuples of shape (2, 5), Own Team first. Required by V2; default
        None suits V1.

    Returns
    -------
    ActorPovFrame
        A new POV V1/V2 frame with the matching feature schema and no privileged
        snapshot.

    Raises
    ------
    ValueError
        Selected fields fail the POV record contract.

    Notes
    -----
    The caller owns source/context/slot coherence. Visibility rows are copied
    exactly; this helper does not expand or infer observations.
    """
    observation = source.base_observation
    previous = observation.previous_timestep_actions
    lifecycle = observation.spawn_lifecycle
    mask = source.action_mask
    frame_type = (
        ActorPovFrameV2 if type(source) is EvaluationFrameV2 else ActorPovFrameV1
    )
    current_fields: dict[str, object] = {}
    if type(source) is EvaluationFrameV2:
        current_fields = {
            "self_ally_index": source.base_observation.self_ally_index[global_slot],
            "class_ids_by_team": class_ids_by_team,
        }
    return frame_type.model_validate(
        dict(
            **current_fields,
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
                    lifecycle.spawn_shield_actual_durations_by_agent_by_team[
                        global_slot
                    ]
                ),
                spawn_shield_configured_duration=(
                    lifecycle.spawn_shield_configured_duration_by_agent[global_slot]
                ),
                spawn_shield_speed=(lifecycle.spawn_shield_speed_by_agent[global_slot]),
                respawn_wave_period_step_count_by_team=(
                    lifecycle.respawn_wave_period_step_count_by_agent_by_team[
                        global_slot
                    ]
                ),
                respawn_wave_countdowns_by_team=(
                    lifecycle.respawn_wave_countdowns_by_agent_by_team[global_slot]
                ),
                active_mask_by_team=(
                    lifecycle.active_mask_by_agent_by_team[global_slot]
                ),
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
    )


def _slice_frame(
    replay: ReplayArtifactV1 | ReplayArtifactV3,
    *,
    global_slot: int,
    public_agent_id: str,
    frame_index: int,
) -> ActorPovFrame:
    """Read and slice a frame from a validated replay.

    Parameters
    ----------
    replay : ReplayArtifactV1 | ReplayArtifactV3
        Valid replay V1 or V3.
    global_slot : int
        Selected simulator slot from 0 through 9.
    public_agent_id : str
        Matching public actor identity.
    frame_index : int
        Existing zero-based replay frame position.

    Returns
    -------
    ActorPovFrame
        Version-matched selected-actor frame, including current public classes when
        applicable.

    Raises
    ------
    IndexError
        The requested frame or actor row is outside the stored sequence.
    """
    return _slice_frame_from_source(
        replay.frames[frame_index],
        global_slot=global_slot,
        public_agent_id=public_agent_id,
        class_ids_by_team=_actor_class_ids(replay.header.context, global_slot),
    )


class _CueIdentity(TypedDict):
    """Hold cue_id, pov_transition_id, and zero-based ordinal before model construction.

    All three fields are required. The two strings are canonical public identities;
    ordinal counts cues already appended to the transition.
    """

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
    """Build the identity fields for a cue at a chosen ordinal.

    Parameters
    ----------
    transition : ActorPovTransitionV1 | None
        Existing POV transition whose ID takes precedence; None builds the ID from the
        supplied episode/actor/index.
    episode_id : str
        Public episode ID used when transition is None.
    public_agent_id : str
        Public actor ID used when transition is None.
    transition_index : int
        Zero-based transition position used when transition is None.
    ordinal : int
        Zero-based cue position within that transition.

    Returns
    -------
    _CueIdentity
        A new dictionary with cue_id, pov_transition_id, and ordinal. Inputs are not
        validated.
    """
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
    start_frame: ActorPovFrame,
    successor_frame: ActorPovFrame,
    has_any_rejection: bool,
    terminated: bool,
    truncated: bool,
    public_end_reason: str | None,
) -> tuple[ActorPovPresentationCueV1, ...]:
    """Derive an ordered cue tuple using only the selected actor's endpoints.

    Parameters
    ----------
    episode_id : str
        Public episode identity used in cue IDs.
    public_agent_id : str
        Selected actor identity used in cue IDs.
    transition_index : int
        Zero-based transition position.
    team_local_slot : int
        Selected actor's Own Team slot, from 0 through 4.
    start_frame : ActorPovFrame
        Selected actor's decision-frame observation.
    successor_frame : ActorPovFrame
        Same actor's immediately following observation.
    has_any_rejection : bool
        Whether any submitted action component was rejected.
    terminated : bool
        Recorded task termination flag for this transition.
    truncated : bool
        Recorded truncation flag for this transition.
    public_end_reason : str | None
        Optional recorded public reason; None leaves it absent.

    Returns
    -------
    tuple[ActorPovPresentationCueV1, ...]
        Action outcome first, then changed own position, health, status, cooldown,
        lifecycle, visible ally/enemy rows, and finally a done cue when applicable.

    Notes
    -----
    The caller supplies coherent endpoints. Adjacent live carriers zero hidden
    relation rows before calling. No cause is inferred from privileged events.
    """
    cues: list[ActorPovPresentationCueV1] = []

    def identity() -> _CueIdentity:
        """Return canonical IDs for the next cue using the current cue-list length as
        its ordinal.
        """
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
    start_frame: ActorPovFrame,
    successor_frame: ActorPovFrame,
) -> ActorPovTransitionV1:
    """Extract this actor's transition fields and derive its local cues.

    Parameters
    ----------
    source : EvaluationTransitionV1
        Valid source transition with simulator-global action and reward arrays.
    global_slot : int
        Selected simulator actor slot, from 0 through 9.
    team_local_slot : int
        Matching Own Team slot, from 0 through 4.
    public_agent_id : str
        Matching public actor identity.
    start_frame : ActorPovFrame
        Already sliced observation at the transition start.
    successor_frame : ActorPovFrame
        Already sliced observation after the transition.

    Returns
    -------
    ActorPovTransitionV1
        New POV transition with this actor's actions, rejection flags, reward, and cues.

    Notes
    -----
    Source/endpoint coherence is a caller precondition. Other actors' reward or
    action rows and the privileged event feed are not copied.
    """
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
    replay: ReplayArtifactV1 | ReplayArtifactV3,
    *,
    global_slot: int,
    team_local_slot: int,
    public_agent_id: str,
    transition_index: int,
    frames: tuple[ActorPovFrame, ...],
) -> ActorPovTransitionV1:
    """Slice a stored transition using an already sliced actor-frame prefix.

    Parameters
    ----------
    replay : ReplayArtifactV1 | ReplayArtifactV3
        Valid replay V1 or V3.
    global_slot : int
        Selected simulator slot from 0 through 9.
    team_local_slot : int
        Matching Own Team slot from 0 through 4.
    public_agent_id : str
        Matching public actor identity.
    transition_index : int
        Existing zero-based transition position.
    frames : tuple[ActorPovFrame, ...]
        Complete ordered selected-actor frames; includes both adjacent endpoints.

    Returns
    -------
    ActorPovTransitionV1
        Selected actor's POV transition.

    Raises
    ------
    IndexError
        The transition or either sliced endpoint is missing.
    """
    return _slice_transition_from_source(
        replay.transitions[transition_index],
        global_slot=global_slot,
        team_local_slot=team_local_slot,
        public_agent_id=public_agent_id,
        start_frame=frames[transition_index],
        successor_frame=frames[transition_index + 1],
    )


def _axis_mapping_from_context(
    context: EvaluationEpisodeContext,
    *,
    global_slot: int,
) -> ActorPovAxisMapping:
    """Resolve public IDs and action labels for one actor's observation axes.

    Parameters
    ----------
    context : EvaluationEpisodeContext
        Valid context with a supported actor projection and mechanics catalog.
    global_slot : int
        Selected simulator slot from 0 through 9.

    Returns
    -------
    ActorPovAxisMapping
        Version-matched immutable axis record; target zero is None, followed by five
        allies and five enemies.

    Raises
    ------
    ValueError
        The projection or constructed axis record is unsupported.
    """
    _require_actor_projection_v1(context)
    catalog = context.static_mechanics_catalog
    ally_slots = catalog.global_slot_by_actor_and_ally_observation_row[global_slot]
    enemy_slots = catalog.global_slot_by_actor_and_enemy_observation_row[global_slot]
    target_slots = catalog.global_recipient_slot_by_actor_and_target_action[global_slot]

    def public_id(slot: int) -> str:
        """Return the public roster identity for an already bounded simulator-global
        slot.
        """
        return context.roster[slot].public_agent_id

    axis_type = (
        ActorPovAxisMappingV2 if context.schema_version == 3 else ActorPovAxisMappingV1
    )
    return axis_type.model_validate(
        dict(
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
    )


def _axis_mapping_from_replay(
    replay: ReplayArtifactV1 | ReplayArtifactV3,
    *,
    global_slot: int,
) -> ActorPovAxisMapping:
    """Read the selected actor's axis record from a validated replay.

    Parameters
    ----------
    replay : ReplayArtifactV1 | ReplayArtifactV3
        Replay V1 or V3 with validated context and mechanics catalog.
    global_slot : int
        Selected simulator slot from 0 through 9.

    Returns
    -------
    ActorPovAxisMapping
        The same version-matched axis mapping as _axis_mapping_from_context.
    """
    return _axis_mapping_from_context(
        replay.header.context,
        global_slot=global_slot,
    )


def _require_actor_projection_v1(context: EvaluationEpisodeContext) -> None:
    """Require the projection version supported by this context's POV path.

    Context V3 requires actor projection version 3; older contexts require version 1.
    Return None when supported; otherwise raise ValueError. Information mode is
    checked separately by the selected-actor/export boundary.
    """
    if context.actor_projection.version != (
        3 if context.schema_version == 3 else ACTOR_POV_SCHEMA_VERSION
    ):
        raise ValueError("actor POV V1 requires actor projection version 1")


def _require_selected_self_topology(
    frame: ActorPovFrame,
    *,
    configured_team_id: int,
    class_id: int,
) -> None:
    """Check the selected self row against public roster identity.

    Parameters
    ----------
    frame : ActorPovFrame
        Sliced POV V1 or V2 frame.
    configured_team_id : int
        Simulator Team A/B ID, respectively 1 or 2.
    class_id : int
        Selected public class ID from 1 through 5.

    Returns
    -------
    None
        None.

    Raises
    ------
    ValueError
        ACTIVE is not one, ALIVE is not binary, or team/class features disagree. V2 uses
        actor-relative self relation zero.
    """
    row = frame.self_features
    if row[_FEATURE_ACTIVE] != 1.0:
        raise ValueError("configured-active POV self rows require ACTIVE=1")
    if row[_FEATURE_ALIVE] not in (0.0, 1.0):
        raise ValueError("POV self ALIVE must be exactly zero or one")
    if row[_FEATURE_TEAM_ID] != (
        float(configured_team_id) if frame.schema_version == 1 else 0.0
    ):
        raise ValueError("POV self team feature must match selected roster metadata")
    if row[_FEATURE_CLASS_ID] != float(class_id):
        raise ValueError("POV self class feature must match selected roster metadata")


def _selected_pov_roster_row(
    context: EvaluationEpisodeContext,
    *,
    global_slot: int,
) -> RosterSlotV1:
    """Select a configured-active actor from supported NoSharedObs source material.

    Parameters
    ----------
    context : EvaluationEpisodeContext
        Valid context whose roster and projection are authoritative.
    global_slot : int
        Exact Python integer from 0 through 9; booleans are rejected.

    Returns
    -------
    RosterSlotV1
        Matching public roster row. A dead configured-active actor remains selectable.

    Raises
    ------
    ValueError
        The slot, projection, information mode, or configured activity is unsupported.
    """
    _require_actor_projection_v1(context)
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


@overload
def build_actor_pov_current_slice_v1(
    context: EvaluationEpisodeContext,
    frame: EvaluationFrameV1,
    *,
    global_slot: int,
    incoming_transition_view: EvaluationTransitionViewV1 | None = None,
) -> ActorPovCurrentSliceV1:
    """Build the selected actor's current observation and incoming transition.

    Parameters
    ----------
    context : EvaluationEpisodeContext
        Valid evaluation episode context. Only supported NoSharedObs projections are
        accepted.
    frame : EvaluationFrame
        Decision frame V1 or V2 from that context. Without an incoming view it must be
        the initial frame.
    global_slot : int
        Exact Python integer from 0 through 9 selecting a configured-active actor; dead
        actors are allowed.
    incoming_transition_view : EvaluationTransitionViewV1 | None
        None for the initial frame. For later frames, the exact
        EvaluationTransitionViewV1 whose successor equals frame and whose context equals
        context.

    Returns
    -------
    ActorPovCurrentSlice
        New ActorPovCurrentSliceV1 for frame V1 or ActorPovCurrentSliceV2 for frame V2.
        Frame zero has no incoming transition; later frames include only their own
        incoming actor transition, not a reconstructed earlier prefix.

    Raises
    ------
    TypeError
        A source model or incoming view is not the exact supported type.
    ValueError
        Source records, actor selection, projection, or time/identity joins fail
        validation.

    Notes
    -----
    Host-only immutable projection; does not run the simulator or write a file.
    The incoming view is reconstructed using EvaluationTransitionViewV1's actual
    version-specific validation. V1 views receive full legacy checks; later view
    versions have narrower structural checks. This function additionally checks
    selected-actor topology and exact context/successor equality.
    Only this actor's base inputs, actions, rejection flags, reward, public done
    flags, and locally derived cues enter the result.
    """
    ...


@overload
def build_actor_pov_current_slice_v1(
    context: EvaluationEpisodeContext,
    frame: EvaluationFrameV2,
    *,
    global_slot: int,
    incoming_transition_view: EvaluationTransitionViewV1 | None = None,
) -> ActorPovCurrentSliceV2:
    """Build the selected actor's current observation and incoming transition.

    Parameters
    ----------
    context : EvaluationEpisodeContext
        Valid evaluation episode context. Only supported NoSharedObs projections are
        accepted.
    frame : EvaluationFrame
        Decision frame V1 or V2 from that context. Without an incoming view it must be
        the initial frame.
    global_slot : int
        Exact Python integer from 0 through 9 selecting a configured-active actor; dead
        actors are allowed.
    incoming_transition_view : EvaluationTransitionViewV1 | None
        None for the initial frame. For later frames, the exact
        EvaluationTransitionViewV1 whose successor equals frame and whose context equals
        context.

    Returns
    -------
    ActorPovCurrentSlice
        New ActorPovCurrentSliceV1 for frame V1 or ActorPovCurrentSliceV2 for frame V2.
        Frame zero has no incoming transition; later frames include only their own
        incoming actor transition, not a reconstructed earlier prefix.

    Raises
    ------
    TypeError
        A source model or incoming view is not the exact supported type.
    ValueError
        Source records, actor selection, projection, or time/identity joins fail
        validation.

    Notes
    -----
    Host-only immutable projection; does not run the simulator or write a file.
    The incoming view is reconstructed using EvaluationTransitionViewV1's actual
    version-specific validation. V1 views receive full legacy checks; later view
    versions have narrower structural checks. This function additionally checks
    selected-actor topology and exact context/successor equality.
    Only this actor's base inputs, actions, rejection flags, reward, public done
    flags, and locally derived cues enter the result.
    """
    ...


@overload
def build_actor_pov_current_slice_v1(
    context: EvaluationEpisodeContext,
    frame: EvaluationFrame,
    *,
    global_slot: int,
    incoming_transition_view: EvaluationTransitionViewV1 | None = None,
) -> ActorPovCurrentSlice:
    """Build the selected actor's current observation and incoming transition.

    Parameters
    ----------
    context : EvaluationEpisodeContext
        Valid evaluation episode context. Only supported NoSharedObs projections are
        accepted.
    frame : EvaluationFrame
        Decision frame V1 or V2 from that context. Without an incoming view it must be
        the initial frame.
    global_slot : int
        Exact Python integer from 0 through 9 selecting a configured-active actor; dead
        actors are allowed.
    incoming_transition_view : EvaluationTransitionViewV1 | None
        None for the initial frame. For later frames, the exact
        EvaluationTransitionViewV1 whose successor equals frame and whose context equals
        context.

    Returns
    -------
    ActorPovCurrentSlice
        New ActorPovCurrentSliceV1 for frame V1 or ActorPovCurrentSliceV2 for frame V2.
        Frame zero has no incoming transition; later frames include only their own
        incoming actor transition, not a reconstructed earlier prefix.

    Raises
    ------
    TypeError
        A source model or incoming view is not the exact supported type.
    ValueError
        Source records, actor selection, projection, or time/identity joins fail
        validation.

    Notes
    -----
    Host-only immutable projection; does not run the simulator or write a file.
    The incoming view is reconstructed using EvaluationTransitionViewV1's actual
    version-specific validation. V1 views receive full legacy checks; later view
    versions have narrower structural checks. This function additionally checks
    selected-actor topology and exact context/successor equality.
    Only this actor's base inputs, actions, rejection flags, reward, public done
    flags, and locally derived cues enter the result.
    """
    ...


def build_actor_pov_current_slice_v1(
    context: EvaluationEpisodeContext,
    frame: EvaluationFrame,
    *,
    global_slot: int,
    incoming_transition_view: EvaluationTransitionViewV1 | None = None,
) -> ActorPovCurrentSlice:
    """Build the selected actor's current observation and incoming transition.

    Parameters
    ----------
    context : EvaluationEpisodeContext
        Valid evaluation episode context. Only supported NoSharedObs projections are
        accepted.
    frame : EvaluationFrame
        Decision frame V1 or V2 from that context. Without an incoming view it must be
        the initial frame.
    global_slot : int
        Exact Python integer from 0 through 9 selecting a configured-active actor; dead
        actors are allowed.
    incoming_transition_view : EvaluationTransitionViewV1 | None
        None for the initial frame. For later frames, the exact
        EvaluationTransitionViewV1 whose successor equals frame and whose context equals
        context.

    Returns
    -------
    ActorPovCurrentSlice
        New ActorPovCurrentSliceV1 for frame V1 or ActorPovCurrentSliceV2 for frame V2.
        Frame zero has no incoming transition; later frames include only their own
        incoming actor transition, not a reconstructed earlier prefix.

    Raises
    ------
    TypeError
        A source model or incoming view is not the exact supported type.
    ValueError
        Source records, actor selection, projection, or time/identity joins fail
        validation.

    Notes
    -----
    Host-only immutable projection; does not run the simulator or write a file.
    The incoming view is reconstructed using EvaluationTransitionViewV1's actual
    version-specific validation. V1 views receive full legacy checks; later view
    versions have narrower structural checks. This function additionally checks
    selected-actor topology and exact context/successor equality.
    Only this actor's base inputs, actions, rejection flags, reward, public done
    flags, and locally derived cues enter the result.
    """
    canonical_context = cast(
        EvaluationEpisodeContext,
        validate_declared_model_tree(
            context,
            record_name="actor POV current context",
            expected_type=evaluation_context_type(context),
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
            class_ids_by_team=_actor_class_ids(canonical_view.context, global_slot),
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
            class_ids_by_team=_actor_class_ids(canonical_context, global_slot),
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
        class_ids_by_team=_actor_class_ids(canonical_context, global_slot),
    )
    _require_selected_self_topology(
        current_frame,
        configured_team_id=roster.configured_team_id,
        class_id=roster.class_id,
    )
    slice_type = (
        ActorPovCurrentSliceV2
        if type(canonical_frame) is EvaluationFrameV2
        else ActorPovCurrentSliceV1
    )
    return slice_type.model_validate(
        dict(
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
    )


def build_actor_pov_adjacent_transition_slice_v1(
    transition_view: EvaluationTransitionViewV1,
    *,
    global_slot: int,
) -> ActorPovAdjacentTransitionSlice:
    """Build one actor's transition with both exact allowed observation endpoints.

    Parameters
    ----------
    transition_view : EvaluationTransitionViewV1
        Exact EvaluationTransitionViewV1 joining a context, start frame, transition, and
        successor frame.
    global_slot : int
        Exact Python integer from 0 through 9 selecting a configured-active NoSharedObs
        actor; dead actors remain selectable.

    Returns
    -------
    ActorPovAdjacentTransitionSlice
        New ActorPovAdjacentTransitionSliceV1 for source frame V1, or V2 for frame V2.
        Contains start_frame, transition, successor_frame, public identity, and actor
        axes.

    Raises
    ------
    TypeError
        transition_view is not the exact supported view type.
    ValueError
        The source view, actor selection, projection, endpoint join, or locally derived
        cues are invalid.

    Notes
    -----
    Host-only and immutable. Reconstructs the source view with its version-specific
    checks, then checks selected self/lifecycle truth and adjacent simulator ticks.
    Hidden relation rows are zeroed for cue comparison; the returned endpoints
    retain their exact source values. No privileged events, earlier prefix, source
    provenance root, or completion claim is carried. No file is written.
    """
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
        class_ids_by_team=_actor_class_ids(canonical_view.context, global_slot),
    )
    successor = _slice_frame_from_source(
        canonical_view.successor_frame,
        global_slot=global_slot,
        public_agent_id=public_agent_id,
        class_ids_by_team=_actor_class_ids(canonical_view.context, global_slot),
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
    slice_type = (
        ActorPovAdjacentTransitionSliceV2
        if type(canonical_view.start_frame) is EvaluationFrameV2
        else ActorPovAdjacentTransitionSliceV1
    )
    return slice_type.model_validate(
        dict(
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
    )


@overload
def slice_actor_pov_current_frame_v1(
    context: EvaluationEpisodeContext,
    frame: EvaluationFrameV1,
    *,
    global_slot: int,
    incoming_transition_view: EvaluationTransitionViewV1 | None = None,
) -> ActorPovFrameV1:
    """Return only the selected actor's current observation frame.

    Parameters
    ----------
    context : EvaluationEpisodeContext
        Valid evaluation episode context. Only supported NoSharedObs projections are
        accepted.
    frame : EvaluationFrame
        Decision frame V1 or V2 from that context. Without an incoming view it must be
        the initial frame.
    global_slot : int
        Exact Python integer from 0 through 9 selecting a configured-active actor; dead
        actors are allowed.
    incoming_transition_view : EvaluationTransitionViewV1 | None
        None for the initial frame. For later frames, the exact
        EvaluationTransitionViewV1 whose successor equals frame and whose context equals
        context.

    Returns
    -------
    ActorPovFrame
        New ActorPovFrameV1 for source frame V1 or ActorPovFrameV2 for source frame V2.
        The current frame includes authorized observations, masks, and lifecycle data.

    Raises
    ------
    TypeError
        A source model or incoming view is not the exact supported type.
    ValueError
        Source records, actor selection, projection, or time/identity joins fail
        validation.

    Notes
    -----
    Uses build_actor_pov_current_slice_v1 and its host-side source checks, including
    the required incoming view after frame zero. Discards the carrier's axis metadata
    and incoming transition from the return value. Does not change inputs or write
    files.
    """
    ...


@overload
def slice_actor_pov_current_frame_v1(
    context: EvaluationEpisodeContext,
    frame: EvaluationFrameV2,
    *,
    global_slot: int,
    incoming_transition_view: EvaluationTransitionViewV1 | None = None,
) -> ActorPovFrameV2:
    """Return only the selected actor's current observation frame.

    Parameters
    ----------
    context : EvaluationEpisodeContext
        Valid evaluation episode context. Only supported NoSharedObs projections are
        accepted.
    frame : EvaluationFrame
        Decision frame V1 or V2 from that context. Without an incoming view it must be
        the initial frame.
    global_slot : int
        Exact Python integer from 0 through 9 selecting a configured-active actor; dead
        actors are allowed.
    incoming_transition_view : EvaluationTransitionViewV1 | None
        None for the initial frame. For later frames, the exact
        EvaluationTransitionViewV1 whose successor equals frame and whose context equals
        context.

    Returns
    -------
    ActorPovFrame
        New ActorPovFrameV1 for source frame V1 or ActorPovFrameV2 for source frame V2.
        The current frame includes authorized observations, masks, and lifecycle data.

    Raises
    ------
    TypeError
        A source model or incoming view is not the exact supported type.
    ValueError
        Source records, actor selection, projection, or time/identity joins fail
        validation.

    Notes
    -----
    Uses build_actor_pov_current_slice_v1 and its host-side source checks, including
    the required incoming view after frame zero. Discards the carrier's axis metadata
    and incoming transition from the return value. Does not change inputs or write
    files.
    """
    ...


@overload
def slice_actor_pov_current_frame_v1(
    context: EvaluationEpisodeContext,
    frame: EvaluationFrame,
    *,
    global_slot: int,
    incoming_transition_view: EvaluationTransitionViewV1 | None = None,
) -> ActorPovFrame:
    """Return only the selected actor's current observation frame.

    Parameters
    ----------
    context : EvaluationEpisodeContext
        Valid evaluation episode context. Only supported NoSharedObs projections are
        accepted.
    frame : EvaluationFrame
        Decision frame V1 or V2 from that context. Without an incoming view it must be
        the initial frame.
    global_slot : int
        Exact Python integer from 0 through 9 selecting a configured-active actor; dead
        actors are allowed.
    incoming_transition_view : EvaluationTransitionViewV1 | None
        None for the initial frame. For later frames, the exact
        EvaluationTransitionViewV1 whose successor equals frame and whose context equals
        context.

    Returns
    -------
    ActorPovFrame
        New ActorPovFrameV1 for source frame V1 or ActorPovFrameV2 for source frame V2.
        The current frame includes authorized observations, masks, and lifecycle data.

    Raises
    ------
    TypeError
        A source model or incoming view is not the exact supported type.
    ValueError
        Source records, actor selection, projection, or time/identity joins fail
        validation.

    Notes
    -----
    Uses build_actor_pov_current_slice_v1 and its host-side source checks, including
    the required incoming view after frame zero. Discards the carrier's axis metadata
    and incoming transition from the return value. Does not change inputs or write
    files.
    """
    ...


def slice_actor_pov_current_frame_v1(
    context: EvaluationEpisodeContext,
    frame: EvaluationFrame,
    *,
    global_slot: int,
    incoming_transition_view: EvaluationTransitionViewV1 | None = None,
) -> ActorPovFrame:
    """Return only the selected actor's current observation frame.

    Parameters
    ----------
    context : EvaluationEpisodeContext
        Valid evaluation episode context. Only supported NoSharedObs projections are
        accepted.
    frame : EvaluationFrame
        Decision frame V1 or V2 from that context. Without an incoming view it must be
        the initial frame.
    global_slot : int
        Exact Python integer from 0 through 9 selecting a configured-active actor; dead
        actors are allowed.
    incoming_transition_view : EvaluationTransitionViewV1 | None
        None for the initial frame. For later frames, the exact
        EvaluationTransitionViewV1 whose successor equals frame and whose context equals
        context.

    Returns
    -------
    ActorPovFrame
        New ActorPovFrameV1 for source frame V1 or ActorPovFrameV2 for source frame V2.
        The current frame includes authorized observations, masks, and lifecycle data.

    Raises
    ------
    TypeError
        A source model or incoming view is not the exact supported type.
    ValueError
        Source records, actor selection, projection, or time/identity joins fail
        validation.

    Notes
    -----
    Uses build_actor_pov_current_slice_v1 and its host-side source checks, including
    the required incoming view after frame zero. Discards the carrier's axis metadata
    and incoming transition from the return value. Does not change inputs or write
    files.
    """
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
    """Return one actor's incoming transition from a coherent source view.

    Parameters
    ----------
    transition_view : EvaluationTransitionViewV1
        Source EvaluationTransitionViewV1 with context and adjacent endpoints.
    global_slot : int
        Exact Python integer from 0 through 9 selecting a configured-active NoSharedObs
        actor.

    Returns
    -------
    ActorPovTransitionV1
        ActorPovTransitionV1 containing own actions, rejections, reward, public done
        flags, and cues.

    Raises
    ------
    TypeError
        Source records do not have the supported exact types.
    ValueError
        The selected actor, projection, context, or incoming endpoint join is invalid.
    AssertionError
        The validated current-slice builder unexpectedly returns no incoming transition.

    Notes
    -----
    Uses build_actor_pov_current_slice_v1's host-side checks. Does not retain other
    actors' action/reward rows or write files.
    """
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
    replay: ReplayArtifactV1 | ReplayArtifactV3,
    *,
    global_slot: int,
) -> ActorPovReplayArtifact:
    """Build actor content and its provenance envelope from a validated replay.

    Parameters
    ----------
    replay : ReplayArtifactV1 | ReplayArtifactV3
        Already validated exact replay V1 or V3.
    global_slot : int
        Exact Python integer from 0 through 9 selecting a configured-active actor.

    Returns
    -------
    ActorPovReplayArtifact
        POV V1 for replay V1, or POV V2 for replay V3, with a complete stored prefix,
        copied completion metadata, separate content/envelope hashes, and source
        reference.

    Raises
    ------
    ValueError
        The slot, projection, information mode, topology, or constructed record is
        invalid.

    Notes
    -----
    Host-only serialization work. No input is changed and no file is written.
    SharedObs is rejected. Public export helpers own full source validation.
    """
    context = replay.header.context
    _require_actor_projection_v1(context)
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
    current = type(replay) is ReplayArtifactV3
    content_type = ActorPovReplayContentV2 if current else ActorPovReplayContentV1
    artifact_type = ActorPovReplayArtifactV2 if current else ActorPovReplayArtifactV1
    content_payload: dict[str, object] = {
        "schema_id": ACTOR_POV_CONTENT_SCHEMA_ID,
        "schema_version": 2 if current else ACTOR_POV_SCHEMA_VERSION,
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
    content = content_type.model_validate(
        {
            **content_payload,
            "canonical_digest_sha256": canonical_digest_sha256(content_payload),
        }
    )
    artifact_payload: dict[str, object] = {
        "schema_id": ACTOR_POV_ARTIFACT_SCHEMA_ID,
        "schema_version": 2 if current else ACTOR_POV_SCHEMA_VERSION,
        "artifact_id": f"{episode_id}:actor-pov:{public_agent_id}",
        "source_replay": _build_replay_reference_from_validated(replay),
        "content": content,
    }
    return artifact_type.model_validate(
        {
            **artifact_payload,
            "canonical_digest_sha256": canonical_digest_sha256(artifact_payload),
        }
    )


def validate_actor_pov_replay_content_v1(
    content: ActorPovReplayContentV1,
) -> None:
    """Validate historical actor content and rederive every local cue.

    Parameters
    ----------
    content : ActorPovReplayContentV1
        Exact ActorPovReplayContentV1 with its complete declared nested model tree.

    Returns
    -------
    None
        None when fixed shapes, canonical identities/digest, T+1/T structure,
        completion,
        and each cue tuple agree.

    Raises
    ------
    TypeError
        A root or nested model does not have its declared exact type.
    ValueError
        A structural, digest, temporal, or cue-derivation check fails.

    Notes
    -----
    Host-only standalone validation. Does not prove equality to a full source
    replay, read a file, or infer hidden causes for observed changes.
    """
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
    """Validate a standalone POV V1 envelope and its local actor content.

    Parameters
    ----------
    artifact : ActorPovReplayArtifactV1
        Exact ActorPovReplayArtifactV1, including its declared nested model types.

    Returns
    -------
    None
        None when canonical structure, IDs, digests, completion, and local cue
        derivation agree.

    Raises
    ------
    TypeError
        The root or a declared nested model is not the exact expected type.
    ValueError
        Stored values, hashes, temporal joins, or cues are inconsistent.

    Notes
    -----
    Host-only; does not read the referenced replay or write files. To establish
    source correspondence, use validate_actor_pov_replay_against_replay_v1.
    """
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
    """Export a configured-active actor's exact NoSharedObs view from replay V1.

    Parameters
    ----------
    replay : ReplayArtifactV1
        Exact ReplayArtifactV1; the full source is validated before export.
    global_slot : int
        Exact Python integer from 0 through 9; booleans and inactive roster slots are
        rejected. A dead configured-active actor is allowed.

    Returns
    -------
    ActorPovReplayArtifactV1
        New ActorPovReplayArtifactV1 with T+1 selected-actor frames, T own transitions,
        public completion metadata, an actor-content digest, and a separate source
        provenance envelope/digest.

    Raises
    ------
    TypeError
        replay is not the exact supported replay type.
    ValueError
        Replay validation, projection/mode, actor selection, topology, or result
        validation fails.

    Notes
    -----
    Host-only; builds and validates immutable models and canonical hashes. No input
    is changed and no file is written. SharedObs exports are unavailable. Historical V1
    feature semantics are preserved.
    Use replay_io for storage. Export cost grows with the retained trajectory.
    """
    if type(replay) is not ReplayArtifactV1:
        raise TypeError("actor POV export requires ReplayArtifactV1")
    validate_replay_artifact_v1(replay)
    artifact = cast(
        ActorPovReplayArtifactV1,
        _export_from_validated_replay(replay, global_slot=global_slot),
    )
    validate_actor_pov_replay_artifact_v1(artifact)
    return artifact


def validate_actor_pov_replay_against_replay_v1(
    artifact: ActorPovReplayArtifactV1,
    replay: ReplayArtifactV1,
) -> None:
    """Prove that this POV artifact equals the selected source-replay export.

    Parameters
    ----------
    artifact : ActorPovReplayArtifactV1
        ActorPovReplayArtifactV1 to compare, including its selected global slot.
    replay : ReplayArtifactV1
        Exact ReplayArtifactV1 declared as the source.

    Returns
    -------
    None
        None when the entire artifact equals the independently rebuilt source
        projection.

    Raises
    ------
    TypeError
        The source replay or declared POV model tree has an unsupported exact type.
    ValueError
        Source validation, actor selection, or exact artifact comparison fails.

    Notes
    -----
    Host-only; recomputes the source projection and hashes without writing files.
    Validates the supplied POV artifact before rebuilding.
    """
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
    """Validate and serialize only the historical actor-authorized content.

    Parameters
    ----------
    content : ActorPovReplayContentV1
        Exact ActorPovReplayContentV1.

    Returns
    -------
    bytes
        Canonical compact, key-sorted UTF-8 JSON bytes including the content digest
        but no full-source provenance envelope.

    Raises
    ------
    TypeError
        A declared model type is unsupported.
    ValueError
        Content validation or canonical serialization fails.

    Notes
    -----
    Useful for comparing the actor's recorded inputs and local outputs across
    different full-source replays. Writes no file.
    """
    validate_actor_pov_replay_content_v1(content)
    return canonical_json_bytes(content)


def canonical_actor_pov_replay_json_bytes_v1(
    artifact: ActorPovReplayArtifactV1,
) -> bytes:
    """Validate and serialize the full POV V1 provenance envelope.

    Parameters
    ----------
    artifact : ActorPovReplayArtifactV1
        Exact ActorPovReplayArtifactV1 with valid local content.

    Returns
    -------
    bytes
        Canonical compact, key-sorted UTF-8 JSON bytes including content, its digest,
        and the full source replay reference. No file is written.

    Raises
    ------
    TypeError
        A declared model type is unsupported.
    ValueError
        Artifact validation or canonical serialization fails.

    Notes
    -----
    Envelope bytes can differ when full-source provenance differs even if the
    actor-authorized content is identical.
    """
    validate_actor_pov_replay_artifact_v1(artifact)
    return canonical_json_bytes(artifact)


def validate_actor_pov_replay_content(content: ActorPovReplayContent) -> None:
    """Validate either supported actor-content version and all local cues.

    Parameters
    ----------
    content : ActorPovReplayContent
        Exact ActorPovReplayContentV1 or ActorPovReplayContentV2.

    Returns
    -------
    None
        None when declared model types, IDs/digest, prefix structure, completion,
        and cues derived from the recorded endpoints agree.

    Raises
    ------
    TypeError
        Root or nested model types are unsupported.
    ValueError
        Stored values, joins, digest, or rederived cues are inconsistent.

    Notes
    -----
    Host-only. This checks actor content without a source provenance reference;
    against-replay validation is the separate source-correspondence check.
    """
    if type(content) not in (ActorPovReplayContentV1, ActorPovReplayContentV2):
        raise TypeError("actor POV content requires an exact supported root")
    validate_declared_model_tree(
        content, record_name="actor POV content", expected_type=type(content)
    )
    for index, transition in enumerate(content.transitions):
        expected = _derive_cues(
            episode_id=content.episode_id,
            public_agent_id=content.public_agent_id,
            transition_index=index,
            team_local_slot=content.selected_team_local_slot,
            start_frame=content.frames[index],
            successor_frame=content.frames[index + 1],
            has_any_rejection=(
                transition.submitted_action_tuple_is_out_of_domain
                or transition.in_domain_move_action_is_rejected
                or transition.in_domain_combat_action_pair_is_rejected
            ),
            terminated=transition.terminated,
            truncated=transition.truncated,
            public_end_reason=transition.public_end_reason,
        )
        if transition.cues != expected:
            raise ValueError("POV cues must be derived from their recorded endpoints")


def export_actor_pov_replay_v2(
    replay: ReplayArtifactV3, *, global_slot: int
) -> ActorPovReplayArtifactV2:
    """Export a configured-active actor's exact NoSharedObs view from replay V3.

    Parameters
    ----------
    replay : ReplayArtifactV3
        Exact ReplayArtifactV3; the full source is validated before export.
    global_slot : int
        Exact Python integer from 0 through 9; booleans and inactive roster slots are
        rejected. A dead configured-active actor is allowed.

    Returns
    -------
    ActorPovReplayArtifactV2
        New ActorPovReplayArtifactV2 with T+1 selected-actor frames, T own transitions,
        public completion metadata, an actor-content digest, and a separate source
        provenance envelope/digest.

    Raises
    ------
    TypeError
        replay is not the exact supported replay type.
    ValueError
        Replay validation, projection/mode, actor selection, topology, or result
        validation fails.

    Notes
    -----
    Host-only; builds and validates immutable models and canonical hashes. No input
    is changed and no file is written. SharedObs exports are unavailable. Current
    actor-relative feature flags and public class topology are preserved.
    Use replay_io for storage. Export cost grows with the retained trajectory.
    """
    from marl_battlegrounds.evaluation.replay_v3 import validate_replay_artifact_v3

    if type(replay) is not ReplayArtifactV3:
        raise TypeError("actor POV V2 export requires ReplayArtifactV3")
    validate_replay_artifact_v3(replay)
    artifact = _export_from_validated_replay(replay, global_slot=global_slot)
    if type(artifact) is not ActorPovReplayArtifactV2:
        raise TypeError("current replay must produce current actor POV")
    validate_actor_pov_replay_artifact_v2(artifact)
    return artifact


def validate_actor_pov_replay_artifact_v2(artifact: ActorPovReplayArtifactV2) -> None:
    """Validate a standalone POV V2 envelope and its local actor content.

    Parameters
    ----------
    artifact : ActorPovReplayArtifactV2
        Exact ActorPovReplayArtifactV2, including its declared nested model types.

    Returns
    -------
    None
        None when canonical structure, IDs, digests, completion, and local cue
        derivation agree.

    Raises
    ------
    TypeError
        The root or a declared nested model is not the exact expected type.
    ValueError
        Stored values, hashes, temporal joins, or cues are inconsistent.

    Notes
    -----
    Host-only; does not read the referenced replay or write files. To establish
    source correspondence, use validate_actor_pov_replay_against_replay_v2.
    """
    validate_declared_model_tree(
        artifact, record_name="actor POV replay", expected_type=ActorPovReplayArtifactV2
    )
    validate_actor_pov_replay_content(artifact.content)


def validate_actor_pov_replay_against_replay_v2(
    artifact: ActorPovReplayArtifactV2, replay: ReplayArtifactV3
) -> None:
    """Prove that this POV artifact equals the selected source-replay export.

    Parameters
    ----------
    artifact : ActorPovReplayArtifactV2
        ActorPovReplayArtifactV2 to compare, including its selected global slot.
    replay : ReplayArtifactV3
        Exact ReplayArtifactV3 declared as the source.

    Returns
    -------
    None
        None when the entire artifact equals the independently rebuilt source
        projection.

    Raises
    ------
    TypeError
        The source replay has an unsupported exact type.
    ValueError
        Source validation, actor selection, or exact artifact comparison fails.

    Notes
    -----
    Host-only; recomputes the source projection and hashes without writing files.
    This version validates the rebuilt export, then compares the supplied artifact by
    model equality; it does not separately call the standalone validator on that
    artifact.
    """
    if artifact != export_actor_pov_replay_v2(
        replay, global_slot=artifact.content.selected_global_slot
    ):
        raise ValueError("actor POV does not match its source replay")


def canonical_actor_pov_replay_json_bytes_v2(
    artifact: ActorPovReplayArtifactV2,
) -> bytes:
    """Validate and serialize the full POV V2 provenance envelope.

    Parameters
    ----------
    artifact : ActorPovReplayArtifactV2
        Exact ActorPovReplayArtifactV2 with valid local content.

    Returns
    -------
    bytes
        Canonical compact, key-sorted UTF-8 JSON bytes including content, its digest,
        and the full source replay reference. No file is written.

    Raises
    ------
    TypeError
        A declared model type is unsupported.
    ValueError
        Artifact validation or canonical serialization fails.

    Notes
    -----
    Envelope bytes can differ when full-source provenance differs even if the
    actor-authorized content is identical.
    """
    validate_actor_pov_replay_artifact_v2(artifact)
    return canonical_json_bytes(artifact)


__all__ = [
    "ACTOR_POV_ADJACENT_TRANSITION_SLICE_SCHEMA_ID",
    "ACTOR_POV_ARTIFACT_SCHEMA_ID",
    "ACTOR_POV_AXIS_MAPPING_SCHEMA_ID",
    "ACTOR_POV_CONTENT_SCHEMA_ID",
    "ACTOR_POV_CURRENT_SLICE_SCHEMA_ID",
    "ACTOR_POV_SCHEMA_VERSION",
    "ACTOR_POV_SCHEMA_VERSION_V2",
    "ActorPovAcceptedActionV1",
    "ActorPovActionMaskV1",
    "ActorPovAdjacentTransitionSlice",
    "ActorPovAdjacentTransitionSliceV1",
    "ActorPovAdjacentTransitionSliceV2",
    "ActorPovAxisMapping",
    "ActorPovAxisMappingV1",
    "ActorPovAxisMappingV2",
    "ActorPovCurrentSlice",
    "ActorPovCurrentSliceV1",
    "ActorPovCurrentSliceV2",
    "ActorPovEpisodeCompletionV1",
    "ActorPovEpisodeEndedCueV1",
    "ActorPovFrame",
    "ActorPovFrameV1",
    "ActorPovFrameV2",
    "ActorPovOwnActionOutcomeCueV1",
    "ActorPovOwnCooldownChangedCueV1",
    "ActorPovOwnHealthChangedCueV1",
    "ActorPovOwnLifecycleChangedCueV1",
    "ActorPovOwnPositionChangedCueV1",
    "ActorPovOwnStatusChangedCueV1",
    "ActorPovPresentationCueV1",
    "ActorPovPreviousTimestepActionsV1",
    "ActorPovReplayArtifact",
    "ActorPovReplayArtifactV1",
    "ActorPovReplayArtifactV2",
    "ActorPovReplayContent",
    "ActorPovReplayContentV1",
    "ActorPovReplayContentV2",
    "ActorPovSpawnLifecycleV1",
    "ActorPovSubmittedActionV1",
    "ActorPovTransitionV1",
    "ActorPovVisibleBodyObservationChangedCueV1",
    "build_actor_pov_adjacent_transition_slice_v1",
    "build_actor_pov_current_slice_v1",
    "canonical_actor_pov_content_json_bytes_v1",
    "canonical_actor_pov_replay_json_bytes_v1",
    "canonical_actor_pov_replay_json_bytes_v2",
    "export_actor_pov_replay_v1",
    "export_actor_pov_replay_v2",
    "slice_actor_pov_current_frame_v1",
    "slice_actor_pov_current_transition_v1",
    "validate_actor_pov_replay_against_replay_v1",
    "validate_actor_pov_replay_against_replay_v2",
    "validate_actor_pov_replay_artifact_v1",
    "validate_actor_pov_replay_artifact_v2",
    "validate_actor_pov_replay_content",
    "validate_actor_pov_replay_content_v1",
]
