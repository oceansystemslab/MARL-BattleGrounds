"""Strict additive contracts for authority-filtered presentation frames.

This module is deliberately transport- and service-neutral.  Its five final
leaves bind an authority-local current endpoint to one source epoch while
keeping incoming history and outgoing inspection in separate branches.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from math import isfinite
from typing import Annotated, Literal, Self, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from marl_battlegrounds.evaluation.models import EvaluationEpisodeContextV1
from marl_battlegrounds.evaluation.pov import (
    ActorPovActionMaskV1,
    ActorPovAxisMappingV1,
)
from marl_battlegrounds.rendering.authorized_incoming import (
    AgentIncomingObservationV1,
    AgentIncomingObservedStatusV1,
    NoSharedObsIncomingSummaryV1,
    NoSharedObsOwnActionOutcomeIncomingCueV1,
    NoSharedObsOwnCooldownChangedIncomingCueV1,
    NoSharedObsOwnHealthChangedIncomingCueV1,
    NoSharedObsOwnLifecycleChangedIncomingCueV1,
    NoSharedObsOwnPositionChangedIncomingCueV1,
    NoSharedObsOwnStatusChangedIncomingCueV1,
    NoSharedObsVisibleBodyChangedIncomingCueV1,
    SharedObsAppearanceIncomingDeltaV1,
    SharedObsDisappearanceIncomingDeltaV1,
    SharedObsIncomingSummaryV1,
    SharedObsObservationProvenanceIncomingDeltaV1,
    SharedObsObservedValuesIncomingDeltaV1,
)
from marl_battlegrounds.rendering.authorized_inspection import (
    AuthorizedAxisOnlyTargetActionV1,
    AuthorizedDecisionMaskV1,
    AuthorizedNoTargetActionV1,
    AuthorizedVisibleTargetActionV1,
    LiveDraftInspectionPresentationV1,
    NoSharedObsReplayTransitionReferenceV1,
    OracleReplayTransitionReferenceV1,
    ReplayInspectionPresentationV1,
    SharedObsReplayTransitionReferenceV1,
)
from marl_battlegrounds.rendering.authorized_pov_scene import (
    NoSharedObsAuthorizedScenePartsV1,
    SharedObsAgentObservationProvenanceV1,
    SharedObsAuthorizedScenePartsV1,
    SharedObsAuthorizedSensorSourceV1,
    pov_presentation_key_v1,
)
from marl_battlegrounds.rendering.authorized_presentation import (
    AcceptedActionTupleV1,
    AgentPovVisualIncomingSummaryV1,
    AuthorizedAgentV1,
    AuthorizedAuraModifierV1,
    AuthorizedBattlefieldSceneV1,
    AuthorizedClassMechanics,
    AuthorizedClassMechanicsV1,
    AuthorizedClassMechanicsV2,
    AuthorizedStatusV1,
    ReplayIncomingActionRejectedEventV1,
    ReplayIncomingAuthorizedAgentIdentityV1,
    ReplayIncomingFeedOnlyAgentIdentityV1,
    ReplayIncomingSummaryV1,
    SubmittedActionTupleV1,
    build_oracle_authorized_scene_v1,
    oracle_presentation_key_v1,
)
from marl_battlegrounds.rendering.scene import BattlefieldSceneV2

PRESENTATION_PROTOCOL_SCHEMA_VERSION = 1
PRESENTATION_TECHNICAL_DIGEST_PREFIX_LENGTH_V1 = 12
PRESENTATION_AUDIENCE_UNAVAILABLE_MESSAGE: Literal[
    "Authorized presentation is unavailable for the active audience."
] = "Authorized presentation is unavailable for the active audience."

_OpaqueId = Annotated[
    str,
    StringConstraints(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$"),
]
_ScientificId = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=512,
        pattern=r"^[A-Za-z0-9_.:/+\-]+$",
    ),
]
_DisplayText = Annotated[
    str,
    StringConstraints(min_length=1, max_length=128, pattern=r"^[^\r\n]+$"),
]
_NonNegativeInt = Annotated[int, Field(ge=0)]
_Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_Sha256Prefix = Annotated[
    str,
    StringConstraints(
        min_length=PRESENTATION_TECHNICAL_DIGEST_PREFIX_LENGTH_V1,
        max_length=PRESENTATION_TECHNICAL_DIGEST_PREFIX_LENGTH_V1,
        pattern=rf"^[0-9a-f]{{{PRESENTATION_TECHNICAL_DIGEST_PREFIX_LENGTH_V1}}}$",
    ),
]


class _PresentationProtocolModel(BaseModel):
    """Strict immutable base for the additive presentation resource."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
    )


def _require_exact_type(value: object, expected: type[object], *, name: str) -> None:
    if type(value) is not expected:
        raise ValueError(f"{name} must use the exact {expected.__name__} root.")


def _require_ordered_unique(values: tuple[object, ...], *, name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must contain unique ordered values.")


class LiveOraclePresentationSourceIdentityV1(_PresentationProtocolModel):
    source_kind: Literal["live_oracle_frame"]
    source_session_id: _OpaqueId
    source_run_generation: _NonNegativeInt
    source_revision: _NonNegativeInt
    source_authority_epoch: _NonNegativeInt
    episode_id: _ScientificId
    source_frame_index: _NonNegativeInt
    source_frame_id: _ScientificId
    source_simulator_step_count: _NonNegativeInt
    source_submission_scope: Literal["joint_turn", "scripted_playback"]
    source_authorized_endpoint_digest_sha256: _Sha256Hex

    @model_validator(mode="after")
    def _validate_source(self) -> Self:
        if self.source_authority_epoch != self.source_revision:
            raise ValueError("source authority epoch must equal source revision.")
        if self.source_frame_id != f"{self.episode_id}:frame:{self.source_frame_index}":
            raise ValueError("live Oracle source frame ID is not canonical.")
        return self


class LiveNoSharedObsPresentationSourceIdentityV1(_PresentationProtocolModel):
    source_kind: Literal["live_no_shared_obs_frame"]
    source_session_id: _OpaqueId
    source_run_generation: _NonNegativeInt
    source_revision: _NonNegativeInt
    source_authority_epoch: _NonNegativeInt
    episode_id: _ScientificId
    source_frame_index: _NonNegativeInt
    source_recipient_public_agent_id: _ScientificId
    source_recipient_frame_id: _ScientificId
    source_simulator_step_count: _NonNegativeInt
    source_submission_scope: Literal["controlled_actor", "scripted_playback"]
    source_authorized_endpoint_digest_sha256: _Sha256Hex

    @model_validator(mode="after")
    def _validate_source(self) -> Self:
        if self.source_authority_epoch != self.source_revision:
            raise ValueError("source authority epoch must equal source revision.")
        expected = (
            f"{self.episode_id}:actor-pov:{self.source_recipient_public_agent_id}:"
            f"frame:{self.source_frame_index}"
        )
        if self.source_recipient_frame_id != expected:
            raise ValueError("live NoSharedObs source frame ID is not canonical.")
        return self


class ReplayOraclePresentationSourceIdentityV1(_PresentationProtocolModel):
    source_kind: Literal["replay_oracle_frame"]
    source_session_id: _OpaqueId
    source_revision: _NonNegativeInt
    source_authority_epoch: _NonNegativeInt
    source_artifact_id: _ScientificId
    source_timeline_id: _ScientificId
    source_replay_schema_version: Literal[1]
    source_context_digest_sha256: _Sha256Hex
    source_trajectory_content_digest_sha256: _Sha256Hex
    source_artifact_digest_sha256: _Sha256Hex
    episode_id: _ScientificId
    source_frame_index: _NonNegativeInt
    source_final_frame_index: _NonNegativeInt
    source_frame_id: _ScientificId
    source_simulator_step_count: _NonNegativeInt
    source_cursor_generation: _NonNegativeInt
    source_choreography_generation: _NonNegativeInt
    source_recorded_ordinary_movement_distance_scale: Annotated[
        float,
        Field(gt=0.0),
    ]
    source_authorized_endpoint_digest_sha256: _Sha256Hex

    @model_validator(mode="after")
    def _validate_source(self) -> Self:
        if self.source_authority_epoch != self.source_revision:
            raise ValueError("source authority epoch must equal source revision.")
        if self.source_frame_index > self.source_final_frame_index:
            raise ValueError("source frame exceeds the retained replay prefix.")
        if self.source_frame_id != f"{self.episode_id}:frame:{self.source_frame_index}":
            raise ValueError("source frame ID is not canonical.")
        if self.source_artifact_id != f"{self.episode_id}:replay":
            raise ValueError("source replay artifact ID is not canonical.")
        if self.source_timeline_id != (
            f"{self.source_artifact_id}:timeline:researcher"
        ):
            raise ValueError("source Oracle timeline ID is not canonical.")
        if self.source_choreography_generation > self.source_cursor_generation:
            raise ValueError(
                "source choreography generation cannot exceed cursor generation."
            )
        if not isfinite(self.source_recorded_ordinary_movement_distance_scale):
            raise ValueError("source recorded movement scale must be finite.")
        return self


class ReplayNoSharedObsPresentationSourceIdentityV1(_PresentationProtocolModel):
    source_kind: Literal["replay_no_shared_obs_frame"]
    source_session_id: _OpaqueId
    source_revision: _NonNegativeInt
    source_authority_epoch: _NonNegativeInt
    episode_id: _ScientificId
    source_frame_index: _NonNegativeInt
    source_final_frame_index: _NonNegativeInt
    source_recipient_public_agent_id: _ScientificId
    source_recipient_frame_id: _ScientificId
    source_simulator_step_count: _NonNegativeInt
    source_observation_mode: Literal["no_shared_obs"]
    source_authorized_endpoint_digest_sha256: _Sha256Hex

    @model_validator(mode="after")
    def _validate_source(self) -> Self:
        _validate_replay_agent_source(self, mode="actor-pov")
        return self


class ReplaySharedObsPresentationSourceIdentityV1(_PresentationProtocolModel):
    source_kind: Literal["replay_shared_obs_visual_union_frame"]
    source_session_id: _OpaqueId
    source_revision: _NonNegativeInt
    source_authority_epoch: _NonNegativeInt
    episode_id: _ScientificId
    source_frame_index: _NonNegativeInt
    source_final_frame_index: _NonNegativeInt
    source_recipient_public_agent_id: _ScientificId
    source_recipient_frame_id: _ScientificId
    source_simulator_step_count: _NonNegativeInt
    source_observation_mode: Literal["shared_obs_visual_union"]
    source_authorized_endpoint_digest_sha256: _Sha256Hex

    @model_validator(mode="after")
    def _validate_source(self) -> Self:
        _validate_replay_agent_source(
            self,
            mode="shared-obs-visual-union",
        )
        return self


def _validate_replay_agent_source(
    source: ReplayNoSharedObsPresentationSourceIdentityV1
    | ReplaySharedObsPresentationSourceIdentityV1,
    *,
    mode: Literal["actor-pov", "shared-obs-visual-union"],
) -> None:
    if source.source_authority_epoch != source.source_revision:
        raise ValueError("source authority epoch must equal source revision.")
    if source.source_frame_index > source.source_final_frame_index:
        raise ValueError("source frame exceeds the retained replay prefix.")
    expected = (
        f"{source.episode_id}:{mode}:{source.source_recipient_public_agent_id}:"
        f"frame:{source.source_frame_index}"
    )
    if source.source_recipient_frame_id != expected:
        raise ValueError("replay Agent source frame ID is not recipient-local.")


class OraclePresentationAuthorityV1(_PresentationProtocolModel):
    authority_kind: Literal["oracle"]
    projection_basis: Literal["global_evaluation_projection"]


class NoSharedObsPresentationAuthorityV1(_PresentationProtocolModel):
    authority_kind: Literal["agent_pov"]
    observation_mode: Literal["no_shared_obs"]
    recipient_public_agent_id: _ScientificId
    recipient_presentation_key: _ScientificId
    projection_basis: Literal["recipient_own_recorded_observation"]
    exact_actor_input_export_available: Literal[True]


class SharedObsPresentationAuthorityV1(_PresentationProtocolModel):
    authority_kind: Literal["agent_pov"]
    observation_mode: Literal["shared_obs_visual_union"]
    recipient_public_agent_id: _ScientificId
    recipient_presentation_key: _ScientificId
    projection_basis: Literal["authorized_same_epoch_sensor_source_visual_union"]
    exact_actor_input_export_available: Literal[False]


class OraclePublicIdentityDirectoryRowV1(_PresentationProtocolModel):
    public_agent_id: _ScientificId
    configured_active: bool
    team_id: Annotated[int, Field(ge=1, le=2)]
    team_local_slot: Annotated[int, Field(ge=0, lt=5)]
    class_id: Annotated[int, Field(ge=1, le=5)] | None
    class_name: _DisplayText | None

    @model_validator(mode="after")
    def _validate_row(self) -> Self:
        if self.configured_active != (self.class_id is not None):
            raise ValueError("active directory rows require a class identity.")
        if (self.class_id is None) != (self.class_name is None):
            raise ValueError("directory class ID and name must be present as a pair.")
        return self


class OraclePublicIdentityDirectoryV1(_PresentationProtocolModel):
    directory_kind: Literal["oracle_public_identity_directory"]
    identities: tuple[OraclePublicIdentityDirectoryRowV1, ...]

    @model_validator(mode="after")
    def _validate_directory(self) -> Self:
        if len(self.identities) != 10 or any(
            type(row) is not OraclePublicIdentityDirectoryRowV1
            for row in self.identities
        ):
            raise ValueError("Oracle directory requires exactly ten exact rows.")
        for index, row in enumerate(self.identities):
            if row.team_id != index // 5 + 1 or row.team_local_slot != index % 5:
                raise ValueError("Oracle directory must retain fixed team topology.")
        _require_ordered_unique(
            tuple(row.public_agent_id for row in self.identities),
            name="Oracle directory public identities",
        )
        return self


class ReplayResearcherRosterAgentV1(_PresentationProtocolModel):
    """Global roster facts that deliberately omit battlefield geometry."""

    presentation_key: _ScientificId
    public_agent_id: _ScientificId
    team_id: Annotated[int, Field(ge=1, le=2)]
    team_local_slot: Annotated[int, Field(ge=0, lt=5)]
    class_id: Annotated[int, Field(ge=1, le=5)]
    class_name: _DisplayText
    life_state: Literal["alive", "corpse"]
    current_health: Annotated[float, Field(ge=0.0)]
    maximum_health: Annotated[float, Field(gt=0.0)]
    effective_movement_speed: Annotated[float, Field(ge=0.0)]
    ultimate_cooldown_remaining: _NonNegativeInt
    spawn_shield_remaining: _NonNegativeInt
    steps_until_out_of_combat: _NonNegativeInt
    out_of_combat_delay_steps: _NonNegativeInt
    statuses: tuple[AuthorizedStatusV1, ...]
    aura_modifiers: tuple[AuthorizedAuraModifierV1, ...]

    @model_validator(mode="after")
    def _validate_row(self) -> Self:
        if self.current_health > self.maximum_health:
            raise ValueError("researcher roster health exceeds maximum health.")
        if self.steps_until_out_of_combat > self.out_of_combat_delay_steps:
            raise ValueError("researcher roster combat duration exceeds its delay.")
        if any(type(row) is not AuthorizedStatusV1 for row in self.statuses):
            raise ValueError("researcher roster statuses require exact roots.")
        if any(
            type(row) is not AuthorizedAuraModifierV1 for row in self.aura_modifiers
        ):
            raise ValueError("researcher roster modifiers require exact roots.")
        return self


class MovementActionDisplayRowV1(_PresentationProtocolModel):
    move_action: Annotated[int, Field(ge=0, lt=9)]
    display_name: _DisplayText


class TargetNoneActionDisplayRowV1(_PresentationProtocolModel):
    target_kind: Literal["target_none"]
    target_action: Literal[0]
    display_name: _DisplayText


class TargetAgentActionDisplayRowV1(_PresentationProtocolModel):
    target_kind: Literal["public_agent"]
    target_action: Annotated[int, Field(ge=1, lt=11)]
    display_name: _DisplayText
    target_public_agent_id: _ScientificId
    target_relation: Literal["same_team", "opponent"]


type TargetActionDisplayRowV1 = Annotated[
    TargetNoneActionDisplayRowV1 | TargetAgentActionDisplayRowV1,
    Field(discriminator="target_kind"),
]


class UltimateChoiceDisplayRowV1(_PresentationProtocolModel):
    use_ultimate_action: Annotated[int, Field(ge=0, lt=2)]
    display_name: _DisplayText


class _ActionAxisBaseV1(_PresentationProtocolModel):
    owner_presentation_key: _ScientificId
    owner_public_agent_id: _ScientificId
    movement_actions: tuple[MovementActionDisplayRowV1, ...]
    target_actions: tuple[TargetActionDisplayRowV1, ...]
    ultimate_choices: tuple[UltimateChoiceDisplayRowV1, ...]

    @model_validator(mode="after")
    def _validate_axis(self) -> Self:
        if (
            len(self.movement_actions) != 9
            or any(
                type(row) is not MovementActionDisplayRowV1
                for row in self.movement_actions
            )
            or tuple(row.move_action for row in self.movement_actions)
            != tuple(range(9))
        ):
            raise ValueError("movement axis must retain exactly nine ordered rows.")
        if (
            len(self.target_actions) != 11
            or type(self.target_actions[0]) is not TargetNoneActionDisplayRowV1
            or any(
                type(row) is not TargetAgentActionDisplayRowV1
                for row in self.target_actions[1:]
            )
            or tuple(row.target_action for row in self.target_actions)
            != tuple(range(11))
        ):
            raise ValueError("target axis must retain exactly eleven ordered rows.")
        positive = cast(
            tuple[TargetAgentActionDisplayRowV1, ...],
            self.target_actions[1:],
        )
        if (
            tuple(row.target_relation for row in positive[:5]) != ("same_team",) * 5
            or tuple(row.target_relation for row in positive[5:]) != ("opponent",) * 5
        ):
            raise ValueError(
                "target axis must retain the exact five-plus-five partition."
            )
        _require_ordered_unique(
            tuple(row.target_public_agent_id for row in positive),
            name="target-axis public identities",
        )
        if (
            len(self.ultimate_choices) != 2
            or any(
                type(row) is not UltimateChoiceDisplayRowV1
                for row in self.ultimate_choices
            )
            or tuple(row.use_ultimate_action for row in self.ultimate_choices) != (0, 1)
        ):
            raise ValueError("Ultimate axis must retain exactly two ordered rows.")
        for rows, name in (
            (self.movement_actions, "movement"),
            (self.target_actions, "target"),
            (self.ultimate_choices, "Ultimate"),
        ):
            _require_ordered_unique(
                tuple(row.display_name for row in rows),
                name=f"{name} display names",
            )
        return self

    @property
    def target_public_agent_id_by_action(self) -> tuple[str | None, ...]:
        return (
            None,
            *(
                cast(TargetAgentActionDisplayRowV1, row).target_public_agent_id
                for row in self.target_actions[1:]
            ),
        )


class OracleActionAxisV1(_ActionAxisBaseV1):
    axis_kind: Literal["oracle_actor_action_axis"]


class AgentPovActionAxisV1(_ActionAxisBaseV1):
    axis_kind: Literal["agent_pov_action_axis"]


class AgentPovDecisionMaskV1(_PresentationProtocolModel):
    """Strict required-field copy of the accepted recipient decision mask."""

    schema_id: Literal["marl_battlegrounds.evaluation.actor_pov_action_mask"]
    schema_version: Literal[1]
    move: tuple[bool, ...]
    select_target: tuple[bool, ...]
    use_ultimate: tuple[bool, ...]
    select_target_use_ultimate_joint: tuple[tuple[bool, ...], ...]

    @model_validator(mode="after")
    def _validate_mask(self) -> Self:
        if (
            len(self.move) != 9
            or len(self.select_target) != 11
            or len(self.use_ultimate) != 2
            or len(self.select_target_use_ultimate_joint) != 11
            or any(len(row) != 2 for row in self.select_target_use_ultimate_joint)
            or any(type(value) is not bool for value in self.move)
            or any(type(value) is not bool for value in self.select_target)
            or any(type(value) is not bool for value in self.use_ultimate)
            or any(
                type(value) is not bool
                for row in self.select_target_use_ultimate_joint
                for value in row
            )
        ):
            raise ValueError(
                "Agent decision mask must retain exact 9/11/2/11x2 shapes."
            )
        target_marginal = tuple(
            any(row) for row in self.select_target_use_ultimate_joint
        )
        ultimate_marginal = tuple(
            any(row[ultimate] for row in self.select_target_use_ultimate_joint)
            for ultimate in range(2)
        )
        if (
            self.select_target != target_marginal
            or self.use_ultimate != ultimate_marginal
        ):
            raise ValueError("Agent decision mask marginals must equal its joint mask.")
        return self


def _strict_agent_mask_v1(mask: ActorPovActionMaskV1) -> AgentPovDecisionMaskV1:
    if type(mask) is not ActorPovActionMaskV1:
        raise TypeError("Agent mask projection requires the exact accepted mask root.")
    validated = ActorPovActionMaskV1.model_validate(mask.model_dump(mode="python"))
    return AgentPovDecisionMaskV1(
        schema_id=validated.schema_id,
        schema_version=validated.schema_version,
        move=validated.move,
        select_target=validated.select_target,
        use_ultimate=validated.use_ultimate,
        select_target_use_ultimate_joint=(validated.select_target_use_ultimate_joint),
    )


def _accepted_agent_mask_v1(mask: AgentPovDecisionMaskV1) -> ActorPovActionMaskV1:
    return ActorPovActionMaskV1(
        schema_id=mask.schema_id,
        schema_version=mask.schema_version,
        move=mask.move,
        select_target=mask.select_target,
        use_ultimate=mask.use_ultimate,
        select_target_use_ultimate_joint=mask.select_target_use_ultimate_joint,
    )


class NoSharedObsPresentationEndpointPartsV1(_PresentationProtocolModel):
    source_episode_id: _ScientificId
    source_frame_index: _NonNegativeInt
    source_recipient_frame_id: _ScientificId
    source_simulator_step_count: _NonNegativeInt
    recipient_public_agent_id: _ScientificId
    recipient_presentation_key: _ScientificId
    scene: AuthorizedBattlefieldSceneV1
    next_decision_action_mask: AgentPovDecisionMaskV1

    @model_validator(mode="after")
    def _validate_parts(self) -> Self:
        _require_exact_type(self.scene, AuthorizedBattlefieldSceneV1, name="scene")
        _require_exact_type(
            self.next_decision_action_mask,
            AgentPovDecisionMaskV1,
            name="next_decision_action_mask",
        )
        expected = (
            f"{self.source_episode_id}:actor-pov:{self.recipient_public_agent_id}:"
            f"frame:{self.source_frame_index}"
        )
        if self.source_recipient_frame_id != expected:
            raise ValueError("NoSharedObs endpoint frame ID is not recipient-local.")
        NoSharedObsAuthorizedScenePartsV1(
            source_episode_id=self.source_episode_id,
            source_frame_index=self.source_frame_index,
            source_recipient_frame_id=self.source_recipient_frame_id,
            source_simulator_step_count=self.source_simulator_step_count,
            recipient_public_agent_id=self.recipient_public_agent_id,
            recipient_presentation_key=self.recipient_presentation_key,
            scene=self.scene,
            next_decision_action_mask=_accepted_agent_mask_v1(
                self.next_decision_action_mask
            ),
        )
        return self


class SharedObsPresentationEndpointPartsV1(_PresentationProtocolModel):
    source_episode_id: _ScientificId
    source_frame_index: _NonNegativeInt
    source_recipient_frame_id: _ScientificId
    source_simulator_step_count: _NonNegativeInt
    recipient_public_agent_id: _ScientificId
    recipient_presentation_key: _ScientificId
    scene: AuthorizedBattlefieldSceneV1
    next_decision_action_mask: AgentPovDecisionMaskV1
    authorized_sensor_sources: tuple[SharedObsAuthorizedSensorSourceV1, ...]
    agent_observation_provenance: tuple[
        SharedObsAgentObservationProvenanceV1,
        ...,
    ]

    @model_validator(mode="after")
    def _validate_parts(self) -> Self:
        _require_exact_type(self.scene, AuthorizedBattlefieldSceneV1, name="scene")
        _require_exact_type(
            self.next_decision_action_mask,
            AgentPovDecisionMaskV1,
            name="next_decision_action_mask",
        )
        expected = (
            f"{self.source_episode_id}:shared-obs-visual-union:"
            f"{self.recipient_public_agent_id}:frame:{self.source_frame_index}"
        )
        if self.source_recipient_frame_id != expected:
            raise ValueError("SharedObs endpoint frame ID is not recipient-local.")
        SharedObsAuthorizedScenePartsV1(
            source_episode_id=self.source_episode_id,
            source_frame_index=self.source_frame_index,
            source_recipient_frame_id=self.source_recipient_frame_id,
            source_simulator_step_count=self.source_simulator_step_count,
            recipient_public_agent_id=self.recipient_public_agent_id,
            recipient_presentation_key=self.recipient_presentation_key,
            scene=self.scene,
            next_decision_action_mask=_accepted_agent_mask_v1(
                self.next_decision_action_mask
            ),
            authorized_sensor_sources=self.authorized_sensor_sources,
            agent_observation_provenance=self.agent_observation_provenance,
        )
        return self


class _OracleAuthorizedCurrentEndpointContentV1(_PresentationProtocolModel):
    endpoint_kind: Literal["oracle_authorized_current"]
    episode_id: _ScientificId
    frame_index: _NonNegativeInt
    frame_id: _ScientificId
    simulator_step_count: _NonNegativeInt
    scene: AuthorizedBattlefieldSceneV1
    identity_directory: OraclePublicIdentityDirectoryV1
    action_axis: OracleActionAxisV1 | None

    @model_validator(mode="after")
    def _validate_content(self) -> Self:
        if self.frame_id != f"{self.episode_id}:frame:{self.frame_index}":
            raise ValueError("Oracle endpoint frame ID is not canonical.")
        _require_exact_type(self.scene, AuthorizedBattlefieldSceneV1, name="scene")
        _require_exact_type(
            self.identity_directory,
            OraclePublicIdentityDirectoryV1,
            name="identity_directory",
        )
        if self.action_axis is not None:
            _require_exact_type(
                self.action_axis, OracleActionAxisV1, name="action_axis"
            )
        directory_by_id = {
            row.public_agent_id: row for row in self.identity_directory.identities
        }
        expected_active = tuple(
            row.public_agent_id
            for row in self.identity_directory.identities
            if row.configured_active
        )
        actual_active = tuple(row.public_agent_id for row in self.scene.agents)
        if actual_active != expected_active or any(
            row.relation != "oracle" for row in self.scene.agents
        ):
            raise ValueError("Oracle scene identities must equal the active directory.")
        class_name_by_id = {
            row.class_id: row.class_name for row in self.scene.class_mechanics
        }
        for agent in self.scene.agents:
            directory_row = directory_by_id[agent.public_agent_id]
            if (
                directory_row.team_id != agent.team_id
                or directory_row.class_id != agent.class_id
                or directory_row.class_name != agent.class_name
                or class_name_by_id.get(agent.class_id) != agent.class_name
            ):
                raise ValueError("Oracle scene identity facts must join the directory.")
        for pad in self.scene.spawn_pads:
            if pad.assigned_public_agent_id is None:
                continue
            directory_row = directory_by_id.get(pad.assigned_public_agent_id)
            if directory_row is None or (
                not directory_row.configured_active
                or directory_row.team_id != pad.team_id
                or directory_row.team_local_slot != pad.team_local_slot
            ):
                raise ValueError(
                    "Oracle spawn-pad assignment must join directory topology."
                )
        if self.action_axis is not None:
            owner = directory_by_id.get(self.action_axis.owner_public_agent_id)
            if owner is None or not owner.configured_active:
                raise ValueError("Oracle action-axis owner must be active.")
            target_rows = cast(
                tuple[TargetAgentActionDisplayRowV1, ...],
                self.action_axis.target_actions[1:],
            )
            for row in target_rows:
                target = directory_by_id.get(row.target_public_agent_id)
                if target is None:
                    raise ValueError(
                        "Oracle action-axis target must join the directory."
                    )
                expected_relation = (
                    "same_team" if target.team_id == owner.team_id else "opponent"
                )
                if row.target_relation != expected_relation:
                    raise ValueError(
                        "Oracle action-axis relation is not actor-relative."
                    )
            if self.action_axis.target_public_agent_id_by_action != (
                None,
                *_oracle_target_public_axis(
                    self.identity_directory,
                    owner_team_id=owner.team_id,
                ),
            ):
                raise ValueError(
                    "Oracle action-axis order must retain team-local target semantics."
                )
        return self


class _NoSharedObsAuthorizedCurrentEndpointContentV1(_PresentationProtocolModel):
    endpoint_kind: Literal["no_shared_obs_authorized_current"]
    parts: NoSharedObsPresentationEndpointPartsV1
    action_axis: AgentPovActionAxisV1

    @model_validator(mode="after")
    def _validate_content(self) -> Self:
        _require_exact_type(
            self.parts,
            NoSharedObsPresentationEndpointPartsV1,
            name="parts",
        )
        _require_exact_type(self.action_axis, AgentPovActionAxisV1, name="action_axis")
        _validate_agent_endpoint_parts(self.parts, self.action_axis)
        return self


class _SharedObsAuthorizedCurrentEndpointContentV1(_PresentationProtocolModel):
    endpoint_kind: Literal["shared_obs_authorized_current"]
    parts: SharedObsPresentationEndpointPartsV1
    action_axis: AgentPovActionAxisV1

    @model_validator(mode="after")
    def _validate_content(self) -> Self:
        _require_exact_type(
            self.parts,
            SharedObsPresentationEndpointPartsV1,
            name="parts",
        )
        _require_exact_type(self.action_axis, AgentPovActionAxisV1, name="action_axis")
        _validate_agent_endpoint_parts(self.parts, self.action_axis)
        return self


def _agent_axis_relation_and_team(
    action_axis: AgentPovActionAxisV1,
    *,
    recipient_public_agent_id: str,
    recipient_team_id: int,
    public_agent_id: str,
) -> tuple[Literal["self", "ally", "opponent"], int]:
    try:
        target_index = action_axis.target_public_agent_id_by_action.index(
            public_agent_id
        )
    except ValueError as error:
        raise ValueError("Agent identity is outside its action axis.") from error
    if public_agent_id == recipient_public_agent_id:
        if not 1 <= target_index <= 5:
            raise ValueError(
                "Agent recipient moved outside its same-team target block."
            )
        return "self", recipient_team_id
    if 1 <= target_index <= 5:
        return "ally", recipient_team_id
    if 6 <= target_index <= 10:
        return "opponent", 1 if recipient_team_id == 2 else 2
    raise ValueError("Agent identity occupies an invalid target-axis row.")


def _validate_shared_sensor_source_axis(
    source: SharedObsAuthorizedSensorSourceV1,
    action_axis: AgentPovActionAxisV1,
) -> None:
    _require_exact_type(
        source,
        SharedObsAuthorizedSensorSourceV1,
        name="SharedObs sensor source",
    )
    try:
        target_index = action_axis.target_public_agent_id_by_action.index(
            source.source_public_agent_id
        )
    except ValueError as error:
        raise ValueError(
            "SharedObs sensor source is outside the action axis."
        ) from error
    if source.source_kind == "recipient_base":
        if (
            source.source_public_agent_id != action_axis.owner_public_agent_id
            or source.source_presentation_key != action_axis.owner_presentation_key
        ):
            raise ValueError("SharedObs recipient-base source is not the fixed owner.")
        return
    if (
        source.source_kind != "shared_sensor_source"
        or source.source_public_agent_id == action_axis.owner_public_agent_id
        or not 1 <= target_index <= 5
    ):
        raise ValueError(
            "SharedObs shared sensor source must be a nonrecipient teammate."
        )


def _validate_agent_endpoint_parts(
    parts: NoSharedObsPresentationEndpointPartsV1
    | SharedObsPresentationEndpointPartsV1,
    action_axis: AgentPovActionAxisV1,
) -> None:
    if (
        action_axis.owner_public_agent_id != parts.recipient_public_agent_id
        or action_axis.owner_presentation_key != parts.recipient_presentation_key
    ):
        raise ValueError("Agent action axis must belong to the fixed recipient.")
    self_rows = tuple(
        row
        for row in parts.scene.agents
        if row.public_agent_id == parts.recipient_public_agent_id
    )
    if (
        len(self_rows) != 1
        or self_rows[0].relation != "self"
        or self_rows[0].presentation_key != parts.recipient_presentation_key
    ):
        raise ValueError("Agent endpoint recipient must join one self scene row.")
    target_ids = set(action_axis.target_public_agent_id_by_action[1:])
    if any(row.public_agent_id not in target_ids for row in parts.scene.agents):
        raise ValueError("Agent scene identities must belong to its action axis.")
    if type(parts) is SharedObsPresentationEndpointPartsV1:
        shared_sources = tuple(parts.authorized_sensor_sources) + tuple(
            source
            for provenance in parts.agent_observation_provenance
            for source in provenance.observation_sources
        )
        for source in shared_sources:
            _validate_shared_sensor_source_axis(source, action_axis)
        shared_public_ids = tuple(
            source.source_public_agent_id for source in parts.authorized_sensor_sources
        ) + tuple(
            public_id
            for provenance in parts.agent_observation_provenance
            for public_id in (
                provenance.agent_public_agent_id,
                *(
                    source.source_public_agent_id
                    for source in provenance.observation_sources
                ),
            )
        )
        if any(public_id not in target_ids for public_id in shared_public_ids):
            raise ValueError(
                "SharedObs endpoint provenance identity is outside its action axis."
            )
    target_index_by_id = {
        public_id: index
        for index, public_id in enumerate(action_axis.target_public_agent_id_by_action)
        if public_id is not None
    }
    recipient_team_id = self_rows[0].team_id
    for row in parts.scene.agents:
        if any(status.direct_sources for status in row.statuses):
            raise ValueError(
                "Agent observation statuses cannot disclose direct source identities."
            )
        target_index_by_id[row.public_agent_id]
        expected_relation, expected_team_id = _agent_axis_relation_and_team(
            action_axis,
            recipient_public_agent_id=parts.recipient_public_agent_id,
            recipient_team_id=recipient_team_id,
            public_agent_id=row.public_agent_id,
        )
        if row.relation != expected_relation or row.team_id != expected_team_id:
            raise ValueError(
                "Agent scene relation/team does not join its target-axis block."
            )


def _oracle_target_public_axis(
    directory: OraclePublicIdentityDirectoryV1,
    *,
    owner_team_id: int,
) -> tuple[str, ...]:
    own = tuple(
        row.public_agent_id
        for row in directory.identities
        if row.team_id == owner_team_id
    )
    opponent = tuple(
        row.public_agent_id
        for row in directory.identities
        if row.team_id != owner_team_id
    )
    return (*own, *opponent)


class OracleAuthorizedCurrentEndpointV1(_OracleAuthorizedCurrentEndpointContentV1):
    authorized_endpoint_digest_sha256: _Sha256Hex

    @model_validator(mode="after")
    def _validate_digest(self) -> Self:
        if self.authorized_endpoint_digest_sha256 != (
            canonical_authorized_endpoint_digest_sha256(self)
        ):
            raise ValueError("Oracle authorized endpoint digest mismatch.")
        return self


class NoSharedObsAuthorizedCurrentEndpointV1(
    _NoSharedObsAuthorizedCurrentEndpointContentV1
):
    authorized_endpoint_digest_sha256: _Sha256Hex

    @model_validator(mode="after")
    def _validate_digest(self) -> Self:
        if self.authorized_endpoint_digest_sha256 != (
            canonical_authorized_endpoint_digest_sha256(self)
        ):
            raise ValueError("NoSharedObs authorized endpoint digest mismatch.")
        return self


class SharedObsAuthorizedCurrentEndpointV1(
    _SharedObsAuthorizedCurrentEndpointContentV1
):
    authorized_endpoint_digest_sha256: _Sha256Hex

    @model_validator(mode="after")
    def _validate_digest(self) -> Self:
        if self.authorized_endpoint_digest_sha256 != (
            canonical_authorized_endpoint_digest_sha256(self)
        ):
            raise ValueError("SharedObs authorized endpoint digest mismatch.")
        return self


_ENDPOINT_CONTENT_TYPES = (
    _OracleAuthorizedCurrentEndpointContentV1,
    _NoSharedObsAuthorizedCurrentEndpointContentV1,
    _SharedObsAuthorizedCurrentEndpointContentV1,
)
_ENDPOINT_TYPES = (
    OracleAuthorizedCurrentEndpointV1,
    NoSharedObsAuthorizedCurrentEndpointV1,
    SharedObsAuthorizedCurrentEndpointV1,
)


def canonical_authorized_endpoint_digest_sha256(
    endpoint: _OracleAuthorizedCurrentEndpointContentV1
    | _NoSharedObsAuthorizedCurrentEndpointContentV1
    | _SharedObsAuthorizedCurrentEndpointContentV1
    | OracleAuthorizedCurrentEndpointV1
    | NoSharedObsAuthorizedCurrentEndpointV1
    | SharedObsAuthorizedCurrentEndpointV1,
) -> str:
    """Hash one exact authority-local endpoint, excluding only its digest."""
    if type(endpoint) not in (*_ENDPOINT_CONTENT_TYPES, *_ENDPOINT_TYPES):
        raise TypeError("endpoint digest requires one exact endpoint content root.")
    _validate_recursive_exact_runtime_types(endpoint)
    content_type = {
        _OracleAuthorizedCurrentEndpointContentV1: (
            _OracleAuthorizedCurrentEndpointContentV1
        ),
        _NoSharedObsAuthorizedCurrentEndpointContentV1: (
            _NoSharedObsAuthorizedCurrentEndpointContentV1
        ),
        _SharedObsAuthorizedCurrentEndpointContentV1: (
            _SharedObsAuthorizedCurrentEndpointContentV1
        ),
        OracleAuthorizedCurrentEndpointV1: _OracleAuthorizedCurrentEndpointContentV1,
        NoSharedObsAuthorizedCurrentEndpointV1: (
            _NoSharedObsAuthorizedCurrentEndpointContentV1
        ),
        SharedObsAuthorizedCurrentEndpointV1: (
            _SharedObsAuthorizedCurrentEndpointContentV1
        ),
    }[type(endpoint)]
    content_values = {
        name: getattr(endpoint, name) for name in content_type.model_fields
    }
    content = content_type.model_construct(**content_values)
    validated_content = content_type.model_validate_json(
        content.model_dump_json(warnings=False)
    )
    _validate_exact_runtime_tree_matches(
        content,
        validated_content,
        path="authorized_endpoint",
    )
    canonical_content = validated_content.model_dump(mode="json")
    encoded = json.dumps(
        canonical_content,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _seal_oracle_authorized_current_endpoint_v1(
    *,
    episode_id: str,
    frame_index: int,
    frame_id: str,
    simulator_step_count: int,
    scene: AuthorizedBattlefieldSceneV1,
    identity_directory: OraclePublicIdentityDirectoryV1,
    action_axis: OracleActionAxisV1 | None,
) -> OracleAuthorizedCurrentEndpointV1:
    content = _OracleAuthorizedCurrentEndpointContentV1(
        endpoint_kind="oracle_authorized_current",
        episode_id=episode_id,
        frame_index=frame_index,
        frame_id=frame_id,
        simulator_step_count=simulator_step_count,
        scene=scene,
        identity_directory=identity_directory,
        action_axis=action_axis,
    )
    return OracleAuthorizedCurrentEndpointV1(
        endpoint_kind=content.endpoint_kind,
        episode_id=content.episode_id,
        frame_index=content.frame_index,
        frame_id=content.frame_id,
        simulator_step_count=content.simulator_step_count,
        scene=content.scene,
        identity_directory=content.identity_directory,
        action_axis=content.action_axis,
        authorized_endpoint_digest_sha256=(
            canonical_authorized_endpoint_digest_sha256(content)
        ),
    )


def _oracle_public_identity_directory_v1(
    context: EvaluationEpisodeContextV1,
) -> OraclePublicIdentityDirectoryV1:
    catalog = context.static_mechanics_catalog
    return OraclePublicIdentityDirectoryV1(
        directory_kind="oracle_public_identity_directory",
        identities=tuple(
            OraclePublicIdentityDirectoryRowV1(
                public_agent_id=row.public_agent_id,
                configured_active=row.configured_active,
                team_id=index // 5 + 1,
                team_local_slot=row.team_local_slot,
                class_id=row.class_id if row.configured_active else None,
                class_name=(
                    catalog.class_name_by_id[row.class_id]
                    if row.configured_active
                    else None
                ),
            )
            for index, row in enumerate(context.roster)
        ),
    )


def _oracle_action_axis_from_context_v1(
    context: EvaluationEpisodeContextV1,
    scene: AuthorizedBattlefieldSceneV1,
    directory: OraclePublicIdentityDirectoryV1,
    *,
    authority_session_id: str,
    selected_internal_slot: int | None,
) -> OracleActionAxisV1 | None:
    if selected_internal_slot is None:
        return None
    if type(selected_internal_slot) is not int or not (
        0 <= selected_internal_slot < len(context.roster)
    ):
        raise ValueError("selected_internal_slot must be an exact configured slot.")
    roster = context.roster[selected_internal_slot]
    if not roster.configured_active:
        raise ValueError("the selected Oracle action-axis owner must be active.")
    directory_by_id = {row.public_agent_id: row for row in directory.identities}
    owner = directory_by_id[roster.public_agent_id]
    owner_scene_rows = tuple(
        row for row in scene.agents if row.public_agent_id == roster.public_agent_id
    )
    if len(owner_scene_rows) != 1:
        raise ValueError("the selected Oracle action-axis owner must join the scene.")
    owner_scene = owner_scene_rows[0]
    expected_owner_key = oracle_presentation_key_v1(
        authority_session_id=authority_session_id,
        public_agent_id=roster.public_agent_id,
    )
    if owner_scene.presentation_key != expected_owner_key:
        raise ValueError("the selected Oracle action-axis owner key is not canonical.")

    catalog = context.static_mechanics_catalog
    target_slots = catalog.global_recipient_slot_by_actor_and_target_action[
        selected_internal_slot
    ]
    if len(target_slots) != 11 or target_slots[0] is not None:
        raise ValueError("the selected Oracle target axis is not canonical.")
    target_rows: list[TargetNoneActionDisplayRowV1 | TargetAgentActionDisplayRowV1] = [
        TargetNoneActionDisplayRowV1(
            target_kind="target_none",
            target_action=0,
            display_name=catalog.target_action_name_by_id[0],
        )
    ]
    for target_action, target_slot in enumerate(target_slots[1:], start=1):
        if type(target_slot) is not int or not 0 <= target_slot < len(context.roster):
            raise ValueError(
                "the selected Oracle target axis contains an invalid slot."
            )
        target_public_agent_id = context.roster[target_slot].public_agent_id
        target = directory_by_id[target_public_agent_id]
        target_rows.append(
            TargetAgentActionDisplayRowV1(
                target_kind="public_agent",
                target_action=target_action,
                display_name=catalog.target_action_name_by_id[target_action],
                target_public_agent_id=target_public_agent_id,
                target_relation=(
                    "same_team" if target.team_id == owner.team_id else "opponent"
                ),
            )
        )
    return OracleActionAxisV1(
        axis_kind="oracle_actor_action_axis",
        owner_presentation_key=expected_owner_key,
        owner_public_agent_id=roster.public_agent_id,
        movement_actions=tuple(
            MovementActionDisplayRowV1(
                move_action=index,
                display_name=name,
            )
            for index, name in enumerate(catalog.movement_action_name_by_id)
        ),
        target_actions=tuple(target_rows),
        ultimate_choices=tuple(
            UltimateChoiceDisplayRowV1(
                use_ultimate_action=index,
                display_name=name,
            )
            for index, name in enumerate(catalog.use_ultimate_action_name_by_id)
        ),
    )


def build_oracle_authorized_current_endpoint_v1(
    *,
    context: EvaluationEpisodeContextV1,
    source_scene: BattlefieldSceneV2,
    authority_session_id: str,
    selected_internal_slot: int | None,
) -> OracleAuthorizedCurrentEndpointV1:
    """Derive the full Oracle endpoint from exact epoch-bearing authority."""
    if type(context) is not EvaluationEpisodeContextV1:
        raise TypeError("context must use the exact evaluation episode root.")
    if type(source_scene) is not BattlefieldSceneV2:
        raise TypeError("source_scene must use the exact BattlefieldSceneV2 root.")
    session = cast(
        str,
        TypeAdapter(_OpaqueId).validate_python(authority_session_id),
    )
    scene = build_oracle_authorized_scene_v1(
        context,
        source_scene,
        authority_session_id=session,
    )
    directory = _oracle_public_identity_directory_v1(context)
    action_axis = _oracle_action_axis_from_context_v1(
        context,
        scene,
        directory,
        authority_session_id=session,
        selected_internal_slot=selected_internal_slot,
    )
    return _seal_oracle_authorized_current_endpoint_v1(
        episode_id=source_scene.episode_id,
        frame_index=source_scene.frame_index,
        frame_id=source_scene.frame_id,
        simulator_step_count=source_scene.simulator_step_count,
        scene=scene,
        identity_directory=directory,
        action_axis=action_axis,
    )


def build_no_shared_obs_authorized_current_endpoint_v1(
    *,
    parts: NoSharedObsAuthorizedScenePartsV1,
    axis_mapping: ActorPovAxisMappingV1,
) -> NoSharedObsAuthorizedCurrentEndpointV1:
    """Bind accepted NoShared parts to their trusted recorded action axis."""
    if type(parts) is not NoSharedObsAuthorizedScenePartsV1:
        raise TypeError("parts must use the exact accepted NoSharedObs root.")
    parts.__post_init__()
    action_axis = _agent_pov_action_axis_v1(
        axis_mapping,
        owner_presentation_key=parts.recipient_presentation_key,
        owner_public_agent_id=parts.recipient_public_agent_id,
    )
    projected_parts = NoSharedObsPresentationEndpointPartsV1(
        source_episode_id=parts.source_episode_id,
        source_frame_index=parts.source_frame_index,
        source_recipient_frame_id=parts.source_recipient_frame_id,
        source_simulator_step_count=parts.source_simulator_step_count,
        recipient_public_agent_id=parts.recipient_public_agent_id,
        recipient_presentation_key=parts.recipient_presentation_key,
        scene=parts.scene,
        next_decision_action_mask=_strict_agent_mask_v1(
            parts.next_decision_action_mask
        ),
    )
    content = _NoSharedObsAuthorizedCurrentEndpointContentV1(
        endpoint_kind="no_shared_obs_authorized_current",
        parts=projected_parts,
        action_axis=action_axis,
    )
    return NoSharedObsAuthorizedCurrentEndpointV1(
        endpoint_kind=content.endpoint_kind,
        parts=content.parts,
        action_axis=content.action_axis,
        authorized_endpoint_digest_sha256=(
            canonical_authorized_endpoint_digest_sha256(content)
        ),
    )


def build_shared_obs_authorized_current_endpoint_v1(
    *,
    parts: SharedObsAuthorizedScenePartsV1,
    axis_mapping: ActorPovAxisMappingV1,
) -> SharedObsAuthorizedCurrentEndpointV1:
    """Bind accepted Shared visual-union parts to the recipient source axis."""
    if type(parts) is not SharedObsAuthorizedScenePartsV1:
        raise TypeError("parts must use the exact accepted SharedObs root.")
    parts.__post_init__()
    action_axis = _agent_pov_action_axis_v1(
        axis_mapping,
        owner_presentation_key=parts.recipient_presentation_key,
        owner_public_agent_id=parts.recipient_public_agent_id,
    )
    projected_parts = SharedObsPresentationEndpointPartsV1(
        source_episode_id=parts.source_episode_id,
        source_frame_index=parts.source_frame_index,
        source_recipient_frame_id=parts.source_recipient_frame_id,
        source_simulator_step_count=parts.source_simulator_step_count,
        recipient_public_agent_id=parts.recipient_public_agent_id,
        recipient_presentation_key=parts.recipient_presentation_key,
        scene=parts.scene,
        next_decision_action_mask=_strict_agent_mask_v1(
            parts.next_decision_action_mask
        ),
        authorized_sensor_sources=parts.authorized_sensor_sources,
        agent_observation_provenance=parts.agent_observation_provenance,
    )
    content = _SharedObsAuthorizedCurrentEndpointContentV1(
        endpoint_kind="shared_obs_authorized_current",
        parts=projected_parts,
        action_axis=action_axis,
    )
    return SharedObsAuthorizedCurrentEndpointV1(
        endpoint_kind=content.endpoint_kind,
        parts=content.parts,
        action_axis=content.action_axis,
        authorized_endpoint_digest_sha256=(
            canonical_authorized_endpoint_digest_sha256(content)
        ),
    )


def _agent_pov_action_axis_v1(
    axis_mapping: ActorPovAxisMappingV1,
    *,
    owner_presentation_key: str,
    owner_public_agent_id: str,
) -> AgentPovActionAxisV1:
    if type(axis_mapping) is not ActorPovAxisMappingV1:
        raise TypeError("axis_mapping must use the exact accepted POV axis root.")
    mapping = ActorPovAxisMappingV1.model_validate(
        axis_mapping.model_dump(mode="python")
    )
    target_public_ids = mapping.target_action_recipient_public_agent_id_by_id
    if len(target_public_ids) != 11 or target_public_ids[0] is not None:
        raise ValueError("accepted POV target axis must retain target-none at zero.")
    targets: list[TargetNoneActionDisplayRowV1 | TargetAgentActionDisplayRowV1] = [
        TargetNoneActionDisplayRowV1(
            target_kind="target_none",
            target_action=0,
            display_name=mapping.target_action_name_by_id[0],
        )
    ]
    for target_action, target_public_agent_id in enumerate(
        target_public_ids[1:],
        start=1,
    ):
        if type(target_public_agent_id) is not str:
            raise ValueError("accepted positive POV target identities must be textual.")
        targets.append(
            TargetAgentActionDisplayRowV1(
                target_kind="public_agent",
                target_action=target_action,
                display_name=mapping.target_action_name_by_id[target_action],
                target_public_agent_id=target_public_agent_id,
                target_relation=("same_team" if target_action <= 5 else "opponent"),
            )
        )
    return AgentPovActionAxisV1(
        axis_kind="agent_pov_action_axis",
        owner_presentation_key=owner_presentation_key,
        owner_public_agent_id=owner_public_agent_id,
        movement_actions=tuple(
            MovementActionDisplayRowV1(
                move_action=move_action,
                display_name=display_name,
            )
            for move_action, display_name in enumerate(
                mapping.movement_action_name_by_id
            )
        ),
        target_actions=tuple(targets),
        ultimate_choices=tuple(
            UltimateChoiceDisplayRowV1(
                use_ultimate_action=use_ultimate_action,
                display_name=display_name,
            )
            for use_ultimate_action, display_name in enumerate(
                mapping.use_ultimate_action_name_by_id
            )
        ),
    )


class LatestTransitionActionRowV1(_PresentationProtocolModel):
    actor_presentation_key: _ScientificId
    actor_public_agent_id: _ScientificId
    target_action_recipient_public_agent_id_by_id: tuple[_ScientificId | None, ...]
    submitted_action: SubmittedActionTupleV1
    accepted_action: AcceptedActionTupleV1

    @model_validator(mode="after")
    def _validate_row(self) -> Self:
        if (
            len(self.target_action_recipient_public_agent_id_by_id) != 11
            or self.target_action_recipient_public_agent_id_by_id[0] is not None
            or any(
                value is None
                for value in self.target_action_recipient_public_agent_id_by_id[1:]
            )
        ):
            raise ValueError("Latest Transition target axis must have 11 exact rows.")
        _require_ordered_unique(
            cast(
                tuple[object, ...],
                self.target_action_recipient_public_agent_id_by_id[1:],
            ),
            name="Latest Transition target public identities",
        )
        _require_exact_type(
            self.submitted_action,
            SubmittedActionTupleV1,
            name="submitted_action",
        )
        _require_exact_type(
            self.accepted_action,
            AcceptedActionTupleV1,
            name="accepted_action",
        )
        return self


class _LatestTransitionBaseV1(_PresentationProtocolModel):
    episode_id: _ScientificId
    incoming_transition_index: _NonNegativeInt
    incoming_transition_id: _ScientificId
    incoming_start_frame_id: _ScientificId
    incoming_successor_frame_id: _ScientificId
    incoming_start_simulator_step_count: _NonNegativeInt
    incoming_successor_simulator_step_count: _NonNegativeInt
    action_rows: tuple[LatestTransitionActionRowV1, ...]

    @model_validator(mode="after")
    def _validate_epoch(self) -> Self:
        if self.incoming_successor_simulator_step_count != (
            self.incoming_start_simulator_step_count + 1
        ):
            raise ValueError("Latest Transition ticks must be adjacent.")
        if not self.action_rows or any(
            type(row) is not LatestTransitionActionRowV1 for row in self.action_rows
        ):
            raise ValueError("Latest Transition requires exact action rows.")
        _require_ordered_unique(
            tuple(row.actor_public_agent_id for row in self.action_rows),
            name="Latest Transition actors",
        )
        return self


class OracleLatestTransitionV1(_LatestTransitionBaseV1):
    transition_kind: Literal["oracle_incoming_submitted_accepted"]

    @model_validator(mode="after")
    def _validate_ids(self) -> Self:
        prefix = self.episode_id
        _validate_latest_transition_ids(self, prefix=prefix)
        return self


class NoSharedObsLatestTransitionV1(_LatestTransitionBaseV1):
    transition_kind: Literal["no_shared_obs_incoming_submitted_accepted"]
    recipient_public_agent_id: _ScientificId
    recipient_presentation_key: _ScientificId

    @model_validator(mode="after")
    def _validate_ids(self) -> Self:
        prefix = f"{self.episode_id}:actor-pov:{self.recipient_public_agent_id}"
        _validate_agent_latest_transition(self, prefix=prefix)
        return self


class SharedObsLatestTransitionV1(_LatestTransitionBaseV1):
    transition_kind: Literal["shared_obs_incoming_submitted_accepted"]
    recipient_public_agent_id: _ScientificId
    recipient_presentation_key: _ScientificId

    @model_validator(mode="after")
    def _validate_ids(self) -> Self:
        prefix = (
            f"{self.episode_id}:shared-obs-visual-union:"
            f"{self.recipient_public_agent_id}"
        )
        _validate_agent_latest_transition(self, prefix=prefix)
        return self


class _UpcomingTransitionBaseV1(_PresentationProtocolModel):
    episode_id: _ScientificId
    outgoing_transition_index: _NonNegativeInt
    outgoing_transition_id: _ScientificId
    outgoing_start_frame_id: _ScientificId
    outgoing_successor_frame_id: _ScientificId
    outgoing_start_simulator_step_count: _NonNegativeInt
    outgoing_successor_simulator_step_count: _NonNegativeInt
    action_rows: tuple[LatestTransitionActionRowV1, ...]

    @model_validator(mode="after")
    def _validate_epoch(self) -> Self:
        if self.outgoing_successor_simulator_step_count != (
            self.outgoing_start_simulator_step_count + 1
        ):
            raise ValueError("Upcoming Transition ticks must be adjacent.")
        if not self.action_rows or any(
            type(row) is not LatestTransitionActionRowV1 for row in self.action_rows
        ):
            raise ValueError("Upcoming Transition requires exact action rows.")
        _require_ordered_unique(
            tuple(row.actor_public_agent_id for row in self.action_rows),
            name="Upcoming Transition actors",
        )
        return self


class OracleUpcomingTransitionV1(_UpcomingTransitionBaseV1):
    transition_kind: Literal["oracle_outgoing_submitted_accepted"]

    @model_validator(mode="after")
    def _validate_ids(self) -> Self:
        _validate_upcoming_transition_ids(self, prefix=self.episode_id)
        return self


class NoSharedObsUpcomingTransitionV1(_UpcomingTransitionBaseV1):
    transition_kind: Literal["no_shared_obs_outgoing_submitted_accepted"]
    recipient_public_agent_id: _ScientificId
    recipient_presentation_key: _ScientificId

    @model_validator(mode="after")
    def _validate_ids(self) -> Self:
        prefix = f"{self.episode_id}:actor-pov:{self.recipient_public_agent_id}"
        _validate_agent_upcoming_transition(self, prefix=prefix)
        return self


class SharedObsUpcomingTransitionV1(_UpcomingTransitionBaseV1):
    transition_kind: Literal["shared_obs_outgoing_submitted_accepted"]
    recipient_public_agent_id: _ScientificId
    recipient_presentation_key: _ScientificId

    @model_validator(mode="after")
    def _validate_ids(self) -> Self:
        prefix = (
            f"{self.episode_id}:shared-obs-visual-union:"
            f"{self.recipient_public_agent_id}"
        )
        _validate_agent_upcoming_transition(self, prefix=prefix)
        return self


def _validate_latest_transition_ids(
    transition: _LatestTransitionBaseV1,
    *,
    prefix: str,
) -> None:
    index = transition.incoming_transition_index
    if (
        transition.incoming_transition_id != f"{prefix}:transition:{index}"
        or transition.incoming_start_frame_id != f"{prefix}:frame:{index}"
        or transition.incoming_successor_frame_id != f"{prefix}:frame:{index + 1}"
    ):
        raise ValueError("Latest Transition IDs do not retain one adjacent epoch.")


def _validate_agent_latest_transition(
    transition: NoSharedObsLatestTransitionV1 | SharedObsLatestTransitionV1,
    *,
    prefix: str,
) -> None:
    _validate_latest_transition_ids(transition, prefix=prefix)
    if len(transition.action_rows) != 1:
        raise ValueError("Agent Latest Transition requires exactly the fixed owner.")
    row = transition.action_rows[0]
    if (
        row.actor_public_agent_id != transition.recipient_public_agent_id
        or row.actor_presentation_key != transition.recipient_presentation_key
    ):
        raise ValueError("Agent Latest Transition row must equal the fixed owner.")


def _validate_upcoming_transition_ids(
    transition: _UpcomingTransitionBaseV1,
    *,
    prefix: str,
) -> None:
    index = transition.outgoing_transition_index
    if (
        transition.outgoing_transition_id != f"{prefix}:transition:{index}"
        or transition.outgoing_start_frame_id != f"{prefix}:frame:{index}"
        or transition.outgoing_successor_frame_id != f"{prefix}:frame:{index + 1}"
    ):
        raise ValueError("Upcoming Transition IDs do not retain one adjacent epoch.")


def _validate_agent_upcoming_transition(
    transition: NoSharedObsUpcomingTransitionV1 | SharedObsUpcomingTransitionV1,
    *,
    prefix: str,
) -> None:
    _validate_upcoming_transition_ids(transition, prefix=prefix)
    if len(transition.action_rows) != 1:
        raise ValueError("Agent Upcoming Transition requires exactly the fixed owner.")
    row = transition.action_rows[0]
    if (
        row.actor_public_agent_id != transition.recipient_public_agent_id
        or row.actor_presentation_key != transition.recipient_presentation_key
    ):
        raise ValueError("Agent Upcoming Transition row must equal the fixed owner.")


class ReplayResearcherSpaceV1(_PresentationProtocolModel):
    """Global researcher facts that deliberately omit battlefield geometry."""

    researcher_space_kind: Literal["global_replay_researcher_space"]
    episode_id: _ScientificId
    frame_index: _NonNegativeInt
    final_frame_index: _NonNegativeInt
    simulator_step_count: _NonNegativeInt
    selected_public_agent_id: _ScientificId
    identity_directory: OraclePublicIdentityDirectoryV1
    roster_agents: tuple[ReplayResearcherRosterAgentV1, ...]
    class_mechanics: tuple[AuthorizedClassMechanics, ...]
    latest_transition: OracleLatestTransitionV1 | None
    upcoming_transition: OracleUpcomingTransitionV1 | None

    @model_validator(mode="after")
    def _validate_space(self) -> Self:
        if self.frame_index > self.final_frame_index:
            raise ValueError("researcher-space frame exceeds the replay prefix.")
        active_directory = tuple(
            row for row in self.identity_directory.identities if row.configured_active
        )
        if len(self.roster_agents) != len(active_directory) or any(
            type(row) is not ReplayResearcherRosterAgentV1 for row in self.roster_agents
        ):
            raise ValueError("researcher roster must exactly cover active identities.")
        for roster, directory in zip(
            self.roster_agents,
            active_directory,
            strict=True,
        ):
            if (
                roster.public_agent_id != directory.public_agent_id
                or roster.team_id != directory.team_id
                or roster.team_local_slot != directory.team_local_slot
                or roster.class_id != directory.class_id
                or roster.class_name != directory.class_name
            ):
                raise ValueError("researcher roster does not join its directory.")
        if self.selected_public_agent_id not in {
            row.public_agent_id for row in self.roster_agents
        }:
            raise ValueError("researcher selection must name an active roster row.")
        represented_classes = tuple(
            sorted({row.class_id for row in self.roster_agents})
        )
        if tuple(
            row.class_id for row in self.class_mechanics
        ) != represented_classes or any(
            type(row) not in (AuthorizedClassMechanicsV1, AuthorizedClassMechanicsV2)
            for row in self.class_mechanics
        ):
            raise ValueError(
                "researcher class mechanics must cover represented classes exactly."
            )
        self._validate_transition(self.latest_transition, incoming=True)
        self._validate_transition(self.upcoming_transition, incoming=False)
        return self

    def _validate_transition(
        self,
        transition: OracleLatestTransitionV1 | OracleUpcomingTransitionV1 | None,
        *,
        incoming: bool,
    ) -> None:
        expected_present = (
            self.frame_index > 0
            if incoming
            else (self.frame_index < self.final_frame_index)
        )
        expected_type = (
            OracleLatestTransitionV1 if incoming else OracleUpcomingTransitionV1
        )
        if not expected_present:
            if transition is not None:
                raise ValueError("researcher transition presence changed at an edge.")
            return
        _require_exact_type(transition, expected_type, name="researcher transition")
        checked = cast(
            OracleLatestTransitionV1 | OracleUpcomingTransitionV1,
            transition,
        )
        if incoming:
            latest = cast(OracleLatestTransitionV1, checked)
            if (
                latest.episode_id != self.episode_id
                or latest.incoming_transition_index != self.frame_index - 1
                or latest.incoming_successor_frame_id
                != f"{self.episode_id}:frame:{self.frame_index}"
                or latest.incoming_successor_simulator_step_count
                != self.simulator_step_count
            ):
                raise ValueError("researcher Latest Transition misses current s_n.")
            action_rows = latest.action_rows
        else:
            upcoming = cast(OracleUpcomingTransitionV1, checked)
            if (
                upcoming.episode_id != self.episode_id
                or upcoming.outgoing_transition_index != self.frame_index
                or upcoming.outgoing_start_frame_id
                != f"{self.episode_id}:frame:{self.frame_index}"
                or upcoming.outgoing_start_simulator_step_count
                != self.simulator_step_count
            ):
                raise ValueError("researcher Upcoming Transition does not leave s_n.")
            action_rows = upcoming.action_rows
        active_ids = tuple(row.public_agent_id for row in self.roster_agents)
        if tuple(row.actor_public_agent_id for row in action_rows) != active_ids:
            raise ValueError("researcher transition rows changed active roster order.")
        roster_by_id = {row.public_agent_id: row for row in self.roster_agents}
        directory_ids = {
            row.public_agent_id for row in self.identity_directory.identities
        }
        directory_by_id = {
            row.public_agent_id: row for row in self.identity_directory.identities
        }
        for row in action_rows:
            actor = roster_by_id[row.actor_public_agent_id]
            directory = directory_by_id[row.actor_public_agent_id]
            target_axis = cast(
                tuple[str, ...],
                row.target_action_recipient_public_agent_id_by_id[1:],
            )
            if (
                row.actor_presentation_key != actor.presentation_key
                or set(target_axis) != directory_ids
                or target_axis
                != _oracle_target_public_axis(
                    self.identity_directory,
                    owner_team_id=directory.team_id,
                )
            ):
                raise ValueError("researcher transition action identity is invalid.")


class LiveOracleTechnicalFrameV1(_PresentationProtocolModel):
    technical_kind: Literal["live_oracle_technical_frame"]
    episode_id: _ScientificId
    evaluation_frame_index: _NonNegativeInt
    simulator_step_count: _NonNegativeInt
    incoming_transition_id: _ScientificId | None


class LiveNoSharedObsTechnicalFrameV1(_PresentationProtocolModel):
    technical_kind: Literal["live_no_shared_obs_technical_frame"]
    episode_id: _ScientificId
    recipient_frame_index: _NonNegativeInt
    simulator_step_count: _NonNegativeInt
    incoming_recipient_transition_id: _ScientificId | None


class ReplayOracleTechnicalFrameV1(_PresentationProtocolModel):
    technical_kind: Literal["replay_oracle_technical_frame"]
    artifact_digest_prefix: _Sha256Prefix
    frame_index: _NonNegativeInt
    simulator_step_count: _NonNegativeInt
    incoming_transition_id: _ScientificId | None
    recorded_ordinary_movement_distance_scale: Annotated[float, Field(gt=0.0)]

    @model_validator(mode="after")
    def _validate_scale(self) -> Self:
        if not isfinite(self.recorded_ordinary_movement_distance_scale):
            raise ValueError("recorded movement scale must be finite.")
        return self


class ReplayNoSharedObsTechnicalFrameV1(_PresentationProtocolModel):
    technical_kind: Literal["replay_no_shared_obs_technical_frame"]
    frame_index: _NonNegativeInt
    simulator_step_count: _NonNegativeInt
    incoming_recipient_transition_id: _ScientificId | None


class ReplaySharedObsTechnicalFrameV1(_PresentationProtocolModel):
    technical_kind: Literal["replay_shared_obs_technical_frame"]
    frame_index: _NonNegativeInt
    simulator_step_count: _NonNegativeInt
    incoming_recipient_transition_id: _ScientificId | None


class PresentationApiErrorV1(_PresentationProtocolModel):
    schema_version: Literal[1]
    error_code: Literal["audience_unavailable"]
    message: Literal["Authorized presentation is unavailable for the active audience."]


_KEY_PUBLIC_FIELD_BY_KEY_FIELD = {
    "presentation_key": "public_agent_id",
    "source_presentation_key": "source_public_agent_id",
    "agent_presentation_key": "agent_public_agent_id",
    "recipient_presentation_key": "recipient_public_agent_id",
    "owner_presentation_key": "owner_public_agent_id",
    "target_presentation_key": "target_public_agent_id",
    "assigned_presentation_key": "assigned_public_agent_id",
    "actor_presentation_key": "actor_public_agent_id",
}

_WIRE_MODEL_MODULE_NAMES = (
    __name__,
    "marl_battlegrounds.evaluation.pov",
    "marl_battlegrounds.rendering.authorized_incoming",
    "marl_battlegrounds.rendering.authorized_inspection",
    "marl_battlegrounds.rendering.authorized_pov_scene",
    "marl_battlegrounds.rendering.authorized_presentation",
)
_allowed_exact_runtime_types_cache: frozenset[type[object]] | None = None


def _object_field_values(value: object) -> Mapping[str, object] | None:
    if isinstance(value, BaseModel):
        field_names = set(type(value).model_fields)
        if (
            set(value.__dict__) != field_names
            or getattr(value, "__pydantic_extra__", None)
            or getattr(value, "__pydantic_private__", None)
        ):
            raise ValueError(
                "presentation model runtime fields must exactly equal its schema."
            )
        return {name: value.__dict__[name] for name in type(value).model_fields}
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: getattr(value, field.name) for field in fields(value)}
    return None


def _validate_exact_runtime_tree_matches(
    candidate: object,
    canonical: object,
    *,
    path: str,
) -> None:
    """Reject any runtime type or value changed by serialize/revalidate."""
    if type(candidate) is not type(canonical):
        raise ValueError(f"presentation runtime type changed at {path}.")
    if candidate is None or type(candidate) in (str, int, float, bool):
        if candidate != canonical:
            raise ValueError(f"presentation scalar value changed at {path}.")
        return
    if type(candidate) is tuple:
        candidate_items = cast(tuple[object, ...], candidate)
        canonical_items = cast(tuple[object, ...], canonical)
        if len(candidate_items) != len(canonical_items):
            raise ValueError(f"presentation tuple length changed at {path}.")
        for index, (candidate_item, canonical_item) in enumerate(
            zip(candidate_items, canonical_items, strict=True)
        ):
            _validate_exact_runtime_tree_matches(
                candidate_item,
                canonical_item,
                path=f"{path}[{index}]",
            )
        return
    candidate_values = _object_field_values(candidate)
    canonical_values = _object_field_values(canonical)
    if candidate_values is None or canonical_values is None:
        raise ValueError(f"unsupported reachable presentation value at {path}.")
    if candidate_values.keys() != canonical_values.keys():
        raise ValueError(f"presentation runtime fields changed at {path}.")
    for name in candidate_values:
        _validate_exact_runtime_tree_matches(
            candidate_values[name],
            canonical_values[name],
            path=f"{path}.{name}",
        )


def _validate_recursive_exact_runtime_types(root: object) -> None:
    """Reject nested subclasses before serialization can normalize them."""
    global _allowed_exact_runtime_types_cache
    if _allowed_exact_runtime_types_cache is None:
        schema = _AUTHORIZED_PRESENTATION_FRAME_ADAPTER.json_schema()
        definition_names = set(cast(dict[str, object], schema["$defs"]))
        exact_types: set[type[object]] = set()
        exact_type_by_definition_name: dict[str, type[object]] = {}
        for module_name in _WIRE_MODEL_MODULE_NAMES:
            module = sys.modules.get(module_name)
            if module is None:
                continue
            for name, candidate in vars(module).items():
                if name in definition_names and isinstance(candidate, type):
                    candidate_type = cast(type[object], candidate)
                    previous = exact_type_by_definition_name.get(name)
                    if previous is not None and previous is not candidate_type:
                        raise RuntimeError(
                            "presentation schema definition names must map to one "
                            "exact runtime type."
                        )
                    exact_type_by_definition_name[name] = candidate_type
                    exact_types.add(candidate_type)
        exact_types.update((*_ENDPOINT_CONTENT_TYPES, PresentationApiErrorV1))
        _allowed_exact_runtime_types_cache = frozenset(exact_types)
    allowed_exact_types = _allowed_exact_runtime_types_cache
    visited: set[int] = set()

    def visit(value: object, *, path: str) -> None:
        if value is None or type(value) in (str, int, float, bool):
            return
        if type(value) is tuple:
            sequence = cast(tuple[object, ...], value)
            for index, item in enumerate(sequence):
                visit(item, path=f"{path}[{index}]")
            return
        if type(value) is list:
            raise ValueError(
                f"presentation tuple field was replaced by a list at {path}."
            )
        values = _object_field_values(value)
        if values is None:
            raise ValueError(f"unsupported reachable presentation value at {path}.")
        identity = id(value)
        if identity in visited:
            return
        visited.add(identity)
        runtime_type = type(value)
        if runtime_type not in allowed_exact_types:
            raise ValueError(
                f"presentation value at {path} must use a canonical exact runtime type."
            )
        for name, nested in values.items():
            visit(nested, path=f"{path}.{name}")

    visit(root, path="presentation")


def _validate_recursive_presentation_keys(
    root: object,
    *,
    source_session_id: str,
    audience: Literal["oracle", "agent_pov"],
    recipient_public_agent_id: str | None,
    excluded_root_fields: frozenset[str] = frozenset(),
) -> None:
    _validate_recursive_exact_runtime_types(root)
    visited: set[int] = set()

    def visit(value: object, *, path: str) -> None:
        if value is None or type(value) in (str, int, float, bool):
            return
        if type(value) is tuple:
            sequence = cast(tuple[object, ...], value)
            for index, item in enumerate(sequence):
                visit(item, path=f"{path}[{index}]")
            return
        values = _object_field_values(value)
        if values is None:
            raise ValueError(f"unsupported reachable presentation value at {path}.")
        identity = id(value)
        if identity in visited:
            return
        visited.add(identity)
        key_fields = tuple(
            name
            for name in values
            if name == "presentation_key" or name.endswith("_presentation_key")
        )
        for key_field in key_fields:
            public_field = _KEY_PUBLIC_FIELD_BY_KEY_FIELD.get(key_field)
            if public_field is None or public_field not in values:
                raise ValueError(
                    f"unrecognized presentation-key field {path}.{key_field}."
                )
            key = values[key_field]
            public_id = values[public_field]
            if (key is None) != (public_id is None):
                raise ValueError(
                    f"nullable presentation identity pair is incomplete at {path}."
                )
            if key is None:
                continue
            if type(key) is not str or type(public_id) is not str:
                raise ValueError(
                    f"presentation identity pair is not textual at {path}."
                )
            if audience == "oracle":
                expected = oracle_presentation_key_v1(
                    authority_session_id=source_session_id,
                    public_agent_id=public_id,
                )
            else:
                if recipient_public_agent_id is None:  # pragma: no cover - narrowed.
                    raise AssertionError("Agent key visitor lost its recipient")
                expected = pov_presentation_key_v1(
                    authority_session_id=source_session_id,
                    recipient_public_agent_id=recipient_public_agent_id,
                    public_agent_id=public_id,
                )
            if key != expected:
                raise ValueError(f"presentation key does not join authority at {path}.")
        for name, nested in values.items():
            if path == "presentation" and name in excluded_root_fields:
                continue
            visit(nested, path=f"{path}.{name}")

    visit(root, path="presentation")


def _validate_source_endpoint_join(
    source: LiveOraclePresentationSourceIdentityV1
    | LiveNoSharedObsPresentationSourceIdentityV1
    | ReplayOraclePresentationSourceIdentityV1
    | ReplayNoSharedObsPresentationSourceIdentityV1
    | ReplaySharedObsPresentationSourceIdentityV1,
    endpoint: OracleAuthorizedCurrentEndpointV1
    | NoSharedObsAuthorizedCurrentEndpointV1
    | SharedObsAuthorizedCurrentEndpointV1,
) -> None:
    digest = source.source_authorized_endpoint_digest_sha256
    if digest != endpoint.authorized_endpoint_digest_sha256:
        raise ValueError("source and current endpoint digests do not join.")


def _validate_decision_axis(
    axis: OracleActionAxisV1 | AgentPovActionAxisV1,
    decision_mask: AuthorizedDecisionMaskV1,
    *,
    scene: AuthorizedBattlefieldSceneV1,
) -> None:
    if (
        decision_mask.owner_presentation_key != axis.owner_presentation_key
        or decision_mask.owner_public_agent_id != axis.owner_public_agent_id
        or decision_mask.movement_action_display_names
        != tuple(row.display_name for row in axis.movement_actions)
        or decision_mask.use_ultimate_action_display_names
        != tuple(row.display_name for row in axis.ultimate_choices)
    ):
        raise ValueError("inspection decision mask does not join its action axis.")
    scene_by_id = {row.public_agent_id: row for row in scene.agents}
    for expected, actual in zip(
        axis.target_actions, decision_mask.target_actions, strict=True
    ):
        if (
            expected.target_action != actual.target_action
            or expected.display_name != actual.display_name
        ):
            raise ValueError("inspection target row does not join its action axis.")
        if type(expected) is TargetNoneActionDisplayRowV1:
            if type(actual) is not AuthorizedNoTargetActionV1:
                raise ValueError("target action zero must remain target-none.")
            continue
        expected_agent = cast(TargetAgentActionDisplayRowV1, expected)
        if type(actual) not in (
            AuthorizedVisibleTargetActionV1,
            AuthorizedAxisOnlyTargetActionV1,
        ):
            raise ValueError(
                "inspection target identity does not join its action axis."
            )
        actual_agent = cast(
            AuthorizedVisibleTargetActionV1 | AuthorizedAxisOnlyTargetActionV1,
            actual,
        )
        if actual_agent.target_public_agent_id != expected_agent.target_public_agent_id:
            raise ValueError(
                "inspection target identity does not join its action axis."
            )
        visible = scene_by_id.get(actual_agent.target_public_agent_id)
        if visible is None:
            if type(actual) is not AuthorizedAxisOnlyTargetActionV1:
                raise ValueError(
                    "a target absent from the scene must remain axis-only."
                )
            continue
        if type(actual) is not AuthorizedVisibleTargetActionV1:
            raise ValueError(
                "a target present in the scene must remain a visible target."
            )
        if (
            visible.presentation_key != actual.target_presentation_key
            or visible.position != actual.target_anchor
        ):
            raise ValueError("visible inspection target does not join the scene.")


def _validate_oracle_incoming(
    *,
    source: LiveOraclePresentationSourceIdentityV1
    | ReplayOraclePresentationSourceIdentityV1,
    endpoint: OracleAuthorizedCurrentEndpointV1,
    latest_events: ReplayIncomingSummaryV1 | None,
    latest_transition: OracleLatestTransitionV1 | None,
) -> None:
    index = source.source_frame_index
    if index == 0:
        if latest_events is not None or latest_transition is not None:
            raise ValueError("frame zero cannot carry incoming presentation facts.")
        return
    _require_exact_type(latest_events, ReplayIncomingSummaryV1, name="latest_events")
    _require_exact_type(
        latest_transition,
        OracleLatestTransitionV1,
        name="latest_transition",
    )
    events = cast(ReplayIncomingSummaryV1, latest_events)
    transition = cast(OracleLatestTransitionV1, latest_transition)
    expected_transition_id = f"{source.episode_id}:transition:{index - 1}"
    expected_start_id = f"{source.episode_id}:frame:{index - 1}"
    if (
        events.incoming_transition_index != index - 1
        or events.incoming_transition_id != expected_transition_id
        or events.incoming_start_frame_id != expected_start_id
        or events.incoming_successor_frame_id != endpoint.frame_id
        or events.incoming_successor_simulator_step_count
        != endpoint.simulator_step_count
        or transition.incoming_transition_index != index - 1
        or transition.incoming_transition_id != expected_transition_id
        or transition.incoming_start_frame_id != expected_start_id
        or transition.incoming_successor_frame_id != endpoint.frame_id
        or transition.incoming_start_simulator_step_count
        != events.incoming_start_simulator_step_count
        or transition.incoming_successor_simulator_step_count
        != events.incoming_successor_simulator_step_count
    ):
        raise ValueError("Oracle incoming branches do not enter the current endpoint.")
    incoming_successors = tuple(
        (
            row.agent_presentation_key,
            row.agent_public_agent_id,
            row.successor.position,
        )
        for row in events.agent_phase_trajectories
    )
    current_agents = tuple(
        (row.presentation_key, row.public_agent_id, row.position)
        for row in endpoint.scene.agents
    )
    if incoming_successors != current_agents:
        raise ValueError("Oracle incoming successors do not join the current scene.")
    directory_by_id = {
        row.public_agent_id: row for row in endpoint.identity_directory.identities
    }
    action_row_by_id = {
        row.actor_public_agent_id: row for row in transition.action_rows
    }
    active_rejection_actor_ids: set[str] = set()
    for event in events.events:
        if type(event) is ReplayIncomingActionRejectedEventV1:
            identity = event.actor_identity
            directory = directory_by_id.get(identity.public_agent_id)
            if directory is None:
                raise ValueError(
                    "incoming rejection identity is outside the directory."
                )
            if type(identity) is ReplayIncomingAuthorizedAgentIdentityV1:
                if not directory.configured_active:
                    raise ValueError("authorized rejection identity must be active.")
                action_row = action_row_by_id.get(identity.public_agent_id)
                if (
                    action_row is None
                    or action_row.submitted_action != event.submitted_action
                ):
                    raise ValueError(
                        "active rejection submitted tuple must join Latest Transition."
                    )
                active_rejection_actor_ids.add(identity.public_agent_id)
            elif type(identity) is ReplayIncomingFeedOnlyAgentIdentityV1:
                if directory.configured_active:
                    raise ValueError("feed-only rejection identity must be inactive.")
            else:  # pragma: no cover - exact rendering DTO owns this.
                raise AssertionError("incoming rejection identity variant disappeared")
    rows_with_rejection = {
        row.actor_public_agent_id
        for row in transition.action_rows
        if (
            row.submitted_action.move_action != row.accepted_action.move_action
            or row.submitted_action.target_action != row.accepted_action.target_action
            or row.submitted_action.use_ultimate_action
            != row.accepted_action.use_ultimate_action
        )
    }
    if rows_with_rejection != active_rejection_actor_ids:
        raise ValueError(
            "Oracle active rejection events do not equal rejected action rows."
        )
    expected_active = tuple(
        row.public_agent_id
        for row in endpoint.identity_directory.identities
        if row.configured_active
    )
    if (
        tuple(row.actor_public_agent_id for row in transition.action_rows)
        != expected_active
    ):
        raise ValueError(
            "Oracle Latest Transition rows must equal active directory order."
        )
    directory_ids = set(directory_by_id)
    for row in transition.action_rows:
        actor = directory_by_id[row.actor_public_agent_id]
        target_axis = cast(
            tuple[str, ...],
            row.target_action_recipient_public_agent_id_by_id[1:],
        )
        if set(target_axis) != directory_ids:
            raise ValueError(
                "Oracle Latest Transition target axis must equal directory identities."
            )
        if target_axis != _oracle_target_public_axis(
            endpoint.identity_directory,
            owner_team_id=actor.team_id,
        ):
            raise ValueError(
                "Oracle Latest Transition target axis order changed action semantics."
            )


def _validate_agent_incoming(
    *,
    source: LiveNoSharedObsPresentationSourceIdentityV1
    | ReplayNoSharedObsPresentationSourceIdentityV1
    | ReplaySharedObsPresentationSourceIdentityV1,
    endpoint: NoSharedObsAuthorizedCurrentEndpointV1
    | SharedObsAuthorizedCurrentEndpointV1,
    latest_events: NoSharedObsIncomingSummaryV1 | SharedObsIncomingSummaryV1 | None,
    latest_transition: NoSharedObsLatestTransitionV1
    | SharedObsLatestTransitionV1
    | None,
    shared: bool,
) -> None:
    index = source.source_frame_index
    parts = endpoint.parts
    if index == 0:
        if latest_events is not None or latest_transition is not None:
            raise ValueError("frame zero cannot carry incoming presentation facts.")
        return
    expected_events_type = (
        SharedObsIncomingSummaryV1 if shared else NoSharedObsIncomingSummaryV1
    )
    expected_transition_type = (
        SharedObsLatestTransitionV1 if shared else NoSharedObsLatestTransitionV1
    )
    _require_exact_type(latest_events, expected_events_type, name="latest_events")
    _require_exact_type(
        latest_transition,
        expected_transition_type,
        name="latest_transition",
    )
    events = cast(
        NoSharedObsIncomingSummaryV1 | SharedObsIncomingSummaryV1,
        latest_events,
    )
    transition = cast(
        NoSharedObsLatestTransitionV1 | SharedObsLatestTransitionV1,
        latest_transition,
    )
    expected_prefix = (
        f"{source.episode_id}:shared-obs-visual-union:"
        if shared
        else f"{source.episode_id}:actor-pov:"
    ) + source.source_recipient_public_agent_id
    expected_transition_id = f"{expected_prefix}:transition:{index - 1}"
    expected_start_id = f"{expected_prefix}:frame:{index - 1}"
    if (
        events.source_episode_id != source.episode_id
        or events.recipient_public_agent_id != source.source_recipient_public_agent_id
        or events.recipient_presentation_key != parts.recipient_presentation_key
        or events.incoming_transition_index != index - 1
        or events.incoming_recipient_transition_id != expected_transition_id
        or events.incoming_start_recipient_frame_id != expected_start_id
        or events.incoming_successor_recipient_frame_id
        != source.source_recipient_frame_id
        or events.incoming_successor_simulator_step_count
        != source.source_simulator_step_count
        or transition.incoming_transition_index != index - 1
        or transition.incoming_transition_id != expected_transition_id
        or transition.incoming_start_frame_id != expected_start_id
        or transition.incoming_successor_frame_id != source.source_recipient_frame_id
        or transition.incoming_start_simulator_step_count
        != events.incoming_start_simulator_step_count
        or transition.incoming_successor_simulator_step_count
        != events.incoming_successor_simulator_step_count
        or transition.recipient_public_agent_id
        != source.source_recipient_public_agent_id
        or transition.recipient_presentation_key != parts.recipient_presentation_key
    ):
        raise ValueError("Agent incoming branches do not enter the current endpoint.")
    if (
        transition.action_rows[0].target_action_recipient_public_agent_id_by_id
        != endpoint.action_axis.target_public_agent_id_by_action
    ):
        raise ValueError("Agent Latest Transition target axis must equal current axis.")
    if not shared:
        outcome_cue = cast(NoSharedObsIncomingSummaryV1, events).cues[0]
        if type(outcome_cue) is not NoSharedObsOwnActionOutcomeIncomingCueV1:
            raise ValueError(
                "NoSharedObs own-action outcome cue must use its exact root."
            )
        action_row = transition.action_rows[0]
        submitted = action_row.submitted_action
        accepted = action_row.accepted_action
        accepted_without_rejection = (
            submitted.move_action == accepted.move_action
            and submitted.target_action == accepted.target_action
            and submitted.use_ultimate_action == accepted.use_ultimate_action
        )
        expected_outcome = "accepted" if accepted_without_rejection else "rejected"
        if outcome_cue.outcome != expected_outcome:
            raise ValueError(
                "NoSharedObs action outcome cue does not join Latest Transition."
            )
    _validate_agent_latest_event_axis(
        cast(
            NoSharedObsIncomingSummaryV1 | SharedObsIncomingSummaryV1,
            latest_events,
        ),
        endpoint.action_axis,
        parts=parts,
    )


def _validate_agent_latest_event_axis(
    latest_events: NoSharedObsIncomingSummaryV1 | SharedObsIncomingSummaryV1,
    action_axis: AgentPovActionAxisV1,
    *,
    parts: NoSharedObsPresentationEndpointPartsV1
    | SharedObsPresentationEndpointPartsV1,
) -> None:
    scene = parts.scene
    recipient_public_agent_id = parts.recipient_public_agent_id
    allowed_public_ids = set(
        cast(tuple[str, ...], action_axis.target_public_agent_id_by_action[1:])
    )
    scene_by_id = {row.public_agent_id: row for row in scene.agents}
    recipient_rows = tuple(
        row for row in scene.agents if row.public_agent_id == recipient_public_agent_id
    )
    if len(recipient_rows) != 1:
        raise ValueError("Agent incoming validation requires one endpoint self row.")
    recipient_team_id = recipient_rows[0].team_id

    def project_observation(agent: AuthorizedAgentV1) -> AgentIncomingObservationV1:
        return AgentIncomingObservationV1(
            presentation_key=agent.presentation_key,
            public_agent_id=agent.public_agent_id,
            relation=cast(Literal["self", "ally", "opponent"], agent.relation),
            team_id=agent.team_id,
            class_id=agent.class_id,
            class_name=agent.class_name,
            position=agent.position,
            radius=agent.radius,
            life_state=agent.life_state,
            current_health=agent.current_health,
            maximum_health=agent.maximum_health,
            base_movement_speed=agent.base_movement_speed,
            effective_movement_speed=agent.effective_movement_speed,
            observation_radius=agent.observation_radius,
            basic_interaction_radius=agent.basic_interaction_radius,
            ultimate_interaction_radius=agent.ultimate_interaction_radius,
            ultimate_cooldown_remaining=agent.ultimate_cooldown_remaining,
            spawn_shield_remaining=agent.spawn_shield_remaining,
            steps_until_out_of_combat=agent.steps_until_out_of_combat,
            out_of_combat_delay_steps=agent.out_of_combat_delay_steps,
            out_of_combat_health_regeneration_fraction_per_step=(
                agent.out_of_combat_health_regeneration_fraction_per_step
            ),
            statuses=tuple(
                AgentIncomingObservedStatusV1(
                    status_channel=status.status_channel,
                    status_id=status.status_id,
                    family=status.family,
                    configured_duration_steps=status.configured_duration_steps,
                    remaining_duration=status.remaining_duration,
                    mechanic_action_component=status.source_action_component,
                    magnitude_kind=status.magnitude_kind,
                    magnitude=status.magnitude,
                    breaks_on_positive_damage=status.breaks_on_positive_damage,
                )
                for status in agent.statuses
            ),
            aura_modifiers=agent.aura_modifiers,
        )

    def validate_observation(observation: AgentIncomingObservationV1) -> None:
        expected_relation, expected_team_id = _agent_axis_relation_and_team(
            action_axis,
            recipient_public_agent_id=recipient_public_agent_id,
            recipient_team_id=recipient_team_id,
            public_agent_id=observation.public_agent_id,
        )
        if (
            observation.relation != expected_relation
            or observation.team_id != expected_team_id
        ):
            raise ValueError(
                "Agent incoming observation relation/team does not join its axis."
            )

    def visit(value: object, *, path: str) -> None:
        if value is None or type(value) in (str, int, float, bool):
            return
        if type(value) is tuple:
            for index, item in enumerate(cast(tuple[object, ...], value)):
                visit(item, path=f"{path}[{index}]")
            return
        if type(value) is AgentIncomingObservationV1:
            validate_observation(value)
        if type(value) is SharedObsAuthorizedSensorSourceV1:
            _validate_shared_sensor_source_axis(
                value,
                action_axis,
            )
        values = _object_field_values(value)
        if values is None:
            raise ValueError(f"unsupported Agent Latest Events value at {path}.")
        for key_field, public_field in _KEY_PUBLIC_FIELD_BY_KEY_FIELD.items():
            if key_field not in values:
                continue
            public_id = values.get(public_field)
            if public_id is not None and public_id not in allowed_public_ids:
                raise ValueError(
                    "Agent Latest Events identity is outside its action axis at "
                    f"{path}."
                )
        for name, nested in values.items():
            visit(nested, path=f"{path}.{name}")

    visit(latest_events, path="latest_events")
    if type(latest_events) is NoSharedObsIncomingSummaryV1:
        self_agent = recipient_rows[0]
        expected_self = project_observation(self_agent)
        for cue in latest_events.cues:
            if type(cue) is NoSharedObsVisibleBodyChangedIncomingCueV1:
                scene_agent = scene_by_id.get(cue.agent_public_agent_id)
                if cue.successor_observation is None:
                    if scene_agent is not None:
                        raise ValueError(
                            "NoSharedObs disappearance identity remains in "
                            "current scene."
                        )
                elif scene_agent is None or cue.successor_observation != (
                    project_observation(scene_agent)
                ):
                    raise ValueError(
                        "NoSharedObs successor observation does not equal "
                        "current scene."
                    )
            elif type(cue) is NoSharedObsOwnPositionChangedIncomingCueV1:
                if cue.successor_position != self_agent.position:
                    raise ValueError(
                        "NoSharedObs own position successor does not join self."
                    )
            elif type(cue) is NoSharedObsOwnHealthChangedIncomingCueV1:
                if cue.successor_health != self_agent.current_health:
                    raise ValueError(
                        "NoSharedObs own health successor does not join self."
                    )
            elif type(cue) is NoSharedObsOwnStatusChangedIncomingCueV1:
                if cue.successor_statuses != expected_self.statuses:
                    raise ValueError(
                        "NoSharedObs own status successor does not join self."
                    )
            elif type(cue) is NoSharedObsOwnCooldownChangedIncomingCueV1:
                if (
                    cue.successor_remaining_ticks
                    != self_agent.ultimate_cooldown_remaining
                ):
                    raise ValueError(
                        "NoSharedObs own cooldown successor does not join self."
                    )
            elif type(cue) is NoSharedObsOwnLifecycleChangedIncomingCueV1:
                if (
                    not cue.successor_active
                    or cue.successor_life_state != self_agent.life_state
                    or cue.successor_spawn_shield_remaining_ticks
                    != self_agent.spawn_shield_remaining
                ):
                    raise ValueError(
                        "NoSharedObs own lifecycle successor does not join self."
                    )
        return
    if type(latest_events) is not SharedObsIncomingSummaryV1:
        raise ValueError("Agent Latest Events root is not canonical.")
    if type(parts) is not SharedObsPresentationEndpointPartsV1:
        raise ValueError("SharedObs Latest Events require SharedObs endpoint parts.")
    provenance_by_id = {
        row.agent_public_agent_id: row for row in parts.agent_observation_provenance
    }
    for delta in latest_events.deltas:
        if type(delta) is SharedObsDisappearanceIncomingDeltaV1:
            if (
                delta.agent_public_agent_id in scene_by_id
                or delta.agent_public_agent_id in provenance_by_id
            ):
                raise ValueError(
                    "SharedObs disappearance identity remains in current endpoint."
                )
            continue
        if type(delta) not in (
            SharedObsAppearanceIncomingDeltaV1,
            SharedObsObservedValuesIncomingDeltaV1,
            SharedObsObservationProvenanceIncomingDeltaV1,
        ):
            raise ValueError("SharedObs incoming delta variant is not canonical.")
        scene_row = scene_by_id.get(delta.agent_public_agent_id)
        provenance = provenance_by_id.get(delta.agent_public_agent_id)
        if (
            scene_row is None
            or provenance is None
            or provenance.agent_presentation_key != scene_row.presentation_key
        ):
            raise ValueError(
                "SharedObs successor delta identity is absent from current endpoint."
            )
        if type(delta) is SharedObsAppearanceIncomingDeltaV1:
            successor_observation = delta.successor_observation
            if successor_observation != project_observation(scene_row):
                raise ValueError(
                    "SharedObs successor observation does not equal current scene."
                )
            if delta.successor_observation_sources != provenance.observation_sources:
                raise ValueError(
                    "SharedObs appearance provenance does not equal current endpoint."
                )
        elif type(delta) is SharedObsObservedValuesIncomingDeltaV1:
            if delta.successor_observation != project_observation(scene_row):
                raise ValueError(
                    "SharedObs successor observation does not equal current scene."
                )
        elif type(delta) is SharedObsObservationProvenanceIncomingDeltaV1:
            if delta.successor_observation_sources != provenance.observation_sources:
                raise ValueError(
                    "SharedObs successor provenance does not equal current endpoint."
                )
        else:  # pragma: no cover - exact variants narrowed above.
            raise AssertionError("SharedObs delta narrowing lost its exact variant.")


def _validate_oracle_inspection(
    *,
    source: LiveOraclePresentationSourceIdentityV1
    | ReplayOraclePresentationSourceIdentityV1,
    endpoint: OracleAuthorizedCurrentEndpointV1,
    inspection: LiveDraftInspectionPresentationV1
    | ReplayInspectionPresentationV1
    | None,
    replay: bool,
) -> None:
    if inspection is None:
        if not replay:
            raise ValueError("a live Oracle endpoint requires its draft inspection.")
        replay_source = cast(ReplayOraclePresentationSourceIdentityV1, source)
        is_final = (
            replay_source.source_frame_index == replay_source.source_final_frame_index
        )
        if not is_final and endpoint.action_axis is not None:
            raise ValueError(
                "a non-final uninspected Oracle endpoint must omit its action axis."
            )
        return
    expected = (
        ReplayInspectionPresentationV1 if replay else LiveDraftInspectionPresentationV1
    )
    _require_exact_type(inspection, expected, name="inspection")
    if endpoint.action_axis is None:
        raise ValueError("an inspected Oracle endpoint requires its action axis.")
    axis = endpoint.action_axis
    if (
        inspection.actor_presentation_key != axis.owner_presentation_key
        or inspection.actor_public_agent_id != axis.owner_public_agent_id
        or inspection.current_simulator_step_count != source.source_simulator_step_count
    ):
        raise ValueError("Oracle inspection owner/epoch does not join the endpoint.")
    scene_actor = next(
        (
            row
            for row in endpoint.scene.agents
            if row.public_agent_id == inspection.actor_public_agent_id
        ),
        None,
    )
    if scene_actor is None or (
        scene_actor.presentation_key != inspection.actor_presentation_key
        or scene_actor.position != inspection.actor_anchor
    ):
        raise ValueError("Oracle inspection actor does not join the current scene.")
    _validate_decision_axis(axis, inspection.decision_mask, scene=endpoint.scene)
    if replay:
        replay_inspection = cast(ReplayInspectionPresentationV1, inspection)
        _require_exact_type(
            replay_inspection.transition_reference,
            OracleReplayTransitionReferenceV1,
            name="Oracle replay transition_reference",
        )
        if (
            replay_inspection.episode_id != source.episode_id
            or replay_inspection.outgoing_transition_index != source.source_frame_index
        ):
            raise ValueError("Oracle replay inspection is not outgoing T_n.")


def _validate_agent_inspection(
    *,
    source: LiveNoSharedObsPresentationSourceIdentityV1
    | ReplayNoSharedObsPresentationSourceIdentityV1
    | ReplaySharedObsPresentationSourceIdentityV1,
    endpoint: NoSharedObsAuthorizedCurrentEndpointV1
    | SharedObsAuthorizedCurrentEndpointV1,
    inspection: LiveDraftInspectionPresentationV1 | ReplayInspectionPresentationV1,
    replay: bool,
) -> None:
    expected = (
        ReplayInspectionPresentationV1 if replay else LiveDraftInspectionPresentationV1
    )
    _require_exact_type(inspection, expected, name="inspection")
    parts = endpoint.parts
    if (
        inspection.actor_public_agent_id != parts.recipient_public_agent_id
        or inspection.actor_presentation_key != parts.recipient_presentation_key
        or inspection.current_simulator_step_count != source.source_simulator_step_count
    ):
        raise ValueError("Agent inspection must belong to the fixed recipient epoch.")
    self_actor = next(
        (
            row
            for row in parts.scene.agents
            if row.public_agent_id == parts.recipient_public_agent_id
        ),
        None,
    )
    if self_actor is None or (
        self_actor.relation != "self"
        or self_actor.presentation_key != inspection.actor_presentation_key
        or self_actor.position != inspection.actor_anchor
    ):
        raise ValueError("Agent inspection actor anchor must join the self scene row.")
    _validate_decision_axis(
        endpoint.action_axis,
        inspection.decision_mask,
        scene=parts.scene,
    )
    source_mask = parts.next_decision_action_mask
    decision_mask = inspection.decision_mask
    if (
        decision_mask.movement_action_mask != source_mask.move
        or decision_mask.target_action_mask != source_mask.select_target
        or decision_mask.use_ultimate_action_mask != source_mask.use_ultimate
        or decision_mask.target_use_ultimate_joint_mask
        != source_mask.select_target_use_ultimate_joint
    ):
        raise ValueError(
            "Agent inspection legality must equal the endpoint decision mask."
        )
    if replay:
        replay_inspection = cast(ReplayInspectionPresentationV1, inspection)
        reference_type = (
            SharedObsReplayTransitionReferenceV1
            if type(source) is ReplaySharedObsPresentationSourceIdentityV1
            else NoSharedObsReplayTransitionReferenceV1
        )
        _require_exact_type(
            replay_inspection.transition_reference,
            reference_type,
            name="Agent replay transition_reference",
        )
        if (
            replay_inspection.episode_id != source.episode_id
            or replay_inspection.outgoing_transition_index != source.source_frame_index
        ):
            raise ValueError("Agent replay inspection is not outgoing T_n.")


class LiveEditableDraftInspectionV1(_PresentationProtocolModel):
    inspection_kind: Literal["editable_live_draft"]
    submission_scope: Literal["joint_turn", "controlled_actor"]
    draft: LiveDraftInspectionPresentationV1

    @model_validator(mode="after")
    def _validate_draft(self) -> Self:
        _require_exact_type(
            self.draft,
            LiveDraftInspectionPresentationV1,
            name="draft",
        )
        return self


class LiveScriptedPlaybackInspectionV1(_PresentationProtocolModel):
    inspection_kind: Literal["scripted_playback_inspection"]
    submission_scope: Literal["scripted_playback"]
    editable_draft_available: Literal[False]
    advance_semantics: Literal["registered_script_frame"]


type LiveInputInspectionV1 = Annotated[
    LiveEditableDraftInspectionV1 | LiveScriptedPlaybackInspectionV1,
    Field(discriminator="inspection_kind"),
]


class LiveOracleInspectionEnvelopeV1(_PresentationProtocolModel):
    envelope_kind: Literal["live_oracle_source_bound_inspection"]
    source_session_id: _OpaqueId
    source_run_generation: _NonNegativeInt
    source_revision: _NonNegativeInt
    source_authority_epoch: _NonNegativeInt
    episode_id: _ScientificId
    source_frame_index: _NonNegativeInt
    source_frame_id: _ScientificId
    source_simulator_step_count: _NonNegativeInt
    inspection: LiveInputInspectionV1

    @model_validator(mode="after")
    def _validate_inspection(self) -> Self:
        if type(self.inspection) not in (
            LiveEditableDraftInspectionV1,
            LiveScriptedPlaybackInspectionV1,
        ):
            raise ValueError("live Oracle inspection uses an unknown strict variant.")
        return self


class LiveNoSharedObsInspectionEnvelopeV1(_PresentationProtocolModel):
    envelope_kind: Literal["live_no_shared_obs_source_bound_inspection"]
    source_session_id: _OpaqueId
    source_run_generation: _NonNegativeInt
    source_revision: _NonNegativeInt
    source_authority_epoch: _NonNegativeInt
    episode_id: _ScientificId
    source_frame_index: _NonNegativeInt
    source_recipient_public_agent_id: _ScientificId
    source_recipient_frame_id: _ScientificId
    source_simulator_step_count: _NonNegativeInt
    inspection: LiveInputInspectionV1

    @model_validator(mode="after")
    def _validate_inspection(self) -> Self:
        if type(self.inspection) not in (
            LiveEditableDraftInspectionV1,
            LiveScriptedPlaybackInspectionV1,
        ):
            raise ValueError("live Agent inspection uses an unknown strict variant.")
        return self


class LiveOracleAuthorizedPresentationFrameV1(_PresentationProtocolModel):
    schema_version: Literal[1]
    presentation_kind: Literal["live_oracle"]
    product_kind: Literal["combat_debugger"]
    source: LiveOraclePresentationSourceIdentityV1
    authority: OraclePresentationAuthorityV1
    analysis_mode: Literal["analysis"]
    current_endpoint: OracleAuthorizedCurrentEndpointV1
    latest_events: ReplayIncomingSummaryV1 | None
    latest_transition: OracleLatestTransitionV1 | None
    technical_frame: LiveOracleTechnicalFrameV1
    live_inspection: LiveOracleInspectionEnvelopeV1

    @model_validator(mode="after")
    def _validate_frame(self) -> Self:
        _require_exact_type(
            self.source, LiveOraclePresentationSourceIdentityV1, name="source"
        )
        _require_exact_type(
            self.authority, OraclePresentationAuthorityV1, name="authority"
        )
        _require_exact_type(
            self.current_endpoint,
            OracleAuthorizedCurrentEndpointV1,
            name="current_endpoint",
        )
        _require_exact_type(
            self.technical_frame, LiveOracleTechnicalFrameV1, name="technical_frame"
        )
        _validate_source_endpoint_join(self.source, self.current_endpoint)
        _validate_oracle_source_endpoint(self.source, self.current_endpoint)
        _validate_oracle_incoming(
            source=self.source,
            endpoint=self.current_endpoint,
            latest_events=self.latest_events,
            latest_transition=self.latest_transition,
        )
        _validate_live_oracle_technical(self)
        _validate_live_oracle_inspection_envelope(
            self.source,
            self.current_endpoint,
            self.live_inspection,
        )
        if type(self.live_inspection.inspection) is LiveEditableDraftInspectionV1:
            _validate_oracle_inspection(
                source=self.source,
                endpoint=self.current_endpoint,
                inspection=self.live_inspection.inspection.draft,
                replay=False,
            )
        _validate_recursive_presentation_keys(
            self,
            source_session_id=self.source.source_session_id,
            audience="oracle",
            recipient_public_agent_id=None,
        )
        return self


class LiveNoSharedObsAuthorizedPresentationFrameV1(_PresentationProtocolModel):
    schema_version: Literal[1]
    presentation_kind: Literal["live_no_shared_obs_agent_pov"]
    product_kind: Literal["combat_debugger"]
    source: LiveNoSharedObsPresentationSourceIdentityV1
    authority: NoSharedObsPresentationAuthorityV1
    analysis_mode: Literal["analysis"]
    current_endpoint: NoSharedObsAuthorizedCurrentEndpointV1
    latest_events: NoSharedObsIncomingSummaryV1 | None
    visual_events: AgentPovVisualIncomingSummaryV1 | None
    latest_transition: NoSharedObsLatestTransitionV1 | None
    technical_frame: LiveNoSharedObsTechnicalFrameV1
    live_inspection: LiveNoSharedObsInspectionEnvelopeV1

    @model_validator(mode="after")
    def _validate_frame(self) -> Self:
        _validate_no_shared_agent_types(self, live=True)
        _validate_agent_common(self, shared=False)
        _validate_live_agent_technical(self)
        _validate_live_no_shared_inspection_envelope(
            self.source,
            self.current_endpoint,
            self.live_inspection,
        )
        if type(self.live_inspection.inspection) is LiveEditableDraftInspectionV1:
            _validate_agent_inspection(
                source=self.source,
                endpoint=self.current_endpoint,
                inspection=self.live_inspection.inspection.draft,
                replay=False,
            )
        return self


class ReplayOracleAuthorizedPresentationFrameV1(_PresentationProtocolModel):
    schema_version: Literal[1]
    presentation_kind: Literal["replay_oracle"]
    product_kind: Literal["replay_viewer"]
    source: ReplayOraclePresentationSourceIdentityV1
    authority: OraclePresentationAuthorityV1
    analysis_mode: Literal["analysis"]
    current_endpoint: OracleAuthorizedCurrentEndpointV1
    latest_events: ReplayIncomingSummaryV1 | None
    latest_transition: OracleLatestTransitionV1 | None
    upcoming_transition: OracleUpcomingTransitionV1 | None
    technical_frame: ReplayOracleTechnicalFrameV1
    replay_inspection: ReplayInspectionPresentationV1 | None

    @model_validator(mode="after")
    def _validate_frame(self) -> Self:
        _require_exact_type(
            self.source, ReplayOraclePresentationSourceIdentityV1, name="source"
        )
        _require_exact_type(
            self.authority, OraclePresentationAuthorityV1, name="authority"
        )
        _require_exact_type(
            self.current_endpoint,
            OracleAuthorizedCurrentEndpointV1,
            name="current_endpoint",
        )
        _require_exact_type(
            self.technical_frame, ReplayOracleTechnicalFrameV1, name="technical_frame"
        )
        _validate_source_endpoint_join(self.source, self.current_endpoint)
        _validate_oracle_source_endpoint(self.source, self.current_endpoint)
        _validate_oracle_incoming(
            source=self.source,
            endpoint=self.current_endpoint,
            latest_events=self.latest_events,
            latest_transition=self.latest_transition,
        )
        _validate_replay_oracle_upcoming(
            source=self.source,
            endpoint=self.current_endpoint,
            upcoming_transition=self.upcoming_transition,
            inspection=self.replay_inspection,
        )
        _validate_replay_oracle_technical(self)
        if (
            self.source.source_frame_index == self.source.source_final_frame_index
            and self.replay_inspection is not None
        ):
            raise ValueError("the final replay frame cannot carry outgoing intent.")
        _validate_oracle_inspection(
            source=self.source,
            endpoint=self.current_endpoint,
            inspection=self.replay_inspection,
            replay=True,
        )
        _validate_recursive_presentation_keys(
            self,
            source_session_id=self.source.source_session_id,
            audience="oracle",
            recipient_public_agent_id=None,
        )
        return self

    @property
    def current_scene(self) -> AuthorizedBattlefieldSceneV1:
        """Temporary Python-only compatibility view; absent from the wire."""
        return self.current_endpoint.scene

    @property
    def incoming_summary(self) -> ReplayIncomingSummaryV1 | None:
        """Temporary Python-only compatibility view; absent from the wire."""
        return self.latest_events

    @property
    def outgoing_inspection(self) -> ReplayInspectionPresentationV1 | None:
        """Temporary Python-only compatibility view; absent from the wire."""
        return self.replay_inspection


class ReplayNoSharedObsAuthorizedPresentationFrameV1(_PresentationProtocolModel):
    schema_version: Literal[1]
    presentation_kind: Literal["replay_no_shared_obs_agent_pov"]
    product_kind: Literal["replay_viewer"]
    source: ReplayNoSharedObsPresentationSourceIdentityV1
    authority: NoSharedObsPresentationAuthorityV1
    analysis_mode: Literal["analysis"]
    current_endpoint: NoSharedObsAuthorizedCurrentEndpointV1
    latest_events: NoSharedObsIncomingSummaryV1 | None
    visual_events: AgentPovVisualIncomingSummaryV1 | None
    latest_transition: NoSharedObsLatestTransitionV1 | None
    upcoming_transition: NoSharedObsUpcomingTransitionV1 | None
    technical_frame: ReplayNoSharedObsTechnicalFrameV1
    replay_inspection: ReplayInspectionPresentationV1 | None
    researcher_space: ReplayResearcherSpaceV1

    @model_validator(mode="after")
    def _validate_frame(self) -> Self:
        _validate_no_shared_agent_types(self, live=False)
        _validate_agent_common(self, shared=False)
        _validate_replay_agent_upcoming(self, shared=False)
        _validate_replay_agent_technical(self, shared=False)
        _validate_replay_agent_inspection_state(self)
        _validate_replay_researcher_space(self)
        return self


class ReplaySharedObsAuthorizedPresentationFrameV1(_PresentationProtocolModel):
    schema_version: Literal[1]
    presentation_kind: Literal["replay_shared_obs_agent_pov"]
    product_kind: Literal["replay_viewer"]
    source: ReplaySharedObsPresentationSourceIdentityV1
    authority: SharedObsPresentationAuthorityV1
    analysis_mode: Literal["analysis"]
    current_endpoint: SharedObsAuthorizedCurrentEndpointV1
    latest_events: SharedObsIncomingSummaryV1 | None
    visual_events: AgentPovVisualIncomingSummaryV1 | None
    latest_transition: SharedObsLatestTransitionV1 | None
    upcoming_transition: SharedObsUpcomingTransitionV1 | None
    technical_frame: ReplaySharedObsTechnicalFrameV1
    replay_inspection: ReplayInspectionPresentationV1 | None
    researcher_space: ReplayResearcherSpaceV1

    @model_validator(mode="after")
    def _validate_frame(self) -> Self:
        _require_exact_type(
            self.source, ReplaySharedObsPresentationSourceIdentityV1, name="source"
        )
        _require_exact_type(
            self.authority, SharedObsPresentationAuthorityV1, name="authority"
        )
        _require_exact_type(
            self.current_endpoint,
            SharedObsAuthorizedCurrentEndpointV1,
            name="current_endpoint",
        )
        _require_exact_type(
            self.technical_frame,
            ReplaySharedObsTechnicalFrameV1,
            name="technical_frame",
        )
        _validate_agent_common(self, shared=True)
        _validate_replay_agent_upcoming(self, shared=True)
        _validate_replay_agent_technical(self, shared=True)
        _validate_replay_agent_inspection_state(self)
        _validate_replay_researcher_space(self)
        return self


def _validate_oracle_source_endpoint(
    source: LiveOraclePresentationSourceIdentityV1
    | ReplayOraclePresentationSourceIdentityV1,
    endpoint: OracleAuthorizedCurrentEndpointV1,
) -> None:
    if (
        endpoint.episode_id != source.episode_id
        or endpoint.frame_index != source.source_frame_index
        or endpoint.frame_id != source.source_frame_id
        or endpoint.simulator_step_count != source.source_simulator_step_count
    ):
        raise ValueError("Oracle current endpoint does not join its source epoch.")


def _validate_no_shared_agent_types(
    frame: LiveNoSharedObsAuthorizedPresentationFrameV1
    | ReplayNoSharedObsAuthorizedPresentationFrameV1,
    *,
    live: bool,
) -> None:
    source_type = (
        LiveNoSharedObsPresentationSourceIdentityV1
        if live
        else ReplayNoSharedObsPresentationSourceIdentityV1
    )
    technical_type = (
        LiveNoSharedObsTechnicalFrameV1 if live else ReplayNoSharedObsTechnicalFrameV1
    )
    _require_exact_type(frame.source, source_type, name="source")
    _require_exact_type(
        frame.authority, NoSharedObsPresentationAuthorityV1, name="authority"
    )
    _require_exact_type(
        frame.current_endpoint,
        NoSharedObsAuthorizedCurrentEndpointV1,
        name="current_endpoint",
    )
    _require_exact_type(frame.technical_frame, technical_type, name="technical_frame")


def _validate_agent_visual_events(
    frame: LiveNoSharedObsAuthorizedPresentationFrameV1
    | ReplayNoSharedObsAuthorizedPresentationFrameV1
    | ReplaySharedObsAuthorizedPresentationFrameV1,
) -> None:
    source = frame.source
    endpoint = frame.current_endpoint
    index = source.source_frame_index
    if index == 0:
        if frame.visual_events is not None:
            raise ValueError("frame zero cannot carry Agent visual events.")
        return
    _require_exact_type(
        frame.visual_events,
        AgentPovVisualIncomingSummaryV1,
        name="visual_events",
    )
    events = cast(AgentPovVisualIncomingSummaryV1, frame.visual_events)
    latest_events = frame.latest_events
    latest_transition = frame.latest_transition
    if latest_events is None or latest_transition is None:
        raise ValueError("Agent visual events require both incoming fact branches.")
    if (
        events.source_episode_id != source.episode_id
        or events.recipient_public_agent_id != source.source_recipient_public_agent_id
        or events.recipient_presentation_key
        != endpoint.parts.recipient_presentation_key
        or events.incoming_transition_index != index - 1
        or events.incoming_recipient_transition_id
        != latest_events.incoming_recipient_transition_id
        or events.incoming_start_recipient_frame_id
        != latest_events.incoming_start_recipient_frame_id
        or events.incoming_successor_recipient_frame_id
        != latest_events.incoming_successor_recipient_frame_id
        or events.incoming_start_simulator_step_count
        != latest_events.incoming_start_simulator_step_count
        or events.incoming_successor_simulator_step_count
        != latest_events.incoming_successor_simulator_step_count
        or events.incoming_recipient_transition_id
        != latest_transition.incoming_transition_id
    ):
        raise ValueError(
            "Agent visual events do not join the existing recipient-local epoch."
        )
    scene_rows = {
        row.presentation_key: (row.public_agent_id, row.class_id, row.position)
        for row in endpoint.parts.scene.agents
    }
    successor_rows = {
        row.agent_presentation_key: (
            row.agent_public_agent_id,
            row.agent_class_id,
            row.successor.position,
        )
        for row in events.agent_phase_trajectories
        if row.successor is not None
    }
    if successor_rows != scene_rows:
        raise ValueError(
            "Agent visual-event successors must equal the current authorized scene."
        )
    action_row = latest_transition.action_rows[0]
    own_rejections = tuple(
        event
        for event in events.events
        if type(event) is ReplayIncomingActionRejectedEventV1
        and type(event.actor_identity) is ReplayIncomingAuthorizedAgentIdentityV1
        and event.actor_identity.public_agent_id
        == source.source_recipient_public_agent_id
    )
    submitted = action_row.submitted_action
    accepted = action_row.accepted_action
    expected_rejected = (
        submitted.move_action != accepted.move_action
        or submitted.target_action != accepted.target_action
        or submitted.use_ultimate_action != accepted.use_ultimate_action
    )
    if len(own_rejections) != int(expected_rejected) or (
        own_rejections
        and own_rejections[0].submitted_action != action_row.submitted_action
    ):
        raise ValueError("Agent own visual rejection does not join Latest Transition.")


def _validate_agent_common(
    frame: LiveNoSharedObsAuthorizedPresentationFrameV1
    | ReplayNoSharedObsAuthorizedPresentationFrameV1
    | ReplaySharedObsAuthorizedPresentationFrameV1,
    *,
    shared: bool,
) -> None:
    source = frame.source
    authority = frame.authority
    endpoint = frame.current_endpoint
    parts = endpoint.parts
    _validate_source_endpoint_join(source, endpoint)
    if (
        source.episode_id != parts.source_episode_id
        or source.source_frame_index != parts.source_frame_index
        or source.source_recipient_frame_id != parts.source_recipient_frame_id
        or source.source_simulator_step_count != parts.source_simulator_step_count
        or source.source_recipient_public_agent_id != parts.recipient_public_agent_id
        or authority.recipient_public_agent_id != parts.recipient_public_agent_id
        or authority.recipient_presentation_key != parts.recipient_presentation_key
    ):
        raise ValueError("Agent source, authority, and endpoint do not join.")
    _validate_agent_incoming(
        source=source,
        endpoint=endpoint,
        latest_events=frame.latest_events,
        latest_transition=frame.latest_transition,
        shared=shared,
    )
    _validate_agent_visual_events(frame)
    _validate_recursive_presentation_keys(
        frame,
        source_session_id=source.source_session_id,
        audience="agent_pov",
        recipient_public_agent_id=parts.recipient_public_agent_id,
        excluded_root_fields=(
            frozenset({"researcher_space"})
            if type(frame)
            in (
                ReplayNoSharedObsAuthorizedPresentationFrameV1,
                ReplaySharedObsAuthorizedPresentationFrameV1,
            )
            else frozenset()
        ),
    )


def _validate_replay_researcher_space(
    frame: ReplayNoSharedObsAuthorizedPresentationFrameV1
    | ReplaySharedObsAuthorizedPresentationFrameV1,
) -> None:
    researcher = frame.researcher_space
    _require_exact_type(
        researcher,
        ReplayResearcherSpaceV1,
        name="researcher_space",
    )
    source = frame.source
    if (
        researcher.episode_id != source.episode_id
        or researcher.frame_index != source.source_frame_index
        or researcher.final_frame_index != source.source_final_frame_index
        or researcher.simulator_step_count != source.source_simulator_step_count
        or researcher.selected_public_agent_id
        != source.source_recipient_public_agent_id
    ):
        raise ValueError(
            "Replay researcher space does not join its Agent presentation epoch."
        )
    _validate_recursive_presentation_keys(
        researcher,
        source_session_id=source.source_session_id,
        audience="oracle",
        recipient_public_agent_id=None,
    )


def _validate_live_oracle_inspection_envelope(
    source: LiveOraclePresentationSourceIdentityV1,
    endpoint: OracleAuthorizedCurrentEndpointV1,
    envelope: LiveOracleInspectionEnvelopeV1,
) -> None:
    _require_exact_type(
        envelope,
        LiveOracleInspectionEnvelopeV1,
        name="live_inspection",
    )
    axis = endpoint.action_axis
    if (
        axis is None
        or envelope.source_session_id != source.source_session_id
        or envelope.source_run_generation != source.source_run_generation
        or envelope.source_revision != source.source_revision
        or envelope.source_authority_epoch != source.source_authority_epoch
        or envelope.episode_id != source.episode_id
        or envelope.source_frame_index != source.source_frame_index
        or envelope.source_frame_id != source.source_frame_id
        or envelope.source_simulator_step_count != source.source_simulator_step_count
    ):
        raise ValueError(
            "live Oracle inspection envelope does not join its source epoch."
        )
    inspection = envelope.inspection
    if inspection.submission_scope != source.source_submission_scope:
        raise ValueError("live Oracle inspection scope does not join its source.")
    if type(inspection) is LiveEditableDraftInspectionV1:
        if (
            inspection.submission_scope != "joint_turn"
            or inspection.draft.actor_public_agent_id != axis.owner_public_agent_id
        ):
            raise ValueError("editable Oracle draft does not join its actor/scope.")
    elif type(inspection) is not LiveScriptedPlaybackInspectionV1:
        raise ValueError("unknown live Oracle inspection variant.")


def _validate_live_no_shared_inspection_envelope(
    source: LiveNoSharedObsPresentationSourceIdentityV1,
    endpoint: NoSharedObsAuthorizedCurrentEndpointV1,
    envelope: LiveNoSharedObsInspectionEnvelopeV1,
) -> None:
    _require_exact_type(
        envelope,
        LiveNoSharedObsInspectionEnvelopeV1,
        name="live_inspection",
    )
    parts = endpoint.parts
    if (
        envelope.source_session_id != source.source_session_id
        or envelope.source_run_generation != source.source_run_generation
        or envelope.source_revision != source.source_revision
        or envelope.source_authority_epoch != source.source_authority_epoch
        or envelope.episode_id != source.episode_id
        or envelope.source_frame_index != source.source_frame_index
        or envelope.source_recipient_public_agent_id
        != source.source_recipient_public_agent_id
        or envelope.source_recipient_public_agent_id != parts.recipient_public_agent_id
        or envelope.source_recipient_frame_id != source.source_recipient_frame_id
        or envelope.source_simulator_step_count != source.source_simulator_step_count
    ):
        raise ValueError(
            "live Agent inspection envelope does not join its source epoch."
        )
    inspection = envelope.inspection
    if inspection.submission_scope != source.source_submission_scope:
        raise ValueError("live Agent inspection scope does not join its source.")
    if type(inspection) is LiveEditableDraftInspectionV1:
        if (
            inspection.submission_scope != "controlled_actor"
            or inspection.draft.actor_public_agent_id != parts.recipient_public_agent_id
        ):
            raise ValueError("editable Agent draft does not join its owner/scope.")
    elif type(inspection) is not LiveScriptedPlaybackInspectionV1:
        raise ValueError("unknown live Agent inspection variant.")


def _validate_live_oracle_technical(
    frame: LiveOracleAuthorizedPresentationFrameV1,
) -> None:
    technical = frame.technical_frame
    incoming_id = (
        None
        if frame.latest_transition is None
        else frame.latest_transition.incoming_transition_id
    )
    if (
        technical.episode_id != frame.source.episode_id
        or technical.evaluation_frame_index != frame.source.source_frame_index
        or technical.simulator_step_count != frame.source.source_simulator_step_count
        or technical.incoming_transition_id != incoming_id
    ):
        raise ValueError("live Oracle Technical Frame does not join its source.")


def _validate_live_agent_technical(
    frame: LiveNoSharedObsAuthorizedPresentationFrameV1,
) -> None:
    technical = frame.technical_frame
    incoming_id = (
        None
        if frame.latest_transition is None
        else frame.latest_transition.incoming_transition_id
    )
    if (
        technical.episode_id != frame.source.episode_id
        or technical.recipient_frame_index != frame.source.source_frame_index
        or technical.simulator_step_count != frame.source.source_simulator_step_count
        or technical.incoming_recipient_transition_id != incoming_id
    ):
        raise ValueError("live Agent Technical Frame does not join its source.")


def _validate_replay_oracle_technical(
    frame: ReplayOracleAuthorizedPresentationFrameV1,
) -> None:
    technical = frame.technical_frame
    incoming_id = (
        None
        if frame.latest_transition is None
        else frame.latest_transition.incoming_transition_id
    )
    if (
        technical.artifact_digest_prefix
        != frame.source.source_artifact_digest_sha256[
            :PRESENTATION_TECHNICAL_DIGEST_PREFIX_LENGTH_V1
        ]
        or technical.frame_index != frame.source.source_frame_index
        or technical.simulator_step_count != frame.source.source_simulator_step_count
        or technical.incoming_transition_id != incoming_id
        or technical.recorded_ordinary_movement_distance_scale
        != frame.source.source_recorded_ordinary_movement_distance_scale
    ):
        raise ValueError("replay Oracle Technical Frame does not join its source.")


def _validate_replay_oracle_upcoming(
    *,
    source: ReplayOraclePresentationSourceIdentityV1,
    endpoint: OracleAuthorizedCurrentEndpointV1,
    upcoming_transition: OracleUpcomingTransitionV1 | None,
    inspection: ReplayInspectionPresentationV1 | None,
) -> None:
    final = source.source_frame_index == source.source_final_frame_index
    if final:
        if upcoming_transition is not None:
            raise ValueError("the final replay frame cannot carry Upcoming Transition.")
        return
    _require_exact_type(
        upcoming_transition,
        OracleUpcomingTransitionV1,
        name="upcoming_transition",
    )
    transition = cast(OracleUpcomingTransitionV1, upcoming_transition)
    if (
        transition.episode_id != source.episode_id
        or transition.outgoing_transition_index != source.source_frame_index
        or transition.outgoing_start_frame_id != endpoint.frame_id
        or transition.outgoing_start_simulator_step_count
        != endpoint.simulator_step_count
    ):
        raise ValueError("Oracle Upcoming Transition does not leave current s_n.")
    directory_by_id = {
        row.public_agent_id: row for row in endpoint.identity_directory.identities
    }
    scene_by_id = {row.public_agent_id: row for row in endpoint.scene.agents}
    expected_active = tuple(
        row.public_agent_id
        for row in endpoint.identity_directory.identities
        if row.configured_active
    )
    if (
        tuple(row.actor_public_agent_id for row in transition.action_rows)
        != expected_active
    ):
        raise ValueError(
            "Oracle Upcoming Transition rows must equal active directory order."
        )
    directory_ids = set(directory_by_id)
    for row in transition.action_rows:
        actor = directory_by_id[row.actor_public_agent_id]
        scene_actor = scene_by_id.get(row.actor_public_agent_id)
        if (
            scene_actor is None
            or row.actor_presentation_key != scene_actor.presentation_key
        ):
            raise ValueError(
                "Oracle Upcoming Transition actor key does not join the directory."
            )
        target_axis = cast(
            tuple[str, ...],
            row.target_action_recipient_public_agent_id_by_id[1:],
        )
        if set(target_axis) != directory_ids or target_axis != (
            _oracle_target_public_axis(
                endpoint.identity_directory,
                owner_team_id=actor.team_id,
            )
        ):
            raise ValueError(
                "Oracle Upcoming Transition target axis changed action semantics."
            )
    if inspection is None:
        return
    selected = next(
        (
            row
            for row in transition.action_rows
            if row.actor_public_agent_id == inspection.actor_public_agent_id
        ),
        None,
    )
    if selected is None or (
        selected.actor_presentation_key != inspection.actor_presentation_key
        or selected.submitted_action != inspection.submitted_action
        or selected.accepted_action != inspection.accepted_action
    ):
        raise ValueError(
            "Oracle replay inspection must equal its Upcoming Transition row."
        )


def _validate_replay_agent_technical(
    frame: ReplayNoSharedObsAuthorizedPresentationFrameV1
    | ReplaySharedObsAuthorizedPresentationFrameV1,
    *,
    shared: bool,
) -> None:
    technical = frame.technical_frame
    expected_type = (
        ReplaySharedObsTechnicalFrameV1 if shared else ReplayNoSharedObsTechnicalFrameV1
    )
    _require_exact_type(technical, expected_type, name="technical_frame")
    incoming_id = (
        None
        if frame.latest_transition is None
        else frame.latest_transition.incoming_transition_id
    )
    if (
        technical.frame_index != frame.source.source_frame_index
        or technical.simulator_step_count != frame.source.source_simulator_step_count
        or technical.incoming_recipient_transition_id != incoming_id
    ):
        raise ValueError("replay Agent Technical Frame does not join its source.")


def _validate_replay_agent_upcoming(
    frame: ReplayNoSharedObsAuthorizedPresentationFrameV1
    | ReplaySharedObsAuthorizedPresentationFrameV1,
    *,
    shared: bool,
) -> None:
    final = frame.source.source_frame_index == frame.source.source_final_frame_index
    transition = frame.upcoming_transition
    if final:
        if transition is not None:
            raise ValueError(
                "the final replay Agent frame cannot carry Upcoming Transition."
            )
        return
    expected_type = (
        SharedObsUpcomingTransitionV1 if shared else NoSharedObsUpcomingTransitionV1
    )
    _require_exact_type(transition, expected_type, name="upcoming_transition")
    outgoing = cast(
        NoSharedObsUpcomingTransitionV1 | SharedObsUpcomingTransitionV1,
        transition,
    )
    source = frame.source
    endpoint = frame.current_endpoint
    parts = endpoint.parts
    if (
        outgoing.episode_id != source.episode_id
        or outgoing.outgoing_transition_index != source.source_frame_index
        or outgoing.outgoing_start_frame_id != source.source_recipient_frame_id
        or outgoing.outgoing_start_simulator_step_count
        != source.source_simulator_step_count
        or outgoing.recipient_public_agent_id != source.source_recipient_public_agent_id
        or outgoing.recipient_presentation_key != parts.recipient_presentation_key
        or outgoing.action_rows[0].target_action_recipient_public_agent_id_by_id
        != endpoint.action_axis.target_public_agent_id_by_action
    ):
        raise ValueError("Agent Upcoming Transition does not leave current s_n.")
    inspection = frame.replay_inspection
    if inspection is None:
        return
    row = outgoing.action_rows[0]
    if (
        row.actor_presentation_key != inspection.actor_presentation_key
        or row.actor_public_agent_id != inspection.actor_public_agent_id
        or row.submitted_action != inspection.submitted_action
        or row.accepted_action != inspection.accepted_action
    ):
        raise ValueError(
            "Agent replay inspection must equal its Upcoming Transition row."
        )


def _validate_replay_agent_inspection_state(
    frame: ReplayNoSharedObsAuthorizedPresentationFrameV1
    | ReplaySharedObsAuthorizedPresentationFrameV1,
) -> None:
    final = frame.source.source_frame_index == frame.source.source_final_frame_index
    if final:
        if frame.replay_inspection is not None:
            raise ValueError(
                "the final replay Agent frame cannot carry outgoing intent."
            )
        return
    if frame.replay_inspection is None:
        raise ValueError("a non-final replay Agent frame requires owner inspection.")
    _validate_agent_inspection(
        source=frame.source,
        endpoint=frame.current_endpoint,
        inspection=frame.replay_inspection,
        replay=True,
    )


type _AuthorizedPresentationFrameConcreteV1 = (
    LiveOracleAuthorizedPresentationFrameV1
    | LiveNoSharedObsAuthorizedPresentationFrameV1
    | ReplayOracleAuthorizedPresentationFrameV1
    | ReplayNoSharedObsAuthorizedPresentationFrameV1
    | ReplaySharedObsAuthorizedPresentationFrameV1
)
type AuthorizedPresentationFrameV1 = Annotated[
    _AuthorizedPresentationFrameConcreteV1,
    Field(discriminator="presentation_kind"),
]

_AUTHORIZED_PRESENTATION_FRAME_ADAPTER: TypeAdapter[AuthorizedPresentationFrameV1] = (
    TypeAdapter(AuthorizedPresentationFrameV1)
)
_FINAL_PRESENTATION_TYPES = (
    LiveOracleAuthorizedPresentationFrameV1,
    LiveNoSharedObsAuthorizedPresentationFrameV1,
    ReplayOracleAuthorizedPresentationFrameV1,
    ReplayNoSharedObsAuthorizedPresentationFrameV1,
    ReplaySharedObsAuthorizedPresentationFrameV1,
)


@dataclass(frozen=True, slots=True)
class PresentationResourceResultV1:
    """Transport-neutral result that fully revalidates every success payload."""

    outcome: Literal["response", "audience_unavailable"]
    payload: _AuthorizedPresentationFrameConcreteV1 | PresentationApiErrorV1

    def __post_init__(self) -> None:
        if type(self) is not PresentationResourceResultV1:
            raise TypeError("presentation resource result requires its exact root")
        if type(self.outcome) is not str:
            raise TypeError("presentation resource outcome requires an exact string")
        if self.outcome == "response":
            if type(self.payload) not in _FINAL_PRESENTATION_TYPES:
                raise TypeError(
                    "a presentation response requires one exact authorized frame root"
                )
            _validate_recursive_exact_runtime_types(self.payload)
            try:
                revalidated = _AUTHORIZED_PRESENTATION_FRAME_ADAPTER.validate_json(
                    self.payload.model_dump_json()
                )
            except ValidationError as error:
                raise ValueError(
                    "presentation response payload failed full strict revalidation"
                ) from error
            if type(revalidated) is not type(self.payload):
                raise TypeError("presentation response discriminator changed its root")
            _validate_exact_runtime_tree_matches(
                self.payload,
                revalidated,
                path="presentation_response",
            )
            object.__setattr__(self, "payload", revalidated)
            return
        if self.outcome == "audience_unavailable":
            if type(self.payload) is not PresentationApiErrorV1:
                raise TypeError(
                    "an unavailable presentation requires the exact API error root"
                )
            _validate_recursive_exact_runtime_types(self.payload)
            try:
                revalidated_error = PresentationApiErrorV1.model_validate_json(
                    self.payload.model_dump_json()
                )
            except ValidationError as error:
                raise ValueError(
                    "presentation error payload failed full strict revalidation"
                ) from error
            _validate_exact_runtime_tree_matches(
                self.payload,
                revalidated_error,
                path="presentation_error",
            )
            object.__setattr__(self, "payload", revalidated_error)
            return
        raise ValueError("unknown presentation resource outcome")


__all__ = [
    "PRESENTATION_AUDIENCE_UNAVAILABLE_MESSAGE",
    "PRESENTATION_PROTOCOL_SCHEMA_VERSION",
    "PRESENTATION_TECHNICAL_DIGEST_PREFIX_LENGTH_V1",
    "AgentPovActionAxisV1",
    "AgentPovDecisionMaskV1",
    "AuthorizedPresentationFrameV1",
    "LatestTransitionActionRowV1",
    "LiveEditableDraftInspectionV1",
    "LiveInputInspectionV1",
    "LiveNoSharedObsAuthorizedPresentationFrameV1",
    "LiveNoSharedObsInspectionEnvelopeV1",
    "LiveNoSharedObsPresentationSourceIdentityV1",
    "LiveNoSharedObsTechnicalFrameV1",
    "LiveOracleAuthorizedPresentationFrameV1",
    "LiveOracleInspectionEnvelopeV1",
    "LiveOraclePresentationSourceIdentityV1",
    "LiveOracleTechnicalFrameV1",
    "LiveScriptedPlaybackInspectionV1",
    "MovementActionDisplayRowV1",
    "NoSharedObsAuthorizedCurrentEndpointV1",
    "NoSharedObsLatestTransitionV1",
    "NoSharedObsPresentationAuthorityV1",
    "NoSharedObsPresentationEndpointPartsV1",
    "NoSharedObsUpcomingTransitionV1",
    "OracleActionAxisV1",
    "OracleAuthorizedCurrentEndpointV1",
    "OracleLatestTransitionV1",
    "OraclePresentationAuthorityV1",
    "OraclePublicIdentityDirectoryRowV1",
    "OraclePublicIdentityDirectoryV1",
    "OracleUpcomingTransitionV1",
    "PresentationApiErrorV1",
    "PresentationResourceResultV1",
    "ReplayNoSharedObsAuthorizedPresentationFrameV1",
    "ReplayNoSharedObsPresentationSourceIdentityV1",
    "ReplayNoSharedObsTechnicalFrameV1",
    "ReplayOracleAuthorizedPresentationFrameV1",
    "ReplayOraclePresentationSourceIdentityV1",
    "ReplayOracleTechnicalFrameV1",
    "ReplayResearcherRosterAgentV1",
    "ReplayResearcherSpaceV1",
    "ReplaySharedObsAuthorizedPresentationFrameV1",
    "ReplaySharedObsPresentationSourceIdentityV1",
    "ReplaySharedObsTechnicalFrameV1",
    "SharedObsAuthorizedCurrentEndpointV1",
    "SharedObsLatestTransitionV1",
    "SharedObsPresentationAuthorityV1",
    "SharedObsPresentationEndpointPartsV1",
    "SharedObsUpcomingTransitionV1",
    "TargetActionDisplayRowV1",
    "TargetAgentActionDisplayRowV1",
    "TargetNoneActionDisplayRowV1",
    "UltimateChoiceDisplayRowV1",
    "build_no_shared_obs_authorized_current_endpoint_v1",
    "build_oracle_authorized_current_endpoint_v1",
    "build_shared_obs_authorized_current_endpoint_v1",
    "canonical_authorized_endpoint_digest_sha256",
]
