"""Strict authority-safe replay inspection and live draft presentation.

Replay inspection describes the recorded action chosen from the displayed
decision epoch ``s_n/m_n`` and accepted in outgoing transition ``T_n``.  Live
draft inspection describes editable intent before submission.  The two roots
are deliberately disjoint, and neither builder accepts browser, service,
simulator, JAX, NumPy, or successor-state objects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import isfinite
from typing import Annotated, ClassVar, Literal, cast

from pydantic import ConfigDict, Field

from marl_battlegrounds.evaluation.models import (
    ActionAcceptanceFactsV1,
    ActionMaskV1,
    EvaluationEpisodeContextV1,
    EvaluationFrameV1,
    EvaluationTransitionV1,
    JointActionV1,
    TransitionFactsV1,
)
from marl_battlegrounds.evaluation.pov import (
    ActorPovActionMaskV1,
    ActorPovAxisMappingV1,
    ActorPovCurrentSliceV1,
    ActorPovTransitionV1,
    validate_actor_pov_replay_content_v1,
)
from marl_battlegrounds.evaluation.wire_shapes import (
    MAX_AGENT_SLOTS_V1,
    NUM_MOVE_ACTIONS_V1,
    NUM_TARGET_ACTIONS_V1,
    NUM_ULTIMATE_ACTIONS_V1,
)
from marl_battlegrounds.rendering.authorized_pov_scene import (
    NoSharedObsAuthorizedScenePartsV1,
    SharedObsAuthorizedScenePartsV1,
)
from marl_battlegrounds.rendering.authorized_presentation import (
    AcceptedActionTupleV1,
    AuthorizedAgentV1,
    AuthorizedBattlefieldSceneV1,
    Point2D,
    SubmittedActionTupleV1,
)
from marl_battlegrounds.rendering.evaluation_adapter import (
    SHARED_OBS_SOURCE_MATERIAL_PROJECTION_SCHEMA_VERSION,
    SharedObsBaseSensorFrameV1,
    SharedObsBaseSensorSceneV1,
    SharedObsSourceMaterialProjectionV1,
)
from marl_battlegrounds.rendering.evaluation_wire_features import (
    AGENT_FEATURE_ACTIVE_V1,
    AGENT_FEATURE_CLASS_ID_V1,
    AGENT_FEATURE_TEAM_ID_V1,
    AGENT_FEATURE_X_V1,
    AGENT_FEATURE_Y_V1,
)
from marl_battlegrounds.rendering.pov_scene import (
    ActorPovProjectionIndexV1,
    ActorPovSelfSceneV1,
)

AUTHORIZED_INSPECTION_SCHEMA_VERSION = 1

_ASCII_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+\-]*$")

_STRICT_WIRE_CONFIG = ConfigDict(
    allow_inf_nan=False,
    extra="forbid",
    strict=True,
)

type CombatLaneV1 = Literal["none", "basic", "ultimate"]
type DraftArmedLaneV1 = Literal["none", "basic", "ultimate"]


def _require_text(value: str, *, name: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a non-empty Python string.")


def _require_ascii_identifier(value: str, *, name: str) -> None:
    if type(value) is not str or _ASCII_IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be an exact ASCII identifier.")


def _require_python_int(
    value: int,
    *,
    name: str,
    minimum: int = 0,
    maximum_exclusive: int | None = None,
) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be a Python int >= {minimum}.")
    if maximum_exclusive is not None and value >= maximum_exclusive:
        raise ValueError(f"{name} must be less than {maximum_exclusive}.")


def _require_point(value: Point2D, *, name: str) -> None:
    if type(value) is not tuple or len(value) != 2:
        raise ValueError(f"{name} must be a two-coordinate Python tuple.")
    if any(type(item) is not float or not isfinite(item) for item in value):
        raise ValueError(f"{name} coordinates must be finite Python floats.")


def _require_text_tuple(
    values: tuple[str, ...],
    *,
    name: str,
    length: int,
) -> None:
    if type(values) is not tuple or len(values) != length:
        raise ValueError(f"{name} must be a {length}-row Python tuple.")
    for value in values:
        _require_text(value, name=f"{name} item")
    if len(set(values)) != length:
        raise ValueError(f"{name} entries must be unique.")


def _require_bool_tuple(
    values: tuple[bool, ...],
    *,
    name: str,
    length: int,
) -> None:
    if type(values) is not tuple or len(values) != length:
        raise ValueError(f"{name} must be a {length}-row Python tuple.")
    if any(type(value) is not bool for value in values):
        raise ValueError(f"{name} must contain exact Python bool values.")


@dataclass(frozen=True, slots=True, kw_only=True)
class AuthorizedNoTargetActionV1:
    """The recorded target-action-zero axis row."""

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_CONFIG

    target_kind: Literal["no_target"]
    target_action: int
    display_name: str

    def __post_init__(self) -> None:
        if self.target_kind != "no_target" or self.target_action != 0:
            raise ValueError("the no-target variant must identify target action zero.")
        if type(self.target_action) is not int:
            raise ValueError("target_action must be an exact Python int.")
        _require_text(self.display_name, name="display_name")


@dataclass(frozen=True, slots=True, kw_only=True)
class AuthorizedVisibleTargetActionV1:
    """One positive target-axis row joined to the authorized current scene."""

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_CONFIG

    target_kind: Literal["visible_authorized_agent"]
    target_action: int
    display_name: str
    target_presentation_key: str
    target_public_agent_id: str
    target_anchor: Point2D

    def __post_init__(self) -> None:
        if self.target_kind != "visible_authorized_agent":
            raise ValueError("unknown visible-target discriminator.")
        _require_python_int(
            self.target_action,
            name="target_action",
            minimum=1,
            maximum_exclusive=NUM_TARGET_ACTIONS_V1,
        )
        for name in (
            "display_name",
            "target_presentation_key",
            "target_public_agent_id",
        ):
            _require_text(cast(str, getattr(self, name)), name=name)
        _require_point(self.target_anchor, name="target_anchor")


@dataclass(frozen=True, slots=True, kw_only=True)
class AuthorizedAxisOnlyTargetActionV1:
    """One authorized positive target axis whose body anchor is not disclosed."""

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_CONFIG

    target_kind: Literal["axis_only_authorized_agent"]
    target_action: int
    display_name: str
    target_public_agent_id: str

    def __post_init__(self) -> None:
        if self.target_kind != "axis_only_authorized_agent":
            raise ValueError("unknown axis-only-target discriminator.")
        _require_python_int(
            self.target_action,
            name="target_action",
            minimum=1,
            maximum_exclusive=NUM_TARGET_ACTIONS_V1,
        )
        _require_text(self.display_name, name="display_name")
        _require_text(
            self.target_public_agent_id,
            name="target_public_agent_id",
        )


type AuthorizedTargetActionV1 = Annotated[
    AuthorizedNoTargetActionV1
    | AuthorizedVisibleTargetActionV1
    | AuthorizedAxisOnlyTargetActionV1,
    Field(discriminator="target_kind"),
]


@dataclass(frozen=True, slots=True, kw_only=True)
class AuthorizedDecisionMaskV1:
    """One complete owner-bound 9/11/2/11x2 decision surface."""

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_CONFIG

    schema_version: Literal[1]
    owner_presentation_key: str
    owner_public_agent_id: str
    movement_action_display_names: tuple[str, ...]
    movement_action_mask: tuple[bool, ...]
    target_actions: tuple[AuthorizedTargetActionV1, ...]
    target_action_mask: tuple[bool, ...]
    use_ultimate_action_display_names: tuple[str, ...]
    use_ultimate_action_mask: tuple[bool, ...]
    target_use_ultimate_joint_mask: tuple[tuple[bool, ...], ...]

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != AUTHORIZED_INSPECTION_SCHEMA_VERSION
        ):
            raise ValueError("unknown authorized decision-mask schema version.")
        _require_text(self.owner_presentation_key, name="owner_presentation_key")
        _require_text(self.owner_public_agent_id, name="owner_public_agent_id")
        _require_text_tuple(
            self.movement_action_display_names,
            name="movement_action_display_names",
            length=NUM_MOVE_ACTIONS_V1,
        )
        _require_bool_tuple(
            self.movement_action_mask,
            name="movement_action_mask",
            length=NUM_MOVE_ACTIONS_V1,
        )
        if (
            type(self.target_actions) is not tuple
            or len(self.target_actions) != NUM_TARGET_ACTIONS_V1
        ):
            raise ValueError("target_actions must retain the exact eleven-row axis.")
        allowed_target_types = (
            AuthorizedNoTargetActionV1,
            AuthorizedVisibleTargetActionV1,
            AuthorizedAxisOnlyTargetActionV1,
        )
        if any(type(row) not in allowed_target_types for row in self.target_actions):
            raise ValueError("target_actions must contain exact target variants.")
        target_order = tuple(row.target_action for row in self.target_actions)
        if target_order != tuple(range(NUM_TARGET_ACTIONS_V1)):
            raise ValueError("target_actions must preserve exact target-action order.")
        if type(self.target_actions[0]) is not AuthorizedNoTargetActionV1:
            raise ValueError("target action zero must use the no-target variant.")
        positive_public_id_rows: list[str] = []
        for row in self.target_actions[1:]:
            if isinstance(
                row,
                (AuthorizedVisibleTargetActionV1, AuthorizedAxisOnlyTargetActionV1),
            ):
                positive_public_id_rows.append(row.target_public_agent_id)
            else:  # pragma: no cover - exact-type check above owns this branch.
                raise AssertionError("positive target variant disappeared")
        positive_public_ids = tuple(positive_public_id_rows)
        if len(positive_public_ids) != NUM_TARGET_ACTIONS_V1 - 1 or len(
            set(positive_public_ids)
        ) != len(positive_public_ids):
            raise ValueError(
                "positive target actions require one-to-one public identities."
            )
        visible_keys = tuple(
            row.target_presentation_key
            for row in self.target_actions
            if isinstance(row, AuthorizedVisibleTargetActionV1)
        )
        if len(visible_keys) != len(set(visible_keys)):
            raise ValueError(
                "visible target actions require one-to-one presentation keys."
            )
        display_names = tuple(row.display_name for row in self.target_actions)
        if len(set(display_names)) != NUM_TARGET_ACTIONS_V1:
            raise ValueError("target action display names must be unique.")
        _require_bool_tuple(
            self.target_action_mask,
            name="target_action_mask",
            length=NUM_TARGET_ACTIONS_V1,
        )
        _require_text_tuple(
            self.use_ultimate_action_display_names,
            name="use_ultimate_action_display_names",
            length=NUM_ULTIMATE_ACTIONS_V1,
        )
        _require_bool_tuple(
            self.use_ultimate_action_mask,
            name="use_ultimate_action_mask",
            length=NUM_ULTIMATE_ACTIONS_V1,
        )
        if (
            type(self.target_use_ultimate_joint_mask) is not tuple
            or len(self.target_use_ultimate_joint_mask) != NUM_TARGET_ACTIONS_V1
        ):
            raise ValueError("joint combat mask must retain eleven target rows.")
        for row in self.target_use_ultimate_joint_mask:
            _require_bool_tuple(
                row,
                name="joint combat mask row",
                length=NUM_ULTIMATE_ACTIONS_V1,
            )
        target_marginal = tuple(any(row) for row in self.target_use_ultimate_joint_mask)
        ultimate_marginal = tuple(
            any(
                self.target_use_ultimate_joint_mask[target][ultimate]
                for target in range(NUM_TARGET_ACTIONS_V1)
            )
            for ultimate in range(NUM_ULTIMATE_ACTIONS_V1)
        )
        if self.target_action_mask != target_marginal:
            raise ValueError("target mask must equal the exact joint-mask marginal.")
        if self.use_ultimate_action_mask != ultimate_marginal:
            raise ValueError("Ultimate mask must equal the exact joint-mask marginal.")


@dataclass(frozen=True, slots=True, kw_only=True)
class OracleReplayTransitionReferenceV1:
    """Canonical Oracle identity for one recorded outgoing transition."""

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_CONFIG

    reference_kind: Literal["oracle_recorded_transition"]
    transition_id: str
    start_frame_id: str
    successor_frame_id: str

    def __post_init__(self) -> None:
        if self.reference_kind != "oracle_recorded_transition":
            raise ValueError("unknown Oracle transition-reference discriminator.")
        for name in ("transition_id", "start_frame_id", "successor_frame_id"):
            _require_text(cast(str, getattr(self, name)), name=name)


@dataclass(frozen=True, slots=True, kw_only=True)
class NoSharedObsReplayTransitionReferenceV1:
    """Recipient-local NoSharedObs identity for an outgoing transition."""

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_CONFIG

    reference_kind: Literal["no_shared_obs_actor_pov_transition"]
    recipient_public_agent_id: str
    transition_id: str
    start_frame_id: str
    successor_frame_id: str

    def __post_init__(self) -> None:
        if self.reference_kind != "no_shared_obs_actor_pov_transition":
            raise ValueError("unknown NoSharedObs transition-reference discriminator.")
        for name in (
            "recipient_public_agent_id",
            "transition_id",
            "start_frame_id",
            "successor_frame_id",
        ):
            _require_text(cast(str, getattr(self, name)), name=name)


@dataclass(frozen=True, slots=True, kw_only=True)
class SharedObsReplayTransitionReferenceV1:
    """Recipient-local visual-union identity for an outgoing transition."""

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_CONFIG

    reference_kind: Literal["shared_obs_visual_union_transition"]
    recipient_public_agent_id: str
    transition_id: str
    start_frame_id: str
    successor_frame_id: str

    def __post_init__(self) -> None:
        if self.reference_kind != "shared_obs_visual_union_transition":
            raise ValueError("unknown SharedObs transition-reference discriminator.")
        for name in (
            "recipient_public_agent_id",
            "transition_id",
            "start_frame_id",
            "successor_frame_id",
        ):
            _require_text(cast(str, getattr(self, name)), name=name)


type ReplayTransitionReferenceV1 = Annotated[
    OracleReplayTransitionReferenceV1
    | NoSharedObsReplayTransitionReferenceV1
    | SharedObsReplayTransitionReferenceV1,
    Field(discriminator="reference_kind"),
]


def _expected_combat_lane(action: AcceptedActionTupleV1) -> CombatLaneV1:
    if action.use_ultimate_action == 1:
        return "ultimate"
    if action.target_action == 0:
        return "none"
    return "basic"


def _validate_reference_epoch(
    reference: ReplayTransitionReferenceV1,
    *,
    episode_id: str,
    transition_index: int,
    actor_public_agent_id: str,
) -> None:
    if type(reference) is OracleReplayTransitionReferenceV1:
        prefix = episode_id
    elif type(reference) is NoSharedObsReplayTransitionReferenceV1:
        if reference.recipient_public_agent_id != actor_public_agent_id:
            raise ValueError("NoSharedObs reference must belong to the recipient.")
        prefix = f"{episode_id}:actor-pov:{actor_public_agent_id}"
    elif type(reference) is SharedObsReplayTransitionReferenceV1:
        if reference.recipient_public_agent_id != actor_public_agent_id:
            raise ValueError("SharedObs reference must belong to the recipient.")
        prefix = f"{episode_id}:shared-obs-visual-union:{actor_public_agent_id}"
    else:  # pragma: no cover - exact-type validation owns this branch.
        raise AssertionError("transition reference variant disappeared")
    if (
        reference.transition_id != f"{prefix}:transition:{transition_index}"
        or reference.start_frame_id != f"{prefix}:frame:{transition_index}"
        or reference.successor_frame_id != f"{prefix}:frame:{transition_index + 1}"
    ):
        raise ValueError(
            "transition reference must retain its exact authority namespace and "
            "adjacent frame IDs."
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayInspectionPresentationV1:
    """Recorded accepted outgoing route and exact current decision surface."""

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_CONFIG

    schema_version: Literal[1]
    inspection_kind: Literal["replay_recorded_outgoing_action"]
    route_display_basis: Literal["accepted_action"]
    episode_id: str
    outgoing_transition_index: int
    current_simulator_step_count: int
    transition_reference: ReplayTransitionReferenceV1
    actor_presentation_key: str
    actor_public_agent_id: str
    actor_anchor: Point2D
    decision_mask: AuthorizedDecisionMaskV1
    submitted_action: SubmittedActionTupleV1
    accepted_action: AcceptedActionTupleV1
    combat_lane: CombatLaneV1
    accepted_target: AuthorizedTargetActionV1

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != AUTHORIZED_INSPECTION_SCHEMA_VERSION
        ):
            raise ValueError("unknown replay-inspection schema version.")
        if self.inspection_kind != "replay_recorded_outgoing_action":
            raise ValueError("unknown replay-inspection discriminator.")
        if self.route_display_basis != "accepted_action":
            raise ValueError("replay routes must use the accepted action.")
        _require_ascii_identifier(self.episode_id, name="episode_id")
        _require_python_int(
            self.outgoing_transition_index,
            name="outgoing_transition_index",
        )
        _require_python_int(
            self.current_simulator_step_count,
            name="current_simulator_step_count",
        )
        if type(self.transition_reference) not in (
            OracleReplayTransitionReferenceV1,
            NoSharedObsReplayTransitionReferenceV1,
            SharedObsReplayTransitionReferenceV1,
        ):
            raise ValueError("transition_reference must use one exact variant.")
        _require_text(self.actor_presentation_key, name="actor_presentation_key")
        _require_text(self.actor_public_agent_id, name="actor_public_agent_id")
        _require_point(self.actor_anchor, name="actor_anchor")
        if type(self.decision_mask) is not AuthorizedDecisionMaskV1:
            raise ValueError("decision_mask must use its exact strict root.")
        if (
            self.decision_mask.owner_presentation_key != self.actor_presentation_key
            or self.decision_mask.owner_public_agent_id != self.actor_public_agent_id
        ):
            raise ValueError("decision mask owner must equal the inspection actor.")
        if type(self.submitted_action) is not SubmittedActionTupleV1:
            raise ValueError("submitted_action must use its exact tuple root.")
        if type(self.accepted_action) is not AcceptedActionTupleV1:
            raise ValueError("accepted_action must use its exact tuple root.")
        if self.combat_lane != _expected_combat_lane(self.accepted_action):
            raise ValueError("combat lane must equal the canonical accepted action.")
        if type(self.accepted_target) not in (
            AuthorizedNoTargetActionV1,
            AuthorizedVisibleTargetActionV1,
            AuthorizedAxisOnlyTargetActionV1,
        ):
            raise ValueError("accepted_target must use one exact target variant.")
        if (
            self.accepted_target.target_action != self.accepted_action.target_action
            or self.accepted_target
            != self.decision_mask.target_actions[self.accepted_action.target_action]
        ):
            raise ValueError("accepted target must equal its exact decision-axis row.")
        accepted = self.accepted_action
        if (
            not self.decision_mask.movement_action_mask[accepted.move_action]
            or not (
                self.decision_mask.target_use_ultimate_joint_mask[
                    accepted.target_action
                ][accepted.use_ultimate_action]
            )
        ):
            raise ValueError("canonical accepted action must be legal under m_n.")
        _validate_reference_epoch(
            self.transition_reference,
            episode_id=self.episode_id,
            transition_index=self.outgoing_transition_index,
            actor_public_agent_id=self.actor_public_agent_id,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class LiveDraftActionTupleV1:
    """Editable category-bounded intent before submission."""

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_CONFIG

    move_action: int
    target_action: int
    armed_lane: DraftArmedLaneV1

    def __post_init__(self) -> None:
        _require_python_int(
            self.move_action,
            name="move_action",
            maximum_exclusive=NUM_MOVE_ACTIONS_V1,
        )
        _require_python_int(
            self.target_action,
            name="target_action",
            maximum_exclusive=NUM_TARGET_ACTIONS_V1,
        )
        if self.armed_lane not in ("none", "basic", "ultimate"):
            raise ValueError("unknown live draft armed lane.")


@dataclass(frozen=True, slots=True, kw_only=True)
class LiveDraftLegalityV1:
    """Marginal and exact-pair legality for one live draft."""

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_CONFIG

    move_action_is_legal: bool
    target_action_is_legal: bool
    armed_lane_is_legal: bool | None
    combat_pair_is_legal: bool | None

    def __post_init__(self) -> None:
        if (
            type(self.move_action_is_legal) is not bool
            or type(self.target_action_is_legal) is not bool
        ):
            raise ValueError("draft marginal legality values must be Python bools.")
        for name in ("armed_lane_is_legal", "combat_pair_is_legal"):
            value = getattr(self, name)
            if value is not None and type(value) is not bool:
                raise ValueError(f"{name} must be a Python bool or None.")
        if (self.armed_lane_is_legal is None) != (self.combat_pair_is_legal is None):
            raise ValueError("draft lane and pair legality must be absent together.")


@dataclass(frozen=True, slots=True, kw_only=True)
class LiveDraftInspectionPresentationV1:
    """Editable live intent with no recorded-transition or acceptance identity."""

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_CONFIG

    schema_version: Literal[1]
    inspection_kind: Literal["live_draft_action"]
    route_display_basis: Literal["draft_action"]
    current_simulator_step_count: int
    actor_presentation_key: str
    actor_public_agent_id: str
    actor_anchor: Point2D
    decision_mask: AuthorizedDecisionMaskV1
    draft_action: LiveDraftActionTupleV1
    draft_target: AuthorizedTargetActionV1
    draft_legality: LiveDraftLegalityV1

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != AUTHORIZED_INSPECTION_SCHEMA_VERSION
        ):
            raise ValueError("unknown live-draft inspection schema version.")
        if self.inspection_kind != "live_draft_action":
            raise ValueError("unknown live-draft inspection discriminator.")
        if self.route_display_basis != "draft_action":
            raise ValueError("live routes must use the current draft action.")
        _require_python_int(
            self.current_simulator_step_count,
            name="current_simulator_step_count",
        )
        _require_text(self.actor_presentation_key, name="actor_presentation_key")
        _require_text(self.actor_public_agent_id, name="actor_public_agent_id")
        _require_point(self.actor_anchor, name="actor_anchor")
        if type(self.decision_mask) is not AuthorizedDecisionMaskV1:
            raise ValueError("decision_mask must use its exact strict root.")
        if (
            self.decision_mask.owner_presentation_key != self.actor_presentation_key
            or self.decision_mask.owner_public_agent_id != self.actor_public_agent_id
        ):
            raise ValueError("decision mask owner must equal the live draft actor.")
        if type(self.draft_action) is not LiveDraftActionTupleV1:
            raise ValueError("draft_action must use its exact tuple root.")
        if type(self.draft_target) not in (
            AuthorizedNoTargetActionV1,
            AuthorizedVisibleTargetActionV1,
            AuthorizedAxisOnlyTargetActionV1,
        ):
            raise ValueError("draft_target must use one exact target variant.")
        if (
            self.draft_target.target_action != self.draft_action.target_action
            or self.draft_target
            != self.decision_mask.target_actions[self.draft_action.target_action]
        ):
            raise ValueError("draft target must equal its exact decision-axis row.")
        if type(self.draft_legality) is not LiveDraftLegalityV1:
            raise ValueError("draft_legality must use its exact strict root.")
        action = self.draft_action
        expected_move = self.decision_mask.movement_action_mask[action.move_action]
        expected_target = self.decision_mask.target_action_mask[action.target_action]
        if (
            self.draft_legality.move_action_is_legal != expected_move
            or self.draft_legality.target_action_is_legal != expected_target
        ):
            raise ValueError("draft marginal legality must equal the decision mask.")
        if action.armed_lane == "none":
            if (
                self.draft_legality.armed_lane_is_legal is not None
                or self.draft_legality.combat_pair_is_legal is not None
            ):
                raise ValueError(
                    "an unarmed draft has no combat lane or pair legality."
                )
            return
        lane = 0 if action.armed_lane == "basic" else 1
        if (
            self.draft_legality.armed_lane_is_legal
            != self.decision_mask.use_ultimate_action_mask[lane]
            or self.draft_legality.combat_pair_is_legal
            != self.decision_mask.target_use_ultimate_joint_mask[action.target_action][
                lane
            ]
        ):
            raise ValueError("armed draft legality must equal the exact joint mask.")


def _validated_context(
    context: EvaluationEpisodeContextV1,
) -> EvaluationEpisodeContextV1:
    if type(context) is not EvaluationEpisodeContextV1:
        raise TypeError("context must be the exact EvaluationEpisodeContextV1 root.")
    return EvaluationEpisodeContextV1.model_validate(context.model_dump(mode="python"))


def _validated_frame(frame: EvaluationFrameV1) -> EvaluationFrameV1:
    if type(frame) is not EvaluationFrameV1:
        raise TypeError("current_frame must be the exact EvaluationFrameV1 root.")
    return EvaluationFrameV1.model_validate(frame.model_dump(mode="python"))


def _validated_transition(
    transition: EvaluationTransitionV1,
) -> EvaluationTransitionV1:
    if type(transition) is not EvaluationTransitionV1:
        raise TypeError("outgoing_transition must be EvaluationTransitionV1.")
    return EvaluationTransitionV1.model_validate(transition.model_dump(mode="python"))


def _shared_recipient_action_rows(
    transition: EvaluationTransitionV1,
    *,
    recipient_global_slot: int,
) -> tuple[SubmittedActionTupleV1, AcceptedActionTupleV1]:
    """Validate only the Shared recipient-owned action surface of ``T_n``."""
    if type(transition) is not EvaluationTransitionV1:
        raise TypeError("outgoing_transition must be EvaluationTransitionV1.")
    if (
        type(transition.schema_id) is not str
        or transition.schema_id != "marl_battlegrounds.evaluation.transition"
        or type(transition.schema_version) is not int
        or transition.schema_version != 1
    ):
        raise ValueError("outgoing transition must retain its exact V1 schema.")
    for name in (
        "episode_id",
        "transition_id",
        "start_frame_id",
        "successor_frame_id",
    ):
        _require_ascii_identifier(
            cast(str, getattr(transition, name)),
            name=f"outgoing_transition.{name}",
        )
    _require_python_int(
        transition.transition_index,
        name="outgoing_transition.transition_index",
    )
    if type(transition.facts) is not TransitionFactsV1:
        raise TypeError("outgoing transition facts must use their exact V1 root.")
    facts = transition.facts
    if (
        type(facts.schema_id) is not str
        or facts.schema_id != "marl_battlegrounds.evaluation.transition_facts"
        or type(facts.schema_version) is not int
        or facts.schema_version != 1
    ):
        raise ValueError("outgoing transition facts must retain their exact V1 schema.")
    if facts.has_transition is not True:
        raise ValueError("outgoing transition facts must describe a transition.")
    _require_python_int(
        facts.transition_start_step_count,
        name="outgoing transition start tick",
    )
    if type(facts.action_acceptance_facts) is not ActionAcceptanceFactsV1:
        raise TypeError("action acceptance must use its exact V1 root.")
    acceptance = facts.action_acceptance_facts
    submitted = acceptance.submitted_joint_action
    accepted = acceptance.accepted_joint_action
    if type(submitted) is not JointActionV1 or type(accepted) is not JointActionV1:
        raise TypeError("Shared submitted/accepted actions must use exact V1 roots.")
    for owner, name in ((submitted, "submitted"), (accepted, "accepted")):
        for values, field_name in (
            (owner.move, "move"),
            (owner.select_target, "select_target"),
            (owner.use_ultimate, "use_ultimate"),
        ):
            if type(values) is not tuple or len(values) != MAX_AGENT_SLOTS_V1:
                raise ValueError(
                    f"Shared {name} {field_name} must retain the ten-row axis."
                )
    return (
        SubmittedActionTupleV1(
            move_action=submitted.move[recipient_global_slot],
            target_action=submitted.select_target[recipient_global_slot],
            use_ultimate_action=submitted.use_ultimate[recipient_global_slot],
        ),
        AcceptedActionTupleV1(
            move_action=accepted.move[recipient_global_slot],
            target_action=accepted.select_target[recipient_global_slot],
            use_ultimate_action=accepted.use_ultimate[recipient_global_slot],
        ),
    )


def _validated_pov_mask(mask: ActorPovActionMaskV1) -> ActorPovActionMaskV1:
    if type(mask) is not ActorPovActionMaskV1:
        raise TypeError("POV decision mask must use its exact V1 root.")
    return ActorPovActionMaskV1.model_validate(mask.model_dump(mode="python"))


def _validated_oracle_mask(mask: ActionMaskV1) -> ActionMaskV1:
    if type(mask) is not ActionMaskV1:
        raise TypeError("Oracle decision mask must use its exact V1 root.")
    return ActionMaskV1.model_validate(mask.model_dump(mode="python"))


def _validated_scene(scene: AuthorizedBattlefieldSceneV1) -> None:
    if type(scene) is not AuthorizedBattlefieldSceneV1:
        raise TypeError("current_scene must be the exact authorized scene root.")
    scene.__post_init__()


def _scene_agent(
    scene: AuthorizedBattlefieldSceneV1,
    *,
    public_agent_id: str,
    relation: Literal["oracle", "self"] | None = None,
) -> AuthorizedAgentV1:
    matches = tuple(
        row for row in scene.agents if row.public_agent_id == public_agent_id
    )
    if len(matches) != 1:
        raise ValueError("inspection actor must join exactly one authorized body.")
    actor = matches[0]
    if relation is not None and actor.relation != relation:
        raise ValueError("inspection actor has the wrong authority relation.")
    return actor


def _target_axis(
    *,
    target_public_agent_id_by_action: tuple[str | None, ...],
    target_display_names: tuple[str, ...],
    scene: AuthorizedBattlefieldSceneV1,
) -> tuple[AuthorizedTargetActionV1, ...]:
    if (
        type(target_public_agent_id_by_action) is not tuple
        or len(target_public_agent_id_by_action) != NUM_TARGET_ACTIONS_V1
        or target_public_agent_id_by_action[0] is not None
    ):
        raise ValueError("target public identities must retain the exact V1 axis.")
    _require_text_tuple(
        target_display_names,
        name="target display names",
        length=NUM_TARGET_ACTIONS_V1,
    )
    positive_ids = target_public_agent_id_by_action[1:]
    if any(type(value) is not str or not value.strip() for value in positive_ids):
        raise ValueError("positive target actions require public identities.")
    public_ids = cast(tuple[str, ...], positive_ids)
    if len(set(public_ids)) != len(public_ids):
        raise ValueError("positive target identities must be one-to-one.")
    scene_by_public_id = {row.public_agent_id: row for row in scene.agents}
    rows: list[AuthorizedTargetActionV1] = [
        AuthorizedNoTargetActionV1(
            target_kind="no_target",
            target_action=0,
            display_name=target_display_names[0],
        )
    ]
    for target_action, public_agent_id in enumerate(public_ids, start=1):
        body = scene_by_public_id.get(public_agent_id)
        if body is None:
            rows.append(
                AuthorizedAxisOnlyTargetActionV1(
                    target_kind="axis_only_authorized_agent",
                    target_action=target_action,
                    display_name=target_display_names[target_action],
                    target_public_agent_id=public_agent_id,
                )
            )
        else:
            rows.append(
                AuthorizedVisibleTargetActionV1(
                    target_kind="visible_authorized_agent",
                    target_action=target_action,
                    display_name=target_display_names[target_action],
                    target_presentation_key=body.presentation_key,
                    target_public_agent_id=body.public_agent_id,
                    target_anchor=body.position,
                )
            )
    return tuple(rows)


def _decision_mask(
    *,
    owner: AuthorizedAgentV1,
    movement_action_display_names: tuple[str, ...],
    target_display_names: tuple[str, ...],
    use_ultimate_action_display_names: tuple[str, ...],
    target_public_agent_id_by_action: tuple[str | None, ...],
    move_mask: tuple[bool, ...],
    target_mask: tuple[bool, ...],
    use_ultimate_mask: tuple[bool, ...],
    joint_mask: tuple[tuple[bool, ...], ...],
    scene: AuthorizedBattlefieldSceneV1,
) -> AuthorizedDecisionMaskV1:
    return AuthorizedDecisionMaskV1(
        schema_version=AUTHORIZED_INSPECTION_SCHEMA_VERSION,
        owner_presentation_key=owner.presentation_key,
        owner_public_agent_id=owner.public_agent_id,
        movement_action_display_names=movement_action_display_names,
        movement_action_mask=move_mask,
        target_actions=_target_axis(
            target_public_agent_id_by_action=target_public_agent_id_by_action,
            target_display_names=target_display_names,
            scene=scene,
        ),
        target_action_mask=target_mask,
        use_ultimate_action_display_names=use_ultimate_action_display_names,
        use_ultimate_action_mask=use_ultimate_mask,
        target_use_ultimate_joint_mask=joint_mask,
    )


def _oracle_target_public_axis(
    context: EvaluationEpisodeContextV1,
    *,
    actor_internal_slot: int,
) -> tuple[str | None, ...]:
    catalog = context.static_mechanics_catalog
    target_axis_by_actor = catalog.global_recipient_slot_by_actor_and_target_action
    target_slots = target_axis_by_actor[actor_internal_slot]
    if len(target_slots) != NUM_TARGET_ACTIONS_V1 or target_slots[0] is not None:
        raise ValueError("Oracle target mapping changed the V1 target axis.")
    result: list[str | None] = [None]
    for target_slot in target_slots[1:]:
        if type(target_slot) is not int or not 0 <= target_slot < len(context.roster):
            raise ValueError("Oracle target mapping contains an invalid slot.")
        result.append(context.roster[target_slot].public_agent_id)
    return tuple(result)


def _oracle_current_join(
    context: EvaluationEpisodeContextV1,
    frame: EvaluationFrameV1,
    scene: AuthorizedBattlefieldSceneV1,
) -> dict[int, AuthorizedAgentV1]:
    if frame.episode_id != context.identity.episode_id:
        raise ValueError("Oracle current frame and context must join one episode.")
    active_roster = tuple(row for row in context.roster if row.configured_active)
    if len(scene.agents) != len(active_roster) or any(
        row.relation != "oracle" for row in scene.agents
    ):
        raise ValueError("Oracle current scene must contain every active Oracle row.")
    scene_by_public_id = {row.public_agent_id: row for row in scene.agents}
    if set(scene_by_public_id) != {row.public_agent_id for row in active_roster}:
        raise ValueError("Oracle scene identity must equal the active context roster.")
    by_slot: dict[int, AuthorizedAgentV1] = {}
    for roster in active_roster:
        agent = scene_by_public_id[roster.public_agent_id]
        if (
            agent.team_id != roster.configured_team_id
            or agent.class_id != roster.class_id
            or agent.position != frame.snapshot.agent_positions[roster.global_slot]
            or (agent.life_state == "alive")
            != frame.snapshot.alive_mask[roster.global_slot]
        ):
            raise ValueError(
                "Oracle scene does not join the displayed evaluation frame."
            )
        by_slot[roster.global_slot] = agent
    return by_slot


def _oracle_decision_mask(
    context: EvaluationEpisodeContextV1,
    frame: EvaluationFrameV1,
    scene: AuthorizedBattlefieldSceneV1,
    *,
    actor_internal_slot: int,
    owner: AuthorizedAgentV1,
) -> AuthorizedDecisionMaskV1:
    mask = _validated_oracle_mask(frame.action_mask)
    catalog = context.static_mechanics_catalog
    return _decision_mask(
        owner=owner,
        movement_action_display_names=catalog.movement_action_name_by_id,
        target_display_names=catalog.target_action_name_by_id,
        use_ultimate_action_display_names=(catalog.use_ultimate_action_name_by_id),
        target_public_agent_id_by_action=_oracle_target_public_axis(
            context,
            actor_internal_slot=actor_internal_slot,
        ),
        move_mask=mask.move_mask[actor_internal_slot],
        target_mask=mask.select_target_mask[actor_internal_slot],
        use_ultimate_mask=mask.use_ultimate_mask[actor_internal_slot],
        joint_mask=mask.select_target_use_ultimate_joint_mask[actor_internal_slot],
        scene=scene,
    )


def _pov_decision_mask(
    *,
    owner: AuthorizedAgentV1,
    axis_mapping: ActorPovAxisMappingV1,
    mask: ActorPovActionMaskV1,
    scene: AuthorizedBattlefieldSceneV1,
) -> AuthorizedDecisionMaskV1:
    if type(axis_mapping) is not ActorPovAxisMappingV1:
        raise TypeError("axis_mapping must be the exact ActorPovAxisMappingV1 root.")
    validated_axis = ActorPovAxisMappingV1.model_validate(
        axis_mapping.model_dump(mode="python")
    )
    validated_mask = _validated_pov_mask(mask)
    return _decision_mask(
        owner=owner,
        movement_action_display_names=validated_axis.movement_action_name_by_id,
        target_display_names=validated_axis.target_action_name_by_id,
        use_ultimate_action_display_names=(
            validated_axis.use_ultimate_action_name_by_id
        ),
        target_public_agent_id_by_action=(
            validated_axis.target_action_recipient_public_agent_id_by_id
        ),
        move_mask=validated_mask.move,
        target_mask=validated_mask.select_target,
        use_ultimate_mask=validated_mask.use_ultimate,
        joint_mask=validated_mask.select_target_use_ultimate_joint,
        scene=scene,
    )


def _replay_root(
    *,
    episode_id: str,
    transition_index: int,
    current_tick: int,
    transition_reference: ReplayTransitionReferenceV1,
    actor: AuthorizedAgentV1,
    decision_mask: AuthorizedDecisionMaskV1,
    submitted_action: SubmittedActionTupleV1,
    accepted_action: AcceptedActionTupleV1,
) -> ReplayInspectionPresentationV1:
    return ReplayInspectionPresentationV1(
        schema_version=AUTHORIZED_INSPECTION_SCHEMA_VERSION,
        inspection_kind="replay_recorded_outgoing_action",
        route_display_basis="accepted_action",
        episode_id=episode_id,
        outgoing_transition_index=transition_index,
        current_simulator_step_count=current_tick,
        transition_reference=transition_reference,
        actor_presentation_key=actor.presentation_key,
        actor_public_agent_id=actor.public_agent_id,
        actor_anchor=actor.position,
        decision_mask=decision_mask,
        submitted_action=submitted_action,
        accepted_action=accepted_action,
        combat_lane=_expected_combat_lane(accepted_action),
        accepted_target=decision_mask.target_actions[accepted_action.target_action],
    )


def _validate_outgoing_epoch(
    transition: EvaluationTransitionV1,
    *,
    episode_id: str,
    frame_index: int,
    frame_id: str,
    simulator_step_count: int,
) -> None:
    if (
        transition.episode_id != episode_id
        or transition.transition_index != frame_index
        or transition.transition_id != f"{episode_id}:transition:{frame_index}"
        or transition.start_frame_id != frame_id
        or transition.successor_frame_id != f"{episode_id}:frame:{frame_index + 1}"
        or transition.facts.transition_start_step_count != simulator_step_count
    ):
        raise ValueError("outgoing transition does not start at displayed s_n/m_n.")


def build_replay_oracle_inspection_v1(
    context: EvaluationEpisodeContextV1,
    current_frame: EvaluationFrameV1,
    current_scene: AuthorizedBattlefieldSceneV1,
    *,
    inspection_internal_slot: int | None,
    outgoing_transition: EvaluationTransitionV1 | None,
    final_frame_index: int,
) -> ReplayInspectionPresentationV1 | None:
    """Build one Oracle outgoing inspection from exact ``s_n/m_n/T_n`` facts."""
    context = _validated_context(context)
    current_frame = _validated_frame(current_frame)
    _validated_scene(current_scene)
    _require_python_int(final_frame_index, name="final_frame_index")
    if current_frame.frame_index > final_frame_index:
        raise ValueError("current frame exceeds the retained replay prefix.")
    agent_by_slot = _oracle_current_join(context, current_frame, current_scene)
    if inspection_internal_slot is not None:
        _require_python_int(
            inspection_internal_slot,
            name="inspection_internal_slot",
            maximum_exclusive=MAX_AGENT_SLOTS_V1,
        )
        roster = context.roster[inspection_internal_slot]
        if (
            not roster.configured_active
            or inspection_internal_slot not in agent_by_slot
        ):
            raise ValueError("Oracle inspection actor must be configured active.")
    should_have_outgoing = (
        inspection_internal_slot is not None
        and current_frame.frame_index < final_frame_index
    )
    if not should_have_outgoing:
        if outgoing_transition is not None:
            raise ValueError("unselected or final Oracle frames cannot receive T_n.")
        return None
    if inspection_internal_slot is None:  # pragma: no cover - narrowed above.
        raise AssertionError("Oracle inspection slot disappeared")
    if outgoing_transition is None:
        raise ValueError("selected non-final Oracle frames require exact T_n.")
    transition = _validated_transition(outgoing_transition)
    _validate_outgoing_epoch(
        transition,
        episode_id=current_frame.episode_id,
        frame_index=current_frame.frame_index,
        frame_id=current_frame.frame_id,
        simulator_step_count=current_frame.simulator_step_count,
    )
    acceptance = transition.facts.action_acceptance_facts
    submitted = acceptance.submitted_joint_action
    accepted = acceptance.accepted_joint_action
    actor = agent_by_slot[inspection_internal_slot]
    decision_mask = _oracle_decision_mask(
        context,
        current_frame,
        current_scene,
        actor_internal_slot=inspection_internal_slot,
        owner=actor,
    )
    return _replay_root(
        episode_id=current_frame.episode_id,
        transition_index=transition.transition_index,
        current_tick=current_frame.simulator_step_count,
        transition_reference=OracleReplayTransitionReferenceV1(
            reference_kind="oracle_recorded_transition",
            transition_id=transition.transition_id,
            start_frame_id=transition.start_frame_id,
            successor_frame_id=transition.successor_frame_id,
        ),
        actor=actor,
        decision_mask=decision_mask,
        submitted_action=SubmittedActionTupleV1(
            move_action=submitted.move[inspection_internal_slot],
            target_action=submitted.select_target[inspection_internal_slot],
            use_ultimate_action=submitted.use_ultimate[inspection_internal_slot],
        ),
        accepted_action=AcceptedActionTupleV1(
            move_action=accepted.move[inspection_internal_slot],
            target_action=accepted.select_target[inspection_internal_slot],
            use_ultimate_action=accepted.use_ultimate[inspection_internal_slot],
        ),
    )


def _validate_no_shared_current(
    source: ActorPovCurrentSliceV1 | ActorPovProjectionIndexV1,
    current: NoSharedObsAuthorizedScenePartsV1,
) -> tuple[ActorPovAxisMappingV1, ActorPovActionMaskV1, AuthorizedAgentV1]:
    if type(current) is not NoSharedObsAuthorizedScenePartsV1:
        raise TypeError("current must be exact NoSharedObs scene parts.")
    current.__post_init__()
    if type(source) is ActorPovProjectionIndexV1:
        validate_actor_pov_replay_content_v1(source.content)
        content = source.content
        index = current.source_frame_index
        if not 0 <= index < len(content.frames):
            raise ValueError("NoSharedObs current frame is outside retained content.")
        frame = content.frames[index]
        axis_mapping = content.axis_mapping
        source_episode_id = content.episode_id
        source_public_id = content.public_agent_id
    elif type(source) is ActorPovCurrentSliceV1:
        validated = ActorPovCurrentSliceV1.model_validate(
            source.model_dump(mode="python")
        )
        frame = validated.frame
        axis_mapping = validated.axis_mapping
        source_episode_id = validated.episode_id
        source_public_id = validated.public_agent_id
    else:
        raise TypeError(
            "NoSharedObs source must be an exact projection index or current slice."
        )
    if (
        current.source_episode_id != source_episode_id
        or current.recipient_public_agent_id != source_public_id
        or current.source_frame_index != frame.frame_index
        or current.source_recipient_frame_id != frame.pov_frame_id
        or current.source_simulator_step_count != frame.simulator_step_count
        or current.next_decision_action_mask != frame.action_mask
    ):
        raise ValueError("NoSharedObs scene parts do not join the recipient frame.")
    actor = _scene_agent(
        current.scene,
        public_agent_id=current.recipient_public_agent_id,
        relation="self",
    )
    if actor.presentation_key != current.recipient_presentation_key:
        raise ValueError("NoSharedObs recipient key does not join its scene body.")
    recorded_anchor = (
        frame.self_features[AGENT_FEATURE_X_V1],
        frame.self_features[AGENT_FEATURE_Y_V1],
    )
    if actor.position != recorded_anchor:
        raise ValueError("NoSharedObs recipient anchor does not join recorded s_n.")
    return axis_mapping, frame.action_mask, actor


def build_replay_no_shared_obs_inspection_v1(
    source: ActorPovProjectionIndexV1,
    current: NoSharedObsAuthorizedScenePartsV1,
) -> ReplayInspectionPresentationV1 | None:
    """Build fixed-recipient replay inspection from exact NoSharedObs content."""
    if type(source) is not ActorPovProjectionIndexV1:
        raise TypeError("source must be the exact ActorPovProjectionIndexV1 root.")
    axis_mapping, mask, actor = _validate_no_shared_current(source, current)
    content = source.content
    index = current.source_frame_index
    if index == len(content.transitions):
        return None
    if index > len(content.transitions):
        raise ValueError("NoSharedObs outgoing index exceeds retained transitions.")
    transition = content.transitions[index]
    if (
        type(transition) is not ActorPovTransitionV1
        or transition.episode_id != current.source_episode_id
        or transition.public_agent_id != current.recipient_public_agent_id
        or transition.transition_index != index
        or transition.start_pov_frame_id != current.source_recipient_frame_id
    ):
        raise ValueError("NoSharedObs outgoing row does not start at current s_n.")
    decision_mask = _pov_decision_mask(
        owner=actor,
        axis_mapping=axis_mapping,
        mask=mask,
        scene=current.scene,
    )
    submitted = transition.submitted_action
    accepted = transition.accepted_action
    return _replay_root(
        episode_id=current.source_episode_id,
        transition_index=index,
        current_tick=current.source_simulator_step_count,
        transition_reference=NoSharedObsReplayTransitionReferenceV1(
            reference_kind="no_shared_obs_actor_pov_transition",
            recipient_public_agent_id=current.recipient_public_agent_id,
            transition_id=transition.pov_transition_id,
            start_frame_id=transition.start_pov_frame_id,
            successor_frame_id=transition.successor_pov_frame_id,
        ),
        actor=actor,
        decision_mask=decision_mask,
        submitted_action=SubmittedActionTupleV1(
            move_action=submitted.move,
            target_action=submitted.select_target,
            use_ultimate_action=submitted.use_ultimate,
        ),
        accepted_action=AcceptedActionTupleV1(
            move_action=accepted.move,
            target_action=accepted.select_target,
            use_ultimate_action=accepted.use_ultimate,
        ),
    )


def _validate_shared_current(
    current: SharedObsAuthorizedScenePartsV1,
    recipient_source_material: SharedObsSourceMaterialProjectionV1,
    *,
    authorized_recipient_global_slot: int,
) -> tuple[ActorPovAxisMappingV1, ActorPovActionMaskV1, AuthorizedAgentV1, int]:
    if type(current) is not SharedObsAuthorizedScenePartsV1:
        raise TypeError("current must be exact SharedObs authorized scene parts.")
    if type(recipient_source_material) is not SharedObsSourceMaterialProjectionV1:
        raise TypeError("recipient source material must use its exact SharedObs root.")
    _require_python_int(
        authorized_recipient_global_slot,
        name="authorized_recipient_global_slot",
        maximum_exclusive=MAX_AGENT_SLOTS_V1,
    )
    authorized_team_id = 1 if authorized_recipient_global_slot < 5 else 2
    authorized_team_local_slot = authorized_recipient_global_slot % 5

    # Selective validation is intentional.  Shared source material is a wider
    # diagnostic root whose incoming identity, history, availability, and
    # contributor branches are not outgoing-decision authority.  Calling its
    # full ``__post_init__`` here would couple Agent output to Oracle history.
    for name in ("source_episode_id", "recipient_public_agent_id"):
        _require_text(cast(str, getattr(current, name)), name=name)
    for name in ("source_frame_index", "source_simulator_step_count"):
        _require_python_int(cast(int, getattr(current, name)), name=name)
    _require_text(
        current.recipient_presentation_key,
        name="recipient_presentation_key",
    )
    _validated_scene(current.scene)
    current_mask = _validated_pov_mask(current.next_decision_action_mask)

    if (
        type(recipient_source_material.schema_version) is not int
        or recipient_source_material.schema_version
        != SHARED_OBS_SOURCE_MATERIAL_PROJECTION_SCHEMA_VERSION
        or recipient_source_material.observation_materialization
        != "source_material_only"
        or recipient_source_material.exact_actor_input_export_available is not False
    ):
        raise ValueError("SharedObs recipient declaration is not canonical.")
    if type(recipient_source_material.axis_mapping) is not ActorPovAxisMappingV1:
        raise ValueError("SharedObs axis mapping must use its exact V1 root.")
    axis_mapping = ActorPovAxisMappingV1.model_validate(
        recipient_source_material.axis_mapping.model_dump(mode="python")
    )
    frame = recipient_source_material.base_sensor_frame
    source_scene = recipient_source_material.base_sensor_scene
    if type(frame) is not SharedObsBaseSensorFrameV1:
        raise ValueError("SharedObs base frame must use its exact scalar root.")
    if type(source_scene) is not SharedObsBaseSensorSceneV1:
        raise ValueError("SharedObs base scene must use its exact scalar root.")
    if (
        type(frame.schema_version) is not int
        or frame.schema_version != SHARED_OBS_SOURCE_MATERIAL_PROJECTION_SCHEMA_VERSION
        or frame.observation_materialization != "source_material_only"
    ):
        raise ValueError("SharedObs base-frame declaration is not canonical.")
    for name in ("episode_id", "public_agent_id"):
        _require_text(cast(str, getattr(frame, name)), name=name)
    for name in ("frame_index", "simulator_step_count"):
        _require_python_int(cast(int, getattr(frame, name)), name=name)
    frame_mask = _validated_pov_mask(frame.action_mask)
    self_actor = source_scene.self_actor
    if type(self_actor) is not ActorPovSelfSceneV1:
        raise ValueError("SharedObs self actor must use its exact scalar root.")
    _require_python_int(
        self_actor.global_slot,
        name="recipient actor slot",
        maximum_exclusive=MAX_AGENT_SLOTS_V1,
    )
    _require_text(self_actor.public_agent_id, name="recipient actor public ID")
    _require_python_int(
        self_actor.team_local_slot,
        name="recipient team-local slot",
        maximum_exclusive=5,
    )
    _require_python_int(
        self_actor.team_id,
        name="recipient team ID",
        minimum=1,
        maximum_exclusive=3,
    )
    _require_python_int(
        self_actor.class_id,
        name="recipient class ID",
        minimum=1,
        maximum_exclusive=6,
    )
    _require_point(self_actor.position, name="recipient actor position")
    actor = _scene_agent(
        current.scene,
        public_agent_id=current.recipient_public_agent_id,
        relation="self",
    )
    if actor.presentation_key != current.recipient_presentation_key:
        raise ValueError("SharedObs recipient key does not join its scene body.")
    recipient_pads = tuple(
        row
        for row in current.scene.spawn_pads
        if row.assigned_public_agent_id == current.recipient_public_agent_id
        and row.assigned_presentation_key == current.recipient_presentation_key
    )
    if len(recipient_pads) != 1:
        raise ValueError("SharedObs recipient must join one current spawn-pad row.")
    recipient_pad = recipient_pads[0]
    if (
        not recipient_pad.configured_active
        or self_actor.global_slot != authorized_recipient_global_slot
        or self_actor.team_id != authorized_team_id
        or self_actor.team_local_slot != authorized_team_local_slot
        or recipient_pad.team_id != authorized_team_id
        or recipient_pad.team_local_slot != authorized_team_local_slot
        or actor.team_id != authorized_team_id
        or actor.class_id != self_actor.class_id
    ):
        raise ValueError(
            "SharedObs self identity does not join its authorized recipient row."
        )
    relation_global_axes = (
        recipient_source_material.ally_observation_row_global_slot_by_id,
        recipient_source_material.enemy_observation_row_global_slot_by_id,
    )
    if any(
        type(axis) is not tuple
        or len(axis) != 5
        or any(
            type(slot) is not int or not 0 <= slot < MAX_AGENT_SLOTS_V1 for slot in axis
        )
        for axis in relation_global_axes
    ):
        raise ValueError("SharedObs relation slot axes must retain exact V1 rows.")
    flattened_relation_slots = (
        *relation_global_axes[0],
        *relation_global_axes[1],
    )
    if set(flattened_relation_slots) != set(range(MAX_AGENT_SLOTS_V1)):
        raise ValueError("SharedObs relation slot axes must partition all ten slots.")
    own_team_start = (authorized_team_id - 1) * 5
    opponent_team_start = 5 if own_team_start == 0 else 0
    if relation_global_axes != (
        tuple(range(own_team_start, own_team_start + 5)),
        tuple(range(opponent_team_start, opponent_team_start + 5)),
    ):
        raise ValueError("SharedObs relation slot axes changed team-local ordering.")
    if (
        relation_global_axes[0][authorized_team_local_slot]
        != authorized_recipient_global_slot
        or axis_mapping.ally_observation_row_public_agent_id_by_id[
            authorized_team_local_slot
        ]
        != self_actor.public_agent_id
    ):
        raise ValueError("SharedObs self identity does not join its fixed topology.")
    if type(frame.self_features) is not tuple or len(frame.self_features) <= max(
        AGENT_FEATURE_X_V1,
        AGENT_FEATURE_Y_V1,
        AGENT_FEATURE_TEAM_ID_V1,
        AGENT_FEATURE_ACTIVE_V1,
        AGENT_FEATURE_CLASS_ID_V1,
    ):
        raise ValueError("SharedObs self features do not retain identity columns.")
    if (
        frame.self_features[AGENT_FEATURE_TEAM_ID_V1] != float(self_actor.team_id)
        or frame.self_features[AGENT_FEATURE_ACTIVE_V1] != 1.0
        or frame.self_features[AGENT_FEATURE_CLASS_ID_V1] != float(self_actor.class_id)
    ):
        raise ValueError("SharedObs recorded self identity changed fixed ownership.")
    recorded_self_anchor = (
        frame.self_features[AGENT_FEATURE_X_V1],
        frame.self_features[AGENT_FEATURE_Y_V1],
    )
    _require_point(recorded_self_anchor, name="recorded recipient position")
    if self_actor.position != recorded_self_anchor:
        raise ValueError("SharedObs base scene does not join its self feature row.")
    if (
        source_scene.episode_id != frame.episode_id
        or source_scene.frame_index != frame.frame_index
        or source_scene.simulator_step_count != frame.simulator_step_count
    ):
        raise ValueError("SharedObs base scene does not join its frame header.")
    if (
        current.source_episode_id != frame.episode_id
        or current.source_frame_index != frame.frame_index
        or current.source_simulator_step_count != frame.simulator_step_count
        or current.recipient_public_agent_id != frame.public_agent_id
        or current.recipient_public_agent_id != self_actor.public_agent_id
        or current_mask != frame_mask
    ):
        raise ValueError(
            "SharedObs current parts do not join recipient source material."
        )
    expected_local_frame_id = (
        f"{frame.episode_id}:shared-obs-visual-union:"
        f"{frame.public_agent_id}:frame:{frame.frame_index}"
    )
    if current.source_recipient_frame_id != expected_local_frame_id:
        raise ValueError("SharedObs current frame identity is not recipient-local.")
    if actor.position != self_actor.position:
        raise ValueError("SharedObs recipient anchor does not join recorded s_n.")
    return (
        axis_mapping,
        frame_mask,
        actor,
        authorized_recipient_global_slot,
    )


def build_replay_shared_obs_inspection_v1(
    current: SharedObsAuthorizedScenePartsV1,
    recipient_source_material: SharedObsSourceMaterialProjectionV1,
    *,
    authorized_recipient_global_slot: int,
    outgoing_transition: EvaluationTransitionV1 | None,
    final_frame_index: int,
) -> ReplayInspectionPresentationV1 | None:
    """Build a recipient-only SharedObs inspection with no canonical IDs."""
    axis_mapping, mask, actor, actor_internal_slot = _validate_shared_current(
        current,
        recipient_source_material,
        authorized_recipient_global_slot=authorized_recipient_global_slot,
    )
    _require_python_int(final_frame_index, name="final_frame_index")
    index = current.source_frame_index
    if index > final_frame_index:
        raise ValueError("SharedObs current frame exceeds the retained prefix.")
    if index == final_frame_index:
        if outgoing_transition is not None:
            raise ValueError("final SharedObs frames cannot receive T_n.")
        return None
    if outgoing_transition is None:
        raise ValueError("non-final SharedObs frames require exact T_n.")
    submitted_action, accepted_action = _shared_recipient_action_rows(
        outgoing_transition,
        recipient_global_slot=actor_internal_slot,
    )
    canonical_frame_id = f"{current.source_episode_id}:frame:{index}"
    _validate_outgoing_epoch(
        outgoing_transition,
        episode_id=current.source_episode_id,
        frame_index=index,
        frame_id=canonical_frame_id,
        simulator_step_count=current.source_simulator_step_count,
    )
    decision_mask = _pov_decision_mask(
        owner=actor,
        axis_mapping=axis_mapping,
        mask=mask,
        scene=current.scene,
    )
    prefix = (
        f"{current.source_episode_id}:shared-obs-visual-union:"
        f"{current.recipient_public_agent_id}"
    )
    return _replay_root(
        episode_id=current.source_episode_id,
        transition_index=index,
        current_tick=current.source_simulator_step_count,
        transition_reference=SharedObsReplayTransitionReferenceV1(
            reference_kind="shared_obs_visual_union_transition",
            recipient_public_agent_id=current.recipient_public_agent_id,
            transition_id=f"{prefix}:transition:{index}",
            start_frame_id=f"{prefix}:frame:{index}",
            successor_frame_id=f"{prefix}:frame:{index + 1}",
        ),
        actor=actor,
        decision_mask=decision_mask,
        submitted_action=submitted_action,
        accepted_action=accepted_action,
    )


def _draft_root(
    *,
    current_tick: int,
    actor: AuthorizedAgentV1,
    decision_mask: AuthorizedDecisionMaskV1,
    draft_move_action: int,
    draft_target_action: int,
    draft_armed_lane: DraftArmedLaneV1,
) -> LiveDraftInspectionPresentationV1:
    draft = LiveDraftActionTupleV1(
        move_action=draft_move_action,
        target_action=draft_target_action,
        armed_lane=draft_armed_lane,
    )
    if draft_armed_lane == "none":
        lane_legal = None
        pair_legal = None
    else:
        lane = 0 if draft_armed_lane == "basic" else 1
        lane_legal = decision_mask.use_ultimate_action_mask[lane]
        pair_legal = decision_mask.target_use_ultimate_joint_mask[draft_target_action][
            lane
        ]
    return LiveDraftInspectionPresentationV1(
        schema_version=AUTHORIZED_INSPECTION_SCHEMA_VERSION,
        inspection_kind="live_draft_action",
        route_display_basis="draft_action",
        current_simulator_step_count=current_tick,
        actor_presentation_key=actor.presentation_key,
        actor_public_agent_id=actor.public_agent_id,
        actor_anchor=actor.position,
        decision_mask=decision_mask,
        draft_action=draft,
        draft_target=decision_mask.target_actions[draft_target_action],
        draft_legality=LiveDraftLegalityV1(
            move_action_is_legal=decision_mask.movement_action_mask[draft_move_action],
            target_action_is_legal=decision_mask.target_action_mask[
                draft_target_action
            ],
            armed_lane_is_legal=lane_legal,
            combat_pair_is_legal=pair_legal,
        ),
    )


def _oracle_target_action_for_internal_slot(
    context: EvaluationEpisodeContextV1,
    *,
    actor_internal_slot: int,
    target_internal_slot: int | None,
) -> int:
    if target_internal_slot is None:
        return 0
    _require_python_int(
        target_internal_slot,
        name="draft_target_internal_slot",
        maximum_exclusive=MAX_AGENT_SLOTS_V1,
    )
    catalog = context.static_mechanics_catalog
    target_axis_by_actor = catalog.global_recipient_slot_by_actor_and_target_action
    target_axis = target_axis_by_actor[actor_internal_slot]
    matches = tuple(
        action
        for action, slot in enumerate(target_axis)
        if slot == target_internal_slot
    )
    if len(matches) != 1 or matches[0] == 0:
        raise ValueError("draft target slot does not join one positive target action.")
    return matches[0]


def build_live_oracle_draft_inspection_v1(
    context: EvaluationEpisodeContextV1,
    current_frame: EvaluationFrameV1,
    current_scene: AuthorizedBattlefieldSceneV1,
    *,
    controlled_internal_slot: int,
    draft_move_action: int,
    draft_target_internal_slot: int | None,
    draft_armed_lane: DraftArmedLaneV1,
) -> LiveDraftInspectionPresentationV1:
    """Build one live Oracle draft without recorded acceptance identity."""
    context = _validated_context(context)
    current_frame = _validated_frame(current_frame)
    _validated_scene(current_scene)
    _require_python_int(
        controlled_internal_slot,
        name="controlled_internal_slot",
        maximum_exclusive=MAX_AGENT_SLOTS_V1,
    )
    agent_by_slot = _oracle_current_join(context, current_frame, current_scene)
    roster = context.roster[controlled_internal_slot]
    if not roster.configured_active or controlled_internal_slot not in agent_by_slot:
        raise ValueError("live Oracle actor must be configured active.")
    actor = agent_by_slot[controlled_internal_slot]
    decision_mask = _oracle_decision_mask(
        context,
        current_frame,
        current_scene,
        actor_internal_slot=controlled_internal_slot,
        owner=actor,
    )
    target_action = _oracle_target_action_for_internal_slot(
        context,
        actor_internal_slot=controlled_internal_slot,
        target_internal_slot=draft_target_internal_slot,
    )
    return _draft_root(
        current_tick=current_frame.simulator_step_count,
        actor=actor,
        decision_mask=decision_mask,
        draft_move_action=draft_move_action,
        draft_target_action=target_action,
        draft_armed_lane=draft_armed_lane,
    )


def build_live_no_shared_obs_draft_inspection_v1(
    source: ActorPovCurrentSliceV1,
    current: NoSharedObsAuthorizedScenePartsV1,
    *,
    draft_move_action: int,
    draft_target_action: int,
    draft_armed_lane: DraftArmedLaneV1,
) -> LiveDraftInspectionPresentationV1:
    """Build one fixed-recipient NoSharedObs live draft."""
    if type(source) is not ActorPovCurrentSliceV1:
        raise TypeError("source must be the exact ActorPovCurrentSliceV1 root.")
    axis_mapping, mask, actor = _validate_no_shared_current(source, current)
    decision_mask = _pov_decision_mask(
        owner=actor,
        axis_mapping=axis_mapping,
        mask=mask,
        scene=current.scene,
    )
    return _draft_root(
        current_tick=current.source_simulator_step_count,
        actor=actor,
        decision_mask=decision_mask,
        draft_move_action=draft_move_action,
        draft_target_action=draft_target_action,
        draft_armed_lane=draft_armed_lane,
    )


def build_live_shared_obs_draft_inspection_v1(
    current: SharedObsAuthorizedScenePartsV1,
    recipient_source_material: SharedObsSourceMaterialProjectionV1,
    *,
    authorized_recipient_global_slot: int,
    draft_move_action: int,
    draft_target_action: int,
    draft_armed_lane: DraftArmedLaneV1,
) -> LiveDraftInspectionPresentationV1:
    """Build one fixed-recipient SharedObs live draft."""
    axis_mapping, mask, actor, _actor_internal_slot = _validate_shared_current(
        current,
        recipient_source_material,
        authorized_recipient_global_slot=authorized_recipient_global_slot,
    )
    decision_mask = _pov_decision_mask(
        owner=actor,
        axis_mapping=axis_mapping,
        mask=mask,
        scene=current.scene,
    )
    return _draft_root(
        current_tick=current.source_simulator_step_count,
        actor=actor,
        decision_mask=decision_mask,
        draft_move_action=draft_move_action,
        draft_target_action=draft_target_action,
        draft_armed_lane=draft_armed_lane,
    )


__all__ = [
    "AUTHORIZED_INSPECTION_SCHEMA_VERSION",
    "AuthorizedAxisOnlyTargetActionV1",
    "AuthorizedDecisionMaskV1",
    "AuthorizedNoTargetActionV1",
    "AuthorizedTargetActionV1",
    "AuthorizedVisibleTargetActionV1",
    "CombatLaneV1",
    "DraftArmedLaneV1",
    "LiveDraftActionTupleV1",
    "LiveDraftInspectionPresentationV1",
    "LiveDraftLegalityV1",
    "NoSharedObsReplayTransitionReferenceV1",
    "OracleReplayTransitionReferenceV1",
    "ReplayInspectionPresentationV1",
    "ReplayTransitionReferenceV1",
    "SharedObsReplayTransitionReferenceV1",
    "build_live_no_shared_obs_draft_inspection_v1",
    "build_live_oracle_draft_inspection_v1",
    "build_live_shared_obs_draft_inspection_v1",
    "build_replay_no_shared_obs_inspection_v1",
    "build_replay_oracle_inspection_v1",
    "build_replay_shared_obs_inspection_v1",
]
