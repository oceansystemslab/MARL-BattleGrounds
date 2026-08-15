"""Versioned loopback protocol models for the live visual debugger.

Command and request roots are strict Pydantic-validated inbound contracts. Live
frame response roots are outbound typed envelopes over exact renderer
dataclasses: they are serialized once for the browser's strict V2 normalizer,
not accepted as replay or artifact input. Canonical replay/artifact loaders
independently validate their evaluation-record inputs before the renderer
adapters construct these outbound Scene/Event projections.
"""

from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from marl_battlegrounds.evaluation.wire_shapes import (
    MAX_AGENT_SLOTS_V1 as MAX_AGENT_SLOTS,
)
from marl_battlegrounds.evaluation.wire_shapes import (
    NUM_MOVE_ACTIONS_V1 as NUM_MOVE_ACTIONS,
)
from marl_battlegrounds.evaluation.wire_shapes import (
    NUM_TARGET_ACTIONS_V1 as NUM_TARGET_ACTIONS,
)
from marl_battlegrounds.rendering.pov_scene import ActorPovAnalyzerProjectionV1
from marl_battlegrounds.rendering.scene import (
    ResearcherAnalyzerProjectionV2,
    TargetDisclosure,
)

PROTOCOL_SCHEMA_VERSION = 1
LIVE_FRAME_SCHEMA_VERSION = 2

type ViewMode = Literal["researcher", "pov"]
type Preset = Literal["presentation", "analysis"]
type PresentationPreset = Preset
type PendingSubmissionScope = Literal[
    "joint_turn",
    "controlled_actor",
    "scripted_playback",
]
type CommandResult = Literal[
    "applied",
    "duplicate",
    "no_op",
    "shutdown_scheduled",
]
type RecordingLifecycleV1 = Literal[
    "recording",
    "sealed",
    "finalized_unsaved",
    "persistence_failed",
    "saved",
    "reviewing",
    "discarded",
]
type RecordingCompletionStateV1 = Literal[
    "complete",
    "partial",
    "interrupted",
    "failed",
]
type RecordingPersistenceErrorCodeV1 = Literal[
    "target_unavailable",
    "publication_failed",
    "verification_failed",
]
type ApiErrorCode = Literal[
    "invalid_request",
    "unauthorized",
    "forbidden_origin",
    "not_found",
    "method_not_allowed",
    "payload_too_large",
    "unsupported_media_type",
    "stale_revision",
    "command_id_conflict",
    "server_shutting_down",
    "internal_error",
]

_OpaqueId = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9_-]+$",
    ),
]
_CanonicalScientificId = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=256,
        pattern=r"^[A-Za-z0-9_.:-]+$",
    ),
]
_ScenarioName = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z0-9_]+$",
    ),
]
_KeyName = Annotated[str, StringConstraints(min_length=1, max_length=64)]
_GlobalSlot = Annotated[int, Field(ge=0, lt=MAX_AGENT_SLOTS)]
_MoveAction = Annotated[int, Field(ge=0, lt=NUM_MOVE_ACTIONS)]
_TargetAction = Annotated[int, Field(ge=0, lt=NUM_TARGET_ACTIONS)]
_NonNegativeInt = Annotated[int, Field(ge=0)]
_FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
_PositiveUnitFloat = Annotated[
    float,
    Field(gt=0.0, le=1.0, allow_inf_nan=False),
]


class _ProtocolModel(BaseModel):
    """Strict immutable base for every value crossing the HTTP boundary."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
    )


class _ModifiedInputV1(_ProtocolModel):
    """Raw browser modifier state accompanying an input event."""

    shift_key: bool = False
    ctrl_key: bool = False
    alt_key: bool = False
    meta_key: bool = False


class KeyboardCommandV1(_ModifiedInputV1):
    command_type: Literal["keyboard"] = "keyboard"
    key: _KeyName
    repeat: bool = False


class BattlefieldPointerCommandV1(_ModifiedInputV1):
    command_type: Literal["battlefield_pointer"] = "battlefield_pointer"
    world_x: _FiniteFloat
    world_y: _FiniteFloat
    button: Literal["primary", "secondary"]


class RosterSelectionCommandV1(_ProtocolModel):
    command_type: Literal["roster_selection"] = "roster_selection"
    role: Literal["control", "target"]
    global_slot: _GlobalSlot


class ActorPovTargetActionCommandV1(_ProtocolModel):
    """Select an authorized POV target without disclosing a global slot."""

    command_type: Literal["actor_pov_target_action"] = "actor_pov_target_action"
    target_action: _TargetAction


class ScenarioSwitchCommandV1(_ProtocolModel):
    command_type: Literal["scenario_switch"] = "scenario_switch"
    scenario_name: _ScenarioName


class ResetCommandV1(_ProtocolModel):
    command_type: Literal["reset"] = "reset"


class SetViewCommandV1(_ProtocolModel):
    command_type: Literal["set_view"] = "set_view"
    view_mode: ViewMode


class SetPresetCommandV1(_ProtocolModel):
    command_type: Literal["set_preset"] = "set_preset"
    preset: Preset

    @field_validator("preset", mode="before")
    @classmethod
    def _canonicalize_legacy_technical_preset(
        cls,
        value: object,
    ) -> object:
        return (
            "analysis"
            if value in ("presentation", "analysis", "technical", "debug")
            else value
        )


class ExitCommandV1(_ProtocolModel):
    command_type: Literal["exit"] = "exit"


class FinishAndReviewCommandV1(_ProtocolModel):
    command_type: Literal["finish_and_review"] = "finish_and_review"


class ReviewReplayCommandV1(_ProtocolModel):
    command_type: Literal["review_replay"] = "review_replay"


class RetrySaveCommandV1(_ProtocolModel):
    command_type: Literal["retry_save"] = "retry_save"


class SaveAsCommandV1(_ProtocolModel):
    command_type: Literal["save_as"] = "save_as"
    file_name: Annotated[
        str,
        StringConstraints(
            min_length=20,
            max_length=160,
            pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*\.marlbg-replay\.json$",
        ),
    ]


type RecordingReplacementCommandV1 = Annotated[
    ResetCommandV1 | ScenarioSwitchCommandV1,
    Field(discriminator="command_type"),
]


class ConfirmDiscardAndReplaceCommandV1(_ProtocolModel):
    command_type: Literal["confirm_discard_and_replace"] = "confirm_discard_and_replace"
    replacement: RecordingReplacementCommandV1


type DebuggerCommandV1 = Annotated[
    KeyboardCommandV1
    | BattlefieldPointerCommandV1
    | RosterSelectionCommandV1
    | ActorPovTargetActionCommandV1
    | ScenarioSwitchCommandV1
    | ResetCommandV1
    | SetViewCommandV1
    | SetPresetCommandV1
    | FinishAndReviewCommandV1
    | ReviewReplayCommandV1
    | RetrySaveCommandV1
    | SaveAsCommandV1
    | ConfirmDiscardAndReplaceCommandV1
    | ExitCommandV1,
    Field(discriminator="command_type"),
]


class CommandRequestV1(_ProtocolModel):
    schema_version: Literal[1] = PROTOCOL_SCHEMA_VERSION
    client_id: _OpaqueId
    command_id: _OpaqueId
    base_revision: _NonNegativeInt
    command: DebuggerCommandV1


class TargetReferenceV1(_ProtocolModel):
    disclosure: TargetDisclosure
    global_slot: _GlobalSlot | None = None

    @model_validator(mode="after")
    def _validate_disclosure(self) -> Self:
        if self.disclosure == "public":
            if self.global_slot is None:
                raise ValueError("public target references require global_slot.")
        elif self.global_slot is not None:
            raise ValueError(
                f"{self.disclosure} target references must omit global_slot."
            )
        return self


class ActionTupleCardV1(_ProtocolModel):
    move_action: int
    target_action: int | None
    use_ultimate_action: int
    target: TargetReferenceV1
    summary: str

    @model_validator(mode="after")
    def _validate_target_action_disclosure(self) -> Self:
        if self.target.disclosure == "public":
            if self.target_action is None or self.target_action <= 0:
                raise ValueError(
                    "public action targets require a positive target_action."
                )
        elif self.target.disclosure == "target_none":
            if self.target_action != 0:
                raise ValueError("target-none actions require target_action zero.")
        elif self.target.disclosure == "redacted":
            if self.target_action is not None:
                raise ValueError("redacted actions must omit target_action.")
        elif self.target.disclosure == "invalid" and self.target_action is None:
            raise ValueError("invalid actions retain their exact target_action.")
        return self


class PendingActionCardV1(_ProtocolModel):
    label: Literal[
        "PENDING / WILL SUBMIT",
        "PLAYBACK / INSPECTION ONLY",
    ] = "PENDING / WILL SUBMIT"
    actor_global_slot: _GlobalSlot
    move_action: _NonNegativeInt
    target_action: _NonNegativeInt | None
    armed_lane: Literal[0, 1] | None
    arm_origin: Literal["automatic", "explicit"] | None
    target: TargetReferenceV1
    movement_mask_value: bool
    pair_mask_value: bool | None
    summary: str

    @model_validator(mode="after")
    def _validate_pending_disclosure(self) -> Self:
        if self.target.disclosure == "public":
            if self.target_action is None or self.target_action <= 0:
                raise ValueError(
                    "public pending targets require a positive target_action."
                )
        elif self.target.disclosure == "target_none":
            if self.target_action != 0:
                raise ValueError(
                    "target-none pending actions require target_action zero."
                )
        elif self.target.disclosure == "redacted":
            if self.target_action is not None or self.pair_mask_value is not None:
                raise ValueError(
                    "redacted pending actions omit target and pair-mask values."
                )
        else:
            raise ValueError("pending actions cannot carry an invalid target.")
        return self


class ActorActionResultV1(_ProtocolModel):
    actor_global_slot: _GlobalSlot
    submitted: ActionTupleCardV1
    accepted: ActionTupleCardV1
    movement_mask_value: bool
    pair_mask_value: bool | None
    movement_accepted: bool
    combat_result: Literal[
        "accepted",
        "rejected",
        "canonical_noop",
        "undisclosed",
    ]


class LatestTransitionCardV1(_ProtocolModel):
    label: Literal["LATEST ACCEPTED RESULT"] = "LATEST ACCEPTED RESULT"
    transition_id: _NonNegativeInt
    submission_kind: Literal["interactive", "scripted"]
    actors: tuple[ActorActionResultV1, ...]


class LatestTransitionCardV2(_ProtocolModel):
    label: Literal["LATEST ACCEPTED RESULT"] = "LATEST ACCEPTED RESULT"
    transition_index: _NonNegativeInt
    transition_id: _CanonicalScientificId
    submission_kind: Literal["interactive", "scripted"]
    actors: tuple[ActorActionResultV1, ...]


class DiagnosticFactV1(_ProtocolModel):
    fact_id: _OpaqueId
    label: str
    value: str
    technical: bool = False


class CandidateLegalityCardV1(_ProtocolModel):
    """One exact controlled-actor target/lane row copied from the current mask."""

    target_action: _TargetAction
    target: TargetReferenceV1
    lane_0_available: bool
    lane_1_available: bool
    basic_available: bool
    ultimate_available: bool

    @model_validator(mode="after")
    def _validate_target(self) -> Self:
        if self.target_action == 0:
            if self.target.disclosure != "target_none":
                raise ValueError("target action zero requires a target-none reference.")
        elif self.target.disclosure != "public":
            raise ValueError(
                "positive candidate target actions require public references."
            )
        expected_basic = self.target_action > 0 and self.lane_0_available
        if self.basic_available != expected_basic:
            raise ValueError(
                "basic availability must exclude canonical target-none lane zero."
            )
        if self.ultimate_available != self.lane_1_available:
            raise ValueError(
                "ultimate availability must match the exact lane-one mask value."
            )
        return self


class MovementLegalityCardV1(_ProtocolModel):
    """One controlled-actor movement category copied from the current mask."""

    move_action: _MoveAction
    available: bool


class HudFrameV1(_ProtocolModel):
    roster_global_slots: tuple[_GlobalSlot, ...]
    controlled_global_slot: _GlobalSlot
    selected_global_slot: _GlobalSlot | None
    pending_submission_scope: PendingSubmissionScope
    pending_actions: tuple[PendingActionCardV1, ...]
    pending_action: PendingActionCardV1
    latest_transition: LatestTransitionCardV1 | None
    movement_legalities: tuple[MovementLegalityCardV1, ...]
    candidate_legalities: tuple[CandidateLegalityCardV1, ...] = ()
    diagnostics: tuple[DiagnosticFactV1, ...] = ()

    @model_validator(mode="after")
    def _validate_slot_references(self) -> Self:
        if len(self.roster_global_slots) != len(set(self.roster_global_slots)):
            raise ValueError("roster_global_slots must be unique.")
        roster = set(self.roster_global_slots)
        if self.controlled_global_slot not in roster:
            raise ValueError("controlled_global_slot must occur in the roster.")
        if (
            self.selected_global_slot is not None
            and self.selected_global_slot not in roster
        ):
            raise ValueError("selected_global_slot must occur in the roster.")
        if self.pending_action.actor_global_slot != self.controlled_global_slot:
            raise ValueError("pending action actor must be the controlled actor.")
        pending_slots = tuple(
            pending.actor_global_slot for pending in self.pending_actions
        )
        if not pending_slots:
            raise ValueError("pending_actions must contain an authorized actor row.")
        if len(pending_slots) != len(set(pending_slots)):
            raise ValueError("pending action actor slots must be unique.")
        if not set(pending_slots).issubset(roster):
            raise ValueError("pending action actors must occur in the roster.")
        controlled_pending = next(
            (
                pending
                for pending in self.pending_actions
                if pending.actor_global_slot == self.controlled_global_slot
            ),
            None,
        )
        if controlled_pending is None or self.pending_action != controlled_pending:
            raise ValueError(
                "pending_action must equal the controlled pending_actions row."
            )
        movement_actions = tuple(
            legality.move_action for legality in self.movement_legalities
        )
        if movement_actions != tuple(range(NUM_MOVE_ACTIONS)):
            raise ValueError(
                "movement legality rows must cover every movement action in "
                "canonical order."
            )
        expected_pending_slots = (
            self.roster_global_slots
            if self.pending_submission_scope == "joint_turn"
            else (self.controlled_global_slot,)
        )
        if pending_slots != expected_pending_slots:
            raise ValueError(
                "pending action rows must exactly match their submission scope."
            )
        expected_pending_label = (
            "PLAYBACK / INSPECTION ONLY"
            if self.pending_submission_scope == "scripted_playback"
            else "PENDING / WILL SUBMIT"
        )
        if any(
            pending.label != expected_pending_label for pending in self.pending_actions
        ):
            raise ValueError("pending action labels must match their submission scope.")
        for pending in self.pending_actions:
            target = pending.target.global_slot
            if target is not None and target not in roster:
                raise ValueError("public pending targets must occur in the roster.")
        if self.candidate_legalities:
            target_actions = tuple(
                candidate.target_action for candidate in self.candidate_legalities
            )
            if len(target_actions) != len(set(target_actions)):
                raise ValueError("candidate target actions must be unique.")
            target_none_count = sum(
                candidate.target.disclosure == "target_none"
                for candidate in self.candidate_legalities
            )
            if target_none_count != 1:
                raise ValueError(
                    "candidate legality rows require exactly one target-none row."
                )
            public_target_rows = tuple(
                candidate.target.global_slot
                for candidate in self.candidate_legalities
                if candidate.target.disclosure == "public"
            )
            if len(public_target_rows) != len(set(public_target_rows)):
                raise ValueError("candidate public target slots must be unique.")
            if set(public_target_rows) != roster:
                raise ValueError(
                    "candidate legality public targets must exactly match the roster."
                )
        latest_actors = (
            () if self.latest_transition is None else self.latest_transition.actors
        )
        latest_actor_slots = tuple(actor.actor_global_slot for actor in latest_actors)
        if len(latest_actor_slots) != len(set(latest_actor_slots)):
            raise ValueError("latest-transition actor slots must be unique.")
        for actor in latest_actors:
            if actor.actor_global_slot not in roster:
                raise ValueError("latest-transition actors must occur in the roster.")
            for action in (actor.submitted, actor.accepted):
                target = action.target.global_slot
                if target is not None and target not in roster:
                    raise ValueError(
                        "public latest-transition targets must occur in the roster."
                    )
        return self


class ResearcherHudFrameV2(_ProtocolModel):
    """Researcher-authorized HUD with canonical transition identity."""

    roster_global_slots: tuple[_GlobalSlot, ...]
    controlled_global_slot: _GlobalSlot
    selected_global_slot: _GlobalSlot | None
    pending_submission_scope: PendingSubmissionScope
    pending_actions: tuple[PendingActionCardV1, ...]
    pending_action: PendingActionCardV1
    latest_transition: LatestTransitionCardV2 | None
    movement_legalities: tuple[MovementLegalityCardV1, ...]
    candidate_legalities: tuple[CandidateLegalityCardV1, ...] = ()
    diagnostics: tuple[DiagnosticFactV1, ...] = ()

    @model_validator(mode="after")
    def _validate_researcher_hud(self) -> Self:
        HudFrameV1(
            roster_global_slots=self.roster_global_slots,
            controlled_global_slot=self.controlled_global_slot,
            selected_global_slot=self.selected_global_slot,
            pending_submission_scope=self.pending_submission_scope,
            pending_actions=self.pending_actions,
            pending_action=self.pending_action,
            latest_transition=None,
            movement_legalities=self.movement_legalities,
            candidate_legalities=self.candidate_legalities,
            diagnostics=self.diagnostics,
        )
        roster = set(self.roster_global_slots)
        latest = self.latest_transition
        if latest is not None:
            actor_slots = tuple(row.actor_global_slot for row in latest.actors)
            if len(actor_slots) != len(set(actor_slots)) or not set(
                actor_slots
            ).issubset(roster):
                raise ValueError(
                    "researcher latest-result actors must be unique roster members."
                )
            for actor in latest.actors:
                for action in (actor.submitted, actor.accepted):
                    target = action.target.global_slot
                    if target is not None and target not in roster:
                        raise ValueError(
                            "researcher latest-result targets must occur in the roster."
                        )
        return self


class ActorPovTargetReferenceV1(_ProtocolModel):
    """Actor-relative target reference with no recipient global-slot disclosure."""

    target_action: int
    public_agent_id: _CanonicalScientificId | None = None

    @model_validator(mode="after")
    def _validate_target(self) -> Self:
        in_domain = 0 <= self.target_action < NUM_TARGET_ACTIONS
        if not in_domain:
            if self.public_agent_id is not None:
                raise ValueError("out-of-domain POV targets omit recipient identity.")
        elif self.target_action == 0:
            if self.public_agent_id is not None:
                raise ValueError("POV target-none action omits recipient identity.")
        elif self.public_agent_id is None:
            raise ValueError("positive POV target actions require a public agent ID.")
        return self


class ActorPovActionTupleCardV1(_ProtocolModel):
    move_action: int
    target: ActorPovTargetReferenceV1
    use_ultimate_action: int
    summary: str


class ActorPovPendingActionCardV1(_ProtocolModel):
    label: Literal[
        "PENDING / WILL SUBMIT",
        "PLAYBACK / INSPECTION ONLY",
    ] = "PENDING / WILL SUBMIT"
    actor_public_agent_id: _CanonicalScientificId
    move_action: _MoveAction
    target: ActorPovTargetReferenceV1
    armed_lane: Literal[0, 1] | None
    arm_origin: Literal["automatic", "explicit"] | None
    movement_mask_value: bool
    pair_mask_value: bool | None
    summary: str


class ActorPovActionResultV1(_ProtocolModel):
    actor_public_agent_id: _CanonicalScientificId
    submitted: ActorPovActionTupleCardV1
    accepted: ActorPovActionTupleCardV1
    submitted_tuple_is_out_of_domain: bool
    movement_rejected: bool
    combat_pair_rejected: bool
    movement_accepted: bool
    combat_result: Literal["accepted", "rejected", "canonical_noop"]

    @model_validator(mode="after")
    def _validate_outcomes(self) -> Self:
        expected_movement = not (
            self.submitted_tuple_is_out_of_domain or self.movement_rejected
        )
        if self.movement_accepted != expected_movement:
            raise ValueError(
                "POV movement outcome must match recorded rejection truth."
            )
        rejected_combat = (
            self.submitted_tuple_is_out_of_domain or self.combat_pair_rejected
        )
        accepted = self.accepted
        expected_combat = (
            "rejected"
            if rejected_combat
            else "canonical_noop"
            if (
                accepted.target.target_action == 0 and accepted.use_ultimate_action == 0
            )
            else "accepted"
        )
        if self.combat_result != expected_combat:
            raise ValueError("POV combat result must match recorded rejection truth.")
        return self


class ActorPovLatestTransitionCardV1(_ProtocolModel):
    label: Literal["LATEST ACCEPTED RESULT"] = "LATEST ACCEPTED RESULT"
    transition_index: _NonNegativeInt
    pov_transition_id: _CanonicalScientificId
    submission_kind: Literal["interactive", "scripted"]
    actor: ActorPovActionResultV1


class ActorPovCandidateLegalityCardV1(_ProtocolModel):
    target: ActorPovTargetReferenceV1
    lane_0_available: bool
    lane_1_available: bool
    basic_available: bool
    ultimate_available: bool

    @model_validator(mode="after")
    def _validate_availability(self) -> Self:
        expected_basic = self.target.target_action > 0 and self.lane_0_available
        if self.basic_available != expected_basic:
            raise ValueError("POV Basic availability must exclude target-none.")
        if self.ultimate_available != self.lane_1_available:
            raise ValueError("POV Ultimate availability must equal lane one.")
        return self


class ActorPovHudFrameV1(_ProtocolModel):
    """Recipient-authorized HUD without researcher roster or global targets."""

    controlled_public_agent_id: _CanonicalScientificId
    pending_submission_scope: Literal["controlled_actor", "scripted_playback"]
    pending_action: ActorPovPendingActionCardV1
    latest_transition: ActorPovLatestTransitionCardV1 | None
    movement_legalities: tuple[MovementLegalityCardV1, ...]
    candidate_legalities: tuple[ActorPovCandidateLegalityCardV1, ...]
    diagnostics: tuple[DiagnosticFactV1, ...] = ()

    @model_validator(mode="after")
    def _validate_actor_hud(self) -> Self:
        if self.pending_action.actor_public_agent_id != self.controlled_public_agent_id:
            raise ValueError("POV pending action must belong to the controlled actor.")
        if self.latest_transition is not None and (
            self.latest_transition.actor.actor_public_agent_id
            != self.controlled_public_agent_id
        ):
            raise ValueError("POV latest result must belong to the controlled actor.")
        if tuple(row.move_action for row in self.movement_legalities) != tuple(
            range(NUM_MOVE_ACTIONS)
        ):
            raise ValueError("POV movement rows must cover the canonical action axis.")
        if tuple(
            row.target.target_action for row in self.candidate_legalities
        ) != tuple(range(NUM_TARGET_ACTIONS)):
            raise ValueError("POV candidate rows must cover the canonical target axis.")
        return self


class ScenarioOptionV1(_ProtocolModel):
    name: _ScenarioName
    title: str
    description: str
    mode: Literal["interactive", "scripted"]
    audience: Literal["researcher", "stress"]


class ScenarioMetadataV1(ScenarioOptionV1):
    ordinary_movement_distance_scale: _PositiveUnitFloat
    completed_frame_count: _NonNegativeInt
    frame_count: _NonNegativeInt
    next_frame_index: _NonNegativeInt | None
    next_frame_label: str | None
    next_frame_description: str | None
    script_complete: bool

    @model_validator(mode="after")
    def _validate_frame_cursor(self) -> Self:
        if self.ordinary_movement_distance_scale != 1.0:
            raise ValueError("live product movement scale must remain exactly 1.0.")
        if self.completed_frame_count > self.frame_count:
            raise ValueError("completed_frame_count must not exceed frame_count.")
        has_next = self.next_frame_index is not None
        if has_next != (self.next_frame_label is not None):
            raise ValueError("next frame index and label must appear together.")
        if has_next != (self.next_frame_description is not None):
            raise ValueError("next frame index and description must appear together.")
        if self.mode == "interactive":
            if (
                self.frame_count != 0
                or self.completed_frame_count != 0
                or has_next
                or self.script_complete
            ):
                raise ValueError(
                    "interactive scenarios do not expose a scripted frame cursor."
                )
            return self
        if self.script_complete is has_next:
            raise ValueError(
                "script_complete must be true exactly when no next frame exists."
            )
        if (
            self.next_frame_index is not None
            and self.next_frame_index != self.completed_frame_count
        ):
            raise ValueError("next_frame_index must equal completed_frame_count.")
        return self


class TerminalStateV1(_ProtocolModel):
    is_terminal: bool
    terminated: bool
    truncated: bool
    reason: Literal["terminated", "truncated"] | None

    @model_validator(mode="after")
    def _validate_terminal_reason(self) -> Self:
        expected = (
            "terminated" if self.terminated else "truncated" if self.truncated else None
        )
        if self.is_terminal != (expected is not None) or self.reason != expected:
            raise ValueError("terminal flags and reason must describe one state.")
        return self


class TerminalStateV2(_ProtocolModel):
    is_sealed: bool
    terminated: bool
    truncated: bool
    reached_declared_horizon: bool
    reason: Literal["terminated", "truncated", "declared_horizon"] | None

    @model_validator(mode="after")
    def _validate_terminal_reason(self) -> Self:
        expected = (
            "terminated"
            if self.terminated
            else "truncated"
            if self.truncated
            else "declared_horizon"
            if self.reached_declared_horizon
            else None
        )
        if self.is_sealed != (expected is not None) or self.reason != expected:
            raise ValueError("terminal flags and reason must describe one state.")
        return self


class RecordingStatusV1(_ProtocolModel):
    """Path-free live recording lifecycle shared by both audience envelopes."""

    schema_version: Literal[1] = PROTOCOL_SCHEMA_VERSION
    lifecycle: RecordingLifecycleV1
    captured_transition_count: _NonNegativeInt
    expected_transition_count: Annotated[int, Field(gt=0)]
    completion_state: RecordingCompletionStateV1 | None = None
    completion_reason: (
        Annotated[
            str,
            StringConstraints(min_length=1, max_length=256),
        ]
        | None
    ) = None
    restart_fenced: bool
    finish_available: bool
    review_available: bool
    retry_available: bool
    save_as_available: bool
    discard_available: bool
    persistence_error_code: RecordingPersistenceErrorCodeV1 | None = None

    @model_validator(mode="after")
    def _validate_recording_lifecycle(self) -> Self:
        if self.captured_transition_count > self.expected_transition_count:
            raise ValueError("recording progress cannot exceed the declared horizon.")
        finalized = self.lifecycle in (
            "sealed",
            "finalized_unsaved",
            "persistence_failed",
            "saved",
            "reviewing",
        )
        if finalized != (self.completion_state is not None):
            raise ValueError(
                "finalized recording lifecycle requires one completion state."
            )
        reason_required = self.completion_state in (
            "partial",
            "interrupted",
            "failed",
        )
        if reason_required != (self.completion_reason is not None):
            raise ValueError(
                "non-complete recording completion requires one stable reason."
            )
        expected_restart_fenced = (
            self.captured_transition_count > 0 or self.lifecycle != "recording"
        )
        if self.restart_fenced != expected_restart_fenced:
            raise ValueError("restart_fenced must follow exact recorder progress.")
        expected_finish = self.lifecycle == "recording"
        expected_review = self.lifecycle == "saved"
        expected_retry = self.lifecycle == "persistence_failed"
        expected_discard = (
            self.lifecycle == "recording" and self.captured_transition_count > 0
        )
        if (
            self.finish_available != expected_finish
            or self.review_available != expected_review
            or self.retry_available != expected_retry
            or self.save_as_available != expected_retry
            or self.discard_available != expected_discard
        ):
            raise ValueError(
                "recording lifecycle availability flags are not canonical."
            )
        if (self.persistence_error_code is not None) != expected_retry:
            raise ValueError(
                "only persistence_failed may expose a coarse persistence error."
            )
        return self


class _LiveDebuggerEnvelopeV2(_ProtocolModel):
    """Transport metadata shared without sharing audience payload authority."""

    session_id: _OpaqueId
    run_generation: _NonNegativeInt
    revision: _NonNegativeInt
    episode_id: _CanonicalScientificId
    frame_index: _NonNegativeInt
    frame_id: _CanonicalScientificId
    simulator_step_count: _NonNegativeInt
    preset: Preset
    verbose: Literal[False] = False
    terminal: TerminalStateV2
    recording: RecordingStatusV1 | None = None

    @model_validator(mode="after")
    def _validate_common_epoch(self) -> Self:
        if self.frame_id != f"{self.episode_id}:frame:{self.frame_index}":
            raise ValueError("live frame ID must remain canonical.")
        if (
            self.recording is not None
            and self.recording.captured_transition_count != self.frame_index
        ):
            raise ValueError("recording progress must equal the live frame index.")
        return self


class ResearcherLiveDebuggerFrameV2(_LiveDebuggerEnvelopeV2):
    """Researcher-authorized live frame with full canonical evaluation truth."""

    frame_kind: Literal["researcher_live_debugger"] = "researcher_live_debugger"
    schema_version: Literal[2] = LIVE_FRAME_SCHEMA_VERSION
    view_mode: Literal["researcher"] = "researcher"
    incoming_transition_index: _NonNegativeInt | None
    incoming_transition_id: _CanonicalScientificId | None
    show_ranges: bool
    scenario: ScenarioMetadataV1
    available_scenarios: tuple[ScenarioOptionV1, ...]
    projection: ResearcherAnalyzerProjectionV2
    hud: ResearcherHudFrameV2

    @model_validator(mode="after")
    def _validate_researcher_frame(self) -> Self:
        if (self.incoming_transition_id is None) != (
            self.incoming_transition_index is None
        ):
            raise ValueError("incoming transition index and ID must appear together.")
        expected_transition_index = (
            None if self.frame_index == 0 else self.frame_index - 1
        )
        if self.incoming_transition_index != expected_transition_index:
            raise ValueError("incoming transition index must enter this frame.")
        expected_transition_id = (
            None
            if expected_transition_index is None
            else f"{self.episode_id}:transition:{expected_transition_index}"
        )
        if self.incoming_transition_id != expected_transition_id:
            raise ValueError("incoming transition ID must remain canonical.")
        projection = self.projection
        scene = projection.scene
        if (
            scene.episode_id != self.episode_id
            or scene.frame_index != self.frame_index
            or scene.frame_id != self.frame_id
            or scene.simulator_step_count != self.simulator_step_count
            or scene.incoming_transition_id != self.incoming_transition_id
        ):
            raise ValueError("researcher projection must join the live envelope.")
        roster_slots = tuple(agent.global_slot for agent in scene.agents)
        if self.hud.roster_global_slots != roster_slots:
            raise ValueError("researcher HUD roster must match the canonical scene.")
        selection = scene.selection
        if selection is None:
            raise ValueError("live researcher scenes require a selection record.")
        if (
            self.hud.controlled_global_slot != selection.controlled_global_slot
            or self.hud.selected_global_slot != selection.selected_global_slot
        ):
            raise ValueError("researcher HUD and scene selection must agree.")
        if self.hud.controlled_global_slot not in self.hud.roster_global_slots:
            raise ValueError(
                "controlled actor must occur in the authorized HUD roster."
            )
        latest = self.hud.latest_transition
        if latest is not None and (
            latest.transition_index != self.incoming_transition_index
            or latest.transition_id != self.incoming_transition_id
        ):
            raise ValueError("latest-transition HUD identity must join the envelope.")
        return self


class ActorPovLiveDebuggerFrameV2(_LiveDebuggerEnvelopeV2):
    """Recipient-authorized live frame with no researcher projection or HUD."""

    frame_kind: Literal["actor_pov_live_debugger"] = "actor_pov_live_debugger"
    schema_version: Literal[2] = LIVE_FRAME_SCHEMA_VERSION
    view_mode: Literal["pov"] = "pov"
    incoming_pov_transition_id: _CanonicalScientificId | None
    projection: ActorPovAnalyzerProjectionV1
    hud: ActorPovHudFrameV1

    @model_validator(mode="after")
    def _validate_pov_frame(self) -> Self:
        scene = self.projection.scene
        expected_pov_transition_id = (
            None
            if self.frame_index == 0
            else (
                f"{self.episode_id}:actor-pov:"
                f"{scene.self_actor.public_agent_id}:transition:{self.frame_index - 1}"
            )
        )
        if (
            scene.episode_id != self.episode_id
            or scene.frame_index != self.frame_index
            or scene.source_frame_id != self.frame_id
            or scene.simulator_step_count != self.simulator_step_count
            or self.projection.incoming_transition_id != expected_pov_transition_id
            or self.incoming_pov_transition_id != expected_pov_transition_id
            or self.hud.controlled_public_agent_id != scene.self_actor.public_agent_id
        ):
            raise ValueError("POV projection and HUD must join the live envelope.")
        latest = self.hud.latest_transition
        if latest is not None and (
            latest.transition_index != self.frame_index - 1
            or latest.pov_transition_id != self.incoming_pov_transition_id
        ):
            raise ValueError("POV latest-result identity must join the envelope.")
        return self


type LiveDebuggerFrame = ResearcherLiveDebuggerFrameV2 | ActorPovLiveDebuggerFrameV2


class CommandResponseV2(_ProtocolModel):
    schema_version: Literal[2] = 2
    result: CommandResult
    frame: Annotated[LiveDebuggerFrame, Field(discriminator="frame_kind")]
    notice: str | None = None


class ApiErrorV2(_ProtocolModel):
    schema_version: Literal[2] = 2
    error_code: ApiErrorCode
    message: str
    latest_frame: (
        Annotated[LiveDebuggerFrame, Field(discriminator="frame_kind")] | None
    ) = None
