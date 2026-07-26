"""Versioned loopback protocol models for the live visual debugger."""

from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StringConstraints,
    ValidationInfo,
    field_serializer,
    field_validator,
    model_validator,
)

from marl_battlegrounds.core.types import (
    MAX_AGENT_SLOTS,
    NUM_MOVE_ACTIONS,
    NUM_TARGET_ACTIONS,
)
from marl_battlegrounds.rendering.scene import (
    BattlefieldSceneV1,
    TargetDisclosure,
    VisualEventBatchV1,
    to_jsonable,
)
from scripts.dev.visual_debugger.model import (
    MOVEMENT_SCALE_MAXIMUM,
    MOVEMENT_SCALE_MINIMUM,
    MOVEMENT_SCALE_STEP,
)
from scripts.dev.visual_debugger.targeting import global_slot_to_target_action

PROTOCOL_SCHEMA_VERSION = 1
FRAME_SCHEMA_VERSION = 1

type ViewMode = Literal["researcher", "pov"]
type Preset = Literal["presentation", "analysis", "debug"]
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
_MovementScale = Annotated[
    StrictFloat,
    Field(
        ge=MOVEMENT_SCALE_MINIMUM,
        le=MOVEMENT_SCALE_MAXIMUM,
        allow_inf_nan=False,
    ),
]
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


class SetMovementScaleCommandV1(_ProtocolModel):
    command_type: Literal["set_movement_scale"] = "set_movement_scale"
    movement_scale: _MovementScale | None

    @field_validator("movement_scale", mode="before")
    @classmethod
    def _validate_input_number(
        cls,
        value: object,
        info: ValidationInfo,
    ) -> object:
        if value is None:
            return value
        if info.mode == "json" and type(value) is int:
            return float(value)
        if type(value) is not float:
            raise ValueError("movement_scale must be a JSON number or Python float.")
        return value


class ExitCommandV1(_ProtocolModel):
    command_type: Literal["exit"] = "exit"


type DebuggerCommandV1 = Annotated[
    KeyboardCommandV1
    | BattlefieldPointerCommandV1
    | RosterSelectionCommandV1
    | ScenarioSwitchCommandV1
    | ResetCommandV1
    | SetViewCommandV1
    | SetPresetCommandV1
    | SetMovementScaleCommandV1
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
            for candidate in self.candidate_legalities:
                if candidate.target.global_slot is None:
                    continue
                expected_target_action = global_slot_to_target_action(
                    self.controlled_global_slot,
                    candidate.target.global_slot,
                )
                if candidate.target_action != expected_target_action:
                    raise ValueError(
                        "candidate target actions must match the controlled "
                        "actor-relative target mapping."
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


class ScenarioOptionV1(_ProtocolModel):
    name: _ScenarioName
    title: str
    description: str
    mode: Literal["interactive", "scripted"]
    audience: Literal["researcher", "stress"]


class ScenarioMetadataV1(ScenarioOptionV1):
    movement_scale_minimum: _PositiveUnitFloat = MOVEMENT_SCALE_MINIMUM
    movement_scale_maximum: _PositiveUnitFloat = MOVEMENT_SCALE_MAXIMUM
    movement_scale_step: _PositiveUnitFloat = MOVEMENT_SCALE_STEP
    ordinary_movement_distance_scale: _PositiveUnitFloat
    scenario_default_movement_scale: _PositiveUnitFloat
    movement_scale_overridden: bool
    completed_frame_count: _NonNegativeInt
    frame_count: _NonNegativeInt
    next_frame_index: _NonNegativeInt | None
    next_frame_label: str | None
    next_frame_description: str | None
    script_complete: bool

    @model_validator(mode="after")
    def _validate_frame_cursor(self) -> Self:
        if (
            self.movement_scale_minimum,
            self.movement_scale_maximum,
            self.movement_scale_step,
        ) != (
            MOVEMENT_SCALE_MINIMUM,
            MOVEMENT_SCALE_MAXIMUM,
            MOVEMENT_SCALE_STEP,
        ):
            raise ValueError(
                "movement-scale bounds and step must match the authoritative "
                "debugger contract."
            )
        if not (
            self.movement_scale_minimum
            <= self.ordinary_movement_distance_scale
            <= self.movement_scale_maximum
            and self.movement_scale_minimum
            <= self.scenario_default_movement_scale
            <= self.movement_scale_maximum
        ):
            raise ValueError(
                "effective and authored movement scales must remain within "
                "the advertised movement-scale bounds."
            )
        expected_override = (
            self.ordinary_movement_distance_scale
            != self.scenario_default_movement_scale
        )
        if self.movement_scale_overridden != expected_override:
            raise ValueError(
                "movement_scale_overridden must exactly describe effective "
                "versus authored movement scale."
            )
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


class DebuggerFrameV1(_ProtocolModel):
    schema_version: Literal[1] = FRAME_SCHEMA_VERSION
    session_id: _OpaqueId
    run_generation: _NonNegativeInt
    revision: _NonNegativeInt
    simulator_step: _NonNegativeInt
    transition_id: _NonNegativeInt | None
    view_mode: ViewMode
    preset: Preset
    scenario: ScenarioMetadataV1
    available_scenarios: tuple[ScenarioOptionV1, ...]
    terminal: TerminalStateV1
    scene: BattlefieldSceneV1
    event_batch: VisualEventBatchV1 | None
    hud: HudFrameV1

    @field_serializer("scene", when_used="json")
    def _serialize_scene(self, scene: BattlefieldSceneV1) -> object:
        return to_jsonable(scene)

    @field_serializer("event_batch", when_used="json")
    def _serialize_event_batch(
        self,
        event_batch: VisualEventBatchV1 | None,
    ) -> object:
        return None if event_batch is None else to_jsonable(event_batch)

    @model_validator(mode="after")
    def _validate_frame_coherence(self) -> Self:
        expected_audience = (
            "researcher" if self.view_mode == "researcher" else "agent_pov"
        )
        if self.scene.audience != expected_audience:
            raise ValueError("view_mode and scene audience must agree.")
        if (self.transition_id is None) != (self.event_batch is None):
            raise ValueError(
                "transition_id and event_batch must be absent or present together."
            )
        if self.event_batch is not None and (
            self.event_batch.transition_id != self.transition_id
            or self.event_batch.simulator_step != self.simulator_step
        ):
            raise ValueError(
                "event batch identity must match the enclosing debugger frame."
            )
        scene_slots = tuple(agent.global_slot for agent in self.scene.agents)
        if self.hud.roster_global_slots != scene_slots:
            raise ValueError("HUD roster order must match scene agent order.")
        selection = self.scene.selection
        if selection is None:
            raise ValueError("live debugger scenes require a selection record.")
        if (
            self.hud.controlled_global_slot != selection.controlled_global_slot
            or self.hud.selected_global_slot != selection.selected_global_slot
        ):
            raise ValueError("HUD and scene selection records must agree.")
        expected_scope: PendingSubmissionScope = (
            "scripted_playback"
            if self.scenario.mode == "scripted"
            else "joint_turn"
            if self.view_mode == "researcher"
            else "controlled_actor"
        )
        if self.hud.pending_submission_scope != expected_scope:
            raise ValueError(
                "scenario mode, view mode, and pending submission scope must agree."
            )
        if self.hud.latest_transition is not None and (
            self.hud.latest_transition.transition_id != self.transition_id
        ):
            raise ValueError("latest-transition card must match transition_id.")
        if (
            self.view_mode == "pov"
            and self.hud.latest_transition is not None
            and any(
                actor.actor_global_slot != self.hud.controlled_global_slot
                for actor in self.hud.latest_transition.actors
            )
        ):
            raise ValueError(
                "POV latest-transition cards may contain only the controlled actor."
            )
        return self


class CommandResponseV1(_ProtocolModel):
    schema_version: Literal[1] = PROTOCOL_SCHEMA_VERSION
    result: CommandResult
    frame: DebuggerFrameV1
    notice: str | None = None


class ApiErrorV1(_ProtocolModel):
    schema_version: Literal[1] = PROTOCOL_SCHEMA_VERSION
    error_code: ApiErrorCode
    message: str
    latest_frame: DebuggerFrameV1 | None = None
