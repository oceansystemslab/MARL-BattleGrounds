"""Immutable host-side contracts for the comprehensive visual debugger."""

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from typing import Literal

from jax import Array

from marl_battlegrounds.core.types import (
    MAX_AGENT_SLOTS,
    MOVE_STAY,
    NUM_MOVE_ACTIONS,
    NUM_TARGET_ACTIONS,
    Action,
    ActionMask,
    DoneFlags,
    EnvConfig,
    EnvState,
    Info,
    Observation,
    Reward,
)
from marl_battlegrounds.rendering import (
    ActivationVisual,
    ChargeTrailVisual,
    HealthDeltaVisual,
    RejectedActionVisual,
)
from marl_battlegrounds.rendering.visuals import ActivationKind, RejectionComponent

type Lane = Literal[0, 1]
type ArmOrigin = Literal["automatic", "explicit"]
type Relation = Literal["self", "ally", "enemy"]
type ScenarioMode = Literal["interactive", "scripted"]
type SubmissionKind = Literal["interactive", "scripted"]
type StatusKind = Literal[
    "slow_warrior_charge",
    "slow_hunter_basic",
    "slow_rogue_poison",
    "stun_warrior_charge",
    "stun_hunter_trap",
    "stun_rogue_poison",
    "anti_heal_rogue_poison",
    "mage_burst",
    "priest_freedom",
]
type StatusChange = Literal[
    "applied",
    "refreshed",
    "decremented",
    "expired",
    "trap_broken",
    "cleared_unclassified",
    "unchanged",
]
type TransientVisual = (
    HealthDeltaVisual | ActivationVisual | ChargeTrailVisual | RejectedActionVisual
)


def _validate_slot(global_slot: int, *, name: str) -> None:
    if not 0 <= global_slot < MAX_AGENT_SLOTS:
        msg = f"{name} must be in [0, {MAX_AGENT_SLOTS}); got {global_slot}."
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class PendingAction:
    move_action: int = MOVE_STAY
    selected_global_target_slot: int | None = None
    armed_lane: Lane | None = 0
    arm_origin: ArmOrigin | None = "automatic"

    def __post_init__(self) -> None:
        if not 0 <= self.move_action < NUM_MOVE_ACTIONS:
            msg = (
                f"move_action must be in [0, {NUM_MOVE_ACTIONS}); "
                f"got {self.move_action}."
            )
            raise ValueError(msg)
        if self.selected_global_target_slot is not None:
            _validate_slot(
                self.selected_global_target_slot,
                name="selected_global_target_slot",
            )
        if self.armed_lane not in (None, 0, 1):
            msg = f"armed_lane must be None, 0, or 1; got {self.armed_lane}."
            raise ValueError(msg)
        if self.arm_origin not in (None, "automatic", "explicit"):
            msg = f"unknown arm_origin: {self.arm_origin!r}."
            raise ValueError(msg)
        if (self.armed_lane is None) != (self.arm_origin is None):
            msg = "arm_origin must be None exactly when armed_lane is None."
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class LaneAvailability:
    target_action: int
    lane_0_available: bool
    lane_1_available: bool
    armed_lane: Lane | None
    armed_pair_legal: bool

    def __post_init__(self) -> None:
        if not 0 <= self.target_action < NUM_TARGET_ACTIONS:
            msg = f"invalid target_action: {self.target_action}."
            raise ValueError(msg)
        if self.armed_lane not in (None, 0, 1):
            msg = f"armed_lane must be None, 0, or 1; got {self.armed_lane}."
            raise ValueError(msg)
        expected = (
            False
            if self.armed_lane is None
            else self.lane_0_available
            if self.armed_lane == 0
            else self.lane_1_available
        )
        if self.armed_pair_legal != expected:
            msg = "armed_pair_legal must match the selected exact lane value."
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class HudSection:
    """One deterministic side-panel section with an explicit typography role."""

    heading: str
    lines: tuple[str, ...]
    technical: bool = False

    def __post_init__(self) -> None:
        if not self.heading.strip():
            raise ValueError("HUD section heading must not be empty.")
        if not self.lines:
            raise ValueError("HUD section must contain at least one line.")
        if any(not line.strip() for line in self.lines):
            raise ValueError("HUD section lines must not be empty.")


@dataclass(frozen=True, slots=True)
class SelectedTargetFacts:
    controlled_global_slot: int
    target_global_slot: int
    target_action: int
    relation: Relation
    center_distance: float
    has_clear_line_of_sight: bool
    observer_visible: bool
    inside_observation_radius: bool
    inside_basic_radius: bool
    inside_ultimate_radius: bool | None
    lane_0_available: bool
    lane_1_available: bool

    def __post_init__(self) -> None:
        _validate_slot(self.controlled_global_slot, name="controlled_global_slot")
        _validate_slot(self.target_global_slot, name="target_global_slot")
        if not 1 <= self.target_action < NUM_TARGET_ACTIONS:
            msg = f"selected target_action must be non-none; got {self.target_action}."
            raise ValueError(msg)
        if self.relation not in ("self", "ally", "enemy"):
            msg = f"unknown target relation: {self.relation!r}."
            raise ValueError(msg)
        if not isfinite(self.center_distance) or self.center_distance < 0:
            msg = (
                "center_distance must be finite and non-negative; "
                f"got {self.center_distance}."
            )
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ActorCommand:
    actor_global_slot: int
    move_action: int = MOVE_STAY
    target_global_slot: int | None = None
    use_ultimate: int = 0

    def __post_init__(self) -> None:
        _validate_slot(self.actor_global_slot, name="actor_global_slot")
        if not 0 <= self.move_action < NUM_MOVE_ACTIONS:
            msg = (
                f"move_action must be in [0, {NUM_MOVE_ACTIONS}); "
                f"got {self.move_action}."
            )
            raise ValueError(msg)
        if self.target_global_slot is not None:
            _validate_slot(self.target_global_slot, name="target_global_slot")
        if self.use_ultimate not in (0, 1):
            msg = f"use_ultimate must be 0 or 1; got {self.use_ultimate}."
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ScenarioFrame:
    label: str
    description: str
    commands: tuple[ActorCommand, ...]

    def __post_init__(self) -> None:
        actor_slots = tuple(command.actor_global_slot for command in self.commands)
        if len(actor_slots) != len(set(actor_slots)):
            msg = f"scenario frame {self.label!r} contains duplicate actor commands."
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class DebuggerScenario:
    name: str
    title: str
    description: str
    mode: ScenarioMode
    build_config: Callable[[], EnvConfig]
    frames: tuple[ScenarioFrame, ...]
    default_controlled_slot: int

    def __post_init__(self) -> None:
        _validate_slot(self.default_controlled_slot, name="default_controlled_slot")
        if self.mode not in ("interactive", "scripted"):
            msg = f"unknown scenario mode: {self.mode!r}."
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ActorTransition:
    actor_global_slot: int
    submitted_move_action: int
    submitted_target_action: int
    submitted_use_ultimate: int
    accepted_move_action: int
    accepted_target_action: int
    accepted_use_ultimate: int
    submitted_tuple_in_domain: bool
    submitted_move_mask_value: bool
    submitted_lane_0_value: bool
    submitted_lane_1_value: bool
    submitted_pair_mask_value: bool
    movement_accepted: bool
    combat_pair_accepted: bool
    position_before: tuple[float, float]
    position_after: tuple[float, float]
    realized_displacement: tuple[float, float]
    health_before: float
    health_after: float
    net_health_delta: float
    cooldown_before: int
    cooldown_after: int
    effective_speed_before: float
    effective_speed_after: float
    mage_aura_before: float
    mage_aura_after: float
    warrior_aura_before: float
    warrior_aura_after: float


@dataclass(frozen=True, slots=True)
class StatusTransition:
    global_slot: int
    status_kind: StatusKind
    source_class_id: int
    duration_before: int
    duration_after: int
    change: StatusChange


@dataclass(frozen=True, slots=True)
class AcceptedActivation:
    kind: ActivationKind
    source_global_slot: int
    target_global_slot: int | None
    target_action: int
    use_ultimate: int


@dataclass(frozen=True, slots=True)
class ActionRejection:
    actor_global_slot: int
    component: RejectionComponent
    submitted_move_action: int
    submitted_target_action: int
    submitted_use_ultimate: int
    movement_mask_value: bool
    pair_mask_value: bool


@dataclass(frozen=True, slots=True)
class TransitionView:
    scenario_name: str
    submission_kind: SubmissionKind
    report_actor_slots: tuple[int, ...]
    before_state: EnvState
    before_observation: Observation
    before_action_mask: ActionMask
    submitted_action: Action
    accepted_action: Action
    after_state: EnvState
    after_observation: Observation
    after_action_mask: ActionMask
    reward: Reward
    done_flags: DoneFlags
    info: Info
    actor_transitions: tuple[ActorTransition, ...]
    status_transitions: tuple[StatusTransition, ...]
    accepted_activations: tuple[AcceptedActivation, ...]
    rejections: tuple[ActionRejection, ...]


@dataclass(frozen=True, slots=True)
class TransientHistoryEntry:
    visual: TransientVisual
    created_after_step: int
    age_submitted_steps: int
    max_age_submitted_steps: int
    sequence_number: int

    def __post_init__(self) -> None:
        if self.created_after_step < 0:
            msg = (
                "created_after_step must be non-negative; "
                f"got {self.created_after_step}."
            )
            raise ValueError(msg)
        if not 0 <= self.age_submitted_steps < self.max_age_submitted_steps:
            msg = (
                "transient age must satisfy 0 <= age < max_age; "
                f"got age={self.age_submitted_steps}, "
                f"max_age={self.max_age_submitted_steps}."
            )
            raise ValueError(msg)
        if self.sequence_number < 0:
            msg = f"sequence_number must be non-negative; got {self.sequence_number}."
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class DebuggerSession:
    scenario_name: str
    seed: int
    config: EnvConfig
    key: Array
    state: EnvState
    observation: Observation
    action_mask: ActionMask
    last_reward: Reward | None
    done_flags: DoneFlags
    info: Info
    controlled_global_slot: int
    pending_action: PendingAction
    next_script_frame_index: int
    last_transition: TransitionView | None
    transient_history: tuple[TransientHistoryEntry, ...]
    next_transient_sequence_number: int
    show_ranges: bool
    verbose_logging: bool

    def __post_init__(self) -> None:
        _validate_slot(self.controlled_global_slot, name="controlled_global_slot")
        if not bool(self.config.agent_profile.active_mask[self.controlled_global_slot]):
            msg = f"controlled slot g{self.controlled_global_slot} is inactive."
            raise ValueError(msg)
        target_slot = self.pending_action.selected_global_target_slot
        if target_slot is not None and not bool(
            self.config.agent_profile.active_mask[target_slot]
        ):
            msg = f"pending target g{target_slot} is inactive."
            raise ValueError(msg)
        if self.next_script_frame_index < 0:
            msg = "next_script_frame_index must be non-negative."
            raise ValueError(msg)
        if self.next_transient_sequence_number < 0:
            msg = "next_transient_sequence_number must be non-negative."
            raise ValueError(msg)
