"""Immutable host-side contracts for the comprehensive visual debugger."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Literal, cast

from marl_battlegrounds.evaluation.wire_shapes import (
    MAX_AGENT_SLOTS_V1 as MAX_AGENT_SLOTS,
)
from marl_battlegrounds.evaluation.wire_shapes import (
    NUM_MOVE_ACTIONS_V1 as NUM_MOVE_ACTIONS,
)
from marl_battlegrounds.evaluation.wire_shapes import (
    NUM_TARGET_ACTIONS_V1 as NUM_TARGET_ACTIONS,
)

if TYPE_CHECKING:
    from jax import Array

    from marl_battlegrounds.core.types import (
        ActionMask,
        EnvConfig,
        EnvState,
        Observation,
    )
    from marl_battlegrounds.evaluation.metrics import EvaluationTransitionViewV1
    from marl_battlegrounds.evaluation.models import (
        EvaluationEpisodeContextV1,
        EvaluationFrameV1,
    )
    from marl_battlegrounds.rendering.scene import StatusSourceEvidenceStateV2

_MOVE_STAY_V1 = 0

type Lane = Literal[0, 1]
type ArmOrigin = Literal["automatic", "explicit"]
type ScenarioMode = Literal["interactive", "scripted"]
type ScenarioAudience = Literal["researcher", "stress"]
type SubmissionKind = Literal["interactive", "scripted"]
type TeamController = Literal["manual", "scripted_tdm", "random_valid"]
type TeamControllerActionSource = Literal["manual", "scripted", "mixed", "policy"]
type ScenarioSourceKind = Literal[
    "current_buffer",
    "saved_draft",
]

SUPPORTED_TEAM_CONTROLLERS: tuple[TeamController, ...] = (
    "manual",
    "scripted_tdm",
    "random_valid",
)


def team_controller_action_source(
    team_a_controller: TeamController,
    team_b_controller: TeamController,
) -> TeamControllerActionSource:
    """Classify one interactive pair without hiding either controller identity."""
    if team_a_controller == team_b_controller == "manual":
        return "manual"
    if team_a_controller == team_b_controller == "scripted_tdm":
        return "scripted"
    if team_a_controller != "manual" and team_b_controller != "manual":
        return "policy"
    return "mixed"


def _validate_slot(global_slot: int, *, name: str) -> None:
    if not 0 <= global_slot < MAX_AGENT_SLOTS:
        msg = f"{name} must be in [0, {MAX_AGENT_SLOTS}); got {global_slot}."
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class PendingAction:
    move_action: int = _MOVE_STAY_V1
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
class ActorCommand:
    actor_global_slot: int
    move_action: int = _MOVE_STAY_V1
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
class DebuggerScenarioProvenance:
    """Exact authored-source identities retained by diagnostic captures."""

    source_kind: ScenarioSourceKind
    source_identity: str
    scenario_semantic_digest: str
    map_semantic_digest: str
    resolved_configuration_digest: str
    resolved_initial_state_digest: str

    def __post_init__(self) -> None:
        if self.source_kind not in ("current_buffer", "saved_draft"):
            raise ValueError("unknown authored scenario source kind.")
        if not self.source_identity:
            raise ValueError("source_identity must be nonempty.")
        for name, value in (
            ("scenario_semantic_digest", self.scenario_semantic_digest),
            ("map_semantic_digest", self.map_semantic_digest),
            ("resolved_configuration_digest", self.resolved_configuration_digest),
            ("resolved_initial_state_digest", self.resolved_initial_state_digest),
        ):
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{name} must be lowercase SHA-256 hex.")


@dataclass(frozen=True, slots=True)
class DebuggerScenario:
    name: str
    title: str
    description: str
    mode: ScenarioMode
    build_scenario: Callable[[], tuple[EnvConfig, EnvState]]
    frames: tuple[ScenarioFrame, ...]
    default_controlled_slot: int
    audience: ScenarioAudience = "researcher"
    provenance: DebuggerScenarioProvenance | None = None

    def __post_init__(self) -> None:
        _validate_slot(self.default_controlled_slot, name="default_controlled_slot")
        if self.mode not in ("interactive", "scripted"):
            msg = f"unknown scenario mode: {self.mode!r}."
            raise ValueError(msg)
        if self.audience not in ("researcher", "stress"):
            msg = f"unknown scenario audience: {self.audience!r}."
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class RawContinuationIdentity:
    """Identity-bind the exact raw objects that feed the next core step."""

    config: EnvConfig
    key: Array
    state: EnvState
    observation: Observation
    action_mask: ActionMask

    def matches(
        self,
        *,
        config: EnvConfig,
        key: Array,
        state: EnvState,
        observation: Observation,
        action_mask: ActionMask,
    ) -> bool:
        """Return whether every raw continuation object is the bound object."""
        return (
            self.config is config
            and self.key is key
            and self.state is state
            and self.observation is observation
            and self.action_mask is action_mask
        )


@dataclass(frozen=True, slots=True)
class DebuggerSession:
    scenario: DebuggerScenario
    seed: int
    run_generation: int
    scenario_default_movement_scale: float
    config: EnvConfig
    key: Array
    state: EnvState
    observation: Observation
    action_mask: ActionMask
    evaluation_context: EvaluationEpisodeContextV1
    current_evaluation_frame: EvaluationFrameV1
    incoming_evaluation_view: EvaluationTransitionViewV1 | None
    status_source_evidence_state: StatusSourceEvidenceStateV2
    last_submission_kind: SubmissionKind | None
    last_report_actor_slots: tuple[int, ...]
    team_a_controller: TeamController
    team_b_controller: TeamController
    controlled_global_slot: int
    pending_actions: tuple[PendingAction, ...]
    next_script_frame_index: int
    show_ranges: bool
    verbose_logging: bool
    raw_continuation_identity: RawContinuationIdentity | None = None

    @property
    def scenario_name(self) -> str:
        """Return the owned scenario's stable display and provenance name."""
        return self.scenario.name

    def __post_init__(self) -> None:
        from marl_battlegrounds.evaluation.metrics import EvaluationTransitionViewV1
        from marl_battlegrounds.evaluation.models import (
            AssignedPolicySlotV1,
            EvaluationEpisodeContextV1,
            EvaluationFrameV1,
        )
        from marl_battlegrounds.rendering.scene import StatusSourceEvidenceStateV2

        if type(self.scenario) is not DebuggerScenario:
            raise TypeError("scenario must be the exact DebuggerScenario type.")
        continuation = self.raw_continuation_identity
        if continuation is None:
            continuation = RawContinuationIdentity(
                config=self.config,
                key=self.key,
                state=self.state,
                observation=self.observation,
                action_mask=self.action_mask,
            )
            object.__setattr__(self, "raw_continuation_identity", continuation)
        elif type(
            continuation
        ) is not RawContinuationIdentity or not continuation.matches(
            config=self.config,
            key=self.key,
            state=self.state,
            observation=self.observation,
            action_mask=self.action_mask,
        ):
            raise ValueError(
                "raw continuation identity must bind the exact next-step objects."
            )
        if type(self.run_generation) is not int or self.run_generation < 0:
            msg = "run_generation must be a non-negative Python int."
            raise ValueError(msg)
        if type(self.evaluation_context) is not EvaluationEpisodeContextV1:
            raise TypeError(
                "evaluation_context must be the exact EvaluationEpisodeContextV1 root."
            )
        if type(self.current_evaluation_frame) is not EvaluationFrameV1:
            raise TypeError(
                "current_evaluation_frame must be the exact EvaluationFrameV1 root."
            )
        if (
            type(self.scenario_default_movement_scale) is not float
            or not isfinite(self.scenario_default_movement_scale)
            or not 0.0 < self.scenario_default_movement_scale <= 1.0
        ):
            msg = (
                "scenario_default_movement_scale must be a finite Python float "
                "in (0.0, 1.0]."
            )
            raise ValueError(msg)
        _validate_slot(self.controlled_global_slot, name="controlled_global_slot")
        roster = self.evaluation_context.roster
        if not roster[self.controlled_global_slot].configured_active:
            msg = f"controlled slot g{self.controlled_global_slot} is inactive."
            raise ValueError(msg)
        if len(self.pending_actions) != MAX_AGENT_SLOTS:
            msg = (
                f"pending_actions must contain {MAX_AGENT_SLOTS} fixed-slot rows; "
                f"got {len(self.pending_actions)}."
            )
            raise ValueError(msg)
        inactive_pending = PendingAction(armed_lane=None, arm_origin=None)
        runtime_pending_actions = cast(tuple[object, ...], self.pending_actions)
        for actor_slot, pending in enumerate(runtime_pending_actions):
            if not isinstance(pending, PendingAction):
                msg = f"pending_actions[{actor_slot}] must be a PendingAction."
                raise TypeError(msg)
            actor_active = roster[actor_slot].configured_active
            if not actor_active and pending != inactive_pending:
                msg = f"inactive pending row g{actor_slot} must remain neutral."
                raise ValueError(msg)
            target_slot = pending.selected_global_target_slot
            if target_slot is not None and not roster[target_slot].configured_active:
                msg = f"pending target g{target_slot} is inactive."
                raise ValueError(msg)
        if self.next_script_frame_index < 0:
            msg = "next_script_frame_index must be non-negative."
            raise ValueError(msg)
        episode_id = self.evaluation_context.identity.episode_id
        current_frame = self.current_evaluation_frame
        if current_frame.episode_id != episode_id:
            raise ValueError("current evaluation frame must join the session episode.")
        if current_frame.frame_index > self.evaluation_context.expected_horizon:
            raise ValueError(
                "current evaluation frame cannot exceed the declared horizon."
            )
        incoming = self.incoming_evaluation_view
        if incoming is None:
            if current_frame.frame_index != 0:
                raise ValueError(
                    "only evaluation frame zero may omit an incoming transition."
                )
            if self.last_submission_kind is not None or self.last_report_actor_slots:
                raise ValueError(
                    "initial sessions cannot retain transition submission metadata."
                )
        else:
            if type(incoming) is not EvaluationTransitionViewV1:
                raise TypeError(
                    "incoming_evaluation_view must be an exact coherent V1 view."
                )
            if (
                incoming.context != self.evaluation_context
                or incoming.successor_frame != current_frame
                or incoming.transition.transition_index != current_frame.frame_index - 1
            ):
                raise ValueError(
                    "incoming evaluation view must enter the current session frame."
                )
            if self.last_submission_kind not in ("interactive", "scripted"):
                raise ValueError(
                    "non-initial sessions require exact submission-kind metadata."
                )
        if type(self.last_report_actor_slots) is not tuple:
            raise TypeError("last_report_actor_slots must be a Python tuple.")
        if self.last_report_actor_slots != tuple(
            sorted(set(self.last_report_actor_slots))
        ):
            raise ValueError("last_report_actor_slots must be sorted and unique.")
        for actor_slot in self.last_report_actor_slots:
            _validate_slot(actor_slot, name="last_report_actor_slot")
            if not self.evaluation_context.roster[actor_slot].configured_active:
                raise ValueError("last report actor slots must be configured active.")
        if self.team_a_controller not in SUPPORTED_TEAM_CONTROLLERS:
            raise ValueError(
                "team_a_controller must be manual, scripted_tdm, or random_valid."
            )
        if self.team_b_controller not in SUPPORTED_TEAM_CONTROLLERS:
            raise ValueError(
                "team_b_controller must be manual, scripted_tdm, or random_valid."
            )
        if self.scenario.mode == "scripted":
            if self.team_a_controller != "manual" or self.team_b_controller != "manual":
                raise ValueError(
                    "registered scripted scenarios do not use interactive team "
                    "controllers."
                )
        else:
            expected_action_source = team_controller_action_source(
                self.team_a_controller,
                self.team_b_controller,
            )
            expected_aggregation = {
                "action_source": expected_action_source,
                "team_a_controller": self.team_a_controller,
                "team_b_controller": self.team_b_controller,
            }
            for name, expected in expected_aggregation.items():
                values = tuple(
                    row.value
                    for row in self.evaluation_context.aggregation_keys
                    if row.name == name
                )
                if values != (expected,):
                    raise ValueError(
                        f"session {name} must join its evaluation context."
                    )
            team_a_id = roster[0].configured_team_id
            for slot, roster_row in enumerate(roster):
                if not roster_row.configured_active:
                    continue
                assignment = self.evaluation_context.policy_assignments[slot]
                expected_policy_kind = (
                    self.team_a_controller
                    if roster_row.configured_team_id == team_a_id
                    else self.team_b_controller
                )
                if (
                    not isinstance(assignment, AssignedPolicySlotV1)
                    or assignment.policy_kind != expected_policy_kind
                ):
                    raise ValueError(
                        "session team controllers must join every active policy "
                        "assignment."
                    )
        evidence_state = self.status_source_evidence_state
        if type(evidence_state) is not StatusSourceEvidenceStateV2:
            raise TypeError("status_source_evidence_state must be the exact V2 state.")
        if (
            evidence_state.episode_id != episode_id
            or evidence_state.frame_index != current_frame.frame_index
            or evidence_state.frame_id != current_frame.frame_id
        ):
            raise ValueError(
                "status-source evidence must join the current evaluation frame."
            )

    @property
    def pending_action(self) -> PendingAction:
        """Return the currently controlled actor's authoritative draft row."""
        return self.pending_actions[self.controlled_global_slot]

    @property
    def terminated(self) -> bool:
        """Return canonical task-termination truth for the current frame."""
        incoming = self.incoming_evaluation_view
        return incoming is not None and incoming.transition.terminated

    @property
    def truncated(self) -> bool:
        """Return canonical truncation truth for the current frame."""
        incoming = self.incoming_evaluation_view
        return incoming is not None and incoming.transition.truncated

    @property
    def reached_declared_horizon(self) -> bool:
        """Return whether the canonical episode prefix reached its exact horizon."""
        return (
            self.current_evaluation_frame.frame_index
            == self.evaluation_context.expected_horizon
        )

    @property
    def episode_sealed(self) -> bool:
        """Return whether another simulator transition is scientifically invalid."""
        return self.terminated or self.truncated or self.reached_declared_horizon
