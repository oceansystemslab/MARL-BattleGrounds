"""Pure debugger state transitions and the single simulator submission boundary."""

from dataclasses import replace
from typing import Literal, cast

import jax
import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.config import validate_product_env_config
from marl_battlegrounds.core.env import initialize_scenario_state, step
from marl_battlegrounds.core.types import (
    MAGE_CLASS_ID,
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    MOVE_STAY,
    NUM_MOVE_ACTIONS,
    NUM_TARGET_ACTIONS,
    TEAM_A_ID,
    TEAM_B_ID,
    Action,
    ActionMask,
    EnvConfig,
    EnvState,
    Observation,
)
from marl_battlegrounds.evaluation.capture import (
    capture_evaluation_transition_unit_v1,
    capture_initial_evaluation_frame_v1,
)
from marl_battlegrounds.evaluation.metrics import EvaluationTransitionViewV1
from marl_battlegrounds.evaluation.models import (
    ActionMaskV1,
    EvaluationEpisodeContextV1,
    ExecutionInformationMode,
)
from marl_battlegrounds.policies.actor import (
    ActorAction,
    build_joint_action_from_actor_actions,
)
from marl_battlegrounds.policies.no_shared_obs import (
    execute_no_shared_obs_team_policy,
)
from marl_battlegrounds.policies.random_valid import random_policy
from marl_battlegrounds.policies.scripted.no_shared_obs import (
    team_deathmatch_no_shared_obs_policy,
)
from marl_battlegrounds.policies.scripted.shared_obs import (
    team_deathmatch_shared_obs_policy,
)
from marl_battlegrounds.policies.shared_obs import (
    SharedObsSensorSourceBankV1,
    build_default_shared_obs_information_availability,
    build_shared_obs_sensor_source_bank,
    execute_shared_obs_team_policy,
)
from marl_battlegrounds.rendering.evaluation_adapter import (
    advance_status_source_evidence_v2,
    initialize_status_source_evidence_v2,
)
from scripts.dev.visual_debugger.evaluation_bridge import (
    DebuggerCaptureProfileV1,
    DebuggerEvaluationLaunchSpecificationV1,
    build_debugger_evaluation_context_v1,
    build_debugger_evaluation_launch_specification_v1,
    debugger_action_source_kind_v1,
)
from scripts.dev.visual_debugger.model import (
    SUPPORTED_TEAM_CONTROLLERS,
    DebuggerScenario,
    DebuggerSession,
    Lane,
    LaneAvailability,
    PendingAction,
    ScenarioFrame,
    SubmissionKind,
    TeamController,
)

type DebuggerTransitionFailureStageV1 = Literal[
    "action_build",
    "simulation",
    "capture",
    "validation",
]
type DebuggerTransitionFailureCodeV1 = Literal[
    "interactive_action_build_failed",
    "policy_action_build_failed",
    "scripted_action_build_failed",
    "invalid_submitted_action",
    "simulator_step_failed",
    "transition_capture_failed",
    "transition_packaging_failed",
]


class DebuggerTransitionFailureV1(RuntimeError):  # noqa: N818 - frozen protocol name
    """Stable submission-stage failure without leaking raw exception detail."""

    __slots__ = ("stable_code", "stage")

    stage: DebuggerTransitionFailureStageV1
    stable_code: DebuggerTransitionFailureCodeV1

    def __init__(
        self,
        stage: DebuggerTransitionFailureStageV1,
        stable_code: DebuggerTransitionFailureCodeV1,
    ) -> None:
        if stage not in ("action_build", "simulation", "capture", "validation"):
            raise ValueError("unknown debugger transition failure stage")
        if stable_code not in (
            "interactive_action_build_failed",
            "policy_action_build_failed",
            "scripted_action_build_failed",
            "invalid_submitted_action",
            "simulator_step_failed",
            "transition_capture_failed",
            "transition_packaging_failed",
        ):
            raise ValueError("unknown debugger transition failure code")
        self.stage = stage
        self.stable_code = stable_code
        super().__init__(f"debugger transition failed during {stage}")


def _transition_failure(
    stage: DebuggerTransitionFailureStageV1,
    stable_code: DebuggerTransitionFailureCodeV1,
    error: Exception,
) -> DebuggerTransitionFailureV1:
    del error
    return DebuggerTransitionFailureV1(stage, stable_code)


def make_neutral_joint_action() -> Action:
    """Return the canonical fixed-shape neutral joint action."""
    return Action(
        move=jnp.full((MAX_AGENT_SLOTS,), MOVE_STAY, dtype=jnp.int32),
        select_target=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
        use_ultimate=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
    )


def _validate_active_context_slot(
    context: EvaluationEpisodeContextV1,
    global_slot: int,
    *,
    name: str,
) -> None:
    if not 0 <= global_slot < MAX_AGENT_SLOTS:
        raise ValueError(f"{name} must be in [0, {MAX_AGENT_SLOTS}).")
    if not context.roster[global_slot].configured_active:
        raise ValueError(f"{name} g{global_slot} is inactive.")


def _active_context_slots(session: DebuggerSession) -> tuple[int, ...]:
    """Return active slots from the already-host canonical roster."""
    return tuple(
        row.global_slot
        for row in session.evaluation_context.roster
        if row.configured_active
    )


def _target_action_from_context(
    context: EvaluationEpisodeContextV1,
    actor_global_slot: int,
    target_global_slot: int | None,
) -> int:
    """Resolve one target category through the serialized catalog authority."""
    if target_global_slot is None:
        return 0
    catalog = context.static_mechanics_catalog
    mapping = catalog.global_recipient_slot_by_actor_and_target_action[
        actor_global_slot
    ]
    try:
        return mapping.index(target_global_slot)
    except ValueError as error:
        raise ValueError("target is absent from the serialized action axis") from error


def lane_availability(
    action_mask: ActionMask | ActionMaskV1,
    actor_global_slot: int,
    target_action: int,
    armed_lane: Lane | None,
) -> LaneAvailability:
    """Read exact lane availability from the authoritative joint mask."""
    if not 0 <= actor_global_slot < MAX_AGENT_SLOTS:
        msg = f"actor_global_slot must be in [0, {MAX_AGENT_SLOTS})."
        raise ValueError(msg)
    if not 0 <= target_action < NUM_TARGET_ACTIONS:
        msg = f"target_action must be in [0, {NUM_TARGET_ACTIONS})."
        raise ValueError(msg)
    if armed_lane not in (None, 0, 1):
        msg = f"armed_lane must be None, 0, or 1; got {armed_lane}."
        raise ValueError(msg)
    if type(action_mask) is ActionMaskV1:
        lane_values = action_mask.select_target_use_ultimate_joint_mask[
            actor_global_slot
        ][target_action]
    else:
        core_mask = cast(ActionMask, action_mask)
        lane_values = core_mask.select_target_use_ultimate_joint_mask[
            actor_global_slot,
            target_action,
        ]
    lane_0_available = bool(lane_values[0])
    lane_1_available = bool(lane_values[1])
    armed_pair_legal = (
        False
        if armed_lane is None
        else lane_0_available
        if armed_lane == 0
        else lane_1_available
    )
    return LaneAvailability(
        target_action=target_action,
        lane_0_available=lane_0_available,
        lane_1_available=lane_1_available,
        armed_lane=armed_lane,
        armed_pair_legal=armed_pair_legal,
    )


def _default_pending_actions(
    context: EvaluationEpisodeContextV1,
    action_mask: ActionMask | ActionMaskV1,
) -> tuple[PendingAction, ...]:
    """Build one exact fixed-slot draft tuple for a fresh decision epoch."""
    pending_actions: list[PendingAction] = []
    for actor_slot in range(MAX_AGENT_SLOTS):
        if not context.roster[actor_slot].configured_active:
            pending_actions.append(PendingAction(armed_lane=None, arm_origin=None))
            continue
        basic_available = lane_availability(
            action_mask,
            actor_slot,
            0,
            0,
        ).lane_0_available
        pending_actions.append(
            PendingAction(
                armed_lane=0 if basic_available else None,
                arm_origin="automatic" if basic_available else None,
            )
        )
    return tuple(pending_actions)


def _replace_controlled_pending_action(
    session: DebuggerSession,
    pending_action: PendingAction,
) -> DebuggerSession:
    """Replace only the controlled actor's row in the immutable draft tuple."""
    controlled_row = session.evaluation_context.roster[session.controlled_global_slot]
    controller = (
        session.team_a_controller
        if controlled_row.configured_team_id == TEAM_A_ID
        else session.team_b_controller
    )
    if controller != "manual":
        return session
    pending_actions = list(session.pending_actions)
    pending_actions[session.controlled_global_slot] = pending_action
    return replace(session, pending_actions=tuple(pending_actions))


def build_interactive_joint_action(
    context: EvaluationEpisodeContextV1,
    pending_actions: tuple[PendingAction, ...],
    *,
    actor_global_slots: tuple[int, ...],
) -> Action:
    """Build one authorized joint request without pre-filtering it by the mask."""
    if len(pending_actions) != MAX_AGENT_SLOTS:
        msg = (
            f"pending_actions must contain {MAX_AGENT_SLOTS} fixed-slot rows; "
            f"got {len(pending_actions)}."
        )
        raise ValueError(msg)
    if not actor_global_slots:
        raise ValueError("actor_global_slots must contain at least one active actor.")
    if len(actor_global_slots) != len(set(actor_global_slots)):
        raise ValueError("actor_global_slots must not contain duplicates.")

    action = make_neutral_joint_action()
    move = action.move
    target = action.select_target
    ultimate = action.use_ultimate
    for actor_slot in actor_global_slots:
        _validate_active_context_slot(
            context,
            actor_slot,
            name="submission actor",
        )
        pending_action = pending_actions[actor_slot]
        move = move.at[actor_slot].set(pending_action.move_action)
        if pending_action.armed_lane is None:
            continue
        if pending_action.selected_global_target_slot is not None:
            _validate_active_context_slot(
                context,
                pending_action.selected_global_target_slot,
                name="pending target",
            )
        target_action = _target_action_from_context(
            context,
            actor_slot,
            pending_action.selected_global_target_slot,
        )
        target = target.at[actor_slot].set(target_action)
        ultimate = ultimate.at[actor_slot].set(pending_action.armed_lane)
    return Action(move=move, select_target=target, use_ultimate=ultimate)


def _team_actor_action(action: Action, team_identity: int) -> ActorAction:
    """Project one fixed team block from an already-built manual action."""
    if team_identity not in (TEAM_A_ID, TEAM_B_ID):
        raise ValueError("team_identity must be Team A or Team B")
    start = 0 if team_identity == TEAM_A_ID else MAX_AGENTS_PER_TEAM
    stop = start + MAX_AGENTS_PER_TEAM
    return ActorAction(
        move=action.move[start:stop],
        select_target=action.select_target[start:stop],
        use_ultimate=action.use_ultimate[start:stop],
    )


def _policy_keys(session: DebuggerSession) -> Array:
    """Derive role-correct actor keys without consuming the environment stream."""
    context = session.evaluation_context
    seeds = context.seed_protocol
    frame_index = session.current_evaluation_frame.frame_index

    def keys_for_seed(seed: object, *, role: str) -> Array:
        if type(seed) is not int:
            raise ValueError(f"{role} actors require a policy seed")
        decision_key = jax.random.fold_in(jax.random.key(seed), frame_index)
        return jax.random.split(decision_key, num=MAX_AGENT_SLOTS)

    focal_slot = session.scenario.default_controlled_slot
    focal_team_id = context.roster[focal_slot].configured_team_id
    focal_keys = keys_for_seed(seeds.focal_policy_seed, role="focal")
    actor_keys = focal_keys
    cooperative_keys: Array | None = None
    adversarial_keys: Array | None = None
    for row in context.roster:
        if not row.configured_active or row.global_slot == focal_slot:
            continue
        if row.configured_team_id == focal_team_id:
            if cooperative_keys is None:
                cooperative_keys = keys_for_seed(
                    seeds.cooperative_partner_seed,
                    role="cooperative",
                )
            role_keys = cooperative_keys
        else:
            if adversarial_keys is None:
                adversarial_keys = keys_for_seed(
                    seeds.adversarial_opponent_seed,
                    role="adversarial",
                )
            role_keys = adversarial_keys
        actor_keys = actor_keys.at[row.global_slot].set(role_keys[row.global_slot])
    return actor_keys


def _random_shared_obs_policy(
    recipient_observation: Observation,
    recipient_action_mask: ActionMask,
    key: Array,
    source_bank: SharedObsSensorSourceBankV1,
    recipient_source_availability: Array,
    recipient_global_slot: Array,
) -> ActorAction:
    """Expose Random through the SharedObs ABI without changing its semantics."""
    del source_bank, recipient_source_availability, recipient_global_slot
    return random_policy(recipient_observation, recipient_action_mask, key)


def _resolve_team_controller_action(
    session: DebuggerSession,
    *,
    team_identity: int,
    controller: TeamController,
    policy_keys: Array,
    source_bank: SharedObsSensorSourceBankV1 | None,
    information_availability: Array | None,
) -> ActorAction:
    """Resolve one explicit team controller from the current decision epoch."""
    if controller == "manual":
        team_slots = tuple(
            row.global_slot
            for row in session.evaluation_context.roster
            if row.configured_active and row.configured_team_id == team_identity
        )
        manual_action = build_interactive_joint_action(
            session.evaluation_context,
            session.pending_actions,
            actor_global_slots=team_slots,
        )
        return _team_actor_action(manual_action, team_identity)

    mode = session.evaluation_context.execution_information_mode
    if mode == "shared_obs":
        if source_bank is None or information_availability is None:
            raise ValueError("SharedObs policy execution requires its source inputs")
        if controller == "scripted_tdm":
            policy = team_deathmatch_shared_obs_policy
        elif controller == "random_valid":
            policy = _random_shared_obs_policy
        else:
            raise ValueError("unsupported team controller")
        return cast(
            ActorAction,
            execute_shared_obs_team_policy(
                session.observation,
                session.action_mask,
                policy_keys,
                source_bank,
                information_availability,
                policy=policy,
                team_identity=team_identity,
            ),
        )
    if mode != "no_shared_obs":
        raise ValueError("unsupported execution information mode")
    if controller == "scripted_tdm":
        policy = team_deathmatch_no_shared_obs_policy
    elif controller == "random_valid":
        policy = random_policy
    else:
        raise ValueError("unsupported team controller")
    return cast(
        ActorAction,
        execute_no_shared_obs_team_policy(
            session.observation,
            session.action_mask,
            policy_keys,
            policy=policy,
            team_identity=team_identity,
        ),
    )


def _build_configured_joint_action(session: DebuggerSession) -> Action:
    """Resolve both team controllers from one current decision epoch."""
    controllers: tuple[tuple[int, TeamController], ...] = (
        (TEAM_A_ID, session.team_a_controller),
        (TEAM_B_ID, session.team_b_controller),
    )
    if all(controller == "manual" for _team_id, controller in controllers):
        raise ValueError("configured policy assembly requires one policy team")

    mode = session.evaluation_context.execution_information_mode
    information_availability: Array | None = None
    source_bank = None
    if mode == "shared_obs":
        information_availability = _captured_information_availability(session)
        assert information_availability is not None
        source_bank = build_shared_obs_sensor_source_bank(session.observation)
    elif mode == "no_shared_obs":
        if _captured_information_availability(session) is not None:
            raise ValueError("NoSharedObs must not expose SharedObs availability.")
    else:
        raise ValueError("unsupported execution information mode")

    policy_keys = _policy_keys(session)
    team_actions = [
        _resolve_team_controller_action(
            session,
            team_identity=team_identity,
            controller=controller,
            policy_keys=policy_keys,
            source_bank=source_bank,
            information_availability=information_availability,
        )
        for team_identity, controller in controllers
    ]
    return build_joint_action_from_actor_actions(
        team_actions[0],
        team_actions[1],
    )


def build_scripted_joint_action(
    context: EvaluationEpisodeContextV1,
    frame: ScenarioFrame,
) -> Action:
    """Build a potentially multi-actor scripted request from neutral defaults."""
    action = make_neutral_joint_action()
    move = action.move
    target = action.select_target
    ultimate = action.use_ultimate
    seen_slots: set[int] = set()
    for command in frame.commands:
        if command.actor_global_slot in seen_slots:
            msg = (
                f"frame {frame.label!r} contains duplicate command for "
                f"g{command.actor_global_slot}."
            )
            raise ValueError(msg)
        seen_slots.add(command.actor_global_slot)
        _validate_active_context_slot(
            context,
            command.actor_global_slot,
            name="command actor",
        )
        if command.target_global_slot is not None:
            _validate_active_context_slot(
                context,
                command.target_global_slot,
                name="command target",
            )
        move = move.at[command.actor_global_slot].set(command.move_action)
        target = target.at[command.actor_global_slot].set(
            _target_action_from_context(
                context,
                command.actor_global_slot,
                command.target_global_slot,
            )
        )
        ultimate = ultimate.at[command.actor_global_slot].set(command.use_ultimate)
    return Action(move=move, select_target=target, use_ultimate=ultimate)


def _fresh_snapshot(
    scenario: DebuggerScenario,
    seed: int,
) -> tuple[float, EnvConfig, EnvState, Observation, ActionMask]:
    """Validate and expose an authored scenario without replacing its state."""
    authored_config, authored_state = scenario.build_scenario()
    validate_product_env_config(authored_config)
    scenario_default_movement_scale = authored_config.ordinary_movement_distance_scale
    config = authored_config
    del seed
    state, observation, action_mask, _info = initialize_scenario_state(
        authored_state,
        config,
    )
    return (
        scenario_default_movement_scale,
        config,
        state,
        observation,
        action_mask,
    )


def _initial_information_availability(
    config: EnvConfig,
    execution_information_mode: ExecutionInformationMode,
) -> Array | None:
    """Build the episode-wide SharedObs topology exactly once per fresh epoch."""
    if execution_information_mode == "shared_obs":
        return build_default_shared_obs_information_availability(
            config.agent_profile.active_mask,
            config.agent_profile.team_ids,
        )
    if execution_information_mode == "no_shared_obs":
        return None
    raise ValueError("execution_information_mode must be shared_obs or no_shared_obs")


def _captured_information_availability(session: DebuggerSession) -> Array | None:
    """Return the exact matrix recorded on the current policy-input frame."""
    captured = session.current_evaluation_frame.shared_obs_information_availability_by_recipient_and_sensor_source  # noqa: E501
    if session.evaluation_context.execution_information_mode == "shared_obs":
        if captured is None:
            raise ValueError("SharedObs execution requires captured availability.")
        return jnp.asarray(captured, dtype=jnp.bool_)
    if captured is not None:
        raise ValueError("NoSharedObs execution must omit SharedObs availability.")
    return None


def _debugger_expected_horizon(
    scenario: DebuggerScenario,
    config: EnvConfig,
    state: EnvState,
) -> int:
    """Use the script length or remaining interactive simulator transitions."""
    return (
        len(scenario.frames)
        if scenario.mode == "scripted"
        else config.max_steps - int(state.step_count)
    )


def create_session(
    scenario: DebuggerScenario,
    *,
    seed: int,
    evaluation_launch_specification: DebuggerEvaluationLaunchSpecificationV1,
    controlled_global_slot: int | None,
    show_ranges: bool,
    verbose_logging: bool,
    team_a_controller: TeamController = "manual",
    team_b_controller: TeamController = "manual",
    execution_information_mode: ExecutionInformationMode = "no_shared_obs",
) -> DebuggerSession:
    """Create one deterministic immutable debugger session.

    Defaults keep low-level diagnostics on their former manual, NoSharedObs
    contract while live launchers pass the researcher-facing values explicitly.
    """
    if seed != evaluation_launch_specification.root_seed:
        raise ValueError("seed must equal the debugger evaluation launch root seed.")
    (
        scenario_default_movement_scale,
        config,
        state,
        observation,
        action_mask,
    ) = _fresh_snapshot(scenario, seed)
    evaluation_context = build_debugger_evaluation_context_v1(
        evaluation_launch_specification,
        scenario=scenario,
        config=config,
        run_generation=0,
        action_source_kind=debugger_action_source_kind_v1(
            scenario,
            team_a_controller,
            team_b_controller,
        ),
        team_a_controller=team_a_controller,
        team_b_controller=team_b_controller,
        execution_information_mode=execution_information_mode,
        expected_horizon=_debugger_expected_horizon(scenario, config, state),
    )
    next_key = jax.random.key(evaluation_context.seed_protocol.environment_seed)
    information_availability = _initial_information_availability(
        config,
        execution_information_mode,
    )
    initial_frame = capture_initial_evaluation_frame_v1(
        evaluation_context,
        state,
        observation,
        action_mask,
        information_availability,
    )
    status_source_evidence_state = initialize_status_source_evidence_v2(
        evaluation_context,
        initial_frame,
    )
    requested_slot = (
        scenario.default_controlled_slot
        if controlled_global_slot is None
        else controlled_global_slot
    )
    if not (
        0 <= requested_slot < MAX_AGENT_SLOTS
        and evaluation_context.roster[requested_slot].configured_active
    ):
        requested_slot = scenario.default_controlled_slot
    _validate_active_context_slot(
        evaluation_context,
        requested_slot,
        name="controlled_global_slot",
    )

    session = DebuggerSession(
        scenario=scenario,
        seed=seed,
        run_generation=0,
        scenario_default_movement_scale=scenario_default_movement_scale,
        config=config,
        key=next_key,
        state=state,
        observation=observation,
        action_mask=action_mask,
        raw_continuation_identity=None,
        evaluation_context=evaluation_context,
        current_evaluation_frame=initial_frame,
        incoming_evaluation_view=None,
        status_source_evidence_state=status_source_evidence_state,
        last_submission_kind=None,
        last_report_actor_slots=(),
        team_a_controller=team_a_controller,
        team_b_controller=team_b_controller,
        controlled_global_slot=requested_slot,
        pending_actions=_default_pending_actions(
            evaluation_context,
            initial_frame.action_mask,
        ),
        next_script_frame_index=0,
        show_ranges=show_ranges,
        verbose_logging=False,
    )
    return session


def select_clicked_target(
    session: DebuggerSession,
    target_global_slot: int,
) -> DebuggerSession:
    """Select an active target and auto-arm Basic only when exact lane zero is legal."""
    _validate_active_context_slot(
        session.evaluation_context,
        target_global_slot,
        name="clicked target",
    )
    target_action = _target_action_from_context(
        session.evaluation_context,
        session.controlled_global_slot,
        target_global_slot,
    )
    availability = lane_availability(
        session.current_evaluation_frame.action_mask,
        session.controlled_global_slot,
        target_action,
        0,
    )
    pending = PendingAction(
        move_action=session.pending_action.move_action,
        selected_global_target_slot=target_global_slot,
        armed_lane=0 if availability.lane_0_available else None,
        arm_origin="automatic" if availability.lane_0_available else None,
    )
    return _replace_controlled_pending_action(session, pending)


def clear_pending_target(session: DebuggerSession) -> DebuggerSession:
    """Clear target selection while preserving an explicit Mage Burst arm."""
    class_id = session.evaluation_context.roster[
        session.controlled_global_slot
    ].class_id
    keep_mage_ultimate = (
        class_id == MAGE_CLASS_ID
        and session.pending_action.armed_lane == 1
        and session.pending_action.arm_origin == "explicit"
    )
    pending = PendingAction(
        move_action=session.pending_action.move_action,
        selected_global_target_slot=None,
        armed_lane=1 if keep_mage_ultimate else 0,
        arm_origin="explicit" if keep_mage_ultimate else "automatic",
    )
    return _replace_controlled_pending_action(session, pending)


def arm_basic(session: DebuggerSession) -> DebuggerSession:
    """Explicitly arm lane zero even when the current pair is unavailable."""
    return _replace_controlled_pending_action(
        session,
        replace(
            session.pending_action,
            armed_lane=0,
            arm_origin="explicit",
        ),
    )


def arm_ultimate(session: DebuggerSession) -> DebuggerSession:
    """Explicitly arm lane one; Mage Burst always uses target-none."""
    class_id = session.evaluation_context.roster[
        session.controlled_global_slot
    ].class_id
    return _replace_controlled_pending_action(
        session,
        replace(
            session.pending_action,
            selected_global_target_slot=(
                None
                if class_id == MAGE_CLASS_ID
                else session.pending_action.selected_global_target_slot
            ),
            armed_lane=1,
            arm_origin="explicit",
        ),
    )


def select_no_combat(session: DebuggerSession) -> DebuggerSession:
    """Stage no-combat intent while preserving movement and target context."""
    return _replace_controlled_pending_action(
        session,
        replace(
            session.pending_action,
            armed_lane=None,
            arm_origin=None,
        ),
    )


def set_pending_movement(
    session: DebuggerSession,
    move_action: int,
) -> DebuggerSession:
    """Set one in-domain pending movement category without inspecting legality."""
    if not 0 <= move_action < NUM_MOVE_ACTIONS:
        msg = f"move_action must be in [0, {NUM_MOVE_ACTIONS}); got {move_action}."
        raise ValueError(msg)
    return _replace_controlled_pending_action(
        session,
        replace(session.pending_action, move_action=move_action),
    )


def cycle_controlled_actor(
    session: DebuggerSession,
    direction: int,
) -> DebuggerSession:
    """Cycle active fixed slots through direct controlled-actor selection."""
    if direction not in (-1, 1):
        msg = f"direction must be -1 or 1; got {direction}."
        raise ValueError(msg)
    active_slots = _active_context_slots(session)
    current_index = active_slots.index(session.controlled_global_slot)
    controlled_slot = active_slots[(current_index + direction) % len(active_slots)]
    return select_controlled_actor(session, controlled_slot)


def select_controlled_actor(
    session: DebuggerSession,
    global_slot: int,
) -> DebuggerSession:
    """Select an active actor without changing any staged draft or simulator epoch."""
    _validate_active_context_slot(
        session.evaluation_context,
        global_slot,
        name="controlled actor",
    )
    return replace(session, controlled_global_slot=global_slot)


def _validate_joint_action(action: Action) -> None:
    for name, head in zip(Action._fields, action, strict=True):
        if head.shape != (MAX_AGENT_SLOTS,):
            msg = (
                f"action.{name} must have shape ({MAX_AGENT_SLOTS},); got {head.shape}."
            )
            raise ValueError(msg)
        if head.dtype != jnp.int32:
            msg = f"action.{name} must have dtype int32; got {head.dtype}."
            raise ValueError(msg)


def _terminal_reason(session: DebuggerSession) -> str | None:
    if session.terminated:
        return "terminated"
    if session.truncated:
        return "truncated"
    if session.reached_declared_horizon:
        return "at its declared horizon"
    return None


def _post_submit_pending(
    session: DebuggerSession,
    action_mask: ActionMaskV1,
) -> tuple[PendingAction, ...]:
    pending_actions: list[PendingAction] = []
    for actor_slot in range(MAX_AGENT_SLOTS):
        if not session.evaluation_context.roster[actor_slot].configured_active:
            pending_actions.append(PendingAction(armed_lane=None, arm_origin=None))
            continue
        target_slot = session.pending_actions[actor_slot].selected_global_target_slot
        target_action = _target_action_from_context(
            session.evaluation_context,
            actor_slot,
            target_slot,
        )
        availability = lane_availability(
            action_mask,
            actor_slot,
            target_action,
            0,
        )
        pending_actions.append(
            PendingAction(
                move_action=MOVE_STAY,
                selected_global_target_slot=target_slot,
                armed_lane=0 if availability.lane_0_available else None,
                arm_origin="automatic" if availability.lane_0_available else None,
            )
        )
    return tuple(pending_actions)


def submit_joint_action(
    session: DebuggerSession,
    submitted_action: Action,
    *,
    submission_kind: SubmissionKind,
    report_actor_slots: tuple[int, ...],
) -> DebuggerSession:
    """Split once, step once, diagnose once, and advance all paired epoch fields."""
    terminal_reason = _terminal_reason(session)
    if terminal_reason is not None:
        return session
    try:
        _validate_joint_action(submitted_action)
        if len(report_actor_slots) != len(set(report_actor_slots)):
            raise ValueError("report_actor_slots must not contain duplicates.")
        for actor_slot in report_actor_slots:
            _validate_active_context_slot(
                session.evaluation_context,
                actor_slot,
                name="report actor",
            )
    except Exception as error:
        raise _transition_failure(
            "action_build",
            "invalid_submitted_action",
            error,
        ) from error

    try:
        next_key, step_key = jax.random.split(session.key)
        (
            next_state,
            next_observation,
            reward,
            done_flags,
            next_action_mask,
            info,
        ) = step(
            session.config,
            session.state,
            session.action_mask,
            submitted_action,
            step_key,
        )
    except Exception as error:
        raise _transition_failure(
            "simulation",
            "simulator_step_failed",
            error,
        ) from error
    try:
        transition, successor_frame = capture_evaluation_transition_unit_v1(
            session.evaluation_context,
            session.current_evaluation_frame,
            next_state,
            next_observation,
            next_action_mask,
            info.transition_facts,
            reward,
            done_flags,
            successor_shared_obs_information_availability_by_recipient_and_sensor_source=(
                _captured_information_availability(session)
            ),
        )
    except Exception as error:
        raise _transition_failure(
            "capture",
            "transition_capture_failed",
            error,
        ) from error
    try:
        coherent_view = EvaluationTransitionViewV1(
            context=session.evaluation_context,
            start_frame=session.current_evaluation_frame,
            transition=transition,
            successor_frame=successor_frame,
        )
        status_source_evidence_state = advance_status_source_evidence_v2(
            session.status_source_evidence_state,
            coherent_view,
        )
        return replace(
            session,
            key=next_key,
            state=next_state,
            observation=next_observation,
            action_mask=next_action_mask,
            evaluation_context=coherent_view.context,
            current_evaluation_frame=coherent_view.successor_frame,
            incoming_evaluation_view=coherent_view,
            status_source_evidence_state=status_source_evidence_state,
            last_submission_kind=submission_kind,
            last_report_actor_slots=tuple(sorted(report_actor_slots)),
            raw_continuation_identity=None,
            pending_actions=_post_submit_pending(
                session,
                coherent_view.successor_frame.action_mask,
            ),
        )
    except Exception as error:
        raise _transition_failure(
            "validation",
            "transition_packaging_failed",
            error,
        ) from error


def submit_interactive(
    session: DebuggerSession,
    *,
    actor_global_slots: tuple[int, ...] | None = None,
) -> DebuggerSession:
    """Submit one authorized collection of same-epoch pending actor rows."""
    if any(
        controller != "manual"
        for controller in (
            session.team_a_controller,
            session.team_b_controller,
        )
    ):
        try:
            action = _build_configured_joint_action(session)
        except Exception as error:
            raise _transition_failure(
                "action_build",
                "policy_action_build_failed",
                error,
            ) from error
        return submit_joint_action(
            session,
            action,
            submission_kind="interactive",
            report_actor_slots=_active_context_slots(session),
        )
    submission_slots = (
        _active_context_slots(session)
        if actor_global_slots is None
        else actor_global_slots
    )
    try:
        action = build_interactive_joint_action(
            session.evaluation_context,
            session.pending_actions,
            actor_global_slots=submission_slots,
        )
    except Exception as error:
        raise _transition_failure(
            "action_build",
            "interactive_action_build_failed",
            error,
        ) from error
    return submit_joint_action(
        session,
        action,
        submission_kind="interactive",
        report_actor_slots=submission_slots,
    )


def submit_next_script_frame(
    session: DebuggerSession,
) -> DebuggerSession:
    """Submit the next registered multi-actor frame through the shared boundary."""
    scenario = session.scenario
    if session.next_script_frame_index >= len(scenario.frames):
        return session
    frame = scenario.frames[session.next_script_frame_index]
    try:
        action = build_scripted_joint_action(session.evaluation_context, frame)
        report_slots = tuple(
            sorted(command.actor_global_slot for command in frame.commands)
        )
    except Exception as error:
        raise _transition_failure(
            "action_build",
            "scripted_action_build_failed",
            error,
        ) from error
    submitted = submit_joint_action(
        session,
        action,
        submission_kind="scripted",
        report_actor_slots=report_slots,
    )
    if (
        submitted.current_evaluation_frame.frame_index
        == session.current_evaluation_frame.frame_index
    ):
        return submitted
    return replace(
        submitted,
        next_script_frame_index=session.next_script_frame_index + 1,
    )


def reset_session(
    session: DebuggerSession,
) -> DebuggerSession:
    """Recreate the deterministic initial epoch at the product scale."""
    scenario = session.scenario
    return _restart_session(
        session,
        scenario,
        preserve_controlled_slot=True,
    )


def _restart_session(
    session: DebuggerSession,
    scenario: DebuggerScenario,
    *,
    preserve_controlled_slot: bool,
    team_a_controller: TeamController | None = None,
    team_b_controller: TeamController | None = None,
    execution_information_mode: ExecutionInformationMode | None = None,
) -> DebuggerSession:
    """Build one coherent fresh epoch without entering the simulator step seam."""
    next_team_a_controller = (
        session.team_a_controller if team_a_controller is None else team_a_controller
    )
    next_team_b_controller = (
        session.team_b_controller if team_b_controller is None else team_b_controller
    )
    next_information_mode = (
        session.evaluation_context.execution_information_mode
        if execution_information_mode is None
        else execution_information_mode
    )
    (
        scenario_default_movement_scale,
        config,
        state,
        observation,
        action_mask,
    ) = _fresh_snapshot(scenario, session.seed)
    run_generation = session.run_generation + 1
    launch_specification = build_debugger_evaluation_launch_specification_v1(
        root_seed=session.evaluation_context.seed_protocol.root_seed,
        code_revision=session.evaluation_context.code_revision,
        capture_profile=cast(
            DebuggerCaptureProfileV1,
            session.evaluation_context.capture_profile,
        ),
    )
    evaluation_context = build_debugger_evaluation_context_v1(
        launch_specification,
        scenario=scenario,
        config=config,
        run_generation=run_generation,
        action_source_kind=debugger_action_source_kind_v1(
            scenario,
            next_team_a_controller,
            next_team_b_controller,
        ),
        team_a_controller=next_team_a_controller,
        team_b_controller=next_team_b_controller,
        execution_information_mode=next_information_mode,
        expected_horizon=_debugger_expected_horizon(scenario, config, state),
    )
    next_key = jax.random.key(evaluation_context.seed_protocol.environment_seed)
    information_availability = _initial_information_availability(
        config,
        next_information_mode,
    )
    initial_frame = capture_initial_evaluation_frame_v1(
        evaluation_context,
        state,
        observation,
        action_mask,
        information_availability,
    )
    status_source_evidence_state = initialize_status_source_evidence_v2(
        evaluation_context,
        initial_frame,
    )
    controlled_slot = (
        session.controlled_global_slot
        if preserve_controlled_slot
        else scenario.default_controlled_slot
    )
    if not (
        0 <= controlled_slot < MAX_AGENT_SLOTS
        and evaluation_context.roster[controlled_slot].configured_active
    ):
        controlled_slot = scenario.default_controlled_slot
    _validate_active_context_slot(
        evaluation_context,
        controlled_slot,
        name="controlled_global_slot",
    )
    restarted = replace(
        session,
        scenario=scenario,
        run_generation=run_generation,
        scenario_default_movement_scale=scenario_default_movement_scale,
        config=config,
        key=next_key,
        state=state,
        observation=observation,
        action_mask=action_mask,
        raw_continuation_identity=None,
        evaluation_context=evaluation_context,
        current_evaluation_frame=initial_frame,
        incoming_evaluation_view=None,
        status_source_evidence_state=status_source_evidence_state,
        last_submission_kind=None,
        last_report_actor_slots=(),
        team_a_controller=next_team_a_controller,
        team_b_controller=next_team_b_controller,
        controlled_global_slot=controlled_slot,
        pending_actions=_default_pending_actions(
            evaluation_context,
            initial_frame.action_mask,
        ),
        next_script_frame_index=0,
    )
    return restarted


def set_combat_configuration(
    session: DebuggerSession,
    *,
    team_a_controller: TeamController,
    team_b_controller: TeamController,
    execution_information_mode: ExecutionInformationMode,
) -> DebuggerSession:
    """Replace the episode only when its controller or information mode changes."""
    if team_a_controller not in SUPPORTED_TEAM_CONTROLLERS:
        raise ValueError(
            "team_a_controller must be manual, scripted_tdm, or random_valid"
        )
    if team_b_controller not in SUPPORTED_TEAM_CONTROLLERS:
        raise ValueError(
            "team_b_controller must be manual, scripted_tdm, or random_valid"
        )
    if execution_information_mode not in ("shared_obs", "no_shared_obs"):
        raise ValueError(
            "execution_information_mode must be shared_obs or no_shared_obs"
        )
    if (
        session.team_a_controller == team_a_controller
        and session.team_b_controller == team_b_controller
        and session.evaluation_context.execution_information_mode
        == execution_information_mode
    ):
        return session
    return _restart_session(
        session,
        session.scenario,
        preserve_controlled_slot=True,
        team_a_controller=team_a_controller,
        team_b_controller=team_b_controller,
        execution_information_mode=execution_information_mode,
    )


def switch_scenario(
    session: DebuggerSession,
    scenario: DebuggerScenario,
) -> DebuggerSession:
    """Start another scenario at the canonical product movement scale."""
    return _restart_session(
        session,
        scenario,
        preserve_controlled_slot=False,
    )
