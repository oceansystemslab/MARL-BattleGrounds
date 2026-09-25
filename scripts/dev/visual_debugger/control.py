"""Apply debugger drafts and scripted commands through the shared simulator boundary.

The input layer calls these helpers to build immutable ``DebuggerSession`` values.
Draft/selection helpers change host controls only. Submission builds both teams'
actions before one Core step, then captures its matching state, observation, mask
and transition record. Reset helpers create a new recorded episode. Random keys
remain explicit in the session; this module does not write replay files or serve
HTTP requests.

Every Submit's Core step runs compiled, for manual, scripted and policy
Submits alike. SharedObs policy teams (ALPHA, BETA, GAMMA and Random) also run
the policy executor compiled. This module wraps the real ``apply_policies`` and
``step`` in ``jax.jit`` once, at import. JAX reuses a compiled program only
while the controller pair and every array shape and dtype stay the same.
Run uncompiled, one Submit took seconds, because every array operation ran on
its own. NoSharedObs Random still runs uncompiled through
``execute_no_shared_obs_team_policy``; it is fast once JAX has warmed up. A
replaced executor or step (for example a test spy) is called as given,
uncompiled.

Compile cost: the first Submit in a DevClient session (the first in this
Python process) compiles for several seconds, about 6 s on CPU. The first
Submit after switching to a controller pair not yet used in this process
compiles the executor again, about 3 s. After that a Submit takes about
0.03 s. The DevClient service applies each command while it holds its lock,
so the browser waits while a compile runs. These times were measured once on
CPU during review; they are examples, not guarantees.
"""

from collections.abc import Callable
from dataclasses import replace
from typing import Literal, cast

import jax
import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core import env as core_env
from marl_battlegrounds.core.config import validate_product_env_config
from marl_battlegrounds.core.env import initialize_scenario_state, step
from marl_battlegrounds.core.types import (
    AGENT_FEATURE_ACTIVE,
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
    DoneFlags,
    EnvConfig,
    EnvState,
    Info,
    Observation,
    Reward,
)
from marl_battlegrounds.evaluation import policy_execution
from marl_battlegrounds.evaluation.capture import (
    capture_evaluation_transition_unit_v3,
    capture_initial_evaluation_frame_v3,
)
from marl_battlegrounds.evaluation.metrics import EvaluationTransitionViewV1
from marl_battlegrounds.evaluation.models import (
    ActionMaskV1,
    CodeRevisionV1,
    EvaluationEpisodeContextV4,
    ExecutionInformationMode,
)
from marl_battlegrounds.evaluation.policy_execution import (
    Policy,
    PolicyTree,
    apply_policies,
    policy,
)
from marl_battlegrounds.policies.actor import (
    ActorAction,
    build_joint_action_from_actor_actions,
)
from marl_battlegrounds.policies.input import ActorInput, Observations
from marl_battlegrounds.policies.no_shared_obs import (
    execute_no_shared_obs_team_policy,
)
from marl_battlegrounds.policies.random_valid import random_policy
from marl_battlegrounds.policies.shared_obs import (
    build_default_shared_obs_information_availability,
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
    SUPPORTED_TEAM_B_CONTROLLERS,
    SUPPORTED_TEAM_CONTROLLERS,
    DebuggerScenario,
    DebuggerSession,
    Lane,
    LaneAvailability,
    PendingAction,
    ScenarioFrame,
    SubmissionKind,
    TeamBController,
    TeamController,
)

# Compiled forms of the real policy executor and Core step. Controller callables
# are static (the registered adapters are stable objects), so each Team A/Team B
# controller pair gets its own program. Observations, masks, keys, states and
# manual actions stay dynamic: new frames, maps and seeds reuse a program only
# while every array shape and dtype stays the same; any change compiles again.
# These functions keep their compile cache for the whole process, so a test that
# patches code inside the compiled path must replace control.apply_policies or
# control.step; patching deeper internals would not reach a cached program.
_COMPILED_APPLY_POLICIES = cast(
    Callable[..., tuple[Action, PolicyTree, PolicyTree]],
    jax.jit(policy_execution.apply_policies, static_argnums=(0, 1)),
)
_COMPILED_STEP = cast(
    Callable[..., tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info]],
    jax.jit(core_env.step),
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
        """Retain a stable failure stage/code without publishing raw exception
        details.
        """
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
    """Convert a caught failure into the public stage/code pair; omit the raw
    message.
    """
    del error
    return DebuggerTransitionFailureV1(stage, stable_code)


def make_neutral_joint_action() -> Action:
    """Create a fixed-size joint action with no movement, target or Ultimate use.

    Returns
    -------
    Action
        Three int32 JAX arrays shaped [10], using Stay and zero combat categories.
        This initializes a request; current masks still determine submitted-action
        acceptance when the request reaches Core.
    """
    return Action(
        move=jnp.full((MAX_AGENT_SLOTS,), MOVE_STAY, dtype=jnp.int32),
        select_target=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
        use_ultimate=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
    )


def _validate_active_context_slot(
    context: EvaluationEpisodeContextV4,
    global_slot: int,
    *,
    name: str,
) -> None:
    """Require a global slot in range whose recorded roster row is active."""
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
    context: EvaluationEpisodeContextV4,
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
    """Read exact lane availability from the authoritative joint mask.

    Parameters
    ----------
    action_mask : ActionMask | ActionMaskV1
        Current simulator mask or its recorded ActionMaskV1 representation.
    actor_global_slot : int
        Global actor index from zero through nine.
    target_action : int
        Target category on the actor-relative fixed target axis.
    armed_lane : Lane | None
        0 selects Basic, 1 selects Ultimate, and None means neither is armed.

    Returns
    -------
    LaneAvailability
        Both exact lane flags and whether the selected target/lane pair is
        legal.

    Raises
    ------
    ValueError
        If an actor, action category or selection argument violates the stated domain.
    """
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
    context: EvaluationEpisodeContextV4,
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
    context: EvaluationEpisodeContextV4,
    pending_actions: tuple[PendingAction, ...],
    *,
    actor_global_slots: tuple[int, ...],
) -> Action:
    """Build one authorized joint request without pre-filtering it by the mask.

    Parameters
    ----------
    context : EvaluationEpisodeContextV4
        Recorded episode roster and target-action mapping.
    pending_actions : tuple[PendingAction, ...]
        Exactly ten fixed-slot pending rows from the current decision.
    actor_global_slots : tuple[int, ...]
        Nonempty tuple of distinct active actors authorized for this
        submission.

    Returns
    -------
    Action
        Fixed-slot joint request. Actions are not prefiltered by masks; Core
        records acceptance.

    Raises
    ------
    ValueError
        If an actor, action category or selection argument violates the stated domain.
    """
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


def _policy_keys(session: DebuggerSession) -> Array:
    """Derive role-correct actor keys without consuming the environment stream."""
    context = session.evaluation_context
    seeds = context.seed_protocol
    frame_index = session.current_evaluation_frame.frame_index

    def keys_for_seed(seed: object, *, role: str) -> Array:
        """Derive this decision's actor keys from the recorded role-specific seed."""
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


def _required_seed(seed: int | None) -> int:
    """Require a recorded seed value before deriving the next random-key stream."""
    if seed is None:
        raise ValueError("Live debugger execution requires a recorded seed")
    return seed


def _manual_policy(
    variables: PolicyTree,
    carry: PolicyTree,
    actor: ActorInput,
    mask: ActionMask,
    key: Array,
) -> tuple[ActorAction, PolicyTree]:
    """Read a precommitted manual row through the same scalar actor interface."""
    del mask, key
    action = cast(Action, variables)
    active = actor.observation.self_features[AGENT_FEATURE_ACTIVE] > 0.0
    return ActorAction(
        *(
            jnp.where(active, value[actor.observation.self_ally_index], 0)
            for value in action
        )
    ), carry


def _configured_policy(
    session: DebuggerSession, team_identity: int, controller: TeamBController
) -> Policy:
    """Adapt existing stateless choices; no learned-policy UI or memory is added."""
    if controller == "manual":
        slots = tuple(
            row.global_slot
            for row in session.evaluation_context.roster
            if row.configured_active and row.configured_team_id == team_identity
        )
        action = build_interactive_joint_action(
            session.evaluation_context,
            session.pending_actions,
            actor_global_slots=slots,
        )
        team_start = 0 if team_identity == TEAM_A_ID else MAX_AGENTS_PER_TEAM
        local_action = Action(
            *(value[team_start : team_start + MAX_AGENTS_PER_TEAM] for value in action)
        )
        return Policy("manual", _manual_policy, variables=local_action)
    return policy(
        {
            "reactive_tdm": "tdm-alpha",
            "random_valid": "random",
            "scenario_5": "tdm-beta",
            "tdm_gamma": "tdm-gamma",
        }[controller]
    )


def _build_configured_joint_action(session: DebuggerSession) -> Action:
    """Resolve both teams before the single existing simulator step.

    SharedObs uses the public five-argument application authority, compiled
    with one program per controller pair, reused only while array shapes and
    dtypes stay the same (see the module description); a replaced executor
    runs uncompiled. Legacy NoSharedObs Random keeps its original
    three-argument, no-source-bank ABI and runs uncompiled.
    """
    controllers = (session.team_a_controller, session.team_b_controller)
    if controllers == ("manual", "manual"):
        raise ValueError("configured policy assembly requires one policy team")
    mode = session.evaluation_context.execution_information_mode
    availability = _captured_information_availability(session)
    keys = _policy_keys(session)
    if mode == "shared_obs":
        if availability is None:
            raise ValueError("SharedObs requires captured information availability")
        first = _configured_policy(session, TEAM_A_ID, controllers[0])
        second = _configured_policy(session, TEAM_B_ID, controllers[1])
        executor = (
            _COMPILED_APPLY_POLICIES
            if apply_policies is policy_execution.apply_policies
            else apply_policies
        )
        action, _, _ = executor(
            first.apply,
            second.apply,
            first.variables,
            second.variables,
            first.initial_carry,
            second.initial_carry,
            Observations(session.observation, availability),
            session.action_mask,
            keys,
        )
        return action
    if mode != "no_shared_obs" or availability is not None:
        raise ValueError("unsupported execution information mode")
    actions: list[ActorAction] = []
    for team_identity, controller in zip(
        (TEAM_A_ID, TEAM_B_ID), controllers, strict=True
    ):
        if controller == "manual":
            manual = _configured_policy(session, team_identity, controller)
            actions.append(ActorAction(*cast(Action, manual.variables)))
        elif controller == "random_valid":
            actions.append(
                cast(
                    ActorAction,
                    execute_no_shared_obs_team_policy(
                        session.observation,
                        session.action_mask,
                        keys,
                        policy=random_policy,
                        team_identity=team_identity,
                    ),
                )
            )
        else:
            raise ValueError("unsupported NoSharedObs team controller")
    return build_joint_action_from_actor_actions(actions[0], actions[1])


def build_scripted_joint_action(
    context: EvaluationEpisodeContextV4,
    frame: ScenarioFrame,
) -> Action:
    """Build a potentially multi-actor scripted request from neutral defaults.

    Parameters
    ----------
    context : EvaluationEpisodeContextV4
        Recorded episode roster and target-action mapping.
    frame : ScenarioFrame
        One script frame containing distinct actor commands; omitted actors
        remain neutral.

    Returns
    -------
    Action
        Fixed-slot joint request with neutral heads for actors omitted from the
        script frame.

    Raises
    ------
    ValueError
        If an actor, action category or selection argument violates the stated domain.
    """
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


class CombatConfigurationRejectedError(ValueError):
    """An expected selection error that leaves the current session usable."""


def _validate_reactive_controller_selection(
    scenario: DebuggerScenario,
    *,
    team_a_controller: TeamController,
    team_b_controller: TeamBController,
    execution_information_mode: ExecutionInformationMode,
) -> None:
    """Check execution boundaries, independently of scenario content."""
    if not any(
        controller in ("reactive_tdm", "scenario_5", "tdm_gamma")
        for controller in (team_a_controller, team_b_controller)
    ):
        return
    if scenario.mode != "interactive":
        raise CombatConfigurationRejectedError(
            "Reactive controllers require an interactive session."
        )
    if execution_information_mode != "shared_obs":
        raise CombatConfigurationRejectedError(
            "Reactive controllers require SharedObs."
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
    team_b_controller: TeamBController = "manual",
    execution_information_mode: ExecutionInformationMode = "no_shared_obs",
) -> DebuggerSession:
    """Create one deterministic immutable debugger session.

    Defaults keep low-level diagnostics on their former manual, NoSharedObs
    contract while live launchers pass the researcher-facing values explicitly.

    Parameters
    ----------
    scenario : DebuggerScenario
        Factory and metadata for the authored initial state or scripted
        demonstration.
    seed : int
        Root seed, which must equal the launch specification root seed.
    evaluation_launch_specification : DebuggerEvaluationLaunchSpecificationV1
        Recorded identity, seed and capture choices for the new episode.
    controlled_global_slot : int | None
        Requested zero-based active actor slot, or None for the scenario
        default. An unavailable request falls back to that default.
    show_ranges : bool
        Whether the initial view displays configured range overlays.
    verbose_logging : bool
        Whether the host session requests verbose diagnostic logging.
    team_a_controller : TeamController
        Team A controller; default manual. Reactive choices require SharedObs.
    team_b_controller : TeamBController
        Team B controller; default manual. Both teams accept ALPHA
        (``reactive_tdm``), BETA (``scenario_5``) and GAMMA (``tdm_gamma``),
        which require SharedObs, as well as Manual and Random.
    execution_information_mode : ExecutionInformationMode
        no_shared_obs by default for direct diagnostics; shared_obs enables the
        composed team input contract.

    Returns
    -------
    DebuggerSession
        Initial matching session with frame-zero evidence, default pending rows
        and explicit random key.

    Raises
    ------
    ValueError
        If the scenario, seed, active selection or controller/input combination is
        invalid.
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
    _validate_reactive_controller_selection(
        scenario,
        team_a_controller=team_a_controller,
        team_b_controller=team_b_controller,
        execution_information_mode=execution_information_mode,
    )
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
    next_key = jax.random.key(
        _required_seed(evaluation_context.seed_protocol.environment_seed)
    )
    information_availability = _initial_information_availability(
        config,
        execution_information_mode,
    )
    initial_frame = capture_initial_evaluation_frame_v3(
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
    """Select an active target and auto-arm Basic only when exact lane zero is legal.

    Parameters
    ----------
    session : DebuggerSession
        Current immutable session and pending action rows.
    target_global_slot : int
        Active global target slot selected in the authorized researcher view.

    Returns
    -------
    DebuggerSession
        Session with the requested draft or controlled-actor selection.
        Simulator state and the environment random key are unchanged.

    Raises
    ------
    ValueError
        If an actor, action category or selection argument violates the stated domain.
    """
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
    """Clear target selection while preserving an explicit Mage Burst arm.

    Parameters
    ----------
    session : DebuggerSession
        Current immutable session and controlled actor draft.

    Returns
    -------
    DebuggerSession
        Session with the requested draft or controlled-actor selection.
        Simulator state and the environment random key are unchanged.
    """
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
    """Explicitly arm lane zero even when the current pair is unavailable.

    Parameters
    ----------
    session : DebuggerSession
        Current immutable session and controlled actor draft.

    Returns
    -------
    DebuggerSession
        Session with the requested draft or controlled-actor selection.
        Simulator state and the environment random key are unchanged.
    """
    return _replace_controlled_pending_action(
        session,
        replace(
            session.pending_action,
            armed_lane=0,
            arm_origin="explicit",
        ),
    )


def arm_ultimate(session: DebuggerSession) -> DebuggerSession:
    """Explicitly arm lane one; Mage Burst always uses target-none.

    Parameters
    ----------
    session : DebuggerSession
        Current immutable session and controlled actor draft.

    Returns
    -------
    DebuggerSession
        Session with the requested draft or controlled-actor selection.
        Simulator state and the environment random key are unchanged.
    """
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
    """Stage no-combat intent while preserving movement and target context.

    Parameters
    ----------
    session : DebuggerSession
        Current immutable session and controlled actor draft.

    Returns
    -------
    DebuggerSession
        Session with the requested draft or controlled-actor selection.
        Simulator state and the environment random key are unchanged.
    """
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
    """Set one in-domain pending movement category without inspecting legality.

    Parameters
    ----------
    session : DebuggerSession
        Current immutable session and controlled actor draft.
    move_action : int
        Movement category in the fixed move-action domain; current legality is
        left to submission.

    Returns
    -------
    DebuggerSession
        Session with the requested draft or controlled-actor selection.
        Simulator state and the environment random key are unchanged.

    Raises
    ------
    ValueError
        If an actor, action category or selection argument violates the stated domain.
    """
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
    """Cycle active fixed slots through direct controlled-actor selection.

    Parameters
    ----------
    session : DebuggerSession
        Current immutable session with its active roster and selection.
    direction : int
        1 selects the next active slot; -1 selects the previous active slot.

    Returns
    -------
    DebuggerSession
        Session with the requested draft or controlled-actor selection.
        Simulator state and the environment random key are unchanged.

    Raises
    ------
    ValueError
        If an actor, action category or selection argument violates the stated domain.
    """
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
    """Select an active actor without changing any staged draft or simulator epoch.

    Parameters
    ----------
    session : DebuggerSession
        Current immutable session and staged joint draft.
    global_slot : int
        Active zero-based global actor slot to control.

    Returns
    -------
    DebuggerSession
        Session with the requested draft or controlled-actor selection.
        Simulator state and the environment random key are unchanged.

    Raises
    ------
    ValueError
        If an actor, action category or selection argument violates the stated domain.
    """
    _validate_active_context_slot(
        session.evaluation_context,
        global_slot,
        name="controlled actor",
    )
    return replace(session, controlled_global_slot=global_slot)


def _validate_joint_action(action: Action) -> None:
    """Require all three submitted heads to be int32 arrays with shape [10]."""
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
    """Return the sealed endpoint reason in priority order, or None while stepping is
    allowed.
    """
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
    """Keep target choices while resetting movement and re-arming only a legal Basic
    pair.
    """
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
    """Split once, step once, diagnose once, and advance all paired epoch fields.

    Parameters
    ----------
    session : DebuggerSession
        Current paired state, observation, action mask, random key and
        evaluation frame.
    submitted_action : Action
        Three int32 action heads shaped [10], chosen before this transition.
    submission_kind : SubmissionKind
        interactive or scripted, retained with the resulting transition.
    report_actor_slots : tuple[int, ...]
        Distinct active global actor slots whose submitted/accepted rows should
        be reported.

    Returns
    -------
    DebuggerSession
        Matching successor session and captured incoming transition, or the
        unchanged session when already sealed.

    Raises
    ------
    DebuggerTransitionFailureV1
        If action construction, simulation, capture or successor packaging fails.
        The stable stage/code identifies the boundary without exposing raw details.

    Notes
    -----
        Numerical work runs through Core and transition capture; no file is written.
        The input session is immutable, so a failed call cannot install a partial
        successor. Higher-level recording code decides how to handle a failure.

        Every Submit's Core step runs compiled here, for manual, scripted and
        policy Submits alike, unless a caller has replaced ``control.step``.
        Policy actions arrive already built: ``submit_interactive`` runs
        SharedObs policy teams through the compiled policy executor, and
        NoSharedObs Random uncompiled through
        ``execute_no_shared_obs_team_policy`` (fast once JAX has warmed up).
        The first Submit in a DevClient session (the first in this Python
        process) compiles for several seconds, about 6 s on CPU, counting the
        executor's compile when a policy plays. The first Submit after
        switching to a controller pair not yet used in this process compiles
        the executor again, about 3 s. After that a Submit takes about 0.03 s
        while array shapes and dtypes stay the same. The DevClient service
        calls this while holding its lock, so the browser waits during a
        compile. These times were measured once on CPU during review; they
        are examples, not guarantees.
    """
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
        stepper = _COMPILED_STEP if step is core_env.step else step
        (
            next_state,
            next_observation,
            reward,
            done_flags,
            next_action_mask,
            info,
        ) = stepper(
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
        transition, successor_frame = capture_evaluation_transition_unit_v3(
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
    """Submit one authorized collection of same-epoch pending actor rows.

    Parameters
    ----------
    session : DebuggerSession
        Current session carrying both team controller choices and staged manual
        rows.
    actor_global_slots : tuple[int, ...] | None
        Optional active manual submission subset. None means all active actors.
        Configured policy mode resolves and reports both teams instead.

    Returns
    -------
    DebuggerSession
        Successor session after one shared submission, or the unchanged sealed
        endpoint.

    Raises
    ------
    DebuggerTransitionFailureV1
        If action construction, simulation, capture or successor packaging fails.
        The stable stage/code identifies the boundary without exposing raw details.

    Notes
    -----
        Numerical work runs through Core and transition capture; no file is written.
        The input session is immutable, so a failed call cannot install a partial
        successor. Higher-level recording code decides how to handle a failure.

        Every Submit's Core step runs compiled, manual Submits included. With
        SharedObs, the policy teams (ALPHA, BETA, GAMMA and Random) also run
        through the compiled policy executor. NoSharedObs Random still runs
        uncompiled through ``execute_no_shared_obs_team_policy``; it is fast
        once JAX has warmed up. The first Submit in a DevClient session (the
        first in this Python process) compiles for several seconds, about 6 s
        on CPU. The first Submit after switching to a controller pair not yet
        used in this process compiles the executor again, about 3 s. After
        that a Submit takes about 0.03 s. The DevClient service calls this
        while holding its lock, so the browser waits during a compile. These
        times were measured once on CPU during review; they are examples, not
        guarantees.
    """
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
    """Submit the next registered multi-actor frame through the shared boundary.

    Parameters
    ----------
    session : DebuggerSession
        Current scripted session and its next-frame cursor.

    Returns
    -------
    DebuggerSession
        Session with one captured scripted transition and advanced script
        cursor, or the original endpoint if no step is possible.

    Raises
    ------
    DebuggerTransitionFailureV1
        If action construction, simulation, capture or successor packaging fails.
        The stable stage/code identifies the boundary without exposing raw details.

    Notes
    -----
        Numerical work runs through Core and transition capture; no file is written.
        The input session is immutable, so a failed call cannot install a partial
        successor. Higher-level recording code decides how to handle a failure.

        The scripted Core step runs compiled through ``submit_joint_action``;
        its Notes give the first-Submit compile cost.
    """
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
    """Recreate the deterministic initial epoch at the product scale.

    Parameters
    ----------
    session : DebuggerSession
        Session to recreate from its loaded scenario and recorded root seed.

    Returns
    -------
    DebuggerSession
        Fresh deterministic initial session with a new run generation and
        preserved control choice when valid.

    Raises
    ------
    ValueError
        If the scenario, seed, active selection or controller/input combination is
        invalid.
    """
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
    team_b_controller: TeamBController | None = None,
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
    _validate_reactive_controller_selection(
        scenario,
        team_a_controller=next_team_a_controller,
        team_b_controller=next_team_b_controller,
        execution_information_mode=next_information_mode,
    )
    run_generation = session.run_generation + 1
    launch_specification = build_debugger_evaluation_launch_specification_v1(
        root_seed=_required_seed(session.evaluation_context.seed_protocol.root_seed),
        code_revision=cast(CodeRevisionV1, session.evaluation_context.code_revision),
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
    next_key = jax.random.key(
        _required_seed(evaluation_context.seed_protocol.environment_seed)
    )
    information_availability = _initial_information_availability(
        config,
        next_information_mode,
    )
    initial_frame = capture_initial_evaluation_frame_v3(
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
    team_b_controller: TeamBController,
    execution_information_mode: ExecutionInformationMode,
) -> DebuggerSession:
    """Replace the episode only when its controller or information mode changes.

    Parameters
    ----------
    session : DebuggerSession
        Current session, preserved if all requested settings already match.
    team_a_controller : TeamController
        Supported Team A controller kind.
    team_b_controller : TeamBController
        Supported Team B controller kind; both teams accept the same kinds.
    execution_information_mode : ExecutionInformationMode
        shared_obs or no_shared_obs; must satisfy the selected controllers'
        input needs.

    Returns
    -------
    DebuggerSession
        New episode when settings change, otherwise the original session.

    Raises
    ------
    ValueError
        If the scenario, seed, active selection or controller/input combination is
        invalid.
    """
    if team_a_controller not in SUPPORTED_TEAM_CONTROLLERS:
        raise CombatConfigurationRejectedError(
            "team_a_controller must be manual, reactive_tdm, random_valid, "
            "scenario_5, or tdm_gamma"
        )
    if team_b_controller not in SUPPORTED_TEAM_B_CONTROLLERS:
        raise CombatConfigurationRejectedError(
            "team_b_controller must be manual, reactive_tdm, random_valid, "
            "scenario_5, or tdm_gamma"
        )
    if execution_information_mode not in ("shared_obs", "no_shared_obs"):
        raise CombatConfigurationRejectedError(
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
    """Start another scenario at the canonical product movement scale.

    Parameters
    ----------
    session : DebuggerSession
        Current host session whose presentation settings carry into the
        replacement.
    scenario : DebuggerScenario
        New validated scenario factory and metadata.

    Returns
    -------
    DebuggerSession
        Fresh episode for the selected scenario at the product movement
        setting.

    Raises
    ------
    ValueError
        If the scenario, seed, active selection or controller/input combination is
        invalid.
    """
    return _restart_session(
        session,
        scenario,
        preserve_controlled_slot=False,
    )
