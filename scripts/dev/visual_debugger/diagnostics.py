"""Conservative debugger diagnostics derived from public simulator artifacts."""

from dataclasses import replace
from typing import cast

import numpy as np

from marl_battlegrounds.core.combat import (
    HUNTER_BASIC_SLOW_DURATION_TICKS,
    HUNTER_TRAP_STUN_DURATION_TICKS,
    MAGE_BURST_DAMAGE_DURATION_TICKS,
    ONLY_NONE_TARGET_ULTIMATE_MODE,
    PRIEST_HEAL_SPEED_FLOOR_DURATION_TICKS,
    ROGUE_POISON_ANTI_HEAL_DURATION_TICKS,
    ROGUE_POISON_SLOW_DURATION_TICKS,
    ROGUE_POISON_STUN_DURATION_TICKS,
    WARRIOR_CHARGE_SLOW_DURATION_TICKS,
    WARRIOR_CHARGE_STUN_DURATION_TICKS,
    get_ultimate_target_mode_by_class_ids,
)
from marl_battlegrounds.core.geometry import has_clear_line_of_sight
from marl_battlegrounds.core.types import (
    AGENT_FEATURE_ACTIVE,
    AGENT_FEATURE_BASE_MOVEMENT_SPEED,
    AGENT_FEATURE_BASIC_INTERACTION_RADIUS,
    AGENT_FEATURE_CLASS_ID,
    AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER,
    AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER,
    AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED,
    AGENT_FEATURE_MAX_HEALTH,
    AGENT_FEATURE_OBSERVATION_RADIUS,
    AGENT_FEATURE_RADIUS,
    AGENT_FEATURE_TEAM_ID,
    AGENT_FEATURE_ULTIMATE_INTERACTION_RADIUS,
    CONTEXT_FEATURE_EPISODE_HORIZON,
    CONTEXT_FEATURE_MAP_HEIGHT,
    CONTEXT_FEATURE_MAP_WIDTH,
    HUNTER_CLASS_ID,
    MAGE_CLASS_ID,
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    MOVE_STAY,
    NUM_MOVE_ACTIONS,
    NUM_TARGET_ACTIONS,
    NUM_ULTIMATE_ACTIONS,
    PRIEST_CLASS_ID,
    ROGUE_CLASS_ID,
    SLOW_CHANNEL_HUNTER_BASIC,
    SLOW_CHANNEL_ROGUE_POISON,
    SLOW_CHANNEL_WARRIOR_CHARGE,
    STUN_CHANNEL_HUNTER_TRAP,
    STUN_CHANNEL_ROGUE_POISON,
    STUN_CHANNEL_WARRIOR_CHARGE,
    TEAM_A_ID,
    WARRIOR_CLASS_ID,
    Action,
    ActionMask,
    DoneFlags,
    EnvConfig,
    EnvState,
    Info,
    Observation,
    ResolvedAgentProfile,
    Reward,
)
from marl_battlegrounds.rendering import (
    ActivationVisual,
    ChargeTrailVisual,
    HealthDeltaVisual,
    RejectedActionVisual,
)
from marl_battlegrounds.rendering.visuals import ActivationKind
from scripts.dev.visual_debugger.model import (
    AcceptedActivation,
    ActionRejection,
    ActorTransition,
    DebuggerSession,
    SelectedTargetFacts,
    StatusChange,
    StatusKind,
    StatusTransition,
    SubmissionKind,
    TransientHistoryEntry,
    TransitionView,
)
from scripts.dev.visual_debugger.targeting import (
    global_slot_to_target_action,
    target_action_to_global_slot,
)

_MOVE_NAMES = (
    "Stay",
    "North",
    "South",
    "East",
    "West",
    "Northeast",
    "Northwest",
    "Southeast",
    "Southwest",
)
_CLASS_NAMES = ("Neutral", "Mage", "Warrior", "Hunter", "Rogue", "Priest")
_TEAM_NAMES = {1: "A", 2: "B"}
_CHARGE_TRAIL_OPACITY = (1.0, 0.65, 0.35)


def _move_name(move_action: int) -> str:
    if 0 <= move_action < len(_MOVE_NAMES):
        return _MOVE_NAMES[move_action]
    return "Invalid"


def _validate_active_slot(
    config: EnvConfig,
    global_slot: int,
    *,
    name: str,
) -> None:
    if not 0 <= global_slot < MAX_AGENT_SLOTS:
        msg = f"{name} must be in [0, {MAX_AGENT_SLOTS}); got {global_slot}."
        raise ValueError(msg)
    if not bool(config.agent_profile.active_mask[global_slot]):
        msg = f"{name} g{global_slot} is not active in this scenario."
        raise ValueError(msg)


def derive_selected_target_facts(
    *,
    config: EnvConfig,
    state: EnvState,
    observation: Observation,
    action_mask: ActionMask,
    controlled_global_slot: int,
    target_global_slot: int | None,
) -> SelectedTargetFacts | None:
    """Inspect one selected target without reconstructing mask or visibility logic."""
    _validate_active_slot(
        config,
        controlled_global_slot,
        name="controlled_global_slot",
    )
    if target_global_slot is None:
        return None
    _validate_active_slot(config, target_global_slot, name="target_global_slot")

    target_action = global_slot_to_target_action(
        controlled_global_slot,
        target_global_slot,
    )
    actor_team = int(config.agent_profile.team_ids[controlled_global_slot])
    target_team = int(config.agent_profile.team_ids[target_global_slot])
    if controlled_global_slot == target_global_slot:
        relation = "self"
    elif actor_team == target_team:
        relation = "ally"
    else:
        relation = "enemy"

    actor_position = state.agent_positions[controlled_global_slot]
    target_position = state.agent_positions[target_global_slot]
    distance = float(np.linalg.norm(np.asarray(target_position - actor_position)))
    clear_los = bool(
        has_clear_line_of_sight(
            actor_position,
            target_position,
            config.obstacles,
        )
    )

    observer_visible = observer_relative_visibility(
        config=config,
        observation=observation,
        observer_global_slot=controlled_global_slot,
        candidate_global_slot=target_global_slot,
    )

    observation_radius = float(
        config.agent_profile.observation_radii[controlled_global_slot]
    )
    basic_radius = float(
        config.agent_profile.basic_interaction_radii[controlled_global_slot]
    )
    ultimate_radius = float(
        config.agent_profile.ultimate_interaction_radii[controlled_global_slot]
    )
    class_id = int(config.agent_profile.class_ids[controlled_global_slot])
    ultimate_mode = int(get_ultimate_target_mode_by_class_ids(class_id))
    inside_ultimate_radius = (
        None
        if ultimate_mode == ONLY_NONE_TARGET_ULTIMATE_MODE or ultimate_radius <= 0
        else distance <= ultimate_radius
    )

    exact_lanes = action_mask.select_target_use_ultimate_joint_mask[
        controlled_global_slot,
        target_action,
    ]
    return SelectedTargetFacts(
        controlled_global_slot=controlled_global_slot,
        target_global_slot=target_global_slot,
        target_action=target_action,
        relation=relation,
        center_distance=distance,
        has_clear_line_of_sight=clear_los,
        observer_visible=observer_visible,
        inside_observation_radius=distance <= observation_radius,
        inside_basic_radius=distance <= basic_radius,
        inside_ultimate_radius=inside_ultimate_radius,
        lane_0_available=bool(exact_lanes[0]),
        lane_1_available=bool(exact_lanes[1]),
    )


def observer_relative_visibility(
    *,
    config: EnvConfig,
    observation: Observation,
    observer_global_slot: int,
    candidate_global_slot: int,
) -> bool:
    """Read one public relation-local visibility entry without reconstructing it."""
    _validate_active_slot(config, observer_global_slot, name="observer_global_slot")
    _validate_active_slot(config, candidate_global_slot, name="candidate_global_slot")
    observer_team = int(config.agent_profile.team_ids[observer_global_slot])
    candidate_team = int(config.agent_profile.team_ids[candidate_global_slot])
    same_team = observer_team == candidate_team
    if same_team:
        relation_row = (
            candidate_global_slot
            if observer_team == TEAM_A_ID
            else candidate_global_slot - MAX_AGENTS_PER_TEAM
        )
        return bool(
            observation.ally_visibility_mask[
                observer_global_slot,
                relation_row,
            ]
        )
    relation_row = (
        candidate_global_slot - MAX_AGENTS_PER_TEAM
        if observer_team == TEAM_A_ID
        else candidate_global_slot
    )
    return bool(
        observation.enemy_visibility_mask[
            observer_global_slot,
            relation_row,
        ]
    )


def accepted_action_from_successor(state: EnvState) -> Action:
    """Read the accepted joint action stored by the public successor contract."""
    if not bool(state.has_previous_timestep_joint_action):
        msg = "successor state does not contain a previous joint action."
        raise ValueError(msg)
    return Action(
        move=state.previous_timestep_move_actions,
        select_target=state.previous_timestep_select_target_actions,
        use_ultimate=state.previous_timestep_use_ultimate_actions,
    )


def _action_heads_in_domain(
    move_action: int,
    target_action: int,
    use_ultimate: int,
) -> bool:
    return (
        0 <= move_action < NUM_MOVE_ACTIONS
        and 0 <= target_action < NUM_TARGET_ACTIONS
        and 0 <= use_ultimate < NUM_ULTIMATE_ACTIONS
    )


def _accepted_activations(
    observation: Observation,
    accepted_action: Action,
) -> tuple[AcceptedActivation, ...]:
    activations: list[AcceptedActivation] = []
    active_mask = np.asarray(
        observation.self_features[:, AGENT_FEATURE_ACTIVE],
        dtype=bool,
    )
    for actor_slot in range(MAX_AGENT_SLOTS):
        if not active_mask[actor_slot]:
            continue
        target_action = int(accepted_action.select_target[actor_slot])
        use_ultimate = int(accepted_action.use_ultimate[actor_slot])
        class_id = int(observation.self_features[actor_slot, AGENT_FEATURE_CLASS_ID])
        target_slot = target_action_to_global_slot(actor_slot, target_action)
        if use_ultimate == 1:
            kind = cast(
                ActivationKind | None,
                {
                    MAGE_CLASS_ID: "mage_burst",
                    WARRIOR_CLASS_ID: "warrior_charge",
                    HUNTER_CLASS_ID: "hunter_trap",
                    ROGUE_CLASS_ID: "rogue_poison",
                    PRIEST_CLASS_ID: "holy_word",
                }.get(class_id),
            )
            if kind is not None:
                activations.append(
                    AcceptedActivation(
                        kind=kind,
                        source_global_slot=actor_slot,
                        target_global_slot=target_slot,
                        target_action=target_action,
                        use_ultimate=use_ultimate,
                    )
                )
        elif target_action != 0:
            kind = "basic_heal" if class_id == PRIEST_CLASS_ID else "basic_damage"
            activations.append(
                AcceptedActivation(
                    kind=kind,
                    source_global_slot=actor_slot,
                    target_global_slot=target_slot,
                    target_action=target_action,
                    use_ultimate=use_ultimate,
                )
            )
    return tuple(activations)


def _positive_damage_targets(
    activations: tuple[AcceptedActivation, ...],
) -> set[int]:
    return {
        activation.target_global_slot
        for activation in activations
        if activation.target_global_slot is not None
        and activation.kind in ("basic_damage", "warrior_charge")
    }


def _status_arrays(
    state: EnvState,
) -> tuple[tuple[StatusKind, int, np.ndarray], ...]:
    return (
        (
            "slow_warrior_charge",
            WARRIOR_CLASS_ID,
            np.asarray(state.slow_durations)[:, SLOW_CHANNEL_WARRIOR_CHARGE],
        ),
        (
            "slow_hunter_basic",
            HUNTER_CLASS_ID,
            np.asarray(state.slow_durations)[:, SLOW_CHANNEL_HUNTER_BASIC],
        ),
        (
            "slow_rogue_poison",
            ROGUE_CLASS_ID,
            np.asarray(state.slow_durations)[:, SLOW_CHANNEL_ROGUE_POISON],
        ),
        (
            "stun_warrior_charge",
            WARRIOR_CLASS_ID,
            np.asarray(state.stun_durations)[:, STUN_CHANNEL_WARRIOR_CHARGE],
        ),
        (
            "stun_hunter_trap",
            HUNTER_CLASS_ID,
            np.asarray(state.stun_durations)[:, STUN_CHANNEL_HUNTER_TRAP],
        ),
        (
            "stun_rogue_poison",
            ROGUE_CLASS_ID,
            np.asarray(state.stun_durations)[:, STUN_CHANNEL_ROGUE_POISON],
        ),
        (
            "anti_heal_rogue_poison",
            ROGUE_CLASS_ID,
            np.asarray(state.rogue_poison_anti_heal_durations),
        ),
        (
            "mage_burst",
            MAGE_CLASS_ID,
            np.asarray(state.mage_burst_damage_amplification_durations),
        ),
        (
            "priest_freedom",
            PRIEST_CLASS_ID,
            np.asarray(state.priest_blessing_of_freedom_slow_floor_durations),
        ),
    )


_STATUS_CATALOG_DURATIONS: dict[StatusKind, int] = {
    "slow_warrior_charge": WARRIOR_CHARGE_SLOW_DURATION_TICKS,
    "slow_hunter_basic": HUNTER_BASIC_SLOW_DURATION_TICKS,
    "slow_rogue_poison": ROGUE_POISON_SLOW_DURATION_TICKS,
    "stun_warrior_charge": WARRIOR_CHARGE_STUN_DURATION_TICKS,
    "stun_hunter_trap": HUNTER_TRAP_STUN_DURATION_TICKS,
    "stun_rogue_poison": ROGUE_POISON_STUN_DURATION_TICKS,
    "anti_heal_rogue_poison": ROGUE_POISON_ANTI_HEAL_DURATION_TICKS,
    "mage_burst": MAGE_BURST_DAMAGE_DURATION_TICKS,
    "priest_freedom": PRIEST_HEAL_SPEED_FLOOR_DURATION_TICKS,
}


def _classify_status_change(
    *,
    global_slot: int,
    status_kind: StatusKind,
    before: int,
    after: int,
    applications: set[tuple[int, StatusKind]],
    positive_damage_targets: set[int],
) -> StatusChange:
    has_application = (global_slot, status_kind) in applications
    if (
        has_application
        and before == 0
        and after == _STATUS_CATALOG_DURATIONS[status_kind]
    ):
        return "applied"
    if (
        has_application
        and before > 0
        and after == _STATUS_CATALOG_DURATIONS[status_kind]
    ):
        return "refreshed"
    if not has_application and before > 1 and after == before - 1:
        return "decremented"
    if (
        status_kind == "stun_hunter_trap"
        and not has_application
        and before > 1
        and after == 0
        and global_slot in positive_damage_targets
    ):
        return "trap_broken"
    if (
        status_kind == "stun_hunter_trap"
        and not has_application
        and before == 1
        and after == 0
        and global_slot in positive_damage_targets
    ):
        return "cleared_unclassified"
    if not has_application and before == 1 and after == 0:
        return "expired"
    if before == after and not has_application:
        return "unchanged"
    return "cleared_unclassified"


def extract_transition_view(
    *,
    scenario_name: str,
    submission_kind: SubmissionKind,
    report_actor_slots: tuple[int, ...],
    before_state: EnvState,
    before_observation: Observation,
    before_action_mask: ActionMask,
    submitted_action: Action,
    after_state: EnvState,
    after_observation: Observation,
    after_action_mask: ActionMask,
    reward: Reward,
    done_flags: DoneFlags,
    info: Info,
) -> TransitionView:
    """Retain public before/after facts and derive conservative host diagnostics."""
    accepted_action = accepted_action_from_successor(after_state)
    active_mask = np.asarray(
        before_observation.self_features[:, AGENT_FEATURE_ACTIVE],
        dtype=bool,
    )
    actor_transitions: list[ActorTransition] = []
    rejections: list[ActionRejection] = []

    for actor_slot in range(MAX_AGENT_SLOTS):
        if not active_mask[actor_slot]:
            continue
        submitted_move = int(submitted_action.move[actor_slot])
        submitted_target = int(submitted_action.select_target[actor_slot])
        submitted_ultimate = int(submitted_action.use_ultimate[actor_slot])
        in_domain = _action_heads_in_domain(
            submitted_move,
            submitted_target,
            submitted_ultimate,
        )
        move_mask_value = (
            bool(before_action_mask.move_mask[actor_slot, submitted_move])
            if 0 <= submitted_move < NUM_MOVE_ACTIONS
            else False
        )
        if 0 <= submitted_target < NUM_TARGET_ACTIONS:
            lanes = before_action_mask.select_target_use_ultimate_joint_mask[
                actor_slot,
                submitted_target,
            ]
            lane_0 = bool(lanes[0])
            lane_1 = bool(lanes[1])
        else:
            lane_0 = False
            lane_1 = False
        pair_mask_value = (
            bool(
                before_action_mask.select_target_use_ultimate_joint_mask[
                    actor_slot,
                    submitted_target,
                    submitted_ultimate,
                ]
            )
            if 0 <= submitted_target < NUM_TARGET_ACTIONS
            and 0 <= submitted_ultimate < NUM_ULTIMATE_ACTIONS
            else False
        )

        accepted_move = int(accepted_action.move[actor_slot])
        accepted_target = int(accepted_action.select_target[actor_slot])
        accepted_ultimate = int(accepted_action.use_ultimate[actor_slot])
        movement_accepted = (
            in_domain and move_mask_value and submitted_move == accepted_move
        )
        combat_pair_accepted = (
            in_domain
            and pair_mask_value
            and submitted_target == accepted_target
            and submitted_ultimate == accepted_ultimate
        )

        before_position = np.asarray(
            before_state.agent_positions[actor_slot],
            dtype=np.float32,
        )
        after_position = np.asarray(
            after_state.agent_positions[actor_slot],
            dtype=np.float32,
        )
        displacement = after_position - before_position
        before_health = float(before_state.current_health[actor_slot])
        after_health = float(after_state.current_health[actor_slot])

        actor_transitions.append(
            ActorTransition(
                actor_global_slot=actor_slot,
                submitted_move_action=submitted_move,
                submitted_target_action=submitted_target,
                submitted_use_ultimate=submitted_ultimate,
                accepted_move_action=accepted_move,
                accepted_target_action=accepted_target,
                accepted_use_ultimate=accepted_ultimate,
                submitted_tuple_in_domain=in_domain,
                submitted_move_mask_value=move_mask_value,
                submitted_lane_0_value=lane_0,
                submitted_lane_1_value=lane_1,
                submitted_pair_mask_value=pair_mask_value,
                movement_accepted=movement_accepted,
                combat_pair_accepted=combat_pair_accepted,
                position_before=(
                    float(before_position[0]),
                    float(before_position[1]),
                ),
                position_after=(
                    float(after_position[0]),
                    float(after_position[1]),
                ),
                realized_displacement=(
                    float(displacement[0]),
                    float(displacement[1]),
                ),
                health_before=before_health,
                health_after=after_health,
                net_health_delta=after_health - before_health,
                cooldown_before=int(before_state.ultimate_cooldowns[actor_slot]),
                cooldown_after=int(after_state.ultimate_cooldowns[actor_slot]),
                effective_speed_before=float(
                    before_observation.self_features[
                        actor_slot,
                        AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED,
                    ]
                ),
                effective_speed_after=float(
                    after_observation.self_features[
                        actor_slot,
                        AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED,
                    ]
                ),
                mage_aura_before=float(
                    before_observation.self_features[
                        actor_slot,
                        AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER,
                    ]
                ),
                mage_aura_after=float(
                    after_observation.self_features[
                        actor_slot,
                        AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER,
                    ]
                ),
                warrior_aura_before=float(
                    before_observation.self_features[
                        actor_slot,
                        AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER,
                    ]
                ),
                warrior_aura_after=float(
                    after_observation.self_features[
                        actor_slot,
                        AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER,
                    ]
                ),
            )
        )

        if not in_domain:
            rejections.append(
                ActionRejection(
                    actor_global_slot=actor_slot,
                    component="complete_tuple_domain",
                    submitted_move_action=submitted_move,
                    submitted_target_action=submitted_target,
                    submitted_use_ultimate=submitted_ultimate,
                    movement_mask_value=move_mask_value,
                    pair_mask_value=pair_mask_value,
                )
            )
        else:
            if not movement_accepted:
                rejections.append(
                    ActionRejection(
                        actor_global_slot=actor_slot,
                        component="movement",
                        submitted_move_action=submitted_move,
                        submitted_target_action=submitted_target,
                        submitted_use_ultimate=submitted_ultimate,
                        movement_mask_value=move_mask_value,
                        pair_mask_value=pair_mask_value,
                    )
                )
            if not combat_pair_accepted:
                rejections.append(
                    ActionRejection(
                        actor_global_slot=actor_slot,
                        component="combat",
                        submitted_move_action=submitted_move,
                        submitted_target_action=submitted_target,
                        submitted_use_ultimate=submitted_ultimate,
                        movement_mask_value=move_mask_value,
                        pair_mask_value=pair_mask_value,
                    )
                )

    activations = _accepted_activations(before_observation, accepted_action)
    applications = _status_applications_from_observation(
        before_observation,
        activations,
    )
    positive_damage_targets = _positive_damage_targets(activations)
    before_statuses = _status_arrays(before_state)
    after_statuses = _status_arrays(after_state)
    status_transitions: list[StatusTransition] = []
    for (kind, source_class, before_values), (
        after_kind,
        _,
        after_values,
    ) in zip(before_statuses, after_statuses, strict=True):
        if kind != after_kind:
            raise AssertionError("status table order drifted")
        for global_slot in range(MAX_AGENT_SLOTS):
            if not active_mask[global_slot]:
                continue
            before = int(before_values[global_slot])
            after = int(after_values[global_slot])
            change = _classify_status_change(
                global_slot=global_slot,
                status_kind=kind,
                before=before,
                after=after,
                applications=applications,
                positive_damage_targets=positive_damage_targets,
            )
            status_transitions.append(
                StatusTransition(
                    global_slot=global_slot,
                    status_kind=kind,
                    source_class_id=source_class,
                    duration_before=before,
                    duration_after=after,
                    change=change,
                )
            )

    return TransitionView(
        scenario_name=scenario_name,
        submission_kind=submission_kind,
        report_actor_slots=report_actor_slots,
        before_state=before_state,
        before_observation=before_observation,
        before_action_mask=before_action_mask,
        submitted_action=submitted_action,
        accepted_action=accepted_action,
        after_state=after_state,
        after_observation=after_observation,
        after_action_mask=after_action_mask,
        reward=reward,
        done_flags=done_flags,
        info=info,
        actor_transitions=tuple(actor_transitions),
        status_transitions=tuple(status_transitions),
        accepted_activations=activations,
        rejections=tuple(rejections),
    )


def _profile_from_observation(observation: Observation) -> ResolvedAgentProfile:
    """Build the resolved profile fields exposed in self observation rows."""
    values = observation.self_features
    return ResolvedAgentProfile(
        class_ids=values[:, AGENT_FEATURE_CLASS_ID].astype(np.int32),
        team_ids=values[:, AGENT_FEATURE_TEAM_ID].astype(np.int32),
        active_mask=values[:, AGENT_FEATURE_ACTIVE].astype(bool),
        agent_radii=values[:, AGENT_FEATURE_RADIUS],
        base_movement_speeds=values[:, AGENT_FEATURE_BASE_MOVEMENT_SPEED],
        observation_radii=values[:, AGENT_FEATURE_OBSERVATION_RADIUS],
        basic_interaction_radii=values[:, AGENT_FEATURE_BASIC_INTERACTION_RADIUS],
        ultimate_interaction_radii=values[
            :,
            AGENT_FEATURE_ULTIMATE_INTERACTION_RADIUS,
        ],
        max_health=values[:, AGENT_FEATURE_MAX_HEALTH],
    )


def _status_applications_from_observation(
    observation: Observation,
    activations: tuple[AcceptedActivation, ...],
) -> set[tuple[int, StatusKind]]:
    applications: set[tuple[int, StatusKind]] = set()
    for activation in activations:
        source_class = int(
            observation.self_features[
                activation.source_global_slot,
                AGENT_FEATURE_CLASS_ID,
            ]
        )
        target = activation.target_global_slot
        if activation.kind == "mage_burst":
            applications.add((activation.source_global_slot, "mage_burst"))
        elif activation.kind == "warrior_charge" and target is not None:
            applications.update(
                ((target, "slow_warrior_charge"), (target, "stun_warrior_charge"))
            )
        elif activation.kind == "hunter_trap" and target is not None:
            applications.add((target, "stun_hunter_trap"))
        elif activation.kind == "rogue_poison" and target is not None:
            applications.update(
                (
                    (target, "slow_rogue_poison"),
                    (target, "stun_rogue_poison"),
                    (target, "anti_heal_rogue_poison"),
                )
            )
        elif (
            activation.kind == "basic_damage"
            and source_class == HUNTER_CLASS_ID
            and target is not None
        ):
            applications.add((target, "slow_hunter_basic"))
        elif activation.kind == "basic_heal" and target is not None:
            applications.add((target, "priest_freedom"))
    return applications


def derive_transient_entries(
    transition: TransitionView,
    *,
    first_sequence_number: int,
) -> tuple[TransientHistoryEntry, ...]:
    """Describe latest public deltas, activations, rejections, and Charge trails."""
    entries: list[TransientHistoryEntry] = []
    sequence_number = first_sequence_number
    for actor in transition.actor_transitions:
        if actor.net_health_delta != 0.0:
            entries.append(
                TransientHistoryEntry(
                    visual=HealthDeltaVisual(
                        global_slot=actor.actor_global_slot,
                        net_delta=actor.net_health_delta,
                    ),
                    created_after_step=int(transition.after_state.step_count),
                    age_submitted_steps=0,
                    max_age_submitted_steps=1,
                    sequence_number=sequence_number,
                )
            )
            sequence_number += 1

    actor_by_slot = {
        actor.actor_global_slot: actor for actor in transition.actor_transitions
    }
    class_ids = np.asarray(
        transition.before_observation.self_features[:, AGENT_FEATURE_CLASS_ID],
        dtype=np.int32,
    )
    for activation in transition.accepted_activations:
        entries.append(
            TransientHistoryEntry(
                visual=ActivationVisual(
                    kind=activation.kind,
                    source_global_slot=activation.source_global_slot,
                    target_global_slot=activation.target_global_slot,
                    source_class_id=int(class_ids[activation.source_global_slot]),
                ),
                created_after_step=int(transition.after_state.step_count),
                age_submitted_steps=0,
                max_age_submitted_steps=1,
                sequence_number=sequence_number,
            )
        )
        sequence_number += 1
        if (
            activation.kind == "warrior_charge"
            and activation.target_global_slot is not None
        ):
            actor = actor_by_slot[activation.source_global_slot]
            path_kind = (
                "charge_only"
                if actor.accepted_move_action == MOVE_STAY
                else "combined_charge_and_movement"
            )
            entries.append(
                TransientHistoryEntry(
                    visual=ChargeTrailVisual(
                        source_global_slot=activation.source_global_slot,
                        start=actor.position_before,
                        end=actor.position_after,
                        target_global_slot=activation.target_global_slot,
                        path_kind=path_kind,
                        opacity=1.0,
                    ),
                    created_after_step=int(transition.after_state.step_count),
                    age_submitted_steps=0,
                    max_age_submitted_steps=3,
                    sequence_number=sequence_number,
                )
            )
            sequence_number += 1

    for rejection in transition.rejections:
        target_slot = (
            target_action_to_global_slot(
                rejection.actor_global_slot,
                rejection.submitted_target_action,
            )
            if 0 <= rejection.submitted_target_action < NUM_TARGET_ACTIONS
            else None
        )
        lane = (
            rejection.submitted_use_ultimate
            if rejection.submitted_use_ultimate in (0, 1)
            else None
        )
        entries.append(
            TransientHistoryEntry(
                visual=RejectedActionVisual(
                    actor_global_slot=rejection.actor_global_slot,
                    component=rejection.component,
                    target_global_slot=target_slot,
                    lane=lane,
                ),
                created_after_step=int(transition.after_state.step_count),
                age_submitted_steps=0,
                max_age_submitted_steps=1,
                sequence_number=sequence_number,
            )
        )
        sequence_number += 1
    return tuple(entries)


def age_transient_history(
    entries: tuple[TransientHistoryEntry, ...],
) -> tuple[TransientHistoryEntry, ...]:
    """Age once for a submitted step, update Charge opacity, and expire entries."""
    aged: list[TransientHistoryEntry] = []
    for entry in entries:
        next_age = entry.age_submitted_steps + 1
        if next_age >= entry.max_age_submitted_steps:
            continue
        visual = entry.visual
        if isinstance(visual, ChargeTrailVisual):
            visual = replace(
                visual,
                opacity=_CHARGE_TRAIL_OPACITY[next_age],
            )
        aged.append(
            replace(
                entry,
                visual=visual,
                age_submitted_steps=next_age,
            )
        )
    return tuple(sorted(aged, key=lambda entry: entry.sequence_number))


def _actor_transition(
    transition: TransitionView,
    actor_slot: int,
) -> ActorTransition:
    return next(
        actor
        for actor in transition.actor_transitions
        if actor.actor_global_slot == actor_slot
    )


def _target_label(actor_slot: int, target_action: int) -> str:
    if not 0 <= target_action < NUM_TARGET_ACTIONS:
        return f"invalid/t{target_action}"
    target_slot = target_action_to_global_slot(actor_slot, target_action)
    return (
        f"none/t{target_action}"
        if target_slot is None
        else f"g{target_slot}/t{target_action}"
    )


def format_concise_transition(transition: TransitionView) -> str:
    """Format stable human-readable lines without raw tensor dumps."""
    lines = [
        (
            f"STEP scenario={transition.scenario_name} "
            f"{int(transition.before_state.step_count)}"
            f"->{int(transition.after_state.step_count)} "
            f"terminated={int(bool(transition.done_flags.terminated))} "
            f"truncated={int(bool(transition.done_flags.truncated))}"
        )
    ]
    class_ids = np.asarray(
        transition.before_observation.self_features[:, AGENT_FEATURE_CLASS_ID],
        dtype=np.int32,
    )
    team_ids = np.asarray(
        transition.before_observation.self_features[:, AGENT_FEATURE_TEAM_ID],
        dtype=np.int32,
    )
    for actor_slot in transition.report_actor_slots:
        actor = _actor_transition(transition, actor_slot)
        submitted_kind = "Ultimate" if actor.submitted_use_ultimate == 1 else "Basic"
        combat_label = (
            "CANONICAL-NOOP"
            if actor.submitted_target_action == 0
            and actor.submitted_use_ultimate == 0
            and actor.combat_pair_accepted
            else "ACCEPTED"
            if actor.combat_pair_accepted
            else "REJECTED"
        )
        lines.append(
            f"ACTOR g{actor_slot} "
            f"{_TEAM_NAMES[int(team_ids[actor_slot])]}/"
            f"{_CLASS_NAMES[int(class_ids[actor_slot])]} "
            f"target={_target_label(actor_slot, actor.submitted_target_action)} "
            f"lanes=0:{int(actor.submitted_lane_0_value)},"
            f"1:{int(actor.submitted_lane_1_value)} "
            f"submitted={_move_name(actor.submitted_move_action)}"
            f"[{actor.submitted_move_action}],"
            f"{submitted_kind}[{actor.submitted_use_ultimate}] "
            f"accepted={_move_name(actor.accepted_move_action)}"
            f"[{actor.accepted_move_action}],"
            f"t{actor.accepted_target_action},"
            f"u{actor.accepted_use_ultimate} "
            f"movement={'ACCEPTED' if actor.movement_accepted else 'REJECTED'} "
            f"combat={combat_label}"
        )
        if actor.realized_displacement != (0.0, 0.0):
            lines.append(
                f"POSITION g{actor_slot} "
                f"({actor.position_before[0]:.2f},{actor.position_before[1]:.2f})"
                f"->({actor.position_after[0]:.2f},{actor.position_after[1]:.2f}) "
                f"delta=({actor.realized_displacement[0]:+.2f},"
                f"{actor.realized_displacement[1]:+.2f})"
            )

    activation_targets = {
        activation.target_global_slot
        for activation in transition.accepted_activations
        if activation.target_global_slot is not None
    }
    health_capable_kinds = {
        "basic_damage",
        "basic_heal",
        "holy_word",
        "warrior_charge",
    }
    contributions: dict[int, list[str]] = {}
    for activation in transition.accepted_activations:
        if (
            activation.target_global_slot is not None
            and activation.kind in health_capable_kinds
        ):
            contributions.setdefault(activation.target_global_slot, []).append(
                activation.kind
            )

    for actor in transition.actor_transitions:
        if actor.net_health_delta != 0.0 or (
            actor.actor_global_slot in activation_targets
            and actor.actor_global_slot in contributions
        ):
            caveat = (
                " gross contributions not exposed"
                if len(contributions.get(actor.actor_global_slot, ())) > 1
                or (
                    actor.actor_global_slot in contributions
                    and actor.net_health_delta == 0.0
                )
                else ""
            )
            lines.append(
                f"HEALTH g{actor.actor_global_slot} "
                f"{actor.health_before:.2f}->{actor.health_after:.2f} "
                f"net={actor.net_health_delta:+.2f}{caveat}"
            )

        if actor.cooldown_before != actor.cooldown_after:
            if actor.cooldown_before == 0 and actor.cooldown_after > 0:
                change = "started"
            elif actor.cooldown_after == 0:
                change = "expired"
            else:
                change = "decremented"
            lines.append(
                f"COOLDOWN g{actor.actor_global_slot} "
                f"{actor.cooldown_before}->{actor.cooldown_after} {change}"
            )

    for status in transition.status_transitions:
        if status.change == "unchanged":
            continue
        lines.append(
            f"STATUS g{status.global_slot} {status.status_kind} "
            f"{status.duration_before}->{status.duration_after} {status.change}"
        )

    for activation in transition.accepted_activations:
        recipient = (
            "none"
            if activation.target_global_slot is None
            else f"g{activation.target_global_slot}"
        )
        detail = ""
        if activation.kind == "warrior_charge":
            actor = _actor_transition(transition, activation.source_global_slot)
            path = (
                "charge_only"
                if actor.accepted_move_action == MOVE_STAY
                else "combined_charge_and_movement"
            )
            detail = f" path={path}"
            if path == "combined_charge_and_movement":
                detail += f" submitted_move={_move_name(actor.submitted_move_action)}"
        lines.append(
            f"EVENT {activation.kind} source=g{activation.source_global_slot} "
            f"recipient={recipient}{detail}"
        )
    for rejection in transition.rejections:
        rejection_mask_value = (
            rejection.movement_mask_value
            if rejection.component == "movement"
            else rejection.pair_mask_value
        )
        lines.append(
            f"EVENT rejection component={rejection.component} "
            f"actor=g{rejection.actor_global_slot} "
            f"mask={int(rejection_mask_value)}"
        )
    return "\n".join(lines)


def format_verbose_transition(transition: TransitionView) -> str:
    """Add independent geometry, mask, aura, speed, and episode facts."""
    lines = [format_concise_transition(transition)]
    for actor_slot in transition.report_actor_slots:
        actor = _actor_transition(transition, actor_slot)
        target_in_domain = 0 <= actor.submitted_target_action < NUM_TARGET_ACTIONS
        target_slot = (
            target_action_to_global_slot(
                actor_slot,
                actor.submitted_target_action,
            )
            if target_in_domain
            else None
        )
        if not target_in_domain:
            lines.extend(
                (
                    (
                        f"TARGET actor=g{actor_slot} global=invalid relative="
                        f"t{actor.submitted_target_action} relation=n/a distance=n/a"
                    ),
                    (
                        f"GEOMETRY actor=g{actor_slot} target=invalid los=n/a "
                        "visible=n/a observation_range=n/a basic_range=n/a "
                        "ultimate_range=n/a"
                    ),
                )
            )
        elif target_slot is None:
            lines.extend(
                (
                    (
                        f"TARGET actor=g{actor_slot} global=none relative="
                        f"t{actor.submitted_target_action} relation=n/a distance=n/a"
                    ),
                    (
                        f"GEOMETRY actor=g{actor_slot} target=none los=n/a "
                        "visible=n/a observation_range=n/a basic_range=n/a "
                        "ultimate_range=n/a"
                    ),
                )
            )
        else:
            facts = derive_selected_target_facts(
                config=_config_from_transition(transition),
                state=transition.before_state,
                observation=transition.before_observation,
                action_mask=transition.before_action_mask,
                controlled_global_slot=actor_slot,
                target_global_slot=target_slot,
            )
            if facts is None:
                raise AssertionError("non-none target unexpectedly produced no facts")
            ultimate = (
                "n/a"
                if facts.inside_ultimate_radius is None
                else str(int(facts.inside_ultimate_radius))
            )
            lines.extend(
                (
                    (
                        f"TARGET actor=g{actor_slot} global=g{target_slot} "
                        f"relative=t{facts.target_action} relation={facts.relation} "
                        f"distance={facts.center_distance:.2f}"
                    ),
                    (
                        f"GEOMETRY actor=g{actor_slot} target=g{target_slot} "
                        f"los={int(facts.has_clear_line_of_sight)} "
                        f"visible={int(facts.observer_visible)} "
                        f"observation_range={int(facts.inside_observation_radius)} "
                        f"basic_range={int(facts.inside_basic_radius)} "
                        f"ultimate_range={ultimate}"
                    ),
                )
            )
        lines.extend(
            (
                (
                    f"MASK actor=g{actor_slot} "
                    f"move[{actor.submitted_move_action}]="
                    f"{int(actor.submitted_move_mask_value)} "
                    f"target=t{actor.submitted_target_action} "
                    f"lane0={int(actor.submitted_lane_0_value)} "
                    f"lane1={int(actor.submitted_lane_1_value)} "
                    f"submitted_pair={int(actor.submitted_pair_mask_value)} "
                    f"tuple_in_domain={int(actor.submitted_tuple_in_domain)}"
                ),
                (
                    f"SUBMITTED actor=g{actor_slot} "
                    f"move={_move_name(actor.submitted_move_action)}"
                    f"[{actor.submitted_move_action}] "
                    "target="
                    f"{_target_label(actor_slot, actor.submitted_target_action)} "
                    f"use_ultimate={actor.submitted_use_ultimate}"
                ),
                (
                    f"ACCEPTED actor=g{actor_slot} "
                    f"move={_move_name(actor.accepted_move_action)}"
                    f"[{actor.accepted_move_action}] "
                    f"target=t{actor.accepted_target_action} "
                    f"use_ultimate={actor.accepted_use_ultimate}"
                ),
                (
                    f"AURA actor=g{actor_slot} "
                    f"mage={actor.mage_aura_before:.2f}"
                    f"->{actor.mage_aura_after:.2f} "
                    f"warrior={actor.warrior_aura_before:.2f}"
                    f"->{actor.warrior_aura_after:.2f}"
                ),
                (
                    f"SPEED actor=g{actor_slot} "
                    f"{actor.effective_speed_before:.2f}"
                    f"->{actor.effective_speed_after:.2f}"
                ),
                (
                    f"EPISODE reward="
                    f"{float(transition.reward.rewards[actor_slot]):+.2f} "
                    f"terminated={int(bool(transition.done_flags.terminated))} "
                    f"truncated={int(bool(transition.done_flags.truncated))}"
                ),
            )
        )
    return "\n".join(lines)


def _config_from_transition(transition: TransitionView) -> EnvConfig:
    context = transition.before_observation.context_features[0]
    return EnvConfig(
        max_steps=int(context[CONTEXT_FEATURE_EPISODE_HORIZON]),
        map_width=float(context[CONTEXT_FEATURE_MAP_WIDTH]),
        map_height=float(context[CONTEXT_FEATURE_MAP_HEIGHT]),
        obstacles=transition.before_observation.map_obstacle_features[0],
        agent_profile=_profile_from_observation(transition.before_observation),
        initial_agent_positions=transition.before_state.agent_positions,
        ordinary_movement_distance_scale=1.0,
    )


def format_reset(session: DebuggerSession) -> str:
    """Format one stable reset/scenario-switch line."""
    actor = session.controlled_global_slot
    class_id = int(session.config.agent_profile.class_ids[actor])
    team_id = int(session.config.agent_profile.team_ids[actor])
    active_count = int(np.sum(np.asarray(session.config.agent_profile.active_mask)))
    return (
        f"RESET scenario={session.scenario_name} seed={session.seed} "
        f"step={int(session.state.step_count)} controlled=g{actor} "
        f"class={_CLASS_NAMES[class_id]} team={_TEAM_NAMES[team_id]} "
        f"active={active_count} "
        f"movement_scale={session.config.ordinary_movement_distance_scale:.1f}"
    )
