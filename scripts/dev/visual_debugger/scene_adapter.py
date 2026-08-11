"""Allowlisted debugger-session adapters for renderer-neutral scene contracts."""

from typing import cast

import numpy as np
import numpy.typing as npt

from marl_battlegrounds.core.axis_mappings import observation_relation_and_row
from marl_battlegrounds.core.combat import HUNTER_TRAP_STUN_DURATION_TICKS
from marl_battlegrounds.core.types import (
    AGENT_FEATURE_ACTIVE,
    AGENT_FEATURE_ALIVE,
    AGENT_FEATURE_ANTI_HEAL_ROGUE_POISON_DURATION,
    AGENT_FEATURE_ANTI_HEAL_ROGUE_POISON_MULTIPLIER,
    AGENT_FEATURE_BASIC_INTERACTION_RADIUS,
    AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_AURA_RADIUS,
    AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_BURST_MULTIPLIER,
    AGENT_FEATURE_CAPABILITY_DAMAGE_MITIGATION_WARRIOR_AURA_RADIUS,
    AGENT_FEATURE_CLASS_ID,
    AGENT_FEATURE_CURRENT_HEALTH,
    AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER,
    AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_BURST_DURATION,
    AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER,
    AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED,
    AGENT_FEATURE_MAX_HEALTH,
    AGENT_FEATURE_OBSERVATION_RADIUS,
    AGENT_FEATURE_RADIUS,
    AGENT_FEATURE_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_DURATION,
    AGENT_FEATURE_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_FRACTION,
    AGENT_FEATURE_SLOW_HUNTER_BASIC_DURATION,
    AGENT_FEATURE_SLOW_ROGUE_POISON_DURATION,
    AGENT_FEATURE_SLOW_WARRIOR_CHARGE_DURATION,
    AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION,
    AGENT_FEATURE_STUN_ROGUE_POISON_DURATION,
    AGENT_FEATURE_STUN_WARRIOR_CHARGE_DURATION,
    AGENT_FEATURE_TEAM_ID,
    AGENT_FEATURE_ULTIMATE_COOLDOWN_REMAINING,
    AGENT_FEATURE_ULTIMATE_INTERACTION_RADIUS,
    AGENT_FEATURE_X,
    AGENT_FEATURE_Y,
    CONTEXT_FEATURE_MAP_HEIGHT,
    CONTEXT_FEATURE_MAP_WIDTH,
    HUNTER_CLASS_ID,
    MAGE_CLASS_ID,
    MAX_AGENT_SLOTS,
    MAX_OBSTACLE_SLOTS,
    MOVE_STAY,
    NUM_TARGET_ACTIONS,
    OBSTACLE_FEATURE_ACTIVE,
    OBSTACLE_FEATURE_HEIGHT,
    OBSTACLE_FEATURE_RADIUS,
    OBSTACLE_FEATURE_THETA,
    OBSTACLE_FEATURE_TYPE,
    OBSTACLE_FEATURE_WIDTH,
    OBSTACLE_FEATURE_X,
    OBSTACLE_FEATURE_Y,
    OBSTACLE_TYPE_PILLAR,
    OBSTACLE_TYPE_WALL,
    WARRIOR_CLASS_ID,
    EnvConfig,
    Observation,
)
from marl_battlegrounds.rendering.scene import (
    EVENT_SCHEMA_VERSION,
    SCENE_SCHEMA_VERSION,
    AcceptedActivationEventV1,
    AgentSceneV1,
    AuraFieldSceneV1,
    BattlefieldSceneV1,
    ChargeDisplacementEventV1,
    MapSceneV1,
    ModifierSceneV1,
    NetHealthEventV1,
    ObserverVisibilitySceneV1,
    ObstacleSceneV1,
    PendingRouteSceneV1,
    RangeSceneV1,
    RejectedActionEventV1,
    SceneAudience,
    SelectedLegalitySceneV1,
    SelectionSceneV1,
    StatusLifecycleEventV1,
    StatusSceneV1,
    VisualEventBatchV1,
)
from marl_battlegrounds.rendering.scene import Lane as SceneLane
from marl_battlegrounds.rendering.vocabulary import (
    lookup_modifier_token,
    lookup_status_token,
)
from scripts.dev.visual_debugger.diagnostics import (
    activation_uses_successor_anchors,
    latest_visual_event_batch,
    observer_relative_visibility,
    visual_event_id,
)
from scripts.dev.visual_debugger.model import DebuggerSession
from scripts.dev.visual_debugger.targeting import (
    global_slot_to_target_action,
    target_action_to_global_slot,
)

type _FeatureRow = npt.NDArray[np.float32]

_STATUS_FEATURES: tuple[tuple[str, int], ...] = (
    ("stun_warrior_charge", AGENT_FEATURE_STUN_WARRIOR_CHARGE_DURATION),
    ("stun_hunter_trap", AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION),
    ("stun_rogue_poison", AGENT_FEATURE_STUN_ROGUE_POISON_DURATION),
    ("slow_warrior_charge", AGENT_FEATURE_SLOW_WARRIOR_CHARGE_DURATION),
    ("slow_hunter_basic", AGENT_FEATURE_SLOW_HUNTER_BASIC_DURATION),
    ("slow_rogue_poison", AGENT_FEATURE_SLOW_ROGUE_POISON_DURATION),
    ("anti_heal_rogue_poison", AGENT_FEATURE_ANTI_HEAL_ROGUE_POISON_DURATION),
    (
        "priest_freedom",
        AGENT_FEATURE_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_DURATION,
    ),
    ("mage_burst", AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_BURST_DURATION),
)


def _candidate_row(
    *,
    config: EnvConfig,
    observation: Observation,
    observer_global_slot: int,
    candidate_global_slot: int,
) -> _FeatureRow | None:
    if candidate_global_slot == observer_global_slot:
        return np.asarray(
            observation.self_features[observer_global_slot],
            dtype=np.float32,
        )
    if not observer_relative_visibility(
        config=config,
        observation=observation,
        observer_global_slot=observer_global_slot,
        candidate_global_slot=candidate_global_slot,
    ):
        return None
    relation, row = observation_relation_and_row(
        observer_global_slot,
        candidate_global_slot,
    )
    values = (
        observation.ally_unit_features[observer_global_slot, row]
        if relation == "ally"
        else observation.enemy_unit_features[observer_global_slot, row]
    )
    return np.asarray(values, dtype=np.float32)


def _status_scenes(row: _FeatureRow) -> tuple[StatusSceneV1, ...]:
    statuses: list[StatusSceneV1] = []
    for token_id, feature in _STATUS_FEATURES:
        duration = int(row[feature])
        if duration <= 0:
            continue
        definition = lookup_status_token(token_id)
        if definition.source_class_id is None:
            raise AssertionError(f"status {token_id!r} lacks a source class")
        statuses.append(
            StatusSceneV1(
                token_id=token_id,
                duration=duration,
                source_class_id=definition.source_class_id,
                label=definition.label,
                short_label=definition.short_label,
                accessible_name=definition.accessible_name,
                priority=definition.priority,
            )
        )
    return tuple(statuses)


def _modifier_scene(token_id: str, multiplier: float) -> ModifierSceneV1:
    definition = lookup_modifier_token(token_id)
    return ModifierSceneV1(
        token_id=token_id,
        multiplier=float(multiplier),
        label=definition.label,
        accessible_name=definition.accessible_name,
    )


def _modifier_scenes(row: _FeatureRow) -> tuple[ModifierSceneV1, ...]:
    modifiers: list[ModifierSceneV1] = []
    mage_aura = float(row[AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER])
    if mage_aura != 1.0:
        modifiers.append(_modifier_scene("mage_amplification", mage_aura))
    warrior_aura = float(row[AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER])
    if warrior_aura != 1.0:
        modifiers.append(_modifier_scene("warrior_mitigation", warrior_aura))
    if int(row[AGENT_FEATURE_ANTI_HEAL_ROGUE_POISON_DURATION]) > 0:
        modifiers.append(
            _modifier_scene(
                "rogue_anti_heal",
                float(row[AGENT_FEATURE_ANTI_HEAL_ROGUE_POISON_MULTIPLIER]),
            )
        )
    if int(row[AGENT_FEATURE_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_DURATION]) > 0:
        modifiers.append(
            _modifier_scene(
                "priest_freedom",
                float(
                    row[AGENT_FEATURE_SLOW_FLOOR_PRIEST_BLESSING_OF_FREEDOM_FRACTION]
                ),
            )
        )
    if int(row[AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_BURST_DURATION]) > 0:
        modifiers.append(
            _modifier_scene(
                "mage_burst",
                float(
                    row[
                        AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_BURST_MULTIPLIER
                    ]
                ),
            )
        )
    return tuple(modifiers)


def _agent_scene(global_slot: int, row: _FeatureRow) -> AgentSceneV1:
    return AgentSceneV1(
        global_slot=global_slot,
        team_id=int(row[AGENT_FEATURE_TEAM_ID]),
        class_id=int(row[AGENT_FEATURE_CLASS_ID]),
        position=(
            float(row[AGENT_FEATURE_X]),
            float(row[AGENT_FEATURE_Y]),
        ),
        radius=float(row[AGENT_FEATURE_RADIUS]),
        active=bool(row[AGENT_FEATURE_ACTIVE]),
        alive=bool(row[AGENT_FEATURE_ALIVE]),
        current_health=float(row[AGENT_FEATURE_CURRENT_HEALTH]),
        max_health=float(row[AGENT_FEATURE_MAX_HEALTH]),
        ultimate_cooldown=int(row[AGENT_FEATURE_ULTIMATE_COOLDOWN_REMAINING]),
        effective_speed=float(row[AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED]),
        statuses=_status_scenes(row),
        modifiers=_modifier_scenes(row),
    )


def _obstacle_scenes(rows: npt.NDArray[np.float32]) -> tuple[ObstacleSceneV1, ...]:
    obstacles: list[ObstacleSceneV1] = []
    for obstacle_slot in range(MAX_OBSTACLE_SLOTS):
        row = rows[obstacle_slot]
        if not bool(row[OBSTACLE_FEATURE_ACTIVE] == 1.0):
            continue
        obstacle_type = int(row[OBSTACLE_FEATURE_TYPE])
        center = (
            float(row[OBSTACLE_FEATURE_X]),
            float(row[OBSTACLE_FEATURE_Y]),
        )
        if obstacle_type == OBSTACLE_TYPE_PILLAR:
            obstacles.append(
                ObstacleSceneV1(
                    obstacle_id=f"obstacle-{obstacle_slot}",
                    kind="pillar",
                    center=center,
                    radius=float(row[OBSTACLE_FEATURE_RADIUS]),
                )
            )
        elif obstacle_type == OBSTACLE_TYPE_WALL:
            obstacles.append(
                ObstacleSceneV1(
                    obstacle_id=f"obstacle-{obstacle_slot}",
                    kind="wall",
                    center=center,
                    width=float(row[OBSTACLE_FEATURE_WIDTH]),
                    height=float(row[OBSTACLE_FEATURE_HEIGHT]),
                    theta=float(row[OBSTACLE_FEATURE_THETA]),
                )
            )
    return tuple(obstacles)


def _map_scene(
    *,
    session: DebuggerSession,
    audience: SceneAudience,
) -> MapSceneV1:
    controlled = session.controlled_global_slot
    if audience == "researcher":
        width = float(session.config.map_width)
        height = float(session.config.map_height)
        obstacle_rows = np.asarray(session.config.obstacles, dtype=np.float32)
    else:
        context = session.observation.context_features[controlled]
        width = float(context[CONTEXT_FEATURE_MAP_WIDTH])
        height = float(context[CONTEXT_FEATURE_MAP_HEIGHT])
        obstacle_rows = np.asarray(
            session.observation.map_obstacle_features[controlled],
            dtype=np.float32,
        )
    return MapSceneV1(
        width=width,
        height=height,
        obstacles=_obstacle_scenes(obstacle_rows),
    )


def _authorized_agents(
    session: DebuggerSession,
    audience: SceneAudience,
) -> tuple[AgentSceneV1, ...]:
    active_mask = np.asarray(
        session.config.agent_profile.active_mask,
        dtype=bool,
    )
    controlled = session.controlled_global_slot
    agents: list[AgentSceneV1] = []
    for global_slot in range(MAX_AGENT_SLOTS):
        if not active_mask[global_slot]:
            continue
        row = (
            np.asarray(
                session.observation.self_features[global_slot],
                dtype=np.float32,
            )
            if audience == "researcher"
            else _candidate_row(
                config=session.config,
                observation=session.observation,
                observer_global_slot=controlled,
                candidate_global_slot=global_slot,
            )
        )
        if row is not None:
            agents.append(_agent_scene(global_slot, row))
    return tuple(agents)


def _aura_fields(
    agents: tuple[AgentSceneV1, ...], rows: dict[int, _FeatureRow]
) -> tuple[AuraFieldSceneV1, ...]:
    fields: list[AuraFieldSceneV1] = []
    for agent in agents:
        if not agent.alive:
            continue
        row = rows[agent.global_slot]
        if agent.class_id == MAGE_CLASS_ID:
            radius = float(
                row[AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_AURA_RADIUS]
            )
            if radius > 0:
                fields.append(
                    AuraFieldSceneV1(
                        source_global_slot=agent.global_slot,
                        token_id="mage_amplification",
                        center=agent.position,
                        radius=radius,
                    )
                )
        elif agent.class_id == WARRIOR_CLASS_ID:
            radius = float(
                row[AGENT_FEATURE_CAPABILITY_DAMAGE_MITIGATION_WARRIOR_AURA_RADIUS]
            )
            if radius > 0:
                fields.append(
                    AuraFieldSceneV1(
                        source_global_slot=agent.global_slot,
                        token_id="warrior_mitigation",
                        center=agent.position,
                        radius=radius,
                    )
                )
    return tuple(fields)


def _scene_rows(
    session: DebuggerSession,
    agents: tuple[AgentSceneV1, ...],
    audience: SceneAudience,
) -> dict[int, _FeatureRow]:
    controlled = session.controlled_global_slot
    rows: dict[int, _FeatureRow] = {}
    for agent in agents:
        if audience == "researcher":
            rows[agent.global_slot] = np.asarray(
                session.observation.self_features[agent.global_slot],
                dtype=np.float32,
            )
        else:
            row = _candidate_row(
                config=session.config,
                observation=session.observation,
                observer_global_slot=controlled,
                candidate_global_slot=agent.global_slot,
            )
            if row is None:
                raise AssertionError("authorized POV agent lost its feature row")
            rows[agent.global_slot] = row
    return rows


def _ranges(
    session: DebuggerSession,
    controlled_row: _FeatureRow,
) -> tuple[RangeSceneV1, ...]:
    if not session.show_ranges:
        return ()
    center = (
        float(controlled_row[AGENT_FEATURE_X]),
        float(controlled_row[AGENT_FEATURE_Y]),
    )
    ranges = [
        RangeSceneV1(
            global_slot=session.controlled_global_slot,
            center=center,
            radius=float(controlled_row[AGENT_FEATURE_OBSERVATION_RADIUS]),
            kind="observation",
        ),
        RangeSceneV1(
            global_slot=session.controlled_global_slot,
            center=center,
            radius=float(controlled_row[AGENT_FEATURE_BASIC_INTERACTION_RADIUS]),
            kind="basic",
        ),
    ]
    ultimate_radius = float(controlled_row[AGENT_FEATURE_ULTIMATE_INTERACTION_RADIUS])
    if ultimate_radius > 0:
        ranges.append(
            RangeSceneV1(
                global_slot=session.controlled_global_slot,
                center=center,
                radius=ultimate_radius,
                kind="ultimate",
            )
        )
    return tuple(ranges)


def _selection_and_pending(
    session: DebuggerSession,
    rows: dict[int, _FeatureRow],
) -> tuple[
    SelectionSceneV1,
    SelectedLegalitySceneV1 | None,
    PendingRouteSceneV1 | None,
]:
    controlled = session.controlled_global_slot
    target = session.pending_action.selected_global_target_slot
    selected = target if target in rows else None
    selection = SelectionSceneV1(
        controlled_global_slot=controlled,
        selected_global_slot=selected,
    )
    if selected is None:
        return selection, None, None
    target_action = global_slot_to_target_action(controlled, selected)
    exact_lanes = session.action_mask.select_target_use_ultimate_joint_mask[
        controlled,
        target_action,
    ]
    armed_lane = session.pending_action.armed_lane
    lane_0 = bool(exact_lanes[0])
    lane_1 = bool(exact_lanes[1])
    armed_pair_legal = (
        False if armed_lane is None else lane_0 if armed_lane == 0 else lane_1
    )
    legality = SelectedLegalitySceneV1(
        controlled_global_slot=controlled,
        target_global_slot=selected,
        target_action=target_action,
        lane_0_available=lane_0,
        lane_1_available=lane_1,
        armed_lane=armed_lane,
        armed_pair_legal=armed_pair_legal,
    )
    route = (
        None
        if armed_lane is None
        else PendingRouteSceneV1(
            source_global_slot=controlled,
            target_global_slot=selected,
            source_anchor=(
                float(rows[controlled][AGENT_FEATURE_X]),
                float(rows[controlled][AGENT_FEATURE_Y]),
            ),
            target_anchor=(
                float(rows[selected][AGENT_FEATURE_X]),
                float(rows[selected][AGENT_FEATURE_Y]),
            ),
            lane=armed_lane,
            legal=armed_pair_legal,
        )
    )
    return selection, legality, route


def build_battlefield_scene(
    session: DebuggerSession,
    *,
    audience: SceneAudience,
) -> BattlefieldSceneV1:
    """Build one independent researcher or observer-authorized scene."""
    agents = _authorized_agents(session, audience)
    rows = _scene_rows(session, agents, audience)
    controlled_row = rows[session.controlled_global_slot]
    selection, legality, route = _selection_and_pending(session, rows)
    visibility = (
        tuple(
            ObserverVisibilitySceneV1(
                observer_global_slot=session.controlled_global_slot,
                candidate_global_slot=agent.global_slot,
                visible=observer_relative_visibility(
                    config=session.config,
                    observation=session.observation,
                    observer_global_slot=session.controlled_global_slot,
                    candidate_global_slot=agent.global_slot,
                ),
            )
            for agent in agents
        )
        if audience == "researcher"
        else ()
    )
    return BattlefieldSceneV1(
        schema_version=SCENE_SCHEMA_VERSION,
        audience=audience,
        audience_badge=(
            "PRIVILEGED RESEARCHER VIEW"
            if audience == "researcher"
            else f"AGENT POV · id_{session.controlled_global_slot}"
        ),
        map=_map_scene(session=session, audience=audience),
        agents=agents,
        aura_fields=_aura_fields(agents, rows),
        ranges=_ranges(session, controlled_row),
        selection=selection,
        selected_legality=legality,
        pending_route=route,
        observer_visibility=visibility,
    )


def _previous_action_visible(
    *,
    session: DebuggerSession,
    observer_global_slot: int,
    actor_global_slot: int,
) -> bool:
    relation, row = observation_relation_and_row(
        observer_global_slot,
        actor_global_slot,
    )
    previous = session.observation.previous_timestep_actions
    if relation == "ally":
        heads = (
            previous.ally_previous_timestep_move_actions_one_hot[
                observer_global_slot,
                row,
            ],
            previous.ally_previous_timestep_select_target_actions_one_hot[
                observer_global_slot,
                row,
            ],
            previous.ally_previous_timestep_use_ultimate_actions_one_hot[
                observer_global_slot,
                row,
            ],
        )
    else:
        heads = (
            previous.enemy_previous_timestep_move_actions_one_hot[
                observer_global_slot,
                row,
            ],
            previous.enemy_previous_timestep_select_target_actions_one_hot[
                observer_global_slot,
                row,
            ],
            previous.enemy_previous_timestep_use_ultimate_actions_one_hot[
                observer_global_slot,
                row,
            ],
        )
    return all(bool(np.asarray(head).sum() == 1) for head in heads)


def _visible_at(
    *,
    session: DebuggerSession,
    observation: Observation,
    candidate_global_slot: int,
) -> bool:
    if candidate_global_slot == session.controlled_global_slot:
        return True
    return observer_relative_visibility(
        config=session.config,
        observation=observation,
        observer_global_slot=session.controlled_global_slot,
        candidate_global_slot=candidate_global_slot,
    )


_POV_DIRECT_HEALTH_ACTIVATIONS = frozenset(
    (
        "basic_damage",
        "basic_heal",
        "holy_word",
        "warrior_charge",
        "hunter_trap",
        "rogue_poison",
    )
)


def _record_pov_status_applications(
    *,
    applications: dict[tuple[int, str], list[str]],
    token_id: str,
    source_global_slot: int,
    source_class_id: int,
    target_global_slot: int | None,
    event_id: str,
) -> None:
    """Index only application events already authorized for this observer."""

    def add(global_slot: int, status_token_id: str) -> None:
        applications.setdefault((global_slot, status_token_id), []).append(event_id)

    if token_id == "mage_burst":
        add(source_global_slot, "mage_burst")
    elif token_id == "warrior_charge" and target_global_slot is not None:
        add(target_global_slot, "slow_warrior_charge")
        add(target_global_slot, "stun_warrior_charge")
    elif token_id == "hunter_trap" and target_global_slot is not None:
        add(target_global_slot, "stun_hunter_trap")
    elif token_id == "rogue_poison" and target_global_slot is not None:
        add(target_global_slot, "slow_rogue_poison")
        add(target_global_slot, "stun_rogue_poison")
        add(target_global_slot, "anti_heal_rogue_poison")
    elif (
        token_id == "basic_damage"
        and source_class_id == HUNTER_CLASS_ID
        and target_global_slot is not None
    ):
        add(target_global_slot, "slow_hunter_basic")
    elif token_id == "basic_heal" and target_global_slot is not None:
        add(target_global_slot, "priest_freedom")


def _pov_event_batch(session: DebuggerSession) -> VisualEventBatchV1:
    """Construct an observer-authorized event batch without a privileged batch."""
    transition = session.last_transition
    if transition is None:
        raise AssertionError("POV event construction requires a transition")

    transition_id = int(transition.after_state.step_count)
    event_scope = f"run-{session.run_generation}:pov-g{session.controlled_global_slot}"
    actor_by_slot = {
        actor.actor_global_slot: actor for actor in transition.actor_transitions
    }
    events: list[
        AcceptedActivationEventV1
        | NetHealthEventV1
        | ChargeDisplacementEventV1
        | StatusLifecycleEventV1
        | RejectedActionEventV1
    ] = []
    applications: dict[tuple[int, str], list[str]] = {}
    visible_direct_health_targets: set[int] = set()
    visible_positive_damage_targets: set[int] = set()
    activation_ordinal = 0
    charge_ordinal = 0

    for activation in transition.accepted_activations:
        if not _previous_action_visible(
            session=session,
            observer_global_slot=session.controlled_global_slot,
            actor_global_slot=activation.source_global_slot,
        ):
            continue
        source_actor = actor_by_slot[activation.source_global_slot]
        use_successor_anchors = activation_uses_successor_anchors(activation.kind)
        anchor_observation = (
            transition.after_observation
            if use_successor_anchors
            else transition.before_observation
        )
        source_visible_at_anchor = _visible_at(
            session=session,
            observation=anchor_observation,
            candidate_global_slot=activation.source_global_slot,
        )
        target_slot = activation.target_global_slot
        target_public = target_slot is not None and _visible_at(
            session=session,
            observation=anchor_observation,
            candidate_global_slot=target_slot,
        )
        target_actor = (
            actor_by_slot[target_slot]
            if target_public and target_slot is not None
            else None
        )
        source_anchor = (
            source_actor.position_after
            if use_successor_anchors
            else source_actor.position_before
        )
        target_anchor = (
            None
            if target_actor is None
            else target_actor.position_after
            if use_successor_anchors
            else target_actor.position_before
        )
        event_id = visual_event_id(
            event_scope,
            transition_id,
            "activation",
            activation_ordinal,
        )
        activation_ordinal += 1
        source_class_id = int(
            transition.before_observation.self_features[
                activation.source_global_slot,
                AGENT_FEATURE_CLASS_ID,
            ]
        )
        disclosed_target = target_slot if target_public else None
        events.append(
            AcceptedActivationEventV1(
                event_id=event_id,
                transition_id=transition_id,
                token_id=activation.kind,
                source_global_slot=activation.source_global_slot,
                target_global_slot=disclosed_target,
                source_anchor=source_anchor if source_visible_at_anchor else None,
                target_anchor=target_anchor,
                target_disclosure=(
                    "target_none"
                    if target_slot is None
                    else "public"
                    if target_public
                    else "redacted"
                ),
                lane=cast(SceneLane, activation.use_ultimate),
                source_class_id=source_class_id,
            )
        )
        _record_pov_status_applications(
            applications=applications,
            token_id=activation.kind,
            source_global_slot=activation.source_global_slot,
            source_class_id=source_class_id,
            target_global_slot=disclosed_target,
            event_id=event_id,
        )
        if (
            activation.kind in _POV_DIRECT_HEALTH_ACTIVATIONS
            and disclosed_target is not None
        ):
            visible_direct_health_targets.add(disclosed_target)
        if (
            activation.kind
            in ("basic_damage", "warrior_charge", "hunter_trap", "rogue_poison")
            and disclosed_target is not None
        ):
            visible_positive_damage_targets.add(disclosed_target)
        if (
            activation.kind == "warrior_charge"
            and target_slot is not None
            and source_visible_at_anchor
            and target_public
            and _visible_at(
                session=session,
                observation=transition.after_observation,
                candidate_global_slot=activation.source_global_slot,
            )
        ):
            events.append(
                ChargeDisplacementEventV1(
                    event_id=visual_event_id(
                        event_scope,
                        transition_id,
                        "charge",
                        charge_ordinal,
                    ),
                    transition_id=transition_id,
                    source_global_slot=activation.source_global_slot,
                    target_global_slot=target_slot,
                    start=source_actor.position_before,
                    end=source_actor.position_after,
                    path_kind=(
                        "charge_only"
                        if source_actor.accepted_move_action == MOVE_STAY
                        else "combined_charge_and_movement"
                    ),
                )
            )
            charge_ordinal += 1

    health_ordinal = 0
    for actor in transition.actor_transitions:
        # This record discloses both health epochs, so successor visibility
        # alone cannot authorize it.
        visible_before = _visible_at(
            session=session,
            observation=transition.before_observation,
            candidate_global_slot=actor.actor_global_slot,
        )
        visible_after = _visible_at(
            session=session,
            observation=transition.after_observation,
            candidate_global_slot=actor.actor_global_slot,
        )
        if not (visible_before and visible_after) or (
            actor.net_health_delta == 0.0
            and actor.actor_global_slot not in visible_direct_health_targets
        ):
            continue
        outcome = (
            "damage"
            if actor.net_health_delta < 0.0
            else "healing"
            if actor.net_health_delta > 0.0
            else "unchanged"
        )
        events.append(
            NetHealthEventV1(
                event_id=visual_event_id(
                    event_scope,
                    transition_id,
                    "net-health",
                    health_ordinal,
                ),
                transition_id=transition_id,
                recipient_global_slot=actor.actor_global_slot,
                recipient_anchor=actor.position_after,
                health_before=actor.health_before,
                health_after=actor.health_after,
                net_delta=actor.net_health_delta,
                outcome=outcome,
            )
        )
        health_ordinal += 1

    status_ordinal = 0
    for status in transition.status_transitions:
        if status.change == "unchanged":
            continue
        # Lifecycle records likewise disclose both duration epochs.
        visible_before = _visible_at(
            session=session,
            observation=transition.before_observation,
            candidate_global_slot=status.global_slot,
        )
        visible_after = _visible_at(
            session=session,
            observation=transition.after_observation,
            candidate_global_slot=status.global_slot,
        )
        if not (visible_before and visible_after):
            continue
        application_event_ids = tuple(
            applications.get((status.global_slot, status.status_kind), ())
        )
        change = status.change
        if (
            status.status_kind == "stun_hunter_trap"
            and application_event_ids
            and status.duration_before > 1
            and status.duration_after == HUNTER_TRAP_STUN_DURATION_TICKS
            and status.global_slot in visible_positive_damage_targets
        ):
            change = "trap_broken_and_reapplied"
        events.append(
            StatusLifecycleEventV1(
                event_id=visual_event_id(
                    event_scope,
                    transition_id,
                    "status",
                    status_ordinal,
                ),
                transition_id=transition_id,
                recipient_global_slot=status.global_slot,
                recipient_anchor=actor_by_slot[status.global_slot].position_after,
                token_id=status.status_kind,
                change=change,
                duration_before=status.duration_before,
                duration_after=status.duration_after,
                source_class_id=status.source_class_id,
                application_event_ids=application_event_ids,
            )
        )
        status_ordinal += 1

    rejection_ordinal = 0
    for rejection in transition.rejections:
        if rejection.actor_global_slot != session.controlled_global_slot:
            continue
        target_action_in_domain = (
            0 <= rejection.submitted_target_action < NUM_TARGET_ACTIONS
        )
        target_slot = (
            target_action_to_global_slot(
                rejection.actor_global_slot,
                rejection.submitted_target_action,
            )
            if target_action_in_domain
            else None
        )
        target_is_active = target_slot is not None and target_slot in actor_by_slot
        target_public = (
            target_is_active
            and target_slot is not None
            and _visible_at(
                session=session,
                observation=transition.before_observation,
                candidate_global_slot=target_slot,
            )
        )
        if not target_action_in_domain or (
            target_slot is not None and not target_is_active
        ):
            disclosure = "invalid"
        elif target_slot is None:
            disclosure = "target_none"
        elif target_public:
            disclosure = "public"
        else:
            disclosure = "redacted"
        if disclosure == "redacted":
            # A hidden target makes the pair-mask result and rejection itself
            # observer-private. The POV HUD already preserves the submitted
            # action while marking its combat result undisclosed.
            continue
        events.append(
            RejectedActionEventV1(
                event_id=visual_event_id(
                    event_scope,
                    transition_id,
                    "rejection",
                    rejection_ordinal,
                ),
                transition_id=transition_id,
                actor_global_slot=rejection.actor_global_slot,
                component=rejection.component,
                actor_anchor=actor_by_slot[rejection.actor_global_slot].position_before,
                target_global_slot=target_slot if target_public else None,
                target_anchor=(
                    actor_by_slot[target_slot].position_before
                    if target_public and target_slot is not None
                    else None
                ),
                target_disclosure=disclosure,
                lane=(
                    rejection.submitted_use_ultimate
                    if rejection.submitted_use_ultimate in (0, 1)
                    else None
                ),
                movement_mask_value=rejection.movement_mask_value,
                pair_mask_value=rejection.pair_mask_value,
            )
        )
        rejection_ordinal += 1

    return VisualEventBatchV1(
        schema_version=EVENT_SCHEMA_VERSION,
        transition_id=transition_id,
        simulator_step=transition_id,
        events=tuple(events),
    )


def build_visual_event_batch(
    session: DebuggerSession,
    *,
    audience: SceneAudience,
) -> VisualEventBatchV1 | None:
    """Build the latest full or observer-authorized transition event batch."""
    if session.last_transition is None:
        return None
    if audience == "researcher":
        return latest_visual_event_batch(session)
    return _pov_event_batch(session)
