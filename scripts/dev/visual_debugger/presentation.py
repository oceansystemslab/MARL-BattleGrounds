"""Debugger-owned overlay assembly and deterministic HUD presentation."""

from typing import Protocol, cast

import numpy as np

from marl_battlegrounds.core.types import (
    AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED,
    MAGE_CLASS_ID,
)
from marl_battlegrounds.rendering import (
    ActivationVisual,
    BattlefieldOverlays,
    ChargeTrailVisual,
    HealthDeltaVisual,
    LaneMarkerVisual,
    ObserverVisibilityVisual,
    RangeVisual,
    RejectedActionVisual,
    SelectionVisual,
    TargetLinkVisual,
    describe_snapshot_overlays,
    merge_battlefield_overlays,
)
from scripts.dev.visual_debugger.control import lane_availability
from scripts.dev.visual_debugger.diagnostics import (
    derive_selected_target_facts,
    observer_relative_visibility,
)
from scripts.dev.visual_debugger.model import DebuggerSession
from scripts.dev.visual_debugger.targeting import global_slot_to_target_action

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


class _AxesLike(Protocol):
    transAxes: object  # noqa: N815 - Matplotlib API name.

    def clear(self) -> object: ...

    def text(self, x: float, y: float, s: str, **kwargs: object) -> object: ...

    def set_axis_off(self) -> object: ...


def _move_name(move_action: int) -> str:
    if 0 <= move_action < len(_MOVE_NAMES):
        return _MOVE_NAMES[move_action]
    return "Invalid"


def build_debugger_overlays(session: DebuggerSession) -> BattlefieldOverlays:
    """Assemble reusable snapshot descriptions and debugger-owned live facts."""
    snapshot = describe_snapshot_overlays(
        session.config,
        session.state,
        session.observation,
    )
    controlled = session.controlled_global_slot
    target = session.pending_action.selected_global_target_slot
    selections = [SelectionVisual(controlled, "controlled")]
    if target is not None:
        selections.append(SelectionVisual(target, "target"))

    active_slots = tuple(
        int(slot)
        for slot in np.flatnonzero(
            np.asarray(session.config.agent_profile.active_mask, dtype=bool)
        )
    )
    visibility = (
        tuple(
            ObserverVisibilityVisual(
                observer_global_slot=controlled,
                candidate_global_slot=candidate,
                observer_visible=observer_relative_visibility(
                    config=session.config,
                    observation=session.observation,
                    observer_global_slot=controlled,
                    candidate_global_slot=candidate,
                ),
            )
            for candidate in active_slots
        )
        if target is not None
        else ()
    )

    ranges: list[RangeVisual] = []
    if session.show_ranges:
        position = session.state.agent_positions[controlled]
        center = (float(position[0]), float(position[1]))
        ranges.extend(
            (
                RangeVisual(
                    controlled,
                    center,
                    float(session.config.agent_profile.observation_radii[controlled]),
                    "observation",
                ),
                RangeVisual(
                    controlled,
                    center,
                    float(
                        session.config.agent_profile.basic_interaction_radii[controlled]
                    ),
                    "basic",
                ),
            )
        )
        ultimate_radius = float(
            session.config.agent_profile.ultimate_interaction_radii[controlled]
        )
        if ultimate_radius > 0:
            ranges.append(
                RangeVisual(
                    controlled,
                    center,
                    ultimate_radius,
                    "ultimate",
                )
            )

    lane_markers: list[LaneMarkerVisual] = []
    for candidate in active_slots:
        target_action = global_slot_to_target_action(controlled, candidate)
        availability = lane_availability(
            session.action_mask,
            controlled,
            target_action,
            session.pending_action.armed_lane,
        )
        lane_markers.extend(
            (
                LaneMarkerVisual(
                    candidate,
                    0,
                    availability.lane_0_available,
                    target == candidate and session.pending_action.armed_lane == 0,
                ),
                LaneMarkerVisual(
                    candidate,
                    1,
                    availability.lane_1_available,
                    target == candidate and session.pending_action.armed_lane == 1,
                ),
            )
        )

    target_links: tuple[TargetLinkVisual, ...] = ()
    if target is not None and session.pending_action.armed_lane is not None:
        target_action = global_slot_to_target_action(controlled, target)
        availability = lane_availability(
            session.action_mask,
            controlled,
            target_action,
            session.pending_action.armed_lane,
        )
        target_links = (
            TargetLinkVisual(
                source_global_slot=controlled,
                target_global_slot=target,
                lane=session.pending_action.armed_lane,
                legal=availability.armed_pair_legal,
            ),
        )

    health_deltas: list[HealthDeltaVisual] = []
    activations: list[ActivationVisual] = []
    charge_trails: list[ChargeTrailVisual] = []
    rejections: list[RejectedActionVisual] = []
    for entry in session.transient_history:
        if isinstance(entry.visual, HealthDeltaVisual):
            health_deltas.append(entry.visual)
        elif isinstance(entry.visual, ActivationVisual):
            activations.append(entry.visual)
        elif isinstance(entry.visual, ChargeTrailVisual):
            charge_trails.append(entry.visual)
        else:
            rejections.append(entry.visual)

    debugger = BattlefieldOverlays(
        selections=tuple(selections),
        observer_visibility=visibility,
        ranges=tuple(ranges),
        target_links=target_links,
        lane_markers=tuple(lane_markers),
        health_deltas=tuple(health_deltas),
        activations=tuple(activations),
        charge_trails=tuple(charge_trails),
        rejections=tuple(rejections),
    )
    return merge_battlefield_overlays(snapshot, debugger)


def _status_summary(session: DebuggerSession, actor: int) -> str:
    slow = tuple(int(value) for value in session.state.slow_durations[actor])
    stun = tuple(int(value) for value in session.state.stun_durations[actor])
    anti = int(session.state.rogue_poison_anti_heal_durations[actor])
    burst = int(session.state.mage_burst_damage_amplification_durations[actor])
    freedom = int(session.state.priest_blessing_of_freedom_slow_floor_durations[actor])
    return (
        f"STATUS slow={slow} stun={stun} anti_heal={anti} "
        f"burst={burst} freedom={freedom}"
    )


def _extend_chunked(
    lines: list[str],
    prefix: str,
    values: list[str],
    *,
    chunk_size: int = 3,
) -> None:
    for start in range(0, len(values), chunk_size):
        lines.append(f"{prefix} {' '.join(values[start : start + chunk_size])}")


def build_hud_lines(session: DebuggerSession) -> tuple[str, ...]:
    """Return compact stable HUD lines with geometry and legality separated."""
    actor = session.controlled_global_slot
    profile = session.config.agent_profile
    class_id = int(profile.class_ids[actor])
    team_id = int(profile.team_ids[actor])
    target = session.pending_action.selected_global_target_slot
    target_action = global_slot_to_target_action(actor, target)
    availability = lane_availability(
        session.action_mask,
        actor,
        target_action,
        session.pending_action.armed_lane,
    )
    pending_name = (
        "none"
        if session.pending_action.armed_lane is None
        else "Basic"
        if session.pending_action.armed_lane == 0
        else "Ultimate"
    )
    effective_speed = float(
        session.observation.self_features[
            actor,
            AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED,
        ]
    )
    lines = [
        (
            f"SCENARIO {session.scenario_name} "
            f"step={int(session.state.step_count)} "
            f"mode={'verbose' if session.verbose_logging else 'concise'}"
        ),
        (
            f"ACTOR g{actor} team={_TEAM_NAMES[team_id]} "
            f"class={_CLASS_NAMES[class_id]} "
            f"health={float(session.state.current_health[actor]):.2f}/"
            f"{float(profile.max_health[actor]):.2f} "
            f"speed={effective_speed:.2f} "
            f"cooldown={int(session.state.ultimate_cooldowns[actor])}"
        ),
        _status_summary(session, actor),
        (
            f"PENDING movement={_MOVE_NAMES[session.pending_action.move_action]}"
            f"[{session.pending_action.move_action}] "
            f"selected={pending_name} "
            f"origin={session.pending_action.arm_origin or 'none'}"
        ),
    ]

    facts = derive_selected_target_facts(
        config=session.config,
        state=session.state,
        observation=session.observation,
        action_mask=session.action_mask,
        controlled_global_slot=actor,
        target_global_slot=target,
    )
    if facts is None:
        lines.extend(
            (
                "TARGET none/t0 relation=n/a distance=n/a",
                (
                    "GEOMETRY los=n/a visible=n/a observation_range=n/a "
                    "basic_range=n/a ultimate_range=n/a"
                ),
                (
                    f"LEGALITY lane0={int(availability.lane_0_available)} "
                    f"lane1={int(availability.lane_1_available)} "
                    f"selected={pending_name} "
                    f"pending_legal={int(availability.armed_pair_legal)}"
                ),
            )
        )
        if class_id == MAGE_CLASS_ID and session.pending_action.armed_lane == 1:
            lines.append("ABILITY Mage Burst: target-none self activation")
    else:
        ultimate = (
            "n/a"
            if facts.inside_ultimate_radius is None
            else str(int(facts.inside_ultimate_radius))
        )
        lines.extend(
            (
                (
                    f"TARGET g{facts.target_global_slot}/t{facts.target_action} "
                    f"relation={facts.relation} "
                    f"distance={facts.center_distance:.2f}"
                ),
                (
                    f"GEOMETRY los={int(facts.has_clear_line_of_sight)} "
                    f"visible={int(facts.observer_visible)} "
                    f"observation_range={int(facts.inside_observation_radius)} "
                    f"basic_range={int(facts.inside_basic_radius)} "
                    f"ultimate_range={ultimate}"
                ),
                (
                    f"LEGALITY lane0={int(facts.lane_0_available)} "
                    f"lane1={int(facts.lane_1_available)} "
                    f"selected={pending_name} "
                    f"pending_legal={int(availability.armed_pair_legal)}"
                ),
            )
        )

    if session.last_transition is not None:
        transition = session.last_transition
        actor_transition = next(
            value
            for value in transition.actor_transitions
            if value.actor_global_slot == actor
        )
        submitted_move_name = _move_name(actor_transition.submitted_move_action)
        accepted_move_name = _move_name(actor_transition.accepted_move_action)
        movement_result = (
            "accepted" if actor_transition.movement_accepted else "rejected"
        )
        combat_result = (
            "canonical-noop"
            if actor_transition.submitted_target_action == 0
            and actor_transition.submitted_use_ultimate == 0
            and actor_transition.combat_pair_accepted
            else "accepted"
            if actor_transition.combat_pair_accepted
            else "rejected"
        )
        lines.extend(
            (
                (
                    f"LAST submitted=({submitted_move_name},"
                    f"t{actor_transition.submitted_target_action},"
                    f"u{actor_transition.submitted_use_ultimate}) "
                    f"accepted=({accepted_move_name},"
                    f"t{actor_transition.accepted_target_action},"
                    f"u{actor_transition.accepted_use_ultimate})"
                ),
                (
                    f"LAST DELTA health={actor_transition.net_health_delta:+.2f} "
                    f"move={movement_result} combat={combat_result}"
                ),
            )
        )
        health_capable = {
            "basic_damage",
            "basic_heal",
            "holy_word",
            "warrior_charge",
        }
        health_targets = {
            activation.target_global_slot
            for activation in transition.accepted_activations
            if activation.kind in health_capable
            and activation.target_global_slot is not None
        }
        health_changes = [
            f"g{value.actor_global_slot}:{value.net_health_delta:+.2f}"
            for value in transition.actor_transitions
            if value.net_health_delta != 0.0
            or value.actor_global_slot in health_targets
        ]
        cooldown_changes = [
            f"g{value.actor_global_slot}:{value.cooldown_before}->{value.cooldown_after}"
            for value in transition.actor_transitions
            if value.cooldown_before != value.cooldown_after
        ]
        status_changes = [
            (
                f"g{value.global_slot}:{value.status_kind}:"
                f"{value.duration_before}->{value.duration_after}:{value.change}"
            )
            for value in transition.status_transitions
            if value.change != "unchanged"
        ]
        events = [
            (
                f"{value.kind}:g{value.source_global_slot}->"
                + (
                    "none"
                    if value.target_global_slot is None
                    else f"g{value.target_global_slot}"
                )
            )
            for value in transition.accepted_activations
        ]
        events.extend(
            f"reject:{value.component}:g{value.actor_global_slot}"
            for value in transition.rejections
        )
        _extend_chunked(lines, "HEALTH Δ", health_changes)
        _extend_chunked(lines, "COOLDOWN", cooldown_changes)
        _extend_chunked(lines, "STATUS Δ", status_changes)
        _extend_chunked(lines, "EVENTS", events)
    if bool(session.done_flags.terminated) or bool(session.done_flags.truncated):
        lines.append("EPISODE COMPLETE: press R to reset or [ / ] to switch scenario.")
    return tuple(lines)


def draw_hud(axes: object, session: DebuggerSession) -> None:
    """Draw deterministic HUD lines on debugger-owned axes."""
    typed_axes = cast(_AxesLike, axes)
    typed_axes.clear()
    typed_axes.set_axis_off()
    lines = build_hud_lines(session)
    line_step = min(0.066, 0.96 / max(len(lines), 1))
    font_size = 8.5 if len(lines) <= 15 else 6.5
    for index, line in enumerate(lines):
        typed_axes.text(
            0.01,
            0.98 - index * line_step,
            line,
            transform=typed_axes.transAxes,
            ha="left",
            va="top",
            fontsize=font_size,
            family="monospace",
            color="#111111",
        )
