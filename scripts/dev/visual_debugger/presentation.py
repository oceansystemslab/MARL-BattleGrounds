"""Debugger-owned overlay assembly and deterministic HUD presentation."""

from dataclasses import replace
from textwrap import TextWrapper
from typing import Protocol, cast

import numpy as np

from marl_battlegrounds.core.types import (
    AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED,
    MAGE_CLASS_ID,
    NUM_TARGET_ACTIONS,
    SLOW_CHANNEL_HUNTER_BASIC,
    SLOW_CHANNEL_ROGUE_POISON,
    SLOW_CHANNEL_WARRIOR_CHARGE,
    STUN_CHANNEL_HUNTER_TRAP,
    STUN_CHANNEL_ROGUE_POISON,
    STUN_CHANNEL_WARRIOR_CHARGE,
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
    build_play_by_play_lines,
    derive_selected_target_facts,
    format_ability_name,
    format_agent_identity,
    observer_relative_visibility,
)
from scripts.dev.visual_debugger.model import DebuggerSession, HudSection
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
    selections: list[SelectionVisual] = []
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


_BODY_WRAPPER = TextWrapper(
    width=58,
    break_long_words=False,
    break_on_hyphens=False,
    subsequent_indent="  ",
)
_TECHNICAL_WRAPPER = TextWrapper(
    width=72,
    break_long_words=False,
    break_on_hyphens=False,
    subsequent_indent="  ",
)


def _identity(session: DebuggerSession, global_slot: int) -> str:
    profile = session.config.agent_profile
    return format_agent_identity(
        int(profile.class_ids[global_slot]),
        int(profile.team_ids[global_slot]),
        global_slot,
    )


def _current_statuses(session: DebuggerSession, global_slot: int) -> str:
    state = session.state
    values = (
        (
            "CHARGE-STUN",
            int(state.stun_durations[global_slot, STUN_CHANNEL_WARRIOR_CHARGE]),
        ),
        (
            "TRAP",
            int(state.stun_durations[global_slot, STUN_CHANNEL_HUNTER_TRAP]),
        ),
        (
            "POISON-STUN",
            int(state.stun_durations[global_slot, STUN_CHANNEL_ROGUE_POISON]),
        ),
        (
            "CHARGE-SLOW",
            int(state.slow_durations[global_slot, SLOW_CHANNEL_WARRIOR_CHARGE]),
        ),
        (
            "HUNTER-SLOW",
            int(state.slow_durations[global_slot, SLOW_CHANNEL_HUNTER_BASIC]),
        ),
        (
            "POISON-SLOW",
            int(state.slow_durations[global_slot, SLOW_CHANNEL_ROGUE_POISON]),
        ),
        (
            "ANTI-HEAL",
            int(state.rogue_poison_anti_heal_durations[global_slot]),
        ),
        (
            "FREEDOM",
            int(state.priest_blessing_of_freedom_slow_floor_durations[global_slot]),
        ),
        (
            "BURST",
            int(state.mage_burst_damage_amplification_durations[global_slot]),
        ),
    )
    active = tuple(f"{label} {duration}" for label, duration in values if duration > 0)
    return "none" if not active else ", ".join(active)


def _wrap_section(
    heading: str,
    lines: tuple[str, ...],
    *,
    technical: bool = False,
) -> HudSection:
    wrapper = _TECHNICAL_WRAPPER if technical else _BODY_WRAPPER
    wrapped: list[str] = []
    for line in lines:
        wrapped.extend(wrapper.wrap(line) or ("—",))
    return HudSection(heading=heading, lines=tuple(wrapped), technical=technical)


def _pending_target_label(session: DebuggerSession) -> str:
    target = session.pending_action.selected_global_target_slot
    if target is not None:
        return _identity(session, target)
    actor = session.controlled_global_slot
    class_id = int(session.config.agent_profile.class_ids[actor])
    if class_id == MAGE_CLASS_ID and session.pending_action.armed_lane == 1:
        return f"{_identity(session, actor)} (self activation)"
    return "no target"


def _latest_action_label(
    session: DebuggerSession,
    *,
    actor_slot: int,
    target_action: int,
    use_ultimate: int,
) -> str:
    class_id = int(session.config.agent_profile.class_ids[actor_slot])
    ability = format_ability_name(class_id, use_ultimate)
    if not 0 <= target_action < NUM_TARGET_ACTIONS:
        return f"{ability} / invalid target"
    target_slot = target_action_to_global_slot(actor_slot, target_action)
    if target_slot is None:
        return "no combat" if use_ultimate == 0 else f"{ability} / self"
    return f"{ability} / {_identity(session, target_slot)}"


def build_hud_sections(session: DebuggerSession) -> tuple[HudSection, ...]:
    """Build the six deterministic semantic sections of the side panel."""
    actor = session.controlled_global_slot
    profile = session.config.agent_profile
    class_id = int(profile.class_ids[actor])
    target = session.pending_action.selected_global_target_slot
    target_action = global_slot_to_target_action(actor, target)
    availability = lane_availability(
        session.action_mask,
        actor,
        target_action,
        session.pending_action.armed_lane,
    )

    if session.last_transition is None:
        play_lines = (
            f"Ready in {session.scenario_name}. Choose movement and a target, "
            "then submit one simulator step.",
        )
    else:
        focused_transition = replace(
            session.last_transition,
            report_actor_slots=(actor,),
        )
        all_play_lines = build_play_by_play_lines(focused_transition)
        play_lines = all_play_lines[:5]
        if len(all_play_lines) > len(play_lines):
            play_lines += ("Additional public effects are listed in the terminal.",)

    effective_speed = float(
        session.observation.self_features[
            actor,
            AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED,
        ]
    )
    controlled_lines = (
        _identity(session, actor),
        (
            f"Health {float(session.state.current_health[actor]):.2f} / "
            f"{float(profile.max_health[actor]):.2f}   "
            f"Speed {effective_speed:.2f}   "
            f"Ultimate CD {int(session.state.ultimate_cooldowns[actor])}"
        ),
        f"Statuses: {_current_statuses(session, actor)}",
    )

    facts = derive_selected_target_facts(
        config=session.config,
        state=session.state,
        observation=session.observation,
        action_mask=session.action_mask,
        controlled_global_slot=actor,
        target_global_slot=target,
    )
    if facts is None:
        target_lines = (
            "No target selected.",
            "Left-click selects a target; right-click or Escape clears it.",
        )
    else:
        target_lines = (
            f"{_identity(session, facts.target_global_slot)} · {facts.relation} · "
            f"distance {facts.center_distance:.2f}",
            (
                f"LOS {'yes' if facts.has_clear_line_of_sight else 'no'} · "
                f"visible {'yes' if facts.observer_visible else 'no'} · "
                "observation "
                f"{'yes' if facts.inside_observation_radius else 'no'} · "
                f"Basic {'yes' if facts.lane_0_available else 'no'} · "
                f"Ultimate {'yes' if facts.lane_1_available else 'no'}"
            ),
        )

    if session.pending_action.armed_lane is None or (
        session.pending_action.armed_lane == 0
        and session.pending_action.selected_global_target_slot is None
    ):
        pending_ability = "NO COMBAT"
    else:
        pending_ability = format_ability_name(
            class_id,
            session.pending_action.armed_lane,
        )
    pair_result = "pair legal" if availability.armed_pair_legal else "pair illegal"
    pending_lines = (
        (
            f"{_move_name(session.pending_action.move_action)} + "
            f"{pending_ability} → {_pending_target_label(session)}"
        ),
        (
            f"Lane 0 {'available' if availability.lane_0_available else 'unavailable'}"
            " · "
            f"Lane 1 {'available' if availability.lane_1_available else 'unavailable'}"
            f" · {pair_result}"
        ),
    )

    if session.last_transition is None:
        latest_lines = ("No action has been submitted.",)
    else:
        actor_transition = next(
            value
            for value in session.last_transition.actor_transitions
            if value.actor_global_slot == actor
        )
        submitted_combat = _latest_action_label(
            session,
            actor_slot=actor,
            target_action=actor_transition.submitted_target_action,
            use_ultimate=actor_transition.submitted_use_ultimate,
        )
        accepted_combat = _latest_action_label(
            session,
            actor_slot=actor,
            target_action=actor_transition.accepted_target_action,
            use_ultimate=actor_transition.accepted_use_ultimate,
        )
        combat_result = (
            "canonical no-op"
            if actor_transition.submitted_target_action == 0
            and actor_transition.submitted_use_ultimate == 0
            and actor_transition.combat_pair_accepted
            else "accepted"
            if actor_transition.combat_pair_accepted
            else "rejected"
        )
        latest_lines = (
            (
                f"Submitted: {_move_name(actor_transition.submitted_move_action)} / "
                f"{submitted_combat}"
            ),
            (
                f"Accepted:  {_move_name(actor_transition.accepted_move_action)} / "
                f"{accepted_combat}"
            ),
            (
                "Movement "
                f"{'accepted' if actor_transition.movement_accepted else 'rejected'}"
                f" · combat {combat_result}"
            ),
        )

    move_mask_value = (
        bool(session.action_mask.move_mask[actor, session.pending_action.move_action])
        if 0 <= session.pending_action.move_action < len(_MOVE_NAMES)
        else False
    )
    technical_lines = (
        (
            f"scenario={session.scenario_name} step={int(session.state.step_count)} "
            f"mode={'verbose' if session.verbose_logging else 'concise'}"
        ),
        (
            f"actor=g{actor} move_mask[{session.pending_action.move_action}]="
            f"{int(move_mask_value)} target=t{target_action} "
            f"lane0={int(availability.lane_0_available)} "
            f"lane1={int(availability.lane_1_available)} "
            f"pair={int(availability.armed_pair_legal)}"
        ),
        "Visual key: lane 0 = Basic; lane 1 = Ultimate; reticle = selected target.",
        "Status chips show successor-state labels and remaining submitted steps.",
        "Cyan dotted upper band = Mage damage amplification aura.",
        "Bronze hatched lower band = Warrior damage mitigation aura.",
        "BURST!/CHARGE!/TRAP!/POISON!/HOLY WORD! = latest accepted activation.",
    )

    return (
        _wrap_section("PLAY-BY-PLAY", play_lines),
        _wrap_section("CONTROLLED AGENT", controlled_lines),
        _wrap_section("SELECTED TARGET", target_lines),
        _wrap_section("PENDING ACTION", pending_lines),
        _wrap_section("LATEST ACCEPTED RESULT", latest_lines),
        _wrap_section(
            "TECHNICAL DETAILS AND VISUAL KEY",
            technical_lines,
            technical=True,
        ),
    )


def build_hud_lines(session: DebuggerSession) -> tuple[str, ...]:
    """Flatten semantic sections for snapshots and lightweight callers."""
    return tuple(
        line
        for section in build_hud_sections(session)
        for line in (section.heading, *section.lines)
    )


def draw_hud(axes: object, session: DebuggerSession) -> None:
    """Draw a styled six-section hierarchy on debugger-owned axes."""
    typed_axes = cast(_AxesLike, axes)
    typed_axes.clear()
    typed_axes.set_axis_off()
    sections = build_hud_sections(session)
    heading_colors = (
        "#1f4e79",
        "#285943",
        "#704214",
        "#5b3f8c",
        "#7b2d26",
        "#374151",
    )
    total_lines = sum(len(section.lines) for section in sections)
    line_step = min(0.026, 0.76 / max(total_lines, 1))
    y = 0.985
    for section, color in zip(sections, heading_colors, strict=True):
        typed_axes.text(
            0.018,
            y,
            section.heading,
            transform=typed_axes.transAxes,
            ha="left",
            va="top",
            fontsize=9.0,
            weight="bold",
            color="#ffffff",
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": color,
                "edgecolor": color,
                "alpha": 0.96,
            },
        )
        y -= 0.033
        typed_axes.text(
            0.025,
            y,
            "\n".join(section.lines),
            transform=typed_axes.transAxes,
            ha="left",
            va="top",
            fontsize=7.8 if section.technical else 8.4,
            family="monospace" if section.technical else "sans-serif",
            color="#111827",
            linespacing=1.18,
        )
        y -= line_step * len(section.lines) + 0.018
