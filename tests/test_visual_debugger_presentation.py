"""Pure tests for debugger overlay and HUD assembly."""

from dataclasses import FrozenInstanceError, replace

import jax.numpy as jnp
import numpy as np
import pytest
from scripts.dev.visual_debugger.control import (
    arm_ultimate,
    clear_pending_target,
    create_session,
    select_clicked_target,
    submit_joint_action,
    submit_next_script_frame,
)
from scripts.dev.visual_debugger.model import DebuggerSession
from scripts.dev.visual_debugger.presentation import (
    build_debugger_overlays,
    build_hud_lines,
    build_hud_sections,
)
from scripts.dev.visual_debugger.scenarios import get_scenario

from marl_battlegrounds.core.types import MAGE_CLASS_ID, Action
from marl_battlegrounds.rendering import (
    ActivationVisual,
    ChargeTrailVisual,
    HealthDeltaVisual,
    RejectedActionVisual,
)


def _session(
    name: str,
    controlled_slot: int | None = None,
) -> DebuggerSession:
    scenario = get_scenario(name)
    return create_session(
        scenario,
        seed=0,
        controlled_global_slot=controlled_slot,
        show_ranges=True,
        verbose_logging=False,
    )


def test_debugger_overlays_limit_ranges_to_controlled_actor() -> None:
    mage = _session("arena_5v5", 0)
    overlay = build_debugger_overlays(mage)

    assert int(mage.config.agent_profile.class_ids[0]) == MAGE_CLASS_ID
    assert {value.global_slot for value in overlay.ranges} == {0}
    assert {value.kind for value in overlay.ranges} == {"observation", "basic"}

    warrior = replace(mage, controlled_global_slot=1)
    overlay = build_debugger_overlays(warrior)
    assert {value.global_slot for value in overlay.ranges} == {1}
    assert {value.kind for value in overlay.ranges} == {
        "observation",
        "basic",
        "ultimate",
    }

    hidden = replace(warrior, show_ranges=False)
    assert build_debugger_overlays(hidden).ranges == ()


def test_debugger_overlays_describe_visibility_for_every_active_candidate() -> None:
    session = select_clicked_target(_session("arena_5v5", 0), 6)
    overlay = build_debugger_overlays(session)
    active_slots = tuple(
        int(slot)
        for slot in np.flatnonzero(
            np.asarray(session.config.agent_profile.active_mask, dtype=bool)
        )
    )

    assert (
        tuple(value.candidate_global_slot for value in overlay.observer_visibility)
        == active_slots
    )
    assert all(value.observer_global_slot == 0 for value in overlay.observer_visibility)
    for value in overlay.observer_visibility:
        candidate = value.candidate_global_slot
        same_team = candidate < 5
        row = candidate if same_team else candidate - 5
        expected = bool(
            session.observation.ally_visibility_mask[0, row]
            if same_team
            else session.observation.enemy_visibility_mask[0, row]
        )
        assert value.observer_visible is expected

    cleared = clear_pending_target(session)
    assert build_debugger_overlays(cleared).observer_visibility == ()


def test_lane_markers_cover_both_lanes_for_every_active_candidate() -> None:
    session = _session("basic_support")
    overlay = build_debugger_overlays(session)

    assert len(overlay.lane_markers) == 12
    assert {
        (value.candidate_global_slot, value.lane) for value in overlay.lane_markers
    } == {(slot, lane) for slot in (0, 1, 2, 5, 6, 7) for lane in (0, 1)}
    target_markers = [
        value for value in overlay.lane_markers if value.candidate_global_slot == 5
    ]
    assert any(value.available for value in target_markers)
    assert not any(value.selected for value in target_markers)


def test_target_selection_and_link_reflect_pending_exact_legality() -> None:
    session = select_clicked_target(_session("arena_5v5", 2), 7)
    assert build_debugger_overlays(session).target_links == ()
    session = arm_ultimate(session)
    overlay = build_debugger_overlays(session)

    assert {
        (selection.global_slot, selection.role) for selection in overlay.selections
    } == {(2, "controlled"), (7, "target")}
    assert len(overlay.target_links) == 1
    link = overlay.target_links[0]
    assert (link.source_global_slot, link.target_global_slot, link.lane) == (
        2,
        7,
        1,
    )
    assert not link.legal


def test_latest_transition_is_converted_without_mutating_session_epoch() -> None:
    session = _session("ultimate_showcase")
    session = submit_next_script_frame(session)
    session = submit_next_script_frame(session)
    transition_before = session.last_transition
    overlay = build_debugger_overlays(session)

    assert session.last_transition is transition_before
    assert any(isinstance(value, HealthDeltaVisual) for value in overlay.health_deltas)
    assert any(isinstance(value, ActivationVisual) for value in overlay.activations)
    assert any(isinstance(value, ChargeTrailVisual) for value in overlay.charge_trails)
    assert not any(
        isinstance(value, RejectedActionVisual) for value in overlay.rejections
    )


def test_legacy_overlay_contains_only_the_latest_transition_charge() -> None:
    session = _session("ultimate_showcase")
    session = submit_next_script_frame(session)
    session = submit_next_script_frame(session)
    overlaid = build_debugger_overlays(session).charge_trails
    assert len(overlaid) == 1
    assert overlaid[0].opacity == 1.0

    session = submit_next_script_frame(session)
    assert build_debugger_overlays(session).charge_trails == ()


def test_hud_separates_selected_target_facts_from_exact_pending_legality() -> None:
    session = select_clicked_target(_session("arena_5v5", 2), 7)
    sections = {section.heading: section for section in build_hud_sections(session)}

    selected = " ".join(sections["SELECTED TARGET"].lines)
    pending = " ".join(sections["PENDING ACTION"].lines)
    assert "TEAM B HUNTER (id_7) · enemy · distance" in selected
    assert "LOS" in selected
    assert "visible" in selected
    assert "observation" in selected
    assert "Basic no" in selected
    assert "Ultimate no" in selected
    assert "Lane 0 unavailable" in pending
    assert "Lane 1 unavailable" in pending
    assert "pair illegal" in pending


def test_hud_target_none_mage_ultimate_is_unambiguous() -> None:
    session = arm_ultimate(_session("arena_5v5", 0))
    sections = {section.heading: section for section in build_hud_sections(session)}
    target = " ".join(sections["SELECTED TARGET"].lines)
    pending = " ".join(sections["PENDING ACTION"].lines)

    assert "No target selected." in target
    assert "Stay + BURST → TEAM A MAGE (id_0) (self activation)" in pending
    assert "Lane 0 available" in pending
    assert "Lane 1 available" in pending
    assert "pair legal" in pending


def test_hud_has_six_frozen_wrapped_sections_and_complete_visual_key() -> None:
    session = submit_next_script_frame(_session("status_stack"))
    sections = build_hud_sections(session)

    assert tuple(section.heading for section in sections) == (
        "PLAY-BY-PLAY",
        "CONTROLLED AGENT",
        "SELECTED TARGET",
        "PENDING ACTION",
        "LATEST ACCEPTED RESULT",
        "TECHNICAL DETAILS AND VISUAL KEY",
    )
    assert all(
        len(line) <= (72 if section.technical else 58)
        for section in sections
        for line in section.lines
    )
    with pytest.raises(FrozenInstanceError):
        sections[0].heading = "changed"  # type: ignore[misc]

    by_heading = {section.heading: section for section in sections}
    controlled = " ".join(by_heading["CONTROLLED AGENT"].lines)
    latest = " ".join(by_heading["LATEST ACCEPTED RESULT"].lines)
    technical = " ".join(by_heading["TECHNICAL DETAILS AND VISUAL KEY"].lines)
    assert "TEAM B HUNTER (id_5)" in controlled
    assert "Health 82.00 / 100.00" in controlled
    assert "CHARGE-STUN 1" in controlled
    assert "TRAP 4" in controlled
    assert "POISON-SLOW 5" in controlled
    assert "ANTI-HEAL 4" in controlled
    assert "FREEDOM 1" in controlled
    assert "Submitted:" in latest
    assert "Accepted:" in latest
    assert "actor=g5" in technical
    assert "Cyan dotted upper band = Mage damage amplification aura." in technical
    assert "Bronze hatched lower band = Warrior damage mitigation aura." in technical
    assert build_hud_lines(session)[0] == "PLAY-BY-PLAY"


def test_snapshot_previous_actions_are_not_drawn_on_the_battlefield() -> None:
    session = submit_next_script_frame(_session("basic_support"))
    overlay = build_debugger_overlays(session)

    assert bool(session.state.has_previous_timestep_joint_action)
    assert not hasattr(overlay, "previous_actions")
    assert [
        (selection.global_slot, selection.role) for selection in overlay.selections
    ] == [(0, "controlled")]


def test_hud_handles_out_of_domain_submitted_movement_without_index_aliasing() -> None:
    session = _session("arena_5v5")
    submitted = Action(
        move=jnp.zeros_like(session.state.previous_timestep_move_actions).at[0].set(-1),
        select_target=jnp.zeros_like(
            session.state.previous_timestep_select_target_actions
        ),
        use_ultimate=jnp.zeros_like(
            session.state.previous_timestep_use_ultimate_actions
        ),
    )
    session = submit_joint_action(
        session,
        submitted,
        submission_kind="interactive",
        report_actor_slots=(0,),
    )

    sections = {section.heading: section for section in build_hud_sections(session)}
    latest = " ".join(sections["LATEST ACCEPTED RESULT"].lines)
    assert "Submitted: Invalid / no combat" in latest
    assert "Accepted:  Stay / no combat" in latest
    assert "Movement rejected · combat rejected" in latest
