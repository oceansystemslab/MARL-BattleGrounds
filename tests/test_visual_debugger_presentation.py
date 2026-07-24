"""Pure tests for debugger overlay and HUD assembly."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
from scripts.dev.visual_debugger.control import (
    arm_ultimate,
    clear_pending_target,
    create_session,
    select_clicked_target,
    submit_joint_action,
    submit_next_script_frame,
)
from scripts.dev.visual_debugger.model import DebuggerSession, TransientHistoryEntry
from scripts.dev.visual_debugger.presentation import (
    build_debugger_overlays,
    build_hud_lines,
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
    } == {(7, "target")}
    assert len(overlay.target_links) == 1
    link = overlay.target_links[0]
    assert (link.source_global_slot, link.target_global_slot, link.lane) == (
        2,
        7,
        1,
    )
    assert not link.legal


def test_transient_history_is_forwarded_without_aging_or_inference() -> None:
    session = _session("ultimate_showcase")
    session = submit_next_script_frame(session)
    session = submit_next_script_frame(session)
    history_before = session.transient_history
    overlay = build_debugger_overlays(session)

    assert session.transient_history is history_before
    assert any(isinstance(value, HealthDeltaVisual) for value in overlay.health_deltas)
    assert any(isinstance(value, ActivationVisual) for value in overlay.activations)
    assert any(isinstance(value, ChargeTrailVisual) for value in overlay.charge_trails)
    assert not any(
        isinstance(value, RejectedActionVisual) for value in overlay.rejections
    )


def test_multiple_charge_trails_preserve_oldest_to_newest_sequence_order() -> None:
    session = _session("ultimate_showcase")
    trails = tuple(
        TransientHistoryEntry(
            visual=ChargeTrailVisual(
                source_global_slot=1,
                start=(float(sequence), 0.0),
                end=(float(sequence + 1), 0.0),
                target_global_slot=7,
                path_kind="charge_only",
                opacity=opacity,
            ),
            created_after_step=sequence,
            age_submitted_steps=age,
            max_age_submitted_steps=3,
            sequence_number=sequence,
        )
        for sequence, age, opacity in ((4, 2, 0.35), (5, 1, 0.65), (6, 0, 1.0))
    )
    overlaid = build_debugger_overlays(
        replace(session, transient_history=trails)
    ).charge_trails

    assert tuple(trail.start[0] for trail in overlaid) == (4.0, 5.0, 6.0)
    assert tuple(trail.opacity for trail in overlaid) == (0.35, 0.65, 1.0)


def test_hud_separates_target_geometry_and_legality() -> None:
    session = _session("acceptance_lane_lab")
    for _ in range(5):
        session = submit_next_script_frame(session)
    session = select_clicked_target(session, 5)
    lines = build_hud_lines(session)

    target_line = next(line for line in lines if line.startswith("TARGET "))
    geometry_line = next(line for line in lines if line.startswith("GEOMETRY "))
    legality_line = next(line for line in lines if line.startswith("LEGALITY "))
    assert target_line == "TARGET g5/t6 relation=enemy distance=4.00"
    assert geometry_line == (
        "GEOMETRY los=1 visible=1 observation_range=1 basic_range=1 ultimate_range=1"
    )
    assert legality_line == ("LEGALITY lane0=1 lane1=1 selected=Basic pending_legal=1")


def test_hud_target_none_mage_ultimate_is_unambiguous() -> None:
    session = arm_ultimate(_session("arena_5v5", 0))
    lines = build_hud_lines(session)

    assert "TARGET none/t0 relation=n/a distance=n/a" in lines
    assert (
        "GEOMETRY los=n/a visible=n/a observation_range=n/a "
        "basic_range=n/a ultimate_range=n/a"
    ) in lines
    assert "LEGALITY lane0=1 lane1=1 selected=Ultimate pending_legal=1" in lines
    assert "ABILITY Mage Burst: target-none self activation" in lines


def test_hud_contains_complete_actor_pending_status_and_last_transition_summary() -> (
    None
):
    session = submit_next_script_frame(_session("status_stack"))
    lines = build_hud_lines(session)

    assert any(line.startswith("SCENARIO status_stack step=1") for line in lines)
    assert any(
        line.startswith("ACTOR g5 team=B class=Hunter health=92.00/100.00")
        for line in lines
    )
    assert any(line.startswith("STATUS slow=(5, 0, 5)") for line in lines)
    assert any(line.startswith("PENDING movement=Stay[0]") for line in lines)
    assert any(line.startswith("LAST submitted=") for line in lines)
    assert any(line.startswith("LAST DELTA health=") for line in lines)
    assert any("g5:-8.00" in line for line in lines if line.startswith("HEALTH Δ"))
    assert any("g0:0->30" in line for line in lines if line.startswith("COOLDOWN"))
    assert any(
        "g5:stun_hunter_trap:0->4:applied" in line
        for line in lines
        if line.startswith("STATUS Δ")
    )
    assert any(
        "warrior_charge:g0->g5" in line for line in lines if line.startswith("EVENTS")
    )


def test_snapshot_previous_actions_are_not_drawn_on_the_battlefield() -> None:
    session = submit_next_script_frame(_session("basic_support"))
    overlay = build_debugger_overlays(session)

    assert bool(session.state.has_previous_timestep_joint_action)
    assert not hasattr(overlay, "previous_actions")
    assert all(selection.role != "controlled" for selection in overlay.selections)


def test_hud_handles_out_of_domain_submitted_movement_without_index_aliasing() -> None:
    session = _session("acceptance_lane_lab")
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

    assert any(
        line.startswith("LAST submitted=(Invalid,t0,u0)")
        for line in build_hud_lines(session)
    )
