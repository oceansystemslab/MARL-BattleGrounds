"""Focused allowlisting tests for canonical live-debugger browser frames."""

import json
from dataclasses import replace
from typing import cast

import jax.numpy as jnp
import pytest
from scripts.dev.visual_debugger.control import (
    arm_ultimate,
    create_session,
    make_neutral_joint_action,
    reset_session,
    select_clicked_target,
    select_controlled_actor,
    set_pending_movement,
    submit_joint_action,
    submit_next_script_frame,
)
from scripts.dev.visual_debugger.frame import LiveDebuggerFrame, build_debugger_frame
from scripts.dev.visual_debugger.model import DebuggerSession
from scripts.dev.visual_debugger.protocol import (
    ActorPovLiveDebuggerFrameV2,
    ApiErrorV2,
    CommandResponseV2,
    RecordingStatusV1,
    ResearcherLiveDebuggerFrameV2,
    ViewMode,
)
from scripts.dev.visual_debugger.scenarios import (
    RESEARCHER_SCENARIOS,
    STRESS_SCENARIOS,
    get_scenario,
)
from scripts.dev.visual_debugger.targeting import global_slot_to_target_action
from tests.visual_debugger_fixtures import debugger_test_launch_specification

from marl_battlegrounds.core.types import MOVE_EAST, MOVE_NORTH, NUM_MOVE_ACTIONS


def _session(
    name: str = "arena_5v5",
    *,
    controlled_global_slot: int = 0,
) -> DebuggerSession:
    return create_session(
        get_scenario(name),
        seed=0,
        evaluation_launch_specification=debugger_test_launch_specification(0),
        controlled_global_slot=controlled_global_slot,
        show_ranges=True,
        verbose_logging=False,
    )


def _frame(
    session: DebuggerSession,
    *,
    revision: int = 0,
    view_mode: ViewMode = "researcher",
    include_stress: bool = False,
    recording_status: RecordingStatusV1 | None = None,
) -> LiveDebuggerFrame:
    return build_debugger_frame(
        session,
        session_id="session-1",
        revision=revision,
        view_mode=view_mode,
        preset="analysis",
        include_stress=include_stress,
        recording_status=recording_status,
    )


def _recursive_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        mapping = cast(dict[str, object], value)
        keys = set(mapping)
        for item in mapping.values():
            keys.update(_recursive_keys(item))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for item in cast(list[object], value):
            keys.update(_recursive_keys(item))
        return keys
    return set()


def test_initial_researcher_frame_joins_canonical_frame_zero_and_menu() -> None:
    session = _session()
    frame = _frame(session)
    assert isinstance(frame, ResearcherLiveDebuggerFrameV2)
    scene = frame.projection.scene

    assert frame.frame_kind == "researcher_live_debugger"
    assert frame.schema_version == 2
    assert frame.revision == 0
    assert frame.run_generation == 0
    assert frame.episode_id == session.evaluation_context.identity.episode_id
    assert frame.frame_index == 0
    assert frame.frame_id == session.current_evaluation_frame.frame_id
    assert frame.simulator_step_count == 0
    assert frame.show_ranges is True
    assert frame.verbose is False
    assert frame.incoming_transition_index is None
    assert frame.incoming_transition_id is None
    assert frame.projection.incoming_events is None
    assert scene.audience == "researcher"
    assert "PRIVILEGED" in scene.audience_badge
    assert scene.frame_id == frame.frame_id
    assert frame.scenario.name == "arena_5v5"
    assert frame.scenario.mode == "interactive"
    assert len(frame.available_scenarios) == len(RESEARCHER_SCENARIOS)
    assert all(option.audience == "researcher" for option in frame.available_scenarios)
    assert frame.hud.roster_global_slots == tuple(
        agent.global_slot for agent in scene.agents
    )
    assert frame.hud.pending_submission_scope == "joint_turn"
    assert (
        tuple(pending.actor_global_slot for pending in frame.hud.pending_actions)
        == frame.hud.roster_global_slots
    )
    assert frame.hud.pending_action == frame.hud.pending_actions[0]


def test_recording_status_is_audience_common_and_joins_live_frame_progress() -> None:
    status = RecordingStatusV1(
        lifecycle="recording",
        captured_transition_count=0,
        expected_transition_count=5,
        restart_fenced=False,
        finish_available=True,
        review_available=False,
        retry_available=False,
        save_as_available=False,
        discard_available=False,
    )

    researcher = _frame(_session(), recording_status=status)
    pov = _frame(_session(), view_mode="pov", recording_status=status)

    assert researcher.recording == status
    assert pov.recording == status
    assert "replay_path" not in _recursive_keys(researcher.model_dump(mode="json"))
    assert "replay_path" not in _recursive_keys(pov.model_dump(mode="json"))

    with pytest.raises(ValueError, match="recording progress"):
        build_debugger_frame(
            _session(),
            session_id="session-1",
            revision=0,
            view_mode="researcher",
            preset="analysis",
            include_stress=False,
            recording_status=status.model_copy(
                update={
                    "captured_transition_count": 1,
                    "restart_fenced": True,
                    "discard_available": True,
                }
            ),
        )


def test_pov_recording_status_redacts_reducer_processing_failure_reason() -> None:
    status = RecordingStatusV1(
        lifecycle="saved",
        captured_transition_count=0,
        expected_transition_count=5,
        completion_state="interrupted",
        completion_reason="evaluation_processing_failure",
        restart_fenced=True,
        finish_available=False,
        review_available=True,
        retry_available=False,
        save_as_available=False,
        discard_available=False,
    )

    researcher = _frame(_session(), recording_status=status)
    pov = _frame(_session(), view_mode="pov", recording_status=status)

    assert researcher.recording == status
    assert researcher.recording is not None
    assert researcher.recording.completion_reason == "evaluation_processing_failure"
    assert pov.recording is not None
    assert pov.recording.completion_reason == "evaluation_unavailable"
    assert pov.recording.model_copy(
        update={"completion_reason": status.completion_reason}
    ) == (status)
    assert "evaluation_processing_failure" not in pov.model_dump_json()


def test_researcher_pending_plan_preserves_each_actor_draft_and_public_target() -> None:
    session = select_clicked_target(_session(), 5)
    session = set_pending_movement(session, MOVE_EAST)
    session = select_controlled_actor(session, 1)
    session = select_clicked_target(session, 6)
    session = set_pending_movement(session, MOVE_NORTH)
    frame = _frame(session, revision=5)
    assert isinstance(frame, ResearcherLiveDebuggerFrameV2)
    pending_by_slot = {
        pending.actor_global_slot: pending for pending in frame.hud.pending_actions
    }

    assert pending_by_slot[0].move_action == MOVE_EAST
    assert pending_by_slot[0].target.global_slot == 5
    assert pending_by_slot[1].move_action == MOVE_NORTH
    assert pending_by_slot[1].target.global_slot == 6
    assert frame.hud.pending_action == pending_by_slot[1]


def test_researcher_selection_preserves_valid_global_slot_zero() -> None:
    session = select_clicked_target(_session(controlled_global_slot=5), 0)
    frame = _frame(session)
    assert isinstance(frame, ResearcherLiveDebuggerFrameV2)

    assert frame.projection.scene.selection is not None
    assert frame.projection.scene.selection.controlled_global_slot == 5
    assert frame.projection.scene.selection.selected_global_slot == 0
    assert frame.hud.selected_global_slot == 0


def test_scripted_frame_marks_controlled_pending_row_inspection_only() -> None:
    frame = _frame(_session("basic_support"))
    assert isinstance(frame, ResearcherLiveDebuggerFrameV2)

    assert frame.hud.pending_submission_scope == "scripted_playback"
    assert len(frame.hud.pending_actions) == 1
    assert frame.hud.pending_actions == (frame.hud.pending_action,)
    assert frame.hud.pending_action.label == "PLAYBACK / INSPECTION ONLY"


@pytest.mark.parametrize("controlled_global_slot", (0, 5))
def test_controlled_movement_legality_copies_canonical_current_mask_row(
    controlled_global_slot: int,
) -> None:
    session = _session(controlled_global_slot=controlled_global_slot)
    frame = _frame(session)
    assert isinstance(frame, ResearcherLiveDebuggerFrameV2)

    assert tuple(
        legality.move_action for legality in frame.hud.movement_legalities
    ) == tuple(range(NUM_MOVE_ACTIONS))
    assert (
        tuple(legality.available for legality in frame.hud.movement_legalities)
        == session.current_evaluation_frame.action_mask.move_mask[
            controlled_global_slot
        ]
    )


@pytest.mark.parametrize("controlled_global_slot", (0, 5))
def test_researcher_candidates_copy_canonical_current_joint_mask(
    controlled_global_slot: int,
) -> None:
    session = _session(controlled_global_slot=controlled_global_slot)
    frame = _frame(session)
    assert isinstance(frame, ResearcherLiveDebuggerFrameV2)
    candidates = frame.hud.candidate_legalities

    assert candidates[0].target_action == 0
    assert candidates[0].target.disclosure == "target_none"
    exact_mask = session.current_evaluation_frame.action_mask
    for candidate in candidates:
        exact_lanes = exact_mask.select_target_use_ultimate_joint_mask[
            controlled_global_slot
        ][candidate.target_action]
        assert candidate.lane_0_available == exact_lanes[0]
        assert candidate.lane_1_available == exact_lanes[1]
        if candidate.target.global_slot is not None:
            assert candidate.target_action == global_slot_to_target_action(
                controlled_global_slot,
                candidate.target.global_slot,
            )


def test_stress_menu_is_explicitly_opt_in() -> None:
    frame = _frame(_session(), include_stress=True)
    assert isinstance(frame, ResearcherLiveDebuggerFrameV2)

    assert len(frame.available_scenarios) == len(RESEARCHER_SCENARIOS) + len(
        STRESS_SCENARIOS
    )
    assert tuple(
        option.name
        for option in frame.available_scenarios
        if option.audience == "stress"
    ) == (
        "moving_basic_crossfire",
        "moving_focus_crossfire",
        "charge_convergence",
        "trap_lifecycle",
        "max_status_stack",
        "lifecycle_density",
    )


def test_researcher_latest_card_copies_canonical_action_acceptance_facts() -> None:
    session = submit_next_script_frame(_session("basic_support"))
    view = session.incoming_evaluation_view
    assert view is not None
    frame = _frame(session, revision=1)
    assert isinstance(frame, ResearcherLiveDebuggerFrameV2)
    latest = frame.hud.latest_transition
    assert latest is not None
    facts = view.transition.facts.action_acceptance_facts

    assert latest.transition_index == view.transition.transition_index
    assert latest.transition_id == view.transition.transition_id
    assert tuple(actor.actor_global_slot for actor in latest.actors) == (
        session.last_report_actor_slots
    )
    for result in latest.actors:
        actor = result.actor_global_slot
        assert (
            result.accepted.move_action,
            result.accepted.target_action,
            result.accepted.use_ultimate_action,
        ) == (
            facts.accepted_joint_action.move[actor],
            facts.accepted_joint_action.select_target[actor],
            facts.accepted_joint_action.use_ultimate[actor],
        )


def test_researcher_latest_card_keeps_submitted_target_and_ultimate_distinct() -> None:
    session = _session()
    action = make_neutral_joint_action()
    action = action._replace(
        select_target=action.select_target.at[0].set(6),
        use_ultimate=action.use_ultimate.at[0].set(1),
    )
    submitted = submit_joint_action(
        session,
        action,
        submission_kind="interactive",
        report_actor_slots=(0,),
    )
    frame = _frame(submitted, revision=1)
    assert isinstance(frame, ResearcherLiveDebuggerFrameV2)
    latest = frame.hud.latest_transition
    assert latest is not None

    result = latest.actors[0]
    assert result.submitted.target_action == 6
    assert result.submitted.use_ultimate_action == 1


def test_pov_latest_card_copies_authorized_own_action_result() -> None:
    session = submit_next_script_frame(_session("basic_support"))
    view = session.incoming_evaluation_view
    assert view is not None
    frame = _frame(session, revision=1, view_mode="pov")
    assert isinstance(frame, ActorPovLiveDebuggerFrameV2)
    latest = frame.hud.latest_transition
    assert latest is not None
    actor = session.controlled_global_slot
    facts = view.transition.facts.action_acceptance_facts

    assert latest.actor.actor_public_agent_id == (
        session.evaluation_context.roster[actor].public_agent_id
    )
    assert (
        latest.actor.accepted.move_action,
        latest.actor.accepted.target.target_action,
        latest.actor.accepted.use_ultimate_action,
    ) == (
        facts.accepted_joint_action.move[actor],
        facts.accepted_joint_action.select_target[actor],
        facts.accepted_joint_action.use_ultimate[actor],
    )


def test_ui_only_frame_changes_preserve_scientific_and_event_identity() -> None:
    session = submit_next_script_frame(_session("basic_support"))
    first = _frame(session, revision=1)
    changed = arm_ultimate(session)
    second = _frame(changed, revision=2)
    assert isinstance(first, ResearcherLiveDebuggerFrameV2)
    assert isinstance(second, ResearcherLiveDebuggerFrameV2)
    assert first.projection.incoming_events is not None
    assert second.projection.incoming_events is not None

    assert first.frame_id == second.frame_id
    assert first.simulator_step_count == second.simulator_step_count
    assert first.incoming_transition_id == second.incoming_transition_id
    assert tuple(
        event.event_id for event in first.projection.incoming_events.events
    ) == tuple(event.event_id for event in second.projection.incoming_events.events)


def test_reset_returns_canonical_frame_zero_and_advances_run_generation() -> None:
    submitted = submit_next_script_frame(_session("basic_support"))
    reset = reset_session(submitted)
    frame = _frame(reset, revision=2)
    assert isinstance(frame, ResearcherLiveDebuggerFrameV2)

    assert frame.run_generation == submitted.run_generation + 1
    assert frame.frame_index == 0
    assert frame.simulator_step_count == 0
    assert frame.incoming_transition_id is None
    assert frame.projection.incoming_events is None
    assert frame.hud.latest_transition is None
    assert reset.incoming_evaluation_view is None


def test_frame_exposes_product_movement_scale_as_read_only_metadata() -> None:
    session = _session()
    frame = _frame(session, revision=1)
    assert isinstance(frame, ResearcherLiveDebuggerFrameV2)
    payload = cast(dict[str, object], json.loads(frame.model_dump_json()))
    scenario_payload = cast(dict[str, object], payload["scenario"])

    assert frame.scenario.ordinary_movement_distance_scale == 1.0
    assert scenario_payload == {
        key: value
        for key, value in scenario_payload.items()
        if key
        not in {
            "movement_scale_minimum",
            "movement_scale_maximum",
            "movement_scale_step",
            "scenario_default_movement_scale",
            "movement_scale_overridden",
        }
    }


def test_pov_frame_uses_dedicated_projection_and_omits_researcher_ranges() -> None:
    session = select_clicked_target(_session(), 5)
    frame = _frame(session, view_mode="pov")
    assert isinstance(frame, ActorPovLiveDebuggerFrameV2)
    payload = cast(dict[str, object], json.loads(frame.model_dump_json()))
    projection_payload = cast(dict[str, object], payload["projection"])
    scene_payload = cast(dict[str, object], projection_payload["scene"])

    assert frame.frame_kind == "actor_pov_live_debugger"
    assert frame.projection.scene.self_actor.public_agent_id == "0"
    assert frame.hud.controlled_public_agent_id == "0"
    assert frame.hud.pending_submission_scope == "joint_turn"
    assert frame.verbose is False
    assert "show_ranges" not in payload
    assert "ranges" not in scene_payload
    assert "selection" not in scene_payload
    assert "class_mechanics" not in scene_payload
    assert "status_source_evidence" not in projection_payload
    assert _recursive_keys(payload).isdisjoint(
        {
            "researcher_snapshot",
            "submitted_joint_action",
            "accepted_joint_action",
            "reward_by_actor",
            "seed_protocol",
            "policy_assignments",
        }
    )


def test_live_presentation_authority_tracks_session_without_changing_identity() -> None:
    original = _session()
    presented = replace(original, show_ranges=False, verbose_logging=True)
    researcher = _frame(presented)
    pov = _frame(presented, view_mode="pov")

    assert isinstance(researcher, ResearcherLiveDebuggerFrameV2)
    assert isinstance(pov, ActorPovLiveDebuggerFrameV2)
    assert researcher.show_ranges is False
    assert researcher.verbose is False
    assert pov.verbose is False
    assert researcher.frame_id == original.current_evaluation_frame.frame_id
    assert pov.frame_id == original.current_evaluation_frame.frame_id
    assert "show_ranges" not in pov.model_dump(mode="json")


def test_post_transition_pov_response_is_serializable_and_audience_distinct() -> None:
    session = submit_next_script_frame(_session("basic_support"))
    frame = _frame(session, revision=1, view_mode="pov")
    assert isinstance(frame, ActorPovLiveDebuggerFrameV2)
    response = CommandResponseV2(result="applied", frame=frame)
    payload = cast(dict[str, object], json.loads(response.model_dump_json()))
    frame_payload = cast(dict[str, object], payload["frame"])
    projection_payload = cast(dict[str, object], frame_payload["projection"])

    assert set(payload) == {"schema_version", "result", "frame", "notice"}
    assert frame_payload["frame_kind"] == "actor_pov_live_debugger"
    assert frame_payload["view_mode"] == "pov"
    assert "incoming_cues" in projection_payload
    assert "incoming_events" not in projection_payload
    assert "available_scenarios" not in frame_payload
    assert "scenario" not in frame_payload


def test_researcher_frame_json_preserves_canonical_event_discriminators() -> None:
    frame = _frame(
        submit_next_script_frame(_session("basic_support")),
        revision=1,
    )
    assert isinstance(frame, ResearcherLiveDebuggerFrameV2)
    payload = cast(dict[str, object], json.loads(frame.model_dump_json()))
    projection = cast(dict[str, object], payload["projection"])
    event_batch = cast(dict[str, object], projection["incoming_events"])
    events = cast(list[dict[str, object]], event_batch["events"])

    assert events
    assert all("event_type" in event for event in events)
    assert tuple(event["event_id"] for event in events) == tuple(
        frame.projection.scene.incoming_event_ids
    )


@pytest.mark.parametrize("view_mode", ("researcher", "pov"))
def test_command_and_error_v2_envelopes_serialize_nested_frames(
    view_mode: ViewMode,
) -> None:
    frame = _frame(_session(), view_mode=view_mode)
    response_payload = cast(
        dict[str, object],
        json.loads(CommandResponseV2(result="applied", frame=frame).model_dump_json()),
    )
    error_payload = cast(
        dict[str, object],
        json.loads(
            ApiErrorV2(
                error_code="stale_revision",
                message="stale",
                latest_frame=frame,
            ).model_dump_json()
        ),
    )

    response_frame = cast(dict[str, object], response_payload["frame"])
    error_frame = cast(dict[str, object], error_payload["latest_frame"])
    assert isinstance(response_frame["projection"], dict)
    assert isinstance(error_frame["projection"], dict)


def test_researcher_envelope_rejects_incoming_identity_on_frame_zero() -> None:
    frame = _frame(_session())
    assert isinstance(frame, ResearcherLiveDebuggerFrameV2)
    values = {name: getattr(frame, name) for name in frame.__class__.model_fields}
    values["incoming_transition_index"] = 0

    with pytest.raises(ValueError, match="appear together"):
        frame.__class__(**values)


def test_researcher_envelope_rejects_hud_scene_selection_mismatch() -> None:
    frame = _frame(_session())
    assert isinstance(frame, ResearcherLiveDebuggerFrameV2)
    values = {name: getattr(frame, name) for name in frame.__class__.model_fields}
    values["hud"] = frame.hud.model_copy(update={"selected_global_slot": 5})

    with pytest.raises(ValueError, match="selection"):
        frame.__class__(**values)


def test_joint_action_test_inputs_keep_exact_int32_heads() -> None:
    """Keep the custom-action regression independent of NumPy coercion."""
    action = make_neutral_joint_action()
    assert action.move.dtype == jnp.int32
    assert action.select_target.dtype == jnp.int32
    assert action.use_ultimate.dtype == jnp.int32
