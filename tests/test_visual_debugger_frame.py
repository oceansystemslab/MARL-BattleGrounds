"""Focused allowlisting tests for browser debugger frame construction."""

import json
from typing import cast

import numpy as np
import pytest
import scripts.dev.visual_debugger.presentation as legacy_presentation
from scripts.dev.visual_debugger.control import (
    arm_basic,
    arm_ultimate,
    create_session,
    reset_session,
    select_clicked_target,
    select_controlled_actor,
    set_pending_movement,
    submit_interactive,
    submit_next_script_frame,
)
from scripts.dev.visual_debugger.frame import build_debugger_frame
from scripts.dev.visual_debugger.model import DebuggerSession
from scripts.dev.visual_debugger.protocol import (
    ApiErrorV1,
    CommandResponseV1,
    DebuggerFrameV1,
)
from scripts.dev.visual_debugger.scenarios import get_scenario
from scripts.dev.visual_debugger.targeting import global_slot_to_target_action

from marl_battlegrounds.core.types import MOVE_EAST, MOVE_NORTH


def _session(
    name: str = "arena_5v5",
    *,
    controlled_global_slot: int = 0,
) -> DebuggerSession:
    return create_session(
        get_scenario(name),
        seed=0,
        controlled_global_slot=controlled_global_slot,
        show_ranges=True,
        verbose_logging=False,
    )


def _frame(
    session: DebuggerSession,
    *,
    revision: int = 0,
    view_mode: str = "researcher",
    include_stress: bool = False,
) -> DebuggerFrameV1:
    return build_debugger_frame(
        session,
        session_id="session-1",
        revision=revision,
        view_mode=view_mode,  # type: ignore[arg-type]
        preset="analysis",
        include_stress=include_stress,
    )


def _json_tree_is_scalar(value: object) -> bool:
    if value is None or type(value) in (str, int, float, bool):
        return True
    if isinstance(value, list):
        return all(_json_tree_is_scalar(item) for item in cast(list[object], value))
    if isinstance(value, dict):
        mapping = cast(dict[str, object], value)
        return all(
            type(key) is str and _json_tree_is_scalar(item)
            for key, item in mapping.items()
        )
    return False


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


def test_initial_researcher_frame_has_exact_metadata_and_filtered_menu() -> None:
    frame = _frame(_session())

    assert frame.revision == 0
    assert frame.run_generation == 0
    assert frame.simulator_step == 0
    assert frame.transition_id is None
    assert frame.event_batch is None
    assert frame.scene.audience == "researcher"
    assert frame.scene.audience_badge == "PRIVILEGED RESEARCHER VIEW"
    assert frame.scenario.name == "arena_5v5"
    assert frame.scenario.mode == "interactive"
    assert not frame.scenario.script_complete
    assert len(frame.available_scenarios) == 7
    assert all(option.audience == "researcher" for option in frame.available_scenarios)
    assert frame.hud.roster_global_slots == tuple(
        agent.global_slot for agent in frame.scene.agents
    )
    assert frame.hud.pending_submission_scope == "joint_turn"
    assert (
        tuple(pending.actor_global_slot for pending in frame.hud.pending_actions)
        == frame.hud.roster_global_slots
    )
    assert frame.hud.pending_action == frame.hud.pending_actions[0]


def test_researcher_pending_plan_preserves_each_actor_draft_and_public_target() -> None:
    session = select_clicked_target(_session(), 5)
    session = set_pending_movement(session, MOVE_EAST)
    session = select_controlled_actor(session, 1)
    session = select_clicked_target(session, 6)
    session = set_pending_movement(session, MOVE_NORTH)
    frame = _frame(session, revision=5)
    pending_by_slot = {
        pending.actor_global_slot: pending for pending in frame.hud.pending_actions
    }

    assert pending_by_slot[0].move_action == MOVE_EAST
    assert pending_by_slot[0].target.global_slot == 5
    assert pending_by_slot[1].move_action == MOVE_NORTH
    assert pending_by_slot[1].target.global_slot == 6
    assert frame.hud.pending_action == pending_by_slot[1]


def test_scripted_frame_marks_controlled_pending_row_inspection_only() -> None:
    frame = _frame(_session("basic_support"))

    assert frame.hud.pending_submission_scope == "scripted_playback"
    assert len(frame.hud.pending_actions) == 1
    assert frame.hud.pending_actions == (frame.hud.pending_action,)
    assert frame.hud.pending_action.label == "PLAYBACK / INSPECTION ONLY"


@pytest.mark.parametrize("controlled_global_slot", (0, 5))
def test_researcher_candidate_legality_copies_exact_current_mask_rows(
    controlled_global_slot: int,
) -> None:
    session = _session(controlled_global_slot=controlled_global_slot)
    frame = _frame(session)
    candidates = frame.hud.candidate_legalities

    assert len(candidates) == len(frame.scene.agents) + 1
    assert candidates[0].target_action == 0
    assert candidates[0].target.disclosure == "target_none"
    assert candidates[0].target.global_slot is None
    assert tuple(candidate.target.global_slot for candidate in candidates[1:]) == tuple(
        agent.global_slot for agent in frame.scene.agents
    )

    controlled = session.controlled_global_slot
    exact_mask = session.action_mask.select_target_use_ultimate_joint_mask
    for candidate in candidates:
        assert candidate.lane_0_available is bool(
            exact_mask[controlled, candidate.target_action, 0]
        )
        assert candidate.lane_1_available is bool(
            exact_mask[controlled, candidate.target_action, 1]
        )
        if candidate.target.global_slot is not None:
            assert candidate.target_action == global_slot_to_target_action(
                controlled,
                candidate.target.global_slot,
            )


def test_stress_menu_is_explicitly_opt_in() -> None:
    frame = _frame(_session(), include_stress=True)

    assert len(frame.available_scenarios) == 10
    assert tuple(
        option.name
        for option in frame.available_scenarios
        if option.audience == "stress"
    ) == ("charge_convergence", "trap_lifecycle", "max_status_stack")


def test_researcher_latest_card_uses_successor_accepted_action() -> None:
    session = submit_next_script_frame(_session("basic_support"))
    transition = session.last_transition
    assert transition is not None

    frame = _frame(session, revision=1)
    latest = frame.hud.latest_transition
    assert latest is not None
    assert latest.transition_id == int(session.state.step_count)
    assert tuple(actor.actor_global_slot for actor in latest.actors) == (
        transition.report_actor_slots
    )
    for result in latest.actors:
        actor = result.actor_global_slot
        assert result.accepted.move_action == int(
            transition.accepted_action.move[actor]
        )
        assert result.accepted.target_action == int(
            transition.accepted_action.select_target[actor]
        )
        assert result.accepted.use_ultimate_action == int(
            transition.accepted_action.use_ultimate[actor]
        )


def test_pov_latest_card_decodes_authorized_previous_action_observation() -> None:
    session = submit_next_script_frame(_session("basic_support"))
    frame = _frame(session, revision=1, view_mode="pov")
    latest = frame.hud.latest_transition
    assert latest is not None
    assert tuple(actor.actor_global_slot for actor in latest.actors) == (
        session.controlled_global_slot,
    )
    accepted = latest.actors[0].accepted

    previous = session.observation.previous_timestep_actions
    assert accepted.move_action == int(
        np.argmax(previous.ally_previous_timestep_move_actions_one_hot[0, 0])
    )
    assert accepted.target_action == int(
        np.argmax(previous.ally_previous_timestep_select_target_actions_one_hot[0, 0])
    )
    assert accepted.use_ultimate_action == int(
        np.argmax(previous.ally_previous_timestep_use_ultimate_actions_one_hot[0, 0])
    )


def test_ui_only_frame_changes_preserve_transition_and_event_identity() -> None:
    session = submit_next_script_frame(_session("basic_support"))
    first = _frame(session, revision=1)
    changed = arm_ultimate(session)
    second = _frame(changed, revision=2)

    assert first.simulator_step == second.simulator_step
    assert first.transition_id == second.transition_id
    assert first.event_batch is not None
    assert second.event_batch is not None
    assert tuple(event.event_id for event in first.event_batch.events) == tuple(
        event.event_id for event in second.event_batch.events
    )


def test_reset_clears_transition_batch_and_advances_run_generation() -> None:
    submitted = submit_next_script_frame(_session("basic_support"))
    reset = reset_session(submitted)
    frame = _frame(reset, revision=2)

    assert frame.run_generation == submitted.run_generation + 1
    assert frame.simulator_step == 0
    assert frame.transition_id is None
    assert frame.event_batch is None
    assert frame.hud.latest_transition is None


def test_pov_whole_frame_redacts_hidden_pending_target_and_raw_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = select_clicked_target(_session(), 5)
    session = select_controlled_actor(session, 1)
    session = select_clicked_target(session, 6)
    session = set_pending_movement(session, MOVE_NORTH)
    session = select_controlled_actor(session, 0)

    def fail_legacy_hud(_: DebuggerSession) -> None:
        raise AssertionError("browser POV must not use the privileged legacy HUD")

    monkeypatch.setattr(
        legacy_presentation,
        "build_hud_sections",
        fail_legacy_hud,
    )
    frame = _frame(session, view_mode="pov")
    payload = cast(dict[str, object], json.loads(frame.model_dump_json()))
    serialized = json.dumps(payload)
    scene_payload = cast(dict[str, object], payload["scene"])
    agents_payload = cast(list[dict[str, object]], scene_payload["agents"])

    assert frame.scene.selection is not None
    assert frame.scene.selection.selected_global_slot is None
    assert frame.hud.selected_global_slot is None
    assert frame.hud.pending_action.target.disclosure == "redacted"
    assert frame.hud.pending_action.target.global_slot is None
    assert frame.hud.pending_action.target_action is None
    assert frame.hud.pending_action.pair_mask_value is None
    assert frame.hud.pending_submission_scope == "controlled_actor"
    assert frame.hud.pending_actions == (frame.hud.pending_action,)
    assert frame.hud.pending_action.actor_global_slot == 0
    assert all(agent["global_slot"] != 5 for agent in agents_payload)
    assert {
        candidate.target.global_slot
        for candidate in frame.hud.candidate_legalities
        if candidate.target.global_slot is not None
    } == {agent.global_slot for agent in frame.scene.agents}
    assert all(
        candidate.target.global_slot != 5
        for candidate in frame.hud.candidate_legalities
    )
    controlled = session.controlled_global_slot
    exact_mask = session.action_mask.select_target_use_ultimate_joint_mask
    assert all(
        candidate.lane_0_available
        is bool(exact_mask[controlled, candidate.target_action, 0])
        and candidate.lane_1_available
        is bool(exact_mask[controlled, candidate.target_action, 1])
        for candidate in frame.hud.candidate_legalities
    )
    assert "id_5" not in serialized
    assert _json_tree_is_scalar(payload)

    forbidden_keys = {
        "config",
        "key",
        "state",
        "observation",
        "action_mask",
        "before_state",
        "after_state",
        "submitted_action",
        "accepted_action",
        "reward",
        "info",
    }
    assert _recursive_keys(payload).isdisjoint(forbidden_keys)


def test_pov_latest_card_omits_hidden_pair_legality_and_result() -> None:
    researcher_pending = arm_basic(select_clicked_target(_session(), 5))
    submitted = submit_interactive(researcher_pending)

    frame = _frame(submitted, revision=1, view_mode="pov")
    latest = frame.hud.latest_transition
    assert latest is not None
    result = latest.actors[0]

    assert result.submitted.target.disclosure == "redacted"
    assert result.submitted.target_action is None
    assert result.pair_mask_value is None
    assert result.combat_result == "undisclosed"


def test_post_transition_pov_response_is_a_positive_allowlisted_envelope() -> None:
    session = _session()
    initial_pov = _frame(session, view_mode="pov")
    authorized_initial_slots = {agent.global_slot for agent in initial_pov.scene.agents}
    hidden_target = next(
        slot
        for slot, active in enumerate(session.config.agent_profile.active_mask)
        if bool(active) and slot not in authorized_initial_slots
    )

    session = select_controlled_actor(session, 1)
    session = set_pending_movement(session, MOVE_NORTH)
    session = arm_basic(select_clicked_target(session, hidden_target))
    other_actor_draft = session.pending_action
    session = select_controlled_actor(session, 0)
    session = arm_basic(select_clicked_target(session, hidden_target))
    submitted = submit_interactive(session)

    assert submitted.last_transition is not None
    assert submitted.last_transition.report_actor_slots == tuple(range(10))
    assert submitted.pending_actions[1].selected_global_target_slot == (
        other_actor_draft.selected_global_target_slot
    )

    frame = _frame(submitted, revision=1, view_mode="pov")
    response = CommandResponseV1(result="applied", frame=frame)
    payload = cast(dict[str, object], json.loads(response.model_dump_json()))
    frame_payload = cast(dict[str, object], payload["frame"])
    scene_payload = cast(dict[str, object], frame_payload["scene"])
    hud_payload = cast(dict[str, object], frame_payload["hud"])
    batch_payload = cast(dict[str, object], frame_payload["event_batch"])
    events_payload = cast(list[dict[str, object]], batch_payload["events"])

    assert set(payload) == {"schema_version", "result", "frame", "notice"}
    assert set(frame_payload) == {
        "schema_version",
        "session_id",
        "run_generation",
        "revision",
        "simulator_step",
        "transition_id",
        "view_mode",
        "preset",
        "scenario",
        "available_scenarios",
        "terminal",
        "scene",
        "event_batch",
        "hud",
    }
    assert set(scene_payload) == {
        "schema_version",
        "audience",
        "audience_badge",
        "map",
        "agents",
        "aura_fields",
        "ranges",
        "selection",
        "selected_legality",
        "pending_route",
        "observer_visibility",
    }
    assert set(hud_payload) == {
        "roster_global_slots",
        "controlled_global_slot",
        "selected_global_slot",
        "pending_submission_scope",
        "pending_actions",
        "pending_action",
        "latest_transition",
        "candidate_legalities",
        "diagnostics",
    }
    assert set(batch_payload) == {
        "schema_version",
        "transition_id",
        "simulator_step",
        "events",
    }

    controlled = submitted.controlled_global_slot
    authorized_slots = tuple(agent.global_slot for agent in frame.scene.agents)
    assert frame.view_mode == "pov"
    assert frame.scene.audience == "agent_pov"
    assert hidden_target not in authorized_slots
    assert frame.scene.observer_visibility == ()
    assert frame.scene.pending_route is None
    assert frame.scene.selection is not None
    assert frame.scene.selection.selected_global_slot is None
    assert frame.hud.roster_global_slots == authorized_slots
    assert frame.hud.pending_submission_scope == "controlled_actor"
    assert tuple(
        pending.actor_global_slot for pending in frame.hud.pending_actions
    ) == (controlled,)
    assert frame.hud.pending_action == frame.hud.pending_actions[0]
    assert frame.hud.pending_action.target.disclosure == "redacted"
    assert frame.hud.pending_action.target.global_slot is None
    assert frame.hud.pending_action.target_action is None
    assert frame.hud.pending_action.pair_mask_value is None

    latest = frame.hud.latest_transition
    assert latest is not None
    assert tuple(result.actor_global_slot for result in latest.actors) == (controlled,)
    result = latest.actors[0]
    assert result.submitted.target.disclosure == "redacted"
    assert result.submitted.target.global_slot is None
    assert result.submitted.target_action is None
    assert result.pair_mask_value is None
    assert result.combat_result == "undisclosed"

    assert events_payload
    for event in events_payload:
        assert event["event_type"] == "rejected_action"
        assert set(event) == {
            "event_type",
            "event_id",
            "transition_id",
            "actor_global_slot",
            "component",
            "actor_anchor",
            "target_global_slot",
            "target_anchor",
            "target_disclosure",
            "lane",
            "movement_mask_value",
            "pair_mask_value",
        }
        assert event["actor_global_slot"] == controlled
        assert event["target_disclosure"] == "redacted"
        assert event["target_global_slot"] is None
        assert event["target_anchor"] is None

    assert _json_tree_is_scalar(payload)
    forbidden_keys = {
        "accepted_action",
        "accepted_activations",
        "action_mask",
        "actor_transitions",
        "after_action_mask",
        "after_observation",
        "after_state",
        "before_action_mask",
        "before_observation",
        "before_state",
        "candidate_global_slot",
        "config",
        "done_flags",
        "info",
        "key",
        "last_transition",
        "observation",
        "observer_global_slot",
        "position_after",
        "position_before",
        "report_actor_slots",
        "rejections",
        "reward",
        "state",
        "status_transitions",
        "submitted_action",
        "transition",
        "visible",
    }
    assert _recursive_keys(payload).isdisjoint(forbidden_keys)


def test_frame_json_preserves_event_discriminators_without_source_amounts() -> None:
    frame = _frame(
        submit_next_script_frame(_session("basic_support")),
        revision=1,
    )
    payload = cast(dict[str, object], json.loads(frame.model_dump_json()))
    event_batch = cast(dict[str, object], payload["event_batch"])
    events = cast(list[dict[str, object]], event_batch["events"])

    assert events
    assert all("event_type" in event for event in events)
    for event in events:
        if event["event_type"] != "accepted_activation":
            continue
        assert "amount" not in event
        assert "damage" not in event
        assert "healing" not in event


def test_command_and_error_envelopes_serialize_nested_frames() -> None:
    frame = _frame(_session())
    response_payload = cast(
        dict[str, object],
        json.loads(
            CommandResponseV1(
                result="applied",
                frame=frame,
            ).model_dump_json()
        ),
    )
    error_payload = cast(
        dict[str, object],
        json.loads(
            ApiErrorV1(
                error_code="stale_revision",
                message="stale",
                latest_frame=frame,
            ).model_dump_json()
        ),
    )

    response_frame = cast(dict[str, object], response_payload["frame"])
    error_frame = cast(dict[str, object], error_payload["latest_frame"])
    assert isinstance(response_frame["scene"], dict)
    assert isinstance(error_frame["scene"], dict)


def test_frame_model_rejects_view_and_scene_audience_mismatch() -> None:
    frame = _frame(_session())
    values = {name: getattr(frame, name) for name in frame.__class__.model_fields}
    values["view_mode"] = "pov"

    with pytest.raises(ValueError, match="view_mode and scene audience"):
        frame.__class__(**values)


def test_frame_model_rejects_pending_scope_incoherent_with_view_and_scenario() -> None:
    frame = _frame(_session())
    values = {name: getattr(frame, name) for name in frame.__class__.model_fields}
    values["hud"] = frame.hud.model_copy(
        update={
            "pending_submission_scope": "controlled_actor",
            "pending_actions": (frame.hud.pending_action,),
        }
    )

    with pytest.raises(ValueError, match="pending submission scope"):
        frame.__class__(**values)
