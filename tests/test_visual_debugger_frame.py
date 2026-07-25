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


def _session(name: str = "arena_5v5") -> DebuggerSession:
    return create_session(
        get_scenario(name),
        seed=0,
        controlled_global_slot=0,
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
    result = latest.actors[0]
    actor = session.controlled_global_slot

    assert result.accepted.move_action == int(transition.accepted_action.move[actor])
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
    assert all(agent["global_slot"] != 5 for agent in agents_payload)
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
