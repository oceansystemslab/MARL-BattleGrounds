"""Renderer-independent input dispatch and authorization tests."""

from dataclasses import replace
from unittest.mock import Mock

import jax.numpy as jnp
import pytest
import scripts.dev.visual_debugger.control as control_module
import scripts.dev.visual_debugger.input as input_module
from scripts.dev.visual_debugger.control import (
    DebuggerTransitionFailureV1,
    arm_ultimate,
    create_session,
    select_clicked_target,
    select_controlled_actor,
    set_pending_movement,
)
from scripts.dev.visual_debugger.input import (
    dispatch_command,
    hit_test_scene_agents,
    normalize_key,
    recording_restart_intent_v1,
)
from scripts.dev.visual_debugger.model import DebuggerSession
from scripts.dev.visual_debugger.protocol import (
    ActorPovTargetActionCommandV1,
    BattlefieldPointerCommandV1,
    DebuggerCommandV1,
    ExitCommandV1,
    FinishAndReviewCommandV1,
    KeyboardCommandV1,
    ResetCommandV1,
    RosterSelectionCommandV1,
    ScenarioSwitchCommandV1,
    SetPresetCommandV1,
    SetViewCommandV1,
)
from scripts.dev.visual_debugger.scenarios import get_scenario
from scripts.dev.visual_debugger.targeting import global_slot_to_target_action
from tests.visual_debugger_fixtures import debugger_test_launch_specification

from marl_battlegrounds.core.types import (
    MAX_AGENT_SLOTS,
    MOVE_EAST,
    MOVE_NORTH,
    MOVE_SOUTH,
    MOVE_STAY,
    MOVE_WEST,
    EnvConfig,
    EnvState,
)
from marl_battlegrounds.evaluation.pov import build_actor_pov_current_slice_v1
from marl_battlegrounds.rendering.evaluation_adapter import (
    build_researcher_analyzer_projection_v2,
)
from marl_battlegrounds.rendering.pov_scene import (
    build_actor_pov_analyzer_projection_v1,
)
from marl_battlegrounds.rendering.scene import AgentSceneV2


def _session(
    name: str = "arena_5v5",
    *,
    controlled_slot: int | None = None,
) -> DebuggerSession:
    return create_session(
        get_scenario(name),
        seed=0,
        evaluation_launch_specification=debugger_test_launch_specification(),
        controlled_global_slot=controlled_slot,
        show_ranges=True,
        verbose_logging=False,
    )


def _researcher_agents(session: DebuggerSession) -> tuple[AgentSceneV2, ...]:
    projection = build_researcher_analyzer_projection_v2(
        session.evaluation_context,
        session.current_evaluation_frame,
        transition_view=session.incoming_evaluation_view,
        status_source_evidence_state=session.status_source_evidence_state,
    )
    return projection.scene.agents


def _authorized_pov_slots(session: DebuggerSession) -> set[int]:
    slice_ = build_actor_pov_current_slice_v1(
        session.evaluation_context,
        session.current_evaluation_frame,
        global_slot=session.controlled_global_slot,
        incoming_transition_view=session.incoming_evaluation_view,
    )
    scene = build_actor_pov_analyzer_projection_v1(slice_).scene
    authorized_public_ids = {
        scene.self_actor.public_agent_id,
        *(body.public_agent_id for body in scene.visible_bodies),
    }
    return {
        row.global_slot
        for row in session.evaluation_context.roster
        if row.public_agent_id in authorized_public_ids
    }


@pytest.mark.parametrize(
    ("command", "expected"),
    (
        (ResetCommandV1(), "reset"),
        (KeyboardCommandV1(key="r"), "reset"),
        (KeyboardCommandV1(key="R", shift_key=True), None),
        (
            ScenarioSwitchCommandV1(scenario_name="basic_support"),
            "scenario_switch",
        ),
        (FinishAndReviewCommandV1(), None),
    ),
)
def test_recording_restart_intent_classifies_before_restart_construction(
    command: DebuggerCommandV1,
    expected: str | None,
) -> None:
    session = _session()
    assert (
        recording_restart_intent_v1(
            session,
            command,
            view_mode="researcher",
            include_stress=False,
        )
        == expected
    )


def test_recording_lifecycle_command_is_safe_when_recording_is_disabled() -> None:
    session = _session()
    result = dispatch_command(
        session,
        FinishAndReviewCommandV1(),
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
    )

    assert result.session is session
    assert result.handled
    assert not result.changed
    assert result.notice == "Replay recording is not enabled for this debugger session."


@pytest.mark.parametrize(
    ("key", "shift_key", "expected"),
    (
        (None, None, None),
        ("R", None, "r"),
        ("R", False, "r"),
        ("r", True, None),
        ("shift+r", None, None),
        ("shift+tab", None, "shift+tab"),
        ("backtab", None, "shift+tab"),
        ("iso_left_tab", None, "shift+tab"),
        ("tab", True, "shift+tab"),
        (" ", None, "space"),
        ("Spacebar", None, "space"),
        ("return", None, "enter"),
        ("Esc", None, "escape"),
        ("ArrowUp", None, "arrowup"),
        ("up", None, "arrowup"),
        ("ArrowDown", None, "arrowdown"),
        ("left", None, "arrowleft"),
        ("RIGHT", None, "arrowright"),
        ("X", None, "x"),
        ("0", None, "0"),
    ),
)
def test_key_normalization_covers_supported_aliases(
    key: str | None,
    shift_key: bool | None,
    expected: str | None,
) -> None:
    assert normalize_key(key, shift_key=shift_key) == expected


def test_shift_r_is_unrecognized_while_ordinary_r_still_resets() -> None:
    session = _session()

    shifted = dispatch_command(
        session,
        KeyboardCommandV1(key="R", shift_key=True),
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
    )
    ordinary = dispatch_command(
        session,
        KeyboardCommandV1(key="r"),
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
    )

    assert not shifted.handled
    assert not shifted.changed
    assert shifted.session is session
    assert shifted.notice is None
    assert ordinary.handled
    assert ordinary.changed
    assert ordinary.episode_restarted


@pytest.mark.parametrize(
    ("key", "expected_move"),
    (
        ("w", MOVE_NORTH),
        ("ArrowUp", MOVE_NORTH),
        ("s", MOVE_SOUTH),
        ("ArrowDown", MOVE_SOUTH),
        ("d", MOVE_EAST),
        ("ArrowRight", MOVE_EAST),
        ("a", MOVE_WEST),
        ("ArrowLeft", MOVE_WEST),
    ),
)
def test_keyboard_movement_edits_only_pending_input(
    key: str,
    expected_move: int,
) -> None:
    session = _session()
    result = dispatch_command(
        session,
        KeyboardCommandV1(key=key),
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
    )

    assert result.handled
    assert result.changed
    assert result.session.pending_action.move_action == expected_move
    assert result.session.state is session.state
    assert result.session.action_mask is session.action_mask
    assert result.session.key is session.key


def test_stay_key_resets_only_the_controlled_actor_draft() -> None:
    session = set_pending_movement(_session(), MOVE_EAST)
    session = select_controlled_actor(session, 1)
    session = set_pending_movement(session, MOVE_NORTH)
    session = select_controlled_actor(session, 0)

    result = dispatch_command(
        session,
        KeyboardCommandV1(key="x"),
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
    )

    assert result.changed
    assert result.session.pending_actions[0].move_action == MOVE_STAY
    assert result.session.pending_actions[1].move_action == MOVE_NORTH
    assert result.session.state is session.state
    assert result.session.key is session.key


def test_keyboard_movement_rejects_an_unavailable_current_mask_category() -> None:
    session = _session()
    actor = session.controlled_global_slot
    move_rows = [
        list(row) for row in session.current_evaluation_frame.action_mask.move_mask
    ]
    move_rows[actor][MOVE_EAST] = False
    canonical_mask = session.current_evaluation_frame.action_mask.model_copy(
        update={"move_mask": tuple(tuple(row) for row in move_rows)}
    )
    masked = replace(
        session,
        action_mask=session.action_mask._replace(
            move_mask=session.action_mask.move_mask.at[
                actor,
                MOVE_EAST,
            ].set(False)
        ),
        current_evaluation_frame=session.current_evaluation_frame.model_copy(
            update={"action_mask": canonical_mask}
        ),
        raw_continuation_identity=None,
    )

    result = dispatch_command(
        masked,
        KeyboardCommandV1(key="d"),
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
    )

    assert result.handled
    assert not result.changed
    assert result.session is masked
    assert result.notice is not None
    assert "current action mask" in result.notice


@pytest.mark.parametrize(("key", "lane"), (("1", 0), ("2", 1)))
def test_keyboard_combat_lane_rejects_an_unavailable_exact_pair(
    key: str,
    lane: int,
) -> None:
    session = _session(controlled_slot=2)
    target_slot = 7
    target_action = global_slot_to_target_action(
        session.controlled_global_slot,
        target_slot,
    )
    joint_mask = session.action_mask.select_target_use_ultimate_joint_mask.at[
        session.controlled_global_slot,
        target_action,
        lane,
    ].set(False)
    canonical_rows = [
        [list(lanes) for lanes in actor_rows]
        for actor_rows in (
            session.current_evaluation_frame.action_mask.select_target_use_ultimate_joint_mask
        )
    ]
    canonical_rows[session.controlled_global_slot][target_action][lane] = False
    canonical_mask = session.current_evaluation_frame.action_mask.model_copy(
        update={
            "select_target_use_ultimate_joint_mask": tuple(
                tuple(tuple(lanes) for lanes in actor_rows)
                for actor_rows in canonical_rows
            )
        }
    )
    masked = replace(
        session,
        action_mask=session.action_mask._replace(
            select_target_use_ultimate_joint_mask=joint_mask
        ),
        current_evaluation_frame=session.current_evaluation_frame.model_copy(
            update={"action_mask": canonical_mask}
        ),
        raw_continuation_identity=None,
    )
    selected = select_clicked_target(masked, target_slot)
    pending_before = selected.pending_action

    result = dispatch_command(
        selected,
        KeyboardCommandV1(key=key),
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
    )

    assert result.handled
    assert not result.changed
    assert result.session is selected
    assert result.session.pending_action == pending_before
    assert result.notice is not None
    assert "current action mask" in result.notice


def test_keyboard_basic_does_not_alias_canonical_target_none_no_combat() -> None:
    session = _session(controlled_slot=0)
    assert bool(
        session.action_mask.select_target_use_ultimate_joint_mask[
            session.controlled_global_slot,
            0,
            0,
        ]
    )

    result = dispatch_command(
        session,
        KeyboardCommandV1(key="1"),
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
    )

    assert result.handled
    assert not result.changed
    assert result.session is session
    assert result.notice is not None
    assert "canonical no-combat tuple" in result.notice


def test_zero_key_stages_explicit_no_combat_without_losing_draft_context() -> None:
    session = _session(controlled_slot=2)
    session = select_clicked_target(session, 7)
    session = set_pending_movement(session, MOVE_EAST)
    session = arm_ultimate(session)

    result = dispatch_command(
        session,
        KeyboardCommandV1(key="0"),
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
    )

    assert result.handled
    assert result.changed
    assert result.session.pending_action.move_action == MOVE_EAST
    assert result.session.pending_action.selected_global_target_slot == 7
    assert result.session.pending_action.armed_lane is None
    assert result.session.pending_action.arm_origin is None


def test_actor_pov_target_action_selects_a_currently_visible_recipient() -> None:
    session = _session("basic_support")
    result = dispatch_command(
        session,
        ActorPovTargetActionCommandV1(target_action=6),
        view_mode="pov",
        preset="analysis",
        include_stress=False,
    )

    assert result.handled
    assert result.changed
    assert result.notice is None
    assert result.session.pending_action.selected_global_target_slot == 5
    assert result.transition_applied is None


def test_actor_pov_target_action_rejects_hidden_recipient_without_disclosure() -> None:
    session = _session("basic_support")
    result = dispatch_command(
        session,
        ActorPovTargetActionCommandV1(target_action=8),
        view_mode="pov",
        preset="analysis",
        include_stress=False,
    )

    assert result.handled
    assert not result.changed
    assert result.session is session
    assert result.notice == (
        "Target action 8 is unavailable in the current authorized POV."
    )
    assert "g7" not in result.notice


def test_actor_pov_target_action_zero_clears_the_pending_target() -> None:
    selected = dispatch_command(
        _session("basic_support"),
        ActorPovTargetActionCommandV1(target_action=6),
        view_mode="pov",
        preset="analysis",
        include_stress=False,
    ).session
    assert selected.pending_action.selected_global_target_slot == 5

    cleared = dispatch_command(
        selected,
        ActorPovTargetActionCommandV1(target_action=0),
        view_mode="pov",
        preset="analysis",
        include_stress=False,
    )

    assert cleared.handled
    assert cleared.changed
    assert cleared.session.pending_action.selected_global_target_slot is None
    assert cleared.transition_applied is None


def test_actor_pov_target_action_is_rejected_in_researcher_mode() -> None:
    session = _session("basic_support")
    result = dispatch_command(
        session,
        ActorPovTargetActionCommandV1(target_action=6),
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
    )

    assert result.handled
    assert not result.changed
    assert result.session is session
    assert result.notice == "Actor-relative target selection is available only in POV."
    assert result.session.state is session.state
    assert result.session.action_mask is session.action_mask
    assert result.session.key is session.key


@pytest.mark.parametrize("key", (" ", "Enter", "n"))
def test_repeat_submission_keys_are_consumed_without_stepping(key: str) -> None:
    session = _session()
    result = dispatch_command(
        session,
        KeyboardCommandV1(key=key, repeat=True),
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
    )

    assert result.handled
    assert not result.changed
    assert result.session is session
    assert result.notice == "Repeated submission input ignored."
    assert int(result.session.state.step_count) == 0
    assert result.session.key is session.key


@pytest.mark.parametrize("key", ("w", "0", "1", "2", "Enter", " "))
def test_scripted_playback_rejects_manual_drafts_and_submit_keys(key: str) -> None:
    session = _session("basic_support")
    result = dispatch_command(
        session,
        KeyboardCommandV1(key=key),
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
    )

    assert result.handled
    assert not result.changed
    assert result.session is session
    assert result.notice is not None
    assert "press N" in result.notice
    assert int(result.session.state.step_count) == 0
    assert result.session.key is session.key


def test_n_advances_only_scripted_playback() -> None:
    interactive = _session()
    interactive_result = dispatch_command(
        interactive,
        KeyboardCommandV1(key="n"),
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
    )
    scripted = _session("basic_support")
    scripted_result = dispatch_command(
        scripted,
        KeyboardCommandV1(key="n"),
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
    )

    assert not interactive_result.changed
    assert interactive_result.session is interactive
    assert interactive_result.notice == "N advances scripted playback only."
    assert scripted_result.changed
    assert int(scripted_result.session.state.step_count) == 1
    assert scripted_result.transition_applied is not None
    assert scripted_result.session.incoming_evaluation_view is (
        scripted_result.transition_applied
    )
    assert scripted_result.session.last_submission_kind == "scripted"


def test_applied_transition_packaging_preserves_typed_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = DebuggerTransitionFailureV1(
        "validation",
        "transition_packaging_failed",
    )

    def fail_result(*_args: object, **_kwargs: object) -> object:
        raise failure

    monkeypatch.setattr(input_module, "_result", fail_result)

    with pytest.raises(DebuggerTransitionFailureV1) as caught:
        dispatch_command(
            _session(),
            KeyboardCommandV1(key="Enter"),
            view_mode="researcher",
            preset="analysis",
            include_stress=False,
        )

    assert caught.value is failure


def test_ui_only_pov_sanitizer_failure_remains_outside_transition_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_sanitizer(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("ui-only sanitizer failure")

    monkeypatch.setattr(input_module, "sanitize_pov_pending_target", fail_sanitizer)

    with pytest.raises(RuntimeError, match="ui-only sanitizer failure"):
        dispatch_command(
            _session(),
            SetViewCommandV1(view_mode="pov"),
            view_mode="researcher",
            preset="analysis",
            include_stress=False,
        )


def test_researcher_and_pov_submit_scopes_are_explicit_and_single_step() -> None:
    researcher = _session()
    researcher_result = dispatch_command(
        researcher,
        KeyboardCommandV1(key="Enter"),
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
    )
    researcher_view = researcher_result.session.incoming_evaluation_view

    assert researcher_result.changed
    assert researcher_view is not None
    assert researcher_result.transition_applied is researcher_view
    assert researcher_result.session.last_report_actor_slots == tuple(
        range(MAX_AGENT_SLOTS)
    )
    assert int(researcher_result.session.state.step_count) == 1

    pov = set_pending_movement(_session(), MOVE_NORTH)
    pov = select_controlled_actor(pov, 1)
    pov = set_pending_movement(pov, MOVE_EAST)
    pov = select_controlled_actor(pov, 0)
    pov_result = dispatch_command(
        pov,
        KeyboardCommandV1(key="Enter"),
        view_mode="pov",
        preset="analysis",
        include_stress=False,
    )
    pov_view = pov_result.session.incoming_evaluation_view

    assert pov_result.changed
    assert pov_view is not None
    assert pov_result.transition_applied is pov_view
    assert pov_result.session.last_report_actor_slots == (0,)
    submitted_action = (
        pov_view.transition.facts.action_acceptance_facts.submitted_joint_action
    )
    assert (
        submitted_action.move[0],
        submitted_action.select_target[0],
        submitted_action.use_ultimate[0],
    ) == (
        MOVE_NORTH,
        0,
        0,
    )
    for actor_slot in range(1, MAX_AGENT_SLOTS):
        assert (
            submitted_action.move[actor_slot],
            submitted_action.select_target[actor_slot],
            submitted_action.use_ultimate[actor_slot],
        ) == (MOVE_STAY, 0, 0)
    assert int(pov_result.session.state.step_count) == 1


def test_ctrl_alt_and_meta_modified_inputs_are_suppressed() -> None:
    session = _session()
    commands = (
        KeyboardCommandV1(key="d", ctrl_key=True),
        KeyboardCommandV1(key="d", alt_key=True),
        KeyboardCommandV1(key="d", meta_key=True),
        BattlefieldPointerCommandV1(
            world_x=15.0,
            world_y=10.0,
            button="primary",
            meta_key=True,
        ),
    )

    for command in commands:
        result = dispatch_command(
            session,
            command,
            view_mode="researcher",
            preset="analysis",
            include_stress=False,
        )
        assert not result.handled
        assert not result.changed
        assert result.session is session


def test_terminal_blocks_pending_edits_but_keeps_actor_inspection() -> None:
    terminal = _session("basic_support")
    for _ in range(2):
        terminal = dispatch_command(
            terminal,
            KeyboardCommandV1(key="n"),
            view_mode="researcher",
            preset="analysis",
            include_stress=False,
        ).session
    assert terminal.reached_declared_horizon
    assert not terminal.terminated
    assert not terminal.truncated
    movement = dispatch_command(
        terminal,
        KeyboardCommandV1(key="d"),
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
    )
    armed = dispatch_command(
        terminal,
        KeyboardCommandV1(key="2"),
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
    )
    cycled = dispatch_command(
        terminal,
        KeyboardCommandV1(key="Tab"),
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
    )

    assert movement.session is terminal
    assert armed.session is terminal
    assert not movement.changed
    assert not armed.changed
    assert cycled.changed
    assert cycled.session.controlled_global_slot == 1
    assert cycled.session.state is terminal.state
    assert cycled.session.key is terminal.key


def test_terminal_pov_submit_sanitizes_draft_without_reusing_incoming_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registered = get_scenario("arena_5v5")
    build_registered_scenario = registered.build_scenario

    def build_one_step_scenario() -> tuple[EnvConfig, EnvState]:
        config, state = build_registered_scenario()
        return config._replace(max_steps=1), state

    session = create_session(
        replace(registered, build_scenario=build_one_step_scenario),
        seed=0,
        evaluation_launch_specification=debugger_test_launch_specification(),
        controlled_global_slot=0,
        show_ranges=True,
        verbose_logging=False,
    )
    terminal = dispatch_command(
        session,
        KeyboardCommandV1(key="Enter"),
        view_mode="pov",
        preset="analysis",
        include_stress=False,
    ).session
    assert terminal.reached_declared_horizon
    hidden_target = min(
        row.global_slot
        for row in terminal.evaluation_context.roster
        if row.configured_active
        and row.global_slot not in _authorized_pov_slots(terminal)
    )
    staged = select_clicked_target(terminal, hidden_target)
    previous_incoming = staged.incoming_evaluation_view
    state = staged.state
    key = staged.key
    step_spy = Mock(side_effect=AssertionError("terminal submit must not step"))
    capture_spy = Mock(side_effect=AssertionError("terminal submit must not capture"))
    monkeypatch.setattr(control_module, "step", step_spy)
    monkeypatch.setattr(
        control_module,
        "capture_evaluation_transition_unit_v1",
        capture_spy,
    )

    result = dispatch_command(
        staged,
        KeyboardCommandV1(key="Enter"),
        view_mode="pov",
        preset="analysis",
        include_stress=False,
    )

    assert result.changed
    assert result.transition_applied is None
    assert result.session.pending_action.selected_global_target_slot is None
    assert result.session.incoming_evaluation_view is previous_incoming
    assert result.session.state is state
    assert result.session.key is key
    assert step_spy.call_count == 0
    assert capture_spy.call_count == 0


def test_scene_hit_test_uses_only_authorized_agents_and_stable_tie_break() -> None:
    agents = _researcher_agents(_session("basic_support"))
    first = agents[0]
    second = replace(
        agents[1],
        position=first.position,
        radius=first.radius,
    )

    assert hit_test_scene_agents((second, first), *first.position) == min(
        first.global_slot, second.global_slot
    )
    assert hit_test_scene_agents((), *first.position) is None


def test_pointer_hit_test_uses_plain_control_and_shift_target() -> None:
    session = _session("arena_5v5")
    controlled = dispatch_command(
        session,
        BattlefieldPointerCommandV1(
            world_x=3.0,
            world_y=6.0,
            button="primary",
        ),
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
    )
    selected = dispatch_command(
        controlled.session,
        BattlefieldPointerCommandV1(
            world_x=15.0,
            world_y=10.0,
            button="primary",
            shift_key=True,
        ),
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
    )
    cleared = dispatch_command(
        selected.session,
        BattlefieldPointerCommandV1(
            world_x=0.0,
            world_y=0.0,
            button="secondary",
        ),
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
    )

    assert controlled.session.controlled_global_slot == 2
    assert controlled.session.pending_action.selected_global_target_slot is None
    assert selected.session.controlled_global_slot == 2
    assert selected.session.pending_action.selected_global_target_slot == 5
    assert selected.session.pending_actions[0].selected_global_target_slot is None
    assert cleared.session.pending_action.selected_global_target_slot is None
    assert cleared.session.state is session.state
    assert cleared.session.key is session.key


def test_pov_pointer_and_roster_cannot_select_hidden_agent() -> None:
    session = _session("arena_5v5")
    assert 5 not in _authorized_pov_slots(session)

    plain_pointer = dispatch_command(
        session,
        BattlefieldPointerCommandV1(
            world_x=15.0,
            world_y=10.0,
            button="primary",
        ),
        view_mode="pov",
        preset="analysis",
        include_stress=False,
    )
    shifted_pointer = dispatch_command(
        session,
        BattlefieldPointerCommandV1(
            world_x=15.0,
            world_y=10.0,
            button="primary",
            shift_key=True,
        ),
        view_mode="pov",
        preset="analysis",
        include_stress=False,
    )
    roster_results = tuple(
        dispatch_command(
            session,
            RosterSelectionCommandV1(role=role, global_slot=5),
            view_mode="pov",
            preset="analysis",
            include_stress=False,
        )
        for role in ("target", "control")
    )

    for pointer in (plain_pointer, shifted_pointer):
        assert pointer.session is session
        assert not pointer.changed
    for roster in roster_results:
        assert roster.session is session
        assert not roster.changed
        assert roster.notice == "Agent g5 is unavailable in this view."


def test_entering_pov_clears_hidden_pending_target_without_advancing_epoch() -> None:
    session = select_clicked_target(_session("arena_5v5"), 5)
    state = session.state
    key = session.key
    result = dispatch_command(
        session,
        SetViewCommandV1(view_mode="pov"),
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
    )

    assert result.changed
    assert result.view_mode == "pov"
    assert result.session.pending_action.selected_global_target_slot is None
    assert result.session.state is state
    assert result.session.key is key
    assert int(result.session.state.step_count) == 0


def test_pov_submit_clears_target_that_leaves_successor_visibility() -> None:
    registered = get_scenario("arena_5v5")
    build_registered_scenario = registered.build_scenario

    def build_boundary_scenario() -> tuple[EnvConfig, EnvState]:
        config, state = build_registered_scenario()
        positions = state.agent_positions.at[5].set(
            jnp.asarray((8.9, 1.0), dtype=jnp.float32)
        )
        return config, state._replace(agent_positions=positions)

    session = create_session(
        replace(registered, build_scenario=build_boundary_scenario),
        seed=0,
        evaluation_launch_specification=debugger_test_launch_specification(),
        controlled_global_slot=0,
        show_ranges=True,
        verbose_logging=False,
    )
    assert 5 in _authorized_pov_slots(session)

    selected = dispatch_command(
        session,
        RosterSelectionCommandV1(role="target", global_slot=5),
        view_mode="pov",
        preset="analysis",
        include_stress=False,
    )
    moving = dispatch_command(
        selected.session,
        KeyboardCommandV1(key="a"),
        view_mode="pov",
        preset="analysis",
        include_stress=False,
    )
    submitted = dispatch_command(
        moving.session,
        KeyboardCommandV1(key="Enter"),
        view_mode="pov",
        preset="analysis",
        include_stress=False,
    )

    assert int(submitted.session.state.step_count) == 1
    assert 5 not in _authorized_pov_slots(submitted.session)
    assert submitted.session.pending_action.selected_global_target_slot is None


def test_scenario_switch_enforces_stress_authorization() -> None:
    session = _session()
    blocked = dispatch_command(
        session,
        ScenarioSwitchCommandV1(scenario_name="charge_convergence"),
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
    )
    allowed = dispatch_command(
        session,
        ScenarioSwitchCommandV1(scenario_name="charge_convergence"),
        view_mode="researcher",
        preset="analysis",
        include_stress=True,
    )

    assert blocked.session is session
    assert not blocked.changed
    assert blocked.notice is not None
    assert allowed.changed
    assert allowed.session.scenario_name == "charge_convergence"
    assert allowed.session.run_generation == session.run_generation + 1


def test_view_preset_reset_and_exit_commands_report_frame_changes_exactly() -> None:
    session = _session()
    same_view = dispatch_command(
        session,
        SetViewCommandV1(view_mode="researcher"),
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
    )
    preset = dispatch_command(
        session,
        SetPresetCommandV1.model_validate({"preset": "debug"}),
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
    )
    reset = dispatch_command(
        session,
        ResetCommandV1(),
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
    )
    exit_result = dispatch_command(
        session,
        ExitCommandV1(),
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
    )

    assert not same_view.changed
    assert not preset.changed
    assert preset.preset == "analysis"
    assert reset.changed
    assert reset.session.run_generation == session.run_generation + 1
    assert not exit_result.changed
    assert exit_result.shutdown_requested
