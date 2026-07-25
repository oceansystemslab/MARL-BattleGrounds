"""Renderer-independent input dispatch and authorization tests."""

from dataclasses import replace

import jax.numpy as jnp
import pytest
from scripts.dev.visual_debugger.control import (
    create_session,
    select_clicked_target,
    select_controlled_actor,
    set_pending_movement,
)
from scripts.dev.visual_debugger.input import (
    dispatch_command,
    hit_test_scene_agents,
    normalize_key,
)
from scripts.dev.visual_debugger.model import DebuggerSession
from scripts.dev.visual_debugger.protocol import (
    BattlefieldPointerCommandV1,
    ExitCommandV1,
    KeyboardCommandV1,
    ResetCommandV1,
    RosterSelectionCommandV1,
    ScenarioSwitchCommandV1,
    SetPresetCommandV1,
    SetViewCommandV1,
)
from scripts.dev.visual_debugger.scenarios import get_scenario
from scripts.dev.visual_debugger.scene_adapter import build_battlefield_scene

from marl_battlegrounds.core.types import (
    MAX_AGENT_SLOTS,
    MOVE_EAST,
    MOVE_NORTH,
    MOVE_SOUTH,
    MOVE_STAY,
    MOVE_WEST,
    DoneFlags,
    EnvConfig,
)


def _session(
    name: str = "arena_5v5",
    *,
    controlled_slot: int | None = None,
) -> DebuggerSession:
    return create_session(
        get_scenario(name),
        seed=0,
        controlled_global_slot=controlled_slot,
        show_ranges=True,
        verbose_logging=False,
    )


@pytest.mark.parametrize(
    ("key", "shift_key", "expected"),
    (
        (None, None, None),
        ("R", None, "shift+r"),
        ("R", False, "r"),
        ("r", True, "shift+r"),
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
    ),
)
def test_key_normalization_covers_browser_and_legacy_aliases(
    key: str | None,
    shift_key: bool | None,
    expected: str | None,
) -> None:
    assert normalize_key(key, shift_key=shift_key) == expected


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


@pytest.mark.parametrize("key", ("w", "1", "2", "Enter", " "))
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
    assert scripted_result.session.last_transition is not None
    assert scripted_result.session.last_transition.submission_kind == "scripted"


def test_researcher_and_pov_submit_scopes_are_explicit_and_single_step() -> None:
    researcher = _session()
    researcher_result = dispatch_command(
        researcher,
        KeyboardCommandV1(key="Enter"),
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
    )
    researcher_transition = researcher_result.session.last_transition

    assert researcher_result.changed
    assert researcher_transition is not None
    assert researcher_transition.report_actor_slots == tuple(range(MAX_AGENT_SLOTS))
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
    pov_transition = pov_result.session.last_transition

    assert pov_result.changed
    assert pov_transition is not None
    assert pov_transition.report_actor_slots == (0,)
    assert tuple(int(head[0]) for head in pov_transition.submitted_action) == (
        MOVE_NORTH,
        0,
        0,
    )
    for actor_slot in range(1, MAX_AGENT_SLOTS):
        assert tuple(
            int(head[actor_slot]) for head in pov_transition.submitted_action
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
    session = _session("basic_support")
    terminal = replace(
        session,
        done_flags=DoneFlags(
            terminated=jnp.asarray(True),
            truncated=jnp.asarray(False),
        ),
    )
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


def test_scene_hit_test_uses_only_authorized_agents_and_stable_tie_break() -> None:
    scene = build_battlefield_scene(_session("basic_support"), audience="researcher")
    first = scene.agents[0]
    second = replace(
        scene.agents[1],
        position=first.position,
        radius=first.radius,
    )

    assert hit_test_scene_agents((second, first), *first.position) == min(
        first.global_slot, second.global_slot
    )
    assert (
        hit_test_scene_agents(
            (replace(first, active=False),),
            *first.position,
        )
        is None
    )
    assert hit_test_scene_agents((), *first.position) is None


def test_pointer_selection_is_server_hit_tested_and_shift_controls_actor() -> None:
    session = _session("arena_5v5")
    selected = dispatch_command(
        session,
        BattlefieldPointerCommandV1(
            world_x=15.0,
            world_y=10.0,
            button="primary",
        ),
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
    )
    controlled = dispatch_command(
        selected.session,
        BattlefieldPointerCommandV1(
            world_x=3.0,
            world_y=6.0,
            button="primary",
            shift_key=True,
        ),
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
    )
    cleared = dispatch_command(
        controlled.session,
        BattlefieldPointerCommandV1(
            world_x=0.0,
            world_y=0.0,
            button="secondary",
        ),
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
    )

    assert selected.session.pending_action.selected_global_target_slot == 5
    assert controlled.session.controlled_global_slot == 2
    assert controlled.session.pending_action.selected_global_target_slot is None
    assert controlled.session.pending_actions[0].selected_global_target_slot == 5
    assert cleared.session.pending_action.selected_global_target_slot is None
    assert cleared.session.state is session.state
    assert cleared.session.key is session.key


def test_pov_pointer_and_roster_cannot_select_hidden_agent() -> None:
    session = _session("arena_5v5")
    assert 5 not in {
        agent.global_slot
        for agent in build_battlefield_scene(
            session,
            audience="agent_pov",
        ).agents
    }

    pointer = dispatch_command(
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
    roster = dispatch_command(
        session,
        RosterSelectionCommandV1(role="target", global_slot=5),
        view_mode="pov",
        preset="analysis",
        include_stress=False,
    )

    assert pointer.session is session
    assert not pointer.changed
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
    build_registered_config = registered.build_config

    def build_boundary_config() -> EnvConfig:
        config = build_registered_config()
        positions = config.initial_agent_positions.at[5].set(
            jnp.asarray((8.9, 2.0), dtype=jnp.float32)
        )
        return config._replace(initial_agent_positions=positions)

    session = create_session(
        replace(registered, build_config=build_boundary_config),
        seed=0,
        controlled_global_slot=0,
        show_ranges=True,
        verbose_logging=False,
    )
    assert 5 in {
        agent.global_slot
        for agent in build_battlefield_scene(session, audience="agent_pov").agents
    }

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
    assert 5 not in {
        agent.global_slot
        for agent in build_battlefield_scene(
            submitted.session,
            audience="agent_pov",
        ).agents
    }
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
        SetPresetCommandV1(preset="debug"),
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
    assert preset.changed
    assert preset.preset == "debug"
    assert reset.changed
    assert reset.session.run_generation == session.run_generation + 1
    assert not exit_result.changed
    assert exit_result.shutdown_requested
