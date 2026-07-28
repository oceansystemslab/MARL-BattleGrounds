"""Exhaustive pure and integration tests for debugger session control."""

from dataclasses import fields, replace
from typing import cast

import jax
import jax.numpy as jnp
import pytest
import scripts.dev.visual_debugger.control as control
from scripts.dev.visual_debugger.control import (
    arm_basic,
    arm_ultimate,
    build_interactive_joint_action,
    build_scripted_joint_action,
    clear_pending_target,
    create_session,
    cycle_controlled_actor,
    lane_availability,
    make_neutral_joint_action,
    reset_session,
    select_clicked_target,
    select_controlled_actor,
    select_no_combat,
    set_movement_scale,
    set_pending_movement,
    submit_interactive,
    submit_joint_action,
    submit_next_script_frame,
    switch_scenario,
)
from scripts.dev.visual_debugger.model import (
    AcceptedActivation,
    ActionRejection,
    ActorCommand,
    ActorTransition,
    DebuggerScenario,
    DebuggerSession,
    LaneAvailability,
    PendingAction,
    ScenarioFrame,
    SelectedTargetFacts,
    StatusTransition,
    TransitionView,
)
from scripts.dev.visual_debugger.scenarios import get_scenario
from scripts.dev.visual_debugger.targeting import global_slot_to_target_action

from marl_battlegrounds.core.types import (
    MAGE_CLASS_ID,
    MAX_AGENT_SLOTS,
    MOVE_EAST,
    MOVE_NORTH,
    MOVE_STAY,
    MOVE_WEST,
    Action,
    DoneFlags,
)


def _session(
    name: str = "arena_5v5",
    *,
    controlled_slot: int | None = None,
    verbose: bool = False,
) -> DebuggerSession:
    scenario = get_scenario(name)
    return create_session(
        scenario,
        seed=7,
        controlled_global_slot=controlled_slot,
        show_ranges=True,
        verbose_logging=verbose,
    )


def _tree_equal(left: object, right: object) -> bool:
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    return len(left_leaves) == len(right_leaves) and all(
        bool(jnp.array_equal(a, b))
        for a, b in zip(left_leaves, right_leaves, strict=True)
    )


def _replace_pending_row(
    session: DebuggerSession,
    actor_slot: int,
    pending_action: PendingAction,
) -> tuple[PendingAction, ...]:
    pending_actions = list(session.pending_actions)
    pending_actions[actor_slot] = pending_action
    return tuple(pending_actions)


def test_make_neutral_joint_action_contract() -> None:
    action = make_neutral_joint_action()
    for head in action:
        assert head.shape == (MAX_AGENT_SLOTS,)
        assert head.dtype == jnp.int32
        assert bool(jnp.all(head == 0))


def test_every_debugger_structure_has_the_exact_audited_field_schema() -> None:
    expected = {
        PendingAction: (
            "move_action",
            "selected_global_target_slot",
            "armed_lane",
            "arm_origin",
        ),
        LaneAvailability: (
            "target_action",
            "lane_0_available",
            "lane_1_available",
            "armed_lane",
            "armed_pair_legal",
        ),
        SelectedTargetFacts: (
            "controlled_global_slot",
            "target_global_slot",
            "target_action",
            "relation",
            "center_distance",
            "has_clear_line_of_sight",
            "observer_visible",
            "inside_observation_radius",
            "inside_basic_radius",
            "inside_ultimate_radius",
            "lane_0_available",
            "lane_1_available",
        ),
        ActorCommand: (
            "actor_global_slot",
            "move_action",
            "target_global_slot",
            "use_ultimate",
        ),
        ScenarioFrame: ("label", "description", "commands"),
        DebuggerScenario: (
            "name",
            "title",
            "description",
            "mode",
            "build_config",
            "frames",
            "default_controlled_slot",
            "audience",
        ),
        ActorTransition: (
            "actor_global_slot",
            "submitted_move_action",
            "submitted_target_action",
            "submitted_use_ultimate",
            "accepted_move_action",
            "accepted_target_action",
            "accepted_use_ultimate",
            "submitted_tuple_in_domain",
            "submitted_move_mask_value",
            "submitted_lane_0_value",
            "submitted_lane_1_value",
            "submitted_pair_mask_value",
            "movement_accepted",
            "combat_pair_accepted",
            "position_before",
            "position_after",
            "realized_displacement",
            "health_before",
            "health_after",
            "net_health_delta",
            "cooldown_before",
            "cooldown_after",
            "effective_speed_before",
            "effective_speed_after",
            "mage_aura_before",
            "mage_aura_after",
            "warrior_aura_before",
            "warrior_aura_after",
        ),
        StatusTransition: (
            "global_slot",
            "status_kind",
            "source_class_id",
            "duration_before",
            "duration_after",
            "change",
        ),
        AcceptedActivation: (
            "kind",
            "source_global_slot",
            "target_global_slot",
            "target_action",
            "use_ultimate",
        ),
        ActionRejection: (
            "actor_global_slot",
            "component",
            "submitted_move_action",
            "submitted_target_action",
            "submitted_use_ultimate",
            "movement_mask_value",
            "pair_mask_value",
        ),
        TransitionView: (
            "scenario_name",
            "submission_kind",
            "report_actor_slots",
            "before_state",
            "before_observation",
            "before_action_mask",
            "submitted_action",
            "accepted_action",
            "after_state",
            "after_observation",
            "after_action_mask",
            "reward",
            "done_flags",
            "info",
            "actor_transitions",
            "status_transitions",
            "accepted_activations",
            "rejections",
        ),
        DebuggerSession: (
            "scenario_name",
            "seed",
            "run_generation",
            "scenario_default_movement_scale",
            "config",
            "key",
            "state",
            "observation",
            "action_mask",
            "last_reward",
            "done_flags",
            "info",
            "controlled_global_slot",
            "pending_actions",
            "next_script_frame_index",
            "last_transition",
            "show_ranges",
            "verbose_logging",
        ),
    }

    for model_type, field_names in expected.items():
        assert tuple(field.name for field in fields(model_type)) == field_names
        assert hasattr(model_type, "__slots__")


def test_pending_action_validates_head_and_arm_invariants() -> None:
    with pytest.raises(ValueError):
        PendingAction(move_action=-1)
    with pytest.raises(ValueError):
        PendingAction(move_action=9)
    with pytest.raises(ValueError):
        PendingAction(armed_lane=None, arm_origin="explicit")
    with pytest.raises(ValueError):
        PendingAction(armed_lane=0, arm_origin=None)
    with pytest.raises(ValueError):
        PendingAction(selected_global_target_slot=MAX_AGENT_SLOTS)
    with pytest.raises(ValueError):
        PendingAction(armed_lane=2, arm_origin="explicit")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        PendingAction(armed_lane=0, arm_origin="implicit")  # type: ignore[arg-type]


def test_create_session_has_exact_initial_pending_and_epoch_contract() -> None:
    session = _session()

    assert session.run_generation == 0
    assert session.scenario_default_movement_scale == 1.0
    assert session.config.ordinary_movement_distance_scale == 1.0
    assert isinstance(session.pending_actions, tuple)
    assert len(session.pending_actions) == MAX_AGENT_SLOTS
    assert session.pending_action is session.pending_actions[0]
    assert session.pending_action == PendingAction(
        move_action=MOVE_STAY,
        selected_global_target_slot=None,
        armed_lane=0,
        arm_origin="automatic",
    )
    assert session.last_reward is None
    assert session.last_transition is None
    assert session.next_script_frame_index == 0
    assert int(session.state.step_count) == 0
    assert not bool(session.done_flags.done)


def test_session_rejects_invalid_fixed_slot_pending_rows() -> None:
    session = _session("basic_support")
    with pytest.raises(ValueError, match="10 fixed-slot rows"):
        replace(session, pending_actions=session.pending_actions[:-1])
    with pytest.raises(ValueError, match="inactive pending row g3"):
        replace(
            session,
            pending_actions=_replace_pending_row(
                session,
                3,
                PendingAction(move_action=MOVE_EAST),
            ),
        )
    with pytest.raises(ValueError, match="pending target g3 is inactive"):
        replace(
            session,
            pending_actions=_replace_pending_row(
                session,
                0,
                PendingAction(selected_global_target_slot=3),
            ),
        )

    inactive = PendingAction(armed_lane=None, arm_origin=None)
    for slot in (3, 4, 8, 9):
        assert session.pending_actions[slot] == inactive


def test_create_session_falls_back_to_scenario_default_for_inactive_slot() -> None:
    session = _session("basic_support", controlled_slot=4)
    assert session.controlled_global_slot == 0


def test_lane_availability_reads_exact_joint_pair_and_disarmed_is_never_legal() -> None:
    session = _session("arena_5v5")
    unavailable = lane_availability(session.action_mask, 0, 6, None)

    assert unavailable.target_action == 6
    assert not unavailable.lane_0_available
    assert not unavailable.lane_1_available
    assert not unavailable.armed_pair_legal

    target_none = lane_availability(session.action_mask, 2, 0, 0)
    assert target_none.lane_0_available
    assert not target_none.lane_1_available
    assert target_none.armed_pair_legal

    with pytest.raises(ValueError):
        lane_availability(session.action_mask, 0, -1, 0)
    with pytest.raises(ValueError):
        lane_availability(session.action_mask, 0, 11, 0)
    with pytest.raises(ValueError):
        lane_availability(session.action_mask, 0, 0, 2)  # type: ignore[arg-type]


def test_interactive_action_populates_only_authorized_pending_rows() -> None:
    session = _session("ultimate_showcase", controlled_slot=1)
    pending = PendingAction(
        move_action=MOVE_NORTH,
        selected_global_target_slot=7,
        armed_lane=1,
        arm_origin="explicit",
    )
    pending_actions = _replace_pending_row(session, 1, pending)
    action = build_interactive_joint_action(
        session.config,
        pending_actions,
        actor_global_slots=(1,),
    )

    assert int(action.move[1]) == MOVE_NORTH
    assert int(action.select_target[1]) == 8
    assert int(action.use_ultimate[1]) == 1
    for slot in set(range(MAX_AGENT_SLOTS)) - {1}:
        assert tuple(int(head[slot]) for head in action) == (0, 0, 0)


def test_disarmed_action_retains_target_for_inspection_but_submits_noop() -> None:
    session = _session("basic_support")
    pending = PendingAction(
        move_action=MOVE_EAST,
        selected_global_target_slot=5,
        armed_lane=None,
        arm_origin=None,
    )
    action = build_interactive_joint_action(
        session.config,
        _replace_pending_row(session, 0, pending),
        actor_global_slots=(0,),
    )

    assert tuple(int(head[0]) for head in action) == (MOVE_EAST, 0, 0)


def test_interactive_builder_rejects_inactive_pending_target() -> None:
    session = _session("basic_support")
    pending = PendingAction(
        selected_global_target_slot=3,
        armed_lane=0,
        arm_origin="explicit",
    )

    with pytest.raises(ValueError, match="inactive"):
        build_interactive_joint_action(
            session.config,
            _replace_pending_row(session, 0, pending),
            actor_global_slots=(0,),
        )


def test_interactive_builder_preserves_all_active_rows_and_neutral_padding() -> None:
    session = _session("basic_support")
    staged = session
    expected_rows: dict[int, tuple[int, int, int]] = {}
    active_slots = (0, 1, 2, 5, 6, 7)
    for index, actor_slot in enumerate(active_slots):
        staged = select_controlled_actor(staged, actor_slot)
        movement = MOVE_EAST if index % 2 == 0 else MOVE_NORTH
        target_slot = active_slots[(index + 1) % len(active_slots)]
        staged = set_pending_movement(staged, movement)
        staged = select_clicked_target(staged, target_slot)
        staged = arm_ultimate(staged) if index % 2 else arm_basic(staged)
        target_action = (
            0
            if staged.pending_action.armed_lane is None
            else int(
                build_interactive_joint_action(
                    staged.config,
                    staged.pending_actions,
                    actor_global_slots=(actor_slot,),
                ).select_target[actor_slot]
            )
        )
        expected_rows[actor_slot] = (
            movement,
            target_action,
            0 if staged.pending_action.armed_lane is None else index % 2,
        )

    action = build_interactive_joint_action(
        staged.config,
        staged.pending_actions,
        actor_global_slots=active_slots,
    )

    for actor_slot, expected in expected_rows.items():
        assert tuple(int(head[actor_slot]) for head in action) == expected
    for actor_slot in (3, 4, 8, 9):
        assert tuple(int(head[actor_slot]) for head in action) == (MOVE_STAY, 0, 0)


def test_scripted_action_supports_multiple_actor_commands() -> None:
    session = _session("basic_support")
    frame = ScenarioFrame(
        "multi",
        "multi",
        (
            ActorCommand(0, MOVE_NORTH, 5, 0),
            ActorCommand(7, MOVE_EAST, 2, 1),
        ),
    )
    action = build_scripted_joint_action(session.config, frame)

    assert tuple(int(head[0]) for head in action) == (MOVE_NORTH, 6, 0)
    assert tuple(int(head[7]) for head in action) == (MOVE_EAST, 8, 1)
    for slot in set(range(MAX_AGENT_SLOTS)) - {0, 7}:
        assert tuple(int(head[slot]) for head in action) == (0, 0, 0)


def test_scripted_builder_rejects_inactive_actor_and_target() -> None:
    session = _session("basic_support")
    inactive_actor = ScenarioFrame("bad", "bad", (ActorCommand(3),))
    inactive_target = ScenarioFrame(
        "bad-target",
        "bad-target",
        (ActorCommand(0, target_global_slot=3),),
    )

    with pytest.raises(ValueError):
        build_scripted_joint_action(session.config, inactive_actor)
    with pytest.raises(ValueError):
        build_scripted_joint_action(session.config, inactive_target)


def test_click_auto_arms_basic_only_when_exact_lane_zero_is_legal() -> None:
    session = _session("arena_5v5")
    illegal = select_clicked_target(session, 5)
    assert illegal.pending_action.selected_global_target_slot == 5
    assert illegal.pending_action.armed_lane is None
    assert illegal.pending_action.arm_origin is None

    target_action = 6
    legal_joint_mask = session.action_mask.select_target_use_ultimate_joint_mask.at[
        0, target_action, 0
    ].set(True)
    exact_legal = replace(
        session,
        action_mask=session.action_mask._replace(
            select_target_use_ultimate_joint_mask=legal_joint_mask
        ),
    )
    legal = select_clicked_target(exact_legal, 5)
    assert legal.pending_action.selected_global_target_slot == 5
    assert legal.pending_action.armed_lane == 0
    assert legal.pending_action.arm_origin == "automatic"


def test_click_rejects_inactive_and_out_of_domain_slots() -> None:
    session = _session("basic_support")
    with pytest.raises(ValueError):
        select_clicked_target(session, 3)
    with pytest.raises(ValueError):
        select_clicked_target(session, MAX_AGENT_SLOTS)


def test_number_keys_arm_illegal_in_domain_pairs() -> None:
    session = select_clicked_target(
        _session("arena_5v5", controlled_slot=2),
        7,
    )
    basic = arm_basic(session)
    ultimate = arm_ultimate(session)

    assert basic.pending_action.armed_lane == 0
    assert basic.pending_action.arm_origin == "explicit"
    assert ultimate.pending_action.armed_lane == 1
    assert ultimate.pending_action.selected_global_target_slot == 7


def test_arm_ultimate_for_mage_clears_target() -> None:
    session = select_clicked_target(_session("arena_5v5", controlled_slot=0), 0)
    assert int(session.config.agent_profile.class_ids[0]) == MAGE_CLASS_ID

    armed = arm_ultimate(session)

    assert armed.pending_action.selected_global_target_slot is None
    assert armed.pending_action.armed_lane == 1
    assert armed.pending_action.arm_origin == "explicit"


def test_explicit_no_combat_preserves_draft_context_but_packages_no_combat() -> None:
    session = select_clicked_target(_session("arena_5v5", controlled_slot=2), 7)
    session = set_pending_movement(session, MOVE_EAST)
    session = arm_ultimate(session)

    no_combat = select_no_combat(session)
    action = build_interactive_joint_action(
        no_combat.config,
        no_combat.pending_actions,
        actor_global_slots=(no_combat.controlled_global_slot,),
    )

    assert no_combat.pending_action.move_action == MOVE_EAST
    assert no_combat.pending_action.selected_global_target_slot == 7
    assert no_combat.pending_action.armed_lane is None
    assert no_combat.pending_action.arm_origin is None
    assert int(action.select_target[2]) == 0
    assert int(action.use_ultimate[2]) == 0


def test_clear_target_applies_class_specific_lane_rule() -> None:
    mage = arm_ultimate(_session("arena_5v5", controlled_slot=0))
    cleared_mage = clear_pending_target(mage)
    assert cleared_mage.pending_action.armed_lane == 1
    assert cleared_mage.pending_action.arm_origin == "explicit"

    hunter = arm_ultimate(_session("arena_5v5", controlled_slot=2))
    cleared_hunter = clear_pending_target(hunter)
    assert cleared_hunter.pending_action.selected_global_target_slot is None
    assert cleared_hunter.pending_action.armed_lane == 0
    assert cleared_hunter.pending_action.arm_origin == "automatic"


def test_cycle_controlled_actor_wraps_both_teams_and_skips_padding() -> None:
    session = _session("basic_support")
    assert cycle_controlled_actor(session, 1).controlled_global_slot == 1
    assert cycle_controlled_actor(session, -1).controlled_global_slot == 7
    last = select_controlled_actor(session, 7)
    assert cycle_controlled_actor(last, 1).controlled_global_slot == 0
    with pytest.raises(ValueError):
        cycle_controlled_actor(session, 0)


def test_actor_selection_preserves_independent_drafts_and_simulator_epoch() -> None:
    session = _session("arena_5v5", controlled_slot=0)
    session = select_clicked_target(session, 6)
    session = set_pending_movement(session, MOVE_EAST)
    session = arm_ultimate(session)
    actor_0_pending = session.pending_action

    switched = select_controlled_actor(session, 1)
    assert switched.pending_action.selected_global_target_slot is None
    switched = select_clicked_target(switched, 5)
    switched = set_pending_movement(switched, MOVE_NORTH)
    switched = arm_basic(switched)
    actor_1_pending = switched.pending_action

    returned = select_controlled_actor(switched, 0)

    assert returned.pending_action == actor_0_pending
    assert returned.pending_actions[1] == actor_1_pending
    assert returned.pending_actions[0] != returned.pending_actions[1]
    assert returned.state is session.state
    assert returned.observation is session.observation
    assert returned.action_mask is session.action_mask
    assert returned.key is session.key


def test_pending_edits_do_not_split_key_or_change_simulator_epoch() -> None:
    session = _session("arena_5v5")
    original_key = session.key
    original_state = session.state
    original_mask = session.action_mask
    edited = set_pending_movement(session, MOVE_EAST)
    edited = arm_basic(edited)
    edited = select_clicked_target(edited, 0)
    edited = clear_pending_target(edited)
    edited = cycle_controlled_actor(edited, 1)

    assert bool(jnp.array_equal(edited.key, original_key))
    assert edited.state is original_state
    assert edited.action_mask is original_mask
    assert int(edited.state.step_count) == 0


def test_submit_joint_action_calls_step_once_and_updates_paired_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session("arena_5v5")
    action = make_neutral_joint_action()
    real_step = control.step
    calls: list[tuple[object, ...]] = []

    def counting_step(*args: object) -> object:
        calls.append(args)
        return real_step(*args)  # type: ignore[arg-type]

    monkeypatch.setattr(control, "step", counting_step)
    submitted = submit_joint_action(
        session,
        action,
        submission_kind="interactive",
        report_actor_slots=(0,),
    )

    assert len(calls) == 1
    assert calls[0][1] is session.state
    assert calls[0][2] is session.action_mask
    assert int(submitted.state.step_count) == 1
    assert submitted.last_transition is not None
    assert submitted.last_transition.after_state is submitted.state
    assert submitted.last_transition.after_observation is submitted.observation
    assert submitted.last_transition.after_action_mask is submitted.action_mask
    assert not bool(jnp.array_equal(submitted.key, session.key))


def test_researcher_joint_submit_sends_all_ten_drafts_in_one_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session("arena_5v5")
    for actor_slot in range(MAX_AGENT_SLOTS):
        session = select_controlled_actor(session, actor_slot)
        session = set_pending_movement(
            session,
            MOVE_EAST if actor_slot % 2 == 0 else MOVE_NORTH,
        )
    expected_action = build_interactive_joint_action(
        session.config,
        session.pending_actions,
        actor_global_slots=tuple(range(MAX_AGENT_SLOTS)),
    )
    real_step = control.step
    calls: list[tuple[object, ...]] = []

    def counting_step(*args: object) -> object:
        calls.append(args)
        return real_step(*args)  # type: ignore[arg-type]

    monkeypatch.setattr(control, "step", counting_step)
    submitted = submit_interactive(session)

    assert len(calls) == 1
    captured_action = cast(Action, calls[0][3])
    assert all(
        bool(jnp.array_equal(actual, expected))
        for actual, expected in zip(captured_action, expected_action, strict=True)
    )
    assert int(submitted.state.step_count) == int(session.state.step_count) + 1
    assert not bool(jnp.array_equal(submitted.key, session.key))
    assert submitted.last_transition is not None
    assert submitted.last_transition.report_actor_slots == tuple(range(MAX_AGENT_SLOTS))
    assert all(
        bool(jnp.array_equal(actual, expected))
        for actual, expected in zip(
            submitted.last_transition.submitted_action,
            expected_action,
            strict=True,
        )
    )


def test_manual_and_scripted_submission_delegate_to_shared_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manual = _session("arena_5v5")
    scripted = _session("basic_support")
    calls: list[tuple[str, tuple[int, ...]]] = []

    def fake_submit(
        session: DebuggerSession,
        _action: object,
        *,
        submission_kind: str,
        report_actor_slots: tuple[int, ...],
    ) -> DebuggerSession:
        calls.append((submission_kind, report_actor_slots))
        return session

    monkeypatch.setattr(control, "submit_joint_action", fake_submit)
    assert submit_interactive(manual) is manual
    assert submit_next_script_frame(scripted) is scripted
    assert calls == [
        ("interactive", tuple(range(MAX_AGENT_SLOTS))),
        ("scripted", (0, 1, 7)),
    ]


def test_post_submit_rebuilds_every_draft_from_the_successor_mask() -> None:
    scenario, session = (
        get_scenario("basic_support"),
        _session("basic_support", controlled_slot=1),
    )
    session = select_clicked_target(session, 6)
    session = set_pending_movement(session, MOVE_EAST)
    session = arm_ultimate(session)
    session = select_controlled_actor(session, 0)
    session = select_clicked_target(session, 5)
    session = set_pending_movement(session, MOVE_WEST)
    before_pending = session.pending_actions
    submitted = submit_interactive(session)

    for actor_slot in (0, 1, 2, 5, 6, 7):
        pending = submitted.pending_actions[actor_slot]
        target_slot = before_pending[actor_slot].selected_global_target_slot
        target_action = global_slot_to_target_action(actor_slot, target_slot)
        basic_available = bool(
            submitted.action_mask.select_target_use_ultimate_joint_mask[
                actor_slot,
                target_action,
                0,
            ]
        )
        assert pending.move_action == MOVE_STAY
        assert pending.selected_global_target_slot == target_slot
        assert pending.armed_lane == (0 if basic_available else None)
        assert pending.arm_origin == ("automatic" if basic_available else None)
    inactive = PendingAction(armed_lane=None, arm_origin=None)
    for actor_slot in (3, 4, 8, 9):
        assert submitted.pending_actions[actor_slot] == inactive
    assert scenario.name == submitted.scenario_name


def test_mage_burst_can_be_explicitly_rearmed_on_cooldown_for_rejection() -> None:
    session = _session("ultimate_showcase", controlled_slot=0)
    session = submit_next_script_frame(session)
    session = submit_next_script_frame(session)
    assert int(session.state.ultimate_cooldowns[0]) == 30

    armed = arm_ultimate(session)
    assert armed.pending_action.selected_global_target_slot is None
    assert armed.pending_action.armed_lane == 1
    assert not lane_availability(armed.action_mask, 0, 0, 1).armed_pair_legal

    submitted = submit_interactive(armed)
    transition = submitted.last_transition
    assert transition is not None
    actor = next(
        value for value in transition.actor_transitions if value.actor_global_slot == 0
    )
    assert actor.movement_accepted
    assert not actor.combat_pair_accepted
    assert (actor.accepted_target_action, actor.accepted_use_ultimate) == (0, 0)
    assert submitted.pending_action.armed_lane == 0
    assert submitted.pending_action.arm_origin == "automatic"


def test_successor_illegal_basic_disarms_but_preserves_target() -> None:
    session = _session("basic_support", controlled_slot=0)
    session = select_clicked_target(session, 5)
    assert session.pending_action.armed_lane == 0
    session = set_pending_movement(session, MOVE_WEST)
    submitted = submit_interactive(session)

    assert submitted.pending_action.selected_global_target_slot == 5
    assert submitted.pending_action.move_action == MOVE_STAY
    assert submitted.pending_action.armed_lane is None
    assert submitted.pending_action.arm_origin is None
    assert not lane_availability(
        submitted.action_mask,
        0,
        6,
        submitted.pending_action.armed_lane,
    ).armed_pair_legal


@pytest.mark.parametrize(
    ("terminated", "truncated", "reason"),
    ((True, False, "terminated"), (False, True, "truncated")),
)
def test_done_session_rejects_submit_without_key_or_state_change(
    capsys: pytest.CaptureFixture[str],
    terminated: bool,
    truncated: bool,
    reason: str,
) -> None:
    session = _session("arena_5v5")
    terminal = replace(
        session,
        done_flags=DoneFlags(
            terminated=jnp.asarray(terminated),
            truncated=jnp.asarray(truncated),
        ),
    )
    result = submit_interactive(terminal)

    assert result is terminal
    assert (
        f"SUBMIT BLOCKED: episode is {reason}; press R or switch scenario."
        in capsys.readouterr().out
    )


def test_script_completion_does_not_step_or_split_key(
    capsys: pytest.CaptureFixture[str],
) -> None:
    session = _session("aura_crossfire")
    session = submit_next_script_frame(session)
    key = session.key
    state = session.state
    completed = submit_next_script_frame(session)

    assert completed is session
    assert completed.state is state
    assert bool(jnp.array_equal(completed.key, key))
    assert "SCRIPT COMPLETE" in capsys.readouterr().out


def test_reset_replays_identical_initial_session_and_clears_history() -> None:
    initial = _session("ultimate_showcase", controlled_slot=2)
    advanced = submit_next_script_frame(initial)
    advanced = select_clicked_target(advanced, 6)
    reset_result = reset_session(advanced)

    assert _tree_equal(reset_result.state, initial.state)
    assert _tree_equal(reset_result.observation, initial.observation)
    assert _tree_equal(reset_result.action_mask, initial.action_mask)
    assert bool(jnp.array_equal(reset_result.key, initial.key))
    assert reset_result.controlled_global_slot == 2
    assert reset_result.pending_action == PendingAction()
    assert reset_result.pending_actions == initial.pending_actions
    assert reset_result.last_transition is None
    assert reset_result.last_reward is None
    assert reset_result.next_script_frame_index == 0
    assert reset_result.run_generation == initial.run_generation + 1


def test_movement_scale_reset_rebuilds_full_epoch_without_stepping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = _session("basic_support", controlled_slot=2, verbose=True)
    advanced = submit_next_script_frame(initial)
    advanced = replace(advanced, show_ranges=False)

    def forbidden_step(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("movement-scale reset must not call step")

    monkeypatch.setattr(control, "step", forbidden_step)
    scaled = set_movement_scale(advanced, 0.25)

    assert scaled is not advanced
    assert scaled.run_generation == advanced.run_generation + 1
    assert scaled.scenario_default_movement_scale == 1.0
    assert scaled.config.ordinary_movement_distance_scale == 0.25
    assert int(scaled.state.step_count) == 0
    assert scaled.state is not advanced.state
    assert scaled.observation is not advanced.observation
    assert scaled.action_mask is not advanced.action_mask
    assert scaled.info is not advanced.info
    assert scaled.last_reward is None
    assert scaled.last_transition is None
    assert scaled.next_script_frame_index == 0
    assert not bool(scaled.done_flags.terminated)
    assert not bool(scaled.done_flags.truncated)
    assert scaled.controlled_global_slot == 2
    assert scaled.show_ranges is False
    assert scaled.verbose_logging is True
    assert scaled.pending_actions == initial.pending_actions


def test_movement_scale_no_op_reset_and_switch_ownership() -> None:
    session = _session("arena_5v5", controlled_slot=3)

    assert set_movement_scale(session, 1.0) is session
    assert set_movement_scale(session, None) is session

    overridden = set_movement_scale(session, 0.1)
    ordinary_reset = reset_session(overridden)
    switched = switch_scenario(
        ordinary_reset,
        get_scenario("status_stack"),
    )
    restored = set_movement_scale(overridden, None)

    assert ordinary_reset.config.ordinary_movement_distance_scale == 0.1
    assert ordinary_reset.scenario_default_movement_scale == 1.0
    assert ordinary_reset.controlled_global_slot == 3
    assert switched.config.ordinary_movement_distance_scale == 1.0
    assert switched.scenario_default_movement_scale == 1.0
    assert switched.controlled_global_slot == 5
    assert restored.config.ordinary_movement_distance_scale == 1.0
    assert restored.scenario_default_movement_scale == 1.0
    assert restored.run_generation == overridden.run_generation + 1


@pytest.mark.parametrize(
    "movement_scale",
    (0.0, 0.009, 1.001, float("nan"), float("inf")),
)
def test_movement_scale_rejects_invalid_direct_values(
    movement_scale: float,
) -> None:
    with pytest.raises(ValueError, match=r"\[0\.01, 1\.0\]"):
        set_movement_scale(_session(), movement_scale)


def test_movement_scale_rejects_non_float_direct_values() -> None:
    with pytest.raises(TypeError, match="Python float"):
        set_movement_scale(_session(), 1)  # type: ignore[arg-type]


def test_switch_scenario_preserves_seed_and_toggles_but_clears_live_state() -> None:
    session = _session("basic_support", verbose=True)
    session = replace(session, show_ranges=False)
    session = submit_next_script_frame(session)
    switched = switch_scenario(session, get_scenario("status_stack"))

    assert switched.scenario_name == "status_stack"
    assert switched.seed == 7
    assert switched.show_ranges is False
    assert switched.verbose_logging is True
    assert switched.controlled_global_slot == 5
    assert int(switched.state.step_count) == 0
    assert switched.last_transition is None
    assert switched.pending_action == PendingAction()
    inactive = PendingAction(armed_lane=None, arm_origin=None)
    assert len(switched.pending_actions) == MAX_AGENT_SLOTS
    for actor_slot, active in enumerate(switched.config.agent_profile.active_mask):
        assert switched.pending_actions[actor_slot] == (
            PendingAction() if bool(active) else inactive
        )
    assert switched.run_generation == session.run_generation + 1


@pytest.mark.parametrize(
    "bad_action",
    (
        make_neutral_joint_action()._replace(
            move=jnp.zeros((MAX_AGENT_SLOTS - 1,), dtype=jnp.int32)
        ),
        make_neutral_joint_action()._replace(
            select_target=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.float32)
        ),
    ),
)
def test_submission_rejects_bad_shape_or_dtype_before_stepping(
    bad_action: object,
) -> None:
    session = _session("arena_5v5")
    with pytest.raises(ValueError):
        submit_joint_action(
            session,
            bad_action,  # type: ignore[arg-type]
            submission_kind="interactive",
            report_actor_slots=(0,),
        )


def test_scenario_contract_rejects_duplicate_frame_actors() -> None:
    with pytest.raises(ValueError):
        ScenarioFrame(
            "duplicate",
            "duplicate",
            (ActorCommand(0), ActorCommand(0)),
        )


def test_debugger_scenario_validates_default_fixed_slot() -> None:
    with pytest.raises(ValueError):
        DebuggerScenario(
            "bad",
            "bad",
            "bad",
            "interactive",
            get_scenario("arena_5v5").build_config,
            (),
            MAX_AGENT_SLOTS,
        )

    with pytest.raises(ValueError, match="audience"):
        DebuggerScenario(
            "bad-audience",
            "bad",
            "bad",
            "interactive",
            get_scenario("arena_5v5").build_config,
            (),
            0,
            "public",  # type: ignore[arg-type]
        )
