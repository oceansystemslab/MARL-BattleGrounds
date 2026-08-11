"""Canonical evaluation-record diagnostics for live debugger transitions."""

import pytest
from scripts.dev.visual_debugger.control import (
    arm_ultimate,
    create_session,
    make_neutral_joint_action,
    reset_session,
    submit_joint_action,
    submit_next_script_frame,
)
from scripts.dev.visual_debugger.model import DebuggerSession
from scripts.dev.visual_debugger.scenarios import get_scenario
from tests.visual_debugger_fixtures import (
    debugger_test_launch_specification,
    rejection_lane_scenario,
    submit_fixture_frame,
)

from marl_battlegrounds.core.types import MOVE_EAST, MOVE_STAY, NUM_TARGET_ACTIONS
from marl_battlegrounds.evaluation.models import ActionRejectedEventV1
from marl_battlegrounds.rendering.evaluation_adapter import (
    build_researcher_analyzer_projection_v2,
    build_visual_event_batch_v2,
)
from marl_battlegrounds.rendering.scene import VisualEventBatchV2


def _session(name: str) -> DebuggerSession:
    return create_session(
        get_scenario(name),
        seed=0,
        evaluation_launch_specification=debugger_test_launch_specification(),
        controlled_global_slot=None,
        show_ranges=True,
        verbose_logging=False,
    )


def _rejection_session() -> DebuggerSession:
    scenario = rejection_lane_scenario()
    return create_session(
        scenario,
        seed=0,
        evaluation_launch_specification=debugger_test_launch_specification(),
        controlled_global_slot=None,
        show_ranges=True,
        verbose_logging=False,
    )


def test_canonical_facts_report_accepted_move_and_rejected_combat_pair() -> None:
    scenario = rejection_lane_scenario()
    submitted = submit_fixture_frame(_rejection_session(), scenario.frames[0])
    view = submitted.incoming_evaluation_view
    assert view is not None
    facts = view.transition.facts.action_acceptance_facts

    assert facts.submitted_joint_action.move[0] == MOVE_EAST
    assert facts.accepted_joint_action.move[0] == MOVE_EAST
    assert facts.submitted_joint_action.select_target[0] == 6
    assert facts.accepted_joint_action.select_target[0] == 0
    assert not facts.submitted_action_tuple_is_out_of_domain_by_actor[0]
    assert not facts.in_domain_move_action_is_rejected_by_actor[0]
    assert facts.in_domain_combat_action_pair_is_rejected_by_actor[0]
    rejections = tuple(
        event
        for event in view.transition.events
        if isinstance(event, ActionRejectedEventV1)
    )
    assert tuple(
        (event.actor_global_slot, event.rejection_component) for event in rejections
    ) == ((0, "combat_pair"),)


@pytest.mark.parametrize(
    ("head_name", "value"),
    (("move", -1), ("select_target", NUM_TARGET_ACTIONS), ("use_ultimate", 2)),
)
def test_out_of_domain_tuple_is_canonicalized_without_mask_indexing(
    head_name: str,
    value: int,
) -> None:
    session = _rejection_session()
    action = make_neutral_joint_action()
    action = action._replace(**{head_name: getattr(action, head_name).at[0].set(value)})

    submitted = submit_joint_action(
        session,
        action,
        submission_kind="interactive",
        report_actor_slots=(0,),
    )
    view = submitted.incoming_evaluation_view
    assert view is not None
    facts = view.transition.facts.action_acceptance_facts

    assert facts.submitted_action_tuple_is_out_of_domain_by_actor[0]
    assert (
        facts.accepted_joint_action.move[0],
        facts.accepted_joint_action.select_target[0],
        facts.accepted_joint_action.use_ultimate[0],
    ) == (MOVE_STAY, 0, 0)
    domain_rejections = tuple(
        event
        for event in view.transition.events
        if isinstance(event, ActionRejectedEventV1)
        and event.rejection_component == "domain"
    )
    assert tuple(event.actor_global_slot for event in domain_rejections) == (0,)


def test_visual_event_batch_is_an_order_preserving_projection_of_cp2_events() -> None:
    session = submit_next_script_frame(_session("ultimate_showcase"))
    session = submit_next_script_frame(session)
    view = session.incoming_evaluation_view
    assert view is not None

    batch = build_visual_event_batch_v2(view)

    assert type(batch) is VisualEventBatchV2
    assert batch.transition_id == view.transition.transition_id
    assert batch.start_frame_id == view.start_frame.frame_id
    assert batch.successor_frame_id == view.successor_frame.frame_id
    assert tuple(event.event_id for event in batch.events) == tuple(
        event.event_id for event in view.transition.events
    )
    assert tuple(event.ordinal for event in batch.events) == tuple(
        range(len(batch.events))
    )


def test_live_researcher_projection_uses_the_exact_incoming_cp2_event_ids() -> None:
    session = submit_next_script_frame(_session("basic_support"))
    view = session.incoming_evaluation_view
    assert view is not None

    projection = build_researcher_analyzer_projection_v2(
        session.evaluation_context,
        session.current_evaluation_frame,
        transition_view=view,
        status_source_evidence_state=session.status_source_evidence_state,
    )

    assert projection.incoming_events is not None
    assert tuple(event.event_id for event in projection.incoming_events.events) == (
        tuple(event.event_id for event in view.transition.events)
    )
    assert projection.scene.incoming_event_ids == tuple(
        event.event_id for event in view.transition.events
    )


def test_ui_only_edits_preserve_canonical_transition_and_projection_events() -> None:
    session = submit_next_script_frame(_session("basic_support"))
    view = session.incoming_evaluation_view
    assert view is not None
    before = build_visual_event_batch_v2(view)

    edited = arm_ultimate(session)
    assert edited.incoming_evaluation_view is view
    after = build_visual_event_batch_v2(view)

    assert after == before
    assert edited.current_evaluation_frame == session.current_evaluation_frame


def test_reset_replaces_the_scientific_episode_and_clears_incoming_events() -> None:
    submitted = submit_next_script_frame(_session("basic_support"))
    reset = reset_session(submitted)

    assert reset.run_generation == submitted.run_generation + 1
    assert reset.evaluation_context.identity.episode_id != (
        submitted.evaluation_context.identity.episode_id
    )
    assert reset.current_evaluation_frame.frame_index == 0
    assert reset.incoming_evaluation_view is None
    projection = build_researcher_analyzer_projection_v2(
        reset.evaluation_context,
        reset.current_evaluation_frame,
        status_source_evidence_state=reset.status_source_evidence_state,
    )
    assert projection.incoming_events is None
    assert projection.scene.incoming_event_ids == ()
