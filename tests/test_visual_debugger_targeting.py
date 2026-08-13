"""Exhaustive debugger parity tests for the canonical target mapping."""

import pytest
from scripts.dev.visual_debugger.control import (
    create_session,
    select_clicked_target,
    submit_interactive,
)
from scripts.dev.visual_debugger.scenarios import get_scenario
from scripts.dev.visual_debugger.targeting import (
    global_slot_to_target_action,
    target_action_to_global_slot,
)
from tests.visual_debugger_fixtures import debugger_test_launch_specification

from marl_battlegrounds.core.types import MAX_AGENT_SLOTS, NUM_TARGET_ACTIONS


@pytest.mark.parametrize("actor_global_slot", range(MAX_AGENT_SLOTS))
def test_target_none_round_trips(actor_global_slot: int) -> None:
    assert global_slot_to_target_action(actor_global_slot, None) == 0
    assert target_action_to_global_slot(actor_global_slot, 0) is None


@pytest.mark.parametrize("actor_global_slot", range(MAX_AGENT_SLOTS))
@pytest.mark.parametrize("target_global_slot", range(MAX_AGENT_SLOTS))
def test_target_mapping_round_trips_all_fixed_slots(
    actor_global_slot: int,
    target_global_slot: int,
) -> None:
    target_action = global_slot_to_target_action(
        actor_global_slot,
        target_global_slot,
    )

    assert 1 <= target_action < NUM_TARGET_ACTIONS
    assert (
        target_action_to_global_slot(actor_global_slot, target_action)
        == target_global_slot
    )


@pytest.mark.parametrize(
    ("actor_global_slot", "target_global_slot", "expected_target_action"),
    (
        (0, 0, 1),
        (0, 4, 5),
        (0, 5, 6),
        (0, 9, 10),
        (5, 5, 1),
        (5, 9, 5),
        (5, 0, 6),
        (5, 4, 10),
    ),
)
def test_target_mapping_matches_fixed_team_blocks(
    actor_global_slot: int,
    target_global_slot: int,
    expected_target_action: int,
) -> None:
    assert (
        global_slot_to_target_action(actor_global_slot, target_global_slot)
        == expected_target_action
    )


@pytest.mark.parametrize("actor_global_slot", (-1, MAX_AGENT_SLOTS))
def test_target_mapping_rejects_invalid_actor_slots(actor_global_slot: int) -> None:
    with pytest.raises(ValueError):
        global_slot_to_target_action(actor_global_slot, None)
    with pytest.raises(ValueError):
        target_action_to_global_slot(actor_global_slot, 0)


@pytest.mark.parametrize("target_global_slot", (-1, MAX_AGENT_SLOTS))
def test_target_mapping_rejects_invalid_target_slots(target_global_slot: int) -> None:
    with pytest.raises(ValueError):
        global_slot_to_target_action(0, target_global_slot)


@pytest.mark.parametrize("target_action", (-1, NUM_TARGET_ACTIONS))
def test_target_mapping_rejects_invalid_target_actions(target_action: int) -> None:
    with pytest.raises(ValueError):
        target_action_to_global_slot(0, target_action)


@pytest.mark.parametrize(
    ("actor_slot", "target_slot", "expected_target_action", "expected_health"),
    (
        (0, 5, 6, 65.05),
        (7, 2, 8, 94.0),
    ),
)
def test_clicked_global_target_routes_to_expected_recipient_for_both_teams(
    actor_slot: int,
    target_slot: int,
    expected_target_action: int,
    expected_health: float,
) -> None:
    scenario = get_scenario("basic_support")
    session = create_session(
        scenario,
        seed=0,
        evaluation_launch_specification=debugger_test_launch_specification(),
        controlled_global_slot=actor_slot,
        show_ranges=False,
        verbose_logging=False,
    )
    session = select_clicked_target(session, target_slot)
    submitted = submit_interactive(session)
    view = submitted.incoming_evaluation_view
    assert view is not None
    acceptance = view.transition.facts.action_acceptance_facts

    assert acceptance.accepted_joint_action.select_target[actor_slot] == (
        expected_target_action
    )
    assert not acceptance.submitted_action_tuple_is_out_of_domain_by_actor[actor_slot]
    assert not acceptance.in_domain_combat_action_pair_is_rejected_by_actor[actor_slot]
    assert view.successor_frame.snapshot.current_health[target_slot] == pytest.approx(
        expected_health
    )
