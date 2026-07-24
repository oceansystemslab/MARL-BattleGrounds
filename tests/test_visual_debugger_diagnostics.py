"""Edge-heavy tests for target facts, transition inference, logs, and history."""

from dataclasses import FrozenInstanceError

import jax.numpy as jnp
import numpy as np
import pytest
from scripts.dev.visual_debugger.control import (
    arm_ultimate,
    build_interactive_joint_action,
    create_session,
    cycle_controlled_actor,
    make_neutral_joint_action,
    select_clicked_target,
    submit_interactive,
    submit_joint_action,
    submit_next_script_frame,
)
from scripts.dev.visual_debugger.diagnostics import (
    accepted_action_from_successor,
    age_transient_history,
    derive_selected_target_facts,
    extract_transition_view,
    format_concise_transition,
    format_verbose_transition,
)
from scripts.dev.visual_debugger.model import (
    DebuggerSession,
    SelectedTargetFacts,
    TransientHistoryEntry,
)
from scripts.dev.visual_debugger.scenarios import get_scenario

from marl_battlegrounds.core.geometry import has_clear_line_of_sight
from marl_battlegrounds.core.types import (
    MAX_AGENT_SLOTS,
    MOVE_EAST,
    MOVE_STAY,
    NUM_TARGET_ACTIONS,
)
from marl_battlegrounds.rendering import ChargeTrailVisual, HealthDeltaVisual


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


def _facts(session: DebuggerSession, target: int) -> SelectedTargetFacts:
    facts = derive_selected_target_facts(
        config=session.config,
        state=session.state,
        observation=session.observation,
        action_mask=session.action_mask,
        controlled_global_slot=session.controlled_global_slot,
        target_global_slot=target,
    )
    assert facts is not None
    return facts


@pytest.mark.parametrize(
    ("actor", "target", "relation"),
    (
        (0, 0, "self"),
        (0, 1, "ally"),
        (0, 5, "enemy"),
        (5, 5, "self"),
        (5, 6, "ally"),
        (5, 0, "enemy"),
    ),
)
def test_selected_target_facts_classify_self_ally_enemy_for_both_teams(
    actor: int,
    target: int,
    relation: str,
) -> None:
    facts = _facts(_session("arena_5v5", actor), target)
    assert facts.relation == relation


def test_selected_target_facts_are_frozen_and_on_demand() -> None:
    session = _session("arena_5v5")
    assert (
        derive_selected_target_facts(
            config=session.config,
            state=session.state,
            observation=session.observation,
            action_mask=session.action_mask,
            controlled_global_slot=0,
            target_global_slot=None,
        )
        is None
    )
    facts = _facts(session, 1)
    with pytest.raises(FrozenInstanceError):
        facts.center_distance = 0.0  # type: ignore[misc]


def test_selected_target_facts_report_exact_center_distance_and_public_los() -> None:
    session = _session("arena_5v5")
    observed_los_values: set[bool] = set()
    for target in range(MAX_AGENT_SLOTS):
        facts = _facts(session, target)
        expected_distance = float(
            np.linalg.norm(
                np.asarray(
                    session.state.agent_positions[target]
                    - session.state.agent_positions[0]
                )
            )
        )
        expected_los = bool(
            has_clear_line_of_sight(
                session.state.agent_positions[0],
                session.state.agent_positions[target],
                session.config.obstacles,
            )
        )
        assert facts.center_distance == pytest.approx(expected_distance)
        assert facts.has_clear_line_of_sight is expected_los
        observed_los_values.add(expected_los)
    assert observed_los_values == {False, True}


@pytest.mark.parametrize(
    ("actor", "target"),
    ((0, 0), (0, 4), (0, 5), (0, 9), (5, 5), (5, 9), (5, 0), (5, 4)),
)
def test_selected_target_facts_read_relation_local_visibility_masks(
    actor: int,
    target: int,
) -> None:
    session = _session("arena_5v5", actor)
    facts = _facts(session, target)
    same_team = int(session.config.agent_profile.team_ids[actor]) == int(
        session.config.agent_profile.team_ids[target]
    )
    if actor < 5:
        row = target if same_team else target - 5
    else:
        row = target - 5 if same_team else target
    expected = bool(
        session.observation.ally_visibility_mask[actor, row]
        if same_team
        else session.observation.enemy_visibility_mask[actor, row]
    )
    assert facts.observer_visible is expected


def test_selected_target_facts_report_individual_inclusive_ranges() -> None:
    scenario = get_scenario("acceptance_lane_lab")
    session = _session("acceptance_lane_lab")
    expected = (
        (False, False, False),
        (False, False, False),
        (False, False, False),
        (True, False, False),
        (True, True, False),
        (True, True, True),
    )
    for expected_membership in expected:
        facts = _facts(session, 5)
        assert (
            facts.inside_observation_radius,
            facts.inside_basic_radius,
            facts.inside_ultimate_radius,
        ) == expected_membership
        if int(session.state.step_count) < len(scenario.frames):
            session = submit_next_script_frame(session)


def test_nontargeted_mage_ultimate_range_is_not_applicable() -> None:
    session = _session("arena_5v5", 0)
    facts = _facts(session, 5)
    assert facts.inside_ultimate_radius is None


def test_selected_target_facts_do_not_derive_lanes_from_geometry() -> None:
    session = _session("acceptance_lane_lab")
    facts = _facts(session, 5)
    assert facts.has_clear_line_of_sight
    assert not facts.inside_observation_radius
    joint = session.action_mask.select_target_use_ultimate_joint_mask.at[
        0,
        6,
    ].set(jnp.asarray((True, True)))
    fake_mask = session.action_mask._replace(
        select_target_use_ultimate_joint_mask=joint
    )
    overridden = derive_selected_target_facts(
        config=session.config,
        state=session.state,
        observation=session.observation,
        action_mask=fake_mask,
        controlled_global_slot=0,
        target_global_slot=5,
    )
    assert overridden is not None
    assert overridden.lane_0_available
    assert overridden.lane_1_available
    assert not overridden.inside_observation_radius


def test_selected_target_facts_reject_inactive_actor_and_target() -> None:
    session = _session("acceptance_lane_lab")
    with pytest.raises(ValueError):
        derive_selected_target_facts(
            config=session.config,
            state=session.state,
            observation=session.observation,
            action_mask=session.action_mask,
            controlled_global_slot=1,
            target_global_slot=5,
        )
    with pytest.raises(ValueError):
        derive_selected_target_facts(
            config=session.config,
            state=session.state,
            observation=session.observation,
            action_mask=session.action_mask,
            controlled_global_slot=0,
            target_global_slot=1,
        )


def test_selected_target_facts_recompute_after_actor_switch_and_successor_step() -> (
    None
):
    session = select_clicked_target(_session("acceptance_lane_lab"), 5)
    before = _facts(session, 5)
    assert before.relation == "enemy"
    assert before.target_action == 6
    assert before.center_distance == pytest.approx(9.0)

    switched = cycle_controlled_actor(session, 1)
    assert switched.pending_action.selected_global_target_slot == 5
    switched_facts = _facts(switched, 5)
    assert switched_facts.controlled_global_slot == 5
    assert switched_facts.relation == "self"
    assert switched_facts.target_action == 1
    assert switched_facts.center_distance == pytest.approx(0.0)
    assert switched_facts.observer_visible

    successor = submit_next_script_frame(session)
    assert successor.pending_action.selected_global_target_slot == 5
    successor_facts = _facts(successor, 5)
    assert successor_facts.center_distance == pytest.approx(8.0)
    assert successor_facts.target_action == 6


def test_accepted_action_comes_from_successor_history() -> None:
    session = _session("acceptance_lane_lab")
    with pytest.raises(ValueError):
        accepted_action_from_successor(session.state)
    submitted = submit_next_script_frame(session)
    accepted = accepted_action_from_successor(submitted.state)
    assert accepted.move is submitted.state.previous_timestep_move_actions
    assert accepted.select_target is (
        submitted.state.previous_timestep_select_target_actions
    )
    assert accepted.use_ultimate is (
        submitted.state.previous_timestep_use_ultimate_actions
    )


def test_diagnostics_report_accepted_move_and_rejected_combat() -> None:
    session = submit_next_script_frame(_session("acceptance_lane_lab"))
    transition = session.last_transition
    assert transition is not None
    actor = transition.actor_transitions[0]
    assert actor.submitted_move_action == MOVE_EAST
    assert actor.accepted_move_action == MOVE_EAST
    assert actor.movement_accepted
    assert actor.submitted_target_action == 6
    assert actor.accepted_target_action == 0
    assert not actor.combat_pair_accepted
    assert [
        (item.component, item.actor_global_slot) for item in transition.rejections
    ] == [("combat", 0)]


def test_out_of_domain_tuple_is_classified_without_unsafe_mask_indexing() -> None:
    session = _session("acceptance_lane_lab")
    action = make_neutral_joint_action()._replace(
        select_target=make_neutral_joint_action()
        .select_target.at[0]
        .set(NUM_TARGET_ACTIONS)
    )
    submitted = submit_joint_action(
        session,
        action,
        submission_kind="interactive",
        report_actor_slots=(0,),
    )
    transition = submitted.last_transition
    assert transition is not None
    actor = transition.actor_transitions[0]
    assert not actor.submitted_tuple_in_domain
    assert not actor.movement_accepted
    assert not actor.combat_pair_accepted
    assert transition.rejections[0].component == "complete_tuple_domain"
    assert tuple(int(head[0]) for head in transition.accepted_action) == (0, 0, 0)


def test_health_delta_is_exact_public_net_without_fabricating_zero_delta_visual() -> (
    None
):
    session = _session("basic_support")
    session = submit_next_script_frame(session)
    session = submit_next_script_frame(session)
    transition = session.last_transition
    assert transition is not None
    actor = next(
        value for value in transition.actor_transitions if value.actor_global_slot == 2
    )
    assert actor.net_health_delta == actor.health_after - actor.health_before == 0.0
    health_visuals = [
        entry.visual
        for entry in session.transient_history
        if isinstance(entry.visual, HealthDeltaVisual)
    ]
    assert health_visuals == []


def test_non_health_activations_do_not_create_zero_health_delta_visuals() -> None:
    session = _session("ultimate_showcase")
    session = submit_next_script_frame(session)
    session = submit_next_script_frame(session)

    health_visual_slots = {
        entry.visual.global_slot
        for entry in session.transient_history
        if isinstance(entry.visual, HealthDeltaVisual)
    }

    assert health_visual_slots == {2, 7}
    assert 5 not in health_visual_slots  # Rogue Poison carries no direct health delta.
    assert 6 not in health_visual_slots  # Hunter Trap carries no direct health delta.


def test_status_application_refresh_decrement_expiration_and_trap_break() -> None:
    session = _session("ultimate_showcase")
    session = submit_next_script_frame(session)
    session = submit_next_script_frame(session)
    transition = session.last_transition
    assert transition is not None
    changed = {
        (status.global_slot, status.status_kind): status.change
        for status in transition.status_transitions
        if status.change != "unchanged"
    }
    assert changed[(0, "mage_burst")] == "applied"
    assert changed[(7, "slow_warrior_charge")] == "applied"
    assert changed[(7, "stun_warrior_charge")] == "applied"
    assert changed[(6, "stun_hunter_trap")] == "applied"
    assert changed[(5, "slow_rogue_poison")] == "applied"
    assert changed[(5, "stun_rogue_poison")] == "applied"
    assert changed[(5, "anti_heal_rogue_poison")] == "applied"

    session = submit_next_script_frame(session)
    transition = session.last_transition
    assert transition is not None
    changed = {
        (status.global_slot, status.status_kind): status.change
        for status in transition.status_transitions
        if status.change != "unchanged"
    }
    assert changed[(6, "stun_hunter_trap")] == "trap_broken"
    assert changed[(6, "slow_hunter_basic")] == "applied"
    assert changed[(7, "slow_warrior_charge")] == "decremented"
    assert changed[(7, "stun_warrior_charge")] == "expired"
    assert changed[(5, "stun_rogue_poison")] == "expired"

    support = _session("basic_support")
    support = submit_next_script_frame(support)
    support = submit_next_script_frame(support)
    transition = support.last_transition
    assert transition is not None
    changed = {
        (status.global_slot, status.status_kind): status.change
        for status in transition.status_transitions
        if status.change != "unchanged"
    }
    assert changed[(2, "slow_hunter_basic")] == "refreshed"
    assert changed[(6, "slow_hunter_basic")] == "expired"
    assert changed[(2, "priest_freedom")] == "applied"


def test_trap_one_to_zero_with_accepted_damage_is_not_overclaimed() -> None:
    scenario = get_scenario("acceptance_lane_lab")
    session = _session("acceptance_lane_lab")
    for _ in scenario.frames:
        session = submit_next_script_frame(session)
    # Trap is now 4. Three canonical neutral steps expose durations 3, 2, 1.
    for _ in range(3):
        session = submit_joint_action(
            session,
            make_neutral_joint_action(),
            submission_kind="interactive",
            report_actor_slots=(0,),
        )
    assert int(session.state.stun_durations[5, 1]) == 1
    session = select_clicked_target(session, 5)
    damaged = submit_interactive(session)
    transition = damaged.last_transition
    assert transition is not None
    trap = next(
        status
        for status in transition.status_transitions
        if status.global_slot == 5 and status.status_kind == "stun_hunter_trap"
    )
    assert trap.duration_before == 1
    assert trap.duration_after == 0
    assert trap.change == "cleared_unclassified"


def test_charge_trail_distinguishes_stay_and_movement_and_uses_realized_endpoints() -> (
    None
):
    showcase = _session("ultimate_showcase")
    showcase = submit_next_script_frame(showcase)
    showcase = submit_next_script_frame(showcase)
    charge = next(
        entry.visual
        for entry in showcase.transient_history
        if isinstance(entry.visual, ChargeTrailVisual)
    )
    assert charge.path_kind == "charge_only"
    assert charge.start == (5.0, 5.0)
    np.testing.assert_allclose(charge.end, (9.0715, 3.3714), atol=1e-4)

    stack = submit_next_script_frame(_session("status_stack"))
    combined = next(
        entry.visual
        for entry in stack.transient_history
        if isinstance(entry.visual, ChargeTrailVisual)
    )
    assert combined.path_kind == "combined_charge_and_movement"
    assert combined.start == (3.0, 6.0)
    assert combined.end == (7.0, 7.0)


def test_transient_history_ages_only_on_steps_with_charge_opacity_schedule() -> None:
    session = _session("ultimate_showcase")
    session = submit_next_script_frame(session)
    session = submit_next_script_frame(session)
    charge_entries = tuple(
        entry
        for entry in session.transient_history
        if isinstance(entry.visual, ChargeTrailVisual)
    )
    assert len(charge_entries) == 1
    assert charge_entries[0].age_submitted_steps == 0
    initial_charge = charge_entries[0].visual
    assert isinstance(initial_charge, ChargeTrailVisual)
    assert initial_charge.opacity == 1.0

    same_entries = session.transient_history
    # Pure inspection/redraw paths consume history but never age it.
    assert session.transient_history is same_entries

    session = submit_next_script_frame(session)
    charge = next(
        entry
        for entry in session.transient_history
        if isinstance(entry.visual, ChargeTrailVisual)
    )
    assert charge.age_submitted_steps == 1
    age_one_visual = charge.visual
    assert isinstance(age_one_visual, ChargeTrailVisual)
    assert age_one_visual.opacity == 0.65
    session = submit_joint_action(
        session,
        make_neutral_joint_action(),
        submission_kind="interactive",
        report_actor_slots=(0,),
    )
    charge = next(
        entry
        for entry in session.transient_history
        if isinstance(entry.visual, ChargeTrailVisual)
    )
    assert charge.age_submitted_steps == 2
    age_two_visual = charge.visual
    assert isinstance(age_two_visual, ChargeTrailVisual)
    assert age_two_visual.opacity == 0.35
    session = submit_joint_action(
        session,
        make_neutral_joint_action(),
        submission_kind="interactive",
        report_actor_slots=(0,),
    )
    assert not any(
        isinstance(entry.visual, ChargeTrailVisual)
        for entry in session.transient_history
    )


def test_age_transient_history_preserves_sequence_and_rejects_invalid_age() -> None:
    entry = TransientHistoryEntry(
        visual=ChargeTrailVisual(0, (0.0, 0.0), (1.0, 0.0), 5, "charge_only", 1.0),
        created_after_step=1,
        age_submitted_steps=0,
        max_age_submitted_steps=3,
        sequence_number=4,
    )
    aged = age_transient_history((entry,))
    assert aged[0].sequence_number == 4
    assert aged[0].age_submitted_steps == 1
    with pytest.raises(ValueError):
        TransientHistoryEntry(
            visual=entry.visual,
            created_after_step=1,
            age_submitted_steps=3,
            max_age_submitted_steps=3,
            sequence_number=0,
        )
    with pytest.raises(ValueError):
        TransientHistoryEntry(
            visual=entry.visual,
            created_after_step=-1,
            age_submitted_steps=0,
            max_age_submitted_steps=1,
            sequence_number=0,
        )


def test_concise_log_schema_covers_accepted_basic_and_rejected_ultimate() -> None:
    basic = submit_next_script_frame(_session("basic_support"))
    transition = basic.last_transition
    assert transition is not None
    text = format_concise_transition(transition)
    assert "STEP scenario=basic_support 0->1 terminated=0 truncated=0" in text
    assert (
        "ACTOR g0 A/Mage target=g5/t6 lanes=0:1,1:0 submitted=Stay[0],Basic[0]"
    ) in text
    assert "HEALTH g5 80.00->66.20 net=-13.80" in text
    assert "EVENT basic_damage source=g0 recipient=g5" in text

    lane = _session("acceptance_lane_lab")
    for _ in range(5):
        lane = submit_next_script_frame(lane)
    transition = lane.last_transition
    assert transition is not None
    text = format_concise_transition(transition)
    assert "submitted=East[3],Ultimate[1]" in text
    assert "accepted=East[3],t0,u0" in text
    assert "movement=ACCEPTED combat=REJECTED" in text
    assert "EVENT rejection component=combat actor=g0 mask=0" in text


def test_concise_logs_cover_all_required_effect_and_lifecycle_examples() -> None:
    showcase = _session("ultimate_showcase")
    showcase = submit_next_script_frame(showcase)
    showcase = submit_next_script_frame(showcase)
    transition = showcase.last_transition
    assert transition is not None
    text = format_concise_transition(transition)
    for expected in (
        "EVENT mage_burst source=g0 recipient=none",
        "EVENT warrior_charge source=g1 recipient=g7 path=charge_only",
        "EVENT hunter_trap source=g2 recipient=g6",
        "EVENT rogue_poison source=g3 recipient=g5",
        "EVENT holy_word source=g4 recipient=g2",
        "COOLDOWN g0 0->30 started",
        "STATUS g6 stun_hunter_trap 0->4 applied",
    ):
        assert expected in text

    showcase = submit_next_script_frame(showcase)
    transition = showcase.last_transition
    assert transition is not None
    text = format_concise_transition(transition)
    assert "STATUS g6 stun_hunter_trap 4->0 trap_broken" in text
    assert "COOLDOWN g0 30->29 decremented" in text
    assert "STATUS g5 stun_rogue_poison 1->0 expired" in text

    support = _session("basic_support")
    support = submit_next_script_frame(support)
    support = submit_next_script_frame(support)
    transition = support.last_transition
    assert transition is not None
    text = format_concise_transition(transition)
    assert "EVENT basic_heal source=g2 recipient=g2" in text
    assert "EVENT basic_damage source=g7 recipient=g2" in text
    assert "HEALTH g2 92.00->92.00 net=+0.00 gross contributions not exposed" in text


def test_verbose_log_adds_geometry_mask_aura_speed_and_episode_fields() -> None:
    session = _session("acceptance_lane_lab")
    for _ in range(6):
        session = submit_next_script_frame(session)
    transition = session.last_transition
    assert transition is not None
    text = format_verbose_transition(transition)

    assert "TARGET actor=g0 global=g5 relative=t6 relation=enemy distance=4.00" in text
    assert (
        "GEOMETRY actor=g0 target=g5 los=1 visible=1 observation_range=1 "
        "basic_range=1 ultimate_range=1"
    ) in text
    assert "MASK actor=g0 move[0]=1 target=t6 lane0=1 lane1=1" in text
    assert "SUBMITTED actor=g0 move=Stay[0] target=g5/t6 use_ultimate=1" in text
    assert "ACCEPTED actor=g0 move=Stay[0] target=t6 use_ultimate=1" in text
    assert "AURA actor=g0 mage=" in text
    assert "SPEED actor=g0" in text
    assert "EPISODE reward=+0.00 terminated=0 truncated=0" in text
    assert "Array(" not in text
    assert "[[" not in text


def test_geometry_facts_are_not_logged_as_rejection_causes() -> None:
    session = submit_next_script_frame(_session("acceptance_lane_lab"))
    transition = session.last_transition
    assert transition is not None
    text = format_verbose_transition(transition)
    assert "los=1" in text
    assert "visible=0" in text
    assert "component=combat" in text
    for forbidden in (
        "rejected because",
        "cause=los",
        "cause=range",
        "cause=visibility",
    ):
        assert forbidden not in text.lower()


def test_extract_transition_retains_every_public_before_after_artifact() -> None:
    before = _session("acceptance_lane_lab")
    after = submit_next_script_frame(before)
    transition = after.last_transition
    assert transition is not None
    reconstructed = extract_transition_view(
        scenario_name=transition.scenario_name,
        submission_kind=transition.submission_kind,
        report_actor_slots=transition.report_actor_slots,
        before_state=transition.before_state,
        before_observation=transition.before_observation,
        before_action_mask=transition.before_action_mask,
        submitted_action=transition.submitted_action,
        after_state=transition.after_state,
        after_observation=transition.after_observation,
        after_action_mask=transition.after_action_mask,
        reward=transition.reward,
        done_flags=transition.done_flags,
        info=transition.info,
    )
    assert reconstructed.before_state is transition.before_state
    assert reconstructed.before_observation is transition.before_observation
    assert reconstructed.before_action_mask is transition.before_action_mask
    assert reconstructed.submitted_action is transition.submitted_action
    assert reconstructed.after_state is transition.after_state
    assert reconstructed.after_observation is transition.after_observation
    assert reconstructed.after_action_mask is transition.after_action_mask
    assert reconstructed.reward is transition.reward
    assert reconstructed.done_flags is transition.done_flags
    assert reconstructed.info is transition.info


def test_interactive_builder_does_not_suppress_illegal_pair_before_diagnostics() -> (
    None
):
    session = select_clicked_target(_session("acceptance_lane_lab"), 5)
    session = arm_ultimate(session)
    action = build_interactive_joint_action(
        session.config,
        session.controlled_global_slot,
        session.pending_action,
    )
    assert tuple(int(head[0]) for head in action) == (MOVE_STAY, 6, 1)


@pytest.mark.parametrize(
    ("head_name", "value"),
    (("move", -1), ("select_target", -1), ("use_ultimate", 2)),
)
def test_every_out_of_domain_head_canonicalizes_complete_tuple_in_diagnostics(
    head_name: str,
    value: int,
) -> None:
    session = _session("acceptance_lane_lab")
    action = make_neutral_joint_action()
    replacement = getattr(action, head_name).at[0].set(value)
    action = action._replace(**{head_name: replacement})
    after = submit_joint_action(
        session,
        action,
        submission_kind="interactive",
        report_actor_slots=(0,),
    )
    transition = after.last_transition
    assert transition is not None
    actor = transition.actor_transitions[0]
    assert not actor.submitted_tuple_in_domain
    assert tuple(int(head[0]) for head in transition.accepted_action) == (0, 0, 0)
    assert {item.component for item in transition.rejections} == {
        "complete_tuple_domain"
    }
    text = format_verbose_transition(transition)
    if head_name == "select_target":
        assert (
            "TARGET actor=g0 global=invalid relative=t-1 relation=n/a distance=n/a"
        ) in text
        assert "GEOMETRY actor=g0 target=invalid los=n/a" in text
