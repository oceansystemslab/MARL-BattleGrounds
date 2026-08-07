"""Edge-heavy tests for target facts, transition inference, logs, and history."""

from dataclasses import FrozenInstanceError, replace

import jax.numpy as jnp
import numpy as np
import pytest
from scripts.dev.visual_debugger.control import (
    arm_ultimate,
    build_interactive_joint_action,
    create_session,
    cycle_controlled_actor,
    make_neutral_joint_action,
    reset_session,
    select_clicked_target,
    select_controlled_actor,
    set_movement_scale,
    submit_interactive,
    submit_joint_action,
    submit_next_script_frame,
)
from scripts.dev.visual_debugger.diagnostics import (
    accepted_action_from_successor,
    derive_selected_target_facts,
    derive_visual_event_batch,
    extract_transition_view,
    format_concise_transition,
    format_reset,
    format_verbose_transition,
    latest_visual_event_batch,
)
from scripts.dev.visual_debugger.model import DebuggerSession, SelectedTargetFacts
from scripts.dev.visual_debugger.scenarios import get_scenario
from scripts.dev.visual_debugger.targeting import global_slot_to_target_action
from tests.visual_debugger_fixtures import (
    rejection_lane_scenario,
    submit_fixture_frame,
)

from marl_battlegrounds.core.geometry import has_clear_line_of_sight
from marl_battlegrounds.core.types import (
    MAX_AGENT_SLOTS,
    MOVE_EAST,
    MOVE_STAY,
    NUM_TARGET_ACTIONS,
)
from marl_battlegrounds.rendering.scene import (
    AcceptedActivationEventV1,
    ChargeDisplacementEventV1,
    NetHealthEventV1,
    StatusLifecycleEventV1,
    to_jsonable,
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


def _rejection_session() -> DebuggerSession:
    scenario = rejection_lane_scenario()
    return create_session(
        scenario,
        seed=0,
        controlled_global_slot=None,
        show_ranges=True,
        verbose_logging=False,
    )


def test_reset_diagnostic_formats_movement_scale_with_fixed_two_decimals() -> None:
    initial = _session("arena_5v5")
    tenth = set_movement_scale(initial, 0.1)
    hundredth = set_movement_scale(initial, 0.01)

    assert format_reset(initial).endswith("movement_scale=1.00")
    assert format_reset(tenth).endswith("movement_scale=0.10")
    assert format_reset(hundredth).endswith("movement_scale=0.01")


def _rejected_ultimate_with_movement() -> DebuggerSession:
    session = _session("arena_5v5", 2)
    target_action = global_slot_to_target_action(2, 7)
    neutral = make_neutral_joint_action()
    action = neutral._replace(
        move=neutral.move.at[2].set(MOVE_EAST),
        select_target=neutral.select_target.at[2].set(target_action),
        use_ultimate=neutral.use_ultimate.at[2].set(1),
    )
    return submit_joint_action(
        session,
        action,
        submission_kind="interactive",
        report_actor_slots=(2,),
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
    session = _rejection_session()
    profile = session.config.agent_profile
    cases = (
        (float(profile.observation_radii[0]), "inside_observation_radius"),
        (float(profile.basic_interaction_radii[0]), "inside_basic_radius"),
        (float(profile.ultimate_interaction_radii[0]), "inside_ultimate_radius"),
    )
    for distance, attribute in cases:
        positions = session.state.agent_positions.at[5].set(
            session.state.agent_positions[0] + jnp.asarray((distance, 0.0))
        )
        exact = replace(
            session,
            state=session.state._replace(agent_positions=positions),
        )
        assert getattr(_facts(exact, 5), attribute) is True


def test_nontargeted_mage_ultimate_range_is_not_applicable() -> None:
    session = _session("arena_5v5", 0)
    facts = _facts(session, 5)
    assert facts.inside_ultimate_radius is None


def test_selected_target_facts_do_not_derive_lanes_from_geometry() -> None:
    session = _rejection_session()
    facts = _facts(session, 5)
    assert facts.has_clear_line_of_sight
    assert facts.inside_observation_radius
    assert not facts.inside_basic_radius
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
    assert overridden.inside_observation_radius
    assert not overridden.inside_basic_radius


def test_selected_target_facts_reject_inactive_actor_and_target() -> None:
    session = _rejection_session()
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
    scenario = rejection_lane_scenario()
    session = select_clicked_target(_rejection_session(), 5)
    before = _facts(session, 5)
    assert before.relation == "enemy"
    assert before.target_action == 6
    assert before.center_distance == pytest.approx(4.0)

    switched = cycle_controlled_actor(session, 1)
    assert switched.pending_action.selected_global_target_slot is None
    assert switched.pending_actions[0].selected_global_target_slot == 5
    switched_facts = _facts(switched, 5)
    assert switched_facts.controlled_global_slot == 5
    assert switched_facts.relation == "self"
    assert switched_facts.target_action == 1
    assert switched_facts.center_distance == pytest.approx(0.0)
    assert switched_facts.observer_visible

    successor = submit_fixture_frame(session, scenario.frames[0])
    assert successor.pending_action.selected_global_target_slot == 5
    successor_facts = _facts(successor, 5)
    assert successor_facts.center_distance == pytest.approx(3.0)
    assert successor_facts.target_action == 6


def test_accepted_action_comes_from_successor_history() -> None:
    scenario = rejection_lane_scenario()
    session = _rejection_session()
    with pytest.raises(ValueError):
        accepted_action_from_successor(session.state)
    submitted = submit_fixture_frame(session, scenario.frames[0])
    accepted = accepted_action_from_successor(submitted.state)
    assert accepted.move is submitted.state.previous_timestep_move_actions
    assert accepted.select_target is (
        submitted.state.previous_timestep_select_target_actions
    )
    assert accepted.use_ultimate is (
        submitted.state.previous_timestep_use_ultimate_actions
    )


def test_diagnostics_report_accepted_move_and_rejected_combat() -> None:
    scenario = rejection_lane_scenario()
    session = submit_fixture_frame(_rejection_session(), scenario.frames[0])
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
    session = _rejection_session()
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


def test_health_outcome_preserves_exact_net_for_competing_health_intents() -> None:
    session = _session("basic_support")
    session = submit_next_script_frame(session)
    session = submit_next_script_frame(session)
    transition = session.last_transition
    assert transition is not None
    actor = next(
        value for value in transition.actor_transitions if value.actor_global_slot == 2
    )
    assert actor.net_health_delta == actor.health_after - actor.health_before == 2.0
    batch = derive_visual_event_batch(transition)
    health_events = tuple(
        event for event in batch.events if isinstance(event, NetHealthEventV1)
    )
    health = next(event for event in health_events if event.recipient_global_slot == 2)
    assert health.net_delta == 2.0
    assert health.outcome == "healing"
    assert not hasattr(health, "source_global_slot")


def test_non_health_activations_do_not_create_zero_health_delta_visuals() -> None:
    session = _session("ultimate_showcase")
    session = submit_next_script_frame(session)
    session = submit_next_script_frame(session)

    batch = latest_visual_event_batch(session)
    assert batch is not None
    health_visual_slots = {
        event.recipient_global_slot
        for event in batch.events
        if isinstance(event, NetHealthEventV1)
    }

    assert health_visual_slots == {2, 5, 6, 7}
    assert 5 in health_visual_slots  # Rogue Poison carries approved direct damage.
    assert 6 in health_visual_slots  # Hunter Trap carries approved direct damage.


def test_activation_events_preserve_multiplicity_successor_anchors_and_no_amount() -> (
    None
):
    session = submit_next_script_frame(_session("moving_basic_crossfire"))
    transition = session.last_transition
    assert transition is not None
    duplicated = replace(
        transition,
        accepted_activations=(
            transition.accepted_activations[0],
            transition.accepted_activations[0],
        ),
    )
    batch = derive_visual_event_batch(duplicated)
    activations = tuple(
        event for event in batch.events if isinstance(event, AcceptedActivationEventV1)
    )
    assert len(activations) == 2
    assert activations[0].token_id == activations[1].token_id
    assert activations[0].event_id != activations[1].event_id
    source = next(
        actor
        for actor in transition.actor_transitions
        if actor.actor_global_slot == activations[0].source_global_slot
    )
    target = next(
        actor
        for actor in transition.actor_transitions
        if actor.actor_global_slot == activations[0].target_global_slot
    )
    assert source.position_before != source.position_after
    assert target.position_before != target.position_after
    assert activations[0].source_anchor == source.position_after
    assert activations[0].target_anchor == target.position_after
    payload = to_jsonable(activations[0])
    assert "amount" not in payload  # type: ignore[operator]
    assert "damage" not in payload  # type: ignore[operator]
    assert "healing" not in payload  # type: ignore[operator]


def test_completed_anchors_use_successor_with_charge_prestate_exception() -> None:
    sessions: list[DebuggerSession] = []
    moving_basics = submit_next_script_frame(_session("moving_basic_crossfire"))
    sessions.append(moving_basics)

    mirrored = _session("mirrored_ultimates")
    for _ in get_scenario("mirrored_ultimates").frames:
        mirrored = submit_next_script_frame(mirrored)
        sessions.append(mirrored)

    observed_kinds: set[str] = set()
    for session in sessions:
        transition = session.last_transition
        assert transition is not None
        batch = latest_visual_event_batch(session)
        assert batch is not None
        actor_by_slot = {
            actor.actor_global_slot: actor for actor in transition.actor_transitions
        }
        activation_by_source = {
            event.source_global_slot: event
            for event in batch.events
            if isinstance(event, AcceptedActivationEventV1)
        }
        for activation in transition.accepted_activations:
            observed_kinds.add(activation.kind)
            event = activation_by_source[activation.source_global_slot]
            source = actor_by_slot[activation.source_global_slot]
            target = (
                None
                if activation.target_global_slot is None
                else actor_by_slot[activation.target_global_slot]
            )
            if activation.kind == "warrior_charge":
                assert target is not None
                assert source.position_before != source.position_after
                assert event.source_anchor == source.position_before
                assert event.target_anchor == target.position_before
            else:
                assert event.source_anchor == source.position_after
                assert event.target_anchor == (
                    None if target is None else target.position_after
                )

    assert observed_kinds == {
        "basic_damage",
        "basic_heal",
        "mage_burst",
        "warrior_charge",
        "hunter_trap",
        "rogue_poison",
        "holy_word",
    }


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
    session = _session("ultimate_showcase")
    session = submit_next_script_frame(session)
    session = submit_next_script_frame(session)
    assert int(session.state.stun_durations[6, 1]) == 4
    # Three canonical neutral steps expose durations 3, 2, 1.
    for _ in range(3):
        session = submit_joint_action(
            session,
            make_neutral_joint_action(),
            submission_kind="interactive",
            report_actor_slots=(0,),
        )
    assert int(session.state.stun_durations[6, 1]) == 1
    session = select_controlled_actor(session, 2)
    session = select_clicked_target(session, 6)
    damaged = submit_interactive(session)
    transition = damaged.last_transition
    assert transition is not None
    trap = next(
        status
        for status in transition.status_transitions
        if status.global_slot == 6 and status.status_kind == "stun_hunter_trap"
    )
    assert trap.duration_before == 1
    assert trap.duration_after == 0
    assert trap.change == "cleared_unclassified"
    batch = derive_visual_event_batch(transition)
    lifecycle = next(
        event
        for event in batch.events
        if isinstance(event, StatusLifecycleEventV1)
        and event.recipient_global_slot == 6
        and event.token_id == "stun_hunter_trap"
    )
    assert lifecycle.change == "cleared_unclassified"


def test_charge_trail_distinguishes_stay_and_movement_and_uses_realized_endpoints() -> (
    None
):
    showcase = _session("ultimate_showcase")
    showcase = submit_next_script_frame(showcase)
    showcase = submit_next_script_frame(showcase)
    batch = latest_visual_event_batch(showcase)
    assert batch is not None
    charge = next(
        event for event in batch.events if isinstance(event, ChargeDisplacementEventV1)
    )
    assert charge.path_kind == "charge_only"
    assert charge.start == (5.0, 5.0)
    np.testing.assert_allclose(charge.end, (9.0715, 3.3714), atol=1e-4)

    stack = submit_next_script_frame(_session("status_stack"))
    batch = latest_visual_event_batch(stack)
    assert batch is not None
    combined = next(
        event for event in batch.events if isinstance(event, ChargeDisplacementEventV1)
    )
    assert combined.path_kind == "combined_charge_and_movement"
    assert combined.start == (3.0, 6.0)
    assert combined.end == (7.0, 7.0)


def test_latest_transition_events_survive_ui_only_edits_and_replace_on_submit() -> None:
    session = _session("ultimate_showcase")
    session = submit_next_script_frame(session)
    session = submit_next_script_frame(session)
    initial = latest_visual_event_batch(session)
    assert initial is not None
    initial_ids = tuple(event.event_id for event in initial.events)
    assert any(isinstance(event, ChargeDisplacementEventV1) for event in initial.events)

    inspected = select_controlled_actor(session, 3)
    inspected = select_clicked_target(inspected, 5)
    same = latest_visual_event_batch(inspected)
    assert same is not None
    assert tuple(event.event_id for event in same.events) == initial_ids

    advanced = submit_next_script_frame(inspected)
    replacement = latest_visual_event_batch(advanced)
    assert replacement is not None
    assert replacement.transition_id == initial.transition_id + 1
    assert not any(
        isinstance(event, ChargeDisplacementEventV1) for event in replacement.events
    )

    reset = reset_session(advanced)
    assert latest_visual_event_batch(reset) is None
    replayed = submit_next_script_frame(reset)
    replayed = submit_next_script_frame(replayed)
    replayed_batch = latest_visual_event_batch(replayed)
    assert replayed_batch is not None
    assert tuple(event.event_id for event in replayed_batch.events) != initial_ids


def test_concise_log_schema_covers_accepted_basic_and_rejected_ultimate() -> None:
    basic = submit_next_script_frame(_session("basic_support"))
    transition = basic.last_transition
    assert transition is not None
    text = format_concise_transition(transition)
    play, technical = text.split("\n\nTECHNICAL DIAGNOSTICS\n")
    assert play.startswith("PLAY-BY-PLAY\n")
    assert "TEAM A MAGE (id_0) attempted BASIC on TEAM B MAGE (id_5)." in play
    assert "TEAM B MAGE (id_5) lost 14.95 HP." in play
    assert "Accepted contributors included TEAM A MAGE (id_0) BASIC." in play
    assert "Active public multipliers included" in play
    assert " g0" not in play
    assert (
        "Transition   scenario=basic_support step=0 -> 1 terminated=0 truncated=0"
        in technical
    )
    assert "Actor id_0 [g0] submitted move=Stay[0] target=t6 ultimate=0" in technical
    assert "Health id_5 80.00 -> 65.05 net=-14.95" in technical
    assert "Activation basic_damage g0->g5" in technical

    rejected = _rejected_ultimate_with_movement()
    transition = rejected.last_transition
    assert transition is not None
    text = format_concise_transition(transition)
    play, technical = text.split("\n\nTECHNICAL DIAGNOSTICS\n")
    assert "TEAM A HUNTER (id_2) attempted TRAP on TEAM B HUNTER (id_7)." in play
    assert "TRAP was rejected." in play
    assert "East movement was accepted." in play
    assert "Actor id_2 [g2] submitted move=East[3] target=t8 ultimate=1" in technical
    assert "accepted  move=East[3] target=t0 ultimate=0" in technical
    assert "mask move=1 lane0=0 lane1=0 pair=0 domain=1" in technical
    assert "Rejection combat actor=g2 mask=0" in technical


def test_concise_logs_cover_all_required_effect_and_lifecycle_examples() -> None:
    showcase = _session("ultimate_showcase")
    showcase = submit_next_script_frame(showcase)
    showcase = submit_next_script_frame(showcase)
    transition = showcase.last_transition
    assert transition is not None
    text = format_concise_transition(transition)
    for expected in (
        "TEAM A MAGE (id_0) attempted BURST as a self activation.",
        "TEAM A WARRIOR (id_1) attempted CHARGE on TEAM B HUNTER (id_7).",
        "TEAM A HUNTER (id_2) attempted TRAP on TEAM B WARRIOR (id_6).",
        "TEAM A ROGUE (id_3) attempted POISON on TEAM B MAGE (id_5).",
        "TEAM A PRIEST (id_4) attempted HOLY WORD on TEAM A HUNTER (id_2).",
        "TEAM B MAGE (id_5) lost 36.00 HP.",
        "Successor state: TEAM A MAGE (id_0) Ultimate cooldown is 30.",
        "Successor state: TEAM B WARRIOR (id_6) has TRAP 4 (applied).",
        "Activation rogue_poison g3->g5",
    ):
        assert expected in text

    showcase = submit_next_script_frame(showcase)
    transition = showcase.last_transition
    assert transition is not None
    text = format_concise_transition(transition)
    assert (
        "Successor state: TEAM B WARRIOR (id_6) has no TRAP "
        "(broken after an accepted damage activation)."
    ) in text
    assert "Cooldown id_0 30->29" in text
    assert "Successor state: TEAM B MAGE (id_5) has no POISON-STUN (expired)." in text

    support = _session("basic_support")
    support = submit_next_script_frame(support)
    support = submit_next_script_frame(support)
    transition = support.last_transition
    assert transition is not None
    text = format_concise_transition(transition)
    assert "TEAM A PRIEST (id_2) gained 2.00 HP." in text
    assert (
        "Accepted contributors included TEAM A PRIEST (id_2) BASIC "
        "and TEAM B HUNTER (id_7) BASIC."
    ) in text
    assert (
        "The public transition does not expose the gross damage/healing split." in text
    )
    assert "Health id_2 94.00 -> 96.00 net=+2.00" in text


def test_verbose_log_adds_geometry_mask_aura_speed_and_episode_fields() -> None:
    session = _rejected_ultimate_with_movement()
    transition = session.last_transition
    assert transition is not None
    text = format_verbose_transition(transition)

    assert "Target g2->g7 t8 relation=enemy distance=" in text
    assert "Geometry g2->g7 los=" in text
    assert "visible=" in text
    assert "observation=" in text
    assert "basic=" in text
    assert "ultimate=" in text
    assert "mask move=1 lane0=0 lane1=0 pair=0 domain=1" in text
    assert "Position g2" in text
    assert "Aura g2 mage=" in text
    assert "Speed g2" in text
    assert "Reward g2 +0.00" in text
    assert "Array(" not in text
    assert "[[" not in text


def test_geometry_facts_are_not_logged_as_rejection_causes() -> None:
    session = _rejected_ultimate_with_movement()
    transition = session.last_transition
    assert transition is not None
    text = format_verbose_transition(transition)
    assert "los=1" in text
    assert "visible=0" in text
    assert "Rejection combat actor=g2 mask=0" in text
    for forbidden in (
        "rejected because",
        "cause=los",
        "cause=range",
        "cause=visibility",
    ):
        assert forbidden not in text.lower()


def test_extract_transition_retains_every_public_before_after_artifact() -> None:
    scenario = rejection_lane_scenario()
    before = _rejection_session()
    after = submit_fixture_frame(before, scenario.frames[0])
    transition = after.last_transition
    assert transition is not None
    assert transition.config is before.config
    reconstructed = extract_transition_view(
        scenario_name=transition.scenario_name,
        config=transition.config,
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
    assert reconstructed.config is transition.config
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
    session = select_clicked_target(_session("arena_5v5", 2), 7)
    session = arm_ultimate(session)
    action = build_interactive_joint_action(
        session.config,
        session.pending_actions,
        actor_global_slots=(session.controlled_global_slot,),
    )
    assert tuple(int(head[2]) for head in action) == (MOVE_STAY, 8, 1)


@pytest.mark.parametrize(
    ("head_name", "value"),
    (("move", -1), ("select_target", -1), ("use_ultimate", 2)),
)
def test_every_out_of_domain_head_canonicalizes_complete_tuple_in_diagnostics(
    head_name: str,
    value: int,
) -> None:
    session = _rejection_session()
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
        assert "Target g0 invalid t-1; relation=n/a distance=n/a" in text
        assert "Geometry g0->invalid los=n/a" in text
