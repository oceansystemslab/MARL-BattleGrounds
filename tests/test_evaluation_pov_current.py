"""Live recipient-slice proofs over the accepted CP2/CP3 coherent view."""

from __future__ import annotations

from typing import cast

import pytest
from pydantic import ValidationError
from tests.evaluation_fixtures import (
    CapturedEvaluationTrajectory,
    captured_evaluation_trajectory,
    neutral_action,
)

from marl_battlegrounds.core.types import Action
from marl_battlegrounds.evaluation.metrics import EvaluationTransitionViewV1
from marl_battlegrounds.evaluation.pov import (
    ActorPovAdjacentTransitionSliceV1,
    ActorPovCurrentSliceV1,
    build_actor_pov_adjacent_transition_slice_v1,
    build_actor_pov_current_slice_v1,
    slice_actor_pov_current_frame_v1,
    slice_actor_pov_current_transition_v1,
)


@pytest.fixture(scope="module")
def trajectory() -> CapturedEvaluationTrajectory:
    return captured_evaluation_trajectory(transition_count=1)


def _view(trajectory: CapturedEvaluationTrajectory) -> EvaluationTransitionViewV1:
    return EvaluationTransitionViewV1(
        context=trajectory.context,
        start_frame=trajectory.frames[0],
        transition=trajectory.transitions[0],
        successor_frame=trajectory.frames[1],
    )


def _view_at(
    trajectory: CapturedEvaluationTrajectory,
    transition_index: int,
) -> EvaluationTransitionViewV1:
    return EvaluationTransitionViewV1(
        context=trajectory.context,
        start_frame=trajectory.frames[transition_index],
        transition=trajectory.transitions[transition_index],
        successor_frame=trajectory.frames[transition_index + 1],
    )


def test_frame_zero_is_an_exact_no_shared_obs_decision_slice(
    trajectory: CapturedEvaluationTrajectory,
) -> None:
    current = build_actor_pov_current_slice_v1(
        trajectory.context,
        trajectory.frames[0],
        global_slot=0,
    )

    assert current.frame.frame_index == 0
    assert current.incoming_transition is None
    assert current.observation_materialization == "exact_no_shared_obs_actor_input"
    assert current.public_agent_id == trajectory.context.roster[0].public_agent_id
    assert (
        current.frame.self_features
        == (trajectory.frames[0].base_observation.self_features[0])
    )
    # The current frame mask governs the action chosen at this decision epoch.
    assert (
        current.frame.action_mask.move == trajectory.frames[0].action_mask.move_mask[0]
    )
    assert (
        current.frame.action_mask.select_target_use_ultimate_joint
        == (trajectory.frames[0].action_mask.select_target_use_ultimate_joint_mask[0])
    )
    assert (
        slice_actor_pov_current_frame_v1(
            trajectory.context,
            trajectory.frames[0],
            global_slot=0,
        )
        == current.frame
    )


def test_successor_slice_pairs_action_t_with_next_decision_mask(
    trajectory: CapturedEvaluationTrajectory,
) -> None:
    view = _view(trajectory)
    current = build_actor_pov_current_slice_v1(
        trajectory.context,
        trajectory.frames[1],
        global_slot=0,
        incoming_transition_view=view,
    )
    incoming = current.incoming_transition
    assert incoming is not None
    acceptance = trajectory.transitions[0].facts.action_acceptance_facts

    assert current.frame.frame_index == 1
    assert incoming.transition_index == 0
    assert incoming.submitted_action.move == acceptance.submitted_joint_action.move[0]
    assert incoming.accepted_action.move == acceptance.accepted_joint_action.move[0]
    assert (
        incoming.canonical_reward
        == trajectory.transitions[0].canonical_reward_by_agent[0]
    )
    # The successor mask is action-1 authority; it does not explain action 0.
    assert (
        current.frame.action_mask.move == trajectory.frames[1].action_mask.move_mask[0]
    )
    assert slice_actor_pov_current_transition_v1(view, global_slot=0) == incoming
    assert tuple(cue.ordinal for cue in incoming.cues) == tuple(
        range(len(incoming.cues))
    )


def test_team_b_current_slice_keeps_actor_relative_own_team_axis(
    trajectory: CapturedEvaluationTrajectory,
) -> None:
    current = build_actor_pov_current_slice_v1(
        trajectory.context,
        trajectory.frames[1],
        global_slot=5,
        incoming_transition_view=_view(trajectory),
    )
    lifecycle = trajectory.frames[1].base_observation.spawn_lifecycle

    assert current.configured_team_id == 2
    assert current.selected_team_local_slot == 0
    assert current.axis_mapping.spawn_lifecycle_team_axis_name_by_id == (
        "Own Team",
        "Opponent Team",
    )
    assert (
        current.frame.spawn_lifecycle.spawn_shield_actual_durations_by_team[0][0]
        == lifecycle.spawn_shield_actual_durations_by_agent_by_team[5][0][0]
    )


def test_hidden_other_actor_event_does_not_change_current_recipient_slice() -> None:
    neutral = neutral_action()
    hidden_rejection = Action(
        move=neutral.move.at[3].set(-7),
        select_target=neutral.select_target,
        use_ultimate=neutral.use_ultimate,
    )
    first = captured_evaluation_trajectory(
        transition_count=1,
        actions=(neutral,),
    )
    second = captured_evaluation_trajectory(
        transition_count=1,
        actions=(hidden_rejection,),
    )
    assert first.frames == second.frames
    assert first.transitions[0].events != second.transitions[0].events

    first_slice = build_actor_pov_current_slice_v1(
        first.context,
        first.frames[1],
        global_slot=0,
        incoming_transition_view=_view(first),
    )
    second_slice = build_actor_pov_current_slice_v1(
        second.context,
        second.frames[1],
        global_slot=0,
        incoming_transition_view=_view(second),
    )
    assert first_slice == second_slice


def test_current_slice_fails_closed_for_shared_obs_and_inactive_actor() -> None:
    shared = captured_evaluation_trajectory(
        transition_count=0,
        execution_information_mode="shared_obs",
    )
    with pytest.raises(ValueError, match="unavailable for shared_obs"):
        build_actor_pov_current_slice_v1(
            shared.context,
            shared.frames[0],
            global_slot=0,
        )

    no_shared = captured_evaluation_trajectory(transition_count=0)
    with pytest.raises(ValueError, match="configured-active"):
        build_actor_pov_current_slice_v1(
            no_shared.context,
            no_shared.frames[0],
            global_slot=3,
        )


@pytest.mark.parametrize("slot", [-1, 10, True])
def test_current_slice_rejects_invalid_actor_slot(slot: object) -> None:
    trajectory = captured_evaluation_trajectory(transition_count=0)
    with pytest.raises(ValueError, match="exact bounded integer"):
        build_actor_pov_current_slice_v1(
            trajectory.context,
            trajectory.frames[0],
            global_slot=slot,  # type: ignore[arg-type]
        )


def test_current_slice_requires_initial_or_exact_incoming_view(
    trajectory: CapturedEvaluationTrajectory,
) -> None:
    with pytest.raises(ValueError, match="initial frame index"):
        build_actor_pov_current_slice_v1(
            trajectory.context,
            trajectory.frames[1],
            global_slot=0,
        )
    with pytest.raises(ValueError, match="must enter the selected frame"):
        build_actor_pov_current_slice_v1(
            trajectory.context,
            trajectory.frames[0],
            global_slot=0,
            incoming_transition_view=_view(trajectory),
        )


def test_current_slice_root_is_strict_and_versioned(
    trajectory: CapturedEvaluationTrajectory,
) -> None:
    current = build_actor_pov_current_slice_v1(
        trajectory.context,
        trajectory.frames[0],
        global_slot=0,
    )
    assert (
        ActorPovCurrentSliceV1.model_validate_json(current.model_dump_json()) == current
    )

    future = current.model_dump(mode="python")
    future["schema_version"] = 2
    with pytest.raises(ValidationError):
        ActorPovCurrentSliceV1.model_validate(future)

    listed = current.model_dump(mode="python")
    listed["axis_mapping"] = {
        **current.axis_mapping.model_dump(mode="python"),
        "movement_action_name_by_id": list(
            current.axis_mapping.movement_action_name_by_id
        ),
    }
    with pytest.raises(ValidationError):
        ActorPovCurrentSliceV1.model_validate(listed)


def test_adjacent_slice_round_trips_one_nonzero_coherent_transition() -> None:
    trajectory = captured_evaluation_trajectory(transition_count=2)
    view = _view_at(trajectory, 1)
    source_before = (
        view.context.model_dump_json(),
        view.start_frame.model_dump_json(),
        view.transition.model_dump_json(),
        view.successor_frame.model_dump_json(),
    )
    carrier = build_actor_pov_adjacent_transition_slice_v1(view, global_slot=0)

    assert carrier.start_frame.frame_index == 1
    assert carrier.transition.transition_index == 1
    assert carrier.successor_frame.frame_index == 2
    assert carrier.successor_frame.simulator_step_count == (
        carrier.start_frame.simulator_step_count + 1
    )
    assert (
        carrier.start_frame.self_features
        == (trajectory.frames[1].base_observation.self_features[0])
    )
    assert (
        carrier.successor_frame.self_features
        == (trajectory.frames[2].base_observation.self_features[0])
    )
    assert (
        ActorPovAdjacentTransitionSliceV1.model_validate_json(carrier.model_dump_json())
        == carrier
    )
    assert set(type(carrier).model_fields).isdisjoint(
        {"events", "status_source_evidence", "source_transition"}
    )
    assert source_before == (
        view.context.model_dump_json(),
        view.start_frame.model_dump_json(),
        view.transition.model_dump_json(),
        view.successor_frame.model_dump_json(),
    )


def test_adjacent_factory_requires_exact_no_shared_coherent_view() -> None:
    trajectory = captured_evaluation_trajectory(transition_count=1)
    view = _view(trajectory)
    with pytest.raises(TypeError, match="exact EvaluationTransitionViewV1"):
        build_actor_pov_adjacent_transition_slice_v1(
            cast(EvaluationTransitionViewV1, object()),
            global_slot=0,
        )
    for invalid_slot in (-1, 3, 10, True):
        with pytest.raises(ValueError):
            build_actor_pov_adjacent_transition_slice_v1(
                view,
                global_slot=invalid_slot,  # type: ignore[arg-type]
            )

    shared = captured_evaluation_trajectory(
        transition_count=1,
        execution_information_mode="shared_obs",
    )
    with pytest.raises(ValueError, match="unavailable for shared_obs"):
        build_actor_pov_adjacent_transition_slice_v1(
            _view(shared),
            global_slot=0,
        )

    other = captured_evaluation_trajectory(
        transition_count=1,
        episode_id="other-episode",
    )
    mismatched = object.__new__(EvaluationTransitionViewV1)
    object.__setattr__(mismatched, "context", other.context)
    object.__setattr__(mismatched, "start_frame", view.start_frame)
    object.__setattr__(mismatched, "transition", view.transition)
    object.__setattr__(mismatched, "successor_frame", view.successor_frame)
    with pytest.raises(ValueError):
        build_actor_pov_adjacent_transition_slice_v1(
            mismatched,
            global_slot=0,
        )


def test_adjacent_slice_rejects_identity_epoch_tick_axis_and_topology_poison() -> None:
    trajectory = captured_evaluation_trajectory(transition_count=2)
    carrier = build_actor_pov_adjacent_transition_slice_v1(
        _view_at(trajectory, 1),
        global_slot=0,
    )

    poisons: list[dict[str, object]] = []

    recipient = carrier.model_dump(mode="python")
    recipient["public_agent_id"] = "different-recipient"
    poisons.append(recipient)

    local_slot = carrier.model_dump(mode="python")
    local_slot["selected_team_local_slot"] = 1
    poisons.append(local_slot)

    wrong_index = carrier.model_dump(mode="python")
    transition = dict(wrong_index["transition"])  # type: ignore[arg-type]
    transition["transition_index"] = 0
    wrong_index["transition"] = transition
    poisons.append(wrong_index)

    wrong_id = carrier.model_dump(mode="python")
    start = dict(wrong_id["start_frame"])  # type: ignore[arg-type]
    start["pov_frame_id"] = "episode-001:actor-pov:agent-slot-0:frame:0"
    wrong_id["start_frame"] = start
    poisons.append(wrong_id)

    wrong_tick = carrier.model_dump(mode="python")
    successor = dict(wrong_tick["successor_frame"])  # type: ignore[arg-type]
    successor["simulator_step_count"] = carrier.start_frame.simulator_step_count + 2
    wrong_tick["successor_frame"] = successor
    poisons.append(wrong_tick)

    wrong_axis = carrier.model_dump(mode="python")
    axis = dict(wrong_axis["axis_mapping"])  # type: ignore[arg-type]
    allies = list(axis["ally_observation_row_public_agent_id_by_id"])  # type: ignore[arg-type]
    allies[0], allies[1] = allies[1], allies[0]
    axis["ally_observation_row_public_agent_id_by_id"] = tuple(allies)
    axis["target_action_recipient_public_agent_id_by_id"] = (
        None,
        *allies,
        *axis["enemy_observation_row_public_agent_id_by_id"],  # type: ignore[misc]
    )
    wrong_axis["axis_mapping"] = axis
    poisons.append(wrong_axis)

    wrong_lifecycle = carrier.model_dump(mode="python")
    successor = dict(wrong_lifecycle["successor_frame"])  # type: ignore[arg-type]
    lifecycle = dict(successor["spawn_lifecycle"])  # type: ignore[arg-type]
    alive = [list(row) for row in lifecycle["alive_mask_by_team"]]  # type: ignore[arg-type]
    alive[0][0] = not alive[0][0]
    lifecycle["alive_mask_by_team"] = tuple(tuple(row) for row in alive)
    successor["spawn_lifecycle"] = lifecycle
    wrong_lifecycle["successor_frame"] = successor
    poisons.append(wrong_lifecycle)

    wrong_visible_self = carrier.model_dump(mode="python")
    start = dict(wrong_visible_self["start_frame"])  # type: ignore[arg-type]
    ally_rows = list(start["ally_unit_features"])  # type: ignore[arg-type]
    visible_self = list(ally_rows[0])  # type: ignore[arg-type]
    visible_self[0] = visible_self[0] + 0.25
    ally_rows[0] = tuple(visible_self)
    start["ally_unit_features"] = tuple(ally_rows)
    wrong_visible_self["start_frame"] = start
    poisons.append(wrong_visible_self)

    wrong_cues = carrier.model_dump(mode="python")
    transition = dict(wrong_cues["transition"])  # type: ignore[arg-type]
    cues = list(transition["cues"])  # type: ignore[arg-type]
    outcome = dict(cues[0])  # type: ignore[arg-type]
    outcome["outcome"] = "rejected" if outcome["outcome"] == "accepted" else "accepted"
    cues[0] = outcome
    transition["cues"] = tuple(cues)
    wrong_cues["transition"] = transition
    poisons.append(wrong_cues)

    for poisoned in poisons:
        with pytest.raises(ValidationError):
            ActorPovAdjacentTransitionSliceV1.model_validate(poisoned)


def test_adjacent_rendering_seam_revalidates_model_constructed_roots() -> None:
    trajectory = captured_evaluation_trajectory(transition_count=1)
    carrier = build_actor_pov_adjacent_transition_slice_v1(
        _view(trajectory),
        global_slot=0,
    )
    forged = carrier.model_copy(
        update={"selected_team_local_slot": 1},
    )
    assert forged.selected_team_local_slot == 1
    with pytest.raises(ValidationError):
        ActorPovAdjacentTransitionSliceV1.model_validate(
            forged.model_dump(mode="python")
        )
