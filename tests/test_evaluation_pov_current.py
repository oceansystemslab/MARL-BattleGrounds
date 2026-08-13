"""Live recipient-slice proofs over the accepted CP2/CP3 coherent view."""

from __future__ import annotations

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
    ActorPovCurrentSliceV1,
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
