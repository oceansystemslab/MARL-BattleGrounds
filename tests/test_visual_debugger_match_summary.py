"""Match UI packages captured facts at the installed frame, outside spatial POV."""

from __future__ import annotations

import pytest
from scripts.dev.visual_debugger.match_summary import build_match_summary_v1
from tests.evaluation_fixtures import (
    captured_evaluation_trajectory,
    captured_team_deathmatch_threshold_trajectory,
)

from marl_battlegrounds.evaluation.models import AssignedPolicySlotV1


def test_match_summary_uses_captured_task_scores_policy_identity_and_result() -> None:
    trajectory = captured_team_deathmatch_threshold_trajectory()
    context = trajectory.context
    before = build_match_summary_v1(context, trajectory.frames[0])
    after = build_match_summary_v1(
        context, trajectory.frames[1], trajectory.transitions[0].events
    )
    assert before.task_mode == 1
    assert (
        before.score_threshold
        == context.resolved_env_config.team_deathmatch_score_threshold
    )
    assert before.scores == trajectory.frames[0].snapshot.team_deathmatch_scores
    assert before.outcome == "in_progress"
    assert after.scores == trajectory.frames[1].snapshot.team_deathmatch_scores
    assert after.outcome == next(
        event.outcome
        for event in trajectory.transitions[0].events
        if event.event_type == "team_deathmatch_completed"
    )
    for team in after.teams:
        assignments = tuple(
            assignment
            for roster, assignment in zip(
                context.roster, context.policy_assignments, strict=True
            )
            if roster.configured_team_id == team.team_id
            and isinstance(assignment, AssignedPolicySlotV1)
        )
        assert team.policy_ids == tuple(
            dict.fromkeys(row.policy_id for row in assignments)
        )
        assert team.checkpoint_digests == tuple(
            dict.fromkeys(
                row.checkpoint_digest for row in assignments if row.checkpoint_digest
            )
        )
    with pytest.raises(ValueError, match="incoming transition"):
        build_match_summary_v1(
            context, trajectory.frames[0], trajectory.transitions[0].events
        )


def test_match_summary_keeps_legacy_combat_diagnostic_truthful() -> None:
    trajectory = captured_evaluation_trajectory(transition_count=0)
    summary = build_match_summary_v1(trajectory.context, trajectory.frames[0])
    assert summary.task_mode == 0
    assert summary.score_threshold == 0
    assert summary.scores == (0, 0)
    assert summary.outcome == "not_applicable"
    unrelated = trajectory.context.model_copy(
        update={
            "identity": trajectory.context.identity.model_copy(
                update={"episode_id": "other"}
            )
        }
    )
    with pytest.raises(ValueError, match="episode context"):
        build_match_summary_v1(unrelated, trajectory.frames[0])
