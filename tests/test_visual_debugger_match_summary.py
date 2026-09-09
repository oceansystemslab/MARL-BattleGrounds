"""Match UI packages captured facts at the installed frame, outside spatial POV."""

from __future__ import annotations

from dataclasses import replace

import pytest
from scripts.dev.visual_debugger.control import create_session
from scripts.dev.visual_debugger.match_summary import build_match_summary_v1
from scripts.dev.visual_debugger.model import DebuggerScenarioProvenance
from scripts.dev.visual_debugger.scenarios import get_scenario
from tests.evaluation_fixtures import (
    captured_evaluation_trajectory,
    captured_team_deathmatch_threshold_trajectory,
)
from tests.visual_debugger_fixtures import debugger_test_launch_specification

from marl_battlegrounds.evaluation.models import (
    AggregationKeyV1,
    AssignedPolicySlotV1,
    ContentAddressedIdentityV1,
)
from marl_battlegrounds.evaluation.replay_v2 import context_v2


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
    assert tuple(death.public_agent_id for death in after.deaths) == tuple(
        context.roster[event.recipient_global_slot].public_agent_id
        for event in trajectory.transitions[0].events
        if event.event_type == "agent_died"
    )
    assert before.deaths == ()
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


@pytest.mark.parametrize("scenario_id", range(1, 9))
def test_scenario_pressure_display_name_uses_recorded_controller(
    scenario_id: int,
) -> None:
    trajectory = captured_evaluation_trajectory(transition_count=0)
    variant = "beta" if scenario_id in (3, 5, 8) else "alpha"
    policy_id = (
        "scenario-5-pressure-controller"
        if variant == "beta"
        else "reactive-team-deathmatch-controller"
    )
    context = trajectory.context.model_copy(
        update={
            "identity": trajectory.context.identity.model_copy(
                update={
                    "scenario": ContentAddressedIdentityV1(
                        identifier=f"tdm-scenario-{scenario_id}",
                        version=1,
                        canonical_digest="a" * 64,
                    )
                }
            ),
            "policy_assignments": tuple(
                row.model_copy(
                    update={"policy_id": policy_id, "policy_kind": "scenario_pressure"}
                )
                if isinstance(row, AssignedPolicySlotV1) and row.global_slot >= 5
                else row
                for row in trajectory.context.policy_assignments
            ),
        }
    )
    summary = build_match_summary_v1(context, trajectory.frames[0])
    assert (
        summary.teams[1].display_name
        == f"tdm-scenario-{scenario_id}-controller-{variant}"
    )
    assert summary.teams[1].policy_ids == (policy_id,)

    # Exercise the real producer too: its policy IDs are slot-specific; the
    # controller descriptor belongs to algorithm_id, and authored scenario
    # identity intentionally remains independent from its readable name.
    source = get_scenario("arena_5v5")
    config, state = source.build_scenario()
    config = config._replace(task_mode=1, team_deathmatch_score_threshold=20)
    scenario = replace(
        source,
        name=f"scenario_{scenario_id}",
        build_scenario=lambda: (config, state),
        provenance=DebuggerScenarioProvenance(
            source_kind="saved_draft",
            source_identity=f"saved_draft:scenario:scenario_{scenario_id}:r1",
            scenario_semantic_digest="b" * 64,
            map_semantic_digest="c" * 64,
            resolved_configuration_digest="d" * 64,
            resolved_initial_state_digest="e" * 64,
        ),
    )
    session = create_session(
        scenario,
        seed=7,
        evaluation_launch_specification=debugger_test_launch_specification(7),
        controlled_global_slot=None,
        show_ranges=False,
        verbose_logging=False,
        team_a_controller="manual",
        team_b_controller="scenario_5" if variant == "beta" else "reactive_tdm",
        execution_information_mode="shared_obs",
    )
    assert session.evaluation_context.identity.scenario is not None
    assert session.evaluation_context.identity.scenario.identifier == (
        "authored-team-deathmatch-scenario"
    )
    for produced in (
        session.evaluation_context,
        context_v2(session.evaluation_context, scenario_name=scenario.name),
    ):
        actual = build_match_summary_v1(produced, session.current_evaluation_frame)
        assert actual.teams[1].display_name == summary.teams[1].display_name
        assert actual.teams[1].policy_ids == tuple(
            f"debugger-action-source:{session.team_b_controller}:slot:{slot}"
            for slot in range(5, 10)
        )
        assert all(
            row.algorithm_id == policy_id
            for row in produced.policy_assignments[5:]
            if row.assignment_status == "assigned"
        )

    current = context_v2(session.evaluation_context, scenario_name=scenario.name)
    callable_name = f"tdm-{variant}"
    current = current.model_copy(
        update={
            "policy_assignments": tuple(
                row.model_copy(
                    update={
                        "policy_id": callable_name,
                        "algorithm_id": None,
                        "policy_kind": "callable",
                    }
                )
                if row.assignment_status == "assigned" and row.global_slot >= 5
                else row
                for row in current.policy_assignments
            ),
        }
    )
    # A familiar policy name alone is not proof of its controller rules.
    assert (
        build_match_summary_v1(current, session.current_evaluation_frame)
        .teams[1]
        .display_name
        == callable_name
    )
    descriptor = ContentAddressedIdentityV1(
        identifier=policy_id, version=1, canonical_digest="f" * 64
    )
    current = current.model_copy(
        update={
            "aggregation_keys": (
                *current.aggregation_keys,
                AggregationKeyV1(
                    name="team_b_controller_identity",
                    value=descriptor.model_dump_json(),
                ),
            )
        }
    )
    observed = build_match_summary_v1(current, session.current_evaluation_frame)
    assert observed.teams[1].display_name == summary.teams[1].display_name
    assert observed.teams[1].policy_ids == (callable_name,)
