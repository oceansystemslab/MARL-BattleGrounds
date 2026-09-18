"""Compare a scenario turn with the saved result of a reviewed solution.

Tests use the same comparison helpers to check recorded state and action facts.
The saved result is a reference for its exact scenario revision.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np

from marl_battlegrounds.core.types import Action, DoneFlags, EnvState, Info, Reward


def assert_witness_tick(
    expected: Mapping[str, Any],
    *,
    state: EnvState,
    action: Action,
    reward: Reward,
    done: DoneFlags,
    info: Info,
    case_name: str,
) -> None:
    label = f"{case_name}, turn {expected['tick']}"
    facts = info.transition_facts
    combat = facts.combat_transition_facts
    physical = facts.physical_facts
    for field, actual in (
        ("positions", state.agent_positions),
        ("health", state.current_health),
        ("damage", combat.total_effective_damage_by_recipient),
        (
            "movement_displacement",
            physical.ordinary_movement_phase_displacement_by_agent,
        ),
        ("charge_displacement", physical.charge_phase_displacement_by_agent),
    ):
        message = f"{label}: {field} changed"
        assert np.shape(actual) == np.shape(expected[field]), message
        tolerance = 1e-5 if field in ("health", "damage") else 0.00016
        np.testing.assert_allclose(
            actual, expected[field], rtol=0, atol=tolerance, err_msg=message
        )

    for field, actual in (
        ("move", action.move),
        ("select_target", action.select_target),
        ("use_ultimate", action.use_ultimate),
        ("basic_activated", combat.basic_effect_is_activated_by_source),
        ("alive", state.alive_mask),
        ("newly_dead", facts.death_facts.is_newly_dead_by_recipient),
        ("scores", state.team_deathmatch_scores),
        ("rewards", reward.rewards),
        ("step_count", state.step_count),
        ("terminated", done.terminated),
        ("truncated", done.truncated),
        ("outcome", facts.team_deathmatch_facts.outcome),
    ):
        message = f"{label}: {field} changed"
        assert np.shape(actual) == np.shape(expected[field]), message
        np.testing.assert_array_equal(actual, expected[field], err_msg=message)

    acceptance = facts.action_acceptance_facts
    for recorded in (
        acceptance.submitted_joint_action,
        acceptance.accepted_joint_action,
    ):
        for field, actual, submitted in zip(
            Action._fields, recorded, action, strict=True
        ):
            message = f"{label}: {field} submission/acceptance changed"
            assert np.shape(actual) == np.shape(submitted), message
            np.testing.assert_array_equal(actual, submitted, err_msg=message)
