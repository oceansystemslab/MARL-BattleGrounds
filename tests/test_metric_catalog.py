"""Qualify the fixed scalar schema without duplicating numerical reducers."""

import re
from collections import Counter
from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from marl_battlegrounds.evaluation.metric_catalog import (
    FAMILY_COLUMN_COUNTS,
    FULL_METRIC_NAMES,
    METRIC_COLUMNS,
    METRIC_COLUMNS_BY_NAME,
    METRIC_FAMILIES,
    PRIORITY_METRIC_COLUMNS,
    PRIORITY_METRIC_NAMES,
)


def test_priority_order_matches_the_public_result_vector() -> None:
    expected = (
        "episode_length",
        "team_a_win",
        "team_a_draw",
        "team_a_loss",
        "team_b_win",
        "team_b_draw",
        "team_b_loss",
        "team_a_return",
        "team_b_return",
        *(f"agent_{slot}_return" for slot in range(10)),
        "team_a_score",
        "team_b_score",
        "score_difference",
        "team_a_kills",
        "team_b_kills",
        "team_a_deaths",
        "team_b_deaths",
    )
    assert expected == PRIORITY_METRIC_NAMES
    assert len(PRIORITY_METRIC_COLUMNS) == 26
    assert METRIC_COLUMNS[:26] == PRIORITY_METRIC_COLUMNS


def test_fixed_catalog_has_unique_names_and_immutable_complete_definitions() -> None:
    assert len(FULL_METRIC_NAMES) == len(set(FULL_METRIC_NAMES)) == 1388
    assert tuple(METRIC_COLUMNS_BY_NAME) == FULL_METRIC_NAMES
    for column in METRIC_COLUMNS:
        assert re.fullmatch(r"[a-z][a-z0-9_]*", column.name)
        assert column.label and column.description and column.missing_when
        assert column.unit
    with pytest.raises(FrozenInstanceError):
        METRIC_COLUMNS[0].name = "changed"  # pyright: ignore[reportAttributeAccessIssue]
    assert isinstance(METRIC_COLUMNS_BY_NAME, MappingProxyType)
    assert isinstance(METRIC_FAMILIES, MappingProxyType)
    assert set(METRIC_FAMILIES) == {
        "overview",
        *(column.family for column in METRIC_COLUMNS),
    }
    for label, description in METRIC_FAMILIES.values():
        assert label and description.endswith(".")
        assert "_" not in label
    assert METRIC_FAMILIES["controlled_damage"][0] == "Damage to Controlled Recipients"


def test_recipient_dimensions_cover_only_the_two_requested_matrices() -> None:
    columns = [
        column for column in METRIC_COLUMNS if column.scope == "source_recipient"
    ]
    assert len(columns) == 200
    assert Counter(column.family for column in columns) == {
        "recipient_damage": 100,
        "recipient_healing": 100,
    }
    for column in columns:
        source, recipient = column.subjects
        assert 0 <= source < 10 and 0 <= recipient < 10
        assert (source // 5 == recipient // 5) == (column.family == "recipient_healing")
        assert column.status_channel is None
    # A recipient has one received amount, shared with its team-to-recipient view.
    assert (
        len([name for name in FULL_METRIC_NAMES if name.endswith("_damage_received")])
        == 12
    )
    assert (
        len(
            [
                name
                for name in FULL_METRIC_NAMES
                if name.endswith("_priest_healing_received")
            ]
        )
        == 12
    )
    assert not any(
        name.startswith("team_") and "_agent_" in name for name in FULL_METRIC_NAMES
    )


def test_formation_keeps_twenty_unordered_ally_pairs_and_two_team_summaries() -> None:
    columns = [column for column in METRIC_COLUMNS if column.family == "formation"]
    assert len(columns) == 44
    pairs = {column.subjects for column in columns if column.scope == "ally_pair"}
    assert len(pairs) == 20
    assert all(first < second and first // 5 == second // 5 for first, second in pairs)
    assert Counter(column.scope for column in columns) == {"ally_pair": 40, "team": 4}


def test_status_dimensions_do_not_expand_into_recipient_or_ability_cross_products() -> (
    None
):
    for family in ("controlled_damage", "controlled_healing", "controlled_kills"):
        columns = [column for column in METRIC_COLUMNS if column.family == family]
        assert {column.status_channel for column in columns} == set(range(7))
        assert {column.scope for column in columns} == {"agent", "team"}
    for family in ("status_applications", "status_active_steps"):
        columns = [column for column in METRIC_COLUMNS if column.family == family]
        assert len(columns) == 108
        assert {column.status_channel for column in columns} == set(range(9))
        assert {column.scope for column in columns} == {"agent", "team"}
    assert not any(
        column.unit == "fraction"
        and column.family in ("controlled_damage", "controlled_healing")
        for column in METRIC_COLUMNS
    )


def test_applicability_distinguishes_general_zeros_from_class_specific_absence() -> (
    None
):
    for name in ("agent_0_damage_done", "agent_0_healing_done"):
        assert METRIC_COLUMNS_BY_NAME[name].required_class_id is None
    assert METRIC_COLUMNS_BY_NAME["agent_0_burst_damage"].required_class_id == 1
    assert METRIC_COLUMNS_BY_NAME["agent_0_wasted_healing"].required_class_id == 5
    assert (
        "is zero" in METRIC_COLUMNS_BY_NAME["agent_0_kill_participation"].missing_when
    )
    assert (
        "is zero"
        in METRIC_COLUMNS_BY_NAME["agent_0_agent_1_ally_distance_mean"].missing_when
    )
    assert "inactive" in METRIC_COLUMNS_BY_NAME["agent_9_return"].missing_when
    assert "agent_0_rescue_opportunities" not in METRIC_COLUMNS_BY_NAME
    assert "agent_0_rescue_rate" not in METRIC_COLUMNS_BY_NAME
    assert (
        "unique team rescues"
        in METRIC_COLUMNS_BY_NAME["agent_0_rescue_participation"].description
    )


def test_family_budget_retains_requested_counts_without_duplicate_aliases() -> None:
    assert dict(FAMILY_COLUMN_COUNTS) == {
        "priority": 26,
        "abilities": 24,
        "deaths": 42,
        "kill_contributions": 30,
        "damage_done": 22,
        "recipient_damage": 100,
        "healing_done": 22,
        "recipient_healing": 100,
        "damage_received": 22,
        "healing_received": 70,
        "wasted_healing": 22,
        "controlled_damage": 84,
        "controlled_healing": 84,
        "controlled_kills": 154,
        "coordination": 12,
        "action_acceptance": 84,
        "status_applications": 108,
        "status_active_steps": 108,
        "trap_breaks": 28,
        "respawn": 4,
        "burst": 48,
        "aura_coverage": 72,
        "aura_benefits": 4,
        "poison": 12,
        "priest_rescue": 26,
        "freedom": 36,
        "formation": 44,
    }
    assert not any("burst_activations" in name for name in FULL_METRIC_NAMES)
    assert not any("class_" in name for name in FULL_METRIC_NAMES)
    assert not any(
        column.family in ("aura_coverage", "aura_benefits")
        and column.scope in ("source_recipient", "ally_pair")
        for column in METRIC_COLUMNS
    )
