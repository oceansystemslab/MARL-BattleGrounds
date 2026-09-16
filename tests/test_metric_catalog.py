"""Check scalar metric names, ordering and schema rules.

The catalog tests share the numerical definitions rather than implementing a
second metric calculator.
"""

import hashlib
import re
from collections import Counter
from dataclasses import FrozenInstanceError, asdict
from types import MappingProxyType

import pytest

from marl_battlegrounds.evaluation.metric_catalog import (
    DIRECTED_METRICS,
    FAMILY_COLUMN_COUNTS,
    FULL_METRIC_NAMES,
    METRIC_COLUMNS,
    METRIC_COLUMNS_BY_NAME,
    METRIC_FAMILIES,
    METRIC_GROUPS,
    METRIC_SCHEMA_VERSION,
    METRIC_TOPICS,
    PRIORITY_METRIC_COLUMNS,
    PRIORITY_METRIC_NAMES,
    RECIPIENT_PAIRS_BY_RELATION,
    STATUS_LABELS,
    STATUS_NAMES,
    ULTIMATE_GROUP_BY_CLASS,
    metric_groups,
    metric_guidance,
    metric_locations,
    metric_order_key,
    metric_primary_location,
    metric_search_facts,
    metric_topic_text,
    metric_view_order,
)


def test_search_facts_cover_the_fixed_catalog_without_changing_columns() -> None:
    before = tuple(asdict(column) for column in METRIC_COLUMNS)
    expected_keys = {
        "kind",
        "ability",
        "status",
        "status_subject",
        "source_class_id",
        "recipient_class_id",
        "subject",
        "source",
        "relation",
        "qualifiers",
    }
    kinds = {
        "damage",
        "healing",
        "regeneration",
        "excess_healing",
        "effective_healing",
        "kill",
        "death",
        "rescue",
        "activation",
        "action",
        "respawn",
        "formation",
        "aura_coverage",
        "aura_benefit",
        "status_application",
        "status_time",
        "freedom",
        "poison_prevention",
        "trap_break",
        "coordination",
        "return",
        "score",
        "episode",
        "outcome",
    }
    seen_kinds: set[object] = set()
    for column in METRIC_COLUMNS:
        facts = metric_search_facts(column)
        assert set(facts) == expected_keys
        assert facts["kind"] in kinds
        seen_kinds.add(facts["kind"])
        assert facts["ability"] in (None, "basic", "ultimate", "burst")
        assert facts["status"] in (None, *STATUS_NAMES)
        assert facts["status_subject"] in (None, "source", "recipient")
        assert (facts["status"] is None) == (facts["status_subject"] is None)
        assert isinstance(facts["qualifiers"], list)
        if column.unit == "fraction":
            assert "fraction" in facts["qualifiers"]
        assert facts["subject"] in ("source", "recipient", "pair", "team", "episode")
        assert facts["source"] in ("none", "agent", "team", "class", "unknown")
        assert facts["relation"] in (None, "ally", "enemy", "self")
        for field in ("source_class_id", "recipient_class_id"):
            assert facts[field] in (None, 1, 2, 3, 4, 5)
        if column.scope in ("source_recipient", "team_recipient"):
            assert facts["subject"] == "source"
        if column.scope == "ally_pair":
            assert facts["subject"] == "pair"
            assert facts["source"] == "none"
    assert seen_kinds == kinds
    assert tuple(asdict(column) for column in METRIC_COLUMNS) == before


def test_search_facts_keep_healers_effects_and_recipients_distinct() -> None:
    cases = {
        "agent_0_regenerated_healing": {
            "kind": "regeneration",
            "source": "none",
            "source_class_id": None,
            "subject": "recipient",
        },
        "agent_0_healing_received": {
            "kind": "healing",
            "source_class_id": None,
            "qualifiers": ["combined"],
        },
        "team_a_effective_healing_received": {
            "kind": "effective_healing",
            "source_class_id": None,
            "qualifiers": ["combined"],
        },
        "agent_0_priest_healing_received": {
            "kind": "healing",
            "source": "class",
            "source_class_id": 5,
            "recipient_class_id": None,
            "relation": "ally",
            "qualifiers": ["priest"],
        },
        "agent_0_to_agent_1_basic_effective_healing_done": {
            "kind": "effective_healing",
            "ability": "basic",
            "source": "agent",
            "source_class_id": 5,
            "recipient_class_id": None,
        },
        "team_b_ultimate_excess_healing": {
            "kind": "excess_healing",
            "ability": "ultimate",
            "source": "team",
            "source_class_id": 5,
        },
        "team_a_priest_holy_word_salvation_applications": {
            "kind": "activation",
            "ability": "ultimate",
            "source": "team",
            "source_class_id": 5,
            "status": None,
            "status_subject": None,
            "subject": "source",
        },
        "agent_0_damage_received_while_hunter_basic_slow": {
            "kind": "damage",
            "ability": None,
            "source_class_id": None,
            "status": "hunter_basic_slow",
            "status_subject": "recipient",
            "relation": "enemy",
        },
        "agent_0_healing_received_while_hunter_trap": {
            "kind": "healing",
            "source_class_id": 5,
            "status": "hunter_trap",
            "recipient_class_id": None,
        },
        "agent_0_hunter_trap_active_steps": {
            "kind": "status_time",
            "source": "unknown",
            "source_class_id": None,
            "recipient_class_id": None,
            "relation": None,
            "status_subject": "recipient",
        },
        "agent_0_to_agent_5_hunter_trap_applications": {
            "kind": "status_application",
            "source": "agent",
            "source_class_id": 3,
            "ability": "ultimate",
            "status": "hunter_trap",
            "status_subject": "recipient",
        },
        "agent_0_burst_damage_received": {
            "kind": "damage",
            "source": "class",
            "source_class_id": 1,
            "recipient_class_id": None,
            "ability": "burst",
            "relation": "enemy",
            "status_subject": "source",
        },
        "team_a_damage_done": {
            "kind": "damage",
            "source": "team",
            "source_class_id": None,
        },
        "team_a_mage_basic_kills": {
            "kind": "kill",
            "ability": "basic",
            "source_class_id": 1,
        },
        "team_a_ultimate_kills": {
            "kind": "kill",
            "ability": "ultimate",
            "source_class_id": None,
        },
        "agent_0_deaths_while_hunter_trap": {
            "kind": "death",
            "subject": "recipient",
            "source_class_id": None,
        },
        "agent_0_to_agent_5_basic_kill_contributions": {
            "kind": "kill",
            "subject": "source",
            "ability": "basic",
            "source_class_id": None,
            "relation": "enemy",
        },
        "agent_0_rescues": {
            "kind": "rescue",
            "subject": "recipient",
            "source_class_id": 5,
            "ability": None,
            "relation": "ally",
        },
        "agent_0_mage_aura_covered_recipient_steps": {
            "kind": "aura_coverage",
            "subject": "recipient",
            "source_class_id": 1,
            "recipient_class_id": None,
            "relation": "ally",
        },
        "team_a_damage_from_mage_aura": {
            "kind": "aura_benefit",
            "source": "class",
            "source_class_id": 1,
        },
        "agent_0_healing_prevented_by_poison": {
            "kind": "poison_prevention",
            "source": "unknown",
            "source_class_id": None,
            "status": "rogue_poison_anti_heal",
        },
        "agent_0_freedom_protected_steps": {
            "kind": "freedom",
            "source": "unknown",
            "source_class_id": None,
            "status": "priest_freedom",
        },
    }
    for name, expected in cases.items():
        facts = metric_search_facts(METRIC_COLUMNS_BY_NAME[name])
        assert {field: facts[field] for field in expected} == expected, name


def test_search_pair_facts_use_the_directed_registry_for_every_amount_and_share() -> (
    None
):
    for metric in DIRECTED_METRICS:
        stems = (metric.amount, metric.allocation, metric.contribution)
        columns = (
            column
            for column in METRIC_COLUMNS
            if column.scope == "source_recipient" and column.stem in stems
        )
        for column in columns:
            facts = metric_search_facts(column)
            assert facts["source"] == "agent", column.name
            assert facts["relation"] == (
                None if metric.relation == "all" else metric.relation
            ), column.name
            assert facts["recipient_class_id"] is None, column.name
            if metric.required_class_id is not None:
                assert facts["source_class_id"] == metric.required_class_id, column.name
            if metric.status_channel is not None:
                assert facts["status"] == STATUS_NAMES[metric.status_channel], (
                    column.name
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
    assert len(FULL_METRIC_NAMES) == len(set(FULL_METRIC_NAMES)) == 11158
    assert METRIC_SCHEMA_VERSION == 13
    additions = {
        f"agent_{slot}_{stem}"
        for slot in range(10)
        for stem in (
            "trap_intervals",
            "trap_break_rate",
            "trap_mean_remaining_steps_at_break",
            "rescue_opportunities",
        )
    }
    assert len(additions) == 40
    assert additions <= set(FULL_METRIC_NAMES)
    basic_shares = {
        *(f"agent_{slot}_basic_rescue_participation" for slot in range(10)),
        "team_a_basic_rescue_fraction",
        "team_b_basic_rescue_fraction",
    }
    assert len(basic_shares) == 12
    assert basic_shares <= set(FULL_METRIC_NAMES)
    ability_counts = {
        f"team_{team}_{stem}"
        for team in ("a", "b")
        for stem in (
            "warrior_charge_applications",
            "rogue_poison_applications",
            "priest_holy_word_salvation_applications",
        )
    }
    assert len(ability_counts) == 6
    assert ability_counts <= set(FULL_METRIC_NAMES)
    earlier_names = [name for name in FULL_METRIC_NAMES if name not in ability_counts]
    # Every earlier name and its relative position stays unchanged.
    assert len(earlier_names) == 11152
    assert hashlib.sha256("\n".join(earlier_names).encode()).hexdigest() == (
        "5217c631d20da81f7e90ba5ef3d17a7a244bdda986c6983a64a387ca7f41296e"
    )
    surviving = sorted(set(earlier_names) - basic_shares)
    assert len(surviving) == 11140
    assert hashlib.sha256("\n".join(surviving).encode()).hexdigest() == (
        "29575079c8dd13c532367461619660b9c219ced7d8db2d44bd6ab9f6a3ab8852"
    )
    directional = Counter(
        column.scope
        for column in METRIC_COLUMNS
        if "_to_agent_" in column.name or "_and_agent_" in column.name
    )
    assert directional == {
        "team_recipient": 120,
        "source_recipient": 9050,
        "ally_pair": 40,
    }
    # These are the approved removals, stated independently of catalog helpers.
    removed = {
        f"team_{team}_{stem}"
        for team in ("a", "b")
        for stem in (
            "mage_ultimate_damage_done",
            "priest_ultimate_damage_done",
            "mage_ultimate_kills",
            "mage_ultimate_kill_fraction",
            "priest_ultimate_activations",
            "warrior_ultimate_damage_done",
            "hunter_ultimate_damage_done",
            "rogue_ultimate_damage_done",
        )
    }
    for slot in range(10):
        pair = f"agent_{slot}_to_agent_{slot}_"
        removed.update(
            pair + f"healing_to_{status}_recipient" + suffix
            for status in ("warrior_charge_stun", "hunter_trap", "rogue_poison_stun")
            for suffix in ("", "_allocation_fraction", "_contribution_fraction")
        )
        removed.update(
            pair + "mage_burst_applications" + suffix
            for suffix in ("", "_allocation_fraction", "_contribution_fraction")
        )
        removed.add(
            f"team_{'a' if slot < 5 else 'b'}_to_agent_{slot}_mage_burst_applications"
        )
        removed.update(
            pair + f"{emitter}_aura_{stem}"
            for emitter in ("mage", "warrior")
            for stem in (
                "eligible_steps",
                "coverage",
                "covered_steps_contribution_fraction",
            )
        )
        assert f"agent_{slot}_mage_burst_applications" in METRIC_COLUMNS_BY_NAME
        for emitter in ("mage", "warrior"):
            assert pair + f"{emitter}_aura_covered_steps" in METRIC_COLUMNS_BY_NAME
    assert len(removed) == 206
    assert not removed.intersection(FULL_METRIC_NAMES)
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
        *ULTIMATE_GROUP_BY_CLASS.values(),
        *(column.family for column in METRIC_COLUMNS),
    }
    for label, description in METRIC_FAMILIES.values():
        assert label and description.endswith(".")
        assert "_" not in label
    assert METRIC_FAMILIES["controlled_damage"][0] == "Damage to Controlled Recipients"


def test_recipient_dimensions_cover_effects_and_declared_ultimate_details() -> None:
    columns = [
        column for column in METRIC_COLUMNS if column.scope == "source_recipient"
    ]
    assert len(columns) == 9050
    assert Counter(column.family for column in columns) == {
        "recipient_healing": 1650,
        "status_applications": 1200,
        "controlled_damage": 1050,
        "controlled_healing": 960,
        "controlled_kills": 1050,
        "abilities": 600,
        "kill_contributions": 600,
        "aura_coverage": 440,
        "recipient_damage": 450,
        "burst": 450,
        "priest_rescue": 450,
        "trap_breaks": 150,
    }
    for column in columns:
        source, recipient = column.subjects
        assert 0 <= source < 10 and 0 <= recipient < 10
        assert column.recipient_role
        if column.family in (
            "recipient_damage",
            "burst",
            "kill_contributions",
            "trap_breaks",
        ):
            assert source // 5 != recipient // 5
        if column.family in ("recipient_healing", "priest_rescue", "aura_coverage"):
            assert source // 5 == recipient // 5
    # Generic recipient totals reuse received amounts rather than team aliases.
    assert "team_a_to_agent_8_damage_done" not in METRIC_COLUMNS_BY_NAME
    assert "team_a_to_agent_0_healing_done" not in METRIC_COLUMNS_BY_NAME
    assert "agent_8_damage_received" in METRIC_COLUMNS_BY_NAME
    assert "agent_0_priest_healing_received" in METRIC_COLUMNS_BY_NAME
    assert "team_a_to_agent_8_ultimate_applications" in METRIC_COLUMNS_BY_NAME
    for slot in range(10):
        for stem, unit, topic in (
            ("trap_intervals", "count", "ultimate_hunter"),
            ("trap_break_rate", "fraction", "ultimate_hunter"),
            ("trap_mean_remaining_steps_at_break", "steps", "ultimate_hunter"),
            ("rescue_opportunities", "count", "priest_rescue"),
        ):
            column = METRIC_COLUMNS_BY_NAME[f"agent_{slot}_{stem}"]
            assert column.scope == "agent"
            assert column.subjects == (slot,)
            assert column.subject_role == "recipient"
            assert column.required_class_id is None
            assert column.unit == unit
            assert metric_primary_location(column) == (topic, "recipients")
            assert (column.denominator is None) == (unit == "count")
            if unit != "count":
                assert column.numerator is not None
                assert "is zero" in column.missing_when


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
        assert {column.scope for column in columns} == {
            "agent",
            "team",
            "source_recipient",
        }
        paired = [column for column in columns if column.scope == "source_recipient"]
        assert len(paired) == 50 * 7 * 3 - (90 if family == "controlled_healing" else 0)
        assert not any(
            column.stem.startswith(("basic_", "ultimate_")) for column in paired
        )
    applications = [
        column for column in METRIC_COLUMNS if column.family == "status_applications"
    ]
    counts = [column for column in applications if column.unit == "count"]
    assert (
        len([column for column in counts if column.scope in ("team", "agent")]) == 108
    )
    assert (
        len([column for column in counts if column.scope == "source_recipient"]) == 400
    )
    active = [
        column for column in METRIC_COLUMNS if column.family == "status_active_steps"
    ]
    assert len(active) == 108
    assert {column.scope for column in active} == {"team", "agent"}


def test_applicability_distinguishes_general_zeros_from_class_specific_absence() -> (
    None
):
    for name in ("agent_0_damage_done", "agent_0_healing_done"):
        assert METRIC_COLUMNS_BY_NAME[name].required_class_id is None
    assert METRIC_COLUMNS_BY_NAME["agent_0_burst_damage"].required_class_id == 1
    assert METRIC_COLUMNS_BY_NAME["agent_0_excess_healing"].required_class_id == 5
    assert (
        "is zero" in METRIC_COLUMNS_BY_NAME["agent_0_kill_participation"].missing_when
    )
    assert (
        "is zero"
        in METRIC_COLUMNS_BY_NAME["agent_0_and_agent_1_ally_distance_mean"].missing_when
    )
    assert "was not included in this game" in (
        METRIC_COLUMNS_BY_NAME["agent_9_return"].missing_when
    )
    opportunity = METRIC_COLUMNS_BY_NAME["agent_0_rescue_opportunities"]
    assert opportunity.subject_role == "recipient"
    assert opportunity.required_class_id is None
    # This ally's chance to be saved is not one Priest's ability to save it alone.
    assert "agent_0_rescue_rate" not in METRIC_COLUMNS_BY_NAME
    assert (
        "Team A's rescues"
        in METRIC_COLUMNS_BY_NAME["agent_0_rescue_participation"].description
    )


def test_family_budget_retains_requested_counts_without_duplicate_aliases() -> None:
    assert dict(FAMILY_COLUMN_COUNTS) == {
        "priority": 26,
        "abilities": 670,
        "deaths": 42,
        "kill_contributions": 734,
        "damage_done": 46,
        "recipient_damage": 450,
        "healing_done": 118,
        "recipient_healing": 1650,
        "damage_received": 42,
        "healing_received": 158,
        "excess_healing": 102,
        "controlled_damage": 1204,
        "controlled_healing": 1114,
        "controlled_kills": 1274,
        "coordination": 12,
        "action_acceptance": 84,
        "status_applications": 1388,
        "status_active_steps": 108,
        "trap_breaks": 218,
        "respawn": 6,
        "burst": 520,
        "aura_coverage": 552,
        "aura_benefits": 4,
        "poison": 12,
        "priest_rescue": 544,
        "freedom": 36,
        "formation": 44,
    }
    assert sum(FAMILY_COLUMN_COUNTS.values()) == len(FULL_METRIC_NAMES)
    assert not any("burst_activations" in name for name in FULL_METRIC_NAMES)
    assert not any("class_" in name for name in FULL_METRIC_NAMES)
    assert not any(
        column.family in ("poison", "aura_benefits", "status_active_steps")
        and column.scope == "source_recipient"
        for column in METRIC_COLUMNS
    )


def test_every_measure_has_its_reviewed_direction_without_a_default() -> None:
    expected: dict[str, dict[str, str]] = {}

    def classify(
        family: str, *, higher: str = "", lower: str = "", context: str = ""
    ) -> None:
        measures: dict[str, str] = {}
        for direction, names in (
            ("higher", higher),
            ("lower", lower),
            ("descriptive", context),
        ):
            for name in names.split():
                assert name not in measures
                measures[name] = direction
        expected[family] = measures

    classify(
        "priority",
        higher="win return score score_difference kills",
        lower="loss deaths",
        context="draw episode_length",
    )
    classify(
        "abilities",
        context="basic_activations ultimate_activations "
        "ultimate_applications ultimate_application_fraction",
    )
    classify(
        "deaths", lower="deaths dead_steps", context="death_fraction dead_step_fraction"
    )
    classify(
        "kill_contributions",
        higher="kill_contributions "
        "kill_contributions_per_death solo_kills "
        "ultimate_kill_contributions ultimate_kills",
        context="solo_kill_fraction basic_kill_fraction ultimate_kill_fraction "
        "kill_participation basic_kill_participation ultimate_kill_participation",
    )
    classify(
        "damage_done",
        higher="damage_done ultimate_damage_done",
        context="damage_done_fraction",
    )
    classify(
        "recipient_damage",
        higher="damage_done basic_damage_done ultimate_damage_done",
        context="damage_done_fraction",
    )
    classify(
        "healing_done",
        higher="effective_healing_done "
        "ultimate_effective_healing_done ultimate_effective_healing_fraction",
        context="healing_done healing_done_fraction ultimate_healing_done "
        "effective_healing_fraction",
    )
    classify(
        "recipient_healing",
        higher="effective_healing_done basic_effective_healing_done "
        "ultimate_effective_healing_done",
        lower="ultimate_excess_healing",
        context="healing_done healing_done_fraction ultimate_healing_done "
        "excess_healing basic_excess_healing",
    )
    classify(
        "damage_received",
        lower="damage_received basic_damage_received ultimate_damage_received",
        context="damage_received_fraction",
    )
    classify(
        "healing_received",
        higher="effective_priest_healing_received",
        context="effective_healing_received priest_healing_received "
        "healing_received regenerated_healing "
        "priest_healing_received_fraction regeneration_healing_received_fraction "
        "healing_received_fraction excess_priest_healing_received_fraction "
        "effective_priest_healing_received_fraction",
    )
    classify(
        "excess_healing",
        lower="ultimate_excess_healing_fraction",
        context="excess_healing excess_healing_received excess_healing_fraction",
    )
    classify(
        "controlled_damage",
        higher=" ".join(
            f"damage_to_{status}_recipient"
            for status in STATUS_NAMES[:7]
            if status != "hunter_trap"
        ),
        context="damage_to_hunter_trap_recipient",
    )
    classify(
        "controlled_healing",
        context=" ".join(
            f"healing_to_{status}_recipient" for status in STATUS_NAMES[:7]
        ),
    )
    classify(
        "controlled_kills",
        higher=" ".join(
            f"{measure}_{status}_recipient"
            for status in STATUS_NAMES[:7]
            for measure in (
                "kills_of",
                "kill_contributions_to",
            )
        ),
        context=" ".join(
            f"kill_participation_in_{status}_recipient" for status in STATUS_NAMES[:7]
        ),
    )
    classify(
        "coordination",
        higher="single_contributor_kills multi_contributor_kills",
        context="single_contributor_kill_fraction multi_contributor_kill_fraction "
        "focus_fire_concentration focus_fire_steps",
    )
    classify(
        "action_acceptance",
        higher="action_acceptance_rate actions_accepted",
        lower="actions_rejected action_domain_rejections action_movement_rejections "
        "action_combat_rejections",
        context="actions_submitted",
    )
    classify(
        "status_applications",
        context=" ".join(f"{status}_applications" for status in STATUS_NAMES),
    )
    classify(
        "status_active_steps",
        lower=" ".join(f"{status}_active_steps" for status in STATUS_NAMES[:7]),
        context=" ".join(f"{status}_active_steps" for status in STATUS_NAMES[7:]),
    )
    classify(
        "trap_breaks",
        context="trap_intervals trap_breaks trap_break_rate "
        "trap_mean_remaining_steps_at_break trap_break_contributions "
        "trap_break_participation",
    )
    classify(
        "respawn",
        context="mean_observed_respawn_wait_steps respawn_waves "
        "mean_agents_per_respawn_wave",
    )
    classify(
        "burst",
        higher="burst_damage burst_damage_contributing_to_kill "
        "burst_kill_contributions",
        context="burst_damage_fraction",
    )
    classify(
        "aura_coverage",
        context="mage_aura_coverage warrior_aura_coverage "
        "mage_aura_covered_steps mage_aura_eligible_steps "
        "warrior_aura_covered_steps warrior_aura_eligible_steps",
    )
    classify(
        "aura_benefits", higher="damage_from_mage_aura damage_prevented_by_warrior_aura"
    )
    classify("poison", lower="healing_prevented_by_poison")
    classify(
        "priest_rescue",
        higher="rescues rescue_rate rescue_contributions "
        "ultimate_rescue_contributions ultimate_rescues",
        context="rescue_opportunities basic_rescue_fraction basic_rescue_participation "
        "ultimate_rescue_fraction "
        "rescue_participation ultimate_rescue_participation",
    )
    classify(
        "freedom",
        context="freedom_protected_steps freedom_eligible_steps "
        "freedom_protection_fraction",
    )
    classify("formation", context="ally_distance_mean ally_distance_observations")

    observed: dict[str, set[str]] = {}
    for column in METRIC_COLUMNS:
        name = column.stem
        if name not in expected[column.family]:
            continue
        if column.scope == "team_recipient":
            continue
        if column.scope == "source_recipient" and not (
            column.family in ("recipient_damage", "recipient_healing", "abilities")
            or (column.family == "burst" and name == "burst_damage")
        ):
            continue
        if column.subject_role == "recipient" and column.family in (
            "priest_rescue",
            "trap_breaks",
        ):
            continue
        observed.setdefault(column.family, set()).add(name)
        direction = expected[column.family][name]
        assert column.direction == direction, column.name
    assert observed == {family: set(names) for family, names in expected.items()}

    effective_counts = {"higher": 0, "descriptive": 0}
    for column in METRIC_COLUMNS:
        if "effective" not in column.stem or "healing" not in column.stem:
            continue
        share = column.stem.endswith(("_allocation_fraction", "_contribution_fraction"))
        basic_or_combined_fraction = column.stem in (
            "effective_healing_fraction",
            "basic_effective_healing_fraction",
            "effective_priest_healing_received_fraction",
        )
        expected_direction = (
            "descriptive"
            if share
            or basic_or_combined_fraction
            or column.stem == "effective_healing_received"
            else "higher"
        )
        direction, guidance = metric_guidance(column)
        assert column.direction == direction == expected_direction, column.name
        assert ("Freedom" in guidance) == basic_or_combined_fraction
        assert "More regeneration" not in guidance
        if column.unit == "fraction" and not share:
            excess = METRIC_COLUMNS_BY_NAME[
                column.name.replace("effective", "excess", 1)
            ]
            assert excess.denominator == column.denominator
            assert excess.subjects == column.subjects
            assert excess.scope == column.scope
            assert excess.direction == (
                "descriptive" if basic_or_combined_fraction else "lower"
            )
        effective_counts[direction] += 1
    assert effective_counts == {"higher": 280, "descriptive": 448}

    # Trap damage can free the enemy. Excess can accompany useful Basic Freedom;
    # Salvation does not give Freedom. Keep those explanations distinct.
    trap_amounts = 0
    excess_columns = 0
    for column in METRIC_COLUMNS:
        _, guidance = metric_guidance(column)
        if column.stem.startswith("damage_to_hunter_trap_recipient"):
            assert column.direction == "descriptive"
            assert "Damage can free a trapped enemy" in guidance
            trap_amounts += column.denominator is None
        elif column.family == "controlled_damage" and column.subject_role == "source":
            assert column.direction == (
                "descriptive" if column.denominator else "higher"
            )
        if "excess" in column.stem:
            excess_columns += 1
            if column.stem.startswith("ultimate_"):
                allocation = column.stem.endswith(
                    ("_allocation_fraction", "_contribution_fraction")
                )
                assert column.direction == ("descriptive" if allocation else "lower")
                assert (
                    "where excess Ultimate healing went"
                    if allocation
                    else "This Ultimate does not give Freedom"
                ) in guidance
            else:
                assert column.direction == "descriptive", column.name
                assert "Basic ability can still give Freedom" in guidance
    assert trap_amounts == 62
    assert excess_columns == 714

    # A recorded Warrior can occupy any slot. Other classes do not inherit its
    # tank role. These checks cover all twelve incoming-damage amount variants.
    incoming_stems = (
        "damage_received",
        "basic_damage_received",
        "ultimate_damage_received",
        "burst_damage_received",
        "burst_damage_contributing_to_kill_received",
        *(f"damage_received_while_{status}" for status in STATUS_NAMES[:7]),
    )
    for slot in range(10):
        team = "A" if slot < 5 else "B"
        for stem in incoming_stems:
            column = METRIC_COLUMNS_BY_NAME[f"agent_{slot}_{stem}"]
            if stem == "damage_received_while_hunter_trap":
                assert column.direction == "descriptive"
                for class_id in range(1, 6):
                    assert metric_guidance(column, (class_id,) * 10) == (
                        "descriptive",
                        f"Context dependent for Team {team}. "
                        "Damage can free this agent from Trap.",
                    )
                continue
            assert column.direction == "lower"
            assert "when this agent is a Warrior" in metric_guidance(column)[1]
            for class_id in range(1, 6):
                # Move this class through one slot at a time. Other slots stay
                # Mage, so reading a fixed slot would give the wrong answer.
                classes = tuple(class_id if i == slot else 1 for i in range(10))
                direction, text = metric_guidance(column, classes)
                assert direction == ("descriptive" if class_id == 2 else "lower")
                assert f"for Team {team}" in text
                assert ("protect teammates" in text) == (class_id == 2)
        assert metric_guidance(
            METRIC_COLUMNS_BY_NAME[f"agent_{slot}_deaths"], (2,) * 10
        ) == ("lower", f"Lower is better for Team {team}.")
    assert metric_guidance(METRIC_COLUMNS_BY_NAME["score_difference"])[1] == (
        "Higher is better for Team A; lower is better for Team B."
    )
    for source, target, team in ((0, 5, "A"), (5, 0, "B")):
        assert metric_guidance(
            METRIC_COLUMNS_BY_NAME[f"agent_{source}_to_agent_{target}_damage_done"],
            (2,) * 10,
        ) == ("higher", f"Higher is better for Team {team}.")


def test_plain_descriptions_separate_effects_counts_and_opportunity_meanings() -> None:
    for ability in ("basic", "ultimate"):
        made = METRIC_COLUMNS_BY_NAME[f"agent_1_{ability}_activations"]
        received = METRIC_COLUMNS_BY_NAME[f"team_a_to_agent_1_{ability}_applications"]
        allocation = METRIC_COLUMNS_BY_NAME[
            f"agent_2_to_agent_1_{ability}_application_fraction"
        ]
        contribution = METRIC_COLUMNS_BY_NAME[
            f"agent_2_to_agent_1_{ability}_application_contribution_fraction"
        ]
        assert made.label == f"{ability.title()} Ability Activations"
        assert made.description.startswith(
            f"How many times this agent used its {ability.title()} ability."
        )
        assert METRIC_COLUMNS_BY_NAME[
            f"team_a_{ability}_activations"
        ].description.startswith(
            f"How many times everyone on Team A used their "
            f"{ability.title()} abilities, added together."
        )
        assert received.label == f"{ability.title()} Ability Activations Received"
        assert received.description.startswith(
            f"How many times agents on Team A used their "
            f"{ability.title()} abilities on this target."
        )
        assert "4/10 = 0.4 (40%)" in allocation.description
        assert (
            allocation.denominator
            == f"All {ability.title()} ability activations by this agent"
        )
        assert "4/8 = 0.5 (50%)" in contribution.description
        assert contribution.denominator == (
            f"All {ability.title()} ability activations by Team A on this target, "
            "including every class"
        )
        assert "includes every class" in contribution.description
    for column in METRIC_COLUMNS:
        if column.family in (
            "damage_done",
            "recipient_damage",
            "damage_received",
            "controlled_damage",
        ):
            assert "overheal" not in column.description.lower()
            assert "regeneration" not in column.description.lower()
        assert not re.search(r"\b(canonical|configured|raw)\b", column.description)
        assert "divided by" not in column.description.lower()
        assert "divide " not in column.description.lower()
        assert not re.search(r"\b(Basic|Ultimate) Uses\b", column.label)
        for text in (
            column.label,
            column.description,
            column.numerator,
            column.denominator,
            column.missing_when,
        ):
            if text:
                assert text == text.strip()
                assert text[0].isupper() or text[0].isdigit(), column.name
                assert not re.search(r"\b(this|its|their|the whole) team\b", text)
        assert "Recipient to Recipient" not in column.label
    for team, source, recipient in (("A", 0, 4), ("B", 5, 9)):
        column = METRIC_COLUMNS_BY_NAME[
            f"agent_{source}_to_agent_{recipient}_healing_to_hunter_trap_recipient_contribution_fraction"
        ]
        assert (
            column.label == f"Share of Team {team} Healing to Target With Hunter Trap"
        )
        assert column.numerator and "this target" in column.numerator
        assert column.denominator and f"Team {team}" in column.denominator
    for emitter in ("mage", "warrior"):
        coverage = METRIC_COLUMNS_BY_NAME[f"agent_0_{emitter}_aura_coverage"]
        assert "out of range" not in coverage.description
        assert coverage.denominator and "out of range" in coverage.denominator
    wait = METRIC_COLUMNS_BY_NAME["team_a_mean_observed_respawn_wait_steps"]
    assert wait.numerator == (
        "Ticks each agent on Team A spent dead during the recording, added together"
    )
    actions = METRIC_COLUMNS_BY_NAME["team_b_action_acceptance_rate"]
    assert actions.denominator == "All actions sent by agents on Team B"
    focus = METRIC_COLUMNS_BY_NAME["team_a_focus_fire_steps"]
    assert focus.label == "Multi-Attacker Ticks"
    assert "not chances they might have had to attack together" in focus.description
    solo = METRIC_COLUMNS_BY_NAME["team_a_single_contributor_kills"]
    assert "exactly one agent helping" in solo.description
    assert "healing on the kill tick counts as help" in solo.description
    assert "at least partly useful" in solo.description
    for team, slots in (("a", range(5)), ("b", range(5, 10))):
        denominator = f"All kills by Team {team.upper()}"
        basic = METRIC_COLUMNS_BY_NAME[f"team_{team}_basic_kill_fraction"]
        ultimate = METRIC_COLUMNS_BY_NAME[f"team_{team}_ultimate_kill_fraction"]
        assert basic.denominator == ultimate.denominator == denominator
        assert basic.unit == "fraction"
        assert basic.required_class_id is None
        for class_id, class_name in enumerate(
            ("mage", "warrior", "hunter", "rogue", "priest"), 1
        ):
            count = METRIC_COLUMNS_BY_NAME[f"team_{team}_{class_name}_basic_kills"]
            fraction = METRIC_COLUMNS_BY_NAME[
                f"team_{team}_{class_name}_basic_kill_fraction"
            ]
            assert count.scope == fraction.scope == "team"
            assert count.required_class_id == fraction.required_class_id == class_id
            assert count.direction == "higher"
            assert fraction.direction == "descriptive"
            assert count.unit == "count" and fraction.unit == "fraction"
            assert count.denominator is None
            assert fraction.denominator == denominator
            assert "is zero" in fraction.missing_when
        for slot in slots:
            basic = METRIC_COLUMNS_BY_NAME[f"agent_{slot}_basic_kill_participation"]
            ultimate = METRIC_COLUMNS_BY_NAME[
                f"agent_{slot}_ultimate_kill_participation"
            ]
            assert basic.denominator == ultimate.denominator == denominator
            assert basic.required_class_id is None
            assert "is zero" in basic.missing_when
    formation = METRIC_COLUMNS_BY_NAME["team_a_ally_distance_observations"]
    assert formation.label == "Ally-Pair Distance Measurements"
    assert "Three living teammates give three measurements" in formation.description
    freedom = METRIC_COLUMNS_BY_NAME["agent_0_freedom_eligible_steps"]
    assert "stood still or was not slowed" in freedom.description
    assert "Team A" in METRIC_COLUMNS_BY_NAME["score_difference"].description
    assert "final" not in METRIC_COLUMNS_BY_NAME["score_difference"].description
    poison = METRIC_COLUMNS_BY_NAME["agent_0_healing_prevented_by_poison"]
    assert poison.label == "Priest Healing Prevented On This Agent"
    assert "before the maximum-health limit" in poison.description
    assert "at the start of the tick" in poison.description
    assert "Does not include regeneration" in poison.description


def test_ultimate_views_reuse_columns_without_inventing_mixed_class_team_totals() -> (
    None
):
    classes = (1, 2, 3, 4, 5, 1, 1, 5, 0, 0)

    def groups(name: str) -> tuple[str, ...]:
        return metric_groups(METRIC_COLUMNS_BY_NAME[name], classes)

    assert "overview" not in METRIC_FAMILIES
    assert set(ULTIMATE_GROUP_BY_CLASS.values()) == {
        "ultimate_mage",
        "ultimate_warrior",
        "ultimate_hunter",
        "ultimate_rogue",
        "ultimate_priest",
    }
    assert groups("agent_1_to_agent_6_ultimate_damage_done") == (
        "recipient_damage",
        "ultimate_warrior",
    )
    assert groups("team_a_ultimate_damage_done") == ("damage_done", "recipient_damage")
    assert groups("team_a_ultimate_kills") == ("kill_contributions",)
    for team in ("a", "b"):
        for stem in (
            "single_contributor_kills",
            "single_contributor_kill_fraction",
            "multi_contributor_kills",
            "multi_contributor_kill_fraction",
        ):
            assert groups(f"team_{team}_{stem}") == (
                "coordination",
                "kill_contributions",
            )
        assert groups(f"team_{team}_basic_kill_fraction") == ("kill_contributions",)
        for class_name in ("mage", "warrior", "hunter", "rogue", "priest"):
            for stem in ("kills", "kill_fraction"):
                assert groups(f"team_{team}_{class_name}_basic_{stem}") == (
                    "kill_contributions",
                )
    assert groups("agent_1_basic_kill_participation") == ("kill_contributions",)
    assert groups("team_a_ultimate_healing_done") == (
        "healing_done",
        "recipient_healing",
        "ultimate_priest",
    )
    assert groups("team_a_ultimate_rescues") == ("priest_rescue", "ultimate_priest")
    for team in ("a", "b"):
        assert f"team_{team}_priest_ultimate_activations" not in METRIC_COLUMNS_BY_NAME
        assert "abilities" in groups(f"team_{team}_ultimate_activations")
    assert groups("agent_0_ultimate_healing_done") == (
        "healing_done",
        "recipient_healing",
    )
    assert groups("agent_0_ultimate_damage_done") == (
        "damage_done",
        "recipient_damage",
    )
    assert groups("team_b_burst_damage") == ("ultimate_mage",)
    assert groups("agent_1_burst_damage") == ("ultimate_mage",)
    assert groups("agent_1_rogue_poison_slow_active_steps") == (
        "status_active_steps",
        "ultimate_rogue",
    )
    assert groups("agent_1_healing_prevented_by_poison") == ("ultimate_rogue",)
    assert groups("agent_1_trap_break_contributions") == ("ultimate_hunter",)
    assert groups("agent_2_hunter_basic_slow_applications") == ("status_applications",)
    assert groups("agent_4_priest_freedom_applications") == ("status_applications",)
    assert groups("agent_1_mage_burst_applications") == ("status_applications",)
    assert groups("agent_0_mage_burst_applications") == ("status_applications",)
    assert groups("team_a_mage_burst_applications") == (
        "status_applications",
        "ultimate_mage",
    )
    assert groups("agent_3_ultimate_priest_healing_received") == (
        "healing_received",
        "recipient_healing",
        "ultimate_priest",
    )
    for name in (
        "agent_0_ultimate_damage_done",
        "agent_0_to_agent_5_ultimate_damage_done",
        "agent_0_ultimate_kill_contributions",
        "agent_0_ultimate_healing_done",
    ):
        assert "ultimate_mage" not in groups(name)
    for name in (
        "agent_4_ultimate_damage_done",
        "agent_4_to_agent_5_ultimate_damage_done",
    ):
        assert "ultimate_priest" not in groups(name)
    for group, names in {
        "recipient_healing": (
            "team_a_healing_done",
            "team_a_basic_healing_done",
            "team_a_ultimate_healing_done",
            "team_a_effective_healing_done",
            "team_a_excess_healing",
            "agent_4_healing_done",
            "agent_4_basic_healing_done",
            "agent_4_ultimate_healing_done",
            "agent_4_excess_healing",
            "agent_0_priest_healing_received",
            "agent_0_basic_priest_healing_received",
            "agent_0_ultimate_priest_healing_received",
            "agent_0_effective_priest_healing_received",
            "agent_0_excess_healing_received",
        ),
        "recipient_damage": (
            "team_a_damage_done",
            "agent_0_damage_done",
            "agent_5_damage_received",
            "agent_5_basic_damage_received",
            "agent_5_ultimate_damage_received",
        ),
        "kill_contributions": ("team_a_kills", "agent_5_deaths"),
        "deaths": ("team_a_deaths", "team_b_deaths"),
    }.items():
        for name in names:
            assert group in groups(name)
    for name, order in (
        ("agent_1_ultimate_activations", 0),
        ("team_a_warrior_charge_applications", 0),
        ("agent_1_to_agent_5_ultimate_application_fraction", 0),
        ("agent_1_ultimate_damage_done", 1),
        ("team_a_burst_damage", 1),
        ("agent_4_ultimate_effective_healing_fraction", 1),
        ("agent_1_ultimate_kill_contributions", 2),
        ("team_a_burst_kills", 2),
        ("team_a_ultimate_rescues", 2),
        ("agent_5_hunter_trap_active_steps", 3),
        ("agent_5_healing_prevented_by_poison", 3),
        ("team_a_trap_breaks", 3),
    ):
        assert metric_view_order(METRIC_COLUMNS_BY_NAME[name]) == order
    other_team_priest = (1, 2, 3, 4, 0, 1, 2, 3, 4, 5)
    for stem in (
        "ultimate_healing_done",
        "ultimate_effective_healing_done",
        "ultimate_effective_healing_fraction",
        "ultimate_excess_healing_fraction",
        "ultimate_rescues",
        "ultimate_rescue_fraction",
    ):
        own = METRIC_COLUMNS_BY_NAME[f"team_a_{stem}"]
        opponent = METRIC_COLUMNS_BY_NAME[f"team_b_{stem}"]
        assert "ultimate_priest" not in metric_groups(own, other_team_priest)
        assert "ultimate_priest" in metric_groups(opponent, other_team_priest)
        assert "ultimate_priest" not in metric_groups(own, (0,) * 10)
        assert own.family in metric_groups(own, other_team_priest)
    for stem in (
        "ultimate_priest_healing_received",
        "ultimate_effective_priest_healing_received",
        "ultimate_excess_healing_received",
    ):
        own = METRIC_COLUMNS_BY_NAME[f"agent_0_{stem}"]
        opponent = METRIC_COLUMNS_BY_NAME[f"agent_5_{stem}"]
        assert "ultimate_priest" not in metric_groups(own, other_team_priest)
        assert "ultimate_priest" in metric_groups(opponent, other_team_priest)
        assert own.family in metric_groups(own, other_team_priest)
    # A killed enemy's shared Burst denominator belongs to the opposing Mage
    # team. A Mage on the victim's own team does not establish that capability.
    for victim in (0, 5):
        column = METRIC_COLUMNS_BY_NAME[f"agent_{victim}_deaths"]
        own_team_mage = (1,) * 5 + (5,) * 5 if victim == 0 else (5,) * 5 + (1,) * 5
        opposing_mage = own_team_mage[5:] + own_team_mage[:5]
        assert "ultimate_mage" not in metric_groups(column, own_team_mage)
        assert "ultimate_mage" in metric_groups(column, opposing_mage)
        assert "ultimate_mage" not in metric_groups(column, (0,) * 10)
        assert "deaths" in metric_groups(column, own_team_mage)
    assert METRIC_COLUMNS_BY_NAME["team_a_ultimate_rescues"].subject_role == "team"
    for class_id, class_name in ((2, "warrior"), (3, "hunter"), (4, "rogue")):
        column = METRIC_COLUMNS_BY_NAME[f"team_a_{class_name}_ultimate_kills"]
        group = f"ultimate_{class_name}"
        assert group not in metric_groups(column, (0,) * 5 + (class_id,) * 5)
        assert group in metric_groups(column, (class_id,) * 5 + (0,) * 5)
    # An initial status is evidence owned by its recipient; no current caster
    # is required to inspect it. The same applies to Trap breaks and Poison.
    without_casters = (5,) * 10
    for name, group in (
        ("team_a_warrior_charge_slow_active_steps", "ultimate_warrior"),
        ("agent_0_hunter_trap_active_steps", "ultimate_hunter"),
        ("team_a_rogue_poison_anti_heal_active_steps", "ultimate_rogue"),
        ("agent_0_mage_burst_active_steps", "ultimate_mage"),
        ("team_a_trap_breaks", "ultimate_hunter"),
        ("agent_0_healing_prevented_by_poison", "ultimate_rogue"),
    ):
        assert group in metric_groups(METRIC_COLUMNS_BY_NAME[name], without_casters)

    canonical = (1, 2, 3, 4, 5) * 2
    for topic in ULTIMATE_GROUP_BY_CLASS.values():
        for team in ("a", "b"):
            assert (
                (topic, "totals")
                in metric_locations(
                    METRIC_COLUMNS_BY_NAME[f"team_{team}_kills"], canonical
                )
            ) == (topic != "ultimate_mage")
        for slot in range(10):
            assert (topic, "recipients") in metric_locations(
                METRIC_COLUMNS_BY_NAME[f"agent_{slot}_deaths"], canonical
            )
        totals = {
            c.name
            for c in METRIC_COLUMNS
            if (topic, "totals") in metric_locations(c, canonical)
        }
        recipients = {
            c.name
            for c in METRIC_COLUMNS
            if (topic, "recipients") in metric_locations(c, canonical)
        }
        assert totals.isdisjoint(recipients)
    for name, view in (
        ("team_a_rescues", "totals"),
        ("agent_0_rescues", "recipients"),
        ("agent_0_rescue_opportunities", "recipients"),
    ):
        assert ("ultimate_priest", view) in metric_locations(
            METRIC_COLUMNS_BY_NAME[name], canonical
        )
    # A Priest-only team cannot help with kills, but it still has ordinary
    # Episode Results. Its victim's class must not create a helper on that team.
    priest_only_team = (5,) * 5 + (1,) * 5
    for name in ("team_a_kills", "agent_5_deaths"):
        assert "ultimate_priest" not in metric_groups(
            METRIC_COLUMNS_BY_NAME[name], priest_only_team
        )

    for topic, stem, ability in (
        ("ultimate_mage", "mage_burst_applications", "Mage Burst"),
        ("ultimate_warrior", "warrior_charge_applications", "Warrior Charge"),
        ("ultimate_hunter", "hunter_trap_applications", "Hunter Trap"),
        ("ultimate_rogue", "rogue_poison_applications", "Rogue Poison"),
    ):
        column = METRIC_COLUMNS_BY_NAME[f"team_b_{stem}"]
        text = metric_topic_text(column, topic, canonical)
        assert text.get("label", column.label) == f"{ability} Applications"
        assert text["subtitle"] == ""
        assert "Team B" in text["description"]
        assert ability in text["description"]
        if column.status_channel is not None:
            assert (
                column.label == STATUS_LABELS[column.status_channel] + " Applications"
            )
        assert metric_topic_text(column, "status_applications", canonical) == {}
    for team in ("a", "b"):
        for class_id, stem, topic, ability in (
            (2, "warrior_charge_applications", "ultimate_warrior", "Warrior Charge"),
            (4, "rogue_poison_applications", "ultimate_rogue", "Rogue Poison"),
            (
                5,
                "priest_holy_word_salvation_applications",
                "ultimate_priest",
                "Priest Salvation",
            ),
        ):
            column = METRIC_COLUMNS_BY_NAME[f"team_{team}_{stem}"]
            assert column.scope == "team" and column.family == "abilities"
            assert (
                column.required_class_id == class_id and column.status_channel is None
            )
            assert column.numerator is None and column.denominator is None
            assert metric_primary_location(column) == (topic, "totals")
            assert metric_locations(column, canonical) == ((topic, "totals"),)
            text = metric_topic_text(column, topic, canonical)
            assert text.get("label", column.label) == f"{ability} Applications"
            assert text["subtitle"] == ""
            assert "Each activation counts once" in text["description"]
        for stem in (
            "warrior_charge_slow",
            "warrior_charge_stun",
            "rogue_poison_slow",
            "rogue_poison_stun",
            "rogue_poison_anti_heal",
        ):
            column = METRIC_COLUMNS_BY_NAME[f"team_{team}_{stem}_applications"]
            assert column.status_channel is not None
            assert (
                column.label == STATUS_LABELS[column.status_channel] + " Applications"
            )
            assert metric_locations(column, canonical) == (
                ("status_applications", "totals"),
            )
            assert metric_topic_text(column, "status_applications", canonical) == {}
    permuted = (5, 3, 1, 4, 2, 2, 5, 4, 1, 3)
    for slot, topic, ability in (
        (0, "ultimate_priest", "Priest Salvation"),
        (1, "ultimate_hunter", "Hunter Trap"),
        (2, "ultimate_mage", "Mage Burst"),
        (3, "ultimate_rogue", "Rogue Poison"),
        (4, "ultimate_warrior", "Warrior Charge"),
    ):
        activation = METRIC_COLUMNS_BY_NAME[f"agent_{slot}_ultimate_activations"]
        text = metric_topic_text(activation, topic, permuted)
        assert ability in text["description"]
        assert text["label"] == f"{ability} Applications"
        assert set(text) <= {"label", "description", "subtitle"}
        if topic == "ultimate_mage":
            assert "itself" in text["description"]
            continue
        contribution = METRIC_COLUMNS_BY_NAME[
            f"agent_{slot}_to_agent_5_ultimate_kill_contributions"
        ]
        text = metric_topic_text(contribution, topic, permuted)
        assert ability in text["description"]
        assert "Mage" not in text["description"]
        if topic == "ultimate_priest":
            assert "useful" in text["description"]
            assert (
                "an ally who damages that enemy on its death tick"
                in text["description"]
            )
            assert "Entirely excess healing does not count" in text["description"]
        else:
            assert "damage" in text["description"]


def test_approved_directed_inventory_is_complete_without_cross_product_expansion() -> (
    None
):
    expected = {
        "basic_applications",
        "ultimate_applications",
        "burst_damage",
        "burst_damage_contributing_to_kill",
        "kill_contributions",
        "basic_kill_contributions",
        "ultimate_kill_contributions",
        "burst_kill_contributions",
        "solo_kills",
        "rescue_contributions",
        "basic_rescue_contributions",
        "ultimate_rescue_contributions",
        "trap_break_contributions",
        *(
            ability + effect
            for ability in ("", "basic_", "ultimate_")
            for effect in (
                "damage_done",
                "healing_done",
                "effective_healing_done",
                "excess_healing",
            )
        ),
        *(status + "_applications" for status in STATUS_NAMES),
        *(
            f"{effect}_to_{status}_recipient"
            for status in STATUS_NAMES[:7]
            for effect in ("damage", "healing")
        ),
        *(f"kill_contributions_to_{status}_recipient" for status in STATUS_NAMES[:7]),
        *(
            f"{emitter}_aura_{quantity}_steps"
            for emitter in ("mage", "warrior")
            for quantity in ("covered", "eligible")
        ),
    }
    assert {metric.key for metric in DIRECTED_METRICS} == expected
    assert len(DIRECTED_METRICS) == len(expected) == 59
    for name, count in (("all", 100), ("ally", 50), ("enemy", 50), ("self", 10)):
        pairs = RECIPIENT_PAIRS_BY_RELATION[name]
        assert len(pairs) == len(set(pairs)) == count
    assert RECIPIENT_PAIRS_BY_RELATION["self"] == tuple(
        (slot, slot) for slot in range(10)
    )


def test_every_directed_amount_has_its_declared_counts_and_distinct_fractions() -> None:
    for metric in DIRECTED_METRICS:
        for source, recipient in RECIPIENT_PAIRS_BY_RELATION[metric.relation]:
            prefix = f"agent_{source}_to_agent_{recipient}_"
            removed_self = source == recipient and (
                metric.key == "mage_burst_applications"
                or metric.key
                in (
                    "healing_to_warrior_charge_stun_recipient",
                    "healing_to_hunter_trap_recipient",
                    "healing_to_rogue_poison_stun_recipient",
                    "mage_aura_eligible_steps",
                    "warrior_aura_eligible_steps",
                )
            )
            if removed_self:
                assert prefix + metric.amount not in METRIC_COLUMNS_BY_NAME
                for fraction in (metric.allocation, metric.contribution):
                    if fraction:
                        assert prefix + fraction not in METRIC_COLUMNS_BY_NAME
                continue
            amount = METRIC_COLUMNS_BY_NAME[prefix + metric.amount]
            assert amount.stem == metric.key
            assert amount.subjects == (source, recipient)
            assert amount.subject_role == metric.subject_role
            if metric.allocation is None:
                assert metric.key.endswith("aura_eligible_steps")
                continue
            assert metric.contribution is not None
            allocation = METRIC_COLUMNS_BY_NAME[prefix + metric.allocation]
            if source == recipient and metric.family == "aura_coverage":
                assert prefix + metric.contribution not in METRIC_COLUMNS_BY_NAME
                assert allocation.denominator and allocation.numerator
                continue
            contribution = METRIC_COLUMNS_BY_NAME[prefix + metric.contribution]
            assert allocation.numerator and contribution.numerator
            assert allocation.denominator
            assert contribution.denominator
            assert allocation.name != contribution.name
            assert allocation.denominator != contribution.denominator


def test_pair_healing_efficiency_is_distinct_from_allocation_and_contribution() -> None:
    for ability in ("", "basic_", "ultimate_"):
        for portion, direction in (
            ("effective", "higher" if ability == "ultimate_" else "descriptive"),
            ("excess", "lower" if ability == "ultimate_" else "descriptive"),
        ):
            stem = f"{ability}{portion}_healing_fraction"
            pair = METRIC_COLUMNS_BY_NAME[f"agent_4_to_agent_0_{stem}"]
            assert pair.scope == "source_recipient"
            assert pair.direction == direction
            assert (
                pair.denominator
                == f"All {ability.replace('_', ' ').title()}healing this agent "
                "gave this target, including excess"
            )
            assert METRIC_COLUMNS_BY_NAME[f"agent_4_{stem}"].scope == "agent"
            assert METRIC_COLUMNS_BY_NAME[f"team_a_{stem}"].scope == "team"
    old_efficiency = METRIC_COLUMNS_BY_NAME["agent_4_effective_healing_fraction"]
    assert old_efficiency.denominator == (
        "All Priest healing this agent provided, including excess"
    )
    allocation = METRIC_COLUMNS_BY_NAME[
        "agent_4_to_agent_0_effective_healing_done_allocation_fraction"
    ]
    contribution = METRIC_COLUMNS_BY_NAME[
        "agent_4_to_agent_0_effective_healing_done_contribution_fraction"
    ]
    assert allocation.denominator is not None
    assert contribution.denominator is not None
    assert "this agent" in allocation.denominator
    assert "Team A" in contribution.denominator
    assert "this target" in contribution.denominator


def test_team_recipient_identity_and_source_class_never_become_patient_class() -> None:
    for name, subject, class_id in (
        ("team_a_to_agent_8_ultimate_applications", (1, 8), None),
        ("team_b_to_agent_2_warrior_charge_stun_applications", (2, 2), 2),
    ):
        column = METRIC_COLUMNS_BY_NAME[name]
        assert column.scope == "team_recipient"
        assert column.subjects == subject
        assert column.required_class_id == class_id
        assert column.requires_ultimate_target
    for name in (
        "agent_0_basic_priest_healing_received",
        "agent_3_ultimate_priest_healing_received",
        "agent_2_rescues",
        "agent_8_burst_damage_received",
        "agent_3_mage_aura_covered_recipient_steps",
    ):
        column = METRIC_COLUMNS_BY_NAME[name]
        assert column.subject_role == "recipient"
        assert column.required_class_id is None
    assert METRIC_COLUMNS_BY_NAME[
        "agent_4_to_agent_0_basic_applications"
    ].requires_basic_target
    assert not METRIC_COLUMNS_BY_NAME[
        "agent_4_to_agent_0_basic_applications"
    ].requires_ultimate_target


def test_dropdown_inventory_has_twenty_nine_distinct_groups() -> None:
    assert set(METRIC_GROUPS) == {
        "priority",
        "abilities",
        "deaths",
        "kill_contributions",
        "damage_done",
        "recipient_damage",
        "healing_done",
        "recipient_healing",
        "damage_received",
        "healing_received",
        "excess_healing",
        "controlled_damage",
        "controlled_healing",
        "controlled_kills",
        "coordination",
        "action_acceptance",
        "status_applications",
        "status_active_steps",
        "respawn",
        "aura_coverage",
        "aura_benefits",
        "priest_rescue",
        "freedom",
        "formation",
        "ultimate_mage",
        "ultimate_warrior",
        "ultimate_hunter",
        "ultimate_rogue",
        "ultimate_priest",
    }
    classes = (1, 2, 3, 4, 5) * 2
    for column in METRIC_COLUMNS:
        groups = metric_groups(column, classes)
        assert groups and len(groups) == len(set(groups))
        assert set(groups) <= set(METRIC_GROUPS)
        for group in groups:
            text = metric_topic_text(column, group, classes)
            if group in ULTIMATE_GROUP_BY_CLASS.values():
                assert set(text) <= {
                    "label",
                    "description",
                    "subtitle",
                    "numerator",
                    "denominator",
                    "guidance",
                    "missing_when",
                }
                assert all("Ultimate" not in value for value in text.values())
            else:
                assert set(text) <= {"description", "subtitle"}
    for slot, status, ultimate in (
        (0, "mage_burst", "ultimate_mage"),
        (1, "warrior_charge_stun", "ultimate_warrior"),
        (3, "rogue_poison_anti_heal", "ultimate_rogue"),
    ):
        assert ultimate in metric_groups(
            METRIC_COLUMNS_BY_NAME[f"agent_{slot}_ultimate_activations"], classes
        )
        assert ultimate not in metric_groups(
            METRIC_COLUMNS_BY_NAME[f"agent_{slot}_{status}_applications"], classes
        )
    # Repeated classes can put otherwise separated rows in the same Ultimate
    # view. Even structurally hidden rows must have unambiguous measure labels.
    for roster in (classes, *((class_id,) * 10 for class_id in range(1, 6))):
        signatures = [
            (
                group,
                column.scope,
                column.subjects,
                metric_topic_text(column, group, roster).get("label", column.label),
                column.status_channel,
                column.unit,
            )
            for column in METRIC_COLUMNS
            for group in metric_groups(column, roster)
        ]
        assert len(signatures) == len(set(signatures))
    solo = METRIC_COLUMNS_BY_NAME["agent_0_to_agent_5_solo_kill_participation"]
    assert "Solo" in solo.label
    for aura in ("mage", "warrior"):
        coverage = METRIC_COLUMNS_BY_NAME[
            f"agent_0_to_agent_1_{aura}_aura_covered_steps_contribution_fraction"
        ]
        assert aura.title() in coverage.label
        assert "Death" not in coverage.label
    assert (
        METRIC_COLUMNS_BY_NAME["team_a_burst_kills"].label
        == "Kills Helped During Burst"
    )
    assert (
        METRIC_COLUMNS_BY_NAME["team_a_burst_kill_contributions"].label
        == "Kill Contributions During Burst"
    )


def test_scope_order_separates_totals_from_recipient_detail() -> None:
    topics = [topic.name for topic in METRIC_TOPICS]
    rank = {
        "episode": 0,
        "team": 0,
        "agent": 1,
        "team_recipient": 2,
        "source_recipient": 3,
        "ally_pair": 4,
    }
    keys: list[tuple[int, bool, int, tuple[int, ...]]] = []
    for column in METRIC_COLUMNS[26:]:
        topic, view = metric_primary_location(column)
        section = (
            2
            if column.scope == "agent" and column.subject_role == "recipient"
            else rank[column.scope]
        )
        keys.append(
            (topics.index(topic), view == "recipients", section, column.subjects)
        )
    assert keys == sorted(keys)
    for column in METRIC_COLUMNS:
        assert column.subject_role in {
            "source",
            "recipient",
            "contributor",
            "emitter",
            "team",
            "episode",
            "ally_pair",
        }
        assert column.stem and column.name.endswith(column.stem)
        assert bool(column.numerator) == bool(column.denominator)
        if column.unit == "fraction":
            assert column.numerator and column.denominator

    def stems(family: str, scope: str, subjects: tuple[int, ...]) -> list[str]:
        # Shared rows follow table order even when their primary CSV homes differ.
        topic = {"recipient_healing": "healing_done"}.get(family, family)
        columns = [
            column
            for column in METRIC_COLUMNS
            if column.family == family
            and column.scope == scope
            and column.subjects == subjects
        ]
        return [
            column.stem
            for column in sorted(columns, key=lambda c: metric_order_key(c, topic))
        ]

    for team in (1, 2):
        coordination = stems("coordination", "team", (team,))
        for kind in ("single", "multi"):
            assert coordination.index(f"{kind}_contributor_kill_fraction") == (
                coordination.index(f"{kind}_contributor_kills") + 1
            )
        kills = stems("kill_contributions", "team", (team,))
        assert kills.index("basic_kill_fraction") == kills.index("basic_kills") + 1
        start = kills.index("basic_kills")
        assert kills[start : start + 12] == [
            "basic_kills",
            "basic_kill_fraction",
            *(
                f"{class_name}_basic_{stem}"
                for class_name in ("mage", "warrior", "hunter", "rogue", "priest")
                for stem in ("kills", "kill_fraction")
            ),
        ]
    for slot in range(10):
        kills = stems("kill_contributions", "agent", (slot,))
        assert kills.index("basic_kill_participation") == (
            kills.index("basic_kill_contributions") + 1
        )

    statuses = [
        "hunter_basic_slow",
        "priest_freedom",
        "mage_burst",
        "warrior_charge_slow",
        "warrior_charge_stun",
        "hunter_trap",
        "rogue_poison_slow",
        "rogue_poison_stun",
        "rogue_poison_anti_heal",
    ]
    assert stems("status_applications", "agent", (0,)) == [
        status + "_applications" for status in statuses
    ]
    for family in ("controlled_damage", "controlled_healing", "controlled_kills"):
        channels = list(
            dict.fromkeys(
                column.status_channel
                for column in METRIC_COLUMNS
                if column.family == family
                and column.scope == "source_recipient"
                and column.subjects
                == ((0, 1) if family == "controlled_healing" else (0, 5))
            )
        )
        assert channels == [1, 0, 3, 4, 2, 5, 6]
    assert stems("aura_coverage", "source_recipient", (0, 1)) == [
        f"{emitter}_aura_{measure}"
        for emitter in ("mage", "warrior")
        for measure in (
            "covered_steps",
            "eligible_steps",
            "coverage",
            "covered_steps_allocation_fraction",
            "covered_steps_contribution_fraction",
        )
    ]
    assert stems("healing_received", "agent", (0,)) == [
        "priest_healing_received",
        "priest_healing_received_fraction",
        "effective_priest_healing_received",
        "effective_priest_healing_received_fraction",
        "excess_priest_healing_received_fraction",
        "healing_received",
        "healing_received_fraction",
        "regenerated_healing",
        "regeneration_healing_received_fraction",
        "effective_healing_received",
        "basic_priest_healing_received",
        "basic_effective_priest_healing_received",
        "ultimate_priest_healing_received",
        "ultimate_effective_priest_healing_received",
    ]
    healing = stems("recipient_healing", "source_recipient", (0, 1))
    assert healing == [
        prefix + stem
        for prefix in ("", "basic_", "ultimate_")
        for stem in (
            "healing_done",
            "healing_done_allocation_fraction",
            "healing_done_contribution_fraction" if prefix else "healing_done_fraction",
            "effective_healing_done",
            "effective_healing_done_allocation_fraction",
            "effective_healing_done_contribution_fraction",
            "effective_healing_fraction",
            "excess_healing",
            "excess_healing_allocation_fraction",
            "excess_healing_contribution_fraction",
            "excess_healing_fraction",
        )
    ]


def test_generated_dictionary_and_manuscript_summary_match_catalog() -> None:
    from pathlib import Path

    from scripts.dev.export_metric_dictionary import (
        dictionary_csv,
        family_summary_markdown,
    )

    assert Path("docs/evaluation/metric_columns.csv").read_text() == dictionary_csv()
    specification = Path("docs/evaluation/metric_specification.md").read_text()
    assert family_summary_markdown() in specification
    assert "**11,158**" in family_summary_markdown()


def test_topics_match_the_approved_researcher_questions() -> None:
    expected = (
        ("priority", "Episode Results", False),
        ("abilities", "Ability Activations", True),
        ("action_acceptance", "Accepted and Rejected Actions", False),
        ("damage_done", "Damage Done", True),
        ("damage_received", "Damage Received", False),
        ("controlled_damage", "Damage to Enemies With Harmful Effects", True),
        ("healing_done", "Healing Done", True),
        ("healing_received", "Healing Received", False),
        ("excess_healing", "Excess Healing", True),
        ("controlled_healing", "Healing to Allies With Harmful Effects", True),
        ("priest_rescue", "Priest Healing Saves", True),
        ("kill_contributions", "Kill Contributions", True),
        ("controlled_kills", "Kills of Enemies With Harmful Effects", True),
        ("deaths", "Deaths and Time Dead", False),
        ("respawn", "Respawning", False),
        ("coordination", "Team Coordination", False),
        ("formation", "Team Formation", False),
        ("aura_coverage", "Aura Coverage", True),
        ("aura_benefits", "Damage Added or Blocked by Auras", False),
        ("status_applications", "Status Applications", True),
        ("status_active_steps", "Time With Status Effects", False),
        ("freedom", "Freedom Against Slows", False),
        ("ultimate_mage", "Burst (Mage Ultimate)", True),
        ("ultimate_warrior", "Charge (Warrior Ultimate)", True),
        ("ultimate_hunter", "Freezing Trap (Hunter Ultimate)", True),
        ("ultimate_rogue", "Crippling Poison (Rogue Ultimate)", True),
        ("ultimate_priest", "Holy Word: Salvation (Priest Ultimate)", True),
    )
    assert tuple((t.name, t.label, t.paired) for t in METRIC_TOPICS) == expected
    assert Counter(t.section for t in METRIC_TOPICS) == {
        "Results and Actions": 3,
        "Damage and Healing": 8,
        "Kills, Deaths and Respawning": 4,
        "Teamwork and Positioning": 4,
        "Status Effects": 3,
        "Ultimates": 5,
    }
    for classes in ((1, 2, 3, 4, 5) * 2, (5,) * 10, (0, 1, 1, 5, 0, 2, 0, 2, 0, 5)):
        for column in METRIC_COLUMNS:
            locations = metric_locations(column, classes)
            # A topic contains each measurement in exactly one of its tables.
            assert len({topic for topic, _ in locations}) == len(locations)
            assert metric_primary_location(column) in locations
            assert all(topic in {t[0] for t in expected} for topic, _ in locations)


def test_primary_homes_and_shared_rows_keep_sources_and_recipients_distinct() -> None:
    expected = {
        "agent_0_ultimate_damage_done": ("damage_done", "totals"),
        "agent_0_to_agent_5_ultimate_damage_done": ("damage_done", "recipients"),
        "agent_0_excess_healing_received": ("excess_healing", "recipients"),
        "agent_0_excess_priest_healing_received_fraction": (
            "excess_healing",
            "recipients",
        ),
        "team_a_excess_priest_healing_received_fraction": (
            "healing_received",
            "single",
        ),
        "agent_0_to_agent_1_excess_healing": ("excess_healing", "recipients"),
        "team_a_single_contributor_kill_fraction": ("kill_contributions", "totals"),
        "team_a_multi_contributor_kills": ("kill_contributions", "totals"),
        "team_a_to_agent_5_basic_applications": ("abilities", "recipients"),
        "agent_0_mage_burst_active_steps": ("status_active_steps", "single"),
        "agent_0_healing_prevented_by_poison": ("ultimate_rogue", "recipients"),
    }
    for name, location in expected.items():
        column = METRIC_COLUMNS_BY_NAME[name]
        assert metric_primary_location(column) == location
    for column in METRIC_COLUMNS:
        if column.subject_role == "recipient":
            locations = metric_locations(column, (1, 2, 3, 4, 5) * 2)
            assert ("healing_done", "totals") not in locations
            assert ("excess_healing", "totals") not in locations
    for slot in range(10):
        amount = METRIC_COLUMNS_BY_NAME[f"agent_{slot}_excess_healing_received"]
        fraction = METRIC_COLUMNS_BY_NAME[
            f"agent_{slot}_excess_priest_healing_received_fraction"
        ]
        for topic, view in (
            ("healing_done", "recipients"),
            ("healing_received", "single"),
            ("excess_healing", "recipients"),
        ):
            rows = sorted(
                (
                    c
                    for c in METRIC_COLUMNS
                    if c.scope == "agent"
                    and c.subjects == (slot,)
                    and (topic, view) in metric_locations(c, (1, 2, 3, 4, 5) * 2)
                ),
                key=lambda c: metric_order_key(c, topic),
            )
            assert rows.index(fraction) == rows.index(amount) + 1
