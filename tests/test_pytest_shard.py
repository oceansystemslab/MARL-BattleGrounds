"""Focused tests for deterministic weighted CI test-work-unit sharding."""

from collections import Counter

import pytest
from scripts.dev.pytest_shard import (
    CI_SHARD_COST_PROFILE,
    ShardCostProfile,
    TestFamilyKey,
    assign_test_families,
    build_test_work_units,
    family_key_from_metadata,
    parse_shard_spec,
    shard_costs,
    shard_loads,
)


def test_parse_shard_spec_uses_one_based_cli_and_zero_based_internal_index() -> None:
    assert parse_shard_spec("1/12") == (0, 12)
    assert parse_shard_spec("12/12") == (11, 12)


@pytest.mark.parametrize("value", ("0/12", "13/12", "1/0", "1", "x/12", "1/x"))
def test_parse_shard_spec_rejects_invalid_selectors(value: str) -> None:
    with pytest.raises(pytest.UsageError):
        parse_shard_spec(value)


def test_family_key_uses_unparameterized_pytest_metadata() -> None:
    path = "tests/groups[test]/test_alpha.py"

    assert family_key_from_metadata(
        path=path,
        parent_nodeid="tests/groups[test]/test_alpha.py::TestAlpha",
        item_name="test_value[case-a]",
        original_name="test_value",
    ) == (path, "tests/groups[test]/test_alpha.py::TestAlpha::test_value")
    assert family_key_from_metadata(
        path=path,
        parent_nodeid="tests/groups[test]/test_alpha.py::TestAlpha",
        item_name="test_value[open[::case]",
        original_name="test_value",
    ) == (path, "tests/groups[test]/test_alpha.py::TestAlpha::test_value")
    assert family_key_from_metadata(
        path=path,
        parent_nodeid="tests/groups[test]/test_alpha.py::TestAlpha",
        item_name="test_value[close]::case]",
        original_name="test_value",
    ) == (path, "tests/groups[test]/test_alpha.py::TestAlpha::test_value")
    assert family_key_from_metadata(
        path=path,
        parent_nodeid="tests/groups[test]/test_alpha.py::TestAlpha",
        item_name="test_value",
        original_name=None,
    ) == (path, "tests/groups[test]/test_alpha.py::TestAlpha::test_value")


def test_family_assignment_is_deterministic_exact_and_balanced() -> None:
    counts = {
        ("tests/test_alpha.py", "tests/test_alpha.py::test_a"): 8,
        ("tests/test_alpha.py", "tests/test_alpha.py::test_b"): 7,
        ("tests/test_beta.py", "tests/test_beta.py::test_c"): 6,
        ("tests/test_beta.py", "tests/test_beta.py::test_d"): 5,
        ("tests/test_gamma.py", "tests/test_gamma.py::test_e"): 4,
        ("tests/test_gamma.py", "tests/test_gamma.py::test_f"): 3,
    }

    first = assign_test_families(counts, 3)
    second = assign_test_families(dict(reversed(tuple(counts.items()))), 3)

    assert first == second
    flattened = tuple(family for shard in first for family in shard)
    assert len(flattened) == len(set(flattened))
    assert set(flattened) == set(counts)
    loads = shard_loads(first, counts)
    assert max(loads) - min(loads) <= max(counts.values())


def test_parameterized_instances_form_one_atomic_family() -> None:
    path = "tests/test_alpha.py"
    nodeids = tuple(
        f"{path}::test_parameterized[{parameter_id}]"
        for parameter_id in ("a", "b", "c")
    )
    family = family_key_from_metadata(
        path=path,
        parent_nodeid=path,
        item_name="test_parameterized[a]",
        original_name="test_parameterized",
    )

    assert all(
        family_key_from_metadata(
            path=path,
            parent_nodeid=path,
            item_name=nodeid.rsplit("::", 1)[-1],
            original_name="test_parameterized",
        )
        == family
        for nodeid in nodeids
    )
    assignments = assign_test_families(
        {
            family: len(nodeids),
            (
                "tests/test_beta.py",
                "tests/test_beta.py::test_independent",
            ): 1,
        },
        2,
    )
    assert sum(family in shard for shard in assignments) == 1


def test_ordinary_independent_families_in_one_file_share_one_shard() -> None:
    path = "tests/test_alpha.py"
    dominant = (path, f"{path}::test_dominant")
    independent = (path, f"{path}::test_independent")
    other = ("tests/test_beta.py", "tests/test_beta.py::test_other")

    assignments = assign_test_families({dominant: 100, independent: 1, other: 1}, 2)

    assert sum(dominant in shard and independent in shard for shard in assignments) == 1
    assert sum(other in shard for shard in assignments) == 1


def test_assignment_rejects_more_shards_than_test_work_units() -> None:
    counts = {
        ("tests/test_alpha.py", f"tests/test_alpha.py::test_{index}"): 1
        for index in range(9)
    }

    with pytest.raises(
        ValueError,
        match="shard_count cannot exceed the number of test work units",
    ):
        assign_test_families(counts, 10)


def test_declared_slow_family_is_atomic_and_residual_file_stays_together() -> None:
    path = "tests/test_slow.py"
    extracted = (path, f"{path}::test_slow_family")
    residual_a = (path, f"{path}::test_residual_a")
    residual_b = (path, f"{path}::test_residual_b")
    ordinary = ("tests/test_ordinary.py", "tests/test_ordinary.py::test_value")
    counts: dict[TestFamilyKey, int] = {
        extracted: 5,
        residual_a: 2,
        residual_b: 3,
        ordinary: 1,
    }
    profile = ShardCostProfile(
        extracted_family_costs={extracted[1]: 100},
        residual_file_costs={path: 40},
        strict=True,
    )

    assignments = assign_test_families(counts, 3, cost_profile=profile)

    assert sum(extracted in shard for shard in assignments) == 1
    assert (
        sum(residual_a in shard and residual_b in shard for shard in assignments) == 1
    )
    assert (
        sum(
            extracted in shard and (residual_a in shard or residual_b in shard)
            for shard in assignments
        )
        == 0
    )


def test_indivisible_measured_file_never_splits_between_shards() -> None:
    path = "tests/test_expensive_fixture.py"
    first = (path, f"{path}::test_first")
    second = (path, f"{path}::test_second")
    other = ("tests/test_other.py", "tests/test_other.py::test_other")
    profile = ShardCostProfile(
        file_cost_overrides={path: 400},
        strict=True,
    )

    assignments = assign_test_families(
        {first: 10, second: 10, other: 1},
        2,
        cost_profile=profile,
    )

    assert sum(first in shard and second in shard for shard in assignments) == 1


def test_weighted_assignment_is_stable_and_reserves_static_gate_capacity() -> None:
    counts = {
        (f"tests/test_{index}.py", f"tests/test_{index}.py::test_value"): cost
        for index, cost in enumerate((100, 90, 80, 70, 60, 50, 40, 30))
    }
    profile = ShardCostProfile(
        reserved_costs_by_shard_count={2: (0, 50)},
    )

    first = assign_test_families(counts, 2, cost_profile=profile)
    second = assign_test_families(
        dict(reversed(tuple(counts.items()))),
        2,
        cost_profile=profile,
    )

    assert first == second
    assert shard_loads(first, counts)[1] < shard_loads(first, counts)[0]
    assert shard_costs(first, counts, cost_profile=profile) == (280, 290)
    duplicated = (first[0] + (first[1][0],), first[1])
    with pytest.raises(
        ValueError,
        match="assignments must own every test family exactly once",
    ):
        shard_costs(duplicated, counts, cost_profile=profile)


def test_weighted_assignment_rejects_an_empty_reserved_shard() -> None:
    counts = {
        (f"tests/test_{index}.py", f"tests/test_{index}.py::test_value"): 1
        for index in range(3)
    }
    profile = ShardCostProfile(
        reserved_costs_by_shard_count={3: (0, 0, 1_000)},
    )

    with pytest.raises(
        ValueError,
        match="weighted CI shard assignment produced an empty shard",
    ):
        assign_test_families(counts, 3, cost_profile=profile)


@pytest.mark.parametrize(
    ("profile", "message"),
    (
        (
            ShardCostProfile(
                file_cost_overrides={"tests/test_missing.py": 10},
                strict=True,
            ),
            "stale CI shard file cost profile",
        ),
        (
            ShardCostProfile(
                extracted_family_costs={
                    "tests/test_present.py::test_missing": 10,
                },
                residual_file_costs={"tests/test_present.py": 5},
                strict=True,
            ),
            "stale CI shard family cost profile",
        ),
        (
            ShardCostProfile(
                residual_file_costs={"tests/test_present.py": 5},
                strict=True,
            ),
            "every split test file must have exactly one residual cost override",
        ),
    ),
)
def test_strict_cost_profile_rejects_stale_entries(
    profile: ShardCostProfile,
    message: str,
) -> None:
    present = (
        "tests/test_present.py",
        "tests/test_present.py::test_present",
    )

    with pytest.raises(ValueError, match=message):
        build_test_work_units({present: 1}, profile)


def test_production_profile_names_and_weights_exactly_four_extracted_families() -> None:
    assert CI_SHARD_COST_PROFILE.file_cost_overrides == {
        "tests/test_visual_debugger_replay_service.py": 400,
        "tests/test_visual_debugger_sample_replays.py": 420,
    }
    assert CI_SHARD_COST_PROFILE.extracted_family_costs == {
        (
            "tests/test_visual_debugger_scenarios.py::"
            "test_every_authoritative_visual_mechanic_has_regular_and_stress_evidence"
        ): 500,
        (
            "tests/test_visual_debugger_scenarios.py::"
            "test_every_registered_scripted_command_matches_authored_acceptance"
        ): 70,
        (
            "tests/test_visual_debugger_scenarios.py::"
            "test_researcher_scenarios_cover_every_canonical_event_kind"
        ): 50,
        (
            "tests/test_visual_debugger_service.py::"
            "test_every_scripted_scenario_preflights_each_successor_in_both_views"
        ): 500,
    }
    assert CI_SHARD_COST_PROFILE.residual_file_costs == {
        "tests/test_visual_debugger_scenarios.py": 160,
        "tests/test_visual_debugger_service.py": 200,
    }
    assert CI_SHARD_COST_PROFILE.reserved_costs_by_shard_count[12] == (
        (0,) * 11 + (50,)
    )


def test_dominant_family_weights_keep_atomic_families_on_separate_shards() -> None:
    semantic_path = "tests/test_semantic_inventory.py"
    preflight_path = "tests/test_hosted_preflight.py"
    semantic = (semantic_path, f"{semantic_path}::test_every_mechanic")
    preflight = (preflight_path, f"{preflight_path}::test_every_case")
    semantic_residual = (semantic_path, f"{semantic_path}::test_residual")
    preflight_residual = (preflight_path, f"{preflight_path}::test_residual")
    ordinary = {
        (f"tests/test_{index}.py", f"tests/test_{index}.py::test_value"): 100
        for index in range(10)
    }
    counts: dict[TestFamilyKey, int] = {
        semantic: 1,
        preflight: 30,
        semantic_residual: 5,
        preflight_residual: 5,
        **ordinary,
    }
    profile = ShardCostProfile(
        extracted_family_costs={semantic[1]: 500, preflight[1]: 500},
        residual_file_costs={semantic_path: 100, preflight_path: 100},
        strict=True,
    )

    assignments = assign_test_families(counts, 12, cost_profile=profile)

    semantic_owner = next(shard for shard in assignments if semantic in shard)
    preflight_owner = next(shard for shard in assignments if preflight in shard)
    assert semantic_owner == (semantic,)
    assert preflight_owner == (preflight,)
    flattened = tuple(family for shard in assignments for family in shard)
    assert set(flattened) == set(counts)
    assert len(flattened) == len(set(flattened))
    assert all(assignments)


def test_collected_items_have_one_owner_across_twelve_nonempty_shards() -> None:
    nodeids = tuple(
        f"tests/test_{index}.py::TestGroup[class-{class_index}]::test_value[case-{case}]"
        for index in range(12)
        for class_index in range(2)
        for case in range(1 + (index % 3))
    )
    families = tuple(
        family_key_from_metadata(
            path=f"tests/test_{index}.py",
            parent_nodeid=f"tests/test_{index}.py::TestGroup[class-{class_index}]",
            item_name=f"test_value[case-{case}]",
            original_name="test_value",
        )
        for index in range(12)
        for class_index in range(2)
        for case in range(1 + (index % 3))
    )
    counts = Counter(families)

    assignments = assign_test_families(counts, 12)
    owned_item_indexes = tuple(
        tuple(index for index, family in enumerate(families) if family in set(shard))
        for shard in assignments
    )
    flattened = tuple(index for shard in owned_item_indexes for index in shard)

    assert all(owned_item_indexes)
    assert len(flattened) == len(set(flattened))
    assert sorted(flattened) == list(range(len(nodeids)))
    for path in {family[0] for family in families}:
        assert (
            sum(any(family[0] == path for family in shard) for shard in assignments)
            == 1
        )
