"""Focused tests for deterministic test-family CI sharding."""

from collections import Counter

import pytest
from scripts.dev.pytest_shard import (
    assign_test_families,
    family_key_from_item,
    parse_shard_spec,
    shard_loads,
)


def test_parse_shard_spec_uses_one_based_cli_and_zero_based_internal_index() -> None:
    assert parse_shard_spec("1/12") == (0, 12)
    assert parse_shard_spec("12/12") == (11, 12)


@pytest.mark.parametrize("value", ("0/12", "13/12", "1/0", "1", "x/12", "1/x"))
def test_parse_shard_spec_rejects_invalid_selectors(value: str) -> None:
    with pytest.raises(pytest.UsageError):
        parse_shard_spec(value)


def test_family_key_removes_only_the_terminal_parameter_suffix() -> None:
    path = "tests/groups[test]/test_alpha.py"

    assert family_key_from_item(
        path,
        "tests/groups[test]/test_alpha.py::TestAlpha::test_value[case-a]",
    ) == (path, "tests/groups[test]/test_alpha.py::TestAlpha::test_value")
    assert family_key_from_item(
        path,
        "tests/groups[test]/test_alpha.py::TestAlpha[param]::test_value[case[a]]",
    ) == (path, "tests/groups[test]/test_alpha.py::TestAlpha::test_value")
    assert family_key_from_item(
        path,
        "tests/groups[test]/test_alpha.py::TestAlpha[class::a]::test_value[case::a]",
    ) == (path, "tests/groups[test]/test_alpha.py::TestAlpha::test_value")
    assert family_key_from_item(
        path,
        "tests/groups[test]/test_alpha.py::TestAlpha::test_value",
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
    family = family_key_from_item(path, nodeids[0])

    assert all(family_key_from_item(path, nodeid) == family for nodeid in nodeids)
    assignments = assign_test_families(
        {
            family: len(nodeids),
            (path, f"{path}::test_independent"): 1,
        },
        2,
    )
    assert sum(family in shard for shard in assignments) == 1


def test_independent_families_in_one_file_can_use_different_shards() -> None:
    path = "tests/test_alpha.py"
    dominant = (path, f"{path}::test_dominant")
    independent = (path, f"{path}::test_independent")

    assignments = assign_test_families({dominant: 100, independent: 1}, 2)

    assert assignments == ((dominant,), (independent,))


def test_assignment_rejects_more_shards_than_test_families() -> None:
    counts = {
        ("tests/test_alpha.py", f"tests/test_alpha.py::test_{index}"): 1
        for index in range(11)
    }

    with pytest.raises(
        ValueError,
        match="shard_count cannot exceed the number of test families",
    ):
        assign_test_families(counts, 12)


def test_collected_items_have_one_owner_across_twelve_nonempty_shards() -> None:
    path = "tests/test_inventory.py"
    nodeids = tuple(
        f"{path}::TestGroup[class-{class_index}]::test_{index}[case-{case}]"
        for index in range(12)
        for class_index in range(2)
        for case in range(1 + (index % 3))
    )
    families = tuple(family_key_from_item(path, nodeid) for nodeid in nodeids)
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
