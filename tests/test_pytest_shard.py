"""Focused tests for deterministic test-file-affinity CI sharding."""

from collections import Counter

import pytest
from scripts.dev.pytest_shard import (
    assign_test_families,
    family_key_from_metadata,
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


def test_independent_families_in_one_file_share_one_shard() -> None:
    path = "tests/test_alpha.py"
    dominant = (path, f"{path}::test_dominant")
    independent = (path, f"{path}::test_independent")
    other = ("tests/test_beta.py", "tests/test_beta.py::test_other")

    assignments = assign_test_families({dominant: 100, independent: 1, other: 1}, 2)

    assert sum(dominant in shard and independent in shard for shard in assignments) == 1
    assert sum(other in shard for shard in assignments) == 1


def test_assignment_rejects_more_shards_than_test_files() -> None:
    counts = {
        ("tests/test_alpha.py", f"tests/test_alpha.py::test_{index}"): 1
        for index in range(9)
    }

    with pytest.raises(
        ValueError,
        match="shard_count cannot exceed the number of test files",
    ):
        assign_test_families(counts, 10)


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
