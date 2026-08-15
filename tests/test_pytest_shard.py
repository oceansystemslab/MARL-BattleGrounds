"""Focused tests for deterministic file-level CI sharding."""

import pytest
from scripts.dev.pytest_shard import (
    assign_test_files,
    parse_shard_spec,
    shard_loads,
)


def test_parse_shard_spec_uses_one_based_cli_and_zero_based_internal_index() -> None:
    assert parse_shard_spec("1/3") == (0, 3)
    assert parse_shard_spec("3/3") == (2, 3)


@pytest.mark.parametrize("value", ("0/3", "4/3", "1/0", "1", "x/3", "1/x"))
def test_parse_shard_spec_rejects_invalid_selectors(value: str) -> None:
    with pytest.raises(pytest.UsageError):
        parse_shard_spec(value)


def test_file_assignment_is_deterministic_exact_and_balanced() -> None:
    counts = {
        "tests/test_alpha.py": 8,
        "tests/test_beta.py": 7,
        "tests/test_gamma.py": 6,
        "tests/test_delta.py": 5,
        "tests/test_epsilon.py": 4,
        "tests/test_zeta.py": 3,
    }

    first = assign_test_files(counts, 3)
    second = assign_test_files(dict(reversed(tuple(counts.items()))), 3)

    assert first == second
    flattened = tuple(path for shard in first for path in shard)
    assert len(flattened) == len(set(flattened))
    assert set(flattened) == set(counts)
    loads = shard_loads(first, counts)
    assert max(loads) - min(loads) <= max(counts.values())


def test_file_assignment_keeps_each_file_atomic() -> None:
    assignments = assign_test_files({"a.py": 100, "b.py": 1, "c.py": 1}, 3)
    assert sum("a.py" in shard for shard in assignments) == 1
