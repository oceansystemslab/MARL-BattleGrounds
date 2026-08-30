"""Deterministically shard pytest by test-file affinity for CI.

Load this module with ``pytest -p scripts.dev.pytest_shard`` and pass an exact
``--ci-shard=N/M`` selector. Every test file has one worker, and every
parameterized instance of one test therefore remains together as well.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

import pytest
from _pytest.config import Config
from _pytest.config.argparsing import Parser
from _pytest.nodes import Item

type TestFamilyKey = tuple[str, str]


def parse_shard_spec(value: str) -> tuple[int, int]:
    """Return a zero-based shard index and positive shard count."""
    try:
        raw_index, raw_count = value.split("/", maxsplit=1)
        index = int(raw_index)
        count = int(raw_count)
    except (TypeError, ValueError) as error:
        raise pytest.UsageError("--ci-shard must use the form N/M") from error
    if count < 1 or index < 1 or index > count:
        raise pytest.UsageError("--ci-shard requires 1 <= N <= M")
    return index - 1, count


def family_key_from_metadata(
    *,
    path: str,
    parent_nodeid: str,
    item_name: str,
    original_name: str | None,
) -> TestFamilyKey:
    """Return one family key from pytest's unparameterized item metadata."""
    family_name = (
        original_name if isinstance(original_name, str) and original_name else item_name
    )
    return path, f"{parent_nodeid}::{family_name}"


def family_key_from_item(item: Item) -> TestFamilyKey:
    """Return one family key without parsing arbitrary parameter-ID text."""
    parent = item.parent
    if parent is None:
        raise pytest.UsageError("CI sharding requires every item to have a collector.")
    return family_key_from_metadata(
        path=item.path.as_posix(),
        parent_nodeid=parent.nodeid,
        item_name=item.name,
        original_name=getattr(item, "originalname", None),
    )


def assign_test_families(
    item_counts: Mapping[TestFamilyKey, int], shard_count: int
) -> tuple[tuple[TestFamilyKey, ...], ...]:
    """LPT-pack whole test files by their collected item counts."""
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    if any(count < 1 for count in item_counts.values()):
        raise ValueError("every test family must contain at least one item")
    families_by_path: dict[str, list[TestFamilyKey]] = {}
    file_loads: Counter[str] = Counter()
    for family, count in item_counts.items():
        path = family[0]
        families_by_path.setdefault(path, []).append(family)
        file_loads[path] += count
    if len(families_by_path) < shard_count:
        raise ValueError("shard_count cannot exceed the number of test files")

    loads = [0] * shard_count
    assignments: list[list[TestFamilyKey]] = [[] for _ in range(shard_count)]
    for path, count in sorted(file_loads.items(), key=lambda row: (-row[1], row[0])):
        shard_index = min(range(shard_count), key=lambda index: (loads[index], index))
        assignments[shard_index].extend(families_by_path[path])
        loads[shard_index] += count
    return tuple(tuple(sorted(families)) for families in assignments)


def pytest_addoption(parser: Parser) -> None:
    """Register the opt-in CI shard selector."""
    group = parser.getgroup("CI sharding")
    group.addoption(
        "--ci-shard",
        metavar="N/M",
        help="Run one deterministic test-file-affinity shard of the collected tests.",
    )


def pytest_collection_modifyitems(config: Config, items: list[Item]) -> None:
    """Deselect test families owned by other CI shards after collection."""
    raw_spec = config.getoption("ci_shard")
    if raw_spec is None:
        return
    if not isinstance(raw_spec, str):
        raise pytest.UsageError("--ci-shard must use the form N/M")

    shard_index, shard_count = parse_shard_spec(raw_spec)
    families = tuple(family_key_from_item(item) for item in items)
    try:
        assignments = assign_test_families(Counter(families), shard_count)
    except ValueError as error:
        raise pytest.UsageError(str(error)) from error
    selected_families = set(assignments[shard_index])
    selected = [
        item
        for item, family in zip(items, families, strict=True)
        if family in selected_families
    ]
    deselected = [
        item
        for item, family in zip(items, families, strict=True)
        if family not in selected_families
    ]
    config.hook.pytest_deselected(items=deselected)
    items[:] = selected


def shard_loads(
    assignments: Sequence[Sequence[TestFamilyKey]],
    item_counts: Mapping[TestFamilyKey, int],
) -> tuple[int, ...]:
    """Return item totals for diagnostics and focused unit tests."""
    return tuple(sum(item_counts[path] for path in paths) for paths in assignments)
