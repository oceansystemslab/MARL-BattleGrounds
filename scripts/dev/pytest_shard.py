"""Deterministically shard pytest by test file for CI.

Load this module with ``pytest -p scripts.dev.pytest_shard`` and pass an exact
``--ci-shard=N/M`` selector.  Tests from one file stay together so module
fixtures and JAX compilation are not repeated across workers.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

import pytest
from _pytest.config import Config
from _pytest.config.argparsing import Parser
from _pytest.nodes import Item


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


def assign_test_files(
    item_counts: Mapping[str, int], shard_count: int
) -> tuple[tuple[str, ...], ...]:
    """Assign whole files with deterministic longest-processing-time packing."""
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    if any(count < 1 for count in item_counts.values()):
        raise ValueError("every test file must contain at least one item")

    loads = [0] * shard_count
    assignments: list[list[str]] = [[] for _ in range(shard_count)]
    for path, count in sorted(item_counts.items(), key=lambda row: (-row[1], row[0])):
        shard_index = min(range(shard_count), key=lambda index: (loads[index], index))
        assignments[shard_index].append(path)
        loads[shard_index] += count
    return tuple(tuple(sorted(paths)) for paths in assignments)


def pytest_addoption(parser: Parser) -> None:
    """Register the opt-in CI shard selector."""
    group = parser.getgroup("CI sharding")
    group.addoption(
        "--ci-shard",
        metavar="N/M",
        help="Run one deterministic file-level shard of the collected tests.",
    )


def pytest_collection_modifyitems(config: Config, items: list[Item]) -> None:
    """Deselect files owned by other CI shards after normal collection."""
    raw_spec = config.getoption("ci_shard")
    if raw_spec is None:
        return
    if not isinstance(raw_spec, str):
        raise pytest.UsageError("--ci-shard must use the form N/M")

    shard_index, shard_count = parse_shard_spec(raw_spec)
    paths = tuple(item.path.as_posix() for item in items)
    assignments = assign_test_files(Counter(paths), shard_count)
    selected_paths = set(assignments[shard_index])
    selected = [
        item for item, path in zip(items, paths, strict=True) if path in selected_paths
    ]
    deselected = [
        item
        for item, path in zip(items, paths, strict=True)
        if path not in selected_paths
    ]
    config.hook.pytest_deselected(items=deselected)
    items[:] = selected


def shard_loads(
    assignments: Sequence[Sequence[str]], item_counts: Mapping[str, int]
) -> tuple[int, ...]:
    """Return item totals for diagnostics and focused unit tests."""
    return tuple(sum(item_counts[path] for path in paths) for paths in assignments)
