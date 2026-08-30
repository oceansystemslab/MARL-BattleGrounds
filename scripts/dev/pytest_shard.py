"""Deterministically shard pytest by weighted test-work-unit affinity for CI.

Load this module with ``pytest -p scripts.dev.pytest_shard`` and pass an exact
``--ci-shard=N/M`` selector. Ordinary test files remain on one worker. A small,
strictly validated cost profile may extract known slow function families while
keeping every parameterized family indivisible.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import pytest
from _pytest.config import Config
from _pytest.config.argparsing import Parser
from _pytest.nodes import Item

type TestFamilyKey = tuple[str, str]


def _empty_string_costs() -> dict[str, int]:
    return {}


def _empty_reserved_costs() -> dict[int, tuple[int, ...]]:
    return {}


@dataclass(frozen=True)
class ShardCostProfile:
    """Declare measured CI costs without changing test ownership semantics."""

    file_cost_overrides: Mapping[str, int] = field(default_factory=_empty_string_costs)
    extracted_family_costs: Mapping[str, int] = field(
        default_factory=_empty_string_costs
    )
    residual_file_costs: Mapping[str, int] = field(default_factory=_empty_string_costs)
    reserved_costs_by_shard_count: Mapping[int, tuple[int, ...]] = field(
        default_factory=_empty_reserved_costs
    )
    strict: bool = False


@dataclass(frozen=True)
class TestWorkUnit:
    """One indivisible, deterministically identified CI scheduling unit."""

    identifier: str
    families: tuple[TestFamilyKey, ...]
    cost: int


CI_SHARD_COST_PROFILE = ShardCostProfile(
    file_cost_overrides={
        "tests/test_visual_debugger_replay_service.py": 400,
        "tests/test_visual_debugger_sample_replays.py": 420,
    },
    extracted_family_costs={
        (
            "tests/test_visual_debugger_scenarios.py::"
            "test_every_authoritative_visual_mechanic_has_regular_and_stress_evidence"
        ): 300,
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
    },
    residual_file_costs={
        "tests/test_visual_debugger_scenarios.py": 160,
        "tests/test_visual_debugger_service.py": 200,
    },
    reserved_costs_by_shard_count={12: (0,) * 11 + (50,)},
    strict=True,
)


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


def logical_path_from_family(family: TestFamilyKey) -> str:
    """Return the repository-relative pytest path encoded in a family node ID."""
    logical_path = family[1].split("::", maxsplit=1)[0]
    if not logical_path:
        raise ValueError("test family node ID must contain a logical test path")
    return logical_path


def _positive_costs(values: Mapping[str, int], label: str) -> None:
    if any(type(value) is not int or value < 1 for value in values.values()):
        raise ValueError(f"{label} must contain positive integer costs")


def _reserved_costs(profile: ShardCostProfile, shard_count: int) -> tuple[int, ...]:
    reserved = profile.reserved_costs_by_shard_count.get(shard_count)
    if reserved is None:
        return (0,) * shard_count
    if len(reserved) != shard_count:
        raise ValueError("reserved shard costs must match their shard count")
    if any(type(cost) is not int or cost < 0 for cost in reserved):
        raise ValueError("reserved shard costs must be nonnegative integers")
    return reserved


def build_test_work_units(
    item_counts: Mapping[TestFamilyKey, int],
    cost_profile: ShardCostProfile | None = None,
) -> tuple[TestWorkUnit, ...]:
    """Build ordinary file units plus explicitly extracted slow families."""
    if any(count < 1 for count in item_counts.values()):
        raise ValueError("every test family must contain at least one item")
    profile = cost_profile or ShardCostProfile()
    _positive_costs(profile.file_cost_overrides, "file cost overrides")
    _positive_costs(profile.extracted_family_costs, "family cost overrides")
    _positive_costs(profile.residual_file_costs, "residual file cost overrides")

    families_by_path: dict[str, list[TestFamilyKey]] = {}
    family_by_nodeid: dict[str, TestFamilyKey] = {}
    for family in item_counts:
        logical_path = logical_path_from_family(family)
        families_by_path.setdefault(logical_path, []).append(family)
        nodeid = family[1]
        if nodeid in family_by_nodeid:
            raise ValueError(f"duplicate logical test family: {nodeid}")
        family_by_nodeid[nodeid] = family

    configured_files = set(profile.file_cost_overrides)
    residual_files = set(profile.residual_file_costs)
    extracted_nodeids = set(profile.extracted_family_costs)
    extracted_paths = {
        nodeid.split("::", maxsplit=1)[0] for nodeid in extracted_nodeids
    }
    if profile.strict:
        missing_files = (configured_files | residual_files) - families_by_path.keys()
        if missing_files:
            raise ValueError(
                f"stale CI shard file cost profile: {sorted(missing_files)}"
            )
        missing_families = extracted_nodeids - family_by_nodeid.keys()
        if missing_families:
            raise ValueError(
                f"stale CI shard family cost profile: {sorted(missing_families)}"
            )
        if residual_files != extracted_paths:
            raise ValueError(
                "every split test file must have exactly one residual cost override"
            )
        indivisible_conflicts = configured_files & extracted_paths
        if indivisible_conflicts:
            raise ValueError(
                "indivisible file cost overrides cannot extract test families: "
                f"{sorted(indivisible_conflicts)}"
            )

    units: list[TestWorkUnit] = []
    for logical_path, raw_families in sorted(families_by_path.items()):
        families = tuple(sorted(raw_families))
        extracted = tuple(
            family for family in families if family[1] in extracted_nodeids
        )
        residual = tuple(family for family in families if family not in extracted)
        for family in extracted:
            units.append(
                TestWorkUnit(
                    identifier=f"family:{family[1]}",
                    families=(family,),
                    cost=profile.extracted_family_costs[family[1]],
                )
            )
        if not residual:
            continue
        collected_items = sum(item_counts[family] for family in residual)
        if logical_path in profile.file_cost_overrides:
            cost = profile.file_cost_overrides[logical_path]
            identifier = f"file:{logical_path}"
        elif extracted:
            cost = profile.residual_file_costs.get(logical_path, collected_items)
            identifier = f"residual:{logical_path}"
        else:
            cost = collected_items
            identifier = f"file:{logical_path}"
        units.append(
            TestWorkUnit(
                identifier=identifier,
                families=residual,
                cost=cost,
            )
        )
    return tuple(sorted(units, key=lambda unit: unit.identifier))


def assign_test_families(
    item_counts: Mapping[TestFamilyKey, int],
    shard_count: int,
    *,
    cost_profile: ShardCostProfile | None = None,
) -> tuple[tuple[TestFamilyKey, ...], ...]:
    """LPT-pack deterministic, indivisible test work units by measured cost."""
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    profile = cost_profile or ShardCostProfile()
    units = build_test_work_units(item_counts, profile)
    if len(units) < shard_count:
        raise ValueError("shard_count cannot exceed the number of test work units")

    loads = list(_reserved_costs(profile, shard_count))
    assignments: list[list[TestFamilyKey]] = [[] for _ in range(shard_count)]
    for unit in sorted(units, key=lambda row: (-row.cost, row.identifier)):
        shard_index = min(range(shard_count), key=lambda index: (loads[index], index))
        assignments[shard_index].extend(unit.families)
        loads[shard_index] += unit.cost
    if any(not families for families in assignments):
        raise ValueError("weighted CI shard assignment produced an empty shard")
    return tuple(tuple(sorted(families)) for families in assignments)


def pytest_addoption(parser: Parser) -> None:
    """Register the opt-in CI shard selector."""
    group = parser.getgroup("CI sharding")
    group.addoption(
        "--ci-shard",
        metavar="N/M",
        help="Run one deterministic weighted work-unit shard of the collected tests.",
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
        assignments = assign_test_families(
            Counter(families),
            shard_count,
            cost_profile=CI_SHARD_COST_PROFILE,
        )
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


def shard_costs(
    assignments: Sequence[Sequence[TestFamilyKey]],
    item_counts: Mapping[TestFamilyKey, int],
    *,
    cost_profile: ShardCostProfile | None = None,
) -> tuple[int, ...]:
    """Return scheduling costs, including any reserved non-pytest work."""
    profile = cost_profile or ShardCostProfile()
    units = build_test_work_units(item_counts, profile)
    ownership_counts = Counter(
        family for families in assignments for family in families
    )
    if set(ownership_counts) != set(item_counts) or any(
        count != 1 for count in ownership_counts.values()
    ):
        raise ValueError("assignments must own every test family exactly once")
    owner_by_family = {
        family: shard_index
        for shard_index, families in enumerate(assignments)
        for family in families
    }
    costs = list(_reserved_costs(profile, len(assignments)))
    for unit in units:
        owners = {owner_by_family[family] for family in unit.families}
        if len(owners) != 1:
            raise ValueError("one test work unit cannot span multiple shards")
        costs[owners.pop()] += unit.cost
    return tuple(costs)
