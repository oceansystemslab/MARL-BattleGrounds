"""Deterministically shard pytest by weighted test-work-unit affinity for CI.

Load this module with ``pytest -p scripts.dev.pytest_shard`` and pass an exact
``--ci-shard=N/M`` selector. Ordinary test files remain on one worker. A small,
strictly validated cost profile may extract known slow function families or
split a declared hotspot at function-family boundaries while keeping every
parameterized family indivisible.
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
type ModuleFixtureKey = tuple[str, str]
type TestWorkUnitRelocation = tuple[str, int, int]


def _empty_string_costs() -> dict[str, int]:
    """Create an independent empty cost map for a new shard profile."""
    return {}


def _empty_reserved_costs() -> dict[int, tuple[int, ...]]:
    """Create an empty map of reserved costs keyed by shard count."""
    return {}


def _empty_relocations() -> dict[int, tuple[TestWorkUnitRelocation, ...]]:
    """Create an empty map of explicitly requested test-unit relocations."""
    return {}


def _empty_module_fixture_keys() -> frozenset[ModuleFixtureKey]:
    """Start with no declared module-fixture dependencies."""
    return frozenset()


@dataclass(frozen=True)
class ShardCostProfile:
    """Declare measured CI costs without changing test ownership semantics."""

    file_cost_overrides: Mapping[str, int] = field(default_factory=_empty_string_costs)
    split_file_family_cost_floors: Mapping[str, int] = field(
        default_factory=_empty_string_costs
    )
    extracted_family_costs: Mapping[str, int] = field(
        default_factory=_empty_string_costs
    )
    residual_file_costs: Mapping[str, int] = field(default_factory=_empty_string_costs)
    reserved_costs_by_shard_count: Mapping[int, tuple[int, ...]] = field(
        default_factory=_empty_reserved_costs
    )
    relocations_by_shard_count: Mapping[int, tuple[TestWorkUnitRelocation, ...]] = (
        field(default_factory=_empty_relocations)
    )
    repeatable_module_fixtures: frozenset[ModuleFixtureKey] = field(
        default_factory=_empty_module_fixture_keys
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
        "tests/test_shared_obs_runtime.py": 110,
        # The current 20-case scenario suite measured 104.322 s on CPU.
        "tests/test_tdm_scenarios.py": 105,
        "tests/test_visual_debugger_replay_service.py": 400,
        # Packet 3's 38-case collection suite measured 286.39 s on CPU.
        "tests/test_training_collection.py": 287,
        # Packet 4 focused CPU suites: 192.54 s, 130.82 s and 218.72 s.
        "tests/test_training_learner.py": 193,
        "tests/test_training_checkpoints.py": 131,
        "tests/test_training_runner.py": 219,
    },
    extracted_family_costs={
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
        (
            "tests/test_visual_debugger_sample_replays.py::"
            "test_checked_samples_match_fresh_cpu_generation_scientific_truth"
        ): 400,
    },
    residual_file_costs={
        "tests/test_visual_debugger_scenarios.py": 160,
        "tests/test_visual_debugger_service.py": 200,
        "tests/test_visual_debugger_sample_replays.py": 500,
    },
    reserved_costs_by_shard_count={12: (0,) * 11 + (50,)},
    # Keep measured twelve-worker destinations while refreshing source owners
    # from actual collection after test additions (M9-M10 Packet 4, 2026-09-19).
    # Source IDs are owners after weighted packing, before these intact moves.
    # Keep the existing split/fixture rules and each unit's collection order.
    relocations_by_shard_count={
        12: (
            ("residual:tests/test_visual_debugger_service.py", 8, 4),
            ("file:tests/test_shared_obs_runtime.py", 8, 3),
            ("file:tests/test_combat_ultimate_effects.py", 4, 5),
            ("file:tests/test_tdm_scenarios.py", 12, 8),
            ("file:tests/test_canonical_tournament.py", 11, 7),
        )
    },
    strict=True,
)


def parse_shard_spec(value: str) -> tuple[int, int]:
    """Parse a one-based CLI selector into a zero-based shard index and count.

    Parameters
    ----------
    value : str
        One-based shard selector in N/M form, such as 3/12.

    Returns
    -------
    tuple of int
        Zero-based selected index and positive shard count.

    Raises
    ------
    pytest.UsageError
        The selector is malformed or does not satisfy 1 <= N <= M.
    """
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
    """Build a stable family identity without parsing parameter labels.

    Parameters
    ----------
    path : str
        Pytest item file path used as the family's physical file identity.
    parent_nodeid : str
        Node ID of the item's collector, before the function name.
    item_name : str
        Collected item name, used when original_name is unavailable.
    original_name : str | None
        Unparameterized function name, or None. A nonempty value keeps all parameter
        cases in one family.

    Returns
    -------
    TestFamilyKey
        Physical path and collector-qualified unparameterized function name.
    """
    family_name = (
        original_name if isinstance(original_name, str) and original_name else item_name
    )
    return path, f"{parent_nodeid}::{family_name}"


def family_key_from_item(item: Item) -> TestFamilyKey:
    """Find an item's unparameterized scheduling family.

    Parameters
    ----------
    item : Item
        Collected pytest item whose collector and resolved fixture metadata are
        inspected.

    Returns
    -------
    TestFamilyKey
        Family key shared by every parameter case of this function.

    Raises
    ------
    pytest.UsageError
        The item has no parent collector.
    """
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
    """Read the repository-relative path from a family node ID.

    Parameters
    ----------
    family : TestFamilyKey
        Pair of physical path and unparameterized family node ID.

    Returns
    -------
    str
        Node ID prefix before the first double colon.

    Raises
    ------
    ValueError
        The node ID contains no logical path.
    """
    logical_path = family[1].split("::", maxsplit=1)[0]
    if not logical_path:
        raise ValueError("test family node ID must contain a logical test path")
    return logical_path


def module_fixture_keys_from_item(item: Item) -> frozenset[ModuleFixtureKey]:
    """Find module-scoped fixtures in the item's resolved dependency closure.

    Parameters
    ----------
    item : Item
        Collected pytest item whose collector and resolved fixture metadata are
        inspected.

    Returns
    -------
    frozenset of ModuleFixtureKey
        Fixture owner node IDs paired with fixture names.

    Notes
    -----
    This reads pytest collection metadata; it does not execute fixtures.
    """
    fixture_info = getattr(item, "_fixtureinfo", None)
    fixture_defs_by_name = getattr(fixture_info, "name2fixturedefs", {})
    fixture_keys: set[ModuleFixtureKey] = set()
    for fixture_name, fixture_defs in fixture_defs_by_name.items():
        for fixture_def in fixture_defs or ():
            if fixture_def.scope == "module":
                fixture_keys.add((fixture_def.baseid, fixture_name))
    return frozenset(fixture_keys)


def dynamic_fixture_request_sites_from_item(item: Item) -> frozenset[str]:
    """Find non-pytest code that can request fixtures dynamically.

    Parameters
    ----------
    item : Item
        Collected pytest item whose collector and resolved fixture metadata are
        inspected.

    Returns
    -------
    frozenset of str
        Item or fixture call-site identities using the request fixture.

    Notes
    -----
    This conservative inspection reads metadata and does not run fixture code.
    """
    fixture_info = getattr(item, "_fixtureinfo", None)
    request_sites: set[str] = set()
    if "request" in getattr(fixture_info, "argnames", ()):
        request_sites.add(item.nodeid)
    fixture_defs_by_name = getattr(fixture_info, "name2fixturedefs", {})
    for fixture_defs in fixture_defs_by_name.values():
        for fixture_def in fixture_defs or ():
            fixture_module = getattr(fixture_def.func, "__module__", "")
            is_pytest_builtin = fixture_module == "pytest" or fixture_module.startswith(
                "_pytest."
            )
            if "request" in fixture_def.argnames and not is_pytest_builtin:
                request_sites.add(f"{fixture_def.baseid}::{fixture_def.argname}")
    return frozenset(request_sites)


def validate_split_fixture_affinity(
    module_fixtures_by_family: Mapping[TestFamilyKey, frozenset[ModuleFixtureKey]],
    cost_profile: ShardCostProfile,
) -> None:
    """Reject unsafe module-fixture sharing across split scheduling units.

    Parameters
    ----------
    module_fixtures_by_family : Mapping[TestFamilyKey, frozenset[ModuleFixtureKey]]
        Exact module-scoped fixture identities used by each collected family.
    cost_profile : ShardCostProfile
        Measured scheduling overrides and split rules. Where optional, None uses item
        counts and ordinary file grouping.

    Raises
    ------
    ValueError
        A split shares a fixture not declared repeatable, or a strict repeatable
        fixture entry is stale.

    Notes
    -----
    This validates the proposed split and changes no assignments.
    """
    extracted_nodeids = set(cost_profile.extracted_family_costs)
    split_files = set(cost_profile.split_file_family_cost_floors)
    extracted_paths = {
        nodeid.split("::", maxsplit=1)[0] for nodeid in extracted_nodeids
    }
    split_paths = extracted_paths | split_files
    discovered_module_fixtures = {
        fixture_key
        for family, fixture_keys in module_fixtures_by_family.items()
        if logical_path_from_family(family) in split_paths
        for fixture_key in fixture_keys
    }
    stale_repeatable_fixtures = (
        cost_profile.repeatable_module_fixtures - discovered_module_fixtures
    )
    if cost_profile.strict and stale_repeatable_fixtures:
        raise ValueError(
            "stale repeatable module fixture profile: "
            f"{sorted(stale_repeatable_fixtures)}"
        )
    for logical_path in sorted(split_paths):
        fixture_owners: dict[ModuleFixtureKey, set[str]] = {}
        for family, fixture_keys in module_fixtures_by_family.items():
            if logical_path_from_family(family) != logical_path:
                continue
            work_unit = (
                f"family:{family[1]}"
                if logical_path in split_files or family[1] in extracted_nodeids
                else f"residual:{logical_path}"
            )
            for fixture_key in fixture_keys:
                fixture_owners.setdefault(fixture_key, set()).add(work_unit)
        shared_fixtures = {
            fixture_key: sorted(owners)
            for fixture_key, owners in fixture_owners.items()
            if len(owners) > 1
            and fixture_key not in cost_profile.repeatable_module_fixtures
        }
        if shared_fixtures:
            labels = {
                f"{baseid}::{name}": owners
                for (baseid, name), owners in sorted(shared_fixtures.items())
            }
            raise ValueError(
                f"split test families cannot share module-scoped fixtures: {labels}"
            )


def validate_split_dynamic_fixture_requests(
    request_sites_by_family: Mapping[TestFamilyKey, frozenset[str]],
    cost_profile: ShardCostProfile,
) -> None:
    """Reject dynamic fixture selection in files selected for splitting.

    Parameters
    ----------
    request_sites_by_family : Mapping[TestFamilyKey, frozenset[str]]
        Known direct or fixture-mediated dynamic request sites for each family.
    cost_profile : ShardCostProfile
        Measured scheduling overrides and split rules. Where optional, None uses item
        counts and ordinary file grouping.

    Raises
    ------
    ValueError
        A selected split file contains a known dynamic request site.

    Notes
    -----
    Dynamic fixture selection cannot prove the static fixture-sharing boundary.
    """
    split_paths = set(cost_profile.split_file_family_cost_floors) | {
        nodeid.split("::", maxsplit=1)[0]
        for nodeid in cost_profile.extracted_family_costs
    }
    request_sites = sorted(
        {
            request_site
            for family, family_request_sites in request_sites_by_family.items()
            if logical_path_from_family(family) in split_paths
            for request_site in family_request_sites
        }
    )
    if request_sites:
        raise ValueError(
            "split test files cannot use pytest's dynamic request fixture API: "
            f"{request_sites}"
        )


def _positive_costs(values: Mapping[str, int], label: str) -> None:
    """Reject Boolean, non-integer, or nonpositive measured costs."""
    if any(type(value) is not int or value < 1 for value in values.values()):
        raise ValueError(f"{label} must contain positive integer costs")


def _reserved_costs(profile: ShardCostProfile, shard_count: int) -> tuple[int, ...]:
    """Return one nonnegative reserved cost per shard, defaulting each to zero."""
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
    """Group test families into indivisible units with declared scheduling costs.

    Parameters
    ----------
    item_counts : Mapping[TestFamilyKey, int]
        Positive collected-item count for every unparameterized test family.
    cost_profile : ShardCostProfile | None
        Measured scheduling overrides and split rules. Where optional, None uses item
        counts and ordinary file grouping.

    Returns
    -------
    tuple of TestWorkUnit
        Stable file, extracted-family, or split-family units.

    Raises
    ------
    ValueError
        Counts, costs, logical IDs, or strict profile entries are invalid.

    Notes
    -----
    Ordinary files remain whole. Declared split/extraction rules never separate
    parameter cases of one family. Cost units are relative scheduling weights,
    not a promise of measured seconds on the current machine.
    """
    if any(count < 1 for count in item_counts.values()):
        raise ValueError("every test family must contain at least one item")
    profile = cost_profile or ShardCostProfile()
    _positive_costs(profile.file_cost_overrides, "file cost overrides")
    _positive_costs(
        profile.split_file_family_cost_floors,
        "split-file family cost floors",
    )
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
    split_files = set(profile.split_file_family_cost_floors)
    residual_files = set(profile.residual_file_costs)
    extracted_nodeids = set(profile.extracted_family_costs)
    extracted_paths = {
        nodeid.split("::", maxsplit=1)[0] for nodeid in extracted_nodeids
    }
    if profile.strict:
        missing_files = (
            configured_files | split_files | residual_files
        ) - families_by_path.keys()
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
        split_conflicts = split_files & (
            configured_files | residual_files | extracted_paths
        )
        if split_conflicts:
            raise ValueError(
                "family-split files cannot use another file or family profile: "
                f"{sorted(split_conflicts)}"
            )

    units: list[TestWorkUnit] = []
    for logical_path, raw_families in sorted(families_by_path.items()):
        families = tuple(sorted(raw_families))
        if logical_path in split_files:
            cost_floor = profile.split_file_family_cost_floors[logical_path]
            units.extend(
                TestWorkUnit(
                    identifier=f"family:{family[1]}",
                    families=(family,),
                    cost=max(item_counts[family], cost_floor),
                )
                for family in families
            )
            continue
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
    """Assign whole work units to shards using largest-cost-first packing.

    Parameters
    ----------
    item_counts : Mapping[TestFamilyKey, int]
        Positive collected-item count for every unparameterized test family.
    shard_count : int
        Positive worker count no larger than the number of indivisible work units.
    cost_profile : ShardCostProfile | None
        Measured scheduling overrides and split rules. Where optional, None uses item
        counts and ordinary file grouping.

    Returns
    -------
    tuple of tuple of TestFamilyKey
        Deterministic nonempty shard memberships, sorted within each shard.

    Raises
    ------
    ValueError
        Counts, costs, shard count, reserved work, or declared relocations are invalid.

    Notes
    -----
    Each unit goes to the current least-loaded shard with stable tie breaking.
    Checked explicit relocations run afterward. This computes membership only;
    it does not execute tests or change their assertion logic.
    """
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

    unit_by_identifier = {unit.identifier: unit for unit in units}
    seen_relocations: set[str] = set()
    for (
        identifier,
        source_shard,
        target_shard,
    ) in profile.relocations_by_shard_count.get(shard_count, ()):
        if identifier in seen_relocations:
            raise ValueError(f"duplicate CI shard relocation: {identifier}")
        seen_relocations.add(identifier)
        if type(source_shard) is not int or type(target_shard) is not int:
            raise ValueError("CI shard relocation indexes must be integer shard IDs")
        if not 1 <= source_shard <= shard_count or not 1 <= target_shard <= shard_count:
            raise ValueError("CI shard relocation indexes must use one-based shard IDs")
        if source_shard == target_shard:
            raise ValueError("CI shard relocation must change the owning shard")
        unit = unit_by_identifier.get(identifier)
        if unit is None:
            raise ValueError(f"stale CI shard relocation work unit: {identifier}")
        source_index = source_shard - 1
        target_index = target_shard - 1
        source_families = assignments[source_index]
        if not all(family in source_families for family in unit.families):
            raise ValueError(
                "CI shard relocation source drift: "
                f"{identifier} is not wholly owned by shard {source_shard}"
            )
        for family in unit.families:
            source_families.remove(family)
        assignments[target_index].extend(unit.families)

    if any(not families for families in assignments):
        raise ValueError("weighted CI shard assignment produced an empty shard")
    return tuple(tuple(sorted(families)) for families in assignments)


def pytest_addoption(parser: Parser) -> None:
    """Register the optional --ci-shard=N/M pytest command-line argument.

    Parameters
    ----------
    parser : Parser
        Pytest option parser modified to register --ci-shard.

    Notes
    -----
    Called by pytest during plugin setup. No selector means no collection filtering.
    """
    group = parser.getgroup("CI sharding")
    group.addoption(
        "--ci-shard",
        metavar="N/M",
        help="Run one deterministic weighted work-unit shard of the collected tests.",
    )


def pytest_collection_modifyitems(config: Config, items: list[Item]) -> None:
    """Keep only the selected shard after checking the complete collection.

    Parameters
    ----------
    config : Config
        Pytest configuration containing the optional shard selector.
    items : list[Item]
        Complete collected item list, replaced in place with this shard's items.

    Raises
    ------
    pytest.UsageError
        The selector, fixture split, cost profile, or assignment is invalid.

    Notes
    -----
    With no selector this leaves items unchanged. Otherwise it reports the
    removed items through pytest_deselected and replaces the list in place.
    Relative collection order of retained items is preserved.
    """
    raw_spec = config.getoption("ci_shard")
    if raw_spec is None:
        return
    if not isinstance(raw_spec, str):
        raise pytest.UsageError("--ci-shard must use the form N/M")

    shard_index, shard_count = parse_shard_spec(raw_spec)
    families = tuple(family_key_from_item(item) for item in items)
    module_fixtures_by_family: dict[TestFamilyKey, set[ModuleFixtureKey]] = {}
    request_sites_by_family: dict[TestFamilyKey, set[str]] = {}
    for item, family in zip(items, families, strict=True):
        module_fixtures_by_family.setdefault(family, set()).update(
            module_fixture_keys_from_item(item)
        )
        request_sites_by_family.setdefault(family, set()).update(
            dynamic_fixture_request_sites_from_item(item)
        )
    try:
        validate_split_fixture_affinity(
            {
                family: frozenset(fixture_keys)
                for family, fixture_keys in module_fixtures_by_family.items()
            },
            CI_SHARD_COST_PROFILE,
        )
        validate_split_dynamic_fixture_requests(
            {
                family: frozenset(request_sites)
                for family, request_sites in request_sites_by_family.items()
            },
            CI_SHARD_COST_PROFILE,
        )
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
    """Count collected items assigned to each shard.

    Parameters
    ----------
    assignments : Sequence[Sequence[TestFamilyKey]]
        Ordered shard memberships whose entries are whole test-family keys.
    item_counts : Mapping[TestFamilyKey, int]
        Positive collected-item count for every unparameterized test family.

    Returns
    -------
    tuple of int
        Item totals in shard order.

    Raises
    ------
    KeyError
        An assigned family is absent from item_counts.

    Notes
    -----
    This is a diagnostic sum, not an exact-cover validator or timing estimate.
    """
    return tuple(sum(item_counts[path] for path in paths) for paths in assignments)


def shard_costs(
    assignments: Sequence[Sequence[TestFamilyKey]],
    item_counts: Mapping[TestFamilyKey, int],
    *,
    cost_profile: ShardCostProfile | None = None,
) -> tuple[int, ...]:
    """Compute assigned scheduling cost after proving whole-unit exact coverage.

    Parameters
    ----------
    assignments : Sequence[Sequence[TestFamilyKey]]
        Ordered shard memberships whose entries are whole test-family keys.
    item_counts : Mapping[TestFamilyKey, int]
        Positive collected-item count for every unparameterized test family.
    cost_profile : ShardCostProfile | None
        Measured scheduling overrides and split rules. Where optional, None uses item
        counts and ordinary file grouping.

    Returns
    -------
    tuple of int
        Work-unit costs plus reserved non-pytest costs, in shard order.

    Raises
    ------
    ValueError
        Membership misses or repeats a family, splits a work unit, or uses an
        invalid cost profile.

    Notes
    -----
    The numbers are scheduling weights. They do not measure elapsed time.
    """
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
