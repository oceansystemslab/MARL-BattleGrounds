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
    return {}


def _empty_reserved_costs() -> dict[int, tuple[int, ...]]:
    return {}


def _empty_relocations() -> dict[int, tuple[TestWorkUnitRelocation, ...]]:
    return {}


def _empty_module_fixture_keys() -> frozenset[ModuleFixtureKey]:
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
        "tests/test_visual_debugger_replay_service.py": 400,
    },
    split_file_family_cost_floors={
        "tests/test_scripted_team_deathmatch_no_shared_obs.py": 20,
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
    relocations_by_shard_count={
        12: (
            (
                "family:tests/test_scripted_team_deathmatch_no_shared_obs.py::"
                "test_eager_jit_vmap_key_forms_and_x64_keep_exact_actions_and_dtypes",
                9,
                1,
            ),
            (
                "family:tests/test_scripted_team_deathmatch_no_shared_obs.py::"
                "test_policy_uses_exact_masks_and_ignores_misleading_marginals",
                10,
                4,
            ),
            (
                "family:tests/test_scripted_team_deathmatch_no_shared_obs.py::"
                "test_dead_inactive_and_stunned_masks_produce_the_canonical_inert_action",
                5,
                2,
            ),
            (
                "family:tests/test_scripted_team_deathmatch_no_shared_obs.py::"
                "test_dormant_task_history_and_lifecycle_fields_do_not_change_the_policy",
                7,
                4,
            ),
            (
                "residual:tests/test_visual_debugger_service.py",
                6,
                10,
            ),
            (
                "family:tests/test_scripted_team_deathmatch_no_shared_obs.py::"
                "test_invalid_damage_modifier_never_suppresses_an_aged_trap",
                7,
                11,
            ),
            (
                "family:tests/test_scripted_team_deathmatch_no_shared_obs.py::"
                "test_mage_burst_uses_the_locked_configured_crowd_and_covering_boundaries",
                7,
                4,
            ),
            (
                "file:tests/test_shared_obs_runtime.py",
                11,
                9,
            ),
        )
    },
    repeatable_module_fixtures=frozenset(
        {
            (
                "tests/test_scripted_team_deathmatch_no_shared_obs.py",
                "class_rows",
            )
        }
    ),
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


def module_fixture_keys_from_item(item: Item) -> frozenset[ModuleFixtureKey]:
    """Return resolved module-scoped fixtures in an item's transitive closure."""
    fixture_info = getattr(item, "_fixtureinfo", None)
    fixture_defs_by_name = getattr(fixture_info, "name2fixturedefs", {})
    fixture_keys: set[ModuleFixtureKey] = set()
    for fixture_name, fixture_defs in fixture_defs_by_name.items():
        for fixture_def in fixture_defs or ():
            if fixture_def.scope == "module":
                fixture_keys.add((fixture_def.baseid, fixture_name))
    return frozenset(fixture_keys)


def dynamic_fixture_request_sites_from_item(item: Item) -> frozenset[str]:
    """Return non-pytest call sites that can select fixtures dynamically."""
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
    """Reject shared module fixtures unless an exact fixture is repeatable."""
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
    """Reject dynamic fixture selection inside any profiled split test file."""
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
