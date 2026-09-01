"""Focused tests for deterministic weighted CI test-work-unit sharding."""

from collections import Counter
from types import SimpleNamespace
from typing import cast

import pytest
from _pytest.nodes import Item
from scripts.dev.pytest_shard import (
    CI_SHARD_COST_PROFILE,
    ShardCostProfile,
    TestFamilyKey,
    assign_test_families,
    build_test_work_units,
    dynamic_fixture_request_sites_from_item,
    family_key_from_metadata,
    module_fixture_keys_from_item,
    parse_shard_spec,
    shard_costs,
    shard_loads,
    validate_split_dynamic_fixture_requests,
    validate_split_fixture_affinity,
)


@pytest.fixture(scope="module")
def module_affinity_anchor() -> object:
    return object()


@pytest.fixture
def indirect_affinity_anchor(module_affinity_anchor: object) -> object:
    return module_affinity_anchor


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


def test_split_fixture_affinity_rejects_every_shared_module_fixture_definition(
    request: pytest.FixtureRequest,
    indirect_affinity_anchor: object,
) -> None:
    del indirect_affinity_anchor
    path = "tests/test_pytest_shard.py"
    extracted = (
        path,
        f"{path}::test_split_fixture_affinity_rejects_every_shared_module_fixture_definition",
    )
    second_extracted = (path, f"{path}::test_second_extracted_family")
    residual = (path, f"{path}::test_fixture_consumer")
    node = cast(Item, request.node)
    fixture_keys = module_fixture_keys_from_item(node)
    request_sites = dynamic_fixture_request_sites_from_item(node)
    module_fixture = (path, "module_affinity_anchor")
    assert module_fixture in fixture_keys
    assert node.nodeid in request_sites
    profile = ShardCostProfile(
        extracted_family_costs={extracted[1]: 100},
        residual_file_costs={path: 40},
        strict=True,
    )

    with pytest.raises(
        ValueError,
        match="split test families cannot share module-scoped fixtures",
    ):
        validate_split_fixture_affinity(
            {
                extracted: fixture_keys,
                residual: frozenset({module_fixture}),
            },
            profile,
        )

    repeatable_profile = ShardCostProfile(
        split_file_family_cost_floors={path: 10},
        repeatable_module_fixtures=frozenset({module_fixture}),
        strict=True,
    )
    validate_split_fixture_affinity(
        {
            extracted: fixture_keys,
            residual: frozenset({module_fixture}),
        },
        repeatable_profile,
    )
    same_name_override = ("tests/parent.py", module_fixture[1])
    with pytest.raises(
        ValueError,
        match="split test families cannot share module-scoped fixtures",
    ):
        validate_split_fixture_affinity(
            {
                extracted: frozenset({module_fixture, same_name_override}),
                residual: frozenset({module_fixture, same_name_override}),
            },
            repeatable_profile,
        )
    with pytest.raises(ValueError, match="stale repeatable module fixture profile"):
        validate_split_fixture_affinity(
            {
                extracted: fixture_keys,
                residual: frozenset({module_fixture}),
            },
            ShardCostProfile(
                split_file_family_cost_floors={path: 10},
                repeatable_module_fixtures=frozenset(
                    {(path, "renamed_module_fixture")}
                ),
                strict=True,
            ),
        )

    extracted_pair_profile = ShardCostProfile(
        extracted_family_costs={
            extracted[1]: 100,
            second_extracted[1]: 90,
        },
        residual_file_costs={path: 40},
        strict=True,
    )
    with pytest.raises(
        ValueError,
        match="split test families cannot share module-scoped fixtures",
    ):
        validate_split_fixture_affinity(
            {
                extracted: fixture_keys,
                second_extracted: fixture_keys,
                residual: frozenset({(path, "independent_module_fixture")}),
            },
            extracted_pair_profile,
        )

    validate_split_fixture_affinity(
        {
            extracted: fixture_keys,
            residual: frozenset({(path, "independent_module_fixture")}),
        },
        profile,
    )

    with pytest.raises(
        ValueError,
        match="split test files cannot use pytest's dynamic request fixture API",
    ):
        validate_split_dynamic_fixture_requests(
            {
                extracted: request_sites,
                residual: frozenset(),
            },
            profile,
        )
    validate_split_dynamic_fixture_requests(
        {
            extracted: frozenset(),
            residual: frozenset(),
        },
        profile,
    )

    overridden_item = cast(
        Item,
        SimpleNamespace(
            _fixtureinfo=SimpleNamespace(
                name2fixturedefs={
                    "shared_fixture": (
                        SimpleNamespace(scope="module", baseid="tests/parent.py"),
                        SimpleNamespace(scope="function", baseid=path),
                    )
                }
            )
        ),
    )
    assert ("tests/parent.py", "shared_fixture") in module_fixture_keys_from_item(
        overridden_item
    )

    def project_fixture() -> None:
        return None

    dynamic_fixture_item = cast(
        Item,
        SimpleNamespace(
            nodeid=f"{path}::test_dynamic_fixture_consumer",
            _fixtureinfo=SimpleNamespace(
                argnames=(),
                name2fixturedefs={
                    "project_fixture": (
                        SimpleNamespace(
                            argname="project_fixture",
                            argnames=("request",),
                            baseid=path,
                            func=project_fixture,
                            scope="function",
                        ),
                    )
                },
            ),
        ),
    )
    assert dynamic_fixture_request_sites_from_item(dynamic_fixture_item) == frozenset(
        {f"{path}::project_fixture"}
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


def test_declared_hotspot_splits_only_at_atomic_family_boundaries() -> None:
    path = "tests/test_hotspot.py"
    parameterized = (path, f"{path}::test_parameterized")
    first = (path, f"{path}::test_first")
    second = (path, f"{path}::test_second")
    profile = ShardCostProfile(
        split_file_family_cost_floors={path: 20},
        strict=True,
    )
    counts: dict[TestFamilyKey, int] = {
        parameterized: 7,
        first: 1,
        second: 2,
    }

    units = build_test_work_units(counts, profile)
    assignments = assign_test_families(counts, 3, cost_profile=profile)

    assert {unit.identifier for unit in units} == {
        f"family:{parameterized[1]}",
        f"family:{first[1]}",
        f"family:{second[1]}",
    }
    assert {unit.cost for unit in units} == {20}
    assert all(len(unit.families) == 1 for unit in units)
    assert sum(parameterized in shard for shard in assignments) == 1
    assert set(family for shard in assignments for family in shard) == set(counts)


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
    families = {
        (f"tests/test_{name}.py", f"tests/test_{name}.py::test_value"): cost
        for name, cost in (("a", 100), ("b", 90), ("c", 80), ("d", 70))
    }
    relocated_identifier = "file:tests/test_d.py"
    profile = ShardCostProfile(
        relocations_by_shard_count={
            3: ((relocated_identifier, 3, 2),),
        }
    )

    assignments = assign_test_families(families, 3, cost_profile=profile)
    relocated_family = ("tests/test_d.py", "tests/test_d.py::test_value")

    assert assignments == assign_test_families(
        dict(reversed(tuple(families.items()))),
        3,
        cost_profile=profile,
    )
    assert relocated_family not in assignments[2]
    assert relocated_family in assignments[1]
    assert sorted(family for shard in assignments for family in shard) == sorted(
        families
    )

    wrong_source = ShardCostProfile(
        relocations_by_shard_count={
            3: ((relocated_identifier, 1, 2),),
        }
    )
    with pytest.raises(ValueError, match="CI shard relocation source drift"):
        assign_test_families(families, 3, cost_profile=wrong_source)

    stale_unit = ShardCostProfile(
        relocations_by_shard_count={
            3: (("file:tests/test_missing.py", 3, 2),),
        }
    )
    with pytest.raises(ValueError, match="stale CI shard relocation work unit"):
        assign_test_families(families, 3, cost_profile=stale_unit)

    invalid_profiles = (
        (
            ShardCostProfile(
                relocations_by_shard_count={
                    3: ((relocated_identifier, 0, 2),),
                }
            ),
            "relocation indexes must use one-based shard IDs",
        ),
        (
            ShardCostProfile(
                relocations_by_shard_count={
                    3: ((relocated_identifier, 3, 3),),
                }
            ),
            "relocation must change the owning shard",
        ),
        (
            ShardCostProfile(
                relocations_by_shard_count={
                    3: (
                        (relocated_identifier, 3, 2),
                        (relocated_identifier, 3, 1),
                    ),
                }
            ),
            "duplicate CI shard relocation",
        ),
        (
            ShardCostProfile(
                relocations_by_shard_count={
                    3: (("file:tests/test_a.py", 1, 2),),
                }
            ),
            "weighted CI shard assignment produced an empty shard",
        ),
    )
    for invalid_profile, message in invalid_profiles:
        with pytest.raises(ValueError, match=message):
            assign_test_families(families, 3, cost_profile=invalid_profile)

    for invalid_index in (True, 1.5, "1"):
        for invalid_relocation in (
            (relocated_identifier, invalid_index, 2),
            (relocated_identifier, 3, invalid_index),
        ):
            invalid_type = ShardCostProfile(
                relocations_by_shard_count={
                    3: (invalid_relocation,),  # type: ignore[arg-type]
                }
            )
            with pytest.raises(
                ValueError,
                match="relocation indexes must be integer shard IDs",
            ):
                assign_test_families(families, 3, cost_profile=invalid_type)


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
                split_file_family_cost_floors={"tests/test_missing.py": 10},
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


def test_production_profile_names_and_weights_exactly_five_extracted_families() -> None:
    assert CI_SHARD_COST_PROFILE.file_cost_overrides == {
        "tests/test_shared_obs_runtime.py": 110,
        "tests/test_visual_debugger_replay_service.py": 400,
    }
    assert CI_SHARD_COST_PROFILE.split_file_family_cost_floors == {
        "tests/test_scripted_team_deathmatch_no_shared_obs.py": 20,
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
        (
            "tests/test_visual_debugger_sample_replays.py::"
            "test_checked_samples_match_fresh_cpu_generation_scientific_truth"
        ): 400,
    }
    assert CI_SHARD_COST_PROFILE.residual_file_costs == {
        "tests/test_visual_debugger_scenarios.py": 160,
        "tests/test_visual_debugger_service.py": 200,
        "tests/test_visual_debugger_sample_replays.py": 500,
    }
    assert CI_SHARD_COST_PROFILE.reserved_costs_by_shard_count[12] == (
        (0,) * 11 + (50,)
    )
    assert set(CI_SHARD_COST_PROFILE.relocations_by_shard_count) == {12}
    tdm_prefix = "family:tests/test_scripted_team_deathmatch_no_shared_obs.py::"
    assert CI_SHARD_COST_PROFILE.relocations_by_shard_count[12] == (
        (
            tdm_prefix
            + "test_eager_jit_vmap_key_forms_and_x64_keep_exact_actions_and_dtypes",
            9,
            1,
        ),
        (
            tdm_prefix
            + "test_policy_uses_exact_masks_and_ignores_misleading_marginals",
            10,
            4,
        ),
        (
            tdm_prefix
            + "test_dead_inactive_and_stunned_masks_produce_the_canonical_inert_action",
            7,
            2,
        ),
        (
            tdm_prefix
            + "test_dormant_task_history_and_lifecycle_fields_do_not_change_the_policy",
            8,
            4,
        ),
        ("residual:tests/test_visual_debugger_service.py", 6, 10),
        (
            tdm_prefix + "test_invalid_damage_modifier_never_suppresses_an_aged_trap",
            8,
            11,
        ),
        (
            tdm_prefix + "test_mage_burst_uses_the_locked_configured_crowd_"
            "and_covering_boundaries",
            8,
            4,
        ),
        ("file:tests/test_shared_obs_runtime.py", 11, 9),
    )
    assert CI_SHARD_COST_PROFILE.repeatable_module_fixtures == frozenset(
        {
            (
                "tests/test_scripted_team_deathmatch_no_shared_obs.py",
                "class_rows",
            )
        }
    )


def test_dominant_units_preserve_hotspot_affinity_and_exact_ownership() -> None:
    semantic_path = "tests/test_semantic_inventory.py"
    preflight_path = "tests/test_hosted_preflight.py"
    sample_path = "tests/test_sample_replays.py"
    semantic = (semantic_path, f"{semantic_path}::test_every_mechanic")
    preflight = (preflight_path, f"{preflight_path}::test_every_case")
    independent_generation = (
        sample_path,
        f"{sample_path}::test_checked_samples_match_fresh_generation",
    )
    semantic_residual = (semantic_path, f"{semantic_path}::test_residual")
    preflight_residual = (preflight_path, f"{preflight_path}::test_residual")
    first_fixture_consumer = (sample_path, f"{sample_path}::test_first_fixture_user")
    second_fixture_consumer = (sample_path, f"{sample_path}::test_second_fixture_user")
    ordinary = {
        (f"tests/test_{index}.py", f"tests/test_{index}.py::test_value"): 100
        for index in range(10)
    }
    counts: dict[TestFamilyKey, int] = {
        semantic: 1,
        preflight: 30,
        independent_generation: 1,
        semantic_residual: 5,
        preflight_residual: 5,
        first_fixture_consumer: 20,
        second_fixture_consumer: 39,
        **ordinary,
    }
    profile = ShardCostProfile(
        extracted_family_costs={
            semantic[1]: 500,
            preflight[1]: 500,
            independent_generation[1]: 400,
        },
        residual_file_costs={
            semantic_path: 100,
            preflight_path: 100,
            sample_path: 500,
        },
        strict=True,
    )

    assignments = assign_test_families(counts, 12, cost_profile=profile)

    semantic_owner = next(shard for shard in assignments if semantic in shard)
    preflight_owner = next(shard for shard in assignments if preflight in shard)
    sample_residual_owner = next(
        shard for shard in assignments if first_fixture_consumer in shard
    )
    assert semantic_owner == (semantic,)
    assert preflight_owner == (preflight,)
    assert sample_residual_owner == (
        first_fixture_consumer,
        second_fixture_consumer,
    )
    assert independent_generation not in sample_residual_owner
    assert sum(independent_generation in shard for shard in assignments) == 1
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
