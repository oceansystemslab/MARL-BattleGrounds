"""Check packaged TDM content, repaired passages and public configuration creation.

Map sources retain approved identities, shapes, splits and paired spawn pads.
The three repaired passages admit a disc with the required 1.05-unit diameter.
Scenario loading preserves its own approved state and configuration. The
eight installed scenarios are republished at Red Zone depth 5.0, each approved
at the revision it installs; the manifest digest equals the resolved config V2
rebuilt from the loaded config. A tampered, unsealed or version-1 scenario
configuration is refused.
Both map factories use DEFAULT_TDM_RED_ZONE_DEPTH (5.0) unless given a depth,
and forward an explicit one. A custom map narrower than the default depth is
refused rather than given a smaller depth.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections.abc import Callable
from operator import itemgetter
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pydantic import ValidationError

from marl_battlegrounds import _tdm_assets
from marl_battlegrounds._tdm_assets import ScenarioContent, scenario_content
from marl_battlegrounds.core.config import (
    resolve_agent_profile,
    validate_env_config,
    validate_product_env_config,
)
from marl_battlegrounds.core.env import initialize_scenario_state, reset, step
from marl_battlegrounds.core.geometry import disc_overlaps_obstacle
from marl_battlegrounds.core.types import (
    HUNTER_CLASS_ID,
    MAGE_CLASS_ID,
    MAX_AGENT_SLOTS,
    MAX_OBSTACLE_SLOTS,
    MOVE_STAY,
    NEUTRAL_CLASS_ID,
    OBSTACLE_FEATURES,
    PRIEST_CLASS_ID,
    ROGUE_CLASS_ID,
    TASK_MODE_TDM,
    WARRIOR_CLASS_ID,
    Action,
    ActionMask,
    DoneFlags,
    EnvConfig,
    EnvState,
    Info,
    Observation,
    Reward,
)
from marl_battlegrounds.evaluation.catalog import build_resolved_env_config_v2
from marl_battlegrounds.evaluation.models import (
    ResolvedEnvConfigV2,
    canonical_digest_sha256,
)
from marl_battlegrounds.tasks import (
    CANONICAL_TDM_EVALUATION_MAP_IDS,
    DEFAULT_TDM_RED_ZONE_DEPTH,
    AgentClassName,
    TDMScenarioInfo,
    balanced_spawn_configs,
    canonical_tournament_rosters,
    list_tdm_maps,
    list_tdm_scenarios,
    load_tdm_scenario,
    make_canonical_team_deathmatch_evaluation_config,
    make_standard_team_deathmatch_config,
    prepare_exact_env_config,
    spawn_locations_for_source,
)


def test_approved_inventory_and_immutable_provenance() -> None:
    maps = list_tdm_maps()
    scenarios = list_tdm_scenarios()
    assert tuple(row.map_id for row in maps) == tuple(range(52))
    assert tuple(row.map_id for row in maps if row.curriculum) == tuple(range(12))
    assert all(row.split == "training" for row in maps if row.curriculum)
    assert tuple(
        row.map_id for row in maps if row.split == "training" and not row.curriculum
    ) == tuple(range(12, 42))
    assert tuple(row.map_id for row in maps if row.split == "validation") == tuple(
        range(42, 47)
    )
    assert tuple(row.map_id for row in maps if row.split == "test") == tuple(
        range(47, 52)
    )
    assert CANONICAL_TDM_EVALUATION_MAP_IDS == (47, 48, 49, 50, 51)
    for info in maps:
        assert info.source.asset_id.startswith(f"tdm_map_id_{info.map_id}_")
        assert re.match(rf"tdm[-_]map[-_]id[-_]{info.map_id}[-_]", info.name)
        assert Path(info.source.source_path).parent.name == info.source.asset_id
    assert tuple(row.scenario_id for row in scenarios) == tuple(range(1, 9))
    assert maps[3].source.revision == 2
    assert maps[33].source.revision == 8
    assert tuple(row.source.revision for row in scenarios) == (
        42,
        20,
        27,
        15,
        16,
        17,
        30,
        19,
    )
    # Each scenario, Scenario 3 included, is approved at the revision it installs.
    for row in scenarios:
        assert row.approved_source == row.source
    with pytest.raises(ValidationError, match="frozen"):
        maps[12].source.revision = 999  # type: ignore[misc]


def test_every_approved_map_constructs_fixed_shape_tdm_with_reflected_pads() -> None:
    for info in list_tdm_maps():
        config = make_standard_team_deathmatch_config(
            map_id=info.map_id, team_a_roster=("mage",), team_b_roster=("priest",)
        )
        validate_product_env_config(config)
        assert config.task_mode == TASK_MODE_TDM
        assert config.obstacles.shape == (MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES)
        assert config.obstacles.dtype == jnp.float32
        assert config.agent_profile.class_ids.shape == (MAX_AGENT_SLOTS,)
        assert config.agent_profile.class_ids.dtype == jnp.int32
        pads = np.asarray(config.team_spawn_pad_positions)
        assert pads.shape == (2, 5, 2)
        np.testing.assert_array_equal(pads[0, :, 1], pads[1, :, 1])
        np.testing.assert_array_equal(pads[0, :, 0] + pads[1, :, 0], config.map_width)


@pytest.mark.parametrize(
    ("map_id", "centers"),
    (
        (
            21,
            ((13.9214467, 5.0), (11.2785536, 5.0), (6.0785535, 5.0), (8.7214467, 5.0)),
        ),
        (
            36,
            ((6.2, 8.0), (6.1785534, 2.0), (13.8000002, 8.0), (13.8214468, 2.0)),
        ),
        (
            48,
            (
                (7.4866027, 3.9866027),
                (12.5133973, 3.9866027),
                (7.4866027, 6.0133973),
                (12.5133973, 6.0133973),
            ),
        ),
    ),
)
def test_repaired_map_passages_admit_the_required_disc(
    map_id: int,
    centers: tuple[tuple[float, float], ...],
) -> None:
    config = make_standard_team_deathmatch_config(
        map_id=map_id, team_a_roster=("warrior",), team_b_roster=("warrior",)
    )
    positions = jnp.asarray(centers, dtype=jnp.float32)
    # Check local room at each repaired passage, including surrounding obstacles.
    overlaps = jax.vmap(
        jax.vmap(disc_overlaps_obstacle, in_axes=(None, None, 0)),
        in_axes=(0, None, None),
    )(positions, 0.525, config.obstacles)
    assert not bool(jnp.any(overlaps))
    assert bool(jnp.all(positions >= 0.525))
    assert bool(
        jnp.all(positions <= jnp.asarray((config.map_width, config.map_height)) - 0.525)
    )


@pytest.mark.parametrize(
    ("team_a", "team_b", "expected_a", "expected_b"),
    (
        (("priest",), ("priest",), (PRIEST_CLASS_ID,), (PRIEST_CLASS_ID,)),
        (
            ("hunter", "hunter"),
            ("warrior", "rogue", "mage"),
            (HUNTER_CLASS_ID, HUNTER_CLASS_ID),
            (WARRIOR_CLASS_ID, ROGUE_CLASS_ID, MAGE_CLASS_ID),
        ),
        (("priest",) * 5, ("mage",), (PRIEST_CLASS_ID,) * 5, (MAGE_CLASS_ID,)),
        (
            ("priest", "rogue", "hunter", "warrior", "mage"),
            ("hunter",),
            (
                PRIEST_CLASS_ID,
                ROGUE_CLASS_ID,
                HUNTER_CLASS_ID,
                WARRIOR_CLASS_ID,
                MAGE_CLASS_ID,
            ),
            (HUNTER_CLASS_ID,),
        ),
    ),
)
def test_standard_factory_preserves_order_duplicates_and_independent_sizes(
    team_a: tuple[AgentClassName, ...],
    team_b: tuple[AgentClassName, ...],
    expected_a: tuple[int, ...],
    expected_b: tuple[int, ...],
) -> None:
    config = make_standard_team_deathmatch_config(
        map_id=12,
        team_a_roster=team_a,
        team_b_roster=team_b,
        score_threshold=7,
        max_steps=43,
    )
    profile = config.agent_profile
    expected = (
        expected_a
        + (NEUTRAL_CLASS_ID,) * (5 - len(expected_a))
        + expected_b
        + (NEUTRAL_CLASS_ID,) * (5 - len(expected_b))
    )
    np.testing.assert_array_equal(profile.class_ids, expected)
    np.testing.assert_array_equal(profile.active_mask, np.asarray(expected) != 0)
    assert config.team_deathmatch_score_threshold == 7
    assert config.max_steps == 43
    state, _, _, _ = reset(config, jax.random.key(0))
    np.testing.assert_array_equal(state.alive_mask, profile.active_mask)
    np.testing.assert_array_equal(
        np.asarray(state.agent_positions)[~profile.active_mask], 0
    )


@pytest.mark.parametrize("map_id", CANONICAL_TDM_EVALUATION_MAP_IDS)
def test_canonical_factory_fixes_mirrored_class_order_and_rules(map_id: int) -> None:
    config = make_canonical_team_deathmatch_evaluation_config(map_id=map_id)
    classes = (
        MAGE_CLASS_ID,
        WARRIOR_CLASS_ID,
        HUNTER_CLASS_ID,
        ROGUE_CLASS_ID,
        PRIEST_CLASS_ID,
    )
    np.testing.assert_array_equal(config.agent_profile.class_ids, classes * 2)
    assert bool(jnp.all(config.agent_profile.active_mask))
    assert config.team_deathmatch_score_threshold == 20
    assert config.team_deathmatch_red_zone_depth == DEFAULT_TDM_RED_ZONE_DEPTH
    assert config.max_steps == 300
    assert config.ordinary_movement_distance_scale == 1.0
    assert config.spawn_shield_duration_steps == 3
    assert config.spawn_shield_movement_speed == 2.0
    np.testing.assert_array_equal(config.team_respawn_wave_period_step_count, (5, 5))
    state, _, _, _ = reset(config, jax.random.key(9))
    positions = np.asarray(state.agent_positions).reshape(2, 5, 2)
    np.testing.assert_array_equal(positions[0, :, 1], positions[1, :, 1])
    np.testing.assert_array_equal(positions[0, :, 0] + positions[1, :, 0], 20)


@pytest.mark.parametrize("red_zone_depth", (6.0, 0.0, 20.0))
def test_factories_default_to_the_shared_red_zone_depth_and_forward_explicit_ones(
    red_zone_depth: float,
) -> None:
    assert DEFAULT_TDM_RED_ZONE_DEPTH == 5.0
    assert type(DEFAULT_TDM_RED_ZONE_DEPTH) is float
    standard = make_standard_team_deathmatch_config(
        map_id=12, team_a_roster=("mage",), team_b_roster=("priest",)
    )
    canonical = make_canonical_team_deathmatch_evaluation_config(map_id=47)
    for config in (standard, canonical):
        assert type(config.team_deathmatch_red_zone_depth) is float
        assert config.team_deathmatch_red_zone_depth == DEFAULT_TDM_RED_ZONE_DEPTH
    explicit_standard = make_standard_team_deathmatch_config(
        map_id=12,
        team_a_roster=("mage",),
        team_b_roster=("priest",),
        red_zone_depth=red_zone_depth,
    )
    explicit_canonical = make_canonical_team_deathmatch_evaluation_config(
        map_id=47, red_zone_depth=red_zone_depth
    )
    for config, default_config in (
        (explicit_standard, standard),
        (explicit_canonical, canonical),
    ):
        assert config.team_deathmatch_red_zone_depth == red_zone_depth
        # Nothing else changes.
        for left, right in zip(
            jax.tree.leaves(
                config._replace(
                    team_deathmatch_red_zone_depth=DEFAULT_TDM_RED_ZONE_DEPTH
                )
            ),
            jax.tree.leaves(default_config),
            strict=True,
        ):
            np.testing.assert_array_equal(left, right)


def test_narrow_custom_map_refuses_the_default_depth_instead_of_shrinking_it() -> None:
    source = make_standard_team_deathmatch_config(
        map_id=12, team_a_roster=("priest",), team_b_roster=("priest",)
    )
    assert source.team_deathmatch_red_zone_depth == 5.0
    rows = jnp.arange(1.0, 10.0, 2.0, dtype=jnp.float32)
    narrow = source._replace(
        map_width=4.5,
        obstacles=jnp.zeros_like(source.obstacles),
        team_spawn_pad_positions=jnp.stack(
            (
                jnp.stack((jnp.full((5,), 1.0, jnp.float32), rows), axis=-1),
                jnp.stack((jnp.full((5,), 3.5, jnp.float32), rows), axis=-1),
            )
        ),
    )
    message = (
        "team_deathmatch_red_zone_depth must not exceed map_width after conversion "
        "to float32, not 5.0 with map_width 4.5."
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        validate_env_config(narrow)
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        prepare_exact_env_config(narrow, num_envs=None)
    # The same map is valid once its depth fits, so the depth was the only fault.
    for depth in (4.5, 0.0):
        validate_env_config(narrow._replace(team_deathmatch_red_zone_depth=depth))


def test_canonical_roster_discovery_reuses_the_factory_order() -> None:
    first, second = canonical_tournament_rosters()
    assert first == second == ("mage", "warrior", "hunter", "rogue", "priest")
    assert isinstance(first, tuple) and isinstance(second, tuple)
    expected = make_canonical_team_deathmatch_evaluation_config(map_id=47)
    actual = make_standard_team_deathmatch_config(
        map_id=47, team_a_roster=first, team_b_roster=second
    )
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("batched", (False, True))
def test_balanced_configs_preserve_sources_and_all_nonpad_values(batched: bool) -> None:
    first = make_standard_team_deathmatch_config(
        map_id=12, team_a_roster=("priest", "mage"), team_b_roster=("rogue",) * 3
    )._replace(ordinary_movement_distance_scale=0.25)
    second = make_standard_team_deathmatch_config(
        map_id=13, team_a_roster=("mage",), team_b_roster=("priest",)
    )
    source = (
        jax.tree.map(lambda *values: jnp.stack(values), first, second)
        if batched
        else first
    )
    before = [np.array(value, copy=True) for value in jax.tree.leaves(source)]
    prepared = balanced_spawn_configs(source, num_envs=2)
    expected_sources = (first, second) if batched else (first, first)
    for lane, expected in enumerate(expected_sources):
        actual = jax.tree.map(itemgetter(lane), prepared)
        for name in EnvConfig._fields:
            value, reference = getattr(actual, name), getattr(expected, name)
            if name == "team_spawn_pad_positions" and lane == 1:
                reference = reference[::-1]
            for left, right in zip(
                jax.tree.leaves(value), jax.tree.leaves(reference), strict=True
            ):
                np.testing.assert_array_equal(left, right)
    for left, right in zip(jax.tree.leaves(source), before, strict=True):
        np.testing.assert_array_equal(left, right)
    repeated = balanced_spawn_configs(prepared, num_envs=2)
    np.testing.assert_array_equal(
        repeated.team_spawn_pad_positions[1],
        expected_sources[1].team_spawn_pad_positions,
    )


@pytest.mark.parametrize("size", (True, 0, -2, 1, 3, 2.0, "2"))
def test_balanced_configs_reject_invalid_native_batch_sizes(size: object) -> None:
    config = make_canonical_team_deathmatch_evaluation_config(map_id=47)
    with pytest.raises(ValueError, match="positive even integer"):
        balanced_spawn_configs(config, num_envs=cast(int, size))


@pytest.mark.parametrize("size", (None, 1, 3))
def test_exact_config_preparation_accepts_custom_scalar_and_odd_batches(
    size: int | None,
) -> None:
    config = make_canonical_team_deathmatch_evaluation_config(map_id=47)._replace(
        ordinary_movement_distance_scale=0.25
    )
    prepared = prepare_exact_env_config(config, num_envs=size)
    expected_prefix = () if size is None else (size,)
    for actual, expected in zip(
        jax.tree.leaves(prepared), jax.tree.leaves(config), strict=True
    ):
        assert not actual.weak_type
        np.testing.assert_array_equal(
            actual, np.broadcast_to(expected, (*expected_prefix, *np.shape(expected)))
        )
    again = prepare_exact_env_config(prepared, num_envs=size)
    for left, right in zip(
        jax.tree.leaves(again), jax.tree.leaves(prepared), strict=True
    ):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("field", ("max_steps", "obstacles", "profile"))
def test_config_preparation_rejects_each_kind_of_mixed_batch_shape(field: str) -> None:
    source = make_canonical_team_deathmatch_evaluation_config(map_id=47)
    batch = prepare_exact_env_config(source, num_envs=2)
    if field == "profile":
        broken = batch._replace(
            agent_profile=batch.agent_profile._replace(
                class_ids=source.agent_profile.class_ids
            )
        )
    else:
        broken = batch._replace(**{field: getattr(source, field)})
    for prepare in (prepare_exact_env_config, balanced_spawn_configs):
        with pytest.raises(ValueError, match="inconsistent scalar or batch shape"):
            prepare(broken, num_envs=2)


def test_config_validation_does_not_silently_narrow_array_dtypes() -> None:
    source = make_canonical_team_deathmatch_evaluation_config(map_id=47)
    broken = source._replace(obstacles=np.asarray(source.obstacles, dtype=np.float64))
    with pytest.raises(TypeError, match="dtype"):
        prepare_exact_env_config(broken, num_envs=None)


def test_exact_source_does_not_validate_an_unused_invalid_bank_exchange() -> None:
    config = make_standard_team_deathmatch_config(
        map_id=12, team_a_roster=("priest",), team_b_roster=("priest",)
    )
    # Neutral custom setup permits an empty team. Its pads can touch the wall,
    # but those same pads cannot receive the other team's nonzero body radius.
    profile = resolve_agent_profile(
        jnp.asarray((PRIEST_CLASS_ID,) + (0,) * 9, jnp.int32),
        jnp.asarray((1, 0), jnp.int32),
    )
    config = config._replace(
        task_mode=0,
        team_deathmatch_score_threshold=0,
        team_deathmatch_red_zone_depth=0.0,
        agent_profile=profile,
        obstacles=jnp.zeros_like(config.obstacles),
        team_spawn_pad_positions=config.team_spawn_pad_positions.at[1, :, 0].set(0),
    )
    validate_env_config(config)
    exact = prepare_exact_env_config(config, num_envs=None)
    np.testing.assert_array_equal(
        exact.team_spawn_pad_positions, config.team_spawn_pad_positions
    )
    swapped = config._replace(
        team_spawn_pad_positions=config.team_spawn_pad_positions[::-1]
    )
    with pytest.raises(ValueError, match="radius-adjusted map bounds"):
        prepare_exact_env_config(swapped, num_envs=None)
    with pytest.raises(ValueError, match="swapped spawn locations"):
        balanced_spawn_configs(config, num_envs=2)


def test_config_preparation_accepts_constants_captured_by_compiled_callers() -> None:
    source = make_canonical_team_deathmatch_evaluation_config(map_id=47)

    def prepare() -> tuple[EnvConfig, EnvConfig]:
        return (
            prepare_exact_env_config(source, num_envs=2),
            balanced_spawn_configs(source, num_envs=2),
        )

    expected = prepare()
    actual = cast(tuple[EnvConfig, EnvConfig], jax.jit(prepare)())
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_array_equal(left, right)


def test_balanced_config_selection_reuses_compilation_without_host_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import marl_battlegrounds.tasks as tasks

    first = make_canonical_team_deathmatch_evaluation_config(map_id=47)
    second = make_canonical_team_deathmatch_evaluation_config(map_id=48)
    expected = [
        balanced_spawn_configs(source, num_envs=2) for source in (first, second)
    ]
    calls = 0

    def prepare(source: EnvConfig) -> EnvConfig:
        nonlocal calls
        calls += 1
        return balanced_spawn_configs(source, num_envs=2)

    def reject_host_validation(_config: EnvConfig) -> None:
        raise AssertionError("host validation entered a compiled config selection")

    monkeypatch.setattr(tasks, "validate_env_config", reject_host_validation)
    compiled = cast(Callable[[EnvConfig], EnvConfig], jax.jit(prepare))
    for source, reference in zip((first, second), expected, strict=True):
        actual = compiled(source)
        for left, right in zip(
            jax.tree.leaves(actual), jax.tree.leaves(reference), strict=True
        ):
            np.testing.assert_array_equal(left, right)
    assert calls == 1


@pytest.mark.parametrize("distinct", (False, True))
def test_config_setup_copies_batch_once_and_validates_distinct_sources(
    distinct: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    import marl_battlegrounds.tasks as tasks

    source = make_canonical_team_deathmatch_evaluation_config(map_id=47)
    batch = prepare_exact_env_config(source, num_envs=128)
    if distinct:
        batch = batch._replace(max_steps=jnp.asarray(batch.max_steps).at[-1].add(1))
    snapshots = 0
    validations = 0
    original_get = jax.device_get
    try:
        validation_platform = cast(str, jax.local_devices(backend="cpu")[0].platform)
    except RuntimeError:
        validation_platform = cast(str, jax.local_devices()[0].platform)

    def capture(config: EnvConfig) -> EnvConfig:
        nonlocal snapshots
        snapshots += 1
        return original_get(config)

    def validate(config: EnvConfig) -> None:
        nonlocal validations
        validations += 1
        assert all(
            cast(str, device.platform) == validation_platform
            for value in jax.tree.leaves(config)
            if isinstance(value, jax.Array)
            for device in value.devices()
        )
        validate_env_config(config)

    monkeypatch.setattr(jax, "device_get", capture)
    monkeypatch.setattr(tasks, "validate_env_config", validate)
    balanced_spawn_configs(batch, num_envs=128)
    assert snapshots == 1
    assert validations == (4 if distinct else 2)


@pytest.mark.parametrize("batched", (False, True))
def test_spawn_source_relationship_preserves_declared_bank_order(batched: bool) -> None:
    source = make_canonical_team_deathmatch_evaluation_config(map_id=47)
    swapped_source = source._replace(
        team_spawn_pad_positions=source.team_spawn_pad_positions[::-1]
    )
    resolved = balanced_spawn_configs(source, num_envs=2) if batched else swapped_source
    compiled = cast(
        Callable[[EnvConfig, EnvConfig], tuple[jax.Array, jax.Array]],
        jax.jit(spawn_locations_for_source),
    )
    for declared, expected in (
        (source, (0, 1) if batched else 1),
        (swapped_source, (1, 0) if batched else 0),
    ):
        matches, choices = compiled(resolved, declared)
        np.testing.assert_array_equal(matches, np.ones((2,) if batched else (), bool))
        np.testing.assert_array_equal(choices, expected)
        assert choices.dtype == jnp.int32
    if batched:
        declared_batch = prepare_exact_env_config(source, num_envs=2)
        matches, choices = compiled(resolved, declared_batch)
        np.testing.assert_array_equal(matches, (True, True))
        np.testing.assert_array_equal(choices, (0, 1))


@pytest.mark.parametrize("field", ("pads", "width", "obstacles", "profile", "rules"))
def test_spawn_source_relationship_rejects_undeclared_edits(field: str) -> None:
    source = make_standard_team_deathmatch_config(
        map_id=12, team_a_roster=("priest",), team_b_roster=("mage",)
    )
    resolved = balanced_spawn_configs(source, num_envs=2)
    if field == "pads":
        # An unused pad still belongs to the declared complete bank.
        resolved = resolved._replace(
            team_spawn_pad_positions=resolved.team_spawn_pad_positions.at[
                1, 0, 4, 1
            ].add(0.25)
        )
    elif field == "width":
        resolved = resolved._replace(
            map_width=jnp.asarray(resolved.map_width).at[1].add(1)
        )
    elif field == "obstacles":
        resolved = resolved._replace(obstacles=resolved.obstacles.at[1, 0, 0].add(1))
    elif field == "profile":
        resolved = resolved._replace(
            agent_profile=resolved.agent_profile._replace(
                class_ids=resolved.agent_profile.class_ids.at[1, 0].set(MAGE_CLASS_ID)
            )
        )
    else:
        resolved = resolved._replace(
            max_steps=jnp.asarray(resolved.max_steps).at[1].add(1)
        )
    matches, choices = cast(
        tuple[jax.Array, jax.Array],
        jax.jit(spawn_locations_for_source)(resolved, source),
    )
    np.testing.assert_array_equal(matches, (True, False))
    np.testing.assert_array_equal(choices, (0, -1))


def test_spawn_source_relationship_distinguishes_unknown_from_invalid() -> None:
    source = make_canonical_team_deathmatch_evaluation_config(map_id=47)
    # The relationship helper checks identity, not physical validity.
    source = source._replace(
        team_spawn_pad_positions=jnp.broadcast_to(
            source.team_spawn_pad_positions[:1], source.team_spawn_pad_positions.shape
        )
    )
    matches, choice = spawn_locations_for_source(source, source)
    assert bool(matches)
    assert int(choice) == -1
    matches, choice = spawn_locations_for_source(
        source._replace(max_steps=source.max_steps + 1), source
    )
    assert not bool(matches)
    assert int(choice) == -1


@pytest.mark.parametrize("map_id", (-1, 52, True, 17.0, "17"))
def test_factory_rejects_invalid_map_ids(map_id: object) -> None:
    with pytest.raises(ValueError, match="map_id"):
        make_standard_team_deathmatch_config(
            map_id=cast(int, map_id), team_a_roster=("mage",), team_b_roster=("mage",)
        )
    with pytest.raises(ValueError, match="map_id"):
        make_canonical_team_deathmatch_evaluation_config(map_id=cast(int, map_id))


@pytest.mark.parametrize("map_id", (12, 42, 0))
def test_canonical_factory_rejects_training_validation_and_curriculum_maps(
    map_id: int,
) -> None:
    with pytest.raises(ValueError, match="canonical evaluation"):
        make_canonical_team_deathmatch_evaluation_config(map_id=map_id)


@pytest.mark.parametrize(
    "roster", ((), ("mage",) * 6, ("Priest",), (1,), "mage", {"mage"})
)
def test_factory_rejects_invalid_or_unordered_rosters(roster: object) -> None:
    with pytest.raises((TypeError, ValueError), match="team_a_roster"):
        make_standard_team_deathmatch_config(
            map_id=12,
            team_a_roster=cast(tuple[AgentClassName, ...], roster),
            team_b_roster=("mage",),
        )


@pytest.mark.parametrize(
    ("threshold", "horizon"), ((0, 300), (-1, 300), (True, 300), (2.5, 300), (20, 0))
)
def test_factory_uses_core_scalar_validation(
    threshold: object, horizon: object
) -> None:
    with pytest.raises((TypeError, ValueError)):
        make_standard_team_deathmatch_config(
            map_id=12,
            team_a_roster=("mage",),
            team_b_roster=("mage",),
            score_threshold=cast(int, threshold),
            max_steps=cast(int, horizon),
        )


@pytest.mark.parametrize("scenario_id", range(1, 9))
def test_scenario_load_restores_approved_config_state_and_public_initialization(
    scenario_id: int,
) -> None:
    scenario = load_tdm_scenario(scenario_id)
    expected_horizon = 10 if scenario_id == 3 else 5
    assert scenario.info.horizon == expected_horizon
    assert scenario.config.max_steps - int(scenario.initial_state.step_count) == (
        expected_horizon
    )
    np.testing.assert_array_equal(
        scenario.config.agent_profile.class_ids, scenario.info.class_ids
    )
    np.testing.assert_array_equal(
        scenario.config.team_respawn_wave_period_step_count, (5, 5)
    )
    assert scenario.config.team_deathmatch_red_zone_depth == DEFAULT_TDM_RED_ZONE_DEPTH
    recorded = scenario_content(scenario.info).configuration
    assert type(recorded) is ResolvedEnvConfigV2
    assert build_resolved_env_config_v2(scenario.config) == recorded
    assert recorded.canonical_digest_sha256 == (
        scenario.info.resolved_configuration_digest
    )
    assert not bool(scenario.initial_state.has_previous_timestep_joint_action)
    restored, _, mask, _ = initialize_scenario_state(
        scenario.initial_state, scenario.config
    )
    for actual, expected in zip(restored, scenario.initial_state, strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert mask.move_mask.shape[0] == MAX_AGENT_SLOTS
    assert scenario.notes


@pytest.mark.parametrize("scenario_id", (0, 9, True, 3.0, "3"))
def test_scenario_load_rejects_nonexact_or_out_of_range_ids(
    scenario_id: object,
) -> None:
    with pytest.raises(ValueError, match="scenario_id"):
        load_tdm_scenario(cast(int, scenario_id))


def test_package_loader_detects_changed_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def changed_bytes(_relative_path: str) -> bytes:
        return b"{}"

    list_tdm_maps()
    monkeypatch.setattr(_tdm_assets, "_resource_bytes", changed_bytes)
    with pytest.raises(ValueError, match="content digest mismatch"):
        make_standard_team_deathmatch_config(
            map_id=12, team_a_roster=("mage",), team_b_roster=("mage",)
        )
    with pytest.raises(ValueError, match="content digest mismatch"):
        load_tdm_scenario(3)


def _resealed(configuration: dict[str, object]) -> dict[str, object]:
    fields = {
        name: value
        for name, value in configuration.items()
        if name != "canonical_digest_sha256"
    }
    return {**fields, "canonical_digest_sha256": canonical_digest_sha256(fields)}


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        ("resealed_depth_0", "disagrees with live mechanics"),
        ("resealed_depth_6", "disagrees with live mechanics"),
        ("unsealed_depth", "canonical digest mismatch"),
        ("negative_depth", "must be nonnegative"),
        ("version_1", "Input should be 2"),
    ),
)
def test_scenario_loader_refuses_tampered_configuration_records(
    monkeypatch: pytest.MonkeyPatch, tamper: str, message: str
) -> None:
    import marl_battlegrounds.tasks as tasks

    info = list_tdm_scenarios()[2]
    content = scenario_content(info).model_dump(mode="json")
    configuration = cast(dict[str, object], content["configuration"])
    changed = {
        "resealed_depth_0": _resealed(
            {**configuration, "team_deathmatch_red_zone_depth": 0.0}
        ),
        "resealed_depth_6": _resealed(
            {**configuration, "team_deathmatch_red_zone_depth": 6.0}
        ),
        "unsealed_depth": {**configuration, "team_deathmatch_red_zone_depth": 6.0},
        "negative_depth": _resealed(
            {**configuration, "team_deathmatch_red_zone_depth": -5.0}
        ),
        "version_1": _resealed(
            {
                **{
                    name: value
                    for name, value in configuration.items()
                    if name != "team_deathmatch_red_zone_depth"
                },
                "schema_version": 1,
            }
        ),
    }[tamper]
    raw = json.dumps({**content, "configuration": changed}).encode()

    def tampered_content(_info: TDMScenarioInfo) -> ScenarioContent:
        return ScenarioContent.model_validate_json(raw)

    monkeypatch.setattr(tasks, "scenario_content", tampered_content)
    with pytest.raises(ValueError, match=message):
        load_tdm_scenario(3)


def test_public_factory_supports_ordinary_and_jitted_transition() -> None:
    config = make_standard_team_deathmatch_config(
        map_id=12, team_a_roster=("priest", "priest"), team_b_roster=("hunter",)
    )
    key = jax.random.key(3)
    state, _, mask, _ = cast(
        tuple[EnvState, Observation, ActionMask, Info], jax.jit(reset)(config, key)
    )
    action = Action(
        move=jnp.full((MAX_AGENT_SLOTS,), MOVE_STAY, dtype=jnp.int32),
        select_target=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
        use_ultimate=jnp.zeros((MAX_AGENT_SLOTS,), dtype=jnp.int32),
    )
    eager = step(config, state, mask, action, key)
    compiled = cast(
        tuple[EnvState, Observation, Reward, DoneFlags, ActionMask, Info],
        jax.jit(step)(config, state, mask, action, key),
    )
    for left, right in zip(
        jax.tree_util.tree_leaves(eager),
        jax.tree_util.tree_leaves(compiled),
        strict=True,
    ):
        np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-6)
    assert int(compiled[0].step_count) == 1
    assert compiled[0].agent_positions.shape == (10, 2)


def test_public_loading_works_away_from_repo_without_development_imports(
    tmp_path: Path,
) -> None:
    script = """
import sys
from marl_battlegrounds import _tdm_assets
from marl_battlegrounds.tasks import (
    load_tdm_scenario, make_canonical_team_deathmatch_evaluation_config,
)
assert make_canonical_team_deathmatch_evaluation_config(map_id=47).max_steps == 300
assert load_tdm_scenario(3).info.horizon == 10
assert len(_tdm_assets.map_history()) == 52
assert _tdm_assets.map_id_aliases().maps[0].current_map_id == 12
assert not any(name == 'scripts' or name.startswith('scripts.') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", script], cwd=tmp_path, check=True)
