"""Approved content integrity and public Team Deathmatch construction contracts."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pydantic import ValidationError

from marl_battlegrounds import _tdm_assets
from marl_battlegrounds.core.config import validate_product_env_config
from marl_battlegrounds.core.env import initialize_scenario_state, reset, step
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
    EnvState,
    Info,
    Observation,
    Reward,
)
from marl_battlegrounds.tasks import (
    CANONICAL_TDM_EVALUATION_MAP_IDS,
    AgentClassName,
    list_tdm_maps,
    list_tdm_scenarios,
    load_tdm_scenario,
    make_canonical_team_deathmatch_evaluation_config,
    make_standard_team_deathmatch_config,
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
    assert maps[33].source.revision == 4
    assert scenarios[2].source.revision == 25
    assert scenarios[2].approved_source.revision == 24
    assert scenarios[2].source.semantic_digest == (
        scenarios[2].approved_source.semantic_digest
    )
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
    assert config.max_steps == 300
    assert config.ordinary_movement_distance_scale == 1.0
    assert config.spawn_shield_duration_steps == 3
    assert config.spawn_shield_movement_speed == 2.0
    np.testing.assert_array_equal(config.team_respawn_wave_period_step_count, (5, 5))
    state, _, _, _ = reset(config, jax.random.key(9))
    positions = np.asarray(state.agent_positions).reshape(2, 5, 2)
    np.testing.assert_array_equal(positions[0, :, 1], positions[1, :, 1])
    np.testing.assert_array_equal(positions[0, :, 0] + positions[1, :, 0], 20)


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
from marl_battlegrounds.tasks import (
    load_tdm_scenario, make_canonical_team_deathmatch_evaluation_config,
)
assert make_canonical_team_deathmatch_evaluation_config(map_id=47).max_steps == 300
assert load_tdm_scenario(3).info.horizon == 10
assert not any(name == 'scripts' or name.startswith('scripts.') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", script], cwd=tmp_path, check=True, timeout=60)
