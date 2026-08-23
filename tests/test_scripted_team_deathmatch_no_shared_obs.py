"""Deterministic contracts for the TDM NoSharedObs scripted policy."""
# pyright: reportPrivateUsage=false, reportUnknownArgumentType=false
# pyright: reportUnknownLambdaType=false, reportUnknownVariableType=false

import ast
import inspect
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from tests.evaluation_fixtures import evaluation_env_config

import marl_battlegrounds.policies.scripted as scripted_module
import marl_battlegrounds.policies.scripted.no_shared_obs as adapter_module
import marl_battlegrounds.policies.scripted.team_deathmatch as policy_module
from marl_battlegrounds.core.env import reset, step
from marl_battlegrounds.core.types import (
    AGENT_FEATURE_ACTIVE,
    AGENT_FEATURE_ALIVE,
    AGENT_FEATURE_ANTI_HEAL_ROGUE_POISON_DURATION,
    AGENT_FEATURE_ANTI_HEAL_ROGUE_POISON_MULTIPLIER,
    AGENT_FEATURE_BASIC_INTERACTION_RADIUS,
    AGENT_FEATURE_CAPABILITY_BASIC_DAMAGE,
    AGENT_FEATURE_CAPABILITY_BASIC_HEALING,
    AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_BURST_MULTIPLIER,
    AGENT_FEATURE_CAPABILITY_SLOW_WARRIOR_CHARGE_DURATION,
    AGENT_FEATURE_CAPABILITY_SLOW_WARRIOR_CHARGE_MULTIPLIER,
    AGENT_FEATURE_CAPABILITY_STUN_WARRIOR_CHARGE_DURATION,
    AGENT_FEATURE_CAPABILITY_ULTIMATE_DAMAGE,
    AGENT_FEATURE_CLASS_ID,
    AGENT_FEATURE_CURRENT_HEALTH,
    AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER,
    AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_BURST_DURATION,
    AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER,
    AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED,
    AGENT_FEATURE_MAX_HEALTH,
    AGENT_FEATURE_RADIUS,
    AGENT_FEATURE_SLOW_HUNTER_BASIC_DURATION,
    AGENT_FEATURE_SLOW_HUNTER_BASIC_MULTIPLIER,
    AGENT_FEATURE_SLOW_WARRIOR_CHARGE_DURATION,
    AGENT_FEATURE_SLOW_WARRIOR_CHARGE_MULTIPLIER,
    AGENT_FEATURE_STEPS_UNTIL_OUT_OF_COMBAT,
    AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION,
    AGENT_FEATURE_STUN_WARRIOR_CHARGE_DURATION,
    AGENT_FEATURE_ULTIMATE_INTERACTION_RADIUS,
    AGENT_FEATURE_X,
    AGENT_FEATURE_Y,
    CONTEXT_FEATURE_MAP_HEIGHT,
    CONTEXT_FEATURE_MAP_WIDTH,
    HUNTER_CLASS_ID,
    MAGE_CLASS_ID,
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    MOVE_EAST,
    MOVE_STAY,
    MOVE_WEST,
    NUM_MOVE_ACTIONS,
    NUM_TARGET_ACTIONS,
    NUM_ULTIMATE_ACTIONS,
    OBSTACLE_FEATURE_ACTIVE,
    OBSTACLE_FEATURE_RADIUS,
    OBSTACLE_FEATURE_TYPE,
    OBSTACLE_FEATURE_X,
    OBSTACLE_FEATURE_Y,
    OBSTACLE_TYPE_PILLAR,
    PRIEST_CLASS_ID,
    ROGUE_CLASS_ID,
    TASK_MODE_TDM,
    TEAM_A_ID,
    TEAM_B_ID,
    WARRIOR_CLASS_ID,
    ActionMask,
    Observation,
)
from marl_battlegrounds.policies.actor import (
    ActorAction,
    build_joint_action_from_actor_actions,
)
from marl_battlegrounds.policies.no_shared_obs import (
    execute_no_shared_obs_team_policy,
)
from marl_battlegrounds.policies.scripted import (
    NO_SHARED_OBS_ADAPTER_ID,
    NO_SHARED_OBS_ADAPTER_VERSION,
    NUMERIC_PROFILE_ID,
    POLICY_ID,
    POLICY_SEMANTIC_VERSION,
    SEMANTIC_PROFILE_ID,
    TASK_HEAD_VERSION,
    TEAM_DEATHMATCH_PROFILE,
    TRACE_ONTOLOGY_VERSION,
    decide_team_deathmatch_no_shared_obs,
    team_deathmatch_no_shared_obs_policy,
)


def _assert_tree_arrays_exact(actual: object, expected: object) -> None:
    assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(
        expected
    )
    for actual_leaf, expected_leaf in zip(
        jax.tree_util.tree_leaves(actual),
        jax.tree_util.tree_leaves(expected),
        strict=True,
    ):
        actual_array = np.asarray(actual_leaf)
        expected_array = np.asarray(expected_leaf)
        assert actual_array.dtype == expected_array.dtype
        assert actual_array.shape == expected_array.shape
        np.testing.assert_array_equal(actual_array, expected_array)


def _scalar_policy_inputs(
    global_slot: int = 0,
    *,
    team_sizes: tuple[int, int] = (5, 5),
) -> tuple[Observation, ActionMask]:
    config = evaluation_env_config(
        team_sizes=team_sizes,
        task_mode=TASK_MODE_TDM,
        team_deathmatch_score_threshold=3,
    )
    _, observation, action_mask, _ = reset(config, jax.random.key(0))
    return (
        cast(Observation, jax.tree.map(lambda leaf: leaf[global_slot], observation)),
        cast(ActionMask, jax.tree.map(lambda leaf: leaf[global_slot], action_mask)),
    )


@pytest.fixture(scope="module")
def class_rows() -> dict[int, Array]:
    """Return one canonical self-observation row for every playable class."""
    config = evaluation_env_config(
        team_sizes=(5, 5),
        task_mode=TASK_MODE_TDM,
        team_deathmatch_score_threshold=3,
    )
    _, observation, _, _ = reset(config, jax.random.key(1))
    rows: dict[int, Array] = {}
    for row in observation.self_features:
        class_id = int(row[AGENT_FEATURE_CLASS_ID])
        rows.setdefault(class_id, row)
    assert set(rows) == {1, 2, 3, 4, 5}
    return rows


def _placed_row(
    row: Array,
    x: float,
    y: float,
    *,
    health: float | None = None,
) -> Array:
    placed = row.at[AGENT_FEATURE_X].set(jnp.float32(x))
    placed = placed.at[AGENT_FEATURE_Y].set(jnp.float32(y))
    if health is not None:
        placed = placed.at[AGENT_FEATURE_CURRENT_HEALTH].set(jnp.float32(health))
    return placed


def _policy_facts(
    focal_row: Array,
    *,
    allies: tuple[Array, ...] = (),
    enemies: tuple[Array, ...] = (),
) -> policy_module.PolicyFacts:
    zero_row = jnp.zeros_like(focal_row)
    ally_rows = (focal_row, *allies)
    enemy_rows = enemies
    ally_features = jnp.stack(
        (*ally_rows, *((zero_row,) * (MAX_AGENTS_PER_TEAM - len(ally_rows))))
    )
    enemy_features = jnp.stack(
        (*enemy_rows, *((zero_row,) * (MAX_AGENTS_PER_TEAM - len(enemy_rows))))
    )
    own_active = jnp.arange(MAX_AGENTS_PER_TEAM) < len(ally_rows)
    enemy_active = jnp.arange(MAX_AGENTS_PER_TEAM) < len(enemy_rows)
    own_class_ids = ally_features[:, AGENT_FEATURE_CLASS_ID].astype(jnp.int32)
    enemy_class_ids = enemy_features[:, AGENT_FEATURE_CLASS_ID].astype(jnp.int32)

    return policy_module.PolicyFacts(
        focal_features=focal_row,
        ally_features=ally_features,
        enemy_features=enemy_features,
        obstacles=jnp.zeros((16, 8), dtype=jnp.float32),
        ally_visible=own_active,
        enemy_visible=enemy_active,
        own_active=own_active,
        enemy_active=enemy_active,
        own_alive=own_active,
        enemy_alive=enemy_active,
        own_spawn_shields=jnp.zeros((5,), dtype=jnp.int32),
        enemy_spawn_shields=jnp.zeros((5,), dtype=jnp.int32),
        own_class_ids=own_class_ids,
        enemy_class_ids=enemy_class_ids,
        own_configured_count=jnp.asarray(len(ally_rows), dtype=jnp.int32),
        enemy_configured_count=jnp.asarray(len(enemy_rows), dtype=jnp.int32),
        map_width=jnp.asarray(20.0, dtype=jnp.float32),
        map_height=jnp.asarray(12.0, dtype=jnp.float32),
        focal_shield_state=jnp.asarray(
            policy_module.FOCAL_SHIELD_KNOWN_FALSE, dtype=jnp.int32
        ),
    )


def _observation_from_facts(facts: policy_module.PolicyFacts) -> Observation:
    observation, _ = _scalar_policy_inputs()
    lifecycle = observation.spawn_lifecycle
    active = (
        lifecycle.active_mask_by_agent_by_team.at[0]
        .set(facts.own_active)
        .at[1]
        .set(facts.enemy_active)
    )
    alive = (
        lifecycle.alive_mask_by_agent_by_team.at[0]
        .set(facts.own_alive)
        .at[1]
        .set(facts.enemy_alive)
    )
    shields = (
        lifecycle.spawn_shield_actual_durations_by_agent_by_team.at[0]
        .set(facts.own_spawn_shields)
        .at[1]
        .set(facts.enemy_spawn_shields)
    )
    class_ids = (
        lifecycle.class_ids_by_agent_by_team.at[0]
        .set(facts.own_class_ids)
        .at[1]
        .set(facts.enemy_class_ids)
    )
    context = observation.context_features.at[CONTEXT_FEATURE_MAP_WIDTH].set(
        facts.map_width
    )
    context = context.at[CONTEXT_FEATURE_MAP_HEIGHT].set(facts.map_height)
    return observation._replace(
        self_features=facts.focal_features,
        ally_unit_features=facts.ally_features,
        enemy_unit_features=facts.enemy_features,
        map_obstacle_features=facts.obstacles,
        ally_visibility_mask=facts.ally_visible,
        enemy_visibility_mask=facts.enemy_visible,
        context_features=context,
        spawn_lifecycle=lifecycle._replace(
            active_mask_by_agent_by_team=active,
            alive_mask_by_agent_by_team=alive,
            spawn_shield_actual_durations_by_agent_by_team=shields,
            class_ids_by_agent_by_team=class_ids,
        ),
    )


def _local_action_mask(
    *,
    move_support: tuple[int, ...] = (MOVE_STAY,),
    combat_support: tuple[tuple[int, int], ...] = ((0, 0),),
) -> ActionMask:
    move_mask = jnp.zeros((NUM_MOVE_ACTIONS,), dtype=jnp.bool_)
    for move_action in move_support:
        move_mask = move_mask.at[move_action].set(True)

    joint_mask = jnp.zeros((NUM_TARGET_ACTIONS, NUM_ULTIMATE_ACTIONS), dtype=jnp.bool_)
    for target_action, ultimate_action in combat_support:
        joint_mask = joint_mask.at[target_action, ultimate_action].set(True)

    return ActionMask(
        move_mask=move_mask,
        select_target_mask=jnp.any(joint_mask, axis=1),
        use_ultimate_mask=jnp.any(joint_mask, axis=0),
        select_target_use_ultimate_joint_mask=joint_mask,
    )


def _decide_facts(
    facts: policy_module.PolicyFacts,
    *,
    combat_support: tuple[tuple[int, int], ...],
    move_support: tuple[int, ...] = (MOVE_STAY,),
    seed: int = 0,
) -> tuple[ActorAction, policy_module.ScriptedTrace]:
    return policy_module.decide_team_deathmatch(
        facts,
        _local_action_mask(
            move_support=move_support,
            combat_support=combat_support,
        ),
        jax.random.key(seed),
    )


def _assert_scalar_policy_input_contract(
    observation: Observation,
    action_mask: ActionMask,
) -> None:
    """Test-owned malformed-input diagnostic for the scalar policy seam."""
    assert observation.self_features.shape == (58,), "self_features shape"
    assert observation.ally_unit_features.shape == (5, 58), "ally feature shape"
    assert observation.enemy_unit_features.shape == (5, 58), "enemy feature shape"
    assert observation.map_obstacle_features.shape == (16, 8), "obstacle shape"
    assert observation.ally_visibility_mask.shape == (5,), "ally visibility shape"
    assert observation.enemy_visibility_mask.shape == (5,), "enemy visibility shape"
    assert action_mask.move_mask.shape == (9,), "move support shape"
    assert action_mask.select_target_use_ultimate_joint_mask.shape == (
        11,
        2,
    ), "combat support shape"
    assert action_mask.move_mask.dtype == jnp.bool_, "move support dtype"
    assert action_mask.select_target_use_ultimate_joint_mask.dtype == jnp.bool_, (
        "combat support dtype"
    )
    assert bool(jnp.any(action_mask.move_mask)), "all-false movement support"
    assert bool(jnp.any(action_mask.select_target_use_ultimate_joint_mask)), (
        "all-false combat support"
    )


def _assert_nonempty_selection_support(
    trace: policy_module.ScriptedTrace,
) -> None:
    assert int(trace.combat_peer_count) > 0, "empty combat selection support"
    assert int(trace.movement_peer_count) > 0, "empty movement selection support"


def _python_profile_values(value: object) -> Iterator[object]:
    if isinstance(value, tuple):
        for item in value:
            yield from _python_profile_values(item)
    else:
        yield value


def test_profile_and_source_identity_are_complete_immutable_host_values() -> None:
    assert set(adapter_module.__all__) == {
        "NO_SHARED_OBS_ADAPTER_ID",
        "NO_SHARED_OBS_ADAPTER_VERSION",
        "decide_team_deathmatch_no_shared_obs",
        "team_deathmatch_no_shared_obs_policy",
    }
    assert set(scripted_module.__all__) == {
        "NO_SHARED_OBS_ADAPTER_ID",
        "NO_SHARED_OBS_ADAPTER_VERSION",
        "NUMERIC_PROFILE_ID",
        "POLICY_ID",
        "POLICY_SEMANTIC_VERSION",
        "SEMANTIC_PROFILE_ID",
        "TASK_HEAD_VERSION",
        "TEAM_DEATHMATCH_PROFILE",
        "TRACE_ONTOLOGY_VERSION",
        "decide_team_deathmatch_no_shared_obs",
        "team_deathmatch_no_shared_obs_policy",
    }
    assert POLICY_ID == "scripted/team_deathmatch"
    assert POLICY_SEMANTIC_VERSION == 1
    assert TASK_HEAD_VERSION == 1
    assert NO_SHARED_OBS_ADAPTER_ID == "no_shared_obs"
    assert NO_SHARED_OBS_ADAPTER_VERSION == 1
    assert SEMANTIC_PROFILE_ID == "scripted-common-v1"
    assert NUMERIC_PROFILE_ID == "scripted-common-f32-v1"
    assert TRACE_ONTOLOGY_VERSION == 1
    assert TEAM_DEATHMATCH_PROFILE == (
        (
            (0.30, 0.30, 0.15, 0.15, 0.10),
            (0.15, 0.20, 0.35, 0.20, 0.10),
            (0.20, 0.20, 0.20, 0.30, 0.10),
            (0.25, 0.35, 0.10, 0.10, 0.20),
            (0.30, 0.30, 0.25, 0.10, 0.05),
        ),
        (
            (0.40, 0.20, 0.15),
            (0.20, 0.40, 0.10),
            (0.45, 0.15, 0.15),
            (0.45, 0.05, 0.20),
            (0.20, 0.40, 0.15),
        ),
        ((0.75, 0.95), (0.35, 0.80), (0.80, 0.95), (0.25, 0.75), (0.60, 0.90)),
        (0.50, 1.00),
        0.25,
        0.20,
        (0.50, 0.50),
        (0.50, 0.50),
        0.15,
        0.50,
        0.50,
        5.0,
        0.50,
        0.12,
        1.25,
        (0.06, 0.00, 0.02, 0.04, 0.08),
        (
            (0, 0, 0, 0, 1),
            (0, 0, 0, 1, 0),
            (0, 1, 0, 0, 0),
            (1, 0, 0, 0, 1),
            (0, 0, 1, 0, 0),
        ),
        (
            (0, 0, 0, 1, 0),
            (0, 0, 1, 0, 0),
            (0, 0, 0, 0, 1),
            (0, 1, 0, 0, 0),
            (0, 0, 0, 1, 0),
        ),
        0.50,
        0.25,
        2.0,
        (1.00, 0.75, 0.50),
        0.05,
        0.30,
        0.05,
        0.10,
        0.10,
        0.70,
        0.03,
        2.0,
        2,
        80.0,
        1.0,
        3.0,
        20.0,
        2,
        30.0,
        1.0,
        0.0,
    )
    assert callable(team_deathmatch_no_shared_obs_policy)
    assert isinstance(hash(team_deathmatch_no_shared_obs_policy), int)
    assert team_deathmatch_no_shared_obs_policy.profile is TEAM_DEATHMATCH_PROFILE

    assert isinstance(hash(TEAM_DEATHMATCH_PROFILE), int)
    assert all(
        isinstance(value, (bool, int, float))
        for value in _python_profile_values(TEAM_DEATHMATCH_PROFILE)
    )
    with pytest.raises(AttributeError):
        TEAM_DEATHMATCH_PROFILE.excess_weight = 0.0  # type: ignore[misc]


def test_bound_profile_changes_behavior_without_mutating_the_canonical_policy(
    class_rows: dict[int, Array],
) -> None:
    focal = _placed_row(class_rows[MAGE_CLASS_ID], 5.0, 5.0)
    near = _placed_row(class_rows[WARRIOR_CLASS_ID], 7.0, 5.0, health=100.0)
    far = _placed_row(class_rows[WARRIOR_CLASS_ID], 12.0, 5.0, health=100.0)
    observation = _observation_from_facts(_policy_facts(focal, enemies=(near, far)))
    action_mask = _local_action_mask(combat_support=((0, 0), (0, 1), (6, 0), (7, 0)))
    key = jax.random.key(101)
    custom_profile = TEAM_DEATHMATCH_PROFILE._replace(mage_burst_crowd=1)
    custom_policy = team_deathmatch_no_shared_obs_policy._replace(
        profile=custom_profile
    )

    canonical_before = team_deathmatch_no_shared_obs_policy(
        observation, action_mask, key
    )
    canonical_traced, canonical_trace = decide_team_deathmatch_no_shared_obs(
        observation, action_mask, key
    )
    custom_action = custom_policy(observation, action_mask, key)
    custom_traced, custom_trace = (
        adapter_module._decide_team_deathmatch_no_shared_obs_with_profile(
            observation,
            action_mask,
            key,
            custom_policy.profile,
        )
    )
    canonical_after = team_deathmatch_no_shared_obs_policy(
        observation, action_mask, key
    )

    _assert_tree_arrays_exact(canonical_before, canonical_traced)
    _assert_tree_arrays_exact(canonical_after, canonical_before)
    _assert_tree_arrays_exact(custom_action, custom_traced)
    assert int(canonical_before.use_ultimate) == 0
    assert int(canonical_trace.combat_reason_id) != policy_module.MAGE_BURST_TRIGGER
    assert int(custom_action.use_ultimate) == 1
    assert int(custom_trace.combat_reason_id) == policy_module.MAGE_BURST_TRIGGER
    assert team_deathmatch_no_shared_obs_policy.profile is TEAM_DEATHMATCH_PROFILE


def test_test_owned_contract_diagnostic_names_the_malformed_support() -> None:
    observation, action_mask = _scalar_policy_inputs()
    _assert_scalar_policy_input_contract(observation, action_mask)

    with pytest.raises(AssertionError, match="all-false movement support"):
        _assert_scalar_policy_input_contract(
            observation,
            action_mask._replace(move_mask=jnp.zeros((9,), dtype=jnp.bool_)),
        )
    with pytest.raises(AssertionError, match="combat support dtype"):
        _assert_scalar_policy_input_contract(
            observation,
            action_mask._replace(
                select_target_use_ultimate_joint_mask=(
                    action_mask.select_target_use_ultimate_joint_mask.astype(jnp.int32)
                )
            ),
        )


def test_adapter_selects_only_authorized_fixed_shape_facts() -> None:
    observation, action_mask = _scalar_policy_inputs()
    facts = adapter_module._build_policy_facts(observation, action_mask)
    lifecycle = observation.spawn_lifecycle

    _assert_tree_arrays_exact(facts.focal_features, observation.self_features)
    _assert_tree_arrays_exact(facts.ally_features, observation.ally_unit_features)
    _assert_tree_arrays_exact(facts.enemy_features, observation.enemy_unit_features)
    _assert_tree_arrays_exact(facts.obstacles, observation.map_obstacle_features)
    _assert_tree_arrays_exact(facts.ally_visible, observation.ally_visibility_mask)
    _assert_tree_arrays_exact(facts.enemy_visible, observation.enemy_visibility_mask)
    _assert_tree_arrays_exact(
        facts.own_class_ids, lifecycle.class_ids_by_agent_by_team[0]
    )
    _assert_tree_arrays_exact(
        facts.enemy_class_ids, lifecycle.class_ids_by_agent_by_team[1]
    )
    assert facts.focal_features.shape == (58,)
    assert facts.ally_features.shape == (5, 58)
    assert facts.enemy_features.shape == (5, 58)
    assert facts.obstacles.shape == (16, 8)
    assert facts.own_active.shape == (5,)
    assert facts.enemy_active.shape == (5,)
    assert facts.own_alive.shape == (5,)
    assert facts.enemy_alive.shape == (5,)
    assert facts.own_spawn_shields.shape == (5,)
    assert facts.enemy_spawn_shields.shape == (5,)
    assert facts.own_class_ids.dtype == jnp.int32
    assert facts.enemy_class_ids.dtype == jnp.int32
    assert facts.own_configured_count.shape == ()
    assert facts.own_configured_count.dtype == jnp.int32
    assert facts.enemy_configured_count.shape == ()
    assert facts.enemy_configured_count.dtype == jnp.int32
    assert facts.map_width == observation.context_features[CONTEXT_FEATURE_MAP_WIDTH]
    assert facts.map_height == observation.context_features[CONTEXT_FEATURE_MAP_HEIGHT]
    assert facts.map_width.dtype == jnp.float32
    assert facts.map_height.dtype == jnp.float32
    assert facts.focal_shield_state.shape == ()
    assert facts.focal_shield_state.dtype == jnp.int32


@pytest.mark.parametrize(
    ("eligible", "shield_durations", "combat_support", "expected_state"),
    (
        (False, (0, 0, 0, 0, 0), ((0, 0),), policy_module.FOCAL_SHIELD_KNOWN_FALSE),
        (True, (2, 2, 2, 2, 2), ((0, 0),), policy_module.FOCAL_SHIELD_KNOWN_TRUE),
        (True, (2, 0, 2, 0, 2), ((0, 0),), policy_module.FOCAL_SHIELD_UNKNOWN),
        (True, (0, 0, 0, 0, 0), ((0, 0),), policy_module.FOCAL_SHIELD_KNOWN_FALSE),
        (
            True,
            (2, 0, 2, 0, 2),
            ((0, 0), (6, 0)),
            policy_module.FOCAL_SHIELD_KNOWN_FALSE,
        ),
    ),
)
def test_focal_shield_state_uses_only_lifecycle_and_exact_support_proofs(
    eligible: bool,
    shield_durations: tuple[int, ...],
    combat_support: tuple[tuple[int, int], ...],
    expected_state: int,
) -> None:
    observation, _ = _scalar_policy_inputs()
    self_features = observation.self_features.at[AGENT_FEATURE_ACTIVE].set(
        float(eligible)
    )
    self_features = self_features.at[AGENT_FEATURE_ALIVE].set(float(eligible))
    lifecycle = observation.spawn_lifecycle._replace(
        active_mask_by_agent_by_team=(
            observation.spawn_lifecycle.active_mask_by_agent_by_team.at[0].set(True)
        ),
        alive_mask_by_agent_by_team=(
            observation.spawn_lifecycle.alive_mask_by_agent_by_team.at[0].set(True)
        ),
        spawn_shield_actual_durations_by_agent_by_team=(
            observation.spawn_lifecycle.spawn_shield_actual_durations_by_agent_by_team.at[
                0
            ].set(jnp.asarray(shield_durations, dtype=jnp.int32))
        ),
    )
    changed_observation = observation._replace(
        self_features=self_features,
        spawn_lifecycle=lifecycle,
    )
    action_mask = _local_action_mask(combat_support=combat_support)

    actual = adapter_module._focal_shield_state(changed_observation, action_mask)
    assert int(actual) == expected_state


def test_focal_shield_tristate_controls_risk_and_aura_without_guessing_duration(
    class_rows: dict[int, Array],
) -> None:
    focal = _placed_row(class_rows[HUNTER_CLASS_ID], 5.0, 5.0, health=100.0)
    mage_ally = _placed_row(class_rows[MAGE_CLASS_ID], 5.5, 5.0)
    enemy = _placed_row(class_rows[MAGE_CLASS_ID], 7.0, 5.0)
    enemy = enemy.at[AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER].set(1.0)
    enemy = enemy.at[AGENT_FEATURE_CAPABILITY_BASIC_DAMAGE].set(8.0)
    enemy = enemy.at[AGENT_FEATURE_CAPABILITY_ULTIMATE_DAMAGE].set(0.0)
    base = _policy_facts(focal, allies=(mage_ally,), enemies=(enemy,))

    traces = {}
    for state in (
        policy_module.FOCAL_SHIELD_KNOWN_FALSE,
        policy_module.FOCAL_SHIELD_KNOWN_TRUE,
        policy_module.FOCAL_SHIELD_UNKNOWN,
    ):
        _, traces[state] = _decide_facts(
            base._replace(focal_shield_state=jnp.asarray(state, dtype=jnp.int32)),
            combat_support=((0, 0),),
        )

    known_false = traces[policy_module.FOCAL_SHIELD_KNOWN_FALSE]
    known_true = traces[policy_module.FOCAL_SHIELD_KNOWN_TRUE]
    unknown = traces[policy_module.FOCAL_SHIELD_UNKNOWN]
    assert known_false.movement_selection_basis_components[3] > 0.0
    assert known_true.movement_selection_basis_components[3] == 0.0
    assert unknown.movement_selection_basis_components[3] == 0.0
    assert known_false.movement_selection_basis_components[4] < 0.0
    assert known_true.movement_selection_basis_components[4] == 0.0
    assert (
        unknown.movement_selection_basis_components[4]
        == known_false.movement_selection_basis_components[4]
    )


@pytest.mark.parametrize("invalid_stun_duration", (-1.0, np.nan, np.inf))
def test_invalid_ally_stun_duration_neutralizes_mage_aura_damage_value(
    class_rows: dict[int, Array],
    invalid_stun_duration: float,
) -> None:
    focal = _placed_row(class_rows[HUNTER_CLASS_ID], 5.0, 5.0, health=100.0)
    mage_ally = _placed_row(class_rows[MAGE_CLASS_ID], 5.5, 5.0)
    enemy = _placed_row(class_rows[WARRIOR_CLASS_ID], 7.0, 5.0, health=100.0)
    valid_facts = _policy_facts(focal, allies=(mage_ally,), enemies=(enemy,))
    invalid_ally = mage_ally.at[AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION].set(
        invalid_stun_duration
    )
    invalid_facts = _policy_facts(
        focal,
        allies=(invalid_ally,),
        enemies=(enemy,),
    )

    valid_value = policy_module._mage_ally_damage_value(
        valid_facts, jnp.asarray(1, dtype=jnp.int32)
    )
    invalid_value = policy_module._mage_ally_damage_value(
        invalid_facts, jnp.asarray(1, dtype=jnp.int32)
    )

    assert valid_value > 0.0
    assert invalid_value == jnp.float32(0.0)


def test_counters_add_offense_while_countered_by_refines_only_soft_risk(
    class_rows: dict[int, Array],
) -> None:
    focal = _placed_row(class_rows[MAGE_CLASS_ID], 5.0, 5.0, health=80.0)
    focal = focal.at[AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER].set(1.0)
    priest = _placed_row(class_rows[PRIEST_CLASS_ID], 7.0, 5.0, health=100.0)
    priest = priest.at[AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION].set(1.0)
    _, counter_trace = _decide_facts(
        _policy_facts(focal, enemies=(priest,)),
        combat_support=((6, 0),),
    )
    assert counter_trace.combat_selection_basis_components[6] == jnp.float32(0.12)

    residual_components = []
    for enemy_class in (MAGE_CLASS_ID, ROGUE_CLASS_ID):
        enemy = _placed_row(class_rows[enemy_class], 7.0, 5.0)
        enemy = enemy.at[AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER].set(
            1.0
        )
        enemy = enemy.at[AGENT_FEATURE_CAPABILITY_BASIC_DAMAGE].set(9.6)
        enemy = enemy.at[AGENT_FEATURE_CAPABILITY_ULTIMATE_DAMAGE].set(0.0)
        enemy = enemy.at[AGENT_FEATURE_BASIC_INTERACTION_RADIUS].set(3.0)
        _, trace = _decide_facts(
            _policy_facts(focal, enemies=(enemy,)),
            combat_support=((0, 0),),
        )
        assert not bool(trace.fired_guards[7])
        residual_components.append(trace.movement_selection_basis_components[4])

    np.testing.assert_allclose(
        np.asarray(residual_components),
        np.asarray((-0.18, -0.225), dtype=np.float32),
        rtol=0.0,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    ("enemy_positions", "first_enemy_health", "expected_ultimate"),
    (
        ((), 100.0, 0),
        (((7.0, 5.0),), 100.0, 1),
        (((7.0, 5.0),), 10.0, 0),
        (((7.0, 5.0), (12.0, 5.0)), 100.0, 0),
        (((7.0, 5.0), (7.0, 6.0)), 100.0, 1),
    ),
)
def test_mage_burst_uses_the_locked_configured_crowd_and_covering_boundaries(
    class_rows: dict[int, Array],
    enemy_positions: tuple[tuple[float, float], ...],
    first_enemy_health: float,
    expected_ultimate: int,
) -> None:
    focal = _placed_row(class_rows[MAGE_CLASS_ID], 5.0, 5.0)
    enemies = tuple(
        _placed_row(
            class_rows[WARRIOR_CLASS_ID],
            x,
            y,
            health=first_enemy_health if index == 0 else 100.0,
        )
        for index, (x, y) in enumerate(enemy_positions)
    )
    facts = _policy_facts(focal, enemies=enemies)
    combat_support = (
        (0, 0),
        (0, 1),
        *((6 + index, 0) for index in range(len(enemies))),
    )

    action, trace = _decide_facts(facts, combat_support=combat_support)

    assert int(action.use_ultimate) == expected_ultimate
    if expected_ultimate:
        assert int(action.select_target) == 0
        assert int(trace.combat_reason_id) == policy_module.MAGE_BURST_TRIGGER
        assert bool(trace.fired_guards[1])
        assert trace.combat_selection_basis_value == jnp.float32(0.0)
        assert bool(jnp.all(trace.combat_selection_basis_components == 0.0))
    else:
        assert not bool(trace.fired_guards[1])


def test_hidden_covering_basic_does_not_suppress_mage_burst(
    class_rows: dict[int, Array],
) -> None:
    focal = _placed_row(class_rows[MAGE_CLASS_ID], 5.0, 5.0)
    enemies = (
        _placed_row(class_rows[WARRIOR_CLASS_ID], 7.0, 5.0, health=100.0),
        _placed_row(class_rows[WARRIOR_CLASS_ID], 7.0, 6.0, health=100.0),
        _placed_row(class_rows[WARRIOR_CLASS_ID], 6.0, 5.0, health=1.0),
    )
    facts = _policy_facts(focal, enemies=enemies)
    facts = facts._replace(enemy_visible=facts.enemy_visible.at[2].set(False))

    action, trace = _decide_facts(
        facts,
        combat_support=((0, 0), (0, 1), (8, 0)),
    )

    assert (int(action.select_target), int(action.use_ultimate)) == (0, 1)
    assert int(trace.combat_reason_id) == policy_module.MAGE_BURST_TRIGGER


def test_invalid_zero_damage_basic_does_not_suppress_mage_burst(
    class_rows: dict[int, Array],
) -> None:
    huge = jnp.float32(3e38)
    focal = _placed_row(class_rows[MAGE_CLASS_ID], 5.0, 5.0)
    focal = focal.at[AGENT_FEATURE_CAPABILITY_BASIC_DAMAGE].set(huge)
    focal = focal.at[AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_BURST_DURATION].set(1.0)
    focal = focal.at[
        AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_BURST_MULTIPLIER
    ].set(2.0)
    focal = focal.at[AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER].set(2.0)
    enemy = _placed_row(class_rows[WARRIOR_CLASS_ID], 7.0, 5.0, health=0.0)
    enemy = enemy.at[AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER].set(2.0)

    action, trace = _decide_facts(
        _policy_facts(focal, enemies=(enemy,)),
        combat_support=((0, 1), (6, 0)),
    )

    assert (int(action.select_target), int(action.use_ultimate)) == (0, 1)
    assert int(trace.combat_reason_id) == policy_module.MAGE_BURST_TRIGGER


def test_visible_living_shielded_enemy_still_counts_for_crowd_triggers(
    class_rows: dict[int, Array],
) -> None:
    focal = _placed_row(class_rows[MAGE_CLASS_ID], 5.0, 5.0)
    enemy = _placed_row(class_rows[WARRIOR_CLASS_ID], 7.0, 5.0)
    facts = _policy_facts(focal, enemies=(enemy,))
    facts = facts._replace(enemy_spawn_shields=facts.enemy_spawn_shields.at[0].set(2))

    in_range = policy_module._visible_enemies_in_focal_basic(facts)
    action, trace = _decide_facts(
        facts,
        combat_support=((0, 0), (0, 1)),
    )

    assert bool(in_range[0])
    assert (int(action.select_target), int(action.use_ultimate)) == (0, 1)
    assert int(trace.combat_reason_id) == policy_module.MAGE_BURST_TRIGGER


@pytest.mark.parametrize(
    ("trap_duration", "target_health", "expected_target", "guard_fired"),
    (
        (2.0, 100.0, 0, True),
        (1.0, 100.0, 6, False),
        (2.0, 10.0, 6, False),
        (np.inf, 100.0, 6, False),
    ),
)
def test_aged_trap_suppresses_only_nonlethal_positive_damage_above_one_tick(
    class_rows: dict[int, Array],
    trap_duration: float,
    target_health: float,
    expected_target: int,
    guard_fired: bool,
) -> None:
    focal = _placed_row(class_rows[MAGE_CLASS_ID], 5.0, 5.0)
    target = (
        _placed_row(class_rows[WARRIOR_CLASS_ID], 7.0, 5.0, health=target_health)
        .at[AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION]
        .set(trap_duration)
    )
    facts = _policy_facts(focal, enemies=(target,))

    action, trace = _decide_facts(
        facts,
        combat_support=((0, 0), (6, 0)),
    )

    assert int(action.select_target) == expected_target
    assert int(action.use_ultimate) == 0
    assert bool(trace.fired_guards[0]) is guard_fired


@pytest.mark.parametrize(
    ("focal_class", "use_ultimate"),
    (
        (MAGE_CLASS_ID, 0),
        (WARRIOR_CLASS_ID, 0),
        (WARRIOR_CLASS_ID, 1),
        (HUNTER_CLASS_ID, 0),
        (HUNTER_CLASS_ID, 1),
        (ROGUE_CLASS_ID, 0),
        (ROGUE_CLASS_ID, 1),
    ),
)
def test_aged_trap_suppresses_every_positive_damage_class_lane(
    class_rows: dict[int, Array],
    focal_class: int,
    use_ultimate: int,
) -> None:
    focal = _placed_row(
        class_rows[focal_class],
        5.0,
        5.0,
        health=100.0,
    )
    target = _placed_row(
        class_rows[WARRIOR_CLASS_ID],
        6.0,
        5.0,
        health=100.0,
    )
    target = target.at[AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION].set(2.0)

    action, trace = _decide_facts(
        _policy_facts(focal, enemies=(target,)),
        combat_support=((0, 0), (6, use_ultimate)),
    )

    assert (int(action.select_target), int(action.use_ultimate)) == (0, 0)
    assert bool(trace.fired_guards[0])


@pytest.mark.parametrize(
    ("focal_health", "trap_duration", "burst_duration", "expected_charge"),
    (
        (80.0, 1.0, 3.0, True),
        (79.0, 1.0, 3.0, False),
        (80.0, 2.0, 3.0, False),
        (80.0, 1.0, 2.0, False),
    ),
)
def test_warrior_charge_locks_health_trap_and_mage_burst_comparators(
    class_rows: dict[int, Array],
    focal_health: float,
    trap_duration: float,
    burst_duration: float,
    expected_charge: bool,
) -> None:
    focal = _placed_row(class_rows[WARRIOR_CLASS_ID], 5.0, 5.0, health=focal_health)
    target = _placed_row(class_rows[MAGE_CLASS_ID], 9.0, 5.0)
    target = target.at[AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION].set(trap_duration)
    target = target.at[AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_BURST_DURATION].set(
        burst_duration
    )
    facts = _policy_facts(focal, enemies=(target,))

    action, trace = _decide_facts(
        facts,
        combat_support=((0, 0), (6, 1)),
    )

    assert bool(int(action.use_ultimate)) is expected_charge
    assert bool(trace.fired_guards[2]) is expected_charge
    if expected_charge:
        assert int(action.select_target) == 6
        assert int(action.move) == MOVE_STAY
        assert int(trace.combat_reason_id) == policy_module.WARRIOR_CHARGE_TRIGGER
        assert int(trace.movement_reason_id) == policy_module.CHARGE_TO_STAY
        assert trace.movement_selection_basis_value == jnp.float32(0.0)
        assert bool(jnp.all(trace.movement_selection_basis_components == 0.0))
        expected_guards = jnp.zeros((10,), dtype=jnp.bool_)
        expected_guards = expected_guards.at[jnp.asarray((2, 6))].set(True)
        np.testing.assert_array_equal(
            np.asarray(trace.fired_guards), np.asarray(expected_guards)
        )
        np.testing.assert_allclose(
            np.asarray(trace.combat_selection_basis_value),
            np.asarray(jnp.sum(trace.combat_selection_basis_components)),
            rtol=0.0,
            atol=1e-6,
        )


@pytest.mark.parametrize("predicate", ("rogue", "priest", "mage", "absence"))
def test_warrior_keeps_the_four_charge_predicates_distinct_and_directional(
    class_rows: dict[int, Array],
    predicate: str,
) -> None:
    focal = _placed_row(class_rows[WARRIOR_CLASS_ID], 5.0, 5.0, health=80.0)
    allies: tuple[Array, ...] = ()
    extra_enemies: tuple[Array, ...] = ()
    if predicate == "rogue":
        target = _placed_row(class_rows[ROGUE_CLASS_ID], 9.0, 5.0)
        allies = (_placed_row(class_rows[PRIEST_CLASS_ID], 7.0, 5.0),)
        extra_enemies = (_placed_row(class_rows[MAGE_CLASS_ID], 15.0, 10.0),)
    elif predicate == "priest":
        target = _placed_row(class_rows[PRIEST_CLASS_ID], 9.0, 5.0)
        allies = (_placed_row(class_rows[WARRIOR_CLASS_ID], 7.0, 5.0),)
    elif predicate == "mage":
        target = (
            _placed_row(class_rows[MAGE_CLASS_ID], 9.0, 5.0)
            .at[AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_BURST_DURATION]
            .set(3.0)
        )
    else:
        target = _placed_row(class_rows[HUNTER_CLASS_ID], 9.0, 5.0)
    facts = _policy_facts(
        focal,
        allies=allies,
        enemies=(target, *extra_enemies),
    )

    action, trace = _decide_facts(
        facts,
        combat_support=((0, 0), (6, 1)),
    )

    assert int(action.select_target) == 6
    assert int(action.use_ultimate) == 1
    assert int(trace.combat_reason_id) == policy_module.WARRIOR_CHARGE_TRIGGER


@pytest.mark.parametrize(
    "predicate", ("rogue-needs-nonfocal", "priest-radius-direction")
)
def test_warrior_directional_charge_predicates_reject_reversed_or_focal_only_proof(
    class_rows: dict[int, Array],
    predicate: str,
) -> None:
    focal = _placed_row(class_rows[WARRIOR_CLASS_ID], 5.0, 5.0, health=80.0)
    if predicate == "rogue-needs-nonfocal":
        target = _placed_row(class_rows[ROGUE_CLASS_ID], 6.0, 5.0)
        allies: tuple[Array, ...] = ()
        extra_enemies = (_placed_row(class_rows[MAGE_CLASS_ID], 15.0, 10.0),)
    else:
        target = _placed_row(class_rows[PRIEST_CLASS_ID], 9.0, 5.0)
        allies = (_placed_row(class_rows[HUNTER_CLASS_ID], 5.75, 5.0),)
        extra_enemies = ()

    action, trace = _decide_facts(
        _policy_facts(focal, allies=allies, enemies=(target, *extra_enemies)),
        combat_support=((0, 0), (6, 1)),
    )

    assert (int(action.select_target), int(action.use_ultimate)) == (0, 0)
    assert not bool(trace.fired_guards[2])


@pytest.mark.parametrize("invalid_value", (-1.0, np.nan, np.inf))
@pytest.mark.parametrize("invalid_fact", ("focal-hp", "target-trap", "target-burst"))
def test_invalid_warrior_health_trap_or_burst_never_fires_charge(
    class_rows: dict[int, Array],
    invalid_fact: str,
    invalid_value: float,
) -> None:
    focal = _placed_row(class_rows[WARRIOR_CLASS_ID], 5.0, 5.0, health=80.0)
    target = _placed_row(class_rows[MAGE_CLASS_ID], 9.0, 5.0)
    target = target.at[AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_BURST_DURATION].set(3.0)
    if invalid_fact == "focal-hp":
        focal = focal.at[AGENT_FEATURE_CURRENT_HEALTH].set(invalid_value)
    elif invalid_fact == "target-trap":
        target = target.at[AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION].set(invalid_value)
    else:
        target = target.at[AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_BURST_DURATION].set(
            invalid_value
        )

    action, trace = _decide_facts(
        _policy_facts(focal, enemies=(target,)),
        combat_support=((0, 0), (6, 1)),
    )

    assert int(action.use_ultimate) == 0
    assert not bool(trace.fired_guards[2])


def test_hidden_target_payload_cannot_create_warrior_charge(
    class_rows: dict[int, Array],
) -> None:
    focal = _placed_row(class_rows[WARRIOR_CLASS_ID], 5.0, 5.0, health=80.0)
    target = _placed_row(class_rows[MAGE_CLASS_ID], 9.0, 5.0)
    target = target.at[AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_BURST_DURATION].set(3.0)
    facts = _policy_facts(focal, enemies=(target,))
    facts = facts._replace(enemy_visible=facts.enemy_visible.at[0].set(False))

    action, trace = _decide_facts(
        facts,
        combat_support=((0, 0), (6, 1)),
    )

    assert (int(action.select_target), int(action.use_ultimate)) == (0, 0)
    assert not bool(trace.fired_guards[2])


def test_charge_does_not_restore_masked_stay_and_diagnostic_names_empty_support(
    class_rows: dict[int, Array],
) -> None:
    focal = _placed_row(class_rows[WARRIOR_CLASS_ID], 5.0, 5.0, health=80.0)
    target = _placed_row(class_rows[HUNTER_CLASS_ID], 9.0, 5.0)
    facts = _policy_facts(focal, enemies=(target,))

    _, trace = _decide_facts(
        facts,
        combat_support=((6, 1),),
        move_support=(MOVE_EAST,),
    )

    assert int(trace.combat_reason_id) == policy_module.WARRIOR_CHARGE_TRIGGER
    assert int(trace.movement_reason_id) == policy_module.CHARGE_TO_STAY
    assert int(trace.movement_peer_count) == 0
    with pytest.raises(AssertionError, match="empty movement selection support"):
        _assert_nonempty_selection_support(trace)


def test_hunter_emergency_precedes_the_priest_crowd_branch_at_raw_hp_below_20(
    class_rows: dict[int, Array],
) -> None:
    focal = _placed_row(class_rows[HUNTER_CLASS_ID], 5.0, 5.0, health=19.0)
    priest = _placed_row(class_rows[PRIEST_CLASS_ID], 7.0, 5.0)
    mage = _placed_row(class_rows[MAGE_CLASS_ID], 7.0, 6.0)
    facts = _policy_facts(focal, enemies=(priest, mage))

    action, trace = _decide_facts(
        facts,
        combat_support=((0, 0), (6, 0), (6, 1), (7, 0), (7, 1)),
    )

    assert int(action.select_target) == 7
    assert int(action.use_ultimate) == 1
    assert int(trace.combat_reason_id) == policy_module.HUNTER_TRAP_EMERGENCY
    assert bool(trace.fired_guards[3])
    assert trace.combat_selection_basis_value > 0.0
    assert bool(jnp.all(trace.combat_selection_basis_components == 0.0))


@pytest.mark.parametrize(
    "invalid_capability",
    ("ultimate-damage", "ultimate-duration", "noncontrolling-slow"),
)
def test_invalid_or_noncontrolling_ultimate_effect_never_qualifies_hunter_emergency(
    class_rows: dict[int, Array],
    invalid_capability: str,
) -> None:
    focal = _placed_row(class_rows[HUNTER_CLASS_ID], 5.0, 5.0, health=19.0)
    enemy = _placed_row(class_rows[WARRIOR_CLASS_ID], 9.0, 5.0)
    enemy = enemy.at[AGENT_FEATURE_BASIC_INTERACTION_RADIUS].set(1.0)
    enemy = enemy.at[AGENT_FEATURE_ULTIMATE_INTERACTION_RADIUS].set(5.0)
    enemy = enemy.at[AGENT_FEATURE_CAPABILITY_ULTIMATE_DAMAGE].set(0.0)
    enemy = enemy.at[AGENT_FEATURE_CAPABILITY_SLOW_WARRIOR_CHARGE_DURATION].set(0.0)
    enemy = enemy.at[AGENT_FEATURE_CAPABILITY_SLOW_WARRIOR_CHARGE_MULTIPLIER].set(1.0)
    enemy = enemy.at[AGENT_FEATURE_CAPABILITY_STUN_WARRIOR_CHARGE_DURATION].set(0.0)
    if invalid_capability == "ultimate-damage":
        enemy = enemy.at[AGENT_FEATURE_CAPABILITY_ULTIMATE_DAMAGE].set(jnp.inf)
    elif invalid_capability == "ultimate-duration":
        enemy = enemy.at[AGENT_FEATURE_CAPABILITY_STUN_WARRIOR_CHARGE_DURATION].set(
            jnp.inf
        )
    else:
        enemy = enemy.at[AGENT_FEATURE_CAPABILITY_SLOW_WARRIOR_CHARGE_DURATION].set(3.0)

    action, trace = _decide_facts(
        _policy_facts(focal, enemies=(enemy,)),
        combat_support=((0, 0), (6, 1)),
    )

    assert int(action.use_ultimate) == 0
    assert not bool(trace.fired_guards[3])


def test_shielded_enemy_never_qualifies_hunter_emergency(
    class_rows: dict[int, Array],
) -> None:
    focal = _placed_row(class_rows[HUNTER_CLASS_ID], 5.0, 5.0, health=19.0)
    enemy = _placed_row(class_rows[WARRIOR_CLASS_ID], 7.0, 5.0)
    facts = _policy_facts(focal, enemies=(enemy,))._replace(
        enemy_spawn_shields=jnp.asarray((2, 0, 0, 0, 0), dtype=jnp.int32)
    )

    action, trace = _decide_facts(
        facts,
        combat_support=((0, 0), (6, 1)),
    )

    assert int(action.use_ultimate) == 0
    assert not bool(trace.fired_guards[3])


def test_shielded_priest_counts_in_crowd_but_cannot_receive_hunter_trap(
    class_rows: dict[int, Array],
) -> None:
    focal = _placed_row(class_rows[HUNTER_CLASS_ID], 5.0, 5.0, health=100.0)
    unshielded_priest = _placed_row(class_rows[PRIEST_CLASS_ID], 7.0, 5.0)
    shielded_priest = _placed_row(class_rows[PRIEST_CLASS_ID], 7.0, 6.0)
    facts = _policy_facts(
        focal,
        enemies=(unshielded_priest, shielded_priest),
    )._replace(enemy_spawn_shields=jnp.asarray((0, 2, 0, 0, 0), dtype=jnp.int32))

    action, trace = _decide_facts(
        facts,
        combat_support=((0, 0), (6, 1), (7, 1)),
    )

    assert (int(action.select_target), int(action.use_ultimate)) == (6, 1)
    assert int(trace.combat_reason_id) == policy_module.HUNTER_TRAP_PRIEST_CROWD


@pytest.mark.parametrize(
    ("focal_health", "expected_reason", "expected_target"),
    (
        (20.0, policy_module.HUNTER_TRAP_PRIEST_CROWD, 6),
        (19.0, policy_module.HUNTER_TRAP_EMERGENCY, 7),
    ),
)
def test_hunter_raw_hp_20_is_not_inside_the_emergency_branch(
    class_rows: dict[int, Array],
    focal_health: float,
    expected_reason: int,
    expected_target: int,
) -> None:
    focal = _placed_row(class_rows[HUNTER_CLASS_ID], 5.0, 5.0, health=focal_health)
    priest = _placed_row(class_rows[PRIEST_CLASS_ID], 7.0, 5.0)
    mage = _placed_row(class_rows[MAGE_CLASS_ID], 7.0, 6.0)
    facts = _policy_facts(focal, enemies=(priest, mage))

    action, trace = _decide_facts(
        facts,
        combat_support=((0, 0), (6, 0), (6, 1), (7, 0), (7, 1)),
    )

    assert int(action.select_target) == expected_target
    assert int(action.use_ultimate) == 1
    assert int(trace.combat_reason_id) == expected_reason


def test_hunter_no_priest_crowd_uses_configured_count_and_highest_raw_hp(
    class_rows: dict[int, Array],
) -> None:
    focal = _placed_row(class_rows[HUNTER_CLASS_ID], 5.0, 5.0, health=100.0)
    hunter = _placed_row(class_rows[HUNTER_CLASS_ID], 7.0, 5.0, health=70.0)
    warrior = _placed_row(class_rows[WARRIOR_CLASS_ID], 7.0, 6.0, health=80.0)
    facts = _policy_facts(focal, enemies=(hunter, warrior))

    action, trace = _decide_facts(
        facts,
        combat_support=((0, 0), (6, 0), (6, 1), (7, 0), (7, 1)),
    )

    assert int(action.select_target) == 7
    assert int(action.use_ultimate) == 1
    assert int(trace.combat_reason_id) == policy_module.HUNTER_TRAP_NO_PRIEST_CROWD
    assert trace.combat_selection_basis_value == jnp.float32(80.0)
    assert bool(jnp.all(trace.combat_selection_basis_components == 0.0))


def test_shielded_enemy_counts_in_no_priest_crowd_but_cannot_receive_trap(
    class_rows: dict[int, Array],
) -> None:
    focal = _placed_row(class_rows[HUNTER_CLASS_ID], 5.0, 5.0, health=100.0)
    hunter = _placed_row(class_rows[HUNTER_CLASS_ID], 7.0, 5.0, health=100.0)
    warrior = _placed_row(class_rows[WARRIOR_CLASS_ID], 7.0, 6.0, health=80.0)
    facts = _policy_facts(focal, enemies=(hunter, warrior))._replace(
        enemy_spawn_shields=jnp.asarray((2, 0, 0, 0, 0), dtype=jnp.int32)
    )

    action, trace = _decide_facts(
        facts,
        combat_support=((0, 0), (6, 1), (7, 1)),
    )

    assert (int(action.select_target), int(action.use_ultimate)) == (7, 1)
    assert int(trace.combat_reason_id) == policy_module.HUNTER_TRAP_NO_PRIEST_CROWD


def test_configured_dead_priest_still_blocks_hunter_no_priest_crowd(
    class_rows: dict[int, Array],
) -> None:
    focal = _placed_row(class_rows[HUNTER_CLASS_ID], 5.0, 5.0, health=100.0)
    priest = _placed_row(class_rows[PRIEST_CLASS_ID], 15.0, 10.0)
    mage = _placed_row(class_rows[MAGE_CLASS_ID], 7.0, 5.0)
    rogue = _placed_row(class_rows[ROGUE_CLASS_ID], 7.0, 6.0)
    facts = _policy_facts(focal, enemies=(priest, mage, rogue))._replace(
        enemy_alive=jnp.asarray((False, True, True, False, False))
    )

    action, trace = _decide_facts(
        facts,
        combat_support=(
            (0, 0),
            (7, 0),
            (7, 1),
            (8, 0),
            (8, 1),
        ),
    )

    assert int(action.use_ultimate) == 0
    assert not bool(trace.fired_guards[3])


@pytest.mark.parametrize("invalid_value", (-1.0, np.nan, np.inf))
@pytest.mark.parametrize("invalid_fact", ("focal-hp", "target-hp"))
def test_invalid_hunter_health_never_fires_a_trap_branch(
    class_rows: dict[int, Array],
    invalid_fact: str,
    invalid_value: float,
) -> None:
    focal = _placed_row(class_rows[HUNTER_CLASS_ID], 5.0, 5.0, health=19.0)
    first = _placed_row(class_rows[MAGE_CLASS_ID], 7.0, 5.0, health=70.0)
    second = _placed_row(class_rows[ROGUE_CLASS_ID], 7.0, 6.0, health=80.0)
    if invalid_fact == "focal-hp":
        focal = focal.at[AGENT_FEATURE_CURRENT_HEALTH].set(invalid_value)
        enemies = (first,)
        combat_support = ((0, 0), (6, 1))
    else:
        focal = focal.at[AGENT_FEATURE_CURRENT_HEALTH].set(100.0)
        first = first.at[AGENT_FEATURE_CURRENT_HEALTH].set(invalid_value)
        second = second.at[AGENT_FEATURE_CURRENT_HEALTH].set(invalid_value)
        enemies = (first, second)
        combat_support = ((0, 0), (6, 1), (7, 1))

    action, trace = _decide_facts(
        _policy_facts(focal, enemies=enemies),
        combat_support=combat_support,
    )

    assert int(action.use_ultimate) == 0
    assert not bool(trace.fired_guards[3])


def test_rogue_poison_is_a_same_target_substitution_after_basic_ranking(
    class_rows: dict[int, Array],
) -> None:
    focal = _placed_row(class_rows[ROGUE_CLASS_ID], 5.0, 5.0)
    target = _placed_row(class_rows[MAGE_CLASS_ID], 6.0, 5.0)
    facts = _policy_facts(focal, enemies=(target,))

    basic_action, basic_trace = _decide_facts(
        facts,
        combat_support=((0, 0), (6, 0)),
        seed=47,
    )
    poison_action, poison_trace = _decide_facts(
        facts,
        combat_support=((0, 0), (6, 0), (6, 1)),
        seed=47,
    )

    assert (int(basic_action.select_target), int(basic_action.use_ultimate)) == (6, 0)
    assert (int(poison_action.select_target), int(poison_action.use_ultimate)) == (
        6,
        1,
    )
    assert int(poison_trace.combat_reason_id) == policy_module.ROGUE_POISON_SUBSTITUTION
    assert bool(poison_trace.fired_guards[4])
    assert int(poison_trace.combat_peer_count) == int(basic_trace.combat_peer_count)
    np.testing.assert_array_equal(
        np.asarray(poison_trace.combat_selection_basis_components),
        np.asarray(basic_trace.combat_selection_basis_components),
    )
    assert (
        poison_trace.combat_selection_basis_value
        == basic_trace.combat_selection_basis_value
    )
    np.testing.assert_allclose(
        np.asarray(poison_trace.combat_selection_basis_value),
        np.asarray(jnp.sum(poison_trace.combat_selection_basis_components)),
        rtol=0.0,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    ("mage_health", "expected_ultimate"),
    ((30.0, 0), (29.0, 1)),
)
def test_priest_holy_word_uses_raw_hp_strictly_below_30(
    class_rows: dict[int, Array],
    mage_health: float,
    expected_ultimate: int,
) -> None:
    focal = _placed_row(class_rows[PRIEST_CLASS_ID], 5.0, 5.0, health=100.0)
    mage = _placed_row(class_rows[MAGE_CLASS_ID], 7.0, 5.0, health=mage_health)
    facts = _policy_facts(focal, allies=(mage,))

    action, trace = _decide_facts(
        facts,
        combat_support=((0, 0), (2, 0), (2, 1)),
    )

    assert int(action.use_ultimate) == expected_ultimate
    if expected_ultimate:
        assert int(action.select_target) == 2
        assert int(trace.combat_reason_id) == policy_module.PRIEST_HOLY_WORD_TRIGGER
        assert bool(trace.fired_guards[5])


def test_priest_holy_word_uses_fixed_class_order_not_health_or_triage(
    class_rows: dict[int, Array],
) -> None:
    focal = _placed_row(class_rows[PRIEST_CLASS_ID], 5.0, 5.0, health=100.0)
    mage = _placed_row(class_rows[MAGE_CLASS_ID], 7.0, 5.0, health=29.0)
    rogue = _placed_row(class_rows[ROGUE_CLASS_ID], 7.0, 6.0, health=1.0)
    facts = _policy_facts(focal, allies=(mage, rogue))

    action, trace = _decide_facts(
        facts,
        combat_support=((0, 0), (2, 0), (2, 1), (3, 0), (3, 1)),
    )

    assert (int(action.select_target), int(action.use_ultimate)) == (2, 1)
    assert int(trace.combat_reason_id) == policy_module.PRIEST_HOLY_WORD_TRIGGER


def test_priest_holy_word_same_class_qualifiers_are_exact_peers(
    class_rows: dict[int, Array],
) -> None:
    focal = _placed_row(class_rows[PRIEST_CLASS_ID], 5.0, 5.0, health=100.0)
    priest_a = _placed_row(class_rows[PRIEST_CLASS_ID], 7.0, 5.0, health=29.0)
    priest_b = _placed_row(class_rows[PRIEST_CLASS_ID], 7.0, 6.0, health=1.0)
    facts = _policy_facts(focal, allies=(priest_a, priest_b))

    action, trace = _decide_facts(
        facts,
        combat_support=((0, 0), (2, 1), (3, 1)),
        seed=53,
    )

    assert int(action.select_target) in (2, 3)
    assert int(action.use_ultimate) == 1
    assert int(trace.combat_peer_count) == 2
    assert trace.combat_selection_basis_value == jnp.float32(0.0)
    assert bool(jnp.all(trace.combat_selection_basis_components == 0.0))


@pytest.mark.parametrize("invalid_health", (-1.0, np.nan, np.inf))
def test_invalid_recipient_health_never_fires_holy_word(
    class_rows: dict[int, Array],
    invalid_health: float,
) -> None:
    focal = _placed_row(class_rows[PRIEST_CLASS_ID], 5.0, 5.0, health=100.0)
    mage = _placed_row(class_rows[MAGE_CLASS_ID], 7.0, 5.0, health=invalid_health)

    action, trace = _decide_facts(
        _policy_facts(focal, allies=(mage,)),
        combat_support=((0, 0), (2, 1)),
    )

    assert int(action.use_ultimate) == 0
    assert not bool(trace.fired_guards[5])


def test_hidden_recipient_health_cannot_create_holy_word(
    class_rows: dict[int, Array],
) -> None:
    focal = _placed_row(class_rows[PRIEST_CLASS_ID], 5.0, 5.0, health=100.0)
    mage = _placed_row(class_rows[MAGE_CLASS_ID], 7.0, 5.0, health=29.0)
    facts = _policy_facts(focal, allies=(mage,))
    facts = facts._replace(ally_visible=facts.ally_visible.at[1].set(False))

    action, trace = _decide_facts(
        facts,
        combat_support=((0, 0), (2, 1)),
    )

    assert (int(action.select_target), int(action.use_ultimate)) == (0, 0)
    assert not bool(trace.fired_guards[5])


def test_damage_formula_applies_burst_aura_and_mitigation_once(
    class_rows: dict[int, Array],
) -> None:
    focal = _placed_row(class_rows[MAGE_CLASS_ID], 5.0, 5.0)
    focal = focal.at[AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_BURST_DURATION].set(1.0)
    focal = focal.at[AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER].set(1.15)
    target = _placed_row(class_rows[WARRIOR_CLASS_ID], 7.0, 5.0, health=100.0)
    target = target.at[AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER].set(
        0.85
    )
    target = target.at[AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION].set(1.0)
    facts = _policy_facts(focal, enemies=(target,))

    action, trace = _decide_facts(
        facts,
        combat_support=((0, 0), (6, 0)),
    )

    raw_damage = focal[AGENT_FEATURE_CAPABILITY_BASIC_DAMAGE]
    burst = focal[AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_BURST_MULTIPLIER]
    payload = raw_damage * burst * jnp.float32(1.15) * jnp.float32(0.85)
    realized = jnp.minimum(payload, jnp.float32(100.0))
    coverage = realized / jnp.float32(100.0)
    potency = jnp.clip(realized / jnp.float32(40.0), 0.0, 1.0)
    impact = jnp.float32(0.5) * (coverage + potency)
    urgency = jnp.float32(0.25)
    access = jnp.float32(1.0 / 3.0)
    expected = jnp.asarray(
        (
            0.30 * impact,
            0.30 * urgency,
            0.0,
            0.0,
            0.10 * access,
            0.0,
            0.0,
            0.0,
        ),
        dtype=jnp.float32,
    )

    assert (int(action.select_target), int(action.use_ultimate)) == (6, 0)
    np.testing.assert_allclose(
        np.asarray(trace.combat_selection_basis_components),
        np.asarray(expected),
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(trace.combat_selection_basis_value),
        np.asarray(jnp.sum(expected)),
        rtol=0.0,
        atol=1e-6,
    )


def test_healing_formula_applies_antiheal_once_and_role_value_only_on_realized_heal(
    class_rows: dict[int, Array],
) -> None:
    focal = _placed_row(class_rows[PRIEST_CLASS_ID], 5.0, 5.0, health=100.0)
    target = _placed_row(class_rows[MAGE_CLASS_ID], 7.0, 5.0, health=40.0)
    target = target.at[AGENT_FEATURE_ANTI_HEAL_ROGUE_POISON_DURATION].set(1.0)
    target = target.at[AGENT_FEATURE_ANTI_HEAL_ROGUE_POISON_MULTIPLIER].set(0.5)
    facts = _policy_facts(focal, allies=(target,))

    action, trace = _decide_facts(
        facts,
        combat_support=((0, 0), (2, 0)),
    )

    raw_healing = focal[AGENT_FEATURE_CAPABILITY_BASIC_HEALING]
    payload = raw_healing * jnp.float32(0.5)
    realized = jnp.minimum(payload, jnp.float32(40.0))
    coverage = realized / jnp.float32(40.0)
    potency = jnp.clip(realized / jnp.float32(16.0), 0.0, 1.0)
    impact = jnp.float32(0.5) * (coverage + potency)
    expected = jnp.asarray(
        (
            0.30 * impact,
            0.30 * 0.50,
            0.0,
            0.0,
            0.05 * (1.0 / 3.0),
            0.0,
            0.06,
            0.0,
        ),
        dtype=jnp.float32,
    )

    assert (int(action.select_target), int(action.use_ultimate)) == (2, 0)
    np.testing.assert_allclose(
        np.asarray(trace.combat_selection_basis_components),
        np.asarray(expected),
        rtol=0.0,
        atol=1e-6,
    )


@pytest.mark.parametrize("payload", ("damage", "healing"))
def test_finite_payload_product_overflow_is_invalid_and_total(
    class_rows: dict[int, Array],
    payload: str,
) -> None:
    huge = jnp.float32(3e38)
    if payload == "damage":
        focal = _placed_row(class_rows[MAGE_CLASS_ID], 5.0, 5.0)
        focal = focal.at[AGENT_FEATURE_CAPABILITY_BASIC_DAMAGE].set(huge)
        focal = focal.at[AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_BURST_DURATION].set(
            1.0
        )
        focal = focal.at[
            AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_BURST_MULTIPLIER
        ].set(2.0)
        focal = focal.at[AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER].set(
            2.0
        )
        target = _placed_row(class_rows[WARRIOR_CLASS_ID], 7.0, 5.0, health=100.0)
        target = target.at[AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER].set(
            2.0
        )
        facts = _policy_facts(focal, enemies=(target,))
        target_action = 6
        offered = policy_module._post_damage(focal, target, huge)
    else:
        focal = _placed_row(class_rows[PRIEST_CLASS_ID], 5.0, 5.0)
        focal = focal.at[AGENT_FEATURE_CAPABILITY_BASIC_HEALING].set(huge)
        target = _placed_row(class_rows[MAGE_CLASS_ID], 7.0, 5.0, health=40.0)
        target = target.at[AGENT_FEATURE_ANTI_HEAL_ROGUE_POISON_DURATION].set(1.0)
        target = target.at[AGENT_FEATURE_ANTI_HEAL_ROGUE_POISON_MULTIPLIER].set(2.0)
        facts = _policy_facts(focal, allies=(target,))
        target_action = 2
        offered = policy_module._post_healing(target, huge)

    workspaces = policy_module._combat_workspaces(facts)
    action, trace = _decide_facts(
        facts,
        combat_support=((target_action, 0),),
    )

    assert offered == jnp.float32(0.0)
    assert bool(jnp.all(jnp.isfinite(workspaces[0])))
    assert bool(jnp.all(jnp.isfinite(workspaces[1])))
    assert bool(jnp.all(jnp.isfinite(trace.combat_selection_basis_components)))
    assert jnp.isfinite(trace.combat_selection_basis_value)
    assert int(trace.combat_peer_count) == 1
    assert (int(action.select_target), int(action.use_ultimate)) == (target_action, 0)


def test_rounded_float32_product_boundary_is_invalid() -> None:
    product, valid = policy_module._finite_nonnegative_product(
        jnp.float32(2.9582604e38),
        jnp.float32(1.1502786),
    )

    assert not bool(valid)
    assert product == jnp.float32(0.0)


def test_subnormal_positive_ratio_scale_is_invalid_and_total(
    class_rows: dict[int, Array],
) -> None:
    tiny = jnp.nextafter(jnp.float32(0.0), jnp.float32(1.0))
    subnormal_ratio = policy_module._safe_ratio(
        jnp.float32(1.0),
        tiny,
        jnp.asarray(True),
    )
    rounded_zero_ratio = policy_module._safe_ratio(
        jnp.float32(1.0),
        jnp.float32(0.20) * tiny,
        jnp.asarray(True),
    )
    focal = _placed_row(class_rows[MAGE_CLASS_ID], 5.0, 5.0)
    target = _placed_row(
        class_rows[WARRIOR_CLASS_ID],
        7.0,
        5.0,
        health=float(tiny),
    )
    target = target.at[AGENT_FEATURE_MAX_HEALTH].set(tiny)
    facts = _policy_facts(focal, enemies=(target,))

    workspaces = policy_module._combat_workspaces(facts)
    _, trace = _decide_facts(facts, combat_support=((6, 0),))

    assert subnormal_ratio == jnp.float32(0.0)
    assert rounded_zero_ratio == jnp.float32(0.0)
    assert bool(jnp.all(jnp.isfinite(workspaces[0])))
    assert bool(jnp.all(jnp.isfinite(workspaces[1])))
    assert bool(jnp.all(jnp.isfinite(trace.combat_selection_basis_components)))
    assert jnp.isfinite(trace.combat_selection_basis_value)
    assert int(trace.combat_peer_count) == 1


@pytest.mark.parametrize(
    ("feature_index", "invalid_value"),
    (
        (AGENT_FEATURE_CURRENT_HEALTH, -1.0),
        (AGENT_FEATURE_CURRENT_HEALTH, np.nan),
        (AGENT_FEATURE_CURRENT_HEALTH, np.inf),
        (AGENT_FEATURE_MAX_HEALTH, 0.0),
        (AGENT_FEATURE_MAX_HEALTH, np.nan),
        (AGENT_FEATURE_MAX_HEALTH, np.inf),
    ),
)
def test_invalid_recipient_health_never_creates_an_excess_penalty(
    class_rows: dict[int, Array],
    feature_index: int,
    invalid_value: float,
) -> None:
    focal = _placed_row(class_rows[MAGE_CLASS_ID], 5.0, 5.0)
    target = _placed_row(class_rows[WARRIOR_CLASS_ID], 7.0, 5.0, health=100.0)
    target = target.at[feature_index].set(invalid_value)
    target = target.at[AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION].set(1.0)

    _, trace = _decide_facts(
        _policy_facts(focal, enemies=(target,)),
        combat_support=((6, 0),),
    )

    assert trace.combat_selection_basis_components[5] == jnp.float32(0.0)


@pytest.mark.parametrize("invalid_value", (-1.0, np.nan, np.inf))
@pytest.mark.parametrize("payload", ("damage", "healing"))
def test_invalid_raw_payload_never_retains_impact_urgency_or_excess(
    class_rows: dict[int, Array],
    payload: str,
    invalid_value: float,
) -> None:
    if payload == "damage":
        focal = _placed_row(class_rows[MAGE_CLASS_ID], 5.0, 5.0)
        focal = focal.at[AGENT_FEATURE_CAPABILITY_BASIC_DAMAGE].set(invalid_value)
        target = _placed_row(class_rows[WARRIOR_CLASS_ID], 7.0, 5.0, health=100.0)
        facts = _policy_facts(focal, enemies=(target,))
        target_action = 6
    else:
        focal = _placed_row(class_rows[PRIEST_CLASS_ID], 5.0, 5.0)
        focal = focal.at[AGENT_FEATURE_CAPABILITY_BASIC_HEALING].set(invalid_value)
        target = _placed_row(class_rows[MAGE_CLASS_ID], 7.0, 5.0, health=40.0)
        facts = _policy_facts(focal, allies=(target,))
        target_action = 2

    _, trace = _decide_facts(
        facts,
        combat_support=((target_action, 0),),
    )

    assert trace.combat_selection_basis_components[0] == jnp.float32(0.0)
    assert trace.combat_selection_basis_components[1] == jnp.float32(0.0)
    assert trace.combat_selection_basis_components[5] == jnp.float32(0.0)


@pytest.mark.parametrize("invalid_value", (-1.0, np.nan, np.inf))
@pytest.mark.parametrize(
    "invalid_modifier", ("burst-multiplier", "mage-aura", "warrior-mitigation")
)
def test_invalid_damage_modifier_never_suppresses_an_aged_trap(
    class_rows: dict[int, Array],
    invalid_modifier: str,
    invalid_value: float,
) -> None:
    focal = _placed_row(class_rows[MAGE_CLASS_ID], 5.0, 5.0)
    target = _placed_row(class_rows[WARRIOR_CLASS_ID], 7.0, 5.0, health=100.0)
    target = target.at[AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION].set(2.0)
    if invalid_modifier == "burst-multiplier":
        focal = focal.at[AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_BURST_DURATION].set(
            1.0
        )
        focal = focal.at[
            AGENT_FEATURE_CAPABILITY_DAMAGE_AMPLIFICATION_MAGE_BURST_MULTIPLIER
        ].set(invalid_value)
    elif invalid_modifier == "mage-aura":
        focal = focal.at[AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER].set(
            invalid_value
        )
    else:
        target = target.at[AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER].set(
            invalid_value
        )

    action, trace = _decide_facts(
        _policy_facts(focal, enemies=(target,)),
        combat_support=((6, 0),),
    )

    assert (int(action.select_target), int(action.use_ultimate)) == (6, 0)
    assert not bool(trace.fired_guards[0])


@pytest.mark.parametrize("invalid_health", (-1.0, np.nan, np.inf))
def test_invalid_recipient_health_never_suppresses_an_aged_trap(
    class_rows: dict[int, Array],
    invalid_health: float,
) -> None:
    focal = _placed_row(class_rows[HUNTER_CLASS_ID], 5.0, 5.0, health=19.0)
    target = _placed_row(
        class_rows[WARRIOR_CLASS_ID],
        7.0,
        5.0,
        health=invalid_health,
    )
    target = target.at[AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION].set(2.0)
    facts = _policy_facts(focal, enemies=(target,))
    workspaces = policy_module._combat_workspaces(facts)

    action, trace = _decide_facts(
        facts,
        combat_support=((6, 1),),
    )

    assert not bool(workspaces[7][6, 1])
    assert workspaces[0][6, 1, 3] > 0.0
    assert workspaces[0][6, 1, 4] > 0.0
    assert (int(action.select_target), int(action.use_ultimate)) == (6, 1)
    assert int(trace.combat_peer_count) == 1
    assert not bool(trace.fired_guards[0])


def test_literal_float32_equality_alone_defines_ordinary_combat_peers(
    class_rows: dict[int, Array],
) -> None:
    focal = _placed_row(class_rows[MAGE_CLASS_ID], 5.0, 5.0)
    target = _placed_row(class_rows[WARRIOR_CLASS_ID], 7.0, 5.0, health=100.0)
    target = target.at[AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION].set(1.0)
    tied_facts = _policy_facts(focal, enemies=(target, target))

    tied_action, tied_trace = _decide_facts(
        tied_facts,
        combat_support=((0, 0), (6, 0), (7, 0)),
        seed=59,
    )
    changed_target = target.at[AGENT_FEATURE_CURRENT_HEALTH].set(99.0)
    changed_facts = _policy_facts(focal, enemies=(target, changed_target))
    changed_action, changed_trace = _decide_facts(
        changed_facts,
        combat_support=((0, 0), (6, 0), (7, 0)),
        seed=59,
    )

    assert int(tied_trace.combat_peer_count) == 2
    combat_key, _ = jax.random.split(jax.random.key(59), 2)
    expected_logits = jnp.full((22,), -jnp.inf, dtype=jnp.float32)
    expected_logits = expected_logits.at[jnp.asarray((12, 14))].set(0.0)
    expected_flat = jax.random.categorical(combat_key, expected_logits)
    assert int(tied_action.select_target) == int(expected_flat) // 2
    assert int(tied_action.use_ultimate) == int(expected_flat) % 2
    assert int(changed_trace.combat_peer_count) == 1
    assert int(changed_action.select_target) == 7


def test_each_decision_unconditionally_splits_once_and_draws_each_head_once(
    class_rows: dict[int, Array],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    focal = _placed_row(class_rows[MAGE_CLASS_ID], 5.0, 5.0)
    target = _placed_row(class_rows[WARRIOR_CLASS_ID], 7.0, 5.0, health=100.0)
    target = target.at[AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION].set(1.0)
    facts = _policy_facts(focal, enemies=(target, target))
    action_mask = _local_action_mask(
        move_support=(MOVE_STAY,),
        combat_support=((0, 0), (6, 0), (7, 0)),
    )
    original_split = jax.random.split
    original_categorical = jax.random.categorical
    split_counts: list[int] = []
    categorical_shapes: list[tuple[int, ...]] = []

    def record_split(key: Array, num: int = 2) -> Array:
        split_counts.append(num)
        return original_split(key, num)

    def record_categorical(key: Array, logits: Array) -> Array:
        categorical_shapes.append(logits.shape)
        return original_categorical(key, logits)

    monkeypatch.setattr(jax.random, "split", record_split)
    monkeypatch.setattr(jax.random, "categorical", record_categorical)

    with jax.disable_jit():
        policy_module.decide_team_deathmatch(facts, action_mask, jax.random.key(61))

    assert split_counts == [2]
    assert categorical_shapes == [(22,), (9,)]


@pytest.mark.parametrize(
    ("enemy_damage", "expected_rejection", "expected_fallback"),
    ((11.2, False, False), (11.21, True, True)),
)
def test_movement_risk_rejects_only_strictly_above_point_70(
    class_rows: dict[int, Array],
    enemy_damage: float,
    expected_rejection: bool,
    expected_fallback: bool,
) -> None:
    focal = _placed_row(class_rows[MAGE_CLASS_ID], 5.0, 5.0, health=80.0)
    focal = focal.at[AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER].set(1.0)
    enemy = _placed_row(class_rows[MAGE_CLASS_ID], 7.0, 5.0)
    enemy = enemy.at[AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER].set(1.0)
    enemy = enemy.at[AGENT_FEATURE_CAPABILITY_BASIC_DAMAGE].set(enemy_damage)
    enemy = enemy.at[AGENT_FEATURE_CAPABILITY_ULTIMATE_DAMAGE].set(0.0)
    facts = _policy_facts(focal, enemies=(enemy,))

    _, trace = _decide_facts(
        facts,
        combat_support=((0, 0),),
        move_support=(MOVE_STAY,),
    )

    assert bool(trace.fired_guards[7]) is expected_rejection
    assert bool(trace.fired_guards[8]) is expected_fallback
    if expected_fallback:
        assert int(trace.movement_reason_id) == policy_module.MIN_RISK_FALLBACK


def test_all_rejected_endpoints_fall_back_to_literal_minimum_risk_support(
    class_rows: dict[int, Array],
) -> None:
    focal = _placed_row(class_rows[MAGE_CLASS_ID], 5.0, 5.0, health=80.0)
    focal = focal.at[AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER].set(1.0)
    always_threat = _placed_row(class_rows[MAGE_CLASS_ID], 7.0, 5.0)
    always_threat = always_threat.at[
        AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER
    ].set(1.0)
    always_threat = always_threat.at[AGENT_FEATURE_CAPABILITY_BASIC_DAMAGE].set(12.8)
    always_threat = always_threat.at[AGENT_FEATURE_CAPABILITY_ULTIMATE_DAMAGE].set(0.0)
    stay_only_threat = _placed_row(class_rows[MAGE_CLASS_ID], 3.0, 5.0)
    stay_only_threat = stay_only_threat.at[
        AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER
    ].set(1.0)
    stay_only_threat = stay_only_threat.at[AGENT_FEATURE_CAPABILITY_BASIC_DAMAGE].set(
        8.0
    )
    stay_only_threat = stay_only_threat.at[
        AGENT_FEATURE_CAPABILITY_ULTIMATE_DAMAGE
    ].set(0.0)
    stay_only_threat = stay_only_threat.at[AGENT_FEATURE_BASIC_INTERACTION_RADIUS].set(
        2.0
    )
    facts = _policy_facts(focal, enemies=(always_threat, stay_only_threat))

    action, trace = _decide_facts(
        facts,
        combat_support=((0, 0),),
        move_support=(MOVE_STAY, MOVE_EAST),
    )

    assert int(action.move) == MOVE_EAST
    assert int(trace.movement_reason_id) == policy_module.MIN_RISK_FALLBACK
    assert int(trace.movement_peer_count) == 1
    assert bool(trace.fired_guards[7])
    assert bool(trace.fired_guards[8])
    np.testing.assert_allclose(
        np.asarray(trace.movement_selection_basis_value),
        np.asarray(jnp.sum(trace.movement_selection_basis_components)),
        rtol=0.0,
        atol=1e-6,
    )


def test_finite_inputs_cannot_leave_a_nonfinite_intended_endpoint(
    class_rows: dict[int, Array],
) -> None:
    huge = jnp.float32(3e38)
    focal = _placed_row(class_rows[MAGE_CLASS_ID], huge, huge, health=40.0)
    focal = focal.at[AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED].set(huge)
    focal = focal.at[AGENT_FEATURE_STEPS_UNTIL_OUT_OF_COMBAT].set(1.0)
    facts = _policy_facts(focal)
    raw_payload = jnp.zeros((11, 2), dtype=jnp.float32)

    components, utility, risk, endpoints = policy_module._movement_workspaces(
        facts,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        raw_payload,
        raw_payload,
    )

    assert bool(jnp.all(jnp.isfinite(endpoints)))
    assert bool(jnp.all(endpoints[MOVE_EAST] == 0.0))
    assert bool(jnp.all(components[MOVE_EAST] == 0.0))
    assert utility[MOVE_EAST] == jnp.float32(0.0)
    assert risk[MOVE_EAST] == jnp.float32(0.0)


@pytest.mark.parametrize(
    ("utility_difference", "stay_supported", "expected_move", "deadband_fired"),
    (
        (np.float32(0.03), True, MOVE_STAY, True),
        (
            np.nextafter(np.float32(0.03), np.float32(np.inf)),
            True,
            MOVE_WEST,
            False,
        ),
        (np.float32(0.03), False, MOVE_WEST, False),
    ),
)
def test_stay_deadband_uses_point_03_equality_without_restoring_masked_stay(
    class_rows: dict[int, Array],
    monkeypatch: pytest.MonkeyPatch,
    utility_difference: np.float32,
    stay_supported: bool,
    expected_move: int,
    deadband_fired: bool,
) -> None:
    focal = _placed_row(class_rows[MAGE_CLASS_ID], 5.0, 5.0, health=80.0)
    facts = _policy_facts(
        focal,
        enemies=(_placed_row(class_rows[WARRIOR_CLASS_ID], 7.0, 5.0),),
    )
    move_support = (MOVE_STAY, MOVE_WEST) if stay_supported else (MOVE_WEST,)

    def controlled_movement(*_: object) -> tuple[Array, Array, Array, Array]:
        components = jnp.zeros((9, 9), dtype=jnp.float32)
        components = components.at[MOVE_WEST, 0].set(utility_difference)
        return (
            components,
            jnp.sum(components, axis=1, dtype=jnp.float32),
            jnp.zeros((9,), dtype=jnp.float32),
            jnp.zeros((9, 2), dtype=jnp.float32),
        )

    monkeypatch.setattr(policy_module, "_movement_workspaces", controlled_movement)

    with jax.disable_jit():
        action, trace = _decide_facts(
            facts,
            combat_support=((0, 0),),
            move_support=move_support,
        )

    assert int(action.move) == expected_move
    assert bool(trace.fired_guards[9]) is deadband_fired
    if deadband_fired:
        assert int(trace.movement_reason_id) == policy_module.STAY_DEADBAND
    else:
        assert int(trace.movement_reason_id) == policy_module.MOVE_DIRECT_SCORE


def test_rejected_stay_is_not_restored_by_the_deadband(
    class_rows: dict[int, Array],
) -> None:
    focal = _placed_row(class_rows[MAGE_CLASS_ID], 5.0, 5.0, health=80.0)
    focal = focal.at[AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER].set(1.0)
    threat = _placed_row(class_rows[MAGE_CLASS_ID], 7.0, 5.0)
    threat = threat.at[AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER].set(1.0)
    threat = threat.at[AGENT_FEATURE_CAPABILITY_BASIC_DAMAGE].set(12.8)
    threat = threat.at[AGENT_FEATURE_CAPABILITY_ULTIMATE_DAMAGE].set(0.0)
    threat = threat.at[AGENT_FEATURE_BASIC_INTERACTION_RADIUS].set(2.5)
    facts = _policy_facts(focal, enemies=(threat,))

    action, trace = _decide_facts(
        facts,
        combat_support=((0, 0),),
        move_support=(MOVE_STAY, MOVE_WEST),
    )

    assert int(action.move) == MOVE_WEST
    assert bool(trace.fired_guards[7])
    assert not bool(trace.fired_guards[9])


def test_formation_uses_selected_ally_then_cohort_then_neutral_no_anchor(
    class_rows: dict[int, Array],
) -> None:
    focal = _placed_row(class_rows[PRIEST_CLASS_ID], 5.0, 5.0, health=100.0)
    ally = _placed_row(class_rows[MAGE_CLASS_ID], 7.0, 5.0, health=40.0)
    anchored_facts = _policy_facts(focal, allies=(ally,))

    _, selected_trace = _decide_facts(
        anchored_facts,
        combat_support=((2, 0),),
    )
    _, cohort_trace = _decide_facts(
        anchored_facts,
        combat_support=((0, 0),),
    )
    _, no_anchor_trace = _decide_facts(
        _policy_facts(focal),
        combat_support=((0, 0),),
    )

    assert selected_trace.movement_selection_basis_components[1] > 0.0
    assert cohort_trace.movement_selection_basis_components[1] > 0.0
    assert no_anchor_trace.movement_selection_basis_components[1] == 0.0
    for trace in (selected_trace, cohort_trace, no_anchor_trace):
        assert int(trace.movement_reason_id) == policy_module.MOVE_DIRECT_SCORE
        assert int(trace.movement_peer_count) == 1


def test_movement_components_count_friendly_crowding_once(
    class_rows: dict[int, Array],
) -> None:
    focal = _placed_row(class_rows[PRIEST_CLASS_ID], 5.0, 5.0, health=100.0)
    close_ally = _placed_row(class_rows[MAGE_CLASS_ID], 5.0, 5.0)
    far_ally = _placed_row(class_rows[MAGE_CLASS_ID], 10.0, 10.0)

    _, close_trace = _decide_facts(
        _policy_facts(focal, allies=(close_ally,)),
        combat_support=((0, 0),),
    )
    _, far_trace = _decide_facts(
        _policy_facts(focal, allies=(far_ally,)),
        combat_support=((0, 0),),
    )

    assert close_trace.movement_selection_basis_components[5] == jnp.float32(-0.05)
    assert far_trace.movement_selection_basis_components[5] == jnp.float32(0.0)


@pytest.mark.parametrize(
    ("move", "expected_obstruction"),
    ((MOVE_STAY, 0.0), (MOVE_WEST, -0.10)),
)
def test_movement_obstruction_scores_the_unprojected_endpoint(
    class_rows: dict[int, Array],
    move: int,
    expected_obstruction: float,
) -> None:
    focal = _placed_row(class_rows[MAGE_CLASS_ID], 0.5, 0.5, health=100.0)

    action, trace = _decide_facts(
        _policy_facts(focal),
        combat_support=((0, 0),),
        move_support=(move,),
    )

    assert int(action.move) == move
    assert trace.movement_selection_basis_components[6] == jnp.float32(
        expected_obstruction
    )


@pytest.mark.parametrize("invalid_dimension", (0.0, -1.0))
def test_nonpositive_map_dimensions_make_obstruction_neutral(
    class_rows: dict[int, Array],
    invalid_dimension: float,
) -> None:
    focal = _placed_row(class_rows[MAGE_CLASS_ID], 0.5, 0.5, health=100.0)
    facts = _policy_facts(focal)._replace(
        map_width=jnp.asarray(invalid_dimension, dtype=jnp.float32),
        map_height=jnp.asarray(invalid_dimension, dtype=jnp.float32),
    )

    action, trace = _decide_facts(
        facts,
        combat_support=((0, 0),),
        move_support=(MOVE_WEST,),
    )

    assert int(action.move) == MOVE_WEST
    assert trace.movement_selection_basis_components[6] == jnp.float32(0.0)


@pytest.mark.parametrize(
    ("obstacle_x", "selected_target", "expected"),
    ((5.3, 0, 0.75), (6.5, 6, 0.50)),
)
def test_invalid_map_only_neutralizes_the_endpoint_obstruction_predicate(
    class_rows: dict[int, Array],
    obstacle_x: float,
    selected_target: int,
    expected: float,
) -> None:
    focal = _placed_row(class_rows[MAGE_CLASS_ID], 5.0, 5.0)
    enemies = (
        (_placed_row(class_rows[WARRIOR_CLASS_ID], 7.0, 5.0),)
        if selected_target > 0
        else ()
    )
    obstacle = jnp.zeros((8,), dtype=jnp.float32)
    obstacle = obstacle.at[OBSTACLE_FEATURE_TYPE].set(OBSTACLE_TYPE_PILLAR)
    obstacle = obstacle.at[OBSTACLE_FEATURE_X].set(obstacle_x)
    obstacle = obstacle.at[OBSTACLE_FEATURE_Y].set(5.0)
    obstacle = obstacle.at[OBSTACLE_FEATURE_RADIUS].set(0.1)
    obstacle = obstacle.at[OBSTACLE_FEATURE_ACTIVE].set(1.0)
    facts = _policy_facts(focal, enemies=enemies)._replace(
        obstacles=jnp.zeros((16, 8), dtype=jnp.float32).at[0].set(obstacle),
        map_width=jnp.asarray(0.0, dtype=jnp.float32),
    )

    actual = policy_module._obstruction(
        facts,
        jnp.asarray((6.0, 5.0), dtype=jnp.float32),
        jnp.asarray(selected_target, dtype=jnp.int32),
    )

    assert actual == jnp.float32(expected)


@pytest.mark.parametrize(
    ("obstacle_x", "selected_target", "expected"),
    ((5.3, 0, 0.75), (6.5, 6, 0.50)),
)
def test_invalid_focal_radius_preserves_independent_los_obstruction(
    class_rows: dict[int, Array],
    obstacle_x: float,
    selected_target: int,
    expected: float,
) -> None:
    focal = _placed_row(class_rows[MAGE_CLASS_ID], 5.0, 5.0)
    focal = focal.at[AGENT_FEATURE_RADIUS].set(jnp.nan)
    enemies = (
        (_placed_row(class_rows[WARRIOR_CLASS_ID], 7.0, 5.0),)
        if selected_target > 0
        else ()
    )
    obstacle = jnp.zeros((8,), dtype=jnp.float32)
    obstacle = obstacle.at[OBSTACLE_FEATURE_TYPE].set(OBSTACLE_TYPE_PILLAR)
    obstacle = obstacle.at[OBSTACLE_FEATURE_X].set(obstacle_x)
    obstacle = obstacle.at[OBSTACLE_FEATURE_Y].set(5.0)
    obstacle = obstacle.at[OBSTACLE_FEATURE_RADIUS].set(0.1)
    obstacle = obstacle.at[OBSTACLE_FEATURE_ACTIVE].set(1.0)
    facts = _policy_facts(focal, enemies=enemies)._replace(
        obstacles=jnp.zeros((16, 8), dtype=jnp.float32).at[0].set(obstacle),
    )

    actual = policy_module._obstruction(
        facts,
        jnp.asarray((6.0, 5.0), dtype=jnp.float32),
        jnp.asarray(selected_target, dtype=jnp.int32),
    )

    assert actual == jnp.float32(expected)


@pytest.mark.parametrize("focal_countdown", (1.0, 2.0))
def test_future_recovery_readiness_ignores_current_participation_inputs(
    class_rows: dict[int, Array],
    focal_countdown: float,
) -> None:
    focal = _placed_row(class_rows[MAGE_CLASS_ID], 5.0, 5.0, health=40.0)
    focal = focal.at[AGENT_FEATURE_STEPS_UNTIL_OUT_OF_COMBAT].set(focal_countdown)
    target = _placed_row(class_rows[WARRIOR_CLASS_ID], 7.0, 5.0, health=100.0)
    target = target.at[AGENT_FEATURE_STEPS_UNTIL_OUT_OF_COMBAT].set(1.0)
    facts = _policy_facts(focal, enemies=(target,))
    raw_damage = jnp.zeros((11, 2), dtype=jnp.float32).at[6, 0].set(13.0)
    raw_healing = jnp.zeros((11, 2), dtype=jnp.float32)
    mechanics_risk = jnp.zeros((9,), dtype=jnp.float32)

    valid = policy_module._recovery_quality(
        facts,
        mechanics_risk,
        jnp.asarray(6, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        raw_damage,
        raw_healing,
    )
    invalid_payload = policy_module._recovery_quality(
        facts,
        mechanics_risk,
        jnp.asarray(6, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        raw_damage.at[6, 0].set(jnp.inf),
        raw_healing,
    )
    invalid_target = target.at[AGENT_FEATURE_STEPS_UNTIL_OUT_OF_COMBAT].set(jnp.nan)
    invalid_facts = _policy_facts(focal, enemies=(invalid_target,))
    invalid_countdown = policy_module._recovery_quality(
        invalid_facts,
        mechanics_risk,
        jnp.asarray(6, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        raw_damage,
        raw_healing,
    )
    target_none = policy_module._recovery_quality(
        invalid_facts,
        mechanics_risk,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        raw_damage,
        raw_healing,
    )

    assert bool(jnp.all(valid > 0.0))
    np.testing.assert_array_equal(np.asarray(invalid_payload), np.asarray(valid))
    np.testing.assert_array_equal(np.asarray(invalid_countdown), np.asarray(valid))
    np.testing.assert_array_equal(np.asarray(target_none), np.asarray(valid))


def test_current_recovery_neutralizes_invalid_participation_inputs(
    class_rows: dict[int, Array],
) -> None:
    focal = _placed_row(class_rows[MAGE_CLASS_ID], 5.0, 5.0, health=40.0)
    focal = focal.at[AGENT_FEATURE_STEPS_UNTIL_OUT_OF_COMBAT].set(0.0)
    target = _placed_row(class_rows[WARRIOR_CLASS_ID], 7.0, 5.0, health=100.0)
    target = target.at[AGENT_FEATURE_STEPS_UNTIL_OUT_OF_COMBAT].set(1.0)
    facts = _policy_facts(focal, enemies=(target,))
    raw_damage = jnp.zeros((11, 2), dtype=jnp.float32).at[6, 0].set(jnp.inf)
    raw_healing = jnp.zeros((11, 2), dtype=jnp.float32)
    mechanics_risk = jnp.zeros((9,), dtype=jnp.float32)

    invalid_payload = policy_module._recovery_quality(
        facts,
        mechanics_risk,
        jnp.asarray(6, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        raw_damage,
        raw_healing,
    )
    target_none = policy_module._recovery_quality(
        facts,
        mechanics_risk,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        raw_damage,
        raw_healing,
    )

    assert bool(jnp.all(invalid_payload == 0.0))
    assert bool(jnp.all(target_none > 0.0))


@pytest.mark.parametrize(
    ("damage", "healing", "recipient_countdown", "expected_positive"),
    (
        (13.0, 0.0, np.nan, False),
        (0.0, 0.0, np.nan, True),
        (0.0, 13.0, np.nan, False),
        (0.0, 13.0, 0.0, True),
        (0.0, 13.0, 1.0, False),
    ),
)
def test_current_recovery_reads_recipient_countdown_only_for_positive_healing(
    class_rows: dict[int, Array],
    damage: float,
    healing: float,
    recipient_countdown: float,
    expected_positive: bool,
) -> None:
    focal = _placed_row(class_rows[MAGE_CLASS_ID], 5.0, 5.0, health=40.0)
    focal = focal.at[AGENT_FEATURE_STEPS_UNTIL_OUT_OF_COMBAT].set(0.0)
    target = _placed_row(class_rows[WARRIOR_CLASS_ID], 7.0, 5.0, health=100.0)
    target = target.at[AGENT_FEATURE_STEPS_UNTIL_OUT_OF_COMBAT].set(recipient_countdown)
    facts = _policy_facts(focal, enemies=(target,))
    raw_damage = jnp.zeros((11, 2), dtype=jnp.float32).at[6, 0].set(damage)
    raw_healing = jnp.zeros((11, 2), dtype=jnp.float32).at[6, 0].set(healing)

    actual = policy_module._recovery_quality(
        facts,
        jnp.zeros((9,), dtype=jnp.float32),
        jnp.asarray(6, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        raw_damage,
        raw_healing,
    )

    assert bool(jnp.all(actual > 0.0)) is expected_positive


@pytest.mark.parametrize(
    ("duration", "expected_control_component"),
    ((0.0, 0.0135), (1.0, 0.0135), (3.0, 0.0180)),
)
def test_hunter_basic_control_handles_zero_one_and_many_tick_refreshes(
    class_rows: dict[int, Array],
    duration: float,
    expected_control_component: float,
) -> None:
    focal = _placed_row(class_rows[HUNTER_CLASS_ID], 5.0, 5.0)
    target = _placed_row(class_rows[WARRIOR_CLASS_ID], 7.0, 5.0, health=100.0)
    target = target.at[AGENT_FEATURE_SLOW_HUNTER_BASIC_DURATION].set(duration)
    target = target.at[AGENT_FEATURE_SLOW_HUNTER_BASIC_MULTIPLIER].set(0.85)
    facts = _policy_facts(focal, enemies=(target,))

    action, trace = _decide_facts(
        facts,
        combat_support=((6, 0),),
    )

    assert (int(action.select_target), int(action.use_ultimate)) == (6, 0)
    np.testing.assert_allclose(
        np.asarray(trace.combat_selection_basis_components[3]),
        np.asarray(jnp.float32(expected_control_component)),
        rtol=0.0,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    ("existing_trap", "expected_control_component"),
    ((0.0, 0.27), (1.0, 0.27), (3.0, 0.06)),
)
def test_fresh_trap_merges_after_age_and_lethal_old_trap_break(
    class_rows: dict[int, Array],
    existing_trap: float,
    expected_control_component: float,
) -> None:
    focal = _placed_row(class_rows[HUNTER_CLASS_ID], 5.0, 5.0, health=100.0)
    target_health = 5.0 if existing_trap > 1.0 else 50.0
    priest = _placed_row(class_rows[PRIEST_CLASS_ID], 7.0, 5.0, health=target_health)
    priest = priest.at[AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION].set(existing_trap)
    mage = _placed_row(class_rows[MAGE_CLASS_ID], 7.0, 6.0)
    facts = _policy_facts(focal, enemies=(priest, mage))

    action, trace = _decide_facts(
        facts,
        combat_support=((6, 1),),
    )

    assert (int(action.select_target), int(action.use_ultimate)) == (6, 1)
    assert int(trace.combat_reason_id) == policy_module.HUNTER_TRAP_PRIEST_CROWD
    np.testing.assert_allclose(
        np.asarray(trace.combat_selection_basis_components[3]),
        np.asarray(jnp.float32(expected_control_component)),
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(trace.combat_selection_basis_value),
        np.asarray(jnp.sum(trace.combat_selection_basis_components)),
        rtol=0.0,
        atol=1e-6,
    )


def test_priest_freedom_gain_is_zero_while_a_successor_stun_remains(
    class_rows: dict[int, Array],
) -> None:
    focal = _placed_row(class_rows[PRIEST_CLASS_ID], 5.0, 5.0, health=100.0)
    slowed = _placed_row(class_rows[MAGE_CLASS_ID], 7.0, 5.0, health=40.0)
    slowed = slowed.at[AGENT_FEATURE_SLOW_WARRIOR_CHARGE_DURATION].set(3.0)
    slowed = slowed.at[AGENT_FEATURE_SLOW_WARRIOR_CHARGE_MULTIPLIER].set(0.5)
    stunned = slowed.at[AGENT_FEATURE_STUN_WARRIOR_CHARGE_DURATION].set(2.0)

    _, freedom_trace = _decide_facts(
        _policy_facts(focal, allies=(slowed,)),
        combat_support=((2, 0),),
    )
    _, stunned_trace = _decide_facts(
        _policy_facts(focal, allies=(stunned,)),
        combat_support=((2, 0),),
    )

    assert freedom_trace.combat_selection_basis_components[3] > 0.0
    assert stunned_trace.combat_selection_basis_components[3] == 0.0


def test_policy_modules_have_no_runtime_validator_or_forbidden_dependencies() -> None:
    adapter_source = inspect.getsource(adapter_module)
    policy_source = inspect.getsource(policy_module)
    source_tree = ast.parse(adapter_source + "\n" + policy_source)

    imported_modules = {
        node.module
        for node in ast.walk(source_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(source_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert not any(
        name.startswith(
            (
                "marl_battlegrounds.core.env",
                "marl_battlegrounds.evaluation",
                "marl_battlegrounds.replay",
            )
        )
        for name in imported_modules
    )
    assert not any(
        isinstance(node, (ast.Assert, ast.Raise)) for node in ast.walk(source_tree)
    )
    assert policy_source.count("jax.lax.switch(") == 1
    assert imported_modules.isdisjoint(
        {
            "marl_battlegrounds.policies.scripted.common",
            "marl_battlegrounds.policies.scripted.shared_obs",
            "marl_battlegrounds.policies.scripted.king_of_the_hill",
            "marl_battlegrounds.policies.scripted.capture_the_flag",
        }
    )
    scripted_directory = Path(policy_module.__file__).parent
    for placeholder in (
        "common.py",
        "king_of_the_hill.py",
        "capture_the_flag.py",
    ):
        assert (scripted_directory / placeholder).read_bytes() == b""
    for forbidden_name in (
        "objective_features",
        "previous_timestep_actions",
        "CONTEXT_FEATURE_EPISODE_HORIZON",
        "CONTEXT_FEATURE_TDM_ALLY_SCORE",
        "CONTEXT_FEATURE_TDM_ENEMY_SCORE",
        "CONTEXT_FEATURE_TDM_SCORE_THRESHOLD",
        "TASK_MODE_",
    ):
        assert forbidden_name not in adapter_source


def test_inert_trace_has_zero_basis_direct_stay_and_no_fired_guard(
    class_rows: dict[int, Array],
) -> None:
    focal = _placed_row(class_rows[MAGE_CLASS_ID], 5.0, 5.0, health=80.0)

    action, trace = _decide_facts(
        _policy_facts(focal),
        combat_support=((0, 0),),
    )

    assert (int(action.select_target), int(action.use_ultimate)) == (0, 0)
    assert int(trace.combat_reason_id) == policy_module.EFFECT_INERT_NOOP
    assert int(trace.movement_reason_id) == policy_module.MOVE_DIRECT_SCORE
    assert trace.combat_selection_basis_value == jnp.float32(0.0)
    assert trace.movement_selection_basis_value == jnp.float32(0.0)
    assert bool(jnp.all(trace.combat_selection_basis_components == 0.0))
    assert bool(jnp.all(trace.movement_selection_basis_components == 0.0))
    assert not bool(jnp.any(trace.fired_guards))
    assert int(trace.combat_peer_count) == 1
    assert int(trace.movement_peer_count) == 1


def test_action_and_trace_entrypoints_return_the_same_scalar_int32_action() -> None:
    observation, action_mask = _scalar_policy_inputs()
    key = jax.random.key(7)

    action = team_deathmatch_no_shared_obs_policy(observation, action_mask, key)
    traced_action, trace = decide_team_deathmatch_no_shared_obs(
        observation, action_mask, key
    )

    _assert_tree_arrays_exact(action, traced_action)
    for leaf in jax.tree_util.tree_leaves(action):
        assert leaf.shape == ()
        assert leaf.dtype == jnp.int32
    assert trace.combat_target.shape == ()
    assert trace.combat_target.dtype == jnp.int32
    assert trace.combat_use_ultimate.shape == ()
    assert trace.combat_use_ultimate.dtype == jnp.int32
    assert trace.movement_action.shape == ()
    assert trace.movement_action.dtype == jnp.int32
    assert trace.combat_selection_basis_value.shape == ()
    assert trace.combat_selection_basis_value.dtype == jnp.float32
    assert trace.movement_selection_basis_value.shape == ()
    assert trace.movement_selection_basis_value.dtype == jnp.float32
    assert trace.combat_selection_basis_components.shape == (8,)
    assert trace.combat_selection_basis_components.dtype == jnp.float32
    assert trace.movement_selection_basis_components.shape == (9,)
    assert trace.movement_selection_basis_components.dtype == jnp.float32
    assert trace.combat_reason_id.shape == ()
    assert trace.combat_reason_id.dtype == jnp.int32
    assert trace.movement_reason_id.shape == ()
    assert trace.movement_reason_id.dtype == jnp.int32
    assert trace.fired_guards.shape == (10,)
    assert trace.fired_guards.dtype == jnp.bool_
    assert trace.combat_peer_count.shape == ()
    assert trace.combat_peer_count.dtype == jnp.int32
    assert trace.movement_peer_count.shape == ()
    assert trace.movement_peer_count.dtype == jnp.int32


def test_dead_inactive_and_stunned_masks_produce_the_canonical_inert_action() -> None:
    observation, _ = _scalar_policy_inputs()
    inert_mask = _local_action_mask()
    variants = (
        observation._replace(
            self_features=observation.self_features.at[AGENT_FEATURE_ACTIVE].set(0.0)
        ),
        observation._replace(
            self_features=observation.self_features.at[AGENT_FEATURE_ALIVE].set(0.0)
        ),
        observation._replace(
            self_features=observation.self_features.at[
                AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION
            ].set(1.0)
        ),
    )
    expected = ActorAction(
        move=jnp.asarray(0, dtype=jnp.int32),
        select_target=jnp.asarray(0, dtype=jnp.int32),
        use_ultimate=jnp.asarray(0, dtype=jnp.int32),
    )

    for variant in variants:
        action = team_deathmatch_no_shared_obs_policy(
            variant, inert_mask, jax.random.key(11)
        )
        _assert_tree_arrays_exact(action, expected)


def test_policy_uses_exact_masks_and_ignores_misleading_marginals() -> None:
    observation, _ = _scalar_policy_inputs()
    action_mask = _local_action_mask(
        move_support=(0, 2, 4, 6, 8),
        combat_support=((0, 0), (6, 0)),
    )
    misleading_marginals = action_mask._replace(
        select_target_mask=jnp.ones((NUM_TARGET_ACTIONS,), dtype=jnp.bool_),
        use_ultimate_mask=jnp.ones((NUM_ULTIMATE_ACTIONS,), dtype=jnp.bool_),
    )
    keys = jax.random.split(jax.random.key(13), 32)
    mapped = jax.vmap(team_deathmatch_no_shared_obs_policy, in_axes=(None, None, 0))

    actions = mapped(observation, action_mask, keys)
    misleading_actions = mapped(observation, misleading_marginals, keys)

    assert bool(jnp.all(action_mask.move_mask[actions.move]))
    assert bool(
        jnp.all(
            action_mask.select_target_use_ultimate_joint_mask[
                actions.select_target, actions.use_ultimate
            ]
        )
    )
    _assert_tree_arrays_exact(actions, misleading_actions)


def test_dormant_task_history_and_lifecycle_fields_do_not_change_the_policy() -> None:
    observation, action_mask = _scalar_policy_inputs()
    key = jax.random.key(17)
    context_change = jnp.arange(observation.context_features.size, dtype=jnp.float32)
    context_change = context_change.at[CONTEXT_FEATURE_MAP_WIDTH].set(
        observation.context_features[CONTEXT_FEATURE_MAP_WIDTH]
    )
    context_change = context_change.at[CONTEXT_FEATURE_MAP_HEIGHT].set(
        observation.context_features[CONTEXT_FEATURE_MAP_HEIGHT]
    )
    lifecycle = observation.spawn_lifecycle
    changed_lifecycle = lifecycle._replace(
        spawn_pad_positions_by_agent_by_team=jnp.full_like(
            lifecycle.spawn_pad_positions_by_agent_by_team, 777.0
        ),
        spawn_shield_configured_duration_by_agent=jnp.asarray(999, dtype=jnp.int32),
        spawn_shield_speed_by_agent=jnp.asarray(999.0, dtype=jnp.float32),
        respawn_wave_period_step_count_by_agent_by_team=jnp.full_like(
            lifecycle.respawn_wave_period_step_count_by_agent_by_team, 999
        ),
        respawn_wave_countdowns_by_agent_by_team=jnp.full_like(
            lifecycle.respawn_wave_countdowns_by_agent_by_team, 999
        ),
    )
    changed_observation = observation._replace(
        objective_features=jnp.full_like(observation.objective_features, 999.0),
        context_features=context_change,
        previous_timestep_actions=jax.tree.map(
            lambda leaf: jnp.full_like(leaf, 1),
            observation.previous_timestep_actions,
        ),
        spawn_lifecycle=changed_lifecycle,
    )

    expected = decide_team_deathmatch_no_shared_obs(observation, action_mask, key)
    actual = decide_team_deathmatch_no_shared_obs(changed_observation, action_mask, key)
    _assert_tree_arrays_exact(actual, expected)


def _zero_nonfocal_rows(leaf: Array, focal_global_slot: int) -> Array:
    keep = (jnp.arange(MAX_AGENT_SLOTS) == focal_global_slot).reshape(
        (MAX_AGENT_SLOTS,) + (1,) * (leaf.ndim - 1)
    )
    return jnp.where(keep, leaf, jnp.zeros_like(leaf))


def test_fixed_team_executor_keeps_each_focal_result_independent() -> None:
    config = evaluation_env_config(
        team_sizes=(5, 5),
        task_mode=TASK_MODE_TDM,
        team_deathmatch_score_threshold=3,
    )
    _, observation, action_mask, _ = reset(config, jax.random.key(19))
    actor_keys = jax.random.split(jax.random.key(23), MAX_AGENT_SLOTS)
    focal_global_slot = 0

    baseline = cast(
        ActorAction,
        execute_no_shared_obs_team_policy(
            observation,
            action_mask,
            actor_keys,
            team_deathmatch_no_shared_obs_policy,
            TEAM_A_ID,
        ),
    )
    changed_observation = cast(
        Observation,
        jax.tree.map(
            lambda leaf: _zero_nonfocal_rows(leaf, focal_global_slot), observation
        ),
    )
    changed_joint = (
        jnp.zeros_like(action_mask.select_target_use_ultimate_joint_mask)
        .at[:, 0, 0]
        .set(True)
    )
    changed_move = jnp.zeros_like(action_mask.move_mask).at[:, MOVE_STAY].set(True)
    changed_mask = ActionMask(
        move_mask=changed_move.at[focal_global_slot].set(
            action_mask.move_mask[focal_global_slot]
        ),
        select_target_mask=jnp.any(changed_joint, axis=2)
        .at[focal_global_slot]
        .set(action_mask.select_target_mask[focal_global_slot]),
        use_ultimate_mask=jnp.any(changed_joint, axis=1)
        .at[focal_global_slot]
        .set(action_mask.use_ultimate_mask[focal_global_slot]),
        select_target_use_ultimate_joint_mask=changed_joint.at[focal_global_slot].set(
            action_mask.select_target_use_ultimate_joint_mask[focal_global_slot]
        ),
    )
    changed_keys = (
        jax.random.split(jax.random.key(29), MAX_AGENT_SLOTS)
        .at[focal_global_slot]
        .set(actor_keys[focal_global_slot])
    )
    changed = cast(
        ActorAction,
        execute_no_shared_obs_team_policy(
            changed_observation,
            changed_mask,
            changed_keys,
            team_deathmatch_no_shared_obs_policy,
            TEAM_A_ID,
        ),
    )

    _assert_tree_arrays_exact(
        jax.tree.map(lambda leaf: leaf[0], changed),
        jax.tree.map(lambda leaf: leaf[0], baseline),
    )


def test_policy_owned_workspace_jaxpr_has_no_widened_intermediate(
    class_rows: dict[int, Array],
) -> None:
    focal = _placed_row(class_rows[MAGE_CLASS_ID], 5.0, 5.0)
    target = _placed_row(class_rows[WARRIOR_CLASS_ID], 7.0, 5.0)
    facts = _policy_facts(focal, enemies=(target,))

    def policy_workspaces(
        scoped_facts: policy_module.PolicyFacts,
    ) -> tuple[Array, ...]:
        combat = policy_module._combat_workspaces(scoped_facts)
        mask = jnp.ones((11, 2), dtype=jnp.bool_)
        ordinary = policy_module._ordinary_combat_support(mask, combat[7])
        hunter = policy_module._hunter_combat(
            scoped_facts,
            mask,
            ordinary,
            combat[1],
            combat[7],
        )
        priest = policy_module._priest_combat(
            scoped_facts,
            mask,
            ordinary,
            combat[1],
        )
        movement = policy_module._movement_workspaces(
            scoped_facts,
            jnp.asarray(6, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            combat[2],
            combat[3],
        )
        return (*combat, *hunter, *priest, *movement)

    with jax.enable_x64(True):
        closed = jax.make_jaxpr(policy_workspaces)(facts)
        workspaces = policy_workspaces(facts)

    assert "f64" not in str(closed)
    assert "i64" not in str(closed)
    for leaf in jax.tree_util.tree_leaves(workspaces):
        if jnp.issubdtype(leaf.dtype, jnp.floating):
            assert leaf.dtype == jnp.float32


def test_eager_jit_vmap_key_forms_and_x64_keep_exact_actions_and_dtypes() -> None:
    observation, action_mask = _scalar_policy_inputs()
    typed_key = jax.random.key(31)
    legacy_key = jax.random.key_data(typed_key)

    eager = decide_team_deathmatch_no_shared_obs(observation, action_mask, typed_key)
    repeated = decide_team_deathmatch_no_shared_obs(observation, action_mask, typed_key)
    compiled = jax.jit(decide_team_deathmatch_no_shared_obs)(
        observation, action_mask, typed_key
    )
    legacy = decide_team_deathmatch_no_shared_obs(observation, action_mask, legacy_key)
    _assert_tree_arrays_exact(repeated, eager)
    _assert_tree_arrays_exact(compiled, eager)
    _assert_tree_arrays_exact(legacy, eager)

    batch_observation = cast(
        Observation,
        jax.tree.map(
            lambda leaf: jnp.broadcast_to(leaf, (MAX_AGENTS_PER_TEAM, *leaf.shape)),
            observation,
        ),
    )
    batch_mask = cast(
        ActionMask,
        jax.tree.map(
            lambda leaf: jnp.broadcast_to(leaf, (MAX_AGENTS_PER_TEAM, *leaf.shape)),
            action_mask,
        ),
    )
    batch_keys = jax.random.split(typed_key, MAX_AGENTS_PER_TEAM)
    mapped = jax.vmap(
        decide_team_deathmatch_no_shared_obs,
        in_axes=(0, 0, 0),
    )(batch_observation, batch_mask, batch_keys)
    first_mapped = jax.tree.map(lambda leaf: leaf[0], mapped)
    first_direct = decide_team_deathmatch_no_shared_obs(
        observation, action_mask, batch_keys[0]
    )
    _assert_tree_arrays_exact(first_mapped, first_direct)

    with jax.enable_x64(False):
        x64_off = decide_team_deathmatch_no_shared_obs(
            observation, action_mask, typed_key
        )
    with jax.enable_x64(True):
        public_jaxpr = jax.make_jaxpr(decide_team_deathmatch_no_shared_obs)(
            observation,
            action_mask,
            typed_key,
        )
        x64_on = decide_team_deathmatch_no_shared_obs(
            observation, action_mask, typed_key
        )
    assert "f64" not in str(public_jaxpr)
    assert "i64" not in str(public_jaxpr)
    _assert_tree_arrays_exact(x64_on, x64_off)
    action, trace = x64_on
    for leaf in jax.tree_util.tree_leaves(action):
        assert leaf.dtype == jnp.int32
    for leaf in (
        trace.combat_selection_basis_value,
        trace.movement_selection_basis_value,
        trace.combat_selection_basis_components,
        trace.movement_selection_basis_components,
    ):
        assert leaf.dtype == jnp.float32


def test_public_team_policy_actions_are_mask_legal_and_core_accepted() -> None:
    config = evaluation_env_config(
        team_sizes=(1, 1),
        task_mode=TASK_MODE_TDM,
        team_deathmatch_score_threshold=3,
        max_steps=10,
    )
    state, observation, action_mask, _ = reset(config, jax.random.key(37))
    actor_keys = jax.random.split(jax.random.key(41), MAX_AGENT_SLOTS)
    team_a_actions = cast(
        ActorAction,
        execute_no_shared_obs_team_policy(
            observation,
            action_mask,
            actor_keys,
            team_deathmatch_no_shared_obs_policy,
            TEAM_A_ID,
        ),
    )
    team_b_actions = cast(
        ActorAction,
        execute_no_shared_obs_team_policy(
            observation,
            action_mask,
            actor_keys,
            team_deathmatch_no_shared_obs_policy,
            TEAM_B_ID,
        ),
    )
    submitted = build_joint_action_from_actor_actions(team_a_actions, team_b_actions)
    slots = jnp.arange(MAX_AGENT_SLOTS, dtype=jnp.int32)

    assert bool(jnp.all(action_mask.move_mask[slots, submitted.move]))
    assert bool(
        jnp.all(
            action_mask.select_target_use_ultimate_joint_mask[
                slots, submitted.select_target, submitted.use_ultimate
            ]
        )
    )
    inactive_slots = jnp.asarray((1, 2, 3, 4, 6, 7, 8, 9), dtype=jnp.int32)
    assert bool(jnp.all(submitted.move[inactive_slots] == MOVE_STAY))
    assert bool(jnp.all(submitted.select_target[inactive_slots] == 0))
    assert bool(jnp.all(submitted.use_ultimate[inactive_slots] == 0))

    _, _, _, _, _, info = step(
        config,
        state,
        action_mask,
        submitted,
        jax.random.key(43),
    )
    acceptance = info.transition_facts.action_acceptance_facts
    _assert_tree_arrays_exact(acceptance.submitted_joint_action, submitted)
    _assert_tree_arrays_exact(acceptance.accepted_joint_action, submitted)
    assert not bool(
        jnp.any(acceptance.submitted_action_tuple_is_out_of_domain_by_actor)
    )
    assert not bool(jnp.any(acceptance.in_domain_move_action_is_rejected_by_actor))
    assert not bool(
        jnp.any(acceptance.in_domain_combat_action_pair_is_rejected_by_actor)
    )
    assert NO_SHARED_OBS_ADAPTER_ID == "no_shared_obs"
