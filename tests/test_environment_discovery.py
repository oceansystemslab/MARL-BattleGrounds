"""Check agent discovery, space descriptions and legal action sampling.

A ready map built by make uses the Red Zone depth 5.0 when none is given and
forwards a supplied depth, including 0.0, to every game's config and to the
depth column every configured actor sees.
"""

from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import marl_battlegrounds as marl_bgs
from marl_battlegrounds import EnvironmentState, tasks
from marl_battlegrounds import types as public_types
from marl_battlegrounds.core import env as core
from marl_battlegrounds.core import types as core_types
from marl_battlegrounds.core.types import (
    CONTEXT_FEATURE_TDM_RED_ZONE_DEPTH,
    Action,
    EnvConfig,
)
from marl_battlegrounds.policies.actor import (
    ActorAction,
    build_joint_action_from_actor_actions,
)
from marl_battlegrounds.policies.no_shared_obs import execute_no_shared_obs_team_policy
from marl_battlegrounds.policies.random_valid import random_policy
from marl_battlegrounds.spaces import ArraySpace, StructuredSpace


def _config() -> EnvConfig:
    return tasks.make_standard_team_deathmatch_config(
        map_id=12,
        team_a_roster=("mage", "mage", "priest"),
        team_b_roster=("hunter",),
        max_steps=3,
    )


def _assert_tree_exact(actual: object, expected: object) -> None:
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        a, b = np.asarray(left), np.asarray(right)
        assert a.shape == b.shape
        assert a.dtype == b.dtype
        np.testing.assert_array_equal(a, b)


def _row[T](tree: T, index: int) -> T:
    def select(value: Array) -> Array:
        return value[index]

    return jax.tree.map(select, tree)


def _reference_actions(key: Array, state: EnvironmentState) -> Action:
    actor_keys = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(
        key, jnp.arange(core_types.MAX_AGENT_SLOTS, dtype=jnp.int32)
    )
    team_a = cast(
        ActorAction,
        execute_no_shared_obs_team_policy(
            state.observation,
            state.action_mask,
            actor_keys,
            random_policy,
            core_types.TEAM_A_ID,
        ),
    )
    team_b = cast(
        ActorAction,
        execute_no_shared_obs_team_policy(
            state.observation,
            state.action_mask,
            actor_keys,
            random_policy,
            core_types.TEAM_B_ID,
        ),
    )
    return build_joint_action_from_actor_actions(team_a, team_b)


def test_public_type_and_discovery_exports_reuse_existing_authorities() -> None:
    for name in ("Action", "ActionMask", "EnvConfig", "DoneFlags", "Observation"):
        assert getattr(public_types, name) is getattr(core_types, name)
    assert public_types.ActorAction is ActorAction
    for name in (
        "list_tdm_maps",
        "list_tdm_scenarios",
        "load_tdm_scenario",
        "canonical_tournament_rosters",
        "balanced_spawn_configs",
    ):
        assert getattr(marl_bgs, name) is getattr(tasks, name)


def test_discovery_describes_real_actor_values_and_keeps_metadata_separate() -> None:
    env = marl_bgs.make("tdm", env_config=_config(), metrics="none")
    observation, state = env.reset(jax.random.key(11))
    actions = env.sample_actions(jax.random.key(12), state)
    assert env.agents == tuple(f"agent_{slot}" for slot in range(10))
    assert env.num_agents == 10
    info = env.agent_info(state)
    assert set(info) == {"class_ids", "team_ids", "active_mask"}
    for name, values in info.items():
        _assert_tree_exact(values, getattr(state.config.agent_profile, name))
    assert int(jnp.sum(info["active_mask"])) == 4
    assert int(info["class_ids"][0]) == int(info["class_ids"][1])

    for slot, name in enumerate(env.agents):
        observation_space = env.observation_space(name)
        action_space = env.action_space(name)
        assert tuple(observation_space.fields) == core_types.Observation._fields
        assert tuple(action_space.fields) == ActorAction._fields
        assert observation_space.contains(_row(observation.observation, slot))
        actor_action = ActorAction(*(value[slot] for value in actions))
        assert action_space.contains(actor_action)
        assert action_space.contains(actor_action._asdict())
        for field, count in (
            ("move", core_types.NUM_MOVE_ACTIONS),
            ("select_target", core_types.NUM_TARGET_ACTIONS),
            ("use_ultimate", core_types.NUM_ULTIMATE_ACTIONS),
        ):
            space = action_space.fields[field]
            assert isinstance(space, ArraySpace)
            assert space.shape == () and space.dtype == np.dtype(np.int32)
            assert space.low == 0 and space.high == count - 1

    # The same descriptors also accept a real successor, including action history.
    successor_observation, successor, *_ = env.step(jax.random.key(13), state, actions)
    for slot, name in enumerate(env.agents):
        assert env.observation_space(name).contains(
            _row(successor_observation.observation, slot)
        )
    _assert_tree_exact(env.agent_info(successor), info)


def test_spaces_reject_wrong_shapes_dtypes_bounds_and_named_fields() -> None:
    env = marl_bgs.make("tdm", env_config=_config(), metrics="none")
    observation, state = env.reset(jax.random.key(11))
    actor = _row(observation.observation, 0)
    observation_space = env.observation_space("agent_0")
    action_space = env.action_space("agent_0")
    valid = ActorAction(*(jnp.asarray(0, jnp.int32) for _ in range(3)))
    assert action_space.contains(valid)
    for invalid in (
        valid._replace(move=np.asarray(0, np.int64)),
        valid._replace(move=np.asarray((0,), np.int32)),
        valid._replace(move=np.asarray(core_types.NUM_MOVE_ACTIONS, np.int32)),
        valid._replace(select_target=np.asarray(-1, np.int32)),
        {"move": valid.move, "select_target": valid.select_target},
        dict(valid._asdict(), extra=np.asarray(0, np.int32)),
    ):
        assert not action_space.contains(invalid)
    assert not observation_space.contains(
        actor._replace(self_features=np.asarray(actor.self_features, np.float64))
    )
    assert not observation_space.contains(
        actor._replace(self_features=actor.self_features[None])
    )
    bad_lifecycle = actor.spawn_lifecycle._replace(
        active_mask_by_agent_by_team=actor.spawn_lifecycle.active_mask_by_agent_by_team.astype(
            jnp.int32
        )
    )
    assert not observation_space.contains(actor._replace(spawn_lifecycle=bad_lifecycle))
    assert not observation_space.contains(dict(actor._asdict(), agent_id="agent_0"))
    assert not observation_space.contains(state)
    assert isinstance(observation_space.fields["spawn_lifecycle"], StructuredSpace)
    for name in ("agent_-1", "agent_10", "agent_00", "mage", ""):
        with pytest.raises(ValueError, match="agent"):
            env.action_space(name)
        with pytest.raises(ValueError, match="agent"):
            env.observation_space(name)


def test_space_membership_does_not_replace_joint_action_legality() -> None:
    env = marl_bgs.make("tdm", env_config=_config(), metrics="none")
    _, state = env.reset(jax.random.key(11))
    joint = jnp.zeros_like(state.action_mask.select_target_use_ultimate_joint_mask)
    joint = joint.at[:, 0, 0].set(True).at[:, 6, 1].set(True)
    state = state._replace(
        action_mask=state.action_mask._replace(
            select_target_mask=jnp.ones_like(state.action_mask.select_target_mask),
            use_ultimate_mask=jnp.ones_like(state.action_mask.use_ultimate_mask),
            select_target_use_ultimate_joint_mask=joint,
        )
    )
    forbidden = ActorAction(
        jnp.asarray(0, jnp.int32), jnp.asarray(6, jnp.int32), jnp.asarray(0, jnp.int32)
    )
    assert env.action_space("agent_0").contains(forbidden)
    assert bool(state.action_mask.select_target_mask[0, forbidden.select_target])
    assert bool(state.action_mask.use_ultimate_mask[0, forbidden.use_ultimate])
    assert not bool(joint[0, forbidden.select_target, forbidden.use_ultimate])
    keys = jax.random.split(jax.random.key(12), 32)
    actions = cast(
        Action, jax.jit(jax.vmap(env.sample_actions, in_axes=(0, None)))(keys, state)
    )
    slots = jnp.arange(10)[None, :]
    assert bool(jnp.all(joint[slots, actions.select_target, actions.use_ultimate]))


@pytest.mark.parametrize("num_envs", (None, 2), ids=("scalar", "native"))
def test_sample_actions_preserves_actor_keys_dead_slots_and_core_acceptance(
    num_envs: int | None,
) -> None:
    config = _config()
    initial, *_ = core.reset(config, jax.random.key(11))
    dead = initial._replace(
        alive_mask=initial.alive_mask.at[0].set(False),
        current_health=initial.current_health.at[0].set(0.0),
    )
    prepared = core.initialize_scenario_state(dead, config)[:3]
    if num_envs is not None:

        def stack(value: Array) -> Array:
            return jnp.stack((value, value))

        prepared = jax.tree.map(stack, prepared)
    env = marl_bgs.make("tdm", env_config=config, num_envs=num_envs, metrics="none")
    ids = jnp.asarray(37 if num_envs is None else (37, 105), jnp.int32)
    _, state = env.reset(jax.random.key(11), episode_id=ids, initial=prepared)
    key = jax.random.key(12)
    compiled = cast(
        Callable[[Array, EnvironmentState], Action], jax.jit(env.sample_actions)
    )
    actions = compiled(key, state)
    _assert_tree_exact(actions, env.sample_actions(key, state))
    _assert_tree_exact(actions, compiled(jax.random.key_data(key), state))
    expected_shape = (10,) if num_envs is None else (num_envs, 10)
    for value in actions:
        assert value.shape == expected_shape and value.dtype == jnp.int32

    if num_envs is None:
        expected = _reference_actions(key, state)
    else:
        lane_keys = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(key, ids)
        expected = jax.vmap(_reference_actions)(lane_keys, state)
        _assert_tree_exact(compiled(lane_keys, state), expected)
        _assert_tree_exact(compiled(jax.random.key_data(lane_keys), state), expected)
        info = env.agent_info(state)
        for name, values in info.items():
            assert values.shape == (num_envs, 10)
            _assert_tree_exact(values, getattr(state.config.agent_profile, name))
    _assert_tree_exact(actions, expected)
    unavailable = ~state.config.agent_profile.active_mask | ~state.core_state.alive_mask
    for value in actions:
        np.testing.assert_array_equal(np.asarray(value)[np.asarray(unavailable)], 0)

    step_key = jax.random.key(13)
    observation, successor, reward, done, _ = env.step(step_key, state, actions)
    for lane in range(num_envs or 1):
        prior = state if num_envs is None else _row(state, lane)
        action = actions if num_envs is None else _row(actions, lane)
        lane_key = (
            step_key if num_envs is None else jax.random.fold_in(step_key, ids[lane])
        )
        reference = core.step(
            prior.config, prior.core_state, prior.action_mask, action, lane_key
        )
        actual = (
            successor.core_state,
            observation.observation,
            reward,
            done,
            successor.action_mask,
        )
        _assert_tree_exact(
            actual if num_envs is None else _row(actual, lane), reference[:5]
        )
        acceptance = reference[5].transition_facts.action_acceptance_facts
        _assert_tree_exact(acceptance.accepted_joint_action, action)
        assert not bool(
            jnp.any(acceptance.submitted_action_tuple_is_out_of_domain_by_actor)
        )
        assert not bool(jnp.any(acceptance.in_domain_move_action_is_rejected_by_actor))
        assert not bool(
            jnp.any(acceptance.in_domain_combat_action_pair_is_rejected_by_actor)
        )


@pytest.mark.parametrize(
    "num_envs", (None, 1, 3, 128), ids=("scalar", "one", "odd", "normal")
)
def test_ready_map_construction_supports_explicit_unbalanced_shapes(
    num_envs: int | None,
) -> None:
    env = marl_bgs.make(
        "tdm",
        map_id=12,
        num_envs=num_envs,
        balance_spawn_locations=False,
        max_steps=1,
        metrics="none",
    )
    observation, state = env.reset(jax.random.key(11))
    actions = env.sample_actions(jax.random.key(12), state)
    prefix = () if num_envs is None else (num_envs,)
    assert state.episode_id.shape == prefix
    assert state.config.agent_profile.active_mask.shape == (*prefix, 10)
    assert observation.observation.self_features.shape == (
        *prefix,
        10,
        core_types.SELF_FEATURES,
    )
    for value in actions:
        assert value.shape == (*prefix, 10) and value.dtype == jnp.int32
    actor = (
        _row(observation.observation, 0)
        if num_envs is None
        else _row(_row(observation.observation, num_envs - 1), 0)
    )
    assert env.observation_space("agent_0").contains(actor)
    if num_envs is not None:
        np.testing.assert_array_equal(
            state.config.team_spawn_pad_positions,
            np.broadcast_to(
                state.config.team_spawn_pad_positions[0], (num_envs, 2, 5, 2)
            ),
        )


def test_ready_map_defaults_balance_banks_and_raw_handles_require_config() -> None:
    env = marl_bgs.make("tdm", map_id=12, num_envs=2)
    _, state = env.reset(jax.random.key(11))
    team_a, team_b = marl_bgs.canonical_tournament_rosters()
    source = tasks.make_standard_team_deathmatch_config(
        map_id=12, team_a_roster=team_a, team_b_roster=team_b
    )
    np.testing.assert_array_equal(
        state.config.team_spawn_pad_positions[0], source.team_spawn_pad_positions
    )
    np.testing.assert_array_equal(
        state.config.team_spawn_pad_positions[1], source.team_spawn_pad_positions[::-1]
    )
    assert env.metrics == "priority"
    assert state.priority is not None and state.full is None
    for num_envs in (None, 1, 3):
        with pytest.raises(ValueError, match=r"even|balance"):
            marl_bgs.make("tdm", map_id=12, num_envs=num_envs)
    raw = marl_bgs.make("tdm", metrics="none")
    with pytest.raises(ValueError, match=r"config|map"):
        raw.reset(jax.random.key(11))
    _, explicit = raw.reset(jax.random.key(11), env_config=source)
    assert int(explicit.episode_id) == 1
    np.testing.assert_array_equal(
        explicit.config.agent_profile.class_ids, source.agent_profile.class_ids
    )


@pytest.mark.parametrize(
    ("depth_choice", "expected_depth"),
    ((None, 5.0), (6.0, 6.0), (0.0, 0.0)),
    ids=("omitted-uses-default", "supplied-six", "supplied-zero"),
)
def test_ready_map_red_zone_depth_defaults_to_five_and_forwards_a_supplied_value(
    depth_choice: float | None, expected_depth: float
) -> None:
    if depth_choice is None:
        env = marl_bgs.make("tdm", map_id=0, num_envs=2, metrics="none")
    else:
        env = marl_bgs.make(
            "tdm", map_id=0, num_envs=2, metrics="none", red_zone_depth=depth_choice
        )
    assert tasks.DEFAULT_TDM_RED_ZONE_DEPTH == 5.0
    observation, state = env.reset(jax.random.key(13))
    np.testing.assert_array_equal(
        state.config.team_deathmatch_red_zone_depth,
        np.full((2,), expected_depth, dtype=np.float32),
    )
    configured = np.asarray(state.config.agent_profile.active_mask)
    np.testing.assert_array_equal(
        np.asarray(observation.observation.context_features)[
            ..., CONTEXT_FEATURE_TDM_RED_ZONE_DEPTH
        ],
        np.where(configured, np.float32(expected_depth), np.float32(0.0)),
    )
