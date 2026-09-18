"""Check permitted System inputs, action joining and explicit-step learner data.

The tests compare team preparation with the existing sensor/redaction authorities,
preserve restricted permissions and distinguish scalar, native and external JAX
batches. They also check submitted actions, terminal/padded data and the separate
privileged training-state route without changing Core's rules.
"""

import subprocess
import sys
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from tests.evaluation_fixtures import evaluation_env_config

from marl_battlegrounds import Environment, EnvironmentState, make
from marl_battlegrounds.core import env as core
from marl_battlegrounds.core.types import (
    AGENT_FEATURE_ACTIVE,
    AGENT_FEATURE_ALIVE,
    TASK_MODE_TDM,
    Action,
    DoneFlags,
    EnvConfig,
    Reward,
)
from marl_battlegrounds.environment import EpisodeInfo
from marl_battlegrounds.evaluation.policy_execution import (
    SystemInput,
    SystemStepData,
    system_step_data,
)
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.input import (
    ActorInput,
    Observations,
    build_actor_input,
    build_team_actor_input,
)
from marl_battlegrounds.policies.shared_obs import (
    build_shared_obs_sensor_source_bank,
    mask_source_bank_for_recipient,
)
from marl_battlegrounds.tasks import (
    canonical_tournament_rosters,
    make_standard_team_deathmatch_config,
)

type StepResult = tuple[Observations, EnvironmentState, Reward, DoneFlags, EpisodeInfo]


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
    def take(value: Array) -> Array:
        return value[index]

    return jax.tree.map(take, tree)


def _squeeze_inner_batch(value: Array) -> Array:
    return value[:, 0]


def _idle(batch: int | None = None) -> Action:
    zero = jnp.zeros((10,) if batch is None else (batch, 10), jnp.int32)
    return Action(zero, zero, zero)


@pytest.fixture(scope="module")
def snapshot() -> tuple[Environment, Observations, EnvironmentState]:
    config = make_standard_team_deathmatch_config(
        map_id=12,
        team_a_roster=("mage", "mage"),
        team_b_roster=("hunter", "priest", "hunter"),
        max_steps=2,
    )
    env = make("tdm", env_config=config, metrics="none")
    observations, state = env.reset(jax.random.key(8), episode_id=37)
    return env, observations, state


@pytest.mark.parametrize("team", (0, 1))
def test_team_inputs_keep_supplied_permissions_and_match_sensor_reference(
    snapshot: tuple[Environment, Observations, EnvironmentState], team: int
) -> None:
    _, observations, state = snapshot
    full = build_shared_obs_sensor_source_bank(observations.observation)
    bank = _row(full, team)
    start = team * 5
    permissions = observations.source_availability
    subsets = (
        permissions,
        jnp.zeros_like(permissions),
        permissions.at[start].set(False),
        permissions & (jnp.arange(10)[None, :] % 2 == 0),
    )
    default = build_actor_input(observations.observation, state.config)
    assert default.observation is observations.observation
    for subset in subsets:
        compact = observations._replace(source_availability=subset)
        actual = build_team_actor_input(compact, team)
        assert actual.source_availability.dtype == jnp.bool_
        np.testing.assert_array_equal(
            actual.source_availability, subset[start : start + 5, start : start + 5]
        )
        for local in range(5):
            expected = ActorInput(
                _row(observations.observation, start + local),
                mask_source_bank_for_recipient(
                    bank, subset[start + local, start : start + 5]
                ),
                subset[start + local, start : start + 5],
            )
            _assert_tree_exact(_row(actual, local), expected)
    for local in range(5):
        _assert_tree_exact(
            _row(build_team_actor_input(observations, team), local),
            _row(default, start + local),
        )


@pytest.mark.parametrize("team", (0, 1))
def test_recipient_does_not_receive_forbidden_rows_or_other_private_features(
    snapshot: tuple[Environment, Observations, EnvironmentState], team: int
) -> None:
    _, observations, _ = snapshot
    own = team * 5
    forbidden = observations.source_availability.at[own].set(False)
    restricted = observations._replace(source_availability=forbidden)
    expected = _row(build_team_actor_input(restricted, team), 0)
    others = jnp.arange(10) != own
    base = observations.observation

    def poison(value: Array) -> Array:
        shape = (10, *((1,) * (value.ndim - 1)))
        return jnp.where(others.reshape(shape), jnp.full_like(value, 73), value)

    changed = jax.tree.map(poison, base)
    actual = _row(
        build_team_actor_input(restricted._replace(observation=changed), team), 0
    )
    _assert_tree_exact(actual, expected)
    allowed = observations.source_availability.at[own, own + 1].set(True)
    delivered = _row(
        build_team_actor_input(
            observations._replace(source_availability=allowed), team
        ),
        0,
    )
    changed_delivered = _row(
        build_team_actor_input(Observations(changed, allowed), team), 0
    )
    assert not np.array_equal(
        delivered.source_bank.unit_features_by_source_and_candidate,
        changed_delivered.source_bank.unit_features_by_source_and_candidate,
    )


def test_dead_and_inactive_sources_keep_permission_distinct_from_sensing(
    snapshot: tuple[Environment, Observations, EnvironmentState],
) -> None:
    env, observations, state = snapshot
    base = observations.observation
    dead = base._replace(
        self_features=base.self_features.at[1, AGENT_FEATURE_ALIVE].set(0.0)
    )
    dead_observations = observations._replace(observation=dead)
    dead_state = state._replace(
        observation=dead,
        core_state=state.core_state._replace(
            alive_mask=state.core_state.alive_mask.at[1].set(False)
        ),
    )
    inputs = env.policy_inputs(dead_observations, dead_state)
    assert bool(inputs.valid[0]) and bool(inputs.active_mask[0, 1])
    assert bool(inputs.actors.source_availability[0, 0, 1])
    assert not bool(inputs.active_mask[0, 2])
    for leaf in jax.tree.leaves(inputs.actors.source_bank):
        assert not np.any(np.asarray(leaf)[0, :, 1])
        assert not np.any(np.asarray(leaf)[0, :, 2:])
    restored = env.policy_inputs(observations, state)
    assert np.any(
        np.asarray(restored.actors.source_bank.unit_features_by_source_and_candidate)[
            0, 0, 1
        ]
    )
    assert not bool(base.self_features[2, AGENT_FEATURE_ACTIVE])


@pytest.mark.parametrize("num_envs", (None, 1, 3))
def test_public_inputs_match_scalar_reference_under_jit_and_native_batches(
    snapshot: tuple[Environment, Observations, EnvironmentState],
    num_envs: int | None,
) -> None:
    scalar, observations, state = snapshot
    env = make("tdm", env_config=state.config, num_envs=num_envs, metrics="none")
    if num_envs is not None:
        keys = jax.random.split(jax.random.key(12), num_envs)
        observations, state = env.reset(keys)
    assert SystemInput._fields == (
        "actors",
        "action_mask",
        "active_mask",
        "episode_start",
        "valid",
    )
    batch = 1 if num_envs is None else num_envs
    for team in (0, 1):

        def prepare(
            obs: Observations, current: EnvironmentState, *, team: int = team
        ) -> SystemInput:
            return env.policy_inputs(obs, current, team=team)

        actual = cast(SystemInput, jax.jit(prepare)(observations, state))
        assert actual.active_mask.shape == (batch, 5)
        assert actual.valid.shape == actual.episode_start.shape == (batch,)
        assert actual.actors.source_availability.shape == (batch, 5, 5)
        np.testing.assert_array_equal(actual.valid, np.ones(batch, np.bool_))
        np.testing.assert_array_equal(actual.episode_start, np.ones(batch, np.bool_))
        for index in range(batch):
            obs = observations if num_envs is None else _row(observations, index)
            current = state if num_envs is None else _row(state, index)
            expected = scalar.policy_inputs(obs, current, team=team)
            _assert_tree_exact(_row(actual, index), _row(expected, 0))
            np.testing.assert_array_equal(
                actual.active_mask[index],
                current.config.agent_profile.active_mask[team * 5 : (team + 1) * 5],
            )
            _assert_tree_exact(
                _row(actual.actors, index), build_team_actor_input(obs, team)
            )
            for supplied, original in zip(
                actual.action_mask, current.action_mask, strict=True
            ):
                np.testing.assert_array_equal(
                    supplied[index], original[team * 5 : (team + 1) * 5]
                )
        if num_envs is not None:
            mapped = cast(
                SystemInput,
                jax.jit(jax.vmap(prepare))(observations, state),
            )
            _assert_tree_exact(jax.tree.map(_squeeze_inner_batch, mapped), actual)
    assert env.training_state(state) is state.core_state


@pytest.mark.parametrize("role", (-1, 2, True, 0.0))
def test_input_helpers_reject_invalid_static_roles(
    snapshot: tuple[Environment, Observations, EnvironmentState], role: object
) -> None:
    env, observations, state = snapshot
    with pytest.raises(ValueError, match="team"):
        build_team_actor_input(observations, cast(int, role))
    with pytest.raises(ValueError, match="team"):
        env.policy_inputs(observations, state, team=cast(int, role))


@pytest.mark.parametrize("num_envs", (None, 1, 3))
def test_join_actions_preserves_submitted_values_and_native_axes(
    num_envs: int | None,
) -> None:
    env = make("tdm", num_envs=num_envs, metrics="none")
    batch = 1 if num_envs is None else num_envs
    a = ActorAction(*(jnp.full((batch, 5), value, jnp.int32) for value in (-3, 12, 2)))
    b = ActorAction(*(jnp.full((batch, 5), value, jnp.int32) for value in (10, -1, 5)))
    actual = cast(Action, jax.jit(env.join_actions)(a, b))
    shape = (10,) if num_envs is None else (batch, 10)
    for joined, left, right in zip(actual, a, b, strict=True):
        assert joined.shape == shape and joined.dtype == jnp.int32
        expected = jnp.concatenate((left, right), axis=1)
        np.testing.assert_array_equal(
            joined, expected[0] if num_envs is None else expected
        )
    if num_envs is None:
        _assert_tree_exact(env.join_actions(_row(a, 0), _row(b, 0)), actual)


def test_join_actions_rejects_wrong_structures_shapes_and_dtypes() -> None:
    scalar = make("tdm", metrics="none")
    native = make("tdm", num_envs=3, metrics="none")
    valid = ActorAction(*(jnp.zeros(5, jnp.int32) for _ in range(3)))
    with pytest.raises(TypeError):
        scalar.join_actions(cast(ActorAction, tuple(valid)), valid)
    with pytest.raises(TypeError):
        scalar.join_actions(valid._replace(move=valid.move.astype(jnp.float32)), valid)
    for value in (0, 2**32 + 1):
        wide = valid._replace(move=cast(Array, np.full(5, value, np.int64)))
        with pytest.raises(TypeError):
            scalar.join_actions(wide, valid)
    for bad in (
        valid._replace(move=jnp.zeros(4, jnp.int32)),
        ActorAction(*(jnp.zeros((2, 5), jnp.int32) for _ in range(3))),
    ):
        with pytest.raises(TypeError):
            scalar.join_actions(bad, valid)
    with pytest.raises(TypeError):
        native.join_actions(valid, valid)
    wrong_batch = ActorAction(*(jnp.zeros((2, 5), jnp.int32) for _ in range(3)))
    with pytest.raises(TypeError, match=r"batch|num_envs"):
        native.join_actions(wrong_batch, wrong_batch)


def test_canonical_native_step_data_matches_external_vmap_and_mixed_completion() -> (
    None
):
    a, b = canonical_tournament_rosters()
    config = make_standard_team_deathmatch_config(
        map_id=12, team_a_roster=a, team_b_roster=b, max_steps=3
    )
    configs = jax.tree.map(
        lambda *values: jnp.stack(values),
        *(config._replace(max_steps=limit) for limit in (1, 2, 3)),
    )
    env = make("tdm", env_config=configs, num_envs=3, metrics="none")
    observations, state = env.reset(jax.random.key(11))
    for team in (0, 1):
        inputs = env.policy_inputs(observations, state, team=team)
        np.testing.assert_array_equal(inputs.active_mask, np.ones((3, 5), np.bool_))
    action = _idle(3)
    for expected in ((True, True, True), (False, True, True)):
        before = state
        result = env.step(jax.random.key(13), before, action)
        state = result[1]
        for team in (0, 1):

            def project(
                current: EnvironmentState,
                submitted: Action,
                step_result: StepResult,
                *,
                team: int = team,
            ) -> SystemStepData:
                return system_step_data(current, submitted, step_result, system=team)

            native = system_step_data(before, action, result, system=team)
            np.testing.assert_array_equal(native.advanced, expected)
            np.testing.assert_array_equal(native.episode_id, (1, 2, 3))
            mapped = cast(
                SystemStepData, jax.jit(jax.vmap(project))(before, action, result)
            )
            _assert_tree_exact(jax.tree.map(_squeeze_inner_batch, mapped), native)
        np.testing.assert_array_equal(
            env.policy_inputs(result[0], state).valid, ~state.done.done
        )


def test_step_data_keeps_terminal_transition_and_excludes_later_padding(
    snapshot: tuple[Environment, Observations, EnvironmentState],
) -> None:
    env, _, state = snapshot
    action = _idle()._replace(select_target=jnp.zeros(10, jnp.int32).at[0].set(1))
    for expected_live in (True, True, False):
        before = state
        result = env.step(jax.random.key(16), before, action)
        _, state, reward, _, _ = result
        for team in (0, 1):
            data = system_step_data(before, action, result, system=team)
            assert data._fields == (
                "actions",
                "rewards",
                "active_mask",
                "episode_id",
                "advanced",
            )
            np.testing.assert_array_equal(data.advanced, (expected_live,))
            np.testing.assert_array_equal(data.episode_id, (37,))
            assert data.rewards.dtype == jnp.float32
            assert data.active_mask.dtype == data.advanced.dtype == jnp.bool_
            for supplied, projected in zip(action, data.actions, strict=True):
                np.testing.assert_array_equal(
                    projected, supplied[team * 5 : (team + 1) * 5][None]
                )
            np.testing.assert_array_equal(
                data.rewards, reward.rewards[team * 5 : (team + 1) * 5][None]
            )
        inputs = env.policy_inputs(result[0], state)
        np.testing.assert_array_equal(inputs.valid, (~state.done.done)[None])
        assert not bool(inputs.episode_start[0])
    assert bool(state.done.done)


def test_authored_terminal_rewards_and_advancement_at_counter_limit() -> None:
    config: EnvConfig = evaluation_env_config(
        team_sizes=(2, 3),
        task_mode=TASK_MODE_TDM,
        team_deathmatch_score_threshold=20,
        max_steps=6,
    )._replace(spawn_shield_duration_steps=0)
    key = jax.random.key(21)
    initial, *_ = core.reset(config, key)
    authored = initial._replace(
        step_count=jnp.asarray(5, jnp.int32),
        team_deathmatch_scores=jnp.asarray((19, 0), jnp.int32),
        agent_positions=initial.agent_positions.at[0]
        .set(jnp.asarray((4.0, 4.0), jnp.float32))
        .at[5]
        .set(jnp.asarray((6.5, 4.0), jnp.float32)),
        current_health=initial.current_health.at[5].set(1.0),
    )
    prepared = core.initialize_scenario_state(authored, config)
    env = make("tdm", env_config=config, metrics="none")
    observations, state = env.reset(key, initial=prepared[:3], episode_id=71)
    state = state._replace(
        cumulative_transition_count=jnp.asarray(np.iinfo(np.int32).max, jnp.int32)
    )
    assert bool(env.policy_inputs(observations, state).episode_start[0])
    action = _idle()._replace(select_target=jnp.zeros(10, jnp.int32).at[0].set(6))
    assert bool(state.action_mask.select_target_use_ultimate_joint_mask[0, 6, 0])
    result = env.step(key, state, action)
    assert bool(result[3].done)
    assert int(result[1].cumulative_transition_count) == np.iinfo(np.int32).max
    for team, expected in (
        (0, (1.0, 1.0, 0.0, 0.0, 0.0)),
        (1, (-1.0, -1.0, -1.0, 0.0, 0.0)),
    ):
        data = system_step_data(state, action, result, system=team)
        np.testing.assert_array_equal(data.rewards, np.asarray((expected,), np.float32))
        np.testing.assert_array_equal(data.episode_id, (71,))
        np.testing.assert_array_equal(data.advanced, (True,))
    expected = core.step(config, state.core_state, state.action_mask, action, key)
    assert bool(expected[5].transition_facts.has_transition)
    _assert_tree_exact(result[1].core_state, expected[0])
    for role in (-1, 2, True):
        with pytest.raises(ValueError):
            system_step_data(state, action, result, system=role)


def test_system_public_exports_are_lazy_and_reuse_their_owning_definitions() -> None:
    script = """
import sys
import marl_battlegrounds as marl_bgs
assert "jax" not in sys.modules
assert "marl_battlegrounds.evaluation.policy_execution" not in sys.modules
from marl_battlegrounds import types
assert "marl_battlegrounds.evaluation.policy_execution" not in sys.modules
assert types.System is marl_bgs.System
from marl_battlegrounds.evaluation import policy_execution as owner
for name in (
    "System", "SystemInput", "SystemOutput", "SystemState", "SystemStepData",
):
    assert getattr(types, name) is getattr(marl_bgs, name) is getattr(owner, name)
for name in (
    "shared_policy", "independent_policies", "init_systems", "apply_systems",
    "system_step_data",
):
    assert getattr(marl_bgs, name) is getattr(owner, name)
assert types.PolicyTrace is owner.PolicyTrace
assert "PolicyTrace" not in marl_bgs.__all__
for module in (types, marl_bgs):
    try:
        getattr(module, "_missing_public_type")
    except AttributeError:
        pass
    else:
        raise AssertionError("unknown export must raise AttributeError")
"""
    result = subprocess.run(
        (sys.executable, "-c", script), capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr or result.stdout
