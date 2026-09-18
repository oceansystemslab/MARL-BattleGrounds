"""Check automatic resets, exact final inputs and episode ownership.

CPU cases compare with explicit reset calls, including scalar/native key layouts,
permission restrictions, recurrent memory and checked lifecycle failures. These
checks do not qualify GPU speed or add new simulator rules.
"""

from dataclasses import FrozenInstanceError
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from tests.evaluation_fixtures import evaluation_env_config

from marl_battlegrounds.autoreset import AutoReset
from marl_battlegrounds.core import env as core
from marl_battlegrounds.core.types import Action, DoneFlags
from marl_battlegrounds.environment import (
    Environment,
    EnvironmentState,
    MetricMode,
    make,
)
from marl_battlegrounds.evaluation.policy_execution import (
    PolicyTree,
    System,
    SystemInput,
    apply_systems,
    init_systems,
    system_step_data,
)
from marl_battlegrounds.policies.actor import ActorAction


def _setup(
    batch: int | None = 3,
    *,
    lengths: tuple[int, ...] | None = None,
    metrics: MetricMode = "none",
    include_training_state: bool = False,
) -> tuple[AutoReset, EnvironmentState]:
    config = evaluation_env_config(team_sizes=(2, 3), max_steps=1)
    env = make("tdm", env_config=config, num_envs=batch, metrics=metrics)
    wrapper = AutoReset(env, include_training_state=include_training_state)
    _, state = wrapper.reset(jax.random.key(11))
    if lengths is not None:
        state = state._replace(
            config=state.config._replace(
                max_steps=jnp.asarray(lengths, jnp.int32),
            )
        )
    return wrapper, state


def _idle(batch: int | None) -> Action:
    zero = jnp.zeros((10,) if batch is None else (batch, 10), jnp.int32)
    return Action(zero, zero, zero)


def _equal(left: object, right: object) -> None:
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for actual, expected in zip(
        jax.tree.leaves(left), jax.tree.leaves(right), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype


def _tagged(key: Array) -> Array:
    if jax.random.key_data(key).ndim == 1:
        return jax.random.fold_in(key, 0x4155544F)
    return jax.vmap(jax.random.fold_in, in_axes=(0, None))(key, 0x4155544F)


@pytest.mark.parametrize("batch", [None, 1, 3])
@pytest.mark.parametrize("legacy", [False, True])
def test_autoreset_matches_manual_step_and_reset(
    batch: int | None, legacy: bool
) -> None:
    wrapper, state = _setup(batch)
    key = jax.random.PRNGKey(12) if legacy else jax.random.key(12)
    action = _idle(batch)
    base = wrapper.env.step(key, state, action)
    expected_obs, expected_state = wrapper.env.reset_done(_tagged(key), base[1])
    actual = wrapper.step(key, state, action)
    _equal(actual[:2], (expected_obs, expected_state))
    _equal(actual[2:4], base[2:4])
    _equal(actual[4]._replace(final=None), base[4])
    final = actual[4].final
    assert final is not None
    _equal(final.observations, base[0])
    _equal(final.action_mask, base[1].action_mask)
    _equal(final.episode_id, state.episode_id)
    _equal(final.done, base[3])
    assert final.training_state is None
    assert np.all(final.valid)
    assert np.all(actual[3].done)
    assert not np.any(actual[1].done.done)
    assert np.all(actual[1].reset_generation == state.reset_generation + 1)
    learning = system_step_data(state, action, actual)
    assert np.all(learning.advanced)
    np.testing.assert_array_equal(learning.episode_id, np.atleast_1d(state.episode_id))


@pytest.mark.parametrize("legacy", [False, True])
def test_per_lane_keys_and_partial_resets_match_manual(legacy: bool) -> None:
    wrapper, state = _setup(lengths=(1, 2, 3))
    root = jax.random.PRNGKey(19) if legacy else jax.random.key(19)
    for tick in range(4):
        keys = jax.random.split(jax.random.fold_in(root, tick), 3)
        before = state
        manual = wrapper.env.step(keys, before, _idle(3))
        expected = wrapper.env.reset_done(_tagged(keys), manual[1])
        actual = wrapper.step(keys, before, _idle(3))
        _equal(actual[:2], expected)
        _equal(actual[4]._replace(final=None), manual[4])
        state = actual[1]
        assert actual[4].final is not None
        np.testing.assert_array_equal(actual[4].final.valid, manual[4].completed)
        np.testing.assert_array_equal(
            state.reset_generation,
            before.reset_generation + manual[4].completed,
        )
        for lane in np.flatnonzero(~np.asarray(manual[4].completed)):

            def take(value: Array, index: int = int(lane)) -> Array:
                return value[index]

            _equal(
                jax.tree.map(take, state),
                jax.tree.map(take, manual[1]),
            )


@pytest.mark.parametrize("team", [0, 1])
def test_final_inputs_preserve_permissions_masks_and_terminal_validity(
    team: int,
) -> None:
    wrapper, state = _setup(lengths=(1, 2, 3))
    state = state._replace(
        source_availability=jnp.zeros_like(state.source_availability)
    )
    key = jax.random.key(21)
    raw = wrapper.env.step(key, state, _idle(3))
    actual = wrapper.step(key, state, _idle(3))
    final = actual[4].final
    assert final is not None
    assert not np.any(final.observations.source_availability)
    terminal_inputs = wrapper.final_policy_inputs(actual[4], team=team)
    expected_inputs = wrapper.env.policy_inputs(
        raw[0],
        raw[1]._replace(
            done=DoneFlags(jnp.zeros(3, bool), jnp.zeros(3, bool)),
            episode_start=jnp.zeros(3, bool),
        ),
        team=team,
    )._replace(valid=raw[4].completed)
    _equal(terminal_inputs, expected_inputs)
    np.testing.assert_array_equal(terminal_inputs.valid, (True, False, False))
    assert not np.any(terminal_inputs.episode_start)
    # The replacement uses its ordinary permissions; the old view stays restricted.
    assert np.any(actual[0].source_availability[0])
    assert not np.any(actual[0].source_availability[1:])


def test_padding_resets_but_never_becomes_a_second_completion() -> None:
    wrapper, state = _setup(None)
    key = jax.random.key(3)
    terminal = wrapper.env.step(key, state, _idle(None))[1]
    result = wrapper.step(key, terminal, _idle(None))
    assert int(result[4].decision_step) == -1
    assert result[4].final is not None
    assert not bool(result[4].final.valid)
    assert not bool(result[4].completed)
    assert not np.any(result[2].rewards)
    assert not bool(result[1].done.done)
    assert int(result[1].cumulative_transition_count) == 1


def test_training_state_is_optional_and_keeps_the_old_epoch() -> None:
    wrapper, state = _setup(1, include_training_state=True)
    result = wrapper.step(jax.random.key(1), state, _idle(1))
    final = result[4].final
    assert final is not None and final.training_state is not None
    assert int(final.training_state.step_count[0]) == 1
    assert int(result[1].core_state.step_count[0]) == 0
    assert wrapper.training_state(result[1]) is result[1].core_state


def test_checked_reset_failures_reach_old_info() -> None:
    wrapper, state = _setup(None)
    state = state._replace(last_reserved_episode_id=jnp.asarray(np.iinfo(np.int32).max))
    result = wrapper.step(jax.random.key(1), state, _idle(None))
    assert bool(result[4].completed)
    assert bool(result[1].lifecycle_error)
    assert bool(result[4].lifecycle_error)
    assert int(result[4].episode_id) == int(state.episode_id)


def test_authored_start_keeps_local_epoch_and_exact_final_snapshot() -> None:
    config = evaluation_env_config(team_sizes=(2, 3), max_steps=8)
    base = make("tdm", env_config=config, metrics="none")
    core_state, _, _, _ = core.reset(config, jax.random.key(4))
    authored = core_state._replace(step_count=jnp.asarray(7, jnp.int32))
    initialized, observations, masks, _ = core.initialize_scenario_state(
        authored, config
    )
    wrapper = AutoReset(base)
    _, state = wrapper.reset(
        jax.random.key(5),
        initial=(initialized, observations, masks),
        episode_id=9,
    )
    result = wrapper.step(jax.random.key(6), state, _idle(None))
    assert int(result[4].decision_step) == 0
    assert int(result[4].episode_length) == 1
    assert result[4].final is not None and bool(result[4].final.valid)
    assert int(result[4].episode_id) == 9
    assert int(result[1].episode_id) == 10
    assert not bool(result[1].authored_start)


def _initialize(variables: PolicyTree, inputs: SystemInput, keys: Array) -> Array:
    del variables, keys
    return jnp.zeros(inputs.valid.shape, jnp.int32)


def _choose(
    variables: PolicyTree,
    memory: Array,
    inputs: SystemInput,
    keys: Array,
) -> tuple[ActorAction, Array]:
    del variables, keys
    zero = jnp.zeros(inputs.active_mask.shape, jnp.int32)
    return ActorAction(zero, zero, zero), memory + inputs.valid.astype(jnp.int32)


def test_recurrent_memory_resets_on_next_decision_only() -> None:
    wrapper, state = _setup(lengths=(1, 2, 3))
    system = System("counter", _choose, init=_initialize)
    observations = wrapper.get_observations(state)
    memory = init_systems(system, system, observations, state, jax.random.key(7))
    actions, memory, _ = apply_systems(
        system,
        system,
        memory,
        observations,
        state,
        jax.random.key(8),
    )
    result = wrapper.step(jax.random.key(9), state, actions)
    np.testing.assert_array_equal(memory.team_a, (1, 1, 1))
    assert np.all(memory.policy_trace.valid)
    np.testing.assert_array_equal(memory.policy_trace.episode_id, state.episode_id)
    _, next_memory, _ = apply_systems(
        system,
        system,
        memory,
        result[0],
        result[1],
        jax.random.key(10),
    )
    np.testing.assert_array_equal(next_memory.team_a, (1, 2, 2))
    np.testing.assert_array_equal(next_memory.team_b, (1, 2, 2))


def test_compiled_scan_and_external_vmap_keep_dynamic_handles() -> None:
    wrapper, state = _setup(3, lengths=(1, 2, 3))

    traces: list[int] = []

    @jax.jit
    def run(handle: AutoReset, initial: EnvironmentState) -> EnvironmentState:
        traces.append(1)

        def advance(
            carry: EnvironmentState, key: Array
        ) -> tuple[EnvironmentState, Array]:
            result = handle.step(key, carry, _idle(3))
            return result[1], result[4].completed

        return jax.lax.scan(advance, initial, jax.random.split(jax.random.key(3), 3))[0]

    result = cast(EnvironmentState, run(wrapper, state))
    np.testing.assert_array_equal(result.cumulative_transition_count, (3, 3, 3))
    np.testing.assert_array_equal(result.reset_generation, (3, 1, 1))
    changed = state._replace(
        config=state.config._replace(max_steps=jnp.full(3, 2, jnp.int32))
    )
    second = cast(EnvironmentState, run(wrapper, changed))
    np.testing.assert_array_equal(second.reset_generation, (1, 1, 1))
    assert traces == [1]

    scalar, _ = _setup(None)
    keys = jax.random.split(jax.random.key(8), 3)
    _, states = jax.vmap(scalar.reset)(keys)
    mapped = cast(
        tuple[object, ...], jax.jit(jax.vmap(scalar.step))(keys, states, _idle(3))
    )
    from marl_battlegrounds.environment import EpisodeInfo

    info = cast(EpisodeInfo, mapped[4])
    assert info.final is not None
    np.testing.assert_array_equal(info.final.valid, (True, True, True))


def test_invalid_wrappers_teams_and_missing_final_are_rejected() -> None:
    wrapper, state = _setup(None)
    with pytest.raises(TypeError, match="nesting"):
        AutoReset(cast(Environment, wrapper))
    with pytest.raises(TypeError, match="bool"):
        AutoReset(wrapper.env, include_training_state=cast(bool, 1))
    with pytest.raises(FrozenInstanceError):
        wrapper.include_training_state = True  # pyright: ignore[reportAttributeAccessIssue]
    result = wrapper.env.step(jax.random.key(2), state, _idle(None))
    with pytest.raises(ValueError, match="AutoReset"):
        wrapper.final_policy_inputs(result[4])
    for team in (True, -1, 2):
        with pytest.raises(ValueError):
            wrapper.final_policy_inputs(result[4], team=team)


def test_forwarded_raw_helpers_keep_base_behavior() -> None:
    wrapper, state = _setup(None)
    key = jax.random.key(33)
    _equal(wrapper.get_observations(state), wrapper.env.get_observations(state))
    _equal(wrapper.get_action_mask(state), wrapper.env.get_action_mask(state))
    _equal(wrapper.agent_info(state), wrapper.env.agent_info(state))
    _equal(wrapper.sample_actions(key, state), wrapper.env.sample_actions(key, state))
    assert wrapper.agents == wrapper.env.agents
    assert wrapper.num_agents == 10 and wrapper.num_envs is None
    assert wrapper.metrics == "none"
    assert wrapper.full_metrics_episodes == wrapper.replay_episodes == ()
    assert wrapper.action_space("agent_0") is wrapper.env.action_space("agent_0")
    assert wrapper.observation_space("agent_0") is wrapper.env.observation_space(
        "agent_0"
    )
    _equal(wrapper.reset_done(key, state), wrapper.env.reset_done(key, state))
    team = ActorAction(*(jnp.zeros(5, jnp.int32) for _ in range(3)))
    _equal(wrapper.join_actions(team, team), wrapper.env.join_actions(team, team))


class _HostMemory:
    def __init__(self) -> None:
        self.decisions = 0

    def __array__(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("opaque memory must not enter final arrays")


def test_host_memory_and_callback_counts_are_untouched_until_next_decision() -> None:
    wrapper, state = _setup(lengths=(1, 2, 3))
    initialized: list[int] = []
    decisions: list[int] = []

    def initialize(
        variables: PolicyTree,
        inputs: SystemInput,
        keys: Array,
    ) -> list[_HostMemory | None]:
        del variables, keys
        initialized.append(1)
        return [_HostMemory() if valid else None for valid in inputs.valid]

    def choose(
        variables: PolicyTree,
        memory: list[_HostMemory | None],
        inputs: SystemInput,
        keys: Array,
    ) -> tuple[ActorAction, list[_HostMemory | None]]:
        del variables, keys
        decisions.append(1)
        zero = cast(Array, np.zeros(inputs.active_mask.shape, np.int32))
        for item, valid in zip(memory, inputs.valid, strict=True):
            if item is not None and valid:
                item.decisions += 1
        return ActorAction(zero, zero, zero), memory

    host = System("host", choose, init=initialize, execution="host")
    numerical = System("counter", _choose, init=_initialize)
    observations = wrapper.get_observations(state)
    memory = init_systems(host, numerical, observations, state, jax.random.key(41))
    actions, memory, _ = apply_systems(
        host,
        numerical,
        memory,
        observations,
        state,
        jax.random.key(42),
    )
    previous = memory.team_a.copy()
    init_count = len(initialized)
    result = wrapper.step(jax.random.key(43), state, actions)
    wrapper.final_policy_inputs(result[4])
    assert len(decisions) == 1 and len(initialized) == init_count
    assert all(
        old is current for old, current in zip(previous, memory.team_a, strict=True)
    )
    _, memory, _ = apply_systems(
        host,
        numerical,
        memory,
        result[0],
        result[1],
        jax.random.key(44),
    )
    assert len(decisions) == 2
    assert memory.team_a[0] is not previous[0]
    assert memory.team_a[1] is previous[1] and memory.team_a[2] is previous[2]
    assert [item.decisions for item in memory.team_a] == [1, 2, 2]


@pytest.mark.parametrize("metrics", ["none", "priority", "full"])
def test_selected_captures_remain_owned_by_old_episode(metrics: MetricMode) -> None:
    config = evaluation_env_config(team_sizes=(2, 3), max_steps=1)
    base = make(
        "tdm",
        env_config=config,
        metrics=metrics,
        full_metrics_episodes=(1,),
        replay_episodes=(1,),
    )
    wrapper = AutoReset(base)
    _, state = wrapper.reset(jax.random.key(23))
    key = jax.random.key(24)
    raw = base.step(key, state, _idle(None))
    result = wrapper.step(key, state, _idle(None))
    _equal(result[4]._replace(final=None), raw[4])
    assert result[4].full is not None and result[4].replay is not None
    assert int(result[4].episode_id) == 1 and int(result[1].episode_id) == 2
    np.testing.assert_array_equal(result[4].config.max_steps, 1)
    assert result[4].final is not None
    assert wrapper.final_policy_inputs(result[4]).active_mask.shape == (1, 5)


def test_no_completion_keeps_state_and_marks_all_final_rows_invalid() -> None:
    wrapper, state = _setup(lengths=(2, 3, 4))
    key = jax.random.key(56)
    raw = wrapper.env.step(key, state, _idle(3))
    result = wrapper.step(key, state, _idle(3))
    _equal(result[:4], raw[:4])
    _equal(result[4]._replace(final=None), raw[4])
    assert result[4].final is not None
    assert not np.any(result[4].final.valid)
    assert not np.any(wrapper.final_policy_inputs(result[4]).valid)
    np.testing.assert_array_equal(result[1].episode_id, state.episode_id)
    np.testing.assert_array_equal(result[1].reset_generation, state.reset_generation)
