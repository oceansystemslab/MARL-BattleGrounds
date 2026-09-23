"""Check a named pinned System inside the compiled Team B opponent wrapper.

With a JAX System pinned, the lanes assigned to history slot 0 act exactly as
Team B of M8's own apply_systems with that System, for rosters of three and
five actors, and the other lanes act exactly as the unpinned wrapper; a step
with no pinned lane never calls the pinned System and leaves its memory
unchanged; Systems that return a 2-tuple, a 3-tuple or a SystemOutput route
alike; a Policy adapter uses the smallest fitting quarter, half or full batch,
with no repeated capacities for tiny batches. Exact actor call counts cover
zero pins and both capacity boundaries; actions, memory and the full random
key words equal M8's in every route; when a
game ends, the new game's pinned lane starts from fresh memory and a lane
moved off the pinned System stops using it, exactly as M8 resets memory, for a
generic System and a Policy adapter, including switches between all three
capacities; a Policy whose NumPy memory template is
changed after setup still starts pinned games from the snapshot; a pinned
System with its own reset_memory hook and a lane-second memory layout runs
through the wrapper's combined hook and matches M8's direct route; and a
host-execution System is refused as the wrapper's pinned System.
"""

from __future__ import annotations

from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from tests.evaluation_fixtures import evaluation_env_config

from marl_battlegrounds.baselines.ppo import initialize_ppo, make_recurrent_mappo_system
from marl_battlegrounds.core.types import Action, ActionMask
from marl_battlegrounds.environment import Environment, EnvironmentState, make
from marl_battlegrounds.evaluation.policy_execution import (
    Policy,
    System,
    SystemApply,
    SystemInput,
    SystemOutput,
    apply_systems,
    init_systems,
    policy,
    shared_policy,
)
from marl_battlegrounds.evaluation.system_evaluation import (
    freeze_evaluation_method,
    prepare_evaluation_system,
)
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.input import ActorInput, Observations
from marl_battlegrounds.training.opponents import (
    OpponentHistory,
    init_opponent_history,
    make_opponent_system,
)

type Tree = Any
HEADS = ("move", "select_target", "use_ultimate")
PINNED = np.array([False, True, False, True])


def _setup(
    team_size: int, num_envs: int = 4
) -> tuple[Environment, Observations, EnvironmentState]:
    env = make(
        "tdm",
        env_config=evaluation_env_config(
            team_sizes=(team_size, team_size), max_steps=20
        ),
        num_envs=num_envs,
        metrics="none",
    )
    observations, state = env.reset(jax.random.key(40))
    return env, observations, state


@pytest.fixture(scope="module")
def weights() -> Tree:
    return initialize_ppo(jax.random.key(30)).actor_params


def _fill_slot_zero(bank: Array, leaf: Array) -> Array:
    return bank.at[0].set(leaf)


def _history(weights: Tree, snapshot: tuple[int, ...]) -> OpponentHistory:
    history = init_opponent_history(weights, num_envs=len(snapshot))
    return history._replace(
        count=jnp.int32(1),
        historical_variables=jax.tree.map(
            _fill_slot_zero, history.historical_variables, weights
        ),
        lane_snapshot=jnp.asarray(snapshot, jnp.int32),
    )


def _team_b(actions: Action) -> dict[str, np.ndarray]:
    return {name: np.asarray(getattr(actions, name))[:, 5:] for name in HEADS}


def _wrapper_step(
    weights: Tree, pinned: System, snapshot: tuple[int, ...], team_size: int = 3
) -> tuple[Action, Tree]:
    _, observations, state = _setup(team_size, len(snapshot))
    actor = make_recurrent_mappo_system(weights, spawn_frame="world")
    _, variables, template = prepare_evaluation_system(pinned)
    wrapper = make_opponent_system(actor, pinned=pinned)
    wrapped = (_history(weights, snapshot), variables, template)
    memory = init_systems(
        actor,
        wrapper,
        observations,
        state,
        jax.random.key(7),
        variables_a=weights,
        variables_b=wrapped,
    )
    actions, after, _ = apply_systems(
        actor,
        wrapper,
        memory,
        observations,
        state,
        jax.random.key(8),
        variables_a=weights,
        variables_b=wrapped,
    )
    return actions, after.team_b


def _direct_step(
    weights: Tree, pinned: System, team_size: int = 3, num_envs: int = 4
) -> tuple[Action, Tree]:
    _, observations, state = _setup(team_size, num_envs)
    actor = make_recurrent_mappo_system(weights, spawn_frame="world")
    memory = init_systems(
        actor, pinned, observations, state, jax.random.key(7), variables_a=weights
    )
    actions, after, _ = apply_systems(
        actor,
        pinned,
        memory,
        observations,
        state,
        jax.random.key(8),
        variables_a=weights,
    )
    return actions, after.team_b


def _run_pinned(
    weights: Tree,
    pinned: System,
    snapshot: tuple[int, ...],
    team_size: int = 3,
) -> tuple[Action, Tree, Action, Tree]:
    actions, memory = _wrapper_step(weights, pinned, snapshot, team_size)
    direct, direct_memory = _direct_step(weights, pinned, team_size, len(snapshot))
    return actions, memory, direct, direct_memory


@pytest.mark.parametrize("team_size", [3, 5])
def test_pinned_lanes_match_m8_and_other_lanes_match_the_plain_wrapper(
    weights: Tree, team_size: int
) -> None:
    alpha = shared_policy(cast(Policy, freeze_evaluation_method(policy("tdm-alpha"))))
    actions, _, direct, _ = _run_pinned(weights, alpha, (-1, 0, -1, 0), team_size)
    wrapped, reference = _team_b(actions), _team_b(direct)
    for name in HEADS:
        np.testing.assert_array_equal(wrapped[name][PINNED], reference[name][PINNED])
    _, observations, state = _setup(team_size)
    actor = make_recurrent_mappo_system(weights, spawn_frame="world")
    plain = make_opponent_system(actor)
    history = _history(weights, (-1, -1, -1, -1))
    memory = init_systems(
        actor,
        plain,
        observations,
        state,
        jax.random.key(7),
        variables_a=weights,
        variables_b=history,
    )
    unpinned, _, _ = apply_systems(
        actor,
        plain,
        memory,
        observations,
        state,
        jax.random.key(8),
        variables_a=weights,
        variables_b=history,
    )
    own = _team_b(unpinned)
    for name in HEADS:
        np.testing.assert_array_equal(wrapped[name][~PINNED], own[name][~PINNED])


SEEN: list[int] = []


def _seen(value: Array) -> None:
    del value
    SEEN.append(1)


def _counter_init(variables: Tree, inputs: SystemInput, keys: Array) -> Array:
    del variables, keys
    return jnp.zeros(inputs.valid.shape, jnp.float32)


def _counter_actions(memory: Array) -> ActorAction:
    count = memory.shape[0]
    move = jnp.broadcast_to((memory.astype(jnp.int32) % 9)[:, None], (count, 5))
    zero = jnp.zeros((count, 5), jnp.int32)
    return ActorAction(move, zero, zero)


def _counter_two(
    variables: Tree, memory: Array, inputs: SystemInput, keys: Array
) -> tuple[ActorAction, Array]:
    del variables, keys
    jax.debug.callback(_seen, memory)
    return _counter_actions(memory), memory + inputs.valid.astype(jnp.float32)


def _counter_three(
    variables: Tree, memory: Array, inputs: SystemInput, keys: Array
) -> tuple[ActorAction, Array, Array]:
    actions, following = _counter_two(variables, memory, inputs, keys)
    return actions, following, jnp.full((4, 5), -1, jnp.int32)


def _counter_output(
    variables: Tree, memory: Array, inputs: SystemInput, keys: Array
) -> SystemOutput:
    actions, following = _counter_two(variables, memory, inputs, keys)
    return SystemOutput(actions, following)


@pytest.mark.parametrize("apply", [_counter_two, _counter_three, _counter_output])
def test_every_return_form_routes_like_m8_and_idle_steps_skip_the_system(
    weights: Tree, apply: SystemApply
) -> None:
    counter = System("Counter", apply, init=_counter_init)
    actions, memory, direct, direct_memory = _run_pinned(
        weights, counter, (-1, 0, -1, 0)
    )
    pinned_memory = np.asarray(memory[1])
    np.testing.assert_array_equal(
        pinned_memory[PINNED], np.asarray(direct_memory)[PINNED]
    )
    np.testing.assert_array_equal(pinned_memory[~PINNED], np.zeros(2, np.float32))
    wrapped, reference = _team_b(actions), _team_b(direct)
    np.testing.assert_array_equal(wrapped["move"][PINNED], reference["move"][PINNED])
    SEEN.clear()
    _, idle_memory = _wrapper_step(weights, counter, (-1, -1, -1, -1))
    jax.effects_barrier()
    assert SEEN == []
    np.testing.assert_array_equal(np.asarray(idle_memory[1]), np.zeros(4, np.float32))


def _steps_apply(
    variables: Tree, carry: Array, actor: ActorInput, mask: ActionMask, key: Array
) -> tuple[ActorAction, Array]:
    del variables, actor, mask
    # Called with an argument, the callback runs once per computed actor row.
    jax.debug.callback(_seen, carry)
    move = (carry.astype(jnp.int32)[0] + jax.random.randint(key, (), 0, 9)) % 9
    zero = jnp.int32(0)
    following = jnp.concatenate(
        (carry[:1] + jnp.int32(1), jax.random.key_data(key).astype(jnp.int32))
    )
    return ActorAction(move, zero, zero), following


def _steps_policy() -> System:
    source = Policy("Steps", _steps_apply, initial_carry=np.zeros(3, np.int32))
    return shared_policy(cast(Policy, freeze_evaluation_method(source)))


@pytest.mark.parametrize(
    ("snapshot", "rows"),
    [
        ((-1,) * 8, 0),
        ((0, -1, -1, -1, -1, -1, -1, -1), 10),
        ((-1, 0, -1, -1, -1, 0, -1, -1), 10),
        ((0, 0, 0, -1, -1, -1, -1, -1), 20),
        ((0, 0, 0, 0, -1, -1, -1, -1), 20),
        ((0, 0, 0, 0, 0, -1, -1, -1), 40),
        ((0,) * 8, 40),
        ((0,), 5),
        ((0, -1), 5),
        ((0, 0), 10),
    ],
    ids=[
        "pins-0",
        "pins-1",
        "pins-2",
        "pins-3",
        "pins-4",
        "pins-5",
        "pins-8",
        "batch-1",
        "batch-2-one-pin",
        "batch-2-all-pins",
    ],
)
def test_a_policy_runs_on_its_games_only_when_they_fit_and_matches_m8(
    weights: Tree, snapshot: tuple[int, ...], rows: int
) -> None:
    steps = _steps_policy()
    SEEN.clear()
    actions, memory = _wrapper_step(weights, steps, snapshot)
    jax.effects_barrier()
    # B8 uses 2, 4 or 8 games times five actor rows; zero pins skip work.
    # B1 and B2 reuse the distinct capacities that exist at those sizes.
    assert len(SEEN) == rows
    direct, direct_memory = _direct_step(weights, steps, num_envs=len(snapshot))
    lanes = np.asarray(snapshot) == 0
    wrapped, reference = _team_b(actions), _team_b(direct)
    for name in HEADS:
        np.testing.assert_array_equal(wrapped[name][lanes], reference[name][lanes])
    pinned_memory = np.asarray(memory[1])
    np.testing.assert_array_equal(
        pinned_memory[lanes], np.asarray(direct_memory)[lanes]
    )
    np.testing.assert_array_equal(pinned_memory[~lanes], 0.0)


ENDS = (1, 1, 3, 3, 3, 3, 3, 3)
BEFORE = (0, -1, 0, -1, -1, -1, -1, -1)
AFTER = (-1, 0, 0, -1, -1, -1, -1, -1)


def _across_a_new_game(
    weights: Tree,
    team_b: System,
    first: Tree,
    second: Tree,
    ends: tuple[int, ...] = ENDS,
) -> tuple[Action, Tree]:
    env, _, state = _setup(3, 8)
    observations, state = env.reset(
        jax.random.key(41),
        state.config._replace(max_steps=jnp.asarray(ends, jnp.int32)),
    )
    actor = make_recurrent_mappo_system(weights, spawn_frame="world")
    memory = init_systems(
        actor,
        team_b,
        observations,
        state,
        jax.random.key(7),
        variables_a=weights,
        variables_b=first,
    )
    actions, memory, _ = apply_systems(
        actor,
        team_b,
        memory,
        observations,
        state,
        jax.random.key(8),
        variables_a=weights,
        variables_b=first,
    )
    observations, state, *_ = env.step(jax.random.key(9), state, actions)
    np.testing.assert_array_equal(state.done.done, np.asarray(ends) == 1)
    observations, state = env.reset_done(jax.random.key(10), state)
    actions, memory, _ = apply_systems(
        actor,
        team_b,
        memory,
        observations,
        state,
        jax.random.key(11),
        variables_a=weights,
        variables_b=second,
    )
    return actions, memory.team_b


@pytest.mark.parametrize("form", ["system", "policy"])
@pytest.mark.parametrize(
    ("before", "after"),
    [
        (BEFORE, AFTER),
        (BEFORE, (-1, 0, 0, 0, -1, -1, -1, -1)),
        ((0, 0, 0, -1, -1, -1, -1, -1), AFTER),
        (BEFORE, (-1, 0, 0, 0, 0, 0, -1, -1)),
        ((0, 0, 0, 0, 0, -1, -1, -1), AFTER),
    ],
    ids=[
        "quarter",
        "quarter-to-half",
        "half-to-quarter",
        "quarter-to-full",
        "full-to-quarter",
    ],
)
def test_a_new_game_resets_and_reassigns_the_pinned_lanes_like_m8(
    weights: Tree, form: str, before: tuple[int, ...], after: tuple[int, ...]
) -> None:
    pinned = (
        System("Counter", _counter_two, init=_counter_init)
        if form == "system"
        else _steps_policy()
    )
    _, variables, template = prepare_evaluation_system(pinned)
    wrapper = make_opponent_system(
        make_recurrent_mappo_system(weights, spawn_frame="world"), pinned=pinned
    )
    ends = tuple(1 if old != new else 3 for old, new in zip(before, after, strict=True))
    actions, memory = _across_a_new_game(
        weights,
        wrapper,
        (_history(weights, before), variables, template),
        (_history(weights, after), variables, template),
        ends,
    )
    direct, direct_memory = _across_a_new_game(weights, pinned, None, None, ends)
    # Newly pinned games restart; continuing pinned games retain memory.
    lanes = np.flatnonzero(np.asarray(after) == 0)
    wrapped, reference = _team_b(actions), _team_b(direct)
    for name in HEADS:
        np.testing.assert_array_equal(wrapped[name][lanes], reference[name][lanes])
    np.testing.assert_array_equal(
        np.asarray(memory[1])[lanes], np.asarray(direct_memory)[lanes]
    )
    if form == "system":
        np.testing.assert_array_equal(
            np.asarray(memory[1])[lanes],
            np.where(np.asarray(before)[lanes] == 0, 2.0, 1.0),
        )


def _template_apply(
    variables: Tree, carry: Array, actor: ActorInput, mask: ActionMask, key: Array
) -> tuple[ActorAction, Array]:
    del variables, actor, mask, key
    move = carry.astype(jnp.int32)[0] % 9
    zero = jnp.int32(0)
    return ActorAction(move, zero, zero), carry


def test_a_policy_template_changed_after_setup_does_not_reach_pinned_games(
    weights: Tree,
) -> None:
    template = np.array([3.0], np.float32)
    source = Policy("Template", _template_apply, initial_carry=template)
    frozen = cast(Policy, freeze_evaluation_method(source))
    pinned = shared_policy(frozen)
    template[0] = 7.0
    _, memory, _, _ = _run_pinned(weights, pinned, (-1, 0, -1, 0))
    pinned_memory = np.asarray(memory[1])
    np.testing.assert_array_equal(pinned_memory[PINNED], np.full((2, 5, 1), 3.0))


def _second_axis_init(variables: Tree, inputs: SystemInput, keys: Array) -> Array:
    del variables, keys
    return jnp.zeros((1, inputs.valid.shape[0]), jnp.float32)


def _second_axis_apply(
    variables: Tree, memory: Array, inputs: SystemInput, keys: Array
) -> tuple[ActorAction, Array]:
    del variables, keys
    following = memory + inputs.valid.astype(jnp.float32)[None, :]
    return _counter_actions(memory[0]), following


def _second_axis_reset(old: Array, fresh: Array, mask: Array) -> Array:
    return jnp.where(mask[None, :], fresh, old)


def test_a_pinned_system_with_its_own_reset_hook_and_layout_matches_m8(
    weights: Tree,
) -> None:
    layered = System(
        "Lane Second",
        _second_axis_apply,
        init=_second_axis_init,
        reset_memory=_second_axis_reset,
    )
    _, observations, state = _setup(3)
    actor = make_recurrent_mappo_system(weights, spawn_frame="world")
    assert make_opponent_system(actor, pinned=layered).reset_memory is not None
    assert (
        make_opponent_system(actor, pinned=shared_policy(policy("random"))).reset_memory
        is None
    )
    del observations, state
    actions, memory, direct, direct_memory = _run_pinned(
        weights, layered, (-1, 0, -1, 0)
    )
    pinned_memory = np.asarray(memory[1])
    np.testing.assert_array_equal(
        pinned_memory[:, PINNED], np.asarray(direct_memory)[:, PINNED]
    )
    np.testing.assert_array_equal(
        _team_b(actions)["move"][PINNED], _team_b(direct)["move"][PINNED]
    )


def test_host_execution_is_refused_as_the_wrappers_pinned_system(
    weights: Tree,
) -> None:
    actor = make_recurrent_mappo_system(weights, spawn_frame="world")
    host = System("Host", _counter_two, init=_counter_init, execution="host")
    with pytest.raises(ValueError, match="JAX execution"):
        make_opponent_system(actor, pinned=host)
    with pytest.raises(TypeError, match="pinned must be a System"):
        make_opponent_system(actor, pinned=policy("random"))  # pyright: ignore[reportArgumentType]
