"""Check score shaping against complete returns and public game trajectories.

Controlled score records check terminal cancellation, authored starts, cutoffs,
padding, dynamic settings and validation. Synthetic authored test games exercise
real wins, losses, draws, death, respawn, team sizes 1-5 and AutoReset. The public
actor loop checks that optional learner feedback changes no actor input, action,
task reward or simulator transition. These CPU cases do not claim learning gains
or GPU speed; the composed collector owns proof that disabled work is omitted.
"""

from collections.abc import Callable
from itertools import pairwise
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from tests.evaluation_fixtures import evaluation_env_config

from marl_battlegrounds.autoreset import AutoReset
from marl_battlegrounds.core import env as core
from marl_battlegrounds.core.types import TASK_MODE_TDM, Action, EnvConfig
from marl_battlegrounds.environment import (
    Environment,
    EnvironmentState,
    EpisodeInfo,
    make,
)
from marl_battlegrounds.episode_tracking import StepResult
from marl_battlegrounds.evaluation.policy_execution import (
    apply_systems,
    init_systems,
    policy,
    shared_policy,
)
from marl_battlegrounds.training.shaping import team_potential_shaping, validate_shaping


def _equal(actual: object, expected: object) -> None:
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        if jnp.issubdtype(left.dtype, jax.dtypes.prng_key):
            left, right = jax.random.key_data(left), jax.random.key_data(right)
        np.testing.assert_array_equal(left, right)


@pytest.fixture(scope="module")
def config() -> EnvConfig:
    return evaluation_env_config()


def _info(
    config: EnvConfig,
    scores: tuple[tuple[int, int], ...],
    *,
    completed: tuple[bool, ...] | None = None,
    steps: tuple[int, ...] | None = None,
) -> EpisodeInfo:
    batch = len(scores)
    return EpisodeInfo(
        episode_id=jnp.arange(1, batch + 1, dtype=jnp.int32),
        completed=jnp.asarray(completed if completed is not None else (False,) * batch),
        outcome=jnp.zeros(batch, jnp.int32),
        config=config,
        priority=None,
        full=None,
        replay=None,
        decision_step=jnp.asarray(
            steps if steps is not None else (0,) * batch, jnp.int32
        ),
        episode_length=jnp.ones(batch, jnp.int32),
        team_scores=jnp.asarray(scores, jnp.int32),
        lifecycle_error=jnp.zeros(batch, jnp.bool_),
    )


@pytest.mark.parametrize("discount", [0.0, 0.9, 1.0])
@pytest.mark.parametrize("coefficient", [0.0, 0.01, 0.3])
@pytest.mark.parametrize("start", [(0, 0), (4, 1), (1, 3)])
def test_full_discounted_return_cancels_intermediate_scores(
    config: EnvConfig, discount: float, coefficient: float, start: tuple[int, int]
) -> None:
    scores = (start, (start[0] + 1, start[1]), (start[0] + 1, start[1] + 2))
    total = np.zeros(2, np.float64)
    for index, after in enumerate((*scores[1:], scores[-1])):
        info = _info(config, (after,), completed=(index == 2,))
        feedback = team_potential_shaping(
            jnp.asarray((scores[index],), jnp.int32),
            info,
            discount=discount,
            coefficient=coefficient,
        )
        total += discount**index * np.asarray(feedback[0], np.float64)
    expected = coefficient * np.asarray((start[1] - start[0], start[0] - start[1]))
    np.testing.assert_allclose(total, expected, atol=2e-7, rtol=2e-6)


def test_nonterminal_score_change_no_change_and_cutoff_keep_potential(
    config: EnvConfig,
) -> None:
    sequence = ((0, 0), (2, 0), (2, 0), (2, 1), (2, 1))
    feedback: list[Array] = []
    for index, (before, after) in enumerate(pairwise(sequence)):
        info = _info(config, (after,), completed=(index == 3,))
        feedback.append(
            team_potential_shaping(
                jnp.asarray((before,), jnp.int32), info, discount=0.5, coefficient=0.1
            )
        )
    result = np.asarray(feedback)[:, 0]
    np.testing.assert_allclose(result[:, 0], (0.1, -0.1, -0.15, -0.1), atol=2e-8)
    np.testing.assert_array_equal(result[:, 1], -result[:, 0])
    # The first cutoff retains a nonzero successor. Cancelling there would lose it.
    assert result[0, 0] > 0
    first_chunk = np.sum(result[:2] * np.asarray((1, 0.5))[:, None], axis=0)
    second_chunk = np.sum(result[2:] * np.asarray((1, 0.5))[:, None], axis=0)
    np.testing.assert_allclose(first_chunk + 0.5**2 * second_chunk, 0, atol=2e-8)


def test_padding_uses_existing_real_transition_authority(config: EnvConfig) -> None:
    info = _info(config, ((10, 3), (0, 20)), completed=(False, True), steps=(-1, -1))
    before = jnp.asarray(((2, 1), (7, 0)), jnp.int32)
    result = team_potential_shaping(before, info, discount=0.9)
    np.testing.assert_array_equal(result, np.zeros((2, 2), np.float32))


def test_jit_reuses_program_for_changed_scalar_values_and_scores(
    config: EnvConfig,
) -> None:
    traces: list[int] = []

    @jax.jit
    def run(before: Array, info: EpisodeInfo, discount: Array, scale: Array) -> Array:
        traces.append(1)
        return team_potential_shaping(
            before, info, discount=discount, coefficient=scale
        )

    before = jnp.asarray(((1, 2), (3, 1)), jnp.int32)
    info = _info(config, ((2, 2), (3, 2)), completed=(True, False))
    for discount, scale in ((0.9, 0.01), (1.0, 0.1), (0.0, 0.0)):
        actual = cast(
            Array, run(before, info, jnp.asarray(discount), jnp.asarray(scale))
        )
        expected = team_potential_shaping(
            before, info, discount=discount, coefficient=scale
        )
        np.testing.assert_allclose(actual, expected, atol=2e-8)
        assert actual.shape == (2, 2) and actual.dtype == jnp.float32
        before = before + 1
        info = info._replace(team_scores=info.team_scores + 2)
    assert traces == [1]


@pytest.mark.parametrize("discount", [0, 0.9, 1, np.float32(0.5), jnp.asarray(0.5)])
def test_host_validation_accepts_real_boundary_settings(
    discount: float | Array,
) -> None:
    assert validate_shaping(discount=discount) is None
    assert validate_shaping(discount=discount, coefficient=0) is None


@pytest.mark.parametrize("field", ["discount", "coefficient"])
@pytest.mark.parametrize("value", [True, np.bool_(False), 1j, "0.5", None])
def test_host_validation_rejects_nonreal_settings(field: str, value: object) -> None:
    settings = {"discount": 0.9, "coefficient": 0.01, field: value}
    with pytest.raises(TypeError, match="real scalar"):
        validate_shaping(**cast(dict[str, float], settings))


@pytest.mark.parametrize("field", ["discount", "coefficient"])
@pytest.mark.parametrize("value", [float("inf"), float("nan"), -0.01, [0.5]])
def test_host_validation_rejects_invalid_values(field: str, value: object) -> None:
    settings = {"discount": 0.9, "coefficient": 0.01, field: value}
    with pytest.raises(ValueError):
        validate_shaping(**cast(dict[str, float], settings))


def test_host_validation_rejects_excess_discount_and_tracing() -> None:
    def validate(value: Array) -> None:
        validate_shaping(discount=value)

    with pytest.raises(ValueError, match="discount"):
        validate_shaping(discount=1.01)
    with pytest.raises(TypeError, match="host"):
        jax.jit(validate)(jnp.asarray(0.9))


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("before_scores", jnp.zeros((2,), jnp.int32), ValueError),
        ("before_scores", jnp.zeros((0, 2), jnp.int32), ValueError),
        ("before_scores", jnp.zeros((1, 2), jnp.float32), TypeError),
        ("team_scores", jnp.zeros((2, 2), jnp.int32), ValueError),
        ("team_scores", jnp.zeros((1, 2), jnp.float32), TypeError),
        ("completed", jnp.zeros((), jnp.bool_), ValueError),
        ("completed", jnp.zeros((1,), jnp.int32), TypeError),
        ("decision_step", jnp.zeros((2,), jnp.int32), ValueError),
        ("decision_step", jnp.zeros((1,), jnp.float32), TypeError),
        ("discount", jnp.ones((1,), jnp.float32), ValueError),
        ("discount", jnp.asarray(1, jnp.int32), TypeError),
        ("coefficient", jnp.asarray(True), TypeError),
        ("coefficient", True, TypeError),
    ],
)
def test_numerical_boundary_checks_static_shapes_and_dtypes(
    config: EnvConfig, field: str, value: Array | bool, error: type[Exception]
) -> None:
    before = jnp.zeros((1, 2), jnp.int32)
    info = _info(config, ((0, 0),))
    settings: dict[str, float | Array] = {"discount": 0.9}
    if field == "before_scores":
        before = cast(Array, value)
    elif field in ("discount", "coefficient"):
        settings[field] = value
    else:
        info = info._replace(**{field: value})
    with pytest.raises(error, match=field):
        team_potential_shaping(before, info, **settings)


def _stack(*leaves: Array) -> Array:
    return jnp.stack(leaves)


def _row[T](tree: T, index: int) -> T:
    def take(value: Array) -> Array:
        return value[index]

    return jax.tree.map(take, tree)


def _setup(*, scores: Array, horizon: int = 4) -> tuple[Environment, EnvironmentState]:
    configs = tuple(
        evaluation_env_config(
            team_sizes=(size, size),
            task_mode=TASK_MODE_TDM,
            team_deathmatch_score_threshold=20,
            max_steps=horizon,
        )._replace(
            spawn_shield_duration_steps=0,
            team_respawn_wave_period_step_count=jnp.asarray((2, 2), jnp.int32),
        )
        for size in range(1, 6)
    )
    config = jax.tree.map(_stack, *configs)
    env = make("tdm", num_envs=5, metrics="none")
    _, state = env.reset(jax.random.key(17), config)
    authored = state.core_state._replace(
        team_deathmatch_scores=scores,
        agent_positions=state.core_state.agent_positions.at[:, 0]
        .set(jnp.asarray((4.0, 4.0), jnp.float32))
        .at[:, 5]
        .set(jnp.asarray((6.5, 4.0), jnp.float32)),
        current_health=state.core_state.current_health.at[:, 0]
        .set(1.0)
        .at[:, 5]
        .set(1.0),
        team_respawn_wave_countdowns=jnp.ones((5, 2), jnp.int32),
    )
    initialized = jax.tree.map(
        _stack,
        *(
            core.initialize_scenario_state(_row(authored, lane), configs[lane])
            for lane in range(5)
        ),
    )
    _, state = env.reset(jax.random.key(18), config, initial=initialized[:3])
    return env, state


def _idle() -> Action:
    zero = jnp.zeros((5, 10), jnp.int32)
    return Action(zero, zero, zero)


@pytest.fixture(scope="module")
def game() -> tuple[Environment, EnvironmentState]:
    return _setup(
        scores=jnp.asarray(((19, 2), (2, 19), (0, 0), (3, 1), (1, 3)), jnp.int32)
    )


@pytest.fixture(scope="module")
def step() -> Callable[[Environment, Array, EnvironmentState, Action], StepResult]:
    return cast(
        Callable[[Environment, Array, EnvironmentState, Action], StepResult],
        jax.jit(Environment.step),
    )


def _attack() -> Action:
    return _idle()._replace(
        select_target=jnp.zeros((5, 10), jnp.int32)
        .at[:, 0]
        .set(6)
        .at[1, 0]
        .set(0)
        .at[1, 5]
        .set(6)
    )


def test_public_episode_wins_losses_draws_death_respawn_and_terminal_padding(
    game: tuple[Environment, EnvironmentState],
    step: Callable[[Environment, Array, EnvironmentState, Action], StepResult],
) -> None:
    env, start = game
    state = start
    total = np.zeros((5, 2), np.float64)
    for tick in range(5):
        before = state
        result = step(
            env, jax.random.key(tick), state, _attack() if tick == 0 else _idle()
        )
        state, info = result[1], result[4]
        feedback = team_potential_shaping(
            before.core_state.team_deathmatch_scores, info, discount=0.9
        )
        total += 0.9**tick * np.asarray(feedback, np.float64)
        if tick == 0:
            np.testing.assert_array_equal(
                info.completed, (True, True, False, False, False)
            )
            np.testing.assert_array_equal(info.outcome[:2], (1, 2))
            np.testing.assert_array_equal(
                info.team_scores - before.core_state.team_deathmatch_scores,
                ((1, 0), (0, 1), (1, 0), (1, 0), (1, 0)),
            )
            assert not np.any(state.core_state.alive_mask[2:, 5])
        if tick == 1:
            assert np.all(state.core_state.alive_mask[2:, 5])
            np.testing.assert_array_equal(feedback[:2], 0)
        if tick == 3:
            np.testing.assert_array_equal(
                info.completed, (False, False, True, True, True)
            )
            np.testing.assert_array_equal(info.outcome[2:], 3)
        if tick == 4:
            np.testing.assert_array_equal(feedback, 0)
            _equal(state, before)
            assert np.all(info.decision_step == -1)
            assert not np.any(info.completed)
    differences = np.asarray(start.core_state.team_deathmatch_scores, np.float64)
    offset = 0.01 * (differences[:, 1] - differences[:, 0])
    np.testing.assert_allclose(total, np.stack((offset, -offset), axis=-1), atol=2e-8)
    np.testing.assert_array_equal(state.cumulative_transition_count, (1, 1, 4, 4, 4))


def test_same_team_signal_for_all_sizes_and_inactive_padding(
    step: Callable[[Environment, Array, EnvironmentState, Action], StepResult],
) -> None:
    env, before = _setup(scores=jnp.tile(jnp.asarray((3, 1), jnp.int32), (5, 1)))
    result = step(env, jax.random.key(1), before, _idle())
    feedback = team_potential_shaping(
        before.core_state.team_deathmatch_scores, result[4], discount=0.9
    )
    np.testing.assert_allclose(feedback, np.tile((-0.002, 0.002), (5, 1)), atol=2e-9)
    np.testing.assert_array_equal(
        np.sum(result[4].active_mask, axis=1), (2, 4, 6, 8, 10)
    )


def test_autoreset_matches_explicit_reset_using_producing_scores(
    game: tuple[Environment, EnvironmentState],
    step: Callable[[Environment, Array, EnvironmentState, Action], StepResult],
) -> None:
    env, state = game
    key = jax.random.key(41)
    manual = step(env, key, state, _attack())
    observations, reset_state = env.reset_done(
        jax.random.fold_in(key, 0x4155544F), manual[1]
    )
    wrapper = AutoReset(env)
    actual = cast(StepResult, jax.jit(AutoReset.step)(wrapper, key, state, _attack()))
    _equal(actual[:2], (observations, reset_state))
    _equal(actual[2:4], manual[2:4])
    _equal(actual[4]._replace(final=None), manual[4])
    before_scores = state.core_state.team_deathmatch_scores
    expected = team_potential_shaping(before_scores, manual[4], discount=0.9)
    np.testing.assert_array_equal(
        team_potential_shaping(before_scores, actual[4], discount=0.9), expected
    )
    np.testing.assert_array_equal(actual[1].core_state.team_deathmatch_scores[:2], 0)
    assert np.all(actual[4].team_scores[:2] != 0)
    np.testing.assert_allclose(expected[:2, 0], (-0.17, 0.17), atol=2e-8)


def test_public_actor_calls_and_task_trajectory_ignore_shaping(
    game: tuple[Environment, EnvironmentState],
    step: Callable[[Environment, Array, EnvironmentState, Action], StepResult],
) -> None:
    env, state = game
    actor = shared_policy(policy("random"))
    observations = env.get_observations(state)
    memory = init_systems(actor, actor, observations, state, jax.random.key(51))
    branches = [(state, observations, memory), (state, observations, memory)]
    for tick in range(3):
        outputs: list[object] = []
        for enabled, (before, inputs, carried) in enumerate(branches):
            actions, updated, _ = apply_systems(
                actor, actor, carried, inputs, before, jax.random.key(60 + tick)
            )
            result = step(env, jax.random.key(70 + tick), before, actions)
            if enabled:
                feedback = team_potential_shaping(
                    before.core_state.team_deathmatch_scores, result[4], discount=0.9
                )
                assert feedback.shape == (5, 2)
            branches[enabled] = (result[1], result[0], updated)
            outputs.append((actions, updated, result))
        _equal(outputs[0], outputs[1])
