"""Check the complete PQN-VDN learner lifecycle on real training collection.

Contracts checked here, all on CPU with four games, blocks of 4 rounds, a
memory window of 2 (so W = 6 initial rounds), one epoch of two minibatches, and
exploration from 0.8 to 0.05 over half the planned learning blocks. The initial
chunks of 4 and 2 rounds are accepted without any optimizer step, statistics
change, key use, version bump or snapshot; the rate stays exactly 1 before W
and becomes ``epsilon_at(0)`` at W. Each learning block takes exactly epochs *
minibatches optimizer steps, starts every game's unroll from the oldest kept
row's stored memory (changing only that memory changes the learned parameters,
and rerunning the same block repeats them exactly), publishes the network,
statistics and ``epsilon_at(k)`` once, and keeps the last H rows, whose
memories equal the rollout's own stored action-time memories; a final block of
one round (fewer than H) is learned too. When Team A's slot 0 is killed in
every game (set in the environment state, so the dead state shows from the next
collected row), it stays configured, has only the neutral action legal while
dead, respawns later in the same episode with no episode start, its stored
memory is never reset and keeps changing, both blocks are accepted, and a
window starting on kept dead rows keeps the dead slot in the loss (changing its
input there changes the loss). Counters follow the boundary table, and every
accepted boundary (0, T and W included) passes ``validate_pqn_learner``;
tampered counters, rates, keys, kept rows, variances and optimizer counts fail.
Used counts equal E * B * (H + m - 1) pairs, E * B * H kept-row pairs and E * B
sequences, and the exposure by stage, source and opponent equals an independent
NumPy recount of the windows' left rows. On three-decision episodes with the
curriculum, shaping and injected win and loss rewards, windows cross episode
endings, include small rosters and several stages, and the batch reward is the
active-slot mean plus shaping (these short games have no combat, so real
shaping is zero; a known shaping value added to collected transitions reaches
the batch reward through the compact rows and the window); changing an inactive
slot's input leaves the loss unchanged while changing a kept row's reward
changes it. Rejections at the collection, boundary (wrong version, invalid real
row, wrong chunk length, a learning attempt below capacity T, changed collected
history), row, action, update (a later minibatch, a statistics leaf, a NaN in
RAdam's second moment below its rectification point, or steps that did not all
run) and publication stages keep every leaf of the previous boundary with a
sticky reason, and the rejected result keeps its summary and attempted step
flags. The updater's own publication check accepts and refuses exactly the
cases the shared ``refresh_opponents`` does (an ordinary publication; rounds
above the total or not above the last refresh; a wrong version or one at the
int32 maximum; a sticky error; an out-of-range lane snapshot; a full bank with
a due capture). An empty block changes nothing. The collection never changes
current or frozen statistics or rates within a block, on a run whose games
really play both frozen and current opponents. A device round trip of a
boundary repeats the next block exactly. Setup rejects budgets without a
learning block before any network is built. The compiled updater holds under
half an opponent bank of temporaries, while a control that passes the bank
through two nested conditionals shows at least one bank. These checks prove
software contracts, not learning or GPU cost.
"""

# pyright: reportPrivateUsage=false, reportUnknownLambdaType=false
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import lru_cache, partial
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from tests.training_learner_helpers import synthetic_horizons

from marl_battlegrounds.baselines import pqn
from marl_battlegrounds.baselines.actions import categorical_action_mask, decode_actions
from marl_battlegrounds.core.types import Action
from marl_battlegrounds.training import (
    PreparedTrainingContent,
    TrainingCarry,
    TrainingCollection,
    TrainingRollout,
    TrainingTransition,
    make_training_schedule,
    prepare_training_content,
    scan_training_rollout,
)
from marl_battlegrounds.training import pqn_learner as learner
from marl_battlegrounds.training.opponents import OpponentHistory, refresh_opponents

type Tree = Any
type Scan = Callable[[TrainingCarry], tuple[TrainingCarry, TrainingRollout]]
type Update = Callable[
    [learner.PQNLearnerState, TrainingCarry, TrainingRollout],
    tuple[learner.PQNLearnerState, learner.PQNUpdateResult],
]
_CONFIG = pqn.PQNConfig(
    rollout_length=4,
    memory_window=2,
    epochs=1,
    num_minibatches=2,
    eps_start=0.8,
    eps_finish=0.05,
    eps_decay_fraction=0.5,
)
_SLOT_VALUES = jnp.arange(1.0, 6.0, dtype=jnp.float32)


def _equal(left: Tree, right: Tree) -> None:
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        if jnp.issubdtype(a.dtype, jax.dtypes.prng_key):
            a, b = jax.random.key_data(a), jax.random.key_data(b)
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def _nan_like(value: Array) -> Array:
    return jnp.full_like(value, jnp.nan)


def _bump_counts(leaf: Array) -> Array:
    return leaf + 1 if jnp.issubdtype(leaf.dtype, jnp.integer) else leaf


@lru_cache
def _scan(collection: TrainingCollection, length: int) -> Scan:
    return cast(
        Scan, jax.jit(partial(scan_training_rollout, collection, length=length))
    )


@lru_cache
def _update(planned: int) -> Update:
    return cast(
        Update,
        jax.jit(
            partial(
                learner.update_pqn_learner, pqn=_CONFIG, planned_learning_blocks=planned
            )
        ),
    )


class _Kept(NamedTuple):
    recent: learner.PQNRecent
    network: pqn.PQNInferenceVariables


@dataclass
class _Run:
    collection: TrainingCollection
    planned: int
    bank: int
    states: list[Any]
    results: list[learner.PQNUpdateResult]
    collected: dict[int, TrainingCarry]
    rollouts: list[TrainingRollout]
    unchanged: list[bool]


def _same(left: Tree, right: Tree) -> bool:
    return all(
        bool(jnp.array_equal(a, b))
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
    )


def _learn(
    collection: TrainingCollection,
    state: learner.PQNLearnerState,
    lengths: tuple[int, ...],
    *,
    rewards: bool = False,
    full: bool = True,
    keep_collected: frozenset[int] = frozenset(),
) -> _Run:
    # Each learner state holds a 369 MB opponent bank, so runs keep full
    # states only where a test needs them and otherwise the kept rows and the
    # network; collected carries are kept only at the named blocks.
    planned = pqn.pqn_planned_learning_blocks(
        int(state.carry.schedule.total_rounds), _CONFIG
    )

    def keep(value: learner.PQNLearnerState) -> Any:  # noqa: ANN401
        if not full:
            return _Kept(value.recent, value.carry.history.current_variables.network)
        previous = record.states[-1] if record.states else None
        bank = value.carry.history.historical_variables
        if previous is not None and _same(
            bank, previous.carry.history.historical_variables
        ):
            # Equal frozen banks share one copy; every value stays the same.
            history = value.carry.history._replace(
                historical_variables=previous.carry.history.historical_variables
            )
            return value._replace(carry=value.carry._replace(history=history))
        return value

    record = _Run(
        collection, planned, learner._bank_size(state.carry), [], [], {}, [], []
    )
    record.states.append(keep(state))
    for index, length in enumerate(lengths):
        carry, rollout = _scan(collection, length)(state.carry)
        # Collection must leave current and frozen actors untouched.
        record.unchanged.append(
            _same(
                carry.history.current_variables, state.carry.history.current_variables
            )
            and _same(
                carry.history.historical_variables,
                state.carry.history.historical_variables,
            )
        )
        if rewards:
            rollout = _with_ending_rewards(rollout)
        state, result = _update(planned)(state, carry, rollout)
        assert bool(result.accepted) and not bool(result.failed)
        record.states.append(keep(state))
        record.results.append(result)
        if index in keep_collected:
            record.collected[index] = carry
        record.rollouts.append(rollout)
    return record


def _with_ending_rewards(rollout: TrainingRollout) -> TrainingRollout:
    rows = rollout.transitions
    # Games 0 and 2 "win" and games 1 and 3 "lose"; each slot has its own
    # value so the team mean depends on which slots are active.
    sign = jnp.asarray([1.0, -1.0, 1.0, -1.0], jnp.float32)[None, :, None]
    ending = rows.ended[..., None] & rows.valid[..., None]
    rewards = jnp.where(ending, sign * _SLOT_VALUES, rows.task_rewards)
    return rollout._replace(transitions=rows._replace(task_rewards=rewards))


@pytest.fixture(scope="module")
def prepared() -> PreparedTrainingContent:
    return prepare_training_content()


@pytest.fixture(scope="module")
def plain(prepared: PreparedTrainingContent) -> _Run:
    collection, state = learner.init_pqn_learner(
        schedule=make_training_schedule(total_env_steps=60, num_envs=4),
        seed=19049131,
        prepared=prepared,
        metrics="none",
        pqn=_CONFIG,
    )
    # 15 rounds: initial chunks of 4 and 2, two blocks of 4, then one round.
    return _learn(collection, state, (4, 2, 4, 4, 4), keep_collected=frozenset({2}))


@pytest.fixture(scope="module")
def episodes(prepared: PreparedTrainingContent) -> _Run:
    collection, state = learner.init_pqn_learner(
        schedule=make_training_schedule(
            total_env_steps=172, num_envs=4, curriculum=True
        ),
        seed=19049132,
        prepared=prepared,
        shaping=True,
        metrics="none",
        pqn=_CONFIG,
    )
    state = cast(
        learner.PQNLearnerState,
        synthetic_horizons(collection, cast(Any, state), horizon=3),
    )
    # 43 rounds: initial chunks of 4 and 2, nine blocks of 4, then one round.
    return _learn(collection, state, (4, 2, *(4,) * 10), rewards=True, full=False)


def _counts(state: learner.PQNLearnerState) -> tuple[int, int, int, int]:
    return (
        int(state.carry.progress.rounds),
        int(state.completed_blocks),
        int(state.learning_blocks),
        int(state.completed_updates),
    )


def _epsilon(blocks: int, planned: int) -> float:
    return float(pqn.epsilon_at(jnp.int32(blocks), planned, _CONFIG))


def test_initial_chunks_then_learning_blocks_follow_the_boundary_table(
    plain: _Run,
) -> None:
    assert plain.planned == 3
    assert [_counts(state) for state in plain.states] == [
        (0, 0, 0, 0),
        (4, 1, 0, 0),
        (6, 2, 0, 0),
        (10, 3, 1, 2),
        (14, 4, 2, 4),
        (15, 5, 3, 6),
    ]
    first = plain.states[0]
    for state, result in zip(plain.states[1:3], plain.results[:2], strict=True):
        assert bool(result.accepted) and not bool(result.performed)
        assert not bool(result.snapshot.created)
        assert int(result.used.used_td_pairs) == 0
        assert not bool(jnp.any(result.metrics.performed))
        _equal(state.opt_state, first.opt_state)
        _equal(
            state.carry.history.current_variables.network,
            first.carry.history.current_variables.network,
        )
        _equal(state.shuffle_root, first.shuffle_root)
        assert int(state.carry.history.current_update) == 0
        assert int(state.carry.history.last_refresh_rounds) == 0
        assert int(state.carry.history.count) == 0
    assert float(plain.states[1].carry.history.current_variables.epsilon) == 1.0
    at_w = float(plain.states[2].carry.history.current_variables.epsilon)
    assert at_w == pytest.approx(_epsilon(0, 3), abs=1e-7)
    assert at_w == pytest.approx(0.8, abs=1e-7)
    for blocks, (state, result) in enumerate(
        zip(plain.states[3:], plain.results[2:], strict=True), start=1
    ):
        assert bool(result.accepted) and bool(result.performed)
        assert bool(jnp.all(result.metrics.performed))
        assert bool(jnp.all(result.metrics.finite))
        history = state.carry.history
        assert int(history.current_update) == blocks
        assert int(history.last_refresh_rounds) == int(state.carry.progress.rounds)
        assert float(history.current_variables.epsilon) == pytest.approx(
            _epsilon(blocks, 3), abs=1e-6
        )
        m = int(plain.rollouts[blocks + 1].real_steps)
        assert int(result.used.used_td_pairs) == 1 * 4 * (2 + m - 1)
        assert int(result.used.used_prefix_td_pairs) == 1 * 4 * 2
        assert int(result.used.used_sequences) == 1 * 4
        assert int(state.recent.size) == 2
    assert [int(r.real_steps) for r in plain.rollouts] == [4, 2, 4, 4, 1]
    assert bool(plain.results[2].snapshot.created)
    assert int(plain.states[3].carry.history.count) == 1
    rate = plain.states[5].carry.history.current_variables.epsilon
    assert float(rate) == float(np.float32(0.05))
    frozen = plain.states[5].carry.history.historical_variables.epsilon
    assert float(frozen[0]) == pytest.approx(_epsilon(1, 3), abs=1e-6)


def test_every_accepted_boundary_validates_and_tampering_fails(plain: _Run) -> None:
    for index, state in enumerate(plain.states):
        learner.validate_pqn_learner(
            plain.collection,
            state,
            pqn=_CONFIG,
            recheck_installed_content=index == len(plain.states) - 1,
        )
    state = plain.states[3]
    history = state.carry.history
    variables = history.current_variables

    def with_history(**changes: object) -> learner.PQNLearnerState:
        return state._replace(
            carry=state.carry._replace(history=history._replace(**changes))
        )

    def with_rows(**changes: object) -> learner.PQNLearnerState:
        return state._replace(
            recent=state.recent._replace(rows=state.recent.rows._replace(**changes))
        )

    variance = cast(dict[str, Any], variables.network.batch_stats)
    negative = {
        **variance,
        "BatchNorm_1": {
            **variance["BatchNorm_1"],
            "var": variance["BatchNorm_1"]["var"].at[0].set(-1.0),
        },
    }
    rows = state.recent.rows
    starts = state.recent.pre_memory.at[0].set(1.0)
    cases: list[tuple[str, learner.PQNLearnerState]] = [
        ("counts disagree", state._replace(completed_blocks=jnp.int32(4))),
        ("counts disagree", state._replace(completed_updates=jnp.int32(3))),
        (
            "exploration rate",
            with_history(
                current_variables=variables._replace(epsilon=jnp.float32(0.9))
            ),
        ),
        ("shuffle key", state._replace(shuffle_root=jax.random.key(3))),
        (
            "running variance",
            with_history(
                current_variables=variables._replace(
                    network=variables.network._replace(batch_stats=negative)
                )
            ),
        ),
        (
            "optimizer count",
            state._replace(opt_state=jax.tree.map(_bump_counts, state.opt_state)),
        ),
        (
            "game order",
            with_rows(decision_step=rows.decision_step.at[1].add(5)),
        ),
        (
            "kept memory",
            state._replace(
                recent=state.recent._replace(
                    pre_memory=starts,
                    rows=rows._replace(
                        episode_start=rows.episode_start.at[0].set(True)
                    ),
                )
            ),
        ),
        ("successor", with_rows(episode_id=rows.episode_id + 1)),
        (
            "rate 1",
            plain.states[1]._replace(
                carry=plain.states[1].carry._replace(
                    history=plain.states[1].carry.history._replace(
                        current_variables=plain.states[
                            1
                        ].carry.history.current_variables._replace(
                            epsilon=jnp.float32(0.8)
                        )
                    )
                )
            ),
        ),
    ]
    for message, tampered in cases:
        with pytest.raises(ValueError, match=message):
            learner.validate_pqn_learner(
                plain.collection, tampered, pqn=_CONFIG, recheck_installed_content=False
            )


def _window(run: _Run, block: int) -> tuple[learner.PQNRow, Array]:
    before, rollout = run.states[block], run.rollouts[block]
    rows, memory = learner._pqn_rows(rollout.transitions)
    joined, joined_memory, count = learner._learning_window(
        before.recent, rows, memory, rollout.real_steps
    )

    def real(value: Array, n: int = int(count)) -> Array:
        return value[:n]

    return cast(learner.PQNRow, jax.tree.map(real, joined)), joined_memory[: int(count)]


def test_used_counts_and_exposure_equal_an_independent_recount(
    episodes: _Run,
) -> None:
    bank = episodes.bank
    stages: set[int] = set()
    rows_seen: list[learner.PQNRow] = []
    for rollout in episodes.rollouts:
        compact, _ = learner._pqn_rows(rollout.transitions)
        real = int(rollout.real_steps)

        def prefix(value: Array, n: int = real) -> Array:
            return value[:n]

        rows_seen.append(cast(learner.PQNRow, jax.tree.map(prefix, compact)))
    history = cast(
        learner.PQNRow, jax.tree.map(lambda *v: np.concatenate(v), *rows_seen)
    )
    total = 0
    for block, (rollout, result) in enumerate(
        zip(episodes.rollouts, episodes.results, strict=True)
    ):
        real = int(rollout.real_steps)
        kept = min(2, total)
        total += real
        if block < 2:
            assert int(result.used.used_td_pairs) == 0
            continue
        window = slice(total - kept - real, total)
        left = slice(total - kept - real, total - 1)
        pairs = (kept + real - 1) * 4
        used = result.used
        assert int(used.used_td_pairs) == pairs
        assert int(used.used_prefix_td_pairs) == kept * 4
        assert int(used.used_sequences) == 4
        active = np.asarray(history.active)[left]
        assert int(used.used_agent_utilities) == int(active.sum())
        assert pairs <= int(used.used_agent_utilities) <= 5 * pairs
        for label, bins, size in (
            (history.episode_stage, used.exposure_by_stage, 17),
            (history.source_index, used.exposure_by_source, bank),
            (np.asarray(history.opponent_snapshot) + 1, used.exposure_by_opponent, 21),
        ):
            expected = np.bincount(np.asarray(label)[left].ravel(), minlength=size)
            np.testing.assert_array_equal(np.asarray(bins), expected)
        stages.update(int(s) for s in np.asarray(history.episode_stage)[window].ravel())
        _, memory = _window(episodes, block)
        np.testing.assert_array_equal(
            memory[0], np.asarray(episodes.states[block].recent.pre_memory[0])
        )
        if real >= 2:
            # The kept memories are the rollout's own stored action-time
            # memories, never recomputed with the new weights.
            stored = rollout.transitions.learning_outputs.pre_memory
            np.testing.assert_array_equal(
                np.asarray(episodes.states[block + 1].recent.pre_memory),
                np.asarray(stored[real - 2 : real]),
            )
    assert len(stages) >= 2


def test_windows_cross_endings_with_team_rewards_and_small_rosters(
    episodes: _Run,
) -> None:
    crossings = endings = small = 0
    for block in range(2, len(episodes.rollouts)):
        rows, memory = _window(episodes, block)
        assert bool(learner._sequence_valid(rows, jnp.int32(block - 2)))
        ended = np.asarray(rows.ended)
        start = np.asarray(rows.episode_start)
        crossings += int((ended[:-1] & start[1:]).sum())
        active = np.asarray(rows.active)
        small += int((active.sum(-1) < 5).sum())
        task = np.asarray(rows.task_reward)
        means = (active * np.asarray(_SLOT_VALUES)).sum(-1) / np.maximum(
            active.sum(-1), 1
        )
        np.testing.assert_allclose(np.abs(task[ended]), means[ended], rtol=1e-6)
        endings += int(ended.sum())
        batch = learner._expand_minibatch(rows, memory[0], _CONFIG)
        np.testing.assert_array_equal(
            np.asarray(batch.rewards), task + np.asarray(rows.shaping_reward)
        )
    assert crossings > 0 and endings > 0 and small > 0
    # These short CPU games have no combat, so potential shaping stays 0 here.
    # A known shaping value added to one block's transitions must reach the
    # batch reward through the compact rows and the window.
    before, rollout = episodes.states[2], episodes.rollouts[2]
    collected = rollout.transitions
    extra = jnp.where(collected.valid, 0.25, 0.0) * jnp.arange(
        1.0, 5.0, dtype=jnp.float32
    )
    moved = collected._replace(shaping_reward=collected.shaping_reward + extra)
    compact, memory = learner._pqn_rows(moved)
    joined, joined_memory, count = learner._learning_window(
        before.recent, compact, memory, rollout.real_steps
    )
    total, real = int(count), int(rollout.real_steps)

    def filled(value: Array) -> Array:
        return value[:total]

    window = cast(learner.PQNRow, jax.tree.map(filled, joined))
    batch = learner._expand_minibatch(window, joined_memory[0], _CONFIG)
    shaping = np.asarray(window.shaping_reward)
    np.testing.assert_array_equal(
        shaping[total - real :], np.asarray(moved.shaping_reward)[:real]
    )
    assert (shaping[total - real :] != 0).all()
    np.testing.assert_array_equal(
        np.asarray(batch.rewards), np.asarray(window.task_reward) + shaping
    )


def test_inactive_slots_are_ignored_and_kept_rows_are_learned(episodes: _Run) -> None:
    found = False
    for block in range(2, len(episodes.rollouts)):
        rows, memory = _window(episodes, block)
        batch = learner._expand_minibatch(rows, memory[0], _CONFIG)
        inactive = np.argwhere(~np.asarray(batch.active))
        if inactive.size == 0:
            continue
        network = episodes.states[block].network

        def loss(
            values: pqn.PQNBatch, net: pqn.PQNInferenceVariables = network
        ) -> float:
            return float(
                pqn._minibatch_loss(net.params, net.batch_stats, values, _CONFIG)[0]
            )

        base = loss(batch)
        t, game, slot = (int(v) for v in inactive[0])
        noisy = batch._replace(
            actor_features=batch.actor_features.at[t, game, slot].set(1000.0)
        )
        assert loss(noisy) == base
        # Row 0 is a kept row from the previous block and a left row of a pair.
        changed = batch._replace(rewards=batch.rewards.at[0, 0].add(3.0))
        assert loss(changed) != base
        found = True
        break
    assert found


def _rejected(
    prior: learner.PQNLearnerState,
    state: learner.PQNLearnerState,
    result: learner.PQNUpdateResult,
    reason: int,
) -> None:
    assert bool(state.failed) and int(state.failure_reason) == reason
    assert not bool(result.accepted) and not bool(result.performed)
    assert bool(result.failed) and int(result.failure_reason) == reason
    assert int(result.used.used_td_pairs) == 0
    _equal(
        state._replace(failed=prior.failed, failure_reason=prior.failure_reason), prior
    )


def _illegal_actions(rows: TrainingTransition) -> Action:
    categories = np.asarray(categorical_action_mask(rows.action_mask))
    real = np.asarray(rows.valid)[..., None, None] & np.asarray(rows.active)[..., None]
    step, lane, slot, category = (int(v) for v in np.argwhere(real & ~categories)[0])
    native = decode_actions(jnp.int32(category))
    return type(rows.actions)(
        *(
            head.at[step, lane, slot].set(value)
            for head, value in zip(rows.actions, native, strict=True)
        )
    )


def test_rejections_keep_the_previous_boundary_with_a_sticky_reason(
    plain: _Run,
) -> None:
    update = _update(plain.planned)
    prior, collected, rollout = plain.states[2], plain.collected[2], plain.rollouts[2]
    good = plain.results[2]
    rows = rollout.transitions
    order = learner.epoch_permutations(
        prior.shuffle_root, prior.learning_blocks, _CONFIG, 4
    )
    later = int(order[0, 2])
    network = prior.carry.history.current_variables.network
    stats = cast(dict[str, Any], network.batch_stats)
    broken_stats = {
        **stats,
        "BatchNorm_2": {
            **stats["BatchNorm_2"],
            "mean": _nan_like(stats["BatchNorm_2"]["mean"]),
        },
    }

    def with_network(net: pqn.PQNInferenceVariables) -> learner.PQNLearnerState:
        variables = prior.carry.history.current_variables._replace(network=net)
        return prior._replace(
            carry=prior.carry._replace(
                history=prior.carry.history._replace(current_variables=variables)
            )
        )

    radam = prior.opt_state
    moments = [
        index
        for index, leaf in enumerate(jax.tree.leaves(radam))
        if jnp.issubdtype(leaf.dtype, jnp.floating)
    ]
    leaves = jax.tree.leaves(radam)
    half = len(moments) // 2
    # RAdam keeps mu then nu; poison the first leaf of nu only.
    poisoned = [
        _nan_like(leaf) if index == moments[half] else leaf
        for index, leaf in enumerate(leaves)
    ]
    structure = cast(Any, jax.tree.structure(radam))
    nu_nan: Tree = jax.tree.unflatten(structure, poisoned)
    full = collected.history._replace(count=jnp.int32(20))
    cases: list[tuple[int, learner.PQNLearnerState, TrainingCarry, TrainingRollout]] = [
        (
            1,
            prior,
            collected._replace(
                tracking=replace(
                    collected.tracking, error_flags=collected.tracking.error_flags | 1
                )
            ),
            rollout,
        ),
        (
            2,
            prior,
            collected,
            rollout._replace(
                transitions=rows._replace(learner_update=rows.learner_update + 1)
            ),
        ),
        (
            2,
            prior,
            collected,
            rollout._replace(
                transitions=rows._replace(valid=rows.valid.at[0, 1].set(False))
            ),
        ),
        (
            2,
            prior,
            collected._replace(
                history=collected.history._replace(current_update=jnp.int32(1))
            ),
            rollout,
        ),
        (
            2,
            prior,
            collected._replace(
                history=collected.history._replace(last_refresh_rounds=jnp.int32(3))
            ),
            rollout,
        ),
        (
            3,
            prior,
            collected,
            rollout._replace(
                transitions=rows._replace(
                    task_rewards=rows.task_rewards.at[0, 0, 0].set(jnp.nan)
                )
            ),
        ),
        (
            4,
            prior,
            collected,
            rollout._replace(transitions=rows._replace(actions=_illegal_actions(rows))),
        ),
        (
            5,
            prior,
            collected,
            rollout._replace(
                transitions=rows._replace(
                    task_rewards=rows.task_rewards.at[0, later].set(3e38)
                )
            ),
        ),
        (
            5,
            with_network(network._replace(batch_stats=broken_stats)),
            collected,
            rollout,
        ),
        (5, prior._replace(opt_state=nu_nan), collected, rollout),
        (6, prior, collected._replace(history=full), rollout),
    ]
    for reason, start, carry, block in cases:
        state, result = update(start, carry, block)
        _rejected(start, state, result, reason)
        again, again_result = update(state, collected, rollout)
        assert int(again.failure_reason) == reason and bool(again_result.failed)
    # The rejected result keeps the block's summary and attempted steps.
    _, huge, block = cases[7][1:]
    assert cases[7][0] == 5
    state, result = update(prior, huge, block)
    loss = np.asarray(result.metrics.loss)
    assert bool(result.metrics.performed.all())
    assert bool(result.metrics.finite[0, 0]) and not bool(result.metrics.finite[0, 1])
    assert np.isfinite(loss[0, 0]) and not np.isfinite(loss[0, 1])
    _equal(
        result.summary._replace(task_reward_sum=good.summary.task_reward_sum),
        good.summary,
    )
    # A NaN second moment below RAdam's rectification point never reaches the
    # parameters or the loss; the whole-candidate check still rejects it.
    state, result = update(prior._replace(opt_state=nu_nan), collected, rollout)
    assert int(result.failure_reason) == 5
    assert bool(np.isfinite(np.asarray(result.metrics.loss)).all())
    assert not bool(result.metrics.finite.any())


def test_early_or_wrong_sized_blocks_are_rejected(plain: _Run) -> None:
    update = _update(plain.planned)
    # After the first chunk of 4, the next chunk must be H = 2 rounds.
    prior = plain.states[1]
    carry, rollout = _scan(plain.collection, 4)(prior.carry)
    state, result = update(prior, carry, rollout)
    _rejected(prior, state, result, 2)
    # The final block has one round; a capacity-2 program must not learn it.
    prior = plain.states[4]
    carry, rollout = _scan(plain.collection, 2)(prior.carry)
    assert int(rollout.real_steps) == 1
    state, result = update(prior, carry, rollout)
    _rejected(prior, state, result, 2)


def test_a_block_whose_steps_did_not_all_run_is_rejected(
    plain: _Run, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = learner.update_pqn

    def skipped(
        state: pqn.PQNTrainState,
        batch: pqn.PQNBatch,
        **settings: Any,  # noqa: ANN401
    ) -> tuple[pqn.PQNTrainState, pqn.PQNMetrics]:
        candidate, metrics = original(state, batch, **settings)
        return candidate, metrics._replace(performed=jnp.bool_(False))

    monkeypatch.setattr(learner, "update_pqn", skipped)
    update = cast(
        Update,
        jax.jit(
            partial(
                learner.update_pqn_learner,
                pqn=_CONFIG,
                planned_learning_blocks=plain.planned,
            )
        ),
    )
    prior = plain.states[2]
    state, result = update(prior, plain.collected[2], plain.rollouts[2])
    _rejected(prior, state, result, 5)


def test_empty_block_changes_nothing(plain: _Run) -> None:
    final = plain.states[-1]
    carry, rollout = _scan(plain.collection, 4)(final.carry)
    assert int(rollout.real_steps) == 0
    state, result = _update(plain.planned)(final, carry, rollout)
    _equal(state, final)
    assert not bool(result.accepted) and not bool(result.performed)
    assert not bool(result.failed)
    for leaf in jax.tree.leaves(result):
        if jnp.issubdtype(leaf.dtype, jnp.inexact):
            assert bool(jnp.all(jnp.isfinite(leaf)))


def test_the_oldest_kept_memory_seeds_the_unroll(plain: _Run) -> None:
    prior, carry, rollout = plain.states[2], plain.collected[2], plain.rollouts[2]
    assert not bool(np.asarray(prior.recent.rows.episode_start)[0].any())
    again, result = _update(plain.planned)(prior, carry, rollout)
    assert bool(result.accepted)
    learned = plain.states[3].carry.history.current_variables.network
    _equal(again.carry.history.current_variables.network, learned)
    # Slot 0 is always configured, so its stored memory enters every game.
    memory = prior.recent.pre_memory.at[0, :, 0].add(0.5)
    moved, result = _update(plain.planned)(
        prior._replace(recent=prior.recent._replace(pre_memory=memory)), carry, rollout
    )
    assert bool(result.accepted)
    assert not _same(
        moved.carry.history.current_variables.network.params, learned.params
    )


def test_death_and_respawn_keep_memory_and_learn_the_dead_slot(plain: _Run) -> None:
    state = plain.states[2]
    core = state.carry.state.core_state
    # Slot 0 of Team A dies in every game; the next collected row shows it.
    dead = jnp.zeros(core.alive_mask.shape, bool).at[:, 0].set(True)
    core = core._replace(
        alive_mask=core.alive_mask & ~dead,
        current_health=jnp.where(dead, 0.0, core.current_health),
    )
    state = state._replace(
        carry=state.carry._replace(state=state.carry.state._replace(core_state=core))
    )
    rollouts: list[TrainingRollout] = []
    before: list[learner.PQNLearnerState] = []
    for _ in range(2):
        carry, rollout = _scan(plain.collection, 4)(state.carry)
        before.append(state)
        state, result = _update(plain.planned)(state, carry, rollout)
        assert bool(result.accepted) and not bool(result.failed)
        rollouts.append(rollout)
    rows = [r.transitions for r in rollouts]
    alive = np.concatenate([np.asarray(r.alive[..., 0]) for r in rows])
    legal = np.concatenate(
        [np.asarray(categorical_action_mask(r.action_mask))[..., 0, :] for r in rows]
    )
    memory = np.concatenate(
        [np.asarray(r.learning_outputs.pre_memory[..., 0, :]) for r in rows]
    )
    starts = np.concatenate([np.asarray(r.episode_start) for r in rows])
    active = np.concatenate([np.asarray(r.active[..., 0]) for r in rows])
    assert active.all() and not starts.any()
    # Rows 1-3 are dead with only the neutral action legal; the slot respawns
    # later in the same episode.
    assert not alive[:4].any() and (legal[1:4].sum(-1) == 1).all()
    assert alive[4:].any()
    # No memory reset at death or respawn: the stored memory keeps moving.
    assert (np.abs(memory).sum(-1) > 0).all()
    assert (np.abs(np.diff(memory, axis=0)).sum(-1) > 0).all()
    # The second block's window starts on two kept dead rows; the dead slot
    # stays in the loss (a changed dead-row input changes it).
    network = before[1].carry.history.current_variables.network
    window, window_memory = _window(
        _Run(plain.collection, plain.planned, plain.bank, before, [], {}, rollouts, []),
        1,
    )
    assert not np.asarray(window.alive)[:2, :, 0].any()
    batch = learner._expand_minibatch(window, window_memory[0], _CONFIG)
    assert bool(batch.active[:2, :, 0].all())

    def loss(values: pqn.PQNBatch) -> float:
        return float(
            pqn._minibatch_loss(network.params, network.batch_stats, values, _CONFIG)[0]
        )

    changed = batch._replace(actor_features=batch.actor_features.at[0, 0, 0].add(1.0))
    assert loss(changed) != loss(batch)


def test_the_publication_check_agrees_with_the_shared_history(plain: _Run) -> None:
    history = plain.states[3].carry.history
    schedule = plain.states[3].carry.schedule
    actor = history.current_variables
    rounds = history.last_refresh_rounds + 4
    index = history.current_update + 1
    top = jnp.int32(np.iinfo(np.int32).max)
    last_lane = history.lane_snapshot.at[0].set(history.count)
    cases = {
        "ordinary": (history, rounds, index),
        "rounds_above_total": (history, schedule.total_rounds + 1, index),
        "rounds_not_above_last": (history, history.last_refresh_rounds, index),
        "wrong_version": (history, rounds, index + 1),
        "version_at_maximum": (
            history._replace(current_update=top),
            rounds,
            jnp.int32(np.iinfo(np.int32).min),
        ),
        "sticky_error": (history._replace(error=jnp.bool_(True)), rounds, index),
        "bad_lane": (history._replace(lane_snapshot=last_lane), rounds, index),
        "full_bank_due": (
            history._replace(
                count=jnp.int32(20),
                threshold_to_snapshot=jnp.full_like(history.threshold_to_snapshot, -1),
            ),
            schedule.total_rounds,
            index,
        ),
    }
    outcomes: dict[str, bool] = {}
    for name, (case, at, version) in cases.items():
        due = (case.threshold_to_snapshot == -1) & (
            at >= schedule.history_threshold_rounds
        )
        mine = bool(learner._publishable(case, at, version, schedule.total_rounds, due))
        shared, _ = refresh_opponents(
            case, actor, completed_rounds=at, update_index=version, schedule=schedule
        )
        assert mine == (not bool(shared.error)), name
        outcomes[name] = mine
    assert outcomes["ordinary"] and not any(
        value for name, value in outcomes.items() if name != "ordinary"
    )


def test_collection_never_changes_statistics_or_rates_within_a_block(
    plain: _Run, episodes: _Run
) -> None:
    assert plain.unchanged == [True] * 5
    assert episodes.unchanged == [True] * 12
    # The episodes run really mixes frozen and current opponents in its blocks.
    frozen = sum(
        int((np.asarray(rollout.transitions.opponent_snapshot) >= 0).sum())
        for rollout in episodes.rollouts
    )
    current = sum(
        int((np.asarray(rollout.transitions.opponent_snapshot) == -1).sum())
        for rollout in episodes.rollouts[2:]
    )
    assert frozen > 0 and current > 0


def test_a_device_round_trip_repeats_the_next_block_exactly(plain: _Run) -> None:
    prior = plain.states[3]
    restored = cast(learner.PQNLearnerState, jax.device_put(jax.device_get(prior)))
    _equal(
        learner.epoch_permutations(
            restored.shuffle_root, restored.learning_blocks, _CONFIG, 4
        ),
        learner.epoch_permutations(
            prior.shuffle_root, prior.learning_blocks, _CONFIG, 4
        ),
    )
    carry, rollout = _scan(plain.collection, 4)(restored.carry)
    _equal(rollout, plain.rollouts[3])
    state, result = _update(plain.planned)(restored, carry, rollout)
    _equal(state, plain.states[4])
    _equal(result, plain.results[3])


def test_setup_rejects_budgets_without_a_learning_block_before_building(
    prepared: PreparedTrainingContent, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("The network was built before the budget check")

    monkeypatch.setattr(learner, "initialize_pqn", refuse)
    with pytest.raises(ValueError, match="at least memory_window"):
        learner.init_pqn_learner(
            schedule=make_training_schedule(total_env_steps=24, num_envs=4),
            prepared=prepared,
            pqn=_CONFIG,
        )
    with pytest.raises(ValueError, match="divide"):
        learner.init_pqn_learner(
            schedule=make_training_schedule(total_env_steps=64, num_envs=2),
            prepared=prepared,
            pqn=pqn.PQNConfig(rollout_length=4, memory_window=2, num_minibatches=4),
        )


def test_the_updater_holds_under_half_a_bank_of_temporaries(plain: _Run) -> None:
    state = plain.states[2]
    bank = sum(
        leaf.nbytes
        for leaf in jax.tree.leaves(state.carry.history.historical_variables)
    )
    shapes = jax.eval_shape(
        partial(scan_training_rollout, plain.collection, length=4), state.carry
    )
    compiled = (
        jax.jit(
            partial(
                learner.update_pqn_learner,
                pqn=_CONFIG,
                planned_learning_blocks=plain.planned,
            )
        )
        .lower(state, *shapes)
        .compile()
    )
    memory = compiled.memory_analysis()
    assert memory is not None
    assert memory.temp_size_in_bytes < bank // 2

    def control(history: OpponentHistory, flags: Array) -> Tree:
        def nested(scale: float) -> Tree:
            def changed(_: None) -> Tree:
                def scaled(leaf: Array) -> Array:
                    return leaf * scale

                return jax.tree.map(scaled, history.historical_variables)

            def kept(_: None) -> Tree:
                return history.historical_variables

            def inner(_: None) -> Tree:
                return cast(Tree, jax.lax.cond(flags[1], changed, kept, None))

            return cast(Tree, jax.lax.cond(flags[0], inner, kept, None))

        def pick(a: Array, b: Array) -> Array:
            return jnp.where(flags[2], a, b)

        return jax.tree.map(pick, nested(2.0), nested(3.0))

    checked = (
        jax.jit(control)
        .lower(state.carry.history, jnp.zeros(3, jnp.bool_))
        .compile()
        .memory_analysis()
    )
    assert checked is not None
    assert checked.temp_size_in_bytes >= bank
