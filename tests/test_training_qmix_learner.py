"""Check the complete QMIX learner lifecycle on real training collection.

Contracts checked here, all on CPU with small replay settings (buffer 12 rows
per game, minimum 6, sequences of 4, batches of 4, 2 epochs, hard copies every
2 steps, epsilon decay over 40 transitions) and two games. A block before
replay readiness is a warmup: accepted, stored, with no optimizer step, key
use, target change, version bump or snapshot; only the stored epsilon moves to
the clock. Each ready block runs exactly ``epochs`` optimizer steps, draws each
sample with ``fold_in(sampling_root, count before the step)``, copies targets
at the donor's pre-step counts, publishes the actor once with the clock's
epsilon, and reports sampled counts equal to the per-step metric sums with
exposure marginals that each sum to the used TD pairs. A padded final block
and a short nonfinal block are accepted and validated. Frozen history keeps
its captured epsilon while the current rate moves. An empty block changes
nothing. Rejections at the collection, boundary, row, action, update,
publication and later-epoch sample-order stages keep the previous boundary
leaf for leaf with a sticky reason. The readiness-aware block-count rule
accepts exactly the reachable counter states. Setup rejects overflowing count
products before allocation. Split and whole collection agree. Every accepted
boundary passes ``validate_qmix_learner``; tampered counters and epsilon fail.
The compiled update's temporary bytes grow by less than 0.75 per replay byte
(measured between 500 and 1,000 rows; about 0.46 now, about 0.99 if one
branch level kept a replay copy): the rows are inserted once and the replay is
chosen once. These checks prove software contracts, not learning or GPU cost.
Warm starts load saved actors, match greedy inference and keep fresh learner state.
"""

# pyright: reportPrivateUsage=false, reportUnknownLambdaType=false
import itertools
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from marl_battlegrounds.baselines import qmix
from marl_battlegrounds.baselines.actions import categorical_action_mask, decode_actions
from marl_battlegrounds.training import (
    PreparedTrainingContent,
    TrainingCarry,
    TrainingCollection,
    TrainingRollout,
    make_training_schedule,
    prepare_training_content,
    scan_training_rollout,
)
from marl_battlegrounds.training import qmix_learner as learner

type Tree = Any
type Update = Callable[
    [learner.QMIXLearnerState, TrainingCarry, TrainingRollout],
    tuple[learner.QMIXLearnerState, learner.QMIXUpdateResult],
]
_CONFIG = qmix.QMIXConfig(
    rollout_length=4,
    buffer_size=12,
    min_buffer_size=6,
    sample_sequence_length=4,
    sample_batch_size=4,
    epochs=2,
    update_period=2,
    eps_decay=40,
)


def _equal(left: Tree, right: Tree) -> None:
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        if jnp.issubdtype(a.dtype, jax.dtypes.prng_key):
            a, b = jax.random.key_data(a), jax.random.key_data(b)
        np.testing.assert_array_equal(a, b)


def _nan_like(value: Array) -> Array:
    return jnp.full_like(value, jnp.nan)


def _join(left: Array, right: Array) -> Array:
    return jnp.concatenate((left, right))


@lru_cache
def _scan(
    collection: TrainingCollection, length: int
) -> Callable[[TrainingCarry], tuple[TrainingCarry, TrainingRollout]]:
    def scan(values: TrainingCarry) -> tuple[TrainingCarry, TrainingRollout]:
        return scan_training_rollout(collection, values, length=length)

    return cast(
        Callable[[TrainingCarry], tuple[TrainingCarry, TrainingRollout]], jax.jit(scan)
    )


@lru_cache
def _update(config: qmix.QMIXConfig) -> Update:
    return cast(Update, jax.jit(partial(learner.update_qmix_learner, qmix=config)))


@dataclass
class _Run:
    collection: TrainingCollection
    states: list[learner.QMIXLearnerState]
    results: list[learner.QMIXUpdateResult]
    collected: list[TrainingCarry]
    rollouts: list[TrainingRollout]


@pytest.fixture(scope="module")
def prepared() -> PreparedTrainingContent:
    return prepare_training_content()


@pytest.fixture(scope="module")
def run(prepared: PreparedTrainingContent) -> _Run:
    collection, state = learner.init_qmix_learner(
        schedule=make_training_schedule(total_env_steps=28, num_envs=2),
        seed=19048301,
        prepared=prepared,
        metrics="none",
        qmix=_CONFIG,
    )
    record = _Run(collection, [state], [], [], [])
    # 14 rounds: a warmup block, two learning blocks and a padded final block.
    for _ in range(4):
        carry, rollout = _scan(collection, 4)(state.carry)
        state, result = _update(_CONFIG)(state, carry, rollout)
        record.states.append(state)
        record.results.append(result)
        record.collected.append(carry)
        record.rollouts.append(rollout)
    return record


def _counts(state: learner.QMIXLearnerState) -> tuple[int, int, int, int]:
    return (
        int(state.carry.progress.rounds),
        int(state.completed_blocks),
        int(state.learning_blocks),
        int(state.completed_updates),
    )


def _eps(rounds: int) -> float:
    return float(qmix.epsilon_at(jnp.int32(rounds), 2, 0.05, 40))


def test_warmup_then_learning_blocks_follow_readiness(run: _Run) -> None:
    assert [_counts(state) for state in run.states] == [
        (0, 0, 0, 0),
        (4, 1, 0, 0),
        (8, 2, 1, 2),
        (12, 3, 2, 4),
        (14, 4, 3, 6),
    ]
    first, warm = run.results[0], run.states[1]
    assert bool(first.accepted) and not bool(first.performed)
    assert not bool(first.failed) and not bool(first.snapshot.created)
    assert (
        int(first.sampled.used_td_pairs) == 0
        and int(first.summary.real_transitions) == 8
    )
    for name in ("mixer_params", "target_q_params", "target_mixer_params", "opt_state"):
        _equal(getattr(warm, name), getattr(run.states[0], name))
    _equal(
        warm.carry.history.current_variables.params,
        run.states[0].carry.history.current_variables.params,
    )
    assert int(warm.carry.history.current_update) == 0
    assert int(warm.carry.history.last_refresh_rounds) == 0
    for state in run.states:
        rounds = int(state.carry.progress.rounds)
        assert float(state.carry.history.current_variables.epsilon) == pytest.approx(
            _eps(rounds), abs=2e-6
        )
        assert int(state.replay.current_index) == rounds % 12
        assert bool(state.replay.is_full) == (rounds >= 12)
    for result in run.results[1:]:
        assert bool(result.accepted) and bool(result.performed)
        assert bool(jnp.all(result.metrics.finite))
        sampled = result.sampled
        assert (
            int(sampled.sampled_sequences)
            == int(jnp.sum(result.metrics.sampled_sequences))
            == 8
        )
        assert (
            int(sampled.used_td_pairs)
            == int(jnp.sum(result.metrics.used_td_pairs))
            == 24
        )
        assert int(sampled.used_agent_utilities) == int(
            jnp.sum(result.metrics.used_agent_utilities)
        )
        for marginal in (
            sampled.exposure_by_stage,
            sampled.exposure_by_source,
            sampled.exposure_by_opponent,
        ):
            assert int(jnp.sum(marginal)) == 24
    assert bool(run.results[1].snapshot.created)
    assert int(run.states[2].carry.history.last_refresh_rounds) == 8
    assert int(run.states[4].carry.history.current_update) == 3
    assert not bool(run.rollouts[3].transitions.valid[2:].any())


def test_every_accepted_boundary_validates(run: _Run) -> None:
    for index, state in enumerate(run.states):
        learner.validate_qmix_learner(
            run.collection,
            state,
            qmix=_CONFIG,
            recheck_installed_content=index == len(run.states) - 1,
        )
    state = run.states[3]
    with pytest.raises(ValueError, match="counts disagree"):
        learner.validate_qmix_learner(
            run.collection,
            state._replace(completed_blocks=jnp.int32(2)),
            qmix=_CONFIG,
            recheck_installed_content=False,
        )
    variables = state.carry.history.current_variables
    tampered = state._replace(
        carry=state.carry._replace(
            history=state.carry.history._replace(
                current_variables=variables._replace(epsilon=jnp.float32(0.9))
            )
        )
    )
    with pytest.raises(ValueError, match="exploration rate"):
        learner.validate_qmix_learner(
            run.collection, tampered, qmix=_CONFIG, recheck_installed_content=False
        )
    with pytest.raises(ValueError, match="replay cursor"):
        learner.validate_qmix_learner(
            run.collection,
            state._replace(replay=replace(state.replay, current_index=jnp.int32(3))),
            qmix=_CONFIG,
            recheck_installed_content=False,
        )


def _largest_gap(left: Tree, right: Tree) -> float:
    return max(
        float(jnp.max(jnp.abs(a - b)))
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
    )


def test_sampling_keys_and_target_copies_follow_the_count(run: _Run) -> None:
    prior, after, result = run.states[1], run.states[2], run.results[1]
    bank = learner._bank_size(prior.carry)
    train = learner._train_state(prior)
    snapshots = [train]
    losses: list[float] = []
    counts: list[learner.QMIXSampledCounts] = []
    for step in range(2):
        key = jax.random.fold_in(
            prior.sampling_root, int(prior.completed_updates) + step
        )
        sample = learner._sample_rows(after.replay, key, _CONFIG, 2)
        train, metrics = qmix.update_qmix(
            train, learner._expand_sample(sample, _CONFIG), config=_CONFIG
        )
        snapshots.append(train)
        losses.append(float(metrics.loss))
        counts.append(
            learner._step_counts(
                sample,
                metrics,
                bank,
                prior.carry.progress.opponent_steps.shape[0],
                prior.carry.history.capture_capacity,
            )
        )
    # Losses depend on the exact samples and their order; integers are exact.
    np.testing.assert_allclose(result.metrics.loss, losses, rtol=1e-4)
    total = jax.tree.map(lambda *values: sum(values), *counts)
    _equal(total, result.sampled)
    assert int(total.exposure_by_opponent[0]) == 24
    assert int(train.optimizer_steps) == int(after.completed_updates) == 2
    # Compiled and eager Adam steps round differently by a few 1e-6; one step
    # moves weights by about the 3e-5 learning rate.
    step_size = _largest_gap(snapshots[2].online_q, snapshots[1].online_q)
    assert step_size > 1e-5
    online = after.carry.history.current_variables.params
    assert _largest_gap(online, train.online_q) < 0.3 * step_size
    moments = (train.opt_state[0][0].mu, after.opt_state[0][0].mu)
    scale = max(float(jnp.max(jnp.abs(leaf))) for leaf in jax.tree.leaves(moments[0]))
    assert _largest_gap(*moments) < 1e-3 * scale
    assert int(after.opt_state[0][0].count) == 2
    # The copy happens at pre-step count 0; the step at count 1 leaves targets.
    near = _largest_gap(after.target_q_params, snapshots[1].online_q)
    far = _largest_gap(after.target_q_params, snapshots[2].online_q)
    assert near < 0.3 * far
    assert _largest_gap(after.target_mixer_params, snapshots[1].online_mixer) < 0.3 * (
        _largest_gap(after.target_mixer_params, snapshots[2].online_mixer)
    )


def test_frozen_history_keeps_its_captured_epsilon(run: _Run) -> None:
    for state in run.states[2:]:
        history = state.carry.history
        assert int(history.count) >= 1
        assert int(history.captured_rounds[0]) == 8
        assert float(history.historical_variables.epsilon[0]) == pytest.approx(
            _eps(8), abs=2e-6
        )
    assert _eps(14) < _eps(8)


def test_empty_block_changes_nothing(run: _Run) -> None:
    final = run.states[-1]
    carry, rollout = _scan(run.collection, 4)(final.carry)
    assert int(rollout.real_steps) == 0
    state, result = _update(_CONFIG)(final, carry, rollout)
    _equal(state, final)
    assert not bool(result.accepted) and not bool(result.performed)
    assert not bool(result.failed)


def _rejected(
    prior: learner.QMIXLearnerState,
    state: learner.QMIXLearnerState,
    result: learner.QMIXUpdateResult,
    reason: int,
) -> None:
    assert bool(state.failed) and int(state.failure_reason) == reason
    assert not bool(result.accepted) and not bool(result.performed)
    assert bool(result.failed) and int(result.failure_reason) == reason
    _equal(
        state._replace(failed=prior.failed, failure_reason=prior.failure_reason), prior
    )


def test_rejections_keep_the_previous_boundary(run: _Run) -> None:
    prior, collected, rollout = run.states[2], run.collected[2], run.rollouts[2]
    update = _update(_CONFIG)
    rows = rollout.transitions
    broken = collected._replace(
        tracking=replace(
            collected.tracking, error_flags=collected.tracking.error_flags | 1
        )
    )
    cases: list[
        tuple[int, learner.QMIXLearnerState, TrainingCarry, TrainingRollout]
    ] = [
        (1, prior, broken, rollout),
        (
            2,
            prior,
            collected,
            rollout._replace(
                transitions=rows._replace(learner_update=rows.learner_update + 1)
            ),
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
            5,
            prior._replace(mixer_params=jax.tree.map(_nan_like, prior.mixer_params)),
            collected,
            rollout,
        ),
        (
            6,
            prior,
            collected._replace(
                schedule=collected.schedule._replace(
                    total_rounds=collected.progress.rounds - 1
                )
            ),
            rollout,
        ),
    ]
    # Submit a real row's illegal category, decoded back into native heads.
    categories = np.asarray(categorical_action_mask(rows.action_mask))
    real = np.asarray(rows.valid)[..., None, None] & np.asarray(rows.active)[..., None]
    step, lane, slot, category = (int(v) for v in np.argwhere(real & ~categories)[0])
    native = decode_actions(jnp.int32(category))
    actions = type(rows.actions)(
        *(
            head.at[step, lane, slot].set(value)
            for head, value in zip(rows.actions, native, strict=True)
        )
    )
    cases.append(
        (
            4,
            prior,
            collected,
            rollout._replace(transitions=rows._replace(actions=actions)),
        )
    )
    for reason, start, carry, block in cases:
        state, result = update(start, carry, block)
        _rejected(start, state, result, reason)
        again, again_result = update(state, collected, rollout)
        assert int(again.failure_reason) == reason and bool(again_result.failed)


def test_a_broken_second_epoch_sample_rejects_the_whole_block(run: _Run) -> None:
    prior, collected, rollout = run.states[2], run.collected[2], run.rollouts[2]
    experience = prior.replay.experience
    lanes, rows = experience.opponent_update.shape
    tags = jnp.arange(lanes * rows, dtype=jnp.int32).reshape(lanes, rows)
    tagged = replace(prior.replay, experience=experience._replace(opponent_update=tags))
    inserted = learner._insert_rollout(tagged, rollout, _CONFIG)
    seen: list[set[int]] = []
    for step in range(2):
        key = jax.random.fold_in(
            prior.sampling_root, int(prior.completed_updates) + step
        )
        sample = learner._sample_rows(inserted, key, _CONFIG, 2)
        seen.append({int(v) for v in np.asarray(sample.opponent_update).ravel()})
    only_second = sorted(tag for tag in seen[1] - seen[0] if tag % rows < 8)
    assert only_second
    lane, position = divmod(only_second[0], rows)
    corrupted = replace(
        prior.replay,
        experience=experience._replace(
            decision_step=experience.decision_step.at[lane, position].add(1000)
        ),
    )
    start = prior._replace(replay=corrupted)
    state, result = _update(_CONFIG)(start, collected, rollout)
    _rejected(start, state, result, learner.LEARNER_ERROR_SEQUENCE)


def _reachable(top: int, minimum: int, most: int) -> set[tuple[int, int, int]]:
    states: set[tuple[int, int, int]] = {(0, 0, 0)}
    for count in range(1, most + 1):
        for sizes in itertools.product(range(1, top + 1), repeat=count):
            rows = learning = 0
            for size in sizes:
                rows += size
                learning += int(rows >= minimum)
            states.add((rows, count, learning))
    return states


def test_block_count_rule_accepts_exactly_the_reachable_states() -> None:
    assert not learner._block_counts_possible(32, 4, 4, rollout_length=8, minimum=32)
    assert learner._block_counts_possible(32, 4, 1, rollout_length=8, minimum=32)
    for top in range(1, 4):
        for minimum in range(1, 7):
            reachable = _reachable(top, minimum, 5)
            for rounds in range(0, 5 * top + 1):
                for blocks in range(0, 6):
                    for learning in range(0, blocks + 1):
                        expected = (rounds, blocks, learning) in reachable
                        actual = learner._block_counts_possible(
                            rounds,
                            blocks,
                            learning,
                            rollout_length=top,
                            minimum=minimum,
                        )
                        assert actual == expected, (
                            top,
                            minimum,
                            rounds,
                            blocks,
                            learning,
                        )


def test_short_nonfinal_block_is_accepted_and_validated(
    prepared: PreparedTrainingContent,
) -> None:
    config = qmix.QMIXConfig(
        rollout_length=8,
        buffer_size=16,
        min_buffer_size=6,
        sample_sequence_length=4,
        sample_batch_size=4,
        epochs=2,
        eps_decay=40,
    )
    collection, state = learner.init_qmix_learner(
        schedule=make_training_schedule(total_env_steps=40, num_envs=2),
        seed=19048302,
        prepared=prepared,
        metrics="none",
        qmix=config,
    )
    for length, expected in ((4, (4, 1, 0, 0)), (8, (12, 2, 1, 2))):
        carry, rollout = _scan(collection, length)(state.carry)
        state, result = _update(config)(state, carry, rollout)
        assert bool(result.accepted) and not bool(result.failed)
        assert _counts(state) == expected
        learner.validate_qmix_learner(
            collection, state, qmix=config, recheck_installed_content=False
        )


def test_split_and_whole_collection_agree(run: _Run) -> None:
    start = run.states[0].carry
    whole, whole_rollout = _scan(run.collection, 4)(start)
    half, first = _scan(run.collection, 2)(start)
    split, second = _scan(run.collection, 2)(half)
    _equal(split, whole)
    joined = jax.tree.map(_join, first.transitions, second.transitions)
    _equal(joined, whole_rollout.transitions)


def test_setup_rejects_overflowing_counts_before_allocation(
    prepared: PreparedTrainingContent,
) -> None:
    with pytest.raises(ValueError, match="int32"):
        learner.init_qmix_learner(
            schedule=make_training_schedule(total_env_steps=2 * (2**30), num_envs=2),
            prepared=prepared,
            qmix=qmix.QMIXConfig(epochs=4),
        )
    huge = qmix.QMIXConfig(rollout_length=2**28, buffer_size=2**28)
    with pytest.raises(ValueError, match="overflow"):
        learner.init_qmix_learner(
            schedule=make_training_schedule(total_env_steps=28, num_envs=2),
            prepared=prepared,
            qmix=huge,
        )


def test_the_update_keeps_no_replay_copy_per_branch(
    prepared: PreparedTrainingContent,
) -> None:
    sizes: list[tuple[int, int]] = []
    for rows in (500, 1000):
        config = replace(_CONFIG, buffer_size=rows)
        collection, state = learner.init_qmix_learner(
            schedule=make_training_schedule(total_env_steps=16, num_envs=2),
            seed=19048303,
            prepared=prepared,
            metrics="none",
            qmix=config,
        )
        shapes = jax.eval_shape(
            partial(scan_training_rollout, collection, length=4), state.carry
        )
        compiled = (
            jax.jit(partial(learner.update_qmix_learner, qmix=config))
            .lower(state, *shapes)
            .compile()
        )
        memory = compiled.memory_analysis()
        assert memory is not None
        replay = sum(leaf.nbytes for leaf in jax.tree.leaves(state.replay))
        sizes.append((replay, memory.temp_size_in_bytes))
    # Temporary bytes added per replay byte, whatever else (such as opponent
    # history) the temporaries hold: about 0.46 now, about 0.99 if one branch
    # level kept a replay copy, and about 1.99 in the old nested layout.
    growth = (sizes[1][1] - sizes[0][1]) / (sizes[1][0] - sizes[0][0])
    assert growth < 0.75


def test_warm_start_copies_online_and_target_but_keeps_fresh_mixer(
    prepared: PreparedTrainingContent,
    tmp_path: Path,
) -> None:
    schedule = make_training_schedule(total_env_steps=28, num_envs=2)
    _, fresh = learner.init_qmix_learner(
        schedule=schedule, qmix=_CONFIG, seed=23, prepared=prepared, metrics="none"
    )
    imported = jax.tree.map(
        partial(jnp.add, jnp.float32(0.125)),
        fresh.carry.history.current_variables.params,
    )
    from marl_battlegrounds.evaluation.policy_execution import (
        apply_systems,
        init_systems,
    )
    from marl_battlegrounds.training import checkpoints

    artifact = checkpoints.export_system(
        imported,
        tmp_path / "actor",
        method="qmix",
        input_scale=_CONFIG.input_scale,
        spawn_frame=_CONFIG.spawn_frame,
        metadata={
            "run_id": "Warm Start Test",
            "seed": 23,
            "env_steps": 0,
            "checkpoint_id": "a" * 64,
            "optimizer_steps": 0,
        },
    )
    loaded, _ = checkpoints.load_initial_actor(
        artifact,
        method="qmix",
        input_scale=_CONFIG.input_scale,
        spawn_frame=_CONFIG.spawn_frame,
    )
    collection, warm = learner.init_qmix_learner(
        schedule=schedule,
        qmix=_CONFIG,
        seed=23,
        prepared=prepared,
        metrics="none",
        initial_actor=loaded.variables.params,
    )
    _equal(warm.carry.history.current_variables.params, imported)
    _equal(warm.target_q_params, imported)
    _equal(warm.mixer_params, fresh.mixer_params)
    _equal(warm.target_mixer_params, fresh.mixer_params)
    _equal(warm.opt_state, fresh.opt_state)
    _equal(warm.replay, fresh.replay)
    assert float(warm.carry.history.current_variables.epsilon) == 1.0
    assert int(warm.completed_updates) == int(warm.carry.history.count) == 0
    variables = warm.carry.history.current_variables
    # Compare greedy inference separately from the learner's fresh exploration.
    variables = variables._replace(epsilon=jnp.float32(0))
    observations, state = warm.carry.observations, warm.carry.state
    memory = init_systems(loaded, loaded, observations, state, jax.random.key(71))
    expected = apply_systems(
        loaded, loaded, memory, observations, state, jax.random.key(72)
    )
    actual = apply_systems(
        collection.actor,
        loaded,
        memory,
        observations,
        state,
        jax.random.key(72),
        variables_a=variables,
    )
    _equal(actual, expected)


def test_named_opponent_keeps_identity_through_qmix_replay_and_learning(
    prepared: PreparedTrainingContent,
) -> None:
    from tests.test_training_collection import (
        _actor,  # pyright: ignore[reportPrivateUsage]
    )

    collection, state = learner.init_qmix_learner(
        schedule=make_training_schedule(total_env_steps=16, num_envs=2),
        prepared=prepared,
        metrics="none",
        qmix=_CONFIG,
        keep_past=0,
        history_capture_capacity=0,
        opponent_population={"fixed": _actor()},
        opponent_selection={"fixed": 1.0},
    )
    results: list[learner.QMIXUpdateResult] = []
    for _ in range(2):
        carry, rollout = _scan(collection, 4)(state.carry)
        np.testing.assert_array_equal(rollout.transitions.opponent_snapshot, -3)
        state, result = _update(_CONFIG)(state, carry, rollout)
        assert bool(result.accepted) and not bool(result.failed)
        results.append(result)
    result = results[-1]
    assert bool(result.performed)
    counts = np.asarray(result.sampled.exposure_by_opponent)
    assert counts.shape[-1] == 3
    assert int(counts[..., :2].sum()) == 0
    assert int(counts[..., 2].sum()) > 0
    learner.validate_qmix_learner(collection, state, qmix=_CONFIG)


def test_reward_reset_refills_replay_before_new_targets(run: _Run) -> None:
    from marl_battlegrounds.training._continuation_schedules import (
        LearnerContinuation,
        RewardReset,
    )

    parent = run.states[2]
    rounds, blocks, learning, updates = _counts(parent)
    context = LearnerContinuation(
        "qmix",
        rounds,
        blocks,
        learning,
        loss_settings=(("gamma", 0.0),),
        reward_reset=RewardReset(rounds, blocks, learning, rounds),
    )
    state = parent._replace(
        replay=learner._init_replay(run.collection, parent.carry, _CONFIG)
    )
    _equal(state.opt_state, parent.opt_state)
    _equal(state.carry, parent.carry)
    _equal(state.target_q_params, parent.target_q_params)
    update = cast(
        Update,
        jax.jit(
            partial(learner.update_qmix_learner, qmix=_CONFIG, continuation=context)
        ),
    )
    for index in range(2):
        carry, rollout = _scan(run.collection, 4)(state.carry)
        rows = rollout.transitions
        rollout = rollout._replace(
            transitions=rows._replace(
                task_rewards=jnp.where(
                    rows.valid[..., None], jnp.float32(40), jnp.float32(0)
                )
                * jnp.ones_like(rows.task_rewards),
            )
        )
        previous = state
        state, result = update(state, carry, rollout)
        assert bool(result.accepted) and not bool(result.failed)
        assert bool(result.performed) == (index == 1)
        if index == 0:
            assert int(state.completed_updates) == updates
            assert int(state.carry.history.last_refresh_rounds) == rounds
            _equal(state.opt_state, previous.opt_state)
            _equal(state.target_q_params, previous.target_q_params)
            _equal(
                state.carry.history.current_variables.params,
                previous.carry.history.current_variables.params,
            )
        else:
            assert int(state.completed_updates) == updates + _CONFIG.epochs
            np.testing.assert_allclose(result.metrics.mean_target, 40, rtol=1e-6)
    assert int(state.replay.current_index) == 6
    assert not bool(state.replay.is_full)
    valid = np.asarray(state.replay.experience.valid)
    np.testing.assert_array_equal(
        np.asarray(state.replay.experience.task_reward)[valid], 40
    )
