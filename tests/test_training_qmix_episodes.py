"""Check QMIX learning on samples that cross episode endings and stages.

One CPU case with two games and small replay settings (blocks of 4 rounds,
12 rows per game, minimum 6, sequences of 4, batches of 4, 2 epochs), the
test helper's three-decision episodes, the size curriculum and shaping.
Every learning block is accepted while its samples cross episode endings
into new episodes (the sample-order check accepts real joins). Win and loss
task rewards injected on ending rows reach the sampled batch as the mean over
the row's active slots, and each batch reward is task plus shaping. Samples
include small rosters and more than one curriculum stage. The learner's
per-block exposure counts by stage, source row and opponent row equal a
separate NumPy count of the left row of every used TD pair in samples redrawn
with the learner's keys. The shortened episodes come from an explicit test
bank, so the verified-content check of validate_qmix_learner is not run here.
Software contracts only; no learning or speed is claimed.
"""

# pyright: reportPrivateUsage=false
from collections.abc import Callable
from functools import partial
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from tests.test_training_collection import _synthetic

from marl_battlegrounds.baselines import qmix
from marl_battlegrounds.training import (
    TrainingCarry,
    TrainingCollection,
    TrainingRollout,
    make_training_schedule,
    prepare_training_content,
    scan_training_rollout,
)
from marl_battlegrounds.training import qmix_learner as learner

type Update = Callable[
    [learner.QMIXLearnerState, TrainingCarry, TrainingRollout],
    tuple[learner.QMIXLearnerState, learner.QMIXUpdateResult],
]
type Block = tuple[
    learner.QMIXLearnerState, learner.QMIXLearnerState, learner.QMIXUpdateResult
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
_SLOT_VALUES = jnp.arange(1.0, 6.0, dtype=jnp.float32)


def _with_ending_rewards(rollout: TrainingRollout) -> TrainingRollout:
    rows = rollout.transitions
    # Lane 0 "wins" and lane 1 "loses"; each slot gets a different value so
    # the team mean depends on which slots are active.
    sign = jnp.asarray([1.0, -1.0], jnp.float32)[None, :, None]
    ending = rows.ended[..., None] & rows.valid[..., None]
    rewards = jnp.where(ending, sign * _SLOT_VALUES, rows.task_rewards)
    return rollout._replace(transitions=rows._replace(task_rewards=rewards))


def _learn(
    collection: TrainingCollection,
    state: learner.QMIXLearnerState,
    blocks: int,
    *,
    rewards: bool,
) -> list[Block]:
    scan = cast(
        Callable[[TrainingCarry], tuple[TrainingCarry, TrainingRollout]],
        jax.jit(partial(scan_training_rollout, collection, length=4)),
    )
    update = cast(Update, jax.jit(partial(learner.update_qmix_learner, qmix=_CONFIG)))
    record: list[Block] = []
    for _ in range(blocks):
        carry, rollout = scan(state.carry)
        if rewards:
            rollout = _with_ending_rewards(rollout)
        before = state
        state, result = update(state, carry, rollout)
        assert bool(result.accepted) and not bool(result.failed)
        record.append((before, state, result))
    return record


def _samples(block: Block) -> list[learner.QMIXReplayRow]:
    before, after, _ = block
    return [
        learner._sample_rows(
            after.replay, jax.random.fold_in(after.sampling_root, count), _CONFIG, 2
        )
        for count in range(int(before.completed_updates), int(after.completed_updates))
    ]


def _check_exposure(block: Block, bank: int) -> tuple[np.ndarray[Any, Any], ...]:
    counts = [np.zeros(17, np.int64), np.zeros(bank, np.int64), np.zeros(21, np.int64)]
    for sample in _samples(block):
        valid = np.asarray(sample.valid)
        pair = valid[:, :-1] & valid[:, 1:]
        labels = (
            np.asarray(sample.episode_stage),
            np.asarray(sample.source_index),
            np.asarray(sample.opponent_snapshot) + 1,
        )
        for bins, label in zip(counts, labels, strict=True):
            np.add.at(bins, label[:, :-1][pair], 1)
    sampled = block[2].sampled
    for bins, learned in zip(
        counts,
        (
            sampled.exposure_by_stage,
            sampled.exposure_by_source,
            sampled.exposure_by_opponent,
        ),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(learned), bins)
    assert int(counts[0].sum()) == int(sampled.used_td_pairs)
    return tuple(counts)


def test_samples_across_endings_stages_and_small_rosters_learn() -> None:
    prepared = prepare_training_content()
    collection, state = learner.init_qmix_learner(
        schedule=make_training_schedule(
            total_env_steps=80, num_envs=2, curriculum=True
        ),
        seed=19048601,
        prepared=prepared,
        shaping=True,
        metrics="none",
        qmix=_CONFIG,
    )
    collection, carry = _synthetic((collection, state.carry), horizon=3)
    blocks = _learn(collection, state._replace(carry=carry), 10, rewards=True)
    learning = [block for block in blocks if bool(block[2].performed)]
    assert len(learning) == 9
    bank = learner._bank_size(state.carry)
    crossings = endings = small = 0
    stages: set[int] = set()
    for block in learning:
        by_stage = _check_exposure(block, bank)[0]
        stages.update(int(stage) for stage in np.flatnonzero(by_stage))
        for sample in _samples(block):
            assert bool(learner._sequence_valid(sample))
            valid = np.asarray(sample.valid)
            ended = np.asarray(sample.ended) & valid
            start = np.asarray(sample.episode_start)
            crossings += int((ended[:, :-1] & start[:, 1:] & valid[:, 1:]).sum())
            active = np.asarray(sample.active)
            small += int((valid & (active.sum(-1) < 5)).sum())
            # A stored ending's team reward is the injected values' mean over
            # the row's active slots; its sign says which lane it came from.
            task = np.asarray(sample.task_reward)
            means = (active * np.asarray(_SLOT_VALUES)).sum(-1) / np.maximum(
                active.sum(-1), 1
            )
            np.testing.assert_allclose(np.abs(task[ended]), means[ended], rtol=1e-6)
            assert np.all(np.abs(task[ended]) > 0)
            endings += int(ended.sum())
            batch = learner._expand_sample(sample, _CONFIG)
            np.testing.assert_array_equal(
                np.asarray(batch.rewards),
                task + np.asarray(sample.shaping_reward),
            )
    assert crossings > 0 and endings > 0 and small > 0 and len(stages) >= 2
