"""Prove public QMIX continuation retains replay, targets and exact clocks.

A tiny CPU run ends a parent and child on partial blocks, changes only future
rate/exploration, compares full state with direct declared continuation, then
resumes the child without taking extra samples. This is no learning claim.
"""

from pathlib import Path

import pytest
from tests.training_continuation_helpers import (
    capture_runs,
    compare_direct_child,
    fixed_source,
    latest,
)
from tests.training_learner_helpers import equal

from marl_battlegrounds.baselines.qmix import QMIXConfig
from marl_battlegrounds.training import TrainConfig, extend_training, train


def test_qmix_partial_child_matches_full_state_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixed_source(monkeypatch, "qmix")
    starts, ends = capture_runs(monkeypatch)
    settings = QMIXConfig(
        rollout_length=2,
        buffer_size=4,
        min_buffer_size=2,
        sample_sequence_length=2,
        sample_batch_size=2,
        epochs=1,
        update_period=2,
        eps_decay=20,
    )
    parent = train(
        TrainConfig(
            method="qmix",
            num_envs=2,
            total_env_steps=10,
            seed=719,
            qmix=settings,
            checkpoint_interval_updates=1,
            metrics="none",
            verbose=False,
        ),
        output_dir=tmp_path / "parent",
    )
    child = extend_training(
        latest(parent.run_dir),
        additional_env_steps=6,
        output_dir=tmp_path / "child",
        changes={
            "learning_rate": {"kind": "constant", "q_lr": 0.00005},
            "exploration": {
                "kind": "linear",
                "epsilon": 0.4,
                "end_epsilon": 0.2,
                "env_steps": 4,
            },
        },
    )
    assert child.completed_env_steps == 16
    assert child.completed_updates == parent.completed_updates + 2
    compare_direct_child(parent.run_dir, child.run_dir, starts, ends, settings, "qmix")
    final = ends[child.run_dir]
    resumed = train(resume_from=latest(child.run_dir))
    assert resumed.completed_env_steps == 16
    assert resumed.completed_updates == child.completed_updates
    equal(starts[child.run_dir][1], final)
