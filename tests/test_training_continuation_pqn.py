"""Prove public PQN continuation keeps its rows, statistics and optimizer tree.

A tiny CPU parent ends on a partial learned block. Added work refuses an
unapproved terminal-rate call, then uses an explicit future linear rate and
exploration rule. Full numerical state matches direct declared continuation;
ordinary child resume takes no extra sample. An unchanged grandchild preserves
the saved exploration value exactly. This is no learning claim.
"""

import json
from pathlib import Path

import pytest
from tests.training_continuation_helpers import (
    capture_runs,
    compare_direct_child,
    fixed_source,
    latest,
)
from tests.training_learner_helpers import equal

from marl_battlegrounds.baselines.pqn import PQNConfig
from marl_battlegrounds.training import TrainConfig, extend_training, train


def test_pqn_partial_child_matches_full_state_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixed_source(monkeypatch, "pqn_vdn")
    starts, ends = capture_runs(monkeypatch)
    settings = PQNConfig(
        rollout_length=2,
        memory_window=1,
        epochs=1,
        num_minibatches=1,
    )
    parent = train(
        TrainConfig(
            method="pqn_vdn",
            num_envs=2,
            total_env_steps=12,
            seed=720,
            pqn=settings,
            checkpoint_interval_updates=1,
            checkpoint_env_steps=(4,),
            metrics="none",
            verbose=False,
        ),
        output_dir=tmp_path / "parent",
    )
    with pytest.raises(ValueError, match="terminal learning rate"):
        extend_training(
            latest(parent.run_dir),
            additional_env_steps=6,
            output_dir=tmp_path / "refused",
        )
    assert not (tmp_path / "refused").exists()
    child = extend_training(
        latest(parent.run_dir),
        additional_env_steps=4,
        output_dir=tmp_path / "child",
        changes={
            "learning_rate": {"kind": "linear", "q_lr": 0.0005, "optimizer_steps": 2},
            "exploration": {
                "kind": "linear",
                "epsilon": 0.37,
                "end_epsilon": 0.04,
                "learning_blocks": 17,
            },
        },
    )
    assert child.completed_env_steps == 16
    assert child.completed_updates == parent.completed_updates + 1
    compare_direct_child(
        parent.run_dir, child.run_dir, starts, ends, settings, "pqn_vdn"
    )
    final = ends[child.run_dir]
    resumed = train(resume_from=latest(child.run_dir))
    assert resumed.completed_env_steps == 16
    assert resumed.completed_updates == child.completed_updates
    equal(starts[child.run_dir][1], final)
    grandchild = extend_training(
        latest(child.run_dir),
        additional_env_steps=2,
        output_dir=tmp_path / "grandchild",
    )
    assert grandchild.completed_env_steps == 18
    assert grandchild.completed_updates == child.completed_updates + 1
    compare_direct_child(
        child.run_dir, grandchild.run_dir, starts, ends, settings, "pqn_vdn"
    )

    warm_parent = next(
        path.parent
        for path in (parent.run_dir / "checkpoints").glob("*/checkpoint_details.json")
        if json.loads(path.read_text())["counters"]["env_steps"] == 4
    )
    warm = extend_training(
        warm_parent, additional_env_steps=2, output_dir=tmp_path / "warm_child"
    )
    assert warm.completed_env_steps == 6
    assert warm.completed_updates == 0
    warm_final = ends[warm.run_dir]
    warm_resumed = train(resume_from=latest(warm.run_dir))
    assert warm_resumed.completed_updates == 0
    equal(starts[warm.run_dir][1], warm_final)
    warm_learned = extend_training(
        latest(warm.run_dir),
        additional_env_steps=2,
        output_dir=tmp_path / "warm_grandchild",
    )
    assert warm_learned.completed_env_steps == 8
    assert warm_learned.completed_updates == 1
    compare_direct_child(
        warm.run_dir, warm_learned.run_dir, starts, ends, settings, "pqn_vdn"
    )
