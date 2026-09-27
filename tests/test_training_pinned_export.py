"""Check a public training run that pins one of our exported actors.

A run pins another run's final export by absolute path. Its record is a
verified export whose source run played self-play only, so its controller
exposure is none; its weights travel in the carry, their digest is recorded,
and its memory rule is saved in the carry. A copy of the run interrupted after
training and resumed from its first checkpoint, which resolves the saved path
again, restores the pinned weights from the checkpoint and checks their digest,
ends with the same actor and the same update log, apart from wall-clock
timings, as the uninterrupted run.
"""

# pyright: reportPrivateUsage=false
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from marl_battlegrounds.baselines.ppo import PPOConfig
from marl_battlegrounds.evaluation.recording_identity import tree_digest
from marl_battlegrounds.training import checkpoints, runner
from marl_battlegrounds.training.runner import TrainConfig, TrainResult

PPO = PPOConfig(rollout_length=2, epochs=1, spawn_frame="world")


@pytest.fixture(scope="module")
def identity() -> Iterator[dict[str, Any]]:
    # Hold source identity fixed while other work may edit files in the tree.
    fixed = checkpoints.runtime_identity()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(checkpoints, "runtime_identity", lambda: fixed)
        yield fixed


@pytest.fixture(scope="module")
def source(identity: dict[str, Any], tmp_path_factory: pytest.TempPathFactory) -> Path:
    del identity
    config = TrainConfig(
        keep_past=0,
        num_envs=4,
        total_env_steps=8,
        seed=741,
        ppo=PPO,
        checkpoint_interval_updates=1,
        metrics="none",
        verbose=False,
    )
    result = runner.train(config, output_dir=tmp_path_factory.mktemp("source") / "run")
    return result.final_actor.resolve()


def _config(export: Path) -> TrainConfig:
    return TrainConfig(
        keep_past=0,
        num_envs=4,
        total_env_steps=16,
        seed=742,
        ppo=PPO,
        checkpoint_interval_updates=1,
        metrics="none",
        verbose=False,
        pinned_opponent_share=0.5,
        pinned_opponent=str(export),
    )


@pytest.fixture(scope="module")
def pinned(
    source: Path,
    identity: dict[str, Any],
    tmp_path_factory: pytest.TempPathFactory,
) -> TrainResult:
    del identity
    return runner.train(
        _config(source), output_dir=tmp_path_factory.mktemp("pinned") / "run"
    )


def _learning_rows(run: Path) -> list[dict[str, object]]:
    # Wall-clock fields and the attempt ID differ when an update is re-run.
    rows = [
        json.loads(line)
        for line in (run / "training_updates.jsonl").read_text().splitlines()
    ]
    return [
        {
            key: value
            for key, value in row.items()
            if key != "attempt_id" and not key.endswith("_seconds")
        }
        for row in rows
    ]


def _checkpoint(run: Path, updates: int) -> Path:
    for path in sorted((run / "checkpoints").iterdir()):
        if (
            checkpoints.read_checkpoint_description(path)["counters"]["updates"]
            == updates
        ):
            return path
    raise AssertionError(f"No checkpoint after {updates} updates")


def test_a_pinned_export_is_verified_with_its_weights_in_the_carry(
    source: Path, pinned: TrainResult
) -> None:
    exposure = json.loads((pinned.run_dir / "exposure.json").read_text())
    record = exposure["pinned_opponent"]
    assert record["reference"] == str(source)
    assert record["execution"] == "jax"
    assert record["memory_rule"] == "saved in the carry"
    assert record["evidence"]["source"] == "verified export"
    assert record["evidence"]["exposure"] == "none"
    assert record["variables_digest"] == tree_digest(
        checkpoints.load_system(source).variables
    )


def test_an_interrupted_run_pinning_an_export_resumes_to_the_same_result(
    source: Path,
    pinned: TrainResult,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    identity: dict[str, Any],
) -> None:
    del identity

    def interrupt_report(self: runner._Run) -> tuple[Path, ...]:
        raise KeyboardInterrupt("Declared interruption after final training")

    original_report = runner._Run.report
    monkeypatch.setattr(runner._Run, "report", interrupt_report)
    destination = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        runner.train(_config(source), output_dir=destination)
    monkeypatch.setattr(runner._Run, "report", original_report)
    before_rows = _learning_rows(destination)
    restored = runner.train(resume_from=_checkpoint(destination, 1))
    assert restored.completed_updates == 2
    assert _learning_rows(destination) == before_rows
    assert (
        checkpoints.artifact_identity(restored.final_actor)["actor_digest"]
        == checkpoints.artifact_identity(pinned.final_actor)["actor_digest"]
    )
