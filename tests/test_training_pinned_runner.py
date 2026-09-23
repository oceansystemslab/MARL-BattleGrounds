"""Check a public training run that pins tdm-alpha and exports its evidence.

A recorded run with pinned_opponent="tdm-alpha" saves the reference in its run
details, the pinned record in every checkpoint's collection block and in the
recording details, labelled opponent rows in exposure.json, and the record in
each exported actor's metadata. The run's own export is a verified export whose
evidence names Alpha's controller, the same one a run pinning Alpha directly
records, and makes all eight protected scenarios familiar, so a later run that
pins this export inherits that exposure; the same weights claimed from another
checkpoint of the run are only a declared export.
"""

# pyright: reportPrivateUsage=false
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from marl_battlegrounds.baselines.ppo import PPOConfig
from marl_battlegrounds.evaluation.policy_execution import policy
from marl_battlegrounds.training import checkpoints, runner
from marl_battlegrounds.training._content import (
    pinned_opponent_evidence,
    prepare_training_content,
)
from marl_battlegrounds.training.runner import TrainConfig, TrainResult


def _config() -> TrainConfig:
    return TrainConfig(
        num_envs=4,
        total_env_steps=16,
        seed=731,
        ppo=PPOConfig(rollout_length=2, epochs=1, spawn_frame="world"),
        checkpoint_interval_updates=1,
        metrics="none",
        recording=True,
        verbose=False,
        pinned_opponent_share=0.5,
        pinned_opponent="tdm-alpha",
    )


@pytest.fixture(scope="module")
def identity() -> Iterator[dict[str, Any]]:
    # Hold source identity fixed while other work may edit files in the tree.
    fixed = checkpoints.runtime_identity()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(checkpoints, "runtime_identity", lambda: fixed)
        yield fixed


@pytest.fixture(scope="module")
def baseline(
    identity: dict[str, Any], tmp_path_factory: pytest.TempPathFactory
) -> TrainResult:
    del identity
    return runner.train(_config(), output_dir=tmp_path_factory.mktemp("alpha") / "run")


def _checkpoint(run: Path, updates: int) -> Path:
    for path in sorted((run / "checkpoints").iterdir()):
        if (
            checkpoints.read_checkpoint_description(path)["counters"]["updates"]
            == updates
        ):
            return path
    raise AssertionError(f"No checkpoint after {updates} updates")


def test_the_pinned_reference_and_record_reach_every_saved_record(
    baseline: TrainResult,
) -> None:
    run = baseline.run_dir
    details = json.loads((run / "run_details.json").read_text())
    assert details["config"]["pinned_opponent"] == "tdm-alpha"
    saved = checkpoints.read_checkpoint_details(_checkpoint(run, 2))
    record = saved["collection"]["pinned_opponent"]
    assert record["name"] == "tdm-alpha" and record["reference"] == "tdm-alpha"
    assert record["memory_rule"] == "saved in the carry"
    assert record["evidence"]["exposure"] == "known"
    assert saved["metadata"]["recording"]["details"]["pinned_opponent"] == record
    exposure = json.loads((run / "exposure.json").read_text())
    assert exposure["pinned_opponent"] == record
    assert exposure["opponent_rows"][:2] == ["current weights", "pinned: tdm-alpha"]
    assert len(exposure["opponent_rows"]) == len(exposure["starts_by_opponent"]) == 21
    export = baseline.final_actor
    metadata = checkpoints.read_checkpoint_description(export)["metadata"]
    assert metadata["pinned_opponent"] == record


def test_exports_are_verified_only_when_linked_to_their_own_checkpoint(
    baseline: TrainResult,
) -> None:
    binding = prepare_training_content().binding
    export = baseline.final_actor
    verified = pinned_opponent_evidence(
        binding, checkpoints.load_system(export), export=export
    )
    assert verified["source"] == "verified export"
    assert verified["exposure"] == "known"
    assert verified["familiar_scenarios"] == list(range(1, 9))
    direct = pinned_opponent_evidence(binding, policy("tdm-alpha"))
    assert len(verified["controllers"]) == 1  # pyright: ignore[reportArgumentType]
    assert verified["controllers"] == direct["controllers"]
    first = _checkpoint(baseline.run_dir, 1)
    metadata = dict(checkpoints.read_checkpoint_description(export)["metadata"])
    metadata["checkpoint_id"] = first.name
    metadata["env_steps"] = checkpoints.read_checkpoint_description(first)["counters"][
        "env_steps"
    ]
    claimed = checkpoints.export_system(
        checkpoints.load_system(export).variables,
        baseline.run_dir / "actors" / "claimed",
        metadata=metadata,
        spawn_frame="world",
    )
    unlinked = pinned_opponent_evidence(
        binding, checkpoints.load_system(claimed), export=claimed
    )
    assert unlinked["source"] == "declared export"
    assert unlinked["exposure"] == "unknown"
