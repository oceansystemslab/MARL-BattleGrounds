"""Check PQN-VDN's offset boundaries through the public runner when H < T.

One CPU case with four games, blocks of 4 rounds, a memory window of 2 (so
W = 6 initial rounds, collected as a chunk of 4 and a chunk of 2), one epoch of
two minibatches, 64 transitions (16 rounds, so the final block has 2 real
rounds), an Alpha panel with one seed pair, validation fractions 0.5 and 1.0,
extra saves at 24 and 40 transitions and a save interval of 3 optimizer
steps. Contracts checked here: the runner requests blocks of 4, 2, 4, 4 and
4 rounds; routine validation runs at 0, 40 and 64 transitions, never at 32
(where the fixed-block rule would put the 0.5 point); saves happen at 0, 24
(W), 40, 56 (the interval: optimizer steps 2 to 4 cross 3) and 64, never at
16. A run stopped right after its save at 40, with that validation still
pending, resumes: recovery checks its saved host state against the same
offset schedule, the pending validation runs, and the run finishes with the
learner numbers, used counts and exposure of an uninterrupted run without a
panel. These CPU checks prove workflow wiring, not learning.
"""

# pyright: reportPrivateUsage=false
import json
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest
from tests.training_learner_helpers import equal

from marl_battlegrounds.baselines.pqn import PQNConfig

type Tree = Any


def _fixed_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    from marl_battlegrounds.evaluation.recording_context import (
        capture_recording_provenance,
    )
    from marl_battlegrounds.training import checkpoints

    identity = checkpoints.runtime_identity(method="pqn_vdn")
    monkeypatch.setattr(
        checkpoints, "runtime_identity", lambda method="mappo": identity
    )
    evaluator = import_module("marl_battlegrounds.evaluation.evaluate")
    provenance = capture_recording_provenance()

    def fixed_provenance(**_: object) -> dict[str, object]:
        return provenance

    monkeypatch.setattr(evaluator, "capture_recording_provenance", fixed_provenance)


def test_offset_schedule_saves_and_recovery_through_the_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marl_battlegrounds import training
    from marl_battlegrounds.training import collection as collection_module
    from marl_battlegrounds.training import runner

    _fixed_identity(monkeypatch)
    config = training.TrainConfig(
        keep_past=0,
        method="pqn_vdn",
        seed=19049105,
        num_envs=4,
        total_env_steps=64,
        pqn=PQNConfig(rollout_length=4, memory_window=2, epochs=1, num_minibatches=2),
        validation_opponents=("tdm-alpha",),
        validation_fractions=(0.5, 1.0),
        routine_seed_pairs=1,
        confirmation_seed_pairs=1,
        checkpoint_interval_updates=3,
        checkpoint_env_steps=(24, 40),
        verbose=False,
    )
    lengths: list[int] = []
    saves: dict[Path, dict[int, Tree]] = {}
    original_collect = collection_module.collect_training_rollout
    original_save = runner._Run.save
    stop = tmp_path / "interrupted"

    def collect(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        lengths.append(kwargs["length"])
        return original_collect(*args, **kwargs)

    def save(execution: runner._Run) -> Path:
        checkpoint = original_save(execution)
        steps = execution.host["env_steps"]
        saves.setdefault(execution.root, {})[steps] = (
            execution.state if steps == 64 else None
        )
        if execution.root == stop and steps == 40:
            raise KeyboardInterrupt("Declared stop after the save at 40")
        return checkpoint

    monkeypatch.setattr(collection_module, "collect_training_rollout", collect)
    monkeypatch.setattr(runner._Run, "save", save)
    reference = training.train(
        replace(config, validation_opponents=None), output_dir=tmp_path / "reference"
    )
    assert lengths == [4, 2, 4, 4, 4]
    assert sorted(saves[reference.run_dir]) == [0, 24, 40, 56, 64]
    rows = [
        json.loads(line)
        for line in (reference.run_dir / "training_updates.jsonl").open()
    ]
    assert [row["env_steps"] for row in rows] == [16, 24, 40, 56, 64]
    assert [row["completed_updates"] for row in rows] == [0, 0, 2, 4, 6]
    with pytest.raises(KeyboardInterrupt, match="save at 40"):
        training.train(config, output_dir=stop)
    status = json.loads((stop / "status.json").read_text())
    assert status["pending_task"] == {"routine": True, "random": False}
    pointer = json.loads((stop / "latest_checkpoint.json").read_text())
    resumed = training.train(resume_from=stop / pointer["relative_path"])
    assert sorted(saves[stop]) == [0, 24, 40, 56, 64]
    validations = json.loads((stop / "validation_results.json").read_text())
    routine = [row for row in validations if row["purpose"] == "routine"]
    assert [row["env_steps"] for row in routine] == [0, 40, 64]
    assert [row["optimizer_steps"] for row in routine] == [0, 2, 6]
    assert resumed.completed_updates == reference.completed_updates == 6
    assert resumed.selected_actor is not None
    equal(saves[stop][64], saves[reference.run_dir][64])
    exposures = [
        json.loads((root / "exposure.json").read_text())["used_exposure"]
        for root in (reference.run_dir, stop)
    ]
    assert exposures[0] == exposures[1]
    last = [
        json.loads((root / "training_updates.jsonl").read_text().splitlines()[-1])
        for root in (reference.run_dir, stop)
    ]
    for key in (
        "completed_blocks",
        "learning_blocks",
        "used_sequences",
        "used_td_pairs",
        "used_prefix_td_pairs",
        "used_agent_utilities",
    ):
        assert last[0][key] == last[1][key]
