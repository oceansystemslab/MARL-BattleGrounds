"""Check recurrent QMIX through complete native-horizon public workflows.

Every case uses four games, blocks of 8 rounds, a 64-row replay that starts
learning at 32 rows, sequences of 20, batches of 4, one epoch (two in the C
case) and hard target copies every 2 steps. A 192-transition run has three
warmup blocks and then one optimizer step per block. Contracts checked here:
the plain run validates against Alpha at initialization, during warmup and
after learning, never selects a warmup actor, confirms, selects, exports,
loads and evaluates the greedy actor with its real M8 registration, runs the
slot comparison, and writes QMIX reports with the QMIX columns and summary
lines; a recorded run interrupted after a save resumes to the same learner
numbers, exposure and sampled-use totals as an uninterrupted run without a
panel, keeping its earlier log rows and games; Random diagnostics run fresh,
are reused by a second run's initialization and pass recovery verification,
while a re-signed checkpoint whose saved source exposure has the wrong length
or whose saved Random result has the wrong optimizer count is refused before
recovery changes any file;
a warmup save resumes exactly, progress output shows the warmup label and
optimizer steps, and no run keeps its first learner state (a whole replay)
alive once training moves on, fresh or resumed; and curriculum, reward
shaping and both together reach the collection and complete, with each
block's logged task and shaping means equal to its rollout's, and with two
optimizer steps per block the saves follow multiples of the interval crossed
mid-block. Saved results carry method and optimizer counts. These CPU cases
establish workflow correctness, not learning or GPU speed.
"""

# pyright: reportPrivateUsage=false
import gc
import hashlib
import json
import weakref
from collections.abc import Callable
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from tests.training_learner_helpers import equal

from marl_battlegrounds.baselines.qmix import QMIXConfig

type Tree = Any
_QMIX = QMIXConfig(
    rollout_length=8,
    buffer_size=64,
    min_buffer_size=32,
    sample_sequence_length=20,
    sample_batch_size=4,
    epochs=1,
    update_period=2,
)


def _fixed_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    from marl_battlegrounds.evaluation.recording_context import (
        capture_recording_provenance,
    )
    from marl_battlegrounds.training import checkpoints

    identity = checkpoints.runtime_identity(method="qmix")
    monkeypatch.setattr(
        checkpoints, "runtime_identity", lambda method="mappo": identity
    )
    evaluator = import_module("marl_battlegrounds.evaluation.evaluate")
    provenance = capture_recording_provenance()

    def fixed_provenance(**_: object) -> dict[str, object]:
        return provenance

    monkeypatch.setattr(evaluator, "capture_recording_provenance", fixed_provenance)


def _config(**changes: Any) -> Any:  # noqa: ANN401
    from marl_battlegrounds import training

    settings: dict[str, Any] = {
        "method": "qmix",
        "seed": 19048101,
        "num_envs": 4,
        "total_env_steps": 192,
        "qmix": _QMIX,
        "validation_fractions": (0.5, 2 / 3, 1.0),
        "routine_seed_pairs": 1,
        "confirmation_seed_pairs": 1,
        "purpose": "development",
        "metrics": "priority",
        "recording": False,
        "verbose": False,
        **changes,
    }
    return training.TrainConfig(**settings)


def _capture_saves(
    monkeypatch: pytest.MonkeyPatch,
    stop_root: Path | None,
    stop_steps: int,
    keep: set[int] | None = None,
) -> dict[Path, dict[int, Tree]]:
    from marl_battlegrounds.training import runner

    saved: dict[Path, dict[int, Tree]] = {}
    original = runner._Run.save

    def save(execution: runner._Run) -> Path:
        checkpoint = original(execution)
        steps = execution.host["env_steps"]
        # Only kept states are held here, so other states can be freed.
        saved.setdefault(execution.root, {})[steps] = (
            execution.state if keep is None or steps in keep else None
        )
        if execution.root == stop_root and steps == stop_steps:
            raise KeyboardInterrupt("Declared stop after a saved block")
        return checkpoint

    monkeypatch.setattr(runner._Run, "save", save)
    return saved


def _resign_host_state(
    checkpoint: Path, change: Callable[[dict[str, Any]], None]
) -> None:
    from marl_battlegrounds.training import checkpoints

    description = checkpoint / "checkpoint_details.json"
    details = json.loads(description.read_text())
    change(details["metadata"]["host_state"])
    unsigned = {key: value for key, value in details.items() if key != "checkpoint_id"}
    details["checkpoint_id"] = hashlib.sha256(
        checkpoints._json_bytes(unsigned)
    ).hexdigest()
    description.write_text(json.dumps(details))


def _run_files(root: Path) -> dict[Path, bytes]:
    return {
        path: path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and path.name != "checkpoint_details.json"
    }


def _latest(root: Path) -> Path:
    pointer = json.loads((root / "latest_checkpoint.json").read_text())
    return root / pointer["relative_path"]


def test_plain_run_validates_selects_exports_loads_and_evaluates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import marl_battlegrounds as marl_bgs
    from marl_battlegrounds import training
    from marl_battlegrounds.evaluation.recording_identity import (
        normalize_system_registration,
        tree_digest,
    )
    from marl_battlegrounds.training import checkpoints
    from marl_battlegrounds.training.validation import run_slot_diagnostic

    _fixed_identity(monkeypatch)
    result = training.train(
        _config(validation_opponents=("tdm-alpha",)), output_dir=tmp_path / "run"
    )
    root = result.run_dir
    assert result.completed_env_steps == 192 and result.completed_updates == 3
    rows = [json.loads(line) for line in (root / "training_updates.jsonl").open()]
    assert [row["completed_blocks"] for row in rows] == [1, 2, 3, 4, 5, 6]
    assert [row["learning_blocks"] for row in rows] == [0, 0, 0, 1, 2, 3]
    assert [row["loss"] is None for row in rows] == [True] * 3 + [False] * 3
    assert rows[-1]["used_td_pairs"] == 3 * 4 * 19
    assert rows[-1]["replay_rows_per_lane"] == 48
    assert "policy_loss" not in rows[-1] and "update_index" not in rows[-1]
    validations = json.loads((root / "validation_results.json").read_text())
    routine = [row for row in validations if row["purpose"] == "routine"]
    assert [row["env_steps"] for row in routine] == [0, 96, 128, 192]
    assert [row["optimizer_steps"] for row in routine] == [0, 0, 1, 3]
    assert all(row["method"] == "qmix" for row in validations)
    assert all(row["complete"] and row["games"] == 10 for row in validations)
    confirmed = {
        row["env_steps"] for row in validations if row["purpose"] == "confirmation"
    }
    assert confirmed <= {128, 160, 192} and 96 not in confirmed
    assert result.selected_actor is not None
    selected = checkpoints.artifact_identity(result.selected_actor)
    assert selected["method"] == "qmix" and selected["optimizer_steps"] in (1, 3)
    loaded = training.load_system(result.selected_actor)
    assert float(loaded.variables.epsilon) == 0.0
    ancestor = checkpoints.read_checkpoint_description(
        root / "checkpoints" / selected["metadata"]["checkpoint_id"]
    )
    assert tree_digest(loaded.variables.params) == ancestor["actor_digest"]
    evaluation = marl_bgs.evaluate(
        loaded,
        "tdm-alpha",
        num_episodes=2,
        num_envs=4,
        maps=[42],
        seed=19048101,
        metrics="priority",
        phase="validation",
        output_dir=tmp_path / "selected-evaluation",
    )
    assert len(evaluation.completed_episode_ids) == 2
    assert evaluation.paths is not None
    evaluated = json.loads(evaluation.paths["run_details"].read_text())
    system_id, registration = normalize_system_registration(
        loaded, phase="validation", frozen=True
    )
    assert evaluated["systems"][system_id] == registration
    slot = run_slot_diagnostic(
        result.final_actor,
        result.selected_actor,
        output_dir=tmp_path / "slot",
        seed_blocks=2,
        num_envs=4,
    )
    final = checkpoints.artifact_identity(result.final_actor)
    assert slot["purpose"] == "slot" and slot["actor_digest"] == final["actor_digest"]
    assert slot["opponent_digest"] == selected["actor_digest"]
    exposure = json.loads((root / "exposure.json").read_text())
    assert exposure["sampled_exposure"]["used_td_pairs"] == 3 * 4 * 19
    report = training.analyze([root], output_dir=tmp_path / "review")
    assert report["complete"] and report["runs"][0]["method"] == "qmix"
    assert Path(report["artifacts"]["qmix_png"]).stat().st_size > 0
    header = Path(report["artifacts"]["curve"]).read_text().splitlines()[0].split(",")
    assert {
        "loss",
        "replay_rows_per_lane",
        "td_pairs_per_transition",
        "epsilon",
    } <= set(header)
    summary = Path(report["artifacts"]["summary"]).read_text()
    assert "Optimizer steps: 3." in summary and "Used TD pairs" in summary


def test_recorded_run_resumes_after_interruption_like_an_uninterrupted_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marl_battlegrounds import training
    from marl_battlegrounds.training import checkpoints

    _fixed_identity(monkeypatch)
    interrupted = tmp_path / "interrupted"
    saved = _capture_saves(monkeypatch, interrupted, 128)
    config = _config(
        recording=True,
        validation_opponents=("tdm-alpha",),
        checkpoint_env_steps=(128,),
    )
    # The reference differs only by its absent panel; validation never changes
    # learner numbers, so its full evaluation work is not repeated.
    reference = training.train(
        replace(config, validation_opponents=None), output_dir=tmp_path / "reference"
    )
    with pytest.raises(KeyboardInterrupt, match="saved block"):
        training.train(config, output_dir=interrupted)
    first_rows = (interrupted / "training_updates.jsonl").read_bytes()
    games = {
        path: path.read_bytes() for path in (interrupted / "validation").rglob("*.csv")
    }
    assert games
    checkpoint = _latest(interrupted)
    assert checkpoints.read_checkpoint_details(checkpoint)["counters"] == {
        "updates": 1,
        "env_steps": 128,
        "completed_blocks": 4,
        "learning_blocks": 1,
    }
    equal(saved[reference.run_dir][128], saved[interrupted][128])
    resumed = training.train(resume_from=checkpoint)
    equal(saved[reference.run_dir][192], saved[interrupted][192])
    assert (interrupted / "training_updates.jsonl").read_bytes().startswith(first_rows)
    assert all(path.read_bytes() == raw for path, raw in games.items())
    assert resumed.completed_updates == reference.completed_updates == 3
    # Exact host totals survive the interruption, not only the learner state.
    totals = ("sampled_exposure",)
    exposures = [
        {key: json.loads((root / "exposure.json").read_text())[key] for key in totals}
        for root in (reference.run_dir, interrupted)
    ]
    assert exposures[0] == exposures[1]
    last = [
        json.loads((root / "training_updates.jsonl").read_text().splitlines()[-1])
        for root in (reference.run_dir, interrupted)
    ]
    for key in (
        "env_steps",
        "completed_blocks",
        "learning_blocks",
        "completed_updates",
        "sampled_sequences",
        "used_td_pairs",
        "used_agent_utilities",
    ):
        assert last[0][key] == last[1][key]
    assert (
        checkpoints.artifact_identity(resumed.final_actor)["actor_digest"]
        == checkpoints.artifact_identity(reference.final_actor)["actor_digest"]
    )
    assert resumed.selected_actor is not None
    report = training.analyze([interrupted], output_dir=tmp_path / "review")
    assert report["complete"]


def test_random_diagnostics_run_fresh_reuse_initialization_and_verify_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marl_battlegrounds import training

    _fixed_identity(monkeypatch)
    first = training.train(
        _config(random_diagnostic_seed_pairs=1), output_dir=tmp_path / "first"
    )
    records = json.loads((first.run_dir / "random_diagnostics.json").read_text())
    assert [row["env_steps"] for row in records] == [0, 192]
    assert all(row["method"] == "qmix" for row in records)
    assert [row["optimizer_steps"] for row in records] == [0, 3]
    initial = tmp_path / "initial-random.json"
    initial.write_text(json.dumps(records[0]))
    second_root = tmp_path / "second"
    _capture_saves(monkeypatch, second_root, 128)
    config = _config(
        random_diagnostic_seed_pairs=1,
        random_initialization_result=str(initial),
        checkpoint_env_steps=(128,),
    )
    with pytest.raises(KeyboardInterrupt):
        training.train(config, output_dir=second_root)
    checkpoint = _latest(second_root)
    description = checkpoint / "checkpoint_details.json"
    original = description.read_bytes()

    def longer_source_exposure(host: dict[str, Any]) -> None:
        host["sampled_exposure"]["by_source"].append(0)

    def wrong_optimizer_count(host: dict[str, Any]) -> None:
        host["random_results"][-1]["optimizer_steps"] += 1

    # Re-signed forgeries of saved host state are refused before recovery
    # changes any file of the run.
    for change, match in (
        (longer_source_exposure, "source bank"),
        (wrong_optimizer_count, "optimizer count"),
    ):
        _resign_host_state(checkpoint, change)
        before = _run_files(second_root)
        with pytest.raises(ValueError, match=match):
            training.train(resume_from=checkpoint)
        assert _run_files(second_root) == before
        assert not (second_root / "checkpoint_recovery.json").exists()
        description.write_bytes(original)
    resumed = training.train(resume_from=checkpoint)
    reused = json.loads((second_root / "random_diagnostics.json").read_text())
    assert [row["env_steps"] for row in reused] == [0, 128, 192]
    assert reused[0]["reused_initialization"] is True
    assert resumed.completed_updates == 3


def test_warmup_save_resumes_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from marl_battlegrounds import training
    from marl_battlegrounds.training import checkpoints, runner

    _fixed_identity(monkeypatch)
    interrupted = tmp_path / "interrupted"
    saved = _capture_saves(monkeypatch, interrupted, 96, keep={192})
    # Once training moves past the first learner state, nothing may keep it
    # (for QMIX, a whole replay) alive; check after each run's second block.
    first: list[tuple[int, weakref.ref[Any]]] = []
    released: list[bool] = []
    original_init, original_count = runner._Run.__init__, runner._Run._count_qmix_block

    def remember_start(execution: runner._Run, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        original_init(execution, *args, **kwargs)
        replay = cast(Any, execution.state).replay.experience.actions
        first.append((execution.host["completed_blocks"], weakref.ref(replay)))

    def count(execution: runner._Run, result: Any) -> None:  # noqa: ANN401
        original_count(execution, result)
        start, reference = first[-1]
        if execution.host["completed_blocks"] - start == 2:
            gc.collect()
            released.append(reference() is None)

    monkeypatch.setattr(runner._Run, "__init__", remember_start)
    monkeypatch.setattr(runner._Run, "_count_qmix_block", count)
    config = _config(checkpoint_env_steps=(96,), verbose=True)
    reference = training.train(config, output_dir=tmp_path / "reference")
    output = capsys.readouterr().out
    assert "Warmup: Filling Replay Before Learning" in output
    assert "Optimizer Steps:" in output and "Learning Blocks:" in output
    with pytest.raises(KeyboardInterrupt):
        training.train(config, output_dir=interrupted)
    checkpoint = _latest(interrupted)
    assert checkpoints.read_checkpoint_details(checkpoint)["counters"] == {
        "updates": 0,
        "env_steps": 96,
        "completed_blocks": 3,
        "learning_blocks": 0,
    }
    training.train(resume_from=checkpoint)
    equal(saved[reference.run_dir][192], saved[interrupted][192])
    # Fresh reference, interrupted fresh run and resumed run each freed it.
    assert released == [True, True, True]


@pytest.mark.parametrize(
    "changes",
    [
        {"curriculum": True},
        {"shaping": True},
        {"curriculum": True, "shaping": True},
    ],
    ids=["C", "RS", "C-RS"],
)
def test_curriculum_and_shaping_treatments_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changes: dict[str, bool]
) -> None:
    from marl_battlegrounds import training
    from marl_battlegrounds.training import collection as collection_module
    from marl_battlegrounds.training.runner import config_from_dict, config_to_dict

    _fixed_identity(monkeypatch)
    original_collect = collection_module.collect_training_rollout
    means: list[tuple[float, float]] = []
    # The C case also takes two optimizer steps per block with a save interval
    # of 3, so a real run's saves must follow multiples crossed mid-block.
    multi_step = changes == {"curriculum": True}
    saves = _capture_saves(monkeypatch, None, -1, keep=set())

    def collect(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        collection = args[0]
        assert collection.shaping is bool(changes.get("shaping"))
        assert (collection.schedule.arrays.stage_count > 1) == bool(
            changes.get("curriculum")
        )
        after, rollout = original_collect(*args, **kwargs)
        rows = rollout.transitions
        valid, active = np.asarray(rows.valid), np.asarray(rows.active)
        taken = valid[..., None] & active
        task = np.where(taken, np.asarray(rows.task_rewards), 0).sum()
        shaping = np.where(valid, np.asarray(rows.shaping_reward), 0).sum()
        means.append(
            (float(task) / int(taken.sum()), float(shaping) / int(valid.sum()))
        )
        return after, rollout

    monkeypatch.setattr(collection_module, "collect_training_rollout", collect)
    extra: dict[str, Any] = (
        {
            "qmix": replace(_QMIX, epochs=2),
            "checkpoint_interval_updates": 3,
            "validation_fractions": (1.0,),
        }
        if multi_step
        else {}
    )
    epochs = 2 if multi_step else 1
    config = _config(seed=19048102, total_env_steps=256, **changes, **extra)
    assert config_from_dict(config_to_dict(config)) == config
    result = training.train(config, output_dir=tmp_path / "run")
    assert result.completed_env_steps == 256
    assert result.completed_updates == 5 * epochs
    if multi_step:
        # Updates after blocks 4 to 8 are 2, 4, 6, 8 and 10: blocks 5 and 6
        # cross 3 and 6; blocks 4 and 7 cross nothing; 256 is the final save.
        steps = set(saves[result.run_dir])
        assert {160, 192, 256} <= steps and not steps & {128, 224}
    rows = [
        json.loads(line) for line in (result.run_dir / "training_updates.jsonl").open()
    ]
    # Tiny random-play runs may never score, so shaping can honestly be zero;
    # each logged mean must equal the rollout it came from instead.
    assert len(rows) == len(means) == 8
    for row, (task, shaping) in zip(rows, means, strict=True):
        assert row["task_reward_mean"] == pytest.approx(task, abs=1e-7)
        assert row["shaping_mean"] == pytest.approx(shaping, abs=1e-7)
        if not changes.get("shaping"):
            assert row["shaping_mean"] == 0
    exposure = json.loads((result.run_dir / "exposure.json").read_text())
    assert exposure["sampled_exposure"]["used_td_pairs"] == 5 * epochs * 4 * 19
