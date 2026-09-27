"""Check each PPO method through one complete native-horizon public workflow.

Real Alpha validation, interrupted resume, checkpoint selection, actor loading
and evaluation share the existing owners. Validation must not change learner
leaves; saved results must keep exact counts and method/actor identity. These
four CPU cases establish workflow correctness, not learning or GPU speed.
"""

import json
from dataclasses import replace
from importlib import import_module
from pathlib import Path

import pytest
from tests.training_learner_helpers import equal

from marl_battlegrounds.baselines.ppo import PPOConfig, PPOMethod
from marl_battlegrounds.training.learner import LearnerState


@pytest.mark.parametrize("method", ("mappo", "ippo", "ff_mappo", "ff_ippo"))
def test_public_variant_train_validate_resume_select_load_and_evaluate(
    method: PPOMethod,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import marl_battlegrounds as marl_bgs
    from marl_battlegrounds import training
    from marl_battlegrounds.evaluation.recording_context import (
        capture_recording_provenance,
    )
    from marl_battlegrounds.evaluation.recording_identity import (
        normalize_system_registration,
        tree_digest,
    )
    from marl_battlegrounds.training import checkpoints, runner
    from marl_battlegrounds.training._compilation import execution_identity

    identity = checkpoints.runtime_identity()
    monkeypatch.setattr(checkpoints, "runtime_identity", lambda: identity)
    evaluator = import_module("marl_battlegrounds.evaluation.evaluate")
    provenance = capture_recording_provenance()

    def fixed_provenance(**_: object) -> dict[str, object]:
        return provenance

    monkeypatch.setattr(evaluator, "capture_recording_provenance", fixed_provenance)
    config = training.TrainConfig(
        keep_past=0,
        method=method,
        seed=19047101,
        num_envs=4,
        total_env_steps=32,
        ppo=PPOConfig(rollout_length=4, groups=2, minibatches=2, epochs=1),
        checkpoint_interval_updates=1,
        validation_fractions=(0.5, 1.0),
        routine_seed_pairs=1,
        confirmation_seed_pairs=1,
        validation_opponents=("tdm-alpha",),
        purpose="development",
        metrics="priority",
        recording=False,
        verbose=False,
    )
    saved_states: dict[Path, dict[int, LearnerState]] = {}
    original_save = runner._Run.save  # pyright: ignore[reportPrivateUsage]
    interrupted = tmp_path / "interrupted"
    stop = False

    def save(execution: runner._Run) -> Path:  # pyright: ignore[reportPrivateUsage]
        checkpoint = original_save(execution)
        update = int(execution.state.completed_updates)
        saved_states.setdefault(execution.root, {})[update] = execution.state
        if stop and execution.root == interrupted and update == 1:
            raise KeyboardInterrupt("Declared stop after the first saved update")
        return checkpoint

    monkeypatch.setattr(runner._Run, "save", save)  # pyright: ignore[reportPrivateUsage]
    # The reference differs only by its absent panel. Real validation must not
    # change any learner leaf. No evaluator horizon or score is replaced.
    reference = training.train(
        replace(config, validation_opponents=None), output_dir=tmp_path / "reference"
    )
    stop = True
    with pytest.raises(KeyboardInterrupt, match="first saved update"):
        training.train(config, output_dir=interrupted)
    first_rows = (interrupted / "training_updates.jsonl").read_bytes()
    previous_games = {
        path: path.read_bytes() for path in (interrupted / "validation").rglob("*.csv")
    }
    assert previous_games
    pointer = json.loads((interrupted / "latest_checkpoint.json").read_text())
    checkpoint = interrupted / pointer["relative_path"]
    assert checkpoints.read_checkpoint_details(checkpoint)["counters"] == {
        "updates": 1,
        "env_steps": 16,
    }
    equal(saved_states[reference.run_dir][1], saved_states[interrupted][1])
    stop = False
    resumed = training.train(resume_from=checkpoint)
    equal(saved_states[reference.run_dir][2], saved_states[interrupted][2])
    assert (interrupted / "training_updates.jsonl").read_bytes().startswith(first_rows)
    assert all(path.read_bytes() == raw for path, raw in previous_games.items())
    assert resumed.completed_env_steps == reference.completed_env_steps == 32
    assert resumed.completed_updates == reference.completed_updates == 2
    assert (
        checkpoints.artifact_identity(resumed.final_actor)["actor_digest"]
        == checkpoints.artifact_identity(reference.final_actor)["actor_digest"]
    )
    details = json.loads((interrupted / "run_details.json").read_text())
    assert details["config"]["method"] == method
    assert details["config"]["validation_opponents"] == ["tdm-alpha"]
    assert details["execution"] == execution_identity()
    assert details["schemas"] == checkpoints.checkpoint_schemas(method)
    validations = json.loads((interrupted / "validation_results.json").read_text())
    assert [row["env_steps"] for row in validations if row["purpose"] == "routine"] == [
        0,
        16,
        32,
    ]
    assert all(row["complete"] and row["games"] == 10 for row in validations)
    assert resumed.selected_actor is not None
    selected = checkpoints.artifact_identity(resumed.selected_actor)
    assert selected["env_steps"] in (16, 32)
    selection = json.loads((interrupted / "selection.json").read_text())
    assert selection["checkpoint_id"] == selected["metadata"]["checkpoint_id"]
    assert selection["actor_digest"] == selected["actor_digest"]
    ancestor = checkpoints.read_checkpoint_description(
        checkpoints.artifact_directory(
            interrupted, "checkpoints", selection["checkpoint_id"]
        )
    )
    assert ancestor["actor_digest"] == selected["weight_digest"]
    assert ancestor["schemas"] == selected["schemas"]
    loaded = training.load_system(resumed.selected_actor)
    assert tree_digest(loaded.variables) == selected["weight_digest"]
    result = marl_bgs.evaluate(
        loaded,
        "tdm-alpha",
        num_episodes=2,
        num_envs=4,
        maps=[42],
        seed=19047101,
        metrics="priority",
        phase="validation",
        output_dir=tmp_path / "selected-evaluation",
    )
    assert len(result.completed_episode_ids) == 2
    assert result.paths is not None
    evaluated = json.loads(result.paths["run_details"].read_text())
    system_id, registration = normalize_system_registration(
        loaded, phase="validation", frozen=True
    )
    assert evaluated["systems"][system_id] == registration
    assert all(
        entry["system_ids"]["team_a"] == system_id
        for entry in evaluated["passes"].values()
    )
    report = training.analyze([interrupted], output_dir=tmp_path / "review")
    assert report["complete"] and report["runs"][0]["method"] == method
