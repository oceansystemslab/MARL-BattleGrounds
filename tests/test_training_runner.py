"""Check declared training settings and complete public execution/recovery.

Settings reject invalid counts and unknown fields before backend or file work.
The composed tests below exercise real CPU collection, updates and persistence;
these are engineering checks, not evidence that a short run learned a policy.
The test-only panel path also checks real validation, interruption, selection,
loading and reports while proving validation leaves training results unchanged.
Changed execution settings reject before array restoration or recording rewind.
Scaled actor exports survive pending-output recovery; a changed export scale is
rejected before recording or logs change, even when weight bytes still match.
The pinned opponent share is validated with the other settings, round-trips
through the saved config, and reaches the schedule, checkpoint and exposure
records of a real short run.
"""

# Failure injection inspects the private host coordinator, not a public API.
# pyright: reportPrivateUsage=false

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from marl_battlegrounds.training.runner import (
    TrainConfig,
    config_from_dict,
    config_to_dict,
)


@pytest.mark.parametrize(
    ("recording", "input_scale"), [(False, 1.0), (True, 1.0), (False, 0.01)]
)
def test_public_run_resume_and_final_pending_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recording: bool,
    input_scale: float,
) -> None:
    from marl_battlegrounds.baselines.ppo import PPOConfig
    from marl_battlegrounds.training import checkpoints, runner

    # Hold source identity fixed while other focused development checks edit files.
    # Separate checkpoint tests assert source mismatch rejection before recovery.
    identity = checkpoints.runtime_identity()
    monkeypatch.setattr(checkpoints, "runtime_identity", lambda: identity)

    def forbidden_estimate(*args: object, **kwargs: object) -> None:
        pytest.fail("Quiet training must skip optional ETA work")

    monkeypatch.setattr(runner, "TrainingSpeedEstimate", forbidden_estimate)
    monkeypatch.setattr(runner._Run, "pending_seconds", forbidden_estimate)
    config = TrainConfig(
        num_envs=4,
        total_env_steps=12,
        seed=710,
        ppo=PPOConfig(rollout_length=2, epochs=1, input_scale=input_scale),
        checkpoint_interval_updates=1,
        metrics="none",
        recording=recording,
        verbose=False,
    )
    baseline = runner.train(config, output_dir=tmp_path / "baseline")
    from marl_battlegrounds.training._compilation import execution_identity

    details = json.loads((baseline.run_dir / "run_details.json").read_text())
    assert details["execution"] == execution_identity()
    assert baseline.completed_env_steps == 12
    assert baseline.completed_updates == 2
    assert baseline.selected_actor is None
    assert all(path.is_file() for path in baseline.evidence_paths)
    original_report = runner._Run.report

    def interrupt_report(self: runner._Run) -> tuple[Path, ...]:
        raise KeyboardInterrupt("Declared interruption after final training")

    monkeypatch.setattr(runner._Run, "report", interrupt_report)
    destination = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        runner.train(config, output_dir=destination)
    assert "Status: failed" in (destination / "run_summary.md").read_text()
    (destination / "exposure.json").write_text('{"abandoned": true}')
    (destination / "slot_diagnostic.json").write_text('{"abandoned": true}')
    before_rows = (destination / "training_updates.jsonl").read_bytes()
    latest = json.loads((destination / "latest_checkpoint.json").read_text())
    checkpoint = destination / latest["relative_path"]
    monkeypatch.setattr(runner._Run, "report", original_report)
    saved_bytes = {
        path: path.read_bytes() for path in destination.rglob("*") if path.is_file()
    }

    def forbidden_restore(*args: object, **kwargs: object) -> None:
        pytest.fail("Changed execution must reject before numerical restore")

    with monkeypatch.context() as changed:
        changed.setenv("XLA_FLAGS", "--xla_gpu_autotune_level=0")
        changed.setattr(checkpoints, "_restore_arrays", forbidden_restore)
        with pytest.raises(ValueError, match="execution"):
            runner.train(resume_from=checkpoint)
    assert saved_bytes == {
        path: path.read_bytes() for path in destination.rglob("*") if path.is_file()
    }
    assert not (destination / "checkpoint_recovery.json").exists()
    if input_scale != 1.0:
        import hashlib

        saved = checkpoints.read_checkpoint_details(checkpoint)
        ancestor_path = Path(
            next(iter(saved["metadata"]["host_state"]["actors"].values()))
        )
        description = ancestor_path / "actor_details.json"
        original_description = description.read_bytes()
        changed_actor = json.loads(original_description)
        changed_actor["input_scale"] = 1.0
        del changed_actor["checkpoint_id"]
        changed_actor["checkpoint_id"] = hashlib.sha256(
            checkpoints._json_bytes(changed_actor)
        ).hexdigest()
        description.write_text(json.dumps(changed_actor))
        before_mismatch = {
            path: path.read_bytes() for path in destination.rglob("*") if path.is_file()
        }
        with pytest.raises(ValueError, match="actor identity"):
            runner.train(resume_from=checkpoint)
        assert before_mismatch == {
            path: path.read_bytes() for path in destination.rglob("*") if path.is_file()
        }
        assert not (destination / "checkpoint_recovery.json").exists()
        description.write_bytes(original_description)
    if recording:
        restore_cursor = runner.restore_log_cursor

        def interrupt_recovery(path: Path, cursor: dict[str, Any]) -> None:
            raise OSError("Declared failure after M8 recording recovery")

        monkeypatch.setattr(runner, "restore_log_cursor", interrupt_recovery)
        with pytest.raises(OSError, match="after M8"):
            runner.train(resume_from=checkpoint)
        marker = json.loads((destination / "checkpoint_recovery.json").read_text())
        assert marker["checkpoint_id"] == checkpoint.name
        assert (destination / "training_updates.jsonl").read_bytes() == before_rows
        monkeypatch.setattr(runner, "restore_log_cursor", restore_cursor)
        finish_recovery = checkpoints.finish_checkpoint_recovery

        def interrupt_clear(
            restored: checkpoints.RestoredCheckpoint, run_dir: Path
        ) -> None:
            status = json.loads((run_dir / "status.json").read_text())
            assert status["status"] == "incomplete"
            assert status["phase"] == "resuming"
            raise OSError("Declared failure before clearing recovery")

        monkeypatch.setattr(checkpoints, "finish_checkpoint_recovery", interrupt_clear)
        with pytest.raises(OSError, match="before clearing"):
            runner.train(resume_from=checkpoint)
        assert (destination / "checkpoint_recovery.json").is_file()
        monkeypatch.setattr(checkpoints, "finish_checkpoint_recovery", finish_recovery)
    restored = runner.train(resume_from=checkpoint)
    assert restored.completed_env_steps == 12
    assert restored.completed_updates == 2
    assert (destination / "training_updates.jsonl").read_bytes() == before_rows
    assert (
        checkpoints.artifact_identity(restored.final_actor)["actor_digest"]
        == checkpoints.artifact_identity(baseline.final_actor)["actor_digest"]
    )
    assert (
        checkpoints.artifact_identity(restored.final_actor)["input_scale"]
        == input_scale
    )
    assert json.loads((destination / "status.json").read_text())["status"] == "complete"
    assert not (destination / "checkpoint_recovery.json").exists()
    assert json.loads((destination / "slot_diagnostic.json").read_text()) is None
    assert "abandoned" not in json.loads((destination / "exposure.json").read_text())
    archived = list((destination / "attempts").rglob("slot_diagnostic-*.json"))
    assert any(path.read_text() == '{"abandoned": true}' for path in archived)


def test_panel_backed_public_run_resume_selection_and_validation_isolation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from importlib import import_module

    import jax

    from marl_battlegrounds import training
    from marl_battlegrounds.baselines.ppo import PPOConfig, initialize_ppo
    from marl_battlegrounds.evaluation.recording_context import (
        capture_recording_provenance,
    )
    from marl_battlegrounds.evaluation.recording_identity import tree_digest
    from marl_battlegrounds.evaluation.results import load_results
    from marl_battlegrounds.training import checkpoints, runner, validation

    identity = checkpoints.runtime_identity()
    monkeypatch.setattr(checkpoints, "runtime_identity", lambda: identity)
    evaluator = import_module("marl_battlegrounds.evaluation.evaluate")
    provenance = capture_recording_provenance()

    def fixed_provenance(**_: object) -> dict[str, object]:
        return provenance

    monkeypatch.setattr(evaluator, "capture_recording_provenance", fixed_provenance)
    original_evaluate = evaluator.evaluate

    def short_horizon(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        return original_evaluate(*args, **kwargs, max_steps=1)

    # Only the fixture's game horizon changes. M8 still owns actual games,
    # saved outcomes and their scores; production selection stays untouched.
    monkeypatch.setattr(evaluator, "evaluate", short_horizon)
    weights = initialize_ppo(jax.random.key(88)).actor_params
    actors = tuple(
        checkpoints.export_system(
            weights,
            tmp_path / f"panel-actor-{index}",
            metadata={
                "run_id": "untrained-test-panel",
                "seed": 88,
                "env_steps": steps,
                "checkpoint_id": str(index + 1) * 64,
            },
        )
        for index, steps in enumerate((4, 8))
    )
    panel = validation.create_panel(
        *actors, output_dir=tmp_path / "panel", test_only=True
    )
    assert not panel.qualified
    frozen_identities = [checkpoints.artifact_identity(actor) for actor in actors]
    config = TrainConfig(
        num_envs=4,
        total_env_steps=12,
        seed=711,
        ppo=PPOConfig(rollout_length=2, epochs=1),
        checkpoint_interval_updates=1,
        validation_fractions=(1.0,),
        routine_seed_pairs=1,
        confirmation_seed_pairs=1,
        metrics="none",
        verbose=False,
    )
    baseline = training.train(config, output_dir=tmp_path / "without-panel")
    details = json.loads((baseline.run_dir / "run_details.json").read_text())
    assert details["schemas"] == checkpoints.checkpoint_schemas()
    from marl_battlegrounds.training._compilation import execution_identity

    assert details["execution"] == execution_identity()
    assert details["runtime"]["provenance"]["backend"] == "cpu"
    assert details["runtime"]["provenance"]["batch_shape"] == [4]
    assert details["runtime"]["provenance"]["device"]
    assert details["runtime"]["provenance"]["precision"] in ("float32", "float64")
    assert "JAX_PLATFORMS" in details["runtime"]["declared_environment"]
    original_report = runner._Run.report

    def interrupt_after_final_routine(self: runner._Run) -> tuple[Path, ...]:
        if self.host["env_steps"] == 12:
            assert [row["env_steps"] for row in self.host["routine_results"]] == [0, 12]
            assert not self.host["confirmation_results"]
            raise KeyboardInterrupt("Declared report failure after final routine")
        return original_report(self)

    monkeypatch.setattr(runner._Run, "report", interrupt_after_final_routine)
    destination = tmp_path / "with-panel"
    with pytest.raises(KeyboardInterrupt, match="after final routine"):
        training.train(
            replace(config, validation_panel=str(panel.path)), output_dir=destination
        )
    interrupted_results = json.loads(
        (destination / "validation_results.json").read_text()
    )
    assert len(interrupted_results) == 2
    assert all(row["complete"] and row["games"] == 20 for row in interrupted_results)
    previous_games = {
        path: path.read_bytes() for path in (destination / "validation").rglob("*.csv")
    }
    assert previous_games
    updates_before = (destination / "training_updates.jsonl").read_bytes()
    pointer = json.loads((destination / "latest_checkpoint.json").read_text())
    checkpoint = destination / pointer["relative_path"]
    monkeypatch.setattr(runner._Run, "report", original_report)
    restored = training.train(resume_from=checkpoint)

    assert restored.completed_env_steps == baseline.completed_env_steps == 12
    assert restored.completed_updates == baseline.completed_updates == 2
    assert (destination / "training_updates.jsonl").read_bytes() == updates_before
    assert all(path.read_bytes() == raw for path, raw in previous_games.items())
    assert [
        checkpoints.artifact_identity(actor) for actor in actors
    ] == frozen_identities
    restored_identity = checkpoints.artifact_identity(restored.final_actor)
    assert (
        restored_identity["actor_digest"]
        == checkpoints.artifact_identity(baseline.final_actor)["actor_digest"]
    )
    baseline_rows = [
        json.loads(line)
        for line in (baseline.run_dir / "training_updates.jsonl")
        .read_text()
        .splitlines()
    ]
    restored_rows = [json.loads(line) for line in updates_before.splitlines()]
    for name in (
        "env_steps",
        "update_index",
        "actor_decisions",
        "used_policy_samples",
        "used_value_samples",
    ):
        assert [row[name] for row in restored_rows] == [
            row[name] for row in baseline_rows
        ]

    results = json.loads((destination / "validation_results.json").read_text())
    assert [(row["purpose"], row["env_steps"]) for row in results] == [
        ("routine", 0),
        ("routine", 12),
        ("confirmation", 12),
    ]
    assert len({row["task_id"] for row in results}) == 3
    assert all(row["complete"] and row["games"] == 20 for row in results)
    played_games = 0
    pass_paths: set[str] = set()
    for result in results:
        assert result["panel_digest"] == panel.digest
        assert result["score"] == 0.5
        for path in result["pass_paths"]:
            assert path not in pass_paths
            pass_paths.add(path)
            saved = load_results(path, phase="validation")
            assert saved.status == "complete"
            episode_ids = [
                int(value)
                for batch in saved.iter_table("episodes")
                for value in batch["episode_id"]
            ]
            assert len(episode_ids) == len(set(episode_ids)) == 10
            played_games += len(episode_ids)
    assert len(pass_paths) == 6 and played_games == 60
    status = json.loads((destination / "status.json").read_text())
    assert set(status["memory"]) == {
        "process_peak_ram_bytes",
        "jax_allocator",
        "errors",
        "scope",
    }
    events = [
        json.loads(line)
        for line in (destination / "run_events.jsonl").read_text().splitlines()
    ]
    attempts = [row for row in events if row["event"] == "attempt_started"]
    assert len(attempts) == 2
    assert all(row["runtime"]["provenance"]["backend"] == "cpu" for row in attempts)
    terminal = [
        row for row in events if row["event"] in ("attempt_failed", "attempt_complete")
    ]
    assert [row["event"] for row in terminal] == ["attempt_failed", "attempt_complete"]
    assert all(set(row["memory"]) == set(status["memory"]) for row in terminal)
    assert terminal[-1]["memory"] == status["memory"]
    assert restored.selected_actor is not None
    assert restored.selected_actor == restored.final_actor
    selection = json.loads((destination / "selection.json").read_text())
    assert selection["purpose"] == "confirmation"
    assert selection["checkpoint_id"] == restored.final_actor.name
    assert selection["actor_digest"] == restored_identity["actor_digest"]
    loaded = training.load_system(restored.selected_actor)
    assert tree_digest(loaded.variables) == restored_identity["actor_digest"]
    assert all(
        path.is_file() and path.stat().st_size for path in restored.evidence_paths
    )
    report = training.analyze([destination], output_dir=tmp_path / "review")
    assert report["complete"]
    assert report["runs"][0]["validation_checkpoints"] == 3
    assert "Status: complete" in Path(report["artifacts"]["summary"]).read_text()
    assert (
        "not held-out test results" in Path(report["artifacts"]["summary"]).read_text()
    )


@pytest.mark.parametrize(
    "change",
    [
        {"num_envs": True},
        {"num_envs": 2},
        {"total_env_steps": 33},
        {"seed": False},
        {"method": "qmix"},
        {"metrics": "full"},
        {"purpose": "demonstration"},
        {"validation_fractions": (0.5, 0.4, 1.0)},
        {"checkpoint_env_steps": (32,)},
        {"shaping_coefficient": float("nan")},
        {"slot_diagnostic": True},
        {"recording": 1},
        {"pinned_opponent_share": 0.9},
        {"pinned_opponent_share": True},
        {"pinned_opponent_share": float("nan")},
    ],
)
def test_invalid_config_rejected(change: dict[str, Any]) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(TrainConfig(), **change)


def test_pinned_share_flows_through_the_public_run_and_its_saved_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marl_battlegrounds.baselines.ppo import PPOConfig
    from marl_battlegrounds.training import checkpoints, runner

    identity = checkpoints.runtime_identity()
    monkeypatch.setattr(checkpoints, "runtime_identity", lambda: identity)
    config = TrainConfig(
        num_envs=4,
        total_env_steps=16,
        seed=711,
        ppo=PPOConfig(rollout_length=2, epochs=1),
        checkpoint_interval_updates=1,
        metrics="none",
        verbose=False,
        pinned_opponent_share=0.1,
    )
    result = runner.train(config, output_dir=tmp_path / "pinned")
    assert result.completed_updates == 2
    details = json.loads((result.run_dir / "run_details.json").read_text())
    assert details["config"]["pinned_opponent_share"] == 0.1
    assert details["schedule"]["history_thresholds"] == (
        "first update, then 5% through 95%"
    )
    latest = json.loads((result.run_dir / "latest_checkpoint.json").read_text())
    saved = checkpoints.read_checkpoint_details(
        result.run_dir / latest["relative_path"]
    )
    assert saved["collection"]["pinned_opponent_share"] == 0.1
    assert saved["metadata"]["config"]["pinned_opponent_share"] == 0.1
    exposure = json.loads((result.run_dir / "exposure.json").read_text())
    assert len(exposure["steps_by_opponent"]) == 21
    assert len(exposure["starts_by_opponent"]) == 21


def test_config_roundtrip_and_unknown_keys() -> None:
    config = TrainConfig(seed=19041901, checkpoint_env_steps=(524288, 1048576))
    assert config_from_dict(config_to_dict(config)) == config
    pinned = TrainConfig(pinned_opponent_share=0.1)
    assert config_from_dict(config_to_dict(pinned)) == pinned
    assert config_to_dict(pinned)["pinned_opponent_share"] == 0.1
    older = {
        key: value
        for key, value in config_to_dict(config).items()
        if key != "pinned_opponent_share"
    }
    assert config_from_dict(older) == config
    with pytest.raises(ValueError, match="Unknown"):
        config_from_dict({"typo": 1})
    with pytest.raises(ValueError, match="schema_version"):
        config_from_dict({"schema_version": 2})


def test_retention_preserves_candidates_and_small_ancestry(tmp_path: Path) -> None:
    from marl_battlegrounds.training import runner

    records = {letter: tmp_path / "checkpoints" / (letter * 64) for letter in "abcd"}
    for directory in records.values():
        (directory / "state").mkdir(parents=True)
        (directory / "actor").mkdir()
        (directory / "checkpoint_details.json").write_text("{}")
    execution = object.__new__(runner._Run)
    execution.root = tmp_path
    execution.metadata = {"attempt_id": "test"}
    execution.host = {
        "actors": {"a" * 64: "actor-a"},
        "recovery_checkpoints": ["a" * 64, "b" * 64],
        "env_steps": 0,
    }
    execution.checkpoint = records["c"]
    execution.prune_recovery()
    execution.checkpoint = records["d"]
    execution.prune_recovery()
    assert (records["a"] / "state").is_dir()
    assert not (records["b"] / "state").exists()
    assert (records["b"] / "checkpoint_details.json").is_file()
    assert all((records[name] / "state").is_dir() for name in "cd")


def test_slot_task_keeps_separate_continuation_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marl_battlegrounds.training import runner, validation

    member = validation.PanelMember(
        "Final", tmp_path / "development-final", "digest", "checkpoint", "run", 1, 8
    )
    execution = object.__new__(runner._Run)
    execution.root = tmp_path
    execution.panel = validation.FrozenPanel(
        tmp_path / "panel.json", "panel", (member,), True
    )
    execution.host = {"env_steps": 8, "final_actor": ""}
    execution.metadata = {"attempt_id": "test"}
    destinations: list[Path] = []

    def diagnostic(actor: Path, opponent: Path, **kwargs: object) -> dict[str, Any]:
        assert opponent == member.path
        destination = kwargs["output_dir"]
        assert isinstance(destination, Path)
        destinations.append(destination)
        return {"actor": str(actor), "complete": True}

    def ignore_status(self: runner._Run, phase: str, **facts: object) -> None:
        pass

    monkeypatch.setattr(validation, "run_slot_diagnostic", diagnostic)
    monkeypatch.setattr(runner._Run, "set_status", ignore_status)
    for identity in ("old-final", "resumed-final"):
        execution.host["final_actor"] = str(tmp_path / "actors" / identity)
        execution.run_slot_check()
    assert destinations == [
        tmp_path / "slot_diagnostic" / "old-final",
        tmp_path / "slot_diagnostic" / "resumed-final",
    ]
    selected = json.loads((tmp_path / "slot_diagnostic.json").read_text())["actor"]
    assert selected.endswith("resumed-final")


def test_report_archive_preserves_raw_bytes_and_is_idempotent(tmp_path: Path) -> None:
    from marl_battlegrounds.training.runner import _preserve_reports

    report = tmp_path / "slot_diagnostic.json"
    report.write_bytes(b'{"interrupted":')
    _preserve_reports(tmp_path, "old-attempt")
    _preserve_reports(tmp_path, "old-attempt")
    archived = list((tmp_path / "attempts" / "old-attempt" / "reports").iterdir())
    assert len(archived) == 1
    assert archived[0].read_bytes() == report.read_bytes()
