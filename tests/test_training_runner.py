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
records of a real short run. A named pinned_opponent round-trips too, an older
config without the key reads as None, and a name without a positive share or
an empty or non-string name is rejected.
The spawn frame rides the same end-to-end case: a "left" run exports its frame,
a tampered export is rejected as a different identity, the resumed run reports
the frame, and the saved config round-trips with and without the nested key; a
config without the key takes the constructor default "left" for a new run, while
saved_training_config still reads it as "world".
All four PPO methods use the same recording recovery checks. Method settings
round-trip through JSON and keep their recurrent or feedforward batch rules.
Changing an exported model tag while preserving its actor weights is rejected
before recording or log recovery changes the saved files.
QMIX settings replace PPO settings: a QMIX config resolves qmix and a 1600-step
save interval (PPO keeps 25), saves no ppo block while PPO saves no qmix block,
round-trips through JSON, and rejects nondefault PPO settings, QMIX settings
under PPO, a budget below the replay minimum, an exploration clock that would
overflow int32 and a malformed block. A save
happens when a block's optimizer steps cross a multiple of the interval.
PQN-VDN settings replace PPO settings the same way (a pqn block, a 1600-step
interval, JSON round trip, and refusal of other methods' blocks, a budget
without a learning block and a batch its minibatches do not divide); its
extra save points must lie on its offset boundaries (8,320 transitions is
accepted, 8,192 is not, while PPO and QMIX keep their fixed-block points).
A rejected PQN-VDN block writes one learner_update_rejected event whose
finite values are kept and whose NaN and infinite values are null with a
named marker, then attempt_failed with the same reason, and stops without a
second update or any change to the host counts.
red_zone_depth defaults to 5.0 (also from an empty JSON config), round-trips at
0.0, 6.0 and 20.0 in config version 1, reads a JSON integer 6 as 6.0, and
refuses negative, -0.0, NaN, infinite, subnormal, integer and Boolean values;
a saved config without it reads as 0.0 while a new run from such a file takes
5.0. A depth wider than a map is refused before any file, even a validation
panel, is written. A panel-backed run at 6.0 records the depth in its config,
content binding, validation tasks (schema 4 for a System panel, 3 for a
historical panel) and every saved game configuration. On resume, a config
declaring another depth, and a checkpoint re-signed with the pre-Red-Zone
schemas (with or without a config), are refused before any file or array is
touched and leave no recovery marker.
Live callbacks receive saved finite rows once per block and isolate mutation.
Children forward them; failed callbacks keep the row and allow normal recovery.
Partner-only matchmaking leaves legacy opponents alone. Matchmaking receives
copies of member records, so editing callback input cannot change saved identity.
"""

# Failure injection inspects the private host coordinator, not a public API.
# pyright: reportPrivateUsage=false

import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import pytest

from marl_battlegrounds.training.runner import (
    TrainConfig,
    config_from_dict,
    config_to_dict,
)


def _schema_1_copy(checkpoint: Path) -> Path:
    # Copy a learner checkpoint with the pre-Red-Zone schemas, re-signed.
    import hashlib

    from marl_battlegrounds.training import checkpoints

    details = checkpoints.read_checkpoint_details(checkpoint)
    method = details["metadata"]["config"]["method"]
    details["schemas"] = dict(checkpoints._ACTOR_INPUT_1_SCHEMAS[method])
    del details["checkpoint_id"]
    details["checkpoint_id"] = hashlib.sha256(
        checkpoints._json_bytes(details)
    ).hexdigest()
    historical = checkpoint.with_name(details["checkpoint_id"])
    shutil.copytree(checkpoint, historical)
    (historical / "checkpoint_details.json").write_text(json.dumps(details))
    return historical


@pytest.mark.parametrize(
    ("method", "recording", "input_scale", "spawn_frame"),
    [
        ("mappo", False, 1.0, "world"),
        ("mappo", True, 1.0, "world"),
        ("mappo", False, 0.01, "left"),
        ("ippo", True, 1.0, "world"),
        ("ff_mappo", True, 1.0, "world"),
        ("ff_ippo", True, 1.0, "world"),
    ],
)
def test_public_run_resume_and_final_pending_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: Literal["mappo", "ippo", "ff_mappo", "ff_ippo"],
    recording: bool,
    input_scale: float,
    spawn_frame: str,
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
        keep_past=0,
        method=method,
        num_envs=4,
        total_env_steps=12,
        seed=710,
        ppo=PPOConfig(
            rollout_length=2, epochs=1, input_scale=input_scale, spawn_frame=spawn_frame
        ),
        checkpoint_interval_updates=1,
        metrics="none",
        recording=recording,
        verbose=False,
    )
    baseline = runner.train(config, output_dir=tmp_path / "baseline")
    from marl_battlegrounds.training._compilation import execution_identity

    details = json.loads((baseline.run_dir / "run_details.json").read_text())
    assert details["execution"] == execution_identity()
    assert details["config"]["method"] == method
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
    if (method, recording, input_scale) == ("mappo", False, 1.0):
        # Red Zone resume refusals happen before any file or array is touched.
        with monkeypatch.context() as refused:
            refused.setattr(checkpoints, "_restore_arrays", forbidden_restore)
            with pytest.raises(
                ValueError,
                match=r"declares red_zone_depth 6\.0, but the saved run uses 5\.0",
            ):
                runner.train(
                    replace(config, red_zone_depth=6.0), resume_from=checkpoint
                )
            assert saved_bytes == {
                path: path.read_bytes()
                for path in destination.rglob("*")
                if path.is_file()
            }
            historical = _schema_1_copy(checkpoint)
            forged = {
                path: path.read_bytes()
                for path in destination.rglob("*")
                if path.is_file()
            }
            for supplied in (None, config):
                with pytest.raises(ValueError, match="saved before Red Zone"):
                    runner.train(supplied, resume_from=historical)
            assert forged == {
                path: path.read_bytes()
                for path in destination.rglob("*")
                if path.is_file()
            }
            shutil.rmtree(historical)
        assert not (destination / "checkpoint_recovery.json").exists()
    if input_scale != 1.0 or method != "mappo":
        import hashlib

        saved = checkpoints.read_checkpoint_details(checkpoint)
        ancestor_path = Path(
            next(iter(saved["metadata"]["host_state"]["actors"].values()))
        )
        description = ancestor_path / "actor_details.json"
        original_description = description.read_bytes()
        saved_actor = json.loads(original_description)
        assert ("spawn_frame" in saved_actor) == (spawn_frame != "world")
        tampered = [{**saved_actor, "input_scale": 1.0}] if input_scale != 1.0 else []
        if spawn_frame != "world":
            tampered.append(
                {
                    key: value
                    for key, value in saved_actor.items()
                    if key != "spawn_frame"
                }
            )
        if method != "mappo":
            other_method = {
                "ippo": "mappo",
                "ff_mappo": "ff_ippo",
                "ff_ippo": "ff_mappo",
            }[method]
            tampered.append(
                {**saved_actor, "schemas": checkpoints.checkpoint_schemas(other_method)}
            )
        for changed_actor in tampered:
            del changed_actor["checkpoint_id"]
            changed_actor["checkpoint_id"] = hashlib.sha256(
                checkpoints._json_bytes(changed_actor)
            ).hexdigest()
            description.write_text(json.dumps(changed_actor))
            before_mismatch = {
                path: path.read_bytes()
                for path in destination.rglob("*")
                if path.is_file()
            }
            with pytest.raises(ValueError, match="actor identity"):
                runner.train(resume_from=checkpoint)
            assert before_mismatch == {
                path: path.read_bytes()
                for path in destination.rglob("*")
                if path.is_file()
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
        assert (
            marker["checkpoint_id"]
            == checkpoints.read_checkpoint_description(checkpoint)["checkpoint_id"]
        )
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
    assert (
        checkpoints.artifact_identity(restored.final_actor)["spawn_frame"]
        == spawn_frame
    )
    assert json.loads((destination / "status.json").read_text())["status"] == "complete"
    assert not (destination / "checkpoint_recovery.json").exists()
    assert json.loads((destination / "slot_diagnostic.json").read_text()) is None
    assert "abandoned" not in json.loads((destination / "exposure.json").read_text())
    archived = list((destination / "attempts").rglob("slot_diagnostic-*.json"))
    assert any(path.read_text() == '{"abandoned": true}' for path in archived)


@pytest.mark.parametrize("system_panel", (False, True))
def test_panel_backed_public_run_resume_selection_and_validation_isolation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    system_panel: bool,
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
    original_verify = evaluator._verify_evaluation

    def short_horizon(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        return original_evaluate(*args, **kwargs, max_steps=1)

    # Only the fixture's game horizon changes. M8 still owns actual games,
    # saved outcomes and their scores; production selection stays untouched.
    monkeypatch.setattr(evaluator, "evaluate", short_horizon)

    def short_verify(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        return original_verify(*args, **kwargs, max_steps=1)

    monkeypatch.setattr(evaluator, "_verify_evaluation", short_verify)
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
            spawn_frame="world",
        )
        for index, steps in enumerate((4, 8))
    )
    panel = (
        validation.create_panel(
            opponents=("tdm-alpha", "tdm-beta"), output_dir=tmp_path / "panel"
        )
        if system_panel
        else validation.create_panel(
            *actors, output_dir=tmp_path / "panel", test_only=True
        )
    )
    assert panel.qualified == system_panel
    frozen_identities = [checkpoints.artifact_identity(actor) for actor in actors]
    config = TrainConfig(
        keep_past=0,
        num_envs=4,
        total_env_steps=12,
        seed=711,
        # World frame: the selected export's identity is then its raw weight digest.
        ppo=PPOConfig(rollout_length=2, epochs=1, spawn_frame="world"),
        checkpoint_interval_updates=1,
        validation_fractions=(1.0,),
        routine_seed_pairs=1,
        confirmation_seed_pairs=1,
        metrics="none",
        verbose=False,
        red_zone_depth=6.0,
    )
    baseline = training.train(config, output_dir=tmp_path / "without-panel")
    details = json.loads((baseline.run_dir / "run_details.json").read_text())
    assert details["schemas"] == checkpoints.checkpoint_schemas()
    # The run's depth reaches its saved config and its content binding.
    assert details["config"]["red_zone_depth"] == 6.0
    assert details["content_binding"]["red_zone_depth"] == 6.0
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
        # Validation tasks record the run's depth: schema 4 (System panel) or 3.
        assert result["red_zone_depth"] == 6.0
        assert result["schema_version"] == (4 if system_panel else 3)
        for path in result["pass_paths"]:
            assert path not in pass_paths
            pass_paths.add(path)
            saved = load_results(path, phase="validation")
            assert saved.status == "complete"
            assert {
                content["team_deathmatch_red_zone_depth"]
                for content in saved.metadata["configurations"].values()
            } == {6.0}
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
    assert (
        selection["checkpoint_id"]
        == checkpoints.artifact_identity(restored.final_actor)["metadata"][
            "checkpoint_id"
        ]
    )
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
        {"method": "qdn"},
        {"metrics": "full"},
        {"validation_fractions": (0.5, 0.4, 1.0)},
        {"checkpoint_env_steps": (32,)},
        {"shaping_coefficient": float("nan")},
        {"slot_diagnostic": True},
        {"recording": 1},
        {"pinned_opponent_share": 0.9},
        {"pinned_opponent_share": True},
        {"pinned_opponent_share": float("nan")},
        {"pinned_opponent": "tdm-alpha"},
        {"pinned_opponent": "", "pinned_opponent_share": 0.1},
        *(
            {"pinned_opponent": value, "pinned_opponent_share": 0.1}
            for value in ("actor", ".", "..", "relative/actor")
        ),
        {"pinned_opponent": 3, "pinned_opponent_share": 0.1},
        *(
            {"red_zone_depth": value}
            for value in (-1.0, -0.0, float("nan"), float("inf"), 1e-40, 5, True)
        ),
    ],
)
def test_invalid_config_rejected(change: dict[str, Any]) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(TrainConfig(), **change)


def test_qmix_settings_replace_ppo_and_resolve_their_own_save_interval() -> None:
    from marl_battlegrounds.baselines.ppo import PPOConfig
    from marl_battlegrounds.baselines.qmix import DEFAULT_QMIX_CONFIG, QMIXConfig
    from marl_battlegrounds.training.runner import _crosses_interval

    ppo_config = TrainConfig()
    assert ppo_config.qmix is None and ppo_config.checkpoint_interval_updates == 25
    ppo_saved = config_to_dict(ppo_config)
    assert "qmix" not in ppo_saved and ppo_saved["checkpoint_interval_updates"] == 25
    qmix_config = TrainConfig(method="qmix")
    assert qmix_config.qmix == DEFAULT_QMIX_CONFIG
    assert qmix_config.checkpoint_interval_updates == 1600
    saved = config_to_dict(qmix_config)
    assert "ppo" not in saved and saved["qmix"]["buffer_size"] == 1000
    assert config_from_dict(json.loads(json.dumps(saved))) == qmix_config
    explicit = TrainConfig(method="qmix", checkpoint_interval_updates=7)
    assert explicit.checkpoint_interval_updates == 7
    for bad in (
        {"method": "qmix", "ppo": PPOConfig(rollout_length=8)},
        {"qmix": QMIXConfig()},
        {"method": "qmix", "num_envs": 2, "total_env_steps": 62},
        {"method": "qmix", "qmix": "default"},
    ):
        with pytest.raises((TypeError, ValueError)):
            TrainConfig(**bad)  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="exploration clock"):
        TrainConfig(method="qmix", num_envs=32, qmix=QMIXConfig(eps_decay=2**31 - 1))
    with pytest.raises(ValueError, match="ppo"):
        config_from_dict({**saved, "ppo": {}})
    with pytest.raises(ValueError, match="qmix"):
        config_from_dict({**ppo_saved, "qmix": {}})
    assert _crosses_interval(24, 28, 25) and not _crosses_interval(20, 24, 25)
    assert [_crosses_interval(n - 1, n, 25) for n in (24, 25, 26, 50)] == [
        False,
        True,
        False,
        True,
    ]


def test_pqn_settings_replace_ppo_and_follow_offset_capture_points() -> None:
    from marl_battlegrounds.baselines.ppo import PPOConfig
    from marl_battlegrounds.baselines.pqn import DEFAULT_PQN_CONFIG, PQNConfig
    from marl_battlegrounds.baselines.qmix import QMIXConfig

    config = TrainConfig(method="pqn_vdn")
    assert config.pqn == DEFAULT_PQN_CONFIG
    assert config.checkpoint_interval_updates == 1600
    saved = config_to_dict(config)
    assert "ppo" not in saved and "qmix" not in saved
    assert saved["pqn"]["memory_window"] == 4
    assert config_from_dict(json.loads(json.dumps(saved))) == config
    assert "pqn" not in config_to_dict(TrainConfig())
    assert "pqn" not in config_to_dict(TrainConfig(method="qmix"))
    explicit = TrainConfig(method="pqn_vdn", checkpoint_interval_updates=7)
    assert explicit.checkpoint_interval_updates == 7
    offsets = TrainConfig(method="pqn_vdn", checkpoint_env_steps=(4224, 8320))
    assert offsets.checkpoint_env_steps == (4224, 8320)
    for steps in ((8192,), (12288,)):
        with pytest.raises(ValueError, match="update boundaries"):
            TrainConfig(method="pqn_vdn", checkpoint_env_steps=steps)
        assert TrainConfig(checkpoint_env_steps=steps).checkpoint_env_steps == steps
        qmix = TrainConfig(method="qmix", checkpoint_env_steps=steps)
        assert qmix.checkpoint_env_steps == steps
    for bad, match in (
        ({"method": "pqn_vdn", "ppo": PPOConfig(rollout_length=8)}, "no PPO"),
        ({"method": "pqn_vdn", "qmix": QMIXConfig()}, "no QMIX"),
        ({"pqn": PQNConfig()}, "PPO run takes no PQN"),
        ({"method": "qmix", "pqn": PQNConfig()}, "QMIX run takes no PQN"),
        (
            {"method": "pqn_vdn", "num_envs": 32, "total_env_steps": 32 * 132},
            "at least",
        ),
        ({"method": "pqn_vdn", "num_envs": 18, "total_env_steps": 18_000}, "divide"),
        ({"method": "pqn_vdn", "pqn": "default"}, "PQNConfig"),
    ):
        with pytest.raises((TypeError, ValueError), match=match):
            TrainConfig(**bad)  # pyright: ignore[reportArgumentType]
    for block in ("ppo", "qmix"):
        with pytest.raises(ValueError, match=block):
            config_from_dict({**saved, block: {}})
    with pytest.raises(ValueError, match="pqn"):
        config_from_dict({**config_to_dict(TrainConfig()), "pqn": {}})


def test_a_rejected_pqn_block_records_its_diagnostics_and_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import jax.numpy as jnp

    from marl_battlegrounds import training
    from marl_battlegrounds.baselines.pqn import PQNConfig
    from marl_battlegrounds.training import pqn_learner, runner

    original = pqn_learner.update_pqn_learner

    def rejected(state: Any, collected: Any, rollout: Any, **settings: Any) -> Any:  # noqa: ANN401
        _, result = original(state, collected, rollout, **settings)
        summary = result.summary._replace(
            task_reward_sum=jnp.float32(jnp.nan),
            shaping_reward_sum=jnp.float32(jnp.inf),
        )
        metrics = result.metrics._replace(
            loss=jnp.asarray([[-jnp.inf, 1.5]], jnp.float32)
        )
        failed = state._replace(failed=jnp.bool_(True), failure_reason=jnp.int32(5))
        return failed, result._replace(
            accepted=jnp.bool_(False),
            failed=jnp.bool_(True),
            failure_reason=jnp.int32(5),
            summary=summary,
            metrics=metrics,
        )

    calls: list[int] = []
    original_init = runner._Run.__init__

    def counting_init(execution: runner._Run, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        original_init(execution, *args, **kwargs)
        updater = execution.updater

        def counted(*values: Any) -> Any:  # noqa: ANN401
            calls.append(1)
            return updater(*values)

        execution.updater = counted

    monkeypatch.setattr(pqn_learner, "update_pqn_learner", rejected)
    monkeypatch.setattr(runner._Run, "__init__", counting_init)
    config = TrainConfig(
        keep_past=0,
        method="pqn_vdn",
        seed=19049151,
        num_envs=4,
        total_env_steps=64,
        pqn=PQNConfig(rollout_length=4, memory_window=2, epochs=1, num_minibatches=2),
        metrics="none",
        verbose=False,
    )
    root = tmp_path / "run"
    with pytest.raises(RuntimeError, match="reason 5"):
        training.train(config, output_dir=root)
    assert calls == [1]
    events = [json.loads(line) for line in (root / "run_events.jsonl").open()]
    names = [event["event"] for event in events]
    position = names.index("learner_update_rejected")
    assert names[position + 1] == "attempt_failed"
    record = events[position]
    assert record["reason"] == 5 and record["generated_transitions"] == 16
    assert record["task_reward_sum"] is None and record["shaping_reward_sum"] is None
    assert record["losses"] == [[None, 1.5]]
    assert record["nonfinite"] == {
        "task_reward_sum": "NaN",
        "shaping_reward_sum": "+Inf",
        "losses[0,0]": "-Inf",
    }
    assert record["steps_performed"] == [[False, False]]
    assert "reason 5" in events[position + 1]["error"]
    status = json.loads((root / "status.json").read_text())
    assert status["env_steps"] == 0 and status["completed_updates"] == 0
    log = root / "training_updates.jsonl"
    assert not log.exists() or not log.read_text()


def test_pinned_share_flows_through_the_public_run_and_its_saved_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marl_battlegrounds.baselines.ppo import PPOConfig
    from marl_battlegrounds.training import checkpoints, runner

    identity = checkpoints.runtime_identity()
    monkeypatch.setattr(checkpoints, "runtime_identity", lambda: identity)
    config = TrainConfig(
        keep_past=0,
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
    assert details["schedule"]["history_thresholds"] == "No explicit capture targets"
    latest = json.loads((result.run_dir / "latest_checkpoint.json").read_text())
    saved = checkpoints.read_checkpoint_details(
        result.run_dir / latest["relative_path"]
    )
    assert saved["collection"]["pinned_opponent_share"] == 0.1
    assert saved["metadata"]["config"]["pinned_opponent_share"] == 0.1
    exposure = json.loads((result.run_dir / "exposure.json").read_text())
    assert len(exposure["steps_by_opponent"]) == 2
    assert len(exposure["starts_by_opponent"]) == 2


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
    from marl_battlegrounds.baselines.ppo import PPOConfig

    named = TrainConfig(pinned_opponent="tdm-alpha", pinned_opponent_share=0.1)
    assert config_from_dict(config_to_dict(named)) == named
    assert config_to_dict(named)["pinned_opponent"] == "tdm-alpha"
    assert "pinned_opponent" in config_to_dict(config)
    older_named = {
        key: value
        for key, value in config_to_dict(config).items()
        if key != "pinned_opponent"
    }
    assert config_from_dict(older_named) == config
    framed = TrainConfig(ppo=PPOConfig(spawn_frame="left"))
    assert config_from_dict(config_to_dict(framed)) == framed
    assert config_to_dict(framed)["ppo"]["spawn_frame"] == "left"
    without_frame = config_to_dict(config)
    without_frame["ppo"] = {
        key: value
        for key, value in without_frame["ppo"].items()
        if key != "spawn_frame"
    }
    assert config_from_dict(without_frame) == config
    assert config.ppo.spawn_frame == "left"
    # Two rules side by side: a new run from a config without a frame takes the
    # constructor default, while a saved record without one still means "world".
    from marl_battlegrounds.training.checkpoints import saved_training_config

    saved = saved_training_config({"metadata": {"config": without_frame}})
    assert saved["ppo"]["spawn_frame"] == "world"
    with pytest.raises(ValueError, match="Unknown"):
        config_from_dict({"typo": 1})
    with pytest.raises(ValueError, match="schema_version"):
        config_from_dict({"schema_version": 2})


def test_red_zone_depth_roundtrips_and_saved_configs_without_it_read_zero() -> None:
    from marl_battlegrounds.training import checkpoints

    assert TrainConfig().red_zone_depth == config_from_dict({}).red_zone_depth == 5.0
    for depth in (0.0, 6.0, 20.0):
        config = TrainConfig(red_zone_depth=depth)
        saved = config_to_dict(config)
        assert saved["schema_version"] == 1 and saved["red_zone_depth"] == depth
        assert config_from_dict(json.loads(json.dumps(saved))) == config
        assert checkpoints._config_red_zone_depth(saved) == depth
    # JSON has one number type: a hand-written 6 is the depth 6.0.
    assert config_from_dict({"red_zone_depth": 6}).red_zone_depth == 6.0
    with pytest.raises(TypeError):
        config_from_dict({"red_zone_depth": True})
    with pytest.raises(ValueError, match="finite"):
        config_from_dict({"red_zone_depth": 10**400})
    older = {
        key: value
        for key, value in config_to_dict(TrainConfig()).items()
        if key != "red_zone_depth"
    }
    # A new run from an older file takes the default; a saved run keeps 0.0.
    assert config_from_dict(older).red_zone_depth == 5.0
    saved_older = checkpoints.saved_training_config({"metadata": {"config": older}})
    assert saved_older["red_zone_depth"] == 0.0
    assert checkpoints._config_red_zone_depth(older) is None


def test_a_depth_wider_than_a_map_is_refused_before_any_file(tmp_path: Path) -> None:
    from marl_battlegrounds.baselines.ppo import PPOConfig
    from marl_battlegrounds.training import runner

    config = TrainConfig(
        keep_past=0,
        num_envs=4,
        total_env_steps=32,
        ppo=PPOConfig(rollout_length=4, epochs=1),
        red_zone_depth=1000.0,
        validation_opponents=("random",),
        verbose=False,
    )
    output = tmp_path / "run"
    output.mkdir()
    # Content preparation checks every map width before a panel is published.
    with pytest.raises(ValueError, match="map_width"):
        runner.train(config, output_dir=output)
    assert not any(output.iterdir())


@pytest.mark.parametrize("method", ("mappo", "ippo", "ff_mappo", "ff_ippo"))
def test_ppo_method_config_roundtrip_and_method_specific_batch_rules(
    method: Literal["mappo", "ippo", "ff_mappo", "ff_ippo"],
) -> None:
    from marl_battlegrounds.baselines.ppo import PPOConfig

    config = TrainConfig(
        keep_past=0,
        method=method,
        num_envs=4,
        total_env_steps=32,
        ppo=PPOConfig(rollout_length=4, epochs=1),
    )
    assert config_from_dict(config_to_dict(config)) == config
    assert config_from_dict({}).method == "mappo"
    threshold = replace(config, score_threshold_curriculum=True)
    assert config_from_dict(config_to_dict(threshold)) == threshold
    if method.startswith("ff_"):
        assert replace(config, num_envs=6, total_env_steps=24).num_envs == 6
    else:
        with pytest.raises(ValueError, match=r"divis|batch|games|B"):
            replace(config, num_envs=6, total_env_steps=24)


def test_retention_preserves_candidates_and_small_ancestry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marl_battlegrounds.training import checkpoints, runner

    # Isolate retention; checkpoint tests resolve real saved payloads.
    def saved_directory(root: Path, kind: str, identifier: str) -> Path:
        return root / kind / identifier

    monkeypatch.setattr(checkpoints, "artifact_directory", saved_directory)

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
    execution.checkpoint_id = "c" * 64
    execution.prune_recovery()
    execution.checkpoint = records["d"]
    execution.checkpoint_id = "d" * 64
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
    execution.config = TrainConfig()
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


def test_actor_warm_start_and_resume_keep_distinct_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.training_continuation_helpers import capture_runs
    from tests.training_learner_helpers import equal

    from marl_battlegrounds.baselines.ppo import PPOConfig
    from marl_battlegrounds.training import checkpoints, runner

    identity = checkpoints.runtime_identity()
    monkeypatch.setattr(checkpoints, "runtime_identity", lambda: identity)
    starts, ends = capture_runs(monkeypatch)
    config = runner.TrainConfig(
        keep_past=0,
        method="ff_ippo",
        num_envs=4,
        total_env_steps=8,
        ppo=PPOConfig(rollout_length=2, epochs=1),
        metrics="none",
        verbose=False,
    )
    parent = runner.train(config, output_dir=tmp_path / "parent")
    warm = runner.train(
        replace(config, initial_actor=str(parent.final_actor)),
        output_dir=tmp_path / "warm",
    )
    imported = starts[warm.run_dir][1]
    equal(
        imported.carry.history.current_variables,
        ends[parent.run_dir].carry.history.current_variables,
    )
    assert int(imported.completed_updates) == 0
    origin = checkpoints.read_checkpoint_description(warm.final_actor)["metadata"][
        "initial_actor"
    ]
    assert (
        origin["actor_digest"]
        == checkpoints.read_checkpoint_description(parent.final_actor)["actor_digest"]
    )
    assert origin["evidence"]["exposure"] == "none"
    parent.final_actor.rename(parent.final_actor.with_name("moved_source"))
    before = ends[warm.run_dir]
    resumed = runner.train(resume_from=warm.final_checkpoint)
    equal(ends[resumed.run_dir], before)
    for field, value in (
        ("method", "ff_mappo"),
        ("ppo", replace(config.ppo, input_scale=0.5)),
        ("ppo", replace(config.ppo, spawn_frame="world")),
    ):
        invalid = replace(config, initial_actor=str(warm.final_actor), **{field: value})
        destination = tmp_path / f"invalid_{field}"
        with pytest.raises(ValueError, match="Initial actor"):
            runner.train(invalid, output_dir=destination)
        assert not destination.exists()


def test_custom_curriculum_config_round_trip_and_rejection() -> None:
    from marl_battlegrounds.training.runner import (
        TrainConfig,
        config_from_dict,
        config_to_dict,
    )

    stages = [
        {"share": 0.5, "maps": [0], "team_size": 2},
        {"share": 0.5, "maps": [0, 1], "team_size": 5, "score_threshold": 3},
    ]
    config = TrainConfig(curriculum=stages)
    assert config_from_dict(config_to_dict(config)) == config
    stages[0]["share"] = 0.25
    assert not isinstance(config.curriculum, bool)
    assert config.curriculum[0]["share"] == 0.5
    with pytest.raises(ValueError, match="shares"):
        TrainConfig(curriculum=stages)
    with pytest.raises(ValueError, match="Choose"):
        TrainConfig(curriculum=config.curriculum, score_threshold_curriculum=True)


def test_custom_stages_extend_and_resume_with_the_full_source_bank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.training_continuation_helpers import capture_runs
    from tests.training_learner_helpers import equal

    from marl_battlegrounds.baselines.ppo import PPOConfig
    from marl_battlegrounds.training import checkpoints, runner

    identity = checkpoints.runtime_identity()
    monkeypatch.setattr(checkpoints, "runtime_identity", lambda: identity)
    starts, ends = capture_runs(monkeypatch)
    parent = runner.train(
        runner.TrainConfig(
            keep_past=0,
            method="ff_ippo",
            num_envs=4,
            total_env_steps=8,
            ppo=PPOConfig(rollout_length=2, epochs=1),
            metrics="none",
            verbose=False,
            curriculum=[
                {"share": 1.0, "maps": [0], "team_size": 2, "score_threshold": 2}
            ],
        ),
        output_dir=tmp_path / "parent",
    )
    child = runner.extend_training(
        parent.final_checkpoint,
        additional_env_steps=8,
        output_dir=tmp_path / "child",
        changes={
            "curriculum": [
                {"share": 1.0, "maps": [0, 1], "team_size": 3, "score_threshold": 3}
            ]
        },
    )
    collection, initial = starts[child.run_dir]
    assert collection.binding.score_thresholds == (2, 3)
    equal(initial.carry.state, ends[parent.run_dir].carry.state)
    equal(initial.carry.source_indices, ends[parent.run_dir].carry.source_indices)
    before = ends[child.run_dir]
    result = runner.train(resume_from=child.final_checkpoint)
    equal(ends[result.run_dir], before)
    assert starts[result.run_dir][0].binding.score_thresholds == (2, 3)


@pytest.mark.parametrize("method", ["ff_ippo", "qmix", "pqn_vdn"])
def test_live_update_callback_matches_saved_rows_and_isolates_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: Literal["ff_ippo", "qmix", "pqn_vdn"],
) -> None:
    from tests.training_continuation_helpers import fixed_source

    from marl_battlegrounds.baselines.ppo import PPOConfig
    from marl_battlegrounds.baselines.pqn import PQNConfig
    from marl_battlegrounds.baselines.qmix import QMIXConfig
    from marl_battlegrounds.training import checkpoints, runner

    fixed_source(monkeypatch, method)
    config = TrainConfig(
        keep_past=0,
        method=method,
        num_envs=4,
        total_env_steps=24 if method == "pqn_vdn" else 16,
        ppo=PPOConfig(rollout_length=2, epochs=1)
        if method == "ff_ippo"
        else PPOConfig(),
        qmix=QMIXConfig(
            rollout_length=2,
            buffer_size=4,
            min_buffer_size=2,
            sample_sequence_length=2,
            sample_batch_size=2,
            epochs=1,
            eps_min=1.0,
        )
        if method == "qmix"
        else None,
        pqn=PQNConfig(
            rollout_length=2,
            memory_window=1,
            epochs=1,
            num_minibatches=1,
            lr_linear_decay=False,
            eps_start=1.0,
            eps_finish=1.0,
        )
        if method == "pqn_vdn"
        else None,
        checkpoint_interval_updates=1,
        metrics="none",
        verbose=False,
    )
    root = tmp_path / "run"
    received: list[dict[str, Any]] = []

    def log(row: dict[str, Any]) -> None:
        saved = [json.loads(line) for line in (root / "training_updates.jsonl").open()]
        assert len(saved) == len(received) + 1 and saved[-1] == row
        received.append(json.loads(json.dumps(row, allow_nan=False)))
        row["env_steps"] = -1
        if "used_exposure" in row:
            row["used_exposure"]["by_stage"][0] = -999

    result = runner.train(config, output_dir=root, on_update=log)
    saved = [json.loads(line) for line in (root / "training_updates.jsonl").open()]
    assert received == saved
    assert [row["env_steps"] for row in saved] == (
        [8, 12, 20, 24] if method == "pqn_vdn" else [8, 16]
    )
    details = checkpoints.read_checkpoint_details(result.final_checkpoint)
    assert {row["run_id"] for row in saved} == {details["metadata"]["run_id"]}
    assert result.completed_env_steps == config.total_env_steps
    assert "on_update" not in details["metadata"]["config"]
    if method == "pqn_vdn":
        assert (
            details["metadata"]["host_state"]["used_exposure"]
            == saved[-1]["used_exposure"]
        )
    elif method == "ff_ippo":
        child_rows: list[dict[str, Any]] = []
        child = runner.extend_training(
            result.final_checkpoint,
            additional_env_steps=8,
            output_dir=tmp_path / "child",
            on_update=child_rows.append,
        )
        child_saved = [
            json.loads(line)
            for line in (child.run_dir / "training_updates.jsonl").open()
        ]
        assert child_rows == child_saved and len(child_rows) == 1
        assert child_rows[0]["env_steps"] == 24
        assert child_rows[0]["run_id"] != received[0]["run_id"]


def test_callback_failure_keeps_saved_row_and_resumes_from_normal_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.training_continuation_helpers import fixed_source, latest

    from marl_battlegrounds.baselines.ppo import PPOConfig
    from marl_battlegrounds.training import checkpoints, runner

    fixed_source(monkeypatch, "ff_ippo")
    config = TrainConfig(
        keep_past=0,
        method="ff_ippo",
        num_envs=4,
        total_env_steps=16,
        ppo=PPOConfig(rollout_length=2, epochs=1),
        checkpoint_interval_updates=1,
        metrics="none",
        verbose=False,
    )
    root = tmp_path / "run"
    calls: list[int] = []
    failure = LookupError("Researcher logger failed")

    def fail_second(row: dict[str, Any]) -> None:
        calls.append(row["env_steps"])
        if len(calls) == 2:
            raise failure

    with pytest.raises(RuntimeError, match="on_update failed after saving") as caught:
        runner.train(config, output_dir=root, on_update=fail_second)
    assert caught.value.__cause__ is failure and calls == [8, 16]
    log = root / "training_updates.jsonl"
    saved = [json.loads(line) for line in log.open()]
    assert [row["env_steps"] for row in saved] == [8, 16]
    first_line = log.read_bytes().splitlines()[0]
    checkpoint = latest(root)
    assert checkpoints.read_checkpoint_details(checkpoint)["counters"]["env_steps"] == 8
    assert json.loads((root / "status.json").read_text())["status"] == "failed"
    resumed_rows: list[dict[str, Any]] = []
    resumed = runner.train(resume_from=checkpoint, on_update=resumed_rows.append)
    after = [json.loads(line) for line in log.open()]
    assert resumed.completed_env_steps == 16
    assert [row["env_steps"] for row in after] == [8, 16]
    assert resumed_rows == after[1:] and len(resumed_rows) == 1
    assert log.read_bytes().splitlines()[0] == first_line
    assert resumed_rows[0]["run_id"] == saved[-1]["run_id"]
    assert resumed_rows[0]["attempt_id"] != saved[-1]["attempt_id"]


def test_history_settings_and_actual_capture_spacing() -> None:
    from types import SimpleNamespace

    import jax.numpy as jnp

    from marl_battlegrounds.baselines.ppo import PPOConfig
    from marl_battlegrounds.training.curriculum import make_training_schedule
    from marl_battlegrounds.training.runner import _history_setup

    base = TrainConfig(
        method="ff_ippo",
        num_envs=2,
        total_env_steps=64,
        ppo=PPOConfig(rollout_length=8, epochs=1, minibatches=1),
        keep_past=2,
        history_capture_env_steps=(2, 18),
    )
    assert config_from_dict(config_to_dict(base)) == base
    schedule = make_training_schedule(total_env_steps=64, num_envs=2)
    prepared: Any = SimpleNamespace(
        source_configs=SimpleNamespace(max_steps=jnp.asarray([8]))
    )
    resolved, options = _history_setup(base, schedule, prepared)
    assert options == {
        "keep_past": 2,
        "history_capture_capacity": 2,
        "minimum_capture_rounds": 8,
        "capture_interval_rounds": 0,
    }
    assert resolved.arrays.history_threshold_count is not None
    assert int(resolved.arrays.history_threshold_count) == 2
    # These distinct requested steps both round to the same publication.
    with pytest.raises(ValueError, match="0 rounds apart"):
        _history_setup(
            replace(base, history_capture_env_steps=(2, 4)), schedule, prepared
        )
    with pytest.raises(ValueError, match="at least 8"):
        _history_setup(
            replace(base, history_capture_env_steps=None, past_capture_interval=8),
            schedule,
            prepared,
        )
    _, recurring = _history_setup(
        replace(base, history_capture_env_steps=None, past_capture_interval=16),
        schedule,
        prepared,
    )
    assert recurring["capture_interval_rounds"] == 8
    assert recurring["history_capture_capacity"] == 4
    _, rounded = _history_setup(
        replace(base, history_capture_env_steps=None, past_capture_interval=15),
        schedule,
        prepared,
    )
    assert rounded["capture_interval_rounds"] == 8
    disabled, none = _history_setup(replace(base, keep_past=0), schedule, prepared)
    assert disabled.arrays.history_threshold_count is not None
    assert int(disabled.arrays.history_threshold_count) == 0
    assert none["history_capture_capacity"] == 0
    with pytest.raises(ValueError, match="not both"):
        replace(base, past_capture_interval=16)


def test_child_history_growth_preserves_named_member_statistics() -> None:
    from marl_battlegrounds.training.runner import _resize_member_statistics

    host: dict[str, Any] = {
        "member_completed": [[1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 3, 4]],
        "member_score_sums": [[5, 4], [3, 3], [2, 6], [9, 8]],
        "last_member_completed": [[0, 0, 0], [0, 0, 0], [0, 0, 1], [1, 1, 1]],
        "last_member_score_sums": [[0, 0], [0, 0], [2, 6], [3, 2]],
        "sampled_exposure": {"by_opponent": [10, 20, 30, 40]},
    }
    _resize_member_statistics(host, old_capacity=1, new_capacity=3, new_rows=7)
    assert host["member_completed"] == [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
        [0, 0, 0],
        [2, 3, 4],
        [0, 0, 0],
    ]
    assert host["member_score_sums"][5:] == [[9, 8], [0, 0]]
    assert host["last_member_completed"][5:] == [[1, 1, 1], [0, 0, 0]]
    assert host["last_member_score_sums"][5:] == [[3, 2], [0, 0]]
    assert host["sampled_exposure"]["by_opponent"] == [10, 20, 30, 0, 0, 40, 0]


@pytest.mark.parametrize("recording", [False, True])
def test_named_population_callback_append_and_exact_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recording: bool
) -> None:
    from tests.training_continuation_helpers import capture_runs, fixed_source
    from tests.training_learner_helpers import equal

    from marl_battlegrounds.baselines.ppo import PPOConfig
    from marl_battlegrounds.training import checkpoints, runner

    fixed_source(monkeypatch, "ff_ippo")
    starts, ends = capture_runs(monkeypatch)
    config = TrainConfig(
        method="ff_ippo",
        num_envs=4,
        total_env_steps=16,
        keep_past=0,
        opponents={"self": 0.6, "past": 0.2, "random_friend": 0.2},
        ppo=PPOConfig(rollout_length=2, epochs=1),
        checkpoint_interval_updates=1,
        recording=recording,
        metrics="none",
        verbose=False,
    )
    seen: list[int] = []

    def choose(statistics: dict[str, Any], steps: int) -> dict[str, Any]:
        seen.append(steps)
        assert {row["name"] for row in statistics["members"]} >= {
            "self",
            "random_friend",
        }
        return {"opponents": ["random_friend", "self", "self"]}

    parent = runner.train(
        config,
        output_dir=tmp_path / "parent",
        opponents={"random_friend": "random"},
        matchmaking=choose,
    )
    assert seen == [0, 8]
    details = checkpoints.read_checkpoint_details(parent.final_checkpoint)
    assert details["metadata"]["opponent_selection"] == [
        "random_friend",
        "self",
        "self",
    ]
    assert details["collection"]["opponent_members"][0]["reference"] == "random"
    child = runner.extend_training(
        parent.final_checkpoint,
        additional_env_steps=8,
        output_dir=tmp_path / "child",
        changes={"opponents": ["new_friend", "random_friend"]},
        opponents={"new_friend": "tdm-alpha"},
    )
    child_collection, child_start = starts[child.run_dir]
    assert child_collection.opponent_names == ("random_friend", "new_friend")
    equal(child_start.carry.state, ends[parent.run_dir].carry.state)
    before = ends[child.run_dir]
    resumed = runner.train(resume_from=child.final_checkpoint)
    equal(ends[resumed.run_dir], before)
    saved = checkpoints.read_checkpoint_details(resumed.final_checkpoint)
    assert [row["reference"] for row in saved["collection"]["opponent_members"]] == [
        "random",
        "tdm-alpha",
    ]


@pytest.mark.parametrize("method", ["ff_ippo", "qmix", "pqn_vdn"])
def test_partner_population_append_preserves_games_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: Literal["ff_ippo", "qmix", "pqn_vdn"],
) -> None:
    from tests.training_continuation_helpers import capture_runs, fixed_source
    from tests.training_learner_helpers import equal

    from marl_battlegrounds.baselines.ppo import PPOConfig
    from marl_battlegrounds.baselines.pqn import PQNConfig
    from marl_battlegrounds.baselines.qmix import QMIXConfig
    from marl_battlegrounds.training import checkpoints, runner

    fixed_source(monkeypatch, method)
    starts, ends = capture_runs(monkeypatch)
    parent = runner.train(
        TrainConfig(
            method=method,
            num_envs=4,
            total_env_steps=24,
            keep_past=0,
            learner_slots=(0, 2),
            partners={"friend": 1.0},
            validation_partners={"unfamiliar_friend": "tdm-beta"},
            validation_partner_labels={"unfamiliar_friend": "held_out"},
            ppo=PPOConfig(rollout_length=2, epochs=1)
            if method == "ff_ippo"
            else PPOConfig(),
            qmix=QMIXConfig(
                rollout_length=2,
                buffer_size=4,
                min_buffer_size=2,
                sample_sequence_length=2,
                sample_batch_size=2,
                epochs=1,
            )
            if method == "qmix"
            else None,
            pqn=PQNConfig(
                rollout_length=2,
                memory_window=1,
                epochs=1,
                num_minibatches=1,
                lr_linear_decay=False,
            )
            if method == "pqn_vdn"
            else None,
            checkpoint_interval_updates=1,
            recording=method == "ff_ippo",
            metrics="none",
            verbose=False,
        ),
        partners={"friend": "random"},
        output_dir=tmp_path / "parent",
    )
    saved = checkpoints.read_checkpoint_details(parent.final_checkpoint)
    assert saved["collection"]["learner_slots"] == [0, 2]
    assert saved["collection"]["partner_members"][0]["reference"] == "random"
    assert [
        member["label"]
        for member in saved["metadata"]["validation_deployment"]["partners"]
    ] == ["familiar", "held_out"]
    assert saved["metadata"]["used_partner_members"][0]["name"] == "friend"
    child = runner.extend_training(
        parent.final_checkpoint,
        additional_env_steps=8,
        output_dir=tmp_path / "child",
        changes={"partners": ["new_friend", "friend"]},
        partners={"new_friend": "tdm-alpha"},
    )
    collection, boundary = starts[child.run_dir]
    assert collection.partner_names == ("friend", "new_friend")
    equal(boundary.carry.state, ends[parent.run_dir].carry.state)
    equal(
        boundary.carry.partner_values[0].members[0],
        ends[parent.run_dir].carry.partner_values[0].members[0],
    )
    for old, new in zip(
        ends[parent.run_dir].carry.partner_selection,
        boundary.carry.partner_selection,
        strict=True,
    ):
        equal(old.choices, new.choices)
        equal(old.game_starts, new.game_starts)
    if method == "qmix":
        equal(boundary.replay, ends[parent.run_dir].replay)
    elif method == "pqn_vdn":
        equal(boundary.recent, ends[parent.run_dir].recent)
    details = checkpoints.read_checkpoint_details(child.final_checkpoint)
    assert details["metadata"]["partner_selection"] == ["new_friend", "friend"]
    assert [
        row["name"] for row in details["metadata"]["validation_deployment"]["partners"]
    ] == ["friend", "new_friend", "unfamiliar_friend"]
    expected = ends[child.run_dir]
    resumed = runner.train(resume_from=child.final_checkpoint)
    assert resumed.completed_env_steps == 32
    equal(ends[child.run_dir], expected)


@pytest.mark.parametrize("choice", [{}, {"partners": {"friend": 1.0}}])
def test_partner_only_matchmaking_keeps_legacy_pinned_opponents(
    monkeypatch: pytest.MonkeyPatch, choice: dict[str, Any]
) -> None:
    from types import SimpleNamespace
    from typing import NamedTuple, cast

    from marl_battlegrounds.training import collection, runner

    class State(NamedTuple):
        carry: object

    execution = object.__new__(runner._Run)
    execution.collection = cast(
        Any, SimpleNamespace(pinned_opponent_share=0.1, partner_records=())
    )
    original_carry = object()
    changed_carry = object()
    execution.state = State(original_carry)
    execution.host = {"env_steps": 8}

    def choose(_stats: dict[str, Any], _steps: int) -> dict[str, Any]:
        return choice

    def statistics(_self: runner._Run) -> list[dict[str, Any]]:
        return []

    execution.matchmaking = choose
    monkeypatch.setattr(runner._Run, "_member_statistics", statistics)
    selections: list[object] = []

    def partners(
        owner: object, carry: object, *, selection: object
    ) -> tuple[object, object]:
        assert owner is execution.collection and carry is original_carry
        selections.append(selection)
        return owner, changed_carry

    monkeypatch.setattr(collection, "update_partner_selection", partners)
    # Leave the real opponent helper installed: it rejects this legacy pin.
    execution._choose_next_games()
    assert selections == [choice.get("partners")]
    assert execution.state.carry is changed_carry
    assert execution.collection.pinned_opponent_share == 0.1


def test_matchmaking_cannot_mutate_frozen_member_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace
    from typing import NamedTuple, cast

    from marl_battlegrounds.training import collection, runner

    class State(NamedTuple):
        carry: object

    partner = {"name": "friend", "evidence": {"actor_id": "frozen_partner"}}
    opponent = {"name": "rival", "evidence": {"actor_id": "frozen_opponent"}}
    execution = object.__new__(runner._Run)
    execution.collection = cast(Any, SimpleNamespace(partner_records=(partner,)))
    execution.state = State(object())
    execution.host = {"env_steps": 8}

    def choose(stats: dict[str, Any], _steps: int) -> dict[str, object]:
        stats["partners"][0].pop("name")
        stats["partners"][0]["evidence"]["actor_id"] = "changed_partner"
        stats["members"][0]["evidence"]["actor_id"] = "changed_opponent"
        return {}

    def statistics(_self: runner._Run) -> list[dict[str, Any]]:
        return [{**opponent}]

    def partners(
        owner: object, carry: object, *, selection: object
    ) -> tuple[object, object]:
        return owner, carry

    execution.matchmaking = choose
    monkeypatch.setattr(runner._Run, "_member_statistics", statistics)
    monkeypatch.setattr(collection, "update_partner_selection", partners)
    execution._choose_next_games()
    assert partner == {"name": "friend", "evidence": {"actor_id": "frozen_partner"}}
    assert opponent == {"name": "rival", "evidence": {"actor_id": "frozen_opponent"}}
