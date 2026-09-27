"""Check public child validation, frozen roots and original parent actor ownership.

Short CPU learners use real schema-2 M8 validation under the fixed H300 rules.
These tests inspect saved games and identities; they make no learning or speed
claim. Small direct selection fixtures prove a better parent can remain selected
while final always belongs to the child. Historical score-only fixtures request
the saved ranking rule; live runs use their recorded rule and checkpoint IDs,
never readable actor-folder names.
"""

import json
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest

from marl_battlegrounds.training import validation


def _files(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_parent_best_is_distinct_from_child_final(tmp_path: Path) -> None:
    panel = validation.FrozenPanel(
        tmp_path / "panel.json",
        "panel",
        (validation.PanelMember("Opponent", registration_id="opponent"),),
        True,
        schema_version=2,
        roots={"routine": 11, "initialization": 12, "confirmation": 13},
    )
    declared: dict[str, Any] = {
        "schema_version": 1,
        "panel_digest": "panel",
        "panel_schema_version": 2,
        "roots": panel.roots,
        "routine_seed_pairs": 1,
        "confirmation_seed_pairs": 1,
        "red_zone_depth": 5.0,
        "allow_different_roots": False,
    }

    def row(identifier: str, steps: int, purpose: str, score: float) -> dict[str, Any]:
        return {
            **validation.declared_panel_task(
                declared,
                panel,
                checkpoint_id=identifier,
                actor_digest=identifier,
                env_steps=steps,
                purpose=purpose,
            ),
            "complete": True,
            "score": score,
            "mean_kill_difference": 0.0,
        }

    parent_routine = row("parent", 8, "routine", 0.9)
    parent_confirmation = row("parent", 8, "confirmation", 0.8)
    child_routine = row("child", 12, "routine", 0.4)
    child_confirmation = row("child", 12, "confirmation", 0.5)
    inherited = {"records": [parent_routine, parent_confirmation], "used_roots": [11]}
    _, confirmed, candidates = validation.selection_validation_results(
        [child_routine],
        [child_confirmation],
        inherited,
        declaration=declared,
        panel=panel,
        final_checkpoint_id="child",
        rule="saved",
    )
    from marl_battlegrounds.training.analysis import select_checkpoint

    assert candidates == ("parent", "child")
    assert select_checkpoint(confirmed, rule="saved")["checkpoint_id"] == "parent"
    assert parent_confirmation["checkpoint_id"] == "parent"
    changed = {**declared, "roots": {**panel.roots, "confirmation": 14}}
    _, confirmed, candidates = validation.selection_validation_results(
        [child_routine],
        [],
        inherited,
        declaration=changed,
        panel=panel,
        final_checkpoint_id="child",
        rule="saved",
    )
    assert candidates == ("parent", "child") and not confirmed


def test_public_child_roots_resume_and_original_selected_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from marl_battlegrounds.baselines.ppo import PPOConfig
    from marl_battlegrounds.evaluation.recording_context import (
        capture_recording_provenance,
    )
    from marl_battlegrounds.training import checkpoints, extend_training, train
    from marl_battlegrounds.training._selection_evidence import (
        freeze_inherited_candidates,
        read_inherited_candidates,
        read_run_evidence,
    )
    from marl_battlegrounds.training.analysis import select_checkpoint
    from marl_battlegrounds.training.runner import TrainConfig
    from marl_battlegrounds.training.selection import reselect_checkpoint

    identity = checkpoints.runtime_identity()
    monkeypatch.setattr(checkpoints, "runtime_identity", lambda: identity)
    provenance = capture_recording_provenance()
    evaluator = import_module("marl_battlegrounds.evaluation.evaluate")

    def fixed_provenance(**_: object) -> dict[str, object]:
        return provenance

    monkeypatch.setattr(evaluator, "capture_recording_provenance", fixed_provenance)
    panel = validation.create_panel(
        opponents=("tdm-alpha",), output_dir=tmp_path / "panel"
    )
    parent = train(
        TrainConfig(
            method="ff_ippo",
            num_envs=4,
            total_env_steps=8,
            keep_past=0,
            seed=811,
            ppo=PPOConfig(rollout_length=2, epochs=1),
            checkpoint_interval_updates=1,
            validation_panel=str(panel.path),
            validation_fractions=(1.0,),
            routine_seed_pairs=1,
            confirmation_seed_pairs=1,
            metrics="none",
            verbose=False,
        ),
        output_dir=tmp_path / "parent",
    )
    before = _files(parent.run_dir)
    pointer = json.loads((parent.run_dir / "latest_checkpoint.json").read_text())
    child = extend_training(
        parent.run_dir / pointer["relative_path"],
        additional_env_steps=4,
        output_dir=tmp_path / "child",
        changes={
            "validation": {
                "roots": {"routine": 7001, "confirmation": 7002},
                "allow_different_roots": True,
            }
        },
    )
    assert _files(parent.run_dir) == before
    details = json.loads((child.run_dir / "run_details.json").read_text())
    declared = details["validation_declaration"]
    assert declared["roots"]["routine"] == 7001
    assert declared["roots"]["confirmation"] == 7002
    assert [point["env_steps"] for point in declared["points"]] == [12]
    assert child.final_actor.parent == child.run_dir / "actors"
    parent_id = checkpoints.read_checkpoint_description(parent.final_actor)["metadata"][
        "checkpoint_id"
    ]
    child_id = checkpoints.read_checkpoint_description(child.final_actor)["metadata"][
        "checkpoint_id"
    ]
    evidence = read_run_evidence(child.run_dir)
    local = json.loads((child.run_dir / "validation_results.json").read_text())
    inherited = read_inherited_candidates(
        details["continuation"]["inherited_candidates"], declaration=declared
    )
    _, confirmations, candidates = validation.selection_validation_results(
        [row for row in local if row["purpose"] == "routine"],
        [row for row in local if row["purpose"] == "confirmation"],
        inherited,
        declaration=declared,
        panel=panel,
        final_checkpoint_id=child_id,
        rule=details["selection_rule"],
    )
    winner = select_checkpoint(confirmations)
    assert child.selected_actor == Path(
        evidence["actors"][winner["checkpoint_id"]]["actor_path"]
    )
    assert parent_id in candidates and child_id in candidates
    decision = reselect_checkpoint(
        [child.run_dir],
        declaration={"name": "Child Evidence", "allow_different_roots": True},
        output_dir=tmp_path / "separate_selection",
    )
    assert decision["status"] == "complete"
    assert decision["runs"][0]["winner"]["root"] == 7002
    assert decision["runs"][0]["winner"]["seed_pairs"] == 1
    assert Path(decision["runs"][0]["winner"]["actor_path"]) == child.selected_actor
    rows = json.loads((child.run_dir / "validation_results.json").read_text())
    assert all(
        row["root"] == (7001 if row["purpose"] == "routine" else 7002) for row in rows
    )
    pointer = json.loads((child.run_dir / "latest_checkpoint.json").read_text())
    child_checkpoint = child.run_dir / pointer["relative_path"]
    next_references = freeze_inherited_candidates(
        child_checkpoint, declaration=declared
    )
    next_evidence = read_inherited_candidates(next_references, declaration=declared)
    assert next_evidence["actors"][parent_id]["actor_path"] == str(parent.final_actor)
    assert any(
        row["root"] == 7002 and row["checkpoint_id"] == parent_id
        for row in next_evidence["records"]
        if row["purpose"] == "confirmation"
    )
    resumed = train(resume_from=child_checkpoint)
    assert resumed.selected_actor == child.selected_actor
    assert resumed.final_actor == child.final_actor
    assert _files(parent.run_dir) == before
    assert (
        json.loads((child.run_dir / "run_details.json").read_text())[
            "validation_declaration"
        ]
        == declared
    )


def test_child_before_learning_finishes_without_a_selected_actor(
    tmp_path: Path,
) -> None:
    panel = validation.FrozenPanel(
        tmp_path / "panel.json",
        "panel",
        (validation.PanelMember("Opponent", registration_id="opponent"),),
        True,
        schema_version=2,
        roots={"routine": 11, "initialization": 12, "confirmation": 13},
    )
    declared = {
        "schema_version": 1,
        "panel_digest": "panel",
        "panel_schema_version": 2,
        "roots": panel.roots,
        "routine_seed_pairs": 1,
        "confirmation_seed_pairs": 1,
        "red_zone_depth": 5.0,
        "allow_different_roots": False,
    }
    row = {
        **validation.declared_panel_task(
            declared,
            panel,
            checkpoint_id="untrained",
            actor_digest="weights",
            env_steps=4,
            purpose="routine",
        ),
        "complete": True,
        "score": 0.5,
        "mean_kill_difference": 0.0,
        "method": "pqn_vdn",
        "optimizer_steps": 0,
    }
    routine, confirmations, candidates = validation.selection_validation_results(
        [row],
        [],
        {"records": [], "used_roots": []},
        declaration=declared,
        panel=panel,
        final_checkpoint_id="untrained",
    )
    assert routine == [row] and confirmations == [] and candidates == ()
    with pytest.raises(ValueError, match="Incomplete validation"):
        validation.selection_validation_results(
            [{**row, "complete": False}],
            [],
            {"records": [], "used_roots": []},
            declaration=declared,
            panel=panel,
            final_checkpoint_id="untrained",
        )


def test_child_host_counts_keep_partial_blocks_and_unfinished_pqn_warmup() -> None:
    from marl_battlegrounds.training import _run_io
    from marl_battlegrounds.training._continuation_schedules import LearnerContinuation

    pqn_host = {
        "env_steps": 20,
        "completed_updates": 0,
        "completed_blocks": 2,
        "learning_blocks": 0,
        "actor_decisions": 0,
        "used_sequences": 0,
        "used_td_pairs": 0,
        "used_agent_utilities": 0,
        "used_prefix_td_pairs": 0,
        "used_exposure": {
            "by_stage": [0] * 17,
            "by_source": [0],
            "by_opponent": [0] * 21,
        },
    }
    _run_io._check_pqn_host_counts(  # pyright: ignore[reportPrivateUsage]
        pqn_host,
        {"env_steps": 20, "updates": 0, "completed_blocks": 2, "learning_blocks": 0},
        {
            "num_envs": 4,
            "total_env_steps": 20,
            "pqn": {
                "rollout_length": 4,
                "memory_window": 2,
                "epochs": 1,
                "num_minibatches": 1,
            },
        },
        continuation=LearnerContinuation(
            "pqn_vdn", 4, 1, 0, pqn_planned_learning_blocks=3
        ),
    )
    qmix_host = {
        "env_steps": 12,
        "completed_updates": 2,
        "completed_blocks": 2,
        "learning_blocks": 1,
        "actor_decisions": 0,
        "sampled_sequences": 4,
        "used_td_pairs": 8,
        "used_agent_utilities": 8,
        "sampled_exposure": {
            "by_stage": [8] + [0] * 16,
            "by_source": [8],
            "by_opponent": [8] + [0] * 20,
        },
    }
    _run_io._check_qmix_host_counts(  # pyright: ignore[reportPrivateUsage]
        qmix_host,
        {"env_steps": 12, "updates": 2, "completed_blocks": 2, "learning_blocks": 1},
        {
            "num_envs": 4,
            "total_env_steps": 12,
            "qmix": {
                "rollout_length": 4,
                "min_buffer_size": 3,
                "epochs": 2,
                "sample_batch_size": 2,
                "sample_sequence_length": 3,
            },
        },
        continuation=LearnerContinuation("qmix", 1, 1, 0),
    )


def test_warmup_child_run_setup_uses_saved_points_and_planned_learning_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace
    from typing import cast

    from marl_battlegrounds.baselines.ppo import PPOConfig
    from marl_battlegrounds.baselines.pqn import PQNConfig
    from marl_battlegrounds.training import _continuation_schedules, pqn_learner

    runner = import_module("marl_battlegrounds.training.runner")
    context = _continuation_schedules.LearnerContinuation(
        "pqn_vdn", 4, 1, 0, pqn_planned_learning_blocks=3
    )
    declared: dict[str, Any] = {
        "points": [{"env_steps": 20}],
        "roots": {"routine": 11, "initialization": 12, "confirmation": 13},
    }

    def saved_context(schedule: object) -> _continuation_schedules.LearnerContinuation:
        return context

    def saved_declaration(run: object, panel: object) -> dict[str, Any]:
        return declared

    def bank_size(carry: object) -> int:
        return 1

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("A child must use its saved plan before fresh planning")

    monkeypatch.setattr(_continuation_schedules, "schedule_continuation", saved_context)
    monkeypatch.setattr(validation, "saved_validation_declaration", saved_declaration)
    monkeypatch.setattr(validation, "resolve_validation_schedule", forbidden)
    monkeypatch.setattr(runner, "pqn_planned_learning_blocks", forbidden)
    monkeypatch.setattr(pqn_learner, "_bank_size", bank_size)
    config = SimpleNamespace(
        method="pqn_vdn",
        ppo=PPOConfig(),
        qmix=None,
        pqn=PQNConfig(rollout_length=4, memory_window=2),
        num_envs=4,
        total_env_steps=20,
        verbose=False,
        random_diagnostic_seed_pairs=None,
        checkpoint_env_steps=(),
        validation_fractions=(1.0,),
    )
    (tmp_path / "run_details.json").write_text(
        '{"created_at":"2026-09-25T00:00:00+00:00"}'
    )
    run = runner._Run(
        tmp_path,
        cast(Any, config),
        cast(Any, SimpleNamespace(schedule=None)),
        SimpleNamespace(
            carry=SimpleNamespace(
                progress=SimpleNamespace(opponent_steps=SimpleNamespace(shape=(2,)))
            )
        ),
        {
            "host_state": {},
            "continuation": {"inherited_candidates": [], "start_env_steps": 16},
        },
        None,
        cast(Any, object()),
        None,
        attempt_started=0.0,
        runtime={},
    )
    assert run.validation_steps == {20}
    assert run.planned_learning_blocks == 3
