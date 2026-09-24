"""Check host-only progress, exact log recovery and exclusive training writes.

The display uses existing summaries, distinguishes unknown whole-run estimates,
does not read a clock when disabled and respects its ten-second output limit.
Recovery validates saved prefixes before truncation; a second writer is rejected.
Saved host records must match their checkpoint ancestry, actors, scheduled
validation and selection before any writer or training log can move backward.
Runtime metadata keeps device selectors distinct from verified hardware, and
optional terminal memory reads preserve missing measurements and read failures.
Forecasts exclude the first compiled update, wait for warmed measurements, count
partial batches exactly and restart their timing window on a new attempt.
PQN-VDN host records follow its offset boundaries at B32, T128 and H4 (W = 132
initial rounds): the counts, optimizer steps, used sequences, TD pairs,
kept-row pairs and exposure marginals must match the closed-form totals,
also beyond int32; routine validation at 4,224 and 8,320 transitions is
accepted while a result at 8,192 (the old rule's point) is rejected; and the
initial-collection phase is labelled "Initial Random Collection".
"""

from __future__ import annotations

# pyright: reportPrivateUsage=false
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import Mock

import pytest

from marl_battlegrounds.training import _run_io as io_helpers
from marl_battlegrounds.training._compilation import execution_identity

if TYPE_CHECKING:
    from marl_battlegrounds.training.validation import FrozenPanel


def test_progress_distinguishes_training_and_whole_run_eta() -> None:
    status = {
        "phase": "training",
        "env_steps": 50,
        "total_env_steps": 100,
        "completed_updates": 3,
        "elapsed_seconds": 40,
        "transitions_per_second": 5,
        "recent_transitions_per_second": 10,
        "estimated_transitions_per_second": 10,
    }
    text = io_helpers.progress_text(status)
    assert "50 / 100 (50.0%)" in text
    assert "5 steps/s overall" in text
    assert "Training Time Left (Estimate): 5s" in text
    assert "Whole Run Time Left (Estimate): Estimating" in text
    assert "Whole Run Time Left (Estimate): 25s" in io_helpers.progress_text(
        {**status, "pending_work_seconds": 20}
    )


def test_eta_waits_for_warm_updates_and_starts_over_after_resume() -> None:
    status = {
        "env_steps": 4096,
        "total_env_steps": 1048576,
        "transitions_per_second": 4096 / 27,
    }
    estimate = io_helpers.TrainingSpeedEstimate()
    estimate.observe(4096, 27)
    estimate.observe(4096, 1)
    assert estimate.rate is None
    assert "Training Time Left (Estimate): Estimating" in io_helpers.progress_text(
        status
    )
    estimate.observe(4096, 1)
    assert estimate.rate == 4096
    resumed = io_helpers.TrainingSpeedEstimate()
    resumed.observe(4096, 30)
    assert resumed.rate is None
    assert "Training Time Left (Estimate): 0s" in io_helpers.progress_text(
        {**status, "env_steps": 1048576, "estimated_transitions_per_second": None}
    )


def test_progress_separates_training_threshold_from_validation() -> None:
    text = io_helpers.progress_text({"score_threshold": 3})
    assert "New Training Games Need 3 Kills To Win" in text
    assert "Validation Still Needs 20" in text
    assert "New Training Games" not in io_helpers.progress_text({})


def test_eta_window_uses_recent_real_transitions_and_rejects_invalid_costs() -> None:
    estimate = io_helpers.TrainingSpeedEstimate()
    estimate.observe(100, 20)
    estimate.observe(100, 10)
    for _ in range(5):
        estimate.observe(100, 1)
    assert estimate.rate == 100
    estimate.observe(50, 1)
    assert estimate.rate == 90
    estimate.observe(100, float("nan"))
    assert estimate.rate is None
    estimate.observe(100, 1)
    assert estimate.rate is None
    estimate.observe(100, 1)
    assert estimate.rate == 100


def test_progress_uses_no_clock_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden() -> float:
        raise AssertionError("disabled display read the clock")

    monkeypatch.setattr(io_helpers.time, "monotonic", forbidden)
    output = io.StringIO()
    io_helpers.ProgressReporter(enabled=False, stream=output).report({})
    assert output.getvalue() == ""


def test_progress_throttles_updates_but_reports_phase_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = iter((0, 1, 10, 11))
    monkeypatch.setattr(io_helpers.time, "monotonic", lambda: next(clock))
    output = io.StringIO()
    reporter = io_helpers.ProgressReporter(enabled=True, stream=output)
    for phase in ("training", "training", "training", "validation"):
        reporter.report({"phase": phase})
    assert output.getvalue().count(" UTC]") == 3


def test_log_recovery_rejects_corruption_before_truncation(tmp_path: Path) -> None:
    log = tmp_path / "updates.jsonl"
    io_helpers.append_jsonl(log, {"step": 1})
    cursor = io_helpers.log_cursor(log)
    io_helpers.append_jsonl(log, {"step": 2})
    original = log.read_bytes()
    with pytest.raises(ValueError, match="match"):
        io_helpers.restore_log_cursor(log, {**cursor, "sha256": "0" * 64})
    assert log.read_bytes() == original
    io_helpers.restore_log_cursor(log, cursor)
    io_helpers.restore_log_cursor(log, cursor)
    assert io_helpers.read_jsonl(log) == [{"step": 1}]


def test_run_lock_rejects_second_writer_and_releases(tmp_path: Path) -> None:
    with (
        io_helpers.run_lock(tmp_path),
        pytest.raises(RuntimeError, match="writer"),
        io_helpers.run_lock(tmp_path),
    ):
        pass
    with io_helpers.run_lock(tmp_path):
        io_helpers.atomic_json(tmp_path / "status.json", {"status": "complete"})
    assert (tmp_path / "status.json").is_file()


def test_process_identity_distinguishes_missing_process() -> None:
    assert io_helpers.process_identity()["start_ticks"] is not None
    assert io_helpers.process_identity(2**30)["start_ticks"] is None


def test_abandoned_suffix_is_preserved_once_before_rewind(tmp_path: Path) -> None:
    log = tmp_path / "updates.jsonl"
    io_helpers.append_jsonl(log, {"step": 1})
    cursor = io_helpers.log_cursor(log)
    io_helpers.append_jsonl(log, {"step": 2})
    with log.open("ab") as stream:
        stream.write(b'{"step":')
    evidence = io_helpers.preserve_log_suffix(log, cursor, tmp_path / "attempt")
    assert evidence is not None
    assert evidence.read_bytes() == b'{"step": 2}\n{"step":'
    assert io_helpers.preserve_log_suffix(log, cursor, tmp_path / "attempt") == evidence
    io_helpers.restore_log_cursor(log, cursor)
    assert io_helpers.read_jsonl(log) == [{"step": 1}]
    assert io_helpers.preserve_log_suffix(log, cursor, tmp_path / "attempt") is None


def test_lifecycle_append_preserves_torn_event_and_separates_new_record(
    tmp_path: Path,
) -> None:
    log = tmp_path / "events.jsonl"
    log.write_bytes(b'{"event":')
    io_helpers.append_jsonl(log, {"event": "resumed"}, durable=True)
    assert log.read_bytes() == b'{"event":\n{"event": "resumed"}\n'


def test_runtime_details_reuses_m8_and_records_only_declared_selectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from marl_battlegrounds.evaluation import runtime_provenance
    from marl_battlegrounds.training import runner

    provenance = {"backend": "gpu", "device": "Fixture Device"}
    captured = Mock()
    captured.model_dump.return_value = provenance
    capture = Mock(return_value=captured)
    monkeypatch.setattr(runtime_provenance, "capture_runtime_provenance", capture)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-fixture")
    monkeypatch.setenv("JAX_PLATFORMS", "cuda,cpu")
    monkeypatch.setenv("JAX_DISABLE_JIT", "false")
    monkeypatch.delenv("XLA_PYTHON_CLIENT_MEM_FRACTION", raising=False)
    monkeypatch.setenv("UNRELATED_PRIVATE_SETTING", "must not be recorded")
    details = runner._runtime_details("0.test", 32)
    capture.assert_called_once_with(
        "0.test", num_envs=32, policy_execution_included=True
    )
    captured.model_dump.assert_called_once_with(mode="json")
    assert details["provenance"] == provenance
    assert details["declared_environment"]["CUDA_VISIBLE_DEVICES"] == "GPU-fixture"
    assert details["declared_environment"]["JAX_PLATFORMS"] == "cuda,cpu"
    assert details["declared_environment"]["JAX_DISABLE_JIT"] == "false"
    assert details["declared_environment"]["XLA_PYTHON_CLIENT_MEM_FRACTION"] is None
    assert "UNRELATED_PRIVATE_SETTING" not in details["declared_environment"]


@pytest.mark.parametrize("mode", ["available", "unavailable", "error", "no_device"])
def test_memory_snapshot_preserves_ram_when_allocator_data_is_missing_or_fails(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    import resource
    import sys

    import jax

    from marl_battlegrounds.training import runner

    get_usage = Mock(return_value=SimpleNamespace(ru_maxrss=7))
    monkeypatch.setattr(resource, "getrusage", get_usage)
    counters = {"bytes_in_use": 10, "peak_bytes_in_use": 20}
    stats = Mock(return_value=counters if mode == "available" else None)
    if mode == "error":
        stats.side_effect = RuntimeError("Allocator read failed")
    devices = Mock(
        return_value=[]
        if mode == "no_device"
        else [SimpleNamespace(memory_stats=stats)]
    )
    monkeypatch.setattr(jax, "devices", devices)
    snapshot = runner._memory_snapshot()
    assert snapshot["process_peak_ram_bytes"] == 7 * (
        1 if sys.platform == "darwin" else 1024
    )
    assert snapshot["jax_allocator"] == (counters if mode == "available" else None)
    get_usage.assert_called_once_with(resource.RUSAGE_SELF)
    devices.assert_called_once_with()
    assert stats.call_count == (0 if mode == "no_device" else 1)
    if mode in {"error", "no_device"}:
        assert len(snapshot["errors"]) == 1
        assert snapshot["errors"][0].startswith("JAX allocator:")
    else:
        assert snapshot["errors"] == []


def test_memory_snapshot_preserves_allocator_data_when_ram_read_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import resource

    import jax

    from marl_battlegrounds.training import runner

    monkeypatch.setattr(
        resource, "getrusage", Mock(side_effect=OSError("RAM read failed"))
    )
    counters = {"bytes_in_use": 10}
    monkeypatch.setattr(
        jax,
        "devices",
        Mock(return_value=[SimpleNamespace(memory_stats=Mock(return_value=counters))]),
    )
    snapshot = runner._memory_snapshot()
    assert snapshot["process_peak_ram_bytes"] is None
    assert snapshot["jax_allocator"] == counters
    assert snapshot["errors"] == ["Process RAM: OSError: RAM read failed"]


def _saved_host(*, steps: int = 0, pending: bool | None = False) -> dict[str, Any]:
    host: dict[str, Any] = {
        "env_steps": steps,
        "completed_updates": (steps + 15) // 16,
        "actor_decisions": steps * 5,
        "used_policy_samples": steps * 5,
        "used_value_samples": steps * 5,
        "training_seconds": float(steps),
        "elapsed_seconds": float(steps + 100),
        "routine_results": [],
        "confirmation_results": [],
        "actors": {},
        "pending": None if pending is None else {"routine": pending},
        "selected_actor": None,
        "final_actor": None,
        "validation_seconds": 0.0,
        "validation_games": 0,
        "save_seconds": 0.0,
        "saves": 0,
        "report_seconds": None,
        "slot_complete": False,
        "selection": None,
        "recovery_checkpoints": [],
    }
    if steps:
        host.update(
            {
                "stage_completed": [[0, 0, 0] for _ in range(17)],
                "stage_score_sums": [[0, 0] for _ in range(17)],
                "stage_length_sum": [0] * 17,
                "stage_k20_count": [0] * 17,
            }
        )
    return host


def _description(
    root: Path,
    host: dict[str, Any],
    *,
    parent: dict[str, Any] | None = None,
    panel: bool = False,
    attempt: str = "original",
) -> dict[str, Any]:
    from marl_battlegrounds.training import checkpoints

    config = {
        "seed": 42,
        "num_envs": 4,
        "total_env_steps": 32,
        "ppo": {"epochs": 1, "rollout_length": 4},
        "validation_panel": "frozen-panel" if panel else None,
        "validation_fractions": [0.5, 1.0],
        "checkpoint_env_steps": [],
        "routine_seed_pairs": 1,
        "confirmation_seed_pairs": 2,
        "slot_diagnostic": False,
    }
    details: dict[str, Any] = {
        "schema_version": 1,
        "kind": "learner",
        "schemas": checkpoints._SCHEMAS,
        "metadata": {
            "run_id": "one-run",
            "attempt_id": attempt,
            "parent_checkpoint": None if parent is None else parent["checkpoint_id"],
            "config": config,
            "source": {"fixed": True},
            "dependencies": {"fixed": True},
            "execution": execution_identity(),
            "host_state": host,
        },
        "actor_layout": [],
        "actor_digest": "b" * 64,
        "files": {},
        "collection": {},
        "layout": [],
        "recording_token": None,
        "counters": {
            "updates": host["completed_updates"],
            "env_steps": host["env_steps"],
        },
    }
    details["checkpoint_id"] = hashlib.sha256(
        checkpoints._json_bytes(details)
    ).hexdigest()
    directory = root / "checkpoints" / details["checkpoint_id"]
    directory.mkdir(parents=True)
    io_helpers.atomic_json(directory / "checkpoint_details.json", details)
    return details


def _actor(root: Path, ancestor: dict[str, Any]) -> Path:
    from marl_battlegrounds.training import checkpoints

    details: dict[str, Any] = {
        "schema_version": 1,
        "kind": "actor",
        "schemas": ancestor["schemas"],
        "metadata": {
            "run_id": "one-run",
            "seed": 42,
            "env_steps": ancestor["counters"]["env_steps"],
            "checkpoint_id": ancestor["checkpoint_id"],
        },
        "actor_layout": [],
        "actor_digest": ancestor["actor_digest"],
        "files": {},
    }
    details["checkpoint_id"] = hashlib.sha256(
        checkpoints._json_bytes(details)
    ).hexdigest()
    path = root / "actors" / ancestor["checkpoint_id"]
    path.mkdir(parents=True)
    io_helpers.atomic_json(path / "actor_details.json", details)
    return path


def _panel() -> FrozenPanel:
    from marl_battlegrounds.training.validation import FrozenPanel, PanelMember

    return FrozenPanel(
        Path("panel.json"),
        "c" * 64,
        tuple(
            PanelMember(
                name, Path(name), digest * 64, digest * 64, "panel-run", 7, index + 1
            )
            for index, (name, digest) in enumerate((("Halfway", "d"), ("Final", "e")))
        ),
        True,
    )


def _validation(
    root: Path,
    ancestor: dict[str, Any],
    panel: FrozenPanel,
    purpose: str,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    from marl_battlegrounds.training import validation

    key = ancestor["checkpoint_id"]
    pairs = 1 if purpose == "routine" else 2
    task = validation.validation_task_description(
        checkpoint_id=key,
        actor_digest=ancestor["actor_digest"],
        env_steps=ancestor["counters"]["env_steps"],
        panel_digest=panel.digest,
        purpose=purpose,
        seed_pairs=pairs,
        members=tuple((member.name, member.actor_digest) for member in panel.members),
    )
    directory = root / "validation" / f"{purpose}-{key}"
    directory.mkdir(parents=True)
    io_helpers.atomic_json(directory / "task.json", task)
    paths = [str(directory / f"opponent-{index}" / "run") for index in range(2)]
    for path in paths:
        Path(path).mkdir(parents=True)
    summary = {
        **task,
        "complete": True,
        "score": 0.5,
        "ci_low": 0.25,
        "ci_high": 0.75,
        "games": 20 * pairs,
        "pass_paths": paths,
    }
    io_helpers.atomic_json(directory / "validation_summary.json", summary)

    def saved(parent: Path, pass_id: str) -> Path:
        return parent / "run"

    monkeypatch.setattr(validation, "_saved_run", saved)
    monkeypatch.setattr(validation, "_pending", Mock(return_value=0))
    return {**summary, "elapsed_seconds": 1.0}


def test_saved_host_accepts_update_zero_pending_export_and_no_panel(
    tmp_path: Path,
) -> None:
    details = _description(tmp_path, _saved_host())
    io_helpers.validate_host_state(tmp_path, details, panel=None)
    assert not (tmp_path / "actors").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("pending", {"routine": "False"}),
        ("pending", {"routine": False, "path": "x"}),
        ("training_seconds", float("nan")),
        ("elapsed_seconds", True),
        ("completed_updates", False),
        ("used_value_samples", 1),
        ("recovery_checkpoints", ["../outside"]),
        ("actors", {"../outside": "/tmp"}),
        ("slot_complete", True),
        ("selected_actor", "/tmp/actor"),
    ],
)
def test_saved_host_rejects_malformed_state_without_mutation(
    tmp_path: Path, field: str, value: object
) -> None:
    host = _saved_host()
    host[field] = value
    # The restore owner has already checked the description; exercise its next
    # host admission boundary directly even for nonfinite malformed fields.
    details = _description(tmp_path, _saved_host())
    details["metadata"]["host_state"] = host
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises(ValueError):
        io_helpers.validate_host_state(tmp_path, details, panel=None)
    assert before == {
        path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
    }


def test_saved_host_keeps_ancestor_candidates_across_attempts_with_pruned_payloads(
    tmp_path: Path,
) -> None:
    origin = _description(tmp_path, _saved_host())
    actor = _actor(tmp_path, origin)
    host = _saved_host(steps=32)
    host["actors"] = {origin["checkpoint_id"]: str(actor)}
    host["recovery_checkpoints"] = [origin["checkpoint_id"]]
    current = _description(tmp_path, host, parent=origin, attempt="new-attempt")
    io_helpers.validate_host_state(tmp_path, current, panel=None)
    assert not (tmp_path / "actors" / current["checkpoint_id"]).exists()
    abandoned = _description(
        tmp_path,
        _saved_host(steps=16, pending=None),
        parent=origin,
        attempt="abandoned",
    )
    abandoned_actor = _actor(tmp_path, abandoned)
    host["actors"][abandoned["checkpoint_id"]] = str(abandoned_actor)
    with pytest.raises(ValueError, match="continuation identity"):
        io_helpers.validate_host_state(tmp_path, current, panel=None)


def test_saved_host_rejects_a_different_ancestor_execution(tmp_path: Path) -> None:
    origin = _description(tmp_path, _saved_host())
    current = _description(tmp_path, _saved_host(steps=32), parent=origin)
    current["metadata"]["execution"]["schema_version"] += 1
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="execution identity"):
        io_helpers.validate_host_state(tmp_path, current, panel=None)
    assert before == {
        path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
    }


def test_saved_host_checks_completed_validation_and_allows_current_pending_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = _panel()
    origin = _description(tmp_path, _saved_host(pending=True), panel=True)
    actor = _actor(tmp_path, origin)
    host = _saved_host(steps=16, pending=True)
    host["actors"] = {origin["checkpoint_id"]: str(actor)}
    host["routine_results"] = [
        _validation(tmp_path, origin, panel, "routine", monkeypatch)
    ]
    host["validation_games"] = 20
    current = _description(tmp_path, host, parent=origin, panel=True)
    io_helpers.validate_host_state(tmp_path, current, panel=panel)
    host["routine_results"][0]["panel_digest"] = "f" * 64
    with pytest.raises(ValueError, match="task identity"):
        io_helpers.validate_host_state(tmp_path, current, panel=panel)


def test_saved_host_rejects_missing_due_validation_before_recovery(
    tmp_path: Path,
) -> None:
    panel = _panel()
    details = _description(tmp_path, _saved_host(steps=16, pending=True), panel=True)
    with pytest.raises(ValueError, match="coverage"):
        io_helpers.validate_host_state(tmp_path, details, panel=panel)


def test_saved_host_rejects_actor_provenance_change(tmp_path: Path) -> None:
    origin = _description(tmp_path, _saved_host())
    actor = _actor(tmp_path, origin)
    host = _saved_host(steps=32)
    host["actors"] = {origin["checkpoint_id"]: str(actor)}
    current = _description(tmp_path, host, parent=origin)
    path = actor / "actor_details.json"
    value = json.loads(path.read_text())
    value["metadata"]["seed"] = 999
    del value["checkpoint_id"]
    from marl_battlegrounds.training import checkpoints

    value["checkpoint_id"] = hashlib.sha256(checkpoints._json_bytes(value)).hexdigest()
    io_helpers.atomic_json(path, value)
    with pytest.raises(ValueError, match="actor identity"):
        io_helpers.validate_host_state(tmp_path, current, panel=None)


def test_saved_selection_requires_all_candidates_and_exact_chosen_actor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from marl_battlegrounds.training.analysis import select_checkpoint

    panel = _panel()
    origin = _description(tmp_path, _saved_host(pending=True), panel=True)
    middle = _description(
        tmp_path, _saved_host(steps=16, pending=True), parent=origin, panel=True
    )
    final = _description(
        tmp_path, _saved_host(steps=32, pending=True), parent=middle, panel=True
    )
    host = _saved_host(steps=32, pending=None)
    host["actors"] = {
        item["checkpoint_id"]: str(_actor(tmp_path, item))
        for item in (origin, middle, final)
    }
    host["routine_results"] = [
        _validation(tmp_path, item, panel, "routine", monkeypatch)
        for item in (origin, middle, final)
    ]
    host["confirmation_results"] = [
        _validation(tmp_path, item, panel, "confirmation", monkeypatch)
        for item in (middle, final)
    ]
    host["validation_games"] = 140
    host["final_actor"] = host["actors"][final["checkpoint_id"]]
    host["selection"] = select_checkpoint(host["confirmation_results"])
    host["selected_actor"] = host["actors"][host["selection"]["checkpoint_id"]]
    current = _description(tmp_path, host, parent=final, panel=True)
    io_helpers.validate_host_state(tmp_path, current, panel=panel)
    assert host["selection"]["checkpoint_id"] == middle["checkpoint_id"]
    host["selected_actor"] = host["final_actor"]
    with pytest.raises(ValueError, match="selected actor identity"):
        io_helpers.validate_host_state(tmp_path, current, panel=panel)
    host["selected_actor"] = host["actors"][middle["checkpoint_id"]]
    host["confirmation_results"].pop()
    host["validation_games"] -= 40
    with pytest.raises(ValueError, match="missing confirmation"):
        io_helpers.validate_host_state(tmp_path, current, panel=panel)


def test_description_only_read_preserves_pruned_ancestry_without_allowing_restore(
    tmp_path: Path,
) -> None:
    from marl_battlegrounds.training import checkpoints

    details = _description(tmp_path, _saved_host())
    details["files"] = {"state/data": {"bytes": 1, "sha256": "a" * 64}}
    del details["checkpoint_id"]
    details["checkpoint_id"] = hashlib.sha256(
        checkpoints._json_bytes(details)
    ).hexdigest()
    directory = tmp_path / "checkpoints" / details["checkpoint_id"]
    directory.mkdir()
    io_helpers.atomic_json(directory / "checkpoint_details.json", details)
    assert checkpoints.read_checkpoint_description(directory) == details
    with pytest.raises(ValueError, match="payload files"):
        checkpoints.read_checkpoint_details(directory)
    details["metadata"]["run_id"] = "changed"
    io_helpers.atomic_json(directory / "checkpoint_details.json", details)
    with pytest.raises(ValueError, match="description digest"):
        checkpoints.read_checkpoint_description(directory)


def test_readable_progress_separates_combat_from_optimizer_numbers() -> None:
    status: dict[str, Any] = {
        "phase": "training",
        "env_steps": 1000,
        "total_env_steps": 2000,
        "policy_loss": -0.02465,
        "value_loss": 0.000073,
        "shaping_mean": 0.0000032,
        "random_validation": {
            "games": 40,
            "env_steps": 1000,
            "wall_seconds": 90,
            "mean_kills_for": 4.0,
            "mean_kills_against": 2.0,
            "kill_margin": 2.0,
            "kill_margin_change": 1.5,
            "wins": 0,
            "draws": 40,
            "losses": 0,
        },
    }
    text = io_helpers.progress_text(status)
    assert "Run Time At That Check: 1m 30s" in text
    assert "Our Team Kills 4.00" in text and "Our Team Deaths 2.00" in text
    assert "Change In Kill Difference Since Untrained: +1.50 Per Game" in text
    assert "0 Wins, 40 Draws, 0 Losses" in text
    assert "Draws Can Still Show Combat Improvement" in text
    assert "Policy Loss" not in text and "Shaping" not in text
    missing = {
        **status,
        "random_validation": {
            **status["random_validation"],
            "mean_kills_for": None,
            "mean_kills_against": None,
            "kill_margin": None,
        },
    }
    assert "Not Available" in io_helpers.progress_text(missing)
    assert "Our Team Kills 0" not in io_helpers.progress_text(missing)


def test_screen_worker_leaves_regular_progress_to_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MARL_BGS_PROGRESS_PHASES_ONLY", "1")
    clock = Mock(return_value=0)
    monkeypatch.setattr(io_helpers.time, "monotonic", clock)
    stream = io.StringIO()
    reporter = io_helpers.ProgressReporter(enabled=True, stream=stream)
    reporter.report({"phase": "training"})
    reporter.report({"phase": "training"})
    reporter.report({"phase": "validation"})
    reporter.report({"phase": "validation"}, force=True)
    assert clock.call_count == 3
    assert stream.getvalue().count(" UTC]") == 3


def test_panel_score_remains_visible_without_random_hook() -> None:
    text = io_helpers.progress_text({"validation_score": 0.75})
    assert "Average Game Score 0.750" in text
    assert "Win = 1; Draw = 0.5; Loss = 0" in text


_PQN = {
    "rollout_length": 128,
    "memory_window": 4,
    "epochs": 4,
    "num_minibatches": 16,
    "q_lr": 0.00025,
    "lr_linear_decay": True,
    "max_grad_norm": 1.0,
    "gamma": 0.99,
    "td_lambda": 0.85,
    "eps_start": 1.0,
    "eps_finish": 0.01,
    "eps_decay_fraction": 0.1,
    "input_scale": 1.0,
    "spawn_frame": "left",
}


def _pqn_counts(steps: int, batch: int) -> tuple[int, int]:
    rounds, initial = steps // batch, 132
    if rounds <= initial:
        return -(-rounds // 128), 0
    learning = -(-(rounds - initial) // 128)
    return 2 + learning, learning


def _pqn_host(
    steps: int, *, batch: int = 32, pending: bool | None = False
) -> dict[str, Any]:
    blocks, learning = _pqn_counts(steps, batch)
    rounds = steps // batch
    pairs = 4 * batch * ((rounds - 132) + learning * 3) if learning else 0
    host = _saved_host(steps=steps, pending=pending)
    for name in ("used_policy_samples", "used_value_samples"):
        del host[name]
    host.update(
        {
            "completed_updates": learning * 4 * 16,
            "completed_blocks": blocks,
            "learning_blocks": learning,
            "used_sequences": learning * 4 * batch,
            "used_td_pairs": pairs,
            "used_agent_utilities": 2 * pairs,
            "used_prefix_td_pairs": learning * 4 * batch * 4,
            "used_exposure": {
                "by_stage": [pairs] + [0] * 16,
                "by_source": [pairs, 0],
                "by_opponent": [pairs] + [0] * 20,
            },
        }
    )
    return host


def _pqn_description(
    root: Path,
    host: dict[str, Any],
    *,
    batch: int = 32,
    total: int = 16_640,
    parent: dict[str, Any] | None = None,
    panel: bool = False,
) -> dict[str, Any]:
    from marl_battlegrounds.training import checkpoints

    config = {
        "seed": 42,
        "method": "pqn_vdn",
        "num_envs": batch,
        "total_env_steps": total,
        "pqn": _PQN,
        "validation_panel": "frozen-panel" if panel else None,
        "validation_fractions": [0.25, 0.5, 1.0],
        "checkpoint_env_steps": [],
        "routine_seed_pairs": 1,
        "confirmation_seed_pairs": 2,
        "slot_diagnostic": False,
    }
    details: dict[str, Any] = {
        "schema_version": 1,
        "kind": "learner",
        "schemas": checkpoints.checkpoint_schemas("pqn_vdn"),
        "metadata": {
            "run_id": "one-run",
            "attempt_id": "original",
            "parent_checkpoint": None if parent is None else parent["checkpoint_id"],
            "config": config,
            "source": {"fixed": True},
            "dependencies": {"fixed": True},
            "execution": execution_identity(),
            "host_state": host,
        },
        "actor_layout": [],
        "actor_digest": hashlib.sha256(str(host["env_steps"]).encode()).hexdigest(),
        "files": {},
        "collection": {},
        "layout": [],
        "recording_token": None,
        "counters": {
            "updates": host["completed_updates"],
            "env_steps": host["env_steps"],
            "completed_blocks": host["completed_blocks"],
            "learning_blocks": host["learning_blocks"],
        },
    }
    details["checkpoint_id"] = hashlib.sha256(
        checkpoints._json_bytes(details)
    ).hexdigest()
    directory = root / "checkpoints" / details["checkpoint_id"]
    directory.mkdir(parents=True)
    io_helpers.atomic_json(directory / "checkpoint_details.json", details)
    return details


def _pqn_actor(root: Path, ancestor: dict[str, Any]) -> Path:
    from marl_battlegrounds.training import checkpoints

    details: dict[str, Any] = {
        "schema_version": 1,
        "kind": "actor",
        "schemas": ancestor["schemas"],
        "metadata": {
            "run_id": "one-run",
            "seed": 42,
            "env_steps": ancestor["counters"]["env_steps"],
            "checkpoint_id": ancestor["checkpoint_id"],
            "optimizer_steps": ancestor["counters"]["updates"],
        },
        "actor_layout": [],
        "actor_digest": ancestor["actor_digest"],
        "input_scale": 1.0,
        "spawn_frame": "left",
        "epsilon": 0.0,
        "tie_rule": "first_legal_maximum",
        "files": {},
    }
    details["checkpoint_id"] = hashlib.sha256(
        checkpoints._json_bytes(details)
    ).hexdigest()
    path = root / "actors" / ancestor["checkpoint_id"]
    path.mkdir(parents=True)
    io_helpers.atomic_json(path / "actor_details.json", details)
    return path


def _pqn_validation(
    root: Path,
    ancestor: dict[str, Any],
    actor: Path,
    panel: FrozenPanel,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    from marl_battlegrounds.training import checkpoints, validation

    identity = checkpoints.artifact_identity(actor)
    key = ancestor["checkpoint_id"]
    task = validation.validation_task_description(
        checkpoint_id=key,
        actor_digest=identity["actor_digest"],
        env_steps=ancestor["counters"]["env_steps"],
        panel_digest=panel.digest,
        purpose="routine",
        seed_pairs=1,
        members=tuple((member.name, member.actor_digest) for member in panel.members),
    )
    directory = root / "validation" / f"routine-{key}"
    directory.mkdir(parents=True)
    io_helpers.atomic_json(directory / "task.json", task)
    paths = [str(directory / f"opponent-{index}" / "run") for index in range(2)]
    for path in paths:
        Path(path).mkdir(parents=True)
    summary = {
        **task,
        "complete": True,
        "score": 0.5,
        "ci_low": 0.25,
        "ci_high": 0.75,
        "games": 20,
        "pass_paths": paths,
        "method": "pqn_vdn",
        "optimizer_steps": identity["optimizer_steps"],
    }
    io_helpers.atomic_json(directory / "validation_summary.json", summary)

    def saved(parent: Path, pass_id: str) -> Path:
        return parent / "run"

    monkeypatch.setattr(validation, "_saved_run", saved)
    monkeypatch.setattr(validation, "_pending", Mock(return_value=0))
    return {**summary, "elapsed_seconds": 1.0}


def test_saved_pqn_host_counts_follow_offset_boundaries_beyond_int32(
    tmp_path: Path,
) -> None:
    for steps in (0, 4096, 4224, 8320, 12416, 16_640):
        root = tmp_path / f"steps-{steps}"
        # A save at the budget still owes its final capture.
        final = False if steps == 16_640 else None
        details = _pqn_description(root, _pqn_host(steps, pending=final))
        io_helpers.validate_host_state(root, details, panel=None)
    batch, rounds = 1024, 2**30
    steps = batch * rounds
    big = _pqn_host(steps, batch=batch, pending=False)
    assert big["used_td_pairs"] > 2**31
    details = _pqn_description(tmp_path / "big", big, batch=batch, total=steps)
    io_helpers.validate_host_state(tmp_path / "big", details, panel=None)
    for field, value in (
        ("learning_blocks", 2),
        ("completed_blocks", 4),
        ("used_sequences", 1),
        ("used_td_pairs", 1),
        ("used_prefix_td_pairs", 0),
        ("used_agent_utilities", 1),
        (
            "used_exposure",
            {"by_stage": [0] * 17, "by_source": [0], "by_opponent": [0] * 21},
        ),
    ):
        host = _pqn_host(8320, pending=None)
        host[field] = value
        root = tmp_path / field
        bad = _pqn_description(root, host)
        with pytest.raises(ValueError, match="host experience or sample counts"):
            io_helpers.validate_host_state(root, bad, panel=None)
    unreachable = _pqn_host(8320, pending=None)
    unreachable.update(env_steps=8192)
    root = tmp_path / "unreachable"
    with pytest.raises(ValueError, match="host experience or sample counts"):
        io_helpers.validate_host_state(
            root, _pqn_description(root, unreachable), panel=None
        )


def _pqn_chain(
    root: Path, steps: tuple[int, ...], monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], FrozenPanel]:
    panel = _panel()
    parent: dict[str, Any] | None = None
    actors: dict[str, str] = {}
    results: list[dict[str, Any]] = []
    for point in steps:
        ancestor = _pqn_description(
            root, _pqn_host(point, pending=True), parent=parent, panel=True
        )
        actor = _pqn_actor(root, ancestor)
        actors[ancestor["checkpoint_id"]] = str(actor)
        results.append(_pqn_validation(root, ancestor, actor, panel, monkeypatch))
        parent = ancestor
    host = _pqn_host(16_640, pending=None)
    host["actors"] = actors
    host["routine_results"] = results
    host["validation_games"] = 20 * len(results)
    final = cast(dict[str, Any], parent)
    host["final_actor"] = actors[final["checkpoint_id"]]
    return _pqn_description(root, host, parent=final, panel=True), panel


def test_saved_pqn_routine_results_follow_the_offset_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    good, panel = _pqn_chain(tmp_path / "good", (0, 4224, 8320, 16_640), monkeypatch)
    io_helpers.validate_host_state(tmp_path / "good", good, panel=panel)
    bad, panel = _pqn_chain(tmp_path / "bad", (0, 4224, 8192, 16_640), monkeypatch)
    with pytest.raises(ValueError, match="routine validation schedule"):
        io_helpers.validate_host_state(tmp_path / "bad", bad, panel=panel)


def test_pqn_initial_collection_has_its_own_label() -> None:
    text = io_helpers.progress_text(
        {"phase": "initial_collection", "learning_blocks": 0, "completed_updates": 0}
    )
    assert "Initial Random Collection" in text
    assert "Optimizer Steps: 0 | Learning Blocks: 0" in text
