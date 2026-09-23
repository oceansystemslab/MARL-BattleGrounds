"""Check the bounded MAPPO screen's host bookkeeping without learning runs.

The fixtures cover all twelve B/T shapes, exact even update budgets and shared
capture points, invalid measurements, frozen package checks, read-only status,
and failure/stop boundaries. Mock jobs never call a policy, evaluator or GPU.
The separate launcher tests own real process-group shutdown and source copies.
"""

from __future__ import annotations

# pyright: reportPrivateUsage=false
import copy
import json
import random
import shutil
import subprocess
import sys
import time
import venv
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from marl_battlegrounds.training import screen
from marl_battlegrounds.training._run_io import atomic_json


@pytest.fixture
def declaration() -> dict[str, Any]:
    return screen.screen_declaration()


def _calibration(declaration: dict[str, Any], seconds: float) -> dict[str, Any]:
    return {
        row["name"]: {"median_update_seconds": seconds} for row in declaration["cases"]
    }


def test_declaration_covers_twelve_fixed_shapes_without_global_random_changes() -> None:
    first = screen.screen_declaration()
    # Optional libraries may initialize their own state on the first import.
    before = random.getstate()
    assert first == screen.screen_declaration()
    assert random.getstate() == before
    assert len(first["cases"]) == 12
    assert {(row["num_envs"], row["rollout_length"]) for row in first["cases"]} == {
        (batch, length) for batch in (32, 512, 1024) for length in (16, 32, 64, 128)
    }
    assert (
        len(
            {
                first[name]
                for name in ("training_seed", "calibration_seed", "order_seed")
            }
        )
        == 3
    )
    assert first["training_seed"] == 19_044_601
    assert first["calibration_seed"] == 19_044_600
    assert first["max_elapsed_seconds"] == 7200
    assert first["numerical_stop_seconds"] == 7020
    assert first["calibration_limit_seconds"] == 1200
    assert first["training_maps"] == list(range(42))
    assert first["validation_maps"] == list(range(42, 47))
    assert first["base_config"]["curriculum"] is False
    assert first["base_config"]["shaping_mode"] == "score_delta"
    assert first["base_config"]["shaping_coefficient"] == 0.01
    assert first["base_config"]["ppo"]["input_scale"] == 0.01
    assert first["base_config"]["ppo"]["value_normalization"] is False


@pytest.mark.parametrize("batch", [32, 512, 1024])
@pytest.mark.parametrize("length", [16, 32, 64, 128])
def test_calibration_counts_cold_reset_warmup_and_five_timed_blocks(
    declaration: dict[str, Any], batch: int, length: int
) -> None:
    from marl_battlegrounds.training.runner import config_from_dict

    config = screen.calibration_config(
        declaration, {"num_envs": batch, "rollout_length": length}
    )
    resolved = config_from_dict(config)
    warm_blocks = (300 + length - 1) // length
    assert resolved.total_env_steps == (1 + warm_blocks + 5) * batch * length
    assert warm_blocks * length >= 300
    assert resolved.seed == declaration["calibration_seed"]
    assert resolved.num_envs == batch and resolved.ppo.rollout_length == length
    assert resolved.random_diagnostic_seed_pairs is None
    assert resolved.random_initialization_result is None
    assert resolved.checkpoint_env_steps == ()


@pytest.mark.parametrize(
    "seconds,anchor", [(1.0, 131_072), (0.1, 1_048_576), (100.0, None)]
)
def test_budgets_freeze_even_updates_and_exact_shared_captures(
    tmp_path: Path, declaration: dict[str, Any], seconds: float, anchor: int | None
) -> None:
    from marl_battlegrounds.training.runner import config_from_dict

    original = copy.deepcopy(declaration)
    calibration = _calibration(declaration, seconds)
    result = screen.resolve_budgets(declaration, calibration, tmp_path)
    assert declaration == original
    assert result["shared_env_steps"] == anchor
    assert result["declaration_digest"] == screen._digest(declaration)
    assert len(result["cases"]) == 12
    expected_passes = 1
    for index, row in enumerate(result["cases"]):
        updates = row["updates"]
        config = config_from_dict(row["config"])
        assert updates >= 2 and updates % 2 == 0
        assert updates * seconds <= 300 < (updates + 2) * seconds
        assert (
            row["total_env_steps"] == updates * row["num_envs"] * row["rollout_length"]
        )
        points = sorted(
            {row["total_env_steps"] // 2, row["total_env_steps"]}
            | ({anchor} if anchor else set())
        )
        assert list(config.checkpoint_env_steps) == points
        assert all(
            point % (config.num_envs * config.ppo.rollout_length) == 0
            for point in points
        )
        assert config.total_env_steps == row["total_env_steps"]
        assert config.seed == declaration["training_seed"]
        assert config.random_diagnostic_seed_pairs == 4
        assert config.random_initialization_result == (
            None if index == 0 else str(tmp_path / "shared/initialization.json")
        )
        assert row["config_digest"] == screen._digest(row["config"])
        expected_passes += len(points)
    assert result["diagnostic_passes"] == expected_passes <= 37
    assert result["diagnostic_games"] == expected_passes * 40 <= 1480


@pytest.mark.parametrize(
    "value", [True, None, "fast", 0, -1, float("inf"), float("nan"), 151.0]
)
def test_invalid_or_infeasible_calibration_never_yields_budgets(
    tmp_path: Path, declaration: dict[str, Any], value: object
) -> None:
    calibration = _calibration(declaration, 1.0)
    calibration[declaration["cases"][0]["name"]]["median_update_seconds"] = value
    with pytest.raises(ValueError):
        screen.resolve_budgets(declaration, calibration, tmp_path)


def _package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "screen with spaces"
    (root / "source").mkdir(parents=True)
    (root / "source/module.py").write_text("fixed = True\n")
    (root / "logs").mkdir()
    atomic_json(root / "declaration.json", {"fixed": True})
    scripts = {f"{name}.sh": "command\n" for name in screen._SCRIPT_ACTIONS}
    for name, text in scripts.items():
        (root / name).write_text(text)
    runtime = {
        "package_file": str(root / "source/module.py"),
        "identity": {"source": "fixed"},
    }
    gpu = {
        "uuid": "GPU-internal",
        "name": "NVIDIA GeForce RTX 5090",
        "pci_bus_id": "01:00",
    }
    monkeypatch.setattr(screen._launch, "_runtime", Mock(return_value=runtime))
    monkeypatch.setattr(screen._launch, "_gpu", Mock(return_value=gpu))
    atomic_json(
        root / "screen_package.json",
        {
            "schema_version": 1,
            "package_path": str(root),
            "origin": {"files": screen._launch._files(root / "source")},
            "runtime": runtime,
            "gpu": gpu,
            "cpu_affinity": [0],
            "declaration_sha256": screen._launch._hash(root / "declaration.json"),
            "scripts": {name: screen._launch._hash(root / name) for name in scripts},
        },
    )
    return root


@pytest.mark.parametrize(
    "changed", ["source", "declaration", "script", "runtime", "gpu"]
)
def test_package_rejects_changed_frozen_inputs_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    root = _package(tmp_path, monkeypatch)
    screen.validate_screen_package(root)
    if changed in {"runtime", "gpu"}:
        monkeypatch.setattr(
            screen._launch, f"_{changed}", Mock(return_value={"changed": True})
        )
    else:
        path = {
            "source": root / "source/module.py",
            "declaration": root / "declaration.json",
            "script": root / "launch.sh",
        }[changed]
        path.write_text("changed")
    spawn = Mock(side_effect=AssertionError("Changed source must not launch"))
    monkeypatch.setattr(screen.subprocess, "Popen", spawn)
    with pytest.raises(ValueError):
        screen.start_screen(root)
    spawn.assert_not_called()
    assert not (root / "study.json").exists()


def test_status_imports_no_jax_and_does_not_change_saved_files(tmp_path: Path) -> None:
    atomic_json(
        tmp_path / "study.json",
        {
            "status": "running",
            "phase": "training",
            "started_at_seconds": time.time() - 5,
            "deadline_at": time.time() + 30,
            "elapsed_seconds": 5.0,
            "cases": {},
        },
    )
    before = (tmp_path / "study.json").read_bytes()
    command = (
        "import sys,json; "
        "from marl_battlegrounds.training.screen import screen_status; "
        f"result=screen_status({str(tmp_path)!r}); "
        "assert 'jax' not in sys.modules; print(json.dumps(result))"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", command],
        check=True,
        text=True,
        capture_output=True,
    )
    status = json.loads(result.stdout)
    assert status["status"] != "complete"
    assert status["alive"] is False
    assert status["elapsed_seconds"] >= 5
    assert (tmp_path / "study.json").read_bytes() == before


@pytest.mark.parametrize(
    "reason,code,exception",
    [
        ("deadline", 124, TimeoutError),
        ("signal", 143, InterruptedError),
        (None, 7, RuntimeError),
    ],
)
def test_job_failure_is_not_success_and_keeps_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reason: str | None,
    code: int,
    exception: type[Exception],
) -> None:
    job = tmp_path / "job"
    job.mkdir()
    record: dict[str, Any] = {"state": "exited", "exit_code": code}
    if reason is not None:
        record.update(stop_reason=reason, stop_signal=15)
    atomic_json(job / "process.json", record)
    child = Mock(returncode=code)
    child.poll.return_value = code
    monkeypatch.setattr(screen.subprocess, "Popen", Mock(return_value=child))
    monkeypatch.setattr(screen, "_write_study", Mock())
    with pytest.raises(exception):
        screen._run_job(
            tmp_path, {}, job, mode="trial", deadline=time.time() + 10, stopped=[False]
        )


@pytest.mark.parametrize("stopped,remaining", [(True, 10), (False, -1)])
def test_stopped_or_expired_job_never_spawns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stopped: bool, remaining: int
) -> None:
    spawn = Mock(side_effect=AssertionError("Stopped work must not start"))
    monkeypatch.setattr(screen.subprocess, "Popen", spawn)
    with pytest.raises(InterruptedError if stopped else TimeoutError):
        screen._run_job(
            tmp_path,
            {},
            tmp_path / "job",
            mode="trial",
            deadline=time.time() + remaining,
            stopped=[stopped],
        )
    spawn.assert_not_called()


def _study(
    root: Path, declaration: dict[str, Any], *, expired: bool = False
) -> dict[str, Any]:
    started = time.time() - (7300 if expired else 10)
    study: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "phase": "training",
        "started_at_seconds": started,
        "deadline_at": started + 7200,
        "stop_at": started + 7020,
        "active_job": None,
        "current_case": None,
        "cases": {
            case["name"]: {"status": "unstarted"} for case in declaration["cases"]
        },
    }
    atomic_json(root / "declaration.json", declaration)
    atomic_json(root / "launch_time.json", {"started_at_seconds": started})
    atomic_json(root / "study.json", study)
    (root / "jobs").mkdir(exist_ok=True)
    (root / "configs").mkdir(exist_ok=True)
    return study


@pytest.mark.parametrize(
    "changed", ["started_at_seconds", "deadline_at", "stop_at", "cases"]
)
def test_saved_clock_and_case_membership_cannot_change_on_resume(
    tmp_path: Path, declaration: dict[str, Any], changed: str
) -> None:
    study = _study(tmp_path, declaration)
    screen._validate_clock(tmp_path, study, declaration)
    if changed == "cases":
        study["cases"].pop(declaration["cases"][0]["name"])
    else:
        study[changed] += 1
    with pytest.raises(ValueError, match="changed"):
        screen._validate_clock(tmp_path, study, declaration)


def _completed_calibration(
    root: Path, declaration: dict[str, Any], case: dict[str, Any]
) -> dict[str, Any]:
    job = root / "calibration" / case["name"] / "attempt-1"
    (job / "run").mkdir(parents=True)
    config = screen.calibration_config(declaration, case)
    updates = config["total_env_steps"] // (case["num_envs"] * case["rollout_length"])
    rows = [
        {"update_index": index + 1, "collection_update_seconds": 1.0}
        for index in range(updates)
    ]
    (job / "run/training_updates.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    result = {
        "schema_version": 1,
        "num_envs": case["num_envs"],
        "rollout_length": case["rollout_length"],
        "median_update_seconds": 1.0,
        "timed_seconds": [1.0] * 5,
        "cold_first_call_seconds": 1.0,
        "overhead_seconds": 0.5,
        "save_seconds": 0.1,
        "memory": None,
        "env_steps": config["total_env_steps"],
        "run_dir": str(job / "run"),
        "validation_seconds": 1.0,
        "engineering_validation_games": 40,
        "scope": "Engineering calibration only; no model selection",
    }
    atomic_json(job / "config.json", config)
    atomic_json(job / "calibration_result.json", result)
    atomic_json(job / "process.json", {"state": "exited", "exit_code": 0})
    atomic_json(job / "run/status.json", {"status": "complete", "phase": "complete"})
    return result


def test_completed_calibration_publication_gap_reuses_original_measurements(
    tmp_path: Path, declaration: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    case = declaration["cases"][0]
    assert screen._read_calibration(tmp_path, case, declaration) is None
    result = _completed_calibration(tmp_path, declaration, case)
    spawn = Mock(side_effect=AssertionError("Completed calibration must not rerun"))
    monkeypatch.setattr(screen.subprocess, "Popen", spawn)
    assert screen._read_calibration(tmp_path, case, declaration) == result
    published = tmp_path / "calibration" / case["name"] / "calibration.json"
    first = published.read_bytes()
    assert screen._read_calibration(tmp_path, case, declaration) == result
    assert published.read_bytes() == first
    spawn.assert_not_called()


@pytest.mark.parametrize(
    "changed", ["config", "samples", "process", "extra_result", "published"]
)
def test_calibration_recovery_rejects_changed_or_ambiguous_evidence(
    tmp_path: Path, declaration: dict[str, Any], changed: str
) -> None:
    case = declaration["cases"][0]
    result = _completed_calibration(tmp_path, declaration, case)
    directory = tmp_path / "calibration" / case["name"]
    job = directory / "attempt-1"
    if changed == "config":
        config = screen._launch._read(job / "config.json")
        config["seed"] += 1
        atomic_json(job / "config.json", config)
    elif changed == "samples":
        result["timed_seconds"][0] = 2.0
        atomic_json(job / "calibration_result.json", result)
    elif changed == "process":
        atomic_json(
            job / "process.json", {"state": "exited", "exit_code": 0, "stop_signal": 15}
        )
    elif changed == "extra_result":
        (directory / "attempt-2").mkdir()
        atomic_json(directory / "attempt-2/calibration_result.json", result)
    else:
        atomic_json(
            directory / "calibration.json", {**result, "median_update_seconds": 2.0}
        )
    with pytest.raises(ValueError):
        screen._read_calibration(tmp_path, case, declaration)


def _frozen_budgets(root: Path, declaration: dict[str, Any]) -> dict[str, Any]:
    calibration = {
        case["name"]: _completed_calibration(root, declaration, case)
        for case in declaration["cases"]
    }
    budgets = screen.resolve_budgets(declaration, calibration, root)
    budgets["forecast"] = {"preserved": True}
    for row in budgets["cases"]:
        atomic_json(root / row["config_path"], row["config"])
    atomic_json(root / "budgets.json", budgets)
    return budgets


@pytest.mark.parametrize(
    "failure,expected_status",
    [
        (RuntimeError, "failed"),
        (InterruptedError, "stopped"),
        (TimeoutError, "expired"),
    ],
)
def test_budget_publication_gap_rebinds_then_stops_without_starting_next_job(
    tmp_path: Path,
    declaration: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    failure: type[Exception],
    expected_status: str,
) -> None:
    from marl_battlegrounds.training import analysis

    _study(tmp_path, declaration)
    budgets = _frozen_budgets(tmp_path, declaration)
    run = Mock(side_effect=failure("injected job stop"))
    monkeypatch.setattr(screen, "_run_job", run)
    monkeypatch.setattr(analysis, "analyze_screen", Mock(return_value={}))
    monkeypatch.setattr(screen, "_print_progress", Mock())
    before = (tmp_path / "budgets.json").read_bytes()
    with pytest.raises(failure, match="injected job stop"):
        screen.run_mappo_screen(tmp_path, resume=True)
    run.assert_called_once()
    after = screen._launch._read(tmp_path / "study.json")
    assert after["status"] == expected_status
    assert after["budgets_digest"] == screen._digest(budgets)
    assert (tmp_path / "budgets.json").read_bytes() == before
    assert run.call_args.kwargs["deadline"] == after["stop_at"]


def test_unbound_changed_budget_is_rejected_before_any_job(
    tmp_path: Path, declaration: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from marl_battlegrounds.training import analysis

    _study(tmp_path, declaration)
    budgets = _frozen_budgets(tmp_path, declaration)
    budgets["cases"][0]["updates"] += 2
    atomic_json(tmp_path / "budgets.json", budgets)
    run = Mock(side_effect=AssertionError("Changed budget must not run"))
    monkeypatch.setattr(screen, "_run_job", run)
    monkeypatch.setattr(analysis, "analyze_screen", Mock(return_value={}))
    monkeypatch.setattr(screen, "_print_progress", Mock())
    with pytest.raises(ValueError, match="Unbound budgets differ"):
        screen.run_mappo_screen(tmp_path, resume=True)
    run.assert_not_called()


def test_expired_resume_only_writes_reports_and_preserves_original_clock(
    tmp_path: Path, declaration: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from marl_battlegrounds.training import analysis

    before = _study(tmp_path, declaration, expired=True)
    run = Mock(side_effect=AssertionError("Expired resume must not collect"))
    report = Mock(return_value={})
    monkeypatch.setattr(screen, "_run_job", run)
    monkeypatch.setattr(analysis, "analyze_screen", report)
    result = screen.run_mappo_screen(tmp_path, resume=True)
    assert result["status"] == "expired"
    assert all(
        result[key] == before[key]
        for key in ("started_at_seconds", "stop_at", "deadline_at")
    )
    run.assert_not_called()
    report.assert_called_once_with(tmp_path)


def test_explicit_resume_reuses_deadline_and_frozen_environment(
    tmp_path: Path, declaration: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _package(tmp_path, monkeypatch)
    original = _study(root, declaration)
    manifest = screen._launch._read(root / "screen_package.json")
    manifest["declaration_sha256"] = screen._launch._hash(root / "declaration.json")
    atomic_json(root / "screen_package.json", manifest)
    spawn = Mock(return_value=Mock(pid=123))
    monkeypatch.setattr(screen.subprocess, "Popen", spawn)
    monkeypatch.setattr(
        screen,
        "process_identity",
        Mock(return_value={"pid": 123, "start_ticks": "fixed", "boot_id": "fixed"}),
    )
    monkeypatch.setattr(screen._launch, "_alive", Mock(return_value=False))
    with pytest.raises(ValueError, match="use resume"):
        screen.start_screen(root)
    spawn.assert_not_called()
    result = screen.start_screen(root, resume=True)
    assert result["mode"] == "resume"
    assert "--resume" in result["command"]
    assert spawn.call_args.kwargs["stdin"] == subprocess.DEVNULL
    assert spawn.call_args.kwargs["start_new_session"] is True
    env = spawn.call_args.kwargs["env"]
    assert env["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.85"
    assert env["JAX_COMPILATION_CACHE_DIR"] == str(root / "compilation-cache")
    saved = screen._launch._read(root / "study.json")
    assert saved == original


def test_active_nested_job_blocks_resume_without_starting_another_owner(
    tmp_path: Path, declaration: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _package(tmp_path, monkeypatch)
    _study(root, declaration)
    manifest = screen._launch._read(root / "screen_package.json")
    manifest["declaration_sha256"] = screen._launch._hash(root / "declaration.json")
    atomic_json(root / "screen_package.json", manifest)
    job = root / "jobs" / declaration["cases"][0]["name"]
    job.mkdir()
    atomic_json(job / "process.json", {"owned_orphan": True})

    def alive(record: dict[str, Any]) -> bool:
        return bool(record.get("owned_orphan"))

    monkeypatch.setattr(screen._launch, "_alive", alive)
    spawn = Mock(side_effect=AssertionError("Live owned work must block resume"))
    monkeypatch.setattr(screen.subprocess, "Popen", spawn)
    with pytest.raises(RuntimeError, match="active owned process"):
        screen.start_screen(root, resume=True)
    spawn.assert_not_called()
    status = screen.screen_status(root)
    assert status["status"] == "orphaned_workers"
    assert status["alive"] is False
    assert status["active_owned_jobs"] == [str(job)]


def test_job_display_failure_stops_and_waits_for_its_fresh_supervisor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = Mock(returncode=None)
    child.poll.return_value = None
    monkeypatch.setattr(screen.subprocess, "Popen", Mock(return_value=child))
    monkeypatch.setattr(screen, "_write_study", Mock())
    monkeypatch.setattr(
        screen, "_print_progress", Mock(side_effect=ValueError("bad status"))
    )
    with pytest.raises(ValueError, match="bad status"):
        screen._run_job(
            tmp_path,
            {},
            tmp_path / "job",
            mode="trial",
            deadline=time.time() + 10,
            stopped=[False],
        )
    child.send_signal.assert_called_once_with(15)
    child.wait.assert_called_once_with(timeout=20)


def test_prepare_builds_isolated_cpu_package_and_generated_status_command(
    tmp_path: Path, declaration: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "temporary candidate"
    package = repository / "src/marl_battlegrounds"
    training = package / "training"
    training.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (training / "__init__.py").write_text("")
    (training / "checkpoints.py").write_text(
        "def runtime_identity():\n"
        "    return {'source':'fixture-source','dependencies':{'stdlib':'only'}}\n"
    )
    for name in ("screen.py", "_launch.py", "_run_io.py"):
        shutil.copy2(Path(screen.__file__).parent / name, training / name)
    (repository / ".gitignore").write_text("docs/dev/milestone*.md\n")
    private = repository / "docs/dev/milestone_fixture.md"
    private.parent.mkdir(parents=True)
    private.write_text("Private plan must stay in the original checkout")
    for arguments in (
        ["init", "--quiet"],
        ["add", "."],
        [
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "Initial fixture",
        ],
    ):
        subprocess.run(
            ["git", "-C", str(repository), *arguments], check=True, capture_output=True
        )
    (training / "candidate.py").write_text("uncommitted = True\n")
    before = screen._launch._files(repository / ".git")
    gpu = {
        "uuid": "GPU-internal",
        "name": "NVIDIA GeForce RTX 5090",
        "pci_bus_id": "01:00",
    }
    monkeypatch.setattr(screen._launch, "_gpu", Mock(return_value=gpu))
    monkeypatch.setattr(screen, "screen_declaration", Mock(return_value=declaration))
    content = Mock(return_value={"fixture": "No simulator content loaded"})
    monkeypatch.setattr(screen, "_content_identity", content)

    def install(source: Path, environment: Path, python: str) -> None:
        assert python == sys.executable
        venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
        site = next((environment / "lib").glob("python*/site-packages"))
        (site / "screen_source.pth").write_text(str(source / "src") + "\n")

    monkeypatch.setattr(screen._launch, "_install", install)
    output = tmp_path / "prepared package with spaces"
    result = screen.prepare_screen(repository, output, gpu_uuid="GPU-internal")
    manifest = screen._launch._read(output / "screen_package.json")
    assert manifest["origin"]["kind"] == "working_tree"
    assert manifest["origin"]["files"] == screen._launch._files(output / "source")
    assert manifest["runtime"]["package_file"] == str(
        output / "source/src/marl_battlegrounds/__init__.py"
    )
    assert (
        output / "source/src/marl_battlegrounds/training/candidate.py"
    ).read_text() == "uncommitted = True\n"
    assert not (output / "source/docs/dev/milestone_fixture.md").exists()
    assert private.read_text() == "Private plan must stay in the original checkout"
    assert before == screen._launch._files(repository / ".git")
    content.assert_called_once_with(output / ".venv/bin/python", output / "source")
    assert not (output / "study.json").exists()
    assert not (output / "process.json").exists()
    for name in screen._SCRIPT_ACTIONS:
        script = output / f"{name}.sh"
        assert script.exists() and script.stat().st_mode & 0o111
        assert str(output / ".venv/bin/python") in script.read_text()
        assert str(output / f"{name}.sh") in result[name]
    executed = subprocess.run(
        ["bash", str(output / "status.sh"), "--json"],
        check=True,
        text=True,
        capture_output=True,
    )
    status = json.loads(executed.stdout)
    assert status["status"] == "not_started"
    assert not status["alive"] and not status["active_owned_jobs"]
    assert screen.validate_screen_package(output) == manifest


@pytest.mark.parametrize("live_owner", ["controller", "job", "none"])
def test_stop_uses_verified_live_owner_and_never_a_saved_orphan_group(
    tmp_path: Path,
    declaration: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    live_owner: str,
) -> None:
    _study(tmp_path, declaration)
    controller = {"pid": 100, "start_ticks": "controller", "boot_id": "same"}
    supervisor = {"pid": 200, "start_ticks": "job", "boot_id": "same"}
    stale = {"pid": 201, "start_ticks": "old", "boot_id": "same"}
    atomic_json(tmp_path / "process.json", {"trainer": controller})
    for name, identity in (("active", supervisor), ("stale", stale)):
        job = tmp_path / "jobs" / name
        job.mkdir()
        atomic_json(job / "process.json", {"process": identity, "orphan_group": 300})

    def alive(record: dict[str, Any]) -> bool:
        owner = record.get("process", record.get("trainer"))
        return (
            (owner == controller and live_owner == "controller")
            or (owner == supervisor and live_owner == "job")
            or record.get("orphan_group") == 300
        )

    verify = Mock(side_effect=alive)
    send = Mock()
    group = Mock(side_effect=AssertionError("A saved group must not authorize signals"))
    monkeypatch.setattr(screen._launch, "_alive", verify)
    monkeypatch.setattr(screen.os, "kill", send)
    monkeypatch.setattr(screen.os, "killpg", group)
    result = screen.stop_screen(tmp_path)
    expected = {"controller": [100], "job": [200], "none": []}[live_owner]
    assert result["stop_requested_pids"] == expected
    assert send.call_args_list == [((pid, 15),) for pid in expected]
    verify.assert_any_call({"process": controller})
    if live_owner != "controller":
        verify.assert_any_call({"process": supervisor})
        verify.assert_any_call({"process": stale})
    if live_owner == "none":
        assert "manual cleanup" in result["stop_notice"]
    group.assert_not_called()


def test_stopped_speed_check_cannot_look_like_finished_experiment() -> None:
    text = screen.screen_status_text(
        {
            "status": "failed",
            "phase": "reporting",
            "calibrations_complete": 12,
            "trials_started": False,
            "cases": {f"case-{i}": {"status": "unstarted"} for i in range(12)},
            "elapsed_seconds": 976,
            "failure": "Estimated work exceeds the time allowance",
            "run": {
                "phase": "complete",
                "env_steps": 589824,
                "total_env_steps": 589824,
            },
        }
    )
    assert "Study Stopped Before Comparison Trials" in text
    assert "Speed Checks Finished: 12 / 12" in text
    assert "Comparison Trials Finished: 0 / 12" in text
    assert "16m 16s" in text and "Reason For Stopping" in text
    assert "Run Finished" not in text and "100.0%" not in text


def test_human_screen_status_shows_work_and_unknown_estimates() -> None:
    text = screen.screen_status_text(
        {
            "status": "running",
            "phase": "training",
            "current_case": "b512-t128",
            "calibrations_complete": 12,
            "cases": {"b512-t128": {"status": "running"}},
            "elapsed_seconds": 100,
            "remaining_deadline_seconds": 90,
        }
    )
    assert "512 Parallel Games; 128 Steps Per Collection" in text
    assert "Study Time Left (Estimate): Estimating" in text
    assert "1m 30s" in text


def test_stop_request_and_orphan_notice_remain_visible() -> None:
    text = screen.screen_status_text(
        {"status": "running", "phase": "training", "stop_requested_pids": [123]}
    )
    assert "Stop Requested; Waiting" in text
    assert "manual cleanup" in screen.screen_status_text(
        {"status": "orphaned_workers", "stop_notice": "manual cleanup required"}
    )
