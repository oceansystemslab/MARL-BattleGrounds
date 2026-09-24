"""Check fixed source preparation and unattended process ownership on CPU.

These small fixtures test source/config/panel identity, isolated imports,
explicit GPU selection without GPU allocation, durable failure status, detached
stdin/log behavior, duplicate launch rejection and recovery marker precedence.
Status distinguishes live ownership, clean completion, failed exit and a killed
process without treating an older run record as the current process's result.
Private trainer groups keep validation workers owned through stop signals,
natural failure, forced shutdown and leader exit. Read-only orphan detection
blocks restart; cleanup never signals a stale PID or an unrelated group.
Working-source copies include current staged, unstaged and new public bytes
without changing Git. Fixed deadlines stop whole trainer groups and cannot be
reported as success when a child exits cleanly during a stop request.
Nested supervisors receive enough grace to stop their separate worker groups
after a controller crash, natural exit, stop request or deadline.
Short UTC lifecycle messages keep full process and cleanup facts in saved JSON.
PPO packages keep their exact import probe; QMIX packages use a probe that also
records Flashbax, and PQN-VDN packages use PPO's probe, whose dependency record
has no Flashbax. They do not install real training dependencies or start a
learning experiment.
"""

from __future__ import annotations

# pyright: reportPrivateUsage=false
import io
import json
import os
import signal
import subprocess
import sys
import tarfile
import time
import venv
from pathlib import Path
from types import SimpleNamespace
from typing import Any, BinaryIO, cast
from unittest.mock import Mock

import pytest

from marl_battlegrounds.training import _launch as launch
from marl_battlegrounds.training._run_io import atomic_json, process_identity


def _git(repository: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        env=dict(os.environ, GIT_OPTIONAL_LOCKS="0"),
    ).stdout


def _working_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "working source"
    repository.mkdir()
    _git(repository, "init", "--quiet")
    (repository / ".gitignore").write_text("artifacts/\nignored*\n__pycache__/\n")
    (repository / "code.py").write_text("version = 'committed'\n")
    (repository / "deleted.py").write_text("old = True\n")
    private = repository / "docs/dev/milestone_private.md"
    private.parent.mkdir(parents=True)
    private.write_text("Private even if accidentally tracked")
    _git(repository, "add", ".")
    _git(
        repository,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "Initial fixture",
    )
    return repository


def test_working_export_freezes_actual_candidate_without_git_changes(
    tmp_path: Path,
) -> None:
    repository = _working_repository(tmp_path)
    (repository / "code.py").write_text("version = 'staged'\n")
    _git(repository, "add", "code.py")
    (repository / "code.py").write_text("version = 'current'\n")
    (repository / "new.py").write_text("new = True\n")
    (repository / "ignored.txt").write_text("private")
    (repository / "deleted.py").unlink()
    (repository / "__pycache__").mkdir()
    (repository / "__pycache__/cache.pyc").write_bytes(b"bytecode")
    (repository / "artifacts").mkdir()
    before = launch._files(repository / ".git")
    destination = repository / "artifacts/source"
    record = launch.export_working_source(repository, destination)
    assert (destination / "code.py").read_text() == "version = 'current'\n"
    assert (destination / "new.py").read_text() == "new = True\n"
    assert record["files"] == launch._files(destination)
    assert set(record["files"]) == {".gitignore", "code.py", "new.py"}
    assert record["kind"] == "working_tree"
    assert record["commit"] == _git(repository, "rev-parse", "HEAD").decode().strip()
    assert (
        record["git_tree"]
        == _git(repository, "rev-parse", "HEAD^{tree}").decode().strip()
    )
    assert "MM code.py" in record["working_tree"]["status_porcelain"]
    assert before == launch._files(repository / ".git")
    assert (repository / "docs/dev/milestone_private.md").is_file()
    assert not (destination / ".git").exists()


@pytest.mark.parametrize("change", ["bytes", "index", "new_file"])
def test_working_export_rejects_changes_during_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    repository = _working_repository(tmp_path)
    copy = launch.shutil.copy2
    changed = False

    def mutate(source: Path, target: Path, *, follow_symlinks: bool = True) -> Path:
        nonlocal changed
        result = copy(source, target, follow_symlinks=follow_symlinks)
        if not changed:
            changed = True
            if change == "new_file":
                (repository / "new.py").write_text("new = True\n")
            else:
                (repository / "code.py").write_text("version = 'changed'\n")
                if change == "index":
                    _git(repository, "add", "code.py")
                    (repository / "code.py").write_text("version = 'committed'\n")
        return Path(result)

    monkeypatch.setattr(launch.shutil, "copy2", mutate)
    with pytest.raises(ValueError, match="changed while copying"):
        launch.export_working_source(repository, tmp_path / "copy")


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_working_export_rejects_source_links(tmp_path: Path, kind: str) -> None:
    repository = _working_repository(tmp_path)
    target = repository / "code.py" if kind == "file" else repository / "docs"
    (repository / "linked").symlink_to(target, target_is_directory=kind == "directory")
    with pytest.raises(ValueError, match="contain links"):
        launch.export_working_source(repository, tmp_path / "copy")


def test_environment_applies_only_explicit_memory_and_cache_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.1")
    monkeypatch.setenv("JAX_COMPILATION_CACHE_DIR", "/old/cache")
    monkeypatch.setenv("JAX_ENABLE_COMPILATION_CACHE", "false")
    cache = tmp_path / "cache"
    env = launch._environment(
        "GPU-internal", memory_fraction=0.85, compilation_cache=cache
    )
    assert env["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.85"
    assert env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert env["JAX_COMPILATION_CACHE_DIR"] == str(cache)
    assert env["JAX_ENABLE_COMPILATION_CACHE"] == "true"
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-internal"
    assert not cache.exists()
    default = launch._environment(None)
    assert "XLA_PYTHON_CLIENT_MEM_FRACTION" not in default
    assert "JAX_COMPILATION_CACHE_DIR" not in default
    assert "JAX_ENABLE_COMPILATION_CACHE" not in default
    assert os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.1"
    with pytest.raises(ValueError, match="absolute"):
        launch._environment(None, compilation_cache=Path("relative"))


@pytest.mark.parametrize(
    "fraction", [True, "0.85", 0, -0.1, 1.1, float("nan"), float("inf")]
)
def test_environment_rejects_invalid_memory_fraction(fraction: object) -> None:
    with pytest.raises(ValueError, match="Memory fraction"):
        launch._environment(None, memory_fraction=cast(float, fraction))


def _package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "package with spaces"
    source = root / "source"
    panel = root / "panel"
    source.mkdir(parents=True)
    panel.mkdir()
    (root / "logs").mkdir()
    (source / "module.py").write_text("source = 1\n")
    (panel / "actor").write_bytes(b"frozen actor")
    atomic_json(root / "config.json", {"seed": 42})
    for name in ("launch.sh", "status.sh", "resume.sh"):
        (root / name).write_text("fixed command\n")
    runtime = {"identity": {"source": "fixed", "dependencies": {"jax": "pinned"}}}
    gpu = {"uuid": "GPU-internal", "name": "NVIDIA GeForce RTX 5090", "pci": "01:00"}
    monkeypatch.setattr(launch, "_runtime", Mock(return_value=runtime))
    monkeypatch.setattr(launch, "_gpu", Mock(return_value=gpu))
    atomic_json(
        root / "launch_package.json",
        {
            "schema_version": 1,
            "package_path": str(root),
            "origin": {"files": launch._files(source)},
            "panel_files": launch._files(panel),
            "config_sha256": launch._hash(root / "config.json"),
            "scripts": {
                name: launch._hash(root / name)
                for name in ("launch.sh", "status.sh", "resume.sh")
            },
            "runtime": runtime,
            "gpu": gpu,
            "execution_environment": {
                name: launch._environment("GPU-internal")[name]
                for name in launch._EXECUTION_KEYS
            },
        },
    )
    return root


@pytest.mark.parametrize("changed", ["source", "panel", "config", "script", "runtime"])
def test_launch_rejects_changed_package_before_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    root = _package(tmp_path, monkeypatch)
    launch.validate_package(root)
    paths = {
        "source": root / "source/module.py",
        "panel": root / "panel/actor",
        "config": root / "config.json",
        "script": root / "launch.sh",
    }
    if changed == "runtime":
        monkeypatch.setattr(launch, "_runtime", Mock(return_value={"different": True}))
    else:
        paths[changed].write_bytes(b"changed")
    with pytest.raises(ValueError):
        launch.start(root)
    assert not (root / "process.json").exists()
    assert not (root / "run").exists()


def test_inventory_rejects_links_and_ignores_bytecode(tmp_path: Path) -> None:
    (tmp_path / "source.py").write_text("fixed")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__/cache.pyc").write_bytes(b"generated")
    assert set(launch._files(tmp_path)) == {"source.py"}
    (tmp_path / "alias").symlink_to(tmp_path / "source.py")
    with pytest.raises(ValueError, match="links"):
        launch._files(tmp_path)


def test_environment_discards_mutable_python_routing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "JAX_PLATFORM_NAME"):
        monkeypatch.setenv(name, "mutable-checkout")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "wrong-device")
    monkeypatch.setenv("JAX_DISABLE_JIT", "true")
    monkeypatch.setenv("XLA_FLAGS", "wrong flags")
    gpu = launch._environment("GPU-internal")
    assert gpu["CUDA_VISIBLE_DEVICES"] == "GPU-internal"
    assert gpu["JAX_PLATFORMS"] == "cuda,cpu"
    assert gpu["PYTHONNOUSERSITE"] == "1"
    assert gpu["JAX_DISABLE_JIT"] == "false"
    assert gpu["JAX_ENABLE_X64"] == "false"
    assert gpu["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert "XLA_FLAGS" not in gpu
    assert not any(
        name in gpu
        for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "JAX_PLATFORM_NAME")
    )
    assert launch._environment(None)["JAX_PLATFORMS"] == "cpu"


def test_gpu_selection_uses_uuid_and_preserves_bus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        launch.subprocess,
        "run",
        Mock(
            return_value=SimpleNamespace(
                stdout="GPU-external, NVIDIA RTX 5090, 00000000:04:00.0\n"
                "GPU-internal, NVIDIA RTX 5090, 00000000:01:00.0\n"
            )
        ),
    )
    assert launch._gpu("GPU-internal")["pci_bus_id"] == "00000000:01:00.0"
    with pytest.raises(ValueError, match="full UUID"):
        launch._gpu("0")


def test_export_records_commit_tree_and_exact_bytes_without_git_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w") as archive:
        entry = tarfile.TarInfo("src/module.py")
        entry.size = 9
        archive.addfile(entry, io.BytesIO(b"value = 1"))
    calls: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(command)
        if "archive" in command:
            cast(BinaryIO, kwargs["stdout"]).write(data.getvalue())
        return SimpleNamespace(
            stdout=b"a" * 40 + b"\n" if "commit}" in command[-1] else b"b" * 40 + b"\n"
        )

    monkeypatch.setattr(launch.subprocess, "run", run)
    result = launch._export_source(
        tmp_path / "checkout", "approved", tmp_path / "fixed"
    )
    assert result["commit"] == "a" * 40
    assert result["git_tree"] == "b" * 40
    assert result["files"] == launch._files(tmp_path / "fixed")
    assert [command[3] for command in calls] == ["rev-parse", "rev-parse", "archive"]
    assert (tmp_path / "fixed/src/module.py").read_bytes() == b"value = 1"


def test_install_uses_snapshot_lock_and_separate_copied_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def run(command: list[str], **kwargs: object) -> None:
        calls.append((command, kwargs))

    monkeypatch.setattr(launch.subprocess, "run", run)
    source, environment = tmp_path / "source", tmp_path / ".venv"
    launch._install(source, environment, sys.executable)
    command, kwargs = calls[0]
    assert command[:4] == ["uv", "sync", "--project", str(source)]
    assert "--frozen" in command and "--no-dev" in command
    assert kwargs["env"]["UV_PROJECT_ENVIRONMENT"] == str(environment)
    assert kwargs["env"]["UV_LINK_MODE"] == "copy"
    assert kwargs["env"]["UV_PYTHON_DOWNLOADS"] == "never"


def test_actual_isolated_import_ignores_later_development_changes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "snapshot"
    package = source / "src/marl_battlegrounds"
    (package / "training").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "training/__init__.py").write_text("")
    (package / "training/checkpoints.py").write_text(
        "def runtime_identity():\n"
        "    return {'source':'original-source','dependencies':{'example':'1.0'}}\n"
    )
    environment = tmp_path / "fixed environment"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
    site = next((environment / "lib").glob("python*/site-packages"))
    (site / "fixed_source.pth").write_text(str(source / "src") + "\n")
    python = environment / "bin/python"
    before = launch._runtime(python, source)
    development = tmp_path / "development"
    development.mkdir()
    (development / "checkpoints.py").write_text("changed development source")
    (development / "dependency.py").write_text("changed development environment")
    after = launch._runtime(python, source)
    assert after == before
    assert Path(after["package_file"]).is_relative_to(source)


def test_duplicate_live_launch_fails_and_pid_reuse_is_not_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _package(tmp_path, monkeypatch)
    identity = process_identity()
    atomic_json(root / "process.json", {"process": identity})
    with pytest.raises(RuntimeError, match="active training"):
        launch.start(root)
    assert launch.status(root)["alive"]
    changed = {**identity, "start_ticks": "different"}
    assert not launch._alive({"process": changed})
    assert launch._alive({"process": changed, "trainer": identity})


def test_resume_requires_marker_checkpoint_and_never_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import marl_battlegrounds.training.checkpoints as checkpoints

    run = tmp_path / "run"
    run.mkdir()
    atomic_json(
        run / "latest_checkpoint.json",
        {
            "schema_version": 1,
            "checkpoint_id": "latest",
            "relative_path": "checkpoints/latest",
        },
    )
    atomic_json(run / "checkpoint_recovery.json", {"checkpoint_id": "recovering"})
    seen: list[Path] = []

    def read(path: Path) -> dict[str, str]:
        seen.append(path)
        return {"checkpoint_id": path.name, "kind": "learner"}

    monkeypatch.setattr(checkpoints, "read_checkpoint_details", read)
    assert launch._checkpoint(tmp_path, None) == run / "checkpoints/recovering"
    assert seen == [run / "checkpoints/recovering"]
    with pytest.raises(ValueError, match="original checkpoint"):
        launch._checkpoint(tmp_path, str(run / "checkpoints/latest"))
    (run / "checkpoint_recovery.json").unlink()
    assert launch._checkpoint(tmp_path, None) == run / "checkpoints/latest"
    monkeypatch.setattr(checkpoints, "read_checkpoint_details", _corrupt)
    with pytest.raises(ValueError, match="Corrupt chosen checkpoint"):
        launch._checkpoint(tmp_path, None)


def test_status_recovery_marker_overrides_old_completed_run(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    old_status = {"phase": "complete", "env_steps": 32, "total_env_steps": 32}
    marker = {"checkpoint_id": "earlier-checkpoint", "phase": "recovering"}
    process = {"state": "exited", "exit_code": 1}
    atomic_json(run / "status.json", old_status)
    atomic_json(run / "checkpoint_recovery.json", marker)
    atomic_json(tmp_path / "process.json", process)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
    result = launch.status(tmp_path)
    assert not result["alive"]
    assert result["run"] == old_status
    assert result["process"] == process
    assert result["recovery"] == marker
    assert result["effective_state"] == "recovering"
    assert result["progress"].startswith("Recovery unfinished")
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize(
    ("process_state", "code", "alive", "phase", "matching", "expected"),
    [
        ("exited", 7, False, "complete", False, "failed"),
        ("exited", -9, False, "training", True, "failed"),
        ("running", None, False, "complete", False, "stopped"),
        ("running", None, False, "training", True, "stopped"),
        ("exited", 0, False, "complete", True, "complete"),
        ("exited", 0, False, "training", True, "stopped"),
        ("exited", 0, False, "complete", False, "stopped"),
        ("exited", None, False, "complete", True, "stopped"),
        ("running", None, True, "complete", False, "running"),
        ("starting", None, True, "complete", False, "starting"),
        ("running", None, True, "training", True, "training"),
        ("running", None, True, "complete", True, "running"),
    ],
)
def test_status_uses_current_process_outcome_before_saved_run_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    process_state: str,
    code: int | None,
    alive: bool,
    phase: str,
    matching: bool,
    expected: str,
) -> None:
    trainer = {"pid": 123, "start_ticks": "current", "boot_id": "boot"}
    process = {"state": process_state, "exit_code": code, "trainer": trainer}
    run = {
        "status": "complete" if phase == "complete" else "running",
        "phase": phase,
        "process": trainer if matching else {**trainer, "start_ticks": "older"},
        "env_steps": 32,
        "total_env_steps": 32,
    }
    atomic_json(tmp_path / "process.json", process)
    (tmp_path / "run").mkdir()
    atomic_json(tmp_path / "run/status.json", run)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
    live = Mock(return_value=alive)
    monkeypatch.setattr(launch, "_alive", live)
    result = launch.status(tmp_path)
    assert result["effective_state"] == expected
    assert result["alive"] is alive
    assert result["process"] == process
    assert result["run"] == run
    live.assert_called_once_with(process)
    if expected in {"failed", "stopped"}:
        assert result["progress"].startswith(expected.capitalize())
    if expected in {"starting", "running"}:
        assert "waiting" in result["progress"]
    assert {path: path.read_bytes() for path in before} == before


def _corrupt(*_args: object) -> None:
    raise ValueError("Corrupt chosen checkpoint")


def _supervisor_command(
    package: Path,
    worker: str,
    *,
    crash_before_registration: bool = False,
    deadline_seconds: float | None = None,
    stop_grace_seconds: float | None = None,
) -> list[str]:
    injection = ""
    if crash_before_registration:
        injection = (
            "\nimport json,os,signal\n"
            "original=module.atomic_json\n"
            "def publish(path,value):\n"
            "    if 'trainer' in value:\n"
            f"        Path({str(package / 'unregistered.json')!r}).write_text("
            "json.dumps(value['trainer']))\n"
            "        os.kill(os.getpid(),signal.SIGKILL)\n"
            "    original(path,value)\n"
            "module.atomic_json=publish\n"
        )
    # Shorter than the launcher's real 10 s and 2 s limits so tests stay quick.
    # One second still gives a trainer that obeys a stop time to exit on a
    # loaded machine before the kill. A trainer that ignores the stop is killed
    # after 1 s and collected almost at once; even if every limit ran out, the
    # stop would end within 3 s, under the 5 s speed check in the stop test.
    # The tests themselves wait for the supervisor with no time limit.
    script = (
        "import importlib.util; from pathlib import Path; "
        f"spec=importlib.util.spec_from_file_location('launch_under_test',"
        f"{launch.__file__!r}); "
        "module=importlib.util.module_from_spec(spec); "
        "spec.loader.exec_module(module); "
        "module._STOP_TIMEOUT_SECONDS=1.0; module._KILL_TIMEOUT_SECONDS=1.0; "
        f"print('Launch module: '+str(module.__file__),flush=True); "
        + injection
        + f"raise SystemExit(module.supervise_command(Path({str(package)!r}),"
        f"{[sys.executable, '-I', '-c', worker]!r},deadline_at="
        + (
            "None"
            if deadline_seconds is None
            else f"module.time.time()+{deadline_seconds!r}"
        )
        + f",stop_grace_seconds={stop_grace_seconds!r}))"
    )
    return [sys.executable, "-I", "-c", script]


@pytest.mark.parametrize("code", [0, 7])
def test_detached_child_survives_parent_exit_and_records_stdout_and_failure(
    tmp_path: Path, code: int
) -> None:
    worker = (
        "import json,os,pathlib,sys,time; time.sleep(0.2); "
        f"pathlib.Path({str(tmp_path / 'worker.json')!r}).write_text("
        "json.dumps({'stdin':sys.stdin.read(),'session':os.getsid(0)})); "
        f"print('worker output',flush=True); sys.exit({code})"
    )
    supervisor = _supervisor_command(tmp_path, worker)
    parent = (
        "import subprocess; "
        f"log=open({str(tmp_path / 'process.log')!r},'ab',buffering=0); "
        f"child=subprocess.Popen({supervisor!r},"
        "stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,"
        "start_new_session=True); print(child.pid)"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", parent],
        check=True,
        capture_output=True,
        text=True,
    )
    pid = int(result.stdout.strip())
    try:
        while True:
            path = tmp_path / "process.json"
            if path.exists() and launch._read(path).get("state") == "exited":
                break
            time.sleep(0.05)
        record = launch._read(tmp_path / "process.json")
        assert record["state"] == "exited"
        assert record["exit_code"] == code
        worker_result = launch._read(tmp_path / "worker.json")
        assert worker_result == {"stdin": "", "session": record["trainer"]["pid"]}
        assert worker_result["session"] != pid
        assert "worker output" in (tmp_path / "process.log").read_text()
        assert record["process"]["pid"] == pid
        assert "finished_at" in record
    finally:
        if launch._alive({"process": process_identity(pid)}):
            os.killpg(pid, signal.SIGTERM)


def test_supervisor_records_preflight_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(launch, "validate_package", _corrupt)
    with pytest.raises(ValueError, match="Corrupt chosen checkpoint"):
        launch.main(["supervise", str(tmp_path)])
    record = launch._read(tmp_path / "process.json")
    assert record["state"] == "exited" and record["exit_code"] == 1
    assert "Corrupt chosen checkpoint" in record["error"]


def test_preparation_copies_panel_and_emits_space_safe_commands_without_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import marl_battlegrounds.training.validation as validation
    from marl_battlegrounds.training.runner import TrainConfig, config_to_dict

    old = tmp_path / "development panel"
    old.mkdir()
    members: list[SimpleNamespace] = []
    for name in ("Halfway", "Final"):
        actor = old / name
        actor.mkdir()
        (actor / "weights").write_text(name)
        members.append(SimpleNamespace(name=name, path=actor))
    manifest = old / "panel.json"
    atomic_json(
        manifest,
        {
            "panel_digest": "fixed-panel",
            "members": [
                {"name": item.name, "path": str(item.path)} for item in members
            ],
        },
    )
    config_path = tmp_path / "config.json"
    atomic_json(
        config_path,
        config_to_dict(
            TrainConfig(
                purpose="demonstration",
                validation_panel=str(manifest),
            )
        ),
    )

    def load_panel(path: str | Path) -> SimpleNamespace:
        return SimpleNamespace(
            path=manifest,
            members=members,
            digest="fixed-panel",
            qualified=True,
            schema_version=1,
        )

    def export(repository: Path, commit: str, destination: Path) -> dict[str, Any]:
        destination.mkdir()
        (destination / "uv.lock").write_text("fixed lock")
        (destination / "pyproject.toml").write_text("fixed source")
        return {
            "commit": "a" * 40,
            "git_tree": "b" * 40,
            "files": launch._files(destination),
        }

    monkeypatch.setattr(validation, "load_panel", load_panel)
    monkeypatch.setattr(launch, "_export_source", export)
    monkeypatch.setattr(launch, "_install", Mock())
    monkeypatch.setattr(launch, "_runtime", Mock(return_value={"fixed": True}))
    monkeypatch.setattr(launch, "_gpu", Mock(return_value={"uuid": "GPU-internal"}))
    target = tmp_path / "fixed package"
    result = launch.prepare_run(
        tmp_path,
        target,
        config_path,
        commit="approved",
        gpu_uuid="GPU-internal",
    )
    assert result["launch"] == f"bash '{target}/launch.sh'"
    content = launch._read(target / "panel/panel.json")
    assert [row["path"] for row in content["members"]] == ["halfway", "final"]
    assert content["panel_digest"] == "fixed-panel"
    assert launch._read(target / "config.json")["validation_panel"] == str(
        target / "panel/panel.json"
    )
    assert not (target / "run").exists()
    assert not (target / "process.json").exists()
    (old / "Halfway/weights").write_text("changed developer data")
    assert (target / "panel/halfway/weights").read_text() == "Halfway"
    assert launch.validate_package(target)["origin"]["commit"] == "a" * 40
    for name in ("launch", "resume", "status"):
        text = (target / f"{name}.sh").read_text()
        assert f"'{target}/.venv/bin/python' -I" in text
        assert '"$@"' in text
    with pytest.raises(ValueError, match="must be new"):
        launch.prepare_run(tmp_path, target, config_path, commit="a", gpu_uuid="GPU-a")


def test_stop_signal_reaches_owned_trainer_and_records_exit(tmp_path: Path) -> None:
    worker = (
        f"import pathlib,time; pathlib.Path({str(tmp_path / 'ready')!r}).touch(); "
        "time.sleep(30)"
    )
    command = _supervisor_command(tmp_path, worker)
    with (tmp_path / "process.log").open("wb") as log:
        child = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        # Wait until the trainer runs. A stop that arrives after the trainer is
        # recorded but before it is released keeps it from starting at all.
        while not (tmp_path / "ready").exists():
            time.sleep(0.02)
        assert (tmp_path / "ready").exists()
        before = launch._read(tmp_path / "process.json")
        assert "trainer" in before
        child.send_signal(signal.SIGTERM)
        child.wait()
        after = launch._read(tmp_path / "process.json")
        assert after["state"] == "exited"
        assert after["stop_signal"] == signal.SIGTERM
        assert after["exit_code"] == -signal.SIGTERM
        assert not launch._alive(after)
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def _family_worker(package: Path, mode: str) -> str:
    ignore = (
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        "signal.signal(signal.SIGINT,signal.SIG_IGN); "
        "signal.signal(signal.SIGHUP,signal.SIG_IGN); "
        if mode == "ignore"
        else ""
    )
    worker = (
        "import json,os,pathlib,signal,time; "
        + ignore
        + f"pathlib.Path({str(package / 'worker.json')!r}).write_text("
        "json.dumps({'pid':os.getpid(),'group':os.getpgrp(),'session':os.getsid(0)})); "
        "time.sleep(30)"
    )
    return (
        "import pathlib,signal,subprocess,sys,time\n"
        + ignore
        + f"worker=subprocess.Popen({[sys.executable, '-I', '-c', worker]!r})\n"
        f"while not pathlib.Path({str(package / 'worker.json')!r}).exists():\n"
        "    time.sleep(0.01)\n"
        + (
            "sys.exit(7)\n"
            if mode == "fail"
            else (
                f"while not pathlib.Path({str(package / 'exit_trainer')!r}).exists():\n"
                "    time.sleep(0.01)\n"
                "sys.exit(7)\n"
                if mode == "orphan"
                else "time.sleep(30)\n"
            )
        )
    )


def _start_family(
    package: Path, mode: str, *, deadline_seconds: float | None = None
) -> subprocess.Popen[bytes]:
    with (package / "process.log").open("wb") as log:
        child = subprocess.Popen(
            _supervisor_command(
                package,
                _family_worker(package, mode),
                deadline_seconds=deadline_seconds,
            ),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    while True:
        record = package / "process.json"
        if (
            (package / "worker.json").exists()
            and record.exists()
            and "trainer_group" in launch._read(record)
        ):
            return child
        time.sleep(0.02)


def _kill_identity(identity: dict[str, Any]) -> None:
    try:
        if process_identity(identity["pid"]) == identity:
            os.kill(identity["pid"], signal.SIGKILL)
    except ProcessLookupError:
        pass


def _cleanup_family(package: Path, child: subprocess.Popen[bytes]) -> None:
    record_path = package / "process.json"
    if record_path.exists():
        record = launch._read(record_path)
        group = record.get("trainer_group")
        if isinstance(group, dict):
            group = cast(dict[str, Any], group)
            for pid in launch._group_members(group["pid"]):
                _kill_identity(process_identity(pid))
    if child.poll() is None:
        child.kill()
    child.wait()


@pytest.mark.parametrize(
    ("number", "mode"),
    [
        (signal.SIGTERM, "wait"),
        (signal.SIGINT, "wait"),
        (signal.SIGHUP, "wait"),
        (signal.SIGTERM, "ignore"),
    ],
)
def test_stop_reaches_trainer_and_worker_with_bounded_force_and_isolation(
    tmp_path: Path, number: int, mode: str
) -> None:
    unrelated = subprocess.Popen(
        [sys.executable, "-I", "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )
    unrelated_identity = process_identity(unrelated.pid)
    child = _start_family(tmp_path, mode)
    try:
        before = launch._read(tmp_path / "process.json")
        worker = launch._read(tmp_path / "worker.json")
        assert worker["group"] == worker["session"] == before["trainer"]["pid"]
        assert before["trainer_group"] == before["trainer"]
        assert worker["group"] != child.pid
        started = time.monotonic()
        child.send_signal(number)
        child.wait()
        assert time.monotonic() - started < 5
        after = launch._read(tmp_path / "process.json")
        assert after["state"] == "exited" and after["stop_signal"] == number
        assert after["cleanup"]["signals"][0] == number
        assert after["cleanup"]["remaining_pids"] == []
        assert not launch._alive(after)
        assert not launch._group_members(worker["group"])
        if mode == "ignore":
            assert after["cleanup"]["signals"] == [number, signal.SIGKILL]
            assert after["exit_code"] == -signal.SIGKILL
        else:
            assert after["cleanup"]["signals"] == [number]
            assert after["exit_code"] != 0
        assert process_identity(unrelated.pid) == unrelated_identity
        assert unrelated.poll() is None
        assert (
            f"Launch module: {launch.__file__}"
            in (tmp_path / "process.log").read_text()
        )
    finally:
        _cleanup_family(tmp_path, child)
        unrelated.kill()
        unrelated.wait()


def test_natural_trainer_failure_cleans_worker_after_leader_exit(
    tmp_path: Path,
) -> None:
    child = _start_family(tmp_path, "fail")
    try:
        child.wait()
        after = launch._read(tmp_path / "process.json")
        assert after["state"] == "exited" and after["exit_code"] == 7
        assert after["cleanup"]["signals"] == [signal.SIGTERM]
        assert after["cleanup"]["remaining_pids"] == []
        assert not launch._alive(after)
        assert not launch._group_members(after["trainer"]["pid"])
    finally:
        _cleanup_family(tmp_path, child)


@pytest.mark.parametrize("mode", ["wait", "ignore"])
def test_deadline_stops_owned_descendants_and_records_distinct_failure(
    tmp_path: Path, mode: str
) -> None:
    child = _start_family(tmp_path, mode, deadline_seconds=2.0)
    try:
        child.wait()
        record = launch._read(tmp_path / "process.json")
        assert child.returncode == record["exit_code"] == 124
        assert record["stop_reason"] == "deadline"
        assert record["stop_signal"] == signal.SIGTERM
        assert record["cleanup"]["remaining_pids"] == []
        assert record["cleanup"]["trainer_exit_code"] < 0
        assert not launch._alive(record)
        assert not launch._group_members(record["trainer"]["pid"])
        if mode == "ignore":
            assert record["cleanup"]["signals"] == [signal.SIGTERM, signal.SIGKILL]
    finally:
        _cleanup_family(tmp_path, child)


@pytest.mark.parametrize("outcome", ["stop", "crash", "exit", "deadline"])
def test_nested_supervisor_cleans_separate_worker_group(
    tmp_path: Path, outcome: str
) -> None:
    inner = tmp_path / "inner"
    inner.mkdir()
    inner_command = _supervisor_command(inner, _family_worker(inner, "ignore"))
    controller = (
        "import os,pathlib,signal,subprocess,sys,time\n"
        f"child=subprocess.Popen({inner_command!r})\n"
        f"ready=pathlib.Path({str(tmp_path / 'controller_ready')!r})\n"
        f"while not pathlib.Path({str(inner / 'worker.json')!r}).exists():\n"
        "    time.sleep(.01)\n"
        "ready.touch()\n"
        f"while not pathlib.Path({str(tmp_path / 'release_controller')!r}).exists():\n"
        "    time.sleep(.01)\n"
        + (
            "os.kill(os.getpid(),signal.SIGKILL)\n"
            if outcome == "crash"
            else "sys.exit(0)\n"
        )
    )
    # Inner test limits total three seconds; five lets it kill and reap its
    # separate worker group before the outer supervisor may force it to stop.
    command = _supervisor_command(
        tmp_path,
        controller,
        deadline_seconds=3.0 if outcome == "deadline" else None,
        stop_grace_seconds=5.0,
    )
    unrelated = subprocess.Popen(
        [sys.executable, "-I", "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )
    unrelated_identity = process_identity(unrelated.pid)
    with (tmp_path / "process.log").open("wb") as log:
        child = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
    try:
        while not (tmp_path / "controller_ready").exists():
            assert child.poll() is None
            time.sleep(0.02)
        outer_before = launch._read(tmp_path / "process.json")
        inner_before = launch._read(inner / "process.json")
        assert (
            os.getpgid(inner_before["process"]["pid"]) == outer_before["trainer"]["pid"]
        )
        assert (
            os.getsid(inner_before["process"]["pid"]) == outer_before["trainer"]["pid"]
        )
        assert inner_before["trainer"]["pid"] != outer_before["trainer"]["pid"]
        if outcome == "stop":
            child.send_signal(signal.SIGTERM)
        elif outcome != "deadline":
            (tmp_path / "release_controller").touch()
        child.wait()
        outer_after = launch._read(tmp_path / "process.json")
        inner_after = launch._read(inner / "process.json")
        assert outer_after["state"] == inner_after["state"] == "exited"
        assert outer_after["cleanup"]["remaining_pids"] == []
        assert inner_after["cleanup"]["remaining_pids"] == []
        assert inner_after["cleanup"]["signals"] == [signal.SIGTERM, signal.SIGKILL]
        assert not launch._alive(inner_after)
        assert not launch._group_members(inner_before["trainer"]["pid"])
        assert not launch._group_members(outer_before["trainer"]["pid"])
        assert process_identity(unrelated.pid) == unrelated_identity
        assert unrelated.poll() is None
        if outcome == "deadline":
            assert child.returncode == 124
        elif outcome == "exit":
            assert child.returncode == 0
        else:
            assert child.returncode != 0
    finally:
        if (inner / "process.json").exists():
            record = launch._read(inner / "process.json")
            for identity in (record.get("trainer"), record.get("process")):
                if isinstance(identity, dict):
                    identity = cast(dict[str, Any], identity)
                    for pid in launch._group_members(identity["pid"]):
                        _kill_identity(process_identity(pid))
                    _kill_identity(identity)
        _cleanup_family(tmp_path, child)
        unrelated.kill()
        unrelated.wait()


def test_cleanup_reserve_covers_nested_grace_and_preserves_flat_defaults() -> None:
    assert launch.cleanup_reserve_seconds() == 15.0
    assert (
        launch.cleanup_reserve_seconds(
            stop_grace_seconds=launch._NESTED_STOP_TIMEOUT_SECONDS
        )
        == 25.0
    )
    assert launch.cleanup_reserve_seconds() < launch._NESTED_STOP_TIMEOUT_SECONDS


@pytest.mark.parametrize("grace", [True, 0.0, -1.0, float("inf"), float("nan")])
def test_invalid_stop_grace_is_rejected_before_process_record(
    tmp_path: Path, grace: object
) -> None:
    with pytest.raises(ValueError, match="Stop grace"):
        launch.supervise_command(
            tmp_path, ["unused"], stop_grace_seconds=cast(float, grace)
        )
    assert not (tmp_path / "process.json").exists()


def test_expired_deadline_never_starts_a_trainer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    spawn = Mock(side_effect=AssertionError("Expired work must not start"))
    monkeypatch.setattr(launch.subprocess, "Popen", spawn)
    assert launch.supervise_command(tmp_path, ["unused"], deadline_at=0.0) == 124
    record = launch._read(tmp_path / "process.json")
    assert record["stop_reason"] == "deadline"
    assert record["exit_code"] == 124
    assert "trainer" not in record
    spawn.assert_not_called()
    output = capsys.readouterr().out
    assert "Process Started" in output
    assert "Process Failed" in output
    assert "Exit Code 124 | Reason Deadline" in output
    assert "process_event" not in output
    assert f"[{record['started_at']}]" in output
    assert f"[{record['finished_at']}]" in output


@pytest.mark.parametrize("code", [0, 7])
def test_supervisor_prints_short_outcome_and_keeps_full_machine_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], code: int
) -> None:
    command = [sys.executable, "-I", "-c", f"raise SystemExit({code})"]
    assert launch.supervise_command(tmp_path, command) == code
    record = launch._read(tmp_path / "process.json")
    output = capsys.readouterr().out.splitlines()
    assert len(output) == 2
    assert "Process Started" in output[0]
    assert ("Process Finished" if code == 0 else "Process Failed") in output[1]
    assert f"Exit Code {code}" in output[1]
    assert all(
        "process_event" not in line and not line.startswith("{") for line in output
    )
    assert record["command"] == command
    assert record["exit_code"] == code
    assert record["trainer"] == record["trainer_group"]
    assert record["cleanup"]["remaining_pids"] == []


@pytest.mark.parametrize("deadline", [True, "later", float("nan"), float("inf")])
def test_invalid_deadline_is_rejected_before_process_record(
    tmp_path: Path, deadline: object
) -> None:
    with pytest.raises(ValueError, match="Deadline"):
        launch.supervise_command(
            tmp_path, ["unused"], deadline_at=cast(float, deadline)
        )
    assert not (tmp_path / "process.json").exists()


@pytest.mark.parametrize("deadline_seconds", [None, 1.5])
def test_graceful_child_zero_exit_does_not_hide_stop_request(
    tmp_path: Path, deadline_seconds: float | None
) -> None:
    worker = (
        "import pathlib,signal,sys,time; "
        "signal.signal(signal.SIGTERM,lambda *_: sys.exit(0)); "
        f"pathlib.Path({str(tmp_path / 'ready')!r}).touch(); time.sleep(30)"
    )
    with (tmp_path / "process.log").open("wb") as log:
        child = subprocess.Popen(
            _supervisor_command(tmp_path, worker, deadline_seconds=deadline_seconds),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    try:
        while not (tmp_path / "ready").exists():
            time.sleep(0.02)
        assert (tmp_path / "ready").exists()
        if deadline_seconds is None:
            child.send_signal(signal.SIGTERM)
        child.wait()
        record = launch._read(tmp_path / "process.json")
        assert record["cleanup"]["trainer_exit_code"] == 0
        assert record["exit_code"] == (143 if deadline_seconds is None else 124)
        assert child.returncode == record["exit_code"]
        assert record["stop_reason"] == (
            "signal" if deadline_seconds is None else "deadline"
        )
    finally:
        _cleanup_family(tmp_path, child)


def test_orphan_worker_blocks_restart_after_supervisor_and_leader_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = _start_family(tmp_path, "orphan")
    try:
        record = launch._read(tmp_path / "process.json")
        child.kill()
        child.wait()
        (tmp_path / "exit_trainer").touch()
        while launch._alive({"trainer": record["trainer"]}):
            time.sleep(0.02)
        assert not launch._alive({"process": record["process"]})
        assert not launch._alive({"trainer": record["trainer"]})
        assert launch._alive(record)
        assert launch.status(tmp_path)["alive"]
        monkeypatch.setattr(launch, "validate_package", Mock(return_value={}))
        with pytest.raises(RuntimeError, match="active training"):
            launch.start(tmp_path, resume=True)
        record.update(state="exited", exit_code=0)
        atomic_json(tmp_path / "process.json", record)
        assert launch.status(tmp_path)["effective_state"] == "stopping"
    finally:
        _cleanup_family(tmp_path, child)


def test_cleanup_never_signals_reaped_child_or_shared_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    send = Mock()
    monkeypatch.setattr(launch.os, "killpg", send)
    reaped = subprocess.Popen([sys.executable, "-I", "-c", "pass"])
    reaped.wait()
    with pytest.raises(ChildProcessError):
        launch._finish_trainer(reaped, signal.SIGTERM)
    shared = subprocess.Popen(
        [sys.executable, "-I", "-c", "import time; time.sleep(30)"]
    )
    try:
        with pytest.raises(RuntimeError, match="private process group"):
            launch._finish_trainer(shared, signal.SIGTERM)
        send.assert_not_called()
    finally:
        shared.kill()
        shared.wait()


@pytest.mark.parametrize("changed", ["pid", "start_ticks", "boot_id"])
def test_saved_group_rejects_missing_or_reused_identity(
    monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    group = process_identity()
    group[changed] = -1 if changed == "pid" else "different"
    members = Mock(return_value=[1])
    monkeypatch.setattr(launch, "_group_members", members)
    assert not launch._alive({"trainer_group": group})
    members.assert_not_called()


def test_unregistered_trainer_exits_if_supervisor_dies_before_release(
    tmp_path: Path,
) -> None:
    command = _supervisor_command(
        tmp_path,
        f"from pathlib import Path; Path({str(tmp_path / 'ran')!r}).touch()",
        crash_before_registration=True,
    )
    with (tmp_path / "process.log").open("wb") as log:
        child = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        child.wait()
        assert child.returncode == -signal.SIGKILL
        identity = launch._read(tmp_path / "unregistered.json")
        while launch._alive({"trainer": identity}):
            time.sleep(0.02)
        assert not launch._alive({"trainer": identity})
        assert not launch._group_members(identity["pid"])
        assert not (tmp_path / "ran").exists()
        assert "trainer" not in launch._read(tmp_path / "process.json")
    finally:
        path = tmp_path / "unregistered.json"
        if path.exists():
            _kill_identity(launch._read(path))
        if child.poll() is None:
            child.kill()
        child.wait()


@pytest.mark.parametrize("method", ("mappo", "ippo", "ff_mappo", "ff_ippo"))
def test_preparation_keeps_builtin_panel_references_and_checks_isolated_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    from marl_battlegrounds.baselines.ppo import validate_ppo_method
    from marl_battlegrounds.training.runner import TrainConfig, config_to_dict
    from marl_battlegrounds.training.validation import create_panel, load_panel

    panel = create_panel(
        opponents=("tdm-alpha", "tdm-beta"), output_dir=tmp_path / "panel"
    )
    config_path = tmp_path / "config.json"
    atomic_json(
        config_path,
        config_to_dict(
            TrainConfig(
                method=validate_ppo_method(method),
                purpose="demonstration",
                validation_panel=str(panel.path),
            )
        ),
    )

    def export(repository: Path, commit: str, destination: Path) -> dict[str, Any]:
        del repository, commit
        destination.mkdir()
        (destination / "pyproject.toml").write_text("fixed source")
        return {
            "commit": "a" * 40,
            "git_tree": "b" * 40,
            "files": launch._files(destination),
        }

    reload_commands: list[list[str]] = []

    def reload(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        assert kwargs["check"] is True
        reload_commands.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(launch, "_export_source", export)
    monkeypatch.setattr(launch, "_install", Mock())
    monkeypatch.setattr(launch, "_runtime", Mock(return_value={"fixed": True}))
    monkeypatch.setattr(launch, "_gpu", Mock(return_value={"uuid": "GPU-internal"}))
    monkeypatch.setattr(launch.subprocess, "run", reload)
    target = tmp_path / "package"
    launch.prepare_run(
        tmp_path, target, config_path, commit="approved", gpu_uuid="GPU-internal"
    )
    copied = load_panel(target / "panel")
    assert launch._read(target / "config.json")["method"] == method
    assert copied.digest == panel.digest
    assert [member.reference for member in copied.members] == ["tdm-alpha", "tdm-beta"]
    assert len(reload_commands) == 1
    assert reload_commands[0][0] == str(target / ".venv/bin/python")
    assert reload_commands[0][1] == "-I"
    assert reload_commands[0][-1] == str(target / "panel")
    assert not (target / "run").exists()


def test_qmix_packages_probe_flashbax_while_ppo_keeps_its_probe_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert launch._PROBE == (
        "import json,pathlib,marl_battlegrounds; "
        "from marl_battlegrounds.training.checkpoints import runtime_identity; "
        "print(json.dumps({'identity':runtime_identity(),"
        "'package_file':str(pathlib.Path(marl_battlegrounds.__file__).resolve())}))"
    )
    expected = launch._PROBE.replace(
        "runtime_identity()", "runtime_identity(method='qmix')"
    )
    assert expected == launch._QMIX_PROBE
    source = tmp_path / "source"
    package = source / "src" / "marl_battlegrounds"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    probes: list[str] = []

    def run(command: list[str], **kwargs: object) -> object:
        del kwargs
        probes.append(command[-1])
        output: str = json.dumps(
            {"identity": {}, "package_file": str((package / "__init__.py").resolve())}
        )
        return SimpleNamespace(stdout=output)

    monkeypatch.setattr(launch.subprocess, "run", run)
    launch._runtime(tmp_path / "python", source)
    launch._runtime(tmp_path / "python", source, method="ippo")
    launch._runtime(tmp_path / "python", source, method="qmix")
    launch._runtime(tmp_path / "python", source, method="pqn_vdn")
    assert probes == [
        launch._PROBE,
        launch._PROBE,
        launch._QMIX_PROBE,
        launch._PROBE,
    ]
    from marl_battlegrounds.training.checkpoints import checkpoint_dependencies

    assert "flashbax" not in checkpoint_dependencies("pqn_vdn")
    assert checkpoint_dependencies("pqn_vdn") == checkpoint_dependencies("mappo")
