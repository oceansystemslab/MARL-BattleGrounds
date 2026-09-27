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
learning experiment. Shared launch clocks keep their first deadlines after a
resume; shared locks reject a duplicate launch. Detached commands keep the
caller's environment and wait for durable process identity before executing.
Failed identity publication starts no command.

The tests' own readiness waits fail instead of hanging when the process they
wait on dies. A readiness wait checks readiness first, keeps waiting with no
time limit while the child lives, looks once more after the child exits, and
then fails with the child's exit code and the last lines of its log. Generated
family leaders and nested controllers exit with code 3 when their own child
ends first. The detached-supervisor wait follows the exact supervisor identity
printed at launch and fails if that process ends before its record says exited.
Its final cleanup signals only that identity; if the wait fails, it also stops
the members of a verified trainer group. Apart from the test's own child
processes that it has not yet reaped, test cleanup kills only captured
identities that still match exactly and members of a saved trainer group whose
leader still matches. It never kills by a bare group number: a process behind a
reused identity or a changed leader is left alive, and any process still in a
saved group number is listed in the failure message. The generated trainer that
kills its own supervisor first proves from process.json that its parent is that
supervisor, with the same PID, start time and boot. Otherwise, for example when
the supervisor already ended and the trainer has a new parent, it signals
nothing and exits with code 4 and a one-line reason.
"""

from __future__ import annotations

# pyright: reportPrivateUsage=false
import inspect
import io
import json
import os
import signal
import subprocess
import sys
import tarfile
import time
import venv
from collections.abc import Callable
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
    latest_id, recovering_id = "a" * 64, "b" * 64
    latest = run / "checkpoints" / latest_id
    recovering = (
        run
        / "checkpoints"
        / checkpoints.artifact_name("mappo", "checkpoint", 8, recovering_id)
    )
    identities = {latest: latest_id, recovering: recovering_id}
    for path in identities:
        path.mkdir(parents=True)
    atomic_json(
        run / "latest_checkpoint.json",
        {
            "schema_version": 1,
            "checkpoint_id": latest_id,
            "relative_path": latest.relative_to(run).as_posix(),
        },
    )
    atomic_json(run / "checkpoint_recovery.json", {"checkpoint_id": recovering_id})
    seen: list[Path] = []

    def describe(path: Path) -> dict[str, str]:
        return {"checkpoint_id": identities[path], "kind": "learner"}

    def read(path: Path) -> dict[str, str]:
        seen.append(path)
        return describe(path)

    monkeypatch.setattr(checkpoints, "read_checkpoint_description", describe)
    monkeypatch.setattr(checkpoints, "read_checkpoint_details", read)
    assert launch._checkpoint(tmp_path, None) == recovering
    assert seen == [recovering]
    with pytest.raises(ValueError, match="original checkpoint"):
        launch._checkpoint(tmp_path, str(latest))
    (run / "checkpoint_recovery.json").unlink()
    assert launch._checkpoint(tmp_path, None) == latest
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
    # The parent has not reaped the supervisor when it reads the identity, so
    # the supervisor's PID cannot belong to another process yet.
    parent = (
        "import importlib.util,json,subprocess; "
        "spec=importlib.util.spec_from_file_location('run_io_under_test',"
        f"{inspect.getfile(process_identity)!r}); "
        "run_io=importlib.util.module_from_spec(spec); "
        "spec.loader.exec_module(run_io); "
        f"log=open({str(tmp_path / 'process.log')!r},'ab',buffering=0); "
        f"child=subprocess.Popen({supervisor!r},"
        "stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,"
        "start_new_session=True); "
        "print(json.dumps(run_io.process_identity(child.pid)))"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", parent],
        check=True,
        capture_output=True,
        text=True,
    )
    identity = json.loads(result.stdout)
    pid = identity["pid"]
    path = tmp_path / "process.json"

    def exited() -> bool:
        return path.exists() and launch._read(path).get("state") == "exited"

    try:
        while not exited():
            if not launch._alive({"process": identity}):
                # The supervisor may have written its last record just before
                # it ended, so look once more before failing.
                if exited():
                    break
                left = _stop_owned(_saved_groups(tmp_path), [])
                pytest.fail(
                    f"The detached supervisor {identity} ended before its record "
                    f"reached exited.\n{_left_text(left)}\n{_log_tail(tmp_path)}"
                )
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
        # Signal only the exact supervisor printed at launch, never a later
        # process that reuses its PID.
        _kill_identity(identity, signal.SIGTERM)


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
    owned: list[dict[str, Any]] = []
    try:
        # Wait until the trainer runs. A stop that arrives after the trainer is
        # recorded but before it is released keeps it from starting at all.
        _wait_for_readiness(child, tmp_path, (tmp_path / "ready").exists)
        owned = _capture_owned(tmp_path)
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
        _cleanup_family(child, owned, tmp_path)


def _family_worker(package: Path, mode: str) -> str:
    ignore = (
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        "signal.signal(signal.SIGINT,signal.SIG_IGN); "
        "signal.signal(signal.SIGHUP,signal.SIG_IGN); "
        if mode == "ignore"
        else ""
    )
    # The worker writes this file once it runs. In "kill_supervisor" mode it
    # uses another name, so the test's family readiness never arrives.
    marker = package / (
        "worker_started.json" if mode == "kill_supervisor" else "worker.json"
    )
    worker = (
        "import sys; sys.exit(9)"
        if mode == "worker_exits"
        else "import json,os,pathlib,signal,time; "
        + ignore
        + f"pathlib.Path({str(marker)!r}).write_text("
        "json.dumps({'pid':os.getpid(),'group':os.getpgrp(),'session':os.getsid(0)})); "
        "time.sleep(30)"
    )
    endings = {
        "fail": "sys.exit(7)\n",
        "orphan": (
            f"while not pathlib.Path({str(package / 'exit_trainer')!r}).exists():\n"
            "    time.sleep(0.01)\n"
            "sys.exit(7)\n"
        ),
        # This ends the supervisor after it recorded the trainer group, while
        # the worker still runs. The trainer's parent should be its supervisor,
        # but if the supervisor already ended, the trainer now has a new parent,
        # such as the user's session manager. So the trainer first proves its
        # parent is the supervisor saved in process.json: same PID, start time
        # and boot. Otherwise it signals nothing, prints why and exits with 4.
        # It checks its parent again just before the kill: while the trainer
        # still has that parent, the parent has not finished exiting, so its PID
        # cannot yet belong to another process. Only the microseconds between
        # that check and the kill stay unguarded.
        "kill_supervisor": (
            "import json\n"
            f"saved=json.loads(pathlib.Path({str(package / 'process.json')!r})"
            ".read_text())['process']\n"
            "parent=os.getppid()\n"
            "try:\n"
            "    stat=pathlib.Path(f'/proc/{parent}/stat').read_text()\n"
            "    boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text()\n"
            "    live={'pid':parent,'start_ticks':stat.rsplit(')',1)[1].split()[19],\n"
            "          'boot_id':boot.strip()}\n"
            "except (OSError,IndexError) as error:\n"
            "    live={'pid':parent,'error':repr(error)}\n"
            "if live!=saved or os.getppid()!=parent:\n"
            "    print(f'Refusing to kill parent {live}: it is not the recorded '\n"
            "          f'supervisor {saved}',file=sys.stderr,flush=True)\n"
            "    sys.exit(4)\n"
            "os.kill(parent,signal.SIGKILL)\n"
            "time.sleep(30)\n"
        ),
    }
    return (
        "import os,pathlib,signal,subprocess,sys,time\n"
        + ignore
        + f"worker=subprocess.Popen({[sys.executable, '-I', '-c', worker]!r})\n"
        f"marker=pathlib.Path({str(marker)!r})\n"
        "while not marker.exists():\n"
        "    if worker.poll() is not None:\n"
        "        if marker.exists():\n"
        "            break\n"
        "        print(f'Family worker exited with code {worker.returncode} '\n"
        "              f'before writing {marker.name}',file=sys.stderr,flush=True)\n"
        "        sys.exit(3)\n"
        "    time.sleep(0.01)\n" + endings.get(mode, "time.sleep(30)\n")
    )


def _log_tail(package: Path) -> str:
    log = package / "process.log"
    lines = log.read_text(errors="replace").splitlines()[-40:] if log.exists() else []
    return f"Last lines of {log}:\n" + ("\n".join(lines) or "(empty or missing)")


def _left_text(left: list[int]) -> str:
    return (
        "Processes left running and not signalled, because this test could not "
        f"prove it owns them: {left or 'none'}"
    )


def _wait_for_readiness(
    child: subprocess.Popen[bytes], package: Path, is_ready: Callable[[], bool]
) -> None:
    # Readiness comes first, so a child that got ready and then exited passes.
    # A live child may take as long as it needs; there is no time limit.
    while not is_ready():
        if child.poll() is not None:
            # The child may have become ready just before it exited.
            if is_ready():
                return
            left = _stop_owned(_saved_groups(package), [])
            pytest.fail(
                f"The process exited with code {child.returncode} before it was "
                f"ready.\n{_left_text(left)}\n{_log_tail(package)}"
            )
        time.sleep(0.02)


def _family_ready(package: Path) -> bool:
    record = package / "process.json"
    return (
        (package / "worker.json").exists()
        and record.exists()
        and "trainer_group" in launch._read(record)
    )


def _start_family(
    package: Path, mode: str, *, deadline_seconds: float | None = None
) -> tuple[subprocess.Popen[bytes], list[dict[str, Any]]]:
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
    _wait_for_readiness(child, package, lambda: _family_ready(package))
    return child, _capture_owned(package)


def _identity_matches(identity: object) -> bool:
    # A saved identity proves ownership only while the live process has the
    # same PID, start time and boot. Missing fields prove nothing.
    if not isinstance(identity, dict):
        return False
    identity = cast(dict[str, Any], identity)
    return (
        type(identity.get("pid")) is int
        and identity["pid"] > 0
        and identity.get("start_ticks") is not None
        and identity.get("boot_id") is not None
        and process_identity(identity["pid"]) == identity
    )


def _kill_identity(identity: dict[str, Any], number: int = signal.SIGKILL) -> bool:
    try:
        if _identity_matches(identity):
            os.kill(identity["pid"], number)
            return True
    except ProcessLookupError:
        pass
    return False


def _saved_groups(*packages: Path) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for package in packages:
        path = package / "process.json"
        group = launch._read(path).get("trainer_group") if path.exists() else None
        if isinstance(group, dict):
            groups.append(cast(dict[str, Any], group))
    return groups


def _owned_members(group: dict[str, Any]) -> list[dict[str, Any]]:
    # Members count only while the group's leader is still the saved process;
    # a leader that is gone cannot vouch for a possibly reused group number.
    # The second scan drops a PID that left the group between the first scan
    # and the identity read, because another process may have reused it.
    if not _identity_matches(group):
        return []
    identities = [process_identity(pid) for pid in launch._group_members(group["pid"])]
    still_members = set(launch._group_members(group["pid"]))
    if not _identity_matches(group):
        return []
    return [
        identity
        for identity in identities
        if identity["pid"] in still_members and identity["start_ticks"] is not None
    ]


def _capture_owned(package: Path) -> list[dict[str, Any]]:
    return [
        identity
        for group in _saved_groups(package)
        for identity in _owned_members(group)
    ]


def _stop_owned(groups: list[dict[str, Any]], owned: list[dict[str, Any]]) -> list[int]:
    # Kill the current members of each saved group whose leader still matches,
    # then the captured identities that still match exactly. Groups go first,
    # because killing a captured leader would leave its group unverifiable.
    # Nothing is ever killed by a bare group number.
    killed: list[dict[str, Any]] = []
    for group in groups:
        killed += [
            identity for identity in _owned_members(group) if _kill_identity(identity)
        ]
    killed += [identity for identity in owned if _kill_identity(identity)]
    # SIGKILL cannot be caught or ignored, so each wait ends once the kernel
    # has finished the kill. A zombie no longer counts as alive.
    for identity in killed:
        while launch._alive({"process": identity}):
            time.sleep(0.01)
    # Anything still in a saved group was not proved to be ours. It is only
    # reported.
    return sorted(
        {pid for group in groups for pid in launch._group_members(group["pid"])}
    )


def _cleanup_family(
    child: subprocess.Popen[bytes], owned: list[dict[str, Any]], *packages: Path
) -> None:
    left = _stop_owned(_saved_groups(*packages), owned)
    if child.poll() is None:
        child.kill()
    child.wait()
    if left:
        pytest.fail(_left_text(left))


def _capture_nested(outer: Path, inner: Path) -> list[dict[str, Any]]:
    owned = [
        identity
        for group in _saved_groups(inner, outer)
        for identity in _owned_members(group)
    ]
    inner_supervisor = launch._read(inner / "process.json").get("process")
    if _identity_matches(inner_supervisor) and inner_supervisor not in owned:
        owned.append(cast(dict[str, Any], inner_supervisor))
    return owned


def _sleeper() -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, "-I", "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )


@pytest.mark.parametrize("code", [0, 7])
def test_readiness_wait_fails_with_exit_code_and_log_when_child_exits_first(
    tmp_path: Path, code: int
) -> None:
    with (tmp_path / "process.log").open("wb") as log:
        child = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-c",
                f"print('Child is leaving',flush=True); raise SystemExit({code})",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    with pytest.raises(
        pytest.fail.Exception, match=f"exited with code {code} before it was ready"
    ) as failure:
        _wait_for_readiness(child, tmp_path, (tmp_path / "ready").exists)
    assert "Child is leaving" in str(failure.value)


def test_readiness_wait_checks_readiness_before_and_after_child_exit(
    tmp_path: Path,
) -> None:
    ready = tmp_path / "ready"
    child = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-c",
            f"import pathlib; pathlib.Path({str(ready)!r}).touch()",
        ]
    )
    child.wait()
    _wait_for_readiness(child, tmp_path, ready.exists)
    looks: list[bool] = []

    def ready_on_second_look() -> bool:
        looks.append(True)
        return len(looks) == 2

    _wait_for_readiness(child, tmp_path, ready_on_second_look)
    assert len(looks) == 2


def test_family_worker_exit_before_its_file_fails_with_leader_code(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        pytest.fail.Exception, match="exited with code 3 before it was ready"
    ) as failure:
        _start_family(tmp_path, "worker_exits")
    assert "Family worker exited with code 9 before writing worker.json" in str(
        failure.value
    )
    record = launch._read(tmp_path / "process.json")
    assert record["exit_code"] == 3
    assert record["cleanup"]["remaining_pids"] == []
    assert not launch._alive(record)


def test_supervisor_death_after_group_record_leaves_no_owned_member(
    tmp_path: Path,
) -> None:
    unrelated = _sleeper()
    unrelated_identity = process_identity(unrelated.pid)
    try:
        with pytest.raises(
            pytest.fail.Exception,
            match=f"exited with code {-signal.SIGKILL} before it was ready",
        ) as failure:
            _start_family(tmp_path, "kill_supervisor")
        assert "could not prove it owns them: none" in str(failure.value)
        record = launch._read(tmp_path / "process.json")
        worker = launch._read(tmp_path / "worker_started.json")
        assert record["state"] == "running"
        assert worker["group"] == record["trainer_group"]["pid"]
        assert not launch._alive(record)
        assert not launch._group_members(worker["group"])
        assert process_identity(unrelated.pid) == unrelated_identity
        assert unrelated.poll() is None
    finally:
        unrelated.kill()
        unrelated.wait()


def test_cleanup_refuses_reused_identity_and_kills_true_identity() -> None:
    unrelated = _sleeper()
    control = _sleeper()
    try:
        unrelated_identity = process_identity(unrelated.pid)
        # Same PID, different start time: the saved process is gone and an
        # unrelated process now holds its PID and group number.
        reused = {**unrelated_identity, "start_ticks": "reused"}
        left = _stop_owned([reused], [reused, process_identity(control.pid)])
        assert control.wait() == -signal.SIGKILL
        assert left == [unrelated.pid]
        assert unrelated.poll() is None
        assert process_identity(unrelated.pid) == unrelated_identity
    finally:
        for process in (unrelated, control):
            process.kill()
            process.wait()


def test_nested_cleanup_never_kills_by_group_number_of_changed_leader(
    tmp_path: Path,
) -> None:
    inner = tmp_path / "inner"
    inner.mkdir()
    unrelated = _sleeper()
    controller = _sleeper()
    try:
        unrelated_identity = process_identity(unrelated.pid)
        stale = {**unrelated_identity, "start_ticks": "older"}
        atomic_json(
            inner / "process.json",
            {"process": stale, "trainer": stale, "trainer_group": stale},
        )
        outer = process_identity(controller.pid)
        atomic_json(
            tmp_path / "process.json", {"trainer": outer, "trainer_group": outer}
        )
        owned = _capture_nested(tmp_path, inner)
        assert owned == [outer]
        with pytest.raises(
            pytest.fail.Exception,
            match=rf"could not prove it owns them: \[{unrelated.pid}\]",
        ):
            _cleanup_family(controller, owned, inner, tmp_path)
        assert controller.returncode == -signal.SIGKILL
        assert unrelated.poll() is None
        assert process_identity(unrelated.pid) == unrelated_identity
    finally:
        for process in (unrelated, controller):
            process.kill()
            process.wait()


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
    try:
        unrelated_identity = process_identity(unrelated.pid)
        child, owned = _start_family(tmp_path, mode)
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
            _cleanup_family(child, owned, tmp_path)
    finally:
        unrelated.kill()
        unrelated.wait()


def test_natural_trainer_failure_cleans_worker_after_leader_exit(
    tmp_path: Path,
) -> None:
    child, owned = _start_family(tmp_path, "fail")
    try:
        child.wait()
        after = launch._read(tmp_path / "process.json")
        assert after["state"] == "exited" and after["exit_code"] == 7
        assert after["cleanup"]["signals"] == [signal.SIGTERM]
        assert after["cleanup"]["remaining_pids"] == []
        assert not launch._alive(after)
        assert not launch._group_members(after["trainer"]["pid"])
    finally:
        _cleanup_family(child, owned, tmp_path)


@pytest.mark.parametrize("mode", ["wait", "ignore"])
def test_deadline_stops_owned_descendants_and_records_distinct_failure(
    tmp_path: Path, mode: str
) -> None:
    child, owned = _start_family(tmp_path, mode, deadline_seconds=2.0)
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
        _cleanup_family(child, owned, tmp_path)


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
        f"inner_ready=pathlib.Path({str(inner / 'worker.json')!r})\n"
        "while not inner_ready.exists():\n"
        "    if child.poll() is not None:\n"
        "        if inner_ready.exists():\n"
        "            break\n"
        "        print(f'Inner supervisor exited with code {child.returncode} '\n"
        "              'before its worker was ready',file=sys.stderr,flush=True)\n"
        "        sys.exit(3)\n"
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
    owned: list[dict[str, Any]] = []
    try:
        _wait_for_readiness(child, tmp_path, (tmp_path / "controller_ready").exists)
        owned = _capture_nested(tmp_path, inner)
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
        unrelated.kill()
        unrelated.wait()
        _cleanup_family(child, owned, inner, tmp_path)


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
    owned: list[dict[str, Any]] = []
    try:
        _wait_for_readiness(child, tmp_path, (tmp_path / "ready").exists)
        owned = _capture_owned(tmp_path)
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
        _cleanup_family(child, owned, tmp_path)


def test_orphan_worker_blocks_restart_after_supervisor_and_leader_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child, owned = _start_family(tmp_path, "orphan")
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
        _cleanup_family(child, owned, tmp_path)


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


def test_shared_launch_clock_keeps_first_deadline_and_original_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "launch_time.json"
    monkeypatch.setattr(launch.time, "time", lambda: 100.0)
    with launch.launch_lock(tmp_path):
        original = launch.launch_clock(
            path,
            durations={"numerical_deadline": 20, "hard_deadline": 30},
            binding={"budget_digest": "original"},
            create=True,
        )
    assert original == {
        "started_at_seconds": 100.0,
        "numerical_deadline": 120.0,
        "hard_deadline": 130.0,
        "budget_digest": "original",
    }
    path.write_text(json.dumps(original, indent=4) + "\n")
    before = path.read_bytes()
    monkeypatch.setattr(launch.time, "time", lambda: 200.0)
    assert (
        launch.launch_clock(
            path,
            durations={"numerical_deadline": 20, "hard_deadline": 30},
            binding={"budget_digest": "original"},
            create=True,
            recorded={"status": "stopped", **original},
        )
        == original
    )
    assert path.read_bytes() == before
    for durations, binding, recorded in (
        (
            {"numerical_deadline": 21, "hard_deadline": 30},
            {"budget_digest": "original"},
            None,
        ),
        (
            {"numerical_deadline": 20, "hard_deadline": 30},
            {"budget_digest": "changed"},
            None,
        ),
        (
            {"numerical_deadline": 20, "hard_deadline": 30},
            {"budget_digest": "original"},
            {**original, "hard_deadline": 131},
        ),
    ):
        with pytest.raises(ValueError, match="clock"):
            launch.launch_clock(
                path, durations=durations, binding=binding, recorded=recorded
            )
        assert path.read_bytes() == before


@pytest.mark.parametrize("duration", [True, 0, -1, float("inf"), float("nan")])
def test_shared_launch_clock_rejects_invalid_duration_without_writing(
    tmp_path: Path, duration: float
) -> None:
    path = tmp_path / "launch_time.json"
    with pytest.raises(ValueError, match="duration"):
        launch.launch_clock(
            path, durations={"deadline_at": duration}, binding={}, create=True
        )
    assert not path.exists()


def test_shared_launch_lock_rejects_a_duplicate_holder(tmp_path: Path) -> None:
    with (
        launch.launch_lock(tmp_path),
        pytest.raises(BlockingIOError),
        launch.launch_lock(tmp_path),
    ):
        pytest.fail("A second launch must not enter the critical section")
    with launch.launch_lock(tmp_path):
        launch.launch_clock(
            tmp_path / "clock.json", durations={}, binding={"id": "case"}, create=True
        )


def test_shared_detached_spawn_publishes_identity_before_inherited_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MARL_LAUNCH_TEST_VALUE", "preserved")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "researcher-modules"))
    output = tmp_path / "observed.json"
    code = (
        "import json,os,pathlib; root=pathlib.Path.cwd(); "
        "record=json.loads((root/'process.json').read_text()); "
        "assert record['process']['pid']==os.getpid(); "
        "assert record['command'][0]; print('Detached command ran',flush=True); "
        "(root/'observed.json').write_text(json.dumps({"
        "'custom':os.environ['MARL_LAUNCH_TEST_VALUE'],"
        "'pythonpath':os.environ['PYTHONPATH'], 'stdin':os.read(0,1).decode()}))"
    )
    with launch.launch_lock(tmp_path):
        record = launch.spawn_detached(
            tmp_path,
            [sys.executable, "-c", code],
            log_path=tmp_path / "process.log",
            mode="start",
            cwd=tmp_path,
        )
    pid = record["process"]["pid"]
    try:
        while not output.exists():
            if not launch._alive({"process": record["process"]}):
                assert output.exists(), (
                    "The saved child exited before writing its output.\n"
                    + _log_tail(tmp_path)
                )
                break
            time.sleep(0.01)
        assert json.loads(output.read_text()) == {
            "custom": "preserved",
            "pythonpath": str(tmp_path / "researcher-modules"),
            "stdin": "",
        }
    finally:
        if launch._alive({"process": record["process"]}):
            os.kill(pid, signal.SIGTERM)
        os.waitpid(pid, 0)
    assert record["mode"] == "start"
    assert "Detached command ran" in (tmp_path / "process.log").read_text()


def test_shared_detached_spawn_does_not_execute_after_publication_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "should-not-exist"

    def fail(path: Path, value: object) -> None:
        raise OSError("Injected publication failure")

    monkeypatch.setattr(launch, "atomic_json", fail)
    with launch.launch_lock(tmp_path), pytest.raises(OSError, match="publication"):
        launch.spawn_detached(
            tmp_path,
            [
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(marker)!r}).touch()",
            ],
            log_path=tmp_path / "process.log",
            mode="start",
        )
    assert not marker.exists()
    assert not (tmp_path / "process.json").exists()


def test_shared_stop_rechecks_live_owner_and_leaves_stale_identity_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live = tmp_path / "live.json"
    stale = tmp_path / "stale.json"
    atomic_json(live, {"process": {"pid": 123, "start_ticks": "live"}})
    atomic_json(stale, {"process": {"pid": 456, "start_ticks": "stale"}})

    def is_alive(record: dict[str, Any]) -> bool:
        return record["process"]["start_ticks"] == "live"

    calls: list[tuple[int, int]] = []

    def stop(pid: int, sig: int) -> None:
        calls.append((pid, sig))

    monkeypatch.setattr(launch, "_alive", is_alive)
    monkeypatch.setattr(launch.os, "kill", stop)
    assert launch.live_process_records([live, stale, tmp_path / "absent.json"]) == [
        live
    ]
    role = tmp_path / "active-role.json"
    atomic_json(role, launch._read(live))
    assert launch.request_stop([stale, live, role, live]) == [123]
    assert calls == [(123, signal.SIGTERM)]


def test_shared_detached_spawn_cleans_owned_session_when_release_is_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "started"
    original_write = launch.os.write

    def interrupted(fd: int, data: bytes) -> int:
        original_write(fd, data)
        record = launch._read(tmp_path / "process.json")
        while not marker.exists():
            if not launch._alive({"process": record["process"]}):
                assert marker.exists(), (
                    "The gated child exited before writing its marker.\n"
                    + _log_tail(tmp_path)
                )
                break
            time.sleep(0.01)
        raise KeyboardInterrupt("Interrupted after releasing the child")

    monkeypatch.setattr(launch.os, "write", interrupted)
    with (
        launch.launch_lock(tmp_path),
        pytest.raises(KeyboardInterrupt, match="releasing"),
    ):
        launch.spawn_detached(
            tmp_path,
            [
                sys.executable,
                "-c",
                "from pathlib import Path; import time; "
                f"Path({str(marker)!r}).touch(); time.sleep(60)",
            ],
            log_path=tmp_path / "process.log",
            mode="start",
        )
    record = launch._read(tmp_path / "process.json")
    assert not launch._alive(record)
    assert launch._group_members(record["process"]["pid"]) == []
    with pytest.raises(ChildProcessError):
        os.waitpid(record["process"]["pid"], os.WNOHANG)
