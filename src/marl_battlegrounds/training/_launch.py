"""Prepare and run a fixed local PPO package without an active Codex session.

Preparation reads a committed Git tree or freezes current working files, builds
a separate pinned environment and copies the frozen panel when needed.
Launching is a separate explicit action. The
detached supervisor owns process logs and exit status; the ordinary trainer
still owns learning, its run lock and scientific completion. Status reads files
and /proc only. This module starts no device when imported.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, cast

from marl_battlegrounds.training._run_io import (
    atomic_json,
    process_identity,
    progress_text,
    utc_now,
)

type Record = dict[str, Any]
_MANIFEST = "launch_package.json"
_STOP_TIMEOUT_SECONDS = 10.0
_KILL_TIMEOUT_SECONDS = 2.0
_NESTED_STOP_TIMEOUT_SECONDS = 20.0
_TRAINER_GATE = (
    "import os,sys; fd=int(sys.argv[1]); ready=os.read(fd,1); os.close(fd); "
    "sys.exit(1) if ready != b'1' else "
    "os.execvpe(sys.argv[2],sys.argv[2:],os.environ)"
)
_EXECUTION_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "JAX_PLATFORMS",
    "JAX_ENABLE_X64",
    "JAX_DISABLE_JIT",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
)
_PROBE = (
    "import json,pathlib,marl_battlegrounds; "
    "from marl_battlegrounds.training.checkpoints import runtime_identity; "
    "print(json.dumps({'identity':runtime_identity(),"
    "'package_file':str(pathlib.Path(marl_battlegrounds.__file__).resolve())}))"
)
# QMIX runs also record Flashbax; the PPO probe above keeps its exact bytes.
_QMIX_PROBE = _PROBE.replace("runtime_identity()", "runtime_identity(method='qmix')")


def _read(path: Path) -> Record:
    """Read a JSON object from a regular package file without following a link."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Required package file is missing: {path}")
    value: object = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return cast(Record, value)


def _hash(path: Path) -> str:
    """Hash one regular file using a bounded read buffer."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular package file: {path}")
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


def _files(root: Path) -> dict[str, str]:
    """Identify immutable source/panel bytes, excluding Python bytecode caches."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Fixed source/panel roots must be real directories")
    values: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if "__pycache__" in path.relative_to(root).parts:
            continue
        if path.is_symlink():
            raise ValueError("A fixed run package must not contain source/panel links")
        if path.is_file():
            values[path.relative_to(root).as_posix()] = _hash(path)
    return values


def _environment(
    gpu_uuid: str | None,
    *,
    memory_fraction: float | None = None,
    compilation_cache: Path | None = None,
) -> dict[str, str]:
    """Use default JAX settings and explicit hardware, without shell overrides.

    Drop inherited JAX/XLA/TF settings as well as mutable Python routing. None
    chooses CPU for bookkeeping and import checks; a UUID chooses that GPU. This
    fixed default policy is shared by preparation, launch and resume.

    Parameters
    ----------
    gpu_uuid : str or None
        Explicit GPU UUID, or None for CPU-only bookkeeping and import checks.
    memory_fraction : float or None, optional
        Optional JAX allocator fraction in (0, 1]. None leaves JAX's default.
        Preallocation remains off. This is a pool limit, not a memory forecast.
    compilation_cache : Path or None, optional
        Absolute directory for JAX's persistent compilation cache. None drops
        inherited cache settings. This helper does not create the directory.

    Returns
    -------
    dict[str, str]
        A new environment with inherited overrides removed, then the requested
        memory and cache settings applied. The parent environment is unchanged.

    Raises
    ------
    ValueError
        The fraction is not a finite number in (0, 1], or the cache is relative.
    """
    if memory_fraction is not None and (
        isinstance(memory_fraction, bool)
        or not isinstance(cast(object, memory_fraction), (int, float))
        or not math.isfinite(memory_fraction)
        or not 0 < memory_fraction <= 1
    ):
        raise ValueError("Memory fraction must be a finite number in (0, 1]")
    if compilation_cache is not None and not compilation_cache.is_absolute():
        raise ValueError("Compilation cache must use an absolute directory")
    env = dict(os.environ)
    for key in tuple(env):
        if key.startswith(("JAX_", "XLA_", "TF_")) or key in (
            "PYTHONPATH",
            "PYTHONHOME",
            "VIRTUAL_ENV",
        ):
            env.pop(key)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["JAX_PLATFORMS"] = "cpu" if gpu_uuid is None else "cuda,cpu"
    env["JAX_ENABLE_X64"] = "false"
    env["JAX_DISABLE_JIT"] = "false"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if gpu_uuid is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    if memory_fraction is not None:
        env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(float(memory_fraction))
    if compilation_cache is not None:
        env["JAX_COMPILATION_CACHE_DIR"] = str(compilation_cache)
        env["JAX_ENABLE_COMPILATION_CACHE"] = "true"
    return env


def _gpu(uuid: str) -> Record:
    """Resolve an explicitly supplied GPU UUID to its current name and PCI bus.

    Query nvidia-smi without allocating a GPU. Missing, duplicate or non-5090
    matches fail. The caller chooses the authorized internal card by UUID;
    device ordinal zero is never used to decide which card is internal.
    """
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,pci.bus_id",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    matches: list[Record] = []
    for line in completed.stdout.splitlines():
        columns = [value.strip() for value in line.split(",")]
        if len(columns) == 3 and columns[0] == uuid:
            matches.append(
                {"uuid": columns[0], "name": columns[1], "pci_bus_id": columns[2]}
            )
    if len(matches) != 1 or "RTX 5090" not in matches[0]["name"]:
        raise ValueError("Choose the verified internal RTX 5090 by its full UUID")
    return matches[0]


def _runtime(python: Path, source: Path, *, method: str = "mappo") -> Record:
    """Probe the fixed environment's actual imports and dependency identity.

    python and source are the package's interpreter and exported source.
    method is the run's training method; "qmix" uses the QMIX probe, which
    also records Flashbax, and any PPO method uses the historical probe.
    Runs one isolated child process on the CPU environment; raises
    ValueError when the child imports a different source package.
    """
    probe = _QMIX_PROBE if method == "qmix" else _PROBE
    completed = subprocess.run(
        [str(python), "-I", "-c", probe],
        cwd=source,
        env=_environment(None),
        check=True,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    result = cast(Record, json.loads(completed.stdout))
    imported = Path(result["package_file"])
    expected = source / "src" / "marl_battlegrounds" / "__init__.py"
    if imported != expected.resolve():
        raise ValueError("The prepared environment imports a different source package")
    return result


def _export_source(repository: Path, commit: str, destination: Path) -> Record:
    """Export one existing commit using read-only Git queries and archive.

    No branch, index, commit or worktree in repository is changed. Tar extraction
    uses Python's data filter and rejects links. The result records exact Git
    commit/tree IDs plus each exported file's bytes.
    """

    def git(*arguments: str) -> bytes:
        """Run one read-only Git command in the explicit source repository."""
        return subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            stdin=subprocess.DEVNULL,
        ).stdout

    resolved = git("rev-parse", "--verify", f"{commit}^{{commit}}").decode().strip()
    tree = git("rev-parse", "--verify", f"{resolved}^{{tree}}").decode().strip()
    destination.mkdir()
    with tempfile.TemporaryFile() as archive:
        subprocess.run(
            ["git", "-C", str(repository), "archive", "--format=tar", resolved],
            check=True,
            stdout=archive,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
        )
        archive.seek(0)
        with tarfile.open(fileobj=archive, mode="r:") as tar:
            if any(member.issym() or member.islnk() for member in tar.getmembers()):
                raise ValueError("The fixed source archive must not contain links")
            tar.extractall(destination, filter="data")
    return {"commit": resolved, "git_tree": tree, "files": _files(destination)}


def export_working_source(repository: Path, destination: Path) -> Record:
    """Copy the current public candidate without changing its Git repository.

    Parameters
    ----------
    repository : Path
        Git worktree root. Tracked files use their current working bytes, even
        when the index holds older bytes. Nonignored new files are included.
        Deleted files, Python bytecode and private docs/dev/milestone*.md files
        are omitted. Links and special files are rejected.
    destination : Path
        New source directory whose parent exists. If it is inside repository,
        it must be ignored by Git. The copy contains no .git directory.

    Returns
    -------
    dict
        commit and git_tree identify the original HEAD only. files identifies
        the actual copied candidate. working_tree records the original status,
        index hashes and file modes. These facts are checked again after copying.
        The result must not be described as the contents of the HEAD commit.

    Raises
    ------
    ValueError
        A path is unsafe, the destination exists, or source/index/status changes
        during the copy. An incomplete destination is retained on failure.
    subprocess.CalledProcessError
        A read-only Git query fails.

    Notes
    -----
    Git optional locks are disabled, so status cannot refresh the index. This
    helper neither installs packages nor changes file permissions in the source.
    The caller must keep the completed copy unchanged and verify its imports.
    """
    repository = repository.absolute()
    destination = destination.absolute()
    for path in (repository, destination, *destination.parents):
        if path.is_symlink():
            raise ValueError("Source export paths must not contain links")
    if destination.exists():
        raise ValueError("Source export destination must be new")
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")

    def git(*arguments: str) -> bytes:
        """Read Git facts without refreshing or writing its index."""
        return subprocess.run(
            ["git", "-C", str(repository), *arguments],
            env=env,
            check=True,
            capture_output=True,
            stdin=subprocess.DEVNULL,
        ).stdout

    if Path(os.fsdecode(git("rev-parse", "--show-toplevel")).strip()) != repository:
        raise ValueError("Source must be the Git worktree root")
    if destination.is_relative_to(repository):
        git("check-ignore", "--quiet", "--", str(destination))

    def snapshot() -> Record:
        """Read source bytes, index bytes and Git state for a stable-copy check."""
        names = git("ls-files", "--cached", "--others", "--exclude-standard", "-z")
        files: dict[str, str] = {}
        modes: dict[str, int] = {}
        for raw in sorted(set(names.split(b"\0")) - {b""}):
            relative = Path(os.fsdecode(raw))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Git returned an unsafe source path")
            if "__pycache__" in relative.parts or (
                relative.parts[:2] == ("docs", "dev")
                and relative.match("milestone*.md")
            ):
                continue
            source = repository / relative
            for part in (source, *source.parents):
                if part == repository:
                    break
                if part.is_symlink():
                    raise ValueError("A fixed source copy must not contain links")
            if not source.exists():
                continue
            if not source.is_file():
                raise ValueError("A fixed source copy needs regular source files")
            name = relative.as_posix()
            files[name] = _hash(source)
            modes[name] = source.stat().st_mode & 0o777
        index = Path(os.fsdecode(git("rev-parse", "--git-path", "index")).strip())
        if not index.is_absolute():
            index = repository / index
        return {
            "commit": git("rev-parse", "--verify", "HEAD^{commit}").decode().strip(),
            "git_tree": git("rev-parse", "--verify", "HEAD^{tree}").decode().strip(),
            "files": files,
            "working_tree": {
                "status_porcelain": os.fsdecode(
                    git("status", "--porcelain=v1", "-z", "--untracked-files=all")
                ),
                "index_sha256": _hash(index),
                "index_entries_sha256": hashlib.sha256(
                    git("ls-files", "--stage", "-z")
                ).hexdigest(),
                "file_modes": modes,
            },
        }

    before = snapshot()
    destination.mkdir()
    for name in before["files"]:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repository / name, target, follow_symlinks=False)
    copied = _files(destination)
    if snapshot() != before or copied != before["files"]:
        raise ValueError("Source or Git state changed while copying the candidate")
    return {"kind": "working_tree", **before}


def _install(source: Path, environment: Path, python: str) -> None:
    """Build one isolated locked environment, copying rather than sharing files.

    Uses installed uv and the snapshot's uv.lock. This may read its package cache
    or download locked packages; it never edits the lockfile or another virtual
    environment. Installation is preparation only, not a training launch.
    """
    env = _environment(None)
    env.update(
        {
            "UV_PROJECT_ENVIRONMENT": str(environment),
            "UV_LINK_MODE": "copy",
            "UV_PYTHON_DOWNLOADS": "never",
        }
    )
    subprocess.run(
        [
            "uv",
            "sync",
            "--project",
            str(source),
            "--frozen",
            "--no-dev",
            "--python",
            python,
            "--extra",
            "training",
            "--extra",
            "viz",
            "--extra",
            "cuda13",
        ],
        env=env,
        check=True,
        stdin=subprocess.DEVNULL,
    )


def prepare_run(
    repository: str | Path,
    destination: str | Path,
    config_path: str | Path,
    *,
    commit: str,
    gpu_uuid: str,
    python: str = sys.executable,
) -> Record:
    """Build a fixed run package and commands without starting its experiment.

    Parameters
    ----------
    repository : str or Path
        Source Git checkout containing the already qualified commit.
    destination : str or Path
        Exact new package directory. Existing directories are never replaced.
    config_path : str or Path
        Versioned demonstration config with a frozen panel path. Its members
        need saved reload references: built-ins, factories or actor exports.
    commit : str
        Explicit qualified commit reference, resolved to a full immutable ID.
    gpu_uuid : str
        Explicit authorized internal RTX 5090 UUID; its PCI identity is saved.
    python : str, default sys.executable
        Existing Python interpreter for uv's isolated environment creation.

    Returns
    -------
    dict
        Absolute package path and exact launch/status/resume shell commands.
        The package owns copied source, panel, config, .venv, logs and run output.

    Raises
    ------
    ValueError, OSError, subprocess.CalledProcessError
        Source, panel, hardware, paths, installation or import checks fail. A
        partial package has no launch manifest and cannot start an experiment.
    """
    target = Path(destination).absolute()
    if target.exists():
        raise ValueError("The launch package destination must be new")
    hardware = _gpu(gpu_uuid)
    from marl_battlegrounds.training.runner import config_from_dict, config_to_dict
    from marl_battlegrounds.training.validation import load_panel

    config = config_from_dict(_read(Path(config_path)))
    if config.purpose != "demonstration" or config.validation_panel is None:
        raise ValueError("The full run package requires a frozen-panel demonstration")
    panel_path = Path(config.validation_panel)
    panel_content = _read(
        panel_path / "panel.json" if panel_path.is_dir() else panel_path
    )
    if panel_content.get("schema_version") == 2 and any(
        member.get("reference") is None for member in panel_content.get("members", [])
    ):
        raise ValueError(
            "A launch package needs reload references; live-only Systems "
            "remain available through train"
        )
    panel = load_panel(config.validation_panel)
    if not panel.qualified:
        raise ValueError("The full run package requires a qualified provisional panel")
    target.mkdir(parents=True)
    source = target / "source"
    origin = _export_source(Path(repository).absolute(), commit, source)
    panel_root = target / "panel"
    panel_root.mkdir()
    content = _read(panel.path)
    for index, (row, member) in enumerate(
        zip(content["members"], panel.members, strict=True)
    ):
        if panel.schema_version == 1:
            assert member.path is not None
            original = member.path
        else:
            assert member.reference is not None
            original = Path(member.reference)
            if not original.is_absolute():
                continue
        output = panel_root / (
            member.name.lower() if panel.schema_version == 1 else f"opponent-{index}"
        )
        shutil.copytree(original, output, symlinks=True)
        _files(output)
        if panel.schema_version == 1:
            row["path"] = output.name
        else:
            row["reference"] = output.name
            row["export_relative"] = True
    atomic_json(panel_root / "panel.json", content)
    if load_panel(panel_root).digest != panel.digest:
        raise ValueError("Copying the panel changed its scientific identity")
    resolved = config_to_dict(config)
    resolved["validation_panel"] = str(panel_root / "panel.json")
    atomic_json(target / "config.json", resolved)
    _install(source, target / ".venv", python)
    interpreter = target / ".venv" / "bin" / "python"
    imported = _runtime(interpreter, source, method=config.method)
    if panel.schema_version == 2:
        subprocess.run(
            [
                str(interpreter),
                "-I",
                "-c",
                "import sys; "
                "from marl_battlegrounds.training.validation import load_panel; "
                "load_panel(sys.argv[1])",
                str(panel_root),
            ],
            cwd=source,
            env=_environment(None),
            check=True,
            stdin=subprocess.DEVNULL,
        )
    if _files(source) != origin["files"]:
        raise ValueError("Environment setup changed the exported source bytes")
    (target / "logs").mkdir()
    for name, action in (
        ("launch", "start"),
        ("status", "status"),
        ("resume", "resume"),
    ):
        command = [
            str(interpreter),
            "-I",
            "-m",
            "marl_battlegrounds.training._launch",
            action,
            str(target),
        ]
        text = (
            "#!/usr/bin/env bash\n"
            "# Use the fixed run package and its own Python environment.\n"
            "set -euo pipefail\nexec " + shlex.join(command) + ' "$@"\n'
        )
        script = target / f"{name}.sh"
        script.write_text(text)
        script.chmod(0o755)
    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "package_path": str(target),
        "origin": origin,
        "runtime": imported,
        "gpu": hardware,
        "execution_environment": {
            name: _environment(gpu_uuid)[name] for name in _EXECUTION_KEYS
        },
        "panel_digest": panel.digest,
        "panel_files": _files(panel_root),
        "config_sha256": _hash(target / "config.json"),
        "scripts": {
            name: _hash(target / name)
            for name in ("launch.sh", "status.sh", "resume.sh")
        },
    }
    atomic_json(target / _MANIFEST, manifest)
    validate_package(target)
    return {
        "package": str(target),
        **{
            name: shlex.join(["bash", str(target / f"{name}.sh")])
            for name in ("launch", "status", "resume")
        },
    }


def validate_package(package: str | Path) -> Record:
    """Verify fixed source, panel, config, imports and hardware before starting.

    This performs no learning, writer recovery or checkpoint selection. It reads
    source/artifact bytes, runs nvidia-smi and probes imports in the pinned
    environment on CPU, using the probe for the method in the hashed config.
    Relocation is rejected because commands and config own absolute paths.
    Ordinary status does not pay this complete verification cost.
    """
    root = Path(package).absolute()
    manifest = _read(root / _MANIFEST)
    if manifest.get("schema_version") != 1 or manifest.get("package_path") != str(root):
        raise ValueError("Launch package schema or absolute location differs")
    if _files(root / "source") != manifest["origin"]["files"]:
        raise ValueError("Fixed source bytes changed")
    if _files(root / "panel") != manifest["panel_files"]:
        raise ValueError("Frozen panel bytes changed")
    if _hash(root / "config.json") != manifest["config_sha256"]:
        raise ValueError("Frozen run settings changed")
    if set(manifest["scripts"]) != {"launch.sh", "status.sh", "resume.sh"}:
        raise ValueError("Prepared command set changed")
    for name, digest in manifest["scripts"].items():
        if (
            name not in ("launch.sh", "status.sh", "resume.sh")
            or _hash(root / name) != digest
        ):
            raise ValueError("Prepared commands changed")
    if _gpu(manifest["gpu"]["uuid"]) != manifest["gpu"]:
        raise ValueError("Selected internal GPU identity changed")
    environment = _environment(manifest["gpu"]["uuid"])
    if manifest["execution_environment"] != {
        name: environment[name] for name in _EXECUTION_KEYS
    }:
        raise ValueError("Frozen JAX execution settings differ")
    method = _read(root / "config.json").get("method", "mappo")
    if (
        _runtime(root / ".venv" / "bin" / "python", root / "source", method=method)
        != manifest["runtime"]
    ):
        raise ValueError("Prepared environment or imported source changed")
    return manifest


def _group_members(group: int) -> list[int]:
    """Read live Linux PIDs in the trainer's private group and session.

    group is the original trainer PID, also its process-group and session ID.
    Zombies cannot write files and are excluded. This read-only scan does not
    prove ownership and must never authorize a signal to a saved group number.
    """
    members: list[int] = []
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal():
            continue
        try:
            fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
            if fields[0] != "Z" and int(fields[2]) == int(fields[3]) == group:
                members.append(int(path.name))
        except OSError, IndexError, ValueError:
            continue
    return members


def _alive(record: Record) -> bool:
    """Check supervisor, trainer and leftover worker liveness without signals.

    Exact PID/start/boot identities guard direct processes. A saved private
    trainer group also blocks restart while workers remain after its leader
    exits. A reused live leader PID or a different boot invalidates that group.
    If the old leader is absent, remaining group members conservatively block
    restart; this cannot grant permission to kill a possibly reused group.
    """
    for name in ("process", "trainer"):
        identity = record.get(name)
        if not isinstance(identity, dict) or not identity.get("start_ticks"):
            continue
        identity = cast(Record, identity)
        if process_identity(identity["pid"]) != identity:
            continue
        try:
            stat = Path(f"/proc/{identity['pid']}/stat").read_text()
            if stat.rsplit(")", 1)[1].split()[0] != "Z":
                return True
        except OSError, IndexError:
            continue
    group = record.get("trainer_group")
    if not isinstance(group, dict):
        return False
    group = cast(Record, group)
    if (
        type(group.get("pid")) is not int
        or group["pid"] <= 0
        or not group.get("start_ticks")
    ):
        return False
    if group.get("boot_id") != process_identity()["boot_id"]:
        return False
    current = process_identity(group["pid"])
    if current["start_ticks"] is not None and current != group:
        return False
    return bool(_group_members(group["pid"]))


def cleanup_reserve_seconds(*, stop_grace_seconds: float | None = None) -> float:
    """Return the time to reserve before a deadline for bounded process cleanup.

    stop_grace_seconds is the positive, finite TERM grace in seconds. None uses
    the usual ten seconds. Add the two-second KILL wait, two-second reap wait
    and one second for polling and publication. A nested outer supervisor uses
    twenty seconds of TERM grace so the inner supervisor can finish its own
    ten-second stop, KILL and reap before its parent can be killed. Invalid
    values raise ValueError. This does no process work; an unkillable OS process
    or blocked file system still needs the saved cleanup error checked.
    """
    grace = _STOP_TIMEOUT_SECONDS if stop_grace_seconds is None else stop_grace_seconds
    if (
        isinstance(grace, bool)
        or not isinstance(cast(object, grace), (int, float))
        or not math.isfinite(grace)
        or grace <= 0
    ):
        raise ValueError("Stop grace must be a finite positive number of seconds")
    return float(grace) + 2 * _KILL_TIMEOUT_SECONDS + 1.0


def _finish_trainer(
    child: subprocess.Popen[bytes],
    number: int,
    *,
    stop_grace_seconds: float | None = None,
) -> Record:
    """Stop the owned trainer session, then reap its direct child within a bound.

    child must be this process's unreaped child and its private session leader.
    Holding that child unreaped keeps its PID/group ID from being reused while
    signals are sent. No saved PID is accepted here. number is the first stop
    signal; live members get stop_grace_seconds (None means ten) before SIGKILL
    and two more seconds to exit, then at most two seconds to reap the trainer.
    Return cleanup
    facts and its exit code, or None if it could not be reaped in that last
    bound. Never wait indefinitely.
    """
    cleanup_reserve_seconds(stop_grace_seconds=stop_grace_seconds)
    grace = _STOP_TIMEOUT_SECONDS if stop_grace_seconds is None else stop_grace_seconds
    os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    if os.getpgid(child.pid) != child.pid or os.getsid(child.pid) != child.pid:
        raise RuntimeError("The owned trainer has no private process group")
    sent: list[int] = []
    remaining: list[int] = []
    for sig, seconds in (
        (number, grace),
        (signal.SIGKILL, _KILL_TIMEOUT_SECONDS),
    ):
        remaining = _group_members(child.pid)
        if not remaining:
            break
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            break
        sent.append(sig)
        deadline = time.monotonic() + seconds
        while remaining and time.monotonic() < deadline:
            time.sleep(0.05)
            remaining = _group_members(child.pid)
    try:
        code = child.wait(timeout=_KILL_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        code = None
    return {"signals": sent, "remaining_pids": remaining, "trainer_exit_code": code}


def _checkpoint(package: Path, explicit: str | None) -> Path:
    """Choose an explicit recovery marker or validated latest pointer.

    A supplied checkpoint wins only if it agrees with an unfinished recovery
    marker. All choices must be complete checkpoints in this package's run.
    There is no fallback from a corrupt chosen checkpoint to an older one.
    """
    from marl_battlegrounds.training.checkpoints import read_checkpoint_details

    run = package / "run"
    marker = run / "checkpoint_recovery.json"
    recovering = _read(marker)["checkpoint_id"] if marker.exists() else None
    if explicit is not None:
        selected = Path(explicit).absolute()
        if recovering is not None and selected.name != recovering:
            raise ValueError("Interrupted recovery requires its original checkpoint")
    elif recovering is not None:
        selected = run / "checkpoints" / recovering
    else:
        pointer = _read(run / "latest_checkpoint.json")
        identifier = pointer.get("checkpoint_id")
        if (
            pointer.get("schema_version") != 1
            or pointer.get("relative_path") != f"checkpoints/{identifier}"
        ):
            raise ValueError("Latest checkpoint pointer is invalid")
        selected = run / "checkpoints" / str(identifier)
    if selected.parent != run / "checkpoints":
        raise ValueError("Resume checkpoint must belong to this package's run")
    details = read_checkpoint_details(selected)
    if details["checkpoint_id"] != selected.name or details["kind"] != "learner":
        raise ValueError("Resume checkpoint identity differs from its directory")
    return selected


def start(
    package: str | Path, *, resume: bool = False, checkpoint: str | None = None
) -> Record:
    """Start one detached supervisor for an explicitly requested launch/resume.

    Validates the package first, then serializes starts using a local file lock.
    A live matching supervisor, trainer or leftover worker blocks either start.
    An existing run requires resume. stdin is closed; stdout/stderr append to
    logs/process.log. Returns
    exact process identity, command and mode after recording them durably.
    This function is only called by the user's launch/resume command.
    """
    root = Path(package).absolute()
    if checkpoint is not None and not resume:
        raise ValueError("An explicit checkpoint requires the resume command")
    manifest = validate_package(root)
    with (root / ".launch.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_path = root / "process.json"
        if status_path.exists() and _alive(_read(status_path)):
            raise RuntimeError("This package already has an active training process")
        if not resume and (root / "run").exists() and any((root / "run").iterdir()):
            raise ValueError("This package has a run already; use its resume command")
        selected = _checkpoint(root, checkpoint) if resume else None
        python = root / ".venv" / "bin" / "python"
        command = [
            str(python),
            "-I",
            "-m",
            "marl_battlegrounds.training._launch",
            "supervise",
            str(root),
        ]
        if selected is not None:
            command += ["--checkpoint", str(selected)]
        with (root / "logs" / "process.log").open("ab", buffering=0) as log:
            child = subprocess.Popen(
                command,
                cwd=root / "source",
                env=_environment(manifest["gpu"]["uuid"]),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        record = {
            "schema_version": 1,
            "state": "starting",
            "started_at": utc_now(),
            "process": process_identity(child.pid),
            "command": command,
            "mode": "resume" if resume else "start",
        }
        atomic_json(status_path, record)
        return record


def supervise_command(
    package: Path,
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    deadline_at: float | None = None,
    stop_grace_seconds: float | None = None,
) -> int:
    """Run one foreground trainer inside the already detached supervisor.

    The trainer starts a private session. Its ordinary workers inherit that
    group, including CPU validation. Stop signals reach the whole group. On
    any trainer exit, cleanup also stops leftover workers before reaping the
    trainer and publishing exit status. Cleanup has a bounded TERM/KILL wait;
    unkillable members remain recorded and block restart. env defaults to the
    inherited isolated environment. This helper never treats process success
    as learned competence. A deliberately detached worker is outside this
    contract; the built-in validation worker does not detach. A pipe gate stops
    the trainer from running before its group identity is durable; if this
    supervisor dies before release, the waiting child exits on pipe closure.
    Print short UTC start/outcome lines. Full machine-readable process identity,
    command, cleanup and exit facts remain in process.json.

    Parameters
    ----------
    package : Path
        Existing job directory. process.json and .launch.lock are written here.
    command : list[str]
        Trainer executable and arguments. No shell expands these strings.
    env : dict[str, str] or None, optional
        Complete child environment, or None to inherit this process's environment.
    deadline_at : float or None, optional
        Absolute UNIX time in seconds. None imposes no deadline. The remaining
        time is converted once to a monotonic clock, so later wall-clock changes
        do not extend this attempt. Expiry requests bounded group cleanup even
        during compilation or evaluation. The caller owns any shared deadline
        across jobs or resumed attempts; this helper never extends it.
        This is when cleanup starts. Subtract cleanup_reserve_seconds from a
        required finish time when all cleanup must fit before that time.
    stop_grace_seconds : float or None, optional
        Positive finite seconds before forced group shutdown. None keeps the
        usual ten-second grace. An outer supervisor with inner supervisors in
        its child group must allow their complete cleanup before forcing them
        to exit; the search uses twenty seconds. The inner supervisor remains
        responsible for its own worker session. No saved group IDs are signaled.

    Returns
    -------
    int
        Child exit code, 124 on deadline expiry, or 128 + signal when a stop
        request caught a child that exited zero. Cleanup errors return failure.
        The raw child code stays in the cleanup record. No stopped job returns
        success merely because its child handled a signal and exited zero.

    Raises
    ------
    ValueError
        deadline_at is not a finite UNIX time, or the stop grace is invalid.
    OSError
        Child launch or process-record publication fails. Cleanup is still tried.
    """
    cleanup_reserve_seconds(stop_grace_seconds=stop_grace_seconds)
    if deadline_at is not None and (
        isinstance(deadline_at, bool)
        or not isinstance(cast(object, deadline_at), (int, float))
        or not math.isfinite(deadline_at)
    ):
        raise ValueError("Deadline must be a finite UNIX timestamp")
    stop_at = (
        None
        if deadline_at is None
        else time.monotonic() + max(0.0, deadline_at - time.time())
    )
    record: Record = {
        "schema_version": 1,
        "state": "running",
        "started_at": utc_now(),
        "process": process_identity(),
        "command": command,
    }
    if deadline_at is not None:
        record["deadline_at"] = deadline_at
    if stop_grace_seconds is not None:
        record["stop_grace_seconds"] = stop_grace_seconds
    with (package / ".launch.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        atomic_json(package / "process.json", record)
    print(f"[{record['started_at']}] Process Started | Job {package.name}", flush=True)
    code = 1
    child: subprocess.Popen[bytes] | None = None
    previous: dict[int, Any] = {}

    def forward(number: int, _frame: object) -> None:
        """Request group cleanup in the main loop without reaping the leader."""
        record.setdefault("stop_signal", number)
        record.setdefault("stop_reason", "signal")

    def deadline_expired() -> bool:
        """Request a deadline stop using the attempt's fixed monotonic clock."""
        if stop_at is not None and time.monotonic() >= stop_at:
            record.setdefault("stop_reason", "deadline")
            record.setdefault("stop_signal", signal.SIGTERM)
            return True
        return False

    try:
        for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            previous[number] = signal.signal(number, forward)
        if deadline_expired():
            return 124
        read_fd, write_fd = os.pipe()
        try:
            child = subprocess.Popen(
                [sys.executable, "-I", "-c", _TRAINER_GATE, str(read_fd), *command],
                env=env,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                pass_fds=(read_fd,),
            )
            record["trainer"] = process_identity(child.pid)
            record["trainer_group"] = record["trainer"]
            atomic_json(package / "process.json", record)
            if not deadline_expired() and "stop_signal" not in record:
                os.write(write_fd, b"1")
        finally:
            os.close(read_fd)
            os.close(write_fd)
        while "stop_signal" not in record:
            if deadline_expired():
                break
            if (
                os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                is not None
            ):
                break
            time.sleep(0.05)
    except BaseException as error:
        record["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
        raise
    finally:
        try:
            if child is not None:
                cleanup = _finish_trainer(
                    child,
                    record.get("stop_signal", signal.SIGTERM),
                    stop_grace_seconds=stop_grace_seconds,
                )
                record["cleanup"] = cleanup
                result = cleanup["trainer_exit_code"]
                code = result if type(result) is int else 1
                if cleanup["remaining_pids"] and code == 0:
                    code = 1
        except Exception as error:
            record["cleanup_error"] = f"{type(error).__name__}: {error}"
            code = 1
            traceback.print_exc()
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)
        if "error" in record and code == 0:
            code = 1
        if record.get("stop_reason") == "deadline":
            code = 124
        elif "stop_signal" in record and code == 0:
            code = 128 + record["stop_signal"]
        record.update({"state": "exited", "exit_code": code, "finished_at": utc_now()})
        atomic_json(package / "process.json", record)
        outcome = "Process Finished" if code == 0 else "Process Failed"
        reason = record.get("stop_reason")
        suffix = f" | Reason {str(reason).capitalize()}" if reason else ""
        print(
            f"[{record['finished_at']}] {outcome} | Job {package.name} | "
            f"Exit Code {code}{suffix}",
            flush=True,
        )
    return code


def status(package: str | Path) -> Record:
    """Read process and scientific status without launching a backend or job.

    Return separate alive/process/run fields plus recovery and effective_state.
    Completion requires the owned trainer's completed run status and a clean
    supervisor exit. A failed exit overrides stale run status; missing live
    ownership without confirmed completion means stopped. A recovery marker
    takes precedence over all older records. Reading status does not recover or
    mutate any saved file or initialize JAX.
    """
    root = Path(package).absolute()
    process = _read(root / "process.json") if (root / "process.json").exists() else {}
    run = (
        _read(root / "run" / "status.json")
        if (root / "run" / "status.json").exists()
        else {}
    )
    marker = root / "run" / "checkpoint_recovery.json"
    recovery = _read(marker) if marker.exists() else None
    alive = _alive(process)
    current_trainer = isinstance(process.get("trainer"), dict) and process[
        "trainer"
    ] == run.get("process")
    if recovery is not None:
        effective_state = "recovering"
        progress = (
            "Recovery unfinished; resume the checkpoint named in the recovery record"
        )
    elif process.get("state") == "exited" and alive:
        effective_state = "stopping"
        progress = "Owned workers remain active; restart is blocked until they exit"
    elif process.get("state") == "exited":
        code = process.get("exit_code")
        if type(code) is int and code != 0:
            effective_state = "failed"
            progress = (
                f"Failed: process exited with code {code}; inspect logs/process.log"
            )
        elif (
            type(code) is int
            and code == 0
            and current_trainer
            and run.get("status") == "complete"
            and run.get("phase") == "complete"
        ):
            effective_state = "complete"
            progress = progress_text(run)
        else:
            effective_state = "stopped"
            progress = (
                "Stopped without a confirmed completed run; inspect logs/process.log"
            )
    elif alive:
        if current_trainer and run.get("phase") != "complete":
            effective_state = run.get("phase", "running")
            progress = progress_text(run)
        else:
            effective_state = (
                "starting" if process.get("state") == "starting" else "running"
            )
            progress = "Process is active; waiting for its current run or exit record"
    else:
        effective_state = "stopped" if process or run else "not_started"
        progress = (
            "Stopped: no matching process is running; completion is unconfirmed"
            if process or run
            else "Not started"
        )
    return {
        "alive": alive,
        "process": process,
        "run": run,
        "recovery": recovery,
        "effective_state": effective_state,
        "progress": progress,
    }


def main(argv: list[str] | None = None) -> int:
    """Dispatch prepared start/status/resume commands and the internal supervisor.

    argv defaults to process arguments. Start/resume perform the explicit job
    action; status only prints records. Errors propagate with nonzero process
    status and logs. The internal supervisor executes the same public train CLI.
    """
    parser = argparse.ArgumentParser(description="Run one fixed PPO package")
    parser.add_argument("action", choices=("start", "resume", "status", "supervise"))
    parser.add_argument("package", type=Path)
    parser.add_argument("--checkpoint")
    arguments = parser.parse_args(argv)
    if arguments.action == "status":
        print(json.dumps(status(arguments.package), indent=2))
        return 0
    if arguments.action != "supervise":
        print(
            json.dumps(
                start(
                    arguments.package,
                    resume=arguments.action == "resume",
                    checkpoint=arguments.checkpoint,
                ),
                indent=2,
            )
        )
        return 0
    package = arguments.package.absolute()
    try:
        manifest = validate_package(package)
        command = [
            str(package / ".venv" / "bin" / "python"),
            "-I",
            "-m",
            "marl_battlegrounds",
            "train",
        ]
        if arguments.checkpoint is None:
            command += [
                "--config",
                str(package / "config.json"),
                "--output-dir",
                str(package / "run"),
            ]
        else:
            command += [
                "--resume-from",
                str(_checkpoint(package, arguments.checkpoint)),
            ]
        return supervise_command(
            package, command, env=_environment(manifest["gpu"]["uuid"])
        )
    except BaseException as error:
        with (package / ".launch.lock").open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            status_path = package / "process.json"
            previous = _read(status_path) if status_path.exists() else {}
            if previous.get("process") != process_identity():
                previous = {}
            atomic_json(
                status_path,
                {
                    **previous,
                    "schema_version": 1,
                    "state": "exited",
                    "process": process_identity(),
                    "exit_code": 1,
                    "finished_at": utc_now(),
                    "error": f"{type(error).__name__}: {error}",
                },
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
