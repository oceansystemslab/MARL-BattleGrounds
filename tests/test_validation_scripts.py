"""Black-box contracts for the local contributor and GPU validation scripts."""

import os
import re
import shutil
import stat
import subprocess
import textwrap
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_DIRECTORY = _REPOSITORY_ROOT / "scripts" / "dev"
_PARALLEL_SCRIPT = "validation_parallel.sh"


def test_hosted_ci_preserves_main_runs_and_uses_runaway_timeout_headroom() -> None:
    workflow = (_REPOSITORY_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )

    assert (
        "group: ci-${{ github.workflow }}-${{ github.ref == 'refs/heads/main' "
        "&& github.run_id || github.ref }}"
    ) in workflow
    assert "cancel-in-progress: true" in workflow
    assert re.search(
        r"(?ms)^  python-test-gates:.*?^    timeout-minutes: 12$",
        workflow,
    )
    assert re.search(
        r"(?ms)^  frontend-browser-gates:.*?^    timeout-minutes: 8$",
        workflow,
    )


def _run(
    arguments: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        cwd=cwd,
        env=env,
        check=check,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _write_executable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _copy_validation_scripts(root: Path, *names: str) -> None:
    destination = root / "scripts" / "dev"
    destination.mkdir(parents=True, exist_ok=True)
    requested = set(names)
    if (_SCRIPT_DIRECTORY / _PARALLEL_SCRIPT).is_file():
        requested.add(_PARALLEL_SCRIPT)
    for name in sorted(requested):
        source = _SCRIPT_DIRECTORY / name
        assert source.is_file(), f"missing validation script: {source}"
        copied = destination / name
        shutil.copy2(source, copied)
        copied.chmod(copied.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return _run(["git", *arguments], cwd=root, check=True)


def _initialize_repository(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "--quiet")
    _git(root, "config", "user.name", "Validation Test")
    _git(root, "config", "user.email", "validation@example.invalid")


def _commit_everything(root: Path, message: str = "test baseline") -> None:
    _git(root, "add", ".")
    _git(root, "commit", "--quiet", "-m", message)


def _fake_tool_script() -> str:
    return r"""
        #!/usr/bin/env bash
        set -eu

        tool="$(basename -- "$0")"
        arguments="$*"
        printf 'start\t%s\t%s\t%s\t%s\t%s\t%s\n' \
          "$tool" "$PWD" "${JAX_PLATFORMS-}" \
          "${PLAYWRIGHT_OUTPUT_DIR-${MARL_PLAYWRIGHT_OUTPUT_DIR-}}" \
          "$arguments" "${CI-}" >> "$FAKE_TOOL_LOG"

        if [[ -n "${FAKE_SLEEP_MATCH-}" && \
              "$arguments" == *"$FAKE_SLEEP_MATCH"* ]]; then
          sleep "${FAKE_SLEEP_SECONDS:-0.25}"
        fi
        if [[ -n "${FAKE_FAIL_MATCH-}" && "$arguments" == *"$FAKE_FAIL_MATCH"* ]]; then
          printf 'failed\t%s\t%s\n' "$tool" "$arguments" >> "$FAKE_TOOL_LOG"
          exit "${FAKE_FAIL_STATUS:-23}"
        fi
        if [[ -n "${FAKE_MUTATE_MATCH-}" && \
              "$arguments" == *"$FAKE_MUTATE_MATCH"* ]]; then
          printf 'changed during validation\n' > "$FAKE_MUTATE_PATH"
        fi

        printf 'finish\t%s\t%s\n' "$tool" "$arguments" >> "$FAKE_TOOL_LOG"
    """


def _install_fake_tools(bin_directory: Path, *names: str) -> None:
    for name in names:
        _write_executable(bin_directory / name, _fake_tool_script())


def _environment(fake_bin: Path, log_path: Path, **updates: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "FAKE_TOOL_LOG": str(log_path),
        }
    )
    env.update(updates)
    return env


def _log_records(log_path: Path, record_type: str = "start") -> list[list[str]]:
    if not log_path.exists():
        return []
    return [
        line.split("\t")
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.startswith(f"{record_type}\t")
    ]


def _outside_directory(tmp_path: Path) -> Path:
    outside = tmp_path / "outside"
    outside.mkdir(exist_ok=True)
    return outside


def test_python_gate_is_root_independent_cpu_only_and_exactly_sharded(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    fake_bin = tmp_path / "fake-bin"
    log_path = tmp_path / "python-gate.log"
    _initialize_repository(repository)
    _copy_validation_scripts(repository, "check.sh")
    _install_fake_tools(fake_bin, "uv")

    for variable, value in (
        ("JAX_PLATFORM_NAME", "gpu"),
        ("JAX_XLA_BACKEND", "gpu"),
        ("PYTEST_ADDOPTS", "--ignore=tests"),
        ("JAX_DISABLE_JIT", "1"),
    ):
        bypass_log = tmp_path / f"python-bypass-{variable}.log"
        rejected = _run(
            [str(repository / "scripts" / "dev" / "check.sh")],
            cwd=_outside_directory(tmp_path),
            env=_environment(fake_bin, bypass_log, **{variable: value}),
        )
        assert rejected.returncode != 0
        assert _log_records(bypass_log) == []

    result = _run(
        [str(repository / "scripts" / "dev" / "check.sh")],
        cwd=_outside_directory(tmp_path),
        env=_environment(fake_bin, log_path),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    records = _log_records(log_path)
    uv_records = [record for record in records if record[1] == "uv"]
    assert len(uv_records) == 15
    assert {record[2] for record in uv_records} == {str(repository)}
    assert {record[3] for record in uv_records} == {"cpu"}

    invocations = [record[5] for record in uv_records]
    shard_selectors = {
        match.group(1)
        for invocation in invocations
        if (match := re.search(r"--ci-shard(?:=|\s+)(\d+/12)(?:\s|$)", invocation))
        is not None
    }
    assert shard_selectors == {f"{index}/12" for index in range(1, 13)}
    assert sum("ruff format --check ." in invocation for invocation in invocations) == 1
    assert sum("ruff check ." in invocation for invocation in invocations) == 1
    assert (
        sum(
            re.search(r"(?:^|\s)pyright(?:\s|$)", invocation) is not None
            for invocation in invocations
        )
        == 1
    )


def test_frontend_gate_is_root_independent_and_isolates_browser_outputs(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    fake_bin = tmp_path / "fake-bin"
    log_path = tmp_path / "frontend-gate.log"
    _initialize_repository(repository)
    _copy_validation_scripts(repository, "check_frontend.sh")
    _install_fake_tools(fake_bin, "npm", "node")

    for variable in (
        "JAX_PLATFORM_NAME",
        "JAX_XLA_BACKEND",
        "JAX_DISABLE_JIT",
        "MARL_CP4_C3_SHIELD_ONLY",
        "MARL_CP4_E_CAPTURE_DIR",
        "MARL_CP5_C_SLICE_ONLY",
        "MARL_CP5_SLICE_5_ONLY",
    ):
        selector_log = tmp_path / f"frontend-selector-{variable}.log"
        rejected = _run(
            [str(repository / "scripts" / "dev" / "check_frontend.sh")],
            cwd=_outside_directory(tmp_path),
            env=_environment(fake_bin, selector_log, **{variable: "1"}),
        )
        assert rejected.returncode != 0
        assert _log_records(selector_log) == []

    result = _run(
        [str(repository / "scripts" / "dev" / "check_frontend.sh")],
        cwd=_outside_directory(tmp_path),
        env=_environment(fake_bin, log_path),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    records = _log_records(log_path)
    assert {record[6] for record in records} == {"1"}
    assert {record[3] for record in records} == {"cpu"}
    npm_invocations = [record[5] for record in records if record[1] == "npm"]
    node_records = [record for record in records if record[1] == "node"]
    assert len(npm_invocations) == 4
    assert len(node_records) == 8
    frontend_root = repository / "web" / "visual_debugger"
    assert all(
        f"--prefix {frontend_root}" in invocation for invocation in npm_invocations
    )
    shard_runner = frontend_root / "e2e" / "support" / "run-ci-shard.js"
    assert all(str(shard_runner) in record[5] for record in node_records)
    assert sum("run format:check" in invocation for invocation in npm_invocations) == 1
    assert sum("run lint" in invocation for invocation in npm_invocations) == 1
    assert sum("run typecheck" in invocation for invocation in npm_invocations) == 1
    assert sum("run test:unit" in invocation for invocation in npm_invocations) == 1

    shard_selectors = {
        match.group(1)
        for record in node_records
        if (match := re.search(r"run-ci-shard\.js\s+(\d+/8)(?:\s|$)", record[5]))
        is not None
    }
    assert shard_selectors == {f"{index}/8" for index in range(1, 9)}

    output_directories: set[str] = set()
    for record in node_records:
        output_directory = record[4]
        if not output_directory:
            match = re.search(r"--output(?:=|\s+)(\S+)", record[5])
            assert match is not None, (
                f"browser shard has no isolated output: {record[5]}"
            )
            output_directory = match.group(1)
        output_directories.add(output_directory)
    assert len(output_directories) == 8

    failure_log = tmp_path / "frontend-early-failure.log"
    failed = _run(
        [str(repository / "scripts" / "dev" / "check_frontend.sh")],
        cwd=_outside_directory(tmp_path),
        env=_environment(
            fake_bin,
            failure_log,
            FAKE_FAIL_MATCH="format:check",
        ),
    )
    assert failed.returncode != 0
    failure_records = _log_records(failure_log)
    assert len([record for record in failure_records if record[1] == "node"]) == 8
    failure_npm_invocations = [
        record[5] for record in failure_records if record[1] == "npm"
    ]
    assert len(failure_npm_invocations) == 1
    assert "format:check" in failure_npm_invocations[0]


def test_python_gate_waits_for_all_workers_before_reporting_failure(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    fake_bin = tmp_path / "fake-bin"
    log_path = tmp_path / "wait-all.log"
    _initialize_repository(repository)
    _copy_validation_scripts(repository, "check.sh")
    _install_fake_tools(fake_bin, "uv")

    result = _run(
        [str(repository / "scripts" / "dev" / "check.sh")],
        cwd=_outside_directory(tmp_path),
        env=_environment(
            fake_bin,
            log_path,
            FAKE_FAIL_MATCH="--ci-shard=1/12 ",
            FAKE_SLEEP_MATCH="--ci-shard=12/12 ",
            FAKE_SLEEP_SECONDS="0.3",
        ),
    )

    assert result.returncode != 0
    starts = _log_records(log_path)
    assert len([record for record in starts if record[1] == "uv"]) == 15
    finishes = _log_records(log_path, "finish")
    assert any("12/12" in record[2] for record in finishes)


def test_precommit_gate_rejects_index_mutation_during_validation(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    log_path = tmp_path / "precommit.log"
    _initialize_repository(repository)
    _copy_validation_scripts(repository, "check_before_commit.sh")
    gate_body = r"""
        #!/usr/bin/env bash
        set -eu
        printf '%s\t%s\n' "$(basename -- "$0")" "$PWD" >> "$FAKE_TOOL_LOG"
        if [[ "$(basename -- "$0")" == "check.sh" ]]; then
          printf 'mutated during validation\n' > "$FAKE_REPOSITORY/candidate.txt"
          git -C "$FAKE_REPOSITORY" add candidate.txt
        fi
        sleep 0.1
    """
    _write_executable(repository / "scripts" / "dev" / "check.sh", gate_body)
    _write_executable(repository / "scripts" / "dev" / "check_frontend.sh", gate_body)
    (repository / "baseline.txt").write_text("baseline\n", encoding="utf-8")
    _commit_everything(repository)
    (repository / "candidate.txt").write_text("candidate\n", encoding="utf-8")
    _git(repository, "add", "candidate.txt")

    env = os.environ.copy()
    env.update(
        {
            "FAKE_REPOSITORY": str(repository),
            "FAKE_TOOL_LOG": str(log_path),
        }
    )
    result = _run(
        [str(repository / "scripts" / "dev" / "check_before_commit.sh")],
        cwd=_outside_directory(tmp_path),
        env=env,
    )

    assert result.returncode != 0
    output = (result.stdout + result.stderr).lower()
    assert "fingerprint" in output or "changed" in output or "modified" in output
    gate_records = log_path.read_text(encoding="utf-8").splitlines()
    assert {line.split("\t", 1)[0] for line in gate_records} == {
        "check.sh",
        "check_frontend.sh",
    }


def test_gpu_gate_distinguishes_clean_qualification_from_dirty_diagnostic(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    fake_bin = tmp_path / "fake-bin"
    clean_log = tmp_path / "gpu-clean.log"
    dirty_log = tmp_path / "gpu-dirty.log"
    _initialize_repository(repository)
    _copy_validation_scripts(repository, "check_gpu.sh")
    (repository / "tracked.txt").write_text("clean\n", encoding="utf-8")
    _commit_everything(repository)
    _install_fake_tools(fake_bin, "nvidia-smi", "uv")

    for variable, value in (
        ("JAX_SKIP_CUDA_CONSTRAINTS_CHECK", "false"),
        ("JAX_DISABLE_JIT", "1"),
        ("PYTEST_ADDOPTS", "--collect-only"),
    ):
        bypass_log = tmp_path / f"gpu-bypass-{variable}.log"
        bypassed = _run(
            [str(repository / "scripts" / "dev" / "check_gpu.sh")],
            cwd=_outside_directory(tmp_path),
            env=_environment(fake_bin, bypass_log, **{variable: value}),
        )
        assert bypassed.returncode != 0
        assert _log_records(bypass_log) == []

    clean = _run(
        [str(repository / "scripts" / "dev" / "check_gpu.sh")],
        cwd=_outside_directory(tmp_path),
        env=_environment(fake_bin, clean_log, JAX_PLATFORMS="cpu"),
    )
    assert clean.returncode == 0, clean.stdout + clean.stderr
    assert "qualification" in (clean.stdout + clean.stderr).lower()
    clean_records = _log_records(clean_log)
    assert clean_records
    assert {record[2] for record in clean_records} == {str(repository)}
    assert {record[3] for record in clean_records} == {"cuda"}
    pytest_invocations = [
        record[5]
        for record in clean_records
        if record[1] == "uv" and re.search(r"(?:^|\s)pytest(?:\s|$)", record[5])
    ]
    assert len(pytest_invocations) == 1
    assert set(re.findall(r"tests/\S+::\S+", pytest_invocations[0])) == {
        "tests/test_core_spine.py::test_that_step_can_be_jit_compiled",
        "tests/test_core_spine.py::test_step_can_run_in_scanned_rollout",
    }

    changed = _run(
        [str(repository / "scripts" / "dev" / "check_gpu.sh")],
        cwd=_outside_directory(tmp_path),
        env=_environment(
            fake_bin,
            clean_log,
            FAKE_MUTATE_MATCH="pytest",
            FAKE_MUTATE_PATH=str(repository / "tracked.txt"),
        ),
    )
    assert changed.returncode != 0
    assert "changed" in (changed.stdout + changed.stderr).lower()
    (repository / "tracked.txt").write_text("clean\n", encoding="utf-8")

    (repository / "tracked.txt").write_text("dirty\n", encoding="utf-8")
    rejected = _run(
        [str(repository / "scripts" / "dev" / "check_gpu.sh")],
        cwd=_outside_directory(tmp_path),
        env=_environment(fake_bin, dirty_log),
    )
    assert rejected.returncode != 0
    assert _log_records(dirty_log) == []

    diagnostic = _run(
        [str(repository / "scripts" / "dev" / "check_gpu.sh"), "--allow-dirty"],
        cwd=_outside_directory(tmp_path),
        env=_environment(fake_bin, dirty_log),
    )
    assert diagnostic.returncode == 0, diagnostic.stdout + diagnostic.stderr
    diagnostic_output = (diagnostic.stdout + diagnostic.stderr).lower()
    assert "diagnostic" in diagnostic_output
    assert "not" in diagnostic_output and "qualification" in diagnostic_output


def test_gpu_probe_failure_is_fatal_and_skips_focused_tests(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    fake_bin = tmp_path / "fake-bin"
    log_path = tmp_path / "gpu-failure.log"
    _initialize_repository(repository)
    _copy_validation_scripts(repository, "check_gpu.sh")
    (repository / "tracked.txt").write_text("clean\n", encoding="utf-8")
    _commit_everything(repository)
    _install_fake_tools(fake_bin, "nvidia-smi", "uv")

    result = _run(
        [str(repository / "scripts" / "dev" / "check_gpu.sh")],
        cwd=_outside_directory(tmp_path),
        env=_environment(fake_bin, log_path, FAKE_FAIL_MATCH="python"),
    )

    assert result.returncode != 0
    invocations = [record[5] for record in _log_records(log_path) if record[1] == "uv"]
    assert any(
        re.search(r"(?:^|\s)python(?:\s|$)", invocation) for invocation in invocations
    )
    assert not any(
        re.search(r"(?:^|\s)pytest(?:\s|$)", invocation) for invocation in invocations
    )
