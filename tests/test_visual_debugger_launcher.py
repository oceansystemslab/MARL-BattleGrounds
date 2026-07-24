"""CLI and shell-launcher regression tests, including dependency isolation."""

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from scripts.dev.geometry_debug_renderer import build_parser, main
from scripts.dev.visual_debugger.scenarios import SCENARIOS

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_PYTHON_ENTRYPOINT = _REPOSITORY_ROOT / "scripts" / "dev" / "geometry_debug_renderer.py"
_SHELL_LAUNCHER = _REPOSITORY_ROOT / "scripts" / "dev" / "run_geometry_renderer.sh"


def test_parser_exposes_complete_cli_contract() -> None:
    parser = build_parser()
    args = parser.parse_args(
        (
            "--scenario",
            "status_stack",
            "--seed",
            "41",
            "--controlled-slot",
            "5",
            "--static",
            "--verbose",
            "--no-ranges",
        )
    )

    assert args.scenario == "status_stack"
    assert args.seed == 41
    assert args.controlled_slot == 5
    assert args.static
    assert args.verbose
    assert args.ranges is False


def test_help_contains_every_option_control_inspector_and_scenario() -> None:
    result = subprocess.run(
        (sys.executable, str(_PYTHON_ENTRYPOINT), "--help"),
        cwd="/tmp",
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    for option in (
        "--scenario",
        "--list-scenarios",
        "--seed",
        "--controlled-slot",
        "--static",
        "--verbose",
        "--ranges",
        "--no-ranges",
    ):
        assert option in result.stdout
    for control in (
        "Tab / Shift+Tab",
        "left click",
        "right click / Escape",
        "Space / Enter",
        "[ / ]",
    ):
        assert control in result.stdout
    for inspector in ("TARGET", "GEOMETRY", "LEGALITY"):
        assert inspector in result.stdout
    for scenario_name in SCENARIOS:
        assert scenario_name in result.stdout


def test_list_scenarios_is_stable_and_does_not_import_matplotlib() -> None:
    code = (
        "import sys; "
        "from scripts.dev.geometry_debug_renderer import main; "
        "status=main(['--list-scenarios']); "
        "print('STATUS', status); "
        "print('MATPLOTLIB', any(n == 'matplotlib' or n.startswith('matplotlib.') "
        "for n in sys.modules))"
    )
    result = subprocess.run(
        (sys.executable, "-c", code),
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "STATUS 0" in result.stdout
    assert "MATPLOTLIB False" in result.stdout
    positions = [result.stdout.index(name) for name in SCENARIOS]
    assert positions == sorted(positions)
    assert "interactive" in result.stdout
    assert "scripted" in result.stdout


def test_lazy_rendering_and_debugger_models_import_without_matplotlib() -> None:
    code = (
        "import sys; "
        "import marl_battlegrounds.rendering; "
        "import scripts.dev.visual_debugger.model; "
        "import scripts.dev.visual_debugger.targeting; "
        "print(any(n == 'matplotlib' or n.startswith('matplotlib.') "
        "for n in sys.modules))"
    )
    result = subprocess.run(
        (sys.executable, "-c", code),
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "False"


def test_missing_matplotlib_is_actionable_and_returns_two(tmp_path: Path) -> None:
    fake_package = tmp_path / "matplotlib"
    fake_package.mkdir()
    (fake_package / "__init__.py").write_text(
        "raise ImportError('blocked by launcher test')\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(tmp_path), str(_REPOSITORY_ROOT)))
    result = subprocess.run(
        (
            sys.executable,
            str(_PYTHON_ENTRYPOINT),
            "--scenario",
            "arena_5v5",
            "--static",
        ),
        cwd=_REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert (
        "error: Matplotlib is required for the visual debugger. "
        "Run 'uv sync --extra viz --extra dev'."
    ) in result.stderr


@pytest.mark.parametrize(
    "argv",
    (
        ("--scenario", "missing"),
        ("--scenario", "acceptance_lane_lab", "--controlled-slot", "1"),
        ("--seed", "not-an-int"),
    ),
)
def test_invalid_cli_inputs_use_argparse_exit_two(argv: tuple[str, ...]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(argv)
    assert exc_info.value.code == 2


def _write_fake_uv(
    tmp_path: Path,
    *,
    exit_code: int,
) -> tuple[Path, Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    record = tmp_path / "uv-args.txt"
    uv = fake_bin / "uv"
    uv.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s\\n\' "$@" > "${UV_RECORD}"\n'
        f"exit {exit_code}\n",
        encoding="utf-8",
    )
    uv.chmod(uv.stat().st_mode | stat.S_IXUSR)
    return fake_bin, record


def test_launcher_resolves_root_outside_repository_cwd_and_forwards_arguments(
    tmp_path: Path,
) -> None:
    fake_bin, record = _write_fake_uv(tmp_path, exit_code=0)
    environment = os.environ.copy()
    environment["PATH"] = os.pathsep.join((str(fake_bin), "/usr/bin", "/bin"))
    environment["UV_RECORD"] = str(record)
    result = subprocess.run(
        (
            str(_SHELL_LAUNCHER),
            "--scenario",
            "status_stack",
            "--seed",
            "19",
            "--verbose",
            "--no-ranges",
        ),
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert record.read_text(encoding="utf-8").splitlines() == [
        "run",
        "--project",
        str(_REPOSITORY_ROOT),
        "--extra",
        "viz",
        "python",
        str(_PYTHON_ENTRYPOINT),
        "--scenario",
        "status_stack",
        "--seed",
        "19",
        "--verbose",
        "--no-ranges",
    ]


def test_launcher_propagates_uv_exit_code(tmp_path: Path) -> None:
    fake_bin, record = _write_fake_uv(tmp_path, exit_code=23)
    environment = os.environ.copy()
    environment["PATH"] = os.pathsep.join((str(fake_bin), "/usr/bin", "/bin"))
    environment["UV_RECORD"] = str(record)
    result = subprocess.run(
        (str(_SHELL_LAUNCHER), "--list-scenarios"),
        cwd=tmp_path,
        env=environment,
        check=False,
    )

    assert result.returncode == 23


def test_launcher_reports_missing_uv() -> None:
    environment = os.environ.copy()
    environment["PATH"] = "/usr/bin:/bin"
    result = subprocess.run(
        ("bash", str(_SHELL_LAUNCHER), "--help"),
        cwd="/tmp",
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 127
    assert (
        "error: uv is required; install uv and run 'uv sync --extra viz --extra dev'."
    ) in result.stderr


def test_launcher_is_executable() -> None:
    assert _SHELL_LAUNCHER.stat().st_mode & stat.S_IXUSR
