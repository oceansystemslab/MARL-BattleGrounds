"""CLI and shell-launcher regression tests, including dependency isolation."""

import os
import stat
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path
from typing import cast

import pytest
from scripts.dev.debug_renderer import build_parser, main
from scripts.dev.visual_debugger.evaluation_bridge import (
    DebuggerEvaluationLaunchSpecificationV1,
)
from scripts.dev.visual_debugger.model import DebuggerScenario
from scripts.dev.visual_debugger.scenarios import (
    RESEARCHER_SCENARIOS,
    STRESS_SCENARIOS,
)
from scripts.dev.visual_debugger.service import DebuggerService

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_PYTHON_ENTRYPOINT = _REPOSITORY_ROOT / "scripts" / "dev" / "debug_renderer.py"
_SHELL_LAUNCHER = _REPOSITORY_ROOT / "scripts" / "dev" / "run_debug_renderer.sh"
_OLD_PYTHON_ENTRYPOINT = (
    _REPOSITORY_ROOT / "scripts" / "dev" / "geometry_debug_renderer.py"
)
_OLD_SHELL_LAUNCHER = _REPOSITORY_ROOT / "scripts" / "dev" / "run_geometry_renderer.sh"
_HAS_MATPLOTLIB = find_spec("matplotlib") is not None
_HAS_PYPLOT = _HAS_MATPLOTLIB and find_spec("matplotlib.pyplot") is not None


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
            "--no-open",
            "--port",
            "8123",
            "--include-stress",
            "--view",
            "pov",
            "--preset",
            "debug",
            "--verbose",
            "--no-ranges",
        )
    )

    assert args.scenario == "status_stack"
    assert args.seed == 41
    assert args.controlled_slot == 5
    assert args.static
    assert args.no_open
    assert args.port == 8123
    assert args.include_stress
    assert args.view == "pov"
    assert args.preset == "debug"
    assert args.verbose
    assert args.ranges is False


def test_parser_exposes_narrow_static_replay_contract() -> None:
    parser = build_parser()
    args = parser.parse_args(
        (
            "--replay",
            "episode.marlbg-replay.json",
            "--static",
            "--frame-index",
            "7",
            "--no-ranges",
        )
    )

    assert args.replay == Path("episode.marlbg-replay.json")
    assert args.static
    assert args.frame_index == 7
    assert args.scenario is None
    assert args.ranges is False


@pytest.mark.parametrize(
    "argv, message",
    (
        (("--frame-index", "1"), "--frame-index requires --replay"),
        (("--replay", "x.marlbg-replay.json"), "--replay currently requires"),
        (
            (
                "--replay",
                "x.marlbg-replay.json",
                "--static",
                "--scenario",
                "arena_5v5",
            ),
            "--scenario is unavailable with --replay",
        ),
    ),
)
def test_static_replay_cli_rejects_live_only_combinations(
    argv: tuple[str, ...],
    message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(argv)
    assert exc_info.value.code == 2
    assert message in capsys.readouterr().err


def test_replay_static_cli_import_path_is_core_and_jax_free() -> None:
    code = """
import sys
from scripts.dev.debug_renderer import build_parser
parser = build_parser()
parser.parse_args(['--replay', 'episode.marlbg-replay.json', '--static'])
for prefix in ('jax', 'jaxlib', 'numpy', 'marl_battlegrounds.core'):
    loaded = any(
        name == prefix or name.startswith(prefix + '.') for name in sys.modules
    )
    assert not loaded, prefix
print('isolated')
"""
    result = subprocess.run(
        (sys.executable, "-c", code),
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "isolated"


def test_parser_rejects_abbreviated_options() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(("--stat",))
    assert exc_info.value.code == 2


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
        "--no-open",
        "--port",
        "--include-stress",
        "--view",
        "--preset",
        "--verbose",
        "--ranges",
        "--no-ranges",
    ):
        assert option in result.stdout
    assert "--ui" not in result.stdout
    for control in (
        "Tab / Shift+Tab",
        "left click",
        "Shift+left click",
        "right click / Escape",
        "Shift+R",
        "arrow keys",
        "X",
        "Space / Enter",
        "P",
        "[ / ]",
        "Scenario/View/Preset",
        "0.5x / 1x / 2x / Off",
        "Reconnect",
    ):
        assert control in result.stdout
    assert "every staged action as one joint turn" in result.stdout
    assert "agent POV: submit only the controlled actor" in result.stdout
    assert "advance the next registered scripted frame" in result.stdout
    for inspector in ("SELECTED TARGET", "PENDING ACTION", "TECHNICAL DETAILS"):
        assert inspector in result.stdout
    for scenario_name in RESEARCHER_SCENARIOS:
        assert scenario_name in result.stdout
    for scenario_name in STRESS_SCENARIOS:
        assert scenario_name not in result.stdout
    assert "acceptance_lane_lab" not in result.stdout
    assert "geometry_debug_renderer" not in result.stdout


def test_list_scenarios_is_stable_and_does_not_import_matplotlib() -> None:
    code = (
        "import sys; "
        "from scripts.dev.debug_renderer import main; "
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
    positions = [result.stdout.index(name) for name in RESEARCHER_SCENARIOS]
    assert positions == sorted(positions)
    for scenario_name in STRESS_SCENARIOS:
        assert scenario_name not in result.stdout
    assert "interactive" in result.stdout
    assert "scripted" in result.stdout


def test_list_scenarios_includes_stress_only_when_explicitly_requested() -> None:
    code = (
        "from scripts.dev.debug_renderer import main; "
        "raise SystemExit(main(['--list-scenarios', '--include-stress']))"
    )
    result = subprocess.run(
        (sys.executable, "-c", code),
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    for scenario_name in (*RESEARCHER_SCENARIOS, *STRESS_SCENARIOS):
        assert scenario_name in result.stdout


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
        "error: Matplotlib is required for static Visual Debugger and Analyzer "
        "snapshots. "
        "Run 'uv sync --extra viz --extra dev'."
    ) in result.stderr


@pytest.mark.parametrize(
    "argv",
    (
        ("--scenario", "missing"),
        ("--scenario", "charge_convergence"),
        ("--scenario", "arena_5v5", "--controlled-slot", "-1"),
        ("--seed", "not-an-int"),
    ),
)
def test_invalid_cli_inputs_use_argparse_exit_two(argv: tuple[str, ...]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(argv)
    assert exc_info.value.code == 2


@pytest.mark.parametrize(
    "argv",
    (
        ("--scenario", "missing"),
        ("--scenario", "charge_convergence"),
        ("--scenario", "arena_5v5", "--controlled-slot", "-1"),
    ),
)
def test_semantic_cli_validation_precedes_runtime_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    argv: tuple[str, ...],
) -> None:
    import scripts.dev.visual_debugger.server as server_module
    import scripts.dev.visual_debugger.static_renderer as static_module

    def fail_if_dispatched(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("runtime must not start before CLI validation")

    monkeypatch.setattr(server_module, "serve_browser_debugger", fail_if_dispatched)
    monkeypatch.setattr(static_module, "run_static_renderer", fail_if_dispatched)

    with pytest.raises(SystemExit) as exc_info:
        main(argv)

    assert exc_info.value.code == 2


def test_browser_default_builds_service_and_forwards_lifecycle_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.dev.visual_debugger.server as server_module

    observed: dict[str, object] = {}

    def fake_serve(
        service: object,
        *,
        asset_root: Path,
        port: int,
        open_browser: bool,
    ) -> int:
        observed.update(
            service=service,
            asset_root=asset_root,
            port=port,
            open_browser=open_browser,
        )
        return 17

    monkeypatch.setattr(server_module, "serve_browser_debugger", fake_serve)
    result = main(
        (
            "--no-open",
            "--port",
            "8123",
            "--view",
            "pov",
            "--preset",
            "debug",
            "--no-ranges",
        )
    )

    assert result == 17
    assert observed["asset_root"] == _REPOSITORY_ROOT / "web" / "visual_debugger"
    assert observed["port"] == 8123
    assert observed["open_browser"] is False
    frame = cast(DebuggerService, observed["service"]).current_frame()
    assert frame.view_mode == "pov"
    assert frame.preset == "debug"
    assert not hasattr(frame.projection.scene, "ranges")


def test_static_flag_uses_only_the_stateless_snapshot_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.dev.visual_debugger.server as server_module
    import scripts.dev.visual_debugger.static_renderer as static_module

    observed: dict[str, object] = {}

    def fake_static(
        *,
        scenario: object,
        seed: int,
        evaluation_launch_specification: object,
        controlled_global_slot: int | None,
        verbose: bool,
        show_ranges: bool,
    ) -> int:
        observed.update(
            scenario=scenario,
            seed=seed,
            evaluation_launch_specification=evaluation_launch_specification,
            controlled_global_slot=controlled_global_slot,
            verbose=verbose,
            show_ranges=show_ranges,
        )
        return 19

    def fail_if_served(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("static mode must not start the browser server")

    monkeypatch.setattr(static_module, "run_static_renderer", fake_static)
    monkeypatch.setattr(server_module, "serve_browser_debugger", fail_if_served)

    result = main(
        (
            "--static",
            "--scenario",
            "status_stack",
            "--seed",
            "13",
            "--controlled-slot",
            "5",
            "--verbose",
            "--no-ranges",
        )
    )

    assert result == 19
    assert cast(DebuggerScenario, observed["scenario"]).name == "status_stack"
    assert observed["seed"] == 13
    assert (
        cast(
            DebuggerEvaluationLaunchSpecificationV1,
            observed["evaluation_launch_specification"],
        ).root_seed
        == 13
    )
    assert observed["controlled_global_slot"] == 5
    assert observed["verbose"] is True
    assert observed["show_ranges"] is False


def test_stress_browser_launch_requires_and_accepts_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.dev.visual_debugger.server as server_module

    def ignore_browser_launch(
        _service: object,
        *,
        asset_root: Path,
        port: int,
        open_browser: bool,
    ) -> int:
        del asset_root, port, open_browser
        return 0

    monkeypatch.setattr(
        server_module,
        "serve_browser_debugger",
        ignore_browser_launch,
    )
    with pytest.raises(SystemExit) as exc_info:
        main(("--scenario", "charge_convergence"))
    assert exc_info.value.code == 2

    assert (
        main(
            (
                "--scenario",
                "charge_convergence",
                "--include-stress",
                "--no-open",
            )
        )
        == 0
    )


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
        "python",
        str(_PYTHON_ENTRYPOINT),
        "--scenario",
        "status_stack",
        "--seed",
        "19",
        "--verbose",
        "--no-ranges",
    ]


def test_launcher_activates_viz_extra_only_for_static_snapshots(
    tmp_path: Path,
) -> None:
    fake_bin, record = _write_fake_uv(tmp_path, exit_code=0)
    environment = os.environ.copy()
    environment["PATH"] = os.pathsep.join((str(fake_bin), "/usr/bin", "/bin"))
    environment["UV_RECORD"] = str(record)

    result = subprocess.run(
        (str(_SHELL_LAUNCHER), "--static", "--scenario", "arena_5v5"),
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
        "--static",
        "--scenario",
        "arena_5v5",
    ]


@pytest.mark.skipif(
    not _HAS_PYPLOT,
    reason="the optional viz extra is not installed",
)
def test_static_cli_completes_with_a_headless_matplotlib_backend() -> None:
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"

    result = subprocess.run(
        (
            sys.executable,
            str(_PYTHON_ENTRYPOINT),
            "--static",
            "--scenario",
            "arena_5v5",
            "--no-ranges",
        ),
        cwd="/tmp",
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr


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
        "error: uv is required to run the Visual Debugger and Analyzer."
        in result.stderr
    )


def test_launcher_is_executable() -> None:
    assert _SHELL_LAUNCHER.stat().st_mode & stat.S_IXUSR


def test_launcher_rename_has_no_compatibility_wrapper() -> None:
    assert _PYTHON_ENTRYPOINT.is_file()
    assert _SHELL_LAUNCHER.is_file()
    assert not _OLD_PYTHON_ENTRYPOINT.exists()
    assert not _OLD_SHELL_LAUNCHER.exists()
