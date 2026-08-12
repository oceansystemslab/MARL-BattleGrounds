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


def _write_valid_replay(tmp_path: Path) -> Path:
    """Write one small canonical replay for launcher-boundary integration tests."""
    from tests.evaluation_fixtures import captured_evaluation_trajectory

    from marl_battlegrounds.evaluation.metrics import build_evaluation_observer_v1
    from marl_battlegrounds.evaluation.replay import (
        RuntimeProvenanceV1,
        build_replay_bundle_v1,
    )
    from marl_battlegrounds.evaluation.replay_io import (
        canonical_replay_json_bytes_v1,
    )

    trajectory = captured_evaluation_trajectory(
        transition_count=1,
        expected_horizon=1,
        episode_id="launcher-replay-episode",
    )
    observer = build_evaluation_observer_v1(trajectory.context)
    observer.start(trajectory.frames[0])
    observer.append(trajectory.transitions[0], trajectory.frames[1])
    report = observer.finalize(completion_state="complete")
    bundle = build_replay_bundle_v1(
        observer,
        report,
        runtime_provenance=RuntimeProvenanceV1(
            python_version="3.14.0",
            package_version="0.0.0",
            jax_version="0.7.0",
            jaxlib_version="0.7.0",
            numpy_version="2.3.0",
            pydantic_version="2.11.0",
            platform="linux",
            machine="x86_64",
            backend="cpu",
            device="generic-cpu",
            precision="float32",
            environment_count=1,
            batch_shape=(1,),
            policy_execution_included=False,
        ),
    )
    replay_path = tmp_path / "launcher.marlbg-replay.json"
    replay_path.write_bytes(canonical_replay_json_bytes_v1(bundle.replay))
    return replay_path


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
        )
    )

    assert args.replay == Path("episode.marlbg-replay.json")
    assert args.static
    assert args.frame_index == 7
    assert not hasattr(args, "scenario")
    assert not hasattr(args, "ranges")


def test_parser_exposes_complete_browser_replay_contract() -> None:
    parser = build_parser()
    args = parser.parse_args(
        (
            "--replay",
            "episode.marlbg-replay.json",
            "--frame-index",
            "7",
            "--pov-slot",
            "5",
            "--view",
            "pov",
            "--preset",
            "debug",
            "--no-ranges",
            "--port",
            "0",
            "--no-open",
        )
    )

    assert args.replay == Path("episode.marlbg-replay.json")
    assert args.frame_index == 7
    assert args.pov_slot == 5
    assert args.view == "pov"
    assert args.preset == "debug"
    assert args.ranges is False
    assert args.port == 0
    assert args.no_open
    assert not hasattr(args, "static")
    assert not hasattr(args, "scenario")


@pytest.mark.parametrize(
    "argv, message",
    (
        (("--frame-index", "1"), "--frame-index requires --replay"),
        (("--pov-slot", "0"), "--pov-slot requires --replay"),
        (
            ("--replay", "x.marlbg-replay.json", "--static"),
            "--frame-index is required with --replay --static",
        ),
    ),
)
def test_replay_scoped_options_and_static_frame_requirements(
    argv: tuple[str, ...],
    message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(argv)
    assert exc_info.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    ("option", "label"),
    (
        (("--scenario", "arena_5v5"), "--scenario"),
        (("--list-scenarios",), "--list-scenarios"),
        (("--include-stress",), "--include-stress"),
        (("--seed", "0"), "--seed"),
        (("--controlled-slot", "0"), "--controlled-slot"),
        (("--pov-slot", "0"), "--pov-slot"),
        (("--view", "researcher"), "--view"),
        (("--preset", "analysis"), "--preset"),
        (("--ranges",), "--ranges/--no-ranges"),
        (("--no-ranges",), "--ranges/--no-ranges"),
        (("--port", "0"), "--port"),
        (("--no-open",), "--no-open"),
        (("--verbose",), "--verbose"),
    ),
)
def test_static_replay_rejects_every_non_exact_option_even_explicit_defaults(
    option: tuple[str, ...],
    label: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv = (
        "--replay",
        "x.marlbg-replay.json",
        "--static",
        "--frame-index",
        "0",
        *option,
    )

    with pytest.raises(SystemExit) as exc_info:
        main(argv)

    assert exc_info.value.code == 2
    error = capsys.readouterr().err
    assert f"{label} is unavailable with --replay --static" in error


@pytest.mark.parametrize(
    ("option", "label"),
    (
        (("--scenario", "arena_5v5"), "--scenario"),
        (("--list-scenarios",), "--list-scenarios"),
        (("--include-stress",), "--include-stress"),
        (("--seed", "0"), "--seed"),
        (("--controlled-slot", "0"), "--controlled-slot"),
        (("--verbose",), "--verbose"),
    ),
)
def test_browser_replay_rejects_live_options_even_explicit_defaults(
    option: tuple[str, ...],
    label: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(("--replay", "x.marlbg-replay.json", *option))

    assert exc_info.value.code == 2
    assert f"{label} is unavailable with --replay" in capsys.readouterr().err


def test_replay_static_cli_import_path_is_core_and_jax_free() -> None:
    code = """
import sys
from scripts.dev.debug_renderer import build_parser
parser = build_parser()
parser.parse_args([
    '--replay', 'episode.marlbg-replay.json',
    '--static', '--frame-index', '0',
])
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


def test_exact_static_replay_dispatches_only_the_stateless_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.dev.visual_debugger.server as server_module
    import scripts.dev.visual_debugger.static_renderer as static_module

    observed: dict[str, object] = {}

    def fake_static_replay(
        *,
        replay_path: Path,
        frame_index: int,
        show_ranges: bool,
    ) -> int:
        observed.update(
            replay_path=replay_path,
            frame_index=frame_index,
            show_ranges=show_ranges,
        )
        return 29

    def fail_if_served(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("static replay must not start the browser server")

    monkeypatch.setattr(
        static_module,
        "run_static_replay_renderer",
        fake_static_replay,
    )
    monkeypatch.setattr(server_module, "serve_browser_debugger", fail_if_served)

    result = main(
        (
            "--replay",
            "episode.marlbg-replay.json",
            "--static",
            "--frame-index",
            "7",
        )
    )

    assert result == 29
    assert observed == {
        "replay_path": Path("episode.marlbg-replay.json"),
        "frame_index": 7,
        "show_ranges": True,
    }


def test_browser_replay_loads_resolves_then_injects_exact_server_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.dev.visual_debugger import replay_service as replay_service_module
    from scripts.dev.visual_debugger import server as server_module
    from scripts.dev.visual_debugger.replay_protocol import (
        ReplayApiErrorV1,
        ReplayCommandRequestV1,
    )
    from scripts.dev.visual_debugger.server import (
        REPLAY_HTTP_ROUTES,
        HttpCoordinatorBinding,
    )

    from marl_battlegrounds.evaluation import replay_io as replay_io_module

    events: list[str] = []
    observed: dict[str, object] = {}
    bundle = object()

    class FakeReplayService:
        def current_frame(self) -> object:
            raise AssertionError("server should not request a frame in this test")

        def current_timeline(self) -> object:
            raise AssertionError("server should not request a timeline in this test")

        def apply_command(self, request: object) -> object:
            del request
            raise AssertionError("server should not apply a command in this test")

    service = FakeReplayService()

    def fake_load(path: Path) -> object:
        events.append("load")
        observed["replay_path"] = path
        return bundle

    def fake_service_factory(loaded: object, **kwargs: object) -> FakeReplayService:
        events.append("resolve")
        observed["bundle"] = loaded
        observed["service_options"] = kwargs
        return service

    def fake_serve(
        resolved_service: object,
        *,
        asset_root: Path,
        port: int,
        open_browser: bool,
        coordinator: HttpCoordinatorBinding,
    ) -> int:
        events.append("serve")
        observed.update(
            service=resolved_service,
            asset_root=asset_root,
            port=port,
            open_browser=open_browser,
            coordinator=coordinator,
        )
        return 31

    monkeypatch.setattr(replay_io_module, "load_replay_bundle_v1", fake_load)
    monkeypatch.setattr(
        replay_service_module,
        "ReplayViewerService",
        fake_service_factory,
    )
    monkeypatch.setattr(server_module, "serve_browser_debugger", fake_serve)

    result = main(
        (
            "--replay",
            "episode.marlbg-replay.json",
            "--frame-index",
            "7",
            "--pov-slot",
            "5",
            "--view",
            "pov",
            "--preset",
            "debug",
            "--no-ranges",
            "--port",
            "0",
            "--no-open",
        )
    )

    assert result == 31
    assert events == ["load", "resolve", "serve"]
    assert observed["replay_path"] == Path("episode.marlbg-replay.json")
    assert observed["bundle"] is bundle
    assert observed["service_options"] == {
        "initial_frame_index": 7,
        "view_mode": "pov",
        "pov_global_slot": 5,
        "preset": "debug",
        "show_ranges": False,
        "verbose": False,
    }
    assert observed["service"] is service
    assert observed["asset_root"] == _REPOSITORY_ROOT / "web" / "visual_debugger"
    assert observed["port"] == 0
    assert observed["open_browser"] is False
    coordinator = cast(HttpCoordinatorBinding, observed["coordinator"])
    assert coordinator.mode == "replay"
    assert coordinator.routes == REPLAY_HTTP_ROUTES
    assert coordinator.request_model is ReplayCommandRequestV1
    assert coordinator.error_factory is ReplayApiErrorV1
    assert coordinator.current_frame == service.current_frame
    assert coordinator.current_timeline == service.current_timeline
    assert coordinator.apply_command == service.apply_command


def test_invalid_missing_and_symlink_replays_fail_before_service_or_server_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts.dev.visual_debugger import replay_service as replay_service_module
    from scripts.dev.visual_debugger import server as server_module

    invalid = tmp_path / "invalid.marlbg-replay.json"
    invalid.write_text("{}", encoding="utf-8")
    target = tmp_path / "target.marlbg-replay.json"
    target.write_text("{}", encoding="utf-8")
    symlink = tmp_path / "symlink.marlbg-replay.json"
    symlink.symlink_to(target)
    missing = tmp_path / "missing.marlbg-replay.json"

    def fail_if_entered(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("invalid replay must fail before service/server entry")

    monkeypatch.setattr(
        replay_service_module,
        "ReplayViewerService",
        fail_if_entered,
    )
    monkeypatch.setattr(server_module, "serve_browser_debugger", fail_if_entered)

    for replay_path, error_code in (
        (missing, "path_not_found"),
        (invalid, "wrong_root_schema"),
        (symlink, "path_is_symlink"),
    ):
        with pytest.raises(SystemExit) as exc_info:
            main(("--replay", str(replay_path), "--no-open"))
        assert exc_info.value.code == 2
        assert error_code in capsys.readouterr().err


def test_invalid_replay_frame_and_pov_fail_before_server_import(
    tmp_path: Path,
) -> None:
    replay_path = _write_valid_replay(tmp_path)
    code = f"""
import sys
from scripts.dev.debug_renderer import main

for argv, message in (
    (
        ['--replay', {str(replay_path)!r}, '--frame-index', '99', '--no-open'],
        'initial frame index is outside the captured replay',
    ),
    (
        [
            '--replay', {str(replay_path)!r}, '--view', 'pov',
            '--pov-slot', '9', '--no-open',
        ],
        'pov_global_slot must name a configured-active replay actor',
    ),
):
    try:
        main(argv)
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError('invalid replay launch unexpectedly succeeded')
    assert 'scripts.dev.visual_debugger.server' not in sys.modules
print('pre-server rejection complete')
"""
    result = subprocess.run(
        (sys.executable, "-c", code),
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "pre-server rejection complete"


def test_valid_replay_browser_path_is_core_jax_and_live_stack_free(
    tmp_path: Path,
) -> None:
    replay_path = _write_valid_replay(tmp_path)
    code = f"""
import sys
import scripts.dev.visual_debugger.server as server

def fake_serve(service, **_kwargs):
    frame = service.current_frame()
    assert frame.cursor.frame_index == 1
    assert frame.view_mode == 'pov'
    assert frame.pov_global_slot == 5
    return 37

server.serve_browser_debugger = fake_serve
from scripts.dev.debug_renderer import main
status = main([
    '--replay', {str(replay_path)!r}, '--frame-index', '1',
    '--view', 'pov', '--pov-slot', '5', '--no-open',
])
assert status == 37
forbidden = (
    'jax',
    'jaxlib',
    'numpy',
    'marl_battlegrounds.core',
    'scripts.dev.visual_debugger.control',
    'scripts.dev.visual_debugger.evaluation_bridge',
    'scripts.dev.visual_debugger.protocol',
    'scripts.dev.visual_debugger.revision',
    'scripts.dev.visual_debugger.scenarios',
    'scripts.dev.visual_debugger.service',
)
loaded = sorted(
    name
    for name in sys.modules
    if any(name == prefix or name.startswith(prefix + '.') for prefix in forbidden)
)
assert loaded == [], loaded
print('isolated replay browser')
"""
    result = subprocess.run(
        (sys.executable, "-c", code),
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "isolated replay browser"


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
        "--replay",
        "--frame-index",
        "--pov-slot",
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
