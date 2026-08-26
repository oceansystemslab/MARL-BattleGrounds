"""CLI and shell-launcher regression tests, including dependency isolation."""

import argparse
import os
import stat
import subprocess
import sys
from collections.abc import Callable
from importlib.util import find_spec
from pathlib import Path
from typing import cast

import pytest
from scripts.dev.debug_renderer import (
    build_parser as build_debugger_parser,
)
from scripts.dev.debug_renderer import (
    main as debug_main,
)
from scripts.dev.replay_viewer import build_parser, main
from scripts.dev.visual_debugger.evaluation_bridge import (
    DebuggerEvaluationLaunchSpecificationV1,
)
from scripts.dev.visual_debugger.model import DebuggerScenario
from scripts.dev.visual_debugger.sample_replays import (
    SAMPLE_REPLAY_DEMO_PROVENANCE_NOTICE,
    SAMPLE_REPLAYS,
)
from scripts.dev.visual_debugger.scenarios import (
    RESEARCHER_SCENARIOS,
    STRESS_SCENARIOS,
)
from scripts.dev.visual_debugger.service import DebuggerService

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_DEBUGGER_PYTHON_ENTRYPOINT = _REPOSITORY_ROOT / "scripts" / "dev" / "debug_renderer.py"
_DEBUGGER_SHELL_LAUNCHER = (
    _REPOSITORY_ROOT / "scripts" / "dev" / "run_debug_renderer.sh"
)
_REPLAY_PYTHON_ENTRYPOINT = _REPOSITORY_ROOT / "scripts" / "dev" / "replay_viewer.py"
_REPLAY_SHELL_LAUNCHER = _REPOSITORY_ROOT / "scripts" / "dev" / "run_replay_viewer.sh"
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


def test_debugger_parser_exposes_live_arena_contract_and_hidden_compatibility() -> None:
    parser = build_debugger_parser()
    args = parser.parse_args(
        (
            "--seed",
            "41",
            "--controlled-slot",
            "5",
            "--static",
            "--no-open",
            "--port",
            "8123",
            "--view",
            "researcher",
            "--preset",
            "debug",
            "--verbose",
            "--no-ranges",
        )
    )

    assert args.seed == 41
    assert args.controlled_slot == 5
    assert args.static
    assert args.no_open
    assert args.port == 8123
    assert args.view == "researcher"
    assert args.preset == "analysis"
    assert args.verbose is False
    assert args.ranges is False


def test_replay_parser_exposes_narrow_static_replay_contract() -> None:
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


def test_replay_parser_exposes_complete_browser_replay_contract() -> None:
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
    assert args.preset == "analysis"
    assert args.ranges is False
    assert args.port == 0
    assert args.no_open
    assert not hasattr(args, "static")
    assert not hasattr(args, "scenario")


def test_debugger_parser_exposes_opt_in_live_recording_target() -> None:
    parser = build_debugger_parser()
    args = parser.parse_args(
        (
            "--record-replay",
            "episode.marlbg-replay.json",
            "--seed",
            "7",
            "--no-open",
        )
    )

    assert args.record_replay == Path("episode.marlbg-replay.json")
    assert args.seed == 7
    assert args.no_open
    assert not hasattr(args, "replay")


def test_recording_rejects_static_mode(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        debug_main(("--record-replay", "episode.marlbg-replay.json", "--static"))

    assert exc_info.value.code == 2
    assert "--record-replay" in capsys.readouterr().err


@pytest.mark.parametrize(
    "moved",
    (
        ("--replay", "other.marlbg-replay.json"),
        ("--scenario", "status_stack"),
        ("--list-scenarios",),
        ("--list-sample-replays",),
        ("--frame-index", "0"),
        ("--pov-slot", "0"),
    ),
)
def test_debugger_rejects_moved_replay_options_with_migration_message(
    moved: tuple[str, ...],
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        debug_main(moved)

    assert exc_info.value.code == 2
    error = capsys.readouterr().err
    assert moved[0] in error
    assert "scripts/dev/run_replay_viewer.sh" in error


def test_debugger_rejects_moved_authority_before_runtime_imports() -> None:
    code = """
import sys
from scripts.dev.debug_renderer import main

for argv in (
    ('--scenario', 'status_stack'),
    ('--replay', 'episode.marlbg-replay.json'),
):
    try:
        main(argv)
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError('moved option unexpectedly launched')

forbidden = (
    'jax',
    'jaxlib',
    'numpy',
    'marl_battlegrounds.core',
    'scripts.dev.visual_debugger',
)
loaded = sorted(
    name
    for name in sys.modules
    if any(name == prefix or name.startswith(prefix + '.') for prefix in forbidden)
)
assert loaded == [], loaded
print('moved options rejected before runtime imports')
"""
    result = subprocess.run(
        (sys.executable, "-c", code),
        cwd=_REPOSITORY_ROOT,
        env={**os.environ, "JAX_PLATFORMS": "cuda"},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "moved options rejected before runtime imports"
    assert result.stderr.count("scripts/dev/run_replay_viewer.sh") == 2


@pytest.mark.parametrize("parser_factory", (build_debugger_parser, build_parser))
@pytest.mark.parametrize(
    ("token", "expected"),
    (("oracle", "researcher"), ("pov", "pov"), ("researcher", "researcher")),
)
def test_view_tokens_canonicalize_without_advertising_legacy_authority(
    parser_factory: Callable[[], argparse.ArgumentParser],
    token: str,
    expected: str,
) -> None:
    assert parser_factory().parse_args(("--view", token)).view == expected


@pytest.mark.parametrize("parser_factory", (build_debugger_parser, build_parser))
@pytest.mark.parametrize("token", ("presentation", "analysis", "debug", "technical"))
def test_legacy_preset_and_verbose_inputs_are_fixed_compatibility_no_ops(
    parser_factory: Callable[[], argparse.ArgumentParser],
    token: str,
) -> None:
    args = parser_factory().parse_args(("--preset", token, "--verbose"))
    assert args.preset == "analysis"
    assert args.verbose is False


@pytest.mark.parametrize(
    "argv, message",
    (
        (("--frame-index", "1"), "choose exactly one replay artifact"),
        (("--pov-slot", "0"), "choose exactly one replay artifact"),
        (
            ("--replay", "x.marlbg-replay.json", "--static"),
            "--frame-index is required with --static",
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
        (("--include-stress",), "--include-stress"),
        (("--seed", "0"), "--seed"),
        (("--pov-slot", "0"), "--pov-slot"),
        (("--view", "researcher"), "--view"),
        (("--port", "0"), "--port"),
        (("--no-open",), "--no-open"),
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
        (("--include-stress",), "--include-stress"),
        (("--seed", "0"), "--seed"),
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
from scripts.dev.replay_viewer import build_parser
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
        env={**os.environ, "JAX_PLATFORMS": "cuda"},
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
            "--no-ranges",
        )
    )

    assert result == 29
    assert observed == {
        "replay_path": Path("episode.marlbg-replay.json"),
        "frame_index": 7,
        "show_ranges": False,
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

        def current_presentation(self) -> object:
            raise AssertionError(
                "server should not request a presentation in this test"
            )

        def current_metric_report(self) -> object:
            raise AssertionError(
                "server should not request a metric report in this test"
            )

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
        "preset": "analysis",
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
    assert coordinator.current_presentation == service.current_presentation
    assert coordinator.current_metric_report == service.current_metric_report
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
from scripts.dev.replay_viewer import main

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
        env={**os.environ, "JAX_PLATFORMS": "cuda"},
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
from scripts.dev.replay_viewer import main
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
        env={**os.environ, "JAX_PLATFORMS": "cuda"},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "isolated replay browser"


def test_checked_sample_replay_browser_path_is_core_jax_and_live_stack_free() -> None:
    sample_name = SAMPLE_REPLAYS[0].name
    code = f"""
import sys
import scripts.dev.visual_debugger.server as server

def fake_serve(service, **_kwargs):
    frame = service.current_frame()
    assert frame.cursor.frame_index == 0
    assert frame.view_mode == 'researcher'
    return 41

server.serve_browser_debugger = fake_serve
from scripts.dev.replay_viewer import main
status = main([
    '--sample-replay', {sample_name!r}, '--no-open',
])
assert status == 41
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
print('isolated checked sample browser')
"""
    result = subprocess.run(
        (sys.executable, "-c", code),
        cwd=_REPOSITORY_ROOT,
        env={**os.environ, "JAX_PLATFORMS": "cuda"},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert SAMPLE_REPLAY_DEMO_PROVENANCE_NOTICE in result.stdout
    assert result.stdout.rstrip().endswith("isolated checked sample browser")


@pytest.mark.parametrize("parser_factory", (build_debugger_parser, build_parser))
def test_parsers_reject_abbreviated_options(
    parser_factory: Callable[[], argparse.ArgumentParser],
) -> None:
    parser = parser_factory()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(("--stat",))
    assert exc_info.value.code == 2


def test_debugger_help_is_live_only_and_hides_legacy_tokens() -> None:
    result = subprocess.run(
        (sys.executable, str(_DEBUGGER_PYTHON_ENTRYPOINT), "--help"),
        cwd="/tmp",
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    for option in (
        "--record-replay",
        "--seed",
        "--controlled-slot",
        "--static",
        "--no-open",
        "--port",
        "--view",
        "--ranges",
        "--no-ranges",
    ):
        assert option in result.stdout
    assert "--ui" not in result.stdout
    for control in (
        "Tab / Shift+Tab",
        "left click",
        "Shift+left click",
        "Escape",
        "arrow keys",
        "X",
        "Space / Enter",
        "Reconnect",
    ):
        assert control in result.stdout
    assert "right click" not in result.stdout
    assert "every staged action as one joint turn" in result.stdout
    assert "submit only the controlled actor" not in result.stdout
    assert "left click            control the clicked authorized actor" in result.stdout
    assert "Shift+left click      select the clicked active target" in result.stdout
    for inspector in (
        "SELECTED TARGET",
        "PENDING AUTHORIZED DRAFT",
        "TECHNICAL FRAME",
    ):
        assert inspector in result.stdout
    assert "exact authority-safe facts permitted by the active leaf" in result.stdout
    for technical_fact in (
        "Episode or Artifact digest prefix",
        "Frame",
        "Simulator step",
        "conditional Incoming transition",
        "replay-only movement scale",
    ):
        assert technical_fact in result.stdout
    assert "raw actor/target indices" not in result.stdout
    assert "same-epoch mask values" not in result.stdout
    for moved_option in (
        "--scenario",
        "--replay",
        "--sample-replay",
        "--frame-index",
        "--pov-slot",
        "--list-scenarios",
        "--list-sample-replays",
        "--include-stress",
    ):
        assert moved_option not in result.stdout
    assert "geometry_debug_renderer" not in result.stdout
    assert "--preset" not in result.stdout
    assert "--verbose" not in result.stdout
    assert "researcher" not in result.stdout
    assert "{oracle,pov}" in result.stdout
    assert "scripts/dev/run_replay_viewer.sh" in result.stdout


def test_replay_help_is_read_only_and_hides_live_and_legacy_tokens() -> None:
    result = subprocess.run(
        (sys.executable, str(_REPLAY_PYTHON_ENTRYPOINT), "--help"),
        cwd="/tmp",
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    for option in (
        "--replay",
        "--sample-replay",
        "--scenario",
        "--list-scenarios",
        "--list-sample-replays",
        "--include-stress",
        "--frame-index",
        "--pov-slot",
        "--view",
    ):
        assert option in result.stdout
    assert "--record-replay" not in result.stdout
    assert "--controlled-slot" not in result.stdout
    assert "--preset" not in result.stdout
    assert "--verbose" not in result.stdout
    assert "researcher" not in result.stdout
    assert "{oracle,pov}" in result.stdout
    for replay_control in (
        "Start / End",
        "-10 / -1 / +1 / +10",
        "Play / Pause",
        "run serialized replay with one request in flight",
        "frame slider",
        "preview without a request; commit one exact seek",
        "Left / Right / Space",
        "unmodified document shortcuts: previous / next /",
        "play or pause",
        "Export PNG",
        "Download Metrics",
        "canonical metric-report download in every visual POV",
        "Tick current / final",
    ):
        assert replay_control in result.stdout
    multiplication_sign = "\N{MULTIPLICATION SIGN}"
    for playback_rate in (
        "0.25",
        "0.50",
        "0.75",
        "1.00",
        "1.25",
        "1.50",
        "1.75",
        "2.00",
    ):
        assert f"{playback_rate}{multiplication_sign}" in result.stdout
    assert "First / -10 / -1" not in result.stdout
    assert "Last / frame slider" not in result.stdout


def test_list_scenarios_is_stable_and_backend_free() -> None:
    code = (
        "import sys; "
        "from scripts.dev.replay_viewer import main; "
        "status=main(['--list-scenarios']); "
        "print('STATUS', status); "
        "forbidden=('jax','jaxlib','numpy','marl_battlegrounds.core','matplotlib'); "
        "loaded=sorted(n for n in sys.modules if any(n == p or n.startswith(p + '.') "
        "for p in forbidden)); "
        "print('FORBIDDEN', loaded)"
    )
    result = subprocess.run(
        (sys.executable, "-c", code),
        cwd=_REPOSITORY_ROOT,
        env={**os.environ, "JAX_PLATFORMS": "cuda"},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "STATUS 0" in result.stdout
    assert "FORBIDDEN []" in result.stdout
    scripted_names = tuple(
        name
        for name, scenario in RESEARCHER_SCENARIOS.items()
        if scenario.mode == "scripted"
    )
    positions = [result.stdout.index(name) for name in scripted_names]
    assert positions == sorted(positions)
    for scenario_name in STRESS_SCENARIOS:
        assert scenario_name not in result.stdout
    assert "arena_5v5" not in result.stdout
    assert "scripted" in result.stdout


def test_list_scenarios_includes_stress_only_when_explicitly_requested() -> None:
    code = (
        "import sys; "
        "from scripts.dev.replay_viewer import main; "
        "status=main(['--list-scenarios', '--include-stress']); "
        "forbidden=('jax','jaxlib','numpy','marl_battlegrounds.core','matplotlib'); "
        "loaded=sorted(n for n in sys.modules if any(n == p or n.startswith(p + '.') "
        "for p in forbidden)); "
        "print('STATUS', status); "
        "print('FORBIDDEN', loaded)"
    )
    result = subprocess.run(
        (sys.executable, "-c", code),
        cwd=_REPOSITORY_ROOT,
        env={**os.environ, "JAX_PLATFORMS": "cuda"},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "STATUS 0" in result.stdout
    assert "FORBIDDEN []" in result.stdout
    scripted_names = tuple(
        name
        for name, scenario in RESEARCHER_SCENARIOS.items()
        if scenario.mode == "scripted"
    )
    for scenario_name in (*scripted_names, *STRESS_SCENARIOS):
        assert scenario_name in result.stdout
    assert "arena_5v5" not in result.stdout


def test_list_sample_replays_is_stable_and_core_free() -> None:
    code = (
        "import sys; "
        "from scripts.dev.replay_viewer import main; "
        "status=main(['--list-sample-replays']); "
        "print('STATUS', status); "
        "print('FORBIDDEN', any(any(n == p or n.startswith(p + '.') "
        "for p in ('jax', 'jaxlib', 'numpy', 'marl_battlegrounds.core')) "
        "for n in sys.modules))"
    )
    result = subprocess.run(
        (sys.executable, "-c", code),
        cwd=_REPOSITORY_ROOT,
        env={**os.environ, "JAX_PLATFORMS": "cuda"},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "STATUS 0" in result.stdout
    assert "FORBIDDEN False" in result.stdout
    positions = [result.stdout.index(sample.name) for sample in SAMPLE_REPLAYS]
    assert positions == sorted(positions)
    for sample in SAMPLE_REPLAYS:
        assert sample.display_name in result.stdout


def test_sample_replay_resolves_integrity_before_standard_replay_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import scripts.dev.replay_viewer as launcher_module
    from scripts.dev.visual_debugger import sample_replays as samples_module

    events: list[object] = []
    verified_bundle = object()

    def fake_load(name: str) -> object:
        events.extend(("verify", name))
        return verified_bundle

    def fake_replay(options: object, *, loaded_bundle: object | None = None) -> int:
        events.extend(
            (
                "dispatch",
                object.__getattribute__(options, "replay"),
                loaded_bundle,
            )
        )
        return 43

    monkeypatch.setattr(samples_module, "load_verified_sample_replay", fake_load)
    monkeypatch.setattr(launcher_module, "_run_browser_replay", fake_replay)

    result = main(("--sample-replay", SAMPLE_REPLAYS[0].name, "--no-open"))

    assert result == 43
    assert events == [
        "verify",
        SAMPLE_REPLAYS[0].name,
        "dispatch",
        None,
        verified_bundle,
    ]
    assert SAMPLE_REPLAY_DEMO_PROVENANCE_NOTICE in capsys.readouterr().out


def test_sample_replay_injects_verified_bundle_without_reopening_source_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.dev.visual_debugger import replay_service as replay_service_module
    from scripts.dev.visual_debugger import sample_replays as samples_module
    from scripts.dev.visual_debugger import server as server_module
    from scripts.dev.visual_debugger.server import HttpCoordinatorBinding

    from marl_battlegrounds.evaluation import replay_io as replay_io_module

    verified_bundle = object()
    observed: dict[str, object] = {}

    class FakeReplayService:
        def current_frame(self) -> object:
            raise AssertionError("server should not request a frame in this test")

        def current_timeline(self) -> object:
            raise AssertionError("server should not request a timeline in this test")

        def current_presentation(self) -> object:
            raise AssertionError(
                "server should not request a presentation in this test"
            )

        def current_metric_report(self) -> object:
            raise AssertionError(
                "server should not request a metric report in this test"
            )

        def apply_command(self, request: object) -> object:
            del request
            raise AssertionError("server should not apply a command in this test")

    service = FakeReplayService()

    def fake_sample_load(name: str) -> object:
        observed["sample_name"] = name
        return verified_bundle

    def fail_path_reopen(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("sample dispatch must not reopen a source path")

    def fake_service_factory(loaded: object, **kwargs: object) -> FakeReplayService:
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
        observed.update(
            service=resolved_service,
            asset_root=asset_root,
            port=port,
            open_browser=open_browser,
            coordinator=coordinator,
        )
        return 47

    monkeypatch.setattr(samples_module, "load_verified_sample_replay", fake_sample_load)
    monkeypatch.setattr(replay_io_module, "load_replay_bundle_v1", fail_path_reopen)
    monkeypatch.setattr(
        replay_service_module,
        "ReplayViewerService",
        fake_service_factory,
    )
    monkeypatch.setattr(server_module, "serve_browser_debugger", fake_serve)

    result = main(("--sample-replay", SAMPLE_REPLAYS[0].name, "--no-open"))

    assert result == 47
    assert observed["sample_name"] == SAMPLE_REPLAYS[0].name
    assert observed["bundle"] is verified_bundle
    assert observed["service"] is service
    assert observed["open_browser"] is False


def test_sample_replay_static_uses_the_verified_in_memory_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.dev.replay_viewer as launcher_module
    from scripts.dev.visual_debugger import sample_replays as samples_module

    verified_bundle = object()
    observed: dict[str, object] = {}

    def fake_sample_load(name: str) -> object:
        del name
        return verified_bundle

    monkeypatch.setattr(
        samples_module,
        "load_verified_sample_replay",
        fake_sample_load,
    )

    def fake_static(options: object, bundle: object) -> int:
        observed["frame_index"] = object.__getattribute__(options, "frame_index")
        observed["ranges"] = object.__getattribute__(options, "ranges")
        observed["bundle"] = bundle
        return 59

    def fail_browser(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("sample --static must not start the browser server")

    monkeypatch.setattr(launcher_module, "_run_static_loaded_replay", fake_static)
    monkeypatch.setattr(launcher_module, "_run_browser_replay", fail_browser)

    result = main(
        (
            "--sample-replay",
            SAMPLE_REPLAYS[0].name,
            "--static",
            "--frame-index",
            "2",
            "--no-ranges",
        )
    )

    assert result == 59
    assert observed == {
        "frame_index": 2,
        "ranges": False,
        "bundle": verified_bundle,
    }


@pytest.mark.parametrize(
    "conflict",
    (
        ("--replay", "other.marlbg-replay.json"),
        ("--scenario", "arena_5v5"),
        ("--seed", "0"),
        ("--record-replay", "capture.marlbg-replay.json"),
    ),
)
def test_sample_replay_rejects_competing_artifact_or_live_authority(
    conflict: tuple[str, ...],
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(("--sample-replay", SAMPLE_REPLAYS[0].name, *conflict))

    assert exc_info.value.code == 2
    assert capsys.readouterr().err


def test_list_sample_replays_rejects_every_launch_option(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(("--list-sample-replays", "--no-open"))

    assert exc_info.value.code == 2
    error = capsys.readouterr().err
    assert "--no-open is unavailable with --list-sample-replays" in error


def test_lazy_rendering_and_debugger_models_are_backend_free() -> None:
    code = (
        "import sys; "
        "import marl_battlegrounds.rendering; "
        "import scripts.dev.visual_debugger.model; "
        "import scripts.dev.visual_debugger.targeting; "
        "forbidden=('jax','jaxlib','numpy','marl_battlegrounds.core','matplotlib'); "
        "loaded=sorted(n for n in sys.modules if any(n == p or n.startswith(p + '.') "
        "for p in forbidden)); "
        "print(loaded)"
    )
    result = subprocess.run(
        (sys.executable, "-c", code),
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


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
            str(_DEBUGGER_PYTHON_ENTRYPOINT),
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
        ("--controlled-slot", "-1"),
        ("--seed", "not-an-int"),
        ("--view", "researcher-visible-typo"),
    ),
)
def test_invalid_cli_inputs_use_argparse_exit_two(argv: tuple[str, ...]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        debug_main(argv)
    assert exc_info.value.code == 2


@pytest.mark.parametrize(
    "argv",
    (("--controlled-slot", "-1"),),
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
        debug_main(argv)

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
    result = debug_main(
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
    assert frame.preset == "analysis"
    assert not hasattr(frame.projection.scene, "ranges")


def test_recording_preflights_before_scenario_provenance_session_or_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts.dev.visual_debugger import revision as revision_module
    from scripts.dev.visual_debugger import runtime_provenance as runtime_module
    from scripts.dev.visual_debugger import scenarios as scenarios_module
    from scripts.dev.visual_debugger import server as server_module
    from scripts.dev.visual_debugger import service as service_module

    from marl_battlegrounds.evaluation import replay_io as replay_io_module
    from marl_battlegrounds.evaluation.replay_io import ReplaySaveError

    target = tmp_path / "missing" / "episode.marlbg-replay.json"

    def fail_preflight(path: Path) -> object:
        raise ReplaySaveError(
            "missing_parent",
            path=path.parent,
            detail="injected missing parent",
        )

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("recording preflight must precede runtime construction")

    monkeypatch.setattr(
        replay_io_module,
        "preflight_replay_bundle_destination_v1",
        fail_preflight,
    )
    monkeypatch.setattr(scenarios_module, "get_scenario", forbidden)
    monkeypatch.setattr(
        revision_module,
        "discover_debugger_code_revision_v1",
        forbidden,
    )
    monkeypatch.setattr(
        runtime_module,
        "capture_debugger_runtime_provenance_v1",
        forbidden,
    )
    monkeypatch.setattr(service_module, "DebuggerService", forbidden)
    monkeypatch.setattr(server_module, "serve_browser_debugger", forbidden)

    with pytest.raises(SystemExit) as exc_info:
        debug_main(("--record-replay", str(target), "--no-open"))

    assert exc_info.value.code == 2
    error = capsys.readouterr().err
    assert "Replay recording target is unavailable" in error
    assert "missing_parent" in error


def test_recording_runtime_provenance_failure_exits_before_router_or_browser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts.dev.visual_debugger import recording as recording_module
    from scripts.dev.visual_debugger import (
        recording_coordinator as coordinator_module,
    )
    from scripts.dev.visual_debugger import runtime_provenance as runtime_module
    from scripts.dev.visual_debugger import server as server_module

    def fail_provenance(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("private device discovery detail")

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("runtime provenance must precede binding and browser open")

    monkeypatch.setattr(
        runtime_module,
        "capture_debugger_runtime_provenance_v1",
        fail_provenance,
    )
    monkeypatch.setattr(recording_module, "DebuggerReplayRecorderV1", forbidden)
    monkeypatch.setattr(coordinator_module, "RecordingDebuggerCoordinator", forbidden)
    monkeypatch.setattr(server_module, "serve_browser_debugger", forbidden)
    monkeypatch.setattr(server_module.webbrowser, "open", forbidden)

    with pytest.raises(SystemExit) as exc_info:
        debug_main(
            (
                "--record-replay",
                str(tmp_path / "episode.marlbg-replay.json"),
            )
        )

    assert exc_info.value.code == 2
    error = capsys.readouterr().err
    assert "Replay recording runtime provenance is unavailable" in error
    assert "private device discovery detail" not in error


def test_recording_launch_injects_retaining_recorder_router_and_graceful_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.dev.visual_debugger import runtime_provenance as runtime_module
    from scripts.dev.visual_debugger import server as server_module
    from scripts.dev.visual_debugger.recording_coordinator import (
        RecordingDebuggerCoordinator,
    )
    from scripts.dev.visual_debugger.server import HttpCoordinatorRouter

    from marl_battlegrounds.evaluation.models import CodeRevisionV1
    from marl_battlegrounds.evaluation.replay import RuntimeProvenanceV1

    target = tmp_path / "recorded.marlbg-replay.json"
    observed: dict[str, object] = {}
    capture_count = 0

    def fake_runtime(
        code_revision: CodeRevisionV1,
    ) -> RuntimeProvenanceV1:
        nonlocal capture_count
        capture_count += 1
        return RuntimeProvenanceV1(
            python_version="3.14.0",
            package_version=code_revision.package_version,
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
        )

    def fake_serve(
        service: object | None = None,
        *,
        asset_root: Path,
        port: int,
        open_browser: bool,
        coordinator_router: HttpCoordinatorRouter,
        graceful_close: object,
    ) -> int:
        observed.update(
            service=service,
            asset_root=asset_root,
            port=port,
            open_browser=open_browser,
            router=coordinator_router,
            graceful_close=graceful_close,
        )
        return 43

    monkeypatch.setattr(
        runtime_module,
        "capture_debugger_runtime_provenance_v1",
        fake_runtime,
    )
    monkeypatch.setattr(server_module, "serve_browser_debugger", fake_serve)

    result = debug_main(
        (
            "--record-replay",
            str(target),
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

    assert result == 43
    assert capture_count == 1
    assert observed["service"] is None
    assert observed["asset_root"] == _REPOSITORY_ROOT / "web" / "visual_debugger"
    assert observed["port"] == 0
    assert observed["open_browser"] is False
    router = cast(HttpCoordinatorRouter, observed["router"])
    snapshot = router.snapshot()
    service = cast(DebuggerService, snapshot.service)
    assert snapshot.binding.mode == "live"
    assert service.session.evaluation_context.capture_profile == (
        "evaluation_metric_complete"
    )
    assert service.current_frame().recording == service.recording_status
    assert service.current_frame().view_mode == "pov"
    assert service.current_frame().preset == "analysis"
    assert getattr(observed["graceful_close"], "__self__", None).__class__ is (
        RecordingDebuggerCoordinator
    )


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

    result = debug_main(
        (
            "--static",
            "--seed",
            "13",
            "--controlled-slot",
            "5",
            "--verbose",
            "--no-ranges",
        )
    )

    assert result == 19
    assert cast(DebuggerScenario, observed["scenario"]).name == "arena_5v5"
    assert observed["seed"] == 13
    assert (
        cast(
            DebuggerEvaluationLaunchSpecificationV1,
            observed["evaluation_launch_specification"],
        ).root_seed
        == 13
    )
    assert observed["controlled_global_slot"] == 5
    assert observed["verbose"] is False
    assert observed["show_ranges"] is False


def test_stress_browser_launch_requires_and_accepts_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.dev.replay_viewer as launcher_module

    bundle = object()

    def fake_materialize(_options: object) -> object:
        return bundle

    def fake_replay(
        _options: object,
        *,
        loaded_bundle: object | None = None,
    ) -> int:
        return 0 if loaded_bundle is bundle else 1

    monkeypatch.setattr(
        launcher_module,
        "_materialize_scripted_bundle",
        fake_materialize,
    )
    monkeypatch.setattr(
        launcher_module,
        "_run_browser_replay",
        fake_replay,
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


def test_scripted_scenario_materializes_in_an_isolated_cpu_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.dev.replay_viewer as launcher_module

    from marl_battlegrounds.evaluation import replay_io as replay_io_module

    observed: dict[str, object] = {}
    bundle = object()

    def fake_subprocess_run(
        command: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        observed["command"] = command
        observed.update(kwargs)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    def fake_load(
        path: Path,
        *,
        require_metric_report: bool = False,
    ) -> object:
        observed["loaded_path"] = path
        observed["require_metric_report"] = require_metric_report
        return bundle

    def fake_browser(
        options: object,
        *,
        loaded_bundle: object | None = None,
    ) -> int:
        observed["browser_options"] = options
        observed["loaded_bundle"] = loaded_bundle
        return 53

    monkeypatch.setattr(launcher_module.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(replay_io_module, "load_replay_bundle_v1", fake_load)
    monkeypatch.setattr(launcher_module, "_run_browser_replay", fake_browser)

    assert main(("--scenario", "basic_support", "--seed", "17", "--no-open")) == 53
    command = cast(list[str], observed["command"])
    assert command[:4] == [
        sys.executable,
        str(_REPLAY_PYTHON_ENTRYPOINT.resolve()),
        "--_materialize-scripted-scenario",
        "basic_support",
    ]
    assert command[4:6] == ["--destination", str(observed["loaded_path"])]
    assert command[6:] == ["--seed", "17"]
    assert observed["cwd"] == _REPOSITORY_ROOT
    environment = cast(dict[str, str], observed["env"])
    assert environment["JAX_PLATFORMS"] == "cpu"
    assert observed["check"] is False
    assert observed["capture_output"] is True
    assert observed["text"] is True
    assert observed["require_metric_report"] is True
    assert observed["loaded_bundle"] is bundle


def test_real_scripted_materializer_publishes_a_publicly_loadable_bundle(
    tmp_path: Path,
) -> None:
    from marl_battlegrounds.evaluation.replay_io import load_replay_bundle_v1

    destination = tmp_path / "basic-support.marlbg-replay.json"
    result = subprocess.run(
        (
            sys.executable,
            str(_REPLAY_PYTHON_ENTRYPOINT.resolve()),
            "--_materialize-scripted-scenario",
            "basic_support",
            "--destination",
            str(destination),
            "--seed",
            "23",
        ),
        cwd=_REPOSITORY_ROOT,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    bundle = load_replay_bundle_v1(destination, require_metric_report=True)
    assert bundle.status == "complete"
    assert bundle.metric_report_artifact is not None
    assert len(bundle.replay.transitions) == 2
    assert len(bundle.replay.frames) == 3


def test_scripted_scenario_static_renders_the_materialized_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.dev.replay_viewer as launcher_module

    bundle = object()
    observed: dict[str, object] = {}

    def fake_materialize(options: object) -> object:
        del options
        return bundle

    monkeypatch.setattr(
        launcher_module,
        "_materialize_scripted_bundle",
        fake_materialize,
    )

    def fake_static(options: object, loaded_bundle: object) -> int:
        observed["frame_index"] = object.__getattribute__(options, "frame_index")
        observed["seed"] = object.__getattribute__(options, "seed")
        observed["ranges"] = object.__getattribute__(options, "ranges")
        observed["bundle"] = loaded_bundle
        return 61

    def fail_browser(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("scenario --static must not start the browser server")

    monkeypatch.setattr(launcher_module, "_run_static_loaded_replay", fake_static)
    monkeypatch.setattr(launcher_module, "_run_browser_replay", fail_browser)

    result = main(
        (
            "--scenario",
            "basic_support",
            "--seed",
            "17",
            "--static",
            "--frame-index",
            "3",
            "--no-ranges",
        )
    )

    assert result == 61
    assert observed == {
        "frame_index": 3,
        "seed": 17,
        "ranges": False,
        "bundle": bundle,
    }


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


def test_debugger_launcher_resolves_root_and_forwards_live_arguments(
    tmp_path: Path,
) -> None:
    fake_bin, record = _write_fake_uv(tmp_path, exit_code=0)
    environment = os.environ.copy()
    environment["PATH"] = os.pathsep.join((str(fake_bin), "/usr/bin", "/bin"))
    environment["UV_RECORD"] = str(record)
    result = subprocess.run(
        (
            str(_DEBUGGER_SHELL_LAUNCHER),
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
        str(_DEBUGGER_PYTHON_ENTRYPOINT),
        "--seed",
        "19",
        "--verbose",
        "--no-ranges",
    ]


def test_debugger_launcher_activates_viz_extra_only_for_static_snapshots(
    tmp_path: Path,
) -> None:
    fake_bin, record = _write_fake_uv(tmp_path, exit_code=0)
    environment = os.environ.copy()
    environment["PATH"] = os.pathsep.join((str(fake_bin), "/usr/bin", "/bin"))
    environment["UV_RECORD"] = str(record)

    result = subprocess.run(
        (str(_DEBUGGER_SHELL_LAUNCHER), "--static"),
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
        str(_DEBUGGER_PYTHON_ENTRYPOINT),
        "--static",
    ]


def test_replay_launcher_resolves_root_and_forwards_replay_arguments(
    tmp_path: Path,
) -> None:
    fake_bin, record = _write_fake_uv(tmp_path, exit_code=0)
    environment = os.environ.copy()
    environment["PATH"] = os.pathsep.join((str(fake_bin), "/usr/bin", "/bin"))
    environment["UV_RECORD"] = str(record)

    result = subprocess.run(
        (
            str(_REPLAY_SHELL_LAUNCHER),
            "--sample-replay",
            SAMPLE_REPLAYS[0].name,
            "--no-open",
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
        str(_REPLAY_PYTHON_ENTRYPOINT),
        "--sample-replay",
        SAMPLE_REPLAYS[0].name,
        "--no-open",
    ]


def test_replay_launcher_activates_viz_extra_for_static_artifacts(
    tmp_path: Path,
) -> None:
    fake_bin, record = _write_fake_uv(tmp_path, exit_code=0)
    environment = os.environ.copy()
    environment["PATH"] = os.pathsep.join((str(fake_bin), "/usr/bin", "/bin"))
    environment["UV_RECORD"] = str(record)

    result = subprocess.run(
        (
            str(_REPLAY_SHELL_LAUNCHER),
            "--replay",
            "episode.marlbg-replay.json",
            "--static",
            "--frame-index",
            "0",
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
        str(_REPLAY_PYTHON_ENTRYPOINT),
        "--replay",
        "episode.marlbg-replay.json",
        "--static",
        "--frame-index",
        "0",
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
            str(_DEBUGGER_PYTHON_ENTRYPOINT),
            "--static",
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


@pytest.mark.parametrize(
    "launcher",
    (_DEBUGGER_SHELL_LAUNCHER, _REPLAY_SHELL_LAUNCHER),
)
def test_launchers_propagate_uv_exit_code(tmp_path: Path, launcher: Path) -> None:
    fake_bin, record = _write_fake_uv(tmp_path, exit_code=23)
    environment = os.environ.copy()
    environment["PATH"] = os.pathsep.join((str(fake_bin), "/usr/bin", "/bin"))
    environment["UV_RECORD"] = str(record)
    result = subprocess.run(
        (str(launcher), "--help"),
        cwd=tmp_path,
        env=environment,
        check=False,
    )

    assert result.returncode == 23


@pytest.mark.parametrize(
    ("launcher", "product"),
    (
        (_DEBUGGER_SHELL_LAUNCHER, "Combat Debugger"),
        (_REPLAY_SHELL_LAUNCHER, "Replay Viewer"),
    ),
)
def test_launchers_report_missing_uv(launcher: Path, product: str) -> None:
    environment = os.environ.copy()
    environment["PATH"] = "/usr/bin:/bin"
    result = subprocess.run(
        ("bash", str(launcher), "--help"),
        cwd="/tmp",
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 127
    assert f"error: uv is required to run the {product}." in result.stderr


def test_launchers_are_executable() -> None:
    assert _DEBUGGER_SHELL_LAUNCHER.stat().st_mode & stat.S_IXUSR
    assert _REPLAY_SHELL_LAUNCHER.stat().st_mode & stat.S_IXUSR


def test_launcher_split_has_both_products_and_no_old_compatibility_wrapper() -> None:
    assert _DEBUGGER_PYTHON_ENTRYPOINT.is_file()
    assert _DEBUGGER_SHELL_LAUNCHER.is_file()
    assert _REPLAY_PYTHON_ENTRYPOINT.is_file()
    assert _REPLAY_SHELL_LAUNCHER.is_file()
    assert not _OLD_PYTHON_ENTRYPOINT.exists()
    assert not _OLD_SHELL_LAUNCHER.exists()
