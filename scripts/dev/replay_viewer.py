"""CLI entry point for immutable replays and scripted demonstrations."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

if TYPE_CHECKING:
    from marl_battlegrounds.evaluation.replay_io import LoadedReplayBundleV1

type _ViewMode = Literal["researcher", "pov"]

_DEBUGGER_LAUNCHER = "scripts/dev/run_debug_renderer.sh"
_PRIVATE_MATERIALIZE_OPTION = "--_materialize-scripted-scenario"


@dataclass(frozen=True, slots=True)
class _LaunchOptions:
    """Resolved replay-viewer values plus exact CLI presence information."""

    replay: Path | None
    sample_replay: str | None
    scenario: str | None
    list_scenarios: bool
    list_sample_replays: bool
    include_stress: bool
    seed: int
    frame_index: int | None
    pov_slot: int | None
    static: bool
    no_open: bool
    port: int
    view: _ViewMode
    ranges: bool
    supplied: frozenset[str]


_EPILOG = """\
artifact selection (choose exactly one):
  --replay PATH          open one validated replay artifact
  --sample-replay NAME  open one checked-in verified sample
  --scenario NAME       materialize one scripted demonstration in isolation
  --list-scenarios      list scripted demonstration launch names
  --list-sample-replays list checked-in sample launch names

read-only replay controls:
  First / -10 / -1      seek backward with one clamped absolute request
  Play/Pause / +1 / +10 serialize playback or seek forward exactly once
  Last / frame slider   seek to an absolute captured frame
  Tick current / final  show the exact captured cursor and terminal tick

The Replay Viewer is read-only and always uses fixed Analysis presentation.
The manual 18x12 combat laboratory remains available through:
  scripts/dev/run_debug_renderer.sh
"""


def _parse_view(value: str) -> _ViewMode:
    """Map the public Oracle name and hidden legacy token to wire authority."""
    if value in ("oracle", "researcher"):
        return "researcher"
    if value == "pov":
        return "pov"
    raise argparse.ArgumentTypeError("invalid choice: expected 'oracle' or 'pov'")


def _parse_compatibility_preset(value: str) -> Literal["analysis"]:
    """Accept historical preset spellings while fixing the product to Analysis."""
    if value in ("analysis", "presentation", "debug", "technical"):
        return "analysis"
    raise argparse.ArgumentTypeError("invalid legacy preset")


def build_parser() -> argparse.ArgumentParser:
    """Build the replay-only CLI without importing simulator or array backends."""
    parser = argparse.ArgumentParser(
        description=(
            "Open immutable MARL-BattleGrounds artifacts and scripted "
            "demonstrations in the read-only Replay Viewer."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--replay",
        metavar="PATH",
        type=Path,
        default=argparse.SUPPRESS,
        help="validated local replay to view or render with --static",
    )
    parser.add_argument(
        "--sample-replay",
        metavar="NAME",
        default=argparse.SUPPRESS,
        help="open one checked-in sample replay by its stable launch name",
    )
    parser.add_argument(
        "--scenario",
        metavar="NAME",
        default=argparse.SUPPRESS,
        help="materialize and open one registered scripted demonstration",
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        default=argparse.SUPPRESS,
        help="list scripted demonstrations, then exit",
    )
    parser.add_argument(
        "--list-sample-replays",
        action="store_true",
        default=argparse.SUPPRESS,
        help="list checked-in sample replay names and descriptions, then exit",
    )
    parser.add_argument(
        "--include-stress",
        action="store_true",
        default=argparse.SUPPRESS,
        help="include developer visual-stress demonstrations",
    )
    parser.add_argument(
        "--seed",
        metavar="N",
        type=int,
        default=argparse.SUPPRESS,
        help="deterministic scripted-demonstration seed (default: 0)",
    )
    parser.add_argument(
        "--frame-index",
        metavar="N",
        type=int,
        default=argparse.SUPPRESS,
        help="initial replay frame (browser default: 0; required with --static)",
    )
    parser.add_argument(
        "--pov-slot",
        metavar="N",
        type=int,
        default=argparse.SUPPRESS,
        help="initial configured-active actor for replay POV authorization",
    )
    parser.add_argument(
        "--static",
        action="store_true",
        default=argparse.SUPPRESS,
        help="render one replay frame through the stateless Matplotlib adapter",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        default=argparse.SUPPRESS,
        help="print the browser URL without opening it automatically",
    )
    parser.add_argument(
        "--port",
        metavar="N",
        type=int,
        default=argparse.SUPPRESS,
        help="loopback browser-server port; 0 chooses an ephemeral port",
    )
    parser.add_argument(
        "--view",
        type=_parse_view,
        metavar="{oracle,pov}",
        default=argparse.SUPPRESS,
        help="initial replay view authorization (default: oracle)",
    )
    parser.add_argument(
        "--ranges",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="show or hide authorized reference ranges (default: show)",
    )

    # Compatibility-only inputs remain accepted but absent from public help.
    parser.add_argument(
        "--preset",
        type=_parse_compatibility_preset,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--verbose",
        action="store_const",
        const=False,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )

    # Diagnose inverse product-boundary mistakes without importing live modules.
    parser.add_argument(
        "--record-replay", default=argparse.SUPPRESS, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--controlled-slot", default=argparse.SUPPRESS, help=argparse.SUPPRESS
    )
    return parser


def _resolve_launch_options(namespace: argparse.Namespace) -> _LaunchOptions:
    supplied = frozenset(vars(namespace)) - {"preset", "verbose"}
    return _LaunchOptions(
        replay=cast(Path | None, getattr(namespace, "replay", None)),
        sample_replay=cast(str | None, getattr(namespace, "sample_replay", None)),
        scenario=cast(str | None, getattr(namespace, "scenario", None)),
        list_scenarios=cast(bool, getattr(namespace, "list_scenarios", False)),
        list_sample_replays=cast(
            bool,
            getattr(namespace, "list_sample_replays", False),
        ),
        include_stress=cast(bool, getattr(namespace, "include_stress", False)),
        seed=cast(int, getattr(namespace, "seed", 0)),
        frame_index=cast(int | None, getattr(namespace, "frame_index", None)),
        pov_slot=cast(int | None, getattr(namespace, "pov_slot", None)),
        static=cast(bool, getattr(namespace, "static", False)),
        no_open=cast(bool, getattr(namespace, "no_open", False)),
        port=cast(int, getattr(namespace, "port", 0)),
        view=cast(_ViewMode, getattr(namespace, "view", "researcher")),
        ranges=cast(bool, getattr(namespace, "ranges", True)),
        supplied=supplied,
    )


_OPTION_LABELS = (
    ("replay", "--replay"),
    ("sample_replay", "--sample-replay"),
    ("scenario", "--scenario"),
    ("list_scenarios", "--list-scenarios"),
    ("list_sample_replays", "--list-sample-replays"),
    ("include_stress", "--include-stress"),
    ("seed", "--seed"),
    ("frame_index", "--frame-index"),
    ("pov_slot", "--pov-slot"),
    ("static", "--static"),
    ("no_open", "--no-open"),
    ("port", "--port"),
    ("view", "--view"),
    ("ranges", "--ranges/--no-ranges"),
)


def _validate_option_matrix(
    parser: argparse.ArgumentParser,
    options: _LaunchOptions,
) -> None:
    if "record_replay" in options.supplied or "controlled_slot" in options.supplied:
        parser.error(
            "manual control and recording use the Combat Debugger; "
            f"run {_DEBUGGER_LAUNCHER}."
        )

    selectors = {
        "replay": options.replay is not None,
        "sample_replay": options.sample_replay is not None,
        "scenario": options.scenario is not None,
        "list_scenarios": options.list_scenarios,
        "list_sample_replays": options.list_sample_replays,
    }
    selected = tuple(name for name, enabled in selectors.items() if enabled)
    if len(selected) != 1:
        parser.error("choose exactly one replay artifact, scenario, or list operation.")
    selector = selected[0]

    if selector == "list_sample_replays":
        allowed = frozenset(("list_sample_replays",))
    elif selector == "list_scenarios":
        allowed = frozenset(("list_scenarios", "include_stress"))
    elif options.static:
        if "frame_index" not in options.supplied:
            parser.error("--frame-index is required with --static.")
        allowed_values = {selector, "static", "frame_index", "ranges"}
        if selector == "scenario":
            allowed_values.update(("seed", "include_stress"))
        allowed = frozenset(allowed_values)
    else:
        allowed_values = {
            selector,
            "frame_index",
            "pov_slot",
            "view",
            "ranges",
            "port",
            "no_open",
        }
        if selector == "scenario":
            allowed_values.update(("seed", "include_stress"))
        allowed = frozenset(allowed_values)

    forbidden = options.supplied - allowed
    for destination, option_label in _OPTION_LABELS:
        if destination in forbidden:
            selector_context = f"--{selector.replace('_', '-')}"
            if options.static:
                selector_context += " --static"
            parser.error(f"{option_label} is unavailable with {selector_context}.")


def _run_browser_replay(
    options: _LaunchOptions,
    *,
    loaded_bundle: LoadedReplayBundleV1 | None = None,
) -> int:
    """Resolve immutable replay authority before importing the HTTP server."""
    bundle = loaded_bundle
    if bundle is None:
        from marl_battlegrounds.evaluation.replay_io import (
            ReplayLoadError,
            load_replay_bundle_v1,
        )

        assert options.replay is not None
        try:
            bundle = load_replay_bundle_v1(options.replay)
        except ReplayLoadError as exc:
            raise ValueError(f"Replay could not be loaded: {exc}") from exc

    from scripts.dev.visual_debugger.replay_service import ReplayViewerService

    service = ReplayViewerService(
        bundle,
        initial_frame_index=(0 if options.frame_index is None else options.frame_index),
        view_mode=options.view,
        pov_global_slot=options.pov_slot,
        preset="analysis",
        show_ranges=options.ranges,
        verbose=False,
    )

    from scripts.dev.visual_debugger.replay_protocol import (
        ReplayApiErrorV1,
        ReplayCommandRequestV1,
    )
    from scripts.dev.visual_debugger.server import (
        REPLAY_HTTP_ROUTES,
        HttpCoordinatorBinding,
        serve_browser_debugger,
    )

    coordinator = HttpCoordinatorBinding(
        mode="replay",
        routes=REPLAY_HTTP_ROUTES,
        request_model=ReplayCommandRequestV1,
        error_factory=ReplayApiErrorV1,
        current_frame=service.current_frame,
        apply_command=service.apply_command,
        current_timeline=service.current_timeline,
        current_presentation=service.current_presentation,
    )
    return serve_browser_debugger(
        service,
        asset_root=_REPOSITORY_ROOT / "web" / "visual_debugger",
        port=options.port,
        open_browser=not options.no_open,
        coordinator=coordinator,
    )


def _run_static_loaded_replay(
    options: _LaunchOptions,
    bundle: LoadedReplayBundleV1,
) -> int:
    """Render one frame from an already-authorized immutable bundle."""
    from scripts.dev.visual_debugger.static_renderer import (
        run_static_replay_artifact_renderer,
    )

    assert options.frame_index is not None
    return run_static_replay_artifact_renderer(
        replay=bundle.replay,
        frame_index=options.frame_index,
        show_ranges=options.ranges,
    )


def _validate_scripted_scenario(
    parser: argparse.ArgumentParser,
    options: _LaunchOptions,
) -> None:
    from scripts.dev.visual_debugger.scenario_catalog import SCENARIO_CATALOG_BY_NAME

    assert options.scenario is not None
    entry = SCENARIO_CATALOG_BY_NAME.get(options.scenario)
    if entry is None:
        parser.error(f"unknown replay scenario {options.scenario!r}.")
    if entry.mode != "scripted":
        parser.error(
            f"scenario {entry.name!r} is the manual combat lab; "
            f"run {_DEBUGGER_LAUNCHER}."
        )
    if entry.audience == "stress" and not options.include_stress:
        parser.error(f"stress scenario {entry.name!r} requires --include-stress.")


def _materialize_scripted_bundle(
    options: _LaunchOptions,
) -> LoadedReplayBundleV1:
    """Run simulator-backed materialization in a child, then load public bytes."""
    assert options.scenario is not None
    with tempfile.TemporaryDirectory(prefix="marlbg-replay-scenario-") as directory:
        destination = Path(directory) / f"{options.scenario}.marlbg-replay.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            _PRIVATE_MATERIALIZE_OPTION,
            options.scenario,
            "--destination",
            str(destination),
            "--seed",
            str(options.seed),
        ]
        if options.include_stress:
            command.append("--include-stress")
        environment = os.environ.copy()
        environment["JAX_PLATFORMS"] = "cpu"
        result = subprocess.run(
            command,
            cwd=_REPOSITORY_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            detail = result.stderr.strip().splitlines()
            message = detail[-1] if detail else "scripted materialization failed"
            raise ValueError(message)

        from marl_battlegrounds.evaluation.replay_io import (
            ReplayLoadError,
            load_replay_bundle_v1,
        )

        try:
            return load_replay_bundle_v1(destination, require_metric_report=True)
        except ReplayLoadError as exc:
            raise ValueError(
                f"Materialized replay could not be publicly loaded: {exc}"
            ) from exc


def _materializer_main(argv: Sequence[str]) -> int:
    """Private child process that owns all simulator-backed construction."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(_PRIVATE_MATERIALIZE_OPTION, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--include-stress", action="store_true")
    options = parser.parse_args(argv)

    try:
        from marl_battlegrounds.evaluation.replay_io import (
            load_replay_bundle_v1,
            preflight_replay_bundle_destination_v1,
        )
        from scripts.dev.visual_debugger.control import create_session
        from scripts.dev.visual_debugger.evaluation_bridge import (
            build_debugger_evaluation_launch_specification_v1,
        )
        from scripts.dev.visual_debugger.protocol import (
            CommandRequestV1,
            CommandResponseV2,
            KeyboardCommandV1,
        )
        from scripts.dev.visual_debugger.recording import (
            DebuggerReplayRecorderV1,
            build_debugger_recording_specification_v1,
        )
        from scripts.dev.visual_debugger.recording_coordinator import (
            RecordingDebuggerCoordinator,
        )
        from scripts.dev.visual_debugger.revision import (
            discover_debugger_code_revision_v1,
        )
        from scripts.dev.visual_debugger.runtime_provenance import (
            capture_debugger_runtime_provenance_v1,
        )
        from scripts.dev.visual_debugger.scenarios import get_scenario
        from scripts.dev.visual_debugger.service import DebuggerService

        scenario = get_scenario(options._materialize_scripted_scenario)
        if scenario.mode != "scripted":
            raise ValueError("only scripted scenarios can become replay artifacts")
        if scenario.audience == "stress" and not options.include_stress:
            raise ValueError(
                f"stress scenario {scenario.name!r} requires --include-stress"
            )
        revision = discover_debugger_code_revision_v1(_REPOSITORY_ROOT)
        launch = build_debugger_evaluation_launch_specification_v1(
            root_seed=options.seed,
            code_revision=revision,
            capture_profile="evaluation_metric_complete",
        )
        session = create_session(
            scenario,
            seed=options.seed,
            evaluation_launch_specification=launch,
            controlled_global_slot=None,
            show_ranges=True,
            verbose_logging=False,
        )
        recorder = DebuggerReplayRecorderV1(
            specification=build_debugger_recording_specification_v1(
                action_source_kind="scripted",
                runtime_provenance=capture_debugger_runtime_provenance_v1(revision),
            ),
            destination=preflight_replay_bundle_destination_v1(options.destination),
            context=session.evaluation_context,
            initial_frame=session.current_evaluation_frame,
        )
        coordinator = RecordingDebuggerCoordinator(
            DebuggerService(
                session,
                view_mode="researcher",
                preset="analysis",
                include_stress=options.include_stress,
                session_id=f"scripted-replay-materializer-{scenario.name}",
                recorder=recorder,
            )
        )
        for frame_index in range(len(scenario.frames)):
            result = coordinator.apply_command(
                CommandRequestV1(
                    client_id="scripted-replay-materializer",
                    command_id=f"{scenario.name}-transition-{frame_index}",
                    base_revision=coordinator.service.revision,
                    command=KeyboardCommandV1(key="n"),
                )
            )
            if not isinstance(result.payload, CommandResponseV2):
                raise RuntimeError("scripted transition returned no live frame")
            if result.payload.result != "applied":
                raise RuntimeError(f"scripted transition {frame_index} was not applied")
        if recorder.lifecycle != "saved":
            raise RuntimeError("scripted scenario did not publish a complete replay")
        load_replay_bundle_v1(options.destination, require_metric_report=True)
        return 0
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    """List or launch one immutable replay authority."""
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == _PRIVATE_MATERIALIZE_OPTION:
        return _materializer_main(arguments)

    parser = build_parser()
    options = _resolve_launch_options(parser.parse_args(arguments))
    _validate_option_matrix(parser, options)

    if options.list_scenarios:
        from scripts.dev.visual_debugger.scenario_catalog import (
            iter_scenario_catalog,
        )

        for entry in iter_scenario_catalog(include_stress=options.include_stress):
            if entry.mode == "scripted":
                print(entry.summary())
        return 0
    if options.list_sample_replays:
        from scripts.dev.visual_debugger.sample_replays import iter_sample_replays

        for sample in iter_sample_replays():
            print(sample.summary())
        return 0

    try:
        if options.replay is not None and options.static:
            from scripts.dev.visual_debugger.static_renderer import (
                run_static_replay_renderer,
            )

            assert options.frame_index is not None
            return run_static_replay_renderer(
                replay_path=options.replay,
                frame_index=options.frame_index,
                show_ranges=options.ranges,
            )

        if options.replay is not None:
            return _run_browser_replay(options)

        if options.sample_replay is not None:
            from scripts.dev.visual_debugger.sample_replays import (
                SAMPLE_REPLAY_DEMO_PROVENANCE_NOTICE,
                load_verified_sample_replay,
            )

            try:
                bundle = load_verified_sample_replay(options.sample_replay)
            except ValueError as exc:
                parser.error(str(exc))
            print(f"Sample replay notice: {SAMPLE_REPLAY_DEMO_PROVENANCE_NOTICE}")
            if options.static:
                return _run_static_loaded_replay(options, bundle)
            return _run_browser_replay(options, loaded_bundle=bundle)

        _validate_scripted_scenario(parser, options)
        bundle = _materialize_scripted_bundle(options)
        if options.static:
            return _run_static_loaded_replay(options, bundle)
        return _run_browser_replay(options, loaded_bundle=bundle)
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: Replay Viewer could not start: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
