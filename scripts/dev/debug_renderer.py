"""CLI entry point for the deterministic Visual Debugger and Analyzer."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

if TYPE_CHECKING:
    from scripts.dev.visual_debugger.scenarios import DebuggerScenario

type _ViewMode = Literal["researcher", "pov"]
type _Preset = Literal["presentation", "analysis", "debug"]


@dataclass(frozen=True, slots=True)
class _LaunchOptions:
    """Resolved values plus exact command-line presence information."""

    scenario: str | None
    replay: Path | None
    frame_index: int | None
    pov_slot: int | None
    list_scenarios: bool
    include_stress: bool
    seed: int
    controlled_slot: int | None
    static: bool
    no_open: bool
    port: int
    view: _ViewMode
    preset: _Preset
    verbose: bool
    ranges: bool
    supplied: frozenset[str]


_EPILOG = """\
battlefield controls (while the battlefield has focus):
  Tab / Shift+Tab       cycle active actors without discarding their drafts
  left click            select an active global target
  Shift+left click      control the clicked active actor
  right click / Escape  clear target to target-none
  1 / 2                 explicitly arm Basic lane 0 / Ultimate lane 1
  W A S D               cardinal movement
  Q E Z C               diagonal movement
  arrow keys            cardinal movement aliases
  X                     select Stay movement
  Space / Enter         researcher: submit every staged action as one joint turn
                        agent POV: submit only the controlled actor
  N                     advance the next registered scripted frame
  R                     deterministic scenario reset
  Shift+R               explain why cooldown clearing is unavailable
  G                     toggle controlled-actor ranges
  V                     toggle concise/verbose logs
  [ / ]                 previous/next scenario
  P                     pause/resume presentation-only motion
  ?                     open browser controls/help

browser controls:
  Scenario/View/Preset  switch authoritative session presentation
  0.5x / 1x / 2x / Off change presentation-only motion speed
  Skip                  settle the current explanation immediately
  Reconnect             fetch the current authoritative frame
  Exit / Ctrl-C         stop the local Visual Debugger and Analyzer

read-only replay controls:
  First / Previous      move to an earlier captured frame, settled
  Play / Next           serialize exact next-frame explanations
  Last / frame slider   seek to an absolute captured frame, settled
  Home / End            keyboard first/last while the timeline has focus
  Left / Right / Space  keyboard previous/next/play-pause

live selected-target inspector:
  SELECTED TARGET       identity, relation, distance, and public geometry
  PENDING ACTION        movement, ability, target, and exact lane legality
  TECHNICAL DETAILS     raw actor/target indices and same-epoch mask values

scenarios:
  arena_5v5             interactive geometry/combat laboratory
  basic_support         scripted Basic damage/healing sequence
  ultimate_showcase     scripted five-Ultimate sequence
  aura_crossfire        scripted amplification/mitigation crossfire
  status_stack          scripted status composition and lifecycle sequence
  team_focus_crossfire  scripted focus fire and coordinated healing
  mirrored_ultimates    scripted reciprocal five-class Ultimate sequence
"""


def build_parser() -> argparse.ArgumentParser:
    """Build the complete analyzer CLI without importing Matplotlib."""
    parser = argparse.ArgumentParser(
        description=(
            "Open the deterministic MARL-BattleGrounds Visual Debugger and Analyzer."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--scenario",
        metavar="NAME",
        default=argparse.SUPPRESS,
        help="scenario to open (default: arena_5v5)",
    )
    parser.add_argument(
        "--replay",
        metavar="PATH",
        type=Path,
        default=argparse.SUPPRESS,
        help="validated local replay to view in a browser or render with --static",
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
        "--list-scenarios",
        action="store_true",
        default=argparse.SUPPRESS,
        help="list scenario names, modes, and descriptions, then exit",
    )
    parser.add_argument(
        "--include-stress",
        action="store_true",
        default=argparse.SUPPRESS,
        help="include developer visual-stress scenarios in lookup and menus",
    )
    parser.add_argument(
        "--seed",
        metavar="N",
        type=int,
        default=argparse.SUPPRESS,
        help="deterministic reset/step key seed (default: 0)",
    )
    parser.add_argument(
        "--controlled-slot",
        metavar="N",
        type=int,
        default=argparse.SUPPRESS,
        help="initial active global slot; otherwise use scenario default",
    )
    parser.add_argument(
        "--static",
        action="store_true",
        default=argparse.SUPPRESS,
        help="render a stateless Matplotlib reset snapshot without a browser server",
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
        choices=("researcher", "pov"),
        default=argparse.SUPPRESS,
        help="initial browser view authorization (default: researcher)",
    )
    parser.add_argument(
        "--preset",
        choices=("presentation", "analysis", "debug"),
        default=argparse.SUPPRESS,
        help="initial browser presentation preset (default: analysis)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=argparse.SUPPRESS,
        help="print verbose transition diagnostics",
    )
    parser.add_argument(
        "--ranges",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="show or hide controlled-actor ranges (default: show)",
    )
    return parser


def _resolve_launch_options(namespace: argparse.Namespace) -> _LaunchOptions:
    """Apply defaults only after preserving which options were explicit."""
    supplied = frozenset(vars(namespace))
    return _LaunchOptions(
        scenario=cast(str | None, getattr(namespace, "scenario", None)),
        replay=cast(Path | None, getattr(namespace, "replay", None)),
        frame_index=cast(int | None, getattr(namespace, "frame_index", None)),
        pov_slot=cast(int | None, getattr(namespace, "pov_slot", None)),
        list_scenarios=cast(bool, getattr(namespace, "list_scenarios", False)),
        include_stress=cast(bool, getattr(namespace, "include_stress", False)),
        seed=cast(int, getattr(namespace, "seed", 0)),
        controlled_slot=cast(
            int | None,
            getattr(namespace, "controlled_slot", None),
        ),
        static=cast(bool, getattr(namespace, "static", False)),
        no_open=cast(bool, getattr(namespace, "no_open", False)),
        port=cast(int, getattr(namespace, "port", 0)),
        view=cast(_ViewMode, getattr(namespace, "view", "researcher")),
        preset=cast(_Preset, getattr(namespace, "preset", "analysis")),
        verbose=cast(bool, getattr(namespace, "verbose", False)),
        ranges=cast(bool, getattr(namespace, "ranges", True)),
        supplied=supplied,
    )


_OPTION_LABELS = (
    ("scenario", "--scenario"),
    ("list_scenarios", "--list-scenarios"),
    ("include_stress", "--include-stress"),
    ("seed", "--seed"),
    ("controlled_slot", "--controlled-slot"),
    ("pov_slot", "--pov-slot"),
    ("view", "--view"),
    ("preset", "--preset"),
    ("ranges", "--ranges/--no-ranges"),
    ("port", "--port"),
    ("no_open", "--no-open"),
    ("verbose", "--verbose"),
)


def _validate_option_matrix(
    parser: argparse.ArgumentParser,
    options: _LaunchOptions,
) -> None:
    if options.replay is None:
        if "frame_index" in options.supplied:
            parser.error("--frame-index requires --replay.")
        if "pov_slot" in options.supplied:
            parser.error("--pov-slot requires --replay.")
        return

    if options.static:
        if "frame_index" not in options.supplied:
            parser.error("--frame-index is required with --replay --static.")
        allowed = frozenset(("replay", "static", "frame_index"))
        launch_label = "--replay --static"
    else:
        allowed = frozenset(
            (
                "replay",
                "frame_index",
                "pov_slot",
                "view",
                "preset",
                "ranges",
                "port",
                "no_open",
            )
        )
        launch_label = "--replay"

    forbidden = options.supplied - allowed
    for destination, option_label in _OPTION_LABELS:
        if destination in forbidden:
            parser.error(f"{option_label} is unavailable with {launch_label}.")


def _run_browser_replay(options: _LaunchOptions) -> int:
    """Load and resolve replay authority before importing the HTTP server."""
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
        preset=options.preset,
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
    )
    return serve_browser_debugger(
        service,
        asset_root=_REPOSITORY_ROOT / "web" / "visual_debugger",
        port=options.port,
        open_browser=not options.no_open,
        coordinator=coordinator,
    )


def _validate_launch(
    scenario: DebuggerScenario,
    *,
    controlled_global_slot: int | None,
    include_stress: bool,
) -> None:
    if scenario.audience == "stress" and not include_stress:
        msg = f"stress scenario {scenario.name!r} requires --include-stress."
        raise ValueError(msg)
    if controlled_global_slot is None:
        return
    config, _ = scenario.build_scenario()
    active_mask = config.agent_profile.active_mask
    if not (
        0 <= controlled_global_slot < len(active_mask)
        and bool(active_mask[controlled_global_slot])
    ):
        msg = (
            f"controlled slot g{controlled_global_slot} is not active in "
            f"scenario {scenario.name!r}."
        )
        raise ValueError(msg)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse, list, or run while preserving standard command exit semantics."""
    parser = build_parser()
    options = _resolve_launch_options(parser.parse_args(argv))
    _validate_option_matrix(parser, options)
    if options.list_scenarios:
        from scripts.dev.visual_debugger.scenarios import iter_scenario_summaries

        for summary in iter_scenario_summaries(
            include_stress=options.include_stress,
        ):
            print(summary)
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
                show_ranges=True,
            )
        if options.replay is not None:
            return _run_browser_replay(options)

        from scripts.dev.visual_debugger.evaluation_bridge import (
            build_debugger_evaluation_launch_specification_v1,
        )
        from scripts.dev.visual_debugger.revision import (
            discover_debugger_code_revision_v1,
        )
        from scripts.dev.visual_debugger.scenarios import get_scenario

        scenario = get_scenario(options.scenario or "arena_5v5")
        _validate_launch(
            scenario,
            controlled_global_slot=options.controlled_slot,
            include_stress=options.include_stress,
        )
        evaluation_launch_specification = (
            build_debugger_evaluation_launch_specification_v1(
                root_seed=options.seed,
                code_revision=discover_debugger_code_revision_v1(
                    _REPOSITORY_ROOT,
                ),
            )
        )
        if options.static:
            from scripts.dev.visual_debugger.static_renderer import (
                run_static_renderer,
            )

            return run_static_renderer(
                scenario=scenario,
                seed=options.seed,
                evaluation_launch_specification=evaluation_launch_specification,
                controlled_global_slot=options.controlled_slot,
                verbose=options.verbose,
                show_ranges=options.ranges,
            )

        from scripts.dev.visual_debugger.control import create_session
        from scripts.dev.visual_debugger.server import serve_browser_debugger
        from scripts.dev.visual_debugger.service import DebuggerService

        session = create_session(
            scenario,
            seed=options.seed,
            evaluation_launch_specification=evaluation_launch_specification,
            controlled_global_slot=options.controlled_slot,
            show_ranges=options.ranges,
            verbose_logging=options.verbose,
        )
        service = DebuggerService(
            session,
            view_mode=options.view,
            preset=options.preset,
            include_stress=options.include_stress,
        )
        return serve_browser_debugger(
            service,
            asset_root=_REPOSITORY_ROOT / "web" / "visual_debugger",
            port=options.port,
            open_browser=not options.no_open,
        )
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(
            f"error: Visual Debugger and Analyzer could not start: {exc}",
            file=sys.stderr,
        )
        return 2
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
