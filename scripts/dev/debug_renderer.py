"""CLI entry point for the deterministic comprehensive visual debugger."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from scripts.dev.visual_debugger.scenarios import (  # noqa: E402 - script bootstrap.
    DebuggerScenario,
    get_scenario,
    iter_scenario_summaries,
)

_EPILOG = """\
controls:
  Tab / Shift+Tab       cycle active actors without discarding their drafts
  left click            select an active global target
  Shift+left click      control the clicked active actor
  right click / Escape  clear target to target-none
  1 / 2                 explicitly arm Basic lane 0 / Ultimate lane 1
  W A S D               cardinal movement
  Q E Z C               diagonal movement
  X                     select Stay movement
  Space / Enter         submit every staged active-agent action as one joint turn
  N                     advance the next registered scripted frame
  R                     deterministic scenario reset
  Shift+R               explain why cooldown clearing is unavailable
  G                     toggle selected-unit ranges
  V                     toggle concise/verbose logs
  [ / ]                 previous/next scenario
  ?                     open browser controls/help
  Exit / Ctrl-C         stop the local browser debugger

selected-target inspector:
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
    """Build the complete debugger CLI without importing Matplotlib."""
    parser = argparse.ArgumentParser(
        description=(
            "Open the deterministic MARL-BattleGrounds Milestone 1-5 visual debugger."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scenario",
        metavar="NAME",
        default="arena_5v5",
        help="scenario to open (default: arena_5v5)",
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="list scenario names, modes, and descriptions, then exit",
    )
    parser.add_argument(
        "--include-stress",
        action="store_true",
        help="include developer visual-stress scenarios in lookup and menus",
    )
    parser.add_argument(
        "--seed",
        metavar="N",
        type=int,
        default=0,
        help="deterministic reset/step key seed (default: 0)",
    )
    parser.add_argument(
        "--controlled-slot",
        metavar="N",
        type=int,
        help="initial active global slot; otherwise use scenario default",
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="render a Matplotlib reset snapshot without callbacks or a server",
    )
    parser.add_argument(
        "--ui",
        choices=("matplotlib", "browser"),
        default="matplotlib",
        help="temporary live UI selector (default: matplotlib)",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="print the browser URL without opening it automatically",
    )
    parser.add_argument(
        "--port",
        metavar="N",
        type=int,
        default=0,
        help="loopback browser-server port; 0 chooses an ephemeral port",
    )
    parser.add_argument(
        "--view",
        choices=("researcher", "pov"),
        default="researcher",
        help="initial browser view authorization (default: researcher)",
    )
    parser.add_argument(
        "--preset",
        choices=("presentation", "analysis", "debug"),
        default="analysis",
        help="initial browser presentation preset (default: analysis)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print verbose transition diagnostics",
    )
    parser.add_argument(
        "--ranges",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show or hide selected-unit ranges (default: show)",
    )
    return parser


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
    config = scenario.build_config()
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
    args = parser.parse_args(argv)
    if args.list_scenarios:
        for summary in iter_scenario_summaries(
            include_stress=args.include_stress,
        ):
            print(summary)
        return 0
    try:
        scenario = get_scenario(args.scenario)
        _validate_launch(
            scenario,
            controlled_global_slot=args.controlled_slot,
            include_stress=args.include_stress,
        )
        if args.static or args.ui == "matplotlib":
            from scripts.dev.visual_debugger.app import run_visual_debugger

            return run_visual_debugger(
                scenario_name=scenario.name,
                seed=args.seed,
                controlled_global_slot=args.controlled_slot,
                static=args.static,
                verbose=args.verbose,
                show_ranges=args.ranges,
                include_stress=args.include_stress,
            )

        from scripts.dev.visual_debugger.control import create_session
        from scripts.dev.visual_debugger.server import serve_browser_debugger
        from scripts.dev.visual_debugger.service import DebuggerService

        session = create_session(
            scenario,
            seed=args.seed,
            controlled_global_slot=args.controlled_slot,
            show_ranges=args.ranges,
            verbose_logging=args.verbose,
        )
        service = DebuggerService(
            session,
            view_mode=args.view,
            preset=args.preset,
            include_stress=args.include_stress,
        )
        return serve_browser_debugger(
            service,
            asset_root=_REPOSITORY_ROOT / "web" / "visual_debugger",
            port=args.port,
            open_browser=not args.no_open,
        )
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: visual debugger could not start: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
