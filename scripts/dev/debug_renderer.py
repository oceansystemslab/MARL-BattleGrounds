"""CLI entry point for the deterministic comprehensive visual debugger."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from scripts.dev.visual_debugger import (  # noqa: E402 - local script bootstrap.
    iter_scenario_summaries,
    run_visual_debugger,
)

_EPILOG = """\
controls:
  Tab / Shift+Tab       cycle active controlled actors
  left click            select an active global target
  Shift+left click      control the clicked active actor
  right click / Escape  clear target to target-none
  1 / 2                 explicitly arm Basic lane 0 / Ultimate lane 1
  W A S D               cardinal movement
  Q E Z C               diagonal movement
  Space / Enter         submit manual action or next scripted frame
  N                     submit the next registered reference/script frame
  R                     deterministic scenario reset
  Shift+R               explain why cooldown clearing is unavailable
  G                     toggle selected-unit ranges
  V                     toggle concise/verbose logs
  [ / ]                 previous/next scenario
  close window          exit

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
        help="render the scenario's reset snapshot without callbacks",
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


def main(argv: Sequence[str] | None = None) -> int:
    """Parse, list, or run while preserving standard command exit semantics."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_scenarios:
        for summary in iter_scenario_summaries():
            print(summary)
        return 0
    try:
        return run_visual_debugger(
            scenario_name=args.scenario,
            seed=args.seed,
            controlled_global_slot=args.controlled_slot,
            static=args.static,
            verbose=args.verbose,
            show_ranges=args.ranges,
        )
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
