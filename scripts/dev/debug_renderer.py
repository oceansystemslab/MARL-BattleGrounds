"""CLI entry point for the manual MARL-BattleGrounds Combat Debugger."""

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
type _ActionSourceKind = Literal["manual", "scripted", "mixed"]

_REPLAY_LAUNCHER = "scripts/dev/run_replay_viewer.sh"
_MOVED_OPTION_LABELS = (
    ("scenario", "--scenario"),
    ("replay", "--replay"),
    ("sample_replay", "--sample-replay"),
    ("frame_index", "--frame-index"),
    ("pov_slot", "--pov-slot"),
    ("list_scenarios", "--list-scenarios"),
    ("list_sample_replays", "--list-sample-replays"),
    ("include_stress", "--include-stress"),
)


@dataclass(frozen=True, slots=True)
class _LaunchOptions:
    """Resolved live-debugger values plus exact CLI presence information."""

    record_replay: Path | None
    seed: int
    controlled_slot: int | None
    static: bool
    no_open: bool
    port: int
    view: _ViewMode
    ranges: bool
    supplied: frozenset[str]


_EPILOG = """\
battlefield controls (while the battlefield has focus):
  Tab / Shift+Tab       cycle active actors without discarding their drafts
  left click            control the clicked authorized actor
  Shift+left click      select the clicked active target
  Escape                clear target to target-none and leave battlefield focus
  1 / 2                 explicitly arm Basic lane 0 / Ultimate lane 1
  W A S D               cardinal movement
  Q E Z C               diagonal movement
  arrow keys            cardinal movement aliases
  X                     select Stay movement
  Space / Enter         submit every staged action as one joint turn
  R                     reset the manual 18x12 arena deterministically
  G                     toggle controlled-actor ranges
  ?                     open browser controls/help

browser controls:
  View                   switch between Oracle and authorized agent POV
  Reconnect              fetch the current authoritative frame
  Exit / Ctrl-C          stop the local Combat Debugger

live selected-target inspector:
  SELECTED TARGET        identity, relation, distance, and public geometry
  PENDING AUTHORIZED DRAFT
                         movement, ability, target, and exact lane legality
  TECHNICAL FRAME        exact authority-safe facts permitted by the active leaf:
                         Episode or Artifact digest prefix; Frame; Simulator step;
                         conditional Incoming transition; replay-only movement scale

The Combat Debugger always opens the manual arena in fixed Analysis
presentation. Replay artifacts and scripted demonstrations now use:
  scripts/dev/run_replay_viewer.sh
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
    """Build the live-only debugger CLI without importing runtime backends."""
    parser = argparse.ArgumentParser(
        description=(
            "Open the manual 18x12 MARL-BattleGrounds Combat Debugger in fixed "
            "Analysis presentation."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--record-replay",
        metavar="PATH",
        type=Path,
        default=argparse.SUPPRESS,
        help=(
            "record one manual arena episode to a canonical replay and metric "
            "sidecar, then review it"
        ),
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
        help="initial active global slot; otherwise use the arena default",
    )
    parser.add_argument(
        "--static",
        action="store_true",
        default=argparse.SUPPRESS,
        help="render a stateless Matplotlib arena snapshot without a browser server",
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
        help="initial browser view authorization (default: oracle)",
    )
    parser.add_argument(
        "--ranges",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="show or hide controlled-actor ranges (default: show)",
    )

    # Compatibility-only inputs remain accepted but are absent from public help.
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

    # Former mixed-product selectors are parsed solely to provide one actionable
    # migration error before any simulator, rendering, or server import occurs.
    parser.add_argument("--scenario", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--replay", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument(
        "--sample-replay", default=argparse.SUPPRESS, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--frame-index", default=argparse.SUPPRESS, help=argparse.SUPPRESS
    )
    parser.add_argument("--pov-slot", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--list-sample-replays",
        action="store_true",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--include-stress",
        action="store_true",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    return parser


def _reject_moved_options(
    parser: argparse.ArgumentParser,
    namespace: argparse.Namespace,
) -> None:
    supplied = vars(namespace)
    for destination, label in _MOVED_OPTION_LABELS:
        if destination in supplied:
            parser.error(f"{label} moved to the Replay Viewer; use {_REPLAY_LAUNCHER}.")


def _resolve_launch_options(namespace: argparse.Namespace) -> _LaunchOptions:
    supplied = frozenset(vars(namespace)) - {"preset", "verbose"}
    return _LaunchOptions(
        record_replay=cast(
            Path | None,
            getattr(namespace, "record_replay", None),
        ),
        seed=cast(int, getattr(namespace, "seed", 0)),
        controlled_slot=cast(
            int | None,
            getattr(namespace, "controlled_slot", None),
        ),
        static=cast(bool, getattr(namespace, "static", False)),
        no_open=cast(bool, getattr(namespace, "no_open", False)),
        port=cast(int, getattr(namespace, "port", 0)),
        view=cast(_ViewMode, getattr(namespace, "view", "researcher")),
        ranges=cast(bool, getattr(namespace, "ranges", True)),
        supplied=supplied,
    )


def _validate_option_matrix(
    parser: argparse.ArgumentParser,
    options: _LaunchOptions,
) -> None:
    if options.record_replay is not None and options.static:
        parser.error("--record-replay is available only in live browser mode.")


def _validate_launch(
    scenario: DebuggerScenario,
    *,
    controlled_global_slot: int | None,
) -> None:
    if scenario.name != "arena_5v5" or scenario.mode != "interactive":
        raise ValueError("the Combat Debugger requires the manual arena_5v5 scenario")
    if controlled_global_slot is None:
        return
    config, _ = scenario.build_scenario()
    active_mask = config.agent_profile.active_mask
    if not (
        0 <= controlled_global_slot < len(active_mask)
        and bool(active_mask[controlled_global_slot])
    ):
        raise ValueError(
            f"controlled slot g{controlled_global_slot} is not active in arena_5v5."
        )


def _recording_action_source_kind(session: object) -> _ActionSourceKind:
    """Read the one path-free action-source contract from a live session."""
    evaluation_context = getattr(session, "evaluation_context", None)
    aggregation_keys = getattr(evaluation_context, "aggregation_keys", ())
    rows = tuple(
        row.value
        for row in aggregation_keys
        if getattr(row, "name", None) == "action_source"
    )
    if len(rows) != 1 or rows[0] not in ("manual", "scripted", "mixed"):
        raise ValueError(
            "recording sessions require one canonical action-source contract."
        )
    return rows[0]


def main(argv: Sequence[str] | None = None) -> int:
    """Launch only the manual live arena or its recording workflow."""
    parser = build_parser()
    namespace = parser.parse_args(argv)
    _reject_moved_options(parser, namespace)
    options = _resolve_launch_options(namespace)
    _validate_option_matrix(parser, options)

    try:
        recording_destination = None
        if options.record_replay is not None:
            from marl_battlegrounds.evaluation.replay_io import (
                ReplaySaveError,
                preflight_replay_bundle_destination_v1,
            )

            try:
                recording_destination = preflight_replay_bundle_destination_v1(
                    options.record_replay
                )
            except ReplaySaveError as exc:
                raise ValueError(
                    f"Replay recording target is unavailable: {exc}"
                ) from exc

        from scripts.dev.visual_debugger.evaluation_bridge import (
            build_debugger_evaluation_launch_specification_v1,
        )
        from scripts.dev.visual_debugger.revision import (
            discover_debugger_code_revision_v1,
        )
        from scripts.dev.visual_debugger.scenarios import get_scenario

        scenario = get_scenario("arena_5v5")
        _validate_launch(
            scenario,
            controlled_global_slot=options.controlled_slot,
        )
        code_revision = discover_debugger_code_revision_v1(_REPOSITORY_ROOT)
        evaluation_launch_specification = (
            build_debugger_evaluation_launch_specification_v1(
                root_seed=options.seed,
                code_revision=code_revision,
                capture_profile=(
                    "evaluation_metric_complete"
                    if recording_destination is not None
                    else "debug"
                ),
            )
        )
        if options.static:
            from scripts.dev.visual_debugger.static_renderer import run_static_renderer

            return run_static_renderer(
                scenario=scenario,
                seed=options.seed,
                evaluation_launch_specification=evaluation_launch_specification,
                controlled_global_slot=options.controlled_slot,
                verbose=False,
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
            verbose_logging=False,
        )
        if recording_destination is None:
            service = DebuggerService(
                session,
                view_mode=options.view,
                preset="analysis",
                include_stress=False,
            )
            return serve_browser_debugger(
                service,
                asset_root=_REPOSITORY_ROOT / "web" / "visual_debugger",
                port=options.port,
                open_browser=not options.no_open,
            )

        from scripts.dev.visual_debugger.recording import (
            DebuggerReplayRecorderV1,
            build_debugger_recording_specification_v1,
        )
        from scripts.dev.visual_debugger.recording_coordinator import (
            RecordingDebuggerCoordinator,
        )
        from scripts.dev.visual_debugger.runtime_provenance import (
            capture_debugger_runtime_provenance_v1,
        )

        try:
            runtime_provenance = capture_debugger_runtime_provenance_v1(code_revision)
        except RuntimeError as exc:
            raise ValueError(
                "Replay recording runtime provenance is unavailable; verify the "
                "selected JAX backend exposes a usable device and precision setting."
            ) from exc

        recorder = DebuggerReplayRecorderV1(
            specification=build_debugger_recording_specification_v1(
                action_source_kind=_recording_action_source_kind(session),
                runtime_provenance=runtime_provenance,
            ),
            destination=recording_destination,
            context=session.evaluation_context,
            initial_frame=session.current_evaluation_frame,
        )
        service = DebuggerService(
            session,
            view_mode=options.view,
            preset="analysis",
            include_stress=False,
            recorder=recorder,
        )
        recording_coordinator = RecordingDebuggerCoordinator(service)
        return serve_browser_debugger(
            asset_root=_REPOSITORY_ROOT / "web" / "visual_debugger",
            port=options.port,
            open_browser=not options.no_open,
            coordinator_router=recording_coordinator.router,
            graceful_close=recording_coordinator.graceful_close,
        )
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: Combat Debugger could not start: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
