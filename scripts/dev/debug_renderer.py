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
    from marl_battlegrounds.evaluation.replay_io import LoadedReplayBundleV1
    from scripts.dev.visual_debugger.scenarios import DebuggerScenario

type _ViewMode = Literal["researcher", "pov"]
type _Preset = Literal["presentation", "analysis"]
type _ActionSourceKind = Literal["manual", "scripted", "mixed"]


@dataclass(frozen=True, slots=True)
class _LaunchOptions:
    """Resolved values plus exact command-line presence information."""

    scenario: str | None
    replay: Path | None
    sample_replay: str | None
    record_replay: Path | None
    frame_index: int | None
    pov_slot: int | None
    list_scenarios: bool
    list_sample_replays: bool
    include_stress: bool
    seed: int
    controlled_slot: int | None
    static: bool
    no_open: bool
    port: int
    view: _ViewMode
    preset: _Preset
    ranges: bool
    supplied: frozenset[str]


_EPILOG = """\
battlefield controls (while the battlefield has focus):
  Tab / Shift+Tab       cycle active actors without discarding their drafts
  left click            control the clicked authorized actor
  Shift+left click      select the clicked active target
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
  G                     toggle controlled-actor ranges
  [ / ]                 previous/next scenario
  P                     pause/resume presentation-only motion
  ?                     open browser controls/help

browser controls:
  Scenario/View/Preset  switch authoritative session presentation
  Motion Off            disable or restore animated explanations
  Skip                  settle the current explanation immediately
  Reconnect             fetch the current authoritative frame
  Exit / Ctrl-C         stop the local Visual Debugger and Analyzer

read-only replay controls:
  First / -10 / -1      seek backward with one clamped absolute request
  Play/Pause / +1 / +10 serialize playback or seek forward exactly once
  Last / frame slider   seek to an absolute captured frame, settled
  Tick current / final  show the exact captured cursor and terminal tick
  Home / End            keyboard first/last while the timeline has focus
  Left / Right / Space  keyboard previous/next/play-pause
  Shift+Left / Right    seek ten captured positions backward/forward, clamped

live selected-target inspector:
  SELECTED TARGET       identity, relation, distance, and public geometry
  PENDING ACTION        movement, ability, target, and exact lane legality
  TECHNICAL FRAME       raw actor/target indices and same-epoch mask values

scenarios:
  arena_5v5             interactive geometry/combat laboratory
  basic_support         scripted Basic damage/healing sequence
  ultimate_showcase     scripted five-Ultimate sequence
  aura_crossfire        scripted amplification/mitigation crossfire
  status_stack          scripted status composition and lifecycle sequence
  team_focus_crossfire  scripted focus fire and coordinated healing
  mirrored_ultimates    scripted reciprocal five-class Ultimate sequence
  death_respawn_cycle   scripted death, respawn, shield, and expiry lifecycle
  recovery_refresh_cycle scripted recovery, rejection, refresh, and expiry lifecycle

checked-in sample replays:
  --list-sample-replays list stable launch names without simulator imports
  --sample-replay NAME open one immutable validated sample artifact
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
        "--sample-replay",
        metavar="NAME",
        default=argparse.SUPPRESS,
        help="open one checked-in sample replay by its stable launch name",
    )
    parser.add_argument(
        "--record-replay",
        metavar="PATH",
        type=Path,
        default=argparse.SUPPRESS,
        help=(
            "record one live browser episode to a canonical replay and metric "
            "sidecar, then review it"
        ),
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
        "--list-sample-replays",
        action="store_true",
        default=argparse.SUPPRESS,
        help="list checked-in sample replay names and descriptions, then exit",
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
        type=_parse_preset,
        metavar="{presentation,analysis}",
        default=argparse.SUPPRESS,
        help="initial browser presentation preset (default: analysis)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--ranges",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="show or hide controlled-actor ranges (default: show)",
    )
    return parser


def _parse_preset(value: str) -> _Preset:
    """Parse canonical presets while accepting the legacy Technical token."""
    if value == "debug":
        return "analysis"
    if value in ("presentation", "analysis"):
        return value
    raise argparse.ArgumentTypeError(
        "invalid choice: expected 'presentation' or 'analysis'"
    )


def _resolve_launch_options(namespace: argparse.Namespace) -> _LaunchOptions:
    """Apply defaults only after preserving which options were explicit."""
    supplied = frozenset(vars(namespace)) - {"verbose"}
    return _LaunchOptions(
        scenario=cast(str | None, getattr(namespace, "scenario", None)),
        replay=cast(Path | None, getattr(namespace, "replay", None)),
        sample_replay=cast(
            str | None,
            getattr(namespace, "sample_replay", None),
        ),
        record_replay=cast(
            Path | None,
            getattr(namespace, "record_replay", None),
        ),
        frame_index=cast(int | None, getattr(namespace, "frame_index", None)),
        pov_slot=cast(int | None, getattr(namespace, "pov_slot", None)),
        list_scenarios=cast(bool, getattr(namespace, "list_scenarios", False)),
        list_sample_replays=cast(
            bool,
            getattr(namespace, "list_sample_replays", False),
        ),
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
        ranges=cast(bool, getattr(namespace, "ranges", True)),
        supplied=supplied,
    )


_OPTION_LABELS = (
    ("scenario", "--scenario"),
    ("sample_replay", "--sample-replay"),
    ("list_scenarios", "--list-scenarios"),
    ("list_sample_replays", "--list-sample-replays"),
    ("include_stress", "--include-stress"),
    ("seed", "--seed"),
    ("controlled_slot", "--controlled-slot"),
    ("pov_slot", "--pov-slot"),
    ("view", "--view"),
    ("preset", "--preset"),
    ("ranges", "--ranges/--no-ranges"),
    ("port", "--port"),
    ("no_open", "--no-open"),
    ("record_replay", "--record-replay"),
)


def _validate_option_matrix(
    parser: argparse.ArgumentParser,
    options: _LaunchOptions,
) -> None:
    if options.list_sample_replays:
        if options.supplied != frozenset(("list_sample_replays",)):
            parser.error(
                "--list-sample-replays cannot be combined with launch options."
            )
        return

    if options.replay is not None and options.sample_replay is not None:
        parser.error("--sample-replay cannot be combined with --replay.")
    if options.sample_replay is not None and options.static:
        parser.error("--static is unavailable with --sample-replay.")

    if options.record_replay is not None:
        if options.replay is not None:
            parser.error("--record-replay cannot be combined with --replay.")
        if options.sample_replay is not None:
            parser.error("--record-replay cannot be combined with --sample-replay.")
        if options.static:
            parser.error("--record-replay is available only in live browser mode.")
        if options.list_scenarios:
            parser.error("--record-replay cannot be combined with --list-scenarios.")
        if "frame_index" in options.supplied:
            parser.error("--frame-index is unavailable with --record-replay.")
        if "pov_slot" in options.supplied:
            parser.error("--pov-slot is unavailable with --record-replay.")

    replay_selected = options.replay is not None or options.sample_replay is not None
    if not replay_selected:
        if "frame_index" in options.supplied:
            parser.error("--frame-index requires --replay.")
        if "pov_slot" in options.supplied:
            parser.error("--pov-slot requires --replay.")
        return

    if options.static:
        if "frame_index" not in options.supplied:
            selection_label = (
                "--sample-replay" if options.sample_replay is not None else "--replay"
            )
            parser.error(f"--frame-index is required with {selection_label} --static.")
        selector = "sample_replay" if options.sample_replay is not None else "replay"
        allowed = frozenset((selector, "static", "frame_index"))
        launch_label = (
            "--sample-replay --static"
            if options.sample_replay is not None
            else "--replay --static"
        )
    else:
        selector = "sample_replay" if options.sample_replay is not None else "replay"
        allowed = frozenset(
            (
                selector,
                "frame_index",
                "pov_slot",
                "view",
                "preset",
                "ranges",
                "port",
                "no_open",
            )
        )
        launch_label = (
            "--sample-replay" if options.sample_replay is not None else "--replay"
        )

    forbidden = options.supplied - allowed
    for destination, option_label in _OPTION_LABELS:
        if destination in forbidden:
            parser.error(f"{option_label} is unavailable with {launch_label}.")


def _run_browser_replay(
    options: _LaunchOptions,
    *,
    loaded_bundle: LoadedReplayBundleV1 | None = None,
) -> int:
    """Load and resolve replay authority before importing the HTTP server."""
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
    """Parse, list, or run while preserving standard command exit semantics."""
    parser = build_parser()
    options = _resolve_launch_options(parser.parse_args(argv))
    _validate_option_matrix(parser, options)
    if options.list_sample_replays:
        from scripts.dev.visual_debugger.sample_replays import iter_sample_replays

        for sample in iter_sample_replays():
            print(sample.summary())
        return 0
    if options.list_scenarios:
        from scripts.dev.visual_debugger.scenario_catalog import (
            iter_scenario_summaries,
        )

        for summary in iter_scenario_summaries(
            include_stress=options.include_stress,
        ):
            print(summary)
        return 0
    sample_replay_bundle = None
    if options.sample_replay is not None:
        from scripts.dev.visual_debugger.sample_replays import (
            SAMPLE_REPLAY_DEMO_PROVENANCE_NOTICE,
            load_verified_sample_replay,
        )

        try:
            sample_replay_bundle = load_verified_sample_replay(options.sample_replay)
        except ValueError as exc:
            parser.error(str(exc))
        print(f"Sample replay notice: {SAMPLE_REPLAY_DEMO_PROVENANCE_NOTICE}")
    try:
        if sample_replay_bundle is not None:
            return _run_browser_replay(
                options,
                loaded_bundle=sample_replay_bundle,
            )
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

        recording_destination = None
        if options.record_replay is not None:
            from marl_battlegrounds.evaluation.replay_io import (
                ReplaySaveError,
                preflight_replay_bundle_destination_v1,
            )

            try:
                recording_destination = preflight_replay_bundle_destination_v1(
                    options.record_replay,
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

        scenario = get_scenario(options.scenario or "arena_5v5")
        _validate_launch(
            scenario,
            controlled_global_slot=options.controlled_slot,
            include_stress=options.include_stress,
        )
        code_revision = discover_debugger_code_revision_v1(
            _REPOSITORY_ROOT,
        )
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
            from scripts.dev.visual_debugger.static_renderer import (
                run_static_renderer,
            )

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
                preset=options.preset,
                include_stress=options.include_stress,
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
            runtime_provenance = capture_debugger_runtime_provenance_v1(
                code_revision,
            )
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
            preset=options.preset,
            include_stress=options.include_stress,
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
        print(
            f"error: Visual Debugger and Analyzer could not start: {exc}",
            file=sys.stderr,
        )
        return 2
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
