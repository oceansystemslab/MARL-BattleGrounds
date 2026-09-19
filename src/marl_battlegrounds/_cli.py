"""Parse package commands and call the existing Python authorities once.

Run ``python -m marl_battlegrounds --help`` for commands. Parsing and help use
only the standard library. Scientific validation, execution and result meanings
belong to the called APIs. Absent scientific options stay absent during resume.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from marl_battlegrounds.evaluation.policy_execution import Policy, System
    from marl_battlegrounds.evaluation.results import (
        CanonicalTournamentResult,
        EvaluationResult,
        TournamentResult,
    )


class _FactoryError(ValueError):
    """Report a trusted factory failure while retaining its original cause."""


def _tokens(text: str, *, empty: bool = False) -> tuple[str, ...]:
    """Split ordered comma tokens; allow empty text only for capture assertions."""
    if empty and not text.strip():
        return ()
    values = tuple(value.strip() for value in text.split(","))
    if any(not value for value in values):
        raise argparse.ArgumentTypeError("Use comma-separated values without gaps")
    return values


def _integer(text: str) -> int:
    """Parse a signed decimal integer without Python expressions or ranges."""
    digits = text.lstrip("+-")
    if not digits or not digits.isascii() or not digits.isdecimal():
        raise argparse.ArgumentTypeError("Use a decimal integer")
    try:
        return int(text, 10)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use a decimal integer") from error


def _ids(text: str) -> tuple[int, ...]:
    """Parse a nonempty ordered map-ID list; the API checks its allowed values."""
    return tuple(_integer(value) for value in _tokens(text))


def _capture_ids(text: str) -> tuple[int, ...]:
    """Parse episode IDs, keeping explicit empty text as an empty assertion."""
    return tuple(_integer(value) for value in _tokens(text, empty=True))


def _execution_options(parser: argparse.ArgumentParser) -> None:
    """Add common recording and execution flags without asserting absent science."""
    for flag, choices, kind, meaning in (
        ("metrics", ("priority", "full", "none"), str, "priority"),
        ("save-replays", None, _integer, "0"),
        ("full-metrics-episodes", None, _capture_ids, "empty"),
        ("replay-episodes", None, _capture_ids, "empty"),
    ):
        parser.add_argument(
            f"--{flag}",
            type=kind,
            choices=choices,
            default=argparse.SUPPRESS,
            help=f"New-run default: {meaning}; omission inherits saved settings",
        )
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--output-dir", help="Save a new run under this directory")
    output.add_argument("--resume-from", help="Resume this existing run")
    parser.add_argument(
        "--num-envs",
        type=_integer,
        default=128,
        help="Parallel environments (default: 128)",
    )
    parser.add_argument(
        "--chunk-size",
        type=_integer,
        default=16,
        help="Steps per compiled chunk (default: 16)",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build all help and syntax rules using only standard-library imports.

    No configurations, assets or researcher modules are read. Scientific limits
    remain with the Python APIs. Subcommands and long flags cannot be abbreviated.
    """
    parser = argparse.ArgumentParser(
        prog="python -m marl_battlegrounds",
        allow_abbrev=False,
        description="Run experiments, prepare assets and view recorded replays.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser(
        "train", allow_abbrev=False, help="Run or resume a declared MAPPO experiment"
    )
    training.add_argument("--config", help="Versioned JSON training settings")
    training_output = training.add_mutually_exclusive_group(required=True)
    training_output.add_argument(
        "--output-dir", help="Exact new or empty run directory"
    )
    training_output.add_argument(
        "--resume-from", help="Complete learner checkpoint path"
    )
    analysis = commands.add_parser(
        "analyze-training",
        allow_abbrev=False,
        help="Plot saved training and validation results",
    )
    analysis.add_argument("run_dirs", nargs="+", help="Saved training run directories")
    analysis.add_argument("--output-dir", required=True, help="Report output directory")
    evaluate = commands.add_parser(
        "evaluate", allow_abbrev=False, help="Evaluate two frozen methods"
    )
    evaluate.add_argument(
        "--system", required=True, help="Team A name or module:factory"
    )
    evaluate.add_argument(
        "--opponent", required=True, help="Team B name or module:factory"
    )
    evaluate.add_argument(
        "--episodes",
        dest="num_episodes",
        type=_integer,
        required=True,
        help="Total games across all maps and spawn choices",
    )
    evaluate.add_argument(
        "--maps",
        type=_ids,
        default=argparse.SUPPRESS,
        help="Ordered comma-separated map IDs; omission uses API defaults",
    )
    for flag in ("system-roster", "opponent-roster"):
        evaluate.add_argument(
            f"--{flag}",
            type=_tokens,
            default=argparse.SUPPRESS,
            help="Ordered comma-separated classes, including repeats",
        )
    evaluate.add_argument(
        "--spawn-mode",
        choices=("paired", "default", "swapped"),
        default=argparse.SUPPRESS,
        help="New-run default: paired; omission inherits saved settings",
    )
    for flag, default in (("seed", 0), ("score-threshold", 20), ("max-steps", 300)):
        evaluate.add_argument(
            f"--{flag}",
            type=_integer,
            default=argparse.SUPPRESS,
            help=f"New-run default: {default}; omission inherits saved settings",
        )
    evaluate.add_argument(
        "--phase", default="evaluation", help="Pass phase (default: evaluation)"
    )
    evaluate.add_argument(
        "--pass-id", default="1", help="Pass ID within phase (default: 1)"
    )
    _execution_options(evaluate)

    canonical = commands.add_parser(
        "canonical",
        allow_abbrev=False,
        help="Compare the released twelve, optionally with a challenger",
    )
    canonical.add_argument(
        "--system",
        default=argparse.SUPPRESS,
        help="Challenger name or module:factory; omission inherits on resume",
    )
    canonical.add_argument(
        "--config",
        default=argparse.SUPPRESS,
        help="Unchanged released JSON snapshot; omission resolves through the API",
    )
    canonical.add_argument(
        "--games-per-opponent",
        type=_integer,
        default=argparse.SUPPRESS,
        help="Uniform game budget; omission inherits snapshot or saved settings",
    )
    canonical.add_argument(
        "--rerun-existing",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Explicitly rerun the whole field; omission inherits on resume",
    )
    _execution_options(canonical)
    tournament = commands.add_parser(
        "tournament", allow_abbrev=False, help="Run a custom configuration's population"
    )
    tournament.add_argument(
        "--config",
        default=argparse.SUPPRESS,
        help="JSON configuration; required for a new run",
    )
    _execution_options(tournament)

    models = commands.add_parser(
        "models",
        allow_abbrev=False,
        help="Explicit asset preparation and cache cleanup",
    )
    operations = models.add_subparsers(dest="operation", required=True)
    download = operations.add_parser(
        "download",
        allow_abbrev=False,
        help="Inspect and confirm missing asset downloads",
    )
    download.add_argument(
        "--config", help="Snapshot/config JSON path; default: installed release"
    )
    download.add_argument(
        "--roles",
        type=_tokens,
        default=("model",),
        help=(
            "Comma-separated model,outcomes_priority,full_report,replay "
            "(default: model)"
        ),
    )
    download.add_argument(
        "--output-config", help="Save verified paths for the next offline command"
    )
    clean = operations.add_parser(
        "clean",
        allow_abbrev=False,
        help="Remove only selected cache digests; stop other cache writers first",
    )
    clean.add_argument(
        "--sha256",
        action="append",
        required=True,
        help="Complete cache digest; repeat for several entries",
    )
    for target in (download, clean):
        target.add_argument(
            "--cache-dir", help="Content cache directory; default: platform user cache"
        )
        target.add_argument(
            "--dry-run", action="store_true", help="Inspect only; do not change files"
        )
        target.add_argument(
            "--yes",
            action="store_true",
            help="Confirm the displayed work without prompting",
        )

    replay = commands.add_parser(
        "replay", allow_abbrev=False, help="Open an existing replay"
    )
    replay.add_argument("path", help="Saved replay path")
    for flag, help_text in (
        (
            "frame-index",
            "Recorded frame index (browser default: 0; required for static)",
        ),
        ("pov-slot", "Global actor slot 0-9 for permitted actor POV"),
        ("port", "Loopback HTTP port (default: 0 selects an available port)"),
    ):
        replay.add_argument(
            f"--{flag}", type=_integer, default=argparse.SUPPRESS, help=help_text
        )
    replay.add_argument(
        "--view",
        choices=("oracle", "pov"),
        default=argparse.SUPPRESS,
        help="Initial browser view (default: oracle); replay permissions apply",
    )
    replay.add_argument(
        "--ranges",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Show ranges (default: off)",
    )
    for flag, help_text in (
        (
            "static",
            "Show one Matplotlib frame; requires --frame-index and the viz extra",
        ),
        ("no-open", "Serve without opening a browser (default: open the browser)"),
    ):
        replay.add_argument(
            f"--{flag}", action="store_true", default=argparse.SUPPRESS, help=help_text
        )
    return parser


def _load_method(reference: str) -> System | Policy | str:
    """Return a bare name or call one trusted factory without choosing an action."""
    if ":" not in reference:
        return reference
    from marl_battlegrounds._method_loading import installed_callable
    from marl_battlegrounds.evaluation.policy_execution import Policy, System

    try:
        value = installed_callable(reference)()
    except Exception as error:
        raise _FactoryError(f"Factory {reference!r} failed: {error}") from error
    if not isinstance(value, (Policy, System)):
        raise _FactoryError(f"Factory {reference!r} must return a System or Policy")
    return value


def _cell(value: object) -> str:
    """Display a stored scalar without inventing a value for unavailable evidence."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "Unavailable"
    return str(value)


def _print_result(
    result: EvaluationResult | TournamentResult, *, tournament: bool
) -> None:
    """Print existing facts and one bounded preview, without fitting or saving."""
    print(f"Status: {result.status.capitalize()}")
    if result.run_dir is None:
        print("Results Were Not Saved")
    else:
        print(f"Run Directory: {result.run_dir}")
    if not tournament:
        print(f"Phase: {result.metadata.get('phase', 'evaluation')}")
        print(f"Pass ID: {result.metadata.get('pass_id', '1')}")
        print(f"Completed Games: {len(getattr(result, 'completed_episode_ids', ()))}")
    for name in (
        "snapshot_id",
        "protocol_compliant",
        "planned_games",
        "reused_games",
        "executed_games",
    ):
        if hasattr(result, name):
            print(f"{name.replace('_', ' ').title()}: {_cell(getattr(result, name))}")
    for name in ("protocol_id", "executed_this_call"):
        if name in result.metadata:
            print(f"{name.replace('_', ' ').title()}: {_cell(result.metadata[name])}")
    table = "tournament_rankings" if tournament else "episodes"
    names = (
        ("rank", "system_name", "policy", "elo", "elo_ci_low", "elo_ci_high")
        if tournament
        else (
            "episode_id",
            "episode_length",
            "team_a_score",
            "team_b_score",
            "system_game_score",
        )
    )
    iterator = result.iter_table(table, rows=32)
    try:
        columns = next(iterator, None)
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()
    print(f"{table.replace('_', ' ').title()} — Up To 32 Rows")
    selected = [name for name in names if columns is not None and name in columns]
    if selected and columns is not None:
        print("\t".join(selected))
        for row in zip(*(columns[name] for name in selected), strict=True):
            print("\t".join(_cell(value) for value in row))
    else:
        print("No Available Rows")
    for path in result.replay_paths:
        print(f"Replay: {path}")
    if result.run_dir is None:
        print("Use --output-dir or Python for complete later result access.")
        if result.replays:
            print(
                "New replay captures were not saved; "
                "use --output-dir to open them later."
            )


def _run_experiment(command: str, arguments: dict[str, Any]) -> None:
    """Load supplied methods once and forward remaining arguments unchanged."""
    import marl_battlegrounds as marl_bgs

    for key in ("system", "opponent"):
        if key in arguments:
            arguments[key] = _load_method(arguments[key])
    if command == "evaluate":
        result = marl_bgs.evaluate(**arguments)
    elif command == "canonical":
        result: EvaluationResult | TournamentResult | CanonicalTournamentResult = (
            marl_bgs.run_canonical_tournament(**arguments)
        )
    else:
        result = marl_bgs.run_tournament(**arguments)
    _print_result(result, tournament=command != "evaluate")


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command, returning 0, 1 or 130; argparse exits with 0 or 2.

    Omitted argv reads process arguments. Expected API/file/factory failures
    print their cause to stderr and return 1. Keyboard interruption returns 130.
    Unexpected programming errors retain a traceback; no operation is retried.
    """
    parser = build_parser()
    namespace = parser.parse_args(argv)
    arguments = vars(namespace).copy()
    command = arguments.pop("command")
    if (
        command == "tournament"
        and "config" not in arguments
        and not arguments["resume_from"]
    ):
        parser.error("tournament requires --config for a new run")
    try:
        if command in {"evaluate", "canonical", "tournament"}:
            _run_experiment(command, arguments)
        elif command == "train":
            from marl_battlegrounds.training.runner import read_config, train

            if not arguments["resume_from"] and not arguments["config"]:
                parser.error("train requires --config for a new run")
            config = (
                read_config(arguments.pop("config")) if arguments["config"] else None
            )
            arguments.pop("config", None)
            result = train(config, **arguments)
            print(f"Training Complete: {result.run_dir}")
        elif command == "analyze-training":
            from marl_battlegrounds.training.analysis import analyze

            result = analyze(**arguments)
            print(f"Training Reports: {result['artifacts']['summary']}")
        elif command == "models":
            from marl_battlegrounds._cli_assets import run_models

            return run_models(namespace, parser)
        else:
            from marl_battlegrounds.viewer.launch import launch_replay
            from marl_battlegrounds.viewer.options import (
                resolve_playback_options,
                validate_playback_options,
            )

            options = resolve_playback_options(namespace)
            try:
                validate_playback_options(options)
            except ValueError as error:
                parser.error(str(error))
            return launch_replay(namespace.path, options=options)
    except (ValueError, OSError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    return 0
