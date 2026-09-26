"""Parse package commands and call the existing Python authorities once.

Run ``python -m marl_battlegrounds --help`` for commands. Parsing and help use
only the standard library. Scientific validation, execution and result meanings
belong to the called APIs. Absent scientific options stay absent during resume.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

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


def _decimal(text: str) -> float:
    """Parse a plain decimal number, such as 6, 6.0 or -0.5, into a float.

    Parameters
    ----------
    text : str
        An optional + or - sign, then ASCII digits with at most one decimal
        point and at least one digit.

    Returns
    -------
    float
        The Python float value; "6" gives 6.0.

    Raises
    ------
    argparse.ArgumentTypeError
        The text has an exponent (1e3), NaN, infinity, underscores (1_0),
        spaces, other characters or a second point. argparse reports it as a
        usage error. The called API checks the allowed range.
    """
    digits = text[1:] if text.startswith(("+", "-")) else text
    whole, _, fraction = digits.partition(".")
    number = whole + fraction
    if not number or not number.isascii() or not number.isdecimal():
        raise argparse.ArgumentTypeError("Use a decimal number such as 5.0")
    return float(text)


def _ids(text: str) -> tuple[int, ...]:
    """Parse a nonempty ordered map-ID list; the API checks its allowed values."""
    return tuple(_integer(value) for value in _tokens(text))


def _capture_ids(text: str) -> tuple[int, ...]:
    """Parse episode IDs, keeping explicit empty text as an empty assertion."""
    return tuple(_integer(value) for value in _tokens(text, empty=True))


def _execution_options(parser: argparse.ArgumentParser) -> None:
    """Add recording and execution flags without asserting absent science.

    Evaluation and tournaments use at most 128 parallel games by default, with
    16 ticks per compiled chunk. Tournaments run one matchup at a time. Omitted
    scientific settings inherit saved run settings.
    """
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
        help="Parallel environments in the active matchup (default: 128)",
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
        "train",
        allow_abbrev=False,
        help="Run or resume a declared PPO, QMIX or PQN-VDN experiment",
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
    analysis.add_argument(
        "--selection", help="Completed separate selection decision JSON"
    )
    analysis.add_argument("--grouping", help="Declared training-run comparison JSON")
    selection = commands.add_parser(
        "reselect-checkpoint",
        allow_abbrev=False,
        help="Write a separate checkpoint decision from saved validation",
    )
    selection.add_argument("run_dirs", nargs="+", help="Saved training run directories")
    selection.add_argument(
        "--declaration", required=True, help="Named selection rule JSON"
    )
    selection.add_argument(
        "--output-dir", required=True, help="Exact new or empty directory"
    )
    extension = commands.add_parser(
        "extend-training",
        allow_abbrev=False,
        help="Continue a full learner checkpoint into a new child run",
    )
    extension.add_argument("checkpoint", help="Complete checkpoints/<id> folder")
    extension.add_argument("--additional-env-steps", required=True, type=_integer)
    extension.add_argument(
        "--output-dir", required=True, help="Exact new or empty child folder"
    )
    extension.add_argument(
        "--changes",
        help="JSON with future validation, rate, exploration or history changes",
    )
    study = commands.add_parser(
        "study", allow_abbrev=False, help="Run or inspect a declared training study"
    )
    study_actions = study.add_subparsers(dest="study_action", required=True)
    for action, description in (
        ("run", "Run the study in the foreground"),
        ("start", "Start the study in the background"),
    ):
        study_action = study_actions.add_parser(
            action, allow_abbrev=False, help=description
        )
        study_action.add_argument("--config", help="Study declaration JSON")
        study_output = study_action.add_mutually_exclusive_group(required=True)
        study_output.add_argument(
            "--output-dir", help="Exact new or empty study directory"
        )
        study_output.add_argument(
            "--resume-from", help="Saved study directory; keep its original deadline"
        )
    for action, description in (
        ("status", "Read saved progress without starting a numerical backend"),
        ("stop", "Ask the study's exact live processes to stop"),
    ):
        study_action = study_actions.add_parser(
            action, allow_abbrev=False, help=description
        )
        study_action.add_argument("study_dir", help="Saved study directory")
    evaluate = commands.add_parser(
        "evaluate", allow_abbrev=False, help="Evaluate two frozen methods"
    )
    evaluate.add_argument(
        "--system",
        required=True,
        help="Team A name, actor export/checkpoint folder or module:factory",
    )
    evaluate.add_argument(
        "--opponent",
        required=True,
        help="Team B name, actor export/checkpoint folder or module:factory",
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
        "--red-zone-depth",
        type=_decimal,
        default=argparse.SUPPRESS,
        help=(
            "Red Zone depth in map units: an agent that dies in its own "
            "team's zone gives the enemy 2 points; 0 turns it off. New-run "
            "default: 5.0; omission inherits saved settings"
        ),
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
        help="Compare the released field, optionally with a challenger",
    )
    canonical.add_argument(
        "--system",
        default=argparse.SUPPRESS,
        help=(
            "Challenger name, actor export/checkpoint folder or module:factory; "
            "omission inherits on resume"
        ),
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
        "tournament", allow_abbrev=False, help="Run an entrant list or saved field"
    )
    population = tournament.add_mutually_exclusive_group()
    population.add_argument(
        "--config",
        default=argparse.SUPPRESS,
        help="JSON field and rules; use this or --entrants for a new run",
    )
    population.add_argument(
        "--entrants",
        dest="policies",
        type=_tokens,
        default=argparse.SUPPRESS,
        help=(
            "Comma-separated names, actor/checkpoint folders or "
            "module:factory references"
        ),
    )
    tournament.add_argument(
        "--challenger",
        default=argparse.SUPPRESS,
        help="System reference to compare against the supplied field",
    )
    for flag in ("games-per-opponent", "episodes-per-pair"):
        tournament.add_argument(
            f"--{flag}",
            type=_integer,
            default=argparse.SUPPRESS,
            help=(
                "Games per opponent; omission inherits field or saved settings"
                if flag == "games-per-opponent"
                else "Alias for --games-per-opponent; both values must agree"
            ),
        )
    tournament.add_argument(
        "--maps",
        type=_ids,
        default=argparse.SUPPRESS,
        help="Ordered comma-separated map IDs; omission inherits field or API defaults",
    )
    for flag in ("seed", "score-threshold", "max-steps"):
        tournament.add_argument(
            f"--{flag}",
            type=_integer,
            default=argparse.SUPPRESS,
            help=(
                "Omission inherits field or API defaults; resume checks saved settings"
            ),
        )
    tournament.add_argument(
        "--red-zone-depth",
        type=_decimal,
        default=argparse.SUPPRESS,
        help=(
            "Red Zone depth in map units; 0 turns it off. Omission inherits "
            "field or API defaults; resume checks saved settings"
        ),
    )
    tournament.add_argument(
        "--rerun-existing",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Explicitly rerun the whole field; omission inherits on resume",
    )
    _execution_options(tournament)

    population_selection = commands.add_parser(
        "select-population",
        allow_abbrev=False,
        help="Freeze the declared initial population from saved tournament results",
    )
    population_selection.add_argument("result", help="Saved tournament run directory")
    population_selection.add_argument(
        "--output-dir", required=True, help="Exact new or empty decision directory"
    )
    population_selection.add_argument(
        "--declaration",
        help="Optional JSON assertion of the selection rule saved before games",
    )

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
    """Resolve saved actors and trusted factories through the shared loader.

    Bare names stay strings for the evaluator. Existing folders take priority
    over factory syntax, including folders containing a colon. Both actor exports
    and complete learner checkpoints load only their actor arrays. Factory errors
    retain their original cause through _FactoryError. Folder errors retain their
    loader error type. No action is chosen and no learner is constructed.
    """
    from marl_battlegrounds._method_loading import load_method

    if Path(reference).is_dir():
        return load_method(reference)
    if ":" not in reference:
        return reference
    try:
        return load_method(reference)
    except TypeError as error:
        raise _FactoryError(str(error)) from error
    except Exception as error:
        raise _FactoryError(f"Factory {reference!r} failed: {error}") from error


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
    """Call the shared API once, preserving tournament references for its loader.

    Evaluation methods load here once. Tournament APIs own entrant and
    challenger loading, including capacity checks, so their references reach
    those APIs unchanged. No parser path imports a researcher factory.
    """
    import marl_battlegrounds as marl_bgs

    if command == "evaluate":
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
        and "policies" not in arguments
        and not arguments["resume_from"]
    ):
        parser.error("tournament requires --config or --entrants for a new run")
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
        elif command == "study":
            from marl_battlegrounds.training import study

            action = arguments.pop("study_action")
            if action in {"run", "start"}:
                if not arguments["resume_from"] and not arguments["config"]:
                    parser.error("study requires --config for a new study")
                function = study.run_study if action == "run" else study.start_study
                result = function(**arguments)
            else:
                function = (
                    study.study_status if action == "status" else study.stop_study
                )
                result = function(arguments["study_dir"])
            print(json.dumps(result, indent=2, allow_nan=False))
            if action == "run":
                return 0 if result["status"] == "complete" else 1
        elif command == "extend-training":
            from marl_battlegrounds.training.runner import extend_training

            changes_path = arguments.pop("changes")
            changes = (
                None
                if changes_path is None
                else json.loads(Path(changes_path).read_text())
            )
            if changes is not None and not isinstance(changes, dict):
                raise ValueError("Continuation changes must be a JSON object")
            changes = cast(dict[str, Any] | None, changes)
            if changes is not None and isinstance(changes.get("validation"), dict):
                validation = cast(dict[str, Any], changes["validation"])
                panel = validation.get("panel")
                if panel is not None and not isinstance(panel, str):
                    raise ValueError("Continuation validation panel must be a path")
                if panel is not None and not Path(panel).is_absolute():
                    validation["panel"] = str(
                        (Path(str(changes_path)).resolve().parent / panel).resolve()
                    )
            result = extend_training(**arguments, changes=changes)
            print(f"Training Continued: {result.run_dir}")
        elif command == "analyze-training":
            from marl_battlegrounds.training.analysis import analyze

            result = analyze(**arguments)
            print(f"Training Reports: {result['artifacts']['summary']}")
        elif command == "reselect-checkpoint":
            from marl_battlegrounds.training.selection import reselect_checkpoint

            result = reselect_checkpoint(**arguments)
            print(f"Selection {result['status']}: {arguments['output_dir']}")
            for run in result["runs"]:
                if run["winner"] is not None:
                    print(f"Selected Actor: {run['winner']['actor_path']}")
                for request in run["needs_confirmation"]:
                    print(request["python"])
        elif command == "select-population":
            import marl_battlegrounds as marl_bgs

            result = marl_bgs.select_initial_population(**arguments)
            print(json.dumps(result, indent=2, allow_nan=False))
            return 0 if result["status"] == "complete" else 1
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
