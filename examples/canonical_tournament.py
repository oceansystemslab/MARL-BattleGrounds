"""Compare immutable tournament records through the public researcher interface.

Run ``python examples/canonical_tournament.py --help`` after installing MARL-BGs.
The fixture command creates clearly artificial local records, then exercises the
custom configuration route. It never installs an official Big 12. The compare
command needs a separately released bundle and prepared assets. The read command
uses the host-only reader and performs no game, fitting or file repair.
"""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal


def fixture(
    directory: Path,
    *,
    entrants: int,
    metrics: Literal["priority", "full", "none"],
    save: bool,
) -> None:
    """Create artificial records, read a complete field and optionally resume it.

    directory is explicit fixture storage. entrants selects 2 through 32 example
    versions; twelve/thirteen show the intended population sizes. metrics is
    priority, full or none. save=False creates no result directory; the source
    fixture files still exist because this command explicitly creates them.
    True saves a result, reloads bounded tables and resumes without new games.
    No learned-method or official qualification claim is made.
    """
    from canonical_fixture import build_record_bundle

    import marl_battlegrounds as marl_bgs
    from marl_battlegrounds.evaluation.tournament_assets import (
        prepare_tournament_assets,
    )

    bundle = build_record_bundle(
        directory / "source", entrants=entrants, full=metrics == "full"
    )
    prepared = prepare_tournament_assets(
        bundle["config"], roles=("outcomes_priority", "full_report"), download=False
    )
    assert not prepared["missing"]
    result = marl_bgs.run_tournament(
        config=prepared["config"],
        metrics=metrics,
        output_dir=directory / "results" if save else None,
    )
    print("Artificial Fixture — Not A Released Big 12")
    print("Imported Package:", marl_bgs.__file__)
    print("Status:", result.status)
    print("Rankings:", result.table("tournament_rankings"))
    if metrics != "none":
        print("Headlines:", result.table("tournament_headline_metrics"))
    for batch in result.iter_table("matches", rows=32):
        print("Original Games In This Batch:", len(batch["episode_id"]))
    if not save:
        assert result.run_dir is None
        return
    assert result.run_dir is not None
    print("Saved Run:", result.run_dir)
    loaded = marl_bgs.load_results(result.run_dir, phase="tournament")
    resumed = marl_bgs.run_tournament(resume_from=result.run_dir)
    assert loaded.status == resumed.status == "complete"
    print("Resumed Status:", resumed.status)


def compare(args: argparse.Namespace) -> None:
    """Run a selected released snapshot with explicit optional effects.

    args supplies the parsed snapshot/challenger, uniform budget, capture,
    save/resume and execution choices. Omitted values inherit saved conditions.
    Explicit --prepare allows downloads first; otherwise execution is offline.
    A missing official snapshot or insufficient reused coverage raises the public
    actionable error. This command never admits or publishes its challenger.
    """
    import marl_battlegrounds as marl_bgs
    from marl_battlegrounds.evaluation.tournament_assets import (
        prepare_tournament_assets,
    )

    options: dict[str, Any] = {}
    if args.config is not None:
        options["config"] = args.config
    if args.system is not None:
        options["system"] = args.system
    for argument in ("games_per_opponent", "metrics", "save_replays", "rerun_existing"):
        value = getattr(args, argument)
        if value is not None:
            options[argument] = value
    if args.prepare:
        from marl_battlegrounds.evaluation.tournament_config import (
            resolve_tournament_config,
        )

        saved = (
            None
            if args.resume_from is None
            else marl_bgs.load_results(args.resume_from).metadata
        )
        resolved = resolve_tournament_config(
            args.config,
            official=True,
            saved=None if saved is None else saved["canonical_config"],
        )
        mode = (
            args.metrics
            if args.metrics is not None
            else "priority"
            if saved is None
            else saved["metrics"]
        )
        fresh = (
            args.rerun_existing
            if args.rerun_existing is not None
            else False
            if saved is None
            else saved["rerun_existing"]
        )
        challenger = args.system is not None or (
            saved is not None and saved.get("challenger_id") is not None
        )
        roles: list[str] = ["model"] if fresh or challenger else []
        if not fresh:
            roles.append("outcomes_priority")
            if mode == "full" or (
                saved is not None and saved.get("full_metrics_episodes")
            ):
                roles.append("full_report")
            if args.save_replays or (
                saved is not None and saved.get("replay_episodes")
            ):
                roles.append("replay")
        prepared = prepare_tournament_assets(
            resolved,
            cache_dir=args.cache_dir,
            roles=tuple(roles),
            download=True,
        )
        if prepared["missing"]:
            raise ValueError(f"Required assets remain missing: {prepared['missing']}")
        options["config"] = prepared["config"]
    result = marl_bgs.run_canonical_tournament(
        **options,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
        num_envs=args.num_envs,
        chunk_size=args.chunk_size,
    )
    print("Snapshot:", result.snapshot_id)
    print(
        "Reused Games:", result.reused_games, "Executed Games:", result.executed_games
    )
    print("Rankings:", result.table("tournament_rankings"))
    print("Replay Files:", result.replay_paths)
    if result.run_dir is not None:
        print("Saved Run:", result.run_dir)


def read(directory: Path, table: str) -> None:
    """Print bounded existing table batches without importing the simulator.

    directory is one exact saved run. table uses the shared public table names.
    Required external source files must remain available for dependent reads.
    No fitting, downloads, recovery or file writes occur.
    """
    import marl_battlegrounds as marl_bgs

    result = marl_bgs.load_results(directory)
    print("Imported Package:", marl_bgs.__file__)
    print("Status:", result.status)
    for batch in result.iter_table(table, rows=32):
        print(batch)


def main() -> None:
    """Parse only example choices and call the shared public Python authorities."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    artificial = commands.add_parser(
        "fixture", help="Create artificial records, without official publication"
    )
    artificial.add_argument("--directory", type=Path)
    artificial.add_argument("--entrants", type=int, default=12)
    artificial.add_argument(
        "--metrics", choices=("priority", "full", "none"), default="priority"
    )
    artificial.add_argument("--save", action="store_true")
    canonical = commands.add_parser(
        "compare", help="Use a separately released snapshot"
    )
    canonical.add_argument("--config", type=Path)
    canonical.add_argument("--system")
    canonical.add_argument("--games-per-opponent", type=int)
    canonical.add_argument("--metrics", choices=("priority", "full", "none"))
    canonical.add_argument("--save-replays", type=int)
    canonical.add_argument("--rerun-existing", action="store_true", default=None)
    canonical.add_argument("--output-dir", type=Path)
    canonical.add_argument("--resume-from", type=Path)
    canonical.add_argument("--num-envs", type=int, default=128)
    canonical.add_argument("--chunk-size", type=int, default=16)
    canonical.add_argument(
        "--prepare",
        action="store_true",
        help="Explicitly download missing declared assets",
    )
    canonical.add_argument("--cache-dir", type=Path)
    saved = commands.add_parser(
        "read", help="Read existing results without games or JAX"
    )
    saved.add_argument("run_dir", type=Path)
    saved.add_argument("--table", default="matches")
    args = parser.parse_args()
    if args.command == "fixture" and args.save and args.directory is None:
        parser.error(
            "fixture --save requires --directory so saved results remain available"
        )
    if args.command == "read":
        read(args.run_dir, args.table)
    elif args.command == "compare":
        compare(args)
    elif args.directory is not None:
        fixture(
            args.directory, entrants=args.entrants, metrics=args.metrics, save=args.save
        )
    else:
        with TemporaryDirectory(prefix="marlbg-canonical-fixture-") as temporary:
            fixture(
                Path(temporary),
                entrants=args.entrants,
                metrics=args.metrics,
                save=args.save,
            )


if __name__ == "__main__":
    main()
