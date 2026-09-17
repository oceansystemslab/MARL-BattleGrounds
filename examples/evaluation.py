"""Run complete matches, repeated validation passes or a small tournament.

Run ``python examples/evaluation.py --help`` after installing MARL-BGs.
The package command ``python -m marl_battlegrounds evaluate --help`` calls the
same evaluator. Copy example files separately; no checkout runtime is required.
Selections are ordinary Python iterables; replace the examples with your own
``range`` or list when integrating these calls into a research script.
"""

import argparse
from pathlib import Path


def main() -> None:
    """Read command-line options and run the selected public workflow.

    Evaluation returns results in memory unless an output directory is given.
    Validation shares one writer across two named passes and requires a directory.
    Tournament mode prints each ranking row and any written file paths. Omitted
    --episodes means 32 total evaluation/validation games or 100 games per
    tournament matchup, covering all five maps with complete spawn pairs.

    Raises
    ------
    SystemExit
        If arguments are invalid or help was requested.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", choices=("evaluate", "validation", "tournament"))
    parser.add_argument(
        "--episodes",
        type=int,
        help=(
            "Games: default 32 total for evaluation/validation, "
            "100 per tournament matchup"
        ),
    )
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument(
        "--metrics", choices=("none", "priority", "full"), default="priority"
    )
    parser.add_argument("--save-replays", type=int, default=0, metavar="FIRST_N")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.episodes is None:
        args.episodes = 100 if args.workflow == "tournament" else 32
    import marl_battlegrounds as marl_bgs
    from marl_battlegrounds.tasks import list_tdm_maps

    if not 0 <= args.save_replays <= args.episodes:
        parser.error("--save-replays must be between zero and --episodes")
    replays = range(1, args.save_replays + 1)

    if args.workflow == "tournament":
        result = marl_bgs.run_tournament(
            ["tdm-alpha", "tdm-beta"],
            episodes_per_pair=args.episodes,
            seed=42,
            num_envs=args.num_envs,
            metrics=args.metrics,
            replay_episodes=replays,
            output_dir=args.output_dir,
        )
        for row in result.tournament_results:
            print(row)
        print("Run Directory:", result.run_dir or "Results Were Not Saved")
        return

    if args.workflow == "validation":
        if args.output_dir is None:
            parser.error("this shared-writer example requires --output-dir")
        validation_maps = [
            item for item in list_tdm_maps() if item.split == "validation"
        ]
        with marl_bgs.RunWriter(args.output_dir) as writer:
            for training_episode in (1_000, 5_000):
                # A real trainer supplies its current frozen checkpoint here.
                # Pass identity separates repeated episode IDs in the same CSV.
                marl_bgs.evaluate(
                    "tdm-alpha",
                    "tdm-beta",
                    maps=validation_maps,
                    num_episodes=args.episodes,
                    seed=training_episode,
                    num_envs=args.num_envs,
                    metrics=args.metrics,
                    replay_episodes=replays,
                    writer=writer,
                    phase="validation",
                    pass_id=f"after-{training_episode}-training-episodes",
                )
            writer.flush()
            print("Files:", writer.paths)
        return

    result = marl_bgs.evaluate(
        "tdm-alpha",
        "tdm-beta",
        num_episodes=args.episodes,
        seed=42,
        num_envs=args.num_envs,
        metrics=args.metrics,
        replay_episodes=replays,
        output_dir=args.output_dir,
    )
    print("Completed episodes:", result.completed_episode_ids)
    print("Run Directory:", result.run_dir or "Results Were Not Saved")


if __name__ == "__main__":
    main()
