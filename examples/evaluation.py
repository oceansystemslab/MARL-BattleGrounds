"""Run complete matches, repeated validation passes or a small tournament.

Run from the repository with ``.venv/bin/python examples/evaluation.py --help``.
Selections are ordinary Python iterables; replace the examples with your own
``range`` or list when integrating these calls into a research script.
"""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", choices=("evaluate", "validation", "tournament"))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument(
        "--metrics", choices=("none", "priority", "full"), default="priority"
    )
    parser.add_argument("--save-replays", type=int, default=0, metavar="FIRST_N")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    from marl_battlegrounds import RunWriter, evaluate, run_tournament
    from marl_battlegrounds.tasks import list_tdm_maps

    if not 0 <= args.save_replays <= args.episodes:
        parser.error("--save-replays must be between zero and --episodes")
    replays = range(1, args.save_replays + 1)

    if args.workflow == "tournament":
        result = run_tournament(
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
        print("Files:", result.paths or "none; results are in memory")
        return

    if args.workflow == "validation":
        if args.output_dir is None:
            parser.error("this shared-writer example requires --output-dir")
        validation_maps = [
            item for item in list_tdm_maps() if item.split == "validation"
        ]
        with RunWriter(args.output_dir) as writer:
            for training_episode in (1_000, 5_000):
                # A real trainer supplies its current frozen checkpoint here.
                # Pass identity separates repeated episode IDs in the same CSV.
                evaluate(
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

    result = evaluate(
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
    print("Files:", result.paths or "none; results are in memory")


if __name__ == "__main__":
    main()
