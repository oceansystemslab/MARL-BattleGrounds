"""Run a small complete CPU recurrent PQN-VDN workflow through the public API.

Install training and viz extras. Run with JAX_PLATFORMS=cpu::

    python examples/pqn_training.py --output-dir artifacts/pqn-example
    python examples/pqn_training.py --output-dir artifacts/pqn-c --curriculum
    python examples/pqn_training.py --resume-from CHECKPOINT --evaluate

The tiny default run uses four games and 176 real transitions: blocks of 4
rounds and a memory window of 2, so the first 6 rounds (a chunk of 4, then a
chunk of 2) are random initial collection, and the next 38 rounds are ten
learning blocks of two optimizer steps each (the last block has 2 rounds).
It checks wiring, not learning. --curriculum and --shaping each add one
setting (together they give C-RS) to that tiny new run; they are refused with
--config or --resume-from, whose settings come from the file or the saved
run. --config reads the same JSON settings as
the package train command; use the training guide's panel-backed
configuration for real checkpoint selection. The loaded actor plays greedily
(epsilon 0). --evaluate plays two games against the scripted "tdm-alpha" and a
two-entrant tournament; these are diagnostics, and no trained competence is
implied.
"""

import argparse
from pathlib import Path


def main() -> None:
    """Parse paths, call public training/loading/evaluation/analysis and show outputs.

    Exactly one new output directory or complete learner checkpoint is required.
    --config overrides the tiny new-run example; resume inherits its saved config.
    --curriculum and --shaping change only the tiny new-run example; with
    --config or --resume-from the parser exits with status 2 before any work.
    --evaluate
    adds two CPU games against "tdm-alpha" and a two-entrant tournament, and
    saves their ordinary M8 records. Invalid settings and execution errors
    propagate without an automatic retry.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    paths = parser.add_mutually_exclusive_group(required=True)
    paths.add_argument("--output-dir", type=Path)
    paths.add_argument("--resume-from", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--shaping", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if (args.curriculum or args.shaping) and (
        args.config is not None or args.resume_from is not None
    ):
        parser.error(
            "--curriculum and --shaping cannot accompany --config or --resume-from"
        )

    import marl_battlegrounds as marl_bgs
    from marl_battlegrounds import training
    from marl_battlegrounds.baselines.pqn import PQNConfig
    from marl_battlegrounds.training.runner import read_config

    config = None
    if args.config:
        config = read_config(args.config)
    elif args.resume_from is None:
        config = training.TrainConfig(
            method="pqn_vdn",
            seed=42,
            num_envs=4,
            total_env_steps=176,
            curriculum=args.curriculum,
            shaping=args.shaping,
            pqn=PQNConfig(
                rollout_length=4,
                memory_window=2,
                epochs=1,
                num_minibatches=2,
            ),
        )
    result = training.train(
        config,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
    )
    actor = training.load_system(result.selected_actor or result.final_actor)
    if args.evaluate:
        marl_bgs.evaluate(
            actor,
            "tdm-alpha",
            num_episodes=2,
            maps=[42],
            seed=43,
            num_envs=2,
            phase="validation",
            output_dir=result.run_dir / "example_evaluation",
        )
        marl_bgs.run_tournament(
            (actor, "tdm-alpha"),
            maps=[47],
            episodes_per_pair=2,
            seed=44,
            num_envs=2,
            output_dir=result.run_dir / "example_tournament",
        )
    reports = training.analyze([result.run_dir], output_dir=result.run_dir / "analysis")
    print(f"Run: {result.run_dir}")
    print(f"Frozen Actor: {result.selected_actor or result.final_actor}")
    print(f"Optimizer Steps: {result.completed_updates}")
    print(f"Summary: {reports['artifacts']['summary']}")


if __name__ == "__main__":
    main()
