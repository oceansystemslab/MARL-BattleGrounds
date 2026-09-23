"""Run a small complete CPU recurrent QMIX workflow through the public API.

Install training and viz extras. Run with JAX_PLATFORMS=cpu::

    python examples/qmix_training.py --output-dir artifacts/qmix-example
    python examples/qmix_training.py --resume-from CHECKPOINT --evaluate

The tiny default run uses four games and 192 real transitions: blocks of 8
rounds, a 64-row replay per game that starts learning at 32 rows, so three
warmup blocks store rows and the last three each take one optimizer step. It
checks wiring, not learning. --config reads the same JSON settings as the
package train command; use a declared panel-backed configuration for a real
demonstration. The loaded actor plays greedily (epsilon 0). Evaluation is
optional and uses Random only as a diagnostic. No trained competence is implied.
"""

import argparse
from pathlib import Path


def main() -> None:
    """Parse paths, call public training/loading/evaluation/analysis and show outputs.

    Exactly one new output directory or complete learner checkpoint is required.
    --config overrides the tiny new-run example; resume inherits its saved config.
    --evaluate adds two CPU diagnostic games and saves their ordinary M8 records.
    Invalid settings and execution errors propagate without an automatic retry.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    paths = parser.add_mutually_exclusive_group(required=True)
    paths.add_argument("--output-dir", type=Path)
    paths.add_argument("--resume-from", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    import marl_battlegrounds as marl_bgs
    from marl_battlegrounds import training
    from marl_battlegrounds.baselines.qmix import QMIXConfig
    from marl_battlegrounds.training.runner import read_config

    config = None
    if args.config:
        config = read_config(args.config)
    elif args.resume_from is None:
        config = training.TrainConfig(
            method="qmix",
            seed=42,
            num_envs=4,
            total_env_steps=192,
            qmix=QMIXConfig(
                rollout_length=8,
                buffer_size=64,
                min_buffer_size=32,
                sample_sequence_length=20,
                sample_batch_size=4,
                epochs=1,
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
            "random",
            num_episodes=2,
            maps=[42],
            seed=43,
            num_envs=2,
            phase="validation",
            output_dir=result.run_dir / "example_evaluation",
        )
    reports = training.analyze([result.run_dir], output_dir=result.run_dir / "analysis")
    print(f"Run: {result.run_dir}")
    print(f"Frozen Actor: {result.selected_actor or result.final_actor}")
    print(f"Completed Updates (Optimizer Steps For QMIX): {result.completed_updates}")
    print(f"Summary: {reports['artifacts']['summary']}")


if __name__ == "__main__":
    main()
