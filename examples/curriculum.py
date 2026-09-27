"""Train with declared curriculum stages, then inspect actual play and evaluate.

Install the training extra. The default CPU run checks the complete workflow::

    JAX_PLATFORMS=cpu python examples/curriculum.py --output-dir artifacts/curriculum

It uses 32 real transitions across four games. That is too short for learning,
so the run may end before any game starts under the second requested stage.
Read the actual exposure printed below; stage shares describe requested work.
For a separately authorized real run, set both the batch and experience budget,
for example --num-envs 512 --total-env-steps 10240000. No budget is changed for you.
The example uses no past-copy bank; its capture spacing does not limit a smoke run.

Each stage names training maps, a share, explicit class rosters and a victory
score. Repeated classes remain distinct agents. Evaluation uses held-out map 42
and the second roster under a declared K20/H300 setup. Evaluation plays one
game per evaluation lane, using min(num_envs, 32) lanes and both spawn ends.
This evaluation budget is separate from the requested training transitions.
A smoke result is not proof of competence, fairness or useful curriculum learning.
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    """Run a declared stage list, load its frozen actor and show actual results.

    --output-dir must name a new run directory. --num-envs and --total-env-steps
    use the ordinary training configuration and validation rules. Errors are
    reported without retrying or changing the requested experiment.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--total-env-steps", type=int, default=32)
    args = parser.parse_args()

    import marl_battlegrounds as marl_bgs
    from marl_battlegrounds import training
    from marl_battlegrounds.baselines.ppo import PPOConfig

    config = training.TrainConfig(
        num_envs=args.num_envs,
        total_env_steps=args.total_env_steps,
        seed=42,
        keep_past=0,
        curriculum=[
            {
                "share": 0.5,
                "maps": [0],
                "rosters": {"system": ["mage", "mage"], "opponent": ["warrior"]},
                "score_threshold": 5,
            },
            {
                "share": 0.5,
                "maps": [1, 2, 3],
                "rosters": {
                    "system": ["mage", "mage", "priest"],
                    "opponent": ["warrior", "hunter"],
                },
                "score_threshold": 20,
            },
        ],
        ppo=PPOConfig(rollout_length=4, epochs=1),
    )
    result = training.train(config, output_dir=args.output_dir)
    actor_path = result.selected_actor or result.final_actor
    actor = training.load_system(actor_path)
    evaluation_dir = result.run_dir / "curriculum_evaluation"
    evaluation_batch = min(args.num_envs, 32)
    evaluation = marl_bgs.evaluate(
        actor,
        "random",
        system_roster=["mage", "mage", "priest"],
        opponent_roster=["warrior", "hunter"],
        maps=[42],
        num_episodes=evaluation_batch,
        num_envs=evaluation_batch,
        score_threshold=20,
        max_steps=300,
        seed=43,
        phase="validation",
        output_dir=evaluation_dir,
    )
    exposure = json.loads((result.run_dir / "exposure.json").read_text())
    print(f"Actor: {actor_path}")
    print(f"Actual Transitions By Stage: {exposure['steps_by_episode_stage']}")
    print(f"Actual Game Starts By Stage: {exposure['starts_by_episode_stage']}")
    print(f"Evaluation: {evaluation.run_dir}")


if __name__ == "__main__":
    main()
