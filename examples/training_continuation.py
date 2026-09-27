"""Continue a full learner checkpoint into a new child run.

Run from the repository with the training extra installed:
python examples/training_continuation.py RUN/checkpoints/ID CHILD --steps 1048576
The matching CLI is python -m marl_battlegrounds extend-training
RUN/checkpoints/ID --output-dir CHILD --additional-env-steps 1048576.
An optional --changes JSON declares future rules;
it never changes the parent's files. Scientific qualification is separate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import marl_battlegrounds as marl_bgs
from marl_battlegrounds import training


def complete_example(output_dir: str | Path) -> dict[str, object]:
    """Train, extend, resume, evaluate and warm-start through the public API.

    ``output_dir`` must be new or empty. This small example uses four CPU lanes
    and 16 transitions to prove the workflow, not learned skill. Set the device
    before calling; GPU examples must use the project's allowed batch sizes.
    Every child has its own folder. Return the parent, child, warm-start result
    and evaluation, including their saved paths and full resumable checkpoints.
    """
    from dataclasses import replace

    from marl_battlegrounds.baselines.ppo import PPOConfig

    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise ValueError("Example output_dir must be new or empty")
    config = training.TrainConfig(
        keep_past=0,
        method="ff_ippo",
        num_envs=4,
        total_env_steps=16,
        ppo=PPOConfig(rollout_length=2, epochs=1),
        metrics="none",
        verbose=False,
    )
    parent = training.train(config, output_dir=root / "parent")
    child = training.extend_training(
        parent.final_checkpoint,
        additional_env_steps=8,
        output_dir=root / "child",
        changes={"seed": 43, "ppo": {"entropy_coefficient": 0.02}},
    )
    resumed = training.train(resume_from=child.final_checkpoint)
    actor = training.load_system(resumed.final_actor)
    results = marl_bgs.evaluate(
        actor, "tdm-alpha", num_episodes=2, maps=[0], seed=7, num_envs=2
    )
    warm = training.train(
        replace(config, initial_actor=str(child.final_actor)),
        output_dir=root / "warm_start",
    )
    return {
        "parent": parent,
        "child": resumed,
        "warm_start": warm,
        "evaluation": results,
    }


def main() -> None:
    """Read a parent, exact added budget and optional future changes, then run.

    Writes only the requested new child directory through extend_training.
    Device selection belongs to the caller's environment before launch.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--changes", type=Path)
    args = parser.parse_args()
    changes_path = cast(Path | None, args.changes)
    changes: dict[str, Any] | None = None
    if changes_path is not None:
        value = json.loads(changes_path.read_text())
        if not isinstance(value, dict):
            raise ValueError("Continuation changes must be a JSON object")
        changes = cast(dict[str, Any], value)
        validation = changes.get("validation")
        if isinstance(validation, dict):
            validation = cast(dict[str, Any], validation)
            panel = validation.get("panel")
            if panel is not None and not isinstance(panel, str):
                raise ValueError("Continuation validation panel must be a path")
            if panel is not None and not Path(panel).is_absolute():
                validation["panel"] = str(
                    (changes_path.resolve().parent / panel).resolve()
                )
    result = training.extend_training(
        args.checkpoint,
        additional_env_steps=args.steps,
        output_dir=args.output_dir,
        changes=changes,
    )
    print(f"Child Run: {result.run_dir}")
    print(f"Selected Actor: {result.selected_actor}")
    print(f"Final Actor: {result.final_actor}")
    print(f"Final Checkpoint: {result.final_checkpoint}")


if __name__ == "__main__":
    main()
