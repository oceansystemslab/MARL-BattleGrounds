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

from marl_battlegrounds import training


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


if __name__ == "__main__":
    main()
