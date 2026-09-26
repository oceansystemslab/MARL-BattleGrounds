"""Write a separate checkpoint choice and print any missing confirmation calls.

Run with saved completed training data, for example::

    JAX_PLATFORMS=cpu uv run --no-sync python examples/training_selection.py RUN \
        --declaration choice.json --output-dir artifacts/new-choice

choice.json contains a schema-1 named selection declaration. Existing artifact
imports can initialize JAX; the CPU setting above applies before those imports.
This reads actor artifacts and saved game tables; it does not train or execute a
model. Output must be new or empty. Missing confirmations are printed, never run
automatically.
The matching package command is ``python -m marl_battlegrounds
reselect-checkpoint RUN --declaration choice.json --output-dir DIRECTORY``.
"""

import argparse
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    """Read saved runs named by argv (or command-line arguments) and print a choice.

    Required --declaration supplies the named rule; --output-dir receives the new
    immutable decision. Invalid declarations or evidence retain their ValueError
    or OSError. The function returns None and performs no game or provider calls.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--declaration", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    from marl_battlegrounds import training

    decision = training.reselect_checkpoint(
        args.run_dirs,
        declaration=args.declaration,
        output_dir=args.output_dir,
    )
    print(f"Selection Status: {decision['status']}")
    for run in decision["runs"]:
        print(f"Run: {run['run_id']}; Status: {run['status']}")
        if run["winner"] is not None:
            print(f"Selected Actor: {run['winner']['actor_path']}")
        for request in run["needs_confirmation"]:
            print(request["python"])


if __name__ == "__main__":
    main()
