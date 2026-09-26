"""Declare a small selection field, freeze two members, then run their test field.

Run from the installed checkout with a new output directory::

    JAX_PLATFORMS=cpu uv run --no-sync python examples/population_selection.py \
        --output-dir artifacts/population-demo

This software demo uses random, tdm-alpha and tdm-beta on validation maps 42 through 46.
It saves the selection rule before games, then tests only the frozen members on
maps 47 through 51. Each matchup has 10 games, capped at 2 steps each. These short games
show the workflow; they do not choose an official population or prove competence.
The default backend is CPU unless JAX_PLATFORMS is already set. Help imports no
numerical modules. Results go into separate selection, decision and test folders.

For an existing declared run, the matching command is ``python -m
marl_battlegrounds select-population RUN --output-dir DECISION``. Its optional
--declaration JSON only checks the rule already saved before games.
See docs/evaluation/population_selection.md for the full declaration contract.
"""

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bounded demo and return 0 on completion, or 1 if evidence is incomplete.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Command arguments; None reads the process arguments. Required
        --output-dir names a parent for new selection, decision and test folders.

    Returns
    -------
    int
        Zero after both fields finish; one if selection or testing is incomplete.
        An incomplete selection never starts the test field.

    Notes
    -----
    Parsing finishes before runtime imports. CPU is selected when no backend was
    set. Both fields use two parallel games and two steps per compiled chunk.
    API and file errors reach the caller; existing output is not overwritten.
    """
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    import marl_battlegrounds as marl_bgs

    entrants = ["random", "tdm-alpha", "tdm-beta"]
    result = marl_bgs.run_tournament(
        config={
            "entrants": entrants,
            "maps": [42, 43, 44, 45, 46],
            "games_per_opponent": 10,
            "max_steps": 2,
            "seed": 17,
            "selection": {
                "size": 2,
                "entrant_order": entrants,
                "failure_policy": "require-complete-field",
            },
        },
        output_dir=args.output_dir / "selection",
        num_envs=2,
        chunk_size=2,
    )
    decision = marl_bgs.select_initial_population(
        result, output_dir=args.output_dir / "decision"
    )
    print(json.dumps(decision, indent=2, allow_nan=False))
    if decision["status"] != "complete":
        return 1
    selected = [row["controller"]["reference"] for row in decision["members"]]
    tested = marl_bgs.run_tournament(
        config={
            "entrants": selected,
            "maps": [47, 48, 49, 50, 51],
            "games_per_opponent": 10,
            "max_steps": 2,
            "seed": 29,
            "selection": {
                "stage": "test",
                "population": str(args.output_dir / "decision" / "population.json"),
            },
        },
        output_dir=args.output_dir / "test",
        num_envs=2,
        chunk_size=2,
    )
    print(f"Test Status: {tested.status.capitalize()}")
    print(f"Population: {decision['population_id']}")
    return 0 if tested.status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
