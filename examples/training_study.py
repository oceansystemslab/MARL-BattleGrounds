"""Run the small declared study through the public Python interface.

From the repository root, run::

    python examples/training_study.py --output-dir artifacts/example-study

The default JSON declares two short FF-IPPO runs. It checks the workflow, not
learning quality. Pass another --config for a real declared experiment. To
resume, use --resume-from instead of --output-dir; the first launch's deadline
still applies. Use --background for a detached launch. The package CLI provides
matching study run/start/status/stop commands. No GPU is selected by this script.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from marl_battlegrounds import training


def main(argv: Sequence[str] | None = None) -> int:
    """Run or start a declared study and print its saved report as JSON.

    argv is an optional argument list; None reads process arguments. A new run
    needs an exact new or empty output directory. Resume reads the declaration
    from the saved study unless --config is supplied as an equality check.
    Return zero after a successful launch or complete foreground study, and one
    for an incomplete foreground result. API errors retain their traceback.
    """
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, help="Study declaration JSON")
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--output-dir", type=Path, help="Exact new or empty folder")
    output.add_argument("--resume-from", type=Path, help="Saved study folder")
    parser.add_argument("--background", action="store_true", help="Launch detached")
    args = parser.parse_args(argv)
    config = args.config
    if config is None and args.resume_from is None:
        config = Path(__file__).with_suffix(".json")
    function = training.start_study if args.background else training.run_study
    report = function(config, output_dir=args.output_dir, resume_from=args.resume_from)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if args.background or report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
