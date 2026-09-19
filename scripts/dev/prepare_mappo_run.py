#!/usr/bin/env python3
"""Prepare fixed MAPPO source, environment, panel and commands without launching.

Run with the installed training/viz extras and uv available:
python scripts/dev/prepare_mappo_run.py --repository REPO --commit QUALIFIED_SHA
    --config CONFIG_JSON --destination NEW_PACKAGE --gpu-uuid INTERNAL_GPU_UUID
Preparation reads Git, writes only the new package and installs its locked
environment. The user separately runs the printed launch command. No branch,
index, commit, training process or remote publication is changed here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from marl_battlegrounds.training._launch import prepare_run


def main() -> None:
    """Parse explicit source/config/hardware choices and print prepared commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    arguments = parser.parse_args()
    result = prepare_run(
        arguments.repository,
        arguments.destination,
        arguments.config,
        commit=arguments.commit,
        gpu_uuid=arguments.gpu_uuid,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
