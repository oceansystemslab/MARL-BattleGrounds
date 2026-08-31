#!/usr/bin/env python3
"""Owner-only promotion of one frozen DevClient candidate into tracked configs."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import cast

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reopen, revalidate, and promote one immutable DevClient candidate."
        )
    )
    parser.add_argument(
        "--candidate",
        required=True,
        help="Frozen candidate SHA-256 identity.",
    )
    parser.add_argument(
        "--asset-id",
        required=True,
        help="Durable lowercase kebab-case asset identity.",
    )
    parser.add_argument(
        "--version",
        required=True,
        type=int,
        help="Positive durable version number.",
    )
    parser.add_argument(
        "--approval-provenance",
        default="owner-cli",
        help="Short owner-approval provenance label (default: owner-cli).",
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=_REPOSITORY_ROOT,
        help=argparse.SUPPRESS,
    )
    return parser


def main(arguments: list[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    from scripts.dev.visual_debugger.authoring_models import SafeAssetId
    from scripts.dev.visual_debugger.authoring_store import DevAssetStore

    store = DevAssetStore(options.repository_root)
    destination = store.promote_candidate(
        options.candidate,
        asset_id=cast(SafeAssetId, options.asset_id),
        version=options.version,
        approval_provenance=options.approval_provenance,
    )
    relative_destination = destination.relative_to(
        options.repository_root.resolve(strict=True)
    )
    print(
        f"Promoted candidate {options.candidate} to {relative_destination.as_posix()}"
    )
    return 0


if __name__ == "__main__":
    os.environ["JAX_PLATFORMS"] = "cpu"
    raise SystemExit(main())
