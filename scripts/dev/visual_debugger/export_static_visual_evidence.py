"""Export deterministic Matplotlib evidence for tracked visual review."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from scripts.dev.visual_debugger.static_renderer import (
    STATIC_VISUAL_VOCABULARY_PATH,
    export_static_visual_vocabulary,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the narrow static-evidence export CLI."""
    parser = argparse.ArgumentParser(
        description="Export the static renderer visual vocabulary at 1440x900.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=STATIC_VISUAL_VOCABULARY_PATH,
        help=f"PNG output path (default: {STATIC_VISUAL_VOCABULARY_PATH})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Export one deterministic vocabulary PNG."""
    args = build_parser().parse_args(argv)
    export_static_visual_vocabulary(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
