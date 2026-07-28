"""Export one registered synthetic renderer fixture as JSON."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from typing import cast

from marl_battlegrounds.rendering.scene import to_jsonable
from scripts.dev.visual_debugger.renderer_fixtures import (
    get_renderer_fixture,
    list_renderer_fixtures,
)

_FIXTURE_NAMES = tuple(fixture.name for fixture in list_renderer_fixtures())


def serialize_renderer_fixture(name: str) -> str:
    """Return one exact registered fixture as a JSON document."""
    fixture = get_renderer_fixture(name)
    return json.dumps(to_jsonable(fixture))


def build_parser() -> argparse.ArgumentParser:
    """Build the narrow synthetic-fixture export CLI."""
    parser = argparse.ArgumentParser(
        description="Export one registered synthetic renderer fixture as JSON.",
    )
    parser.add_argument(
        "fixture_name",
        choices=_FIXTURE_NAMES,
        help="exact registered synthetic fixture name",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse one fixture name and write its exact JSON payload to stdout."""
    args = build_parser().parse_args(argv)
    print(serialize_renderer_fixture(cast(str, args.fixture_name)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
