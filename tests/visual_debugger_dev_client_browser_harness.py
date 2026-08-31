"""Run the real DevClient launcher with an isolated authoring artifact root."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Delegate to the public launcher after replacing only its store root."""
    options = _parser().parse_args(argv)

    from scripts.dev import debug_renderer
    from scripts.dev.visual_debugger import authoring_store

    store_type = authoring_store.DevAssetStore

    def isolated_store(repository_root: Path) -> object:
        return store_type(repository_root, artifact_root=options.artifact_root)

    authoring_store.DevAssetStore = isolated_store  # type: ignore[assignment]
    return debug_renderer.main(
        (
            "--no-open",
            "--port",
            str(options.port),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
