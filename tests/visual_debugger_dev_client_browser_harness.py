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
    parser.add_argument("--seed-map-count", type=int, default=0)
    parser.add_argument("--seed-scenario-count", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Delegate to the public launcher after replacing only its store root."""
    options = _parser().parse_args(argv)

    from scripts.dev import debug_renderer
    from scripts.dev.visual_debugger import authoring_store
    from scripts.dev.visual_debugger.authoring_models import (
        new_map_draft,
        new_scenario_draft,
    )

    store_type = authoring_store.DevAssetStore
    for label, count in (
        ("seed-map-count", options.seed_map_count),
        ("seed-scenario-count", options.seed_scenario_count),
    ):
        if count < 0:
            raise ValueError(f"{label} must be nonnegative")

    if options.seed_map_count or options.seed_scenario_count:
        seed_store = store_type(
            _REPOSITORY_ROOT,
            artifact_root=options.artifact_root,
        )
        for index in range(options.seed_map_count):
            draft = new_map_draft(f"seed_map_{index}")
            draft = draft.model_copy(
                update={
                    "content": draft.content.model_copy(
                        update={"name": f"Seed map {index}"}
                    )
                }
            )
            seed_store.save_draft(draft, expected_revision=0)
        for index in range(options.seed_scenario_count):
            draft = new_scenario_draft(f"seed_scenario_{index}")
            draft = draft.model_copy(
                update={
                    "content": draft.content.model_copy(
                        update={"name": f"Seed scenario {index}"}
                    )
                }
            )
            seed_store.save_draft(draft, expected_revision=0)

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
