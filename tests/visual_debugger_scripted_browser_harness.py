"""Run one registered scripted DebuggerService for browser-only causal testing."""

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
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--scenario", default="aura_crossfire")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Serve a registered script without adding public launcher hooks."""
    options = _parser().parse_args(argv)

    from scripts.dev.visual_debugger.control import create_session
    from scripts.dev.visual_debugger.scenarios import get_scenario
    from scripts.dev.visual_debugger.server import serve_browser_debugger
    from scripts.dev.visual_debugger.service import DebuggerService
    from tests.visual_debugger_fixtures import (
        debugger_test_launch_specification,
    )

    session = create_session(
        get_scenario(options.scenario),
        seed=0,
        evaluation_launch_specification=debugger_test_launch_specification(),
        controlled_global_slot=(
            0 if options.scenario == "death_respawn_cycle" else None
        ),
        show_ranges=True,
        verbose_logging=False,
    )
    service = DebuggerService(
        session,
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
        session_id="scripted-browser-causal-test",
    )
    return serve_browser_debugger(
        service,
        asset_root=_REPOSITORY_ROOT / "web" / "visual_debugger",
        port=options.port,
        open_browser=False,
    )


if __name__ == "__main__":
    raise SystemExit(main())
