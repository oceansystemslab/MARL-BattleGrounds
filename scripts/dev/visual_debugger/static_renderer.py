"""One-shot Matplotlib rendering for an authorized debugger reset snapshot."""

from importlib import import_module
from typing import Protocol, cast

from marl_battlegrounds.rendering import render_scene_geometry
from scripts.dev.visual_debugger.control import create_session
from scripts.dev.visual_debugger.scenarios import DebuggerScenario
from scripts.dev.visual_debugger.scene_adapter import (
    build_battlefield_scene,
    build_visual_event_batch,
)


class _PyplotLike(Protocol):
    def show(self) -> object: ...


def _load_pyplot() -> _PyplotLike:
    try:
        return cast(_PyplotLike, import_module("matplotlib.pyplot"))
    except ImportError as exc:
        msg = (
            "Matplotlib is required for static debugger snapshots. "
            "Run 'uv sync --extra viz --extra dev'."
        )
        raise ImportError(msg) from exc


def run_static_renderer(
    *,
    scenario: DebuggerScenario,
    seed: int,
    controlled_global_slot: int | None,
    verbose: bool,
    show_ranges: bool,
) -> int:
    """Render one reset scene without callbacks, a server, or a transition."""
    pyplot = _load_pyplot()
    session = create_session(
        scenario,
        seed=seed,
        controlled_global_slot=controlled_global_slot,
        show_ranges=show_ranges,
        verbose_logging=verbose,
    )
    scene = build_battlefield_scene(session, audience="researcher")
    event_batch = build_visual_event_batch(session, audience="researcher")
    render_scene_geometry(scene, event_batch=event_batch)
    pyplot.show()
    return 0


__all__ = ["run_static_renderer"]
