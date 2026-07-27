"""One-shot Matplotlib rendering for an authorized debugger reset snapshot."""

from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

from marl_battlegrounds.rendering import SceneRenderOptions, render_scene_geometry
from scripts.dev.visual_debugger.control import create_session
from scripts.dev.visual_debugger.renderer_fixtures import get_renderer_fixture
from scripts.dev.visual_debugger.scenarios import DebuggerScenario
from scripts.dev.visual_debugger.scene_adapter import (
    build_battlefield_scene,
    build_visual_event_batch,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
STATIC_VISUAL_VOCABULARY_PATH = (
    _REPOSITORY_ROOT
    / "web"
    / "visual_debugger"
    / "e2e"
    / "visual-regression.spec.js-snapshots"
    / "static-renderer-visual-vocabulary-1440x900.png"
)


class _PyplotLike(Protocol):
    def show(self) -> object: ...

    def close(self, figure: object) -> object: ...


class _FigureLike(Protocol):
    def set_size_inches(
        self,
        width: float,
        height: float,
        *,
        forward: bool,
    ) -> object: ...

    def savefig(self, filename: Path, **kwargs: object) -> object: ...


def _load_pyplot() -> _PyplotLike:
    try:
        return cast(_PyplotLike, import_module("matplotlib.pyplot"))
    except ImportError as exc:
        msg = (
            "Matplotlib is required for static Visual Debugger and Analyzer "
            "snapshots. "
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


def export_static_visual_vocabulary(
    output_path: Path = STATIC_VISUAL_VOCABULARY_PATH,
) -> Path:
    """Export the canonical static vocabulary evidence at exactly 1440x900."""
    fixture = get_renderer_fixture("visual_vocabulary")
    result = render_scene_geometry(
        fixture.scene,
        event_batch=fixture.event_batch,
        options=SceneRenderOptions(show_agent_ids=True),
    )
    figure = cast(_FigureLike, result.figure)
    figure.set_size_inches(14.4, 9.0, forward=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        dpi=100,
        edgecolor="none",
        facecolor="#0B1020",
        metadata={
            "Description": (
                "Deterministic MARL-BattleGrounds static visual vocabulary evidence"
            ),
            "Software": "MARL-BattleGrounds Visual Debugger and Analyzer",
        },
    )
    _load_pyplot().close(result.figure)
    return output_path


__all__ = [
    "STATIC_VISUAL_VOCABULARY_PATH",
    "export_static_visual_vocabulary",
    "run_static_renderer",
]
