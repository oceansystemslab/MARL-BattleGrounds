"""One-shot Matplotlib rendering for an authorized debugger reset snapshot."""

from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from marl_battlegrounds.rendering import SceneRenderOptions, render_scene_geometry

if TYPE_CHECKING:
    from scripts.dev.visual_debugger.evaluation_bridge import (
        DebuggerEvaluationLaunchSpecificationV1,
    )
    from scripts.dev.visual_debugger.scenarios import DebuggerScenario

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
    evaluation_launch_specification: DebuggerEvaluationLaunchSpecificationV1,
    controlled_global_slot: int | None,
    verbose: bool,
    show_ranges: bool,
) -> int:
    """Render one reset scene without callbacks, a server, or a transition."""
    from marl_battlegrounds.rendering.evaluation_adapter import (
        EvaluationScenePresentationStateV1,
        build_researcher_analyzer_projection_v2,
    )
    from scripts.dev.visual_debugger.control import create_session

    pyplot = _load_pyplot()
    session = create_session(
        scenario,
        seed=seed,
        evaluation_launch_specification=evaluation_launch_specification,
        controlled_global_slot=controlled_global_slot,
        show_ranges=show_ranges,
        verbose_logging=verbose,
    )
    projection = build_researcher_analyzer_projection_v2(
        session.evaluation_context,
        session.current_evaluation_frame,
        transition_view=session.incoming_evaluation_view,
        presentation=EvaluationScenePresentationStateV1(
            controlled_global_slot=session.controlled_global_slot,
            selected_global_slot=session.controlled_global_slot,
            show_ranges=show_ranges,
        ),
        status_source_evidence_state=session.status_source_evidence_state,
    )
    render_scene_geometry(
        projection.scene,
        event_batch=projection.incoming_events,
    )
    pyplot.show()
    return 0


def run_static_replay_renderer(
    *,
    replay_path: Path,
    frame_index: int,
    show_ranges: bool,
) -> int:
    """Validate, project, and render one canonical replay frame offline."""
    from marl_battlegrounds.evaluation.metrics import EvaluationTransitionViewV1
    from marl_battlegrounds.evaluation.replay_io import (
        ReplayLoadError,
        load_replay_artifact_v1,
    )
    from marl_battlegrounds.rendering.evaluation_adapter import (
        EvaluationScenePresentationStateV1,
        build_researcher_analyzer_projection_v2,
        build_status_source_evidence_index_v2,
    )

    if type(frame_index) is not int:
        raise ValueError("replay frame index must be a Python integer.")
    try:
        replay = load_replay_artifact_v1(replay_path)
    except ReplayLoadError as exc:
        raise ValueError(f"Replay could not be loaded: {exc}") from exc
    if not 0 <= frame_index < len(replay.frames):
        raise ValueError(
            f"replay frame index must be in [0, {len(replay.frames) - 1}]."
        )
    frame = replay.frames[frame_index]
    transition_view = (
        None
        if frame_index == 0
        else EvaluationTransitionViewV1(
            context=replay.header.context,
            start_frame=replay.frames[frame_index - 1],
            transition=replay.transitions[frame_index - 1],
            successor_frame=frame,
        )
    )
    status_index = build_status_source_evidence_index_v2(
        replay.header.context,
        replay.frames,
        replay.transitions,
    )
    projection = build_researcher_analyzer_projection_v2(
        replay.header.context,
        frame,
        transition_view=transition_view,
        presentation=EvaluationScenePresentationStateV1(show_ranges=show_ranges),
        status_source_evidence_state=status_index.state_for_frame(frame_index),
    )
    pyplot = _load_pyplot()
    render_scene_geometry(
        projection.scene,
        event_batch=projection.incoming_events,
    )
    pyplot.show()
    return 0


def export_static_visual_vocabulary(
    output_path: Path = STATIC_VISUAL_VOCABULARY_PATH,
) -> Path:
    """Export the canonical static vocabulary evidence at exactly 1440x900."""
    from marl_battlegrounds.rendering.scene import BattlefieldSceneV2
    from scripts.dev.visual_debugger.renderer_fixtures import get_renderer_fixture

    fixture = get_renderer_fixture("visual_vocabulary")
    if (
        fixture.audience != "researcher"
        or type(fixture.scene) is not BattlefieldSceneV2
    ):
        raise ValueError(
            "the static visual vocabulary must remain a researcher V2 fixture"
        )
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
    "run_static_replay_renderer",
]
