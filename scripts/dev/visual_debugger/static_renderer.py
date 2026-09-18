"""Display live developer scenarios and retain replay renderer imports.

Live session construction stays in this repository tool. Saved replay rendering
uses the package authority and imports Matplotlib only when requested.
"""

from typing import TYPE_CHECKING

from marl_battlegrounds.rendering import render_scene_geometry
from marl_battlegrounds.viewer.static import (
    _load_pyplot,  # pyright: ignore[reportPrivateUsage]
)
from marl_battlegrounds.viewer.static import (
    run_static_replay_artifact_renderer as run_static_replay_artifact_renderer,
)
from marl_battlegrounds.viewer.static import (
    run_static_replay_renderer as run_static_replay_renderer,
)

if TYPE_CHECKING:
    from scripts.dev.visual_debugger.evaluation_bridge import (
        DebuggerEvaluationLaunchSpecificationV1,
    )
    from scripts.dev.visual_debugger.scenarios import DebuggerScenario


def run_static_renderer(
    *,
    scenario: DebuggerScenario,
    seed: int,
    evaluation_launch_specification: DebuggerEvaluationLaunchSpecificationV1,
    controlled_global_slot: int | None,
    verbose: bool,
    show_ranges: bool,
) -> int:
    """Display a scenario's initial researcher view without taking a simulator step.

    Parameters
    ----------
    scenario : DebuggerScenario
        Scenario factory used to create the initial endpoint.
    seed : int
        Root random seed passed to the live-session constructor.
    evaluation_launch_specification : DebuggerEvaluationLaunchSpecificationV1
        Recording identity and capture contract for that endpoint.
    controlled_global_slot : int or None
        Global actor slot to select; None uses the scenario default.
    verbose : bool
        Whether the session enables verbose host logging.
    show_ranges : bool
        Whether the view includes interaction and observation ranges.

    Returns
    -------
    int
        Zero after the display call returns.

    Raises
    ------
    ImportError
        If Matplotlib is unavailable.

    Notes
    -----
    Creates a session and opens a Matplotlib window. The numerical session constructor
    owns scenario and seed validation. No transition or replay file is written.
    """
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
        team_b_controller="manual",
        execution_information_mode="no_shared_obs",
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
