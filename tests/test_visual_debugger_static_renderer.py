"""Check one-shot debugger images built from scene records."""

from pathlib import Path

import pytest
import scripts.dev.visual_debugger.static_renderer as static_renderer
from scripts.dev.visual_debugger.scenarios import get_scenario
from tests.visual_debugger_fixtures import debugger_test_launch_specification

from marl_battlegrounds.rendering.scene import (
    BattlefieldSceneV2,
    VisualEventBatchV2,
)


class _FakePyplot:
    def __init__(self) -> None:
        self.show_calls = 0

    def show(self) -> None:
        self.show_calls += 1

    def close(self, figure: object) -> None:
        del figure


def test_static_renderer_builds_one_reset_scene_and_shows_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pyplot = _FakePyplot()
    rendered: list[tuple[BattlefieldSceneV2, VisualEventBatchV2 | None]] = []

    def capture_render(
        scene: BattlefieldSceneV2,
        *,
        event_batch: VisualEventBatchV2 | None = None,
    ) -> object:
        rendered.append((scene, event_batch))
        return object()

    monkeypatch.setattr(static_renderer, "_load_pyplot", lambda: pyplot)
    monkeypatch.setattr(static_renderer, "render_scene_geometry", capture_render)

    result = static_renderer.run_static_renderer(
        scenario=get_scenario("arena_5v5"),
        seed=9,
        evaluation_launch_specification=debugger_test_launch_specification(9),
        controlled_global_slot=1,
        verbose=False,
        show_ranges=False,
    )

    assert result == 0
    assert pyplot.show_calls == 1
    assert len(rendered) == 1
    scene, event_batch = rendered[0]
    assert scene.audience == "researcher"
    assert scene.selection is not None
    assert scene.selection.controlled_global_slot == 1
    assert scene.ranges == ()
    assert event_batch is None


def test_static_renderer_opens_current_replay_without_a_metric_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.evaluation_fixtures import current_captured_evaluation_trajectory
    from tests.test_evaluation_pov import (
        _runtime_provenance,  # pyright: ignore[reportPrivateUsage]
    )

    import marl_battlegrounds.viewer.static as static_renderer
    from marl_battlegrounds.evaluation.replay_io import save_replay
    from marl_battlegrounds.evaluation.replay_v3 import build_replay_v3

    trajectory = current_captured_evaluation_trajectory(transition_count=1)
    replay = build_replay_v3(
        trajectory.context,
        trajectory.frames,
        trajectory.transitions,
        runtime_provenance=_runtime_provenance(),
    )
    path = tmp_path / "current.marlbg-replay.json"
    save_replay(replay, path)
    pyplot = _FakePyplot()
    scenes: list[BattlefieldSceneV2] = []

    def capture_render(
        scene: BattlefieldSceneV2,
        *,
        event_batch: VisualEventBatchV2 | None = None,
    ) -> object:
        scenes.append(scene)
        assert event_batch is not None
        return object()

    monkeypatch.setattr(static_renderer, "_load_pyplot", lambda: pyplot)
    monkeypatch.setattr(static_renderer, "render_scene_geometry", capture_render)
    assert (
        static_renderer.run_static_replay_renderer(
            replay_path=path,
            frame_index=1,
            show_ranges=False,
        )
        == 0
    )
    assert pyplot.show_calls == 1
    assert len(scenes) == 1
    assert scenes[0].frame_index == 1
    assert scenes[0].audience == "researcher"
