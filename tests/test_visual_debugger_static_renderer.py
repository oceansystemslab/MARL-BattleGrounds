"""Focused proof for the one-shot scene-native debugger snapshot adapter."""

import pytest
import scripts.dev.visual_debugger.static_renderer as static_renderer
from scripts.dev.visual_debugger.scenarios import get_scenario

from marl_battlegrounds.rendering.scene import (
    BattlefieldSceneV1,
    VisualEventBatchV1,
)


class _FakePyplot:
    def __init__(self) -> None:
        self.show_calls = 0

    def show(self) -> None:
        self.show_calls += 1


def test_static_renderer_builds_one_reset_scene_and_shows_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pyplot = _FakePyplot()
    rendered: list[tuple[BattlefieldSceneV1, VisualEventBatchV1 | None]] = []

    def capture_render(
        scene: BattlefieldSceneV1,
        *,
        event_batch: VisualEventBatchV1 | None = None,
    ) -> object:
        rendered.append((scene, event_batch))
        return object()

    monkeypatch.setattr(static_renderer, "_load_pyplot", lambda: pyplot)
    monkeypatch.setattr(static_renderer, "render_scene_geometry", capture_render)

    result = static_renderer.run_static_renderer(
        scenario=get_scenario("arena_5v5"),
        seed=9,
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
