"""Focused proof for the one-shot scene-native debugger snapshot adapter."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import scripts.dev.visual_debugger.static_renderer as static_renderer
from scripts.dev.visual_debugger.scenarios import get_scenario

from marl_battlegrounds.rendering import SceneRenderOptions
from marl_battlegrounds.rendering.scene import (
    BattlefieldSceneV1,
    VisualEventBatchV1,
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


def test_static_visual_vocabulary_export_has_exact_canvas_and_semantics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pyplot = _FakePyplot()
    render_calls: list[
        tuple[BattlefieldSceneV1, VisualEventBatchV1 | None, SceneRenderOptions | None]
    ] = []
    size_calls: list[tuple[float, float, bool]] = []
    save_calls: list[tuple[Path, dict[str, object]]] = []

    class _FakeFigure:
        def set_size_inches(
            self,
            width: float,
            height: float,
            *,
            forward: bool,
        ) -> None:
            size_calls.append((width, height, forward))

        def savefig(self, filename: Path, **kwargs: object) -> None:
            save_calls.append((filename, kwargs))

    figure = _FakeFigure()

    def capture_render(
        scene: BattlefieldSceneV1,
        *,
        event_batch: VisualEventBatchV1 | None = None,
        options: SceneRenderOptions | None = None,
    ) -> object:
        render_calls.append((scene, event_batch, options))
        return SimpleNamespace(figure=figure)

    monkeypatch.setattr(static_renderer, "_load_pyplot", lambda: pyplot)
    monkeypatch.setattr(static_renderer, "render_scene_geometry", capture_render)
    output = tmp_path / "static-renderer-visual-vocabulary-1440x900.png"

    exported = static_renderer.export_static_visual_vocabulary(output)

    assert exported == output
    assert size_calls == [(14.4, 9.0, True)]
    assert len(render_calls) == 1
    scene, event_batch, options = render_calls[0]
    assert scene.audience_badge.endswith("SYNTHETIC VISUAL VOCABULARY")
    assert event_batch is not None
    assert len(event_batch.events) == 12
    assert options == SceneRenderOptions(show_agent_ids=True)
    assert tuple(
        (field.source_global_slot, field.token_id) for field in scene.aura_fields
    ) == (
        (0, "mage_amplification"),
        (1, "warrior_mitigation"),
    )
    assert tuple(status.token_id for status in scene.agents[0].statuses) == (
        "stun_warrior_charge",
        "stun_hunter_trap",
        "stun_rogue_poison",
    )
    assert tuple(status.token_id for status in scene.agents[2].statuses) == (
        "slow_warrior_charge",
        "slow_hunter_basic",
        "slow_rogue_poison",
    )
    assert save_calls == [
        (
            output,
            {
                "dpi": 100,
                "edgecolor": "none",
                "facecolor": "#0B1020",
                "metadata": {
                    "Description": (
                        "Deterministic MARL-BattleGrounds static visual vocabulary "
                        "evidence"
                    ),
                    "Software": "MARL-BattleGrounds Visual Debugger and Analyzer",
                },
            },
        )
    ]
