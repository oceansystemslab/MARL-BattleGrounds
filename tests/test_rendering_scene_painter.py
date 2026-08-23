"""Focused semantic checks for the stateless canonical V2 scene painter."""

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from importlib import import_module
from typing import Protocol, cast

import pytest
from scripts.dev.visual_debugger.control import create_session, submit_next_script_frame
from scripts.dev.visual_debugger.scenarios import get_scenario
from tests.evaluation_fixtures import captured_evaluation_trajectory
from tests.visual_debugger_fixtures import debugger_test_launch_specification

from marl_battlegrounds.evaluation.metrics import EvaluationTransitionViewV1
from marl_battlegrounds.evaluation.pov import build_actor_pov_current_slice_v1
from marl_battlegrounds.rendering import (
    RenderResult,
    SceneRenderOptions,
    redraw_scene_geometry,
    render_scene_geometry,
)
from marl_battlegrounds.rendering.evaluation_adapter import (
    EvaluationScenePresentationStateV1,
    build_researcher_analyzer_projection_v2,
)
from marl_battlegrounds.rendering.pov_scene import (
    ActorPovAnalyzerProjectionV1,
    build_actor_pov_analyzer_projection_v1,
)
from marl_battlegrounds.rendering.scene import (
    BattlefieldSceneV2,
    ResearcherAnalyzerProjectionV2,
)
from marl_battlegrounds.rendering.scene_geometry import (
    _aura_color,  # pyright: ignore[reportPrivateUsage]
)
from marl_battlegrounds.rendering.vocabulary import class_token_from_id

_CANONICAL_EVENT_TYPES = frozenset(
    {
        "action_rejected",
        "ability_activated",
        "source_damage_output",
        "source_healing_output",
        "recipient_health_resolution",
        "combat_countdown_reset",
        "agent_left_combat",
        "health_regenerated",
        "cooldown_started",
        "cooldown_ready",
        "charge_phase_displacement",
        "ordinary_movement_phase_displacement",
        "agent_died",
        "lethal_damage_contribution",
        "status_aged_to_zero",
        "status_broken_by_damage",
        "status_applied",
        "status_refreshed_or_extended",
        "status_cleared_by_new_death",
        "spawn_shield_expired",
        "respawn_wave_occurred",
        "agent_respawned",
    }
)


class _ArtistLike(Protocol):
    def get_edgecolor(self) -> object: ...

    def get_facecolor(self) -> object: ...

    def get_gid(self) -> str | None: ...

    def get_linestyle(self) -> str: ...

    def get_linewidth(self) -> float: ...


class _TextLike(_ArtistLike, Protocol):
    def get_bbox_patch(self) -> _ArtistLike: ...

    def get_position(self) -> tuple[float, float]: ...

    def get_text(self) -> str: ...


class _AxesLike(Protocol):
    patches: list[_ArtistLike]
    lines: list[_ArtistLike]
    texts: list[_TextLike]


@dataclass(frozen=True, slots=True)
class _ProductionPainterCases:
    researcher: ResearcherAnalyzerProjectionV2
    vocabulary: tuple[ResearcherAnalyzerProjectionV2, ...]
    pov: ActorPovAnalyzerProjectionV1


@pytest.fixture(scope="module")
def production_painter_cases() -> _ProductionPainterCases:
    def scenario_projections(
        name: str,
        selected_frame_indexes: frozenset[int],
    ) -> tuple[ResearcherAnalyzerProjectionV2, ...]:
        scenario = get_scenario(name)
        session = create_session(
            scenario,
            seed=0,
            evaluation_launch_specification=debugger_test_launch_specification(),
            controlled_global_slot=0,
            show_ranges=True,
            verbose_logging=False,
        )
        projections: list[ResearcherAnalyzerProjectionV2] = []
        for frame_index in range(len(scenario.frames)):
            session = submit_next_script_frame(session)
            if frame_index not in selected_frame_indexes:
                continue
            projections.append(
                build_researcher_analyzer_projection_v2(
                    session.evaluation_context,
                    session.current_evaluation_frame,
                    transition_view=session.incoming_evaluation_view,
                    presentation=EvaluationScenePresentationStateV1(
                        controlled_global_slot=0,
                        selected_global_slot=5,
                        show_ranges=True,
                    ),
                    status_source_evidence_state=(session.status_source_evidence_state),
                )
            )
        if len(projections) != len(selected_frame_indexes):
            raise AssertionError("selected production scenario frame is unavailable")
        return tuple(projections)

    moving = scenario_projections("moving_basic_crossfire", frozenset({0}))
    vocabulary = (
        *moving,
        *scenario_projections("recovery_refresh_cycle", frozenset({0, 1})),
        *scenario_projections("charge_convergence", frozenset({0})),
        *scenario_projections("death_respawn_cycle", frozenset({0, 2, 5})),
    )

    trajectory = captured_evaluation_trajectory(
        transition_count=1,
        expected_horizon=1,
    )
    incoming = EvaluationTransitionViewV1(
        context=trajectory.context,
        start_frame=trajectory.frames[0],
        transition=trajectory.transitions[0],
        successor_frame=trajectory.frames[1],
    )
    pov = build_actor_pov_analyzer_projection_v1(
        build_actor_pov_current_slice_v1(
            trajectory.context,
            trajectory.frames[1],
            global_slot=0,
            incoming_transition_view=incoming,
        )
    )
    return _ProductionPainterCases(
        researcher=moving[0],
        vocabulary=vocabulary,
        pov=pov,
    )


def _skip_if_matplotlib_unavailable() -> None:
    pytest.importorskip("matplotlib.pyplot")


def _close(result: RenderResult) -> None:
    pyplot = import_module("matplotlib.pyplot")
    cast(Callable[[object], object], pyplot.close)(result.figure)


def _artists(result: RenderResult) -> tuple[_ArtistLike, ...]:
    axes = cast(_AxesLike, result.axes)
    return (*axes.patches, *axes.lines, *axes.texts)


def _gids(result: RenderResult) -> tuple[str, ...]:
    return tuple(
        gid for artist in _artists(result) if (gid := artist.get_gid()) is not None
    )


def _texts(result: RenderResult) -> tuple[_TextLike, ...]:
    return tuple(cast(_AxesLike, result.axes).texts)


def _render_projection(
    projection: object,
    *,
    options: SceneRenderOptions | None = None,
) -> RenderResult:
    _skip_if_matplotlib_unavailable()
    if type(projection) is not ResearcherAnalyzerProjectionV2:
        raise ValueError("the static painter accepts researcher V2 projections only")
    return render_scene_geometry(
        projection.scene,
        event_batch=projection.incoming_events,
        options=options,
    )


def test_production_v2_scene_preserves_agents_statuses_and_event_multiplicity(
    production_painter_cases: _ProductionPainterCases,
) -> None:
    projection = production_painter_cases.researcher
    result = _render_projection(projection)
    try:
        scene = projection.scene
        assert type(scene) is BattlefieldSceneV2
        gids = _gids(result)
        assert {gid for gid in gids if gid.endswith(":body")} == {
            f"scene:v2:agent:{agent.public_agent_id}:body" for agent in scene.agents
        }
        assert any(agent.statuses for agent in scene.agents)
        for agent in scene.agents:
            assert tuple(
                gid
                for gid in gids
                if gid.startswith(f"scene:v2:agent:{agent.public_agent_id}:status:")
            ) == tuple(
                f"scene:v2:agent:{agent.public_agent_id}:status:{status.status_id}"
                for status in agent.statuses
            )
        event_batch = projection.incoming_events
        assert event_batch is not None
        event_gids = tuple(gid for gid in gids if gid.startswith("scene:v2:event:"))
        assert event_gids == tuple(
            f"scene:v2:event:{event.event_id}" for event in event_batch.events
        )
        assert len(event_gids) == len(set(event_gids)) == len(event_batch.events)
        assert len(event_gids) >= 50
        assert "scene:audience:researcher" in gids
    finally:
        _close(result)


def test_production_event_batch_draws_every_event_row_exactly_once(
    production_painter_cases: _ProductionPainterCases,
) -> None:
    rendered_event_types: set[str] = set()
    for projection in production_painter_cases.vocabulary:
        result = _render_projection(projection)
        try:
            event_batch = projection.incoming_events
            assert event_batch is not None
            assert event_batch.events
            event_gids = tuple(
                gid for gid in _gids(result) if gid.startswith("scene:v2:event:")
            )
            assert event_gids == tuple(
                f"scene:v2:event:{event.event_id}" for event in event_batch.events
            )
            assert len(event_gids) == len(event_batch.events)
            assert len(event_gids) == len(set(event_gids))
            labels = tuple(
                artist.get_text()
                for artist in _texts(result)
                if (artist.get_gid() or "").startswith("scene:v2:event:")
            )
            assert labels == tuple(
                event.event_type.replace("_", " ").upper()
                for event in event_batch.events
            )
            rendered_event_types.update(
                event.event_type for event in event_batch.events
            )
        finally:
            _close(result)
    assert rendered_event_types == _CANONICAL_EVENT_TYPES


def test_visual_vocabulary_static_evidence_keeps_v2_scene_grammar(
    production_painter_cases: _ProductionPainterCases,
) -> None:
    projection = production_painter_cases.researcher
    result = _render_projection(
        projection,
        options=SceneRenderOptions(show_agent_ids=True),
    )
    try:
        scene = projection.scene
        assert type(scene) is BattlefieldSceneV2
        gids = _gids(result)
        by_gid = {
            gid: artist
            for artist in _artists(result)
            if (gid := artist.get_gid()) is not None
        }
        assert tuple(
            cast(
                _TextLike, by_gid[f"scene:v2:agent:{agent.public_agent_id}:class"]
            ).get_text()
            for agent in scene.agents
        ) == tuple(
            class_token_from_id(agent.class_id).fallback for agent in scene.agents
        )
        assert "scene:v2:selection:controlled:0" in gids
        assert "scene:v2:selection:target:5" in gids
        assert {
            f"scene:v2:range:{row.global_slot}:{row.kind}"
            for row in scene.ranges
            if row.radius > 0.0
        }.issubset(gids)
        assert {
            f"scene:v2:aura:{row.source_public_agent_id}:{row.aura_id}"
            for row in scene.aura_fields
        }.issubset(gids)
        event_batch = projection.incoming_events
        assert event_batch is not None
        assert len([gid for gid in gids if gid.startswith("scene:v2:event:")]) == len(
            event_batch.events
        )
    finally:
        _close(result)


def test_production_event_labels_keep_each_canonical_event_without_joining(
    production_painter_cases: _ProductionPainterCases,
) -> None:
    projection = production_painter_cases.researcher
    result = _render_projection(projection)
    try:
        event_batch = projection.incoming_events
        assert event_batch is not None
        event_labels = {
            artist.get_gid(): artist.get_text()
            for artist in _texts(result)
            if (artist.get_gid() or "").startswith("scene:v2:event:")
        }
        assert tuple(event_labels) == tuple(
            f"scene:v2:event:{event.event_id}" for event in event_batch.events
        )
        assert tuple(event_labels.values()) == tuple(
            event.event_type.replace("_", " ").upper() for event in event_batch.events
        )
        assert len(event_labels) == len(event_batch.events)
    finally:
        _close(result)


def test_pov_projection_is_not_routed_through_privileged_static_painter(
    production_painter_cases: _ProductionPainterCases,
) -> None:
    projection = production_painter_cases.pov
    assert "AGENT POV" in projection.scene.audience_badge
    with pytest.raises(ValueError, match="researcher V2 projections"):
        _render_projection(projection)


def test_redraw_reuses_result_and_recreates_deterministic_semantic_ids(
    production_painter_cases: _ProductionPainterCases,
) -> None:
    projection = production_painter_cases.researcher
    result = _render_projection(projection)
    try:
        assert type(projection.scene) is BattlefieldSceneV2
        initial_gids = _gids(result)
        redrawn = redraw_scene_geometry(
            projection.scene,
            result,
            event_batch=projection.incoming_events,
        )
        assert redrawn is result
        assert _gids(redrawn) == initial_gids
    finally:
        _close(result)


def test_render_options_remove_optional_clutter_without_hiding_bodies(
    production_painter_cases: _ProductionPainterCases,
) -> None:
    projection = production_painter_cases.researcher
    result = _render_projection(
        projection,
        options=SceneRenderOptions(
            show_agent_ids=False,
            show_ranges=False,
            show_statuses=False,
            show_modifiers=False,
            show_observer_visibility=False,
            show_events=False,
        ),
    )
    try:
        assert type(projection.scene) is BattlefieldSceneV2
        gids = _gids(result)
        absent_prefixes: Iterable[str] = (
            "scene:v2:range:",
            "scene:v2:event:",
        )
        assert all(
            not gid.startswith(prefix) for prefix in absent_prefixes for gid in gids
        )
        assert all(":status:" not in gid for gid in gids)
        assert all(":aura:" not in gid for gid in gids if ":agent:" in gid)
        assert {gid for gid in gids if gid.endswith(":body")} == {
            f"scene:v2:agent:{agent.public_agent_id}:body"
            for agent in projection.scene.agents
        }
    finally:
        _close(result)


def test_human_visible_v2_float_labels_never_exceed_two_decimals(
    production_painter_cases: _ProductionPainterCases,
) -> None:
    _skip_if_matplotlib_unavailable()
    projection = production_painter_cases.researcher
    assert type(projection.scene) is BattlefieldSceneV2
    first_agent = projection.scene.agents[0]
    assert first_agent.aura_modifiers
    first_modifier = replace(first_agent.aura_modifiers[0], multiplier=1.234567)
    scene = replace(
        projection.scene,
        agents=(
            replace(
                first_agent,
                aura_modifiers=(first_modifier, *first_agent.aura_modifiers[1:]),
            ),
            *projection.scene.agents[1:],
        ),
    )
    result = render_scene_geometry(scene, event_batch=projection.incoming_events)
    try:
        visible_text = {
            artist.get_gid(): artist.get_text() for artist in _texts(result)
        }
        modifier_gid = (
            f"scene:v2:agent:{first_agent.public_agent_id}:aura:"
            f"{first_modifier.aura_id}"
        )
        assert visible_text[modifier_gid].endswith("x1.23")
        assert all(
            re.search(r"\d+\.\d{3,}", text) is None for text in visible_text.values()
        )
    finally:
        _close(result)


def test_unknown_legacy_aura_token_uses_neutral_color() -> None:
    assert _aura_color("future_aura") == "#9AA7B8"
