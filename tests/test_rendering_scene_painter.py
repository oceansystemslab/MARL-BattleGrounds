"""Focused semantic checks for the stateless canonical V2 scene painter."""

import re
from collections.abc import Callable, Iterable
from dataclasses import replace
from importlib import import_module
from typing import Protocol, cast

import pytest
from scripts.dev.visual_debugger.renderer_fixtures import (
    RendererFixtureV2,
    get_renderer_fixture,
)

from marl_battlegrounds.rendering import (
    RenderResult,
    SceneRenderOptions,
    redraw_scene_geometry,
    render_scene_geometry,
)
from marl_battlegrounds.rendering.scene import BattlefieldSceneV2
from marl_battlegrounds.rendering.scene_geometry import (
    _aura_color,  # pyright: ignore[reportPrivateUsage]
)
from marl_battlegrounds.rendering.vocabulary import class_token_from_id


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


def _render_fixture(
    name: str,
    *,
    options: SceneRenderOptions | None = None,
) -> tuple[RendererFixtureV2, RenderResult]:
    _skip_if_matplotlib_unavailable()
    fixture = get_renderer_fixture(name)
    if (
        fixture.audience != "researcher"
        or type(fixture.scene) is not BattlefieldSceneV2
    ):
        raise ValueError("the static painter accepts researcher V2 fixtures only")
    result = render_scene_geometry(
        fixture.scene,
        event_batch=fixture.event_batch,
        options=options,
    )
    return fixture, result


def test_crowded_v2_scene_preserves_agents_statuses_and_event_multiplicity() -> None:
    fixture, result = _render_fixture("crowded_teamfight")
    try:
        assert type(fixture.scene) is BattlefieldSceneV2
        gids = _gids(result)
        assert {gid for gid in gids if gid.endswith(":body")} == {
            f"scene:v2:agent:{agent.public_agent_id}:body"
            for agent in fixture.scene.agents
        }
        first_agent = fixture.scene.agents[0]
        assert tuple(
            gid for gid in gids if gid.startswith("scene:v2:agent:0:status:")
        ) == tuple(
            f"scene:v2:agent:0:status:{status.status_id}"
            for status in first_agent.statuses
        )
        assert fixture.event_batch is not None
        event_gids = tuple(gid for gid in gids if gid.startswith("scene:v2:event:"))
        assert event_gids == tuple(
            f"scene:v2:event:{event.event_id}" for event in fixture.event_batch.events
        )
        assert len(event_gids) == len(set(event_gids)) == 32
        assert "scene:audience:researcher" in gids
    finally:
        _close(result)


def test_canonical_event_vocabulary_draws_all_21_events_exactly_once() -> None:
    fixture, result = _render_fixture("canonical_event_vocabulary")
    try:
        assert fixture.event_batch is not None
        event_gids = tuple(
            gid for gid in _gids(result) if gid.startswith("scene:v2:event:")
        )
        assert event_gids == tuple(
            f"scene:v2:event:{event.event_id}" for event in fixture.event_batch.events
        )
        assert len(event_gids) == 21
        labels = tuple(
            artist.get_text()
            for artist in _texts(result)
            if (artist.get_gid() or "").startswith("scene:v2:event:")
        )
        assert labels == tuple(
            event.event_type.replace("_", " ").upper()
            for event in fixture.event_batch.events
        )
    finally:
        _close(result)


def test_visual_vocabulary_static_evidence_keeps_v2_scene_grammar() -> None:
    fixture, result = _render_fixture(
        "visual_vocabulary",
        options=SceneRenderOptions(show_agent_ids=True),
    )
    try:
        assert type(fixture.scene) is BattlefieldSceneV2
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
            for agent in fixture.scene.agents
        ) == tuple(
            class_token_from_id(agent.class_id).fallback
            for agent in fixture.scene.agents
        )
        assert "scene:v2:selection:controlled:0" in gids
        assert "scene:v2:selection:target:5" in gids
        assert {
            "scene:v2:aura:0:mage_damage_amplification",
            "scene:v2:aura:1:warrior_damage_mitigation",
        }.issubset(gids)
        assert {
            f"scene:v2:range:{row.global_slot}:{row.kind}"
            for row in fixture.scene.ranges
        }.issubset(gids)
        assert fixture.event_batch is not None
        assert len([gid for gid in gids if gid.startswith("scene:v2:event:")]) == len(
            fixture.event_batch.events
        )
    finally:
        _close(result)


def test_mixed_zero_net_keeps_each_canonical_event_without_joining() -> None:
    fixture, result = _render_fixture("mixed_net_zero")
    try:
        assert fixture.event_batch is not None
        event_labels = {
            artist.get_gid(): artist.get_text()
            for artist in _texts(result)
            if (artist.get_gid() or "").startswith("scene:v2:event:")
        }
        assert tuple(event_labels) == tuple(
            f"scene:v2:event:{event.event_id}" for event in fixture.event_batch.events
        )
        assert tuple(event_labels.values()) == (
            "ABILITY ACTIVATED",
            "ABILITY ACTIVATED",
            "RECIPIENT HEALTH RESOLUTION",
        )
    finally:
        _close(result)


def test_pov_fixture_is_not_routed_through_privileged_static_painter() -> None:
    fixture = get_renderer_fixture("pov_redaction")
    assert fixture.audience == "agent_pov"
    with pytest.raises(ValueError, match="researcher V2"):
        _render_fixture("pov_redaction")


def test_redraw_reuses_result_and_recreates_deterministic_semantic_ids() -> None:
    fixture, result = _render_fixture("mixed_net_zero")
    try:
        assert type(fixture.scene) is BattlefieldSceneV2
        initial_gids = _gids(result)
        redrawn = redraw_scene_geometry(
            fixture.scene,
            result,
            event_batch=fixture.event_batch,
        )
        assert redrawn is result
        assert _gids(redrawn) == initial_gids
    finally:
        _close(result)


def test_render_options_remove_optional_clutter_without_hiding_bodies() -> None:
    fixture, result = _render_fixture(
        "crowded_teamfight",
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
        assert type(fixture.scene) is BattlefieldSceneV2
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
            for agent in fixture.scene.agents
        }
    finally:
        _close(result)


def test_human_visible_v2_float_labels_never_exceed_two_decimals() -> None:
    _skip_if_matplotlib_unavailable()
    fixture = get_renderer_fixture("crowded_teamfight")
    assert type(fixture.scene) is BattlefieldSceneV2
    first_agent = fixture.scene.agents[0]
    first_modifier = replace(first_agent.aura_modifiers[0], multiplier=1.234567)
    scene = replace(
        fixture.scene,
        agents=(
            replace(
                first_agent,
                aura_modifiers=(first_modifier, *first_agent.aura_modifiers[1:]),
            ),
            *fixture.scene.agents[1:],
        ),
    )
    result = render_scene_geometry(scene, event_batch=fixture.event_batch)
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
