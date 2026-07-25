"""Focused semantic checks for the stateless scene-native Matplotlib painter."""

from collections.abc import Callable, Iterable
from dataclasses import replace
from importlib import import_module
from typing import Protocol, cast

import pytest
from scripts.dev.visual_debugger.renderer_fixtures import (
    RendererFixtureV1,
    get_renderer_fixture,
)

from marl_battlegrounds.rendering import (
    RenderResult,
    SceneRenderOptions,
    redraw_scene_geometry,
    render_scene_geometry,
)
from marl_battlegrounds.rendering.scene import AcceptedActivationEventV1
from marl_battlegrounds.rendering.scene_geometry import (
    _aura_color,  # pyright: ignore[reportPrivateUsage]
)


class _ArtistLike(Protocol):
    def get_gid(self) -> str | None: ...

    def get_linestyle(self) -> str: ...


class _TextLike(_ArtistLike, Protocol):
    xy: tuple[float, float]

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


def _texts_with_gid_prefix(
    result: RenderResult,
    prefix: str,
) -> tuple[_TextLike, ...]:
    return tuple(
        artist
        for artist in _texts(result)
        if (artist.get_gid() or "").startswith(prefix)
    )


def _render_fixture(
    name: str,
    *,
    options: SceneRenderOptions | None = None,
) -> tuple[RendererFixtureV1, RenderResult]:
    _skip_if_matplotlib_unavailable()
    fixture = get_renderer_fixture(name)
    result = render_scene_geometry(
        fixture.scene,
        event_batch=fixture.event_batch,
        options=options,
    )
    return fixture, result


def test_crowded_scene_preserves_agents_status_order_and_event_multiplicity() -> None:
    fixture, result = _render_fixture("crowded_teamfight")
    try:
        gids = _gids(result)
        body_gids = {
            gid
            for gid in gids
            if gid.startswith("scene:agent:") and gid.endswith(":body")
        }
        assert body_gids == {
            f"scene:agent:{agent.global_slot}:body" for agent in fixture.scene.agents
        }

        first_agent = fixture.scene.agents[0]
        assert tuple(
            gid for gid in gids if gid.startswith("scene:agent:0:status:")
        ) == tuple(
            f"scene:agent:0:status:{status.token_id}" for status in first_agent.statuses
        )

        assert fixture.event_batch is not None
        event_gids = tuple(gid for gid in gids if gid.startswith("event:"))
        assert event_gids == tuple(
            f"event:{event.event_type}:{event.event_id}"
            for event in fixture.event_batch.events
        )
        assert "scene:audience:researcher" in gids
        assert "scene:pending:0:7:lane:1" in gids
        assert {
            "scene:legality:0:7:lane:0",
            "scene:legality:0:7:lane:1",
        }.issubset(gids)

        charge = next(
            event
            for event in fixture.event_batch.events
            if type(event) is AcceptedActivationEventV1
            and event.token_id == "warrior_charge"
        )
        charge_artist = next(
            artist
            for artist in _texts(result)
            if artist.get_gid() == f"event:accepted_activation:{charge.event_id}"
        )
        assert charge_artist.get_position() == charge.source_anchor
        assert charge_artist.xy == charge.target_anchor
        successor = next(
            agent
            for agent in fixture.scene.agents
            if agent.global_slot == charge.source_global_slot
        )
        assert charge.source_anchor != successor.position
    finally:
        _close(result)


def test_route_collision_keeps_one_primary_artist_per_accepted_activation() -> None:
    fixture, result = _render_fixture("route_collision")
    try:
        assert fixture.event_batch is not None
        accepted = tuple(
            event
            for event in fixture.event_batch.events
            if type(event) is AcceptedActivationEventV1
        )
        primary_event_gids = tuple(
            gid for gid in _gids(result) if gid.startswith("event:accepted_activation:")
        )

        assert primary_event_gids == tuple(
            f"event:accepted_activation:{event.event_id}" for event in accepted
        )
        assert len(primary_event_gids) == len(set(primary_event_gids))

        for event in accepted:
            artist = next(
                candidate
                for candidate in _texts(result)
                if candidate.get_gid() == f"event:accepted_activation:{event.event_id}"
            )
            assert artist.get_position() == event.source_anchor
            assert artist.xy == event.target_anchor
    finally:
        _close(result)


def test_mixed_zero_net_separates_activation_intent_from_recipient_outcome() -> None:
    fixture, result = _render_fixture("mixed_net_zero")
    try:
        assert fixture.event_batch is not None
        event_text = {
            artist.get_gid(): artist.get_text()
            for artist in _texts_with_gid_prefix(result, "event:")
        }
        activation_ids = tuple(
            event.event_id
            for event in fixture.event_batch.events
            if type(event) is AcceptedActivationEventV1
        )
        for event_id in activation_ids:
            label = event_text[f"event:accepted_activation:{event_id}"]
            assert "NET" not in label
            assert "HP unchanged" not in label

        net_text = event_text["event:net_health:synthetic:mixed_net_zero:net-health-0"]
        assert net_text == "HP unchanged"
    finally:
        _close(result)


def test_agent_pov_renders_only_already_authorized_scene_and_event_facts() -> None:
    fixture, result = _render_fixture(
        "pov_redaction",
        options=SceneRenderOptions(show_agent_ids=True),
    )
    try:
        gids = _gids(result)
        text = tuple(artist.get_text() for artist in _texts(result))

        assert {gid for gid in gids if gid.endswith(":body")} == {
            "scene:agent:0:body",
            "scene:agent:1:body",
        }
        assert all("scene:agent:5:" not in gid for gid in gids)
        assert all("id_5" not in value for value in text)
        assert any("redacted" in value for value in text)
        assert all("PRIVILEGED" not in value for value in text)
        assert fixture.event_batch is not None
        assert tuple(gid for gid in gids if gid.startswith("event:")) == tuple(
            f"event:{event.event_type}:{event.event_id}"
            for event in fixture.event_batch.events
        )
        assert "scene:audience:agent_pov" in gids
        assert all(not gid.startswith("scene:visibility:") for gid in gids)
    finally:
        _close(result)


def test_redraw_reuses_result_and_recreates_deterministic_semantic_ids() -> None:
    fixture, result = _render_fixture("mixed_net_zero")
    try:
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


def test_render_options_remove_optional_debug_clutter_without_hiding_truth() -> None:
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
        gids = _gids(result)
        absent_prefixes: Iterable[str] = (
            "scene:range:",
            "scene:visibility:",
            "event:",
        )
        assert all(
            not gid.startswith(prefix) for prefix in absent_prefixes for gid in gids
        )
        assert {gid for gid in gids if gid.endswith(":identity")} == {
            "scene:agent:0:identity",
            "scene:agent:7:identity",
        }
        assert all(":status:" not in gid for gid in gids)
        assert all(":modifier:" not in gid for gid in gids)
        assert {gid for gid in gids if gid.endswith(":body")} == {
            f"scene:agent:{agent.global_slot}:body" for agent in fixture.scene.agents
        }
        assert "scene:audience:researcher" in gids
    finally:
        _close(result)


def test_unknown_team_uses_neutral_pattern_instead_of_team_b_styling() -> None:
    _skip_if_matplotlib_unavailable()
    fixture = get_renderer_fixture("mixed_net_zero")
    unknown_agent = replace(fixture.scene.agents[0], team_id=999)
    scene = replace(
        fixture.scene,
        agents=(unknown_agent, *fixture.scene.agents[1:]),
    )
    result = render_scene_geometry(scene, event_batch=fixture.event_batch)
    try:
        team_ring = next(
            artist
            for artist in _artists(result)
            if artist.get_gid() == "scene:agent:0:team"
        )
        assert team_ring.get_linestyle() == ":"
    finally:
        _close(result)


def test_unknown_aura_token_uses_neutral_color() -> None:
    assert _aura_color("future_aura") == "#9AA7B8"
