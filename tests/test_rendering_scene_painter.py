"""Focused semantic checks for the stateless scene-native Matplotlib painter."""

import re
from collections.abc import Callable, Iterable
from dataclasses import replace
from importlib import import_module
from math import hypot
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
from marl_battlegrounds.rendering.scene import (
    AcceptedActivationEventV1,
    NetHealthEventV1,
)
from marl_battlegrounds.rendering.scene_geometry import (
    _aura_color,  # pyright: ignore[reportPrivateUsage]
)
from marl_battlegrounds.rendering.vocabulary import (
    class_token_from_id,
    lookup_activation_token,
    team_token_from_id,
)


class _ArtistLike(Protocol):
    def get_edgecolor(self) -> object: ...

    def get_facecolor(self) -> object: ...

    def get_gid(self) -> str | None: ...

    def get_linestyle(self) -> str: ...

    def get_linewidth(self) -> float: ...


class _TextLike(_ArtistLike, Protocol):
    xy: tuple[float, float]

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


def _texts_with_gid_prefix(
    result: RenderResult,
    prefix: str,
) -> tuple[_TextLike, ...]:
    return tuple(
        artist
        for artist in _texts(result)
        if (artist.get_gid() or "").startswith(prefix)
    )


def _hex_color(value: object) -> str:
    colors = import_module("matplotlib.colors")
    to_hex = cast(Callable[..., str], colors.to_hex)
    return to_hex(value, keep_alpha=False).upper()


def _alpha(value: object) -> float:
    colors = import_module("matplotlib.colors")
    to_rgba = cast(Callable[..., tuple[float, float, float, float]], colors.to_rgba)
    return to_rgba(value)[3]


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
        status_artists = tuple(
            artist
            for artist in _texts(result)
            if (artist.get_gid() or "").startswith("scene:agent:0:status:")
        )
        assert {artist.get_text().split()[0][0] for artist in status_artists[:3]} == {
            "⬢"
        }
        assert {artist.get_text().split()[0][0] for artist in status_artists[3:6]} == {
            "↻"
        }
        assert {artist.get_text().split()[0][1:] for artist in status_artists[:3]} == {
            "W",
            "H",
            "R",
        }
        assert {artist.get_text().split()[0][1:] for artist in status_artists[3:6]} == {
            "W",
            "H",
            "R",
        }
        assert tuple(
            _hex_color(artist.get_bbox_patch().get_edgecolor())
            for artist in status_artists[:6]
        ) == (
            "#D18B47",
            "#84CC16",
            "#FACC15",
            "#D18B47",
            "#84CC16",
            "#FACC15",
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


def test_visual_vocabulary_static_evidence_keeps_semantic_grammar() -> None:
    fixture, result = _render_fixture(
        "visual_vocabulary",
        options=SceneRenderOptions(show_agent_ids=True),
    )
    try:
        assert fixture.event_batch is not None
        by_gid = {
            gid: artist
            for artist in _texts(result)
            if (gid := artist.get_gid()) is not None
        }
        assert tuple(
            by_gid[f"scene:agent:{agent.global_slot}:class"].get_text()
            for agent in fixture.scene.agents
        ) == tuple(
            class_token_from_id(agent.class_id).fallback
            for agent in fixture.scene.agents
        )
        assert "scene:selection:controlled:0" in _gids(result)
        assert "scene:selection:target:5" in _gids(result)

        artists_by_gid = {
            gid: artist
            for artist in _artists(result)
            if (gid := artist.get_gid()) is not None
        }
        assert tuple(
            _hex_color(artists_by_gid[f"scene:range:{slot}:basic"].get_edgecolor())
            for slot in range(5)
        ) == (
            "#22D3EE",
            "#D18B47",
            "#84CC16",
            "#FACC15",
            "#F472B6",
        )
        assert all(
            artists_by_gid[f"scene:range:{slot}:basic"].get_linestyle() == "--"
            for slot in range(5)
        )
        for aura_gid in (
            "scene:aura:0:mage_amplification",
            "scene:aura:1:warrior_mitigation",
        ):
            aura = artists_by_gid[aura_gid]
            assert aura.get_linewidth() == 0.0
            assert _alpha(aura.get_edgecolor()) == 0.0

        assert {
            by_gid[f"scene:agent:0:status:{token_id}"].get_text().split()[0][0]
            for token_id in (
                "stun_warrior_charge",
                "stun_hunter_trap",
                "stun_rogue_poison",
            )
        } == {"⬢"}
        assert {
            by_gid[f"scene:agent:2:status:{token_id}"].get_text().split()[0][0]
            for token_id in (
                "slow_warrior_charge",
                "slow_hunter_basic",
                "slow_rogue_poison",
            )
        } == {"↻"}

        activations = tuple(
            event
            for event in fixture.event_batch.events
            if type(event) is AcceptedActivationEventV1
        )
        assert len(activations) == 10
        for event in activations:
            primary = by_gid[f"event:accepted_activation:{event.event_id}"].get_text()
            secondary = by_gid.get(f"scene:event-label:{event.event_id}")
            visible_label = primary or (secondary.get_text() if secondary else "")
            assert visible_label.startswith(
                lookup_activation_token(event.token_id).short_label
            )
            if event.token_id == "basic_damage":
                assert secondary is not None
                expected_color = {
                    "mage": "#22D3EE",
                    "warrior": "#D18B47",
                    "hunter": "#84CC16",
                    "rogue": "#FACC15",
                }[class_token_from_id(event.source_class_id).token_id]
                assert (
                    _hex_color(secondary.get_bbox_patch().get_edgecolor())
                    == expected_color
                )
            impact = by_gid.get(f"scene:event-impact:{event.event_id}")
            if event.token_id in (
                "basic_damage",
                "warrior_charge",
                "rogue_poison",
            ):
                assert impact is not None
                assert impact.get_text() == "\N{MINUS SIGN}"
                assert _hex_color(impact.get_bbox_patch().get_edgecolor()) == "#FB7185"
            elif event.token_id in ("basic_heal", "holy_word"):
                assert impact is not None
                assert impact.get_text() == "+"
                assert _hex_color(impact.get_bbox_patch().get_edgecolor()) == "#34D399"
            else:
                assert impact is None
            if impact is not None:
                recipient = next(
                    agent
                    for agent in fixture.scene.agents
                    if agent.global_slot == event.target_global_slot
                )
                assert event.target_anchor is not None
                assert (
                    hypot(
                        impact.xy[0] - event.target_anchor[0],
                        impact.xy[1] - event.target_anchor[1],
                    )
                    >= recipient.radius
                )

        assert (
            by_gid["event:net_health:synthetic:visual_vocabulary:net-damage"].get_text()
            == "NET -12.35"
        )
        assert (
            by_gid[
                "event:net_health:synthetic:visual_vocabulary:net-healing"
            ].get_text()
            == "NET +8.5"
        )
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


def test_visual_language_aligns_aura_ranges_teams_and_hunter_color() -> None:
    _skip_if_matplotlib_unavailable()
    fixture = get_renderer_fixture("crowded_teamfight")
    hunter = next(
        agent
        for agent in fixture.scene.agents
        if class_token_from_id(agent.class_id).token_id == "hunter"
    )
    scene = replace(
        fixture.scene,
        ranges=tuple(
            replace(
                range_record,
                global_slot=hunter.global_slot,
                center=hunter.position,
            )
            for range_record in fixture.scene.ranges
        ),
    )
    result = render_scene_geometry(scene, event_batch=fixture.event_batch)
    try:
        artists = _artists(result)
        by_gid = {
            gid: artist for artist in artists if (gid := artist.get_gid()) is not None
        }
        hunter_body = by_gid[f"scene:agent:{hunter.global_slot}:body"]
        assert _hex_color(hunter_body.get_facecolor()) == "#84CC16"

        aura_artists = tuple(
            artist for gid, artist in by_gid.items() if gid.startswith("scene:aura:")
        )
        assert aura_artists
        assert all(artist.get_linewidth() == 0.0 for artist in aura_artists)
        assert all(
            cast(tuple[float, ...], artist.get_edgecolor())[-1] == 0.0
            for artist in aura_artists
        )

        observation = by_gid[f"scene:range:{hunter.global_slot}:observation"]
        basic = by_gid[f"scene:range:{hunter.global_slot}:basic"]
        ultimate = by_gid[f"scene:range:{hunter.global_slot}:ultimate"]
        assert observation.get_linestyle() == ":"
        assert _hex_color(observation.get_edgecolor()) == "#F4F7FB"
        assert basic.get_linestyle() == "--"
        assert _hex_color(basic.get_edgecolor()) == "#84CC16"
        assert ultimate.get_linestyle() == "-."
        assert _hex_color(ultimate.get_edgecolor()) == "#A78BFA"

        known_team_rings = tuple(
            by_gid[f"scene:agent:{agent.global_slot}:team"]
            for agent in fixture.scene.agents
        )
        assert all(ring.get_linestyle() == "-" for ring in known_team_rings)
        team_b_slots = {
            agent.global_slot
            for agent in fixture.scene.agents
            if team_token_from_id(agent.team_id).token_id == "team_b"
        }
        assert {
            int(gid.split(":")[2]) for gid in by_gid if gid.endswith(":team-marker")
        } == team_b_slots
    finally:
        _close(result)


def test_human_visible_float_labels_never_exceed_two_decimals() -> None:
    _skip_if_matplotlib_unavailable()
    fixture = get_renderer_fixture("crowded_teamfight")
    assert fixture.event_batch is not None
    first_agent = fixture.scene.agents[0]
    first_modifier = replace(
        first_agent.modifiers[0],
        multiplier=1.234567,
    )
    scene = replace(
        fixture.scene,
        agents=(
            replace(
                first_agent,
                modifiers=(first_modifier, *first_agent.modifiers[1:]),
            ),
            *fixture.scene.agents[1:],
        ),
    )
    events = list(fixture.event_batch.events)
    net_index = next(
        index for index, event in enumerate(events) if type(event) is NetHealthEventV1
    )
    net_event = cast(NetHealthEventV1, events[net_index])
    events[net_index] = replace(
        net_event,
        health_after=net_event.health_before - 12.34567,
        net_delta=-12.34567,
        outcome="damage",
    )
    event_batch = replace(fixture.event_batch, events=tuple(events))

    result = render_scene_geometry(scene, event_batch=event_batch)
    try:
        visible_text = {
            artist.get_gid(): artist.get_text() for artist in _texts(result)
        }
        modifier_gid = (
            f"scene:agent:{first_agent.global_slot}:modifier:{first_modifier.token_id}"
        )
        assert visible_text[modifier_gid].endswith("x1.23")
        assert visible_text[f"event:net_health:{net_event.event_id}"] == "NET -12.35"
        assert all(
            re.search(r"\d+\.\d{3,}", text) is None for text in visible_text.values()
        )
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
