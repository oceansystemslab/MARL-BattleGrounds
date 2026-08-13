"""Integration proof that researcher scene mechanics remain catalog-authored."""

from __future__ import annotations

import json
from typing import cast

import pytest
from scripts.dev.visual_debugger.control import create_session
from scripts.dev.visual_debugger.scenarios import get_scenario
from tests.catalog_propagation_fixture import (
    CatalogPropagationExpectationV1,
    build_catalog_propagation_fixture,
    catalog_propagation_values,
    derive_catalog_propagation_context,
    replace_catalog_propagation_values,
)
from tests.visual_debugger_fixtures import debugger_test_launch_specification

from marl_battlegrounds.evaluation.models import EvaluationEpisodeContextV1
from marl_battlegrounds.rendering.scene import to_jsonable


def test_catalog_mechanic_mutation_reaches_serialized_researcher_scene() -> None:
    """Keep browser mechanics factual instead of duplicating catalog tuning."""
    live_frame, expected = build_catalog_propagation_fixture()
    scene = live_frame.projection.scene
    scene_payload = cast(
        dict[str, object],
        json.loads(json.dumps(to_jsonable(scene))),
    )
    serialized_rows = cast(
        list[dict[str, object]],
        scene_payload["class_mechanics"],
    )

    assert (
        live_frame.projection.scene.class_mechanics[0].basic_raw_damage
        == expected.basic_raw_damage
    )
    assert scene.class_mechanics[0].class_id == 1
    assert scene.class_mechanics[0].basic_raw_damage == expected.basic_raw_damage
    assert serialized_rows[0]["class_id"] == 1
    assert serialized_rows[0]["basic_raw_damage"] == expected.basic_raw_damage
    mage_scene = scene.class_mechanics[0]
    burst_scene = next(
        row
        for row in mage_scene.status_mechanics
        if row.status_id == "mage_burst_damage_amplification"
    )
    aura_scene = mage_scene.aura_mechanics[0]
    assert burst_scene.duration_steps == expected.burst_duration_steps
    assert burst_scene.magnitude == expected.burst_multiplier
    assert aura_scene.radius == expected.aura_radius
    assert aura_scene.per_emitter_multiplier == expected.aura_multiplier
    serialized_burst = cast(
        list[dict[str, object]], serialized_rows[0]["status_mechanics"]
    )[0]
    serialized_aura = cast(
        list[dict[str, object]], serialized_rows[0]["aura_mechanics"]
    )[0]
    assert serialized_burst["duration_steps"] == expected.burst_duration_steps
    assert serialized_burst["magnitude"] == expected.burst_multiplier
    assert serialized_aura["radius"] == expected.aura_radius
    assert serialized_aura["per_emitter_multiplier"] == expected.aura_multiplier


@pytest.mark.parametrize("source_basic_damage", [17.25, 13.1])
def test_catalog_mechanic_mutation_is_relative_to_any_valid_source_catalog(
    source_basic_damage: float,
) -> None:
    """A production tune must not collide with fixed test sentinels."""
    session = create_session(
        get_scenario("arena_5v5"),
        seed=0,
        evaluation_launch_specification=debugger_test_launch_specification(0),
        controlled_global_slot=0,
        show_ranges=True,
        verbose_logging=False,
    )
    former_sentinels = CatalogPropagationExpectationV1(
        basic_raw_damage=source_basic_damage,
        burst_duration_steps=7,
        burst_multiplier=1.73,
        aura_radius=4.75,
        aura_multiplier=1.17,
    )
    source_context = replace_catalog_propagation_values(
        session.evaluation_context,
        former_sentinels,
    )
    varied_context, expected = derive_catalog_propagation_context(source_context)

    assert catalog_propagation_values(source_context) == former_sentinels
    assert expected != former_sentinels
    assert expected.basic_raw_damage == pytest.approx(source_basic_damage + 4.25)
    assert expected.burst_duration_steps == 9
    assert expected.burst_multiplier == 1.96
    assert expected.aura_radius == 7.5
    assert expected.aura_multiplier == 1.19
    assert varied_context == EvaluationEpisodeContextV1.model_validate_json(
        varied_context.model_dump_json()
    )
    assert varied_context.static_mechanics_catalog == type(
        varied_context.static_mechanics_catalog
    ).model_validate_json(varied_context.static_mechanics_catalog.model_dump_json())
