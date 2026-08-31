"""Deterministic authorized-presentation browser-contract regeneration proofs."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from scripts.dev.visual_debugger.export_authorized_presentation_browser_schema import (
    _AUTHORIZED_PRESENTATION_FRAME_ADAPTER,
    _SCHEMA_KEYWORDS,
    _compact_schema,
    _validate_schema_keywords,
    render_browser_schema_module,
)
from tests.export_authorized_presentation_browser_fixture import render_fixture

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SCHEMA_ASSET = (
    _REPOSITORY_ROOT
    / "web"
    / "visual_debugger"
    / "src"
    / "authorized-presentation-schema.js"
)
_FIXTURE_ASSET = (
    _REPOSITORY_ROOT
    / "web"
    / "visual_debugger"
    / "tests"
    / "fixtures"
    / "authorized-presentations-v1.json"
)


def test_generated_browser_schema_and_fixture_are_exact() -> None:
    """Checked assets must equal fresh Python output byte for byte."""
    assert _SCHEMA_ASSET.read_text(encoding="utf-8") == render_browser_schema_module()
    rendered_fixture = render_fixture()
    assert _FIXTURE_ASSET.read_text(encoding="utf-8") == rendered_fixture

    payload = json.loads(rendered_fixture)
    assert set(payload["presentations"]) == {
        "live_oracle",
        "live_no_shared_obs_agent_pov",
        "live_shared_obs_agent_pov",
        "replay_oracle",
        "replay_no_shared_obs_agent_pov",
        "replay_shared_obs_agent_pov",
    }
    assert set(payload["compatibility_cases"]) == {"legacy_v1"}
    presentation_roots = [
        *payload["presentations"].values(),
        *(pair["presentation"] for pair in payload["pairs"].values()),
        *(pair["presentation"] for pair in payload["continuity_pairs"].values()),
        *payload["state_cases"].values(),
        *payload["compatibility_cases"].values(),
    ]
    for root in presentation_roots:
        validated = _AUTHORIZED_PRESENTATION_FRAME_ADAPTER.validate_json(
            json.dumps(root, ensure_ascii=True, separators=(",", ":"))
        )
        assert validated.model_dump(mode="json") == root

    legacy_scene = payload["compatibility_cases"]["legacy_v1"]["current_endpoint"][
        "scene"
    ]
    assert all(
        "mechanics_version" not in row and "documentation_profile" not in row
        for row in legacy_scene["class_mechanics"]
    )
    assert legacy_scene["spawn_shield_mechanics"]["availability_kind"] == ("available")


def test_schema_compaction_and_keyword_universe_are_fail_closed() -> None:
    """Metadata stripping must preserve field names and reject new keywords."""
    source_schema = _AUTHORIZED_PRESENTATION_FRAME_ADAPTER.json_schema()
    _validate_schema_keywords(source_schema)
    assert {
        "$defs",
        "$ref",
        "additionalProperties",
        "anyOf",
        "const",
        "description",
        "discriminator",
        "enum",
        "exclusiveMaximum",
        "exclusiveMinimum",
        "items",
        "maxItems",
        "maxLength",
        "maximum",
        "minItems",
        "minLength",
        "minimum",
        "oneOf",
        "pattern",
        "prefixItems",
        "properties",
        "required",
        "title",
        "type",
    } == _SCHEMA_KEYWORDS

    synthetic = {
        "title": "root metadata",
        "description": "root metadata",
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "field metadata"},
            "description": {"type": "string", "title": "field metadata"},
        },
        "$defs": {"title": {"type": "string", "title": "definition metadata"}},
    }
    compact = _compact_schema(synthetic)
    assert isinstance(compact, dict)
    assert "title" not in compact
    assert "description" not in compact
    properties = compact["properties"]
    definitions = compact["$defs"]
    assert isinstance(properties, dict)
    assert isinstance(definitions, dict)
    assert set(properties) == {"title", "description"}
    assert set(definitions) == {"title"}
    assert properties["title"] == {"type": "string"}
    assert properties["description"] == {"type": "string"}
    assert definitions["title"] == {"type": "string"}

    poisoned = copy.deepcopy(source_schema)
    poisoned["futureIgnoredKeyword"] = True
    with pytest.raises(ValueError, match="unsupported authorized-presentation"):
        _validate_schema_keywords(poisoned)
