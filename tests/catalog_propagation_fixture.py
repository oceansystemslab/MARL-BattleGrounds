"""One Python-authored catalog mutation consumed by host and browser tests."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from decimal import ROUND_HALF_UP, Decimal
from math import isfinite
from typing import cast

from scripts.dev.visual_debugger.control import create_session
from scripts.dev.visual_debugger.frame import build_debugger_frame
from scripts.dev.visual_debugger.protocol import ResearcherLiveDebuggerFrameV2
from scripts.dev.visual_debugger.scenarios import get_scenario
from tests.visual_debugger_fixtures import debugger_test_launch_specification

from marl_battlegrounds.evaluation.capture import (
    capture_initial_evaluation_frame_v1,
)
from marl_battlegrounds.evaluation.models import (
    EvaluationEpisodeContextV1,
    StaticMechanicsCatalogV1,
    canonical_digest_sha256,
)
from marl_battlegrounds.rendering.evaluation_adapter import (
    initialize_status_source_evidence_v2,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class CatalogPropagationExpectationV1:
    """Dynamic values deliberately changed at the Python catalog authority."""

    basic_raw_damage: float
    burst_duration_steps: int
    burst_multiplier: float
    aura_radius: float
    aura_multiplier: float


@dataclass(frozen=True, slots=True, kw_only=True)
class CatalogPropagationDeltaV1:
    """Test-owned variation applied to whatever catalog production supplies."""

    basic_raw_damage: float = 4.25
    burst_duration_steps: int = 2
    burst_multiplier: float = 0.23
    aura_radius: float = 2.75
    aura_multiplier: float = 0.02


def _display_grid_sum(value: float, delta: float) -> float:
    """Add a variation and quantize it to the browser's two-decimal grid."""
    if not isfinite(value) or not isfinite(delta):
        raise ValueError("catalog propagation values must remain finite")
    return float(
        (Decimal(str(value)) + Decimal(str(delta))).quantize(
            Decimal("0.01"),
            rounding=ROUND_HALF_UP,
        )
    )


def catalog_propagation_values(
    context: EvaluationEpisodeContextV1,
) -> CatalogPropagationExpectationV1:
    """Read the five mechanics varied by the cross-boundary proof."""
    catalog = context.static_mechanics_catalog
    mage = catalog.class_mechanics[1]
    burst = catalog.status_channels[7]
    mage_aura = catalog.aura_mechanics[0]
    if mage.class_name != "Mage":
        raise ValueError("catalog propagation fixture requires canonical Mage row")
    if burst.status_id != "mage_burst_damage_amplification":
        raise ValueError("catalog propagation fixture requires canonical Burst row")
    if mage_aura.aura_id != "mage_damage_amplification":
        raise ValueError("catalog propagation fixture requires canonical Mage aura row")
    if burst.magnitude is None:
        raise ValueError("catalog propagation fixture requires a Burst magnitude")
    return CatalogPropagationExpectationV1(
        basic_raw_damage=mage.basic_raw_damage,
        burst_duration_steps=burst.duration_steps,
        burst_multiplier=burst.magnitude,
        aura_radius=mage_aura.radius,
        aura_multiplier=mage_aura.per_emitter_multiplier,
    )


def _derived_expectation(
    context: EvaluationEpisodeContextV1,
) -> CatalogPropagationExpectationV1:
    """Derive a distinct reasonable variation from the validated source catalog."""
    source = catalog_propagation_values(context)
    delta = CatalogPropagationDeltaV1()
    expected = CatalogPropagationExpectationV1(
        basic_raw_damage=_display_grid_sum(
            source.basic_raw_damage,
            delta.basic_raw_damage,
        ),
        burst_duration_steps=(source.burst_duration_steps + delta.burst_duration_steps),
        burst_multiplier=_display_grid_sum(
            source.burst_multiplier,
            delta.burst_multiplier,
        ),
        aura_radius=_display_grid_sum(source.aura_radius, delta.aura_radius),
        aura_multiplier=_display_grid_sum(
            source.aura_multiplier,
            delta.aura_multiplier,
        ),
    )
    changed_pairs = (
        ("Mage Basic damage", source.basic_raw_damage, expected.basic_raw_damage),
        (
            "Burst duration",
            source.burst_duration_steps,
            expected.burst_duration_steps,
        ),
        ("Burst multiplier", source.burst_multiplier, expected.burst_multiplier),
        ("Mage aura radius", source.aura_radius, expected.aura_radius),
        (
            "Mage aura multiplier",
            source.aura_multiplier,
            expected.aura_multiplier,
        ),
    )
    for name, source_value, expected_value in changed_pairs:
        if not isfinite(expected_value) or expected_value == source_value:
            raise ValueError(f"catalog propagation must vary finite {name}")
    return expected


def replace_catalog_propagation_values(
    original_context: EvaluationEpisodeContextV1,
    expected: CatalogPropagationExpectationV1,
) -> EvaluationEpisodeContextV1:
    """Return a fully revalidated context with one reasonable Mage tune."""
    catalog_payload = original_context.static_mechanics_catalog.model_dump(mode="json")

    class_rows = cast(list[dict[str, object]], catalog_payload["class_mechanics"])
    mutated_class_rows = [dict(row) for row in class_rows]
    mage = mutated_class_rows[1]
    if mage["class_name"] != "Mage":
        raise ValueError("catalog propagation fixture requires canonical Mage row")
    mage["basic_raw_damage"] = expected.basic_raw_damage
    catalog_payload["class_mechanics"] = mutated_class_rows

    status_rows = cast(list[dict[str, object]], catalog_payload["status_channels"])
    mutated_status_rows = [dict(row) for row in status_rows]
    burst = mutated_status_rows[7]
    if burst["status_id"] != "mage_burst_damage_amplification":
        raise ValueError("catalog propagation fixture requires canonical Burst row")
    burst["duration_steps"] = expected.burst_duration_steps
    burst["magnitude"] = expected.burst_multiplier
    catalog_payload["status_channels"] = mutated_status_rows

    aura_rows = cast(list[dict[str, object]], catalog_payload["aura_mechanics"])
    mutated_aura_rows = [dict(row) for row in aura_rows]
    mage_aura = mutated_aura_rows[0]
    if mage_aura["aura_id"] != "mage_damage_amplification":
        raise ValueError("catalog propagation fixture requires canonical Mage aura row")
    mage_aura["radius"] = expected.aura_radius
    mage_aura["per_emitter_multiplier"] = expected.aura_multiplier
    catalog_payload["aura_mechanics"] = mutated_aura_rows

    catalog_payload["canonical_digest_sha256"] = canonical_digest_sha256(
        catalog_payload,
        exclude={"canonical_digest_sha256"},
    )
    mutated_catalog = StaticMechanicsCatalogV1.model_validate_json(
        json.dumps(catalog_payload)
    )
    context_payload = original_context.model_dump(mode="json")
    context_payload["static_mechanics_catalog"] = mutated_catalog.model_dump(
        mode="json"
    )
    return EvaluationEpisodeContextV1.model_validate_json(json.dumps(context_payload))


def derive_catalog_propagation_context(
    original_context: EvaluationEpisodeContextV1,
) -> tuple[EvaluationEpisodeContextV1, CatalogPropagationExpectationV1]:
    """Apply one source-relative variation and return its exact expectations."""
    expected = _derived_expectation(original_context)
    context = replace_catalog_propagation_values(original_context, expected)
    if catalog_propagation_values(context) != expected:
        raise ValueError("catalog propagation context did not retain its variation")
    return context, expected


def build_catalog_propagation_fixture() -> tuple[
    ResearcherLiveDebuggerFrameV2,
    CatalogPropagationExpectationV1,
]:
    """Build the exact mutated Python envelope served to the browser proof."""
    session = create_session(
        get_scenario("arena_5v5"),
        seed=0,
        evaluation_launch_specification=debugger_test_launch_specification(0),
        controlled_global_slot=0,
        show_ranges=True,
        verbose_logging=False,
    )
    context, expected = derive_catalog_propagation_context(session.evaluation_context)
    initial_frame = capture_initial_evaluation_frame_v1(
        context,
        session.state,
        session.observation,
        session.action_mask,
    )
    evidence = initialize_status_source_evidence_v2(context, initial_frame)
    mutated_session = replace(
        session,
        evaluation_context=context,
        current_evaluation_frame=initial_frame,
        status_source_evidence_state=evidence,
    )
    frame = build_debugger_frame(
        mutated_session,
        session_id="catalog-propagation",
        revision=0,
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
    )
    if not isinstance(frame, ResearcherLiveDebuggerFrameV2):
        raise TypeError("catalog propagation fixture must remain researcher-authored")
    return frame, expected


def catalog_propagation_wire_payload() -> dict[str, object]:
    """Return the test-only JSON envelope shared across the process boundary."""
    frame, expected = build_catalog_propagation_fixture()
    return {
        "live_frame": frame.model_dump(mode="json"),
        "expected": {
            "basic_raw_damage": expected.basic_raw_damage,
            "burst_duration_steps": expected.burst_duration_steps,
            "burst_multiplier": expected.burst_multiplier,
            "aura_radius": expected.aura_radius,
            "aura_multiplier": expected.aura_multiplier,
        },
    }


if __name__ == "__main__":
    print(json.dumps(catalog_propagation_wire_payload(), sort_keys=True))
