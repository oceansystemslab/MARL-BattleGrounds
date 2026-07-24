"""Pure tests for replay-compatible rendering descriptions."""

from dataclasses import FrozenInstanceError, fields
from inspect import getsource

import jax
import jax.numpy as jnp
import pytest

from marl_battlegrounds.core.config import resolve_agent_profile
from marl_battlegrounds.core.env import reset
from marl_battlegrounds.core.types import (
    AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER,
    AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER,
    HUNTER_CLASS_ID,
    MAGE_CLASS_ID,
    MAX_AGENT_SLOTS,
    MAX_OBSTACLE_SLOTS,
    NUM_SLOW_CHANNELS,
    NUM_STUN_CHANNELS,
    OBSTACLE_FEATURES,
    PRIEST_CLASS_ID,
    ROGUE_CLASS_ID,
    TEAM_A_ID,
    TEAM_B_ID,
    WARRIOR_CLASS_ID,
    EnvConfig,
    EnvState,
    Observation,
)
from marl_battlegrounds.rendering.visuals import (
    BASIC_COLOR,
    DAMAGE_COLOR,
    HEALING_COLOR,
    HUNTER_COLOR,
    MAGE_COLOR,
    PERSISTENT_BODY_LOCAL_RADIAL_BOUNDS,
    PRIEST_COLOR,
    ROGUE_COLOR,
    TARGET_COLOR,
    TEAM_A_COLOR,
    TEAM_B_COLOR,
    ULTIMATE_COLOR,
    UNAVAILABLE_COLOR,
    WARRIOR_COLOR,
    ActivationVisual,
    AuraCueVisual,
    BattlefieldOverlays,
    ChargeTrailVisual,
    HealthDeltaVisual,
    LaneMarkerVisual,
    ObserverVisibilityVisual,
    PersistentEffectVisual,
    RangeVisual,
    RejectedActionVisual,
    SelectionVisual,
    StatusCueVisual,
    TargetLinkVisual,
    class_color,
    describe_snapshot_overlays,
    merge_battlefield_overlays,
    team_color,
)


def _snapshot() -> tuple[EnvConfig, EnvState, Observation]:
    roster = jnp.asarray(
        (
            MAGE_CLASS_ID,
            WARRIOR_CLASS_ID,
            HUNTER_CLASS_ID,
            ROGUE_CLASS_ID,
            PRIEST_CLASS_ID,
            MAGE_CLASS_ID,
            WARRIOR_CLASS_ID,
            HUNTER_CLASS_ID,
            ROGUE_CLASS_ID,
            PRIEST_CLASS_ID,
        ),
        dtype=jnp.int32,
    )
    profile = resolve_agent_profile(roster, jnp.asarray((5, 5), dtype=jnp.int32))
    positions = jnp.stack(
        (
            jnp.linspace(1.0, 9.0, MAX_AGENT_SLOTS, dtype=jnp.float32),
            jnp.full((MAX_AGENT_SLOTS,), 2.0, dtype=jnp.float32),
        ),
        axis=1,
    )
    config = EnvConfig(
        max_steps=100,
        map_width=12.0,
        map_height=8.0,
        obstacles=jnp.zeros(
            (MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES),
            dtype=jnp.float32,
        ),
        agent_profile=profile,
        initial_agent_positions=positions,
        ordinary_movement_distance_scale=1.0,
    )
    state, observation, _, _ = reset(config, jax.random.key(0))
    return config, state, observation


def test_class_and_team_palette_is_exact() -> None:
    assert MAGE_COLOR == "#62D5D3"
    assert WARRIOR_COLOR == "#9A6B3F"
    assert HUNTER_COLOR == "#4FAE67"
    assert ROGUE_COLOR == "#C49A2C"
    assert PRIEST_COLOR == "#E88AB7"
    assert TEAM_A_COLOR == "#1E88FF"
    assert TEAM_B_COLOR == "#FF3B3B"
    assert BASIC_COLOR == "#2E7D32"
    assert ULTIMATE_COLOR == "#7E57C2"
    assert UNAVAILABLE_COLOR == "#707070"
    assert TARGET_COLOR == "#E040FB"
    assert DAMAGE_COLOR == "#D32F2F"
    assert HEALING_COLOR == "#00A86B"

    assert class_color(MAGE_CLASS_ID) == MAGE_COLOR
    assert class_color(WARRIOR_CLASS_ID) == WARRIOR_COLOR
    assert class_color(HUNTER_CLASS_ID) == HUNTER_COLOR
    assert class_color(ROGUE_CLASS_ID) == ROGUE_COLOR
    assert class_color(PRIEST_CLASS_ID) == PRIEST_COLOR
    assert team_color(TEAM_A_ID) == TEAM_A_COLOR
    assert team_color(TEAM_B_ID) == TEAM_B_COLOR


@pytest.mark.parametrize(("helper", "unknown_id"), ((class_color, 0), (team_color, 0)))
def test_palette_helpers_reject_unknown_ids(
    helper: object,
    unknown_id: int,
) -> None:
    with pytest.raises(ValueError):
        helper(unknown_id)  # type: ignore[operator]


def test_all_visual_descriptions_are_frozen_and_slotted() -> None:
    visual = SelectionVisual(global_slot=0, role="target")

    assert hasattr(SelectionVisual, "__slots__")
    with pytest.raises(FrozenInstanceError):
        visual.global_slot = 1  # type: ignore[misc]


def test_every_rendering_description_has_the_exact_audited_field_schema() -> None:
    expected = {
        SelectionVisual: ("global_slot", "role"),
        ObserverVisibilityVisual: (
            "observer_global_slot",
            "candidate_global_slot",
            "observer_visible",
        ),
        RangeVisual: ("global_slot", "center", "radius", "kind"),
        TargetLinkVisual: (
            "source_global_slot",
            "target_global_slot",
            "lane",
            "legal",
        ),
        LaneMarkerVisual: (
            "candidate_global_slot",
            "lane",
            "available",
            "selected",
        ),
        StatusCueVisual: (
            "global_slot",
            "family",
            "source_class_id",
            "channel_index",
            "duration",
        ),
        AuraCueVisual: ("global_slot", "kind", "multiplier"),
        PersistentEffectVisual: (
            "global_slot",
            "kind",
            "duration",
            "magnitude",
        ),
        HealthDeltaVisual: ("global_slot", "net_delta"),
        ActivationVisual: (
            "kind",
            "source_global_slot",
            "target_global_slot",
            "source_class_id",
        ),
        ChargeTrailVisual: (
            "source_global_slot",
            "start",
            "end",
            "target_global_slot",
            "path_kind",
            "opacity",
        ),
        RejectedActionVisual: (
            "actor_global_slot",
            "component",
            "target_global_slot",
            "lane",
        ),
        BattlefieldOverlays: (
            "selections",
            "observer_visibility",
            "ranges",
            "target_links",
            "lane_markers",
            "statuses",
            "auras",
            "persistent_effects",
            "health_deltas",
            "activations",
            "charge_trails",
            "rejections",
        ),
    }

    for description_type, field_names in expected.items():
        assert tuple(field.name for field in fields(description_type)) == field_names
        assert hasattr(description_type, "__slots__")


@pytest.mark.parametrize("global_slot", (-1, MAX_AGENT_SLOTS))
def test_slot_bearing_visuals_reject_out_of_domain_slots(global_slot: int) -> None:
    with pytest.raises(ValueError):
        SelectionVisual(global_slot=global_slot, role="target")

    with pytest.raises(ValueError):
        ObserverVisibilityVisual(
            observer_global_slot=0,
            candidate_global_slot=global_slot,
            observer_visible=False,
        )


@pytest.mark.parametrize("radius", (-0.01, float("inf"), float("nan")))
def test_range_visual_rejects_invalid_radii(radius: float) -> None:
    with pytest.raises(ValueError):
        RangeVisual(global_slot=0, center=(0.0, 0.0), radius=radius, kind="basic")


def test_visual_description_enums_and_coordinates_reject_invalid_runtime_values() -> (
    None
):
    with pytest.raises(ValueError):
        SelectionVisual(0, "primary")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        RangeVisual(0, (float("nan"), 0.0), 1.0, "basic")
    with pytest.raises(ValueError):
        RangeVisual(0, (0.0, 0.0), 1.0, "attack")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        TargetLinkVisual(0, 5, 2, True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        LaneMarkerVisual(5, -1, True, False)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        AuraCueVisual(0, "unknown", 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        AuraCueVisual(0, "mage_amplification", 1.0)
    with pytest.raises(ValueError):
        AuraCueVisual(0, "warrior_mitigation", 1.0)
    with pytest.raises(ValueError):
        PersistentEffectVisual(0, "unknown", 1, None)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ActivationVisual("unknown", 0, None, MAGE_CLASS_ID)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ActivationVisual("mage_burst", 0, 5, MAGE_CLASS_ID)
    with pytest.raises(ValueError):
        ActivationVisual("hunter_trap", 0, None, HUNTER_CLASS_ID)
    with pytest.raises(ValueError):
        ActivationVisual("holy_word", 0, 0, MAGE_CLASS_ID)
    with pytest.raises(ValueError):
        RejectedActionVisual(0, "unknown", None, None)  # type: ignore[arg-type]


def test_status_cues_enforce_source_class_channel_correspondence() -> None:
    for source_class_id, channel_index in (
        (WARRIOR_CLASS_ID, 0),
        (HUNTER_CLASS_ID, 1),
        (ROGUE_CLASS_ID, 2),
    ):
        assert (
            StatusCueVisual(
                0,
                "stun",
                source_class_id,
                channel_index,
                1,
            ).channel_index
            == channel_index
        )
        with pytest.raises(ValueError):
            StatusCueVisual(
                0,
                "slow",
                source_class_id,
                (channel_index + 1) % 3,
                1,
            )


def test_visual_descriptions_reject_nonfinite_numeric_payloads() -> None:
    with pytest.raises(ValueError):
        HealthDeltaVisual(0, float("inf"))
    with pytest.raises(ValueError):
        HealthDeltaVisual(0, 0.0)
    with pytest.raises(ValueError):
        PersistentEffectVisual(0, "mage_burst", 1, float("nan"))
    with pytest.raises(ValueError):
        ChargeTrailVisual(
            0,
            (float("inf"), 0.0),
            (1.0, 1.0),
            5,
            "charge_only",
            1.0,
        )


@pytest.mark.parametrize("opacity", (-0.01, 1.01))
def test_charge_trail_rejects_invalid_opacity(opacity: float) -> None:
    with pytest.raises(ValueError):
        ChargeTrailVisual(
            source_global_slot=0,
            start=(0.0, 0.0),
            end=(1.0, 1.0),
            target_global_slot=5,
            path_kind="charge_only",
            opacity=opacity,
        )


def test_overlay_merge_concatenates_every_field_in_argument_order() -> None:
    first = BattlefieldOverlays(
        selections=(SelectionVisual(0, "target"),),
        health_deltas=(HealthDeltaVisual(5, -2.0),),
    )
    second = BattlefieldOverlays(
        selections=(SelectionVisual(5, "target"),),
        auras=(AuraCueVisual(0, "mage_amplification", 1.15),),
    )

    merged = merge_battlefield_overlays(first, second)

    assert merged.selections == (*first.selections, *second.selections)
    assert merged.health_deltas == first.health_deltas
    assert merged.auras == second.auras
    for field in fields(BattlefieldOverlays):
        assert isinstance(getattr(merged, field.name), tuple)


def test_snapshot_description_preserves_all_status_channels_and_effects() -> None:
    config, reset_state, reset_observation = _snapshot()
    state = reset_state._replace(
        slow_durations=jnp.ones(
            (MAX_AGENT_SLOTS, NUM_SLOW_CHANNELS),
            dtype=jnp.int32,
        ),
        stun_durations=jnp.full(
            (MAX_AGENT_SLOTS, NUM_STUN_CHANNELS),
            2,
            dtype=jnp.int32,
        ),
        rogue_poison_anti_heal_durations=jnp.ones(
            (MAX_AGENT_SLOTS,),
            dtype=jnp.int32,
        ),
        mage_burst_damage_amplification_durations=jnp.ones(
            (MAX_AGENT_SLOTS,),
            dtype=jnp.int32,
        ),
        priest_blessing_of_freedom_slow_floor_durations=jnp.ones(
            (MAX_AGENT_SLOTS,),
            dtype=jnp.int32,
        ),
        previous_timestep_move_actions=jnp.arange(
            MAX_AGENT_SLOTS,
            dtype=jnp.int32,
        )
        % 9,
        previous_timestep_select_target_actions=jnp.arange(
            MAX_AGENT_SLOTS,
            dtype=jnp.int32,
        ),
        previous_timestep_use_ultimate_actions=jnp.arange(
            MAX_AGENT_SLOTS,
            dtype=jnp.int32,
        )
        % 2,
        has_previous_timestep_joint_action=jnp.asarray(True),
    )
    self_features = reset_observation.self_features
    self_features = self_features.at[
        :, AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER
    ].set(1.15)
    self_features = self_features.at[
        :, AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER
    ].set(0.85)
    observation = reset_observation._replace(self_features=self_features)

    overlay = describe_snapshot_overlays(config, state, observation)

    assert len(overlay.statuses) == MAX_AGENT_SLOTS * 6
    assert {cue.channel_index for cue in overlay.statuses} == {0, 1, 2}
    assert {cue.family for cue in overlay.statuses} == {"slow", "stun"}
    assert len(overlay.auras) == MAX_AGENT_SLOTS * 2
    assert len(overlay.persistent_effects) == MAX_AGENT_SLOTS * 3
    assert PersistentEffectVisual(0, "rogue_anti_heal", 1, 0.5) in (
        overlay.persistent_effects
    )


def test_snapshot_description_omits_identity_auras_and_zero_durations() -> None:
    config, state, observation = _snapshot()

    overlay = describe_snapshot_overlays(config, state, observation)

    assert overlay.statuses == ()
    assert overlay.persistent_effects == ()
    assert all(aura.multiplier != 1.0 for aura in overlay.auras)


def test_snapshot_aura_description_uses_named_feature_authorities() -> None:
    source = getsource(describe_snapshot_overlays)

    assert "AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER" in source
    assert "AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER" in source
    assert "[:, 29]" not in source
    assert "[:, 30]" not in source


def test_body_local_radial_bounds_do_not_exceed_radius() -> None:
    assert dict(
        (name, (lower, upper))
        for name, lower, upper in PERSISTENT_BODY_LOCAL_RADIAL_BOUNDS
    ) == {
        "team_boundary": (1.00, 1.00),
        "health": (0.73, 0.86),
        "aura": (0.61, 0.69),
        "lane": (0.52, 0.58),
        "class_identity": (0.00, 0.48),
    }
    for _, lower, upper in PERSISTENT_BODY_LOCAL_RADIAL_BOUNDS:
        assert 0.0 <= lower <= upper <= 1.0


@pytest.mark.parametrize(
    ("kind", "expected"),
    (
        ("rogue_anti_heal", 0.5),
        ("priest_freedom", 0.85),
        ("mage_burst", 1.5),
    ),
)
def test_persistent_effect_magnitudes_are_semantically_fixed(
    kind: str,
    expected: float,
) -> None:
    assert PersistentEffectVisual(0, kind, 1, expected).magnitude == expected  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        PersistentEffectVisual(0, kind, 1, None)  # type: ignore[arg-type]
