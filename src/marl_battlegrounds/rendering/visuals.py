"""Immutable visual descriptions shared by debug and replay presentation."""

from dataclasses import dataclass, fields
from typing import Literal

import numpy as np

from marl_battlegrounds.core.combat import (
    MAGE_BURST_DAMAGE_MULTIPLIER,
    PRIEST_HEAL_SPEED_FLOOR,
    ROGUE_POISON_ANTI_HEAL_MULTIPLIER,
)
from marl_battlegrounds.core.types import (
    AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER,
    AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER,
    HUNTER_CLASS_ID,
    MAGE_CLASS_ID,
    MAX_AGENT_SLOTS,
    PRIEST_CLASS_ID,
    ROGUE_CLASS_ID,
    TEAM_A_ID,
    TEAM_B_ID,
    WARRIOR_CLASS_ID,
    EnvConfig,
    EnvState,
    Observation,
)
from marl_battlegrounds.rendering.scene import (
    ChargePathKind,
    Lane,
    RejectionComponent,
)
from marl_battlegrounds.rendering.vocabulary import ActivationTokenId

type Point2D = tuple[float, float]
type RangeKind = Literal["observation", "basic", "ultimate"]
type SelectionRole = Literal["controlled", "target"]
type StatusFamily = Literal["stun", "slow"]
type AuraKind = Literal["mage_amplification", "warrior_mitigation"]
type PersistentEffectKind = Literal["rogue_anti_heal", "priest_freedom", "mage_burst"]
type ActivationKind = ActivationTokenId

MAGE_COLOR = "#62D5D3"
WARRIOR_COLOR = "#9A6B3F"
HUNTER_COLOR = "#4FAE67"
ROGUE_COLOR = "#C49A2C"
PRIEST_COLOR = "#E88AB7"
TEAM_A_COLOR = "#1E88FF"
TEAM_B_COLOR = "#FF3B3B"
BASIC_COLOR = "#2E7D32"
ULTIMATE_COLOR = "#7E57C2"
UNAVAILABLE_COLOR = "#707070"
TARGET_COLOR = "#E040FB"
DAMAGE_COLOR = "#D32F2F"
HEALING_COLOR = "#00A86B"

# Auditable normalized radial allocation for persistent body-local presentation.
PERSISTENT_BODY_LOCAL_RADIAL_BOUNDS: tuple[tuple[str, float, float], ...] = (
    ("team_boundary", 1.00, 1.00),
    ("health", 0.73, 0.86),
    ("aura", 0.61, 0.69),
    ("lane", 0.52, 0.58),
    ("class_identity", 0.00, 0.48),
)

_CLASS_COLORS = {
    MAGE_CLASS_ID: MAGE_COLOR,
    WARRIOR_CLASS_ID: WARRIOR_COLOR,
    HUNTER_CLASS_ID: HUNTER_COLOR,
    ROGUE_CLASS_ID: ROGUE_COLOR,
    PRIEST_CLASS_ID: PRIEST_COLOR,
}
_TEAM_COLORS = {
    TEAM_A_ID: TEAM_A_COLOR,
    TEAM_B_ID: TEAM_B_COLOR,
}


def _validate_global_slot(global_slot: int) -> None:
    if not 0 <= global_slot < MAX_AGENT_SLOTS:
        msg = f"global_slot must be in [0, {MAX_AGENT_SLOTS}); got {global_slot}."
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class SelectionVisual:
    global_slot: int
    role: SelectionRole

    def __post_init__(self) -> None:
        _validate_global_slot(self.global_slot)
        if self.role not in ("controlled", "target"):
            msg = f"unknown selection role: {self.role!r}."
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ObserverVisibilityVisual:
    observer_global_slot: int
    candidate_global_slot: int
    observer_visible: bool

    def __post_init__(self) -> None:
        _validate_global_slot(self.observer_global_slot)
        _validate_global_slot(self.candidate_global_slot)


@dataclass(frozen=True, slots=True)
class RangeVisual:
    global_slot: int
    center: Point2D
    radius: float
    kind: RangeKind

    def __post_init__(self) -> None:
        _validate_global_slot(self.global_slot)
        if self.kind not in ("observation", "basic", "ultimate"):
            msg = f"unknown range kind: {self.kind!r}."
            raise ValueError(msg)
        if len(self.center) != 2 or not all(
            np.isfinite(value) for value in self.center
        ):
            msg = f"center must contain two finite coordinates; got {self.center!r}."
            raise ValueError(msg)
        if not np.isfinite(self.radius) or self.radius < 0:
            msg = f"radius must be finite and non-negative; got {self.radius}."
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class TargetLinkVisual:
    source_global_slot: int
    target_global_slot: int
    lane: Lane
    legal: bool

    def __post_init__(self) -> None:
        _validate_global_slot(self.source_global_slot)
        _validate_global_slot(self.target_global_slot)
        if self.lane not in (0, 1):
            msg = f"lane must be 0 or 1; got {self.lane}."
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class LaneMarkerVisual:
    candidate_global_slot: int
    lane: Lane
    available: bool
    selected: bool

    def __post_init__(self) -> None:
        _validate_global_slot(self.candidate_global_slot)
        if self.lane not in (0, 1):
            msg = f"lane must be 0 or 1; got {self.lane}."
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class StatusCueVisual:
    global_slot: int
    family: StatusFamily
    source_class_id: int
    channel_index: int
    duration: int

    def __post_init__(self) -> None:
        _validate_global_slot(self.global_slot)
        if self.family not in ("stun", "slow"):
            msg = f"unknown status family: {self.family!r}."
            raise ValueError(msg)
        if self.source_class_id not in (
            WARRIOR_CLASS_ID,
            HUNTER_CLASS_ID,
            ROGUE_CLASS_ID,
        ):
            msg = f"unknown status source class: {self.source_class_id}."
            raise ValueError(msg)
        if self.channel_index not in (0, 1, 2):
            msg = f"status channel must be 0, 1, or 2; got {self.channel_index}."
            raise ValueError(msg)
        expected_channel = {
            WARRIOR_CLASS_ID: 0,
            HUNTER_CLASS_ID: 1,
            ROGUE_CLASS_ID: 2,
        }[self.source_class_id]
        if self.channel_index != expected_channel:
            msg = (
                f"source class {self.source_class_id} uses status channel "
                f"{expected_channel}, not {self.channel_index}."
            )
            raise ValueError(msg)
        if self.duration <= 0:
            msg = f"status duration must be positive; got {self.duration}."
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class AuraCueVisual:
    global_slot: int
    kind: AuraKind
    multiplier: float

    def __post_init__(self) -> None:
        _validate_global_slot(self.global_slot)
        if self.kind not in ("mage_amplification", "warrior_mitigation"):
            msg = f"unknown aura kind: {self.kind!r}."
            raise ValueError(msg)
        if not np.isfinite(self.multiplier):
            msg = f"aura multiplier must be finite; got {self.multiplier}."
            raise ValueError(msg)
        if (self.kind == "mage_amplification" and self.multiplier <= 1.0) or (
            self.kind == "warrior_mitigation" and self.multiplier >= 1.0
        ):
            msg = (
                f"{self.kind} must describe a non-identity active multiplier; "
                f"got {self.multiplier}."
            )
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class PersistentEffectVisual:
    global_slot: int
    kind: PersistentEffectKind
    duration: int
    magnitude: float | None

    def __post_init__(self) -> None:
        _validate_global_slot(self.global_slot)
        if self.kind not in ("rogue_anti_heal", "priest_freedom", "mage_burst"):
            msg = f"unknown persistent effect kind: {self.kind!r}."
            raise ValueError(msg)
        if self.duration <= 0:
            msg = f"effect duration must be positive; got {self.duration}."
            raise ValueError(msg)
        expected_magnitude = {
            "rogue_anti_heal": ROGUE_POISON_ANTI_HEAL_MULTIPLIER,
            "priest_freedom": PRIEST_HEAL_SPEED_FLOOR,
            "mage_burst": MAGE_BURST_DAMAGE_MULTIPLIER,
        }[self.kind]
        if self.magnitude != expected_magnitude:
            msg = (
                f"{self.kind} magnitude must be {expected_magnitude}; "
                f"got {self.magnitude}."
            )
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class HealthDeltaVisual:
    global_slot: int
    net_delta: float

    def __post_init__(self) -> None:
        _validate_global_slot(self.global_slot)
        if not np.isfinite(self.net_delta) or self.net_delta == 0:
            msg = f"net_delta must be finite and non-zero; got {self.net_delta}."
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ActivationVisual:
    kind: ActivationKind
    source_global_slot: int
    target_global_slot: int | None
    source_class_id: int

    def __post_init__(self) -> None:
        _validate_global_slot(self.source_global_slot)
        if self.target_global_slot is not None:
            _validate_global_slot(self.target_global_slot)
        if self.kind not in (
            "basic_damage",
            "basic_heal",
            "holy_word",
            "mage_burst",
            "warrior_charge",
            "hunter_trap",
            "rogue_poison",
        ):
            msg = f"unknown activation kind: {self.kind!r}."
            raise ValueError(msg)
        if self.source_class_id not in _CLASS_COLORS:
            msg = f"unknown activation source class: {self.source_class_id}."
            raise ValueError(msg)
        expected_class = {
            "basic_heal": PRIEST_CLASS_ID,
            "holy_word": PRIEST_CLASS_ID,
            "mage_burst": MAGE_CLASS_ID,
            "warrior_charge": WARRIOR_CLASS_ID,
            "hunter_trap": HUNTER_CLASS_ID,
            "rogue_poison": ROGUE_CLASS_ID,
        }.get(self.kind)
        if expected_class is not None and self.source_class_id != expected_class:
            msg = (
                f"{self.kind} must originate from class {expected_class}; "
                f"got {self.source_class_id}."
            )
            raise ValueError(msg)
        if self.kind == "basic_damage" and self.source_class_id == PRIEST_CLASS_ID:
            msg = "Priest Basic must be described as basic_heal."
            raise ValueError(msg)
        if (self.kind == "mage_burst") != (self.target_global_slot is None):
            msg = "only Mage Burst is a target-none activation."
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ChargeTrailVisual:
    source_global_slot: int
    start: Point2D
    end: Point2D
    target_global_slot: int
    path_kind: ChargePathKind
    opacity: float

    def __post_init__(self) -> None:
        _validate_global_slot(self.source_global_slot)
        _validate_global_slot(self.target_global_slot)
        if self.path_kind not in (
            "charge_only",
            "combined_charge_and_movement",
        ):
            msg = f"unknown Charge path kind: {self.path_kind!r}."
            raise ValueError(msg)
        if not all(
            np.isfinite(value) for point in (self.start, self.end) for value in point
        ):
            msg = "Charge trail endpoints must be finite."
            raise ValueError(msg)
        if not np.isfinite(self.opacity) or not 0 <= self.opacity <= 1:
            msg = f"opacity must be in [0, 1]; got {self.opacity}."
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class RejectedActionVisual:
    actor_global_slot: int
    component: RejectionComponent
    target_global_slot: int | None
    lane: Lane | None

    def __post_init__(self) -> None:
        _validate_global_slot(self.actor_global_slot)
        if self.component not in ("movement", "combat", "complete_tuple_domain"):
            msg = f"unknown rejection component: {self.component!r}."
            raise ValueError(msg)
        if self.target_global_slot is not None:
            _validate_global_slot(self.target_global_slot)
        if self.lane not in (None, 0, 1):
            msg = f"lane must be None, 0, or 1; got {self.lane}."
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class BattlefieldOverlays:
    selections: tuple[SelectionVisual, ...] = ()
    observer_visibility: tuple[ObserverVisibilityVisual, ...] = ()
    ranges: tuple[RangeVisual, ...] = ()
    target_links: tuple[TargetLinkVisual, ...] = ()
    lane_markers: tuple[LaneMarkerVisual, ...] = ()
    statuses: tuple[StatusCueVisual, ...] = ()
    auras: tuple[AuraCueVisual, ...] = ()
    persistent_effects: tuple[PersistentEffectVisual, ...] = ()
    health_deltas: tuple[HealthDeltaVisual, ...] = ()
    activations: tuple[ActivationVisual, ...] = ()
    charge_trails: tuple[ChargeTrailVisual, ...] = ()
    rejections: tuple[RejectedActionVisual, ...] = ()


def class_color(class_id: int) -> str:
    """Return the fixed class presentation color."""
    try:
        return _CLASS_COLORS[class_id]
    except KeyError as exc:
        msg = f"unknown class ID: {class_id}."
        raise ValueError(msg) from exc


def team_color(team_id: int) -> str:
    """Return the fixed team presentation color."""
    try:
        return _TEAM_COLORS[team_id]
    except KeyError as exc:
        msg = f"unknown team ID: {team_id}."
        raise ValueError(msg) from exc


def source_effect_color(class_id: int) -> str:
    """Return the source-class color used by status and activation cues."""
    return class_color(class_id)


def describe_snapshot_overlays(
    config: EnvConfig,
    state: EnvState,
    observation: Observation,
) -> BattlefieldOverlays:
    """Describe durable persistent presentation from one public snapshot."""
    active_mask = np.asarray(config.agent_profile.active_mask, dtype=bool)
    slow_durations = np.asarray(state.slow_durations, dtype=np.int32)
    stun_durations = np.asarray(state.stun_durations, dtype=np.int32)
    mage_auras = np.asarray(
        observation.self_features[
            :, AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER
        ],
        dtype=np.float32,
    )
    warrior_auras = np.asarray(
        observation.self_features[
            :, AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER
        ],
        dtype=np.float32,
    )
    source_classes = (WARRIOR_CLASS_ID, HUNTER_CLASS_ID, ROGUE_CLASS_ID)

    statuses: list[StatusCueVisual] = []
    auras: list[AuraCueVisual] = []
    persistent_effects: list[PersistentEffectVisual] = []

    for global_slot in range(MAX_AGENT_SLOTS):
        if not active_mask[global_slot]:
            continue

        for channel_index, source_class_id in enumerate(source_classes):
            stun_duration = int(stun_durations[global_slot, channel_index])
            if stun_duration > 0:
                statuses.append(
                    StatusCueVisual(
                        global_slot=global_slot,
                        family="stun",
                        source_class_id=source_class_id,
                        channel_index=channel_index,
                        duration=stun_duration,
                    )
                )

            slow_duration = int(slow_durations[global_slot, channel_index])
            if slow_duration > 0:
                statuses.append(
                    StatusCueVisual(
                        global_slot=global_slot,
                        family="slow",
                        source_class_id=source_class_id,
                        channel_index=channel_index,
                        duration=slow_duration,
                    )
                )

        mage_multiplier = float(mage_auras[global_slot])
        if mage_multiplier > 1.0:
            auras.append(
                AuraCueVisual(
                    global_slot=global_slot,
                    kind="mage_amplification",
                    multiplier=mage_multiplier,
                )
            )

        warrior_multiplier = float(warrior_auras[global_slot])
        if warrior_multiplier < 1.0:
            auras.append(
                AuraCueVisual(
                    global_slot=global_slot,
                    kind="warrior_mitigation",
                    multiplier=warrior_multiplier,
                )
            )

        anti_heal_duration = int(state.rogue_poison_anti_heal_durations[global_slot])
        if anti_heal_duration > 0:
            persistent_effects.append(
                PersistentEffectVisual(
                    global_slot=global_slot,
                    kind="rogue_anti_heal",
                    duration=anti_heal_duration,
                    magnitude=ROGUE_POISON_ANTI_HEAL_MULTIPLIER,
                )
            )

        freedom_duration = int(
            state.priest_blessing_of_freedom_slow_floor_durations[global_slot]
        )
        if freedom_duration > 0:
            persistent_effects.append(
                PersistentEffectVisual(
                    global_slot=global_slot,
                    kind="priest_freedom",
                    duration=freedom_duration,
                    magnitude=PRIEST_HEAL_SPEED_FLOOR,
                )
            )

        burst_duration = int(
            state.mage_burst_damage_amplification_durations[global_slot]
        )
        if burst_duration > 0:
            persistent_effects.append(
                PersistentEffectVisual(
                    global_slot=global_slot,
                    kind="mage_burst",
                    duration=burst_duration,
                    magnitude=MAGE_BURST_DAMAGE_MULTIPLIER,
                )
            )

    return BattlefieldOverlays(
        statuses=tuple(statuses),
        auras=tuple(auras),
        persistent_effects=tuple(persistent_effects),
    )


def merge_battlefield_overlays(
    *overlays: BattlefieldOverlays,
) -> BattlefieldOverlays:
    """Concatenate every overlay family without deriving new semantics."""
    merged: dict[str, tuple[object, ...]] = {
        field.name: () for field in fields(BattlefieldOverlays)
    }
    for overlay in overlays:
        for field in fields(BattlefieldOverlays):
            merged[field.name] += getattr(overlay, field.name)

    return BattlefieldOverlays(**merged)  # pyright: ignore[reportArgumentType]
