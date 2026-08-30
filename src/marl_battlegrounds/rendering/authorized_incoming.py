"""Recipient-safe incoming presentation summaries for Agent POV.

NoSharedObs summaries preserve the canonical recipient-local cue inventory and
enrich it only with facts independently authorized at the adjacent endpoints.
SharedObs summaries are deliberately weaker: they report deterministic
observation deltas and observation provenance, never scientific event causes.

This module owns no replay transport, HTTP, simulator, JAX, NumPy, Oracle
event, or raw-frame dependency.  Its public NoSharedObs entry points accept
either a validated replay projection index or the exact live adjacent carrier,
so arbitrary standalone transitions/endpoints can never be presented as a
complete cue inventory.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Annotated, ClassVar, Literal, TypedDict, cast

from pydantic import ConfigDict, Field, TypeAdapter

from marl_battlegrounds.evaluation.models import StaticMechanicsCatalogV1
from marl_battlegrounds.evaluation.pov import (
    ActorPovAdjacentTransitionSliceV1,
    ActorPovAxisMappingV1,
    ActorPovEpisodeEndedCueV1,
    ActorPovOwnActionOutcomeCueV1,
    ActorPovOwnCooldownChangedCueV1,
    ActorPovOwnHealthChangedCueV1,
    ActorPovOwnLifecycleChangedCueV1,
    ActorPovOwnPositionChangedCueV1,
    ActorPovOwnStatusChangedCueV1,
    ActorPovPresentationCueV1,
    ActorPovTransitionV1,
    ActorPovVisibleBodyObservationChangedCueV1,
)
from marl_battlegrounds.rendering.authorized_pov_scene import (
    NoSharedObsAuthorizedScenePartsV1,
    SharedObsAgentObservationProvenanceV1,
    SharedObsAuthorizedScenePartsV1,
    SharedObsAuthorizedSensorSourceV1,
    build_no_shared_obs_authorized_scene_v1,
)
from marl_battlegrounds.rendering.authorized_presentation import (
    AUTHORIZED_PRESENTATION_SCHEMA_VERSION,
    AuthorizedAgentV1,
    AuthorizedAuraFieldV1,
    AuthorizedAuraModifierV1,
    AuthorizedBattlefieldSceneV1,
    AuthorizedStatusV1,
)
from marl_battlegrounds.rendering.pov_scene import ActorPovProjectionIndexV1
from marl_battlegrounds.rendering.vocabulary import (
    CATALOG_STATUS_ID_BY_CHANNEL,
    status_sort_key,
    status_token_id_from_catalog_status_id,
)

_STRICT_WIRE_CONFIG = ConfigDict(
    allow_inf_nan=False,
    extra="forbid",
    strict=True,
)

type AgentIncomingRelationV1 = Literal["self", "ally", "opponent"]
type AgentIncomingLifeStateV1 = Literal["alive", "corpse"]
type AgentIncomingStatusFamilyV1 = Literal[
    "slow",
    "stun",
    "anti_heal",
    "damage_amplification",
    "movement_floor",
]
type AgentIncomingMagnitudeKindV1 = Literal[
    "movement_multiplier",
    "none",
    "healing_multiplier",
    "damage_multiplier",
    "movement_floor",
]
type AgentIncomingActionComponentV1 = Literal["basic", "ultimate"]
type SharedObsDynamicFieldV1 = Literal[
    "position",
    "life_state",
    "current_health",
    "effective_movement_speed",
    "ultimate_cooldown_remaining",
    "spawn_shield_remaining",
    "steps_until_out_of_combat",
    "statuses",
    "aura_modifiers",
]

_DYNAMIC_FIELD_ORDER: tuple[SharedObsDynamicFieldV1, ...] = (
    "position",
    "life_state",
    "current_health",
    "effective_movement_speed",
    "ultimate_cooldown_remaining",
    "spawn_shield_remaining",
    "steps_until_out_of_combat",
    "statuses",
    "aura_modifiers",
)
_CLASS_NAME_BY_ID = {
    1: "Mage",
    2: "Warrior",
    3: "Hunter",
    4: "Rogue",
    5: "Priest",
}


def _require_text(value: str, *, name: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a non-empty Python string.")


def _require_int(value: int, *, name: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be a Python int >= {minimum}.")


def _require_float(value: float, *, name: str, minimum: float | None = None) -> None:
    if type(value) is not float or not isfinite(value):
        raise ValueError(f"{name} must be a finite Python float.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")


def _require_point(value: tuple[float, float], *, name: str) -> None:
    if type(value) is not tuple or len(value) != 2:
        raise ValueError(f"{name} must be an exact two-coordinate tuple.")
    for coordinate in value:
        _require_float(coordinate, name=f"{name} coordinate")


def _require_bool(value: bool, *, name: str) -> None:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a Python bool.")


def _require_opaque_pov_key(value: str, *, name: str) -> None:
    _require_text(value, name=name)
    digest = value.removeprefix("pov_")
    if (
        not value.startswith("pov_")
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{name} must be one canonical opaque POV key.")


def _require_exact_tuple(
    value: tuple[object, ...],
    *,
    name: str,
    item_types: type[object] | tuple[type[object], ...],
) -> None:
    if type(value) is not tuple:
        raise ValueError(f"{name} must be a Python tuple.")
    if any(type(item) not in _as_type_tuple(item_types) for item in value):
        raise ValueError(f"{name} contains a non-canonical item type.")


def _as_type_tuple(
    item_types: type[object] | tuple[type[object], ...],
) -> tuple[type[object], ...]:
    if isinstance(item_types, tuple):
        return item_types
    return (item_types,)


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentIncomingObservedStatusV1:
    """One endpoint-authorized status without emitter/source evidence."""

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_CONFIG

    status_channel: int
    status_id: str
    family: AgentIncomingStatusFamilyV1
    configured_duration_steps: int
    remaining_duration: int
    mechanic_action_component: AgentIncomingActionComponentV1
    magnitude_kind: AgentIncomingMagnitudeKindV1
    magnitude: float | None
    breaks_on_positive_damage: bool

    def __post_init__(self) -> None:
        _require_int(self.status_channel, name="status_channel")
        if self.status_channel >= 9:
            raise ValueError("status_channel must remain on the V1 status axis.")
        _require_text(self.status_id, name="status_id")
        if CATALOG_STATUS_ID_BY_CHANNEL[self.status_channel] != self.status_id:
            raise ValueError("incoming status channel and ID are not canonical.")
        if self.family not in (
            "slow",
            "stun",
            "anti_heal",
            "damage_amplification",
            "movement_floor",
        ):
            raise ValueError("unknown incoming status family.")
        _require_int(
            self.configured_duration_steps,
            name="configured_duration_steps",
            minimum=1,
        )
        _require_int(self.remaining_duration, name="remaining_duration", minimum=1)
        if self.remaining_duration > self.configured_duration_steps:
            raise ValueError("status remaining duration exceeds its configuration.")
        if self.mechanic_action_component not in ("basic", "ultimate"):
            raise ValueError("unknown incoming status action component.")
        if self.magnitude_kind not in (
            "movement_multiplier",
            "none",
            "healing_multiplier",
            "damage_multiplier",
            "movement_floor",
        ):
            raise ValueError("unknown incoming status magnitude kind.")
        if self.magnitude is None:
            if self.magnitude_kind != "none":
                raise ValueError("non-none magnitude kind requires a value.")
        else:
            _require_float(self.magnitude, name="magnitude")
            if self.magnitude_kind == "none":
                raise ValueError("none magnitude kind must omit its value.")
        _require_bool(
            self.breaks_on_positive_damage,
            name="breaks_on_positive_damage",
        )


def _require_canonical_status_order(
    statuses: tuple[AgentIncomingObservedStatusV1, ...],
    *,
    name: str,
) -> None:
    channels = tuple(status.status_channel for status in statuses)
    presentation_keys = tuple(
        status_sort_key(status_token_id_from_catalog_status_id(status.status_id))
        for status in statuses
    )
    if len(channels) != len(set(channels)) or presentation_keys != tuple(
        sorted(presentation_keys)
    ):
        raise ValueError(f"{name} must use unique canonical status order.")


@dataclass(frozen=True, slots=True, kw_only=True)
class AgentIncomingObservationV1:
    """One generic endpoint observation stripped of causal source claims."""

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_CONFIG

    presentation_key: str
    public_agent_id: str
    relation: AgentIncomingRelationV1
    team_id: int
    class_id: int
    class_name: str
    position: tuple[float, float]
    radius: float
    life_state: AgentIncomingLifeStateV1
    current_health: float
    maximum_health: float
    base_movement_speed: float
    effective_movement_speed: float
    observation_radius: float
    basic_interaction_radius: float
    ultimate_interaction_radius: float
    ultimate_cooldown_remaining: int
    spawn_shield_remaining: int
    steps_until_out_of_combat: int
    out_of_combat_delay_steps: int
    out_of_combat_health_regeneration_fraction_per_step: float
    statuses: tuple[AgentIncomingObservedStatusV1, ...]
    aura_modifiers: tuple[AuthorizedAuraModifierV1, ...]

    def __post_init__(self) -> None:
        _require_opaque_pov_key(self.presentation_key, name="presentation_key")
        _require_text(self.public_agent_id, name="public_agent_id")
        if self.relation not in ("self", "ally", "opponent"):
            raise ValueError("Agent incoming observations cannot use Oracle relation.")
        _require_int(self.team_id, name="team_id", minimum=1)
        if self.team_id not in (1, 2):
            raise ValueError("team_id must be one or two.")
        _require_int(self.class_id, name="class_id", minimum=1)
        if self.class_id > 5:
            raise ValueError("class_id must be in the V1 class domain.")
        _require_text(self.class_name, name="class_name")
        if _CLASS_NAME_BY_ID[self.class_id] != self.class_name:
            raise ValueError("incoming class ID and name are not canonical.")
        _require_point(self.position, name="position")
        for name in (
            "radius",
            "current_health",
            "maximum_health",
            "base_movement_speed",
            "effective_movement_speed",
            "observation_radius",
            "basic_interaction_radius",
            "ultimate_interaction_radius",
            "out_of_combat_health_regeneration_fraction_per_step",
        ):
            _require_float(cast(float, getattr(self, name)), name=name, minimum=0.0)
        if self.radius <= 0.0 or self.maximum_health <= 0.0:
            raise ValueError("agent radius and maximum health must be positive.")
        if self.current_health > self.maximum_health:
            raise ValueError("current health cannot exceed maximum health.")
        if self.life_state not in ("alive", "corpse"):
            raise ValueError("unknown incoming life state.")
        for name in (
            "ultimate_cooldown_remaining",
            "spawn_shield_remaining",
            "steps_until_out_of_combat",
            "out_of_combat_delay_steps",
        ):
            _require_int(cast(int, getattr(self, name)), name=name)
        if self.steps_until_out_of_combat > self.out_of_combat_delay_steps:
            raise ValueError("combat countdown exceeds its configured delay.")
        if self.out_of_combat_health_regeneration_fraction_per_step > 1.0:
            raise ValueError("incoming regeneration fraction cannot exceed one.")
        _require_exact_tuple(
            cast(tuple[object, ...], self.statuses),
            name="statuses",
            item_types=AgentIncomingObservedStatusV1,
        )
        _require_canonical_status_order(self.statuses, name="incoming statuses")
        _require_exact_tuple(
            cast(tuple[object, ...], self.aura_modifiers),
            name="aura_modifiers",
            item_types=AuthorizedAuraModifierV1,
        )
        aura_ids = tuple(row.aura_id for row in self.aura_modifiers)
        if aura_ids != tuple(sorted(aura_ids)) or len(aura_ids) != len(set(aura_ids)):
            raise ValueError(
                "incoming aura modifiers must have canonical unique identities."
            )
        if any(row.multiplier == 1.0 for row in self.aura_modifiers):
            raise ValueError("neutral incoming aura modifiers must be omitted.")


def _validate_local_cue_identity(
    *,
    cue_type: str,
    cue_id: str,
    pov_transition_id: str,
    ordinal: int,
) -> None:
    _require_text(cue_type, name="cue_type")
    _require_text(cue_id, name="cue_id")
    _require_text(pov_transition_id, name="pov_transition_id")
    _require_int(ordinal, name="ordinal")
    if cue_id != f"{pov_transition_id}:cue:{ordinal}":
        raise ValueError("NoSharedObs cue ID is not recipient-local canonical.")


@dataclass(frozen=True, slots=True, kw_only=True)
class NoSharedObsOwnActionOutcomeIncomingCueV1:
    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_CONFIG

    cue_type: Literal["own_action_outcome"]
    cue_id: str
    pov_transition_id: str
    ordinal: int
    outcome: Literal["accepted", "rejected"]

    def __post_init__(self) -> None:
        _validate_local_cue_identity(
            cue_type=self.cue_type,
            cue_id=self.cue_id,
            pov_transition_id=self.pov_transition_id,
            ordinal=self.ordinal,
        )
        if self.cue_type != "own_action_outcome" or self.outcome not in (
            "accepted",
            "rejected",
        ):
            raise ValueError("invalid own-action outcome cue.")


@dataclass(frozen=True, slots=True, kw_only=True)
class NoSharedObsOwnPositionChangedIncomingCueV1:
    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_CONFIG

    cue_type: Literal["own_position_changed"]
    cue_id: str
    pov_transition_id: str
    ordinal: int
    start_position: tuple[float, float]
    successor_position: tuple[float, float]

    def __post_init__(self) -> None:
        _validate_local_cue_identity(
            cue_type=self.cue_type,
            cue_id=self.cue_id,
            pov_transition_id=self.pov_transition_id,
            ordinal=self.ordinal,
        )
        if self.cue_type != "own_position_changed":
            raise ValueError("invalid own-position cue discriminator.")
        _require_point(self.start_position, name="start_position")
        _require_point(self.successor_position, name="successor_position")
        if self.start_position == self.successor_position:
            raise ValueError("position-change cue requires changed positions.")


@dataclass(frozen=True, slots=True, kw_only=True)
class NoSharedObsOwnHealthChangedIncomingCueV1:
    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_CONFIG

    cue_type: Literal["own_health_changed"]
    cue_id: str
    pov_transition_id: str
    ordinal: int
    start_health: float
    successor_health: float

    def __post_init__(self) -> None:
        _validate_local_cue_identity(
            cue_type=self.cue_type,
            cue_id=self.cue_id,
            pov_transition_id=self.pov_transition_id,
            ordinal=self.ordinal,
        )
        if self.cue_type != "own_health_changed":
            raise ValueError("invalid own-health cue discriminator.")
        _require_float(self.start_health, name="start_health", minimum=0.0)
        _require_float(self.successor_health, name="successor_health", minimum=0.0)
        if self.start_health == self.successor_health:
            raise ValueError("health-change cue requires changed health.")


@dataclass(frozen=True, slots=True, kw_only=True)
class NoSharedObsOwnStatusChangedIncomingCueV1:
    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_CONFIG

    cue_type: Literal["own_status_changed"]
    cue_id: str
    pov_transition_id: str
    ordinal: int
    start_statuses: tuple[AgentIncomingObservedStatusV1, ...]
    successor_statuses: tuple[AgentIncomingObservedStatusV1, ...]

    def __post_init__(self) -> None:
        _validate_local_cue_identity(
            cue_type=self.cue_type,
            cue_id=self.cue_id,
            pov_transition_id=self.pov_transition_id,
            ordinal=self.ordinal,
        )
        if self.cue_type != "own_status_changed":
            raise ValueError("invalid own-status cue discriminator.")
        for name in ("start_statuses", "successor_statuses"):
            value = cast(tuple[object, ...], getattr(self, name))
            _require_exact_tuple(
                value,
                name=name,
                item_types=AgentIncomingObservedStatusV1,
            )
            _require_canonical_status_order(
                cast(tuple[AgentIncomingObservedStatusV1, ...], value),
                name=name,
            )
        if self.start_statuses == self.successor_statuses:
            raise ValueError("status-change cue requires changed status observations.")
        start_by_channel = {
            status.status_channel: status for status in self.start_statuses
        }
        successor_by_channel = {
            status.status_channel: status for status in self.successor_statuses
        }
        if any(
            _status_static_profile(start_by_channel[channel])
            != _status_static_profile(successor_by_channel[channel])
            for channel in start_by_channel.keys() & successor_by_channel.keys()
        ):
            raise ValueError("retained status cue changed static mechanic profile.")


@dataclass(frozen=True, slots=True, kw_only=True)
class NoSharedObsOwnCooldownChangedIncomingCueV1:
    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_CONFIG

    cue_type: Literal["own_cooldown_changed"]
    cue_id: str
    pov_transition_id: str
    ordinal: int
    start_remaining_ticks: int
    successor_remaining_ticks: int

    def __post_init__(self) -> None:
        _validate_local_cue_identity(
            cue_type=self.cue_type,
            cue_id=self.cue_id,
            pov_transition_id=self.pov_transition_id,
            ordinal=self.ordinal,
        )
        if self.cue_type != "own_cooldown_changed":
            raise ValueError("invalid own-cooldown cue discriminator.")
        _require_int(self.start_remaining_ticks, name="start_remaining_ticks")
        _require_int(
            self.successor_remaining_ticks,
            name="successor_remaining_ticks",
        )
        if self.start_remaining_ticks == self.successor_remaining_ticks:
            raise ValueError("cooldown-change cue requires changed values.")


@dataclass(frozen=True, slots=True, kw_only=True)
class NoSharedObsOwnLifecycleChangedIncomingCueV1:
    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_CONFIG

    cue_type: Literal["own_lifecycle_changed"]
    cue_id: str
    pov_transition_id: str
    ordinal: int
    start_active: bool
    successor_active: bool
    start_life_state: AgentIncomingLifeStateV1
    successor_life_state: AgentIncomingLifeStateV1
    start_spawn_shield_remaining_ticks: int
    successor_spawn_shield_remaining_ticks: int

    def __post_init__(self) -> None:
        _validate_local_cue_identity(
            cue_type=self.cue_type,
            cue_id=self.cue_id,
            pov_transition_id=self.pov_transition_id,
            ordinal=self.ordinal,
        )
        if self.cue_type != "own_lifecycle_changed":
            raise ValueError("invalid own-lifecycle cue discriminator.")
        _require_bool(self.start_active, name="start_active")
        _require_bool(self.successor_active, name="successor_active")
        if not self.start_active or not self.successor_active:
            raise ValueError("configured recipient lifecycle must remain active.")
        if self.start_life_state not in ("alive", "corpse") or (
            self.successor_life_state not in ("alive", "corpse")
        ):
            raise ValueError("unknown lifecycle state.")
        _require_int(
            self.start_spawn_shield_remaining_ticks,
            name="start_spawn_shield_remaining_ticks",
        )
        _require_int(
            self.successor_spawn_shield_remaining_ticks,
            name="successor_spawn_shield_remaining_ticks",
        )
        if (
            self.start_active,
            self.start_life_state,
            self.start_spawn_shield_remaining_ticks,
        ) == (
            self.successor_active,
            self.successor_life_state,
            self.successor_spawn_shield_remaining_ticks,
        ):
            raise ValueError("lifecycle cue requires changed lifecycle facts.")


@dataclass(frozen=True, slots=True, kw_only=True)
class NoSharedObsVisibleBodyChangedIncomingCueV1:
    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_CONFIG

    cue_type: Literal["visible_body_observation_changed"]
    cue_id: str
    pov_transition_id: str
    ordinal: int
    observation_change_kind: Literal[
        "appearance",
        "disappearance",
        "observed_values_change",
    ]
    agent_presentation_key: str
    agent_public_agent_id: str
    observed_payload_changed: bool
    start_observation: AgentIncomingObservationV1 | None
    successor_observation: AgentIncomingObservationV1 | None

    def __post_init__(self) -> None:
        _validate_local_cue_identity(
            cue_type=self.cue_type,
            cue_id=self.cue_id,
            pov_transition_id=self.pov_transition_id,
            ordinal=self.ordinal,
        )
        if self.cue_type != "visible_body_observation_changed":
            raise ValueError("invalid visible-body cue discriminator.")
        if self.observation_change_kind not in (
            "appearance",
            "disappearance",
            "observed_values_change",
        ):
            raise ValueError("unknown NoSharedObs body change kind.")
        _require_opaque_pov_key(
            self.agent_presentation_key,
            name="agent_presentation_key",
        )
        _require_text(self.agent_public_agent_id, name="agent_public_agent_id")
        _require_bool(self.observed_payload_changed, name="observed_payload_changed")
        expected_presence = {
            "appearance": (False, True),
            "disappearance": (True, False),
            "observed_values_change": (True, True),
        }[self.observation_change_kind]
        actual_presence = (
            self.start_observation is not None,
            self.successor_observation is not None,
        )
        if actual_presence != expected_presence:
            raise ValueError("body change kind and endpoint observations disagree.")
        endpoint_rows = tuple(
            row
            for row in (self.start_observation, self.successor_observation)
            if row is not None
        )
        if any(type(row) is not AgentIncomingObservationV1 for row in endpoint_rows):
            raise ValueError("body endpoints require exact incoming observations.")
        if any(
            row.presentation_key != self.agent_presentation_key
            or row.public_agent_id != self.agent_public_agent_id
            for row in endpoint_rows
        ):
            raise ValueError("body endpoint observations do not join cue identity.")
        if self.observation_change_kind == "observed_values_change":
            if not self.observed_payload_changed:
                raise ValueError("retained-body cue requires a changed payload.")
            if self.start_observation == self.successor_observation:
                raise ValueError("retained-body endpoint observations did not change.")
            if self.start_observation is None or self.successor_observation is None:
                raise AssertionError("retained-body endpoint observations disappeared.")
            _validate_incoming_observation_static_profile(
                self.start_observation,
                self.successor_observation,
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class NoSharedObsEpisodeEndedIncomingCueV1:
    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_CONFIG

    cue_type: Literal["episode_ended"]
    cue_id: str
    pov_transition_id: str
    ordinal: int
    terminated: bool
    truncated: bool
    public_end_reason: str | None

    def __post_init__(self) -> None:
        _validate_local_cue_identity(
            cue_type=self.cue_type,
            cue_id=self.cue_id,
            pov_transition_id=self.pov_transition_id,
            ordinal=self.ordinal,
        )
        if self.cue_type != "episode_ended":
            raise ValueError("invalid episode-ended cue discriminator.")
        _require_bool(self.terminated, name="terminated")
        _require_bool(self.truncated, name="truncated")
        if not (self.terminated or self.truncated):
            raise ValueError("episode-ended cue requires a done flag.")
        if self.public_end_reason is not None:
            _require_text(self.public_end_reason, name="public_end_reason")


type NoSharedObsIncomingCueV1 = Annotated[
    NoSharedObsOwnActionOutcomeIncomingCueV1
    | NoSharedObsOwnPositionChangedIncomingCueV1
    | NoSharedObsOwnHealthChangedIncomingCueV1
    | NoSharedObsOwnStatusChangedIncomingCueV1
    | NoSharedObsOwnCooldownChangedIncomingCueV1
    | NoSharedObsOwnLifecycleChangedIncomingCueV1
    | NoSharedObsVisibleBodyChangedIncomingCueV1
    | NoSharedObsEpisodeEndedIncomingCueV1,
    Field(discriminator="cue_type"),
]

_NO_SHARED_CUE_TYPES: tuple[type[object], ...] = (
    NoSharedObsOwnActionOutcomeIncomingCueV1,
    NoSharedObsOwnPositionChangedIncomingCueV1,
    NoSharedObsOwnHealthChangedIncomingCueV1,
    NoSharedObsOwnStatusChangedIncomingCueV1,
    NoSharedObsOwnCooldownChangedIncomingCueV1,
    NoSharedObsOwnLifecycleChangedIncomingCueV1,
    NoSharedObsVisibleBodyChangedIncomingCueV1,
    NoSharedObsEpisodeEndedIncomingCueV1,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class NoSharedObsIncomingSummaryV1:
    """Exact recipient-local cues entering one NoSharedObs replay frame."""

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_CONFIG

    schema_version: Literal[1]
    summary_kind: Literal["no_shared_obs_recipient_cues"]
    source_episode_id: str
    recipient_public_agent_id: str
    recipient_presentation_key: str
    incoming_transition_index: int
    incoming_recipient_transition_id: str
    incoming_start_recipient_frame_id: str
    incoming_successor_recipient_frame_id: str
    incoming_start_simulator_step_count: int
    incoming_successor_simulator_step_count: int
    cues: tuple[NoSharedObsIncomingCueV1, ...]
    cue_count: int

    def __post_init__(self) -> None:
        if self.schema_version != AUTHORIZED_PRESENTATION_SCHEMA_VERSION:
            raise ValueError("unknown Agent incoming schema version.")
        if self.summary_kind != "no_shared_obs_recipient_cues":
            raise ValueError("unknown NoSharedObs incoming summary kind.")
        _require_text(self.source_episode_id, name="source_episode_id")
        _require_text(
            self.recipient_public_agent_id,
            name="recipient_public_agent_id",
        )
        _require_opaque_pov_key(
            self.recipient_presentation_key,
            name="recipient_presentation_key",
        )
        _require_int(self.incoming_transition_index, name="incoming_transition_index")
        expected_transition_id = (
            f"{self.source_episode_id}:actor-pov:"
            f"{self.recipient_public_agent_id}:transition:"
            f"{self.incoming_transition_index}"
        )
        if self.incoming_recipient_transition_id != expected_transition_id:
            raise ValueError("NoSharedObs incoming transition ID is not canonical.")
        expected_start_id = (
            f"{self.source_episode_id}:actor-pov:"
            f"{self.recipient_public_agent_id}:frame:"
            f"{self.incoming_transition_index}"
        )
        expected_successor_id = (
            f"{self.source_episode_id}:actor-pov:"
            f"{self.recipient_public_agent_id}:frame:"
            f"{self.incoming_transition_index + 1}"
        )
        if (
            self.incoming_start_recipient_frame_id != expected_start_id
            or self.incoming_successor_recipient_frame_id != expected_successor_id
        ):
            raise ValueError("NoSharedObs incoming endpoint IDs are not canonical.")
        for name in (
            "incoming_start_simulator_step_count",
            "incoming_successor_simulator_step_count",
            "cue_count",
        ):
            _require_int(cast(int, getattr(self, name)), name=name)
        if self.incoming_successor_simulator_step_count != (
            self.incoming_start_simulator_step_count + 1
        ):
            raise ValueError("NoSharedObs incoming endpoint ticks are not adjacent.")
        _require_exact_tuple(
            cast(tuple[object, ...], self.cues),
            name="cues",
            item_types=_NO_SHARED_CUE_TYPES,
        )
        if not self.cues or self.cue_count != len(self.cues):
            raise ValueError("NoSharedObs incoming cue count is not exact.")
        if self.cues[0].cue_type != "own_action_outcome":
            raise ValueError(
                "NoSharedObs cue inventory must begin with action outcome."
            )
        for ordinal, cue in enumerate(self.cues):
            if (
                cue.ordinal != ordinal
                or cue.pov_transition_id != self.incoming_recipient_transition_id
                or cue.cue_id
                != f"{self.incoming_recipient_transition_id}:cue:{ordinal}"
            ):
                raise ValueError("NoSharedObs cue inventory is not exact and ordered.")
        family_rank = {
            "own_action_outcome": 0,
            "own_position_changed": 1,
            "own_health_changed": 2,
            "own_status_changed": 3,
            "own_cooldown_changed": 4,
            "own_lifecycle_changed": 5,
            "visible_body_observation_changed": 6,
            "episode_ended": 7,
        }
        ranks = tuple(family_rank[cue.cue_type] for cue in self.cues)
        if ranks != tuple(sorted(ranks)):
            raise ValueError("NoSharedObs cue families are not canonically ordered.")
        singleton_types = (
            "own_action_outcome",
            "own_position_changed",
            "own_health_changed",
            "own_status_changed",
            "own_cooldown_changed",
            "own_lifecycle_changed",
            "episode_ended",
        )
        for cue_type in singleton_types:
            count = sum(cue.cue_type == cue_type for cue in self.cues)
            expected_maximum = 1
            if count > expected_maximum or (
                cue_type == "own_action_outcome" and count != 1
            ):
                raise ValueError("NoSharedObs cue kind multiplicity is invalid.")
        episode_rows = tuple(
            cue for cue in self.cues if cue.cue_type == "episode_ended"
        )
        if episode_rows and self.cues[-1].cue_type != "episode_ended":
            raise ValueError("NoSharedObs episode-ended cue must be final.")
        body_public_ids = tuple(
            cue.agent_public_agent_id
            for cue in self.cues
            if type(cue) is NoSharedObsVisibleBodyChangedIncomingCueV1
        )
        if len(body_public_ids) != len(set(body_public_ids)):
            raise ValueError("NoSharedObs body cue identities must be unique.")
        body_keys = tuple(
            cue.agent_presentation_key
            for cue in self.cues
            if type(cue) is NoSharedObsVisibleBodyChangedIncomingCueV1
        )
        if len(body_keys) != len(set(body_keys)):
            raise ValueError("NoSharedObs body cue keys must be unique.")
        for cue in self.cues:
            if type(cue) is not NoSharedObsVisibleBodyChangedIncomingCueV1:
                continue
            is_recipient_public = (
                cue.agent_public_agent_id == self.recipient_public_agent_id
            )
            is_recipient_key = (
                cue.agent_presentation_key == self.recipient_presentation_key
            )
            if is_recipient_public != is_recipient_key:
                raise ValueError(
                    "NoSharedObs recipient body identity is not bijective."
                )
            rows = tuple(
                row
                for row in (cue.start_observation, cue.successor_observation)
                if row is not None
            )
            if is_recipient_public:
                if (
                    cue.observation_change_kind != "observed_values_change"
                    or len(rows) != 2
                    or any(row.relation != "self" for row in rows)
                ):
                    raise ValueError(
                        "recipient self-body cue must be retained self observation."
                    )
            elif any(row.relation not in ("ally", "opponent") for row in rows):
                raise ValueError("nonrecipient body cue has invalid relation.")


def _validate_shared_delta_identity(
    *,
    delta_kind: str,
    cue_id: str,
    recipient_transition_id: str,
    ordinal: int,
    agent_presentation_key: str,
    agent_public_agent_id: str,
) -> None:
    _require_text(delta_kind, name="delta_kind")
    _require_text(cue_id, name="cue_id")
    _require_text(recipient_transition_id, name="recipient_transition_id")
    _require_int(ordinal, name="ordinal")
    if cue_id != f"{recipient_transition_id}:cue:{ordinal}":
        raise ValueError("SharedObs cue ID is not recipient-local canonical.")
    _require_opaque_pov_key(
        agent_presentation_key,
        name="agent_presentation_key",
    )
    _require_text(agent_public_agent_id, name="agent_public_agent_id")


def _validate_sources(
    sources: tuple[SharedObsAuthorizedSensorSourceV1, ...],
    *,
    name: str,
) -> None:
    _require_exact_tuple(
        cast(tuple[object, ...], sources),
        name=name,
        item_types=SharedObsAuthorizedSensorSourceV1,
    )
    if not sources:
        raise ValueError(f"{name} must retain at least one observation source.")
    expected = tuple(
        sorted(
            sources,
            key=lambda row: (
                0 if row.source_kind == "recipient_base" else 1,
                row.source_public_agent_id,
            ),
        )
    )
    if sources != expected:
        raise ValueError(f"{name} is not canonical.")
    public_ids = tuple(source.source_public_agent_id for source in sources)
    keys = tuple(source.source_presentation_key for source in sources)
    if len(public_ids) != len(set(public_ids)) or len(keys) != len(set(keys)):
        raise ValueError(f"{name} repeats an observation source identity.")


@dataclass(frozen=True, slots=True, kw_only=True)
class SharedObsAppearanceIncomingDeltaV1:
    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_CONFIG

    delta_kind: Literal["appearance"]
    cue_id: str
    recipient_transition_id: str
    ordinal: int
    agent_presentation_key: str
    agent_public_agent_id: str
    successor_observation: AgentIncomingObservationV1
    successor_observation_sources: tuple[SharedObsAuthorizedSensorSourceV1, ...]

    def __post_init__(self) -> None:
        _validate_shared_delta_identity(
            delta_kind=self.delta_kind,
            cue_id=self.cue_id,
            recipient_transition_id=self.recipient_transition_id,
            ordinal=self.ordinal,
            agent_presentation_key=self.agent_presentation_key,
            agent_public_agent_id=self.agent_public_agent_id,
        )
        if self.delta_kind != "appearance":
            raise ValueError("invalid SharedObs appearance discriminator.")
        if type(self.successor_observation) is not AgentIncomingObservationV1:
            raise ValueError("appearance requires an exact successor observation.")
        if (
            self.successor_observation.presentation_key != self.agent_presentation_key
            or self.successor_observation.public_agent_id != self.agent_public_agent_id
        ):
            raise ValueError("appearance observation does not join delta identity.")
        _validate_sources(
            self.successor_observation_sources,
            name="successor_observation_sources",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SharedObsDisappearanceIncomingDeltaV1:
    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_CONFIG

    delta_kind: Literal["disappearance"]
    cue_id: str
    recipient_transition_id: str
    ordinal: int
    agent_presentation_key: str
    agent_public_agent_id: str
    start_observation: AgentIncomingObservationV1
    start_observation_sources: tuple[SharedObsAuthorizedSensorSourceV1, ...]

    def __post_init__(self) -> None:
        _validate_shared_delta_identity(
            delta_kind=self.delta_kind,
            cue_id=self.cue_id,
            recipient_transition_id=self.recipient_transition_id,
            ordinal=self.ordinal,
            agent_presentation_key=self.agent_presentation_key,
            agent_public_agent_id=self.agent_public_agent_id,
        )
        if self.delta_kind != "disappearance":
            raise ValueError("invalid SharedObs disappearance discriminator.")
        if type(self.start_observation) is not AgentIncomingObservationV1:
            raise ValueError("disappearance requires an exact start observation.")
        if (
            self.start_observation.presentation_key != self.agent_presentation_key
            or self.start_observation.public_agent_id != self.agent_public_agent_id
        ):
            raise ValueError("disappearance observation does not join identity.")
        _validate_sources(
            self.start_observation_sources,
            name="start_observation_sources",
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SharedObsObservedValuesIncomingDeltaV1:
    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_CONFIG

    delta_kind: Literal["observed_values_change"]
    cue_id: str
    recipient_transition_id: str
    ordinal: int
    agent_presentation_key: str
    agent_public_agent_id: str
    changed_dynamic_fields: tuple[SharedObsDynamicFieldV1, ...]
    start_observation: AgentIncomingObservationV1
    successor_observation: AgentIncomingObservationV1

    def __post_init__(self) -> None:
        _validate_shared_delta_identity(
            delta_kind=self.delta_kind,
            cue_id=self.cue_id,
            recipient_transition_id=self.recipient_transition_id,
            ordinal=self.ordinal,
            agent_presentation_key=self.agent_presentation_key,
            agent_public_agent_id=self.agent_public_agent_id,
        )
        if self.delta_kind != "observed_values_change":
            raise ValueError("invalid SharedObs observed-values discriminator.")
        if type(self.changed_dynamic_fields) is not tuple or not (
            self.changed_dynamic_fields
        ):
            raise ValueError("observed-values delta requires changed field names.")
        if self.changed_dynamic_fields != tuple(
            name for name in _DYNAMIC_FIELD_ORDER if name in self.changed_dynamic_fields
        ) or len(self.changed_dynamic_fields) != len(set(self.changed_dynamic_fields)):
            raise ValueError("changed dynamic fields are not canonical.")
        for observation in (self.start_observation, self.successor_observation):
            if type(observation) is not AgentIncomingObservationV1 or (
                observation.presentation_key != self.agent_presentation_key
                or observation.public_agent_id != self.agent_public_agent_id
            ):
                raise ValueError("observed-values endpoint does not join identity.")
        if self.start_observation == self.successor_observation:
            raise ValueError("observed-values endpoints must differ.")
        _validate_incoming_observation_static_profile(
            self.start_observation,
            self.successor_observation,
        )
        if (
            _changed_dynamic_fields(
                self.start_observation,
                self.successor_observation,
            )
            != self.changed_dynamic_fields
        ):
            raise ValueError("changed dynamic field inventory is not exact.")


@dataclass(frozen=True, slots=True, kw_only=True)
class SharedObsObservationProvenanceIncomingDeltaV1:
    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_CONFIG

    delta_kind: Literal["observation_provenance_change"]
    cue_id: str
    recipient_transition_id: str
    ordinal: int
    agent_presentation_key: str
    agent_public_agent_id: str
    start_observation_sources: tuple[SharedObsAuthorizedSensorSourceV1, ...]
    successor_observation_sources: tuple[SharedObsAuthorizedSensorSourceV1, ...]

    def __post_init__(self) -> None:
        _validate_shared_delta_identity(
            delta_kind=self.delta_kind,
            cue_id=self.cue_id,
            recipient_transition_id=self.recipient_transition_id,
            ordinal=self.ordinal,
            agent_presentation_key=self.agent_presentation_key,
            agent_public_agent_id=self.agent_public_agent_id,
        )
        if self.delta_kind != "observation_provenance_change":
            raise ValueError("invalid SharedObs provenance discriminator.")
        _validate_sources(
            self.start_observation_sources,
            name="start_observation_sources",
        )
        _validate_sources(
            self.successor_observation_sources,
            name="successor_observation_sources",
        )
        if self.start_observation_sources == self.successor_observation_sources:
            raise ValueError("provenance delta requires changed observation sources.")


type SharedObsIncomingDeltaV1 = Annotated[
    SharedObsAppearanceIncomingDeltaV1
    | SharedObsDisappearanceIncomingDeltaV1
    | SharedObsObservedValuesIncomingDeltaV1
    | SharedObsObservationProvenanceIncomingDeltaV1,
    Field(discriminator="delta_kind"),
]

_SHARED_DELTA_TYPES: tuple[type[object], ...] = (
    SharedObsAppearanceIncomingDeltaV1,
    SharedObsDisappearanceIncomingDeltaV1,
    SharedObsObservedValuesIncomingDeltaV1,
    SharedObsObservationProvenanceIncomingDeltaV1,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class SharedObsIncomingSummaryV1:
    """Generic recipient-local observation deltas entering one SharedObs frame."""

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_WIRE_CONFIG

    schema_version: Literal[1]
    summary_kind: Literal["shared_obs_recipient_observation_deltas"]
    source_episode_id: str
    recipient_public_agent_id: str
    recipient_presentation_key: str
    incoming_transition_index: int
    incoming_recipient_transition_id: str
    incoming_start_recipient_frame_id: str
    incoming_successor_recipient_frame_id: str
    incoming_start_simulator_step_count: int
    incoming_successor_simulator_step_count: int
    deltas: tuple[SharedObsIncomingDeltaV1, ...]
    delta_count: int

    def __post_init__(self) -> None:
        if self.schema_version != AUTHORIZED_PRESENTATION_SCHEMA_VERSION:
            raise ValueError("unknown Agent incoming schema version.")
        if self.summary_kind != "shared_obs_recipient_observation_deltas":
            raise ValueError("unknown SharedObs incoming summary kind.")
        _require_text(self.source_episode_id, name="source_episode_id")
        _require_text(
            self.recipient_public_agent_id,
            name="recipient_public_agent_id",
        )
        _require_opaque_pov_key(
            self.recipient_presentation_key,
            name="recipient_presentation_key",
        )
        _require_int(self.incoming_transition_index, name="incoming_transition_index")
        expected_transition_id = (
            f"{self.source_episode_id}:shared-obs-visual-union:"
            f"{self.recipient_public_agent_id}:transition:"
            f"{self.incoming_transition_index}"
        )
        if self.incoming_recipient_transition_id != expected_transition_id:
            raise ValueError("SharedObs incoming transition ID is not canonical.")
        expected_start_id = (
            f"{self.source_episode_id}:shared-obs-visual-union:"
            f"{self.recipient_public_agent_id}:frame:"
            f"{self.incoming_transition_index}"
        )
        expected_successor_id = (
            f"{self.source_episode_id}:shared-obs-visual-union:"
            f"{self.recipient_public_agent_id}:frame:"
            f"{self.incoming_transition_index + 1}"
        )
        if (
            self.incoming_start_recipient_frame_id != expected_start_id
            or self.incoming_successor_recipient_frame_id != expected_successor_id
        ):
            raise ValueError("SharedObs incoming endpoint IDs are not canonical.")
        for name in (
            "incoming_start_simulator_step_count",
            "incoming_successor_simulator_step_count",
            "delta_count",
        ):
            _require_int(cast(int, getattr(self, name)), name=name)
        if self.incoming_successor_simulator_step_count != (
            self.incoming_start_simulator_step_count + 1
        ):
            raise ValueError("SharedObs incoming endpoint ticks are not adjacent.")
        _require_exact_tuple(
            cast(tuple[object, ...], self.deltas),
            name="deltas",
            item_types=_SHARED_DELTA_TYPES,
        )
        if self.delta_count != len(self.deltas):
            raise ValueError("SharedObs incoming delta count is not exact.")
        for ordinal, delta in enumerate(self.deltas):
            if (
                delta.ordinal != ordinal
                or delta.recipient_transition_id
                != self.incoming_recipient_transition_id
                or delta.cue_id
                != f"{self.incoming_recipient_transition_id}:cue:{ordinal}"
            ):
                raise ValueError("SharedObs delta inventory is not exact and ordered.")
        public_to_key: dict[str, str] = {}
        key_to_public: dict[str, str] = {}
        source_kind_by_public: dict[str, str] = {}
        relation_by_public: dict[str, AgentIncomingRelationV1] = {}
        groups: list[list[SharedObsIncomingDeltaV1]] = []
        seen_public_ids: set[str] = set()
        for delta in self.deltas:
            existing_key = public_to_key.setdefault(
                delta.agent_public_agent_id,
                delta.agent_presentation_key,
            )
            existing_public = key_to_public.setdefault(
                delta.agent_presentation_key,
                delta.agent_public_agent_id,
            )
            if (
                existing_key != delta.agent_presentation_key
                or existing_public != delta.agent_public_agent_id
            ):
                raise ValueError("SharedObs delta identity mapping is not one-to-one.")
            is_recipient_public = (
                delta.agent_public_agent_id == self.recipient_public_agent_id
            )
            is_recipient_key = (
                delta.agent_presentation_key == self.recipient_presentation_key
            )
            if is_recipient_public != is_recipient_key:
                raise ValueError("SharedObs recipient delta identity is not bijective.")
            observations: tuple[AgentIncomingObservationV1, ...]
            source_lists: tuple[tuple[SharedObsAuthorizedSensorSourceV1, ...], ...]
            if type(delta) is SharedObsAppearanceIncomingDeltaV1:
                if is_recipient_public:
                    raise ValueError("SharedObs recipient cannot appear.")
                observations = (delta.successor_observation,)
                source_lists = (delta.successor_observation_sources,)
            elif type(delta) is SharedObsDisappearanceIncomingDeltaV1:
                if is_recipient_public:
                    raise ValueError("SharedObs recipient cannot disappear.")
                observations = (delta.start_observation,)
                source_lists = (delta.start_observation_sources,)
            elif type(delta) is SharedObsObservedValuesIncomingDeltaV1:
                observations = (
                    delta.start_observation,
                    delta.successor_observation,
                )
                source_lists = ()
            elif type(delta) is SharedObsObservationProvenanceIncomingDeltaV1:
                observations = ()
                source_lists = (
                    delta.start_observation_sources,
                    delta.successor_observation_sources,
                )
            else:  # pragma: no cover - exact root union checked above.
                raise TypeError("unknown SharedObs delta variant.")
            if is_recipient_public:
                if any(row.relation != "self" for row in observations):
                    raise ValueError("SharedObs recipient observations must be self.")
            elif any(row.relation == "self" for row in observations):
                raise ValueError("SharedObs nonrecipient observation cannot be self.")
            for observation in observations:
                prior_relation = relation_by_public.setdefault(
                    observation.public_agent_id,
                    observation.relation,
                )
                if prior_relation != observation.relation:
                    raise ValueError(
                        "SharedObs observation relation changed within the summary."
                    )
            for sources in source_lists:
                for source in sources:
                    prior_key = public_to_key.setdefault(
                        source.source_public_agent_id,
                        source.source_presentation_key,
                    )
                    prior_public = key_to_public.setdefault(
                        source.source_presentation_key,
                        source.source_public_agent_id,
                    )
                    prior_kind = source_kind_by_public.setdefault(
                        source.source_public_agent_id,
                        source.source_kind,
                    )
                    if (
                        prior_key != source.source_presentation_key
                        or prior_public != source.source_public_agent_id
                        or prior_kind != source.source_kind
                    ):
                        raise ValueError(
                            "SharedObs source identity is not stable and bijective."
                        )
                    source_is_recipient_public = (
                        source.source_public_agent_id == self.recipient_public_agent_id
                    )
                    source_is_recipient_key = (
                        source.source_presentation_key
                        == self.recipient_presentation_key
                    )
                    if source.source_kind == "recipient_base":
                        if not (source_is_recipient_public and source_is_recipient_key):
                            raise ValueError(
                                "SharedObs recipient-base source identity is not exact."
                            )
                    elif source_is_recipient_public or source_is_recipient_key:
                        raise ValueError(
                            "SharedObs shared source cannot alias recipient identity."
                        )
            if not groups or (
                groups[-1][0].agent_public_agent_id != delta.agent_public_agent_id
            ):
                if delta.agent_public_agent_id in seen_public_ids:
                    raise ValueError(
                        "SharedObs delta identity groups are not contiguous."
                    )
                seen_public_ids.add(delta.agent_public_agent_id)
                groups.append([])
            groups[-1].append(delta)
        for public_id, source_kind in source_kind_by_public.items():
            relation = relation_by_public.get(public_id)
            if relation is None:
                continue
            expected_relation = "self" if source_kind == "recipient_base" else "ally"
            if relation != expected_relation:
                raise ValueError(
                    "SharedObs source kind conflicts with observed relation."
                )
        for group in groups:
            kinds = tuple(delta.delta_kind for delta in group)
            if kinds not in (
                ("appearance",),
                ("disappearance",),
                ("observed_values_change",),
                ("observation_provenance_change",),
                (
                    "observed_values_change",
                    "observation_provenance_change",
                ),
            ):
                raise ValueError("SharedObs per-agent delta group is not canonical.")


def _status_snapshot(status: AuthorizedStatusV1) -> AgentIncomingObservedStatusV1:
    if type(status) is not AuthorizedStatusV1:
        raise ValueError("incoming status source must be an authorized status row.")
    return AgentIncomingObservedStatusV1(
        status_channel=status.status_channel,
        status_id=status.status_id,
        family=status.family,
        configured_duration_steps=status.configured_duration_steps,
        remaining_duration=status.remaining_duration,
        mechanic_action_component=status.source_action_component,
        magnitude_kind=status.magnitude_kind,
        magnitude=status.magnitude,
        breaks_on_positive_damage=status.breaks_on_positive_damage,
    )


def _observation(agent: AuthorizedAgentV1) -> AgentIncomingObservationV1:
    if type(agent) is not AuthorizedAgentV1:
        raise ValueError("incoming observation requires an authorized agent row.")
    if agent.relation == "oracle":
        raise ValueError("Agent incoming summaries cannot consume Oracle agents.")
    return AgentIncomingObservationV1(
        presentation_key=agent.presentation_key,
        public_agent_id=agent.public_agent_id,
        relation=agent.relation,
        team_id=agent.team_id,
        class_id=agent.class_id,
        class_name=agent.class_name,
        position=agent.position,
        radius=agent.radius,
        life_state=agent.life_state,
        current_health=agent.current_health,
        maximum_health=agent.maximum_health,
        base_movement_speed=agent.base_movement_speed,
        effective_movement_speed=agent.effective_movement_speed,
        observation_radius=agent.observation_radius,
        basic_interaction_radius=agent.basic_interaction_radius,
        ultimate_interaction_radius=agent.ultimate_interaction_radius,
        ultimate_cooldown_remaining=agent.ultimate_cooldown_remaining,
        spawn_shield_remaining=agent.spawn_shield_remaining,
        steps_until_out_of_combat=agent.steps_until_out_of_combat,
        out_of_combat_delay_steps=agent.out_of_combat_delay_steps,
        out_of_combat_health_regeneration_fraction_per_step=(
            agent.out_of_combat_health_regeneration_fraction_per_step
        ),
        statuses=tuple(_status_snapshot(status) for status in agent.statuses),
        aura_modifiers=agent.aura_modifiers,
    )


def _static_profile(agent: AuthorizedAgentV1) -> tuple[object, ...]:
    return (
        agent.presentation_key,
        agent.public_agent_id,
        agent.relation,
        agent.team_id,
        agent.class_id,
        agent.class_name,
        agent.radius,
        agent.maximum_health,
        agent.base_movement_speed,
        agent.observation_radius,
        agent.basic_interaction_radius,
        agent.ultimate_interaction_radius,
        agent.out_of_combat_delay_steps,
        agent.out_of_combat_health_regeneration_fraction_per_step,
    )


def _status_static_profile(
    status: AgentIncomingObservedStatusV1,
) -> tuple[object, ...]:
    return (
        status.status_channel,
        status.status_id,
        status.family,
        status.configured_duration_steps,
        status.mechanic_action_component,
        status.magnitude_kind,
        status.magnitude,
        status.breaks_on_positive_damage,
    )


def _incoming_observation_static_profile(
    observation: AgentIncomingObservationV1,
) -> tuple[object, ...]:
    return (
        observation.presentation_key,
        observation.public_agent_id,
        observation.relation,
        observation.team_id,
        observation.class_id,
        observation.class_name,
        observation.radius,
        observation.maximum_health,
        observation.base_movement_speed,
        observation.observation_radius,
        observation.basic_interaction_radius,
        observation.ultimate_interaction_radius,
        observation.out_of_combat_delay_steps,
        observation.out_of_combat_health_regeneration_fraction_per_step,
    )


def _validate_incoming_observation_static_profile(
    start: AgentIncomingObservationV1,
    successor: AgentIncomingObservationV1,
) -> None:
    if _incoming_observation_static_profile(
        start
    ) != _incoming_observation_static_profile(successor):
        raise ValueError("retained incoming observation changed static profile.")
    start_statuses = {status.status_channel: status for status in start.statuses}
    successor_statuses = {
        status.status_channel: status for status in successor.statuses
    }
    if any(
        _status_static_profile(start_statuses[channel])
        != _status_static_profile(successor_statuses[channel])
        for channel in start_statuses.keys() & successor_statuses.keys()
    ):
        raise ValueError("retained incoming status changed static profile.")


def _validate_retained_static_profile(
    start: AuthorizedAgentV1,
    successor: AuthorizedAgentV1,
) -> None:
    if _static_profile(start) != _static_profile(successor):
        raise ValueError("retained Agent observation changed static profile.")
    start_statuses = {
        status.status_channel: _status_snapshot(status) for status in start.statuses
    }
    successor_statuses = {
        status.status_channel: _status_snapshot(status) for status in successor.statuses
    }
    for channel in start_statuses.keys() & successor_statuses.keys():
        if _status_static_profile(start_statuses[channel]) != _status_static_profile(
            successor_statuses[channel]
        ):
            raise ValueError("retained status changed static mechanic profile.")


def _agent_maps(
    agents: tuple[AuthorizedAgentV1, ...],
) -> tuple[dict[str, AuthorizedAgentV1], dict[str, AuthorizedAgentV1]]:
    by_public = {agent.public_agent_id: agent for agent in agents}
    by_key = {agent.presentation_key: agent for agent in agents}
    if len(by_public) != len(agents) or len(by_key) != len(agents):
        raise ValueError("authorized Agent endpoint repeats an identity.")
    return by_public, by_key


def _validate_cross_epoch_agent_identity(
    start_agents: tuple[AuthorizedAgentV1, ...],
    successor_agents: tuple[AuthorizedAgentV1, ...],
) -> tuple[dict[str, AuthorizedAgentV1], dict[str, AuthorizedAgentV1]]:
    start_by_public, start_by_key = _agent_maps(start_agents)
    successor_by_public, successor_by_key = _agent_maps(successor_agents)
    for public_id in start_by_public.keys() & successor_by_public.keys():
        _validate_retained_static_profile(
            start_by_public[public_id],
            successor_by_public[public_id],
        )
    for key in start_by_key.keys() & successor_by_key.keys():
        if start_by_key[key].public_agent_id != successor_by_key[key].public_agent_id:
            raise ValueError("opaque Agent key changed public identity across epochs.")
    return start_by_public, successor_by_public


def _recipient_agent(
    parts: NoSharedObsAuthorizedScenePartsV1,
) -> AuthorizedAgentV1:
    matches = tuple(
        agent
        for agent in parts.scene.agents
        if agent.public_agent_id == parts.recipient_public_agent_id
        and agent.presentation_key == parts.recipient_presentation_key
    )
    if len(matches) != 1 or matches[0].relation != "self":
        raise ValueError("NoSharedObs endpoint lacks its exact recipient body.")
    return matches[0]


def _validate_no_shared_endpoints(
    start: NoSharedObsAuthorizedScenePartsV1,
    transition: ActorPovTransitionV1,
    successor: NoSharedObsAuthorizedScenePartsV1,
) -> None:
    if (
        type(start) is not NoSharedObsAuthorizedScenePartsV1
        or type(successor) is not NoSharedObsAuthorizedScenePartsV1
    ):
        raise TypeError("NoSharedObs incoming compositor requires exact endpoints.")
    if type(transition) is not ActorPovTransitionV1:
        raise TypeError("NoSharedObs incoming compositor requires an exact transition.")
    if (
        start.source_episode_id != successor.source_episode_id
        or transition.episode_id != start.source_episode_id
        or start.recipient_public_agent_id != successor.recipient_public_agent_id
        or transition.public_agent_id != start.recipient_public_agent_id
        or start.recipient_presentation_key != successor.recipient_presentation_key
    ):
        raise ValueError("NoSharedObs adjacent endpoint authority does not join.")
    if (
        successor.source_frame_index != start.source_frame_index + 1
        or transition.transition_index != start.source_frame_index
        or transition.start_pov_frame_id != start.source_recipient_frame_id
        or transition.successor_pov_frame_id != successor.source_recipient_frame_id
    ):
        raise ValueError("NoSharedObs incoming endpoint frame epochs are not adjacent.")
    if successor.source_simulator_step_count != start.source_simulator_step_count + 1:
        raise ValueError("NoSharedObs incoming simulator ticks are not adjacent.")
    _validate_cross_epoch_agent_identity(start.scene.agents, successor.scene.agents)
    _validate_retained_static_profile(
        _recipient_agent(start),
        _recipient_agent(successor),
    )
    _validate_static_scene_continuity(start.scene, successor.scene)


class _CueIdentityV1(TypedDict):
    cue_id: str
    pov_transition_id: str
    ordinal: int


def _cue_identity(cue: ActorPovPresentationCueV1) -> _CueIdentityV1:
    return {
        "cue_id": cue.cue_id,
        "pov_transition_id": cue.pov_transition_id,
        "ordinal": cue.ordinal,
    }


def _body_public_id(
    cue: ActorPovVisibleBodyObservationChangedCueV1,
    axis_mapping: ActorPovAxisMappingV1,
) -> str:
    if cue.relation == "ally":
        return axis_mapping.ally_observation_row_public_agent_id_by_id[
            cue.observation_row
        ]
    return axis_mapping.enemy_observation_row_public_agent_id_by_id[cue.observation_row]


def _visible_observation(
    *,
    public_id: str,
    visible: bool,
    agents: dict[str, AuthorizedAgentV1],
    recipient_public_id: str,
) -> AgentIncomingObservationV1 | None:
    agent = agents.get(public_id)
    if visible:
        if agent is None:
            raise ValueError("visible recipient-local body is absent from endpoint.")
        return _observation(agent)
    if agent is not None and public_id != recipient_public_id:
        raise ValueError("masked recipient-local body leaked into endpoint scene.")
    return None


def _compose_no_shared_obs_incoming_summary_v1(
    start: NoSharedObsAuthorizedScenePartsV1,
    transition: ActorPovTransitionV1,
    successor: NoSharedObsAuthorizedScenePartsV1,
    axis_mapping: ActorPovAxisMappingV1,
) -> NoSharedObsIncomingSummaryV1:
    """Compose from a coherent index-owned transition; intentionally private."""
    _validate_no_shared_endpoints(start, transition, successor)
    if type(axis_mapping) is not ActorPovAxisMappingV1:
        raise TypeError("NoSharedObs cue compositor requires exact axis mapping.")
    start_by_public, successor_by_public = _validate_cross_epoch_agent_identity(
        start.scene.agents,
        successor.scene.agents,
    )
    start_owner = _recipient_agent(start)
    successor_owner = _recipient_agent(successor)
    output: list[NoSharedObsIncomingCueV1] = []
    for cue in transition.cues:
        identity = _cue_identity(cue)
        if type(cue) is ActorPovOwnActionOutcomeCueV1:
            output.append(
                NoSharedObsOwnActionOutcomeIncomingCueV1(
                    cue_type=cue.cue_type,
                    outcome=cue.outcome,
                    **identity,
                )
            )
        elif type(cue) is ActorPovOwnPositionChangedCueV1:
            if (
                cue.start_position != start_owner.position
                or cue.successor_position != successor_owner.position
            ):
                raise ValueError("own position cue does not join authorized endpoints.")
            output.append(
                NoSharedObsOwnPositionChangedIncomingCueV1(
                    cue_type=cue.cue_type,
                    start_position=cue.start_position,
                    successor_position=cue.successor_position,
                    **identity,
                )
            )
        elif type(cue) is ActorPovOwnHealthChangedCueV1:
            if (
                cue.start_health != start_owner.current_health
                or cue.successor_health != successor_owner.current_health
            ):
                raise ValueError("own health cue does not join authorized endpoints.")
            output.append(
                NoSharedObsOwnHealthChangedIncomingCueV1(
                    cue_type=cue.cue_type,
                    start_health=cue.start_health,
                    successor_health=cue.successor_health,
                    **identity,
                )
            )
        elif type(cue) is ActorPovOwnStatusChangedCueV1:
            start_statuses = tuple(
                _status_snapshot(status) for status in start_owner.statuses
            )
            successor_statuses = tuple(
                _status_snapshot(status) for status in successor_owner.statuses
            )
            output.append(
                NoSharedObsOwnStatusChangedIncomingCueV1(
                    cue_type=cue.cue_type,
                    start_statuses=start_statuses,
                    successor_statuses=successor_statuses,
                    **identity,
                )
            )
        elif type(cue) is ActorPovOwnCooldownChangedCueV1:
            start_ticks = start_owner.ultimate_cooldown_remaining
            successor_ticks = successor_owner.ultimate_cooldown_remaining
            if cue.start_remaining_ticks != float(
                start_ticks
            ) or cue.successor_remaining_ticks != float(successor_ticks):
                raise ValueError("own cooldown cue does not join authorized endpoints.")
            output.append(
                NoSharedObsOwnCooldownChangedIncomingCueV1(
                    cue_type=cue.cue_type,
                    start_remaining_ticks=start_ticks,
                    successor_remaining_ticks=successor_ticks,
                    **identity,
                )
            )
        elif type(cue) is ActorPovOwnLifecycleChangedCueV1:
            start_alive = start_owner.life_state == "alive"
            successor_alive = successor_owner.life_state == "alive"
            if (
                not cue.start_active
                or not cue.successor_active
                or cue.start_alive != start_alive
                or cue.successor_alive != successor_alive
                or cue.start_spawn_shield_remaining_ticks
                != start_owner.spawn_shield_remaining
                or cue.successor_spawn_shield_remaining_ticks
                != successor_owner.spawn_shield_remaining
            ):
                raise ValueError(
                    "own lifecycle cue does not join authorized endpoints."
                )
            output.append(
                NoSharedObsOwnLifecycleChangedIncomingCueV1(
                    cue_type=cue.cue_type,
                    start_active=cue.start_active,
                    successor_active=cue.successor_active,
                    start_life_state=start_owner.life_state,
                    successor_life_state=successor_owner.life_state,
                    start_spawn_shield_remaining_ticks=(
                        cue.start_spawn_shield_remaining_ticks
                    ),
                    successor_spawn_shield_remaining_ticks=(
                        cue.successor_spawn_shield_remaining_ticks
                    ),
                    **identity,
                )
            )
        elif type(cue) is ActorPovVisibleBodyObservationChangedCueV1:
            public_id = _body_public_id(cue, axis_mapping)
            is_recipient = public_id == start.recipient_public_agent_id
            if is_recipient:
                # The relation diagonal may be masked for a dead observer, but
                # recipient self_features remain authorized at both endpoints.
                start_observation = _observation(_recipient_agent(start))
                successor_observation = _observation(_recipient_agent(successor))
            else:
                start_observation = _visible_observation(
                    public_id=public_id,
                    visible=cue.start_visible,
                    agents=start_by_public,
                    recipient_public_id=start.recipient_public_agent_id,
                )
                successor_observation = _visible_observation(
                    public_id=public_id,
                    visible=cue.successor_visible,
                    agents=successor_by_public,
                    recipient_public_id=start.recipient_public_agent_id,
                )
            identity_agent = start_by_public.get(public_id) or successor_by_public.get(
                public_id
            )
            if identity_agent is None:
                raise ValueError(
                    "body cue cannot join an authorized endpoint identity."
                )
            if is_recipient:
                change_kind = "observed_values_change"
            elif not cue.start_visible:
                change_kind = "appearance"
            elif not cue.successor_visible:
                change_kind = "disappearance"
            else:
                change_kind = "observed_values_change"
            output.append(
                NoSharedObsVisibleBodyChangedIncomingCueV1(
                    cue_type=cue.cue_type,
                    observation_change_kind=change_kind,
                    agent_presentation_key=identity_agent.presentation_key,
                    agent_public_agent_id=identity_agent.public_agent_id,
                    observed_payload_changed=cue.observed_payload_changed,
                    start_observation=start_observation,
                    successor_observation=successor_observation,
                    **identity,
                )
            )
        elif type(cue) is ActorPovEpisodeEndedCueV1:
            output.append(
                NoSharedObsEpisodeEndedIncomingCueV1(
                    cue_type=cue.cue_type,
                    terminated=cue.terminated,
                    truncated=cue.truncated,
                    public_end_reason=cue.public_end_reason,
                    **identity,
                )
            )
        else:  # pragma: no cover - the validated V1 union is closed.
            raise TypeError("unknown recipient-local POV cue type.")
    return NoSharedObsIncomingSummaryV1(
        schema_version=AUTHORIZED_PRESENTATION_SCHEMA_VERSION,
        summary_kind="no_shared_obs_recipient_cues",
        source_episode_id=start.source_episode_id,
        recipient_public_agent_id=start.recipient_public_agent_id,
        recipient_presentation_key=start.recipient_presentation_key,
        incoming_transition_index=transition.transition_index,
        incoming_recipient_transition_id=transition.pov_transition_id,
        incoming_start_recipient_frame_id=start.source_recipient_frame_id,
        incoming_successor_recipient_frame_id=successor.source_recipient_frame_id,
        incoming_start_simulator_step_count=start.source_simulator_step_count,
        incoming_successor_simulator_step_count=(successor.source_simulator_step_count),
        cues=tuple(output),
        cue_count=len(output),
    )


def build_replay_no_shared_obs_incoming_summary_v1(
    source: ActorPovProjectionIndexV1,
    *,
    successor_frame_index: int,
    public_catalog: StaticMechanicsCatalogV1,
    authority_session_id: str,
) -> NoSharedObsIncomingSummaryV1 | None:
    """Build one replay incoming summary from a complete validated POV prefix."""
    if type(source) is not ActorPovProjectionIndexV1:
        raise TypeError("NoSharedObs replay incoming requires an exact POV index.")
    if type(successor_frame_index) is not int or not (
        0 <= successor_frame_index < len(source.content.frames)
    ):
        raise IndexError("successor_frame_index is outside the captured POV prefix.")
    # Re-run the index's complete content/cue validation at this trust boundary.
    validated_source = ActorPovProjectionIndexV1(content=source.content)
    successor = build_no_shared_obs_authorized_scene_v1(
        validated_source,
        public_catalog=public_catalog,
        authority_session_id=authority_session_id,
        frame_index=successor_frame_index,
    )
    if successor_frame_index == 0:
        return None
    start = build_no_shared_obs_authorized_scene_v1(
        validated_source,
        public_catalog=public_catalog,
        authority_session_id=authority_session_id,
        frame_index=successor_frame_index - 1,
    )
    transition = validated_source.content.transitions[successor_frame_index - 1]
    return _compose_no_shared_obs_incoming_summary_v1(
        start,
        transition,
        successor,
        validated_source.content.axis_mapping,
    )


def build_live_no_shared_obs_incoming_summary_v1(
    source: ActorPovAdjacentTransitionSliceV1,
    *,
    public_catalog: StaticMechanicsCatalogV1,
    authority_session_id: str,
) -> NoSharedObsIncomingSummaryV1:
    """Build one live incoming summary from the exact adjacent POV carrier."""
    if type(source) is not ActorPovAdjacentTransitionSliceV1:
        raise TypeError(
            "NoSharedObs live incoming requires an exact adjacent POV slice."
        )
    validated_source = ActorPovAdjacentTransitionSliceV1.model_validate(
        source.model_dump(mode="python")
    )
    start = build_no_shared_obs_authorized_scene_v1(
        validated_source,
        public_catalog=public_catalog,
        authority_session_id=authority_session_id,
        frame_index=validated_source.start_frame.frame_index,
    )
    successor = build_no_shared_obs_authorized_scene_v1(
        validated_source,
        public_catalog=public_catalog,
        authority_session_id=authority_session_id,
        frame_index=validated_source.successor_frame.frame_index,
    )
    return _compose_no_shared_obs_incoming_summary_v1(
        start,
        validated_source.transition,
        successor,
        validated_source.axis_mapping,
    )


def _validated_shared_parts(
    value: SharedObsAuthorizedScenePartsV1,
    *,
    name: str,
) -> SharedObsAuthorizedScenePartsV1:
    if type(value) is not SharedObsAuthorizedScenePartsV1:
        raise TypeError(f"{name} must be exact SharedObs authorized scene parts.")
    adapter = TypeAdapter(SharedObsAuthorizedScenePartsV1)
    validated = adapter.validate_json(adapter.dump_json(value))
    if validated != value:
        raise ValueError(f"{name} changes under strict recursive revalidation.")
    return validated


def _provenance_by_public_id(
    parts: SharedObsAuthorizedScenePartsV1,
) -> dict[str, SharedObsAgentObservationProvenanceV1]:
    rows = {
        row.agent_public_agent_id: row for row in parts.agent_observation_provenance
    }
    if len(rows) != len(parts.agent_observation_provenance):
        raise ValueError("SharedObs endpoint repeats agent provenance.")
    return rows


def _changed_dynamic_fields(
    start: AgentIncomingObservationV1,
    successor: AgentIncomingObservationV1,
) -> tuple[SharedObsDynamicFieldV1, ...]:
    return tuple(
        field_name
        for field_name in _DYNAMIC_FIELD_ORDER
        if getattr(start, field_name) != getattr(successor, field_name)
    )


def _validate_shared_endpoints(
    start: SharedObsAuthorizedScenePartsV1,
    successor: SharedObsAuthorizedScenePartsV1,
) -> None:
    if (
        start.source_episode_id != successor.source_episode_id
        or start.recipient_public_agent_id != successor.recipient_public_agent_id
        or start.recipient_presentation_key != successor.recipient_presentation_key
    ):
        raise ValueError("SharedObs adjacent endpoint authority does not join.")
    if successor.source_frame_index != start.source_frame_index + 1:
        raise ValueError("SharedObs incoming frame indexes are not adjacent.")
    if successor.source_simulator_step_count != start.source_simulator_step_count + 1:
        raise ValueError("SharedObs incoming simulator ticks are not adjacent.")
    _validate_cross_epoch_agent_identity(start.scene.agents, successor.scene.agents)
    _validate_static_scene_continuity(start.scene, successor.scene)


def _validate_static_scene_continuity(
    start: AuthorizedBattlefieldSceneV1,
    successor: AuthorizedBattlefieldSceneV1,
) -> None:
    """Reject contradictory static facts while allowing lifecycle dynamics."""
    if start.map != successor.map:
        raise ValueError("Agent adjacent endpoints changed static map facts.")
    if start.spawn_shield_mechanics != successor.spawn_shield_mechanics:
        raise ValueError("Agent adjacent endpoints changed spawn-shield mechanics.")

    start_pads = {(pad.team_id, pad.team_local_slot): pad for pad in start.spawn_pads}
    successor_pads = {
        (pad.team_id, pad.team_local_slot): pad for pad in successor.spawn_pads
    }
    if start_pads.keys() != successor_pads.keys() or any(
        (start_pads[key].position, start_pads[key].configured_active)
        != (successor_pads[key].position, successor_pads[key].configured_active)
        for key in start_pads
    ):
        raise ValueError("Agent adjacent endpoints changed static spawn-pad facts.")

    start_waves = {wave.team_index: wave for wave in start.respawn_waves}
    successor_waves = {wave.team_index: wave for wave in successor.respawn_waves}
    if start_waves.keys() != successor_waves.keys() or any(
        (
            start_waves[key].team_index,
            start_waves[key].team_id,
            start_waves[key].period_steps,
        )
        != (
            successor_waves[key].team_index,
            successor_waves[key].team_id,
            successor_waves[key].period_steps,
        )
        for key in start_waves
    ):
        raise ValueError("Agent adjacent endpoints changed respawn-wave profile.")

    start_mechanics = {row.class_id: row for row in start.class_mechanics}
    successor_mechanics = {row.class_id: row for row in successor.class_mechanics}
    if any(
        start_mechanics[class_id] != successor_mechanics[class_id]
        for class_id in start_mechanics.keys() & successor_mechanics.keys()
    ):
        raise ValueError("Agent adjacent endpoints changed common class mechanics.")

    def aura_static_profile(field: AuthorizedAuraFieldV1) -> tuple[object, ...]:
        return (
            field.aura_id,
            field.source_presentation_key,
            field.source_public_agent_id,
            field.source_class_id,
            field.source_class_name,
            field.radius,
            field.beneficiary_relation,
            field.per_emitter_multiplier,
            field.stacking_rule,
            field.clamp_kind,
            field.clamp_value,
        )

    start_fields = {
        (field.source_public_agent_id, field.aura_id): field
        for field in start.aura_fields
    }
    successor_fields = {
        (field.source_public_agent_id, field.aura_id): field
        for field in successor.aura_fields
    }
    if any(
        aura_static_profile(start_fields[key])
        != aura_static_profile(successor_fields[key])
        for key in start_fields.keys() & successor_fields.keys()
    ):
        raise ValueError("Agent adjacent endpoints changed common aura profile.")


def build_shared_obs_incoming_summary_v1(
    start: SharedObsAuthorizedScenePartsV1 | None,
    successor: SharedObsAuthorizedScenePartsV1,
) -> SharedObsIncomingSummaryV1 | None:
    """Build generic deltas solely from two independently authorized endpoints."""
    successor = _validated_shared_parts(successor, name="successor")
    if successor.source_frame_index == 0:
        if start is not None:
            raise ValueError("SharedObs frame zero cannot have a start endpoint.")
        return None
    if start is None:
        raise ValueError("non-initial SharedObs summary requires its prior endpoint.")
    start = _validated_shared_parts(start, name="start")
    _validate_shared_endpoints(start, successor)

    start_by_public, successor_by_public = _validate_cross_epoch_agent_identity(
        start.scene.agents,
        successor.scene.agents,
    )
    start_provenance = _provenance_by_public_id(start)
    successor_provenance = _provenance_by_public_id(successor)
    transition_id = (
        f"{start.source_episode_id}:shared-obs-visual-union:"
        f"{start.recipient_public_agent_id}:transition:{start.source_frame_index}"
    )
    ordered_public_ids = tuple(
        sorted(
            start_by_public.keys() | successor_by_public.keys(),
            key=lambda public_id: (
                (
                    successor_by_public.get(public_id) or start_by_public[public_id]
                ).team_id,
                public_id,
            ),
        )
    )
    deltas: list[SharedObsIncomingDeltaV1] = []

    class _SharedDeltaIdentityV1(TypedDict):
        cue_id: str
        recipient_transition_id: str
        ordinal: int
        agent_presentation_key: str
        agent_public_agent_id: str

    def identity(agent: AuthorizedAgentV1) -> _SharedDeltaIdentityV1:
        ordinal = len(deltas)
        return {
            "cue_id": f"{transition_id}:cue:{ordinal}",
            "recipient_transition_id": transition_id,
            "ordinal": ordinal,
            "agent_presentation_key": agent.presentation_key,
            "agent_public_agent_id": agent.public_agent_id,
        }

    for public_id in ordered_public_ids:
        start_agent = start_by_public.get(public_id)
        successor_agent = successor_by_public.get(public_id)
        if start_agent is None:
            if successor_agent is None:  # pragma: no cover - set union fence.
                raise AssertionError("SharedObs union identity disappeared.")
            provenance = successor_provenance[public_id]
            deltas.append(
                SharedObsAppearanceIncomingDeltaV1(
                    delta_kind="appearance",
                    successor_observation=_observation(successor_agent),
                    successor_observation_sources=provenance.observation_sources,
                    **identity(successor_agent),
                )
            )
            continue
        if successor_agent is None:
            provenance = start_provenance[public_id]
            deltas.append(
                SharedObsDisappearanceIncomingDeltaV1(
                    delta_kind="disappearance",
                    start_observation=_observation(start_agent),
                    start_observation_sources=provenance.observation_sources,
                    **identity(start_agent),
                )
            )
            continue

        start_observation = _observation(start_agent)
        successor_observation = _observation(successor_agent)
        changed_fields = _changed_dynamic_fields(
            start_observation,
            successor_observation,
        )
        if changed_fields:
            deltas.append(
                SharedObsObservedValuesIncomingDeltaV1(
                    delta_kind="observed_values_change",
                    changed_dynamic_fields=changed_fields,
                    start_observation=start_observation,
                    successor_observation=successor_observation,
                    **identity(start_agent),
                )
            )
        start_sources = start_provenance[public_id].observation_sources
        successor_sources = successor_provenance[public_id].observation_sources
        if start_sources != successor_sources:
            deltas.append(
                SharedObsObservationProvenanceIncomingDeltaV1(
                    delta_kind="observation_provenance_change",
                    start_observation_sources=start_sources,
                    successor_observation_sources=successor_sources,
                    **identity(start_agent),
                )
            )

    return SharedObsIncomingSummaryV1(
        schema_version=AUTHORIZED_PRESENTATION_SCHEMA_VERSION,
        summary_kind="shared_obs_recipient_observation_deltas",
        source_episode_id=start.source_episode_id,
        recipient_public_agent_id=start.recipient_public_agent_id,
        recipient_presentation_key=start.recipient_presentation_key,
        incoming_transition_index=start.source_frame_index,
        incoming_recipient_transition_id=transition_id,
        incoming_start_recipient_frame_id=start.source_recipient_frame_id,
        incoming_successor_recipient_frame_id=successor.source_recipient_frame_id,
        incoming_start_simulator_step_count=start.source_simulator_step_count,
        incoming_successor_simulator_step_count=(successor.source_simulator_step_count),
        deltas=tuple(deltas),
        delta_count=len(deltas),
    )


__all__ = [
    "AgentIncomingObservationV1",
    "AgentIncomingObservedStatusV1",
    "NoSharedObsEpisodeEndedIncomingCueV1",
    "NoSharedObsIncomingCueV1",
    "NoSharedObsIncomingSummaryV1",
    "NoSharedObsOwnActionOutcomeIncomingCueV1",
    "NoSharedObsOwnCooldownChangedIncomingCueV1",
    "NoSharedObsOwnHealthChangedIncomingCueV1",
    "NoSharedObsOwnLifecycleChangedIncomingCueV1",
    "NoSharedObsOwnPositionChangedIncomingCueV1",
    "NoSharedObsOwnStatusChangedIncomingCueV1",
    "NoSharedObsVisibleBodyChangedIncomingCueV1",
    "SharedObsAppearanceIncomingDeltaV1",
    "SharedObsDisappearanceIncomingDeltaV1",
    "SharedObsIncomingDeltaV1",
    "SharedObsIncomingSummaryV1",
    "SharedObsObservationProvenanceIncomingDeltaV1",
    "SharedObsObservedValuesIncomingDeltaV1",
    "build_live_no_shared_obs_incoming_summary_v1",
    "build_replay_no_shared_obs_incoming_summary_v1",
    "build_shared_obs_incoming_summary_v1",
]
