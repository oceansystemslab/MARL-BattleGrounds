"""Strict private contracts for DevClient map and scenario authoring.

These models are deliberately host-internal.  They describe editable product
assets, not simulator or public replay schemas.  The browser may retain local
mutable copies, while every value accepted by the host is reparsed into one of
these immutable, extra-forbidding models.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

_MAX_NAME_LENGTH = 120
_MAX_DESCRIPTION_LENGTH = 2_000
_MAX_NOTES_LENGTH = 8_000
_MAX_ID_LENGTH = 64
_MAX_AGENT_SLOTS = 10
MAX_DEV_ASSET_SEQUENCE = 2**31 - 1

type SafeAssetId = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=_MAX_ID_LENGTH,
        pattern=r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$",
    ),
]
type ObjectId = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=_MAX_ID_LENGTH,
        pattern=r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$",
    ),
]
type SemanticDigest = Annotated[
    str,
    StringConstraints(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"),
]
type DevDraftRevision = Annotated[
    int,
    Field(ge=0, le=MAX_DEV_ASSET_SEQUENCE),
]
type DevSavedRevision = Annotated[
    int,
    Field(ge=1, le=MAX_DEV_ASSET_SEQUENCE),
]
type ShortText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=256),
]
type ClassName = Literal["mage", "warrior", "hunter", "rogue", "priest"]


class _AuthoringModel(BaseModel):
    """Strict immutable base used for all persisted host documents."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        serialize_by_alias=True,
        strict=True,
    )


class DevAuthoringProblemV1(_AuthoringModel):
    """One stable, UI-linkable authoring validation result."""

    severity: Literal["error", "warning"]
    stable_code: ShortText
    message: Annotated[str, StringConstraints(min_length=1, max_length=1_000)]
    object_id: ObjectId | None = None
    field_path: Annotated[str, StringConstraints(min_length=1, max_length=512)]


class DevPointV1(_AuthoringModel):
    x: float
    y: float


class DevWallV1(_AuthoringModel):
    kind: Literal["wall"] = "wall"
    object_id: ObjectId
    name: Annotated[
        str,
        StringConstraints(strip_whitespace=True, max_length=_MAX_NAME_LENGTH),
    ] = ""
    center_x: float
    center_y: float
    width: float
    height: float
    rotation_degrees: float = 0.0


class DevPillarV1(_AuthoringModel):
    kind: Literal["pillar"] = "pillar"
    object_id: ObjectId
    name: Annotated[
        str,
        StringConstraints(strip_whitespace=True, max_length=_MAX_NAME_LENGTH),
    ] = ""
    center_x: float
    center_y: float
    radius: float


type DevObstacleV1 = Annotated[
    DevWallV1 | DevPillarV1,
    Field(discriminator="kind"),
]


class DevSpawnPadV1(_AuthoringModel):
    """One fixed team-local pad whose identity cannot be edited away."""

    object_id: ObjectId
    team: Literal["A", "B"]
    team_local_slot: Annotated[int, Field(ge=1, le=5)]
    position: DevPointV1


class DevMapContentV1(_AuthoringModel):
    """Complete reusable map content; list order is fixed-slot semantics."""

    schema_id: Literal["dev-map-content@1"] = Field(
        default="dev-map-content@1",
        alias="schema",
        serialization_alias="schema",
    )
    name: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=1,
            max_length=_MAX_NAME_LENGTH,
        ),
    ]
    description: Annotated[
        str,
        StringConstraints(strip_whitespace=True, max_length=_MAX_DESCRIPTION_LENGTH),
    ] = ""
    width: float = 20.0
    height: float = 10.0
    obstacles: tuple[DevObstacleV1, ...] = ()
    spawn_pads: Annotated[
        tuple[DevSpawnPadV1, ...],
        Field(min_length=_MAX_AGENT_SLOTS, max_length=_MAX_AGENT_SLOTS),
    ]

    @model_validator(mode="after")
    def _validate_object_and_pad_identity(self) -> DevMapContentV1:
        object_ids = tuple(obstacle.object_id for obstacle in self.obstacles) + tuple(
            pad.object_id for pad in self.spawn_pads
        )
        if len(object_ids) != len(set(object_ids)):
            raise ValueError("map object_id values must be unique")

        expected = tuple(
            (team, local_slot) for team in ("A", "B") for local_slot in range(1, 6)
        )
        actual = tuple((pad.team, pad.team_local_slot) for pad in self.spawn_pads)
        if actual != expected:
            raise ValueError(
                "spawn_pads must contain ordered A1-A5 then B1-B5 identities"
            )
        return self


class DevMapDraftV1(_AuthoringModel):
    schema_id: Literal["dev-map-draft@1"] = Field(
        default="dev-map-draft@1",
        alias="schema",
        serialization_alias="schema",
    )
    asset_id: SafeAssetId
    revision: DevDraftRevision = 0
    content: DevMapContentV1


class DevSourceMapProvenanceV1(_AuthoringModel):
    asset_id: SafeAssetId | None = None
    revision: DevDraftRevision | None = None


class DevTeamDeathmatchTaskV1(_AuthoringModel):
    task: Literal["team_deathmatch"] = "team_deathmatch"
    score_threshold: int = 5


class DevEpisodeConfigurationV1(_AuthoringModel):
    max_steps: int = 300
    spawn_shield_duration_steps: int = 3
    spawn_shield_movement_speed: float = 2.0
    team_a_respawn_wave_period_steps: int = 5
    team_b_respawn_wave_period_steps: int = 5


class DevScenarioGlobalStateV1(_AuthoringModel):
    step_count: int = 0
    team_a_score: int = 0
    team_b_score: int = 0
    team_a_respawn_countdown: int = 4
    team_b_respawn_countdown: int = 4


class DevRosterSlotV1(_AuthoringModel):
    object_id: ObjectId
    team: Literal["A", "B"]
    team_local_slot: Annotated[int, Field(ge=1, le=5)]
    global_slot: Annotated[int, Field(ge=0, le=9)]
    class_name: ClassName | Literal["not_applicable"]


class DevAgentStateV1(_AuthoringModel):
    object_id: ObjectId
    position: DevPointV1
    alive: bool
    current_health: float
    ultimate_cooldown_remaining: int = 0
    spawn_shield_duration_remaining: int = 0
    steps_until_out_of_combat: int = 0
    warrior_charge_slow_duration: int = 0
    hunter_basic_slow_duration: int = 0
    rogue_poison_slow_duration: int = 0
    warrior_charge_stun_duration: int = 0
    hunter_trap_stun_duration: int = 0
    rogue_poison_stun_duration: int = 0
    rogue_poison_anti_heal_duration: int = 0
    mage_burst_duration: int = 0
    priest_blessing_of_freedom_duration: int = 0


class DevScenarioContentV1(_AuthoringModel):
    schema_id: Literal["dev-scenario-content@1"] = Field(
        default="dev-scenario-content@1",
        alias="schema",
        serialization_alias="schema",
    )
    name: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=1,
            max_length=_MAX_NAME_LENGTH,
        ),
    ]
    description: Annotated[
        str,
        StringConstraints(strip_whitespace=True, max_length=_MAX_DESCRIPTION_LENGTH),
    ] = ""
    notes: Annotated[
        str,
        StringConstraints(strip_whitespace=True, max_length=_MAX_NOTES_LENGTH),
    ] = ""
    embedded_map: DevMapContentV1
    source_map_provenance: DevSourceMapProvenanceV1 | None = None
    task: DevTeamDeathmatchTaskV1 = DevTeamDeathmatchTaskV1()
    episode: DevEpisodeConfigurationV1 = DevEpisodeConfigurationV1()
    global_state: DevScenarioGlobalStateV1 = DevScenarioGlobalStateV1()
    team_a_size: int = 5
    team_b_size: int = 5
    roster: Annotated[
        tuple[DevRosterSlotV1, ...],
        Field(min_length=_MAX_AGENT_SLOTS, max_length=_MAX_AGENT_SLOTS),
    ]
    agent_states: Annotated[
        tuple[DevAgentStateV1, ...],
        Field(min_length=_MAX_AGENT_SLOTS, max_length=_MAX_AGENT_SLOTS),
    ]

    @model_validator(mode="after")
    def _validate_fixed_slot_topology(self) -> DevScenarioContentV1:
        expected = tuple(
            ("A" if global_slot < 5 else "B", global_slot % 5 + 1, global_slot)
            for global_slot in range(10)
        )
        actual = tuple(
            (slot.team, slot.team_local_slot, slot.global_slot) for slot in self.roster
        )
        if actual != expected:
            raise ValueError("roster must contain ordered fixed slots A1-A5 then B1-B5")
        roster_ids = tuple(slot.object_id for slot in self.roster)
        state_ids = tuple(state.object_id for state in self.agent_states)
        if roster_ids != state_ids:
            raise ValueError("agent_states must join roster rows by ordered object_id")
        if len(roster_ids) != len(set(roster_ids)):
            raise ValueError("roster object_id values must be unique")
        map_ids = {
            *(obstacle.object_id for obstacle in self.embedded_map.obstacles),
            *(pad.object_id for pad in self.embedded_map.spawn_pads),
        }
        if map_ids.intersection(roster_ids):
            raise ValueError(
                "scenario object_id values must be unique across map and agent objects"
            )

        return self


class DevScenarioDraftV1(_AuthoringModel):
    schema_id: Literal["dev-scenario-draft@1"] = Field(
        default="dev-scenario-draft@1",
        alias="schema",
        serialization_alias="schema",
    )
    asset_id: SafeAssetId
    revision: DevDraftRevision = 0
    content: DevScenarioContentV1


def default_spawn_pads(
    *, width: float = 20.0, height: float = 10.0
) -> tuple[DevSpawnPadV1, ...]:
    """Return the fixed two-team edge formation used for ergonomic new drafts."""
    step = (height - 3.0) / 4.0
    y_coordinates = tuple(1.5 + step * index for index in range(5))
    return tuple(
        DevSpawnPadV1(
            object_id=f"pad-{team.lower()}{local_slot}",
            team=team,
            team_local_slot=local_slot,
            position=DevPointV1(
                x=1.5 if team == "A" else width - 1.5,
                y=y_coordinates[local_slot - 1],
            ),
        )
        for team in ("A", "B")
        for local_slot in range(1, 6)
    )


def new_map_draft(asset_id: SafeAssetId = "untitled-map") -> DevMapDraftV1:
    """Create the minimal ergonomic blank-map draft."""
    return DevMapDraftV1(
        asset_id=asset_id,
        content=DevMapContentV1(
            name="Untitled map",
            spawn_pads=default_spawn_pads(),
        ),
    )


_DEFAULT_CLASSES: tuple[ClassName, ...] = (
    "mage",
    "warrior",
    "hunter",
    "rogue",
    "priest",
)
_DEFAULT_MAX_HEALTH = {
    "mage": 80.0,
    "warrior": 200.0,
    "hunter": 100.0,
    "rogue": 100.0,
    "priest": 100.0,
}


def _new_agent_object_ids(embedded_map: DevMapContentV1) -> tuple[str, ...]:
    """Choose stable agent IDs without changing copied map object identities."""
    reserved = {
        *(obstacle.object_id for obstacle in embedded_map.obstacles),
        *(pad.object_id for pad in embedded_map.spawn_pads),
    }
    object_ids: list[str] = []
    for team in ("a", "b"):
        for local_slot in range(1, 6):
            base = f"agent-{team}{local_slot}"
            object_id = base
            suffix = 2
            while object_id in reserved:
                object_id = f"{base}-{suffix}"
                suffix += 1
            reserved.add(object_id)
            object_ids.append(object_id)
    return tuple(object_ids)


def new_scenario_draft(
    asset_id: SafeAssetId = "untitled-scenario",
    *,
    source_map: DevMapDraftV1 | None = None,
) -> DevScenarioDraftV1:
    """Create the canonical ergonomic blank TDM draft or independent map copy."""
    if source_map is None:
        embedded_map = DevMapContentV1(
            name="Untitled scenario map",
            spawn_pads=default_spawn_pads(),
        )
        source_provenance = None
    else:
        embedded_map = source_map.content.model_copy(deep=True)
        source_provenance = DevSourceMapProvenanceV1(
            asset_id=source_map.asset_id,
            revision=source_map.revision,
        )

    pads = embedded_map.spawn_pads
    agent_object_ids = _new_agent_object_ids(embedded_map)
    roster = tuple(
        DevRosterSlotV1(
            object_id=agent_object_ids[(0 if team == "A" else 5) + local_slot - 1],
            team=team,
            team_local_slot=local_slot,
            global_slot=(0 if team == "A" else 5) + local_slot - 1,
            class_name=_DEFAULT_CLASSES[local_slot - 1],
        )
        for team in ("A", "B")
        for local_slot in range(1, 6)
    )
    states = tuple(
        DevAgentStateV1(
            object_id=roster_slot.object_id,
            position=pads[global_slot].position.model_copy(deep=True),
            alive=True,
            current_health=_DEFAULT_MAX_HEALTH[roster_slot.class_name],
        )
        for global_slot, roster_slot in enumerate(roster)
    )
    return DevScenarioDraftV1(
        asset_id=asset_id,
        content=DevScenarioContentV1(
            name="Untitled TDM scenario",
            embedded_map=embedded_map,
            source_map_provenance=source_provenance,
            roster=roster,
            agent_states=states,
        ),
    )


def duplicate_scenario_draft(
    source: DevScenarioDraftV1,
    *,
    asset_id: SafeAssetId,
) -> DevScenarioDraftV1:
    """Return an independent mutable-draft representation of one exact scenario."""
    return DevScenarioDraftV1(
        asset_id=asset_id,
        revision=0,
        content=source.content.model_copy(deep=True),
    )
