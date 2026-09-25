"""Define immutable drafts for the private map and scenario editor.

The browser may edit local copies; the host parses submitted content with these
strict models before compiling or saving it. Unknown fields and nonfinite values
are rejected. Use ``new_map_draft`` and ``new_scenario_draft`` for starter content.
Factories return in-memory drafts and never write files. Geometry and simulator
validity remain the compiler's responsibility.

Scenario drafts come in two versions. Version 1 (``dev-scenario-draft@1``) has no
Red Zone depth and always keeps its original one-point scoring, that is depth
0.0. Version 2 (``dev-scenario-draft@2``) declares ``task.red_zone_depth``. The
editor edits only version 2; ``upgrade_scenario_draft`` turns an old draft into
version 2 at depth 0.0. ``declared_red_zone_depth`` is the one owner of the
depth a draft means. This module imports only Pydantic, so callers pass any
default depth in themselves.
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
        pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*$",
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
    """A finite map-space point with x and y coordinates."""

    x: float
    y: float


class DevWallV1(_AuthoringModel):
    """An authored rectangular wall with a stable ID, center, size, and rotation.

    Rotation is in degrees. The compiler checks geometry after float32 conversion.
    """

    kind: Literal["wall"] = "wall"
    object_id: ObjectId
    center_x: float
    center_y: float
    width: float
    height: float
    rotation_degrees: float = 0.0


class DevPillarV1(_AuthoringModel):
    """An authored circular pillar with a stable ID, center, and radius."""

    kind: Literal["pillar"] = "pillar"
    object_id: ObjectId
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
        """Require unique object IDs and all ten pads ordered A1-A5 then B1-B5."""
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
    """An editable map revision containing its stable asset ID and complete content.

    Revision zero describes a draft that has not yet been saved.
    """

    schema_id: Literal["dev-map-draft@1"] = Field(
        default="dev-map-draft@1",
        alias="schema",
        serialization_alias="schema",
    )
    asset_id: SafeAssetId
    revision: DevDraftRevision = 0
    content: DevMapContentV1


class DevSourceMapProvenanceV1(_AuthoringModel):
    """Optional source-map identity retained after embedding its content in a
    scenario.
    """

    asset_id: SafeAssetId | None = None
    revision: DevDraftRevision | None = None


class DevTeamDeathmatchTaskV1(_AuthoringModel):
    """Authored Team Deathmatch settings; the compiler validates the score threshold."""

    task: Literal["team_deathmatch"] = "team_deathmatch"
    score_threshold: int = 5


class DevTeamDeathmatchTaskV2(_AuthoringModel):
    """Authored Team Deathmatch settings with a declared Red Zone depth.

    Attributes
    ----------
    task : {"team_deathmatch"}
        The task name, default ``"team_deathmatch"``.
    score_threshold : int
        Points a team needs to win, default 5. The compiler checks its range.
    red_zone_depth : float
        Required, in map units. How far each team's Red Zone reaches in from its
        own spawn edge. When an agent dies inside its own team's Red Zone, the
        enemy team gets 2 points instead of 1. 0.0 turns the rule off. A JSON
        integer such as 5 reads as 5.0; booleans, strings, NaN and infinity are
        rejected here. The compiler checks the raw value before it rounds it to
        float32: it must not be negative (-0.0 included), a positive value must
        not round to zero or a subnormal float32, and it must not exceed the map
        width.
    """

    task: Literal["team_deathmatch"] = "team_deathmatch"
    score_threshold: int = 5
    red_zone_depth: float


class DevEpisodeConfigurationV1(_AuthoringModel):
    """Authored episode limits, spawn shield settings, and team respawn periods.

    Durations are measured in simulator steps. The compiler owns numeric limits.
    """

    max_steps: int = 300
    spawn_shield_duration_steps: int = 3
    spawn_shield_movement_speed: float = 2.0
    team_a_respawn_wave_period_steps: int = 5
    team_b_respawn_wave_period_steps: int = 5


class DevScenarioGlobalStateV1(_AuthoringModel):
    """Starting step, scores, and respawn countdowns for an authored scenario."""

    step_count: int = 0
    team_a_score: int = 0
    team_b_score: int = 0
    team_a_respawn_countdown: int = 4
    team_b_respawn_countdown: int = 4


class DevRosterSlotV1(_AuthoringModel):
    """One fixed A1-A5 or B1-B5 roster position and its selected class.

    Global slots are zero-based; team-local slots are one-based. Inactive positions
    use ``not_applicable`` rather than removing a row.
    """

    object_id: ObjectId
    team: Literal["A", "B"]
    team_local_slot: Annotated[int, Field(ge=1, le=5)]
    global_slot: Annotated[int, Field(ge=0, le=9)]
    class_name: ClassName | Literal["not_applicable"]


class DevAgentStateV1(_AuthoringModel):
    """Authored starting state for one agent, joined to its roster row by object ID.

    Position uses map coordinates. Cooldowns and remaining status durations use
    simulator steps. The compiler checks health, activity, and status consistency.
    """

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
    """A self-contained scenario with embedded map, task settings, and ten agent rows.

    Roster and state rows share fixed A1-A5 then B1-B5 order. Source-map provenance
    records where the embedded map came from; it does not resolve mutable content.
    """

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
        """Require fixed slot order, matching roster/state IDs, and unique object
        IDs.
        """
        _validate_scenario_topology(self.embedded_map, self.roster, self.agent_states)
        return self


def _validate_scenario_topology(
    embedded_map: DevMapContentV1,
    roster: tuple[DevRosterSlotV1, ...],
    agent_states: tuple[DevAgentStateV1, ...],
) -> None:
    """Check the fixed-slot rules that every scenario content version shares.

    Parameters
    ----------
    embedded_map : DevMapContentV1
        The scenario's map; its obstacle and pad IDs must not reuse agent IDs.
    roster : tuple of DevRosterSlotV1
        Ten rows that must be ordered A1-A5 then B1-B5 with global slots 0-9.
    agent_states : tuple of DevAgentStateV1
        Ten rows that must use the roster's object IDs in the same order.

    Raises
    ------
    ValueError
        If the roster order is wrong, the state rows do not join the roster by
        ordered object ID, a roster ID repeats, or an agent ID is also a map ID.
        Pydantic reports the error as a validation error of the content model.
    """
    expected = tuple(
        ("A" if global_slot < 5 else "B", global_slot % 5 + 1, global_slot)
        for global_slot in range(10)
    )
    actual = tuple(
        (slot.team, slot.team_local_slot, slot.global_slot) for slot in roster
    )
    if actual != expected:
        raise ValueError("roster must contain ordered fixed slots A1-A5 then B1-B5")
    roster_ids = tuple(slot.object_id for slot in roster)
    state_ids = tuple(state.object_id for state in agent_states)
    if roster_ids != state_ids:
        raise ValueError("agent_states must join roster rows by ordered object_id")
    if len(roster_ids) != len(set(roster_ids)):
        raise ValueError("roster object_id values must be unique")
    map_ids = {
        *(obstacle.object_id for obstacle in embedded_map.obstacles),
        *(pad.object_id for pad in embedded_map.spawn_pads),
    }
    if map_ids.intersection(roster_ids):
        raise ValueError(
            "scenario object_id values must be unique across map and agent objects"
        )


class DevScenarioDraftV1(_AuthoringModel):
    """An editable scenario revision with a stable asset ID and self-contained
    content.

    This is the old version with no Red Zone depth; it always means depth 0.0.
    Saved version 1 files stay readable and are never rewritten.
    """

    schema_id: Literal["dev-scenario-draft@1"] = Field(
        default="dev-scenario-draft@1",
        alias="schema",
        serialization_alias="schema",
    )
    asset_id: SafeAssetId
    revision: DevDraftRevision = 0
    content: DevScenarioContentV1


class DevScenarioContentV2(_AuthoringModel):
    """A self-contained scenario that declares its Red Zone depth.

    It has every field of ``DevScenarioContentV1`` with the same meaning and
    defaults, except ``task``, which is a required ``DevTeamDeathmatchTaskV2``.
    Roster and state rows share fixed A1-A5 then B1-B5 order, checked by the same
    rule as version 1. Source-map provenance records where the embedded map came
    from; it does not resolve mutable content.

    Attributes
    ----------
    task : DevTeamDeathmatchTaskV2
        Required task settings, including ``red_zone_depth`` in map units.
    """

    schema_id: Literal["dev-scenario-content@2"] = Field(
        default="dev-scenario-content@2",
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
    task: DevTeamDeathmatchTaskV2
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
    def _validate_fixed_slot_topology(self) -> DevScenarioContentV2:
        """Apply the fixed-slot rules shared with version 1."""
        _validate_scenario_topology(self.embedded_map, self.roster, self.agent_states)
        return self


class DevScenarioDraftV2(_AuthoringModel):
    """An editable scenario revision whose content declares its Red Zone depth.

    Revision zero describes a draft that has not yet been saved. This is the only
    scenario version the editor edits.
    """

    schema_id: Literal["dev-scenario-draft@2"] = Field(
        default="dev-scenario-draft@2",
        alias="schema",
        serialization_alias="schema",
    )
    asset_id: SafeAssetId
    revision: DevDraftRevision = 0
    content: DevScenarioContentV2


# Either scenario content version; ``declared_red_zone_depth`` reads its depth.
type DevScenarioContent = DevScenarioContentV1 | DevScenarioContentV2
# Either scenario draft version. Use a plain union, as the store does, so each
# draft's schema tag picks its class.
type DevScenarioDraft = DevScenarioDraftV1 | DevScenarioDraftV2


def declared_red_zone_depth(content: DevScenarioContent) -> float:
    """Return the Red Zone depth that scenario content means, in map units.

    Parameters
    ----------
    content : DevScenarioContentV1 or DevScenarioContentV2
        Scenario content of either version.

    Returns
    -------
    float
        0.0 for version 1 content, its original one-point rule. The
        ``task.red_zone_depth`` value, unchanged, for version 2 content.

    Notes
    -----
    This is the one owner of that rule; the compiler, the service and the
    semantic digest all read the depth through it. It checks nothing: the
    compiler checks the value before use.
    """
    if isinstance(content, DevScenarioContentV1):
        return 0.0
    return content.task.red_zone_depth


def _scenario_content_v2(
    content: DevScenarioContent,
    *,
    red_zone_depth: float,
) -> DevScenarioContentV2:
    """Copy scenario content of either version into version 2 at one depth.

    Every field except the schema tag and the task's new ``red_zone_depth`` is
    copied deeply and keeps its value. Strict model validation may raise
    ``pydantic.ValidationError`` for a depth that is not a float.
    """
    return DevScenarioContentV2(
        name=content.name,
        description=content.description,
        notes=content.notes,
        embedded_map=content.embedded_map.model_copy(deep=True),
        source_map_provenance=(
            None
            if content.source_map_provenance is None
            else content.source_map_provenance.model_copy(deep=True)
        ),
        task=DevTeamDeathmatchTaskV2(
            task=content.task.task,
            score_threshold=content.task.score_threshold,
            red_zone_depth=red_zone_depth,
        ),
        episode=content.episode.model_copy(deep=True),
        global_state=content.global_state.model_copy(deep=True),
        team_a_size=content.team_a_size,
        team_b_size=content.team_b_size,
        roster=tuple(row.model_copy(deep=True) for row in content.roster),
        agent_states=tuple(row.model_copy(deep=True) for row in content.agent_states),
    )


def upgrade_scenario_draft(draft: DevScenarioDraft) -> DevScenarioDraftV2:
    """Return the version 2 form of a scenario draft, the form the editor edits.

    Parameters
    ----------
    draft : DevScenarioDraftV1 or DevScenarioDraftV2
        A parsed scenario draft of either version.

    Returns
    -------
    DevScenarioDraftV2
        The same object for a version 2 draft. For a version 1 draft, a new
        version 2 draft with the same asset ID, revision and content at Red Zone
        depth 0.0, so it keeps its original one-point rule and its semantic,
        map, configuration and state digests.

    Notes
    -----
    Nothing is saved. Keeping the revision lets the next Save write revision
    N + 1 as version 2 while saved revision N stays version 1.
    """
    if isinstance(draft, DevScenarioDraftV2):
        return draft
    return DevScenarioDraftV2(
        asset_id=draft.asset_id,
        revision=draft.revision,
        content=_scenario_content_v2(
            draft.content,
            red_zone_depth=declared_red_zone_depth(draft.content),
        ),
    )


def default_spawn_pads(
    *, width: float = 20.0, height: float = 10.0
) -> tuple[DevSpawnPadV1, ...]:
    """Build the ten fixed team pads used by a new editor draft.

    Parameters
    ----------
    width : float, optional
        Map width, default 20.0. Pads are placed 1.5 units inside each vertical edge.
    height : float, optional
        Map height, default 10.0. Each five-pad bank spans the interior vertical range.

    Returns
    -------
    tuple of DevSpawnPadV1
        Ten pads ordered A1-A5 then B1-B5, with stable IDs and map-space positions.

    Notes
    -----
    This supplies starter coordinates, not a geometry-validity guarantee. The compiler
    checks the resulting map and body-radius constraints before use.
    """
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


def new_map_draft(asset_id: SafeAssetId = "untitled_map") -> DevMapDraftV1:
    """Create an unsaved blank map with the standard ten spawn pads.

    Parameters
    ----------
    asset_id : str, optional
        Safe lowercase snake_case identity, default ``untitled_map``.

    Returns
    -------
    DevMapDraftV1
        Revision-zero map with default 20 by 10 dimensions and no obstacles.

    Raises
    ------
    ValueError
        If the model rejects the supplied asset identity.

    Notes
    -----
    The returned draft stays in memory until the caller explicitly saves it.
    """
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
    asset_id: SafeAssetId = "untitled_scenario",
    *,
    source_map: DevMapDraftV1 | None = None,
    red_zone_depth: float,
) -> DevScenarioDraftV2:
    """Create an unsaved Team Deathmatch scenario with two complete starter teams.

    Parameters
    ----------
    asset_id : str, optional
        Safe scenario identity, default ``untitled_scenario``.
    source_map : DevMapDraftV1 or None, optional
        Map to copy into the scenario. None creates a default blank map. A supplied
        map is copied deeply and its source identity is retained as provenance.
    red_zone_depth : float
        Required keyword, in map units: the declared Red Zone depth. 0.0 keeps
        one-point scoring. This module has no default; the DevClient passes
        ``marl_battlegrounds.tasks.DEFAULT_TDM_RED_ZONE_DEPTH`` (5.0).

    Returns
    -------
    DevScenarioDraftV2
        Revision-zero draft with ten living agents at their assigned pads, default
        class health, the declared default episode settings, score threshold 5
        and the given Red Zone depth.

    Raises
    ------
    ValueError
        If strict draft models reject an identity or content field, including a
        depth that is not a float (``pydantic.ValidationError`` is a
        ``ValueError``).

    Notes
    -----
    Copied maps are independent of later source edits. The compiler must still check
    geometry, the depth's range and simulator-state validity. No files are written.
    """
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
    return DevScenarioDraftV2(
        asset_id=asset_id,
        content=DevScenarioContentV2(
            name="Untitled TDM scenario",
            embedded_map=embedded_map,
            source_map_provenance=source_provenance,
            task=DevTeamDeathmatchTaskV2(red_zone_depth=red_zone_depth),
            roster=roster,
            agent_states=states,
        ),
    )


def duplicate_scenario_draft(
    source: DevScenarioDraft,
    *,
    asset_id: SafeAssetId,
    red_zone_depth: float,
) -> DevScenarioDraftV2:
    """Copy complete scenario content into an independent unsaved identity.

    Parameters
    ----------
    source : DevScenarioDraftV1 or DevScenarioDraftV2
        Validated scenario whose embedded content is copied deeply.
    asset_id : str
        Safe lowercase snake_case identity for the copy.
    red_zone_depth : float
        Required keyword, in map units: the copy's declared Red Zone depth. To
        keep the source's rule, pass ``declared_red_zone_depth(source.content)``,
        which is 0.0 for a version 1 source.

    Returns
    -------
    DevScenarioDraftV2
        Revision-zero draft with independent copied content and the given depth.
        Every other field keeps the source's value.

    Raises
    ------
    ValueError
        If the destination identity or depth fails strict model validation.

    Notes
    -----
    The source is unchanged and no storage operation is performed.
    """
    return DevScenarioDraftV2(
        asset_id=asset_id,
        revision=0,
        content=_scenario_content_v2(source.content, red_zone_depth=red_zone_depth),
    )
