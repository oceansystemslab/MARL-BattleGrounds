"""Handle private DevClient authoring commands and validated live loads.

The live HTTP binding parses commands with ``DevAuthoringCommandRequestV1`` and
calls ``DevClientAuthoringBinding.apply_command``. Draft storage, compilation and
live installation stay in their dedicated owners. The binding serializes requests
with a lock and returns field-linked errors. The load service keeps a fixed compiled
snapshot so later edits cannot change a running scenario. Import builds the shared
read-only mechanics catalog; commands may allocate JAX arrays, read/write local
drafts, or replace the live endpoint according to their explicit operation.

Scenario drafts: the editor edits only version 2, which declares a Red Zone
depth. New blank scenarios, scenarios copied from a map and the default map
preview use ``marl_battlegrounds.tasks.DEFAULT_TDM_RED_ZONE_DEPTH`` (5.0).
Duplicate keeps the source's depth. Open turns a saved version 1 scenario into
version 2 at depth 0.0 in memory, keeping its revision; the saved file is not
changed. Commands still accept version 1 payloads, which Save and Save As store
as version 1, and Combat can load a saved version 1 revision, which plays at 0.0.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock
from typing import Annotated, Literal

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    StringConstraints,
    model_validator,
)

from marl_battlegrounds.core import combat
from marl_battlegrounds.core.config import CANONICAL_PRODUCT_MOVEMENT_SCALE
from marl_battlegrounds.core.types import MAX_OBSTACLE_SLOTS, EnvConfig, EnvState
from marl_battlegrounds.evaluation.catalog import build_static_mechanics_catalog_v1
from marl_battlegrounds.evaluation.map_identity import approved_map_id
from marl_battlegrounds.evaluation.models import (
    AuraMechanicV1,
    ClassMechanicsV1,
    StatusMechanicV1,
    red_zone_team_on_right,
    red_zone_x_range,
)
from marl_battlegrounds.tasks import DEFAULT_TDM_RED_ZONE_DEPTH
from scripts.dev.visual_debugger.authoring_compiler import (
    CompiledDevScenarioV1,
    DevAuthoringValidationError,
    compile_dev_scenario,
    map_semantic_digest,
    validate_map_content,
)
from scripts.dev.visual_debugger.authoring_models import (
    DevAuthoringProblemV1,
    DevDraftRevision,
    DevMapDraftV1,
    DevSavedRevision,
    DevScenarioDraftV1,
    DevScenarioDraftV2,
    SafeAssetId,
    SemanticDigest,
    declared_red_zone_depth,
    duplicate_scenario_draft,
    new_map_draft,
    new_scenario_draft,
    upgrade_scenario_draft,
)
from scripts.dev.visual_debugger.authoring_store import (
    DevAssetAlreadyExistsError,
    DevAssetIntegrityError,
    DevAssetKind,
    DevAssetNotFoundError,
    DevAssetStore,
    DevAssetStoreError,
    DevDraftRevisionConflictError,
)
from scripts.dev.visual_debugger.model import (
    DebuggerScenario,
    DebuggerScenarioProvenance,
)

# Every draft a command may carry or return. A plain union, so each payload's
# schema tag picks its class: a version 1 scenario with a depth, or a version 2
# scenario without one, is rejected.
type DevDraftPayload = DevMapDraftV1 | DevScenarioDraftV1 | DevScenarioDraftV2
type CommandType = Literal[
    "list",
    "new_map",
    "new_scenario",
    "open",
    "save",
    "save_as",
    "validate",
    "delete",
    "open_in_debug",
]


class _ServiceModel(BaseModel):
    """Strict immutable authoring-service model that rejects extra fields and
    nonfinite numbers.
    """

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        serialize_by_alias=True,
        strict=True,
    )


class DevSavedDraftSourceV1(_ServiceModel):
    """Select one exact saved map or scenario ID and positive revision."""

    source_kind: Literal["saved_draft"] = "saved_draft"
    asset_kind: DevAssetKind
    asset_id: SafeAssetId
    revision: DevSavedRevision


type DevPersistedSourceV1 = DevSavedDraftSourceV1


class DevCurrentBufferSourceV1(_ServiceModel):
    """Carry a complete unsaved editor buffer with its matching map/scenario kind."""

    source_kind: Literal["current_buffer"] = "current_buffer"
    asset_kind: DevAssetKind
    draft: DevDraftPayload

    @model_validator(mode="after")
    def _validate_asset_kind(self) -> DevCurrentBufferSourceV1:
        """Require asset_kind to agree with the concrete draft model."""
        expected_kind: DevAssetKind = (
            "map" if isinstance(self.draft, DevMapDraftV1) else "scenario"
        )
        if self.asset_kind != expected_kind:
            raise ValueError("current_buffer asset_kind must match the supplied draft")
        return self


type DevDebugAssetSourceV1 = Annotated[
    DevCurrentBufferSourceV1 | DevSavedDraftSourceV1,
    Field(discriminator="source_kind"),
]


class DevListCommandV1(_ServiceModel):
    """List saved maps, saved scenarios, or both; the default selection is all."""

    command_type: Literal["list"] = "list"
    asset_kind: Literal["map", "scenario", "all"] = "all"


class DevNewMapCommandV1(_ServiceModel):
    """Create an unsaved blank map with the requested safe asset ID."""

    command_type: Literal["new_map"] = "new_map"
    asset_id: SafeAssetId = "untitled_map"


class DevNewScenarioCommandV1(_ServiceModel):
    """Create a blank scenario, copy a saved map, or duplicate a saved scenario.

    Blank creation accepts no source. Copy and duplicate modes require an exact
    saved revision of the corresponding asset kind. The new draft is always
    version 2: blank and map-copy drafts use Red Zone depth
    ``DEFAULT_TDM_RED_ZONE_DEPTH`` (5.0), and a duplicate keeps its source's
    depth (0.0 for a version 1 source).

    Attributes
    ----------
    command_type : Literal["new_scenario"]
        Fixed command tag.
    asset_id : str, default="untitled_scenario"
        Safe ID for the new unsaved draft: lowercase letters and digits in
        words joined by single underscores, 1 to 64 characters.
    creation_mode : {"blank", "copy_saved_map", "duplicate_saved_scenario"}
        How the draft is made; default "blank". "blank" starts from a default
        blank map; "copy_saved_map" copies a saved map; and
        "duplicate_saved_scenario" copies a saved scenario's content.
    source : DevPersistedSourceV1 or None, default=None
        The exact saved revision to copy. It must be None for "blank", a map
        for "copy_saved_map" and a scenario for "duplicate_saved_scenario";
        any other combination fails validation.
    """

    command_type: Literal["new_scenario"] = "new_scenario"
    asset_id: SafeAssetId = "untitled_scenario"
    creation_mode: Literal[
        "blank",
        "copy_saved_map",
        "duplicate_saved_scenario",
    ] = "blank"
    source: DevPersistedSourceV1 | None = None

    @model_validator(mode="after")
    def _validate_source(self) -> DevNewScenarioCommandV1:
        """Require the source kind appropriate to the selected scenario-creation
        mode.
        """
        if self.creation_mode == "blank":
            if self.source is not None:
                raise ValueError("blank scenario creation does not accept a source")
            return self
        if self.source is None:
            raise ValueError(f"{self.creation_mode} requires a persisted source")
        expected_kind = "map" if self.creation_mode == "copy_saved_map" else "scenario"
        if self.source.asset_kind != expected_kind:
            raise ValueError(f"{self.creation_mode} requires a {expected_kind} source")
        return self


class DevOpenCommandV1(_ServiceModel):
    """Open one exact saved asset revision into an editor buffer.

    A saved version 1 scenario opens as version 2 at Red Zone depth 0.0 with the
    same revision, so the next Save writes the following revision as version 2.
    The saved file itself is not changed.
    """

    command_type: Literal["open"] = "open"
    source: DevPersistedSourceV1


class DevSaveCommandV1(_ServiceModel):
    """Save a complete draft only if its expected current revision still matches.

    The draft is stored in the version it has.
    """

    command_type: Literal["save"] = "save"
    draft: DevDraftPayload
    expected_revision: DevDraftRevision


class DevSaveAsCommandV1(_ServiceModel):
    """Save a copy under a new safe identity without overwriting an existing draft.

    The copy keeps the payload's version, so a version 1 scenario payload is
    stored as version 1.
    """

    command_type: Literal["save_as"] = "save_as"
    draft: DevDraftPayload
    asset_id: SafeAssetId


class DevValidateCommandV1(_ServiceModel):
    """Check a complete draft without saving it or replacing the running scenario."""

    command_type: Literal["validate"] = "validate"
    draft: DevDraftPayload


class DevDeleteCommandV1(_ServiceModel):
    """Delete an exact saved asset identity using its last observed revision as a
    guard.
    """

    command_type: Literal["delete"] = "delete"
    source: DevSavedDraftSourceV1


class DevOpenInDebugCommandV1(_ServiceModel):
    """Compile a current buffer or exact saved revision and request live
    installation.
    """

    command_type: Literal["open_in_debug"] = "open_in_debug"
    source: DevDebugAssetSourceV1


type DevAuthoringCommandV1 = Annotated[
    DevListCommandV1
    | DevNewMapCommandV1
    | DevNewScenarioCommandV1
    | DevOpenCommandV1
    | DevSaveCommandV1
    | DevSaveAsCommandV1
    | DevValidateCommandV1
    | DevDeleteCommandV1
    | DevOpenInDebugCommandV1,
    Field(discriminator="command_type"),
]


class DevAuthoringCommandRequestV1(RootModel[DevAuthoringCommandV1]):
    """Strict root command model consumed directly by the live-only endpoint."""

    model_config = ConfigDict(frozen=True, strict=True)


class DevRedZoneStripsV1(_ServiceModel):
    """Both teams' Red Zone floor strips, for the Scenario Author canvas tint.

    Each team's Red Zone is a full-height strip on its own spawn side. When an
    agent dies inside its own team's strip, the enemy team gets 2 points. The
    host computes the strips from the compiled config; the browser only draws
    them. This is a live service reply, never saved.

    Attributes
    ----------
    team_a_x_range : tuple[float, float]
        Team A strip as inclusive (x_min, x_max) world x bounds, exactly the
        float32 bounds Core scores with (models.red_zone_x_range). A collapsed
        range (x_min equal to x_max) is legal and paints nothing.
    team_b_x_range : tuple[float, float]
        Team B strip, with the same meaning.
    """

    team_a_x_range: tuple[float, float]
    team_b_x_range: tuple[float, float]


class DevValidationSummaryV1(_ServiceModel):
    """Report whether a draft is executable, with linked problems and available digests.

    Scenario summaries may include ten effective movement speeds and, when the
    draft declares a positive Red Zone depth, both teams' Red Zone strips
    (red_zone). Missing runtime fields remain None for invalid drafts and
    map-only validation; red_zone is also None at depth 0.

    Attributes
    ----------
    asset_kind : {"map", "scenario"}
        Which kind of draft was validated.
    execution_valid : bool
        True when a map has no error-level problem, or a scenario compiles.
        Warnings alone keep it True.
    semantic_digest : str or None, default=None
        Lowercase 64-character SHA-256 hex digest of the draft's normalized
        content. None for an invalid draft.
    map_semantic_digest : str or None, default=None
        The same kind of digest for the map content alone; for a valid map it
        equals semantic_digest. None for an invalid draft.
    resolved_configuration_digest, resolved_initial_state_digest : str or None
        Digests of a valid scenario's compiled EnvConfig and initial state.
        None (the default) for maps and invalid scenarios.
    effective_movement_speeds : tuple of 10 float or None, default=None
        For a valid scenario, the distance each global slot (0 to 9) can move
        in one step at the compiled initial state, in world units. Dead and
        unused slots are 0.0. None for maps and invalid scenarios.
    problems : tuple of DevAuthoringProblemV1, default=()
        Every error and warning found, each linked to a field path.
    red_zone : DevRedZoneStripsV1 or None, default=None
        Both teams' Red Zone strips for a valid scenario with a positive
        depth (map units). None at depth 0.0, for maps and for invalid
        scenarios.
    """

    asset_kind: DevAssetKind
    execution_valid: bool
    semantic_digest: SemanticDigest | None = None
    map_semantic_digest: SemanticDigest | None = None
    resolved_configuration_digest: SemanticDigest | None = None
    resolved_initial_state_digest: SemanticDigest | None = None
    effective_movement_speeds: (
        Annotated[
            tuple[float, ...],
            Field(min_length=10, max_length=10),
        ]
        | None
    ) = None
    problems: tuple[DevAuthoringProblemV1, ...] = ()
    red_zone: DevRedZoneStripsV1 | None = None


class DevAssetSummaryV1(_ServiceModel):
    """Describe one saved asset revision for discovery, including dimensions and
    validity.
    """

    asset_kind: DevAssetKind
    source_kind: Literal["saved_draft"] = "saved_draft"
    asset_id: SafeAssetId
    revision: DevSavedRevision
    name: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=120),
    ]
    map_width: float
    map_height: float
    execution_valid: bool


class DevDeletedAssetSummaryV1(_ServiceModel):
    """Report the deleted asset identity, greatest revision, and number of removed
    revisions.
    """

    asset_kind: DevAssetKind
    asset_id: SafeAssetId
    latest_revision: DevSavedRevision
    deleted_revision_count: Annotated[int, Field(ge=1)]


class DevDebugLoadSummaryV1(_ServiceModel):
    """Identify the exact compiled map preview or scenario installed in the live
    debugger.

    Source IDs describe the input. Semantic and resolved digests bind the normalized
    physical content and runtime endpoint actually loaded.
    """

    source_kind: Literal["current_buffer", "saved_draft"]
    asset_kind: DevAssetKind
    debug_profile: Literal["authored_scenario", "default_tdm_map_preview"]
    asset_id: SafeAssetId | None = None
    revision: DevDraftRevision | None = None
    source_name: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=120),
    ]
    scenario_name: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=120),
    ]
    map_width: float
    map_height: float
    scenario_semantic_digest: SemanticDigest
    map_semantic_digest: SemanticDigest
    resolved_configuration_digest: SemanticDigest
    resolved_initial_state_digest: SemanticDigest


class DevAuthoringCatalogV1(_ServiceModel):
    """Read-only existing mechanics exposed to numeric authoring inspectors."""

    schema_id: Literal["dev-authoring-catalog@1"] = "dev-authoring-catalog@1"
    mechanics_catalog_digest: SemanticDigest
    maximum_obstacle_slots: int
    canonical_product_movement_scale: float
    fixed_grid_world_units: Annotated[float, Field(ge=1.0, le=1.0)] = 1.0
    fixed_snap_world_units: Annotated[float, Field(ge=0.5, le=0.5)] = 0.5
    class_mechanics: tuple[ClassMechanicsV1, ...]
    status_channels: tuple[StatusMechanicV1, ...]
    aura_mechanics: tuple[AuraMechanicV1, ...]


def _build_authoring_catalog() -> DevAuthoringCatalogV1:
    """Expose the shared mechanics catalog with editor capacities and fixed grid
    settings.
    """
    catalog = build_static_mechanics_catalog_v1()
    return DevAuthoringCatalogV1(
        mechanics_catalog_digest=catalog.canonical_digest_sha256,
        maximum_obstacle_slots=MAX_OBSTACLE_SLOTS,
        canonical_product_movement_scale=float(CANONICAL_PRODUCT_MOVEMENT_SCALE),
        class_mechanics=catalog.class_mechanics[1:],
        status_channels=catalog.status_channels,
        aura_mechanics=catalog.aura_mechanics,
    )


_AUTHORING_CATALOG = _build_authoring_catalog()


class DevAuthoringCommandResponseV1(_ServiceModel):
    """JSON-safe result for every live-only authoring command."""

    ok: bool
    command_type: CommandType
    draft: DevDraftPayload | None = None
    deleted: DevDeletedAssetSummaryV1 | None = None
    assets: tuple[DevAssetSummaryV1, ...] = ()
    validation: DevValidationSummaryV1 | None = None
    debug_load: DevDebugLoadSummaryV1 | None = None
    pending_operation_id: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    problems: tuple[DevAuthoringProblemV1, ...] = ()
    catalog: DevAuthoringCatalogV1 = _AUTHORING_CATALOG


@dataclass(frozen=True, slots=True)
class LoadedDevScenarioSnapshotV1:
    """Host-only immutable object that becomes Combat Debugger reset authority."""

    source: DevDebugAssetSourceV1
    compiled: CompiledDevScenarioV1
    summary: DevDebugLoadSummaryV1


def debugger_scenario_from_snapshot(
    snapshot: LoadedDevScenarioSnapshotV1,
) -> DebuggerScenario:
    """Adapt one compiled snapshot to the live debugger's scenario interface.

    Parameters
    ----------
    snapshot : LoadedDevScenarioSnapshotV1
        Successfully compiled source, runtime endpoint and matching load summary.

    Returns
    -------
    DebuggerScenario
        Interactive scenario with a factory that deep-copies the frozen config/state.
        It retains authored provenance and selects the first active Team A slot.

    Notes
    -----
    Does not load new source bytes or install the scenario. Future resets use the same
    compiled source, even when the editor or saved draft changes. A map preview's
    source identity ends with ``:profile:default-tdm-map-preview@2``; version 2 of
    the profile plays at Red Zone depth ``DEFAULT_TDM_RED_ZONE_DEPTH``.
    """
    compiled = snapshot.compiled
    source = snapshot.source
    if isinstance(source, DevCurrentBufferSourceV1):
        source_identity = (
            f"{source.asset_kind}:current_buffer:{source.draft.asset_id}:"
            f"revision:{source.draft.revision}"
        )
    else:
        source_identity = (
            f"{source.asset_kind}:saved_draft:{source.asset_id}:"
            f"revision:{source.revision}"
        )
    if snapshot.summary.debug_profile == "default_tdm_map_preview":
        source_identity += ":profile:default-tdm-map-preview@2"

    controlled_slot = next(
        row.global_slot
        for row in compiled.content.roster
        if row.team == "A" and row.team_local_slot <= compiled.content.team_a_size
    )
    map_source = compiled.content.source_map_provenance
    map_id = approved_map_id(
        None if map_source is None else map_source.asset_id,
        compiled.map_semantic_digest,
    )

    def build_scenario() -> tuple[EnvConfig, EnvState]:
        """Deep-copy the frozen compiled config/state for each new live reset."""
        return copy.deepcopy(compiled.config), copy.deepcopy(compiled.initial_state)

    return DebuggerScenario(
        name=(
            "dev_map_preview_"
            if snapshot.summary.debug_profile == "default_tdm_map_preview"
            else "dev_scenario_"
        )
        + compiled.semantic_digest[:16],
        title=compiled.content.name,
        description=compiled.content.description,
        mode="interactive",
        build_scenario=build_scenario,
        frames=(),
        default_controlled_slot=controlled_slot,
        provenance=DebuggerScenarioProvenance(
            source_kind=source.source_kind,
            source_identity=source_identity,
            scenario_semantic_digest=compiled.semantic_digest,
            map_semantic_digest=compiled.map_semantic_digest,
            resolved_configuration_digest=compiled.resolved_configuration_digest,
            resolved_initial_state_digest=compiled.resolved_initial_state_digest,
            map_id=map_id,
        ),
    )


class DevScenarioLoadAttemptV1(_ServiceModel):
    """Report installed success, pending work, or errors without a partial snapshot.

    A pending operation has ok=False and no summary or problems. Only the live
    service can adopt it; its accepted callback updates the loader snapshot.
    """

    pending_operation_id: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    ok: bool
    summary: DevDebugLoadSummaryV1 | None = None
    problems: tuple[DevAuthoringProblemV1, ...] = ()


def _service_problem(
    code: str,
    message: str,
    *,
    field_path: str = "command",
) -> DevAuthoringProblemV1:
    """Build a stable authoring-command error linked to a field, defaulting to the
    command root.
    """
    return DevAuthoringProblemV1(
        severity="error",
        stable_code=code,
        message=message,
        field_path=field_path,
    )


def _red_zone_strips(config: EnvConfig) -> DevRedZoneStripsV1 | None:
    """Return both teams' Red Zone strips for a compiled scenario config.

    Parameters
    ----------
    config : EnvConfig
        Validated config from compile_dev_scenario. Its depth is a Python float;
        its spawn pads are float32 (2, 5, 2) in Team A, Team B order.

    Returns
    -------
    DevRedZoneStripsV1 | None
        None when the depth is 0.0 (rule off). Otherwise each team's side comes
        from models.red_zone_team_on_right on its five pad x values (unused
        pads included) and its strip from models.red_zone_x_range, the same
        host rules the Viewer's scene records use.

    Notes
    -----
    Host-only NumPy and Python arithmetic; no JAX program runs.
    """
    depth = config.team_deathmatch_red_zone_depth
    if depth == 0.0:
        return None
    pads = np.asarray(config.team_spawn_pad_positions, dtype=np.float64)
    team_a_range, team_b_range = (
        red_zone_x_range(
            config.map_width,
            depth,
            red_zone_team_on_right(config.map_width, pads[team, :, 0].tolist()),
        )
        for team in range(2)
    )
    return DevRedZoneStripsV1(team_a_x_range=team_a_range, team_b_x_range=team_b_range)


def _validation_summary(draft: DevDraftPayload) -> DevValidationSummaryV1:
    """Validate a map or compile a scenario and return truthful available summary
    fields.

    A valid scenario also reports its Red Zone strips (None at depth 0); map
    drafts and invalid scenarios report none.
    """
    if isinstance(draft, DevMapDraftV1):
        problems = validate_map_content(draft.content)
        valid = not any(problem.severity == "error" for problem in problems)
        digest = map_semantic_digest(draft.content) if valid else None
        return DevValidationSummaryV1(
            asset_kind="map",
            execution_valid=valid,
            semantic_digest=digest,
            map_semantic_digest=digest,
            problems=problems,
        )
    try:
        compiled = compile_dev_scenario(draft)
    except DevAuthoringValidationError as error:
        return DevValidationSummaryV1(
            asset_kind="scenario",
            execution_valid=False,
            problems=error.problems,
        )
    problems = compiled.problems
    config = compiled.config
    state = compiled.initial_state
    effective_movement_speeds = tuple(
        float(value)
        for value in np.asarray(
            combat.derive_effective_movement_speeds(
                state.slow_durations,
                state.priest_blessing_of_freedom_slow_floor_durations,
                state.stun_durations,
                state.spawn_shield_durations,
                config.agent_profile.base_movement_speeds,
                config.spawn_shield_movement_speed,
                config.agent_profile.active_mask & state.alive_mask,
                config.ordinary_movement_distance_scale,
            )
        )
    )
    return DevValidationSummaryV1(
        asset_kind="scenario",
        execution_valid=True,
        semantic_digest=compiled.semantic_digest,
        map_semantic_digest=compiled.map_semantic_digest,
        resolved_configuration_digest=compiled.resolved_configuration_digest,
        resolved_initial_state_digest=compiled.resolved_initial_state_digest,
        effective_movement_speeds=effective_movement_speeds,
        problems=problems,
        red_zone=_red_zone_strips(config),
    )


class DevScenarioLoadService:
    """Compile exact authored assets before replacing the current live snapshot.

    Parameters
    ----------
    store : DevAssetStore
        Protected local store used to resolve saved revisions.
    install_snapshot : callable or None, optional
        Host callback receiving a fully compiled snapshot. None retains the snapshot
        locally without installing it in another live service. A callback returns
        None after installation, or a pending System operation ID. For pending
        work it later calls accept_installed_snapshot only after real adoption.

    Notes
    -----
    The current snapshot changes only after compilation and the installer succeed.
    An installer must apply its own atomic replacement contract; this service cannot
    undo arbitrary external callback side effects.
    """

    def __init__(
        self,
        store: DevAssetStore,
        *,
        install_snapshot: Callable[[LoadedDevScenarioSnapshotV1], str | None]
        | None = None,
    ) -> None:
        """Keep the store and optional installer; start with no loaded snapshot."""
        self._store = store
        self._install_snapshot = install_snapshot
        self._current_snapshot: LoadedDevScenarioSnapshotV1 | None = None

    @property
    def current_snapshot(self) -> LoadedDevScenarioSnapshotV1 | None:
        """Return the last successfully installed compiled snapshot, or None before a
        load.
        """
        return self._current_snapshot

    def _open_source(
        self,
        source: DevDebugAssetSourceV1,
    ) -> DevDraftPayload:
        """Deep-copy a current buffer or load the exact requested saved revision."""
        if isinstance(source, DevCurrentBufferSourceV1):
            return source.draft.model_copy(deep=True)
        return self._store.load_draft(
            source.asset_kind,
            source.asset_id,
            revision=source.revision,
        )

    @staticmethod
    def _compile_source(
        source: DevDebugAssetSourceV1,
        opened: DevDraftPayload,
    ) -> tuple[
        CompiledDevScenarioV1,
        str,
        Literal["authored_scenario", "default_tdm_map_preview"],
    ]:
        """Compile a scenario directly or build a temporary default TDM preview of a
        map.

        A scenario of either version compiles at its declared depth (0.0 for
        version 1). The preview is a version 2 scenario at Red Zone depth
        ``DEFAULT_TDM_RED_ZONE_DEPTH``. Preview creation does not modify or save the
        source map.
        """
        if source.asset_kind == "scenario":
            if not isinstance(opened, (DevScenarioDraftV1, DevScenarioDraftV2)):
                raise TypeError("scenario Debug source resolved a non-scenario asset")
            return (
                compile_dev_scenario(opened),
                opened.content.name,
                "authored_scenario",
            )

        if not isinstance(opened, DevMapDraftV1):
            raise TypeError("map Debug source resolved a non-map asset")
        map_problems = validate_map_content(opened.content)
        if any(problem.severity == "error" for problem in map_problems):
            raise DevAuthoringValidationError(map_problems)
        preview = new_scenario_draft(
            "default_tdm_map_preview",
            source_map=opened,
            red_zone_depth=DEFAULT_TDM_RED_ZONE_DEPTH,
        )
        preview = preview.model_copy(
            update={
                "content": preview.content.model_copy(
                    update={
                        "name": "Default TDM map preview",
                        "description": (
                            f"Transient deterministic 5v5 Team Deathmatch preview "
                            f"of {opened.content.name}; the source map is not "
                            "modified or saved."
                        ),
                    }
                )
            }
        )
        return (
            compile_dev_scenario(preview),
            opened.content.name,
            "default_tdm_map_preview",
        )

    @staticmethod
    def _summary(
        source: DevDebugAssetSourceV1,
        compiled: CompiledDevScenarioV1,
        *,
        source_name: str,
        debug_profile: Literal["authored_scenario", "default_tdm_map_preview"],
    ) -> DevDebugLoadSummaryV1:
        """Bind the loaded source identity to the exact compiled runtime and semantic
        digests.
        """
        asset_id: str | None = None
        revision: int | None = None
        if isinstance(source, DevCurrentBufferSourceV1):
            asset_id = source.draft.asset_id
            revision = source.draft.revision
        else:
            asset_id = source.asset_id
            revision = source.revision
        return DevDebugLoadSummaryV1(
            source_kind=source.source_kind,
            asset_kind=source.asset_kind,
            debug_profile=debug_profile,
            asset_id=asset_id,
            revision=revision,
            source_name=source_name,
            scenario_name=compiled.content.name,
            map_width=compiled.config.map_width,
            map_height=compiled.config.map_height,
            scenario_semantic_digest=compiled.semantic_digest,
            map_semantic_digest=compiled.map_semantic_digest,
            resolved_configuration_digest=compiled.resolved_configuration_digest,
            resolved_initial_state_digest=compiled.resolved_initial_state_digest,
        )

    def load(self, source: DevDebugAssetSourceV1) -> DevScenarioLoadAttemptV1:
        """Resolve, compile and install one source before exposing it as current.

        Parameters
        ----------
        source : DevDebugAssetSourceV1
            Complete current buffer or exact saved map/scenario revision.

        Returns
        -------
        DevScenarioLoadAttemptV1
            Success with the exact loaded summary, or failure with linked problems.
            Handled failures leave this service's current snapshot unchanged.

        Notes
        -----
        May read a saved draft and perform compilation/reset work. The optional
        installer
        runs only after the full snapshot is ready. Expected storage, validation and
        install
        errors become result problems; unexpected failures propagate.
        """
        try:
            opened = self._open_source(source)
            compiled, source_name, debug_profile = self._compile_source(source, opened)
            summary = self._summary(
                source,
                compiled,
                source_name=source_name,
                debug_profile=debug_profile,
            )
        except DevAuthoringValidationError as error:
            return DevScenarioLoadAttemptV1(ok=False, problems=error.problems)
        except DevAssetIntegrityError as error:
            if error.problems:
                return DevScenarioLoadAttemptV1(ok=False, problems=error.problems)
            return DevScenarioLoadAttemptV1(
                ok=False,
                problems=(
                    _service_problem(
                        "debug-asset-load-failed",
                        str(error),
                        field_path="source",
                    ),
                ),
            )
        except (DevAssetStoreError, ValueError) as error:
            return DevScenarioLoadAttemptV1(
                ok=False,
                problems=(
                    _service_problem(
                        "debug-asset-load-failed",
                        str(error),
                        field_path="source",
                    ),
                ),
            )
        snapshot = LoadedDevScenarioSnapshotV1(
            source=source,
            compiled=compiled,
            summary=summary,
        )
        try:
            if self._install_snapshot is not None:
                pending = self._install_snapshot(snapshot)
                if pending is not None:
                    return DevScenarioLoadAttemptV1(
                        ok=False, pending_operation_id=pending
                    )
        except (RuntimeError, TypeError, ValueError) as error:
            return DevScenarioLoadAttemptV1(
                ok=False,
                problems=(
                    _service_problem(
                        "debug-asset-install-failed",
                        str(error),
                        field_path="source",
                    ),
                ),
            )
        self.accept_installed_snapshot(snapshot)
        return DevScenarioLoadAttemptV1(ok=True, summary=summary)

    def accept_installed_snapshot(self, snapshot: LoadedDevScenarioSnapshotV1) -> None:
        """Record the exact compiled snapshot after the live service adopts it.

        This trusted host callback performs no I/O or compilation. A deferred
        installer calls it once on adoption, never on cancellation or failure.
        Ordinary synchronous loads call it before returning ok=True.
        """
        self._current_snapshot = snapshot

    def list_persisted(
        self,
        asset_kind: DevAssetKind = "scenario",
        *,
        include_invalid_drafts: bool = False,
    ) -> tuple[DevAssetSummaryV1, ...]:
        """Summarize the latest parseable revision of each asset in one collection.

        Parameters
        ----------
        asset_kind : {"map", "scenario"}, optional
            Collection to inspect; defaults to scenarios.
        include_invalid_drafts : bool, optional
            False includes only executable drafts. True also includes parseable drafts
            whose execution checks fail, so the editor can offer them for repair.

        Returns
        -------
        tuple of DevAssetSummaryV1
            Stable saved-asset summaries. Missing, corrupt, or unreadable draft entries
            caught by the expected validation path are skipped.

        Notes
        -----
        Reads and validates saved content without saving drafts or installing a
        scenario.
        """
        summaries: list[DevAssetSummaryV1] = []
        for reference in self._store.iter_draft_references(
            asset_kind,
            latest_only=True,
        ):
            try:
                draft = self._store.load_draft(
                    asset_kind,
                    reference.asset_id,
                    revision=reference.revision,
                )
                validation = _validation_summary(draft)
            except DevAssetStoreError, DevAuthoringValidationError, ValueError:
                continue
            if not validation.execution_valid and not include_invalid_drafts:
                continue
            summaries.append(
                DevAssetSummaryV1(
                    asset_kind=asset_kind,
                    source_kind="saved_draft",
                    asset_id=reference.asset_id,
                    revision=reference.revision,
                    name=draft.content.name,
                    map_width=(
                        draft.content.width
                        if isinstance(draft, DevMapDraftV1)
                        else draft.content.embedded_map.width
                    ),
                    map_height=(
                        draft.content.height
                        if isinstance(draft, DevMapDraftV1)
                        else draft.content.embedded_map.height
                    ),
                    execution_valid=validation.execution_valid,
                )
            )
        return tuple(summaries)

    def discover(self) -> tuple[DevAssetSummaryV1, ...]:
        """List executable saved scenarios followed by executable saved maps.

        Returns
        -------
        tuple of DevAssetSummaryV1
            Latest valid saved identities available to the live Combat interface.

        Notes
        -----
        Uses the same validation path as collection listing; no live snapshot is
        replaced.
        """
        return self.list_persisted("scenario") + self.list_persisted("map")


class DevClientAuthoringBinding:
    """Serialize parsed editor commands for one local draft store.

    Parameters
    ----------
    store : DevAssetStore
        Owner of saved map/scenario revisions.
    scenario_loader : DevScenarioLoadService or None, optional
        Loader connected to the live service. None creates a loader using this store.

    Notes
    -----
    The lock serializes command execution in this process. The store separately owns
    cross-process mutation locks and revision checks.
    """

    def __init__(
        self,
        store: DevAssetStore,
        *,
        scenario_loader: DevScenarioLoadService | None = None,
    ) -> None:
        """Bind a store, a command lock, and the supplied or default scenario loader."""
        self._store = store
        self._lock = Lock()
        self._scenario_loader = (
            DevScenarioLoadService(store)
            if scenario_loader is None
            else scenario_loader
        )

    @property
    def scenario_loader(self) -> DevScenarioLoadService:
        """Return the loader that owns this binding's live compiled snapshot."""
        return self._scenario_loader

    def _load_persisted(
        self,
        source: DevPersistedSourceV1,
    ) -> DevDraftPayload:
        """Load one exact saved source revision through the protected draft store."""
        return self._store.load_draft(
            source.asset_kind,
            source.asset_id,
            revision=source.revision,
        )

    def _list_assets(
        self,
        requested: Literal["map", "scenario", "all"],
    ) -> tuple[DevAssetSummaryV1, ...]:
        """List requested asset kinds with parseable invalid drafts retained for
        editing.
        """
        summaries: list[DevAssetSummaryV1] = []
        if requested in ("scenario", "all"):
            summaries.extend(
                self._scenario_loader.list_persisted(
                    "scenario",
                    include_invalid_drafts=True,
                )
            )
        if requested in ("map", "all"):
            summaries.extend(
                self._scenario_loader.list_persisted(
                    "map",
                    include_invalid_drafts=True,
                )
            )
        return tuple(summaries)

    def _new_scenario(self, command: DevNewScenarioCommandV1) -> DevScenarioDraftV2:
        """Resolve the validated creation mode using the shared blank/copy factories.

        Blank and map-copy drafts use ``DEFAULT_TDM_RED_ZONE_DEPTH``; a duplicate
        keeps ``declared_red_zone_depth`` of its source.
        """
        if command.creation_mode == "blank":
            return new_scenario_draft(
                command.asset_id,
                red_zone_depth=DEFAULT_TDM_RED_ZONE_DEPTH,
            )
        if command.source is None:
            raise AssertionError("validated nonblank creation requires a source")
        source = self._load_persisted(command.source)
        if command.creation_mode == "copy_saved_map":
            if not isinstance(source, DevMapDraftV1):
                raise TypeError("copy_saved_map resolved a non-map source")
            return new_scenario_draft(
                command.asset_id,
                source_map=source,
                red_zone_depth=DEFAULT_TDM_RED_ZONE_DEPTH,
            )
        if isinstance(source, DevMapDraftV1):
            raise TypeError("duplicate_saved_scenario resolved a non-scenario source")
        return duplicate_scenario_draft(
            source,
            asset_id=command.asset_id,
            red_zone_depth=declared_red_zone_depth(source.content),
        )

    def apply_command(
        self,
        request: DevAuthoringCommandRequestV1,
    ) -> DevAuthoringCommandResponseV1:
        """Apply one parsed editor request while holding the binding's command lock.

        Parameters
        ----------
        request : DevAuthoringCommandRequestV1
            Strict command root selecting one supported authoring operation.

        Returns
        -------
        DevAuthoringCommandResponseV1
            Command-specific draft, listing, validation, deletion or load data. Expected
            storage/validation failures return ``ok=False`` with linked problems.

        Notes
        -----
        Only the selected operation runs. Save and delete commands change local draft
        files; Open In Debug may replace the live scenario. New and Validate commands do
        not save content. Open returns a saved version 1 scenario as version 2 at Red
        Zone depth 0.0 without changing the file. Unexpected failures are not silently
        converted into success.
        """
        with self._lock:
            return self._apply_command(request)

    def _apply_command(
        self,
        request: DevAuthoringCommandRequestV1,
    ) -> DevAuthoringCommandResponseV1:
        """Dispatch one command under the caller-held lock and translate expected
        failures.
        """
        command = request.root
        try:
            if isinstance(command, DevListCommandV1):
                return DevAuthoringCommandResponseV1(
                    ok=True,
                    command_type=command.command_type,
                    assets=self._list_assets(command.asset_kind),
                )
            if isinstance(command, DevNewMapCommandV1):
                draft = new_map_draft(command.asset_id)
                return DevAuthoringCommandResponseV1(
                    ok=True,
                    command_type=command.command_type,
                    draft=draft,
                    validation=_validation_summary(draft),
                )
            if isinstance(command, DevNewScenarioCommandV1):
                draft = self._new_scenario(command)
                return DevAuthoringCommandResponseV1(
                    ok=True,
                    command_type=command.command_type,
                    draft=draft,
                    validation=_validation_summary(draft),
                )
            if isinstance(command, DevOpenCommandV1):
                opened = self._load_persisted(command.source)
                if not isinstance(opened, DevMapDraftV1):
                    # The editor edits only version 2; the saved file is unchanged.
                    opened = upgrade_scenario_draft(opened)
                return DevAuthoringCommandResponseV1(
                    ok=True,
                    command_type=command.command_type,
                    draft=opened,
                    validation=_validation_summary(opened),
                )
            if isinstance(command, DevSaveCommandV1):
                saved = self._store.save_draft(
                    command.draft,
                    expected_revision=command.expected_revision,
                )
                return DevAuthoringCommandResponseV1(
                    ok=True,
                    command_type=command.command_type,
                    draft=saved,
                    validation=_validation_summary(saved),
                )
            if isinstance(command, DevSaveAsCommandV1):
                saved = self._store.save_draft_as(
                    command.draft,
                    asset_id=command.asset_id,
                )
                return DevAuthoringCommandResponseV1(
                    ok=True,
                    command_type=command.command_type,
                    draft=saved,
                    validation=_validation_summary(saved),
                )
            if isinstance(command, DevValidateCommandV1):
                validation = _validation_summary(command.draft)
                return DevAuthoringCommandResponseV1(
                    ok=validation.execution_valid,
                    command_type=command.command_type,
                    draft=command.draft,
                    validation=validation,
                    problems=validation.problems,
                )
            if isinstance(command, DevDeleteCommandV1):
                deleted = self._store.delete_draft(
                    command.source.asset_kind,
                    command.source.asset_id,
                    expected_revision=command.source.revision,
                )
                return DevAuthoringCommandResponseV1(
                    ok=True,
                    command_type=command.command_type,
                    deleted=DevDeletedAssetSummaryV1(
                        asset_kind=command.source.asset_kind,
                        asset_id=command.source.asset_id,
                        latest_revision=command.source.revision,
                        deleted_revision_count=len(deleted),
                    ),
                )
            attempt = self._scenario_loader.load(command.source)
            return DevAuthoringCommandResponseV1(
                ok=attempt.ok,
                command_type=command.command_type,
                debug_load=attempt.summary,
                pending_operation_id=attempt.pending_operation_id,
                problems=attempt.problems,
            )
        except DevAuthoringValidationError as error:
            return DevAuthoringCommandResponseV1(
                ok=False,
                command_type=command.command_type,
                problems=error.problems,
            )
        except DevDraftRevisionConflictError as error:
            problem = _service_problem(
                "draft-revision-conflict",
                str(error),
                field_path=(
                    "source.revision"
                    if isinstance(command, DevDeleteCommandV1)
                    else "expected_revision"
                ),
            )
        except DevAssetAlreadyExistsError as error:
            problem = _service_problem("asset-already-exists", str(error))
        except DevAssetNotFoundError as error:
            problem = _service_problem(
                "asset-not-found", str(error), field_path="source"
            )
        except DevAssetIntegrityError as error:
            if error.problems:
                return DevAuthoringCommandResponseV1(
                    ok=False,
                    command_type=command.command_type,
                    problems=error.problems,
                )
            problem = _service_problem(
                "asset-integrity-failure", str(error), field_path="source"
            )
        except (DevAssetStoreError, TypeError, ValueError) as error:
            problem = _service_problem("authoring-command-failed", str(error))
        return DevAuthoringCommandResponseV1(
            ok=False,
            command_type=command.command_type,
            problems=(problem,),
        )


__all__ = [
    "DevAssetSummaryV1",
    "DevAuthoringCatalogV1",
    "DevAuthoringCommandRequestV1",
    "DevAuthoringCommandResponseV1",
    "DevAuthoringCommandV1",
    "DevClientAuthoringBinding",
    "DevCurrentBufferSourceV1",
    "DevDebugAssetSourceV1",
    "DevDebugLoadSummaryV1",
    "DevDeleteCommandV1",
    "DevDeletedAssetSummaryV1",
    "DevListCommandV1",
    "DevNewMapCommandV1",
    "DevNewScenarioCommandV1",
    "DevOpenCommandV1",
    "DevOpenInDebugCommandV1",
    "DevPersistedSourceV1",
    "DevSaveAsCommandV1",
    "DevSaveCommandV1",
    "DevSavedDraftSourceV1",
    "DevScenarioLoadAttemptV1",
    "DevScenarioLoadService",
    "DevValidateCommandV1",
    "DevValidationSummaryV1",
    "LoadedDevScenarioSnapshotV1",
    "debugger_scenario_from_snapshot",
]
