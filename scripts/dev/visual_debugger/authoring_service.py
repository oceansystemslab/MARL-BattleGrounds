"""One live-only DevClient authoring binding and authoritative load service."""

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
from marl_battlegrounds.evaluation.models import (
    AuraMechanicV1,
    ClassMechanicsV1,
    StatusMechanicV1,
)
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
    SafeAssetId,
    SemanticDigest,
    duplicate_scenario_draft,
    new_map_draft,
    new_scenario_draft,
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

type DevDraftPayload = DevMapDraftV1 | DevScenarioDraftV1
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
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        serialize_by_alias=True,
        strict=True,
    )


class DevSavedDraftSourceV1(_ServiceModel):
    source_kind: Literal["saved_draft"] = "saved_draft"
    asset_kind: DevAssetKind
    asset_id: SafeAssetId
    revision: DevSavedRevision


type DevPersistedSourceV1 = DevSavedDraftSourceV1


class DevCurrentBufferSourceV1(_ServiceModel):
    source_kind: Literal["current_buffer"] = "current_buffer"
    asset_kind: DevAssetKind
    draft: DevDraftPayload

    @model_validator(mode="after")
    def _validate_asset_kind(self) -> DevCurrentBufferSourceV1:
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
    command_type: Literal["list"] = "list"
    asset_kind: Literal["map", "scenario", "all"] = "all"


class DevNewMapCommandV1(_ServiceModel):
    command_type: Literal["new_map"] = "new_map"
    asset_id: SafeAssetId = "untitled_map"


class DevNewScenarioCommandV1(_ServiceModel):
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
    command_type: Literal["open"] = "open"
    source: DevPersistedSourceV1


class DevSaveCommandV1(_ServiceModel):
    command_type: Literal["save"] = "save"
    draft: DevDraftPayload
    expected_revision: DevDraftRevision


class DevSaveAsCommandV1(_ServiceModel):
    command_type: Literal["save_as"] = "save_as"
    draft: DevDraftPayload
    asset_id: SafeAssetId


class DevValidateCommandV1(_ServiceModel):
    command_type: Literal["validate"] = "validate"
    draft: DevDraftPayload


class DevDeleteCommandV1(_ServiceModel):
    command_type: Literal["delete"] = "delete"
    source: DevSavedDraftSourceV1


class DevOpenInDebugCommandV1(_ServiceModel):
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


class DevValidationSummaryV1(_ServiceModel):
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


class DevAssetSummaryV1(_ServiceModel):
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
    asset_kind: DevAssetKind
    asset_id: SafeAssetId
    latest_revision: DevSavedRevision
    deleted_revision_count: Annotated[int, Field(ge=1)]


class DevDebugLoadSummaryV1(_ServiceModel):
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
    """Adapt one validated host snapshot to the existing debugger contract."""
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
        source_identity += ":profile:default-tdm-map-preview@1"

    controlled_slot = next(
        row.global_slot
        for row in compiled.content.roster
        if row.team == "A" and row.team_local_slot <= compiled.content.team_a_size
    )

    def build_scenario() -> tuple[EnvConfig, EnvState]:
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
        ),
    )


class DevScenarioLoadAttemptV1(_ServiceModel):
    ok: bool
    summary: DevDebugLoadSummaryV1 | None = None
    problems: tuple[DevAuthoringProblemV1, ...] = ()


def _service_problem(
    code: str,
    message: str,
    *,
    field_path: str = "command",
) -> DevAuthoringProblemV1:
    return DevAuthoringProblemV1(
        severity="error",
        stable_code=code,
        message=message,
        field_path=field_path,
    )


def _validation_summary(draft: DevDraftPayload) -> DevValidationSummaryV1:
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
    )


class DevScenarioLoadService:
    """Discover and load exact authored assets without partial replacement."""

    def __init__(
        self,
        store: DevAssetStore,
        *,
        install_snapshot: Callable[[LoadedDevScenarioSnapshotV1], None] | None = None,
    ) -> None:
        self._store = store
        self._install_snapshot = install_snapshot
        self._current_snapshot: LoadedDevScenarioSnapshotV1 | None = None

    @property
    def current_snapshot(self) -> LoadedDevScenarioSnapshotV1 | None:
        return self._current_snapshot

    def _open_source(
        self,
        source: DevDebugAssetSourceV1,
    ) -> DevDraftPayload:
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
        if source.asset_kind == "scenario":
            if not isinstance(opened, DevScenarioDraftV1):
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
        """Reopen, compile, revalidate, then atomically replace current snapshot."""
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
                self._install_snapshot(snapshot)
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
        self._current_snapshot = snapshot
        return DevScenarioLoadAttemptV1(ok=True, summary=summary)

    def list_persisted(
        self,
        asset_kind: DevAssetKind = "scenario",
        *,
        include_invalid_drafts: bool = False,
    ) -> tuple[DevAssetSummaryV1, ...]:
        """Summarize one asset kind for authoring and Debug discovery."""
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
        """Return exact valid saved maps and scenarios for Combat."""
        return self.list_persisted("scenario") + self.list_persisted("map")


class DevClientAuthoringBinding:
    """Single callable live-only host authority for the authoring endpoint."""

    def __init__(
        self,
        store: DevAssetStore,
        *,
        scenario_loader: DevScenarioLoadService | None = None,
    ) -> None:
        self._store = store
        self._lock = Lock()
        self._scenario_loader = (
            DevScenarioLoadService(store)
            if scenario_loader is None
            else scenario_loader
        )

    @property
    def scenario_loader(self) -> DevScenarioLoadService:
        return self._scenario_loader

    def _load_persisted(
        self,
        source: DevPersistedSourceV1,
    ) -> DevDraftPayload:
        return self._store.load_draft(
            source.asset_kind,
            source.asset_id,
            revision=source.revision,
        )

    def _list_assets(
        self,
        requested: Literal["map", "scenario", "all"],
    ) -> tuple[DevAssetSummaryV1, ...]:
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

    def _new_scenario(self, command: DevNewScenarioCommandV1) -> DevScenarioDraftV1:
        if command.creation_mode == "blank":
            return new_scenario_draft(command.asset_id)
        if command.source is None:
            raise AssertionError("validated nonblank creation requires a source")
        source = self._load_persisted(command.source)
        if command.creation_mode == "copy_saved_map":
            if not isinstance(source, DevMapDraftV1):
                raise TypeError("copy_saved_map resolved a non-map source")
            return new_scenario_draft(command.asset_id, source_map=source)
        if not isinstance(source, DevScenarioDraftV1):
            raise TypeError("duplicate_saved_scenario resolved a non-scenario source")
        return duplicate_scenario_draft(source, asset_id=command.asset_id)

    def apply_command(
        self,
        request: DevAuthoringCommandRequestV1,
    ) -> DevAuthoringCommandResponseV1:
        """Serialize and apply one strictly parsed live authoring command."""
        with self._lock:
            return self._apply_command(request)

    def _apply_command(
        self,
        request: DevAuthoringCommandRequestV1,
    ) -> DevAuthoringCommandResponseV1:
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
