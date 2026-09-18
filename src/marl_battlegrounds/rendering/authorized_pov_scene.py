"""Build neutral scenes without crossing a recipient's information boundary.

build_no_shared_obs_authorized_scene_v1 uses exact recipient POV records and
a separately validated public mechanics catalog. The SharedObs builder
checks every supplied source's identity and epoch, then reads unit payloads
only from the recipient and its recorded admitted allies. It keeps body
provenance and the recipient-owned next-decision mask.

These host builders accept no Oracle scene, full episode context, event
batch, simulator state or renderer. Public class documentation is separate
from observed per-slot profile values. Opaque presentation keys are scoped
to a viewing session and recipient; they are display IDs, not credentials.
Strong boundary checks may rebuild records and perform JSON validation in
memory. No file I/O, JAX execution, hidden-source inference or mutation occurs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from json import dumps
from struct import pack, unpack
from typing import ClassVar, Literal, cast

from pydantic import ConfigDict, TypeAdapter

from marl_battlegrounds.evaluation.models import (
    AuraMechanicV1,
    ClassMechanicsV1,
    StaticMechanicsCatalogV1,
    StatusMechanicV1,
)
from marl_battlegrounds.evaluation.pov import (
    ActorPovActionMaskV1,
    ActorPovAdjacentTransitionSlice,
    ActorPovAdjacentTransitionSliceV1,
    ActorPovAdjacentTransitionSliceV2,
    ActorPovAxisMapping,
    ActorPovAxisMappingV1,
    ActorPovAxisMappingV2,
    ActorPovCurrentSlice,
    ActorPovCurrentSliceV1,
    ActorPovCurrentSliceV2,
    ActorPovFrame,
    ActorPovReplayContent,
    ActorPovReplayContentV1,
    ActorPovReplayContentV2,
    ActorPovSpawnLifecycleV1,
)
from marl_battlegrounds.evaluation.wire_shapes import (
    MAX_AGENT_SLOTS_V1,
    MAX_AGENTS_PER_TEAM_V1,
)
from marl_battlegrounds.rendering.authorized_presentation import (
    AUTHORIZED_PRESENTATION_SCHEMA_VERSION,
    AuthorizedAgentV1,
    AuthorizedAuraFieldV1,
    AuthorizedAuraModifierV1,
    AuthorizedBattlefieldSceneV1,
    AuthorizedClassAuraMechanicV1,
    AuthorizedClassDocumentationProfileV1,
    AuthorizedClassMechanicsV2,
    AuthorizedClassStatusMechanicV1,
    AuthorizedMapV1,
    AuthorizedObstacleV1,
    AuthorizedRespawnWaveV1,
    AuthorizedSpawnPadV1,
    AuthorizedSpawnShieldMechanicsAvailableV2,
    AuthorizedStatusV1,
    authorized_class_documentation_profile_v1,
)
from marl_battlegrounds.rendering.evaluation_adapter import (
    SHARED_OBS_SOURCE_MATERIAL_PROJECTION_SCHEMA_VERSION,
    SharedObsBaseSensorFrameV1,
    SharedObsBaseSensorFrameV2,
    SharedObsBaseSensorSceneV1,
    SharedObsSensorSourceAvailabilityV1,
    SharedObsSourceMaterialProjection,
    SharedObsSourceMaterialProjectionV1,
    SharedObsSourceMaterialProjectionV2,
    _shared_obs_base_sensor_scene,  # pyright: ignore[reportPrivateUsage]
)
from marl_battlegrounds.rendering.evaluation_wire_features import (
    DecodedAgentFeatureRowV1,
    decode_agent_feature_row,
)
from marl_battlegrounds.rendering.pov_scene import (
    ActorPovProjectionIndexV1,
    ActorPovSelfSceneV1,
    _build_actor_pov_battlefield_scene_v1,  # pyright: ignore[reportPrivateUsage]
    build_actor_pov_analyzer_projection_v1,
    build_actor_pov_projection_index_v1,
)
from marl_battlegrounds.rendering.scene import MapSceneV1, ObstacleSceneV1
from marl_battlegrounds.rendering.vocabulary import (
    status_sort_key,
    status_token_id_from_catalog_status_id,
)

type NoSharedObsPovSourceV1 = (
    ActorPovProjectionIndexV1
    | ActorPovReplayContent
    | ActorPovCurrentSlice
    | ActorPovAdjacentTransitionSlice
)

_STRICT_SHARED_WIRE_CONFIG = ConfigDict(
    allow_inf_nan=False,
    extra="forbid",
    strict=True,
)
_SHARED_SOURCE_MATERIAL_DISCLOSURE_V1 = (
    "SOURCE MATERIAL ONLY · NOT MATERIALIZED SHAREDOBS ACTOR INPUT"
)


def _require_text(value: str, *, name: str) -> None:
    """Require value to be a nonblank exact Python string.

    name labels ValueError for a wrong type or blank value. Return None without
    trimming or converting valid text.
    """
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a non-empty Python string.")


def _require_opaque_pov_key(value: str, *, name: str) -> None:
    """Check value has the canonical pov_ plus 64-lowercase-hex format.

    name labels ValueError. Return None on success; format validity does not
    prove that a key was issued by a particular authority session.
    """
    _require_text(value, name=name)
    digest = value.removeprefix("pov_")
    if (
        not value.startswith("pov_")
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{name} must be one canonical opaque POV key.")


def _validate_pov_scene_envelope(
    scene: AuthorizedBattlefieldSceneV1,
    *,
    recipient_public_agent_id: str,
    recipient_presentation_key: str,
) -> AuthorizedAgentV1:
    """Revalidate a neutral scene and join its one recipient self row.

    scene must be the exact authorized scene root. recipient_public_agent_id
    and recipient_presentation_key must identify exactly one self body; all
    body keys must be opaque POV keys and relations must match recipient/team
    identity. Return that AuthorizedAgentV1. Raise ValueError for bad roots or
    joins. JSON structural revalidation is host work, with no file I/O.
    """
    if type(scene) is not AuthorizedBattlefieldSceneV1:
        raise ValueError("scene must be the exact authorized neutral scene.")
    validated_scene = TypeAdapter(AuthorizedBattlefieldSceneV1).validate_json(
        dumps(
            asdict(scene),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    if validated_scene != scene:
        raise ValueError("authorized scene changes under structural revalidation.")
    recipient_rows = tuple(
        row for row in scene.agents if row.public_agent_id == recipient_public_agent_id
    )
    if len(recipient_rows) != 1:
        raise ValueError("recipient identity must join exactly one scene row.")
    recipient = recipient_rows[0]
    if (
        recipient.presentation_key != recipient_presentation_key
        or recipient.relation != "self"
    ):
        raise ValueError("recipient identity must join exactly one self row.")
    for agent in scene.agents:
        _require_opaque_pov_key(
            agent.presentation_key,
            name="agent presentation key",
        )
        expected_relation: Literal["self", "ally", "opponent"]
        if agent.public_agent_id == recipient_public_agent_id:
            expected_relation = "self"
        elif agent.team_id == recipient.team_id:
            expected_relation = "ally"
        else:
            expected_relation = "opponent"
        if agent.relation != expected_relation:
            raise ValueError(
                "Agent POV rows must use exact recipient-relative relations."
            )
    return recipient


def _catalog_or_exact_f32(recorded: float, catalog: float) -> bool:
    """Compare recorded with catalog or its exactly rounded float32 value.

    Return True for either exact equality, False otherwise or when float32
    conversion overflows. This allows recorded float32 roundoff without a
    broad tolerance that could admit changed mechanics. Inputs are host floats.
    """
    if recorded == catalog:
        return True
    try:
        catalog_as_f32 = unpack(">f", pack(">f", catalog))[0]
    except OverflowError:
        return False
    return recorded == catalog_as_f32


def _catalog_as_f32(catalog: float, *, name: str) -> float:
    """Round catalog to its exact IEEE float32 value and return a Python float.

    name labels ValueError if packing overflows. This host conversion does not
    allocate a JAX array or apply an approximate tolerance.
    """
    try:
        return unpack(">f", pack(">f", catalog))[0]
    except OverflowError as error:
        raise ValueError(f"{name} is outside the V1 float32 wire domain.") from error


def _require_catalog_float_join(
    recorded: float,
    catalog: float,
    *,
    name: str,
) -> None:
    """Require recorded to equal catalog or its exact float32 rounding.

    name labels ValueError on mismatch. Return None without replacing the
    recorded value or allowing arbitrary near-equality.
    """
    if not _catalog_or_exact_f32(recorded, catalog):
        raise ValueError(f"{name} does not join the public V1 mechanics catalog.")


def pov_presentation_key_v1(
    *,
    authority_session_id: str,
    recipient_public_agent_id: str,
    public_agent_id: str,
) -> str:
    """Create a stable opaque body key within one recipient authority session.

    Parameters
    ----------
    authority_session_id : str
        Nonblank host-owned identifier for the exact viewing authority session.
    recipient_public_agent_id : str
        Nonblank public ID of the recipient whose view owns the key.
    public_agent_id : str
        Nonblank public ID of the body being presented.

    Returns
    -------
    str
        pov_ followed by the lowercase SHA-256 digest of the scoped identities.
        The same inputs give the same key; changing scope changes its input.

    Raises
    ------
    ValueError
        Any input is not a nonblank Python string.

    Notes
    -----
    This deterministic display identity is not an access credential. It draws
    no randomness, stores no registry and writes no files.
    """
    _require_text(authority_session_id, name="authority_session_id")
    _require_text(
        recipient_public_agent_id,
        name="recipient_public_agent_id",
    )
    _require_text(public_agent_id, name="public_agent_id")
    payload = (
        "agent_pov\x00"
        f"{authority_session_id}\x00{recipient_public_agent_id}\x00{public_agent_id}"
    ).encode()
    return f"pov_{sha256(payload).hexdigest()}"


@dataclass(frozen=True, slots=True, kw_only=True)
class NoSharedObsAuthorizedScenePartsV1:
    """Bind a NoSharedObs scene to its recipient and exact decision mask.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    source_episode_id : str
        Nonblank recorded source episode ID.
    source_frame_index : int
        Nonnegative Python recorded frame index.
    source_recipient_frame_id : str
        Canonical episode:actor-pov:recipient:frame:index ID.
    source_simulator_step_count : int
        Nonnegative Python simulator step count for the selected frame.
    recipient_public_agent_id : str
        Nonblank public ID of the one recipient whose authority is represented.
    recipient_presentation_key : str
        Canonical pov_ key with 64 lowercase hexadecimal digest characters.
    scene : AuthorizedBattlefieldSceneV1
        Exact AuthorizedBattlefieldSceneV1 restricted to the recipient, with one
        matching self row.
    next_decision_action_mask : ActorPovActionMaskV1
        Exact ActorPovActionMaskV1 copied from this recipient at the selected decision.

    Raises
    ------
    ValueError
        Source identity/counters, canonical frame/key, recipient scene joins or
        exact mask validation fails.

    Notes
    -----
    Scene and mask are structurally revalidated. This object does not contain an
    incoming event batch or a full researcher snapshot.
    """

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_SHARED_WIRE_CONFIG
    """Class-level Pydantic settings: strict values, no extra fields and no
    nonfinite numbers.
    """

    source_episode_id: str
    """Nonblank recorded source episode ID."""
    source_frame_index: int
    """Nonnegative Python recorded frame index."""
    source_recipient_frame_id: str
    """Canonical episode:actor-pov:recipient:frame:index ID."""
    source_simulator_step_count: int
    """Nonnegative Python simulator step count for the selected frame."""
    recipient_public_agent_id: str
    """Nonblank public ID of the one recipient whose authority is represented."""
    recipient_presentation_key: str
    """Canonical pov_ key with 64 lowercase hexadecimal digest characters."""
    scene: AuthorizedBattlefieldSceneV1
    """Exact AuthorizedBattlefieldSceneV1 restricted to the recipient, with one
    matching self row.
    """
    next_decision_action_mask: ActorPovActionMaskV1
    """Exact ActorPovActionMaskV1 copied from this recipient at the selected
    decision.
    """

    def __post_init__(self) -> None:
        """Validate NoSharedObsAuthorizedScenePartsV1 at the presentation boundary.

        Raise ValueError if source identity/counters, canonical frame/key,
        recipient scene joins or exact mask validation fails.
        Return None without changing valid values.
        """
        _require_text(self.source_episode_id, name="source_episode_id")
        for name in ("source_frame_index", "source_simulator_step_count"):
            value = cast(int, getattr(self, name))
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative Python int.")
        _require_text(
            self.recipient_public_agent_id,
            name="recipient_public_agent_id",
        )
        expected_recipient_frame_id = (
            f"{self.source_episode_id}:actor-pov:"
            f"{self.recipient_public_agent_id}:frame:{self.source_frame_index}"
        )
        if self.source_recipient_frame_id != expected_recipient_frame_id:
            raise ValueError("NoSharedObs recipient frame ID is not canonical.")
        _require_opaque_pov_key(
            self.recipient_presentation_key,
            name="recipient_presentation_key",
        )
        _validate_pov_scene_envelope(
            self.scene,
            recipient_public_agent_id=self.recipient_public_agent_id,
            recipient_presentation_key=self.recipient_presentation_key,
        )
        if type(self.next_decision_action_mask) is not ActorPovActionMaskV1:
            raise ValueError("next_decision_action_mask must be the exact POV mask.")
        validated_mask = ActorPovActionMaskV1.model_validate(
            self.next_decision_action_mask.model_dump(mode="python")
        )
        if validated_mask != self.next_decision_action_mask:
            raise ValueError("next-decision mask changes under revalidation.")


@dataclass(frozen=True, slots=True, kw_only=True)
class SharedObsAuthorizedSensorSourceV1:
    """Identify one admitted sensor source in a recipient visual union.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    source_kind : Literal['recipient_base', 'shared_sensor_source']
        recipient_base for self input or shared_sensor_source for an admitted ally.
    source_presentation_key : str
        Canonical opaque key scoped to the same recipient authority.
    source_public_agent_id : str
        Nonblank public ID of the contributing sensor source.

    Raises
    ------
    ValueError
        Source kind, opaque key format or public ID is invalid.

    Notes
    -----
    This row records admission already decided from the recipient's availability
    matrix; it does not decide permission itself.
    """

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_SHARED_WIRE_CONFIG
    """Class-level Pydantic settings: strict values, no extra fields and no
    nonfinite numbers.
    """

    source_kind: Literal["recipient_base", "shared_sensor_source"]
    """recipient_base for self input or shared_sensor_source for an admitted ally."""
    source_presentation_key: str
    """Canonical opaque key scoped to the same recipient authority."""
    source_public_agent_id: str
    """Nonblank public ID of the contributing sensor source."""

    def __post_init__(self) -> None:
        """Validate SharedObsAuthorizedSensorSourceV1 at the presentation boundary.

        Raise ValueError if source kind, opaque key format or public ID is invalid.
        Return None without changing valid values.
        """
        if self.source_kind not in ("recipient_base", "shared_sensor_source"):
            raise ValueError("unknown SharedObs authorized source kind.")
        _require_opaque_pov_key(
            self.source_presentation_key,
            name="source_presentation_key",
        )
        _require_text(
            self.source_public_agent_id,
            name="source_public_agent_id",
        )


def _shared_source_sort_key(
    source: SharedObsAuthorizedSensorSourceV1,
) -> tuple[int, str]:
    """Order source with the recipient first, then by source public ID.

    Return (kind rank, public ID), where recipient_base has rank zero and
    shared_sensor_source rank one. The source record is already validated.
    """
    return (
        0 if source.source_kind == "recipient_base" else 1,
        source.source_public_agent_id,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class SharedObsAgentObservationProvenanceV1:
    """List the admitted sensors that supplied one deduplicated body.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    agent_presentation_key : str
        Canonical opaque key for the displayed agent.
    agent_public_agent_id : str
        Nonblank public ID of the displayed agent.
    observation_sources : tuple[SharedObsAuthorizedSensorSourceV1, ...]
        Nonempty tuple of exact source rows, recipient first then shared sources
        by public ID, with no repeated source IDs.

    Raises
    ------
    ValueError
        Agent identity, source record types, nonempty ordering or unique source
        IDs are invalid.
    """

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_SHARED_WIRE_CONFIG
    """Class-level Pydantic settings: strict values, no extra fields and no
    nonfinite numbers.
    """

    agent_presentation_key: str
    """Canonical opaque key for the displayed agent."""
    agent_public_agent_id: str
    """Nonblank public ID of the displayed agent."""
    observation_sources: tuple[SharedObsAuthorizedSensorSourceV1, ...]
    """Nonempty tuple of exact source rows, recipient first then shared sources by
    public ID, with no repeated source IDs.
    """

    def __post_init__(self) -> None:
        """Validate SharedObsAgentObservationProvenanceV1 at the presentation
        boundary.

        Raise ValueError if agent identity, source record types, nonempty
        ordering or unique source IDs are invalid.
        Return None without changing valid values.
        """
        _require_opaque_pov_key(
            self.agent_presentation_key,
            name="agent_presentation_key",
        )
        _require_text(
            self.agent_public_agent_id,
            name="agent_public_agent_id",
        )
        if type(self.observation_sources) is not tuple or not self.observation_sources:
            raise ValueError(
                "SharedObs agent provenance requires a non-empty source tuple."
            )
        if any(
            type(source) is not SharedObsAuthorizedSensorSourceV1
            for source in self.observation_sources
        ):
            raise ValueError(
                "SharedObs agent provenance requires exact authorized sources."
            )
        if self.observation_sources != tuple(
            sorted(self.observation_sources, key=_shared_source_sort_key)
        ):
            raise ValueError("SharedObs observation sources are not canonical.")
        source_ids = tuple(
            source.source_public_agent_id for source in self.observation_sources
        )
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("SharedObs observation provenance repeats a source.")


@dataclass(frozen=True, slots=True, kw_only=True)
class SharedObsAuthorizedScenePartsV1:
    """Bind a SharedObs visual union to one recipient's decision authority.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    source_episode_id : str
        Nonblank recorded source episode ID.
    source_frame_index : int
        Nonnegative Python recorded frame index.
    source_recipient_frame_id : str
        Canonical episode:shared-obs-visual-union:recipient:frame:index ID.
    source_simulator_step_count : int
        Nonnegative Python simulator step count for the selected frame.
    recipient_public_agent_id : str
        Nonblank public ID of the one recipient whose authority is represented.
    recipient_presentation_key : str
        Canonical pov_ key with 64 lowercase hexadecimal digest characters.
    scene : AuthorizedBattlefieldSceneV1
        Exact AuthorizedBattlefieldSceneV1 restricted to the recipient, with one
        matching self row.
    next_decision_action_mask : ActorPovActionMaskV1
        Exact ActorPovActionMaskV1 copied from this recipient at the selected decision.
    authorized_sensor_sources : tuple[SharedObsAuthorizedSensorSourceV1, ...]
        Nonempty ordered tuple containing exactly one recipient-base source plus
        unique admitted ally sources.
    agent_observation_provenance : tuple[SharedObsAgentObservationProvenanceV1, ...]
        Ordered tuple covering scene agents exactly and naming only authorized sources.

    Raises
    ------
    ValueError
        Frame/key/recipient joins, scene/mask revalidation, ordered unique
        source set, exact body coverage or provenance/source joins fail.

    Notes
    -----
    Agents are sorted by physical team ID then public ID. Every admitted source
    contributes its own self row. This is a visual union with the recipient's
    mask, not an export of composed policy tensors.
    """

    __pydantic_config__: ClassVar[ConfigDict] = _STRICT_SHARED_WIRE_CONFIG
    """Class-level Pydantic settings: strict values, no extra fields and no
    nonfinite numbers.
    """

    source_episode_id: str
    """Nonblank recorded source episode ID."""
    source_frame_index: int
    """Nonnegative Python recorded frame index."""
    source_recipient_frame_id: str
    """Canonical episode:shared-obs-visual-union:recipient:frame:index ID."""
    source_simulator_step_count: int
    """Nonnegative Python simulator step count for the selected frame."""
    recipient_public_agent_id: str
    """Nonblank public ID of the one recipient whose authority is represented."""
    recipient_presentation_key: str
    """Canonical pov_ key with 64 lowercase hexadecimal digest characters."""
    scene: AuthorizedBattlefieldSceneV1
    """Exact AuthorizedBattlefieldSceneV1 restricted to the recipient, with one
    matching self row.
    """
    next_decision_action_mask: ActorPovActionMaskV1
    """Exact ActorPovActionMaskV1 copied from this recipient at the selected
    decision.
    """
    authorized_sensor_sources: tuple[SharedObsAuthorizedSensorSourceV1, ...]
    """Nonempty ordered tuple containing exactly one recipient-base source plus
    unique admitted ally sources.
    """
    agent_observation_provenance: tuple[SharedObsAgentObservationProvenanceV1, ...]
    """Ordered tuple covering scene agents exactly and naming only authorized
    sources.
    """

    def __post_init__(self) -> None:
        """Validate SharedObsAuthorizedScenePartsV1 at the presentation boundary.

        Raise ValueError if frame/key/recipient joins, scene/mask revalidation,
        ordered unique source set, exact body coverage or provenance/source
        joins fail.
        Return None without changing valid values.
        """
        _require_text(self.source_episode_id, name="source_episode_id")
        for name in ("source_frame_index", "source_simulator_step_count"):
            value = cast(int, getattr(self, name))
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative Python int.")
        _require_text(
            self.recipient_public_agent_id,
            name="recipient_public_agent_id",
        )
        expected_recipient_frame_id = (
            f"{self.source_episode_id}:shared-obs-visual-union:"
            f"{self.recipient_public_agent_id}:frame:{self.source_frame_index}"
        )
        if self.source_recipient_frame_id != expected_recipient_frame_id:
            raise ValueError("SharedObs recipient frame ID is not canonical.")
        _require_opaque_pov_key(
            self.recipient_presentation_key,
            name="recipient_presentation_key",
        )
        _validate_pov_scene_envelope(
            self.scene,
            recipient_public_agent_id=self.recipient_public_agent_id,
            recipient_presentation_key=self.recipient_presentation_key,
        )
        if type(self.next_decision_action_mask) is not ActorPovActionMaskV1:
            raise ValueError("next_decision_action_mask must be the exact POV mask.")
        validated_mask = ActorPovActionMaskV1.model_validate(
            self.next_decision_action_mask.model_dump(mode="python")
        )
        if validated_mask != self.next_decision_action_mask:
            raise ValueError("next-decision mask changes under revalidation.")
        if type(self.authorized_sensor_sources) is not tuple or any(
            type(source) is not SharedObsAuthorizedSensorSourceV1
            for source in self.authorized_sensor_sources
        ):
            raise ValueError("authorized_sensor_sources must use exact source rows.")
        if (
            not self.authorized_sensor_sources
            or self.authorized_sensor_sources
            != tuple(
                sorted(self.authorized_sensor_sources, key=_shared_source_sort_key)
            )
        ):
            raise ValueError("SharedObs authorized sources are not canonical.")
        recipient_sources = tuple(
            source
            for source in self.authorized_sensor_sources
            if source.source_kind == "recipient_base"
        )
        if len(recipient_sources) != 1 or (
            recipient_sources[0].source_public_agent_id
            != self.recipient_public_agent_id
            or recipient_sources[0].source_presentation_key
            != self.recipient_presentation_key
        ):
            raise ValueError("SharedObs requires one exact recipient-base source.")
        source_public_ids = tuple(
            source.source_public_agent_id for source in self.authorized_sensor_sources
        )
        source_keys = tuple(
            source.source_presentation_key for source in self.authorized_sensor_sources
        )
        if len(source_public_ids) != len(set(source_public_ids)) or len(
            source_keys
        ) != len(set(source_keys)):
            raise ValueError("SharedObs authorized sources must be unique.")
        if type(self.agent_observation_provenance) is not tuple or any(
            type(row) is not SharedObsAgentObservationProvenanceV1
            for row in self.agent_observation_provenance
        ):
            raise ValueError("agent provenance must use exact SharedObs rows.")

        agent_order = tuple(
            (agent.team_id, agent.public_agent_id) for agent in self.scene.agents
        )
        if agent_order != tuple(sorted(agent_order)):
            raise ValueError("SharedObs scene agents are not canonical.")
        provenance_ids = tuple(
            row.agent_public_agent_id for row in self.agent_observation_provenance
        )
        if provenance_ids != tuple(
            agent.public_agent_id for agent in self.scene.agents
        ):
            raise ValueError("SharedObs provenance must exactly cover scene agents.")
        agent_by_public_id = {
            agent.public_agent_id: agent for agent in self.scene.agents
        }
        source_set = set(self.authorized_sensor_sources)
        for row in self.agent_observation_provenance:
            agent = agent_by_public_id[row.agent_public_agent_id]
            if agent.presentation_key != row.agent_presentation_key:
                raise ValueError("SharedObs provenance does not join its agent.")
            if any(source not in source_set for source in row.observation_sources):
                raise ValueError("SharedObs provenance names an unauthorized source.")
        for source in self.authorized_sensor_sources:
            source_agent = agent_by_public_id.get(source.source_public_agent_id)
            if source_agent is None or (
                source_agent.presentation_key != source.source_presentation_key
                or (
                    source.source_kind == "shared_sensor_source"
                    and source_agent.relation != "ally"
                )
            ):
                raise ValueError("SharedObs source must join its authorized body.")
            source_provenance = next(
                row
                for row in self.agent_observation_provenance
                if row.agent_public_agent_id == source.source_public_agent_id
            )
            if source not in source_provenance.observation_sources:
                raise ValueError("SharedObs source must contribute its own self row.")


@dataclass(frozen=True, slots=True, kw_only=True)
class _SourceSelectionV1:
    """Keep the selected POV frame and its declared recipient identity.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    frame : ActorPovFrame
        Exact supported ActorPovFrame selected from a validated source.
    public_agent_id : str
        Selected recipient's public ID.
    selected_team_local_slot : int
        Recipient roster index in 0..4.
    configured_team_id : int
        Recipient physical team ID 1 or 2.
    class_id : int
        Recipient class ID in 1..5.
    axis_mapping : ActorPovAxisMapping
        Version-matched public action and observation-row mapping.

    Notes
    -----
    Construction performs no validation. Callers supply already checked values.
    """

    frame: ActorPovFrame
    """Exact supported ActorPovFrame selected from a validated source."""
    public_agent_id: str
    """Selected recipient's public ID."""
    selected_team_local_slot: int
    """Recipient roster index in 0..4."""
    configured_team_id: int
    """Recipient physical team ID 1 or 2."""
    class_id: int
    """Recipient class ID in 1..5."""
    axis_mapping: ActorPovAxisMapping
    """Version-matched public action and observation-row mapping."""


@dataclass(frozen=True, slots=True, kw_only=True)
class _AuthorizedRowV1:
    """Keep one decoded body row after its authority and identity joins.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    public_agent_id : str
        Joined public body ID.
    relation : Literal['self', 'ally', 'opponent']
        self, ally or opponent relative to the fixed recipient.
    decoded : DecodedAgentFeatureRowV1
        Decoded visible/self feature row with physical team metadata.
    spawn_shield_remaining : int
        Nonnegative shield steps from the recipient-owned lifecycle row.

    Notes
    -----
    Construction performs no validation. Callers supply already checked values.
    """

    public_agent_id: str
    """Joined public body ID."""
    relation: Literal["self", "ally", "opponent"]
    """self, ally or opponent relative to the fixed recipient."""
    decoded: DecodedAgentFeatureRowV1
    """Decoded visible/self feature row with physical team metadata."""
    spawn_shield_remaining: int
    """Nonnegative shield steps from the recipient-owned lifecycle row."""


@dataclass(frozen=True, slots=True, kw_only=True)
class _SharedSourceHeaderV1:
    """Keep a source projection after metadata joins and before admission.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    projection : SharedObsSourceMaterialProjection
        Original source-material projection; unit payload may still be unadmitted.
    source_global_slot : int
        Source global slot in 0..9 established by recipient topology.
    source_public_agent_id : str
        Source public ID joined to the same topology and frame epoch.

    Notes
    -----
    Construction performs no validation. Callers supply already checked values.
    """

    projection: SharedObsSourceMaterialProjection
    """Original source-material projection; unit payload may still be unadmitted."""
    source_global_slot: int
    """Source global slot in 0..9 established by recipient topology."""
    source_public_agent_id: str
    """Source public ID joined to the same topology and frame epoch."""


@dataclass(frozen=True, slots=True, kw_only=True)
class _SharedContributionV1:
    """Pair one authorized decoded body with its contributing sensor.

    This is a frozen, slotted, keyword-only host record.

    Attributes
    ----------
    row : _AuthorizedRowV1
        Authorized body facts relative to the fixed recipient.
    source : SharedObsAuthorizedSensorSourceV1
        Admitted recipient-base or ally-source identity that supplied row.

    Notes
    -----
    Construction performs no validation. Callers supply already checked values.
    """

    row: _AuthorizedRowV1
    """Authorized body facts relative to the fixed recipient."""
    source: SharedObsAuthorizedSensorSourceV1
    """Admitted recipient-base or ally-source identity that supplied row."""


def _validated_catalog(
    catalog: StaticMechanicsCatalogV1,
) -> StaticMechanicsCatalogV1:
    """Revalidate the exact public mechanics catalog and its content digest.

    catalog must be StaticMechanicsCatalogV1, not a subclass. Return a validated
    copy from its Python payload. Raise TypeError for a wrong root and
    ValueError for invalid data/digest, including bypassed model construction.
    """
    if type(catalog) is not StaticMechanicsCatalogV1:
        raise TypeError("catalog must be the exact StaticMechanicsCatalogV1 root.")
    # Revalidation closes the `model_construct` escape hatch at this authority
    # boundary and rechecks the content digest as well as all V1 axes.
    return StaticMechanicsCatalogV1.model_validate(catalog.model_dump(mode="python"))


def _validate_shared_projection_declaration(
    projection: SharedObsSourceMaterialProjection,
) -> None:
    """Check a SharedObs projection's used version, identity and frame headers.

    projection must be an exact supported source-material root. Verify the
    source-only disclosure, version, canonical frame IDs and self/header joins.
    Raise TypeError for a wrong root or ValueError for contradictory metadata;
    return None. This deliberately does not validate all unit payloads before
    source availability has been checked.
    """
    if (
        type(projection) is not SharedObsSourceMaterialProjectionV1
        and type(projection) is not SharedObsSourceMaterialProjectionV2
    ):
        raise TypeError("SharedObs source material must use its exact projection root.")
    if (
        type(projection.schema_version) is not int
        or projection.schema_version
        != (2 if type(projection) is SharedObsSourceMaterialProjectionV2 else 1)
        or projection.disclosure_label != _SHARED_SOURCE_MATERIAL_DISCLOSURE_V1
        or projection.observation_materialization != "source_material_only"
        or projection.exact_actor_input_export_available is not False
    ):
        raise ValueError("SharedObs source declaration is not canonical.")
    frame = projection.base_sensor_frame
    scene = projection.base_sensor_scene
    if (
        (
            type(frame) is not SharedObsBaseSensorFrameV1
            and type(frame) is not SharedObsBaseSensorFrameV2
        )
        and type(frame) is not SharedObsBaseSensorFrameV2
    ) or type(scene) is not SharedObsBaseSensorSceneV1:
        raise ValueError("SharedObs source uses an invalid frame or scene root.")
    if (
        type(frame.schema_version) is not int
        or frame.schema_version != projection.schema_version
        or frame.observation_materialization != "source_material_only"
        or type(frame.episode_id) is not str
        or not frame.episode_id.strip()
        or type(frame.public_agent_id) is not str
        or not frame.public_agent_id.strip()
        or type(frame.frame_index) is not int
        or frame.frame_index < 0
        or type(frame.simulator_step_count) is not int
        or frame.simulator_step_count < 0
        or frame.source_material_frame_id
        != (
            f"{frame.episode_id}:shared-obs-source-material:"
            f"{frame.public_agent_id}:frame:{frame.frame_index}"
        )
        or frame.source_frame_id != f"{frame.episode_id}:frame:{frame.frame_index}"
    ):
        raise ValueError("SharedObs base-frame declaration is not canonical.")
    if (
        type(scene.schema_version) is not int
        or scene.schema_version != SHARED_OBS_SOURCE_MATERIAL_PROJECTION_SCHEMA_VERSION
        or scene.audience_badge != _SHARED_SOURCE_MATERIAL_DISCLOSURE_V1
        or scene.observation_materialization != "source_material_only"
        or scene.episode_id != frame.episode_id
        or scene.frame_index != frame.frame_index
        or scene.source_frame_id != frame.source_frame_id
        or scene.simulator_step_count != frame.simulator_step_count
        or type(scene.self_actor) is not ActorPovSelfSceneV1
        or scene.self_actor.public_agent_id != frame.public_agent_id
    ):
        raise ValueError("SharedObs base-scene declaration does not join its frame.")


def _shared_public_id_by_global_slot(
    projection: SharedObsSourceMaterialProjection,
) -> dict[int, str]:
    """Validate and return the declared ten-slot public identity mapping.

    projection supplies two five-row relation axes and their versioned public
    mapping. Revalidate that mapping, require a partition of slots and unique
    public IDs, then return a fresh dictionary. Raise ValueError on mismatch.
    No unit feature row is read.
    """
    if (
        type(projection.axis_mapping) is not ActorPovAxisMappingV1
        and type(projection.axis_mapping) is not ActorPovAxisMappingV2
    ):
        raise ValueError("SharedObs source axis must use the exact POV mapping.")
    validated_axis = type(projection.axis_mapping).model_validate(
        projection.axis_mapping.model_dump(mode="python")
    )
    if validated_axis != projection.axis_mapping:
        raise ValueError("SharedObs source axis changes under revalidation.")
    for values, name in (
        (
            projection.ally_observation_row_global_slot_by_id,
            "ally_observation_row_global_slot_by_id",
        ),
        (
            projection.enemy_observation_row_global_slot_by_id,
            "enemy_observation_row_global_slot_by_id",
        ),
    ):
        if type(values) is not tuple or len(values) != MAX_AGENTS_PER_TEAM_V1:
            raise ValueError(f"{name} must retain the five-row V1 axis.")
        if any(
            type(value) is not int or not 0 <= value < MAX_AGENT_SLOTS_V1
            for value in values
        ):
            raise ValueError(f"{name} contains an invalid global slot.")
    pairs = tuple(
        zip(
            projection.ally_observation_row_global_slot_by_id,
            projection.axis_mapping.ally_observation_row_public_agent_id_by_id,
            strict=True,
        )
    ) + tuple(
        zip(
            projection.enemy_observation_row_global_slot_by_id,
            projection.axis_mapping.enemy_observation_row_public_agent_id_by_id,
            strict=True,
        )
    )
    if {slot for slot, _public_id in pairs} != set(range(MAX_AGENT_SLOTS_V1)):
        raise ValueError("SharedObs source axes must partition global slots.")
    public_ids = tuple(public_id for _slot, public_id in pairs)
    if len(public_ids) != len(set(public_ids)):
        raise ValueError("SharedObs source axes must partition public identities.")
    return dict(pairs)


def _shared_source_header(
    projection: SharedObsSourceMaterialProjection,
    *,
    recipient_projection: SharedObsSourceMaterialProjection,
    recipient_public_id_by_slot: dict[int, str],
    recipient_topology_by_slot: dict[int, SharedObsSensorSourceAvailabilityV1],
) -> _SharedSourceHeaderV1:
    """Join one source's identity and frame epoch before admission filtering.

    projection is a candidate source. recipient_projection supplies the selected
    epoch/version; recipient_public_id_by_slot and recipient_topology_by_slot
    supply the validated ten-slot authority. Return a _SharedSourceHeaderV1
    when source/frame/self metadata agrees. Raise TypeError/ValueError through
    root/header checks. Do not decode an unavailable source's unit payload.
    """
    _validate_shared_projection_declaration(projection)
    for name in (
        "schema_version",
        "disclosure_label",
        "observation_materialization",
        "exact_actor_input_export_available",
    ):
        if getattr(projection, name) != getattr(recipient_projection, name):
            raise ValueError("SharedObs source declaration does not join recipient.")
    source_public_id_by_slot = _shared_public_id_by_global_slot(projection)
    if source_public_id_by_slot != recipient_public_id_by_slot:
        raise ValueError("SharedObs source remaps the canonical actor topology.")
    frame = projection.base_sensor_frame
    scene = projection.base_sensor_scene
    if type(frame) is not type(recipient_projection.base_sensor_frame) or type(
        scene
    ) is not type(recipient_projection.base_sensor_scene):
        raise ValueError("SharedObs source uses an invalid frame or scene root.")
    recipient_frame = recipient_projection.base_sensor_frame
    for name in (
        "episode_id",
        "frame_index",
        "source_frame_id",
        "simulator_step_count",
    ):
        if getattr(frame, name) != getattr(recipient_frame, name):
            raise ValueError("SharedObs source does not join the recipient epoch.")
    if (
        type(frame.public_agent_id) is not str
        or not frame.public_agent_id.strip()
        or frame.source_material_frame_id
        != (
            f"{frame.episode_id}:shared-obs-source-material:"
            f"{frame.public_agent_id}:frame:{frame.frame_index}"
        )
    ):
        raise ValueError("SharedObs source material identity is not canonical.")
    if (
        scene.episode_id != frame.episode_id
        or scene.frame_index != frame.frame_index
        or scene.source_frame_id != frame.source_frame_id
        or scene.simulator_step_count != frame.simulator_step_count
        or type(scene.self_actor) is not ActorPovSelfSceneV1
        or scene.self_actor.public_agent_id != frame.public_agent_id
    ):
        raise ValueError("SharedObs source scene does not join its frame header.")
    source_global_slot = scene.self_actor.global_slot
    if type(source_global_slot) is not int or source_global_slot not in range(
        MAX_AGENT_SLOTS_V1
    ):
        raise ValueError("SharedObs source self slot is outside the topology.")
    topology = recipient_topology_by_slot[source_global_slot]
    if (
        frame.public_agent_id != topology.sensor_source_public_agent_id
        or source_public_id_by_slot[source_global_slot] != frame.public_agent_id
        or scene.self_actor.team_local_slot != topology.sensor_source_team_local_slot
        or scene.self_actor.team_id != topology.sensor_source_configured_team_id
    ):
        raise ValueError("SharedObs source self identity does not join topology.")
    return _SharedSourceHeaderV1(
        projection=projection,
        source_global_slot=source_global_slot,
        source_public_agent_id=frame.public_agent_id,
    )


def _validate_shared_unit_axes(
    projection: SharedObsSourceMaterialProjection,
) -> None:
    """Check the containers needed to read one admitted source's unit rows.

    projection must have a tuple self row, two five-entry unit tuples and two
    five-entry bool visibility tuples. Raise ValueError for bad containers;
    return None. Row features are decoded later only for self or visible rows.
    """
    frame = projection.base_sensor_frame
    if type(frame.self_features) is not tuple:
        raise ValueError("SharedObs self features must use an exact tuple row.")
    for rows, visibility, name in (
        (
            frame.ally_unit_features,
            frame.ally_visibility_mask,
            "ally",
        ),
        (
            frame.enemy_unit_features,
            frame.enemy_visibility_mask,
            "enemy",
        ),
    ):
        if type(rows) is not tuple or len(rows) != MAX_AGENTS_PER_TEAM_V1:
            raise ValueError(f"SharedObs {name} rows must retain five entries.")
        if type(visibility) is not tuple or len(visibility) != MAX_AGENTS_PER_TEAM_V1:
            raise ValueError(f"SharedObs {name} visibility must retain five entries.")
        if any(type(value) is not bool for value in visibility):
            raise ValueError(
                f"SharedObs {name} visibility must contain exact booleans."
            )


def _validated_recipient_topology(
    projection: SharedObsSourceMaterialProjection,
) -> tuple[
    tuple[SharedObsSensorSourceAvailabilityV1, ...],
    dict[int, SharedObsSensorSourceAvailabilityV1],
    dict[int, str],
    SharedObsSensorSourceAvailabilityV1,
    MapSceneV1,
]:
    """Validate the recipient's source topology and its own base scene.

    projection supplies the ten ordered availability cells, public mappings
    and recipient base frame. Return (rows, rows by slot, public IDs by slot,
    self topology row, map). Require canonical team/slot/self joins, decode
    recipient self and rebuild its base scene. Raise TypeError/ValueError for
    invalid roots, identity or payload. This is the only availability matrix
    used to decide which other sources may contribute.
    """
    _validate_shared_projection_declaration(projection)
    public_id_by_slot = _shared_public_id_by_global_slot(projection)
    rows = projection.sensor_source_availability
    if (
        type(rows) is not tuple
        or len(rows) != MAX_AGENT_SLOTS_V1
        or any(type(row) is not SharedObsSensorSourceAvailabilityV1 for row in rows)
    ):
        raise ValueError("SharedObs recipient topology must retain ten exact rows.")
    if tuple(row.sensor_source_global_slot for row in rows) != tuple(
        range(MAX_AGENT_SLOTS_V1)
    ):
        raise ValueError("SharedObs recipient topology must retain source ordering.")
    for row in rows:
        row.__post_init__()
    self_rows = tuple(
        row
        for row in rows
        if row.relation_to_recipient == "self"
        and row.sensor_source_public_agent_id
        == projection.base_sensor_frame.public_agent_id
    )
    if len(self_rows) != 1:
        raise ValueError("SharedObs recipient topology requires one self row.")
    self_row = self_rows[0]
    recipient_team_id = self_row.sensor_source_configured_team_id
    axis_keys: list[tuple[str, int]] = []
    for row in rows:
        team_local_slot = row.sensor_source_global_slot % MAX_AGENTS_PER_TEAM_V1
        absolute_team_id = (
            1 if row.sensor_source_global_slot < MAX_AGENTS_PER_TEAM_V1 else 2
        )
        expected_configured_team = (
            absolute_team_id if row.sensor_source_configured_active else 0
        )
        if (
            row.sensor_source_team_local_slot != team_local_slot
            or row.sensor_source_configured_team_id != expected_configured_team
        ):
            raise ValueError("SharedObs recipient source topology is contradictory.")
        if not row.sensor_source_configured_active:
            expected_relation = "inactive"
        elif row.sensor_source_global_slot == self_row.sensor_source_global_slot:
            expected_relation = "self"
        elif absolute_team_id == recipient_team_id:
            expected_relation = "ally"
        else:
            expected_relation = "opponent"
        expected_axis = "ally" if absolute_team_id == recipient_team_id else "enemy"
        global_slots = (
            projection.ally_observation_row_global_slot_by_id
            if expected_axis == "ally"
            else projection.enemy_observation_row_global_slot_by_id
        )
        if (
            row.relation_to_recipient != expected_relation
            or row.base_sensor_relation_axis != expected_axis
            or global_slots[row.base_sensor_observation_row]
            != row.sensor_source_global_slot
            or public_id_by_slot[row.sensor_source_global_slot]
            != row.sensor_source_public_agent_id
        ):
            raise ValueError("SharedObs recipient relation topology is contradictory.")
        axis_keys.append(
            (row.base_sensor_relation_axis, row.base_sensor_observation_row)
        )
    if len(set(axis_keys)) != MAX_AGENT_SLOTS_V1:
        raise ValueError("SharedObs recipient rows must partition relation axes.")
    scene_self = projection.base_sensor_scene.self_actor
    if (
        scene_self.global_slot != self_row.sensor_source_global_slot
        or scene_self.public_agent_id != self_row.sensor_source_public_agent_id
        or scene_self.team_local_slot != self_row.sensor_source_team_local_slot
        or scene_self.team_id != self_row.sensor_source_configured_team_id
    ):
        raise ValueError("SharedObs recipient scene self does not join topology.")
    _validate_shared_unit_axes(projection)
    self_decoded = decode_agent_feature_row(
        projection.base_sensor_frame.self_features,
        schema_version=projection.base_sensor_frame.schema_version,
        team_id=self_row.sensor_source_configured_team_id,
        is_enemy=False,
    )
    if (
        not self_decoded.configured_active
        or self_decoded.team_id != self_row.sensor_source_configured_team_id
    ):
        raise ValueError("SharedObs recipient self row does not join topology.")
    expected_scene = _shared_obs_base_sensor_scene(
        projection.base_sensor_frame,
        selected_global_slot=self_row.sensor_source_global_slot,
        selected_team_local_slot=self_row.sensor_source_team_local_slot,
        configured_team_id=self_row.sensor_source_configured_team_id,
        class_id=self_decoded.class_id,
        axis_mapping=projection.axis_mapping,
    )
    if expected_scene != projection.base_sensor_scene:
        raise ValueError("SharedObs recipient scene must derive from its used frame.")
    return (
        rows,
        {row.sensor_source_global_slot: row for row in rows},
        public_id_by_slot,
        self_row,
        expected_scene.map,
    )


def _validated_source(source: NoSharedObsPovSourceV1) -> NoSharedObsPovSourceV1:
    """Revalidate an exact NoSharedObs source before reading selected rows.

    source may be a current/adjacent slice, replay content or projection index
    using supported versions. Return a structurally revalidated slice or a new
    validated replay index. Even an existing index is revalidated at this
    stronger authority boundary. Raise TypeError for an unsupported root and
    ValueError for invalid content/digests. No file is read.
    """
    if (
        type(source) is ActorPovAdjacentTransitionSliceV1
        or type(source) is ActorPovAdjacentTransitionSliceV2
    ):
        return type(source).model_validate(source.model_dump(mode="python"))
    if type(source) is ActorPovCurrentSliceV1 or type(source) is ActorPovCurrentSliceV2:
        return type(source).model_validate(source.model_dump(mode="python"))
    if (
        type(source) is ActorPovReplayContentV1
        or type(source) is ActorPovReplayContentV2
    ):
        validated = type(source).model_validate(source.model_dump(mode="python"))
        # The index constructor additionally checks the declared model tree and
        # content digest, then remains reusable by interactive callers.
        return build_actor_pov_projection_index_v1(validated)
    if type(source) is ActorPovProjectionIndexV1:
        validated_content = type(source.content).model_validate(
            source.content.model_dump(mode="python")
        )
        return build_actor_pov_projection_index_v1(validated_content)
    raise TypeError(
        "source must be an exact POV projection index, replay content, or "
        "live current/adjacent slice."
    )


def _select_source(
    source: NoSharedObsPovSourceV1,
    *,
    frame_index: int | None,
) -> _SourceSelectionV1:
    """Select one frame and identity from an already validated POV source.

    source may be a replay/index or current/adjacent slice. frame_index is
    required and range-checked for replay/index, optional but equal to the
    current frame for a current slice, and required to match one exact endpoint
    for an adjacent slice. Return _SourceSelectionV1. Bad replay indices raise
    IndexError, mismatched live indices ValueError, and unsupported roots
    TypeError. No preceding history is invented for an adjacent start frame.
    """
    if type(source) is ActorPovProjectionIndexV1:
        content = source.content
        if type(frame_index) is not int or not 0 <= frame_index < len(content.frames):
            raise IndexError("frame_index is outside the captured POV prefix.")
        frame = content.frames[frame_index]
        return _SourceSelectionV1(
            frame=frame,
            public_agent_id=content.public_agent_id,
            selected_team_local_slot=content.selected_team_local_slot,
            configured_team_id=content.configured_team_id,
            class_id=content.class_id,
            axis_mapping=content.axis_mapping,
        )
    if (
        type(source) is ActorPovReplayContentV1
        or type(source) is ActorPovReplayContentV2
    ):
        if type(frame_index) is not int or not 0 <= frame_index < len(source.frames):
            raise IndexError("frame_index is outside the captured POV prefix.")
        return _SourceSelectionV1(
            frame=source.frames[frame_index],
            public_agent_id=source.public_agent_id,
            selected_team_local_slot=source.selected_team_local_slot,
            configured_team_id=source.configured_team_id,
            class_id=source.class_id,
            axis_mapping=source.axis_mapping,
        )
    if type(source) is ActorPovCurrentSliceV1 or type(source) is ActorPovCurrentSliceV2:
        if frame_index is not None and frame_index != source.frame.frame_index:
            raise ValueError(
                "a live current slice accepts only its own canonical frame index."
            )
        return _SourceSelectionV1(
            frame=source.frame,
            public_agent_id=source.public_agent_id,
            selected_team_local_slot=source.selected_team_local_slot,
            configured_team_id=source.configured_team_id,
            class_id=source.class_id,
            axis_mapping=source.axis_mapping,
        )
    if (
        type(source) is ActorPovAdjacentTransitionSliceV1
        or type(source) is ActorPovAdjacentTransitionSliceV2
    ):
        if type(frame_index) is not int or frame_index not in (
            source.start_frame.frame_index,
            source.successor_frame.frame_index,
        ):
            raise ValueError(
                "a live adjacent slice accepts only its exact endpoint indexes."
            )
        frame = (
            source.start_frame
            if frame_index == source.start_frame.frame_index
            else source.successor_frame
        )
        return _SourceSelectionV1(
            frame=frame,
            public_agent_id=source.public_agent_id,
            selected_team_local_slot=source.selected_team_local_slot,
            configured_team_id=source.configured_team_id,
            class_id=source.class_id,
            axis_mapping=source.axis_mapping,
        )
    raise TypeError(
        "source must be an exact POV projection index, replay content, or "
        "live current/adjacent slice."
    )


def _authorized_map(source_map: MapSceneV1) -> AuthorizedMapV1:
    # The old scalar POV projection remains the compatibility oracle for map
    # decoding.  Exact concrete row checks occur in its constructors.
    """Copy an exact scalar POV map into the neutral authorized map schema.

    source_map must be MapSceneV1 with exact ObstacleSceneV1 rows. Return a new
    AuthorizedMapV1 with unchanged world dimensions and obstacle geometry.
    Raise TypeError for wrong roots; nested constructors reject invalid values.
    """
    if type(source_map) is not MapSceneV1 or any(
        type(row) is not ObstacleSceneV1 for row in source_map.obstacles
    ):
        raise TypeError("source_map must contain exact scalar POV map rows.")
    return AuthorizedMapV1(
        width=source_map.width,
        height=source_map.height,
        obstacles=tuple(
            AuthorizedObstacleV1(
                obstacle_id=row.obstacle_id,
                kind=row.kind,
                center=row.center,
                radius=row.radius,
                width=row.width,
                height=row.height,
                theta=row.theta,
            )
            for row in source_map.obstacles
        ),
    )


def _validate_row_against_class_catalog(
    row: DecodedAgentFeatureRowV1,
    class_catalog: ClassMechanicsV1,
    catalog: StaticMechanicsCatalogV1,
) -> None:
    """Check active identity and fixed class capabilities without erasing profiles.

    row is decoded authorized data; class_catalog and catalog are its validated
    public class and mechanic authorities. Check raw ability outputs, cooldown
    capability, status/aura ownership and strengths, and countdown upper bounds.
    Resolved per-slot radius, speed, health, range and recovery profile values
    need not equal generic class documentation. Return None or raise ValueError
    for an invalid identity or changed fixed capability.
    """
    if (
        not row.configured_active
        or not 1 <= row.class_id <= 5
        or row.class_id != class_catalog.class_id
        or row.team_id not in (1, 2)
        or row.radius <= 0.0
        or row.maximum_health <= 0.0
    ):
        raise ValueError("authorized POV agent row has invalid active identity facts.")
    # Radius, speed, range, health, and OOC values are resolved per-slot profile
    # facts on the observation wire.  They need not equal the generic public
    # class documentation.  Only fixed class-authored capability columns join
    # the static catalog here.
    for name, recorded, expected in (
        ("basic raw damage", row.basic_raw_damage, class_catalog.basic_raw_damage),
        (
            "basic raw healing",
            row.basic_raw_healing,
            class_catalog.basic_raw_healing,
        ),
        (
            "ultimate raw damage",
            row.ultimate_raw_damage,
            class_catalog.ultimate_raw_damage,
        ),
        (
            "ultimate raw healing",
            row.ultimate_raw_healing,
            class_catalog.ultimate_raw_healing,
        ),
    ):
        _require_catalog_float_join(recorded, expected, name=name)
    if row.ultimate_cooldown_steps != class_catalog.ultimate_cooldown_steps:
        raise ValueError("ultimate cooldown capability changed catalog semantics.")
    if row.ultimate_cooldown_remaining > row.ultimate_cooldown_steps:
        raise ValueError("ultimate cooldown remaining exceeds its configured duration.")
    if row.steps_until_out_of_combat > row.out_of_combat_delay_steps:
        raise ValueError("out-of-combat countdown exceeds its configured delay.")

    status_by_channel = {
        status.status_channel_id: status for status in catalog.status_channels
    }
    for channel, status in status_by_channel.items():
        row_duration = row.status_capability_duration_by_channel[channel]
        row_magnitude = row.status_capability_magnitude_by_channel[channel]
        if status.source_class_id == row.class_id:
            if row_duration != status.duration_steps:
                raise ValueError(
                    "status duration capability changed public catalog semantics."
                )
            if row_magnitude is not None:
                if status.magnitude is None:
                    raise ValueError("status magnitude capability is unexpectedly set.")
                _require_catalog_float_join(
                    row_magnitude,
                    status.magnitude,
                    name=f"status channel {channel} magnitude capability",
                )
        elif row_duration != 0 or (row_magnitude is not None and row_magnitude != 0.0):
            raise ValueError("row advertises another class's status capability.")

    aura_by_id = {aura.aura_id: aura for aura in catalog.aura_mechanics}
    for aura_id, radius, multiplier in (
        (
            "mage_damage_amplification",
            row.mage_aura_radius,
            row.mage_aura_per_emitter_multiplier,
        ),
        (
            "warrior_damage_mitigation",
            row.warrior_aura_radius,
            row.warrior_aura_per_emitter_multiplier,
        ),
    ):
        aura = aura_by_id[aura_id]
        if aura.emitter_class_id == row.class_id:
            _require_catalog_float_join(
                radius,
                aura.radius,
                name=f"{aura_id} radius capability",
            )
            _require_catalog_float_join(
                multiplier,
                aura.per_emitter_multiplier,
                name=f"{aura_id} multiplier capability",
            )
        elif radius != 0.0 or multiplier != 0.0:
            raise ValueError("row advertises another class's aura capability.")


def _status_mechanic(
    row: StatusMechanicV1,
) -> AuthorizedClassStatusMechanicV1:
    """Copy one public status catalog row into a neutral class-card record.

    row is a validated StatusMechanicV1. Return AuthorizedClassStatusMechanicV1
    with unchanged channel, timing, strength and damage-break declarations.
    """
    return AuthorizedClassStatusMechanicV1(
        status_channel=row.status_channel_id,
        status_id=row.status_id,
        family=row.family,
        source_action_component=row.source_action_component,
        duration_steps=row.duration_steps,
        magnitude_kind=row.magnitude_kind,
        magnitude=row.magnitude,
        breaks_on_positive_damage=row.breaks_on_positive_damage,
    )


def _catalog_aura_mechanic(
    row: AuraMechanicV1,
) -> AuthorizedClassAuraMechanicV1:
    """Copy one catalog aura into a neutral class-card mechanic.

    row is a validated AuraMechanicV1. Return its unchanged radius, per-emitter
    strength and stacking/clamp declarations as AuthorizedClassAuraMechanicV1.
    """
    return AuthorizedClassAuraMechanicV1(
        aura_id=row.aura_id,
        radius=row.radius,
        per_emitter_multiplier=row.per_emitter_multiplier,
        stacking_rule=row.stacking_rule,
        clamp_kind=row.clamp_kind,
        clamp_value=row.clamp_value,
    )


def _row_aura_mechanic(
    row: AuraMechanicV1,
    decoded: DecodedAgentFeatureRowV1,
) -> AuthorizedClassAuraMechanicV1:
    """Combine a declared aura rule with its authorized wire capability values.

    row supplies aura ID/stacking/clamp semantics; decoded supplies that class's
    wire radius and multiplier. Return an AuthorizedClassAuraMechanicV1.
    The caller has already joined wire capabilities to the public catalog.
    """
    if row.aura_id == "mage_damage_amplification":
        radius = decoded.mage_aura_radius
        multiplier = decoded.mage_aura_per_emitter_multiplier
    else:
        radius = decoded.warrior_aura_radius
        multiplier = decoded.warrior_aura_per_emitter_multiplier
    return AuthorizedClassAuraMechanicV1(
        aura_id=row.aura_id,
        radius=radius,
        per_emitter_multiplier=multiplier,
        stacking_rule=row.stacking_rule,
        clamp_kind=row.clamp_kind,
        clamp_value=row.clamp_value,
    )


def _class_mechanics(
    decoded: DecodedAgentFeatureRowV1,
    catalog: StaticMechanicsCatalogV1,
    *,
    documentation_profile: AuthorizedClassDocumentationProfileV1,
) -> AuthorizedClassMechanicsV2:
    """Build a generic public class card after checking an authorized row.

    decoded supplies a class in 1..5; catalog supplies class/status/aura facts;
    documentation_profile supplies the public wording version. Return an
    AuthorizedClassMechanicsV2 card with catalog values, keeping per-agent
    resolved profile values on body records. Raise ValueError for invalid
    class IDs or mismatched fixed wire capabilities.
    """
    if not 1 <= decoded.class_id <= 5:
        raise ValueError("authorized POV class ID must be in the V1 range 1..5.")
    class_catalog = catalog.class_mechanics[decoded.class_id]
    _validate_row_against_class_catalog(decoded, class_catalog, catalog)
    return AuthorizedClassMechanicsV2(
        class_id=decoded.class_id,
        class_name=class_catalog.class_name,
        maximum_health=class_catalog.maximum_health,
        body_radius=class_catalog.body_radius,
        base_movement_speed=class_catalog.base_movement_speed,
        observation_radius=class_catalog.observation_radius,
        basic_target_mode=class_catalog.basic_target_mode,
        basic_interaction_radius=class_catalog.basic_interaction_radius,
        basic_raw_damage=class_catalog.basic_raw_damage,
        basic_raw_healing=class_catalog.basic_raw_healing,
        ultimate_target_mode=class_catalog.ultimate_target_mode,
        ultimate_interaction_radius=class_catalog.ultimate_interaction_radius,
        ultimate_cooldown_steps=class_catalog.ultimate_cooldown_steps,
        ultimate_raw_damage=class_catalog.ultimate_raw_damage,
        ultimate_raw_healing=class_catalog.ultimate_raw_healing,
        out_of_combat_delay_steps=class_catalog.out_of_combat_delay_steps,
        out_of_combat_health_regeneration_fraction_per_step=(
            class_catalog.out_of_combat_health_regeneration_fraction_per_step
        ),
        status_mechanics=tuple(
            _status_mechanic(status)
            for status in catalog.status_channels
            if status.source_class_id == decoded.class_id
        ),
        aura_mechanics=tuple(
            _catalog_aura_mechanic(aura)
            for aura in catalog.aura_mechanics
            if aura.emitter_class_id == decoded.class_id
        ),
        mechanics_version=2,
        documentation_profile=documentation_profile,
    )


def _active_statuses(
    decoded: DecodedAgentFeatureRowV1,
    catalog: StaticMechanicsCatalogV1,
) -> tuple[AuthorizedStatusV1, ...]:
    """Build active status labels from authorized durations and public mechanics.

    decoded supplies nine remaining durations and available strength columns;
    catalog supplies channel meaning and missing-column strength. Return
    positive-duration AuthorizedStatusV1 rows in display order, with empty
    direct_sources. Recorded strengths must exactly join catalog/f32 values
    or raise ValueError. A class label is never used to guess a source agent.
    """
    statuses: list[AuthorizedStatusV1] = []
    for mechanic in catalog.status_channels:
        remaining = decoded.status_remaining_duration_by_channel[
            mechanic.status_channel_id
        ]
        if remaining == 0:
            continue
        active_magnitude = decoded.status_active_magnitude_by_channel[
            mechanic.status_channel_id
        ]
        if active_magnitude is None:
            magnitude = mechanic.magnitude
        else:
            if mechanic.magnitude is None:
                raise ValueError("stun status unexpectedly carries a magnitude.")
            _require_catalog_float_join(
                active_magnitude,
                mechanic.magnitude,
                name=f"status channel {mechanic.status_channel_id} active magnitude",
            )
            magnitude = active_magnitude
        statuses.append(
            AuthorizedStatusV1(
                status_channel=mechanic.status_channel_id,
                status_id=mechanic.status_id,
                family=mechanic.family,
                configured_duration_steps=mechanic.duration_steps,
                remaining_duration=remaining,
                source_class_id=mechanic.source_class_id,
                source_class_name=(catalog.class_name_by_id[mechanic.source_class_id]),
                source_action_component=mechanic.source_action_component,
                magnitude_kind=mechanic.magnitude_kind,
                magnitude=magnitude,
                breaks_on_positive_damage=mechanic.breaks_on_positive_damage,
                direct_sources=(),
            )
        )
    return tuple(
        sorted(
            statuses,
            key=lambda row: status_sort_key(
                status_token_id_from_catalog_status_id(row.status_id)
            ),
        )
    )


def _aura_modifiers(
    decoded: DecodedAgentFeatureRowV1,
    catalog: StaticMechanicsCatalogV1,
) -> tuple[AuthorizedAuraModifierV1, ...]:
    """Return only non-neutral authorized aura multipliers after bounds checks.

    decoded supplies current recipient multipliers; catalog supplies the
    float32-rounded Mage ceiling and Warrior floor. Return an ordered tuple
    for values different from 1.0. Raise ValueError when a multiplier exceeds
    its declared range. No hidden emitter list is inferred.
    """
    aura_by_id = {row.aura_id: row for row in catalog.aura_mechanics}
    mage_clamp = _catalog_as_f32(
        aura_by_id["mage_damage_amplification"].clamp_value,
        name="Mage aura clamp",
    )
    warrior_clamp = _catalog_as_f32(
        aura_by_id["warrior_damage_mitigation"].clamp_value,
        name="Warrior aura clamp",
    )
    if not 1.0 <= decoded.mage_aura_damage_multiplier <= mage_clamp:
        raise ValueError("Mage aggregate aura multiplier exceeds catalog bounds.")
    if not warrior_clamp <= decoded.warrior_aura_damage_multiplier <= 1.0:
        raise ValueError("Warrior aggregate aura multiplier exceeds catalog bounds.")
    rows: list[AuthorizedAuraModifierV1] = []
    if decoded.mage_aura_damage_multiplier != 1.0:
        rows.append(
            AuthorizedAuraModifierV1(
                aura_id="mage_damage_amplification",
                multiplier=decoded.mage_aura_damage_multiplier,
            )
        )
    if decoded.warrior_aura_damage_multiplier != 1.0:
        rows.append(
            AuthorizedAuraModifierV1(
                aura_id="warrior_damage_mitigation",
                multiplier=decoded.warrior_aura_damage_multiplier,
            )
        )
    return tuple(rows)


def _agent(
    row: _AuthorizedRowV1,
    *,
    catalog: StaticMechanicsCatalogV1,
    authority_session_id: str,
    recipient_public_agent_id: str,
) -> AuthorizedAgentV1:
    """Build one neutral agent from an already authorized decoded row.

    row owns public identity, recipient relation and shield counter. catalog
    supplies class wording and status/aura meaning. authority_session_id and
    recipient_public_agent_id scope the opaque key. Return AuthorizedAgentV1
    with observed profile/dynamic values; direct status sources remain empty.
    Constructors reject invalid values. No hidden facts are added.
    """
    decoded = row.decoded
    class_catalog = catalog.class_mechanics[decoded.class_id]
    return AuthorizedAgentV1(
        presentation_key=pov_presentation_key_v1(
            authority_session_id=authority_session_id,
            recipient_public_agent_id=recipient_public_agent_id,
            public_agent_id=row.public_agent_id,
        ),
        public_agent_id=row.public_agent_id,
        relation=row.relation,
        team_id=decoded.team_id,
        class_id=decoded.class_id,
        class_name=class_catalog.class_name,
        position=decoded.position,
        radius=decoded.radius,
        life_state="alive" if decoded.alive else "corpse",
        current_health=decoded.current_health,
        maximum_health=decoded.maximum_health,
        base_movement_speed=decoded.base_movement_speed,
        effective_movement_speed=decoded.effective_movement_speed,
        observation_radius=decoded.observation_radius,
        basic_interaction_radius=decoded.basic_interaction_radius,
        ultimate_interaction_radius=decoded.ultimate_interaction_radius,
        ultimate_cooldown_remaining=decoded.ultimate_cooldown_remaining,
        spawn_shield_remaining=row.spawn_shield_remaining,
        steps_until_out_of_combat=decoded.steps_until_out_of_combat,
        out_of_combat_delay_steps=decoded.out_of_combat_delay_steps,
        out_of_combat_health_regeneration_fraction_per_step=(
            decoded.out_of_combat_health_regeneration_fraction_per_step
        ),
        statuses=_active_statuses(decoded, catalog),
        aura_modifiers=_aura_modifiers(decoded, catalog),
    )


def _absolute_team_id(*, recipient_team_id: int, actor_relative_index: int) -> int:
    """Translate an own/opponent team axis to physical team ID.

    recipient_team_id is 1/2 and actor_relative_index is 0/1. Return recipient
    team for zero, opposing team for one; raise ValueError outside these axes.
    """
    if recipient_team_id not in (1, 2) or actor_relative_index not in (0, 1):
        raise ValueError("POV lifecycle team identity is outside the V1 axes.")
    return recipient_team_id if actor_relative_index == 0 else 3 - recipient_team_id


def _shared_relation(
    *,
    public_agent_id: str,
    team_id: int,
    recipient_public_agent_id: str,
    recipient_team_id: int,
) -> Literal["self", "ally", "opponent"]:
    """Label a body relative to the fixed SharedObs recipient.

    public_agent_id equal to recipient_public_agent_id yields self. Otherwise,
    team_id equal to recipient_team_id yields ally, and all others opponent.
    Return the label; callers already validated IDs and team values.
    """
    if public_agent_id == recipient_public_agent_id:
        return "self"
    return "ally" if team_id == recipient_team_id else "opponent"


def _shared_authorized_row(
    *,
    public_agent_id: str,
    decoded: DecodedAgentFeatureRowV1,
    topology: SharedObsSensorSourceAvailabilityV1,
    recipient_public_agent_id: str,
    recipient_team_id: int,
    lifecycle: ActorPovSpawnLifecycleV1,
    relation_row_must_be_alive: bool,
) -> _AuthorizedRowV1:
    """Join a decoded visible/self body to the recipient's topology and lifecycle.

    public_agent_id/decoded identify the candidate; topology supplies canonical
    membership/team/local-slot identity. recipient_public_agent_id and
    recipient_team_id fix relation. lifecycle is the recipient-owned record
    used for life/shield truth. relation_row_must_be_alive=True rejects corpses;
    False allows a source's own corpse row. Return _AuthorizedRowV1 or raise
    ValueError for identity, active/life or visibility contradictions.
    """
    if (
        not topology.sensor_source_configured_active
        or not decoded.configured_active
        or decoded.team_id != topology.sensor_source_configured_team_id
        or public_agent_id != topology.sensor_source_public_agent_id
    ):
        raise ValueError("SharedObs unit row conflicts with recipient topology.")
    actor_relative_team_index = 0 if decoded.team_id == recipient_team_id else 1
    team_local_slot = topology.sensor_source_team_local_slot
    lifecycle_active = lifecycle.active_mask_by_team[actor_relative_team_index][
        team_local_slot
    ]
    lifecycle_alive = lifecycle.alive_mask_by_team[actor_relative_team_index][
        team_local_slot
    ]
    if not lifecycle_active or decoded.alive != lifecycle_alive:
        raise ValueError("SharedObs unit row conflicts with recipient lifecycle.")
    if relation_row_must_be_alive and not decoded.alive:
        raise ValueError("SharedObs visible relation rows must be alive.")
    return _AuthorizedRowV1(
        public_agent_id=public_agent_id,
        relation=_shared_relation(
            public_agent_id=public_agent_id,
            team_id=decoded.team_id,
            recipient_public_agent_id=recipient_public_agent_id,
            recipient_team_id=recipient_team_id,
        ),
        decoded=decoded,
        spawn_shield_remaining=(
            lifecycle.spawn_shield_actual_durations_by_team[actor_relative_team_index][
                team_local_slot
            ]
        ),
    )


def _shared_projection_contributions(
    projection: SharedObsSourceMaterialProjection,
    *,
    source: SharedObsAuthorizedSensorSourceV1,
    source_global_slot: int,
    topology_by_global_slot: dict[int, SharedObsSensorSourceAvailabilityV1],
    recipient_public_agent_id: str,
    recipient_team_id: int,
    recipient_lifecycle: ActorPovSpawnLifecycleV1,
) -> tuple[_SharedContributionV1, ...]:
    """Decode only self and visible rows from one already admitted source.

    projection supplies source material; source/source_global_slot identify
    its admission. topology_by_global_slot, recipient_public_agent_id,
    recipient_team_id and recipient_lifecycle supply fixed recipient authority.
    Return body/source pairs, skipping hidden rows and a matching self diagonal.
    Raise ValueError for conflicting identities, self copies or lifecycle facts.
    Call this only after complete source-set/epoch checks and availability gating.
    """
    _validate_shared_unit_axes(projection)
    frame = projection.base_sensor_frame
    public_id_by_slot = _shared_public_id_by_global_slot(projection)
    self_public_id = public_id_by_slot[source_global_slot]
    if (
        self_public_id != source.source_public_agent_id
        or self_public_id != frame.public_agent_id
    ):
        raise ValueError("SharedObs admitted source self identity changed.")
    self_row = _shared_authorized_row(
        public_agent_id=self_public_id,
        decoded=decode_agent_feature_row(
            frame.self_features,
            schema_version=frame.schema_version,
            team_id=topology_by_global_slot[
                source_global_slot
            ].sensor_source_configured_team_id,
            is_enemy=False,
        ),
        topology=topology_by_global_slot[source_global_slot],
        recipient_public_agent_id=recipient_public_agent_id,
        recipient_team_id=recipient_team_id,
        lifecycle=recipient_lifecycle,
        relation_row_must_be_alive=False,
    )
    rows: list[_SharedContributionV1] = [
        _SharedContributionV1(row=self_row, source=source)
    ]
    for relation_axis, global_slots, public_ids, visibility, unit_rows in (
        (
            "ally",
            projection.ally_observation_row_global_slot_by_id,
            projection.axis_mapping.ally_observation_row_public_agent_id_by_id,
            frame.ally_visibility_mask,
            frame.ally_unit_features,
        ),
        (
            "enemy",
            projection.enemy_observation_row_global_slot_by_id,
            projection.axis_mapping.enemy_observation_row_public_agent_id_by_id,
            frame.enemy_visibility_mask,
            frame.enemy_unit_features,
        ),
    ):
        for observation_row, is_visible in enumerate(visibility):
            if not is_visible:
                continue
            global_slot = global_slots[observation_row]
            public_agent_id = public_ids[observation_row]
            raw_row = unit_rows[observation_row]
            if public_agent_id == self_public_id:
                if relation_axis != "ally" or raw_row != frame.self_features:
                    raise ValueError(
                        "SharedObs visible self diagonal conflicts with self row."
                    )
                continue
            topology = topology_by_global_slot[global_slot]
            if topology.sensor_source_public_agent_id != public_agent_id:
                raise ValueError("SharedObs relation row remaps a public identity.")
            rows.append(
                _SharedContributionV1(
                    row=_shared_authorized_row(
                        public_agent_id=public_agent_id,
                        decoded=decode_agent_feature_row(
                            raw_row,
                            schema_version=frame.schema_version,
                            team_id=topology.sensor_source_configured_team_id,
                            is_enemy=relation_axis == "enemy",
                        ),
                        topology=topology,
                        recipient_public_agent_id=recipient_public_agent_id,
                        recipient_team_id=recipient_team_id,
                        lifecycle=recipient_lifecycle,
                        relation_row_must_be_alive=True,
                    ),
                    source=source,
                )
            )
    return tuple(rows)


def _merge_shared_contributions(
    contributions: tuple[_SharedContributionV1, ...],
) -> tuple[
    tuple[_AuthorizedRowV1, ...],
    dict[str, tuple[SharedObsAuthorizedSensorSourceV1, ...]],
]:
    """Deduplicate authorized bodies while retaining every contributing sensor.

    contributions is a tuple of admitted body/source pairs. Equal public IDs
    must have exactly equal decoded facts; otherwise raise ValueError. Return
    (rows sorted by physical team/public ID, provenance by public ID), with
    unique sources ordered recipient-first then public ID. Inputs are unchanged.
    """
    row_by_public_id: dict[str, _AuthorizedRowV1] = {}
    sources_by_public_id: dict[str, dict[str, SharedObsAuthorizedSensorSourceV1]] = {}
    for contribution in contributions:
        public_agent_id = contribution.row.public_agent_id
        existing = row_by_public_id.setdefault(public_agent_id, contribution.row)
        if existing != contribution.row:
            raise ValueError(
                "SharedObs sources disagree about one authorized agent row."
            )
        source_by_id = sources_by_public_id.setdefault(public_agent_id, {})
        previous_source = source_by_id.setdefault(
            contribution.source.source_public_agent_id,
            contribution.source,
        )
        if previous_source != contribution.source:
            raise ValueError("SharedObs source identity is internally inconsistent.")
    rows = tuple(
        sorted(
            row_by_public_id.values(),
            key=lambda row: (row.decoded.team_id, row.public_agent_id),
        )
    )
    provenance = {
        public_agent_id: tuple(
            sorted(source_by_id.values(), key=_shared_source_sort_key)
        )
        for public_agent_id, source_by_id in sources_by_public_id.items()
    }
    return rows, provenance


def build_no_shared_obs_authorized_scene_v1(
    source: NoSharedObsPovSourceV1,
    *,
    public_catalog: StaticMechanicsCatalogV1,
    authority_session_id: str,
    frame_index: int | None = None,
) -> NoSharedObsAuthorizedScenePartsV1:
    """Build a neutral NoSharedObs scene from recipient-authorized records.

    Parameters
    ----------
    source : NoSharedObsPovSourceV1
        Exact supported POV replay/index, current slice or adjacent-transition
        slice. The authority boundary revalidates even an existing replay index.
    public_catalog : StaticMechanicsCatalogV1
        Exact public catalog; contents and digest are revalidated independently.
    authority_session_id : str
        Nonblank host-owned viewing-session ID used to scope opaque body keys.
    frame_index : int or None, optional
        Required for replay/index and adjacent slices. For an adjacent slice it
        must equal one endpoint index. A current slice accepts None or its index.

    Returns
    -------
    NoSharedObsAuthorizedScenePartsV1
        New neutral scene, recipient/frame identity and the recipient's exact
        next-decision mask. Bodies contain self plus visible living relation rows.

    Raises
    ------
    TypeError
        Source or catalog has the wrong exact supported root.
    IndexError
        A replay/index frame index is missing, invalid or outside its prefix.
    ValueError
        Session text, live index, source/catalog validation, identity, lifecycle
        or fixed class-capability joins fail.

    Notes
    -----
    This host path never accepts an Oracle scene or full episode context.
    It uses public catalog explanations while keeping observed per-slot profile
    values on agent rows. Hidden bodies are absent; unshown spawn-pad assignments
    have no body identity. Current statuses have no invented direct source IDs.
    No simulator, policy or renderer runs, and input records remain unchanged.
    """
    _require_text(authority_session_id, name="authority_session_id")
    catalog = _validated_catalog(public_catalog)
    documentation_profile = authorized_class_documentation_profile_v1(catalog)
    validated_source = _validated_source(source)
    selection = _select_source(validated_source, frame_index=frame_index)
    frame = selection.frame
    if (
        type(validated_source) is ActorPovAdjacentTransitionSliceV1
        or type(validated_source) is ActorPovAdjacentTransitionSliceV2
    ):
        # A nonzero start endpoint intentionally has no prior incoming
        # transition.  Decode only its battlefield facts; do not fabricate a
        # legacy AnalyzerProjection incoming identity.
        legacy_scene = _build_actor_pov_battlefield_scene_v1(
            frame,
            episode_id=validated_source.episode_id,
            selected_global_slot=validated_source.selected_global_slot,
            selected_team_local_slot=validated_source.selected_team_local_slot,
            public_agent_id=validated_source.public_agent_id,
            configured_team_id=validated_source.configured_team_id,
            class_id=validated_source.class_id,
            observation_materialization=(validated_source.observation_materialization),
            axis_mapping=validated_source.axis_mapping,
        )
    else:
        # The compatibility projection performs the exact current-slice/content
        # validation and decodes map/lifecycle facts without any Oracle input.
        legacy_source = cast(
            ActorPovProjectionIndexV1
            | ActorPovReplayContentV1
            | ActorPovCurrentSliceV1,
            validated_source,
        )
        legacy_scene = build_actor_pov_analyzer_projection_v1(
            legacy_source,
            frame_index=frame_index,
        ).scene
    lifecycle = frame.spawn_lifecycle
    self_decoded = decode_agent_feature_row(
        frame.self_features,
        schema_version=frame.schema_version,
        team_id=selection.configured_team_id,
        is_enemy=False,
    )
    if (
        self_decoded.team_id != selection.configured_team_id
        or self_decoded.class_id != selection.class_id
    ):
        raise ValueError("POV self row does not join recipient identity.")
    if (
        not lifecycle.active_mask_by_team[0][selection.selected_team_local_slot]
        or self_decoded.alive
        != lifecycle.alive_mask_by_team[0][selection.selected_team_local_slot]
    ):
        raise ValueError("POV self row conflicts with recipient lifecycle input.")

    authorized_rows: list[_AuthorizedRowV1] = [
        _AuthorizedRowV1(
            public_agent_id=selection.public_agent_id,
            relation="self",
            decoded=self_decoded,
            spawn_shield_remaining=(
                lifecycle.spawn_shield_actual_durations_by_team[0][
                    selection.selected_team_local_slot
                ]
            ),
        )
    ]
    seen_public_ids = {selection.public_agent_id}
    for body in legacy_scene.visible_bodies:
        raw_rows = (
            frame.ally_unit_features
            if body.relation == "ally"
            else frame.enemy_unit_features
        )
        raw_row = raw_rows[body.observation_row]
        if body.public_agent_id == selection.public_agent_id:
            if body.relation != "ally" or raw_row != frame.self_features:
                raise ValueError("visible self diagonal conflicts with self_features.")
            continue
        if body.public_agent_id in seen_public_ids:
            raise ValueError("authorized POV rows repeat one public identity.")
        decoded = decode_agent_feature_row(
            raw_row,
            schema_version=frame.schema_version,
            team_id=body.team_id,
            is_enemy=body.relation == "enemy",
        )
        relative_team_index = 0 if body.relation == "ally" else 1
        expected_team_id = _absolute_team_id(
            recipient_team_id=selection.configured_team_id,
            actor_relative_index=relative_team_index,
        )
        if decoded.team_id != expected_team_id:
            raise ValueError("visible relation and recorded team ID disagree.")
        if not decoded.configured_active or not decoded.alive:
            raise ValueError(
                "visible POV rows must remain configured active and alive."
            )
        lifecycle_alive = lifecycle.alive_mask_by_team[relative_team_index][
            body.observation_row
        ]
        lifecycle_active = lifecycle.active_mask_by_team[relative_team_index][
            body.observation_row
        ]
        if not lifecycle_active or not lifecycle_alive:
            raise ValueError("visible body life state conflicts with lifecycle input.")
        authorized_rows.append(
            _AuthorizedRowV1(
                public_agent_id=body.public_agent_id,
                relation="ally" if body.relation == "ally" else "opponent",
                decoded=decoded,
                spawn_shield_remaining=(
                    lifecycle.spawn_shield_actual_durations_by_team[
                        relative_team_index
                    ][body.observation_row]
                ),
            )
        )
        seen_public_ids.add(body.public_agent_id)

    mechanics_by_class: dict[int, AuthorizedClassMechanicsV2] = {}
    for row in authorized_rows:
        mechanics = _class_mechanics(
            row.decoded,
            catalog,
            documentation_profile=documentation_profile,
        )
        previous = mechanics_by_class.setdefault(row.decoded.class_id, mechanics)
        if previous != mechanics:
            raise ValueError(
                "same-class authorized rows carry conflicting static capabilities."
            )
    agents = tuple(
        _agent(
            row,
            catalog=catalog,
            authority_session_id=authority_session_id,
            recipient_public_agent_id=selection.public_agent_id,
        )
        for row in authorized_rows
    )
    agent_by_public_id = {row.public_agent_id: row for row in agents}

    aura_fields = tuple(
        AuthorizedAuraFieldV1(
            aura_id=mechanic.aura_id,
            source_presentation_key=agent.presentation_key,
            source_public_agent_id=agent.public_agent_id,
            source_class_id=agent.class_id,
            source_class_name=agent.class_name,
            source_alive=agent.life_state == "alive",
            center=agent.position,
            radius=mechanic.radius,
            beneficiary_relation=catalog_mechanic.beneficiary_relation,
            per_emitter_multiplier=mechanic.per_emitter_multiplier,
            stacking_rule=mechanic.stacking_rule,
            clamp_kind=mechanic.clamp_kind,
            clamp_value=mechanic.clamp_value,
        )
        for agent, authorized_row in zip(agents, authorized_rows, strict=True)
        for catalog_mechanic in catalog.aura_mechanics
        if catalog_mechanic.emitter_class_id == agent.class_id
        for mechanic in (_row_aura_mechanic(catalog_mechanic, authorized_row.decoded),)
    )

    spawn_pads: list[AuthorizedSpawnPadV1] = []
    for pad in legacy_scene.spawn_pads:
        team_id = _absolute_team_id(
            recipient_team_id=selection.configured_team_id,
            actor_relative_index=pad.actor_relative_team_index,
        )
        public_axis = (
            selection.axis_mapping.ally_observation_row_public_agent_id_by_id
            if pad.actor_relative_team_index == 0
            else selection.axis_mapping.enemy_observation_row_public_agent_id_by_id
        )
        candidate_public_id = public_axis[pad.team_local_slot]
        assigned = agent_by_public_id.get(candidate_public_id)
        if not pad.configured_active:
            assigned = None
        spawn_pads.append(
            AuthorizedSpawnPadV1(
                team_id=team_id,
                team_local_slot=pad.team_local_slot,
                assigned_presentation_key=(
                    None if assigned is None else assigned.presentation_key
                ),
                assigned_public_agent_id=(
                    None if assigned is None else assigned.public_agent_id
                ),
                position=pad.position,
                configured_active=pad.configured_active,
                currently_alive=pad.currently_alive,
                spawn_shield_remaining=pad.spawn_shield_remaining,
            )
        )
    spawn_pads.sort(key=lambda row: (row.team_id, row.team_local_slot))

    respawn_waves = tuple(
        sorted(
            (
                AuthorizedRespawnWaveV1(
                    team_index=(
                        _absolute_team_id(
                            recipient_team_id=selection.configured_team_id,
                            actor_relative_index=wave.actor_relative_team_index,
                        )
                        - 1
                    ),
                    team_id=_absolute_team_id(
                        recipient_team_id=selection.configured_team_id,
                        actor_relative_index=wave.actor_relative_team_index,
                    ),
                    period_steps=wave.period_steps,
                    countdown_steps=wave.countdown_steps,
                )
                for wave in legacy_scene.respawn_waves
            ),
            key=lambda row: row.team_index,
        )
    )
    scene = AuthorizedBattlefieldSceneV1(
        schema_version=AUTHORIZED_PRESENTATION_SCHEMA_VERSION,
        map=_authorized_map(legacy_scene.map),
        agents=agents,
        aura_fields=aura_fields,
        class_mechanics=tuple(
            mechanics_by_class[class_id] for class_id in sorted(mechanics_by_class)
        ),
        spawn_shield_mechanics=AuthorizedSpawnShieldMechanicsAvailableV2(
            availability_kind="available_v2",
            configured_duration_steps=lifecycle.spawn_shield_configured_duration,
            movement_speed=lifecycle.spawn_shield_speed,
            protection_effect="invulnerable",
            visibility_effect="concealed_from_opponents",
            targetability_effect="untargetable",
            action_scope="movement_only",
            aura_effect="excluded_as_emitter_and_beneficiary",
            agent_collision_effect="phased_until_expiring_endpoint_rejoin",
            ordinary_application_mechanism=("end_of_transition_respawn_lifecycle"),
        ),
        spawn_pads=tuple(spawn_pads),
        respawn_waves=respawn_waves,
    )
    recipient_key = pov_presentation_key_v1(
        authority_session_id=authority_session_id,
        recipient_public_agent_id=selection.public_agent_id,
        public_agent_id=selection.public_agent_id,
    )
    return NoSharedObsAuthorizedScenePartsV1(
        source_episode_id=frame.episode_id,
        source_frame_index=frame.frame_index,
        source_recipient_frame_id=frame.pov_frame_id,
        source_simulator_step_count=frame.simulator_step_count,
        recipient_public_agent_id=selection.public_agent_id,
        recipient_presentation_key=recipient_key,
        scene=scene,
        next_decision_action_mask=frame.action_mask,
    )


def build_shared_obs_authorized_scene_v1(
    recipient_source_material: SharedObsSourceMaterialProjection,
    *,
    all_active_nonrecipient_source_material: tuple[
        SharedObsSourceMaterialProjection, ...
    ],
    public_catalog: StaticMechanicsCatalogV1,
    authority_session_id: str,
) -> SharedObsAuthorizedScenePartsV1:
    """Build one recipient's visual union from admitted recorded SharedObs sources.

    Parameters
    ----------
    recipient_source_material : SharedObsSourceMaterialProjectionV1 or V2
        Exact selected recipient projection. Its availability, lifecycle, map and
        next-decision mask own the resulting view.
    all_active_nonrecipient_source_material : tuple of SharedObsSourceMaterialProjection
        Every and only configured-active nonrecipient source, including opponents
        and unavailable sources. Roots, headers, identity topology and epoch are
        checked before admission. Order is arbitrary; duplicates are rejected.
    public_catalog : StaticMechanicsCatalogV1
        Exact public catalog; structure and digest are revalidated.
    authority_session_id : str
        Nonblank host-owned viewing-session ID used to scope recipient body keys.

    Returns
    -------
    SharedObsAuthorizedScenePartsV1
        Neutral scene sorted by physical team/public ID, recipient-owned mask,
        admitted sensor list and per-body observation provenance. This is a
        visual union, not an export of exact composed SharedObs policy tensors.

    Raises
    ------
    TypeError
        Catalog or source container/root types are unsupported.
    ValueError
        Source set, headers, topology, epoch, availability, lifecycle, catalog
        or duplicate-body facts disagree.

    Notes
    -----
    Read body payloads only from the recipient and allies admitted by that
    recipient's recorded availability. Unavailable source unit payloads are
    not validated, decoded or composed. Admitted self/visible rows must agree
    with recipient lifecycle; duplicate body facts must match exactly. The
    recipient's map, lifecycle and mask are never replaced by another source.
    No Oracle snapshot, policy execution, file I/O or input mutation is used.
    """
    _require_text(authority_session_id, name="authority_session_id")
    catalog = _validated_catalog(public_catalog)
    documentation_profile = authorized_class_documentation_profile_v1(catalog)
    (
        topology_rows,
        topology_by_global_slot,
        recipient_public_id_by_slot,
        recipient_topology,
        recipient_map,
    ) = _validated_recipient_topology(recipient_source_material)
    recipient_frame = recipient_source_material.base_sensor_frame
    recipient_public_agent_id = recipient_frame.public_agent_id
    recipient_global_slot = recipient_topology.sensor_source_global_slot
    recipient_team_id = recipient_topology.sensor_source_configured_team_id
    lifecycle = recipient_frame.spawn_lifecycle
    if type(lifecycle) is not ActorPovSpawnLifecycleV1:
        raise ValueError("SharedObs recipient lifecycle must use its exact root.")
    validated_lifecycle = ActorPovSpawnLifecycleV1.model_validate(
        lifecycle.model_dump(mode="python")
    )
    if validated_lifecycle != lifecycle:
        raise ValueError("SharedObs recipient lifecycle changes under revalidation.")
    if type(recipient_frame.action_mask) is not ActorPovActionMaskV1:
        raise ValueError("SharedObs recipient action mask must use its exact root.")
    validated_mask = ActorPovActionMaskV1.model_validate(
        recipient_frame.action_mask.model_dump(mode="python")
    )
    if validated_mask != recipient_frame.action_mask:
        raise ValueError("SharedObs recipient action mask changes under revalidation.")
    for topology in topology_rows:
        absolute_team_id = (
            1 if topology.sensor_source_global_slot < MAX_AGENTS_PER_TEAM_V1 else 2
        )
        actor_relative_team_index = 0 if absolute_team_id == recipient_team_id else 1
        team_local_slot = topology.sensor_source_team_local_slot
        configured_active = lifecycle.active_mask_by_team[actor_relative_team_index][
            team_local_slot
        ]
        currently_alive = lifecycle.alive_mask_by_team[actor_relative_team_index][
            team_local_slot
        ]
        shield_remaining = lifecycle.spawn_shield_actual_durations_by_team[
            actor_relative_team_index
        ][team_local_slot]
        if configured_active != topology.sensor_source_configured_active:
            raise ValueError(
                "SharedObs recipient lifecycle and source topology disagree."
            )
        if not configured_active and (currently_alive or shield_remaining != 0):
            raise ValueError("inactive SharedObs lifecycle rows must remain empty.")

    if type(all_active_nonrecipient_source_material) is not tuple or any(
        type(source)
        not in (
            SharedObsSourceMaterialProjectionV1,
            SharedObsSourceMaterialProjectionV2,
        )
        for source in all_active_nonrecipient_source_material
    ):
        raise TypeError(
            "all active nonrecipient SharedObs sources must use exact projections."
        )
    source_headers = tuple(
        _shared_source_header(
            source,
            recipient_projection=recipient_source_material,
            recipient_public_id_by_slot=recipient_public_id_by_slot,
            recipient_topology_by_slot=topology_by_global_slot,
        )
        for source in all_active_nonrecipient_source_material
    )
    expected_nonrecipient_slots = {
        row.sensor_source_global_slot
        for row in topology_rows
        if row.sensor_source_configured_active
        and row.sensor_source_global_slot != recipient_global_slot
    }
    actual_nonrecipient_slots = {header.source_global_slot for header in source_headers}
    if len(actual_nonrecipient_slots) != len(source_headers):
        raise ValueError("SharedObs nonrecipient source projections are duplicated.")
    if actual_nonrecipient_slots != expected_nonrecipient_slots:
        raise ValueError(
            "SharedObs requires every and only active nonrecipient source projection."
        )
    header_by_slot = {header.source_global_slot: header for header in source_headers}

    recipient_key = pov_presentation_key_v1(
        authority_session_id=authority_session_id,
        recipient_public_agent_id=recipient_public_agent_id,
        public_agent_id=recipient_public_agent_id,
    )
    recipient_sensor_source = SharedObsAuthorizedSensorSourceV1(
        source_kind="recipient_base",
        source_presentation_key=recipient_key,
        source_public_agent_id=recipient_public_agent_id,
    )
    admitted_topology = tuple(
        sorted(
            (row for row in topology_rows if row.recorded_available),
            key=lambda row: row.sensor_source_public_agent_id,
        )
    )
    authorized_sensor_sources = (
        recipient_sensor_source,
        *(
            SharedObsAuthorizedSensorSourceV1(
                source_kind="shared_sensor_source",
                source_presentation_key=pov_presentation_key_v1(
                    authority_session_id=authority_session_id,
                    recipient_public_agent_id=recipient_public_agent_id,
                    public_agent_id=row.sensor_source_public_agent_id,
                ),
                source_public_agent_id=row.sensor_source_public_agent_id,
            )
            for row in admitted_topology
        ),
    )

    contributions = list(
        _shared_projection_contributions(
            recipient_source_material,
            source=recipient_sensor_source,
            source_global_slot=recipient_global_slot,
            topology_by_global_slot=topology_by_global_slot,
            recipient_public_agent_id=recipient_public_agent_id,
            recipient_team_id=recipient_team_id,
            recipient_lifecycle=lifecycle,
        )
    )
    source_by_public_id = {
        source.source_public_agent_id: source for source in authorized_sensor_sources
    }
    for topology in admitted_topology:
        if (
            topology.relation_to_recipient != "ally"
            or not topology.sensor_source_configured_active
            or topology.sensor_source_global_slot == recipient_global_slot
        ):
            raise ValueError("SharedObs admitted a non-allied sensor source.")
        header = header_by_slot[topology.sensor_source_global_slot]
        # This is deliberately after the complete source-set/epoch join.  No
        # unavailable source unit payload is validated, decoded, or composed.
        contributions.extend(
            _shared_projection_contributions(
                header.projection,
                source=source_by_public_id[topology.sensor_source_public_agent_id],
                source_global_slot=topology.sensor_source_global_slot,
                topology_by_global_slot=topology_by_global_slot,
                recipient_public_agent_id=recipient_public_agent_id,
                recipient_team_id=recipient_team_id,
                recipient_lifecycle=lifecycle,
            )
        )
    authorized_rows, provenance_by_public_id = _merge_shared_contributions(
        tuple(contributions)
    )

    mechanics_by_class: dict[int, AuthorizedClassMechanicsV2] = {}
    for row in authorized_rows:
        mechanics = _class_mechanics(
            row.decoded,
            catalog,
            documentation_profile=documentation_profile,
        )
        previous = mechanics_by_class.setdefault(row.decoded.class_id, mechanics)
        if previous != mechanics:
            raise ValueError(
                "same-class SharedObs rows carry conflicting public mechanics."
            )
    agents = tuple(
        _agent(
            row,
            catalog=catalog,
            authority_session_id=authority_session_id,
            recipient_public_agent_id=recipient_public_agent_id,
        )
        for row in authorized_rows
    )
    agent_by_public_id = {agent.public_agent_id: agent for agent in agents}

    aura_fields = tuple(
        sorted(
            (
                AuthorizedAuraFieldV1(
                    aura_id=mechanic.aura_id,
                    source_presentation_key=agent.presentation_key,
                    source_public_agent_id=agent.public_agent_id,
                    source_class_id=agent.class_id,
                    source_class_name=agent.class_name,
                    source_alive=agent.life_state == "alive",
                    center=agent.position,
                    radius=mechanic.radius,
                    beneficiary_relation=catalog_mechanic.beneficiary_relation,
                    per_emitter_multiplier=mechanic.per_emitter_multiplier,
                    stacking_rule=mechanic.stacking_rule,
                    clamp_kind=mechanic.clamp_kind,
                    clamp_value=mechanic.clamp_value,
                )
                for row in authorized_rows
                for agent in (agent_by_public_id[row.public_agent_id],)
                for catalog_mechanic in catalog.aura_mechanics
                if catalog_mechanic.emitter_class_id == agent.class_id
                for mechanic in (_row_aura_mechanic(catalog_mechanic, row.decoded),)
            ),
            key=lambda field: (field.source_public_agent_id, field.aura_id),
        )
    )

    spawn_pads: list[AuthorizedSpawnPadV1] = []
    for actor_relative_team_index in range(2):
        team_id = _absolute_team_id(
            recipient_team_id=recipient_team_id,
            actor_relative_index=actor_relative_team_index,
        )
        for team_local_slot in range(MAX_AGENTS_PER_TEAM_V1):
            pad_position = lifecycle.spawn_pad_positions_by_team[
                actor_relative_team_index
            ][team_local_slot]
            topology = next(
                row
                for row in topology_rows
                if row.sensor_source_team_local_slot == team_local_slot
                and (1 if row.sensor_source_global_slot < MAX_AGENTS_PER_TEAM_V1 else 2)
                == team_id
            )
            configured_active = lifecycle.active_mask_by_team[
                actor_relative_team_index
            ][team_local_slot]
            assigned = agent_by_public_id.get(topology.sensor_source_public_agent_id)
            if not configured_active:
                assigned = None
            spawn_pads.append(
                AuthorizedSpawnPadV1(
                    team_id=team_id,
                    team_local_slot=team_local_slot,
                    assigned_presentation_key=(
                        None if assigned is None else assigned.presentation_key
                    ),
                    assigned_public_agent_id=(
                        None if assigned is None else assigned.public_agent_id
                    ),
                    position=(pad_position[0], pad_position[1]),
                    configured_active=configured_active,
                    currently_alive=lifecycle.alive_mask_by_team[
                        actor_relative_team_index
                    ][team_local_slot],
                    spawn_shield_remaining=(
                        lifecycle.spawn_shield_actual_durations_by_team[
                            actor_relative_team_index
                        ][team_local_slot]
                    ),
                )
            )
    spawn_pads.sort(key=lambda row: (row.team_id, row.team_local_slot))
    respawn_waves = tuple(
        sorted(
            (
                AuthorizedRespawnWaveV1(
                    team_index=(
                        _absolute_team_id(
                            recipient_team_id=recipient_team_id,
                            actor_relative_index=actor_relative_team_index,
                        )
                        - 1
                    ),
                    team_id=_absolute_team_id(
                        recipient_team_id=recipient_team_id,
                        actor_relative_index=actor_relative_team_index,
                    ),
                    period_steps=lifecycle.respawn_wave_period_step_count_by_team[
                        actor_relative_team_index
                    ],
                    countdown_steps=lifecycle.respawn_wave_countdowns_by_team[
                        actor_relative_team_index
                    ],
                )
                for actor_relative_team_index in range(2)
            ),
            key=lambda row: row.team_index,
        )
    )
    scene = AuthorizedBattlefieldSceneV1(
        schema_version=AUTHORIZED_PRESENTATION_SCHEMA_VERSION,
        map=_authorized_map(recipient_map),
        agents=agents,
        aura_fields=aura_fields,
        class_mechanics=tuple(
            mechanics_by_class[class_id] for class_id in sorted(mechanics_by_class)
        ),
        spawn_shield_mechanics=AuthorizedSpawnShieldMechanicsAvailableV2(
            availability_kind="available_v2",
            configured_duration_steps=lifecycle.spawn_shield_configured_duration,
            movement_speed=lifecycle.spawn_shield_speed,
            protection_effect="invulnerable",
            visibility_effect="concealed_from_opponents",
            targetability_effect="untargetable",
            action_scope="movement_only",
            aura_effect="excluded_as_emitter_and_beneficiary",
            agent_collision_effect="phased_until_expiring_endpoint_rejoin",
            ordinary_application_mechanism=("end_of_transition_respawn_lifecycle"),
        ),
        spawn_pads=tuple(spawn_pads),
        respawn_waves=respawn_waves,
    )
    provenance = tuple(
        SharedObsAgentObservationProvenanceV1(
            agent_presentation_key=agent.presentation_key,
            agent_public_agent_id=agent.public_agent_id,
            observation_sources=provenance_by_public_id[agent.public_agent_id],
        )
        for agent in agents
    )
    return SharedObsAuthorizedScenePartsV1(
        source_episode_id=recipient_frame.episode_id,
        source_frame_index=recipient_frame.frame_index,
        source_recipient_frame_id=(
            f"{recipient_frame.episode_id}:shared-obs-visual-union:"
            f"{recipient_public_agent_id}:frame:{recipient_frame.frame_index}"
        ),
        source_simulator_step_count=recipient_frame.simulator_step_count,
        recipient_public_agent_id=recipient_public_agent_id,
        recipient_presentation_key=recipient_key,
        scene=scene,
        next_decision_action_mask=recipient_frame.action_mask,
        authorized_sensor_sources=authorized_sensor_sources,
        agent_observation_provenance=provenance,
    )


__all__ = [
    "NoSharedObsAuthorizedScenePartsV1",
    "NoSharedObsPovSourceV1",
    "SharedObsAgentObservationProvenanceV1",
    "SharedObsAuthorizedScenePartsV1",
    "SharedObsAuthorizedSensorSourceV1",
    "build_no_shared_obs_authorized_scene_v1",
    "build_shared_obs_authorized_scene_v1",
    "pov_presentation_key_v1",
]
