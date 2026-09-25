"""Represent historical V1 replays and their separate metric-report artifacts.

A replay stores context once, T+1 frames, and T transitions, including an initial
frame for an empty prefix. It contains host records, not live EnvState objects,
renderer images, policy internals, or local paths. Content digests bind the
trajectory and metric sidecar without a circular reference. Current frame V3
recording uses replay_v4; these V1 readers preserve historical compatibility.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from math import prod
from typing import Annotated, Literal, cast

from pydantic import BeforeValidator, Field, StringConstraints, model_validator

from marl_battlegrounds.evaluation.metrics import (
    EPISODE_COMPLETION_SCHEMA_ID,
    METRIC_REPORT_SCHEMA_ID,
    PROCESSING_STATUS_SCHEMA_ID,
    RAW_SUFFICIENT_STATISTIC_SCHEMA_ID,
    EvaluationEpisodeCompletionV1,
    EvaluationEpisodeObserverV1,
    EvaluationMetricReportV1,
    EvaluationProcessingStatusV1,
    EvaluationTransitionViewV1,
    validate_evaluation_processing_progress_v1,
)
from marl_battlegrounds.evaluation.models import (
    REQUIRED_SCHEMA_BINDINGS_V1,
    EvaluationEpisodeContextV1,
    EvaluationFrameV1,
    EvaluationModel,
    EvaluationTransitionV1,
    SchemaVersionEntryV1,
    canonical_digest_sha256,
    canonical_json_bytes,
)
from marl_battlegrounds.evaluation.validation import (
    validate_declared_model_tree,
    validate_initial_evaluation_frame_v1,
)

REPLAY_SCHEMA_VERSION: Literal[1] = 1
CANONICAL_REPLAY_JSON_PROFILE_V1 = "marl_battlegrounds.canonical_json.v1"

RUNTIME_PROVENANCE_SCHEMA_ID = "marl_battlegrounds.evaluation.runtime_provenance"
REPLAY_WRAPPER_METADATA_SCHEMA_ID = (
    "marl_battlegrounds.evaluation.replay_wrapper_metadata"
)
REPLAY_TRAJECTORY_CONTENT_REFERENCE_SCHEMA_ID = (
    "marl_battlegrounds.evaluation.replay_trajectory_content_reference"
)
METRIC_REPORT_ARTIFACT_SCHEMA_ID = (
    "marl_battlegrounds.evaluation.metric_report_artifact"
)
METRIC_REPORT_REFERENCE_SCHEMA_ID = (
    "marl_battlegrounds.evaluation.metric_report_reference"
)
REPLAY_ARTIFACT_REFERENCE_SCHEMA_ID = (
    "marl_battlegrounds.evaluation.replay_artifact_reference"
)
REPLAY_HEADER_SCHEMA_ID = "marl_battlegrounds.evaluation.replay_header"
REPLAY_ARTIFACT_SCHEMA_ID = "marl_battlegrounds.evaluation.replay_artifact"

REQUIRED_REPLAY_ENVELOPE_SCHEMA_BINDINGS_V1 = (
    (EPISODE_COMPLETION_SCHEMA_ID, 1),
    (PROCESSING_STATUS_SCHEMA_ID, 1),
    (RAW_SUFFICIENT_STATISTIC_SCHEMA_ID, 1),
    (METRIC_REPORT_SCHEMA_ID, 1),
    (RUNTIME_PROVENANCE_SCHEMA_ID, 1),
    (REPLAY_WRAPPER_METADATA_SCHEMA_ID, 1),
    (REPLAY_TRAJECTORY_CONTENT_REFERENCE_SCHEMA_ID, 1),
    (METRIC_REPORT_ARTIFACT_SCHEMA_ID, 1),
    (METRIC_REPORT_REFERENCE_SCHEMA_ID, 1),
    (REPLAY_HEADER_SCHEMA_ID, 1),
    (REPLAY_ARTIFACT_SCHEMA_ID, 1),
)


def _require_schema_version_one(value: object) -> object:
    """Accept only the exact Python integer 1, rejecting bool and coercible values.

    Return the unchanged value for Pydantic; raise ValueError otherwise.
    """
    if type(value) is not int or value != 1:
        raise ValueError("schema_version must be the exact integer 1")
    return value


_SchemaVersionV1 = Annotated[
    Literal[1],
    BeforeValidator(_require_schema_version_one),
]
_AsciiText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        pattern=r"^[\x20-\x7e]+$",
    ),
]
_AsciiIdentifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/+\-]*$",
    ),
]
_Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_NonNegativeInt = Annotated[int, Field(ge=0)]
_PositiveInt = Annotated[int, Field(gt=0)]


def _schema_bindings(
    rows: tuple[SchemaVersionEntryV1, ...],
) -> tuple[tuple[str, int], ...]:
    """Return schema-ID/version pairs in the supplied row order without sorting.

    Header validation compares this ordered tuple with the frozen V1 bindings.
    """
    return tuple((row.schema_id, row.schema_version) for row in rows)


def _require_exact_nested_model(
    model: EvaluationModel,
    *,
    expected_type: type[EvaluationModel],
    record_name: str,
) -> None:
    """Revalidate a nested record's exact declared type and unchanged field values.

    Raise ValueError through the shared strict-tree validator; no subtype or
    unchecked Pydantic construction is admitted merely because it compares equal.
    """
    validate_declared_model_tree(
        model,
        record_name=record_name,
        expected_type=expected_type,
    )


class RuntimeProvenanceV1(EvaluationModel):
    """Record software, device, precision, and batch facts for one execution.

    Attributes
    ----------
    schema_id : Literal['marl_battlegrounds.evaluation.runtime_provenance']
        Fixed runtime-provenance schema identifier.
    schema_version : _SchemaVersionV1
        Exact integer 1; defaults to 1.
    python_version : _AsciiText
        Recorded Python version text.
    package_version : _AsciiText
        Recorded MARL-BGs package version text.
    jax_version : _AsciiText
        Recorded JAX version text.
    jaxlib_version : _AsciiText
        Recorded jaxlib version text.
    numpy_version : _AsciiText
        Recorded NumPy version text.
    pydantic_version : _AsciiText
        Recorded Pydantic version text.
    platform : _AsciiText
        Recorded operating-system/platform text.
    machine : _AsciiText
        Recorded machine architecture text.
    backend : _AsciiText
        Recorded execution backend, such as cpu or gpu.
    device : _AsciiText
        Recorded device description.
    driver_version : _AsciiText | None
        Optional discovered driver text; None means unrecorded.
    runtime_version : _AsciiText | None
        Optional discovered runtime text; None means unrecorded.
    precision : _AsciiIdentifier
        Nonempty identifier for the declared numeric precision.
    environment_count : _PositiveInt
        Positive number of environments in this execution batch.
    batch_shape : tuple[_PositiveInt, ...]
        Positive batch dimensions whose product equals environment_count.
        An empty tuple represents one scalar environment.
    policy_execution_included : bool
        Whether the recorded execution includes policy work.

    Notes
    -----
    This model records supplied facts; it does not discover software or hardware.
    Nonempty printable ASCII text is required. Construction validates batch size.
    """

    schema_id: Literal["marl_battlegrounds.evaluation.runtime_provenance"] = (
        RUNTIME_PROVENANCE_SCHEMA_ID
    )
    schema_version: _SchemaVersionV1 = REPLAY_SCHEMA_VERSION
    python_version: _AsciiText
    package_version: _AsciiText
    jax_version: _AsciiText
    jaxlib_version: _AsciiText
    numpy_version: _AsciiText
    pydantic_version: _AsciiText
    platform: _AsciiText
    machine: _AsciiText
    backend: _AsciiText
    device: _AsciiText
    driver_version: _AsciiText | None = None
    runtime_version: _AsciiText | None = None
    precision: _AsciiIdentifier
    environment_count: _PositiveInt
    batch_shape: tuple[_PositiveInt, ...]
    policy_execution_included: bool

    @model_validator(mode="after")
    def _validate_environment_shape(self) -> RuntimeProvenanceV1:
        """Require batch-shape product to equal the recorded environment count.

        Return this validated model or raise ValueError. An empty batch shape has
        product one and is valid for scalar execution.
        """
        if prod(self.batch_shape) != self.environment_count:
            raise ValueError("runtime batch shape product must equal environment count")
        return self


class ReplayWrapperMetadataV1(EvaluationModel):
    """Identify one wrapper in the recorded outer execution stack.

    Attributes
    ----------
    schema_id : Literal['marl_battlegrounds.evaluation.replay_wrapper_metadata']
        Fixed wrapper-metadata schema identifier.
    schema_version : _SchemaVersionV1
        Exact integer 1; defaults to 1.
    position : _NonNegativeInt
        Zero-based position in the ordered wrapper stack.
    wrapper_id : _AsciiIdentifier
        Stable nonempty wrapper identifier.
    wrapper_version : _PositiveInt
        Positive version of its behavior contract.
    configuration_digest_sha256 : _Sha256Hex
        Lowercase SHA-256 of normalized wrapper settings.

    Notes
    -----
    The replay header checks that positions are ordered and gap-free. This
    record does not store local paths or executable wrapper objects.
    """

    schema_id: Literal["marl_battlegrounds.evaluation.replay_wrapper_metadata"] = (
        REPLAY_WRAPPER_METADATA_SCHEMA_ID
    )
    schema_version: _SchemaVersionV1 = REPLAY_SCHEMA_VERSION
    position: _NonNegativeInt
    wrapper_id: _AsciiIdentifier
    wrapper_version: _PositiveInt
    configuration_digest_sha256: _Sha256Hex


class ReplayTrajectoryContentReferenceV1(EvaluationModel):
    """Name replay content before adding the metric-report reference.

    Attributes
    ----------
    schema_id :
    Literal['marl_battlegrounds.evaluation.replay_trajectory_content_reference']
        Fixed trajectory-content-reference schema identifier.
    schema_version : _SchemaVersionV1
        Exact integer 1; defaults to 1.
    replay_artifact_id : _AsciiIdentifier
        Canonical episode_id followed by ":replay".
    episode_id : _AsciiIdentifier
        Episode identity shared by the replay and report.
    replay_schema_version : _SchemaVersionV1
        Exact integer 1; defaults to 1.
    context_digest_sha256 : _Sha256Hex
        Lowercase SHA-256 of the recorded context.
    trajectory_content_digest_sha256 : _Sha256Hex
        Lowercase SHA-256 of header, completion,
        processing status, frames, and transitions, excluding the report link.

    Notes
    -----
    This intermediate identity prevents circular hashing between replay and report.
    """

    schema_id: Literal[
        "marl_battlegrounds.evaluation.replay_trajectory_content_reference"
    ] = REPLAY_TRAJECTORY_CONTENT_REFERENCE_SCHEMA_ID
    schema_version: _SchemaVersionV1 = REPLAY_SCHEMA_VERSION
    replay_artifact_id: _AsciiIdentifier
    episode_id: _AsciiIdentifier
    replay_schema_version: _SchemaVersionV1 = REPLAY_SCHEMA_VERSION
    context_digest_sha256: _Sha256Hex
    trajectory_content_digest_sha256: _Sha256Hex

    @model_validator(mode="after")
    def _validate_identity(self) -> ReplayTrajectoryContentReferenceV1:
        """Require the canonical episode-based replay ID or raise ValueError.

        Return the same validated reference.
        """
        if self.replay_artifact_id != f"{self.episode_id}:replay":
            raise ValueError("trajectory reference replay artifact ID is not canonical")
        return self


class EvaluationMetricReportArtifactV1(EvaluationModel):
    """Bind a historical metric report to the exact trajectory it describes.

    Attributes
    ----------
    schema_id : Literal['marl_battlegrounds.evaluation.metric_report_artifact']
        Fixed metric-report-artifact schema identifier.
    schema_version : _SchemaVersionV1
        Exact integer 1; defaults to 1.
    report_artifact_id : _AsciiIdentifier
        Canonical episode ID followed by ":metric-report-artifact".
    canonical_digest_sha256 : _Sha256Hex
        Lowercase SHA-256 of every other field.
    source_trajectory : ReplayTrajectoryContentReferenceV1
        Path-free context and trajectory-content identity.
    report : EvaluationMetricReportV1
        Exact V1 metric report, including context, completion, and processing facts.

    Notes
    -----
    Construction checks strict nested types, episode/context joins, and the
    digest. Pair validation against an actual replay is a separate check.
    """

    schema_id: Literal["marl_battlegrounds.evaluation.metric_report_artifact"] = (
        METRIC_REPORT_ARTIFACT_SCHEMA_ID
    )
    schema_version: _SchemaVersionV1 = REPLAY_SCHEMA_VERSION
    report_artifact_id: _AsciiIdentifier
    canonical_digest_sha256: _Sha256Hex
    source_trajectory: ReplayTrajectoryContentReferenceV1
    report: EvaluationMetricReportV1

    @model_validator(mode="after")
    def _validate_artifact(self) -> EvaluationMetricReportArtifactV1:
        """Check exact nested report types, canonical IDs, context join, and digest.

        Return this artifact or raise ValueError. This local check does not load a
        replay.
        """
        _require_exact_nested_model(
            self.source_trajectory,
            expected_type=ReplayTrajectoryContentReferenceV1,
            record_name="metric report source trajectory",
        )
        _require_exact_nested_model(
            self.report,
            expected_type=EvaluationMetricReportV1,
            record_name="metric report artifact report",
        )
        episode_id = self.report.context.identity.episode_id
        if self.report_artifact_id != f"{episode_id}:metric-report-artifact":
            raise ValueError("metric report artifact ID is not canonical")
        if self.source_trajectory.episode_id != episode_id:
            raise ValueError("metric report artifact must join its trajectory episode")
        if self.source_trajectory.context_digest_sha256 != canonical_digest_sha256(
            self.report.context
        ):
            raise ValueError("metric report context digest must join the trajectory")
        if self.canonical_digest_sha256 != canonical_digest_sha256(
            self,
            exclude={"canonical_digest_sha256"},
        ):
            raise ValueError("metric report artifact digest is not canonical")
        return self


class MetricReportReferenceV1(EvaluationModel):
    """Describe the separate report bytes without embedding a filesystem path.

    Attributes
    ----------
    schema_id : Literal['marl_battlegrounds.evaluation.metric_report_reference']
        Fixed metric-report-reference schema identifier.
    schema_version : _SchemaVersionV1
        Exact integer 1; defaults to 1.
    report_artifact_id : _AsciiIdentifier
        Canonical episode ID followed by ":metric-report-artifact".
    episode_id : _AsciiIdentifier
        Shared episode identity.
    report_artifact_schema_id :
    Literal['marl_battlegrounds.evaluation.metric_report_artifact']
        Fixed metric-report-artifact schema identifier.
    report_artifact_schema_version : _SchemaVersionV1
        Exact integer 1; defaults to 1.
    metric_report_id : _AsciiIdentifier
        Canonical episode ID followed by ":metric-report".
    metric_report_schema_version : _SchemaVersionV1
        Exact integer 1; defaults to 1.
    trajectory_content_digest_sha256 : _Sha256Hex
        Digest of the trajectory described by the report.
    canonical_digest_sha256 : _Sha256Hex
        Digest of the complete referenced report artifact.
    canonical_byte_length : _PositiveInt
        Positive byte length of its canonical JSON encoding.

    Notes
    -----
    Construction checks identity format. Matching the referenced bytes and replay
    requires sidecar validation; this record performs no file access.
    """

    schema_id: Literal["marl_battlegrounds.evaluation.metric_report_reference"] = (
        METRIC_REPORT_REFERENCE_SCHEMA_ID
    )
    schema_version: _SchemaVersionV1 = REPLAY_SCHEMA_VERSION
    report_artifact_id: _AsciiIdentifier
    episode_id: _AsciiIdentifier
    report_artifact_schema_id: Literal[
        "marl_battlegrounds.evaluation.metric_report_artifact"
    ] = METRIC_REPORT_ARTIFACT_SCHEMA_ID
    report_artifact_schema_version: _SchemaVersionV1 = REPLAY_SCHEMA_VERSION
    metric_report_id: _AsciiIdentifier
    metric_report_schema_version: _SchemaVersionV1 = REPLAY_SCHEMA_VERSION
    trajectory_content_digest_sha256: _Sha256Hex
    canonical_digest_sha256: _Sha256Hex
    canonical_byte_length: _PositiveInt

    @model_validator(mode="after")
    def _validate_identity(self) -> MetricReportReferenceV1:
        """Check canonical report and report-artifact IDs for this episode.

        Return the same reference or raise ValueError on either mismatch.
        """
        if self.report_artifact_id != f"{self.episode_id}:metric-report-artifact":
            raise ValueError("metric report reference artifact ID is not canonical")
        if self.metric_report_id != f"{self.episode_id}:metric-report":
            raise ValueError("metric report reference report ID is not canonical")
        return self


class ReplayArtifactReferenceV1(EvaluationModel):
    """Identify complete replay bytes independently of their storage location.

    Attributes
    ----------
    schema_id : Literal['marl_battlegrounds.evaluation.replay_artifact_reference']
        Fixed replay-artifact-reference schema identifier.
    schema_version : _SchemaVersionV1
        Exact integer 1; defaults to 1.
    artifact_id : _AsciiIdentifier
        Canonical episode ID followed by ":replay".
    episode_id : _AsciiIdentifier
        Shared episode identity.
    replay_schema_version : _SchemaVersionV1
        Exact integer 1; defaults to 1.
    context_digest_sha256 : _Sha256Hex
        Digest of the stored episode context.
    trajectory_content_digest_sha256 : _Sha256Hex
        Digest before adding the report link.
    canonical_digest_sha256 : _Sha256Hex
        Digest of the complete replay artifact.
    canonical_byte_length : _PositiveInt
        Positive byte length of canonical replay JSON.

    Notes
    -----
    A reference may describe a valid partial replay; "artifact" does not imply
    the simulated episode reached termination. No path or file is opened.
    """

    schema_id: Literal["marl_battlegrounds.evaluation.replay_artifact_reference"] = (
        REPLAY_ARTIFACT_REFERENCE_SCHEMA_ID
    )
    schema_version: _SchemaVersionV1 = REPLAY_SCHEMA_VERSION
    artifact_id: _AsciiIdentifier
    episode_id: _AsciiIdentifier
    replay_schema_version: _SchemaVersionV1 = REPLAY_SCHEMA_VERSION
    context_digest_sha256: _Sha256Hex
    trajectory_content_digest_sha256: _Sha256Hex
    canonical_digest_sha256: _Sha256Hex
    canonical_byte_length: _PositiveInt

    @model_validator(mode="after")
    def _validate_identity(self) -> ReplayArtifactReferenceV1:
        """Require the canonical episode-based replay ID or raise ValueError.

        Return the same validated reference.
        """
        if self.artifact_id != f"{self.episode_id}:replay":
            raise ValueError("replay artifact reference ID is not canonical")
        return self


class ReplayArtifactHeaderV1(EvaluationModel):
    """Describe a historical replay's records, schemas, and execution metadata.

    Attributes
    ----------
    schema_id : Literal['marl_battlegrounds.evaluation.replay_header']
        Fixed replay-header schema identifier.
    schema_version : _SchemaVersionV1
        Exact integer 1; defaults to 1.
    header_id : _AsciiIdentifier
        Canonical episode ID followed by ":replay-header".
    canonical_json_profile : Literal['marl_battlegrounds.canonical_json.v1']
        Fixed canonical JSON V1 profile identifier.
    source_schema_versions : tuple[SchemaVersionEntryV1, ...]
        Exact ordered context V1 schema bindings.
    envelope_schema_versions : tuple[SchemaVersionEntryV1, ...]
        Exact ordered V1 replay-envelope bindings.
    context : EvaluationEpisodeContextV1
        Exact V1 context, stored once for all frames and transitions.
    context_digest_sha256 : _Sha256Hex
        Digest of that complete context.
    expected_transition_count : _PositiveInt
        Positive context horizon in artifact transitions.
    recorded_transition_count : _NonNegativeInt
        Recorded prefix length, from zero to expected count.
    recorded_frame_count : _PositiveInt
        Recorded transition count plus one.
    first_frame_id : _AsciiIdentifier
        Canonical artifact frame-zero ID.
    last_frame_id : _AsciiIdentifier
        Canonical ID at recorded_transition_count.
    runtime_provenance : RuntimeProvenanceV1
        Exact execution metadata; package version must match context.
    wrapper_stack : tuple[ReplayWrapperMetadataV1, ...]
        Gap-free ordered wrapper metadata; defaults to an empty tuple.

    Notes
    -----
    Construction checks these joins and strict nested types. It does not yet
    validate every transition because the body is stored in ReplayArtifactV1.
    """

    schema_id: Literal["marl_battlegrounds.evaluation.replay_header"] = (
        REPLAY_HEADER_SCHEMA_ID
    )
    schema_version: _SchemaVersionV1 = REPLAY_SCHEMA_VERSION
    header_id: _AsciiIdentifier
    canonical_json_profile: Literal["marl_battlegrounds.canonical_json.v1"] = (
        CANONICAL_REPLAY_JSON_PROFILE_V1
    )
    source_schema_versions: tuple[SchemaVersionEntryV1, ...]
    envelope_schema_versions: tuple[SchemaVersionEntryV1, ...]
    context: EvaluationEpisodeContextV1
    context_digest_sha256: _Sha256Hex
    expected_transition_count: _PositiveInt
    recorded_transition_count: _NonNegativeInt
    recorded_frame_count: _PositiveInt
    first_frame_id: _AsciiIdentifier
    last_frame_id: _AsciiIdentifier
    runtime_provenance: RuntimeProvenanceV1
    wrapper_stack: tuple[ReplayWrapperMetadataV1, ...] = ()

    @model_validator(mode="after")
    def _validate_header(self) -> ReplayArtifactHeaderV1:
        """Check header schemas, context/runtime joins, counts, IDs, and wrapper order.

        Return this header or raise ValueError. Expected counts are artifact
        transitions,
        which need not start at simulator tick zero.
        """
        _require_exact_nested_model(
            self.context,
            expected_type=EvaluationEpisodeContextV1,
            record_name="replay header context",
        )
        _require_exact_nested_model(
            self.runtime_provenance,
            expected_type=RuntimeProvenanceV1,
            record_name="replay runtime provenance",
        )
        for wrapper in self.wrapper_stack:
            _require_exact_nested_model(
                wrapper,
                expected_type=ReplayWrapperMetadataV1,
                record_name="replay wrapper metadata",
            )
        episode_id = self.context.identity.episode_id
        if self.header_id != f"{episode_id}:replay-header":
            raise ValueError("replay header ID is not canonical")
        if _schema_bindings(self.source_schema_versions) != REQUIRED_SCHEMA_BINDINGS_V1:
            raise ValueError("replay source schemas must equal the eight CP2 roots")
        if (
            _schema_bindings(self.envelope_schema_versions)
            != REQUIRED_REPLAY_ENVELOPE_SCHEMA_BINDINGS_V1
        ):
            raise ValueError("replay envelope schemas must equal the V1 bindings")
        if self.source_schema_versions != self.context.schema_versions:
            raise ValueError("replay source schemas must equal context bindings")
        if self.context_digest_sha256 != canonical_digest_sha256(self.context):
            raise ValueError("replay context digest is not canonical")
        if (
            self.runtime_provenance.package_version
            != self.context.code_revision.package_version
        ):
            raise ValueError("runtime package version must equal context code revision")
        if self.expected_transition_count != self.context.expected_horizon:
            raise ValueError("replay expected count must equal context horizon")
        if self.recorded_transition_count > self.expected_transition_count:
            raise ValueError("replay transition count cannot exceed its horizon")
        if self.recorded_frame_count != self.recorded_transition_count + 1:
            raise ValueError("replay frame count must equal transition count plus one")
        if self.first_frame_id != f"{episode_id}:frame:0":
            raise ValueError("replay first frame ID is not canonical")
        if self.last_frame_id != (
            f"{episode_id}:frame:{self.recorded_transition_count}"
        ):
            raise ValueError("replay last frame ID is not canonical")
        positions = tuple(row.position for row in self.wrapper_stack)
        if positions != tuple(range(len(self.wrapper_stack))):
            raise ValueError("replay wrapper positions must be gap-free and ordered")
        return self


def _trajectory_content_payload(
    *,
    header: ReplayArtifactHeaderV1,
    completion: EvaluationEpisodeCompletionV1,
    processing_status: EvaluationProcessingStatusV1,
    frames: tuple[EvaluationFrameV1, ...],
    transitions: tuple[EvaluationTransitionV1, ...],
) -> dict[str, object]:
    """Build the non-circular digest payload from header and trajectory records.

    Include completion and processing status as supplied. Exclude report reference,
    artifact ID, and final artifact digest. Returned dictionary references immutable
    models and tuples; this helper does not serialize or validate them.
    """
    return {
        "header": header,
        "completion": completion,
        "processing_status": processing_status,
        "frames": frames,
        "transitions": transitions,
    }


class ReplayArtifactV1(EvaluationModel):
    """Store a content-addressed V1 replay and its path-free report reference.

    Attributes
    ----------
    schema_id : Literal['marl_battlegrounds.evaluation.replay_artifact']
        Fixed replay-artifact schema identifier.
    schema_version : _SchemaVersionV1
        Exact integer 1; defaults to 1.
    artifact_id : _AsciiIdentifier
        Canonical episode ID followed by ":replay".
    canonical_digest_sha256 : _Sha256Hex
        Digest of all other artifact fields.
    trajectory_content_digest_sha256 : _Sha256Hex
        Digest of the trajectory before its report link.
    header : ReplayArtifactHeaderV1
        Exact V1 header with context and execution metadata.
    completion : EvaluationEpisodeCompletionV1
        Episode completion or partial-prefix status.
    processing_status : EvaluationProcessingStatusV1
        Metric-processing progress, separate from trajectory validity.
    metric_report_reference : MetricReportReferenceV1
        Exact reference to the separately stored metric report.
    frames : tuple[EvaluationFrameV1, ...]
        T+1 ordered V1 frame records, including frame zero.
    transitions : tuple[EvaluationTransitionV1, ...]
        T ordered V1 transition records.

    Notes
    -----
    Construction validates the local envelope and digests. Call
    validate_replay_artifact_v1 for the O(T) adjacency, task, and event checks.
    A valid partial prefix is allowed; this model does not mean the episode finished.
    """

    schema_id: Literal["marl_battlegrounds.evaluation.replay_artifact"] = (
        REPLAY_ARTIFACT_SCHEMA_ID
    )
    schema_version: _SchemaVersionV1 = REPLAY_SCHEMA_VERSION
    artifact_id: _AsciiIdentifier
    canonical_digest_sha256: _Sha256Hex
    trajectory_content_digest_sha256: _Sha256Hex
    header: ReplayArtifactHeaderV1
    completion: EvaluationEpisodeCompletionV1
    processing_status: EvaluationProcessingStatusV1
    metric_report_reference: MetricReportReferenceV1
    frames: tuple[EvaluationFrameV1, ...]
    transitions: tuple[EvaluationTransitionV1, ...]

    @model_validator(mode="after")
    def _validate_local_envelope(self) -> ReplayArtifactV1:
        """Check strict envelope types, episode/report joins, counts, and both digests.

        Return this artifact or raise ValueError. Full transition semantics remain in
        the explicit replay validation pass.
        """
        _require_exact_nested_model(
            self.header,
            expected_type=ReplayArtifactHeaderV1,
            record_name="replay header",
        )
        _require_exact_nested_model(
            self.completion,
            expected_type=EvaluationEpisodeCompletionV1,
            record_name="replay completion",
        )
        _require_exact_nested_model(
            self.processing_status,
            expected_type=EvaluationProcessingStatusV1,
            record_name="replay processing status",
        )
        _require_exact_nested_model(
            self.metric_report_reference,
            expected_type=MetricReportReferenceV1,
            record_name="replay metric report reference",
        )
        episode_id = self.header.context.identity.episode_id
        if self.artifact_id != f"{episode_id}:replay":
            raise ValueError("replay artifact ID is not canonical")
        if self.completion.episode_id != episode_id:
            raise ValueError("replay completion must join the context episode")
        if self.metric_report_reference.episode_id != episode_id:
            raise ValueError("replay report reference must join the context episode")
        if (
            self.metric_report_reference.trajectory_content_digest_sha256
            != self.trajectory_content_digest_sha256
        ):
            raise ValueError("replay report reference must join trajectory content")
        if len(self.frames) != self.header.recorded_frame_count:
            raise ValueError("replay frame tuple must equal its recorded count")
        if len(self.transitions) != self.header.recorded_transition_count:
            raise ValueError("replay transition tuple must equal its recorded count")
        expected_content_digest = canonical_digest_sha256(
            _trajectory_content_payload(
                header=self.header,
                completion=self.completion,
                processing_status=self.processing_status,
                frames=self.frames,
                transitions=self.transitions,
            )
        )
        if self.trajectory_content_digest_sha256 != expected_content_digest:
            raise ValueError("replay trajectory-content digest is not canonical")
        if self.canonical_digest_sha256 != canonical_digest_sha256(
            self,
            exclude={"canonical_digest_sha256"},
        ):
            raise ValueError("replay artifact digest is not canonical")
        return self


@dataclass(frozen=True, slots=True)
class ReplayBundleV1:
    """Keep a V1 replay and the metric artifact referenced by that replay.

    Attributes
    ----------
    replay : ReplayArtifactV1
        Historical replay with its metric-report reference.
    metric_report_artifact : EvaluationMetricReportArtifactV1
        Matching report sidecar to persist separately.

    Notes
    -----
    Direct dataclass construction validates both members and their joins.
    The internal builder reuses an already validated observer trajectory.
    """

    replay: ReplayArtifactV1
    metric_report_artifact: EvaluationMetricReportArtifactV1

    def __post_init__(self) -> None:
        """Validate the replay, sidecar, and all cross-artifact references after
        construction.

        Raise ValueError on an invalid member or mismatched join.
        """
        validate_metric_report_artifact_against_replay_v1(
            self.metric_report_artifact,
            self.replay,
        )


def _schema_version_rows(
    bindings: tuple[tuple[str, int], ...],
) -> tuple[SchemaVersionEntryV1, ...]:
    """Construct strict V1 schema rows from ordered identifier/version bindings.

    Preserve input order and let model validation reject versions other than one.
    """
    return tuple(
        SchemaVersionEntryV1.model_validate(
            {"schema_id": schema_id, "schema_version": version}
        )
        for schema_id, version in bindings
    )


def _require_finalized_retained_observer(
    observer: EvaluationEpisodeObserverV1,
    report: EvaluationMetricReportV1,
) -> tuple[
    EvaluationEpisodeContextV1,
    tuple[EvaluationFrameV1, ...],
    tuple[EvaluationTransitionV1, ...],
    EvaluationMetricReportV1,
]:
    """Require a finalized V1 observer with retained history and its exact report.

    Return context, frame tuple, transition tuple, and the committed report after
    checking counts and progress. Wrong concrete types raise TypeError; lifecycle,
    profile, or report disagreement raises ValueError. Training-light history is
    insufficient for replay construction.
    """
    if type(observer) is not EvaluationEpisodeObserverV1:
        raise TypeError("replay builder requires EvaluationEpisodeObserverV1")
    if observer.lifecycle_state != "finalized":
        raise ValueError("replay builder requires a finalized observer")
    if type(report) is not EvaluationMetricReportV1:
        raise TypeError("replay builder requires EvaluationMetricReportV1")
    canonical_report = cast(
        EvaluationMetricReportV1,
        validate_declared_model_tree(
            report,
            record_name="replay metric report",
            expected_type=EvaluationMetricReportV1,
        ),
    )
    finalized_report = observer.finalized_report
    if finalized_report is None:
        raise ValueError("finalized observer is missing its committed report")
    if canonical_report != finalized_report:
        raise ValueError("replay builder report must equal the observer report")
    frames = observer.retained_frames
    transitions = observer.retained_transitions
    if frames is None or transitions is None:
        raise ValueError(
            "replay construction requires a metric-complete retaining profile"
        )
    context = observer.context
    if finalized_report.context != context:
        raise ValueError("replay report context must equal observer context")
    if canonical_report.completion.validated_transition_count != len(transitions):
        raise ValueError(
            "replay report validated count must equal retained transitions"
        )
    if canonical_report.processing_status.processed_transition_count != (
        observer.processed_transition_count
    ):
        raise ValueError("replay report processed count must equal observer progress")
    if observer.validated_transition_count != len(transitions):
        raise ValueError("observer validated count must equal retained transitions")
    if len(frames) != len(transitions) + 1:
        raise ValueError("retained replay history must have exactly T+1/T records")
    return (
        context,
        frames,
        transitions,
        finalized_report,
    )


def _build_replay_bundle_v1(
    observer: EvaluationEpisodeObserverV1,
    report: EvaluationMetricReportV1,
    *,
    runtime_provenance: RuntimeProvenanceV1,
    wrapper_stack: tuple[ReplayWrapperMetadataV1, ...],
) -> ReplayBundleV1:
    """Assemble both artifact envelopes from one already validated retained trajectory.

    Validate supplied runtime/wrappers, derive non-circular trajectory and sidecar
    identities, then build final replay bytes' digest. Reuse detached observer
    records without revalidating every transition. No files are written and the
    observer is not changed.
    """
    context, frames, transitions, canonical_report = (
        _require_finalized_retained_observer(
            observer,
            report,
        )
    )
    if type(runtime_provenance) is not RuntimeProvenanceV1:
        raise TypeError("runtime provenance must use RuntimeProvenanceV1")
    canonical_runtime = cast(
        RuntimeProvenanceV1,
        validate_declared_model_tree(
            runtime_provenance,
            record_name="replay runtime provenance",
            expected_type=RuntimeProvenanceV1,
        ),
    )
    if type(wrapper_stack) is not tuple:
        raise TypeError("replay wrapper stack must be an immutable tuple")
    canonical_wrappers = tuple(
        cast(
            ReplayWrapperMetadataV1,
            validate_declared_model_tree(
                wrapper,
                record_name="replay wrapper metadata",
                expected_type=ReplayWrapperMetadataV1,
            ),
        )
        for wrapper in wrapper_stack
    )
    episode_id = context.identity.episode_id
    header = ReplayArtifactHeaderV1(
        header_id=f"{episode_id}:replay-header",
        source_schema_versions=context.schema_versions,
        envelope_schema_versions=_schema_version_rows(
            REQUIRED_REPLAY_ENVELOPE_SCHEMA_BINDINGS_V1
        ),
        context=context,
        context_digest_sha256=canonical_digest_sha256(context),
        expected_transition_count=context.expected_horizon,
        recorded_transition_count=len(transitions),
        recorded_frame_count=len(frames),
        first_frame_id=frames[0].frame_id,
        last_frame_id=frames[-1].frame_id,
        runtime_provenance=canonical_runtime,
        wrapper_stack=canonical_wrappers,
    )
    trajectory_content_digest = canonical_digest_sha256(
        _trajectory_content_payload(
            header=header,
            completion=canonical_report.completion,
            processing_status=canonical_report.processing_status,
            frames=frames,
            transitions=transitions,
        )
    )
    source_trajectory = ReplayTrajectoryContentReferenceV1(
        replay_artifact_id=f"{episode_id}:replay",
        episode_id=episode_id,
        context_digest_sha256=header.context_digest_sha256,
        trajectory_content_digest_sha256=trajectory_content_digest,
    )
    report_artifact_payload: dict[str, object] = {
        "schema_id": METRIC_REPORT_ARTIFACT_SCHEMA_ID,
        "schema_version": REPLAY_SCHEMA_VERSION,
        "report_artifact_id": f"{episode_id}:metric-report-artifact",
        "source_trajectory": source_trajectory,
        "report": canonical_report,
    }
    report_artifact = EvaluationMetricReportArtifactV1.model_construct(
        _fields_set=None,
        **report_artifact_payload,
        canonical_digest_sha256=canonical_digest_sha256(report_artifact_payload),
    )
    metric_reference = MetricReportReferenceV1(
        report_artifact_id=report_artifact.report_artifact_id,
        episode_id=episode_id,
        metric_report_id=canonical_report.report_id,
        trajectory_content_digest_sha256=trajectory_content_digest,
        canonical_digest_sha256=report_artifact.canonical_digest_sha256,
        canonical_byte_length=len(canonical_json_bytes(report_artifact)),
    )
    replay_payload: dict[str, object] = {
        "schema_id": REPLAY_ARTIFACT_SCHEMA_ID,
        "schema_version": REPLAY_SCHEMA_VERSION,
        "artifact_id": f"{episode_id}:replay",
        "trajectory_content_digest_sha256": trajectory_content_digest,
        "header": header,
        "completion": canonical_report.completion,
        "processing_status": canonical_report.processing_status,
        "metric_report_reference": metric_reference,
        "frames": frames,
        "transitions": transitions,
    }
    replay = ReplayArtifactV1.model_construct(
        _fields_set=None,
        **replay_payload,
        canonical_digest_sha256=canonical_digest_sha256(replay_payload),
    )
    # The observer owns the validated trajectory and report; these envelopes
    # contain only those detached records and the identities generated above.
    # Public constructors and import validators still check arbitrary records.
    bundle = object.__new__(ReplayBundleV1)
    object.__setattr__(bundle, "replay", replay)
    object.__setattr__(bundle, "metric_report_artifact", report_artifact)
    return bundle


def build_replay_bundle_v1(
    observer: EvaluationEpisodeObserverV1,
    report: EvaluationMetricReportV1,
    *,
    runtime_provenance: RuntimeProvenanceV1,
    wrapper_stack: tuple[ReplayWrapperMetadataV1, ...] = (),
) -> ReplayBundleV1:
    """Build a historical replay and its report sidecar.

    Parameters
    ----------
    observer : EvaluationEpisodeObserverV1
        Exact finalized EvaluationEpisodeObserverV1 with a metric-complete
        retaining profile and T+1/T validated records.
    report : EvaluationMetricReportV1
        Exact V1 report equal to the observer's committed finalized report.
    runtime_provenance : RuntimeProvenanceV1
        Explicit RuntimeProvenanceV1 matching context package version.
    wrapper_stack : tuple[ReplayWrapperMetadataV1, ...]
        Immutable ordered wrapper tuple, positions zero onward.
        Defaults to an empty tuple.

    Returns
    -------
    ReplayBundleV1
        ReplayBundleV1 containing the replay and separately persistable metric artifact.

    Raises
    ------
    TypeError
        Observer/report/runtime types are wrong or wrapper_stack is not a tuple.
    ValueError
        Finalization, retained history, report identity, metadata, or counts disagree.

    Notes
    -----
    Host-only construction performs no file I/O and does not mutate the observer.
    Persistence should publish the sidecar before the replay that references it.
    """
    return _build_replay_bundle_v1(
        observer,
        report,
        runtime_provenance=runtime_provenance,
        wrapper_stack=wrapper_stack,
    )


def build_replay_artifact_v1(
    observer: EvaluationEpisodeObserverV1,
    report: EvaluationMetricReportV1,
    *,
    runtime_provenance: RuntimeProvenanceV1,
    wrapper_stack: tuple[ReplayWrapperMetadataV1, ...] = (),
) -> ReplayArtifactV1:
    """Build the replay member of a historical recording bundle.

    Parameters
    ----------
    observer : EvaluationEpisodeObserverV1
        Exact finalized EvaluationEpisodeObserverV1 with a metric-complete
        retaining profile and T+1/T validated records.
    report : EvaluationMetricReportV1
        Exact V1 report equal to the observer's committed finalized report.
    runtime_provenance : RuntimeProvenanceV1
        Explicit RuntimeProvenanceV1 matching context package version.
    wrapper_stack : tuple[ReplayWrapperMetadataV1, ...]
        Immutable ordered wrapper tuple, positions zero onward.
        Defaults to an empty tuple.

    Returns
    -------
    ReplayArtifactV1
        ReplayArtifactV1 containing a report reference but not the sidecar bytes.

    Raises
    ------
    TypeError
        Observer/report/runtime types are wrong or wrapper_stack is not a tuple.
    ValueError
        Finalization, retained history, report identity, metadata, or counts disagree.

    Notes
    -----
    Host-only construction performs no file I/O and does not mutate the observer.
    Persistence callers should use build_replay_bundle_v1 so they can publish the
    referenced sidecar too.
    """
    return _build_replay_bundle_v1(
        observer,
        report,
        runtime_provenance=runtime_provenance,
        wrapper_stack=wrapper_stack,
    ).replay


def build_replay_artifact_reference_v1(
    artifact: ReplayArtifactV1,
) -> ReplayArtifactReferenceV1:
    """Validate a historical replay and return its path-free content reference.

    Parameters
    ----------
    artifact : ReplayArtifactV1
        Exact ReplayArtifactV1 to validate, including every transition.

    Returns
    -------
    ReplayArtifactReferenceV1
        Reference containing episode/context/trajectory/artifact digests and the
        canonical JSON byte length. Partial but valid artifacts are supported.

    Raises
    ------
    ValueError
        Strict types, digests, trajectory semantics, or completion joins fail.

    Notes
    -----
    Performs an O(T) host validation pass and canonical serialization for byte
    length. It does not read or write a file.
    """
    validate_replay_artifact_v1(artifact)
    return _build_replay_artifact_reference_from_validated_v1(artifact)


def _build_replay_artifact_reference_from_validated_v1(
    artifact: ReplayArtifactV1,
) -> ReplayArtifactReferenceV1:
    """Build a path-free reference after the caller has validated the replay.

    Compute canonical JSON byte length and reuse its recorded digests. This helper
    does not repeat the full O(T) semantic pass.
    """
    context = artifact.header.context
    return ReplayArtifactReferenceV1(
        artifact_id=artifact.artifact_id,
        episode_id=context.identity.episode_id,
        context_digest_sha256=artifact.header.context_digest_sha256,
        trajectory_content_digest_sha256=(artifact.trajectory_content_digest_sha256),
        canonical_digest_sha256=artifact.canonical_digest_sha256,
        canonical_byte_length=len(canonical_json_bytes(artifact)),
    )


def _validate_replay_semantics(artifact: ReplayArtifactV1) -> None:
    """Revalidate a V1 replay's full trajectory and completion/progress joins.

    Check exact model trees, gap-free indices, initial frame, every adjacent
    transition view, no continuation after done, and matching tail flags/counts.
    Raise ValueError on disagreement; work grows with recorded transition count.
    """
    canonical_artifact = cast(
        ReplayArtifactV1,
        validate_declared_model_tree(
            artifact,
            record_name="replay artifact",
            expected_type=ReplayArtifactV1,
        ),
    )
    header = canonical_artifact.header
    context = header.context
    frames = canonical_artifact.frames
    transitions = canonical_artifact.transitions
    if not frames:
        raise ValueError("replay must contain its initial frame")
    if tuple(frame.frame_index for frame in frames) != tuple(range(len(frames))):
        raise ValueError("replay frame positions must equal artifact frame indices")
    if tuple(transition.transition_index for transition in transitions) != tuple(
        range(len(transitions))
    ):
        raise ValueError(
            "replay transition positions must equal artifact transition indices"
        )
    if frames[0].frame_id != header.first_frame_id:
        raise ValueError("replay first frame must equal its header reference")
    if frames[-1].frame_id != header.last_frame_id:
        raise ValueError("replay last frame must equal its header reference")
    validate_initial_evaluation_frame_v1(context, frames[0])
    for transition_index, transition in enumerate(transitions):
        if transition_index > 0:
            previous = transitions[transition_index - 1]
            if previous.terminated or previous.truncated:
                raise ValueError("replay cannot continue after a done transition")
        EvaluationTransitionViewV1(
            context=context,
            start_frame=frames[transition_index],
            transition=transition,
            successor_frame=frames[transition_index + 1],
        )
    completion = canonical_artifact.completion
    transition_count = len(transitions)
    if completion.expected_transition_count != context.expected_horizon:
        raise ValueError("replay completion must use the context horizon")
    if completion.validated_transition_count != transition_count:
        raise ValueError("replay completion count must equal recorded transitions")
    if completion.last_valid_frame_index != frames[-1].frame_index:
        raise ValueError("replay completion must name the final frame index")
    if completion.last_valid_frame_id != frames[-1].frame_id:
        raise ValueError("replay completion must name the final frame ID")
    final_transition = transitions[-1] if transitions else None
    terminated = False if final_transition is None else final_transition.terminated
    truncated = False if final_transition is None else final_transition.truncated
    if completion.terminated != terminated or completion.truncated != truncated:
        raise ValueError("replay completion done flags must equal the transition tail")
    if (
        final_transition is not None
        and final_transition.owning_task_end_reason is not None
        and completion.end_or_failure_reason != final_transition.owning_task_end_reason
    ):
        raise ValueError("replay completion reason must equal authoritative tail truth")
    validate_evaluation_processing_progress_v1(
        transition_count,
        canonical_artifact.processing_status,
    )


def validate_replay_artifact_v1(artifact: ReplayArtifactV1) -> None:
    """Check the full semantics of one historical replay.

    Parameters
    ----------
    artifact : ReplayArtifactV1
        Exact V1 replay model, including its context, frames, and transitions.

    Returns
    -------
    None
        None.

    Raises
    ------
    ValueError
        Strict types, hashes, indices, task/event facts, completion, or
        processing progress is inconsistent.

    Notes
    -----
    Runs an explicit O(T) host pass without simulation or file access. A valid
    report reference is checked locally; validating actual sidecar bytes needs
    validate_metric_report_artifact_against_replay_v1.
    """
    _validate_replay_semantics(artifact)


def iter_replay_transition_views_v1(
    artifact: ReplayArtifactV1,
) -> Iterator[EvaluationTransitionViewV1]:
    """Validate a historical replay and iterate its joined transition views.

    Parameters
    ----------
    artifact : ReplayArtifactV1
        Exact V1 replay to validate before producing views.

    Yields
    ------
    EvaluationTransitionViewV1
        Context/start-frame/transition/successor views in increasing transition index.
        An initial-frame-only replay yields no views.

    Raises
    ------
    ValueError
        Replay validation or construction of a joined transition view fails.

    Notes
    -----
    The full O(T) replay validation occurs when this function is called.
    Individual strict views are then constructed lazily as the iterator advances.
    No simulator execution or file access occurs.
    """
    _validate_replay_semantics(artifact)
    context = artifact.header.context
    return (
        EvaluationTransitionViewV1(
            context=context,
            start_frame=artifact.frames[transition_index],
            transition=transition,
            successor_frame=artifact.frames[transition_index + 1],
        )
        for transition_index, transition in enumerate(artifact.transitions)
    )


def validate_metric_report_artifact_against_replay_v1(
    report_artifact: EvaluationMetricReportArtifactV1,
    replay: ReplayArtifactV1,
) -> None:
    """Check a historical metric sidecar and every join to its replay.

    Parameters
    ----------
    report_artifact : EvaluationMetricReportArtifactV1
        Exact V1 report artifact with its source-trajectory reference.
    replay : ReplayArtifactV1
        Exact V1 replay expected to own that report.

    Returns
    -------
    None
        None.

    Raises
    ------
    ValueError
        Either artifact is invalid, or context, trajectory, completion,
        progress, identity, digest, or byte-length references disagree.

    Notes
    -----
    Performs the replay's O(T) host semantic pass before sidecar comparison.
    It checks supplied models and does not open files.
    """
    validate_replay_artifact_v1(replay)
    _validate_metric_report_artifact_against_validated_replay_v1(
        report_artifact,
        replay,
    )


def _validate_metric_report_artifact_against_validated_replay_v1(
    report_artifact: EvaluationMetricReportArtifactV1,
    replay: ReplayArtifactV1,
) -> None:
    """Check a strict report sidecar against a replay already semantically validated.

    Require matching trajectory identity, byte-equal context, completion, processing
    status, report digest, and canonical byte length. Raise ValueError on mismatch
    without rerunning the replay's transition checks.
    """
    canonical_report_artifact = cast(
        EvaluationMetricReportArtifactV1,
        validate_declared_model_tree(
            report_artifact,
            record_name="metric report artifact",
            expected_type=EvaluationMetricReportArtifactV1,
        ),
    )
    report = canonical_report_artifact.report
    context = replay.header.context
    if canonical_report_artifact.source_trajectory != (
        ReplayTrajectoryContentReferenceV1(
            replay_artifact_id=replay.artifact_id,
            episode_id=context.identity.episode_id,
            context_digest_sha256=replay.header.context_digest_sha256,
            trajectory_content_digest_sha256=(replay.trajectory_content_digest_sha256),
        )
    ):
        raise ValueError("metric report source trajectory does not match replay")
    if canonical_json_bytes(report.context) != canonical_json_bytes(context):
        raise ValueError("metric report context does not match replay context")
    if report.completion != replay.completion:
        raise ValueError("metric report completion does not match replay completion")
    if report.processing_status != replay.processing_status:
        raise ValueError("metric report processing does not match replay processing")
    expected_reference = MetricReportReferenceV1(
        report_artifact_id=canonical_report_artifact.report_artifact_id,
        episode_id=context.identity.episode_id,
        metric_report_id=report.report_id,
        trajectory_content_digest_sha256=replay.trajectory_content_digest_sha256,
        canonical_digest_sha256=(canonical_report_artifact.canonical_digest_sha256),
        canonical_byte_length=len(canonical_json_bytes(canonical_report_artifact)),
    )
    if replay.metric_report_reference != expected_reference:
        raise ValueError("replay metric report reference does not match sidecar")


__all__ = [
    "CANONICAL_REPLAY_JSON_PROFILE_V1",
    "REPLAY_ARTIFACT_REFERENCE_SCHEMA_ID",
    "REPLAY_ARTIFACT_SCHEMA_ID",
    "REPLAY_HEADER_SCHEMA_ID",
    "REPLAY_SCHEMA_VERSION",
    "REQUIRED_REPLAY_ENVELOPE_SCHEMA_BINDINGS_V1",
    "EvaluationMetricReportArtifactV1",
    "MetricReportReferenceV1",
    "ReplayArtifactHeaderV1",
    "ReplayArtifactReferenceV1",
    "ReplayArtifactV1",
    "ReplayBundleV1",
    "ReplayTrajectoryContentReferenceV1",
    "ReplayWrapperMetadataV1",
    "RuntimeProvenanceV1",
    "build_replay_artifact_reference_v1",
    "build_replay_artifact_v1",
    "build_replay_bundle_v1",
    "iter_replay_transition_views_v1",
    "validate_metric_report_artifact_against_replay_v1",
    "validate_replay_artifact_v1",
]
