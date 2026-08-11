"""Strict in-memory replay contracts over accepted CP2/CP3 records.

The standard replay is a host-only semantic artifact.  It stores the accepted
episode context once and the exact ``T + 1`` frame / ``T`` transition normal
form.  It never stores ``EnvState``, renderer frames, policy internals, local
paths, or simulator-owned reconstruction logic.
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
    return tuple((row.schema_id, row.schema_version) for row in rows)


def _require_exact_nested_model(
    model: EvaluationModel,
    *,
    expected_type: type[EvaluationModel],
    record_name: str,
) -> None:
    validate_declared_model_tree(
        model,
        record_name=record_name,
        expected_type=expected_type,
    )


class RuntimeProvenanceV1(EvaluationModel):
    """Versioned runtime facts captured once for one replay."""

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
        if prod(self.batch_shape) != self.environment_count:
            raise ValueError("runtime batch shape product must equal environment count")
        return self


class ReplayWrapperMetadataV1(EvaluationModel):
    """One ordered wrapper or adapter in the recorded host execution stack."""

    schema_id: Literal["marl_battlegrounds.evaluation.replay_wrapper_metadata"] = (
        REPLAY_WRAPPER_METADATA_SCHEMA_ID
    )
    schema_version: _SchemaVersionV1 = REPLAY_SCHEMA_VERSION
    position: _NonNegativeInt
    wrapper_id: _AsciiIdentifier
    wrapper_version: _PositiveInt
    configuration_digest_sha256: _Sha256Hex


class ReplayTrajectoryContentReferenceV1(EvaluationModel):
    """Non-circular reference to replay content before its report link exists."""

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
        if self.replay_artifact_id != f"{self.episode_id}:replay":
            raise ValueError("trajectory reference replay artifact ID is not canonical")
        return self


class EvaluationMetricReportArtifactV1(EvaluationModel):
    """Content-addressed metric-report sidecar bound to trajectory content."""

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
    """Path-free durable reference to one metric-report artifact."""

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
        if self.report_artifact_id != f"{self.episode_id}:metric-report-artifact":
            raise ValueError("metric report reference artifact ID is not canonical")
        if self.metric_report_id != f"{self.episode_id}:metric-report":
            raise ValueError("metric report reference report ID is not canonical")
        return self


class ReplayArtifactReferenceV1(EvaluationModel):
    """Path-free content-addressed reference to a completed replay artifact."""

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
        if self.artifact_id != f"{self.episode_id}:replay":
            raise ValueError("replay artifact reference ID is not canonical")
        return self


class ReplayArtifactHeaderV1(EvaluationModel):
    """One self-describing header for a semantic replay normal form."""

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
    return {
        "header": header,
        "completion": completion,
        "processing_status": processing_status,
        "frames": frames,
        "transitions": transitions,
    }


class ReplayArtifactV1(EvaluationModel):
    """Content-addressed canonical semantic replay with exactly T+1/T records."""

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
    """In-memory replay plus its independently persisted metric sidecar."""

    replay: ReplayArtifactV1
    metric_report_artifact: EvaluationMetricReportArtifactV1

    def __post_init__(self) -> None:
        validate_metric_report_artifact_against_replay_v1(
            self.metric_report_artifact,
            self.replay,
        )


def _schema_version_rows(
    bindings: tuple[tuple[str, int], ...],
) -> tuple[SchemaVersionEntryV1, ...]:
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
    if canonical_json_bytes(canonical_report) != canonical_json_bytes(finalized_report):
        raise ValueError("replay builder report must equal the observer report")
    frames = observer.retained_frames
    transitions = observer.retained_transitions
    if frames is None or transitions is None:
        raise ValueError(
            "replay construction requires a metric-complete retaining profile"
        )
    context = observer.context
    if canonical_json_bytes(canonical_report.context) != canonical_json_bytes(context):
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
    return context, frames, transitions, canonical_report


def _build_replay_bundle_v1(
    observer: EvaluationEpisodeObserverV1,
    report: EvaluationMetricReportV1,
    *,
    runtime_provenance: RuntimeProvenanceV1,
    wrapper_stack: tuple[ReplayWrapperMetadataV1, ...],
) -> ReplayBundleV1:
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
    report_artifact = EvaluationMetricReportArtifactV1.model_validate(
        {
            **report_artifact_payload,
            "canonical_digest_sha256": canonical_digest_sha256(report_artifact_payload),
        }
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
    replay = ReplayArtifactV1.model_validate(
        {
            **replay_payload,
            "canonical_digest_sha256": canonical_digest_sha256(replay_payload),
        }
    )
    return ReplayBundleV1(
        replay=replay,
        metric_report_artifact=report_artifact,
    )


def build_replay_bundle_v1(
    observer: EvaluationEpisodeObserverV1,
    report: EvaluationMetricReportV1,
    *,
    runtime_provenance: RuntimeProvenanceV1,
    wrapper_stack: tuple[ReplayWrapperMetadataV1, ...] = (),
) -> ReplayBundleV1:
    """Build one replay and its report sidecar from a finalized observer."""
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
    """Build only the replay member of a deterministic in-memory bundle.

    This convenience result contains a report reference but not the referenced
    sidecar bytes.  Persistence callers must use :func:`build_replay_bundle_v1`
    so the report can be published before the replay.
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
    """Return a path-free content reference for a validated replay artifact."""
    validate_replay_artifact_v1(artifact)
    return _build_replay_artifact_reference_from_validated_v1(artifact)


def _build_replay_artifact_reference_from_validated_v1(
    artifact: ReplayArtifactV1,
) -> ReplayArtifactReferenceV1:
    """Build a replay reference after the caller's semantic validation pass."""
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
    """Run the explicit O(T) semantic validation pass for one replay."""
    _validate_replay_semantics(artifact)


def iter_replay_transition_views_v1(
    artifact: ReplayArtifactV1,
) -> Iterator[EvaluationTransitionViewV1]:
    """Yield gap-free canonical CP3 views from one validated replay."""
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
    """Validate a metric sidecar and all of its joins to one replay."""
    validate_replay_artifact_v1(replay)
    _validate_metric_report_artifact_against_validated_replay_v1(
        report_artifact,
        replay,
    )


def _validate_metric_report_artifact_against_validated_replay_v1(
    report_artifact: EvaluationMetricReportArtifactV1,
    replay: ReplayArtifactV1,
) -> None:
    """Validate report joins after the caller's semantic replay validation."""
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
