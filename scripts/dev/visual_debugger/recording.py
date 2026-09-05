"""Path-independent debugger recording and replay publication lifecycle.

The recorder owns exactly one retaining CP3 observer and immutable prepared
replay bytes.  It deliberately owns no service lock, browser state, simulator
state, or command protocol: callers serialize access externally and commit the
recorder only after their candidate response has passed validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import StringConstraints, model_validator

from marl_battlegrounds.evaluation.metrics import (
    EvaluationEpisodeObserverV1,
    EvaluationMetricReducerV1,
    ObserverLifecycleState,
    RolloutFailureOrigin,
)
from marl_battlegrounds.evaluation.models import (
    AssignedPolicySlotV1,
    EvaluationEpisodeContextV1,
    EvaluationFrameV1,
    EvaluationModel,
    EvaluationTransitionV1,
    canonical_digest_sha256,
)
from marl_battlegrounds.evaluation.replay import (
    ReplayBundleV1,
    ReplayWrapperMetadataV1,
    RuntimeProvenanceV1,
    build_replay_bundle_v1,
)
from marl_battlegrounds.evaluation.replay_io import (
    REPLAY_FILE_SUFFIX_V1,
    LoadedReplayBundleV1,
    PreparedReplayBundleV1,
    ReplayBundleDestinationV1,
    ReplayIOErrorCodeV1,
    ReplayLoadError,
    ReplaySaveError,
    SavedReplayBundleV1,
    load_replay_bundle_v1,
    preflight_replay_bundle_destination_v1,
    prepare_replay_bundle_v1,
    publish_prepared_replay_bundle_v1,
)
from marl_battlegrounds.evaluation.validation import validate_declared_model_tree
from scripts.dev.visual_debugger.evaluation_bridge import (
    DebuggerActionSourceKindV1,
)
from scripts.dev.visual_debugger.protocol import (
    RecordingLifecycleV1,
    RecordingPersistenceErrorCodeV1,
    RecordingStatusV1,
)

DEBUGGER_RECORDING_SCHEMA_VERSION: Literal[1] = 1
DEBUGGER_RECORDING_SPECIFICATION_SCHEMA_ID = (
    "marl_battlegrounds.visual_debugger.recording_specification"
)

type DebuggerRecordingCloseCauseV1 = Literal[
    "endpoint",
    "finish_and_review",
    "user_exit",
    "keyboard_interrupt",
    "truncation",
    "processing_failure",
    "simulation_failure",
    "policy_failure",
    "capture_failure",
    "validation_failure",
]
type DebuggerRecordingPublicationOutcomeV1 = Literal[
    "saved",
    "persistence_failed",
]

_Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_FAILURE_ORIGIN_BY_CAUSE: dict[
    DebuggerRecordingCloseCauseV1,
    RolloutFailureOrigin,
] = {
    "simulation_failure": "simulation",
    "policy_failure": "policy",
    "capture_failure": "capture",
    "validation_failure": "validation",
}
_DEFAULT_REASON_BY_CAUSE: dict[DebuggerRecordingCloseCauseV1, str] = {
    "finish_and_review": "user_finish_and_review",
    "user_exit": "user_exit",
    "keyboard_interrupt": "keyboard_interrupt",
    "truncation": "environment_truncated",
    "processing_failure": "evaluation_processing_failure",
    "simulation_failure": "simulation_failure",
    "policy_failure": "policy_failure",
    "capture_failure": "capture_failure",
    "validation_failure": "validation_failure",
}
_TARGET_ERROR_CODES: frozenset[ReplayIOErrorCodeV1] = frozenset(
    {
        "invalid_argument",
        "unsupported_platform",
        "invalid_filename",
        "missing_parent",
        "path_not_found",
        "path_is_symlink",
        "path_not_regular_file",
        "path_not_directory",
        "replay_target_exists",
        "metric_report_conflict",
        "companion_target_exists",
    }
)


class _RecordingMaterializationError(RuntimeError):
    """Expected replay build/preparation failure after observer finalization."""


class DebuggerRecordingSpecificationV1(EvaluationModel):
    """Immutable scientific recording inputs, excluding local destinations."""

    schema_id: Literal["marl_battlegrounds.visual_debugger.recording_specification"] = (
        DEBUGGER_RECORDING_SPECIFICATION_SCHEMA_ID
    )
    schema_version: Literal[1] = DEBUGGER_RECORDING_SCHEMA_VERSION
    specification_id: Annotated[
        str,
        StringConstraints(pattern=r"^debugger-recording:[0-9a-f]{64}$"),
    ]
    recording_content_digest_sha256: _Sha256Hex
    canonical_digest_sha256: _Sha256Hex
    action_source_kind: DebuggerActionSourceKindV1
    capture_profile: Literal["evaluation_metric_complete"] = (
        "evaluation_metric_complete"
    )
    runtime_provenance: RuntimeProvenanceV1
    wrapper_stack: tuple[ReplayWrapperMetadataV1, ...] = ()

    @model_validator(mode="after")
    def _validate_recording_specification(
        self,
    ) -> DebuggerRecordingSpecificationV1:
        validate_declared_model_tree(
            self.runtime_provenance,
            record_name="debugger recording runtime provenance",
            expected_type=RuntimeProvenanceV1,
        )
        if tuple(row.position for row in self.wrapper_stack) != tuple(
            range(len(self.wrapper_stack))
        ):
            raise ValueError("recording wrapper positions must be gap-free and ordered")
        for row in self.wrapper_stack:
            validate_declared_model_tree(
                row,
                record_name="debugger recording wrapper metadata",
                expected_type=ReplayWrapperMetadataV1,
            )
        content_payload = _recording_content_payload(
            action_source_kind=self.action_source_kind,
            runtime_provenance=self.runtime_provenance,
            wrapper_stack=self.wrapper_stack,
        )
        content_digest = canonical_digest_sha256(content_payload)
        if self.recording_content_digest_sha256 != content_digest:
            raise ValueError("debugger recording content digest mismatch")
        if self.specification_id != f"debugger-recording:{content_digest}":
            raise ValueError("debugger recording specification ID is not canonical")
        if self.canonical_digest_sha256 != canonical_digest_sha256(
            self,
            exclude={"canonical_digest_sha256"},
        ):
            raise ValueError("debugger recording specification digest mismatch")
        return self


def _recording_content_payload(
    *,
    action_source_kind: DebuggerActionSourceKindV1,
    runtime_provenance: RuntimeProvenanceV1,
    wrapper_stack: tuple[ReplayWrapperMetadataV1, ...],
) -> dict[str, object]:
    return {
        "schema_id": DEBUGGER_RECORDING_SPECIFICATION_SCHEMA_ID,
        "schema_version": DEBUGGER_RECORDING_SCHEMA_VERSION,
        "action_source_kind": action_source_kind,
        "capture_profile": "evaluation_metric_complete",
        "runtime_provenance": runtime_provenance,
        "wrapper_stack": wrapper_stack,
    }


def build_debugger_recording_specification_v1(
    *,
    action_source_kind: DebuggerActionSourceKindV1,
    runtime_provenance: RuntimeProvenanceV1,
    wrapper_stack: tuple[ReplayWrapperMetadataV1, ...] = (),
) -> DebuggerRecordingSpecificationV1:
    """Build one content-addressed recording specification without paths."""
    if type(wrapper_stack) is not tuple:
        raise TypeError("wrapper_stack must be an immutable tuple")
    content_payload = _recording_content_payload(
        action_source_kind=action_source_kind,
        runtime_provenance=runtime_provenance,
        wrapper_stack=wrapper_stack,
    )
    content_digest = canonical_digest_sha256(content_payload)
    payload: dict[str, object] = {
        **content_payload,
        "specification_id": f"debugger-recording:{content_digest}",
        "recording_content_digest_sha256": content_digest,
    }
    payload["canonical_digest_sha256"] = canonical_digest_sha256(payload)
    return DebuggerRecordingSpecificationV1.model_validate(payload)


def _context_action_source(context: EvaluationEpisodeContextV1) -> str:
    rows = tuple(
        row.value for row in context.aggregation_keys if row.name == "action_source"
    )
    if len(rows) != 1:
        raise ValueError("recording context requires exactly one action_source key")
    return rows[0]


def _context_policy_execution_included(context: EvaluationEpisodeContextV1) -> bool:
    """Derive actual policy execution from exact per-slot assignments."""
    return any(
        isinstance(row, AssignedPolicySlotV1)
        and row.policy_kind in ("scripted_tdm", "random_valid", "scenario_1")
        for row in context.policy_assignments
    )


def _persistence_error_code(
    error: ReplaySaveError | ReplayLoadError,
) -> RecordingPersistenceErrorCodeV1:
    if error.code in _TARGET_ERROR_CODES:
        return "target_unavailable"
    if isinstance(error, ReplayLoadError) or (
        error.code == "replay_publication_verification_failed"
    ):
        return "verification_failed"
    return "publication_failed"


def _completion_for_cause(
    cause: DebuggerRecordingCloseCauseV1,
    *,
    failure_reason: str | None,
) -> tuple[
    Literal["complete", "partial", "interrupted", "failed"],
    str | None,
    RolloutFailureOrigin | None,
]:
    if cause == "endpoint":
        if failure_reason is not None:
            raise ValueError("endpoint closeout forbids a failure reason")
        return "complete", None, None
    if cause in _FAILURE_ORIGIN_BY_CAUSE:
        reason = (
            _DEFAULT_REASON_BY_CAUSE[cause]
            if failure_reason is None
            else failure_reason
        )
        if (
            type(reason) is not str
            or not 1 <= len(reason) <= 256
            or any(ord(character) < 32 or ord(character) > 126 for character in reason)
        ):
            raise ValueError(
                "failure reason must contain 1..256 printable ASCII characters"
            )
        return "failed", reason, _FAILURE_ORIGIN_BY_CAUSE[cause]
    if failure_reason is not None:
        raise ValueError(
            "nonfailure closeout reasons are fixed by the recorder contract"
        )
    if cause == "finish_and_review":
        return "partial", _DEFAULT_REASON_BY_CAUSE[cause], None
    return "interrupted", _DEFAULT_REASON_BY_CAUSE[cause], None


class DebuggerReplayRecorderV1:
    """Lock-external owner of one retaining observer and cached replay bytes."""

    __slots__ = (
        "_bundle",
        "_close_cause",
        "_close_reason",
        "_context",
        "_current_destination",
        "_current_frame",
        "_last_io_error_code",
        "_lifecycle",
        "_observer",
        "_original_destination",
        "_persistence_error_code",
        "_prepared_bundle",
        "_publication_outcome",
        "_reducers",
        "_saved_bundle",
        "_specification",
        "_verified_loaded_bundle",
        "_verify_existing_on_retry",
    )

    def __init__(
        self,
        *,
        specification: DebuggerRecordingSpecificationV1,
        destination: ReplayBundleDestinationV1,
        context: EvaluationEpisodeContextV1,
        initial_frame: EvaluationFrameV1,
        reducers: tuple[EvaluationMetricReducerV1, ...] = (),
    ) -> None:
        if type(specification) is not DebuggerRecordingSpecificationV1:
            raise TypeError(
                "specification must be exact DebuggerRecordingSpecificationV1"
            )
        if type(destination) is not ReplayBundleDestinationV1:
            raise TypeError("destination must be exact ReplayBundleDestinationV1")
        if type(context) is not EvaluationEpisodeContextV1:
            raise TypeError("context must be exact EvaluationEpisodeContextV1")
        if type(initial_frame) is not EvaluationFrameV1:
            raise TypeError("initial_frame must be exact EvaluationFrameV1")
        if type(reducers) is not tuple:
            raise TypeError("reducers must be an immutable tuple")
        if context.capture_profile != "evaluation_metric_complete":
            raise ValueError("debugger recording requires metric-complete capture")
        if _context_action_source(context) != specification.action_source_kind:
            raise ValueError("recording action source must match episode context")
        if (
            specification.runtime_provenance.policy_execution_included
            != _context_policy_execution_included(context)
        ):
            raise ValueError("recording policy provenance must match episode context")
        if (
            context.code_revision.package_version
            != specification.runtime_provenance.package_version
        ):
            raise ValueError("runtime package version must match episode provenance")

        observer = EvaluationEpisodeObserverV1(context, reducers=reducers)
        observer.start(initial_frame)

        self._specification = specification
        self._context = context
        self._current_frame = initial_frame
        self._original_destination = destination
        self._current_destination = destination
        self._reducers = reducers
        self._observer = observer
        self._lifecycle: RecordingLifecycleV1 = "recording"
        self._close_cause: DebuggerRecordingCloseCauseV1 | None = None
        self._close_reason: str | None = None
        self._bundle: ReplayBundleV1 | None = None
        self._prepared_bundle: PreparedReplayBundleV1 | None = None
        self._saved_bundle: SavedReplayBundleV1 | None = None
        self._verified_loaded_bundle: LoadedReplayBundleV1 | None = None
        self._persistence_error_code: RecordingPersistenceErrorCodeV1 | None = None
        self._last_io_error_code: ReplayIOErrorCodeV1 | None = None
        self._verify_existing_on_retry = False
        self._publication_outcome: DebuggerRecordingPublicationOutcomeV1 | None = None

    @property
    def specification(self) -> DebuggerRecordingSpecificationV1:
        return self._specification

    @property
    def lifecycle(self) -> RecordingLifecycleV1:
        return self._lifecycle

    @property
    def observer_lifecycle_state(self) -> ObserverLifecycleState:
        """Expose observer diagnostics without exposing its mutable instance."""
        return self._observer.lifecycle_state

    @property
    def context(self) -> EvaluationEpisodeContextV1:
        """Return the exact frozen episode context owned by the observer."""
        return self._context

    @property
    def current_frame(self) -> EvaluationFrameV1:
        """Return the latest validated frozen frame without trajectory exposure."""
        return self._current_frame

    @property
    def expected_transition_count(self) -> int:
        return self._context.expected_horizon

    @property
    def close_cause(self) -> DebuggerRecordingCloseCauseV1 | None:
        return self._close_cause

    @property
    def persistence_error_code(self) -> RecordingPersistenceErrorCodeV1 | None:
        return self._persistence_error_code

    @property
    def publication_outcome(self) -> DebuggerRecordingPublicationOutcomeV1 | None:
        return self._publication_outcome

    @property
    def validated_transition_count(self) -> int:
        return self._observer.validated_transition_count

    @property
    def retained_frame_count(self) -> int:
        frames = self._observer.retained_frames
        if frames is None:
            raise RuntimeError("recording observer unexpectedly lacks retained frames")
        return len(frames)

    @property
    def retained_transition_count(self) -> int:
        transitions = self._observer.retained_transitions
        if transitions is None:
            raise RuntimeError(
                "recording observer unexpectedly lacks retained transitions"
            )
        return len(transitions)

    @property
    def bundle(self) -> ReplayBundleV1 | None:
        return self._bundle

    @property
    def prepared_bundle(self) -> PreparedReplayBundleV1 | None:
        return self._prepared_bundle

    @property
    def saved_bundle(self) -> SavedReplayBundleV1 | None:
        return self._saved_bundle

    @property
    def verified_loaded_bundle(self) -> LoadedReplayBundleV1 | None:
        return self._verified_loaded_bundle

    def _status_for(
        self,
        *,
        captured_transition_count: int,
        lifecycle: RecordingLifecycleV1,
        close_cause: DebuggerRecordingCloseCauseV1 | None,
        close_reason: str | None,
        persistence_error_code: RecordingPersistenceErrorCodeV1 | None,
    ) -> RecordingStatusV1:
        completion_state: (
            Literal["complete", "partial", "interrupted", "failed"] | None
        ) = None
        completion_reason: str | None = None
        if close_cause is not None:
            completion_state, mapped_reason, _origin = _completion_for_cause(
                close_cause,
                failure_reason=(
                    close_reason if close_cause in _FAILURE_ORIGIN_BY_CAUSE else None
                ),
            )
            if completion_state != "complete":
                completion_reason = close_reason or mapped_reason
        return RecordingStatusV1(
            lifecycle=lifecycle,
            captured_transition_count=captured_transition_count,
            expected_transition_count=self._context.expected_horizon,
            completion_state=completion_state,
            completion_reason=completion_reason,
            restart_fenced=(captured_transition_count > 0 or lifecycle != "recording"),
            finish_available=lifecycle == "recording",
            review_available=lifecycle == "saved",
            retry_available=lifecycle == "persistence_failed",
            save_as_available=lifecycle == "persistence_failed",
            discard_available=(
                lifecycle == "recording" and captured_transition_count > 0
            ),
            persistence_error_code=persistence_error_code,
        )

    @property
    def status(self) -> RecordingStatusV1:
        """Return the exact path-free status for the committed recorder state."""
        return self._status_for(
            captured_transition_count=self.validated_transition_count,
            lifecycle=self._lifecycle,
            close_cause=self._close_cause,
            close_reason=self._close_reason,
            persistence_error_code=self._persistence_error_code,
        )

    def preview_status_v1(
        self,
        *,
        captured_transition_count: int,
        lifecycle: RecordingLifecycleV1,
        close_cause: DebuggerRecordingCloseCauseV1 | None = None,
        completion_reason: str | None = None,
        persistence_error_code: RecordingPersistenceErrorCodeV1 | None = None,
    ) -> RecordingStatusV1:
        """Build a pure candidate status without mutating observer or recorder."""
        return self._status_for(
            captured_transition_count=captured_transition_count,
            lifecycle=lifecycle,
            close_cause=close_cause,
            close_reason=completion_reason,
            persistence_error_code=persistence_error_code,
        )

    def preview_status_after_append_v1(
        self,
        transition: EvaluationTransitionV1,
    ) -> RecordingStatusV1:
        """Predict the canonical status for the next already-validated unit."""
        if type(transition) is not EvaluationTransitionV1:
            raise TypeError("transition must be exact EvaluationTransitionV1")
        if self._lifecycle != "recording":
            raise RuntimeError("append preview requires an active recording")
        next_count = self.validated_transition_count + 1
        if transition.transition_index != self.validated_transition_count:
            raise ValueError("append preview transition index is not gap-free")
        if transition.episode_id != self._context.identity.episode_id:
            raise ValueError("append preview transition belongs to another episode")
        if transition.terminated or next_count == self._context.expected_horizon:
            return self.preview_status_v1(
                captured_transition_count=next_count,
                lifecycle="sealed",
                close_cause="endpoint",
            )
        if transition.truncated:
            return self.preview_status_v1(
                captured_transition_count=next_count,
                lifecycle="sealed",
                close_cause="truncation",
                completion_reason=(
                    transition.owning_task_end_reason or "environment_truncated"
                ),
            )
        return self.preview_status_v1(
            captured_transition_count=next_count,
            lifecycle="recording",
        )

    def replacement_for(
        self,
        context: EvaluationEpisodeContextV1,
        initial_frame: EvaluationFrameV1,
    ) -> DebuggerReplayRecorderV1:
        """Build an uncommitted frame-zero recorder without mutating this one."""
        if self._lifecycle not in ("recording", "sealed") or self._bundle is not None:
            raise RuntimeError("only an unfinalized recorder may build a replacement")
        action_source = _context_action_source(context)
        if action_source not in ("manual", "scripted", "mixed", "policy"):
            raise ValueError("replacement context has an unsupported action source")
        replacement_specification = build_debugger_recording_specification_v1(
            action_source_kind=action_source,
            runtime_provenance=self._specification.runtime_provenance.model_copy(
                update={
                    "policy_execution_included": (
                        _context_policy_execution_included(context)
                    )
                }
            ),
            wrapper_stack=self._specification.wrapper_stack,
        )
        return DebuggerReplayRecorderV1(
            specification=replacement_specification,
            destination=self._original_destination,
            context=context,
            initial_frame=initial_frame,
            reducers=self._reducers,
        )

    def append(
        self,
        transition: EvaluationTransitionV1,
        successor_frame: EvaluationFrameV1,
    ) -> None:
        """Delegate one coherent unit to the sole observer; perform no I/O."""
        if self._lifecycle != "recording":
            raise RuntimeError("append requires an active recording")
        previous_count = self._observer.validated_transition_count
        try:
            self._observer.append(transition, successor_frame)
        finally:
            if self._observer.validated_transition_count == previous_count + 1:
                self._current_frame = successor_frame
            self._synchronize_sealed_state()

    def _synchronize_sealed_state(self) -> None:
        transitions = self._observer.retained_transitions
        if not transitions:
            return
        tail = transitions[-1]
        count = self._observer.validated_transition_count
        if tail.terminated or count == self._context.expected_horizon:
            self._lifecycle = "sealed"
            self._close_cause = "endpoint"
            self._close_reason = None
        elif tail.truncated:
            self._lifecycle = "sealed"
            self._close_cause = "truncation"
            self._close_reason = tail.owning_task_end_reason or "environment_truncated"

    def _finalize_once(
        self,
        close_cause: DebuggerRecordingCloseCauseV1,
        *,
        failure_reason: str | None,
    ) -> None:
        if self._prepared_bundle is not None:
            return
        if self._lifecycle == "discarded":
            raise RuntimeError("discarded recording cannot be finalized")
        report = self._observer.finalized_report
        if report is None:
            completion_state, reason, origin = _completion_for_cause(
                close_cause,
                failure_reason=failure_reason,
            )
            if close_cause == "truncation" and self._close_reason is not None:
                reason = self._close_reason
            report = self._observer.finalize(
                completion_state=completion_state,
                end_or_failure_reason=reason,
                failure_origin=origin,
            )
            self._close_cause = close_cause
            self._close_reason = report.completion.end_or_failure_reason
        elif self._close_cause != close_cause:
            raise RuntimeError("recording close cause cannot change after finalize")
        self._lifecycle = "finalized_unsaved"

        if self._bundle is None:
            try:
                self._bundle = build_replay_bundle_v1(
                    self._observer,
                    report,
                    runtime_provenance=self._specification.runtime_provenance,
                    wrapper_stack=self._specification.wrapper_stack,
                )
            except (TypeError, ValueError) as error:
                raise _RecordingMaterializationError(
                    "replay bundle construction failed"
                ) from error
        try:
            self._prepared_bundle = prepare_replay_bundle_v1(self._bundle)
        except ReplaySaveError as error:
            raise _RecordingMaterializationError(
                "replay byte preparation failed"
            ) from error

    def _mark_materialization_failure(self) -> None:
        self._lifecycle = "persistence_failed"
        self._persistence_error_code = "publication_failed"
        self._last_io_error_code = None
        self._verify_existing_on_retry = False
        self._publication_outcome = "persistence_failed"

    def finalize_and_save(
        self,
        close_cause: DebuggerRecordingCloseCauseV1,
        *,
        failure_reason: str | None = None,
    ) -> DebuggerRecordingPublicationOutcomeV1:
        """Finalize exactly once, publish, and publicly reload the exact bytes."""
        if self._close_cause is not None and self._close_cause != close_cause:
            raise RuntimeError("recording close cause cannot change after finalize")
        if self._publication_outcome is not None:
            return self._publication_outcome
        try:
            self._finalize_once(close_cause, failure_reason=failure_reason)
        except _RecordingMaterializationError:
            self._mark_materialization_failure()
            return "persistence_failed"
        return self._publish(verify_existing=False)

    def _publish(
        self,
        *,
        verify_existing: bool,
    ) -> DebuggerRecordingPublicationOutcomeV1:
        prepared = self._prepared_bundle
        if prepared is None:
            raise RuntimeError("recording must be finalized before publication")
        try:
            saved = publish_prepared_replay_bundle_v1(
                prepared,
                self._current_destination,
                verify_existing_replay=verify_existing,
            )
            loaded = load_replay_bundle_v1(
                saved.replay_path,
                require_metric_report=True,
                max_file_size_bytes=prepared.max_file_size_bytes,
            )
            if (
                loaded.status != "complete"
                or loaded.replay != prepared.bundle.replay
                or loaded.metric_report_artifact
                != prepared.bundle.metric_report_artifact
            ):
                raise ReplayLoadError(
                    "semantic_validation_failed",
                    path=saved.replay_path,
                    detail="publicly reloaded bundle differs from prepared bytes",
                )
        except (ReplaySaveError, ReplayLoadError) as error:
            self._lifecycle = "persistence_failed"
            self._persistence_error_code = _persistence_error_code(error)
            self._last_io_error_code = error.code
            self._verify_existing_on_retry = (
                verify_existing
                or error.code == "replay_publication_verification_failed"
                or isinstance(error, ReplayLoadError)
            )
            self._publication_outcome = "persistence_failed"
            return "persistence_failed"

        self._saved_bundle = saved
        self._verified_loaded_bundle = loaded
        self._persistence_error_code = None
        self._last_io_error_code = None
        self._verify_existing_on_retry = False
        self._lifecycle = "saved"
        self._publication_outcome = "saved"
        return "saved"

    def retry_save(self) -> DebuggerRecordingPublicationOutcomeV1:
        """Retry exact cached bytes, verifying an uncertain prior publication."""
        if self._lifecycle != "persistence_failed":
            raise RuntimeError("retry requires persistence_failed lifecycle")
        if not self._resume_materialization():
            return "persistence_failed"
        return self._publish(verify_existing=self._verify_existing_on_retry)

    def _resume_materialization(self) -> bool:
        if self._prepared_bundle is not None:
            return True
        close_cause = self._close_cause
        if close_cause is None:
            raise RuntimeError("retryable materialization lacks a close cause")
        try:
            self._finalize_once(close_cause, failure_reason=None)
        except _RecordingMaterializationError:
            self._mark_materialization_failure()
            return False
        return True

    def save_as(self, basename: str) -> DebuggerRecordingPublicationOutcomeV1:
        """Publish cached bytes under one basename in the original directory."""
        if self._lifecycle != "persistence_failed":
            raise RuntimeError("Save As requires persistence_failed lifecycle")
        if type(basename) is not str or not basename:
            raise ValueError("Save As requires a nonempty basename")
        if (
            Path(basename).name != basename
            or "/" in basename
            or "\\" in basename
            or not basename.endswith(REPLAY_FILE_SUFFIX_V1)
        ):
            raise ValueError(
                "Save As accepts only a replay basename in the original directory"
            )
        candidate_path = self._original_destination.replay_path.parent / basename
        try:
            destination = preflight_replay_bundle_destination_v1(candidate_path)
        except ReplaySaveError as error:
            self._persistence_error_code = _persistence_error_code(error)
            self._last_io_error_code = error.code
            self._publication_outcome = "persistence_failed"
            return "persistence_failed"
        self._current_destination = destination
        self._verify_existing_on_retry = False
        if not self._resume_materialization():
            return "persistence_failed"
        return self._publish(verify_existing=False)

    def save_recovery_copy(self) -> DebuggerRecordingPublicationOutcomeV1:
        """Publish a deterministic Ctrl-C recovery sibling without overwrite."""
        if self._lifecycle != "persistence_failed":
            raise RuntimeError("recovery save requires persistence_failed lifecycle")
        if not self._resume_materialization():
            return "persistence_failed"
        prepared = self._prepared_bundle
        if prepared is None:
            raise RuntimeError("recovery save requires immutable prepared bytes")
        original_name = self._original_destination.replay_path.name
        stem = original_name[: -len(REPLAY_FILE_SUFFIX_V1)]
        recovery_name = (
            f"{stem}.recovery-{prepared.replay_payload_sha256[:16]}"
            f"{REPLAY_FILE_SUFFIX_V1}"
        )
        recovery_path = self._original_destination.replay_path.parent / recovery_name
        try:
            destination = preflight_replay_bundle_destination_v1(recovery_path)
        except ReplaySaveError as error:
            self._persistence_error_code = _persistence_error_code(error)
            self._last_io_error_code = error.code
            self._publication_outcome = "persistence_failed"
            return "persistence_failed"
        self._current_destination = destination
        self._verify_existing_on_retry = False
        return self._publish(verify_existing=False)

    def begin_review(self) -> LoadedReplayBundleV1:
        """Enter read-only review after the service has prebuilt its response."""
        loaded = self._verified_loaded_bundle
        if loaded is None or self._lifecycle not in ("saved", "reviewing"):
            raise RuntimeError("review requires one publicly verified saved bundle")
        self._lifecycle = "reviewing"
        return loaded

    def discard(self) -> None:
        """Mark an unfinalized draft/prefix discarded without deleting anything."""
        if self._lifecycle not in ("recording", "sealed") or self._bundle is not None:
            raise RuntimeError("only an unfinalized recording may be discarded")
        self._lifecycle = "discarded"
        self._close_cause = None
        self._close_reason = None


__all__ = [
    "DEBUGGER_RECORDING_SCHEMA_VERSION",
    "DEBUGGER_RECORDING_SPECIFICATION_SCHEMA_ID",
    "DebuggerRecordingCloseCauseV1",
    "DebuggerRecordingPublicationOutcomeV1",
    "DebuggerRecordingSpecificationV1",
    "DebuggerReplayRecorderV1",
    "build_debugger_recording_specification_v1",
]
