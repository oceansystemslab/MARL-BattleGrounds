"""Current debugger recording: captured frames and a single immutable replay file."""

from __future__ import annotations

from pathlib import Path

from marl_battlegrounds.evaluation.metrics import ObserverLifecycleState
from marl_battlegrounds.evaluation.models import (
    EvaluationEpisodeContextV1,
    EvaluationFrameV1,
    EvaluationTransitionV1,
)
from marl_battlegrounds.evaluation.replay_io import (
    REPLAY_FILE_SUFFIX_V1,
    LoadedReplay,
    PreparedReplay,
    ReplayDestination,
    ReplayLoadError,
    ReplaySaveError,
    SavedReplay,
    load_replay,
    preflight_replay_destination,
    publish_prepared_replay,
)
from marl_battlegrounds.evaluation.replay_v2 import (
    ReplayArtifactV2,
    build_replay_v2,
    context_v2,
)
from scripts.dev.visual_debugger.protocol import (
    RecordingLifecycleV1,
    RecordingPersistenceErrorCodeV1,
    RecordingStatusV1,
)
from scripts.dev.visual_debugger.recording import (
    DebuggerRecordingCloseCauseV1,
    DebuggerRecordingPublicationOutcomeV1,
    DebuggerRecordingSpecificationV1,
    build_debugger_recording_specification_v1,
    build_recording_status,
    recording_action_source,
    recording_completion_for_cause,
    recording_persistence_error_code,
    recording_policy_execution_included,
)


class DebuggerReplayRecorder:
    """Capture only; compute optional metrics later through the shared viewer path."""

    def __init__(
        self,
        *,
        specification: DebuggerRecordingSpecificationV1,
        destination: ReplayDestination,
        context: EvaluationEpisodeContextV1,
        initial_frame: EvaluationFrameV1,
        scenario_name: str | None = None,
    ) -> None:
        if (
            type(specification) is not DebuggerRecordingSpecificationV1
            or type(destination) is not ReplayDestination
        ):
            raise TypeError(
                "recording requires an exact specification and replay destination"
            )
        if (
            type(context) is not EvaluationEpisodeContextV1
            or type(initial_frame) is not EvaluationFrameV1
        ):
            raise TypeError(
                "recording requires an exact live context and initial frame"
            )
        if (
            initial_frame.episode_id != context.identity.episode_id
            or initial_frame.frame_index != 0
        ):
            raise ValueError("recording must start at its episode's initial frame")
        if (
            context.capture_profile != "evaluation_metric_complete"
            or recording_action_source(context) != specification.action_source_kind
        ):
            raise ValueError(
                "recording capture profile and action source must match its context"
            )
        if (
            specification.runtime_provenance.policy_execution_included
            != recording_policy_execution_included(context)
        ):
            raise ValueError("recording policy execution must match its context")
        if (
            specification.runtime_provenance.package_version
            != context.code_revision.package_version
        ):
            raise ValueError("recording runtime must match its source revision")
        self.specification = specification
        self.context = context
        self.scenario_name = scenario_name
        self._original_destination = destination
        self._destination = destination
        self._frames = [initial_frame]
        self._transitions: list[EvaluationTransitionV1] = []
        self.lifecycle: RecordingLifecycleV1 = "recording"
        self.close_cause: DebuggerRecordingCloseCauseV1 | None = None
        self._close_reason: str | None = None
        self._failed_append = False
        self.replay: ReplayArtifactV2 | None = None
        self.prepared_replay: PreparedReplay | None = None
        self.saved_bundle: SavedReplay | None = None
        self.verified_loaded_bundle: LoadedReplay | None = None
        self.persistence_error_code: RecordingPersistenceErrorCodeV1 | None = None
        self.publication_outcome: DebuggerRecordingPublicationOutcomeV1 | None = None
        self._verify_existing = False

    @property
    def current_frame(self) -> EvaluationFrameV1:
        return self._frames[-1]

    @property
    def expected_transition_count(self) -> int:
        return self.context.expected_horizon

    @property
    def validated_transition_count(self) -> int:
        """Structurally accepted capture count, retained for the service interface."""
        return len(self._transitions)

    @property
    def retained_frame_count(self) -> int:
        return len(self._frames)

    @property
    def retained_transition_count(self) -> int:
        return len(self._transitions)

    @property
    def observer_lifecycle_state(self) -> ObserverLifecycleState:
        """Report capture lifecycle without constructing a metric observer."""
        if self.replay is not None:
            return "finalized"
        if self._failed_append:
            return "poisoned"
        return "sealed" if self.lifecycle == "sealed" else "open"

    @property
    def status(self) -> RecordingStatusV1:
        return self.preview_status_v1(
            captured_transition_count=self.validated_transition_count,
            lifecycle=self.lifecycle,
            close_cause=self.close_cause,
            completion_reason=self._close_reason,
            persistence_error_code=self.persistence_error_code,
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
        return build_recording_status(
            self.expected_transition_count,
            captured_transition_count=captured_transition_count,
            lifecycle=lifecycle,
            close_cause=close_cause,
            close_reason=completion_reason,
            persistence_error_code=persistence_error_code,
        )

    def preview_status_after_append_v1(
        self, transition: EvaluationTransitionV1
    ) -> RecordingStatusV1:
        if type(transition) is not EvaluationTransitionV1:
            raise TypeError("recording append requires an exact transition")
        if self.lifecycle != "recording" or self._failed_append:
            raise RuntimeError("append requires an active recording")
        if (
            transition.episode_id != self.context.identity.episode_id
            or transition.transition_index != self.validated_transition_count
        ):
            raise ValueError(
                "recording append must extend its contiguous episode prefix"
            )
        count = self.validated_transition_count + 1
        cause: DebuggerRecordingCloseCauseV1 | None = None
        reason = None
        if transition.terminated or count == self.expected_transition_count:
            cause = "endpoint"
        elif transition.truncated:
            cause = "truncation"
            reason = transition.owning_task_end_reason or "environment_truncated"
        return self.preview_status_v1(
            captured_transition_count=count,
            lifecycle="recording" if cause is None else "sealed",
            close_cause=cause,
            completion_reason=reason,
        )

    def append(
        self, transition: EvaluationTransitionV1, successor_frame: EvaluationFrameV1
    ) -> None:
        """Retain the live bridge's facts after checking adjacent frame identities."""
        try:
            status = self.preview_status_after_append_v1(transition)
            if type(successor_frame) is not EvaluationFrameV1:
                raise TypeError("recording append requires an exact successor frame")
            if (
                transition.start_frame_id != self.current_frame.frame_id
                or transition.successor_frame_id != successor_frame.frame_id
                or successor_frame.episode_id != self.context.identity.episode_id
                or successor_frame.frame_index != len(self._frames)
                or successor_frame.simulator_step_count
                != self.current_frame.simulator_step_count + 1
            ):
                raise ValueError(
                    "recording transition must join adjacent captured frames"
                )
        except TypeError, ValueError:
            self._failed_append = True
            raise
        self._transitions.append(transition)
        self._frames.append(successor_frame)
        self.lifecycle = status.lifecycle
        if self.lifecycle == "sealed":
            self.close_cause = (
                "truncation"
                if transition.truncated
                and not transition.terminated
                and len(self._transitions) < self.expected_transition_count
                else "endpoint"
            )
            self._close_reason = status.completion_reason

    def replacement_for(
        self, context: EvaluationEpisodeContextV1, initial_frame: EvaluationFrameV1
    ) -> DebuggerReplayRecorder:
        if self.lifecycle not in ("recording", "sealed") or self.replay is not None:
            raise RuntimeError("only an unfinalized recording may be replaced")
        source = recording_action_source(context)
        if source not in ("manual", "scripted", "mixed", "policy"):
            raise ValueError("unsupported recording action source")
        specification = build_debugger_recording_specification_v1(
            action_source_kind=source,
            runtime_provenance=self.specification.runtime_provenance.model_copy(
                update={
                    "policy_execution_included": recording_policy_execution_included(
                        context
                    )
                }
            ),
            wrapper_stack=self.specification.wrapper_stack,
        )
        return DebuggerReplayRecorder(
            specification=specification,
            destination=self._original_destination,
            context=context,
            initial_frame=initial_frame,
            scenario_name=self.scenario_name,
        )

    def _prepare(self) -> None:
        if self.prepared_replay is not None:
            return
        if self.close_cause is None:
            raise RuntimeError("recording must close before materialization")
        if self.replay is None:
            state, reason, origin = recording_completion_for_cause(
                self.close_cause,
                failure_reason=self._close_reason
                if self.close_cause.endswith("_failure")
                and self.close_cause != "processing_failure"
                else None,
            )
            self.replay = build_replay_v2(
                context_v2(self.context, scenario_name=self.scenario_name),
                self._frames,
                self._transitions,
                runtime_provenance=self.specification.runtime_provenance,
                wrapper_stack=self.specification.wrapper_stack,
                completion_state=state,
                end_or_failure_reason=self._close_reason or reason,
                failure_origin=origin,
            )
            self._close_reason = self.replay.completion.end_or_failure_reason
        self.prepared_replay = PreparedReplay(self.replay)

    def finalize_and_save(
        self,
        close_cause: DebuggerRecordingCloseCauseV1,
        *,
        failure_reason: str | None = None,
    ) -> DebuggerRecordingPublicationOutcomeV1:
        if self.lifecycle == "discarded":
            raise RuntimeError("discarded recording cannot be finalized")
        if self.close_cause is not None and self.close_cause != close_cause:
            raise RuntimeError("recording close cause cannot change after finalization")
        recording_completion_for_cause(close_cause, failure_reason=failure_reason)
        if self.publication_outcome is not None:
            return self.publication_outcome
        self.close_cause = close_cause
        self._close_reason = failure_reason or self._close_reason
        return self._publish()

    def _publish(self) -> DebuggerRecordingPublicationOutcomeV1:
        try:
            self._prepare()
            prepared = self.prepared_replay
            if prepared is None:
                raise RuntimeError("recording publication requires prepared bytes")
            self.lifecycle = "finalized_unsaved"
            saved = publish_prepared_replay(
                prepared,
                self._destination,
                verify_existing_replay=self._verify_existing,
            )
            loaded = load_replay(
                saved.replay_path, max_file_size_bytes=prepared.max_file_size_bytes
            )
            if loaded.replay != prepared.replay or loaded.status != "not_recorded":
                raise ReplayLoadError(
                    "semantic_validation_failed",
                    path=saved.replay_path,
                    detail="publicly reloaded replay differs from prepared capture",
                )
        except (ReplayLoadError, ReplaySaveError, TypeError, ValueError) as error:
            self.lifecycle = "persistence_failed"
            self.persistence_error_code = (
                recording_persistence_error_code(error)
                if isinstance(error, (ReplayLoadError, ReplaySaveError))
                else "publication_failed"
            )
            self._verify_existing = (
                self._verify_existing
                or isinstance(error, ReplayLoadError)
                or (
                    isinstance(error, ReplaySaveError)
                    and error.code == "replay_publication_verification_failed"
                )
            )
            self.publication_outcome = "persistence_failed"
            return "persistence_failed"
        self.saved_bundle = saved
        self.verified_loaded_bundle = loaded
        self.lifecycle = "saved"
        self.persistence_error_code = None
        self._verify_existing = False
        self.publication_outcome = "saved"
        return "saved"

    def retry_save(self) -> DebuggerRecordingPublicationOutcomeV1:
        if self.lifecycle != "persistence_failed":
            raise RuntimeError("retry requires a failed publication")
        return self._publish()

    def save_as(self, basename: str) -> DebuggerRecordingPublicationOutcomeV1:
        if self.lifecycle != "persistence_failed":
            raise RuntimeError("Save As requires a failed publication")
        if (
            type(basename) is not str
            or not basename
            or Path(basename).name != basename
            or "/" in basename
            or "\\" in basename
            or not basename.endswith(REPLAY_FILE_SUFFIX_V1)
        ):
            raise ValueError(
                "Save As requires one replay basename in the original directory"
            )
        try:
            self._destination = preflight_replay_destination(
                self._original_destination.replay_path.parent / basename
            )
        except ReplaySaveError as error:
            self.persistence_error_code = recording_persistence_error_code(error)
            return "persistence_failed"
        self._verify_existing = False
        return self._publish()

    def save_recovery_copy(self) -> DebuggerRecordingPublicationOutcomeV1:
        if self.lifecycle != "persistence_failed":
            raise RuntimeError("recovery requires a failed publication")
        try:
            self._prepare()
        except ReplaySaveError, TypeError, ValueError:
            return "persistence_failed"
        prepared = self.prepared_replay
        if prepared is None:
            raise RuntimeError("recovery requires prepared bytes")
        stem = self._original_destination.replay_path.name[
            : -len(REPLAY_FILE_SUFFIX_V1)
        ]
        return self.save_as(
            f"{stem}.recovery-{prepared.replay_payload_sha256[:16]}{REPLAY_FILE_SUFFIX_V1}"
        )

    def begin_review(self) -> LoadedReplay:
        if (
            self.lifecycle not in ("saved", "reviewing")
            or self.verified_loaded_bundle is None
        ):
            raise RuntimeError("review requires a publicly reloaded recording")
        self.lifecycle = "reviewing"
        return self.verified_loaded_bundle

    def discard(self) -> None:
        if self.lifecycle not in ("recording", "sealed") or self.replay is not None:
            raise RuntimeError("only an unfinalized recording may be discarded")
        self.lifecycle = "discarded"
        self.close_cause = None
        self._close_reason = None


__all__ = ["DebuggerReplayRecorder"]
