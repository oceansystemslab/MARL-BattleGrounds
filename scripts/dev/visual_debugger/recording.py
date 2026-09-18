"""Share debugger recording contracts and the version-1 bundle recorder.

The current live recorder imports the specification and status helpers here.
``DebuggerReplayRecorderV1`` preserves the older observer-backed bundle path.
It retains capture, prepares immutable replay and metric content, and verifies
saved files through the public loader. Callers own locking and commit a
recording only after validating their candidate response. No simulator step,
service lock, or browser state belongs to this module.
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
    AssignedPolicySlotV2,
    EvaluationEpisodeContext,
    EvaluationEpisodeContextV1,
    EvaluationFrameV1,
    EvaluationModel,
    EvaluationTransitionV1,
    canonical_digest_sha256,
)
from marl_battlegrounds.evaluation.reducers import build_tdm_metric_reducers
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
        """Require exact nested metadata, ordered wrappers, and matching content
        hashes.
        """
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
    """Build the path-free scientific fields used to derive a recording identity."""
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
    """Build an immutable recording description whose ID comes from its content.

    Parameters
    ----------
    action_source_kind : DebuggerActionSourceKindV1
        Whether manual input, scripts, policies, or a mixture supplies actions.
    runtime_provenance : RuntimeProvenanceV1
        Exact runtime metadata, including package and policy execution facts.
    wrapper_stack : tuple of ReplayWrapperMetadataV1, optional
        Ordered wrapper records with positions 0 through length minus one.
        Defaults to an empty tuple.

    Returns
    -------
    DebuggerRecordingSpecificationV1
        Validated specification with content and whole-record SHA-256 digests.

    Raises
    ------
    TypeError
        The wrapper collection is not an immutable tuple.
    ValueError
        Nested records, wrapper positions, or declared field values are invalid.

    Notes
    -----
    Local file paths are excluded. This call performs no replay I/O.
    """
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


def recording_action_source(context: EvaluationEpisodeContext) -> str:
    """Read the recording action source declared by an episode.

    Parameters
    ----------
    context : EvaluationEpisodeContext
        Episode context containing aggregation keys.

    Returns
    -------
    str
        Value of its unique ``action_source`` key.

    Raises
    ------
    ValueError
        The context contains zero or multiple action-source keys.
    """
    rows = tuple(
        row.value for row in context.aggregation_keys if row.name == "action_source"
    )
    if len(rows) != 1:
        raise ValueError("recording context requires exactly one action_source key")
    return rows[0]


def recording_policy_execution_included(context: EvaluationEpisodeContext) -> bool:
    """Check whether recorded slot assignments include an executed policy.

    Parameters
    ----------
    context : EvaluationEpisodeContext
        Context whose supported slot assignments declare their policy kind.

    Returns
    -------
    bool
        True for a reactive TDM, random-valid, or Scenario 5 policy assignment.
        Manual and scripted assignments do not count as policy execution.
    """
    return any(
        isinstance(row, (AssignedPolicySlotV1, AssignedPolicySlotV2))
        and row.policy_kind in ("reactive_tdm", "random_valid", "scenario_5")
        for row in context.policy_assignments
    )


def recording_persistence_error_code(
    error: ReplaySaveError | ReplayLoadError,
) -> RecordingPersistenceErrorCodeV1:
    """Map a detailed replay I/O failure to a public save-error category.

    Parameters
    ----------
    error : ReplaySaveError or ReplayLoadError
        Typed error from destination checks, publication, or public reload.

    Returns
    -------
    RecordingPersistenceErrorCodeV1
        Target unavailable, verification failed, or publication failed. Detailed
        local paths and exception text are not included in this value.
    """
    if error.code in _TARGET_ERROR_CODES:
        return "target_unavailable"
    if isinstance(error, ReplayLoadError) or (
        error.code == "replay_publication_verification_failed"
    ):
        return "verification_failed"
    return "publication_failed"


def recording_completion_for_cause(
    cause: DebuggerRecordingCloseCauseV1,
    *,
    failure_reason: str | None,
) -> tuple[
    Literal["complete", "partial", "interrupted", "failed"],
    str | None,
    RolloutFailureOrigin | None,
]:
    """Map a close cause to completion state, reason, and failure owner.

    Parameters
    ----------
    cause : DebuggerRecordingCloseCauseV1
        Supported reason capture ended.
    failure_reason : str or None
        Optional override for simulation, policy, capture, or validation failure.
        An override must contain 1..256 printable ASCII characters. Pass None
        for nonfailure causes so their fixed reason is used.

    Returns
    -------
    tuple
        Completion state, optional reason, and optional failure origin.

    Raises
    ------
    ValueError
        A reason is supplied for a nonfailure cause or has invalid text.
    """
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


def build_recording_status(
    expected_transition_count: int,
    *,
    captured_transition_count: int,
    lifecycle: RecordingLifecycleV1,
    close_cause: DebuggerRecordingCloseCauseV1 | None,
    close_reason: str | None,
    persistence_error_code: RecordingPersistenceErrorCodeV1 | None,
) -> RecordingStatusV1:
    """Build validated display facts and recording controls from lifecycle state.

    Parameters
    ----------
    expected_transition_count : int
        Planned maximum decision count.
    captured_transition_count : int
        Number of accepted decisions.
    lifecycle : RecordingLifecycleV1
        Current or proposed recording state.
    close_cause : DebuggerRecordingCloseCauseV1 or None
        Why capture ended, or None while open.
    close_reason : str or None
        Stored completion reason, if present.
    persistence_error_code : RecordingPersistenceErrorCodeV1 or None
        Public save-error category, if publication failed.

    Returns
    -------
    RecordingStatusV1
        Validated counts, completion facts, restart fence, and available actions.

    Raises
    ------
    ValueError
        Counts, completion facts, or action availability break the status contract.

    Notes
    -----
    This function performs no mutation or I/O.
    """
    completion_state: Literal["complete", "partial", "interrupted", "failed"] | None = (
        None
    )
    completion_reason: str | None = None
    if close_cause is not None:
        completion_state, mapped_reason, _origin = recording_completion_for_cause(
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
        expected_transition_count=expected_transition_count,
        completion_state=completion_state,
        completion_reason=completion_reason,
        restart_fenced=(captured_transition_count > 0 or lifecycle != "recording"),
        finish_available=lifecycle == "recording",
        review_available=lifecycle == "saved",
        retry_available=lifecycle == "persistence_failed",
        save_as_available=lifecycle == "persistence_failed",
        discard_available=(lifecycle == "recording" and captured_transition_count > 0),
        persistence_error_code=persistence_error_code,
    )


class DebuggerReplayRecorderV1:
    """Retain version-1 capture and publish a verified replay-and-metrics bundle.

    One observer owns accepted frames and transitions. With no supplied reducers,
    standard TDM metrics are evaluated from retained capture during finalization.
    Callers serialize access; the recorder has no lock and advances no simulator.
    """

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
        "_offline_metrics_evaluated",
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
        """Start a version-1 observer at the episode's initial frame.

        Parameters
        ----------
        specification : DebuggerRecordingSpecificationV1
            Exact recording metadata with matching action source and runtime facts.
        destination : ReplayBundleDestinationV1
            Exact preflighted destination for the replay and metric companion.
        context : EvaluationEpisodeContextV1
            Exact metric-complete context for this capture.
        initial_frame : EvaluationFrameV1
            Exact initial frame accepted by the context's observer.
        reducers : tuple of EvaluationMetricReducerV1, optional
            Immutable reducer tuple. Empty by default; standard TDM metrics are then
            evaluated once from retained capture at finalization.

        Raises
        ------
        TypeError
            An exact model type or immutable reducer tuple is missing.
        ValueError
            Capture profile, provenance, action source, or initial frame is invalid.

        Notes
        -----
        Construction starts an in-memory observer and writes no bundle files.
        """
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
        if recording_action_source(context) != specification.action_source_kind:
            raise ValueError("recording action source must match episode context")
        if (
            specification.runtime_provenance.policy_execution_included
            != recording_policy_execution_included(context)
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
        self._offline_metrics_evaluated = False
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
        """Return the immutable path-free recording specification."""
        return self._specification

    @property
    def lifecycle(self) -> RecordingLifecycleV1:
        """Return the current capture, publication, or review lifecycle state."""
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
        """Return the planned maximum number of captured decisions."""
        return self._context.expected_horizon

    @property
    def close_cause(self) -> DebuggerRecordingCloseCauseV1 | None:
        """Return why capture ended, or None while its end is undecided."""
        return self._close_cause

    @property
    def persistence_error_code(self) -> RecordingPersistenceErrorCodeV1 | None:
        """Return the latest public save-error category, or None."""
        return self._persistence_error_code

    @property
    def publication_outcome(self) -> DebuggerRecordingPublicationOutcomeV1 | None:
        """Return the cached publication result, or None before publication."""
        return self._publication_outcome

    @property
    def validated_transition_count(self) -> int:
        """Return the number of transitions accepted by this recorder."""
        return self._observer.validated_transition_count

    @property
    def retained_frame_count(self) -> int:
        """Return the retained state count, including the initial state."""
        return self._observer.validated_transition_count + 1

    @property
    def retained_transition_count(self) -> int:
        """Return the number of retained transition records."""
        return self._observer.validated_transition_count

    @property
    def bundle(self) -> ReplayBundleV1 | None:
        """Return the built immutable replay bundle, or None before materialization."""
        return self._bundle

    @property
    def prepared_bundle(self) -> PreparedReplayBundleV1 | None:
        """Return cached serialized bundle content, or None before preparation."""
        return self._prepared_bundle

    @property
    def saved_bundle(self) -> SavedReplayBundleV1 | None:
        """Return published file identities, or None before verified publication."""
        return self._saved_bundle

    @property
    def verified_loaded_bundle(self) -> LoadedReplayBundleV1 | None:
        """Return the public reload result, or None before verification."""
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
        """Build a proposed public status using this episode's expected horizon."""
        return build_recording_status(
            self._context.expected_horizon,
            captured_transition_count=captured_transition_count,
            lifecycle=lifecycle,
            close_cause=close_cause,
            close_reason=close_reason,
            persistence_error_code=persistence_error_code,
        )

    @property
    def status(self) -> RecordingStatusV1:
        """Return current counts, lifecycle, completion, and available save actions."""
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
        """Build a possible status without changing the recording.

        Parameters
        ----------
        captured_transition_count : int
            Number of captured decisions in the proposed status.
        lifecycle : RecordingLifecycleV1
            Proposed recording state.
        close_cause : DebuggerRecordingCloseCauseV1 or None, optional
            Why capture ended. None means it has not ended.
        completion_reason : str or None, optional
            Stored end or failure reason. Defaults to None.
        persistence_error_code : RecordingPersistenceErrorCodeV1 or None, optional
            Public save error category. Defaults to None.

        Returns
        -------
        RecordingStatusV1
            Validated counts, completion facts, and available recording actions.

        Raises
        ------
        ValueError
            The proposed fields break the status or completion contract.
        """
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
        """Check the next transition and predict whether capture will close.

        Parameters
        ----------
        transition : EvaluationTransitionV1
            Exact transition model for this episode and the next contiguous index.

        Returns
        -------
        RecordingStatusV1
            Proposed state after one more decision. Termination or the expected
            horizon seals a complete recording; earlier truncation seals an
            interrupted recording.

        Raises
        ------
        TypeError
            The transition is not the exact supported model.
        ValueError
            The episode or transition index does not match this recording.
        RuntimeError
            This recording cannot accept another transition.

        Notes
        -----
        This method changes no recorder state and does not validate successor data.
        """
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
        """Build an uncommitted recorder for a replacement episode at frame zero.

        Parameters
        ----------
        context : EvaluationEpisodeContextV1
            New context with a supported action source and valid recording provenance.
        initial_frame : EvaluationFrameV1
            Matching initial frame for that context.

        Returns
        -------
        DebuggerReplayRecorderV1
            Fresh recorder with the original destination, reducers, and wrapper stack.

        Raises
        ------
        RuntimeError
            This recorder has already been finalized or discarded.
        TypeError
            The replacement models use unsupported exact types.
        ValueError
            Replacement inputs fail constructor validation.

        Notes
        -----
        This recorder is unchanged; the caller decides whether to install the result.
        """
        if self._lifecycle not in ("recording", "sealed") or self._bundle is not None:
            raise RuntimeError("only an unfinalized recorder may build a replacement")
        action_source = recording_action_source(context)
        if action_source not in ("manual", "scripted", "mixed", "policy"):
            raise ValueError("replacement context has an unsupported action source")
        replacement_specification = build_debugger_recording_specification_v1(
            action_source_kind=action_source,
            runtime_provenance=self._specification.runtime_provenance.model_copy(
                update={
                    "policy_execution_included": (
                        recording_policy_execution_included(context)
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
        """Give one transition and successor frame to the retaining observer.

        Parameters
        ----------
        transition : EvaluationTransitionV1
            Next contiguous transition for this episode.
        successor_frame : EvaluationFrameV1
            Matching next frame, validated by the shared observer.

        Raises
        ------
        RuntimeError
            Capture is not active or the observer rejects its lifecycle.
        TypeError
            A supplied record has an unsupported type.
        ValueError
            The records fail the shared observer's capture contract.

        Notes
        -----
        This performs no file I/O. If the observer accepts the transition before a
        later reducer failure, the current frame and sealed state still advance to
        that accepted transition.
        """
        if self._lifecycle != "recording":
            raise RuntimeError("append requires an active recording")
        previous_count = self._observer.validated_transition_count
        try:
            self._observer.append(transition, successor_frame)
        finally:
            if self._observer.validated_transition_count == previous_count + 1:
                self._current_frame = successor_frame
                self._synchronize_sealed_state(transition)

    def _synchronize_sealed_state(self, tail: EvaluationTransitionV1) -> None:
        """Seal capture at termination, the planned horizon, or earlier truncation."""
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
        """Finalize metrics and cache immutable bundle content at most once.

        Reuse any existing report and prepared bytes. A changed close cause or a
        discarded capture is rejected. Expected materialization failures are wrapped
        so publication status can report them without rerunning completed work.
        """
        if self._prepared_bundle is not None:
            return
        if self._lifecycle == "discarded":
            raise RuntimeError("discarded recording cannot be finalized")
        report = self._observer.finalized_report
        if report is None:
            if not self._reducers and not self._offline_metrics_evaluated:
                self._observer.evaluate_retained(build_tdm_metric_reducers())
                self._offline_metrics_evaluated = True
            completion_state, reason, origin = recording_completion_for_cause(
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
        """Record a preparation failure without claiming that files were published."""
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
        """Close capture, publish its replay, and reload the saved result.

        Parameters
        ----------
        close_cause : DebuggerRecordingCloseCauseV1
            Declared reason capture ended. It cannot change after finalization.
        failure_reason : str or None, optional
            Optional 1..256 printable ASCII characters for a failure cause.
            Nonfailure causes use their fixed reason. Defaults to None.

        Returns
        -------
        DebuggerRecordingPublicationOutcomeV1
            ``saved`` after public reload verification, or ``persistence_failed``.
            Repeated calls return the existing publication result.

        Raises
        ------
        ValueError
            The cause and supplied failure reason are inconsistent.
        RuntimeError
            The recording was discarded or its existing close cause differs.

        Notes
        -----
        This call changes lifecycle state and may write files. It retains prepared
        content for a later retry after a save failure. Callers serialize access.
        """
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
        """Publish cached bytes and verify both replay and metrics through reload.

        ``verify_existing`` permits checking an uncertain earlier publication against
        the same prepared content. Expected I/O errors update retry state and return
        persistence_failed; no new rollout or metric reduction is performed here.
        """
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
            self._persistence_error_code = recording_persistence_error_code(error)
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
        """Retry the retained replay after a failed publication.

        Returns
        -------
        DebuggerRecordingPublicationOutcomeV1
            ``saved`` after verification, or ``persistence_failed`` again.

        Raises
        ------
        RuntimeError
            The previous publication did not fail.

        Notes
        -----
        This may write files and updates save status. An uncertain earlier write is
        verified against the retained content before it can count as a success.
        """
        if self._lifecycle != "persistence_failed":
            raise RuntimeError("retry requires persistence_failed lifecycle")
        if not self._resume_materialization():
            return "persistence_failed"
        return self._publish(verify_existing=self._verify_existing_on_retry)

    def _resume_materialization(self) -> bool:
        """Resume a failed preparation, reusing completed work; return whether ready."""
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
        """Retry publication under a new filename in the original directory.

        Parameters
        ----------
        basename : str
            One nonempty filename ending in the replay suffix. Directory separators
            and paths outside the original destination directory are rejected.

        Returns
        -------
        DebuggerRecordingPublicationOutcomeV1
            ``saved`` or ``persistence_failed``. Saved content is reloaded and checked.

        Raises
        ------
        ValueError
            The basename is not an allowed replay filename.
        RuntimeError
            The recorder is not in the failed-publication state.

        Notes
        -----
        This changes the retry destination and may create files. Existing unrelated
        files are not overwritten.
        """
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
            self._persistence_error_code = recording_persistence_error_code(error)
            self._last_io_error_code = error.code
            self._publication_outcome = "persistence_failed"
            return "persistence_failed"
        self._current_destination = destination
        self._verify_existing_on_retry = False
        if not self._resume_materialization():
            return "persistence_failed"
        return self._publish(verify_existing=False)

    def save_recovery_copy(self) -> DebuggerRecordingPublicationOutcomeV1:
        """Try a deterministic recovery filename beside the original replay.

        Returns
        -------
        DebuggerRecordingPublicationOutcomeV1
            ``saved`` or ``persistence_failed``.

        Raises
        ------
        RuntimeError
            The recorder has no failed publication to recover.

        Notes
        -----
        The name includes the first 16 characters of the replay content hash. This
        may create a file, but does not overwrite an unrelated existing replay.
        """
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
            self._persistence_error_code = recording_persistence_error_code(error)
            self._last_io_error_code = error.code
            self._publication_outcome = "persistence_failed"
            return "persistence_failed"
        self._current_destination = destination
        self._verify_existing_on_retry = False
        return self._publish(verify_existing=False)

    def begin_review(self) -> LoadedReplayBundleV1:
        """Enter review using the already verified saved bundle.

        Returns
        -------
        LoadedReplayBundleV1
            Cached replay and metric bundle from successful public reload.

        Raises
        ------
        RuntimeError
            No saved and verified bundle is available.

        Notes
        -----
        This changes lifecycle to reviewing without reading files again.
        """
        loaded = self._verified_loaded_bundle
        if loaded is None or self._lifecycle not in ("saved", "reviewing"):
            raise RuntimeError("review requires one publicly verified saved bundle")
        self._lifecycle = "reviewing"
        return loaded

    def discard(self) -> None:
        """Mark an unfinished recording discarded without deleting files.

        Raises
        ------
        RuntimeError
            Capture has already been finalized or discarded.

        Notes
        -----
        This changes lifecycle state and clears its close reason. It does not free
        all retained capture objects or remove a published file.
        """
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
