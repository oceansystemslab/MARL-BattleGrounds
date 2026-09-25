"""Build and read historical V3 replays with exact relative observation frames.

V3 binds episode context V3 to V2 frames (19 context columns) and unchanged
transition facts. It was the current format before the Red Zone rule; new
recordings use replay_v4 (context V4, frame V3), and V3 files stay readable.
It reuses the V2 envelope's ordering, completion and digest authorities rather
than duplicating them. Builders operate on captured records; metrics, simulator
execution and file publication belong to separate modules.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Literal

from pydantic import model_validator

from marl_battlegrounds.evaluation.metrics import CompletionState, RolloutFailureOrigin
from marl_battlegrounds.evaluation.models import (
    EvaluationEpisodeContextV3,
    EvaluationFrameV2,
    EvaluationTransitionV1,
    canonical_digest_sha256,
    canonical_json_bytes,
)
from marl_battlegrounds.evaluation.replay import (
    ReplayWrapperMetadataV1,
    RuntimeProvenanceV1,
)
from marl_battlegrounds.evaluation.replay_v2 import (
    ReplayArtifactHeaderV2,
    ReplayArtifactReferenceV2,
    ReplayArtifactV2,
    _build_replay_content,  # pyright: ignore[reportPrivateUsage]
    _frames_from_packets,  # pyright: ignore[reportPrivateUsage]
)
from marl_battlegrounds.evaluation.validation import (
    _validate_frame_information_regime,  # pyright: ignore[reportPrivateUsage]
    validate_declared_model_tree,
)

if TYPE_CHECKING:
    from marl_battlegrounds.evaluation.replay_capture import ReplayPackets


class ReplayArtifactHeaderV3(ReplayArtifactHeaderV2):
    """Current replay header with exact V3 context.

    Attributes
    ----------
    schema_version : Literal[3]
        Fixed integer 3.
    context : EvaluationEpisodeContextV3
        Exact EvaluationEpisodeContextV3 (the pre-Red-Zone context version).

    Notes
    -----
    All other identity, count, runtime and wrapper fields follow
    ReplayArtifactHeaderV2. Its validators still require T+1 frames for T
    transitions, matching context/runtime versions and ordered wrapper positions.
    This is a separate frozen wire root, not an alias for a V2 header.
    """

    # These exact wire roots share validators, not substitutable version values.
    schema_version: Literal[3] = 3  # pyright: ignore[reportIncompatibleVariableOverride]
    context: EvaluationEpisodeContextV3  # pyright: ignore[reportIncompatibleVariableOverride]


class ReplayArtifactV3(ReplayArtifactV2):
    """Current replay envelope with exact relative observation frames.

    Attributes
    ----------
    schema_version : Literal[3]
        Fixed integer 3.
    header : ReplayArtifactHeaderV3
        Exact ReplayArtifactHeaderV3.
    frames : tuple[EvaluationFrameV2, ...]
        Tuple of EvaluationFrameV2 containing the V3 replay's observation leaves.

    Notes
    -----
    Artifact IDs, completion, transitions and digests follow ReplayArtifactV2's
    common contract. Additional validation checks each frame against the current
    recorded information regime. This frozen record holds captured evidence,
    not a policy input, metric report or open file.
    """

    schema_version: Literal[3] = 3  # pyright: ignore[reportIncompatibleVariableOverride]
    header: ReplayArtifactHeaderV3  # pyright: ignore[reportIncompatibleVariableOverride]
    frames: tuple[EvaluationFrameV2, ...]  # pyright: ignore[reportIncompatibleVariableOverride]

    @model_validator(mode="after")
    def _validate_current_frames(self) -> ReplayArtifactV3:
        """Check every V2 frame against the recorded execution information
        contract.
        """
        for frame in self.frames:
            _validate_frame_information_regime(self.header.context, frame)
        return self


class ReplayArtifactReferenceV3(ReplayArtifactReferenceV2):
    """Path-free reference to original V3 replay bytes.

    Attributes
    ----------
    schema_version : Literal[3]
        Fixed reference version 3.
    replay_schema_version : Literal[3]
        Fixed replay version 3.

    Notes
    -----
    IDs, content digests and positive canonical byte length use the shared
    ReplayArtifactReferenceV2 field contract. The version values identify real
    V3 bytes; this reference does not convert or relabel historical artifacts.
    """

    schema_version: Literal[3] = 3  # pyright: ignore[reportIncompatibleVariableOverride]
    replay_schema_version: Literal[3] = 3  # pyright: ignore[reportIncompatibleVariableOverride]


def build_replay_v3(
    context: EvaluationEpisodeContextV3,
    frames: Sequence[EvaluationFrameV2],
    transitions: Sequence[EvaluationTransitionV1],
    *,
    runtime_provenance: RuntimeProvenanceV1,
    wrapper_stack: tuple[ReplayWrapperMetadataV1, ...] = (),
    completion_state: CompletionState | None = None,
    end_or_failure_reason: str | None = None,
    failure_origin: RolloutFailureOrigin | None = None,
) -> ReplayArtifactV3:
    """Seal context V3 captured rows into a validated V3 replay.

    Parameters
    ----------
    context : EvaluationEpisodeContextV3
        Exact EvaluationEpisodeContextV3 for this captured episode.
    frames : Sequence[EvaluationFrameV2]
        Nonempty sequence of exact EvaluationFrameV2, contiguous from zero.
    transitions : Sequence[EvaluationTransitionV1]
        EvaluationTransitionV1 sequence with one fewer row than frames.
    runtime_provenance : RuntimeProvenanceV1
        Actual runtime matching the context's package version.
    wrapper_stack : tuple[ReplayWrapperMetadataV1, ...]
        Ordered wrapper metadata, default ().
    completion_state : CompletionState | None
        Optional explicit status. None infers complete only from
        task termination or the declared horizon, otherwise partial.
    end_or_failure_reason : str | None
        Optional truthful reason. None uses the recorded
        task ending reason when complete, otherwise captured_prefix.
    failure_origin : RolloutFailureOrigin | None
        Optional recorded failure origin, default None.

    Returns
    -------
    ReplayArtifactV3
        Frozen ReplayArtifactV3 with checked frame/transition joins, information
        regime, completion and canonical context/trajectory/artifact digests.

    Raises
    ------
    TypeError
        Context is not exact V3 or a frame is not exact V2.
    ValueError
        Capture counts, identities, information regime, completion
        or recorded metadata are inconsistent.

    Notes
    -----
    Host-only; no game execution, metric calculation or file publication.
    An initial-only partial prefix is allowed. Existing immutable input records
    are not rewritten or silently upgraded from historical observations.
    """
    if type(context) is not EvaluationEpisodeContextV3:
        raise TypeError("current replay capture requires exact context V3")
    if any(type(frame) is not EvaluationFrameV2 for frame in frames):
        raise TypeError("current replay capture requires exact frame V2")
    content = _build_replay_content(
        context,
        tuple(frames),
        tuple(transitions),
        runtime_provenance=runtime_provenance,
        wrapper_stack=wrapper_stack,
        completion_state=completion_state,
        end_or_failure_reason=end_or_failure_reason,
        failure_origin=failure_origin,
        header_type=ReplayArtifactHeaderV3,
    )
    payload = {
        "schema_id": "marl_battlegrounds.evaluation.replay_artifact",
        "schema_version": 3,
        "artifact_id": f"{context.identity.episode_id}:replay",
        "trajectory_content_digest_sha256": canonical_digest_sha256(content),
        **content,
    }
    return ReplayArtifactV3.model_validate(
        {**payload, "canonical_digest_sha256": canonical_digest_sha256(payload)}
    )


def replay_reference_v3(replay: ReplayArtifactV3) -> ReplayArtifactReferenceV3:
    """Describe the original canonical bytes of an exact V3 replay.

    Parameters
    ----------
    replay : ReplayArtifactV3
        Exact ReplayArtifactV3.

    Returns
    -------
    ReplayArtifactReferenceV3
        ReplayArtifactReferenceV3 containing existing identities/digests and the
        canonical JSON byte length. It includes no machine-local path.

    Raises
    ------
    TypeError
        replay is not exact V3.

    Notes
    -----
    Host-only serialization. This does not read/write files or change content.
    """
    if type(replay) is not ReplayArtifactV3:
        raise TypeError("current replay references require exact ReplayArtifactV3")
    return ReplayArtifactReferenceV3(
        artifact_id=replay.artifact_id,
        episode_id=replay.header.context.identity.episode_id,
        context_digest_sha256=replay.header.context_digest_sha256,
        trajectory_content_digest_sha256=replay.trajectory_content_digest_sha256,
        canonical_digest_sha256=replay.canonical_digest_sha256,
        canonical_byte_length=len(canonical_json_bytes(replay)),
    )


def validate_replay_artifact_v3(replay: ReplayArtifactV3) -> None:
    """Validate a V3 replay's exact model tree without converting versions.

    Parameters
    ----------
    replay : ReplayArtifactV3
        Expected exact ReplayArtifactV3 with declared nested model types.

    Returns
    -------
    None
        None when the complete declared tree is valid.

    Raises
    ------
    TypeError
        The root or a nested model has the wrong exact type.
    ValueError
        Stored fields violate current replay/model invariants.

    Notes
    -----
    Host-only and read-only. Revalidation can detect data created through
    unchecked model construction; it does not repair or rewrite a replay.
    """
    validate_declared_model_tree(
        replay, record_name="replay", expected_type=ReplayArtifactV3
    )


def replay_from_packets(
    context: EvaluationEpisodeContextV3,
    packets: Iterable[ReplayPackets],
    *,
    runtime_provenance: RuntimeProvenanceV1,
    wrapper_stack: tuple[ReplayWrapperMetadataV1, ...] = (),
    completion_state: CompletionState | None = None,
    end_or_failure_reason: str | None = None,
    failure_origin: RolloutFailureOrigin | None = None,
) -> ReplayArtifactV3:
    """Build a V3 replay from one ordered stream of scalar capture packets.

    Parameters
    ----------
    context : EvaluationEpisodeContextV3
        Exact EvaluationEpisodeContextV3 matching the initial packet config.
    packets : Iterable[ReplayPackets]
        Iterable of valid scalar ReplayPackets for one positive episode ID.
        Start at zero with initial=True; indexes must be consecutive and no row
        may follow completion.
    runtime_provenance : RuntimeProvenanceV1
        Actual runtime matching the context's package version.
    wrapper_stack : tuple[ReplayWrapperMetadataV1, ...]
        Ordered wrapper metadata, default ().
    completion_state : CompletionState | None
        Optional explicit status. None infers complete from task
        termination/horizon, otherwise partial.
    end_or_failure_reason : str | None
        Optional reason, otherwise task reason or captured_prefix.
    failure_origin : RolloutFailureOrigin | None
        Optional failure origin, default None.

    Returns
    -------
    ReplayArtifactV3
        Validated ReplayArtifactV3 with current V2 observation frames, unchanged
        authoritative transition facts and canonical digests.

    Raises
    ------
    TypeError
        The context or captured records have incompatible exact types.
    ValueError
        The stream is empty, invalid, mixed, out of order, inconsistent
        with context or incompatible with declared completion.

    Notes
    -----
    Host-only. Consumes the iterable and may read device arrays during capture
    conversion. Shared ordering/type restoration is owned by the common packet
    path. It does not edit inputs, retain extra policy projections, calculate
    metrics or publish files.
    """
    from marl_battlegrounds.evaluation.capture import (
        capture_evaluation_transition_unit_v2,
        capture_initial_evaluation_frame_v2,
    )

    frames, transitions = _frames_from_packets(
        context,
        packets,
        capture_initial=capture_initial_evaluation_frame_v2,
        capture_transition=capture_evaluation_transition_unit_v2,
    )
    return build_replay_v3(
        context,
        frames,
        transitions,
        runtime_provenance=runtime_provenance,
        wrapper_stack=wrapper_stack,
        completion_state=completion_state,
        end_or_failure_reason=end_or_failure_reason,
        failure_origin=failure_origin,
    )


__all__ = [
    "ReplayArtifactHeaderV3",
    "ReplayArtifactReferenceV3",
    "ReplayArtifactV3",
    "build_replay_v3",
    "replay_from_packets",
    "replay_reference_v3",
    "validate_replay_artifact_v3",
]
