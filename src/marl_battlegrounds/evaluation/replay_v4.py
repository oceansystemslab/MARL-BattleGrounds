"""Build current replays that record the Team Deathmatch Red Zone rule.

V4 binds episode context V4 (resolved config V2, which records the Red Zone
depth) to frame V3 (20 context columns; column 19 is the depth) and unchanged
transition facts. It reuses the V2 envelope's ordering, completion and digest
authorities rather than duplicating them. Replay V3 and older stay readable
with their original meanings. Builders operate on captured records; metrics,
simulator execution and file publication belong to separate modules. Capture
imports stay local to replay_from_packets so replay readers do not import JAX.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Literal

from pydantic import model_validator

from marl_battlegrounds.evaluation.metrics import CompletionState, RolloutFailureOrigin
from marl_battlegrounds.evaluation.models import (
    EvaluationEpisodeContextV4,
    EvaluationFrameV3,
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


class ReplayArtifactHeaderV4(ReplayArtifactHeaderV2):
    """Current replay header with exact V4 context.

    Attributes
    ----------
    schema_version : Literal[4]
        Fixed integer 4.
    context : EvaluationEpisodeContextV4
        Exact EvaluationEpisodeContextV4, including the recorded Red Zone depth.

    Notes
    -----
    All other identity, count, runtime and wrapper fields follow
    ReplayArtifactHeaderV2. Its validators still require T+1 frames for T
    transitions, matching context/runtime versions and ordered wrapper positions.
    This is a separate frozen wire root, not an alias for an older header.
    """

    # These exact wire roots share validators, not substitutable version values.
    schema_version: Literal[4] = 4  # pyright: ignore[reportIncompatibleVariableOverride]
    context: EvaluationEpisodeContextV4  # pyright: ignore[reportIncompatibleVariableOverride]


class ReplayArtifactV4(ReplayArtifactV2):
    """Current replay envelope with 20-column observation frames.

    Attributes
    ----------
    schema_version : Literal[4]
        Fixed integer 4.
    header : ReplayArtifactHeaderV4
        Exact ReplayArtifactHeaderV4.
    frames : tuple[EvaluationFrameV3, ...]
        Tuple of EvaluationFrameV3; context column 19 is the Red Zone depth.

    Notes
    -----
    Artifact IDs, completion, transitions and digests follow ReplayArtifactV2's
    common contract. Additional validation checks each frame against the
    recorded information regime. This frozen record holds captured evidence,
    not a policy input, metric report or open file.
    """

    schema_version: Literal[4] = 4  # pyright: ignore[reportIncompatibleVariableOverride]
    header: ReplayArtifactHeaderV4  # pyright: ignore[reportIncompatibleVariableOverride]
    frames: tuple[EvaluationFrameV3, ...]  # pyright: ignore[reportIncompatibleVariableOverride]

    @model_validator(mode="after")
    def _validate_current_frames(self) -> ReplayArtifactV4:
        """Check every frame against the recorded execution information contract."""
        for frame in self.frames:
            _validate_frame_information_regime(self.header.context, frame)
        return self


class ReplayArtifactReferenceV4(ReplayArtifactReferenceV2):
    """Path-free reference to original replay V4 bytes.

    Attributes
    ----------
    schema_version : Literal[4]
        Fixed reference version 4.
    replay_schema_version : Literal[4]
        Fixed replay version 4.

    Notes
    -----
    IDs, content digests and positive canonical byte length use the shared
    ReplayArtifactReferenceV2 field contract. The version values identify real
    V4 bytes; this reference does not convert or relabel historical artifacts.
    """

    schema_version: Literal[4] = 4  # pyright: ignore[reportIncompatibleVariableOverride]
    replay_schema_version: Literal[4] = 4  # pyright: ignore[reportIncompatibleVariableOverride]


def build_replay_v4(
    context: EvaluationEpisodeContextV4,
    frames: Sequence[EvaluationFrameV3],
    transitions: Sequence[EvaluationTransitionV1],
    *,
    runtime_provenance: RuntimeProvenanceV1,
    wrapper_stack: tuple[ReplayWrapperMetadataV1, ...] = (),
    completion_state: CompletionState | None = None,
    end_or_failure_reason: str | None = None,
    failure_origin: RolloutFailureOrigin | None = None,
) -> ReplayArtifactV4:
    """Seal current captured rows into a validated V4 replay.

    Parameters
    ----------
    context : EvaluationEpisodeContextV4
        Exact EvaluationEpisodeContextV4 for this captured episode.
    frames : Sequence[EvaluationFrameV3]
        Nonempty sequence of exact EvaluationFrameV3, contiguous from zero.
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
    ReplayArtifactV4
        Frozen ReplayArtifactV4 with checked frame/transition joins, information
        regime, completion and canonical context/trajectory/artifact digests.

    Raises
    ------
    TypeError
        Context is not exact V4 or a frame is not exact V3.
    ValueError
        Capture counts, identities, information regime, completion
        or recorded metadata are inconsistent.

    Notes
    -----
    Host-only; no game execution, metric calculation or file publication.
    An initial-only partial prefix is allowed. Existing immutable input records
    are not rewritten or silently upgraded from historical observations.
    """
    if type(context) is not EvaluationEpisodeContextV4:
        raise TypeError("current replay capture requires exact context V4")
    if any(type(frame) is not EvaluationFrameV3 for frame in frames):
        raise TypeError("current replay capture requires exact frame V3")
    content = _build_replay_content(
        context,
        tuple(frames),
        tuple(transitions),
        runtime_provenance=runtime_provenance,
        wrapper_stack=wrapper_stack,
        completion_state=completion_state,
        end_or_failure_reason=end_or_failure_reason,
        failure_origin=failure_origin,
        header_type=ReplayArtifactHeaderV4,
    )
    payload = {
        "schema_id": "marl_battlegrounds.evaluation.replay_artifact",
        "schema_version": 4,
        "artifact_id": f"{context.identity.episode_id}:replay",
        "trajectory_content_digest_sha256": canonical_digest_sha256(content),
        **content,
    }
    return ReplayArtifactV4.model_validate(
        {**payload, "canonical_digest_sha256": canonical_digest_sha256(payload)}
    )


def replay_reference_v4(replay: ReplayArtifactV4) -> ReplayArtifactReferenceV4:
    """Describe the original canonical bytes of an exact V4 replay.

    Parameters
    ----------
    replay : ReplayArtifactV4
        Exact ReplayArtifactV4.

    Returns
    -------
    ReplayArtifactReferenceV4
        ReplayArtifactReferenceV4 containing existing identities/digests and the
        canonical JSON byte length. It includes no machine-local path.

    Raises
    ------
    TypeError
        replay is not exact V4.

    Notes
    -----
    Host-only serialization. This does not read/write files or change content.
    """
    if type(replay) is not ReplayArtifactV4:
        raise TypeError("current replay references require exact ReplayArtifactV4")
    return ReplayArtifactReferenceV4(
        artifact_id=replay.artifact_id,
        episode_id=replay.header.context.identity.episode_id,
        context_digest_sha256=replay.header.context_digest_sha256,
        trajectory_content_digest_sha256=replay.trajectory_content_digest_sha256,
        canonical_digest_sha256=replay.canonical_digest_sha256,
        canonical_byte_length=len(canonical_json_bytes(replay)),
    )


def validate_replay_artifact_v4(replay: ReplayArtifactV4) -> None:
    """Validate a V4 replay's exact model tree without converting versions.

    Parameters
    ----------
    replay : ReplayArtifactV4
        Expected exact ReplayArtifactV4 with declared nested model types.

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
        replay, record_name="replay", expected_type=ReplayArtifactV4
    )


def replay_from_packets(
    context: EvaluationEpisodeContextV4,
    packets: Iterable[ReplayPackets],
    *,
    runtime_provenance: RuntimeProvenanceV1,
    wrapper_stack: tuple[ReplayWrapperMetadataV1, ...] = (),
    completion_state: CompletionState | None = None,
    end_or_failure_reason: str | None = None,
    failure_origin: RolloutFailureOrigin | None = None,
) -> ReplayArtifactV4:
    """Build a current replay from one ordered stream of scalar capture packets.

    Parameters
    ----------
    context : EvaluationEpisodeContextV4
        Exact EvaluationEpisodeContextV4 matching the initial packet config,
        including its Red Zone depth.
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
    ReplayArtifactV4
        Validated ReplayArtifactV4 with 20-column V3 observation frames,
        unchanged authoritative transition facts and canonical digests.

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
        capture_evaluation_transition_unit_v3,
        capture_initial_evaluation_frame_v3,
    )

    frames, transitions = _frames_from_packets(
        context,
        packets,
        capture_initial=capture_initial_evaluation_frame_v3,
        capture_transition=capture_evaluation_transition_unit_v3,
    )
    return build_replay_v4(
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
    "ReplayArtifactHeaderV4",
    "ReplayArtifactReferenceV4",
    "ReplayArtifactV4",
    "build_replay_v4",
    "replay_from_packets",
    "replay_reference_v4",
    "validate_replay_artifact_v4",
]
