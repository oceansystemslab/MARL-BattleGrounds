"""Current replays with exact relative observations and shared trajectory rules."""

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
    """The existing header fields with current context and version identity."""

    # These exact wire roots share validators, not substitutable version values.
    schema_version: Literal[3] = 3  # pyright: ignore[reportIncompatibleVariableOverride]
    context: EvaluationEpisodeContextV3  # pyright: ignore[reportIncompatibleVariableOverride]


class ReplayArtifactV3(ReplayArtifactV2):
    """The same trajectory envelope with actual V2 observation frames."""

    schema_version: Literal[3] = 3  # pyright: ignore[reportIncompatibleVariableOverride]
    header: ReplayArtifactHeaderV3  # pyright: ignore[reportIncompatibleVariableOverride]
    frames: tuple[EvaluationFrameV2, ...]  # pyright: ignore[reportIncompatibleVariableOverride]

    @model_validator(mode="after")
    def _validate_current_frames(self) -> ReplayArtifactV3:
        for frame in self.frames:
            _validate_frame_information_regime(self.header.context, frame)
        return self


class ReplayArtifactReferenceV3(ReplayArtifactReferenceV2):
    """A reference preserving the actual V3 replay and context identities."""

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
    """Seal current captured rows using the shared completion and digest rules."""
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
    """Describe original canonical V3 bytes without rewriting their content."""
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
    """Validate the exact current root and every nested recorded type."""
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
    """Use the shared packet ordering and transfer path for current recordings."""
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
