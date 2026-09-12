"""Self-contained replays: recorded trajectories and provenance, without metrics."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Annotated, Literal, cast

from pydantic import Field, StringConstraints, model_validator

from marl_battlegrounds.evaluation.metrics import (
    CompletionState,
    EvaluationEpisodeCompletionV1,
    RolloutFailureOrigin,
)
from marl_battlegrounds.evaluation.models import (
    REQUIRED_SCHEMA_BINDINGS_V2,
    AssignedPolicySlotV1,
    AssignedPolicySlotV2,
    EvaluationEpisodeContext,
    EvaluationEpisodeContextV2,
    EvaluationFrame,
    EvaluationFrameV1,
    EvaluationModel,
    EvaluationTransitionV1,
    SchemaVersionEntryV2,
    canonical_digest_sha256,
    canonical_json_bytes,
    evaluation_context_type,
)
from marl_battlegrounds.evaluation.replay import (
    CANONICAL_REPLAY_JSON_PROFILE_V1,
    ReplayWrapperMetadataV1,
    RuntimeProvenanceV1,
)

if TYPE_CHECKING:
    from marl_battlegrounds.evaluation.replay_capture import ReplayPackets

type _Count = Annotated[int, Field(ge=0)]
type _Positive = Annotated[int, Field(gt=0)]
type _Identifier = Annotated[
    str, StringConstraints(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/+\-]*$")
]
type _Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


def context_v2(
    context: EvaluationEpisodeContext, *, scenario_name: str | None = None
) -> EvaluationEpisodeContextV2:
    """Keep known legacy provenance when recording a new V2 artifact."""
    evaluation_context_type(context)
    if context.schema_version not in (1, 2):
        raise ValueError("legacy V2 recording cannot change a current input contract")
    if type(context) is EvaluationEpisodeContextV2:
        return (
            context
            if scenario_name is None
            else EvaluationEpisodeContextV2.model_validate(
                {**context.model_dump(), "scenario_name": scenario_name}
            )
        )
    assignments = tuple(
        AssignedPolicySlotV2(
            **row.model_dump(
                exclude={"training_run_id", "training_step"}
                if row.policy_kind
                in ("manual", "scripted", "random_valid", "reactive_tdm", "scenario_5")
                else set()
            ),
            lifecycle="frozen",
        )
        if isinstance(row, AssignedPolicySlotV1)
        else row
        for row in context.policy_assignments
    )
    return EvaluationEpisodeContextV2(
        **context.model_dump(
            mode="python",
            exclude={"schema_version", "schema_versions", "policy_assignments"},
        ),
        schema_versions=tuple(
            SchemaVersionEntryV2(schema_id=name, schema_version=version)
            for name, version in REQUIRED_SCHEMA_BINDINGS_V2
        ),
        policy_assignments=assignments,
        scenario_name=scenario_name,
    )


class ReplayArtifactHeaderV2(EvaluationModel):
    """Recorded environment, policy and runtime identity for one trajectory."""

    schema_id: Literal["marl_battlegrounds.evaluation.replay_header"] = (
        "marl_battlegrounds.evaluation.replay_header"
    )
    schema_version: Literal[2] = 2
    header_id: _Identifier
    canonical_json_profile: Literal["marl_battlegrounds.canonical_json.v1"] = (
        CANONICAL_REPLAY_JSON_PROFILE_V1
    )
    context: EvaluationEpisodeContextV2
    context_digest_sha256: _Digest
    expected_transition_count: _Positive
    recorded_transition_count: _Count
    recorded_frame_count: _Positive
    first_frame_id: _Identifier
    last_frame_id: _Identifier
    runtime_provenance: RuntimeProvenanceV1
    wrapper_stack: tuple[ReplayWrapperMetadataV1, ...] = ()

    @model_validator(mode="after")
    def _validate_header(self) -> ReplayArtifactHeaderV2:
        episode_id = self.context.identity.episode_id
        if self.header_id != f"{episode_id}:replay-header":
            raise ValueError("replay header must join its episode")
        if self.context_digest_sha256 != canonical_digest_sha256(self.context):
            raise ValueError("replay context digest is not canonical")
        if self.expected_transition_count != self.context.expected_horizon:
            raise ValueError("replay horizon must match its context")
        if self.recorded_transition_count > self.expected_transition_count:
            raise ValueError("replay cannot exceed its declared horizon")
        if self.recorded_frame_count != self.recorded_transition_count + 1:
            raise ValueError("replay requires exactly T+1 frames for T transitions")
        if (self.first_frame_id, self.last_frame_id) != (
            f"{episode_id}:frame:0",
            f"{episode_id}:frame:{self.recorded_transition_count}",
        ):
            raise ValueError("replay frame bounds must join its episode")
        if (
            self.runtime_provenance.package_version
            != self.context.code_revision.package_version
        ):
            raise ValueError("runtime package version must match recorded source")
        if tuple(row.position for row in self.wrapper_stack) != tuple(
            range(len(self.wrapper_stack))
        ):
            raise ValueError("wrapper positions must be ordered without gaps")
        return self


class ReplayArtifactV2(EvaluationModel):
    """A strict, content-addressed trajectory with no metric-report dependency."""

    schema_id: Literal["marl_battlegrounds.evaluation.replay_artifact"] = (
        "marl_battlegrounds.evaluation.replay_artifact"
    )
    schema_version: Literal[2] = 2
    artifact_id: _Identifier
    canonical_digest_sha256: _Digest
    trajectory_content_digest_sha256: _Digest
    header: ReplayArtifactHeaderV2
    completion: EvaluationEpisodeCompletionV1
    frames: tuple[EvaluationFrameV1, ...]
    transitions: tuple[EvaluationTransitionV1, ...]

    @model_validator(mode="after")
    def _validate_artifact(self) -> ReplayArtifactV2:
        episode_id = self.header.context.identity.episode_id
        count = len(self.transitions)
        if self.artifact_id != f"{episode_id}:replay":
            raise ValueError("replay artifact must join its episode")
        if (
            len(self.frames) != self.header.recorded_frame_count
            or count != self.header.recorded_transition_count
        ):
            raise ValueError("replay counts must agree with the recorded tuples")
        if (
            self.completion.episode_id != episode_id
            or self.completion.validated_transition_count != count
        ):
            raise ValueError("replay completion must join its captured prefix")
        if (
            self.completion.expected_transition_count
            != self.header.expected_transition_count
        ):
            raise ValueError("replay completion horizon must match its context")
        for index, frame in enumerate(self.frames):
            if (frame.episode_id, frame.frame_index, frame.frame_id) != (
                episode_id,
                index,
                f"{episode_id}:frame:{index}",
            ):
                raise ValueError(
                    "replay frames must retain their contiguous episode identities"
                )
            if (
                frame.simulator_step_count
                != self.frames[0].simulator_step_count + index
            ):
                raise ValueError(
                    "replay frames must retain consecutive simulator epochs"
                )
            shared = (
                frame.shared_obs_information_availability_by_recipient_and_sensor_source
            )
            if (shared is not None) != (
                self.header.context.execution_information_mode == "shared_obs"
            ):
                raise ValueError(
                    "replay sensor availability must match its information regime"
                )
        for index, transition in enumerate(self.transitions):
            if (
                transition.episode_id,
                transition.transition_index,
                transition.transition_id,
            ) != (episode_id, index, f"{episode_id}:transition:{index}"):
                raise ValueError(
                    "replay transitions must retain their contiguous episode identities"
                )
            if index < count - 1 and (transition.terminated or transition.truncated):
                raise ValueError("replay cannot continue after an episode ends")
        tail = self.transitions[-1] if self.transitions else None
        if (self.completion.terminated, self.completion.truncated) != (
            False if tail is None else tail.terminated,
            False if tail is None else tail.truncated,
        ):
            raise ValueError("replay completion flags must match the captured tail")
        if self.trajectory_content_digest_sha256 != canonical_digest_sha256(
            self._trajectory_payload()
        ):
            raise ValueError("replay trajectory digest is not canonical")
        if self.canonical_digest_sha256 != canonical_digest_sha256(
            self, exclude={"canonical_digest_sha256"}
        ):
            raise ValueError("replay artifact digest is not canonical")
        return self

    def _trajectory_payload(self) -> dict[str, object]:
        return {
            "header": self.header,
            "completion": self.completion,
            "frames": self.frames,
            "transitions": self.transitions,
        }


class ReplayArtifactReferenceV2(EvaluationModel):
    """Path-free provenance retaining the real V2 replay identity."""

    schema_id: Literal["marl_battlegrounds.evaluation.replay_artifact_reference"] = (
        "marl_battlegrounds.evaluation.replay_artifact_reference"
    )
    schema_version: Literal[2] = 2
    artifact_id: _Identifier
    episode_id: _Identifier
    replay_schema_version: Literal[2] = 2
    context_digest_sha256: _Digest
    trajectory_content_digest_sha256: _Digest
    canonical_digest_sha256: _Digest
    canonical_byte_length: _Positive

    @model_validator(mode="after")
    def _validate_identity(self) -> ReplayArtifactReferenceV2:
        if self.artifact_id != f"{self.episode_id}:replay":
            raise ValueError("replay reference must join its episode")
        return self


def _build_replay_content(
    context: EvaluationEpisodeContext,
    frames: Sequence[EvaluationFrame],
    transitions: Sequence[EvaluationTransitionV1],
    *,
    runtime_provenance: RuntimeProvenanceV1,
    wrapper_stack: tuple[ReplayWrapperMetadataV1, ...],
    completion_state: CompletionState | None,
    end_or_failure_reason: str | None,
    failure_origin: RolloutFailureOrigin | None,
    header_type: type[EvaluationModel],
) -> dict[str, object]:
    """Share completion and content assembly across explicit replay versions."""
    frames = tuple(frames)
    transitions = tuple(transitions)
    if not frames or len(frames) != len(transitions) + 1:
        raise ValueError("replay requires a nonempty T+1/T captured prefix")
    count = len(transitions)
    episode_id = context.identity.episode_id
    tail = transitions[-1] if transitions else None
    terminated = False if tail is None else tail.terminated
    truncated = False if tail is None else tail.truncated
    bases = tuple(
        basis
        for basis, present in (
            ("task_terminal", terminated),
            ("declared_horizon", count == context.expected_horizon),
        )
        if present
    )
    completion = EvaluationEpisodeCompletionV1(
        episode_id=episode_id,
        completion_state=completion_state or ("complete" if bases else "partial"),
        expected_transition_count=context.expected_horizon,
        validated_transition_count=count,
        last_valid_frame_index=count,
        last_valid_frame_id=frames[-1].frame_id,
        terminated=terminated,
        truncated=truncated,
        completion_bases=cast(
            "tuple[Literal['task_terminal', 'declared_horizon'], ...]", bases
        ),
        end_or_failure_reason=end_or_failure_reason
        if end_or_failure_reason is not None
        else (
            tail.owning_task_end_reason
            if bases and tail is not None
            else "captured_prefix"
            if not bases
            else None
        ),
        failure_origin=failure_origin,
    )
    header = header_type.model_validate(
        {
            "header_id": f"{episode_id}:replay-header",
            "context": context,
            "context_digest_sha256": canonical_digest_sha256(context),
            "expected_transition_count": context.expected_horizon,
            "recorded_transition_count": count,
            "recorded_frame_count": len(frames),
            "first_frame_id": frames[0].frame_id,
            "last_frame_id": frames[-1].frame_id,
            "runtime_provenance": runtime_provenance,
            "wrapper_stack": wrapper_stack,
        }
    )
    content: dict[str, object] = {
        "header": header,
        "completion": completion,
        "frames": frames,
        "transitions": transitions,
    }
    return content


def build_replay_v2(
    context: EvaluationEpisodeContext,
    frames: Sequence[EvaluationFrameV1],
    transitions: Sequence[EvaluationTransitionV1],
    *,
    runtime_provenance: RuntimeProvenanceV1,
    wrapper_stack: tuple[ReplayWrapperMetadataV1, ...] = (),
    completion_state: CompletionState | None = None,
    end_or_failure_reason: str | None = None,
    failure_origin: RolloutFailureOrigin | None = None,
) -> ReplayArtifactV2:
    """Seal captured facts once; never compute metrics or repeat simulator rules."""
    context = context_v2(context)
    if any(type(frame) is not EvaluationFrameV1 for frame in frames):
        raise TypeError("legacy replay V2 requires exact V1 observation frames")
    episode_id = context.identity.episode_id
    content = _build_replay_content(
        context,
        frames,
        transitions,
        runtime_provenance=runtime_provenance,
        wrapper_stack=wrapper_stack,
        completion_state=completion_state,
        end_or_failure_reason=end_or_failure_reason,
        failure_origin=failure_origin,
        header_type=ReplayArtifactHeaderV2,
    )
    payload: dict[str, object] = {
        "schema_id": "marl_battlegrounds.evaluation.replay_artifact",
        "schema_version": 2,
        "artifact_id": f"{episode_id}:replay",
        "trajectory_content_digest_sha256": canonical_digest_sha256(content),
        **content,
    }
    return ReplayArtifactV2.model_validate(
        {**payload, "canonical_digest_sha256": canonical_digest_sha256(payload)}
    )


def replay_reference_v2(replay: ReplayArtifactV2) -> ReplayArtifactReferenceV2:
    """Build a reference to the actual canonical replay bytes."""
    if type(replay) is not ReplayArtifactV2:
        raise TypeError("V2 references require exact ReplayArtifactV2")
    return ReplayArtifactReferenceV2(
        artifact_id=replay.artifact_id,
        episode_id=replay.header.context.identity.episode_id,
        context_digest_sha256=replay.header.context_digest_sha256,
        trajectory_content_digest_sha256=replay.trajectory_content_digest_sha256,
        canonical_digest_sha256=replay.canonical_digest_sha256,
        canonical_byte_length=len(canonical_json_bytes(replay)),
    )


def replay_from_packets(
    context: EvaluationEpisodeContextV2,
    packets: Iterable[ReplayPackets],
    *,
    runtime_provenance: RuntimeProvenanceV1,
    wrapper_stack: tuple[ReplayWrapperMetadataV1, ...] = (),
    completion_state: CompletionState | None = None,
    end_or_failure_reason: str | None = None,
    failure_origin: RolloutFailureOrigin | None = None,
) -> ReplayArtifactV2:
    """Seal one ordered stream of scalar captured packets using existing wire leaves."""
    from marl_battlegrounds.evaluation.capture import (
        capture_evaluation_transition_unit_v1,
        capture_initial_evaluation_frame_v1,
    )

    if type(context) is not EvaluationEpisodeContextV2:
        raise TypeError("legacy packet capture requires exact context V2")
    frames, transitions = _frames_from_packets(
        context,
        packets,
        capture_initial=capture_initial_evaluation_frame_v1,
        capture_transition=capture_evaluation_transition_unit_v1,
    )
    return build_replay_v2(
        context,
        frames,
        transitions,
        runtime_provenance=runtime_provenance,
        wrapper_stack=wrapper_stack,
        completion_state=completion_state,
        end_or_failure_reason=end_or_failure_reason,
        failure_origin=failure_origin,
    )


def _frames_from_packets[FrameT: EvaluationFrame](
    context: EvaluationEpisodeContext,
    packets: Iterable[ReplayPackets],
    *,
    capture_initial: Callable[..., FrameT],
    capture_transition: Callable[..., tuple[EvaluationTransitionV1, FrameT]],
) -> tuple[list[FrameT], list[EvaluationTransitionV1]]:
    """Decode one packet stream with the requested exact observation capture."""
    from marl_battlegrounds.evaluation.catalog import build_resolved_env_config_v1
    from marl_battlegrounds.evaluation.recording_context import restore_recording_config

    frames: list[FrameT] = []
    transitions: list[EvaluationTransitionV1] = []
    episode_id: int | None = None
    availability = None
    for packet in packets:
        if not bool(packet.valid):
            raise ValueError("spooled replay packets must be valid scalar rows")
        if context.execution_information_mode == "shared_obs":
            availability = packet.source_availability
        index = int(packet.transition_index)
        if index != len(transitions) or bool(packet.initial) != (index == 0):
            raise ValueError(
                "replay packets must start at zero without gaps or duplicates"
            )
        if episode_id is None:
            episode_id = int(packet.episode_id)
            if episode_id <= 0:
                raise ValueError("captured episode IDs must be positive")
            if (
                build_resolved_env_config_v1(restore_recording_config(packet.config))
                != context.resolved_env_config
            ):
                raise ValueError(
                    "initial replay packet configuration must match its context"
                )
            frames.append(
                capture_initial(
                    context,
                    packet.initial_state,
                    packet.initial_observation,
                    packet.initial_action_mask,
                    availability,
                )
            )
        if int(packet.episode_id) != episode_id:
            raise ValueError("replay packets cannot mix episode identities")
        if transitions and (transitions[-1].terminated or transitions[-1].truncated):
            raise ValueError("replay packets cannot continue after completion")
        transition, frame = capture_transition(
            context,
            frames[-1],
            packet.state,
            packet.observation,
            packet.action_mask,
            packet.info.transition_facts,
            packet.reward,
            packet.done,
            successor_shared_obs_information_availability_by_recipient_and_sensor_source=availability,
        )
        if transitions and availability is not None:
            # Capture validated this matrix against the same frozen roster. It
            # also completes the preceding frame's next-decision source choice.
            field = "shared_obs_information_availability_by_recipient_and_sensor_source"
            admitted = getattr(frame, field)
            if getattr(frames[-1], field) != admitted:
                frames[-1] = frames[-1].model_copy(update={field: admitted})
        transitions.append(transition)
        frames.append(frame)
    return frames, transitions
