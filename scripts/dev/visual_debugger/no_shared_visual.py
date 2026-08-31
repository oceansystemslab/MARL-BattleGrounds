"""Visual-only NoSharedObs V1 adapter over canonical V2 episode records.

The evaluation context retains the canonical V2 actor-input identity so policy
capture and reconstruction remain truthful.  The existing debugger renderer
needs only the established V1 recipient-safe visual slice, which predates the
public class-ID leaf added to the policy projection.  This module confines that
presentation compatibility boundary; it never changes captured records or
policy inputs.
"""

from marl_battlegrounds.evaluation.actor_projection import (
    NO_SHARED_OBS_ACTOR_PROJECTION_ID,
    NO_SHARED_OBS_ACTOR_PROJECTION_V2,
)
from marl_battlegrounds.evaluation.metrics import EvaluationTransitionViewV1
from marl_battlegrounds.evaluation.models import (
    EvaluationEpisodeContextV1,
    EvaluationFrameV1,
    VersionedIdentityV1,
    canonical_digest_sha256,
)
from marl_battlegrounds.evaluation.pov import (
    ACTOR_POV_CONTENT_SCHEMA_ID,
    ACTOR_POV_SCHEMA_VERSION,
    ActorPovAdjacentTransitionSliceV1,
    ActorPovCurrentSliceV1,
    ActorPovEpisodeCompletionV1,
    ActorPovReplayContentV1,
    build_actor_pov_adjacent_transition_slice_v1,
    build_actor_pov_current_slice_v1,
)
from marl_battlegrounds.evaluation.replay import (
    ReplayArtifactV1,
    validate_replay_artifact_v1,
)

_VISUAL_PROJECTION_V1 = VersionedIdentityV1(
    identifier=NO_SHARED_OBS_ACTOR_PROJECTION_ID,
    version=1,
)


def _visual_context_v1(
    context: EvaluationEpisodeContextV1,
) -> EvaluationEpisodeContextV1:
    if type(context) is not EvaluationEpisodeContextV1:
        raise TypeError("visual context requires exact EvaluationEpisodeContextV1")
    if context.execution_information_mode != "no_shared_obs":
        raise ValueError("NoSharedObs visual slices require no_shared_obs execution")
    if context.actor_projection == _VISUAL_PROJECTION_V1:
        return context
    if context.actor_projection != NO_SHARED_OBS_ACTOR_PROJECTION_V2:
        raise ValueError("unsupported NoSharedObs actor projection")
    return context.model_copy(update={"actor_projection": _VISUAL_PROJECTION_V1})


def build_live_no_shared_obs_visual_current_slice_v1(
    context: EvaluationEpisodeContextV1,
    frame: EvaluationFrameV1,
    *,
    global_slot: int,
    incoming_transition_view: EvaluationTransitionViewV1 | None = None,
) -> ActorPovCurrentSliceV1:
    """Build the established visual slice without changing capture authority."""
    visual_context = _visual_context_v1(context)
    visual_incoming = (
        None
        if incoming_transition_view is None
        else EvaluationTransitionViewV1(
            context=visual_context,
            start_frame=incoming_transition_view.start_frame,
            transition=incoming_transition_view.transition,
            successor_frame=incoming_transition_view.successor_frame,
        )
    )
    return build_actor_pov_current_slice_v1(
        visual_context,
        frame,
        global_slot=global_slot,
        incoming_transition_view=visual_incoming,
    )


def build_live_no_shared_obs_visual_adjacent_slice_v1(
    view: EvaluationTransitionViewV1,
    *,
    global_slot: int,
) -> ActorPovAdjacentTransitionSliceV1:
    """Build one visual incoming carrier over an unchanged transition unit."""
    if type(view) is not EvaluationTransitionViewV1:
        raise TypeError("visual transition requires exact EvaluationTransitionViewV1")
    visual_view = EvaluationTransitionViewV1(
        context=_visual_context_v1(view.context),
        start_frame=view.start_frame,
        transition=view.transition,
        successor_frame=view.successor_frame,
    )
    return build_actor_pov_adjacent_transition_slice_v1(
        visual_view,
        global_slot=global_slot,
    )


def build_replay_no_shared_obs_visual_content_v1(
    replay: ReplayArtifactV1,
    *,
    global_slot: int,
) -> ActorPovReplayContentV1:
    """Build an ephemeral V1 renderer view over one canonical V2 replay.

    The returned content is the established recipient-safe visual subset.  It
    is not an actor-POV artifact, must not be persisted, and must not be
    advertised as an exact V2 policy input: the canonical replay retains the
    V2 projection identity and its separately reconstructable class-ID leaf.
    """
    if type(replay) is not ReplayArtifactV1:
        raise TypeError("NoSharedObs visual replay requires ReplayArtifactV1")
    validate_replay_artifact_v1(replay)
    context = replay.header.context
    if context.execution_information_mode != "no_shared_obs":
        raise ValueError("NoSharedObs visual replay requires no_shared_obs execution")
    if context.actor_projection != NO_SHARED_OBS_ACTOR_PROJECTION_V2:
        raise ValueError("NoSharedObs visual replay requires actor projection V2")

    slices = tuple(
        build_live_no_shared_obs_visual_current_slice_v1(
            context,
            frame,
            global_slot=global_slot,
            incoming_transition_view=(
                None
                if frame_index == 0
                else EvaluationTransitionViewV1(
                    context=context,
                    start_frame=replay.frames[frame_index - 1],
                    transition=replay.transitions[frame_index - 1],
                    successor_frame=frame,
                )
            ),
        )
        for frame_index, frame in enumerate(replay.frames)
    )
    first = slices[0]
    transitions = tuple(
        transition
        for slice_ in slices[1:]
        if (transition := slice_.incoming_transition) is not None
    )
    if len(transitions) != len(replay.transitions):
        raise RuntimeError("NoSharedObs visual replay lost an incoming transition")

    source_completion = replay.completion
    completion = ActorPovEpisodeCompletionV1(
        completion_state=source_completion.completion_state,
        expected_transition_count=source_completion.expected_transition_count,
        captured_transition_count=source_completion.validated_transition_count,
        terminated=source_completion.terminated,
        truncated=source_completion.truncated,
        completion_bases=source_completion.completion_bases,
        public_end_or_failure_reason=source_completion.end_or_failure_reason,
    )
    content_payload: dict[str, object] = {
        "schema_id": ACTOR_POV_CONTENT_SCHEMA_ID,
        "schema_version": ACTOR_POV_SCHEMA_VERSION,
        "content_id": (f"{first.episode_id}:actor-pov:{first.public_agent_id}:content"),
        "episode_id": first.episode_id,
        "selected_global_slot": first.selected_global_slot,
        "selected_team_local_slot": first.selected_team_local_slot,
        "public_agent_id": first.public_agent_id,
        "configured_team_id": first.configured_team_id,
        "class_id": first.class_id,
        "observation_materialization": first.observation_materialization,
        "axis_mapping": first.axis_mapping,
        "completion": completion,
        "frames": tuple(slice_.frame for slice_ in slices),
        "transitions": transitions,
    }
    return ActorPovReplayContentV1.model_validate(
        {
            **content_payload,
            "canonical_digest_sha256": canonical_digest_sha256(content_payload),
        }
    )


__all__ = [
    "build_live_no_shared_obs_visual_adjacent_slice_v1",
    "build_live_no_shared_obs_visual_current_slice_v1",
    "build_replay_no_shared_obs_visual_content_v1",
]
