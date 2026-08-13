"""Recipient-sliced actor-POV export, privacy, and validation proofs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import pytest
from pydantic import ValidationError
from tests.evaluation_fixtures import (
    CapturedEvaluationTrajectory,
    captured_evaluation_trajectory,
    neutral_action,
)

from marl_battlegrounds.core.types import Action
from marl_battlegrounds.evaluation.metrics import build_evaluation_observer_v1
from marl_battlegrounds.evaluation.models import (
    BaseObservationV1,
    EvaluationFrameV1,
    GlobalAnalysisSnapshotV1,
    SpawnLifecycleObservationV1,
    canonical_digest_sha256,
    canonical_json_bytes,
)
from marl_battlegrounds.evaluation.pov import (
    ActorPovFrameV1,
    ActorPovReplayArtifactV1,
    ActorPovReplayContentV1,
    ActorPovSpawnLifecycleV1,
    ActorPovTransitionV1,
    canonical_actor_pov_content_json_bytes_v1,
    canonical_actor_pov_replay_json_bytes_v1,
    export_actor_pov_replay_v1,
    validate_actor_pov_replay_against_replay_v1,
    validate_actor_pov_replay_artifact_v1,
    validate_actor_pov_replay_content_v1,
)
from marl_battlegrounds.evaluation.replay import (
    MetricReportReferenceV1,
    ReplayArtifactV1,
    RuntimeProvenanceV1,
    build_replay_bundle_v1,
    validate_replay_artifact_v1,
)


def _runtime_provenance() -> RuntimeProvenanceV1:
    return RuntimeProvenanceV1(
        python_version="3.13.0",
        package_version="0.0.0",
        jax_version="0.7.0",
        jaxlib_version="0.7.0",
        numpy_version="2.3.0",
        pydantic_version="2.11.0",
        platform="linux",
        machine="x86_64",
        backend="cpu",
        device="generic-cpu",
        precision="float32",
        environment_count=1,
        batch_shape=(1,),
        policy_execution_included=False,
    )


def _build_replay(
    trajectory: CapturedEvaluationTrajectory,
    *,
    completion_state: str = "partial",
    reason: str | None = "fixture_prefix",
) -> ReplayArtifactV1:
    observer = build_evaluation_observer_v1(trajectory.context)
    observer.start(trajectory.frames[0])
    for transition, successor in zip(
        trajectory.transitions,
        trajectory.frames[1:],
        strict=True,
    ):
        observer.append(transition, successor)
    report = observer.finalize(
        completion_state=completion_state,  # type: ignore[arg-type]
        end_or_failure_reason=reason,
    )
    return build_replay_bundle_v1(
        observer,
        report,
        runtime_provenance=_runtime_provenance(),
    ).replay


@pytest.fixture(scope="module")
def trajectory() -> CapturedEvaluationTrajectory:
    return captured_evaluation_trajectory(transition_count=2)


@pytest.fixture(scope="module")
def replay(trajectory: CapturedEvaluationTrajectory) -> ReplayArtifactV1:
    return _build_replay(trajectory)


def _rebuild_replay(
    replay: ReplayArtifactV1,
    *,
    frames: tuple[EvaluationFrameV1, ...] | None = None,
) -> ReplayArtifactV1:
    replacement_frames = replay.frames if frames is None else frames
    trajectory_payload: dict[str, object] = {
        "header": replay.header,
        "completion": replay.completion,
        "processing_status": replay.processing_status,
        "frames": replacement_frames,
        "transitions": replay.transitions,
    }
    trajectory_digest = canonical_digest_sha256(trajectory_payload)
    reference_payload = replay.metric_report_reference.model_dump(mode="python")
    reference_payload["trajectory_content_digest_sha256"] = trajectory_digest
    report_reference = MetricReportReferenceV1.model_validate(reference_payload)
    replay_payload: dict[str, object] = {
        "schema_id": replay.schema_id,
        "schema_version": replay.schema_version,
        "artifact_id": replay.artifact_id,
        "trajectory_content_digest_sha256": trajectory_digest,
        "header": replay.header,
        "completion": replay.completion,
        "processing_status": replay.processing_status,
        "metric_report_reference": report_reference,
        "frames": replacement_frames,
        "transitions": replay.transitions,
    }
    rebuilt = ReplayArtifactV1.model_validate(
        {
            **replay_payload,
            "canonical_digest_sha256": canonical_digest_sha256(replay_payload),
        }
    )
    validate_replay_artifact_v1(rebuilt)
    return rebuilt


def _rebuild_content(
    content: ActorPovReplayContentV1,
    **updates: object,
) -> ActorPovReplayContentV1:
    payload = content.model_dump(
        mode="python",
        exclude={"canonical_digest_sha256"},
    )
    payload.update(updates)
    return ActorPovReplayContentV1.model_validate(
        {
            **payload,
            "canonical_digest_sha256": canonical_digest_sha256(payload),
        }
    )


def _rebuild_artifact(
    artifact: ActorPovReplayArtifactV1,
    **updates: object,
) -> ActorPovReplayArtifactV1:
    payload = artifact.model_dump(
        mode="python",
        exclude={"canonical_digest_sha256"},
    )
    payload.update(updates)
    return ActorPovReplayArtifactV1.model_validate(
        {
            **payload,
            "canonical_digest_sha256": canonical_digest_sha256(payload),
        }
    )


def _replace_tuple_item[T](
    values: tuple[T, ...], index: int, value: T
) -> tuple[T, ...]:
    mutable = list(values)
    mutable[index] = value
    return tuple(mutable)


def _all_mapping_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        for key, item in mapping.items():
            keys.add(str(key))
            keys.update(_all_mapping_keys(item))
    elif isinstance(value, (list, tuple)):
        sequence = cast(list[object] | tuple[object, ...], value)
        for item in sequence:
            keys.update(_all_mapping_keys(item))
    return keys


def test_export_slices_every_configured_active_actor_exactly(
    replay: ReplayArtifactV1,
) -> None:
    context = replay.header.context
    active_slots = tuple(
        row.global_slot for row in context.roster if row.configured_active
    )
    assert active_slots == (0, 1, 2, 5, 6)

    for slot in active_slots:
        artifact = export_actor_pov_replay_v1(replay, global_slot=slot)
        validate_actor_pov_replay_against_replay_v1(artifact, replay)
        content = artifact.content
        roster = context.roster[slot]
        catalog = context.static_mechanics_catalog
        assert content.selected_global_slot == slot
        assert content.selected_team_local_slot == roster.team_local_slot
        assert content.public_agent_id == roster.public_agent_id
        assert content.configured_team_id == roster.configured_team_id
        assert content.class_id == roster.class_id
        assert len(content.frames) == len(replay.frames)
        assert len(content.transitions) == len(replay.transitions)

        mapping = content.axis_mapping
        ally_slots = catalog.global_slot_by_actor_and_ally_observation_row[slot]
        enemy_slots = catalog.global_slot_by_actor_and_enemy_observation_row[slot]
        assert mapping.ally_observation_row_public_agent_id_by_id == tuple(
            context.roster[index].public_agent_id for index in ally_slots
        )
        assert mapping.enemy_observation_row_public_agent_id_by_id == tuple(
            context.roster[index].public_agent_id for index in enemy_slots
        )
        assert mapping.target_action_recipient_public_agent_id_by_id == (
            None,
            *mapping.ally_observation_row_public_agent_id_by_id,
            *mapping.enemy_observation_row_public_agent_id_by_id,
        )

        for source, frame in zip(replay.frames, content.frames, strict=True):
            observation = source.base_observation
            previous = observation.previous_timestep_actions
            lifecycle = observation.spawn_lifecycle
            assert frame.self_features == observation.self_features[slot]
            assert frame.ally_unit_features == observation.ally_unit_features[slot]
            assert frame.enemy_unit_features == observation.enemy_unit_features[slot]
            assert (
                frame.map_obstacle_features == (observation.map_obstacle_features[slot])
            )
            assert frame.objective_features == observation.objective_features[slot]
            assert frame.context_features == observation.context_features[slot]
            assert frame.ally_visibility_mask == observation.ally_visibility_mask[slot]
            assert (
                frame.enemy_visibility_mask == (observation.enemy_visibility_mask[slot])
            )
            assert (
                frame.previous_timestep_actions.ally_move_actions_one_hot
                == (previous.ally_previous_timestep_move_actions_one_hot[slot])
            )
            assert (
                frame.previous_timestep_actions.enemy_move_actions_one_hot
                == (previous.enemy_previous_timestep_move_actions_one_hot[slot])
            )
            assert (
                frame.previous_timestep_actions.ally_select_target_actions_one_hot
                == previous.ally_previous_timestep_select_target_actions_one_hot[slot]
            )
            assert (
                frame.previous_timestep_actions.enemy_select_target_actions_one_hot
                == previous.enemy_previous_timestep_select_target_actions_one_hot[slot]
            )
            assert (
                frame.previous_timestep_actions.ally_use_ultimate_actions_one_hot
                == previous.ally_previous_timestep_use_ultimate_actions_one_hot[slot]
            )
            assert (
                frame.previous_timestep_actions.enemy_use_ultimate_actions_one_hot
                == previous.enemy_previous_timestep_use_ultimate_actions_one_hot[slot]
            )
            assert (
                frame.spawn_lifecycle.spawn_pad_positions_by_team
                == (lifecycle.spawn_pad_positions_by_agent_by_team[slot])
            )
            assert (
                frame.spawn_lifecycle.spawn_shield_actual_durations_by_team
                == lifecycle.spawn_shield_actual_durations_by_agent_by_team[slot]
            )
            assert (
                frame.spawn_lifecycle.spawn_shield_configured_duration
                == (lifecycle.spawn_shield_configured_duration_by_agent[slot])
            )
            assert (
                frame.spawn_lifecycle.spawn_shield_speed
                == (lifecycle.spawn_shield_speed_by_agent[slot])
            )
            assert (
                frame.spawn_lifecycle.respawn_wave_period_step_count_by_team
                == lifecycle.respawn_wave_period_step_count_by_agent_by_team[slot]
            )
            assert (
                frame.spawn_lifecycle.respawn_wave_countdowns_by_team
                == (lifecycle.respawn_wave_countdowns_by_agent_by_team[slot])
            )
            assert (
                frame.spawn_lifecycle.active_mask_by_team
                == (lifecycle.active_mask_by_agent_by_team[slot])
            )
            assert (
                frame.spawn_lifecycle.alive_mask_by_team
                == (lifecycle.alive_mask_by_agent_by_team[slot])
            )
            assert frame.action_mask.move == source.action_mask.move_mask[slot]
            assert (
                frame.action_mask.select_target
                == (source.action_mask.select_target_mask[slot])
            )
            assert (
                frame.action_mask.use_ultimate
                == (source.action_mask.use_ultimate_mask[slot])
            )
            assert (
                frame.action_mask.select_target_use_ultimate_joint
                == (source.action_mask.select_target_use_ultimate_joint_mask[slot])
            )

        for source, transition in zip(
            replay.transitions,
            content.transitions,
            strict=True,
        ):
            acceptance = source.facts.action_acceptance_facts
            assert (
                transition.submitted_action.move
                == (acceptance.submitted_joint_action.move[slot])
            )
            assert (
                transition.submitted_action.select_target
                == (acceptance.submitted_joint_action.select_target[slot])
            )
            assert (
                transition.submitted_action.use_ultimate
                == (acceptance.submitted_joint_action.use_ultimate[slot])
            )
            assert (
                transition.accepted_action.move
                == (acceptance.accepted_joint_action.move[slot])
            )
            assert (
                transition.accepted_action.select_target
                == (acceptance.accepted_joint_action.select_target[slot])
            )
            assert (
                transition.accepted_action.use_ultimate
                == (acceptance.accepted_joint_action.use_ultimate[slot])
            )
            assert (
                transition.submitted_action_tuple_is_out_of_domain
                == (acceptance.submitted_action_tuple_is_out_of_domain_by_actor[slot])
            )
            assert (
                transition.in_domain_move_action_is_rejected
                == (acceptance.in_domain_move_action_is_rejected_by_actor[slot])
            )
            assert (
                transition.in_domain_combat_action_pair_is_rejected
                == (acceptance.in_domain_combat_action_pair_is_rejected_by_actor[slot])
            )
            assert transition.canonical_reward == source.canonical_reward_by_agent[slot]
            assert transition.terminated == source.terminated
            assert transition.truncated == source.truncated
            assert transition.public_end_reason == source.owning_task_end_reason
            assert tuple(cue.ordinal for cue in transition.cues) == tuple(
                range(len(transition.cues))
            )
            assert all(
                cue.cue_id == f"{transition.pov_transition_id}:cue:{cue.ordinal}"
                for cue in transition.cues
            )


@pytest.mark.parametrize("slot", [3, 4, 7, 8, 9])
def test_inactive_actor_export_rejects(
    replay: ReplayArtifactV1,
    slot: int,
) -> None:
    with pytest.raises(ValueError, match="configured-active"):
        export_actor_pov_replay_v1(replay, global_slot=slot)


@pytest.mark.parametrize("slot", [-1, 10, True])
def test_unbounded_or_bool_actor_slot_rejects(
    replay: ReplayArtifactV1,
    slot: object,
) -> None:
    with pytest.raises(ValueError, match="exact bounded integer"):
        export_actor_pov_replay_v1(replay, global_slot=slot)  # type: ignore[arg-type]


def test_exact_shared_obs_export_fails_closed() -> None:
    shared = captured_evaluation_trajectory(
        transition_count=1,
        execution_information_mode="shared_obs",
    )
    replay = _build_replay(shared)
    with pytest.raises(ValueError, match="unavailable for shared_obs"):
        export_actor_pov_replay_v1(replay, global_slot=0)


def test_submitted_out_of_domain_actions_remain_exact() -> None:
    neutral = neutral_action()
    submitted = Action(
        move=neutral.move.at[0].set(-7),
        select_target=neutral.select_target.at[0].set(99),
        use_ultimate=neutral.use_ultimate.at[0].set(-3),
    )
    trajectory = captured_evaluation_trajectory(
        transition_count=1,
        actions=(submitted,),
    )
    replay = _build_replay(trajectory)
    transition = export_actor_pov_replay_v1(
        replay,
        global_slot=0,
    ).content.transitions[0]
    assert transition.submitted_action.model_dump() == {
        "schema_id": "marl_battlegrounds.evaluation.actor_pov_submitted_action",
        "schema_version": 1,
        "move": -7,
        "select_target": 99,
        "use_ultimate": -3,
    }
    assert transition.accepted_action.move == 0
    assert transition.submitted_action_tuple_is_out_of_domain
    assert transition.cues[0].cue_type == "own_action_outcome"
    assert transition.cues[0].outcome == "rejected"


def test_team_b_spawn_axis_zero_is_own_team(
    replay: ReplayArtifactV1,
) -> None:
    artifact = export_actor_pov_replay_v1(replay, global_slot=5)
    content = artifact.content
    assert content.configured_team_id == 2
    assert content.selected_team_local_slot == 0
    assert content.axis_mapping.spawn_lifecycle_team_axis_name_by_id == (
        "Own Team",
        "Opponent Team",
    )
    for source, frame in zip(replay.frames, content.frames, strict=True):
        lifecycle = source.base_observation.spawn_lifecycle
        source_row = lifecycle.spawn_shield_actual_durations_by_agent_by_team[5]
        assert (
            frame.spawn_lifecycle.spawn_shield_actual_durations_by_team[0][0]
            == (source_row[0][0])
        )
        # Axis zero is actor-relative Own Team for Team B; it is not indexed by
        # configured_team_id - 1.
        assert content.axis_mapping.ally_observation_row_public_agent_id_by_id[0] == (
            content.public_agent_id
        )


def test_content_and_artifact_json_round_trip_and_disclosure_allowlist(
    replay: ReplayArtifactV1,
) -> None:
    artifact = export_actor_pov_replay_v1(replay, global_slot=0)
    content_bytes = canonical_actor_pov_content_json_bytes_v1(artifact.content)
    artifact_bytes = canonical_actor_pov_replay_json_bytes_v1(artifact)
    assert (
        ActorPovReplayContentV1.model_validate_json(content_bytes) == artifact.content
    )
    loaded = ActorPovReplayArtifactV1.model_validate_json(artifact_bytes)
    assert loaded == artifact
    validate_actor_pov_replay_artifact_v1(loaded)
    assert canonical_actor_pov_replay_json_bytes_v1(loaded) == artifact_bytes

    keys = _all_mapping_keys(artifact.model_dump(mode="python"))
    forbidden = {
        "snapshot",
        "events",
        "event_id",
        "event_type",
        "facts",
        "canonical_reward_by_team",
        "policy_assignments",
        "seed_protocol",
        "runtime_provenance",
        "wrapper_stack",
        "processing_status",
        "statistics",
        "global_analysis_snapshot",
    }
    assert keys.isdisjoint(forbidden)
    assert "source_transition_id" not in keys
    assert "source_event_ordinal" not in keys


def test_json_unknown_version_extra_field_and_python_lists_reject(
    replay: ReplayArtifactV1,
) -> None:
    artifact = export_actor_pov_replay_v1(replay, global_slot=0)
    future = artifact.model_dump(mode="json")
    future["schema_version"] = 2
    with pytest.raises(ValidationError):
        ActorPovReplayArtifactV1.model_validate(future)
    extra = artifact.model_dump(mode="python")
    extra["future_field"] = "forbidden"
    with pytest.raises(ValidationError):
        ActorPovReplayArtifactV1.model_validate(extra)
    frame = artifact.content.frames[0]
    frame_payload = frame.model_dump(mode="python")
    frame_payload["self_features"] = list(frame.self_features)
    with pytest.raises(ValidationError):
        ActorPovFrameV1.model_validate(frame_payload)


def test_stale_digest_and_recomputed_cue_tampering_reject(
    replay: ReplayArtifactV1,
) -> None:
    artifact = export_actor_pov_replay_v1(replay, global_slot=0)
    frame = artifact.content.frames[0]
    context_features = _replace_tuple_item(frame.context_features, 0, 1234.0)
    changed_frame = ActorPovFrameV1.model_validate(
        {**frame.model_dump(mode="python"), "context_features": context_features}
    )
    stale_payload = artifact.content.model_dump(mode="python")
    stale_payload["frames"] = (changed_frame, *artifact.content.frames[1:])
    with pytest.raises(ValidationError, match="digest"):
        ActorPovReplayContentV1.model_validate(stale_payload)

    transition = artifact.content.transitions[0]
    no_cues = ActorPovTransitionV1.model_validate(
        {**transition.model_dump(mode="python"), "cues": ()}
    )
    tampered_content = _rebuild_content(
        artifact.content,
        transitions=(no_cues, *artifact.content.transitions[1:]),
    )
    with pytest.raises(ValueError, match="authorized local rederivation"):
        validate_actor_pov_replay_content_v1(tampered_content)


def test_recomputed_simulator_gap_and_source_reference_tampering_reject(
    replay: ReplayArtifactV1,
) -> None:
    artifact = export_actor_pov_replay_v1(replay, global_slot=0)
    successor = artifact.content.frames[1]
    gapped = ActorPovFrameV1.model_validate(
        {
            **successor.model_dump(mode="python"),
            "simulator_step_count": successor.simulator_step_count + 5,
        }
    )
    with pytest.raises(ValidationError, match="epochs must be adjacent"):
        _rebuild_content(
            artifact.content,
            frames=(artifact.content.frames[0], gapped, *artifact.content.frames[2:]),
        )

    reference_payload = artifact.source_replay.model_dump(mode="python")
    reference_payload["context_digest_sha256"] = "f" * 64
    forged_reference = type(artifact.source_replay).model_validate(reference_payload)
    forged_artifact = _rebuild_artifact(
        artifact,
        source_replay=forged_reference,
    )
    validate_actor_pov_replay_artifact_v1(forged_artifact)
    with pytest.raises(ValueError, match="does not match its source replay"):
        validate_actor_pov_replay_against_replay_v1(forged_artifact, replay)


def test_hidden_snapshot_changes_preserve_authorized_content_bytes(
    replay: ReplayArtifactV1,
) -> None:
    selected_slot = 0
    source_frame = replay.frames[0]
    snapshot = source_frame.snapshot
    changed_health = _replace_tuple_item(
        snapshot.current_health,
        1,
        snapshot.current_health[1] + 0.125,
    )
    changed_snapshot = GlobalAnalysisSnapshotV1.model_validate(
        {**snapshot.model_dump(mode="python"), "current_health": changed_health}
    )
    changed_frame = EvaluationFrameV1.model_validate(
        {
            **source_frame.model_dump(mode="python"),
            "snapshot": changed_snapshot,
        }
    )
    hidden_variant = _rebuild_replay(
        replay,
        frames=(changed_frame, *replay.frames[1:]),
    )
    assert hidden_variant.canonical_digest_sha256 != replay.canonical_digest_sha256
    assert hidden_variant.frames[0].base_observation == (
        replay.frames[0].base_observation
    )

    first = export_actor_pov_replay_v1(replay, global_slot=selected_slot)
    second = export_actor_pov_replay_v1(hidden_variant, global_slot=selected_slot)
    assert first.content.canonical_digest_sha256 == (
        second.content.canonical_digest_sha256
    )
    assert canonical_actor_pov_content_json_bytes_v1(first.content) == (
        canonical_actor_pov_content_json_bytes_v1(second.content)
    )
    assert first.source_replay != second.source_replay
    assert first.canonical_digest_sha256 != second.canonical_digest_sha256
    assert canonical_actor_pov_replay_json_bytes_v1(first) != (
        canonical_actor_pov_replay_json_bytes_v1(second)
    )
    first_outer = first.model_dump(
        mode="python",
        exclude={"canonical_digest_sha256", "source_replay"},
    )
    second_outer = second.model_dump(
        mode="python",
        exclude={"canonical_digest_sha256", "source_replay"},
    )
    assert first_outer == second_outer


def test_hidden_other_actor_event_changes_preserve_authorized_content_bytes() -> None:
    neutral = neutral_action()
    hidden_rejection = Action(
        move=neutral.move.at[3].set(-7),
        select_target=neutral.select_target,
        use_ultimate=neutral.use_ultimate,
    )
    neutral_source = captured_evaluation_trajectory(
        transition_count=1,
        actions=(neutral,),
    )
    hidden_event_source = captured_evaluation_trajectory(
        transition_count=1,
        actions=(hidden_rejection,),
    )
    assert neutral_source.frames == hidden_event_source.frames
    assert neutral_source.transitions[0].events == ()
    assert tuple(
        event.event_type for event in hidden_event_source.transitions[0].events
    ) == ("action_rejected",)

    neutral_replay = _build_replay(neutral_source)
    hidden_event_replay = _build_replay(hidden_event_source)
    first = export_actor_pov_replay_v1(neutral_replay, global_slot=0)
    second = export_actor_pov_replay_v1(hidden_event_replay, global_slot=0)
    assert canonical_actor_pov_content_json_bytes_v1(first.content) == (
        canonical_actor_pov_content_json_bytes_v1(second.content)
    )
    assert first.content.canonical_digest_sha256 == (
        second.content.canonical_digest_sha256
    )
    assert first.source_replay != second.source_replay
    assert canonical_actor_pov_replay_json_bytes_v1(first) != (
        canonical_actor_pov_replay_json_bytes_v1(second)
    )


def test_zero_transition_prefix_and_horizon_completion_are_honest() -> None:
    zero = captured_evaluation_trajectory(transition_count=0)
    zero_pov = export_actor_pov_replay_v1(
        _build_replay(zero),
        global_slot=0,
    )
    assert len(zero_pov.content.frames) == 1
    assert zero_pov.content.transitions == ()
    assert zero_pov.content.completion.completion_state == "partial"

    horizon = captured_evaluation_trajectory(
        transition_count=1,
        expected_horizon=1,
    )
    horizon_replay = _build_replay(
        horizon,
        completion_state="complete",
        reason=None,
    )
    horizon_pov = export_actor_pov_replay_v1(horizon_replay, global_slot=0)
    assert horizon_pov.content.completion.completion_state == "complete"
    assert horizon_pov.content.completion.completion_bases == ("declared_horizon",)
    assert not horizon_pov.content.transitions[-1].terminated
    assert not horizon_pov.content.transitions[-1].truncated
    assert "episode_ended" not in {
        cue.cue_type for cue in horizon_pov.content.transitions[-1].cues
    }


def test_selected_self_topology_fails_closed_before_boolean_cues(
    replay: ReplayArtifactV1,
) -> None:
    frame = replay.frames[0]
    observation = frame.base_observation
    self_row = _replace_tuple_item(observation.self_features[0], 5, 0.5)
    self_features = _replace_tuple_item(observation.self_features, 0, self_row)
    changed_observation = BaseObservationV1.model_validate(
        {**observation.model_dump(mode="python"), "self_features": self_features}
    )
    changed_frame = EvaluationFrameV1.model_validate(
        {**frame.model_dump(mode="python"), "base_observation": changed_observation}
    )
    malformed = _rebuild_replay(
        replay,
        frames=(changed_frame, *replay.frames[1:]),
    )
    with pytest.raises(ValueError, match="ALIVE must be exactly"):
        export_actor_pov_replay_v1(malformed, global_slot=0)


def test_typed_models_do_not_reuse_full_replay_records() -> None:
    assert "snapshot" not in ActorPovFrameV1.model_fields
    assert "events" not in ActorPovTransitionV1.model_fields
    assert "facts" not in ActorPovTransitionV1.model_fields
    assert "context" not in ActorPovReplayContentV1.model_fields
    assert "processing_status" not in ActorPovReplayContentV1.model_fields
    assert "metric_report" not in ActorPovReplayArtifactV1.model_fields
    assert ActorPovFrameV1.model_fields["spawn_lifecycle"].annotation is (
        ActorPovSpawnLifecycleV1
    )
    assert ActorPovFrameV1.model_fields["spawn_lifecycle"].annotation is not (
        SpawnLifecycleObservationV1
    )
    assert canonical_json_bytes({"proof": "pov"}) == b'{"proof":"pov"}'
