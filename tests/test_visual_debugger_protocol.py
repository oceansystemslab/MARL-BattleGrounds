"""Focused validation tests for the browser debugger protocol."""

import json

import pytest
from pydantic import ValidationError
from scripts.dev.visual_debugger.protocol import (
    ActionTupleCardV1,
    ActorActionResultV1,
    ActorPovTargetActionCommandV1,
    BattlefieldPointerCommandV1,
    CandidateLegalityCardV1,
    CombatConfigurationV1,
    CommandRequestV1,
    ConfirmDiscardAndReplaceCommandV1,
    DebuggerCommandV1,
    ExitCommandV1,
    FinishAndReviewCommandV1,
    HudFrameV1,
    KeyboardCommandV1,
    LatestTransitionCardV1,
    MovementLegalityCardV1,
    PendingActionCardV1,
    RecordingStatusV1,
    ResetCommandV1,
    RetrySaveCommandV1,
    ReviewReplayCommandV1,
    RosterSelectionCommandV1,
    SaveAsCommandV1,
    ScenarioMetadataV1,
    ScenarioSwitchCommandV1,
    SetCombatConfigurationCommandV1,
    SetPresetCommandV1,
    SetViewCommandV1,
    TargetReferenceV1,
)

from marl_battlegrounds.core.types import NUM_MOVE_ACTIONS


def _movement_legalities() -> tuple[MovementLegalityCardV1, ...]:
    return tuple(
        MovementLegalityCardV1(move_action=move_action, available=True)
        for move_action in range(NUM_MOVE_ACTIONS)
    )


def test_hud_movement_legality_requires_exact_canonical_action_rows() -> None:
    no_target = TargetReferenceV1(disclosure="target_none", global_slot=None)
    pending = PendingActionCardV1(
        actor_global_slot=0,
        move_action=0,
        target_action=0,
        armed_lane=None,
        arm_origin=None,
        target=no_target,
        movement_mask_value=True,
        pair_mask_value=None,
        summary="no combat",
    )
    legalities = _movement_legalities()

    hud = HudFrameV1(
        roster_global_slots=(0,),
        controlled_global_slot=0,
        selected_global_slot=None,
        pending_submission_scope="joint_turn",
        pending_actions=(pending,),
        pending_action=pending,
        latest_transition=None,
        movement_legalities=legalities,
    )

    assert tuple(row.move_action for row in hud.movement_legalities) == tuple(
        range(NUM_MOVE_ACTIONS)
    )
    retired_scope = hud.model_dump(mode="python")
    retired_scope["pending_submission_scope"] = "controlled_actor"
    with pytest.raises(ValidationError, match=r"joint_turn.*scripted_playback"):
        HudFrameV1.model_validate(retired_scope)
    for invalid_rows in (
        legalities[:-1],
        tuple(reversed(legalities)),
        (*legalities[:-1], legalities[0]),
    ):
        with pytest.raises(ValidationError, match="canonical order"):
            HudFrameV1(
                roster_global_slots=(0,),
                controlled_global_slot=0,
                selected_global_slot=None,
                pending_submission_scope="joint_turn",
                pending_actions=(pending,),
                pending_action=pending,
                latest_transition=None,
                movement_legalities=invalid_rows,
            )
    with pytest.raises(ValidationError):
        MovementLegalityCardV1(
            move_action=NUM_MOVE_ACTIONS,
            available=True,
        )


@pytest.mark.parametrize(
    "command",
    (
        KeyboardCommandV1(
            key="R",
            repeat=False,
            shift_key=True,
        ),
        BattlefieldPointerCommandV1(
            world_x=3.5,
            world_y=6.0,
            button="primary",
            shift_key=True,
        ),
        RosterSelectionCommandV1(role="control", global_slot=1),
        ActorPovTargetActionCommandV1(target_action=6),
        ScenarioSwitchCommandV1(scenario_name="basic_support"),
        ResetCommandV1(),
        SetViewCommandV1(view_mode="pov"),
        SetPresetCommandV1(preset="analysis"),
        SetCombatConfigurationCommandV1(
            team_a_controller="manual",
            team_b_controller="scripted_tdm",
            execution_information_mode="shared_obs",
        ),
        SetCombatConfigurationCommandV1(
            team_a_controller="random_valid",
            team_b_controller="random_valid",
            execution_information_mode="no_shared_obs",
        ),
        FinishAndReviewCommandV1(),
        ReviewReplayCommandV1(),
        RetrySaveCommandV1(),
        SaveAsCommandV1(file_name="a.marlbg-replay.json"),
        ConfirmDiscardAndReplaceCommandV1(replacement=ResetCommandV1()),
        ConfirmDiscardAndReplaceCommandV1(
            replacement=ScenarioSwitchCommandV1(scenario_name="basic_support")
        ),
        ConfirmDiscardAndReplaceCommandV1(
            replacement=SetCombatConfigurationCommandV1(
                team_a_controller="scripted_tdm",
                team_b_controller="scripted_tdm",
                execution_information_mode="no_shared_obs",
            )
        ),
        ExitCommandV1(),
    ),
)
def test_command_request_round_trips_every_discriminated_variant(
    command: DebuggerCommandV1,
) -> None:
    request = CommandRequestV1(
        client_id="client-1",
        command_id="command-1",
        base_revision=0,
        command=command,
    )

    encoded = request.model_dump_json()
    decoded = CommandRequestV1.model_validate_json(encoded)

    assert decoded == request
    assert json.loads(encoded)["command"]["command_type"] == command.command_type


def test_combat_configuration_requires_both_symmetric_controller_values() -> None:
    configuration = CombatConfigurationV1(
        team_a_controller="random_valid",
        team_b_controller="scripted_tdm",
        execution_information_mode="shared_obs",
    )

    assert (
        CombatConfigurationV1.model_validate_json(configuration.model_dump_json())
        == configuration
    )
    for missing in ("team_a_controller", "team_b_controller"):
        payload = configuration.model_dump(mode="json")
        del payload[missing]
        with pytest.raises(ValidationError):
            CombatConfigurationV1.model_validate(payload)


def test_recording_status_enforces_exact_lifecycle_availability() -> None:
    capturing = RecordingStatusV1(
        lifecycle="recording",
        captured_transition_count=0,
        expected_transition_count=5,
        restart_fenced=False,
        finish_available=True,
        review_available=False,
        retry_available=False,
        save_as_available=False,
        discard_available=False,
    )
    failed = RecordingStatusV1(
        lifecycle="persistence_failed",
        captured_transition_count=2,
        expected_transition_count=5,
        completion_state="partial",
        completion_reason="user_finish_and_review",
        restart_fenced=True,
        finish_available=False,
        review_available=False,
        retry_available=True,
        save_as_available=True,
        discard_available=False,
        persistence_error_code="publication_failed",
    )
    saved = RecordingStatusV1(
        lifecycle="saved",
        captured_transition_count=5,
        expected_transition_count=5,
        completion_state="complete",
        restart_fenced=True,
        finish_available=False,
        review_available=True,
        retry_available=False,
        save_as_available=False,
        discard_available=False,
    )

    assert (
        RecordingStatusV1.model_validate_json(capturing.model_dump_json()) == capturing
    )
    assert RecordingStatusV1.model_validate_json(failed.model_dump_json()) == failed
    assert RecordingStatusV1.model_validate_json(saved.model_dump_json()) == saved

    for mutation, message in (
        ({"restart_fenced": True}, "restart_fenced"),
        ({"retry_available": True}, "availability"),
        ({"completion_state": "partial"}, "completion state"),
        ({"persistence_error_code": "publication_failed"}, "persistence_failed"),
    ):
        with pytest.raises(ValidationError, match=message):
            RecordingStatusV1.model_validate(
                {**capturing.model_dump(mode="python"), **mutation}
            )


@pytest.mark.parametrize(
    "file_name",
    (
        "../escape.marlbg-replay.json",
        "/tmp/escape.marlbg-replay.json",
        ".hidden.marlbg-replay.json",
        "missing.json",
        "bad name.marlbg-replay.json",
    ),
)
def test_recording_save_as_accepts_only_one_safe_replay_basename(
    file_name: str,
) -> None:
    with pytest.raises(ValidationError):
        SaveAsCommandV1(file_name=file_name)


@pytest.mark.parametrize(
    "payload",
    (
        {
            "schema_version": 2,
            "client_id": "client-1",
            "command_id": "command-1",
            "base_revision": 0,
            "command": {"command_type": "reset"},
        },
        {
            "schema_version": 1,
            "client_id": "client/1",
            "command_id": "command-1",
            "base_revision": 0,
            "command": {"command_type": "reset"},
        },
        {
            "schema_version": 1,
            "client_id": "client-1",
            "command_id": "command-1",
            "base_revision": True,
            "command": {"command_type": "reset"},
        },
        {
            "schema_version": 1,
            "client_id": "client-1",
            "command_id": "command-1",
            "base_revision": 0,
            "unexpected": "field",
            "command": {"command_type": "reset"},
        },
        {
            "schema_version": 1,
            "client_id": "client-1",
            "command_id": "command-1",
            "base_revision": 0,
            "command": {
                "command_type": "reset",
                "unexpected": "field",
            },
        },
        {
            "schema_version": 1,
            "client_id": "client-1",
            "command_id": "command-1",
            "base_revision": 0,
            "command": {"command_type": "unknown"},
        },
        {
            "schema_version": 1,
            "client_id": "client-1",
            "command_id": "command-1",
            "base_revision": 0,
            "command": {
                "command_type": "roster_selection",
                "role": "target",
                "global_slot": True,
            },
        },
    ),
)
def test_command_request_rejects_version_type_shape_and_extra_field_drift(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        CommandRequestV1.model_validate(payload)


def test_actor_pov_target_action_json_is_strict_bounded_and_global_slot_free() -> None:
    payload = json.dumps(
        {
            "schema_version": 1,
            "client_id": "client-1",
            "command_id": "pov-target-1",
            "base_revision": 0,
            "command": {
                "command_type": "actor_pov_target_action",
                "target_action": 6,
            },
        }
    )

    request = CommandRequestV1.model_validate_json(payload)
    assert isinstance(request.command, ActorPovTargetActionCommandV1)
    assert request.command.target_action == 6
    encoded_command = json.loads(request.model_dump_json())["command"]
    assert encoded_command == {
        "command_type": "actor_pov_target_action",
        "target_action": 6,
    }
    assert "global_slot" not in encoded_command

    for invalid_command in (
        {"command_type": "actor_pov_target_action", "target_action": -1},
        {"command_type": "actor_pov_target_action", "target_action": 11},
        {"command_type": "actor_pov_target_action", "target_action": True},
        {"command_type": "pov_target", "target_action": 6},
        {
            "command_type": "actor_pov_target_action",
            "target_action": 6,
            "global_slot": 5,
        },
    ):
        with pytest.raises(ValidationError):
            CommandRequestV1.model_validate_json(
                json.dumps(
                    {
                        "schema_version": 1,
                        "client_id": "client-1",
                        "command_id": "pov-target-invalid",
                        "base_revision": 0,
                        "command": invalid_command,
                    }
                )
            )


@pytest.mark.parametrize("coordinate", (float("nan"), float("inf"), -float("inf")))
def test_pointer_command_rejects_nonfinite_world_coordinates(
    coordinate: float,
) -> None:
    with pytest.raises(ValidationError):
        BattlefieldPointerCommandV1(
            world_x=coordinate,
            world_y=1.0,
            button="primary",
        )


def test_removed_movement_scale_command_is_rejected_at_the_wire_boundary() -> None:
    payload = json.dumps(
        {
            "schema_version": 1,
            "client_id": "client-1",
            "command_id": "scale-max",
            "base_revision": 0,
            "command": {
                "command_type": "set_movement_scale",
                "movement_scale": 1,
            },
        }
    )

    with pytest.raises(ValidationError):
        CommandRequestV1.model_validate_json(payload)


@pytest.mark.parametrize(
    "legacy_preset",
    ("presentation", "analysis", "technical", "debug"),
)
def test_legacy_preset_canonicalizes_to_analysis(legacy_preset: str) -> None:
    request = CommandRequestV1.model_validate_json(
        json.dumps(
            {
                "schema_version": 1,
                "client_id": "client-1",
                "command_id": "legacy-technical",
                "base_revision": 0,
                "command": {
                    "command_type": "set_preset",
                    "preset": legacy_preset,
                },
            }
        )
    )

    assert isinstance(request.command, SetPresetCommandV1)
    assert request.command.preset == "analysis"


def test_scenario_exposes_movement_scale_as_read_only_recorded_truth() -> None:
    payload = {
        "name": "arena_5v5",
        "title": "Arena",
        "description": "Interactive arena.",
        "mode": "interactive",
        "audience": "researcher",
        "ordinary_movement_distance_scale": 1.0,
        "completed_frame_count": 0,
        "frame_count": 0,
        "next_frame_index": None,
        "next_frame_label": None,
        "next_frame_description": None,
        "script_complete": False,
    }

    metadata = ScenarioMetadataV1.model_validate(payload)

    assert metadata.ordinary_movement_distance_scale == 1.0
    with pytest.raises(ValidationError, match=r"exactly 1\.0"):
        ScenarioMetadataV1.model_validate(
            payload | {"ordinary_movement_distance_scale": 0.5}
        )
    for obsolete_field in (
        "movement_scale_minimum",
        "movement_scale_maximum",
        "movement_scale_step",
        "scenario_default_movement_scale",
        "movement_scale_overridden",
    ):
        with pytest.raises(ValidationError):
            ScenarioMetadataV1.model_validate(payload | {obsolete_field: 1.0})


def test_scenario_metadata_json_accepts_integer_and_rejects_coercion() -> None:
    payload = {
        "name": "arena_5v5",
        "title": "Arena",
        "description": "Interactive arena.",
        "mode": "interactive",
        "audience": "researcher",
        "ordinary_movement_distance_scale": 1,
        "completed_frame_count": 0,
        "frame_count": 0,
        "next_frame_index": None,
        "next_frame_label": None,
        "next_frame_description": None,
        "script_complete": False,
    }

    metadata = ScenarioMetadataV1.model_validate_json(json.dumps(payload))

    assert metadata.ordinary_movement_distance_scale == 1.0
    assert type(metadata.ordinary_movement_distance_scale) is float
    for invalid in ("1", None, True, [1], {"value": 1}):
        with pytest.raises(ValidationError):
            ScenarioMetadataV1.model_validate_json(
                json.dumps(payload | {"ordinary_movement_distance_scale": invalid})
            )
    for mutation in (
        {"unexpected": True},
        {"script_complete": None},
        {"completed_frame_count": False},
    ):
        with pytest.raises(ValidationError):
            ScenarioMetadataV1.model_validate_json(json.dumps(payload | mutation))


def test_capability_token_is_not_part_of_the_request_body_schema() -> None:
    schema_text = json.dumps(CommandRequestV1.model_json_schema())
    assert "capability" not in schema_text.lower()
    assert "token" not in schema_text.lower()


def test_target_reference_disclosure_is_structural() -> None:
    assert (
        TargetReferenceV1(
            disclosure="public",
            global_slot=5,
        ).global_slot
        == 5
    )
    assert (
        TargetReferenceV1(
            disclosure="redacted",
            global_slot=None,
        ).global_slot
        is None
    )
    with pytest.raises(ValidationError, match="require global_slot"):
        TargetReferenceV1(disclosure="public", global_slot=None)
    with pytest.raises(ValidationError, match="must omit global_slot"):
        TargetReferenceV1(disclosure="redacted", global_slot=5)


def test_hud_candidate_legality_requires_target_none_and_exact_roster() -> None:
    no_target = TargetReferenceV1(
        disclosure="target_none",
        global_slot=None,
    )
    pending = PendingActionCardV1(
        actor_global_slot=0,
        move_action=0,
        target_action=0,
        armed_lane=0,
        arm_origin="automatic",
        target=no_target,
        movement_mask_value=True,
        pair_mask_value=True,
        summary="pending",
    )
    target_none = CandidateLegalityCardV1(
        target_action=0,
        target=no_target,
        lane_0_available=True,
        lane_1_available=False,
        basic_available=False,
        ultimate_available=False,
    )
    public_actor = CandidateLegalityCardV1(
        target_action=1,
        target=TargetReferenceV1(
            disclosure="public",
            global_slot=0,
        ),
        lane_0_available=False,
        lane_1_available=True,
        basic_available=False,
        ultimate_available=True,
    )

    hud = HudFrameV1(
        roster_global_slots=(0,),
        controlled_global_slot=0,
        selected_global_slot=None,
        pending_submission_scope="joint_turn",
        pending_actions=(pending,),
        pending_action=pending,
        latest_transition=None,
        movement_legalities=_movement_legalities(),
        candidate_legalities=(target_none, public_actor),
    )
    assert hud.candidate_legalities == (target_none, public_actor)

    with pytest.raises(ValidationError, match="canonical target-none"):
        CandidateLegalityCardV1(
            target_action=0,
            target=no_target,
            lane_0_available=True,
            lane_1_available=False,
            basic_available=True,
            ultimate_available=False,
        )
    with pytest.raises(ValidationError, match="exactly one target-none"):
        HudFrameV1(
            roster_global_slots=(0,),
            controlled_global_slot=0,
            selected_global_slot=None,
            pending_submission_scope="joint_turn",
            pending_actions=(pending,),
            pending_action=pending,
            latest_transition=None,
            movement_legalities=_movement_legalities(),
            candidate_legalities=(public_actor,),
        )
    with pytest.raises(ValidationError, match="exactly match the roster"):
        HudFrameV1(
            roster_global_slots=(0, 1),
            controlled_global_slot=0,
            selected_global_slot=None,
            pending_submission_scope="joint_turn",
            pending_actions=(
                pending,
                pending.model_copy(update={"actor_global_slot": 1}),
            ),
            pending_action=pending,
            latest_transition=None,
            movement_legalities=_movement_legalities(),
            candidate_legalities=(target_none, public_actor),
        )
    with pytest.raises(ValidationError, match="target actions must be unique"):
        HudFrameV1(
            roster_global_slots=(0,),
            controlled_global_slot=0,
            selected_global_slot=None,
            pending_submission_scope="joint_turn",
            pending_actions=(pending,),
            pending_action=pending,
            latest_transition=None,
            movement_legalities=_movement_legalities(),
            candidate_legalities=(target_none, public_actor, public_actor),
        )
    with pytest.raises(ValidationError, match="public target slots must be unique"):
        HudFrameV1(
            roster_global_slots=(0,),
            controlled_global_slot=0,
            selected_global_slot=None,
            pending_submission_scope="joint_turn",
            pending_actions=(pending,),
            pending_action=pending,
            latest_transition=None,
            movement_legalities=_movement_legalities(),
            candidate_legalities=(
                target_none,
                public_actor,
                CandidateLegalityCardV1(
                    target_action=2,
                    target=public_actor.target,
                    lane_0_available=True,
                    lane_1_available=True,
                    basic_available=True,
                    ultimate_available=True,
                ),
            ),
        )

    team_b_pending = pending.model_copy(
        update={"actor_global_slot": 5},
    )
    team_b_self = CandidateLegalityCardV1(
        target_action=1,
        target=TargetReferenceV1(disclosure="public", global_slot=5),
        lane_0_available=True,
        lane_1_available=False,
        basic_available=True,
        ultimate_available=False,
    )
    team_b_team_a_target = CandidateLegalityCardV1(
        target_action=6,
        target=TargetReferenceV1(disclosure="public", global_slot=0),
        lane_0_available=False,
        lane_1_available=True,
        basic_available=False,
        ultimate_available=True,
    )
    team_b_hud = HudFrameV1(
        roster_global_slots=(0, 5),
        controlled_global_slot=5,
        selected_global_slot=None,
        pending_submission_scope="joint_turn",
        pending_actions=(pending, team_b_pending),
        pending_action=team_b_pending,
        latest_transition=None,
        movement_legalities=_movement_legalities(),
        candidate_legalities=(
            target_none,
            team_b_self,
            team_b_team_a_target,
        ),
    )
    assert tuple(
        candidate.target_action for candidate in team_b_hud.candidate_legalities
    ) == (0, 1, 6)


def test_hud_pending_scope_requires_exact_order_alias_and_playback_label() -> None:
    no_target = TargetReferenceV1(disclosure="target_none", global_slot=None)
    pending_0 = PendingActionCardV1(
        actor_global_slot=0,
        move_action=0,
        target_action=0,
        armed_lane=0,
        arm_origin="automatic",
        target=no_target,
        movement_mask_value=True,
        pair_mask_value=True,
        summary="g0",
    )
    pending_1 = pending_0.model_copy(update={"actor_global_slot": 1, "summary": "g1"})

    joint = HudFrameV1(
        roster_global_slots=(0, 1),
        controlled_global_slot=0,
        selected_global_slot=None,
        pending_submission_scope="joint_turn",
        pending_actions=(pending_0, pending_1),
        pending_action=pending_0,
        latest_transition=None,
        movement_legalities=_movement_legalities(),
    )
    assert (
        tuple(pending.actor_global_slot for pending in joint.pending_actions)
        == joint.roster_global_slots
    )

    with pytest.raises(ValidationError, match="submission scope"):
        HudFrameV1(
            roster_global_slots=(0, 1),
            controlled_global_slot=0,
            selected_global_slot=None,
            pending_submission_scope="joint_turn",
            pending_actions=(pending_0,),
            pending_action=pending_0,
            latest_transition=None,
            movement_legalities=_movement_legalities(),
        )
    with pytest.raises(ValidationError, match="must be unique"):
        HudFrameV1(
            roster_global_slots=(0, 1),
            controlled_global_slot=0,
            selected_global_slot=None,
            pending_submission_scope="joint_turn",
            pending_actions=(pending_0, pending_0),
            pending_action=pending_0,
            latest_transition=None,
            movement_legalities=_movement_legalities(),
        )
    with pytest.raises(ValidationError, match="must equal"):
        HudFrameV1(
            roster_global_slots=(0, 1),
            controlled_global_slot=0,
            selected_global_slot=None,
            pending_submission_scope="joint_turn",
            pending_actions=(pending_0, pending_1),
            pending_action=pending_0.model_copy(update={"summary": "stale"}),
            latest_transition=None,
            movement_legalities=_movement_legalities(),
        )
    with pytest.raises(ValidationError, match="labels must match"):
        HudFrameV1(
            roster_global_slots=(0,),
            controlled_global_slot=0,
            selected_global_slot=None,
            pending_submission_scope="scripted_playback",
            pending_actions=(pending_0,),
            pending_action=pending_0,
            latest_transition=None,
            movement_legalities=_movement_legalities(),
        )

    playback = pending_0.model_copy(update={"label": "PLAYBACK / INSPECTION ONLY"})
    assert (
        HudFrameV1(
            roster_global_slots=(0,),
            controlled_global_slot=0,
            selected_global_slot=None,
            pending_submission_scope="scripted_playback",
            pending_actions=(playback,),
            pending_action=playback,
            latest_transition=None,
            movement_legalities=_movement_legalities(),
        ).pending_action.label
        == "PLAYBACK / INSPECTION ONLY"
    )


def test_hud_rejects_public_action_endpoints_absent_from_authorized_roster() -> None:
    no_target = TargetReferenceV1(
        disclosure="target_none",
        global_slot=None,
    )
    hidden_target = TargetReferenceV1(
        disclosure="public",
        global_slot=5,
    )
    submitted = ActionTupleCardV1(
        move_action=0,
        target_action=6,
        use_ultimate_action=0,
        target=hidden_target,
        summary="submitted",
    )
    accepted = ActionTupleCardV1(
        move_action=0,
        target_action=0,
        use_ultimate_action=0,
        target=no_target,
        summary="accepted",
    )
    latest = LatestTransitionCardV1(
        transition_id=1,
        submission_kind="interactive",
        actors=(
            ActorActionResultV1(
                actor_global_slot=0,
                submitted=submitted,
                accepted=accepted,
                movement_mask_value=True,
                pair_mask_value=False,
                movement_accepted=True,
                combat_result="rejected",
            ),
        ),
    )
    pending = PendingActionCardV1(
        actor_global_slot=0,
        move_action=0,
        target_action=0,
        armed_lane=0,
        arm_origin="automatic",
        target=no_target,
        movement_mask_value=True,
        pair_mask_value=True,
        summary="pending",
    )

    with pytest.raises(ValidationError, match="targets must occur in the roster"):
        HudFrameV1(
            roster_global_slots=(0,),
            controlled_global_slot=0,
            selected_global_slot=None,
            pending_submission_scope="joint_turn",
            pending_actions=(pending,),
            pending_action=pending,
            latest_transition=latest,
            movement_legalities=_movement_legalities(),
        )
