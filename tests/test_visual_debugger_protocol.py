"""Focused validation tests for the browser debugger protocol."""

import json

import pytest
from pydantic import ValidationError
from scripts.dev.visual_debugger.protocol import (
    ActionTupleCardV1,
    ActorActionResultV1,
    BattlefieldPointerCommandV1,
    CandidateLegalityCardV1,
    CommandRequestV1,
    DebuggerCommandV1,
    ExitCommandV1,
    HudFrameV1,
    KeyboardCommandV1,
    LatestTransitionCardV1,
    PendingActionCardV1,
    ResetCommandV1,
    RosterSelectionCommandV1,
    ScenarioSwitchCommandV1,
    SetPresetCommandV1,
    SetViewCommandV1,
    TargetReferenceV1,
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
        ScenarioSwitchCommandV1(scenario_name="basic_support"),
        ResetCommandV1(),
        SetViewCommandV1(view_mode="pov"),
        SetPresetCommandV1(preset="analysis"),
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


def test_hud_candidate_legality_requires_target_none_roster_and_actor_mapping() -> None:
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
    )
    public_actor = CandidateLegalityCardV1(
        target_action=1,
        target=TargetReferenceV1(
            disclosure="public",
            global_slot=0,
        ),
        lane_0_available=False,
        lane_1_available=True,
    )

    hud = HudFrameV1(
        roster_global_slots=(0,),
        controlled_global_slot=0,
        selected_global_slot=None,
        pending_action=pending,
        latest_transition=None,
        candidate_legalities=(target_none, public_actor),
    )
    assert hud.candidate_legalities == (target_none, public_actor)

    with pytest.raises(ValidationError, match="exactly one target-none"):
        HudFrameV1(
            roster_global_slots=(0,),
            controlled_global_slot=0,
            selected_global_slot=None,
            pending_action=pending,
            latest_transition=None,
            candidate_legalities=(public_actor,),
        )
    with pytest.raises(ValidationError, match="exactly match the roster"):
        HudFrameV1(
            roster_global_slots=(0, 1),
            controlled_global_slot=0,
            selected_global_slot=None,
            pending_action=pending,
            latest_transition=None,
            candidate_legalities=(target_none, public_actor),
        )
    with pytest.raises(ValidationError, match="target actions must be unique"):
        HudFrameV1(
            roster_global_slots=(0,),
            controlled_global_slot=0,
            selected_global_slot=None,
            pending_action=pending,
            latest_transition=None,
            candidate_legalities=(target_none, public_actor, public_actor),
        )
    with pytest.raises(ValidationError, match="public target slots must be unique"):
        HudFrameV1(
            roster_global_slots=(0,),
            controlled_global_slot=0,
            selected_global_slot=None,
            pending_action=pending,
            latest_transition=None,
            candidate_legalities=(
                target_none,
                public_actor,
                CandidateLegalityCardV1(
                    target_action=2,
                    target=public_actor.target,
                    lane_0_available=True,
                    lane_1_available=True,
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
    )
    team_b_team_a_target = CandidateLegalityCardV1(
        target_action=6,
        target=TargetReferenceV1(disclosure="public", global_slot=0),
        lane_0_available=False,
        lane_1_available=True,
    )
    team_b_hud = HudFrameV1(
        roster_global_slots=(0, 5),
        controlled_global_slot=5,
        selected_global_slot=None,
        pending_action=team_b_pending,
        latest_transition=None,
        candidate_legalities=(
            target_none,
            team_b_self,
            team_b_team_a_target,
        ),
    )
    assert tuple(
        candidate.target_action for candidate in team_b_hud.candidate_legalities
    ) == (0, 1, 6)

    with pytest.raises(ValidationError, match="actor-relative target mapping"):
        HudFrameV1(
            roster_global_slots=(0, 5),
            controlled_global_slot=5,
            selected_global_slot=None,
            pending_action=team_b_pending,
            latest_transition=None,
            candidate_legalities=(
                target_none,
                team_b_self,
                team_b_team_a_target.model_copy(update={"target_action": 2}),
            ),
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
            pending_action=pending,
            latest_transition=latest,
        )
