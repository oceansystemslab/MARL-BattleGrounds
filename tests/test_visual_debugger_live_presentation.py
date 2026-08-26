"""Focused live packaging and locked-service presentation proofs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from inspect import signature
from pathlib import Path
from threading import Event
from typing import cast

import pytest
import scripts.dev.visual_debugger.live_presentation as live_presentation_module
import scripts.dev.visual_debugger.service as service_module
from scripts.dev.visual_debugger.live_presentation import (
    build_live_no_shared_obs_authorized_presentation_v1,
    build_live_oracle_authorized_presentation_v1,
    build_live_researcher_space_v1,
)
from scripts.dev.visual_debugger.presentation_protocol import (
    LiveEditableDraftInspectionV1,
    LiveNoSharedObsAuthorizedPresentationFrameV1,
    LiveOracleAuthorizedPresentationFrameV1,
    LiveResearcherEditableDraftInspectionV1,
    LiveScriptedPlaybackInspectionV1,
    PresentationResourceResultV1,
)
from scripts.dev.visual_debugger.protocol import (
    ActorPovLiveDebuggerFrameV2,
    CommandResponseV2,
    FinishAndReviewCommandV1,
    KeyboardCommandV1,
    ResearcherLiveDebuggerFrameV2,
    ResetCommandV1,
    SetViewCommandV1,
)
from tests.test_visual_debugger_service import (
    _recording_service,  # pyright: ignore[reportPrivateUsage]
    _request,  # pyright: ignore[reportPrivateUsage]
    _service,  # pyright: ignore[reportPrivateUsage]
)

from marl_battlegrounds.evaluation.metrics import EvaluationTransitionViewV1
from marl_battlegrounds.evaluation.models import (
    EvaluationEpisodeContextV1,
    EvaluationFrameV1,
)
from marl_battlegrounds.evaluation.pov import (
    build_actor_pov_adjacent_transition_slice_v1,
    build_actor_pov_current_slice_v1,
)
from marl_battlegrounds.rendering.evaluation_adapter import build_visual_event_batch_v2


def _step_once(service: service_module.DebuggerService) -> None:
    result = service.apply_command(
        _request(
            "presentation-step",
            base_revision=service.revision,
            command=KeyboardCommandV1(key=" "),
        )
    )
    assert result.outcome == "response"


def _switch_to_pov(service: service_module.DebuggerService) -> None:
    result = service.apply_command(
        _request(
            "presentation-pov",
            base_revision=service.revision,
            command=SetViewCommandV1(view_mode="pov"),
        )
    )
    assert result.outcome == "response"


@pytest.mark.parametrize("view_mode", ("researcher", "pov"))
def test_live_frame_zero_presentation_is_exact_and_repeatable(view_mode: str) -> None:
    service = _service()
    if view_mode == "pov":
        _switch_to_pov(service)

    before_session = service.session
    before_raw = service.current_frame()
    before_revision = service.revision
    before_observer_count = service.evaluation_validated_transition_count
    before_command_count = service.command_cache_size
    before_observer = service._evaluation_observer  # pyright: ignore[reportPrivateUsage]
    before_command_records = dict(
        service._command_records  # pyright: ignore[reportPrivateUsage]
    )
    before_shutting_down = service.shutting_down
    before_faulted = service.faulted

    first = service.current_presentation()
    second = service.current_presentation()

    assert type(first) is PresentationResourceResultV1
    assert first.outcome == second.outcome == "response"
    assert type(first.payload) is type(second.payload)
    assert isinstance(
        first.payload,
        (
            LiveOracleAuthorizedPresentationFrameV1,
            LiveNoSharedObsAuthorizedPresentationFrameV1,
        ),
    )
    assert isinstance(
        second.payload,
        (
            LiveOracleAuthorizedPresentationFrameV1,
            LiveNoSharedObsAuthorizedPresentationFrameV1,
        ),
    )
    payload = first.payload
    assert payload.model_dump_json() == second.payload.model_dump_json()
    assert payload.source.source_frame_index == 0
    assert payload.latest_events is None
    assert payload.latest_transition is None
    assert service.session is before_session
    assert service.current_frame() is before_raw
    assert service.revision == before_revision
    assert service.evaluation_validated_transition_count == before_observer_count
    assert service.command_cache_size == before_command_count
    assert service._evaluation_observer is before_observer  # pyright: ignore[reportPrivateUsage]
    assert service._command_records == before_command_records  # pyright: ignore[reportPrivateUsage]
    assert service.shutting_down is before_shutting_down
    assert service.faulted is before_faulted

    if view_mode == "researcher":
        assert type(payload) is LiveOracleAuthorizedPresentationFrameV1
        assert payload.technical_frame.incoming_transition_id is None
    else:
        assert type(payload) is LiveNoSharedObsAuthorizedPresentationFrameV1
        assert payload.technical_frame.incoming_recipient_transition_id is None
        local_encoded = payload.model_dump_json(exclude={"researcher_space"})
        assert ":frame:0" in local_encoded
        assert "oracle_" not in local_encoded
        researcher = payload.researcher_space
        assert researcher.latest_transition is None
        assert (
            researcher.selected_public_agent_id
            == payload.source.source_recipient_public_agent_id
        )
        assert type(researcher.pending_inspection) is (
            LiveResearcherEditableDraftInspectionV1
        )
        assert researcher.pending_inspection.submission_scope == "joint_turn"
        assert all(
            not hasattr(row, "target_anchor")
            for row in researcher.pending_inspection.draft.decision_mask.target_actions
        )


def test_live_nonzero_oracle_and_no_shared_enter_exact_same_transition() -> None:
    service = _service()
    _step_once(service)
    incoming = service.session.incoming_evaluation_view
    assert incoming is not None

    oracle_result = service.current_presentation()
    assert oracle_result.outcome == "response"
    assert type(oracle_result.payload) is LiveOracleAuthorizedPresentationFrameV1
    oracle = oracle_result.payload
    assert oracle.source.source_frame_index == 1
    assert oracle.latest_events is not None
    assert oracle.latest_transition is not None
    assert (
        oracle.latest_transition.incoming_transition_id
        == incoming.transition.transition_id
    )
    assert oracle.latest_transition.incoming_successor_frame_id == (
        incoming.successor_frame.frame_id
    )
    assert oracle.technical_frame.incoming_transition_id == (
        incoming.transition.transition_id
    )

    _switch_to_pov(service)
    pov_result = service.current_presentation()
    assert pov_result.outcome == "response"
    assert type(pov_result.payload) is LiveNoSharedObsAuthorizedPresentationFrameV1
    pov = pov_result.payload
    assert pov.source.source_frame_index == 1
    assert pov.latest_events is not None
    assert pov.latest_transition is not None
    assert pov.latest_transition.incoming_transition_index == 0
    assert pov.latest_transition.incoming_successor_frame_id == (
        pov.source.source_recipient_frame_id
    )
    assert len(pov.latest_transition.action_rows) == 1
    assert (
        pov.latest_transition.action_rows[0].actor_public_agent_id
        == pov.source.source_recipient_public_agent_id
    )
    assert "oracle_" not in pov.model_dump_json(exclude={"researcher_space"})
    assert pov.researcher_space.latest_transition == oracle.latest_transition
    assert (
        pov.researcher_space.technical_frame.incoming_transition_id
        == incoming.transition.transition_id
    )


@pytest.mark.parametrize("view_mode", ("researcher", "pov"))
def test_live_draft_edit_changes_only_revision_scoped_draft_truth(
    view_mode: str,
) -> None:
    service = _service()
    if view_mode == "pov":
        _switch_to_pov(service)
    before = service.current_presentation()
    assert isinstance(
        before.payload,
        (
            LiveOracleAuthorizedPresentationFrameV1,
            LiveNoSharedObsAuthorizedPresentationFrameV1,
        ),
    )
    result = service.apply_command(
        _request(
            f"{view_mode}-draft-east",
            base_revision=service.revision,
            command=KeyboardCommandV1(key="d"),
        )
    )
    assert result.outcome == "response"
    after = service.current_presentation()
    assert type(after.payload) is type(before.payload)
    assert isinstance(
        after.payload,
        (
            LiveOracleAuthorizedPresentationFrameV1,
            LiveNoSharedObsAuthorizedPresentationFrameV1,
        ),
    )
    assert after.payload.current_endpoint == before.payload.current_endpoint
    assert after.payload.latest_events == before.payload.latest_events
    assert after.payload.latest_transition == before.payload.latest_transition
    assert after.payload.technical_frame == before.payload.technical_frame
    assert after.payload.source.source_revision == (
        before.payload.source.source_revision + 1
    )
    assert after.payload.source.source_authority_epoch == (
        before.payload.source.source_authority_epoch + 1
    )
    assert after.payload.source.source_authorized_endpoint_digest_sha256 == (
        before.payload.source.source_authorized_endpoint_digest_sha256
    )
    before_inspection = before.payload.live_inspection.inspection
    after_inspection = after.payload.live_inspection.inspection
    assert type(before_inspection) is LiveEditableDraftInspectionV1
    assert type(after_inspection) is LiveEditableDraftInspectionV1
    assert after_inspection.draft != before_inspection.draft
    assert after_inspection.draft.draft_action.move_action != (
        before_inspection.draft.draft_action.move_action
    )


@pytest.mark.parametrize("view_mode", ("researcher", "pov"))
def test_scripted_live_presentation_is_explicitly_inspection_only(
    view_mode: str,
) -> None:
    service = _service("basic_support")
    if view_mode == "pov":
        _switch_to_pov(service)
    raw = service.current_frame()
    assert raw.hud.pending_submission_scope == "scripted_playback"

    result = service.current_presentation()

    assert result.outcome == "response"
    assert isinstance(
        result.payload,
        (
            LiveOracleAuthorizedPresentationFrameV1,
            LiveNoSharedObsAuthorizedPresentationFrameV1,
        ),
    )
    inspection = result.payload.live_inspection.inspection
    assert type(inspection) is LiveScriptedPlaybackInspectionV1
    assert inspection.submission_scope == "scripted_playback"
    assert inspection.editable_draft_available is False
    assert inspection.advance_semantics == "registered_script_frame"
    encoded = result.payload.model_dump_json()
    assert '"draft_action"' not in encoded
    assert '"decision_mask"' not in encoded


@pytest.mark.parametrize(
    ("target_action", "raw_armed_lane", "expected"),
    (
        (0, None, "none"),
        (0, 0, "none"),
        (1, 0, "basic"),
        (0, 1, "ultimate"),
    ),
)
def test_live_draft_lane_translation_preserves_canonical_action_semantics(
    target_action: int,
    raw_armed_lane: int | None,
    expected: str,
) -> None:
    assert (
        live_presentation_module._draft_lane_v1(  # pyright: ignore[reportPrivateUsage]
            target_action=target_action,
            raw_armed_lane=raw_armed_lane,
        )
        == expected
    )


@pytest.mark.parametrize("view_mode", ("researcher", "pov"))
def test_live_editable_draft_submit_is_observed_once_in_latest_transition(
    view_mode: str,
) -> None:
    service = _service()
    if view_mode == "pov":
        _switch_to_pov(service)
    before = service.current_presentation()
    assert isinstance(
        before.payload,
        (
            LiveOracleAuthorizedPresentationFrameV1,
            LiveNoSharedObsAuthorizedPresentationFrameV1,
        ),
    )
    before_inspection = before.payload.live_inspection.inspection
    assert type(before_inspection) is LiveEditableDraftInspectionV1
    draft = before_inspection.draft
    assert draft.current_simulator_step_count == 0
    assert draft.draft_action.target_action == 0
    assert draft.draft_action.armed_lane == "none"

    submit = _request(
        f"{view_mode}-presentation-submit",
        base_revision=service.revision,
        command=KeyboardCommandV1(key=" "),
    )
    first = service.apply_command(submit)
    duplicate = service.apply_command(submit)
    assert first.outcome == duplicate.outcome == "response"
    assert type(first.payload) is CommandResponseV2
    assert type(duplicate.payload) is CommandResponseV2
    assert first.payload.result == "applied"
    assert duplicate.payload.result == "duplicate"
    assert duplicate.payload.frame is first.payload.frame
    assert service.session.current_evaluation_frame.frame_index == 1

    after = service.current_presentation()
    assert type(after.payload) is type(before.payload)
    assert isinstance(
        after.payload,
        (
            LiveOracleAuthorizedPresentationFrameV1,
            LiveNoSharedObsAuthorizedPresentationFrameV1,
        ),
    )
    assert after.payload.source.source_frame_index == 1
    assert after.payload.latest_transition is not None
    action_row = next(
        row
        for row in after.payload.latest_transition.action_rows
        if row.actor_public_agent_id == draft.actor_public_agent_id
    )
    assert action_row.submitted_action.move_action == draft.draft_action.move_action
    assert action_row.submitted_action.target_action == draft.draft_action.target_action
    assert action_row.submitted_action.use_ultimate_action == 0
    after_inspection = after.payload.live_inspection.inspection
    assert type(after_inspection) is LiveEditableDraftInspectionV1
    assert after_inspection.draft.current_simulator_step_count == 1


@pytest.mark.parametrize("view_mode", ("researcher", "pov"))
def test_scripted_inspection_advances_only_the_registered_script(
    view_mode: str,
) -> None:
    service = _service("basic_support")
    if view_mode == "pov":
        _switch_to_pov(service)
    before = service.current_presentation()
    assert isinstance(
        before.payload,
        (
            LiveOracleAuthorizedPresentationFrameV1,
            LiveNoSharedObsAuthorizedPresentationFrameV1,
        ),
    )
    assert type(before.payload.live_inspection.inspection) is (
        LiveScriptedPlaybackInspectionV1
    )

    advanced = service.apply_command(
        _request(
            f"{view_mode}-scripted-presentation-advance",
            base_revision=service.revision,
            command=KeyboardCommandV1(key="n"),
        )
    )
    assert advanced.outcome == "response"
    after = service.current_presentation()
    assert type(after.payload) is type(before.payload)
    assert isinstance(
        after.payload,
        (
            LiveOracleAuthorizedPresentationFrameV1,
            LiveNoSharedObsAuthorizedPresentationFrameV1,
        ),
    )
    assert after.payload.source.source_frame_index == 1
    assert after.payload.latest_transition is not None
    assert type(after.payload.live_inspection.inspection) is (
        LiveScriptedPlaybackInspectionV1
    )
    encoded = after.payload.model_dump_json()
    assert '"draft_action"' not in encoded
    assert '"decision_mask"' not in encoded


def test_live_presentation_is_read_only_for_active_recorder_on_success_and_error(
    tmp_path: Path,
) -> None:
    service, recorder = _recording_service(tmp_path)
    submitted = service.apply_command(
        _request(
            "recorded-presentation-submit",
            base_revision=service.revision,
            command=KeyboardCommandV1(key=" "),
        )
    )
    assert submitted.outcome == "response"
    before_raw = service.current_frame()
    before_session = service.session
    before_revision = service.revision
    before_commands = service.command_cache_size
    before_status = recorder.status
    before_lifecycle = recorder.lifecycle
    before_observer_lifecycle = recorder.observer_lifecycle_state
    before_validated = recorder.validated_transition_count
    before_frames = recorder.retained_frame_count
    before_transitions = recorder.retained_transition_count
    before_recorder_frame = recorder.current_frame

    first = service.current_presentation()
    second = service.current_presentation()
    assert first == second
    assert service.current_frame() is before_raw
    assert service.session is before_session
    assert service.revision == before_revision
    assert service.command_cache_size == before_commands
    assert recorder.status == before_status
    assert recorder.lifecycle == before_lifecycle
    assert recorder.observer_lifecycle_state == before_observer_lifecycle
    assert recorder.validated_transition_count == before_validated
    assert recorder.retained_frame_count == before_frames
    assert recorder.retained_transition_count == before_transitions
    assert recorder.current_frame is before_recorder_frame

    poisoned = before_raw.model_copy(update={"revision": before_revision + 1})
    service._frame = poisoned  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(RuntimeError, match="diverged from service-owned state"):
        service.current_presentation()
    assert service.session is before_session
    assert service.revision == before_revision
    assert service.command_cache_size == before_commands
    assert recorder.status == before_status
    assert recorder.lifecycle == before_lifecycle
    assert recorder.observer_lifecycle_state == before_observer_lifecycle
    assert recorder.validated_transition_count == before_validated
    assert recorder.retained_frame_count == before_frames
    assert recorder.retained_transition_count == before_transitions
    assert recorder.current_frame is before_recorder_frame


@pytest.mark.parametrize("view_mode", ("researcher", "pov"))
def test_live_packaging_exception_leaves_committed_service_untouched(
    view_mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    if view_mode == "pov":
        _switch_to_pov(service)
    before_raw = service.current_frame()
    before_session = service.session
    before_revision = service.revision
    before_observer = service._evaluation_observer  # pyright: ignore[reportPrivateUsage]
    before_observer_count = service.evaluation_validated_transition_count
    before_commands = dict(
        service._command_records  # pyright: ignore[reportPrivateUsage]
    )
    before_shutting_down = service.shutting_down
    before_faulted = service.faulted

    def fail_packaging(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("injected presentation packaging failure")

    builder_name = (
        "build_live_oracle_authorized_presentation_v1"
        if view_mode == "researcher"
        else "build_live_no_shared_obs_authorized_presentation_v1"
    )
    monkeypatch.setattr(service_module, builder_name, fail_packaging)
    with pytest.raises(RuntimeError, match="injected presentation packaging failure"):
        service.current_presentation()

    assert service.current_frame() is before_raw
    assert service.session is before_session
    assert service.revision == before_revision
    assert service._evaluation_observer is before_observer  # pyright: ignore[reportPrivateUsage]
    assert service.evaluation_validated_transition_count == before_observer_count
    assert service._command_records == before_commands  # pyright: ignore[reportPrivateUsage]
    assert service.shutting_down is before_shutting_down
    assert service.faulted is before_faulted


@pytest.mark.parametrize(
    ("initial_view", "command", "builder_name"),
    (
        (
            "researcher",
            KeyboardCommandV1(key="Enter"),
            "build_live_oracle_authorized_presentation_v1",
        ),
        (
            "researcher",
            KeyboardCommandV1(key="g"),
            "build_live_oracle_authorized_presentation_v1",
        ),
        (
            "researcher",
            ResetCommandV1(),
            "build_live_oracle_authorized_presentation_v1",
        ),
        (
            "researcher",
            SetViewCommandV1(view_mode="pov"),
            "build_live_no_shared_obs_authorized_presentation_v1",
        ),
        (
            "pov",
            KeyboardCommandV1(key="Enter"),
            "build_live_no_shared_obs_authorized_presentation_v1",
        ),
        (
            "pov",
            KeyboardCommandV1(key="g"),
            "build_live_no_shared_obs_authorized_presentation_v1",
        ),
        (
            "pov",
            SetViewCommandV1(view_mode="researcher"),
            "build_live_oracle_authorized_presentation_v1",
        ),
    ),
)
def test_changed_live_command_categories_require_a_buildable_candidate_presentation(
    initial_view: str,
    command: object,
    builder_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    if initial_view == "pov":
        _switch_to_pov(service)
    before_presentation = service.current_presentation()
    before_raw = service.current_frame()
    before_session = service.session
    before_revision = service.revision
    before_observer = service._evaluation_observer  # pyright: ignore[reportPrivateUsage]
    before_observer_count = service.evaluation_validated_transition_count
    before_command_count = service.command_cache_size
    real_builder = getattr(service_module, builder_name)
    builder_calls = 0

    def fail_first_candidate(*args: object, **kwargs: object) -> object:
        nonlocal builder_calls
        builder_calls += 1
        if builder_calls == 1:
            raise RuntimeError("injected candidate-presentation failure")
        return real_builder(  # pyright: ignore[reportArgumentType]
            *args,
            **kwargs,
        )

    monkeypatch.setattr(service_module, builder_name, fail_first_candidate)
    with pytest.raises(RuntimeError, match="candidate-presentation failure"):
        service.apply_command(
            _request(
                f"{initial_view}-{type(command).__name__}-presentation-failure",
                base_revision=before_revision,
                command=command,
            )
        )

    assert service.faulted
    assert service.current_frame() is before_raw
    assert service.session is before_session
    assert service.revision == before_revision
    assert service._evaluation_observer is before_observer  # pyright: ignore[reportPrivateUsage]
    assert service.evaluation_validated_transition_count == before_observer_count
    assert service.command_cache_size == before_command_count + 1
    assert service.current_presentation() == before_presentation


def test_candidate_presentation_failure_precedes_recorder_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, recorder = _recording_service(tmp_path, view_mode="pov")
    before_presentation = service.current_presentation()
    before_raw = service.current_frame()
    before_session = service.session
    before_revision = service.revision
    before_status = recorder.status
    before_lifecycle = recorder.lifecycle
    before_observer_lifecycle = recorder.observer_lifecycle_state
    before_validated = recorder.validated_transition_count
    before_frames = recorder.retained_frame_count
    before_transitions = recorder.retained_transition_count
    before_recorder_frame = recorder.current_frame
    real_builder = service_module.build_live_no_shared_obs_authorized_presentation_v1
    builder_calls = 0

    def fail_first_candidate(*args: object, **kwargs: object) -> object:
        nonlocal builder_calls
        builder_calls += 1
        if builder_calls == 1:
            raise RuntimeError("injected recorded candidate-presentation failure")
        return real_builder(*args, **kwargs)  # type: ignore[no-any-return]

    monkeypatch.setattr(
        service_module,
        "build_live_no_shared_obs_authorized_presentation_v1",
        fail_first_candidate,
    )
    with pytest.raises(
        RuntimeError,
        match="recorded candidate-presentation failure",
    ):
        service.apply_command(
            _request(
                "recorded-candidate-presentation-failure",
                base_revision=before_revision,
                command=KeyboardCommandV1(key="Enter"),
            )
        )

    assert service.faulted
    assert service.current_frame() is before_raw
    assert service.session is before_session
    assert service.revision == before_revision
    assert recorder.status == before_status
    assert recorder.lifecycle == before_lifecycle
    assert recorder.observer_lifecycle_state == before_observer_lifecycle
    assert recorder.validated_transition_count == before_validated
    assert recorder.retained_frame_count == before_frames
    assert recorder.retained_transition_count == before_transitions
    assert recorder.current_frame is before_recorder_frame
    assert service.current_presentation() == before_presentation


def test_recording_lifecycle_presentation_preflight_precedes_recorder_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, recorder = _recording_service(tmp_path)
    before_presentation = service.current_presentation()
    before_raw = service.current_frame()
    before_status = recorder.status
    before_lifecycle = recorder.lifecycle
    before_validated = recorder.validated_transition_count
    real_builder = service_module.build_live_oracle_authorized_presentation_v1
    builder_calls = 0

    def fail_first_candidate(
        context: EvaluationEpisodeContextV1,
        current_frame: EvaluationFrameV1,
        incoming_transition_view: EvaluationTransitionViewV1 | None,
        raw_frame: ResearcherLiveDebuggerFrameV2,
    ) -> LiveOracleAuthorizedPresentationFrameV1:
        nonlocal builder_calls
        builder_calls += 1
        if builder_calls == 1:
            raise RuntimeError("injected lifecycle candidate-presentation failure")
        return real_builder(
            context,
            current_frame,
            incoming_transition_view,
            raw_frame,
        )

    monkeypatch.setattr(
        service_module,
        "build_live_oracle_authorized_presentation_v1",
        fail_first_candidate,
    )
    with pytest.raises(
        RuntimeError,
        match="lifecycle candidate-presentation failure",
    ):
        service.apply_command(
            _request(
                "lifecycle-candidate-presentation-failure",
                base_revision=service.revision,
                command=FinishAndReviewCommandV1(),
            )
        )

    assert service.current_frame() is before_raw
    assert service.revision == 0
    assert recorder.status == before_status
    assert recorder.lifecycle == before_lifecycle
    assert recorder.validated_transition_count == before_validated
    assert service.current_presentation() == before_presentation


def test_agent_charge_candidate_is_presentable_before_observer_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service("charge_convergence", include_stress=True)
    _switch_to_pov(service)
    before_revision = service.revision
    real_builder = service_module.build_live_no_shared_obs_authorized_presentation_v1
    preflight_observations: list[tuple[int, int, int]] = []

    def observe_candidate(
        *args: object,
        **kwargs: object,
    ) -> LiveNoSharedObsAuthorizedPresentationFrameV1:
        raw_frame = cast(ActorPovLiveDebuggerFrameV2, args[2])
        preflight_observations.append(
            (
                raw_frame.frame_index,
                service.session.current_evaluation_frame.frame_index,
                service.evaluation_validated_transition_count,
            )
        )
        return real_builder(*args, **kwargs)  # type: ignore[no-any-return]

    monkeypatch.setattr(
        service_module,
        "build_live_no_shared_obs_authorized_presentation_v1",
        observe_candidate,
    )
    advanced = service.apply_command(
        _request(
            "agent-charge-preflight",
            base_revision=before_revision,
            command=KeyboardCommandV1(key="n"),
        )
    )

    assert advanced.outcome == "response"
    assert isinstance(advanced.payload, CommandResponseV2)
    assert advanced.payload.result == "applied"
    assert preflight_observations == [(1, 0, 0)]
    assert service.revision == before_revision + 1
    assert service.session.current_evaluation_frame.frame_index == 1
    assert service.evaluation_validated_transition_count == 1
    assert not service.faulted
    settled = service.current_presentation()
    assert settled.outcome == "response"
    assert isinstance(
        settled.payload,
        LiveNoSharedObsAuthorizedPresentationFrameV1,
    )


def test_live_public_builders_derive_epoch_and_reject_cross_audience_inputs() -> None:
    service = _service()
    raw = service.current_frame()
    assert type(raw) is ResearcherLiveDebuggerFrameV2
    assert (
        "source_authority_epoch"
        not in signature(build_live_oracle_authorized_presentation_v1).parameters
    )
    assert (
        "source_authority_epoch"
        not in signature(build_live_no_shared_obs_authorized_presentation_v1).parameters
    )

    _switch_to_pov(service)
    pov_session = service.session
    pov_raw = service.current_frame()
    assert type(pov_raw) is ActorPovLiveDebuggerFrameV2
    current_slice = build_actor_pov_current_slice_v1(
        pov_session.evaluation_context,
        pov_session.current_evaluation_frame,
        global_slot=pov_session.controlled_global_slot,
        incoming_transition_view=pov_session.incoming_evaluation_view,
    )
    with pytest.raises(TypeError, match="ActorPovLiveDebuggerFrameV2"):
        build_live_no_shared_obs_authorized_presentation_v1(
            current_slice,
            None,
            raw,  # pyright: ignore[reportArgumentType]
            public_catalog=pov_session.evaluation_context.static_mechanics_catalog,
            incoming_visual_events=None,
            researcher_space=build_live_researcher_space_v1(
                build_live_oracle_authorized_presentation_v1(
                    pov_session.evaluation_context,
                    pov_session.current_evaluation_frame,
                    pov_session.incoming_evaluation_view,
                    raw,
                )
            ),
        )


def test_live_no_shared_builder_requires_exact_adjacent_carrier() -> None:
    service = _service()
    _step_once(service)
    _switch_to_pov(service)
    session = service.session
    raw = service.current_frame()
    assert type(raw) is ActorPovLiveDebuggerFrameV2
    current_slice = build_actor_pov_current_slice_v1(
        session.evaluation_context,
        session.current_evaluation_frame,
        global_slot=session.controlled_global_slot,
        incoming_transition_view=session.incoming_evaluation_view,
    )
    current_presentation = service.current_presentation()
    assert type(current_presentation.payload) is (
        LiveNoSharedObsAuthorizedPresentationFrameV1
    )
    researcher_space = current_presentation.payload.researcher_space
    with pytest.raises(TypeError, match="require an exact carrier"):
        build_live_no_shared_obs_authorized_presentation_v1(
            current_slice,
            None,
            raw,
            public_catalog=session.evaluation_context.static_mechanics_catalog,
            incoming_visual_events=build_visual_event_batch_v2(
                cast(
                    EvaluationTransitionViewV1,
                    session.incoming_evaluation_view,
                )
            ),
            researcher_space=researcher_space,
        )
    assert session.incoming_evaluation_view is not None
    carrier = build_actor_pov_adjacent_transition_slice_v1(
        session.incoming_evaluation_view,
        global_slot=session.controlled_global_slot,
    )
    accepted = build_live_no_shared_obs_authorized_presentation_v1(
        current_slice,
        carrier,
        raw,
        public_catalog=session.evaluation_context.static_mechanics_catalog,
        incoming_visual_events=build_visual_event_batch_v2(
            session.incoming_evaluation_view
        ),
        researcher_space=researcher_space,
    )
    assert accepted.latest_events is not None


@pytest.mark.parametrize("field", ("revision", "session_id", "frame_index"))
def test_live_getter_rejects_a_swapped_committed_raw_snapshot(field: str) -> None:
    service = _service()
    raw = service.current_frame()
    replacement: object = (
        "donor-session" if field == "session_id" else getattr(raw, field) + 1
    )
    poisoned = raw.model_copy(update={field: replacement})
    service._frame = poisoned  # pyright: ignore[reportPrivateUsage]
    before_session = service.session
    before_revision = service.revision
    before_observer = service.evaluation_validated_transition_count
    before_commands = service.command_cache_size

    with pytest.raises(RuntimeError, match="diverged from service-owned state"):
        service.current_presentation()

    assert service.session is before_session
    assert service.revision == before_revision
    assert service.evaluation_validated_transition_count == before_observer
    assert service.command_cache_size == before_commands
    assert service.current_frame() is poisoned


def test_live_presentation_getter_holds_lock_against_submit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    entered = Event()
    release = Event()
    original = service_module.build_live_oracle_authorized_presentation_v1

    def blocked_builder(
        context: EvaluationEpisodeContextV1,
        current_frame: EvaluationFrameV1,
        incoming_transition_view: EvaluationTransitionViewV1 | None,
        raw_frame: ResearcherLiveDebuggerFrameV2,
    ) -> LiveOracleAuthorizedPresentationFrameV1:
        entered.set()
        assert release.wait(timeout=10)
        return original(
            context,
            current_frame,
            incoming_transition_view,
            raw_frame,
        )

    monkeypatch.setattr(
        service_module,
        "build_live_oracle_authorized_presentation_v1",
        blocked_builder,
    )
    submit = _request(
        "submit-during-presentation",
        base_revision=0,
        command=KeyboardCommandV1(key=" "),
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        presentation_future = pool.submit(service.current_presentation)
        assert entered.wait(timeout=10)
        submit_future = pool.submit(service.apply_command, submit)
        assert not submit_future.done()
        release.set()
        presentation = presentation_future.result(timeout=20)
        submitted = submit_future.result(timeout=20)

    assert presentation.outcome == "response"
    assert isinstance(
        presentation.payload,
        LiveOracleAuthorizedPresentationFrameV1,
    )
    assert presentation.payload.source.source_frame_index == 0
    assert submitted.outcome == "response"
    assert service.current_frame().frame_index == 1
    settled = service.current_presentation()
    assert isinstance(settled.payload, LiveOracleAuthorizedPresentationFrameV1)
    assert settled.payload.source.source_frame_index == 1
    assert settled.payload.latest_transition is not None


def test_live_module_exposes_no_shared_product_builder() -> None:
    import scripts.dev.visual_debugger.live_presentation as module

    assert module.__all__ == [
        "build_live_no_shared_obs_authorized_presentation_v1",
        "build_live_oracle_authorized_presentation_v1",
        "build_live_researcher_space_v1",
    ]
    assert not any(
        "shared_obs" in name and "no_shared_obs" not in name for name in dir(module)
    )
