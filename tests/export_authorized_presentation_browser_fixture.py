"""Generate exact Python-owned authorized-presentation browser fixtures."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from copy import deepcopy
from dataclasses import fields, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol, cast

from scripts.dev.visual_debugger.control import create_session
from scripts.dev.visual_debugger.presentation_protocol import (
    LiveNoSharedObsAuthorizedPresentationFrameV1,
    LiveOracleAuthorizedPresentationFrameV1,
    LiveOraclePresentationSourceIdentityV1,
    ReplayNoSharedObsAuthorizedPresentationFrameV1,
    _seal_oracle_authorized_current_endpoint_v1,
)
from scripts.dev.visual_debugger.replay_service import ReplayViewerService
from scripts.dev.visual_debugger.scenarios import get_scenario
from scripts.dev.visual_debugger.service import DebuggerService
from tests.export_visual_debugger_replay_artifacts import build_corpse_overlay_bundle
from tests.test_rendering_authorized_inspection import (
    _InspectionCases,
    inspection_cases,
)
from tests.test_visual_debugger_presentation_protocol_v1 import (
    _FiveFrames,
    _live_no_shared_at,
    _live_oracle_at,
    _replay_no_shared_at,
    _replay_oracle_at,
    _replay_shared_at,
    five_frames,
)
from tests.test_visual_debugger_replay_service import (
    _presentation_service,
    _ServiceCases,
    service_cases,
)
from tests.test_visual_debugger_service import _service
from tests.visual_debugger_fixtures import debugger_test_launch_specification

from marl_battlegrounds.core.env import initialize_scenario_state
from marl_battlegrounds.evaluation.capture import capture_initial_evaluation_frame_v1
from marl_battlegrounds.evaluation.replay_io import (
    REPLAY_FILE_SUFFIX_V1,
    load_replay_bundle_v1,
    save_replay_bundle_v1,
)
from marl_battlegrounds.rendering.authorized_pov_scene import pov_presentation_key_v1
from marl_battlegrounds.rendering.authorized_presentation import (
    AgentPovVisualIncomingAgentPhaseTrajectoryV1,
    AuthorizedBattlefieldSceneV1,
    AuthorizedClassMechanicsV1,
    AuthorizedClassMechanicsV2,
    AuthorizedSpawnShieldMechanicsAvailableV1,
    AuthorizedSpawnShieldMechanicsAvailableV2,
    ReplayIncomingAbilityActivatedEventV1,
    ReplayIncomingAgentAnchorV1,
    ReplayIncomingAgentLeftCombatEventV1,
)


class _WrappedFixture0[T](Protocol):
    __wrapped__: Callable[[], T]


class _WrappedFixture1[T, R](Protocol):
    __wrapped__: Callable[[T], R]


def _live_pov_service() -> DebuggerService:
    session = create_session(
        get_scenario("arena_5v5"),
        seed=0,
        evaluation_launch_specification=debugger_test_launch_specification(),
        controlled_global_slot=None,
        show_ranges=True,
        verbose_logging=False,
    )
    return DebuggerService(
        session,
        view_mode="pov",
        preset="analysis",
        include_stress=False,
        session_id="cp2-7-live-pov-fixture",
    )


def _live_corpse_overlay_frame() -> LiveNoSharedObsAuthorizedPresentationFrameV1:
    """Build one editable live Agent frame with an authorized local corpse."""
    session = create_session(
        get_scenario("arena_5v5"),
        seed=0,
        evaluation_launch_specification=debugger_test_launch_specification(),
        controlled_global_slot=0,
        show_ranges=True,
        verbose_logging=False,
    )
    authored_state = session.state._replace(
        agent_positions=session.state.agent_positions.at[5].set((4.0, 1.0)),
        alive_mask=session.state.alive_mask.at[5].set(False),
        current_health=session.state.current_health.at[5].set(0.0),
    )
    state, observation, action_mask, _ = initialize_scenario_state(
        authored_state,
        session.config,
    )
    frame = capture_initial_evaluation_frame_v1(
        session.evaluation_context,
        state,
        observation,
        action_mask,
    )
    service = DebuggerService(
        replace(
            session,
            state=state,
            observation=observation,
            action_mask=action_mask,
            current_evaluation_frame=frame,
            raw_continuation_identity=None,
        ),
        view_mode="pov",
        preset="analysis",
        include_stress=False,
        session_id="browser-live-no-shared-corpse-overlay",
    )
    result = service.current_presentation()
    if (
        result.outcome != "response"
        or type(result.payload) is not LiveNoSharedObsAuthorizedPresentationFrameV1
        or len(result.payload.local_oracle_corpse_overlay.corpse_observations) != 1
    ):
        raise RuntimeError("live corpse-overlay fixture did not authorize one corpse")
    return result.payload


def _pair(service: DebuggerService | ReplayViewerService) -> dict[str, object]:
    raw = service.current_frame()
    result = service.current_presentation()
    if result.outcome != "response":
        raise RuntimeError("fixture service did not produce a presentation response")
    pair: dict[str, object] = {
        "transport": raw.model_dump(mode="json"),
        "presentation": result.payload.model_dump(mode="json"),
    }
    if isinstance(service, ReplayViewerService):
        pair["timeline"] = service.current_timeline().model_dump(mode="json")
    return pair


def _corpse_overlay_browser_cases() -> tuple[
    ReplayNoSharedObsAuthorizedPresentationFrameV1,
    ReplayNoSharedObsAuthorizedPresentationFrameV1,
    dict[str, str],
]:
    """Return initial/persistent overlays plus digest-valid poison cases."""
    bundle = build_corpse_overlay_bundle(execution_information_mode="no_shared_obs")
    with TemporaryDirectory(prefix="marl-corpse-overlay-fixture-") as directory:
        path = Path(directory) / f"corpse{REPLAY_FILE_SUFFIX_V1}"
        save_replay_bundle_v1(bundle, path)
        loaded = load_replay_bundle_v1(path, require_metric_report=True)
        service = ReplayViewerService(
            loaded,
            initial_frame_index=0,
            view_mode="pov",
            pov_global_slot=0,
            viewer_session_id="browser-replay-no-shared-corpse-overlay",
        )
        result = service.current_presentation()
        persistent_service = ReplayViewerService(
            loaded,
            initial_frame_index=1,
            view_mode="pov",
            pov_global_slot=0,
            viewer_session_id="browser-replay-no-shared-persistent-corpse-overlay",
        )
        persistent_result = persistent_service.current_presentation()
    if (
        result.outcome != "response"
        or type(result.payload) is not ReplayNoSharedObsAuthorizedPresentationFrameV1
    ):
        raise RuntimeError(
            "corpse-overlay fixture did not produce NoSharedObs Agent POV"
        )
    frame = result.payload
    if len(frame.local_oracle_corpse_overlay.corpse_observations) != 1:
        raise RuntimeError(
            "corpse-overlay fixture requires exactly one projected corpse"
        )
    if (
        persistent_result.outcome != "response"
        or type(persistent_result.payload)
        is not ReplayNoSharedObsAuthorizedPresentationFrameV1
    ):
        raise RuntimeError(
            "persistent corpse-overlay fixture did not produce NoSharedObs Agent POV"
        )
    persistent_frame = persistent_result.payload
    persistent_visual = persistent_frame.visual_events
    persistent_corpses = (
        persistent_frame.local_oracle_corpse_overlay.corpse_observations
    )
    if persistent_visual is None or len(persistent_corpses) != 1:
        raise RuntimeError(
            "persistent corpse-overlay fixture requires one projected corpse"
        )
    persistent_corpse_id = persistent_corpses[0].corpse.public_agent_id
    if any(
        row.agent_public_agent_id == persistent_corpse_id
        for row in persistent_visual.agent_phase_trajectories
    ):
        raise RuntimeError(
            "persistent corpse overlay entered the causal trajectory inventory"
        )

    def reseal(payload: dict[str, object]) -> None:
        overlay = cast(dict[str, object], payload["local_oracle_corpse_overlay"])
        content = {
            key: value
            for key, value in overlay.items()
            if key != "authorized_overlay_digest_sha256"
        }
        overlay["authorized_overlay_digest_sha256"] = hashlib.sha256(
            json.dumps(
                content,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

    corpse_fact_mismatch = cast(
        dict[str, object], deepcopy(frame.model_dump(mode="json"))
    )
    mismatch_overlay = cast(
        dict[str, object], corpse_fact_mismatch["local_oracle_corpse_overlay"]
    )
    mismatch_observation = cast(
        dict[str, object],
        cast(list[object], mismatch_overlay["corpse_observations"])[0],
    )
    mismatch_corpse = cast(dict[str, object], mismatch_observation["corpse"])
    mismatch_position = cast(list[float], mismatch_corpse["position"])
    mismatch_position[0] += 0.25
    reseal(corpse_fact_mismatch)

    class_mechanics_mismatch = cast(
        dict[str, object], deepcopy(frame.model_dump(mode="json"))
    )
    class_overlay = cast(
        dict[str, object], class_mechanics_mismatch["local_oracle_corpse_overlay"]
    )
    class_observation = cast(
        dict[str, object], cast(list[object], class_overlay["corpse_observations"])[0]
    )
    class_corpse = cast(dict[str, object], class_observation["corpse"])
    class_facts = cast(dict[str, object], class_observation["oracle_public_facts"])
    class_corpse["radius"] = cast(float, class_corpse["radius"]) + 0.125
    class_facts["radius"] = cast(float, class_facts["radius"]) + 0.125
    reseal(class_mechanics_mismatch)

    return (
        frame,
        persistent_frame,
        {
            "resealed_corpse_public_facts_mismatch": cast(
                str,
                cast(
                    dict[str, object],
                    corpse_fact_mismatch["local_oracle_corpse_overlay"],
                )["authorized_overlay_digest_sha256"],
            ),
            "resealed_corpse_class_mechanics_mismatch": cast(
                str,
                cast(
                    dict[str, object],
                    class_mechanics_mismatch["local_oracle_corpse_overlay"],
                )["authorized_overlay_digest_sha256"],
            ),
        },
    )


def _legacy_v1_scene(
    scene: AuthorizedBattlefieldSceneV1,
) -> AuthorizedBattlefieldSceneV1:
    """Downgrade only the two additive nested contracts through V1 models."""
    shield = scene.spawn_shield_mechanics
    if type(shield) is not AuthorizedSpawnShieldMechanicsAvailableV2:
        raise RuntimeError("canonical fixture requires available Spawn Shield V2")
    legacy_class_rows: list[AuthorizedClassMechanicsV1] = []
    for row in scene.class_mechanics:
        if type(row) is not AuthorizedClassMechanicsV2:
            raise RuntimeError("canonical fixture requires class mechanics V2")
        legacy_class_rows.append(
            AuthorizedClassMechanicsV1(
                **{
                    field.name: getattr(row, field.name)
                    for field in fields(AuthorizedClassMechanicsV1)
                }
            )
        )
    return replace(
        scene,
        class_mechanics=tuple(legacy_class_rows),
        spawn_shield_mechanics=AuthorizedSpawnShieldMechanicsAvailableV1(
            availability_kind="available",
            configured_duration_steps=shield.configured_duration_steps,
            movement_speed=shield.movement_speed,
        ),
    )


def _legacy_v1_compatibility_presentation(
    frame: LiveOracleAuthorizedPresentationFrameV1,
) -> LiveOracleAuthorizedPresentationFrameV1:
    """Construct and reseal one authoritative full-frame legacy V1 case."""
    endpoint = frame.current_endpoint
    legacy_endpoint = _seal_oracle_authorized_current_endpoint_v1(
        episode_id=endpoint.episode_id,
        frame_index=endpoint.frame_index,
        frame_id=endpoint.frame_id,
        simulator_step_count=endpoint.simulator_step_count,
        scene=_legacy_v1_scene(endpoint.scene),
        identity_directory=endpoint.identity_directory,
        action_axis=endpoint.action_axis,
    )
    source_values = {
        name: getattr(frame.source, name)
        for name in LiveOraclePresentationSourceIdentityV1.model_fields
    }
    source_values["source_authorized_endpoint_digest_sha256"] = (
        legacy_endpoint.authorized_endpoint_digest_sha256
    )
    legacy_source = LiveOraclePresentationSourceIdentityV1(**source_values)
    frame_values = {
        name: getattr(frame, name)
        for name in LiveOracleAuthorizedPresentationFrameV1.model_fields
    }
    frame_values["source"] = legacy_source
    frame_values["current_endpoint"] = legacy_endpoint
    return LiveOracleAuthorizedPresentationFrameV1(**frame_values)


def _with_visual_events(
    frame: ReplayNoSharedObsAuthorizedPresentationFrameV1,
    *,
    trajectories: tuple[AgentPovVisualIncomingAgentPhaseTrajectoryV1, ...],
    event: ReplayIncomingAbilityActivatedEventV1 | ReplayIncomingAgentLeftCombatEventV1,
) -> ReplayNoSharedObsAuthorizedPresentationFrameV1:
    """Revalidate one branded adjacent-fog browser contract case."""
    visual = frame.visual_events
    if visual is None or visual.events:
        raise RuntimeError("adjacent-fog fixture requires an empty incoming inventory")
    event_id = f"{visual.incoming_recipient_transition_id}:visual-event:0000"
    bound_event = replace(event, event_id=event_id)
    bound_visual = replace(
        visual,
        agent_phase_trajectories=trajectories,
        ordered_event_ids=(event_id,),
        ordered_event_kinds=(bound_event.event_kind,),
        events=(bound_event,),
        event_count=1,
    )
    values = {
        name: getattr(frame, name)
        for name in ReplayNoSharedObsAuthorizedPresentationFrameV1.model_fields
    }
    values["visual_events"] = bound_visual
    return ReplayNoSharedObsAuthorizedPresentationFrameV1(**values)


def _adjacent_fog_state_cases(
    frame: ReplayNoSharedObsAuthorizedPresentationFrameV1,
) -> dict[str, ReplayNoSharedObsAuthorizedPresentationFrameV1]:
    """Generate positive browser cases for both adjacent fog-set directions."""
    visual = frame.visual_events
    if visual is None or visual.events:
        raise RuntimeError("adjacent-fog fixture requires a noninitial empty frame")
    recipient = next(
        row
        for row in visual.agent_phase_trajectories
        if row.agent_presentation_key == visual.recipient_presentation_key
    )
    if recipient.transition_start is None:
        raise RuntimeError("adjacent-fog fixture recipient requires a start anchor")

    disappearing_public_id = "agent-slot-5"
    disappearing_key = pov_presentation_key_v1(
        authority_session_id=frame.source.source_session_id,
        recipient_public_agent_id=visual.recipient_public_agent_id,
        public_agent_id=disappearing_public_id,
    )
    disappearing_anchor = ReplayIncomingAgentAnchorV1(
        phase="transition_start",
        presentation_key=disappearing_key,
        public_agent_id=disappearing_public_id,
        position=(8.0, 4.0),
    )
    disappearing = AgentPovVisualIncomingAgentPhaseTrajectoryV1(
        agent_presentation_key=disappearing_key,
        agent_public_agent_id=disappearing_public_id,
        agent_class_id=1,
        transition_start=disappearing_anchor,
        successor=None,
    )
    disappearance = _with_visual_events(
        frame,
        trajectories=(*visual.agent_phase_trajectories, disappearing),
        event=ReplayIncomingAbilityActivatedEventV1(
            event_id="pending-local-id",
            ordinal=0,
            phase_rank=20,
            event_kind="ability_activated",
            ability_component="basic",
            source_anchor=disappearing_anchor,
            recipient_anchor=recipient.transition_start,
        ),
    )

    appearing_index = next(
        index
        for index, row in enumerate(visual.agent_phase_trajectories)
        if row.agent_presentation_key != visual.recipient_presentation_key
        and row.successor is not None
    )
    appearing = replace(
        visual.agent_phase_trajectories[appearing_index],
        transition_start=None,
    )
    appearance_trajectories = (
        *(
            row
            for index, row in enumerate(visual.agent_phase_trajectories)
            if index != appearing_index
        ),
        appearing,
    )
    if appearing.successor is None:
        raise RuntimeError("adjacent-fog fixture appearance requires a successor")
    appearance = _with_visual_events(
        frame,
        trajectories=appearance_trajectories,
        event=ReplayIncomingAgentLeftCombatEventV1(
            event_id="pending-local-id",
            ordinal=0,
            phase_rank=50,
            event_kind="agent_left_combat",
            agent_anchor=appearing.successor,
        ),
    )
    return {
        "replay_no_shared_agent_disappearance": disappearance,
        "replay_no_shared_agent_appearance": appearance,
    }


def render_fixture() -> str:
    cases = cast("_WrappedFixture0[_InspectionCases]", inspection_cases).__wrapped__()
    frames = cast(
        "_WrappedFixture1[_InspectionCases, _FiveFrames]", five_frames
    ).__wrapped__(cases)
    replay_cases = cast("_WrappedFixture0[_ServiceCases]", service_cases).__wrapped__()
    (
        corpse_overlay_frame,
        persistent_corpse_overlay_frame,
        corpse_overlay_negative_digests,
    ) = _corpse_overlay_browser_cases()
    pair_services = {
        "live_oracle": _service(),
        "live_no_shared_obs_agent_pov": _live_pov_service(),
        "replay_oracle": _presentation_service(
            replay_cases,
            "oracle",
            viewer_session_id="cp2-7-replay-oracle-fixture",
        ),
        "replay_no_shared_obs_agent_pov": _presentation_service(
            replay_cases,
            "no_shared_obs",
            viewer_session_id="cp2-7-replay-no-shared-fixture",
        ),
        "replay_shared_obs_agent_pov": _presentation_service(
            replay_cases,
            "shared_obs",
            viewer_session_id="cp2-7-replay-shared-fixture",
        ),
    }
    pairs = {kind: _pair(service) for kind, service in pair_services.items()}
    continuity_session = "cp2-7-replay-audience-switch-fixture"
    continuity_pairs = {
        "oracle": _pair(
            ReplayViewerService(
                replay_cases.shared.bundle,
                initial_frame_index=1,
                view_mode="researcher",
                selected_global_slot=0,
                viewer_session_id=continuity_session,
            )
        ),
        "shared_obs": _pair(
            ReplayViewerService(
                replay_cases.shared.bundle,
                initial_frame_index=1,
                view_mode="pov",
                pov_global_slot=0,
                viewer_session_id=continuity_session,
            )
        ),
    }
    final_index = len(cases.no_shared.transitions)
    shared_final_index = len(cases.shared.transitions)
    replay_no_shared_final = _replay_no_shared_at(
        cases,
        frame_index=final_index,
        session="cp2-7-replay-no-shared-final",
    )
    state_cases = {
        "live_oracle_frame_zero": _live_oracle_at(
            cases,
            frame_index=0,
            session="cp2-7-live-oracle-zero",
        ),
        "live_no_shared_frame_zero": _live_no_shared_at(
            cases,
            frame_index=0,
            session="cp2-7-live-no-shared-zero",
        ),
        "live_no_shared_editable_corpse_overlay": _live_corpse_overlay_frame(),
        "replay_oracle_frame_zero": _replay_oracle_at(
            cases,
            frame_index=0,
            selected_internal_slot=0,
            session="cp2-7-replay-oracle-zero",
        ),
        "replay_oracle_final_selected": _replay_oracle_at(
            cases,
            frame_index=final_index,
            selected_internal_slot=0,
            session="cp2-7-replay-oracle-final-selected",
        ),
        "replay_oracle_final_unselected": _replay_oracle_at(
            cases,
            frame_index=final_index,
            selected_internal_slot=None,
            session="cp2-7-replay-oracle-final-unselected",
        ),
        "replay_no_shared_frame_zero": _replay_no_shared_at(
            cases,
            frame_index=0,
            session="cp2-7-replay-no-shared-zero",
        ),
        "replay_no_shared_final": replay_no_shared_final,
        "replay_shared_frame_zero": _replay_shared_at(
            cases,
            frame_index=0,
            session="browser-replay-shared-source-material-metric-processing",
        ),
        "replay_shared_final": _replay_shared_at(
            cases,
            frame_index=shared_final_index,
            session="cp2-7-replay-shared-final",
        ),
        "replay_no_shared_corpse_overlay": corpse_overlay_frame,
        "replay_no_shared_persistent_corpse_overlay": (persistent_corpse_overlay_frame),
    }
    state_cases.update(_adjacent_fog_state_cases(replay_no_shared_final))
    payload = {
        "schema_version": 1,
        "presentations": {
            frame.presentation_kind: frame.model_dump(mode="json")
            for frame in frames.rows
        },
        "pairs": pairs,
        "continuity_pairs": continuity_pairs,
        "state_cases": {
            name: frame.model_dump(mode="json") for name, frame in state_cases.items()
        },
        "compatibility_cases": {
            "legacy_v1": _legacy_v1_compatibility_presentation(
                frames.live_oracle
            ).model_dump(mode="json")
        },
        "corpse_overlay_negative_digests": corpse_overlay_negative_digests,
    }
    return json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    destination = (
        repository_root
        / "web"
        / "visual_debugger"
        / "tests"
        / "fixtures"
        / "authorized-presentations-v1.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_fixture(), encoding="utf-8")


if __name__ == "__main__":
    main()
