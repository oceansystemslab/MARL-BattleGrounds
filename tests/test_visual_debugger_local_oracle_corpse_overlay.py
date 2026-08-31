"""Focused presentation-only local-Oracle corpse authorization proofs."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pydantic import ValidationError
from scripts.dev.visual_debugger.control import create_session
from scripts.dev.visual_debugger.local_oracle_corpse_overlay import (
    _host_has_clear_line_of_sight_v1,
    _host_within_observation_radius_v1,
    build_local_oracle_corpse_overlay_v1,
    validate_local_oracle_corpse_overlay_against_source_v1,
)
from scripts.dev.visual_debugger.presentation_protocol import (
    LocalOracleCorpseObservationV1,
    LocalOracleCorpseOverlayV1,
    ReplayNoSharedObsAuthorizedPresentationFrameV1,
    _validate_local_oracle_corpse_overlay,
    seal_local_oracle_corpse_overlay_v1,
)
from scripts.dev.visual_debugger.replay_service import ReplayViewerService
from scripts.dev.visual_debugger.scenarios import get_scenario
from tests.evaluation_fixtures import (
    CapturedEvaluationTrajectory,
    captured_evaluation_trajectory,
    evaluation_env_config,
    valid_shared_availability,
)
from tests.export_visual_debugger_replay_artifacts import build_corpse_overlay_bundle
from tests.visual_debugger_fixtures import debugger_test_launch_specification

from marl_battlegrounds.core.env import (
    _build_global_visibility_mask_and_distances,
    initialize_scenario_state,
    reset,
)
from marl_battlegrounds.core.geometry import has_clear_line_of_sight
from marl_battlegrounds.evaluation.capture import (
    capture_initial_evaluation_frame_v1,
)
from marl_battlegrounds.evaluation.models import (
    EvaluationFrameV1,
    ResolvedObstacleV1,
)
from marl_battlegrounds.evaluation.pov import build_actor_pov_current_slice_v1
from marl_battlegrounds.evaluation.replay_io import LoadedReplayBundleV1
from marl_battlegrounds.evaluation.wire_shapes import MAX_OBSTACLE_SLOTS_V1
from marl_battlegrounds.rendering.authorized_pov_scene import (
    SharedObsAuthorizedScenePartsV1,
    build_no_shared_obs_authorized_scene_v1,
    build_shared_obs_authorized_scene_v1,
)
from marl_battlegrounds.rendering.authorized_presentation import (
    AuthorizedBattlefieldSceneV1,
)
from marl_battlegrounds.rendering.evaluation_adapter import (
    build_shared_obs_source_material_projection_v1,
)


def _frame_with_agents(
    trajectory: CapturedEvaluationTrajectory,
    *,
    positions: dict[int, tuple[float, float]],
    dead_slots: tuple[int, ...],
) -> EvaluationFrameV1:
    if trajectory.frames[0].frame_index != 0:
        raise ValueError("corpse-overlay fixture requires an initial evaluation frame.")
    config = evaluation_env_config()
    state, observation, action_mask, _ = reset(config, jax.random.PRNGKey(0))
    availability = (
        valid_shared_availability(trajectory.context)
        if trajectory.context.execution_information_mode == "shared_obs"
        else None
    )
    baseline = capture_initial_evaluation_frame_v1(
        trajectory.context,
        state,
        observation,
        action_mask,
        availability,
    )
    if baseline != trajectory.frames[0]:
        raise ValueError("corpse-overlay fixture does not match its reset authority.")
    updated_positions = state.agent_positions
    updated_alive = state.alive_mask
    updated_health = state.current_health
    for slot, position in positions.items():
        updated_positions = updated_positions.at[slot].set(position)
    for slot in dead_slots:
        updated_alive = updated_alive.at[slot].set(False)
        updated_health = updated_health.at[slot].set(0.0)
    authored_state = state._replace(
        agent_positions=updated_positions,
        alive_mask=updated_alive,
        current_health=updated_health,
    )
    coherent_state, coherent_observation, coherent_mask, _ = initialize_scenario_state(
        authored_state,
        config,
    )
    return capture_initial_evaluation_frame_v1(
        trajectory.context,
        coherent_state,
        coherent_observation,
        coherent_mask,
        availability,
    )


def _no_shared_scene(
    trajectory: CapturedEvaluationTrajectory,
    frame: EvaluationFrameV1,
) -> AuthorizedBattlefieldSceneV1:
    current = build_actor_pov_current_slice_v1(
        trajectory.context,
        frame,
        global_slot=0,
    )
    return build_no_shared_obs_authorized_scene_v1(
        current,
        public_catalog=trajectory.context.static_mechanics_catalog,
        authority_session_id="corpse-overlay-no-shared",
    ).scene


def _shared_scene(
    trajectory: CapturedEvaluationTrajectory,
    frame: EvaluationFrameV1,
) -> SharedObsAuthorizedScenePartsV1:
    active_slots = tuple(
        row.global_slot for row in trajectory.context.roster if row.configured_active
    )
    projections = {
        slot: build_shared_obs_source_material_projection_v1(
            trajectory.context,
            frame,
            selected_global_slot=slot,
        )
        for slot in active_slots
    }
    return build_shared_obs_authorized_scene_v1(
        projections[0],
        all_active_nonrecipient_source_material=tuple(
            projections[slot] for slot in active_slots if slot != 0
        ),
        public_catalog=trajectory.context.static_mechanics_catalog,
        authority_session_id="corpse-overlay-shared",
    )


def _overlay(
    trajectory: CapturedEvaluationTrajectory,
    frame: EvaluationFrameV1,
    scene: AuthorizedBattlefieldSceneV1,
    *,
    authority: str = "corpse-overlay-no-shared",
    sensors: tuple[str, ...] = ("agent-slot-0",),
) -> LocalOracleCorpseOverlayV1:
    return build_local_oracle_corpse_overlay_v1(
        trajectory.context,
        frame,
        scene,
        authority_session_id=authority,
        source_authority_epoch=0,
        recipient_public_agent_id="agent-slot-0",
        living_sensor_public_agent_ids=sensors,
    )


def test_no_shared_corpse_projection_is_local_and_policy_byte_inert() -> None:
    trajectory = captured_evaluation_trajectory(transition_count=0)
    candidate = _frame_with_agents(
        trajectory,
        positions={5: (4.0, 1.5)},
        dead_slots=(5,),
    )
    scene = _no_shared_scene(trajectory, candidate)
    observation_bytes = candidate.base_observation.model_dump_json()
    mask_bytes = candidate.action_mask.model_dump_json()
    source_frame_bytes = candidate.model_dump_json()
    scene_before = repr(scene)

    overlay = _overlay(trajectory, candidate, scene)

    assert tuple(row.corpse.public_agent_id for row in overlay.corpse_observations) == (
        "agent-slot-5",
    )
    assert overlay.corpse_observations[0].observing_sensor_public_agent_ids == (
        "agent-slot-0",
    )
    assert overlay.corpse_observations[0].corpse.life_state == "corpse"
    assert overlay.corpse_observations[0].corpse.position == (4.0, 1.5)
    assert candidate.base_observation.model_dump_json() == observation_bytes
    assert candidate.action_mask.model_dump_json() == mask_bytes
    assert candidate.model_dump_json() == source_frame_bytes
    assert repr(scene) == scene_before


@pytest.mark.parametrize(
    ("candidate_slot", "candidate_position"),
    (
        (5, (15.0, 10.0)),
        (3, (0.0, 0.0)),
    ),
    ids=("out-of-range", "inactive"),
)
def test_no_shared_corpse_projection_excludes_unauthorized_candidates(
    candidate_slot: int,
    candidate_position: tuple[float, float],
) -> None:
    trajectory = captured_evaluation_trajectory(transition_count=0)
    candidate = _frame_with_agents(
        trajectory,
        positions={candidate_slot: candidate_position},
        dead_slots=(candidate_slot,),
    )
    scene = _no_shared_scene(trajectory, candidate)

    assert _overlay(trajectory, candidate, scene).corpse_observations == ()


def test_no_shared_corpse_projection_respects_static_line_of_sight() -> None:
    session = create_session(
        get_scenario("arena_5v5"),
        seed=0,
        evaluation_launch_specification=debugger_test_launch_specification(),
        controlled_global_slot=0,
        show_ranges=True,
        verbose_logging=False,
    )
    authored_state = session.state._replace(
        agent_positions=session.state.agent_positions.at[0]
        .set((8.0, 3.0))
        .at[5]
        .set((12.0, 3.0)),
        alive_mask=session.state.alive_mask.at[5].set(False),
        current_health=session.state.current_health.at[5].set(0.0),
    )
    coherent_state, coherent_observation, coherent_mask, _ = initialize_scenario_state(
        authored_state,
        session.config,
    )
    frame = capture_initial_evaluation_frame_v1(
        session.evaluation_context,
        coherent_state,
        coherent_observation,
        coherent_mask,
    )
    current = build_actor_pov_current_slice_v1(
        session.evaluation_context,
        frame,
        global_slot=0,
    )
    scene = build_no_shared_obs_authorized_scene_v1(
        current,
        public_catalog=session.evaluation_context.static_mechanics_catalog,
        authority_session_id="corpse-overlay-obstacle-no-shared",
    ).scene
    recipient_id = session.evaluation_context.roster[0].public_agent_id

    overlay = build_local_oracle_corpse_overlay_v1(
        session.evaluation_context,
        frame,
        scene,
        authority_session_id="corpse-overlay-obstacle-no-shared",
        source_authority_epoch=0,
        recipient_public_agent_id=recipient_id,
        living_sensor_public_agent_ids=(recipient_id,),
    )
    assert overlay.corpse_observations == ()


def test_team_b_recipient_can_authorize_a_locally_visible_opponent_corpse() -> None:
    trajectory = captured_evaluation_trajectory(transition_count=0)
    frame = _frame_with_agents(
        trajectory,
        positions={0: (16.0, 1.5)},
        dead_slots=(0,),
    )
    recipient_slot = 5
    current = build_actor_pov_current_slice_v1(
        trajectory.context,
        frame,
        global_slot=recipient_slot,
    )
    scene = build_no_shared_obs_authorized_scene_v1(
        current,
        public_catalog=trajectory.context.static_mechanics_catalog,
        authority_session_id="corpse-overlay-team-b",
    ).scene
    recipient_id = trajectory.context.roster[recipient_slot].public_agent_id

    overlay = build_local_oracle_corpse_overlay_v1(
        trajectory.context,
        frame,
        scene,
        authority_session_id="corpse-overlay-team-b",
        source_authority_epoch=0,
        recipient_public_agent_id=recipient_id,
        living_sensor_public_agent_ids=(recipient_id,),
    )

    assert tuple(row.corpse.public_agent_id for row in overlay.corpse_observations) == (
        "agent-slot-0",
    )
    assert overlay.corpse_observations[0].observing_sensor_public_agent_ids == (
        recipient_id,
    )


def test_dead_recipient_casts_no_corpse_visibility() -> None:
    trajectory = captured_evaluation_trajectory(transition_count=0)
    dead_frame = _frame_with_agents(
        trajectory,
        positions={5: (4.0, 1.5)},
        dead_slots=(0, 5),
    )
    dead_scene = _no_shared_scene(trajectory, dead_frame)

    overlay = _overlay(trajectory, dead_frame, dead_scene, sensors=())
    assert overlay.living_sensor_public_agent_ids == ()
    assert overlay.corpse_observations == ()
    assert any(
        row.relation == "self" and row.life_state == "corpse"
        for row in dead_scene.agents
    )


def test_shared_obs_uses_living_allied_sensor_union_without_broadening_no_shared() -> (
    None
):
    trajectory = captured_evaluation_trajectory(
        transition_count=0,
        execution_information_mode="shared_obs",
    )
    frame = _frame_with_agents(
        trajectory,
        positions={5: (1.5, 9.5)},
        dead_slots=(5,),
    )
    shared_scene = _shared_scene(trajectory, frame).scene
    overlay = _overlay(
        trajectory,
        frame,
        shared_scene,
        authority="corpse-overlay-shared",
        sensors=("agent-slot-0", "agent-slot-1", "agent-slot-2"),
    )

    assert len(overlay.corpse_observations) == 1
    assert overlay.corpse_observations[0].corpse.public_agent_id == "agent-slot-5"
    assert overlay.corpse_observations[0].observing_sensor_public_agent_ids == (
        "agent-slot-1",
        "agent-slot-2",
    )

    no_shared_trajectory = captured_evaluation_trajectory(transition_count=0)
    no_shared_frame = _frame_with_agents(
        no_shared_trajectory,
        positions={5: (1.5, 9.5)},
        dead_slots=(5,),
    )
    assert (
        _overlay(
            no_shared_trajectory,
            no_shared_frame,
            _no_shared_scene(no_shared_trajectory, no_shared_frame),
        ).corpse_observations
        == ()
    )


def test_corpse_overlay_digest_rejects_forged_content() -> None:
    trajectory = captured_evaluation_trajectory(transition_count=0)
    frame = _frame_with_agents(
        trajectory,
        positions={5: (4.0, 1.5)},
        dead_slots=(5,),
    )
    overlay = _overlay(
        trajectory,
        frame,
        _no_shared_scene(trajectory, frame),
    )
    payload = {name: getattr(overlay, name) for name in type(overlay).model_fields}
    payload["source_authority_epoch"] = 9

    with pytest.raises(ValidationError, match="overlay digest mismatch"):
        LocalOracleCorpseOverlayV1.model_validate(payload)


def test_resealed_corpse_geometry_must_match_authorized_oracle_public_facts() -> None:
    trajectory = captured_evaluation_trajectory(transition_count=0)
    frame = _frame_with_agents(
        trajectory,
        positions={5: (4.0, 1.5)},
        dead_slots=(5,),
    )
    overlay = _overlay(
        trajectory,
        frame,
        _no_shared_scene(trajectory, frame),
    )
    payload = overlay.model_dump(mode="json")
    payload["corpse_observations"][0]["corpse"]["position"][0] += 0.25
    content = {
        key: value
        for key, value in payload.items()
        if key != "authorized_overlay_digest_sha256"
    }
    payload["authorized_overlay_digest_sha256"] = hashlib.sha256(
        json.dumps(
            content,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    with pytest.raises(
        ValidationError,
        match="changed from its authorized Oracle public facts",
    ):
        LocalOracleCorpseOverlayV1.model_validate_json(json.dumps(payload))


def test_coordinated_resealed_corpse_geometry_fails_same_epoch_source_validation() -> (
    None
):
    trajectory = captured_evaluation_trajectory(transition_count=0)
    frame = _frame_with_agents(
        trajectory,
        positions={5: (4.0, 1.5)},
        dead_slots=(5,),
    )
    scene = _no_shared_scene(trajectory, frame)
    overlay = _overlay(trajectory, frame, scene)
    original = overlay.corpse_observations[0]
    forged_position = (original.corpse.position[0] + 0.25, original.corpse.position[1])
    forged_observation = LocalOracleCorpseObservationV1(
        corpse=replace(original.corpse, position=forged_position),
        oracle_public_facts=original.oracle_public_facts.model_copy(
            update={"position": forged_position}
        ),
        observing_sensor_public_agent_ids=(original.observing_sensor_public_agent_ids),
    )
    forged_overlay = seal_local_oracle_corpse_overlay_v1(
        source_episode_id=overlay.source_episode_id,
        source_frame_index=overlay.source_frame_index,
        source_simulator_step_count=overlay.source_simulator_step_count,
        source_authority_epoch=overlay.source_authority_epoch,
        recipient_public_agent_id=overlay.recipient_public_agent_id,
        recipient_presentation_key=overlay.recipient_presentation_key,
        living_sensor_public_agent_ids=overlay.living_sensor_public_agent_ids,
        corpse_observations=(forged_observation,),
    )

    with pytest.raises(
        ValueError,
        match="changed from its authoritative same-epoch source",
    ):
        validate_local_oracle_corpse_overlay_against_source_v1(
            forged_overlay,
            trajectory.context,
            frame,
            scene,
            authority_session_id="corpse-overlay-no-shared",
            source_authority_epoch=0,
            recipient_public_agent_id="agent-slot-0",
            living_sensor_public_agent_ids=("agent-slot-0",),
        )


def test_resealed_corpse_static_facts_must_match_authorized_class_mechanics() -> None:
    bundle = build_corpse_overlay_bundle(execution_information_mode="no_shared_obs")
    service = ReplayViewerService(
        LoadedReplayBundleV1(
            replay=bundle.replay,
            metric_report_artifact=bundle.metric_report_artifact,
            status="complete",
        ),
        initial_frame_index=0,
        view_mode="pov",
        pov_global_slot=0,
        viewer_session_id="corpse-overlay-static-forgery",
    )
    result = service.current_presentation()
    assert result.outcome == "response"
    assert type(result.payload) is ReplayNoSharedObsAuthorizedPresentationFrameV1
    frame = result.payload
    overlay = frame.local_oracle_corpse_overlay
    original = overlay.corpse_observations[0]
    forged_corpse = replace(original.corpse, radius=original.corpse.radius + 0.125)
    forged_facts = original.oracle_public_facts.model_copy(
        update={"radius": original.oracle_public_facts.radius + 0.125}
    )
    forged_observation = LocalOracleCorpseObservationV1(
        corpse=forged_corpse,
        oracle_public_facts=forged_facts,
        observing_sensor_public_agent_ids=(original.observing_sensor_public_agent_ids),
    )
    forged_overlay = seal_local_oracle_corpse_overlay_v1(
        source_episode_id=overlay.source_episode_id,
        source_frame_index=overlay.source_frame_index,
        source_simulator_step_count=overlay.source_simulator_step_count,
        source_authority_epoch=overlay.source_authority_epoch,
        recipient_public_agent_id=overlay.recipient_public_agent_id,
        recipient_presentation_key=overlay.recipient_presentation_key,
        living_sensor_public_agent_ids=overlay.living_sensor_public_agent_ids,
        corpse_observations=(forged_observation,),
    )
    forged_frame = frame.model_copy(
        update={"local_oracle_corpse_overlay": forged_overlay}
    )

    with pytest.raises(ValueError, match="changed from class mechanics"):
        _validate_local_oracle_corpse_overlay(forged_frame, shared=False)


def _resolved_obstacle(
    *,
    obstacle_slot: int,
    obstacle_type_id: int = 0,
    x: float = 0.0,
    y: float = 0.0,
    radius: float = 0.0,
    width: float = 0.0,
    height: float = 0.0,
    theta: float = 0.0,
    is_active: bool = False,
) -> ResolvedObstacleV1:
    return ResolvedObstacleV1(
        obstacle_slot=obstacle_slot,
        obstacle_type_id=obstacle_type_id,
        x=x,
        y=y,
        radius=radius,
        width=width,
        height=height,
        theta=theta,
        is_active=is_active,
    )


def _padded_obstacles(
    *active_rows: ResolvedObstacleV1,
) -> tuple[ResolvedObstacleV1, ...]:
    by_slot = {row.obstacle_slot: row for row in active_rows}
    return tuple(
        by_slot.get(slot, _resolved_obstacle(obstacle_slot=slot))
        for slot in range(MAX_OBSTACLE_SLOTS_V1)
    )


def _jax_obstacle_array(obstacles: tuple[ResolvedObstacleV1, ...]) -> jax.Array:
    return jnp.asarray(
        tuple(
            (
                row.obstacle_type_id,
                row.x,
                row.y,
                row.radius,
                row.width,
                row.height,
                row.theta,
                1.0 if row.is_active else 0.0,
            )
            for row in obstacles
        ),
        dtype=jnp.float32,
    )


@pytest.mark.parametrize(
    ("start", "end", "obstacle", "expected"),
    (
        (
            (-2.0, 0.0),
            (2.0, 0.0),
            _resolved_obstacle(
                obstacle_slot=0,
                obstacle_type_id=1,
                radius=0.5,
                is_active=True,
            ),
            False,
        ),
        (
            (-2.0, 0.5),
            (2.0, 0.5),
            _resolved_obstacle(
                obstacle_slot=0,
                obstacle_type_id=1,
                radius=0.5,
                is_active=True,
            ),
            False,
        ),
        (
            (-2.0, 0.5002),
            (2.0, 0.5002),
            _resolved_obstacle(
                obstacle_slot=0,
                obstacle_type_id=1,
                radius=0.5,
                is_active=True,
            ),
            True,
        ),
        (
            (-2.0, 0.0),
            (2.0, 0.0),
            _resolved_obstacle(
                obstacle_slot=0,
                obstacle_type_id=2,
                width=1.0,
                height=2.0,
                is_active=True,
            ),
            False,
        ),
        (
            (-2.0, -2.0),
            (2.0, 2.0),
            _resolved_obstacle(
                obstacle_slot=0,
                obstacle_type_id=2,
                width=0.5,
                height=3.0,
                theta=np.pi / 4.0,
                is_active=True,
            ),
            False,
        ),
        (
            (0.0, 0.0),
            (0.0, 0.0),
            _resolved_obstacle(
                obstacle_slot=0,
                obstacle_type_id=1,
                radius=0.5,
                is_active=True,
            ),
            False,
        ),
        (
            (2.0, 2.0),
            (2.0, 2.0),
            _resolved_obstacle(
                obstacle_slot=0,
                obstacle_type_id=2,
                width=1.0,
                height=1.0,
                theta=np.pi / 6.0,
                is_active=True,
            ),
            True,
        ),
    ),
    ids=(
        "pillar-crossing",
        "pillar-tangent",
        "pillar-clear",
        "wall-crossing",
        "rotated-wall-crossing",
        "zero-length-inside",
        "zero-length-outside",
    ),
)
def test_host_los_matches_canonical_core_float32_cases(
    start: tuple[float, float],
    end: tuple[float, float],
    obstacle: ResolvedObstacleV1,
    expected: bool,
) -> None:
    obstacles = _padded_obstacles(obstacle)
    host = _host_has_clear_line_of_sight_v1(start, end, obstacles)
    canonical = bool(
        has_clear_line_of_sight(
            jnp.asarray(start, dtype=jnp.float32),
            jnp.asarray(end, dtype=jnp.float32),
            _jax_obstacle_array(obstacles),
        )
    )

    assert host is expected
    assert host is canonical


def test_host_los_matches_canonical_core_across_deterministic_float32_corpus() -> None:
    obstacles = _padded_obstacles(
        _resolved_obstacle(
            obstacle_slot=0,
            obstacle_type_id=1,
            x=-0.75,
            y=0.5,
            radius=0.6,
            is_active=True,
        ),
        _resolved_obstacle(
            obstacle_slot=1,
            obstacle_type_id=2,
            x=1.0,
            y=-0.5,
            width=1.25,
            height=2.5,
            theta=np.pi / 7.0,
            is_active=True,
        ),
    )
    obstacle_array = _jax_obstacle_array(obstacles)
    generator = np.random.default_rng(20260827)
    points = generator.uniform(-4.0, 4.0, size=(512, 2, 2)).astype(np.float32)

    for start_array, end_array in points:
        start = (float(start_array[0]), float(start_array[1]))
        end = (float(end_array[0]), float(end_array[1]))
        assert _host_has_clear_line_of_sight_v1(start, end, obstacles) is bool(
            has_clear_line_of_sight(
                jnp.asarray(start, dtype=jnp.float32),
                jnp.asarray(end, dtype=jnp.float32),
                obstacle_array,
            )
        )


def test_host_radius_gate_matches_canonical_core_distance_at_float32_boundary() -> None:
    config = evaluation_env_config()
    state, _, _, _ = reset(config, jax.random.PRNGKey(0))
    radius = np.float32(config.agent_profile.observation_radii[0])
    distances = (
        np.nextafter(radius, np.float32(-np.inf), dtype=np.float32),
        radius,
        np.nextafter(radius, np.float32(np.inf), dtype=np.float32),
    )

    for distance in distances:
        candidate_state = state._replace(
            agent_positions=state.agent_positions.at[0]
            .set((0.0, 0.0))
            .at[5]
            .set((distance, 0.0)),
            spawn_shield_durations=jnp.zeros_like(state.spawn_shield_durations),
        )
        _, canonical_distances = _build_global_visibility_mask_and_distances(
            candidate_state,
            config,
        )
        canonical = bool(canonical_distances[0, 5] <= radius)
        host = _host_within_observation_radius_v1(
            (0.0, 0.0),
            (float(distance), 0.0),
            float(radius),
        )
        assert host is canonical
