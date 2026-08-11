"""Focused proof for canonical replay and recipient-POV presentation adapters."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts.dev.visual_debugger import static_renderer
from tests.evaluation_fixtures import (
    CapturedEvaluationTrajectory,
    captured_evaluation_trajectory,
)

from marl_battlegrounds.evaluation.metrics import (
    EvaluationEpisodeObserverV1,
    EvaluationTransitionViewV1,
    build_evaluation_observer_v1,
)
from marl_battlegrounds.evaluation.pov import export_actor_pov_replay_v1
from marl_battlegrounds.evaluation.replay import (
    ReplayArtifactV1,
    ReplayBundleV1,
    RuntimeProvenanceV1,
    build_replay_bundle_v1,
)
from marl_battlegrounds.evaluation.replay_io import (
    load_replay_artifact_v1,
    save_replay_bundle_v1,
)
from marl_battlegrounds.rendering.evaluation_adapter import (
    EvaluationScenePresentationStateV1,
    SharedObsBaseSensorFrameV1,
    SharedObsSourceMaterialProjectionV1,
    build_evaluation_battlefield_scene_v2,
    build_shared_obs_source_material_projection_v1,
)
from marl_battlegrounds.rendering.evaluation_wire_features import (
    AGENT_FEATURE_ACTIVE_V1,
    AGENT_FEATURE_CLASS_ID_V1,
    AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER_V1,
    AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER_V1,
    AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED_V1,
    AGENT_FEATURE_TEAM_ID_V1,
    AGENT_FEATURE_ULTIMATE_COOLDOWN_REMAINING_V1,
    OBSTACLE_FEATURE_ACTIVE_V1,
    OBSTACLE_FEATURE_TYPE_V1,
)
from marl_battlegrounds.rendering.pov_scene import (
    ActorPovBattlefieldSceneV1,
    ActorPovProjectionIndexV1,
    build_actor_pov_analyzer_projection_v1,
    build_actor_pov_projection_index_v1,
)
from marl_battlegrounds.rendering.scene import BattlefieldSceneV2

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class _ReplaySceneCase:
    trajectory: CapturedEvaluationTrajectory
    observer: EvaluationEpisodeObserverV1
    bundle: ReplayBundleV1


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


@pytest.fixture(scope="module")
def replay_scene_case() -> _ReplaySceneCase:
    trajectory = captured_evaluation_trajectory(
        transition_count=2,
        expected_horizon=2,
    )
    observer = build_evaluation_observer_v1(trajectory.context)
    observer.start(trajectory.frames[0])
    for transition, successor in zip(
        trajectory.transitions,
        trajectory.frames[1:],
        strict=True,
    ):
        observer.append(transition, successor)
    report = observer.finalize(completion_state="complete")
    return _ReplaySceneCase(
        trajectory=trajectory,
        observer=observer,
        bundle=build_replay_bundle_v1(
            observer,
            report,
            runtime_provenance=_runtime_provenance(),
        ),
    )


def _incoming_view(
    replay: ReplayArtifactV1,
    frame_index: int,
) -> EvaluationTransitionViewV1 | None:
    if frame_index == 0:
        return None
    return EvaluationTransitionViewV1(
        context=replay.header.context,
        start_frame=replay.frames[frame_index - 1],
        transition=replay.transitions[frame_index - 1],
        successor_frame=replay.frames[frame_index],
    )


def test_researcher_scene_v2_projects_only_canonical_context_and_frame_truth(
    replay_scene_case: _ReplaySceneCase,
) -> None:
    trajectory = replay_scene_case.trajectory
    presentation = EvaluationScenePresentationStateV1(
        controlled_global_slot=0,
        selected_global_slot=5,
        armed_lane=0,
    )
    view = EvaluationTransitionViewV1(
        context=trajectory.context,
        start_frame=trajectory.frames[0],
        transition=trajectory.transitions[0],
        successor_frame=trajectory.frames[1],
    )

    scene = build_evaluation_battlefield_scene_v2(
        trajectory.context,
        trajectory.frames[1],
        transition_view=view,
        presentation=presentation,
    )

    assert type(scene) is BattlefieldSceneV2
    assert scene.frame_id == trajectory.frames[1].frame_id
    assert scene.incoming_transition_id == trajectory.transitions[0].transition_id
    assert scene.incoming_event_ids == tuple(
        event.event_id for event in trajectory.transitions[0].events
    )
    assert tuple(agent.global_slot for agent in scene.agents) == (0, 1, 2, 5, 6)
    assert tuple(agent.public_agent_id for agent in scene.agents) == (
        "agent-slot-0",
        "agent-slot-1",
        "agent-slot-2",
        "agent-slot-5",
        "agent-slot-6",
    )
    assert tuple(row.class_id for row in scene.class_mechanics) == (1, 2, 3, 4, 5)
    assert len(scene.spawn_pads) == len(scene.agents)
    assert tuple(wave.team_index for wave in scene.respawn_waves) == (0, 1)
    assert tuple(field.aura_id for field in scene.aura_fields) == (
        "mage_damage_amplification",
        "warrior_damage_mitigation",
    )
    frame = trajectory.frames[1]
    first_agent = scene.agents[0]
    self_row = frame.base_observation.self_features[0]
    assert (
        first_agent.effective_movement_speed
        == self_row[AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED_V1]
    )
    assert tuple(row.multiplier for row in first_agent.aura_modifiers) == (
        self_row[AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER_V1],
        self_row[AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER_V1],
    )
    assert scene.selection is not None
    assert scene.next_decision_selected_legality is not None
    target_action = scene.next_decision_selected_legality.target_action
    assert (
        scene.next_decision_selected_legality.lane_0_available
        == frame.action_mask.select_target_use_ultimate_joint_mask[0][target_action][0]
    )


def test_researcher_adapter_requires_the_exact_incoming_view(
    replay_scene_case: _ReplaySceneCase,
) -> None:
    trajectory = replay_scene_case.trajectory
    with pytest.raises(ValueError, match="require their coherent incoming"):
        build_evaluation_battlefield_scene_v2(
            trajectory.context,
            trajectory.frames[1],
        )
    wrong_view = EvaluationTransitionViewV1(
        context=trajectory.context,
        start_frame=trajectory.frames[1],
        transition=trajectory.transitions[1],
        successor_frame=trajectory.frames[2],
    )
    with pytest.raises(ValueError, match="successor must equal"):
        build_evaluation_battlefield_scene_v2(
            trajectory.context,
            trajectory.frames[1],
            transition_view=wrong_view,
        )
    with pytest.raises(ValueError, match="cannot be reused"):
        build_evaluation_battlefield_scene_v2(
            trajectory.context,
            trajectory.frames[0],
            audience="agent_pov",  # type: ignore[arg-type]
        )


def test_scene_v2_rejects_contradictory_nested_identity(
    replay_scene_case: _ReplaySceneCase,
) -> None:
    replay = replay_scene_case.bundle.replay
    scene = build_evaluation_battlefield_scene_v2(
        replay.header.context,
        replay.frames[0],
    )
    bad_pad = replace(
        scene.spawn_pads[0],
        assigned_public_agent_id="wrong-agent",
    )
    with pytest.raises(ValueError, match="spawn pads must join"):
        replace(scene, spawn_pads=(bad_pad, *scene.spawn_pads[1:]))
    with pytest.raises(ValueError, match="at least 1"):
        replace(scene.class_mechanics[0], class_id=0)


def test_loaded_and_direct_records_produce_equal_static_scene(
    replay_scene_case: _ReplaySceneCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_path = tmp_path / "adapter.marlbg-replay.json"
    save_replay_bundle_v1(replay_scene_case.bundle, replay_path)
    loaded = load_replay_artifact_v1(replay_path)
    frame_index = 1
    direct_scene = build_evaluation_battlefield_scene_v2(
        replay_scene_case.bundle.replay.header.context,
        replay_scene_case.bundle.replay.frames[frame_index],
        transition_view=_incoming_view(replay_scene_case.bundle.replay, frame_index),
    )
    loaded_scene = build_evaluation_battlefield_scene_v2(
        loaded.header.context,
        loaded.frames[frame_index],
        transition_view=_incoming_view(loaded, frame_index),
    )
    assert loaded_scene == direct_scene

    rendered: list[BattlefieldSceneV2] = []
    pyplot = SimpleNamespace(show=lambda: None)

    def capture_render(scene: BattlefieldSceneV2) -> object:
        rendered.append(scene)
        return object()

    monkeypatch.setattr(static_renderer, "_load_pyplot", lambda: pyplot)
    monkeypatch.setattr(
        static_renderer,
        "render_scene_geometry",
        capture_render,
    )
    assert (
        static_renderer.run_static_replay_renderer(
            replay_path=replay_path,
            frame_index=frame_index,
            show_ranges=True,
        )
        == 0
    )
    assert rendered == [loaded_scene]


def test_static_replay_validates_before_loading_matplotlib(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def fail_if_loaded() -> object:
        nonlocal called
        called = True
        raise AssertionError("Matplotlib must not load for an invalid replay")

    monkeypatch.setattr(static_renderer, "_load_pyplot", fail_if_loaded)
    with pytest.raises(ValueError, match="Replay could not be loaded"):
        static_renderer.run_static_replay_renderer(
            replay_path=tmp_path / "missing.marlbg-replay.json",
            frame_index=0,
            show_ranges=True,
        )
    assert not called


def test_actor_pov_projection_is_independent_and_recipient_sliced(
    replay_scene_case: _ReplaySceneCase,
) -> None:
    pov = export_actor_pov_replay_v1(
        replay_scene_case.bundle.replay,
        global_slot=5,
    )
    index = build_actor_pov_projection_index_v1(pov.content)
    projection = build_actor_pov_analyzer_projection_v1(index, frame_index=1)

    assert type(projection.scene) is ActorPovBattlefieldSceneV1
    assert not isinstance(projection.scene, BattlefieldSceneV2)
    assert projection.scene.self_actor.public_agent_id == pov.content.public_agent_id
    assert (
        projection.scene.self_actor.spawn_shield_remaining
        == (
            pov.content.frames[1].spawn_lifecycle.spawn_shield_actual_durations_by_team[
                0
            ][pov.content.selected_team_local_slot]
        )
    )
    assert tuple(row.team_label for row in projection.scene.respawn_waves) == (
        "Own Team",
        "Opponent Team",
    )
    for body in projection.scene.visible_bodies:
        mapping = (
            pov.content.axis_mapping.ally_observation_row_public_agent_id_by_id
            if body.relation == "ally"
            else pov.content.axis_mapping.enemy_observation_row_public_agent_id_by_id
        )
        assert body.public_agent_id == mapping[body.observation_row]
    assert projection.incoming_cues == pov.content.transitions[0].cues
    assert not hasattr(projection.scene, "incoming_event_ids")
    assert not hasattr(projection.scene, "class_mechanics")
    assert not hasattr(projection.scene, "aura_fields")


def test_shared_obs_projection_discloses_only_source_material_for_team_b() -> None:
    trajectory = captured_evaluation_trajectory(
        transition_count=1,
        expected_horizon=1,
        execution_information_mode="shared_obs",
    )
    selected_global_slot = 5
    view = EvaluationTransitionViewV1(
        context=trajectory.context,
        start_frame=trajectory.frames[0],
        transition=trajectory.transitions[0],
        successor_frame=trajectory.frames[1],
    )
    projection = build_shared_obs_source_material_projection_v1(
        trajectory.context,
        trajectory.frames[1],
        selected_global_slot=selected_global_slot,
        transition_view=view,
    )

    assert type(projection) is SharedObsSourceMaterialProjectionV1
    assert type(projection.base_sensor_frame) is SharedObsBaseSensorFrameV1
    assert projection.observation_materialization == "source_material_only"
    assert projection.exact_actor_input_export_available is False
    assert "NOT MATERIALIZED SHAREDOBS ACTOR INPUT" in projection.disclosure_label
    assert "actor-pov" not in projection.base_sensor_frame.source_material_frame_id
    assert projection.base_sensor_scene.self_actor.public_agent_id == "agent-slot-5"
    lifecycle = projection.base_sensor_frame.spawn_lifecycle
    assert (
        projection.base_sensor_scene.self_actor.spawn_shield_remaining
        == lifecycle.spawn_shield_actual_durations_by_team[0][0]
    )
    assert not hasattr(projection.base_sensor_frame, "snapshot")
    assert not hasattr(projection.base_sensor_scene, "incoming_event_ids")
    rows = projection.sensor_source_availability
    assert tuple(row.sensor_source_global_slot for row in rows) == tuple(range(10))
    availability = trajectory.frames[
        1
    ].shared_obs_information_availability_by_recipient_and_sensor_source
    assert availability is not None
    assert (
        tuple(row.recorded_available for row in rows)
        == availability[selected_global_slot]
    )
    assert rows[5].relation_to_recipient == "self"
    assert rows[5].base_sensor_relation_axis == "ally"
    assert rows[6].relation_to_recipient == "ally"
    assert rows[6].recorded_available
    assert rows[0].relation_to_recipient == "opponent"
    assert not rows[0].recorded_available
    assert rows[3].relation_to_recipient == "inactive"
    assert rows[3].sensor_source_configured_team_id == 0
    assert not rows[3].recorded_available

    relabelled = replace(rows[0], sensor_source_public_agent_id="forged-agent")
    with pytest.raises(ValueError, match="public ID does not join"):
        replace(
            projection,
            sensor_source_availability=(relabelled, *rows[1:]),
        )
    self_relabelled = replace(rows[5], sensor_source_public_agent_id="agent-slot-6")
    with pytest.raises(ValueError, match="public ID does not join"):
        replace(
            projection,
            sensor_source_availability=(*rows[:5], self_relabelled, *rows[6:]),
        )


def test_shared_obs_source_material_builder_rejects_no_shared_obs(
    replay_scene_case: _ReplaySceneCase,
) -> None:
    replay = replay_scene_case.bundle.replay
    with pytest.raises(ValueError, match="requires a shared_obs episode"):
        build_shared_obs_source_material_projection_v1(
            replay.header.context,
            replay.frames[0],
            selected_global_slot=0,
        )


@pytest.mark.parametrize(
    "feature_index, tampered_value",
    (
        (AGENT_FEATURE_ACTIVE_V1, 0.5),
        (AGENT_FEATURE_TEAM_ID_V1, 1.5),
        (AGENT_FEATURE_CLASS_ID_V1, 1.5),
        (AGENT_FEATURE_ULTIMATE_COOLDOWN_REMAINING_V1, 1.5),
    ),
)
def test_actor_pov_projection_fails_closed_on_fractional_self_wire_values(
    replay_scene_case: _ReplaySceneCase,
    feature_index: int,
    tampered_value: float,
) -> None:
    pov = export_actor_pov_replay_v1(
        replay_scene_case.bundle.replay,
        global_slot=0,
    )
    content = pov.content
    frame = content.frames[0]
    self_values = list(frame.self_features)
    self_values[feature_index] = tampered_value
    tampered_frame = frame.model_copy(update={"self_features": tuple(self_values)})
    tampered_content = content.model_copy(
        update={"frames": (tampered_frame, *content.frames[1:])}
    )
    forged_index = object.__new__(ActorPovProjectionIndexV1)
    object.__setattr__(forged_index, "content", tampered_content)

    with pytest.raises(ValueError, match="wire"):
        build_actor_pov_analyzer_projection_v1(forged_index, frame_index=0)


def test_actor_pov_projection_fails_closed_on_fractional_obstacle_type(
    replay_scene_case: _ReplaySceneCase,
) -> None:
    pov = export_actor_pov_replay_v1(
        replay_scene_case.bundle.replay,
        global_slot=0,
    )
    content = pov.content
    frame = content.frames[0]
    obstacle_rows = list(frame.map_obstacle_features)
    first_obstacle = list(obstacle_rows[0])
    first_obstacle[OBSTACLE_FEATURE_ACTIVE_V1] = 1.0
    first_obstacle[OBSTACLE_FEATURE_TYPE_V1] = 1.5
    obstacle_rows[0] = tuple(first_obstacle)
    tampered_frame = frame.model_copy(
        update={"map_obstacle_features": tuple(obstacle_rows)}
    )
    tampered_content = content.model_copy(
        update={"frames": (tampered_frame, *content.frames[1:])}
    )
    forged_index = object.__new__(ActorPovProjectionIndexV1)
    object.__setattr__(forged_index, "content", tampered_content)

    with pytest.raises(ValueError, match="integer-valued"):
        build_actor_pov_analyzer_projection_v1(forged_index, frame_index=0)


def test_offline_rendering_wire_constants_match_v1_core_layout() -> None:
    from marl_battlegrounds.core import types as core_types

    assert AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED_V1 == (
        core_types.AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED
    )
    assert AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER_V1 == (
        core_types.AGENT_FEATURE_DAMAGE_AMPLIFICATION_MAGE_AURA_MULTIPLIER
    )
    assert AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER_V1 == (
        core_types.AGENT_FEATURE_DAMAGE_MITIGATION_WARRIOR_AURA_MULTIPLIER
    )


def test_offline_adapter_and_static_module_import_without_core_or_array_backends() -> (
    None
):
    code = """
import sys
import marl_battlegrounds.rendering.evaluation_adapter
import marl_battlegrounds.rendering.pov_scene
import scripts.dev.visual_debugger.static_renderer
for prefix in ('jax', 'jaxlib', 'numpy', 'marl_battlegrounds.core'):
    loaded = any(
        name == prefix or name.startswith(prefix + '.') for name in sys.modules
    )
    assert not loaded, prefix
print('isolated')
"""
    result = subprocess.run(
        (sys.executable, "-c", code),
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "isolated"


def test_cp63_rendering_surface_exports_only_stabilized_projection_seams() -> None:
    import marl_battlegrounds.rendering as rendering

    assert (
        rendering.build_evaluation_battlefield_scene_v2
        is build_evaluation_battlefield_scene_v2
    )
    assert (
        rendering.build_shared_obs_source_material_projection_v1
        is build_shared_obs_source_material_projection_v1
    )
    assert rendering.SharedObsBaseSensorFrameV1 is SharedObsBaseSensorFrameV1
    assert (
        rendering.SharedObsSourceMaterialProjectionV1
        is SharedObsSourceMaterialProjectionV1
    )
    assert not hasattr(rendering, "ResearcherAnalyzerProjectionV2")
    assert not hasattr(rendering, "VisualEventBatchV2")
