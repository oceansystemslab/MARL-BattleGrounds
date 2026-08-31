"""Focused proof for canonical replay and recipient-POV presentation adapters."""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import jax.numpy as jnp
import pytest
from pydantic import TypeAdapter
from scripts.dev.visual_debugger import static_renderer
from tests.evaluation_fixtures import (
    CapturedEvaluationTrajectory,
    captured_evaluation_trajectory,
    evaluation_env_config,
)

from marl_battlegrounds.core.types import (
    MAX_OBSTACLE_SLOTS,
    OBSTACLE_FEATURE_ACTIVE,
    OBSTACLE_FEATURE_RADIUS,
    OBSTACLE_FEATURE_TYPE,
    OBSTACLE_FEATURE_X,
    OBSTACLE_FEATURE_Y,
    OBSTACLE_TYPE_PILLAR,
)
from marl_battlegrounds.evaluation.metrics import (
    EvaluationEpisodeObserverV1,
    EvaluationTransitionViewV1,
    build_evaluation_observer_v1,
)
from marl_battlegrounds.evaluation.models import (
    EvaluationEpisodeContextV1,
    EvaluationFrameV1,
    ResolvedEnvConfigV1,
    StaticMechanicsCatalogV1,
    canonical_digest_sha256,
)
from marl_battlegrounds.evaluation.pov import (
    build_actor_pov_current_slice_v1,
    export_actor_pov_replay_v1,
)
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
    advance_status_source_evidence_v2,
    build_evaluation_battlefield_scene_v2,
    build_researcher_analyzer_projection_v2,
    build_shared_obs_authority_source_material_projection_v1,
    build_shared_obs_source_material_projection_v1,
    build_status_source_evidence_index_v2,
    build_visual_event_batch_v2,
    initialize_status_source_evidence_v2,
    validate_oracle_scene_static_authority_v1,
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
from marl_battlegrounds.rendering.scene import (
    BattlefieldSceneV2,
    ResearcherAnalyzerProjectionV2,
    StatusSourceChannelEvidenceV2,
    StatusSourceEvidenceSceneV2,
    StatusSourceEvidenceStateV2,
    VisualEventBatchV2,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_status_source_state_rejects_noncanonical_prior_transition_text() -> None:
    evidence = StatusSourceEvidenceSceneV2(
        source_global_slot=0,
        source_public_agent_id="public-zero",
        event_id="episode:transition:00:event:0000",
    )
    channel = StatusSourceChannelEvidenceV2(
        recipient_global_slot=1,
        recipient_public_agent_id="public-one",
        status_channel=1,
        status_id="hunter_basic_slow",
        direct_source_evidence=(evidence,),
    )
    with pytest.raises(ValueError, match="canonical event before"):
        StatusSourceEvidenceStateV2(
            schema_version=2,
            episode_id="episode",
            frame_index=1,
            frame_id="episode:frame:1",
            active_statuses=(channel,),
        )


@dataclass(frozen=True, slots=True)
class _ReplaySceneCase:
    trajectory: CapturedEvaluationTrajectory
    observer: EvaluationEpisodeObserverV1
    bundle: ReplayBundleV1


class _PoisonEvaluationContextV1(EvaluationEpisodeContextV1):
    """No-extra subtype used to prove exact authority input ownership."""


class _PoisonEvaluationFrameV1(EvaluationFrameV1):
    """No-extra subtype used to prove exact authority input ownership."""


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


@pytest.fixture(scope="module")
def shared_projection_trajectory() -> CapturedEvaluationTrajectory:
    return captured_evaluation_trajectory(
        transition_count=2,
        expected_horizon=2,
        execution_information_mode="shared_obs",
        episode_id="shared-authority-source",
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


def _context_with_map_width(
    context: EvaluationEpisodeContextV1,
    *,
    map_width: float,
) -> EvaluationEpisodeContextV1:
    config_payload = context.resolved_env_config.model_dump(mode="json")
    config_payload["map_width"] = map_width
    config_payload["canonical_digest_sha256"] = canonical_digest_sha256(
        config_payload,
        exclude={"canonical_digest_sha256"},
    )
    config = ResolvedEnvConfigV1.model_validate_json(json.dumps(config_payload))
    context_payload = context.model_dump(mode="json")
    context_payload["resolved_env_config"] = config.model_dump(mode="json")
    return EvaluationEpisodeContextV1.model_validate_json(json.dumps(context_payload))


def _context_with_spawn_shield_speed(
    context: EvaluationEpisodeContextV1,
    *,
    movement_speed: float,
) -> EvaluationEpisodeContextV1:
    config_payload = context.resolved_env_config.model_dump(mode="json")
    config_payload["spawn_shield_movement_speed"] = movement_speed
    config_payload["canonical_digest_sha256"] = canonical_digest_sha256(
        config_payload,
        exclude={"canonical_digest_sha256"},
    )
    config = ResolvedEnvConfigV1.model_validate_json(json.dumps(config_payload))
    context_payload = context.model_dump(mode="json")
    context_payload["resolved_env_config"] = config.model_dump(mode="json")
    return EvaluationEpisodeContextV1.model_validate_json(json.dumps(context_payload))


def _context_with_mage_basic_damage(
    context: EvaluationEpisodeContextV1,
) -> EvaluationEpisodeContextV1:
    catalog_payload = context.static_mechanics_catalog.model_dump(mode="json")
    class_rows = catalog_payload["class_mechanics"]
    class_rows[1]["basic_raw_damage"] += 1.0
    catalog_payload["canonical_digest_sha256"] = canonical_digest_sha256(
        catalog_payload,
        exclude={"canonical_digest_sha256"},
    )
    catalog = StaticMechanicsCatalogV1.model_validate_json(json.dumps(catalog_payload))
    context_payload = context.model_dump(mode="json")
    context_payload["static_mechanics_catalog"] = catalog.model_dump(mode="json")
    return EvaluationEpisodeContextV1.model_validate_json(json.dumps(context_payload))


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
    assert tuple(
        status.status_channel
        for mechanics in scene.class_mechanics
        for status in mechanics.status_mechanics
    ) == (7, 0, 3, 1, 4, 2, 5, 6, 8)
    assert tuple(
        aura.aura_id
        for mechanics in scene.class_mechanics
        for aura in mechanics.aura_mechanics
    ) == (
        "mage_damage_amplification",
        "warrior_damage_mitigation",
    )
    assert len(scene.spawn_pads) == len(scene.agents)
    assert tuple(wave.team_index for wave in scene.respawn_waves) == (0, 1)
    assert tuple(field.aura_id for field in scene.aura_fields) == (
        "mage_damage_amplification",
        "warrior_damage_mitigation",
    )
    frame = trajectory.frames[1]
    catalog = trajectory.context.static_mechanics_catalog
    visible_by_slot = dict(
        zip(
            (
                *catalog.global_slot_by_actor_and_ally_observation_row[0],
                *catalog.global_slot_by_actor_and_enemy_observation_row[0],
            ),
            (
                *frame.base_observation.ally_visibility_mask[0],
                *frame.base_observation.enemy_visibility_mask[0],
            ),
            strict=True,
        )
    )
    assert tuple(
        (row.observer_global_slot, row.candidate_global_slot, row.visible)
        for row in scene.observer_visibility
    ) == tuple((0, slot, visible_by_slot[slot]) for slot in (0, 1, 2, 5, 6))
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


def test_slot_31_reaches_capture_actor_pov_and_researcher_rendering() -> None:
    """The expanded obstacle tail stays visible through every public join."""
    config = evaluation_env_config()
    pillar = jnp.zeros((8,), dtype=jnp.float32)
    pillar = pillar.at[OBSTACLE_FEATURE_TYPE].set(OBSTACLE_TYPE_PILLAR)
    pillar = pillar.at[OBSTACLE_FEATURE_X].set(10.0)
    pillar = pillar.at[OBSTACLE_FEATURE_Y].set(6.0)
    pillar = pillar.at[OBSTACLE_FEATURE_RADIUS].set(0.75)
    pillar = pillar.at[OBSTACLE_FEATURE_ACTIVE].set(1.0)
    config = config._replace(
        obstacles=config.obstacles.at[MAX_OBSTACLE_SLOTS - 1].set(pillar)
    )
    trajectory = captured_evaluation_trajectory(
        transition_count=0,
        expected_horizon=config.max_steps,
        config=config,
    )
    frame = trajectory.frames[0]

    actor_pov = build_actor_pov_current_slice_v1(
        trajectory.context,
        frame,
        global_slot=0,
    )
    researcher = build_researcher_analyzer_projection_v2(
        trajectory.context,
        frame,
    )

    captured_row = frame.base_observation.map_obstacle_features[0][-1]
    assert captured_row[OBSTACLE_FEATURE_ACTIVE] == 1.0
    assert actor_pov.frame.map_obstacle_features[-1] == captured_row
    assert tuple(row.obstacle_id for row in researcher.scene.map.obstacles) == (
        "obstacle-31",
    )
    assert researcher.scene.map.obstacles[0].kind == "pillar"


def test_oracle_scene_static_authority_rejects_cross_context_and_static_poison(
    replay_scene_case: _ReplaySceneCase,
) -> None:
    trajectory = replay_scene_case.trajectory
    context = trajectory.context
    view = EvaluationTransitionViewV1(
        context=context,
        start_frame=trajectory.frames[0],
        transition=trajectory.transitions[0],
        successor_frame=trajectory.frames[1],
    )
    scene = build_evaluation_battlefield_scene_v2(
        context,
        trajectory.frames[1],
        transition_view=view,
    )
    context_before = context.model_dump_json()
    scene_adapter = TypeAdapter(BattlefieldSceneV2)
    scene_before = scene_adapter.dump_json(scene)
    assert validate_oracle_scene_static_authority_v1(context, scene) is None
    assert context.model_dump_json() == context_before
    assert scene_adapter.dump_json(scene) == scene_before

    map_context = _context_with_map_width(
        context,
        map_width=context.resolved_env_config.map_width + 1.0,
    )
    assert map_context.resolved_env_config.map_width != scene.map.width
    with pytest.raises(ValueError, match=r"map.*context static authority"):
        validate_oracle_scene_static_authority_v1(map_context, scene)

    class_context = _context_with_mage_basic_damage(context)
    assert (
        class_context.static_mechanics_catalog.class_mechanics[1].basic_raw_damage
        != context.static_mechanics_catalog.class_mechanics[1].basic_raw_damage
    )
    with pytest.raises(
        ValueError,
        match=r"class mechanics.*context static authority",
    ):
        validate_oracle_scene_static_authority_v1(class_context, scene)

    agent = scene.agents[0]
    pad = scene.spawn_pads[0]
    wave = scene.respawn_waves[0]
    aura = scene.aura_fields[0]
    static_poisons = (
        replace(
            scene,
            agents=(replace(agent, radius=agent.radius + 0.25), *scene.agents[1:]),
        ),
        replace(
            scene,
            agents=(
                replace(
                    agent,
                    spawn_shield_remaining=(
                        context.resolved_env_config.spawn_shield_duration_steps + 1
                    ),
                ),
                *scene.agents[1:],
            ),
        ),
        replace(
            scene,
            spawn_pads=(
                replace(pad, position=(pad.position[0] + 0.25, pad.position[1])),
                *scene.spawn_pads[1:],
            ),
        ),
        replace(
            scene,
            respawn_waves=(
                replace(wave, period_steps=wave.period_steps + 1),
                *scene.respawn_waves[1:],
            ),
        ),
        replace(
            scene,
            aura_fields=(
                replace(aura, radius=aura.radius + 0.25),
                *scene.aura_fields[1:],
            ),
        ),
    )
    for poisoned in static_poisons:
        with pytest.raises(
            ValueError, match=r"context static authority|counters exceed"
        ):
            validate_oracle_scene_static_authority_v1(context, poisoned)

    shielded_agent = replace(
        agent,
        life_state="alive",
        spawn_shield_remaining=1,
        effective_movement_speed=(
            context.resolved_env_config.spawn_shield_movement_speed
        ),
    )
    shielded_scene = replace(
        scene,
        agents=(shielded_agent, *scene.agents[1:]),
    )
    shielded_before = scene_adapter.dump_json(shielded_scene)
    assert validate_oracle_scene_static_authority_v1(context, shielded_scene) is None
    assert scene_adapter.dump_json(shielded_scene) == shielded_before

    different_shield_speed = _context_with_spawn_shield_speed(
        context,
        movement_speed=context.resolved_env_config.spawn_shield_movement_speed + 1.0,
    )
    with pytest.raises(ValueError, match=r"active spawn shield.*speed authority"):
        validate_oracle_scene_static_authority_v1(
            different_shield_speed,
            shielded_scene,
        )

    unshielded_dynamic_speed = replace(
        scene,
        agents=(
            replace(
                agent,
                spawn_shield_remaining=0,
                effective_movement_speed=(agent.effective_movement_speed + 0.125),
            ),
            *scene.agents[1:],
        ),
    )
    assert (
        validate_oracle_scene_static_authority_v1(
            different_shield_speed,
            unshielded_dynamic_speed,
        )
        is None
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
    mage_mechanics = scene.class_mechanics[0]
    warrior_mechanics = scene.class_mechanics[1]
    with pytest.raises(ValueError, match="canonical V1 identity"):
        replace(mage_mechanics, class_name="Warrior")
    tampered_mage_mechanics = replace(mage_mechanics)
    object.__setattr__(tampered_mage_mechanics, "class_name", "Warrior")
    with pytest.raises(ValueError, match="canonical V1 class identities"):
        replace(
            scene,
            class_mechanics=(tampered_mage_mechanics, *scene.class_mechanics[1:]),
        )
    with pytest.raises(ValueError, match="source class"):
        replace(
            scene,
            class_mechanics=(
                replace(
                    mage_mechanics,
                    status_mechanics=warrior_mechanics.status_mechanics,
                ),
                replace(
                    warrior_mechanics, status_mechanics=mage_mechanics.status_mechanics
                ),
                *scene.class_mechanics[2:],
            ),
        )
    with pytest.raises(ValueError, match=r"ordered aura catalog|emitter class"):
        replace(
            scene,
            class_mechanics=(
                replace(
                    mage_mechanics, aura_mechanics=warrior_mechanics.aura_mechanics
                ),
                replace(
                    warrior_mechanics, aura_mechanics=mage_mechanics.aura_mechanics
                ),
                *scene.class_mechanics[2:],
            ),
        )

    selected_scene = build_evaluation_battlefield_scene_v2(
        replay.header.context,
        replay.frames[0],
        presentation=EvaluationScenePresentationStateV1(
            controlled_global_slot=0,
            selected_global_slot=5,
        ),
    )
    with pytest.raises(ValueError, match="ordered scene roster exactly"):
        replace(
            selected_scene,
            observer_visibility=selected_scene.observer_visibility[:-1],
        )
    self_visibility = selected_scene.observer_visibility[0]
    assert self_visibility.candidate_global_slot == 0
    dead_observer_compatible = replace(
        selected_scene,
        observer_visibility=tuple(
            replace(row, visible=False) for row in selected_scene.observer_visibility
        ),
    )
    assert not any(row.visible for row in dead_observer_compatible.observer_visibility)
    with pytest.raises(ValueError, match="controlled researcher"):
        replace(
            selected_scene,
            observer_visibility=(
                replace(self_visibility, observer_global_slot=5),
                *selected_scene.observer_visibility[1:],
            ),
        )


def test_researcher_visibility_uses_team_b_axes_and_omits_inactive_padding(
    replay_scene_case: _ReplaySceneCase,
) -> None:
    trajectory = replay_scene_case.trajectory
    frame = trajectory.frames[0]
    controlled = 5
    scene = build_evaluation_battlefield_scene_v2(
        trajectory.context,
        frame,
        presentation=EvaluationScenePresentationStateV1(
            controlled_global_slot=controlled,
            selected_global_slot=0,
        ),
    )
    catalog = trajectory.context.static_mechanics_catalog
    expected_by_slot = dict(
        zip(
            (
                *catalog.global_slot_by_actor_and_ally_observation_row[controlled],
                *catalog.global_slot_by_actor_and_enemy_observation_row[controlled],
            ),
            (
                *frame.base_observation.ally_visibility_mask[controlled],
                *frame.base_observation.enemy_visibility_mask[controlled],
            ),
            strict=True,
        )
    )
    active_slots = tuple(
        row.global_slot for row in trajectory.context.roster if row.configured_active
    )
    assert tuple(
        (row.observer_global_slot, row.candidate_global_slot, row.visible)
        for row in scene.observer_visibility
    ) == tuple(
        (controlled, candidate, expected_by_slot[candidate])
        for candidate in active_slots
    )


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

    rendered: list[tuple[BattlefieldSceneV2, VisualEventBatchV2 | None]] = []
    pyplot = SimpleNamespace(show=lambda: None)

    def capture_render(
        scene: BattlefieldSceneV2,
        *,
        event_batch: VisualEventBatchV2 | None,
    ) -> object:
        rendered.append((scene, event_batch))
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
    incoming_view = _incoming_view(loaded, frame_index)
    assert incoming_view is not None
    expected_events = build_visual_event_batch_v2(incoming_view)
    assert rendered == [(loaded_scene, expected_events)]


def test_loaded_and_live_records_produce_equal_researcher_projection(
    replay_scene_case: _ReplaySceneCase,
    tmp_path: Path,
) -> None:
    replay_path = tmp_path / "researcher-parity.marlbg-replay.json"
    save_replay_bundle_v1(replay_scene_case.bundle, replay_path)
    direct = replay_scene_case.bundle.replay
    loaded = load_replay_artifact_v1(replay_path)
    frame_index = 1
    presentation = EvaluationScenePresentationStateV1(
        controlled_global_slot=0,
        selected_global_slot=5,
        show_ranges=True,
    )
    direct_status = build_status_source_evidence_index_v2(
        direct.header.context,
        direct.frames,
        direct.transitions,
    )
    loaded_status = build_status_source_evidence_index_v2(
        loaded.header.context,
        loaded.frames,
        loaded.transitions,
    )

    live_projection = build_researcher_analyzer_projection_v2(
        direct.header.context,
        direct.frames[frame_index],
        transition_view=_incoming_view(direct, frame_index),
        presentation=presentation,
        status_source_evidence_state=direct_status.state_for_frame(frame_index),
    )
    loaded_projection = build_researcher_analyzer_projection_v2(
        loaded.header.context,
        loaded.frames[frame_index],
        transition_view=_incoming_view(loaded, frame_index),
        presentation=presentation,
        status_source_evidence_state=loaded_status.state_for_frame(frame_index),
    )

    assert loaded_projection == live_projection


def test_loaded_pov_projection_equals_live_current_slice_projection(
    replay_scene_case: _ReplaySceneCase,
    tmp_path: Path,
) -> None:
    replay_path = tmp_path / "pov-parity.marlbg-replay.json"
    save_replay_bundle_v1(replay_scene_case.bundle, replay_path)
    direct = replay_scene_case.bundle.replay
    loaded = load_replay_artifact_v1(replay_path)
    frame_index = 1
    incoming = _incoming_view(direct, frame_index)
    assert incoming is not None
    live_slice = build_actor_pov_current_slice_v1(
        direct.header.context,
        direct.frames[frame_index],
        global_slot=5,
        incoming_transition_view=incoming,
    )
    live_projection = build_actor_pov_analyzer_projection_v1(live_slice)

    loaded_pov = export_actor_pov_replay_v1(loaded, global_slot=5)
    loaded_projection = build_actor_pov_analyzer_projection_v1(
        build_actor_pov_projection_index_v1(loaded_pov.content),
        frame_index=frame_index,
    )

    assert loaded_projection == live_projection


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


@pytest.mark.parametrize("frame_index", (0, 1, 2))
def test_shared_authority_source_factory_matches_diagnostic_projection_bytes(
    shared_projection_trajectory: CapturedEvaluationTrajectory,
    frame_index: int,
) -> None:
    trajectory = shared_projection_trajectory
    context = trajectory.context
    frame = trajectory.frames[frame_index]
    context_bytes = context.model_dump_json()
    frame_bytes = frame.model_dump_json()
    view = (
        None
        if frame_index == 0
        else EvaluationTransitionViewV1(
            context=context,
            start_frame=trajectory.frames[frame_index - 1],
            transition=trajectory.transitions[frame_index - 1],
            successor_frame=frame,
        )
    )
    diagnostic = build_shared_obs_source_material_projection_v1(
        context,
        frame,
        selected_global_slot=5,
        transition_view=view,
    )
    authority = build_shared_obs_authority_source_material_projection_v1(
        context,
        frame,
        selected_global_slot=5,
    )
    adapter = TypeAdapter(SharedObsSourceMaterialProjectionV1)

    assert authority == diagnostic
    assert adapter.dump_json(authority) == adapter.dump_json(diagnostic)
    assert context.model_dump_json() == context_bytes
    assert frame.model_dump_json() == frame_bytes


def test_shared_authority_source_factory_has_no_transition_input(
    shared_projection_trajectory: CapturedEvaluationTrajectory,
) -> None:
    parameters = inspect.signature(
        build_shared_obs_authority_source_material_projection_v1
    ).parameters
    assert tuple(parameters) == ("context", "frame", "selected_global_slot")
    trajectory = shared_projection_trajectory
    frame = trajectory.frames[1]
    before = build_shared_obs_authority_source_material_projection_v1(
        trajectory.context,
        frame,
        selected_global_slot=0,
    )
    transition = trajectory.transitions[0]
    mutated_transition = transition.model_copy(
        update={
            "events": (),
            "canonical_reward_by_agent": tuple(
                value + 1.0 for value in transition.canonical_reward_by_agent
            ),
            "canonical_reward_by_team": SimpleNamespace(forbidden="team-reward"),
            "facts": SimpleNamespace(forbidden="history"),
        }
    )
    assert mutated_transition != transition
    after = build_shared_obs_authority_source_material_projection_v1(
        trajectory.context,
        frame,
        selected_global_slot=0,
    )

    assert after == before


def test_shared_authority_source_factory_rejects_exact_input_poisons(
    shared_projection_trajectory: CapturedEvaluationTrajectory,
) -> None:
    trajectory = shared_projection_trajectory
    context = trajectory.context
    frame = trajectory.frames[1]
    context_subclass = _PoisonEvaluationContextV1.model_validate_json(
        context.model_dump_json()
    )
    frame_subclass = _PoisonEvaluationFrameV1.model_validate_json(
        frame.model_dump_json()
    )
    frame_bool = cast(Any, frame).model_construct(
        **{
            **frame.model_dump(mode="python"),
            "frame_index": True,
        }
    )
    snapshot_list = frame.snapshot.model_copy(
        update={"alive_mask": list(frame.snapshot.alive_mask)}
    )
    frame_list = frame.model_copy(update={"snapshot": snapshot_list})
    cases = (
        (context_subclass, frame, 0),
        (context, frame_subclass, 0),
        (context, frame_bool, 0),
        (context, frame_list, 0),
        (context, frame, True),
    )

    for candidate_context, candidate_frame, selected_slot in cases:
        with pytest.raises((TypeError, ValueError)):
            build_shared_obs_authority_source_material_projection_v1(
                candidate_context,
                candidate_frame,
                selected_global_slot=selected_slot,
            )


def test_shared_authority_source_factory_rejects_context_and_regime_mismatch(
    shared_projection_trajectory: CapturedEvaluationTrajectory,
    replay_scene_case: _ReplaySceneCase,
) -> None:
    trajectory = shared_projection_trajectory
    other = captured_evaluation_trajectory(
        transition_count=1,
        expected_horizon=1,
        execution_information_mode="shared_obs",
        episode_id="other-shared-authority-source",
    )
    availability = [
        list(row)
        for row in cast(
            tuple[tuple[bool, ...], ...],
            trajectory.frames[
                1
            ].shared_obs_information_availability_by_recipient_and_sensor_source,
        )
    ]
    availability[0][0] = True
    invalid_availability = trajectory.frames[1].model_copy(
        update={
            "shared_obs_information_availability_by_recipient_and_sensor_source": (
                tuple(tuple(row) for row in availability)
            )
        }
    )

    with pytest.raises(ValueError, match="join to the context episode"):
        build_shared_obs_authority_source_material_projection_v1(
            trajectory.context,
            other.frames[1],
            selected_global_slot=0,
        )
    with pytest.raises(ValueError, match="requires a shared_obs episode"):
        build_shared_obs_authority_source_material_projection_v1(
            replay_scene_case.trajectory.context,
            replay_scene_case.trajectory.frames[0],
            selected_global_slot=0,
        )
    with pytest.raises(ValueError, match="availability must be false"):
        build_shared_obs_authority_source_material_projection_v1(
            trajectory.context,
            invalid_availability,
            selected_global_slot=0,
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


def test_rendering_surface_exports_stabilized_cp7_projection_seams() -> None:
    import marl_battlegrounds.rendering as rendering

    assert (
        rendering.build_evaluation_battlefield_scene_v2
        is build_evaluation_battlefield_scene_v2
    )
    assert (
        rendering.build_shared_obs_authority_source_material_projection_v1
        is build_shared_obs_authority_source_material_projection_v1
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
    assert rendering.ResearcherAnalyzerProjectionV2 is ResearcherAnalyzerProjectionV2
    assert rendering.StatusSourceEvidenceStateV2 is StatusSourceEvidenceStateV2
    assert rendering.VisualEventBatchV2 is VisualEventBatchV2
    assert (
        rendering.advance_status_source_evidence_v2 is advance_status_source_evidence_v2
    )
    assert (
        rendering.build_researcher_analyzer_projection_v2
        is build_researcher_analyzer_projection_v2
    )
    assert (
        rendering.build_status_source_evidence_index_v2
        is build_status_source_evidence_index_v2
    )
    assert rendering.build_visual_event_batch_v2 is build_visual_event_batch_v2
    assert (
        rendering.initialize_status_source_evidence_v2
        is initialize_status_source_evidence_v2
    )
