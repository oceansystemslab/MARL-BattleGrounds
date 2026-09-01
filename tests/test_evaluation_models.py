"""Contract tests for strict versioned evaluation context models."""

import inspect
import json
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, cast

import numpy as np
import pytest
from jax import Array
from pydantic import ValidationError
from tests.evaluation_fixtures import (
    evaluation_context,
    evaluation_env_config,
)

import marl_battlegrounds.evaluation as evaluation_api
from marl_battlegrounds.core.axis_mappings import (
    GLOBAL_RECIPIENT_SLOT_BY_ACTOR_AND_TARGET_ACTION,
    GLOBAL_SLOT_BY_ACTOR_AND_ALLY_OBSERVATION_ROW,
    GLOBAL_SLOT_BY_ACTOR_AND_ENEMY_OBSERVATION_ROW,
    MOVEMENT_ACTION_NAME_BY_ID,
    TARGET_ACTION_NAME_BY_ID,
    UNIT_DIRECTION_VECTOR_BY_MOVEMENT_ACTION,
)
from marl_battlegrounds.core.types import (
    Action as CoreAction,
)
from marl_battlegrounds.core.types import (
    ActionAcceptanceFacts as CoreActionAcceptanceFacts,
)
from marl_battlegrounds.core.types import (
    AuraTransitionFacts as CoreAuraTransitionFacts,
)
from marl_battlegrounds.core.types import (
    CombatTransitionFacts as CoreCombatTransitionFacts,
)
from marl_battlegrounds.core.types import (
    DeathTransitionFacts as CoreDeathTransitionFacts,
)
from marl_battlegrounds.core.types import (
    PhysicalTransitionFacts as CorePhysicalTransitionFacts,
)
from marl_battlegrounds.core.types import (
    RegenerationTransitionFacts as CoreRegenerationTransitionFacts,
)
from marl_battlegrounds.core.types import (
    RespawnTransitionFacts as CoreRespawnTransitionFacts,
)
from marl_battlegrounds.core.types import (
    SpawnShieldTransitionFacts as CoreSpawnShieldTransitionFacts,
)
from marl_battlegrounds.core.types import (
    StatusLifecycleTransitionFacts as CoreStatusLifecycleTransitionFacts,
)
from marl_battlegrounds.core.types import (
    TeamDeathmatchTransitionFacts as CoreTeamDeathmatchTransitionFacts,
)
from marl_battlegrounds.core.types import (
    TransitionFacts as CoreTransitionFacts,
)
from marl_battlegrounds.evaluation.catalog import (
    build_code_revision_v1,
    build_resolved_env_config_v1,
    build_static_mechanics_catalog_v1,
)
from marl_battlegrounds.evaluation.models import (
    CATALOG_SCHEMA_ID,
    CONTEXT_SCHEMA_ID,
    REQUIRED_SCHEMA_BINDINGS_V1,
    RESOLVED_ENV_CONFIG_SCHEMA_ID,
    ActionAcceptanceFactsV1,
    ActionMaskV1,
    ActionRejectedEventV1,
    AssignedPolicySlotV1,
    AuraTransitionFactsV1,
    BaseObservationV1,
    CodeRevisionV1,
    CombatTransitionFactsV1,
    DeathTransitionFactsV1,
    EvaluationEpisodeContextV1,
    EvaluationFrameV1,
    EvaluationTransitionV1,
    ExecutionInformationMode,
    GlobalAnalysisSnapshotV1,
    JointActionV1,
    NotApplicablePolicySlotV1,
    PhysicalTransitionFactsV1,
    PreviousTimestepActionObservationV1,
    RegenerationTransitionFactsV1,
    ResolvedEnvConfigV1,
    RespawnTransitionFactsV1,
    SourceDamageOutputEventV1,
    SpawnLifecycleObservationV1,
    SpawnShieldTransitionFactsV1,
    StaticMechanicsCatalogV1,
    StatusLifecycleTransitionFactsV1,
    TeamDeathmatchTransitionFactsV1,
    TransitionFactsV1,
    VersionedIdentityV1,
    canonical_digest_sha256,
    canonical_json_bytes,
)
from marl_battlegrounds.evaluation.wire_shapes import MAX_OBSTACLE_SLOTS_V1


def _json_payload(
    model: StaticMechanicsCatalogV1 | EvaluationEpisodeContextV1,
) -> dict[str, Any]:
    payload = json.loads(model.model_dump_json())
    return cast(dict[str, Any], payload)


def _nested_keys(value: object) -> Iterator[str]:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        for key, item in mapping.items():
            yield str(key)
            yield from _nested_keys(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        sequence = cast(Sequence[object], value)
        for item in sequence:
            yield from _nested_keys(item)


def _nested_values(value: object) -> Iterator[object]:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        for item in mapping.values():
            yield from _nested_values(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        sequence = cast(Sequence[object], value)
        for item in sequence:
            yield from _nested_values(item)
    else:
        yield value


def _filled_tuple(shape: tuple[int, ...], value: object) -> object:
    if not shape:
        return value
    return tuple(_filled_tuple(shape[1:], value) for _index in range(shape[0]))


def _valid_snapshot() -> GlobalAnalysisSnapshotV1:
    zeros_10 = cast(tuple[int, ...], _filled_tuple((10,), 0))
    return GlobalAnalysisSnapshotV1(
        team_deathmatch_scores=(0, 0),
        alive_mask=cast(tuple[bool, ...], _filled_tuple((10,), False)),
        agent_positions=cast(
            tuple[tuple[float, ...], ...], _filled_tuple((10, 2), 0.0)
        ),
        current_health=cast(tuple[float, ...], _filled_tuple((10,), 0.0)),
        ultimate_cooldowns=zeros_10,
        slow_durations=cast(tuple[tuple[int, ...], ...], _filled_tuple((10, 3), 0)),
        stun_durations=cast(tuple[tuple[int, ...], ...], _filled_tuple((10, 3), 0)),
        rogue_poison_anti_heal_durations=zeros_10,
        mage_burst_damage_amplification_durations=zeros_10,
        priest_blessing_of_freedom_slow_floor_durations=zeros_10,
        team_respawn_wave_countdowns=(0, 0),
        spawn_shield_durations=zeros_10,
        steps_until_out_of_combat=zeros_10,
        previous_timestep_move_actions=zeros_10,
        previous_timestep_select_target_actions=zeros_10,
        previous_timestep_use_ultimate_actions=zeros_10,
        has_previous_timestep_joint_action=False,
    )


def _valid_base_observation() -> BaseObservationV1:
    previous = PreviousTimestepActionObservationV1(
        ally_previous_timestep_move_actions_one_hot=cast(
            tuple[tuple[tuple[float, ...], ...], ...],
            _filled_tuple((10, 5, 9), 0.0),
        ),
        enemy_previous_timestep_move_actions_one_hot=cast(
            tuple[tuple[tuple[float, ...], ...], ...],
            _filled_tuple((10, 5, 9), 0.0),
        ),
        ally_previous_timestep_select_target_actions_one_hot=cast(
            tuple[tuple[tuple[float, ...], ...], ...],
            _filled_tuple((10, 5, 11), 0.0),
        ),
        enemy_previous_timestep_select_target_actions_one_hot=cast(
            tuple[tuple[tuple[float, ...], ...], ...],
            _filled_tuple((10, 5, 11), 0.0),
        ),
        ally_previous_timestep_use_ultimate_actions_one_hot=cast(
            tuple[tuple[tuple[float, ...], ...], ...],
            _filled_tuple((10, 5, 2), 0.0),
        ),
        enemy_previous_timestep_use_ultimate_actions_one_hot=cast(
            tuple[tuple[tuple[float, ...], ...], ...],
            _filled_tuple((10, 5, 2), 0.0),
        ),
    )
    lifecycle = SpawnLifecycleObservationV1(
        spawn_pad_positions_by_agent_by_team=cast(
            tuple[tuple[tuple[tuple[float, ...], ...], ...], ...],
            _filled_tuple((10, 2, 5, 2), 0.0),
        ),
        spawn_shield_actual_durations_by_agent_by_team=cast(
            tuple[tuple[tuple[int, ...], ...], ...],
            _filled_tuple((10, 2, 5), 0),
        ),
        spawn_shield_configured_duration_by_agent=cast(
            tuple[int, ...], _filled_tuple((10,), 0)
        ),
        spawn_shield_speed_by_agent=cast(tuple[float, ...], _filled_tuple((10,), 0.0)),
        respawn_wave_period_step_count_by_agent_by_team=cast(
            tuple[tuple[int, ...], ...], _filled_tuple((10, 2), 0)
        ),
        respawn_wave_countdowns_by_agent_by_team=cast(
            tuple[tuple[int, ...], ...], _filled_tuple((10, 2), 0)
        ),
        active_mask_by_agent_by_team=cast(
            tuple[tuple[tuple[bool, ...], ...], ...],
            _filled_tuple((10, 2, 5), False),
        ),
        alive_mask_by_agent_by_team=cast(
            tuple[tuple[tuple[bool, ...], ...], ...],
            _filled_tuple((10, 2, 5), False),
        ),
    )
    return BaseObservationV1(
        self_features=cast(tuple[tuple[float, ...], ...], _filled_tuple((10, 58), 0.0)),
        ally_unit_features=cast(
            tuple[tuple[tuple[float, ...], ...], ...],
            _filled_tuple((10, 5, 58), 0.0),
        ),
        enemy_unit_features=cast(
            tuple[tuple[tuple[float, ...], ...], ...],
            _filled_tuple((10, 5, 58), 0.0),
        ),
        map_obstacle_features=cast(
            tuple[tuple[tuple[float, ...], ...], ...],
            _filled_tuple((10, MAX_OBSTACLE_SLOTS_V1, 8), 0.0),
        ),
        objective_features=cast(
            tuple[tuple[tuple[float, ...], ...], ...],
            _filled_tuple((10, 8, 12), 0.0),
        ),
        context_features=cast(
            tuple[tuple[float, ...], ...], _filled_tuple((10, 19), 0.0)
        ),
        ally_visibility_mask=cast(
            tuple[tuple[bool, ...], ...], _filled_tuple((10, 5), False)
        ),
        enemy_visibility_mask=cast(
            tuple[tuple[bool, ...], ...], _filled_tuple((10, 5), False)
        ),
        previous_timestep_actions=previous,
        spawn_lifecycle=lifecycle,
    )


def _valid_action_mask() -> ActionMaskV1:
    move_rows = tuple((True, *([False] * 8)) for _actor in range(10))
    target_rows = tuple((True, *([False] * 10)) for _actor in range(10))
    ultimate_rows = tuple((True, False) for _actor in range(10))
    joint_rows = tuple(
        ((True, False), *((False, False) for _target in range(10)))
        for _actor in range(10)
    )
    return ActionMaskV1(
        move_mask=move_rows,
        select_target_mask=target_rows,
        use_ultimate_mask=ultimate_rows,
        select_target_use_ultimate_joint_mask=joint_rows,
    )


def _valid_transition_facts(*, has_transition: bool = True) -> TransitionFactsV1:
    bool_10 = cast(tuple[bool, ...], _filled_tuple((10,), False))
    float_10 = cast(tuple[float, ...], _filled_tuple((10,), 0.0))
    int_10 = cast(tuple[int, ...], _filled_tuple((10,), 0))
    action = JointActionV1(move=int_10, select_target=int_10, use_ultimate=int_10)
    return TransitionFactsV1(
        has_transition=has_transition,
        transition_start_step_count=0 if has_transition else -1,
        action_acceptance_facts=ActionAcceptanceFactsV1(
            submitted_joint_action=action,
            accepted_joint_action=action,
            submitted_action_tuple_is_out_of_domain_by_actor=bool_10,
            in_domain_move_action_is_rejected_by_actor=bool_10,
            in_domain_combat_action_pair_is_rejected_by_actor=bool_10,
        ),
        combat_transition_facts=CombatTransitionFactsV1(
            basic_effect_is_activated_by_source=bool_10,
            ultimate_effect_is_activated_by_source=bool_10,
            combat_effect_has_recipient_by_source=bool_10,
            combat_effect_recipient_global_slot_by_source=cast(
                tuple[int | None, ...], _filled_tuple((10,), None)
            ),
            raw_damage_output_by_source=float_10,
            source_modified_damage_output_by_source=float_10,
            recipient_damage_modifier_by_source=float_10,
            total_effective_damage_by_recipient=float_10,
            raw_healing_output_by_source=float_10,
            source_modified_healing_output_by_source=float_10,
            recipient_healing_modifier_by_source=float_10,
            total_effective_healing_by_recipient=float_10,
            health_after_combat_resolution_by_recipient=float_10,
            slow_is_applied_by_source_and_channel=cast(
                tuple[tuple[bool, ...], ...], _filled_tuple((10, 3), False)
            ),
            stun_is_applied_by_source_and_channel=cast(
                tuple[tuple[bool, ...], ...], _filled_tuple((10, 3), False)
            ),
            rogue_poison_anti_heal_is_applied_by_source=bool_10,
            mage_burst_damage_amplification_is_applied_by_source=bool_10,
            priest_blessing_of_freedom_is_applied_by_source=bool_10,
        ),
        death_facts=DeathTransitionFactsV1(
            is_newly_dead_by_recipient=bool_10,
            contributed_to_new_death_by_source=bool_10,
            attributed_death_damage_by_source=float_10,
        ),
        spawn_shield_facts=SpawnShieldTransitionFactsV1(
            was_active_at_transition_start_by_agent=bool_10,
            expired_at_transition_end_by_agent=bool_10,
        ),
        respawn_facts=RespawnTransitionFactsV1(
            respawn_wave_occurred_this_transition_by_team=(False, False),
            was_respawned_this_transition_by_agent=bool_10,
        ),
        regeneration_facts=RegenerationTransitionFactsV1(
            combat_countdown_was_reset_by_agent=bool_10,
            actual_health_regenerated_this_step_by_agent=float_10,
        ),
        physical_facts=PhysicalTransitionFactsV1(
            charge_phase_displacement_by_agent=cast(
                tuple[tuple[float, ...], ...], _filled_tuple((10, 2), 0.0)
            ),
            ordinary_movement_phase_displacement_by_agent=cast(
                tuple[tuple[float, ...], ...], _filled_tuple((10, 2), 0.0)
            ),
        ),
        aura_facts=AuraTransitionFactsV1(
            is_covered_by_mage_damage_aura_by_emitter_and_beneficiary=cast(
                tuple[tuple[bool, ...], ...], _filled_tuple((10, 10), False)
            ),
            is_covered_by_warrior_mitigation_aura_by_emitter_and_beneficiary=cast(
                tuple[tuple[bool, ...], ...], _filled_tuple((10, 10), False)
            ),
        ),
        status_lifecycle_facts=StatusLifecycleTransitionFactsV1(
            aged_to_zero_by_recipient_and_status_channel=cast(
                tuple[tuple[bool, ...], ...], _filled_tuple((10, 9), False)
            ),
            refreshed_or_extended_by_recipient_and_status_channel=cast(
                tuple[tuple[bool, ...], ...], _filled_tuple((10, 9), False)
            ),
            broken_by_damage_by_recipient_and_status_channel=cast(
                tuple[tuple[bool, ...], ...], _filled_tuple((10, 9), False)
            ),
            cleared_by_new_death_by_recipient_and_status_channel=cast(
                tuple[tuple[bool, ...], ...], _filled_tuple((10, 9), False)
            ),
        ),
        team_deathmatch_facts=TeamDeathmatchTransitionFactsV1(outcome=0),
    )


def _valid_frame(*, frame_index: int = 0) -> EvaluationFrameV1:
    return EvaluationFrameV1(
        episode_id="episode-001",
        frame_index=frame_index,
        frame_id=f"episode-001:frame:{frame_index}",
        simulator_step_count=frame_index,
        snapshot=_valid_snapshot(),
        base_observation=_valid_base_observation(),
        action_mask=_valid_action_mask(),
        shared_obs_information_availability_by_recipient_and_sensor_source=None,
    )


def test_static_catalog_projects_exact_supported_public_authorities() -> None:
    catalog = build_static_mechanics_catalog_v1()

    assert catalog.schema_id == CATALOG_SCHEMA_ID
    assert catalog.schema_version == 1
    assert catalog.duration_unit == "transition_ticks"
    assert catalog.health_unit == "hit_points"
    assert catalog.spatial_unit == "world_units"
    assert catalog.movement_action_name_by_id == MOVEMENT_ACTION_NAME_BY_ID
    assert catalog.target_action_name_by_id == TARGET_ACTION_NAME_BY_ID
    assert catalog.global_recipient_slot_by_actor_and_target_action == (
        GLOBAL_RECIPIENT_SLOT_BY_ACTOR_AND_TARGET_ACTION
    )
    assert catalog.global_slot_by_actor_and_ally_observation_row == (
        GLOBAL_SLOT_BY_ACTOR_AND_ALLY_OBSERVATION_ROW
    )
    assert catalog.global_slot_by_actor_and_enemy_observation_row == (
        GLOBAL_SLOT_BY_ACTOR_AND_ENEMY_OBSERVATION_ROW
    )
    assert catalog.unit_direction_vector_by_movement_action == (
        UNIT_DIRECTION_VECTOR_BY_MOVEMENT_ACTION
    )
    assert catalog.use_ultimate_action_name_by_id == (
        "Do Not Use Ultimate",
        "Use Ultimate",
    )
    assert catalog.health_effect_stage_name_by_id == (
        "raw_source",
        "source_modified_gross",
        "recipient_modified_gross",
        "combat_resolution_health",
        "realized_net_health_change",
        "actual_regeneration",
    )


def test_static_catalog_retains_mapping_shapes_null_and_float32_values() -> None:
    catalog = build_static_mechanics_catalog_v1()

    assert np.asarray(
        catalog.global_recipient_slot_by_actor_and_target_action,
        dtype=object,
    ).shape == (10, 11)
    assert np.asarray(
        catalog.global_slot_by_actor_and_ally_observation_row,
    ).shape == (10, 5)
    assert np.asarray(
        catalog.global_slot_by_actor_and_enemy_observation_row,
    ).shape == (10, 5)
    assert np.asarray(
        catalog.unit_direction_vector_by_movement_action,
        dtype=np.float32,
    ).shape == (9, 2)
    assert all(
        row[0] is None
        for row in catalog.global_recipient_slot_by_actor_and_target_action
    )
    assert catalog.unit_direction_vector_by_movement_action[5][0] == float(
        np.float32(1.0 / np.sqrt(2.0))
    )
    assert catalog.status_channels[4].status_id == "hunter_trap_stun"
    assert catalog.status_channels[4].breaks_on_positive_damage is True


def test_static_catalog_digest_and_json_roundtrip_are_canonical() -> None:
    catalog = build_static_mechanics_catalog_v1()

    assert catalog.canonical_digest_sha256 == canonical_digest_sha256(
        catalog,
        exclude={"canonical_digest_sha256"},
    )
    assert (
        StaticMechanicsCatalogV1.model_validate_json(catalog.model_dump_json())
        == catalog
    )
    assert canonical_json_bytes({"negative_zero": -0.0}) == (b'{"negative_zero":0.0}')
    with pytest.raises(ValueError):
        canonical_json_bytes({"nonfinite": float("nan")})


def test_catalog_rejects_digest_tampering_and_rehashed_unknown_mapping() -> None:
    catalog = build_static_mechanics_catalog_v1()
    payload = _json_payload(catalog)
    payload["class_mechanics"][1]["maximum_health"] += 1.0
    with pytest.raises(ValidationError, match="canonical digest mismatch"):
        StaticMechanicsCatalogV1.model_validate_json(json.dumps(payload))

    payload = _json_payload(catalog)
    payload["global_recipient_slot_by_actor_and_target_action"][0][1] = 9
    payload["canonical_digest_sha256"] = canonical_digest_sha256(
        payload,
        exclude={"canonical_digest_sha256"},
    )
    with pytest.raises(ValidationError, match="target categories"):
        StaticMechanicsCatalogV1.model_validate_json(json.dumps(payload))

    payload = _json_payload(catalog)
    payload["global_slot_by_actor_and_ally_observation_row"][0][4] = 5
    payload["global_slot_by_actor_and_enemy_observation_row"][0][0] = 4
    payload["global_recipient_slot_by_actor_and_target_action"][0][5] = 5
    payload["global_recipient_slot_by_actor_and_target_action"][0][6] = 4
    payload["canonical_digest_sha256"] = canonical_digest_sha256(
        payload,
        exclude={"canonical_digest_sha256"},
    )
    with pytest.raises(ValidationError, match="serialized team ranges"):
        StaticMechanicsCatalogV1.model_validate_json(json.dumps(payload))

    payload = _json_payload(catalog)
    payload["team_global_slot_half_open_ranges"].reverse()
    payload["canonical_digest_sha256"] = canonical_digest_sha256(
        payload,
        exclude={"canonical_digest_sha256"},
    )
    with pytest.raises(ValidationError, match="ordered V1 team blocks"):
        StaticMechanicsCatalogV1.model_validate_json(json.dumps(payload))


def test_redigested_historical_catalog_validates_without_live_global_equality() -> None:
    payload = _json_payload(build_static_mechanics_catalog_v1())
    payload["team_name_by_id"][1] = "Historical Team A"
    payload["canonical_digest_sha256"] = canonical_digest_sha256(
        payload,
        exclude={"canonical_digest_sha256"},
    )

    historical = StaticMechanicsCatalogV1.model_validate_json(json.dumps(payload))

    assert historical.team_name_by_id[1] == "Historical Team A"
    assert (
        StaticMechanicsCatalogV1.model_validate_json(historical.model_dump_json())
        == historical
    )


def test_root_models_reject_unknown_versions_and_extra_fields() -> None:
    catalog_payload = _json_payload(build_static_mechanics_catalog_v1())
    catalog_payload["schema_version"] = 2
    with pytest.raises(ValidationError):
        StaticMechanicsCatalogV1.model_validate_json(json.dumps(catalog_payload))

    context_payload = _json_payload(evaluation_context())
    context_payload["unknown_field"] = True
    with pytest.raises(ValidationError, match="extra_forbidden"):
        EvaluationEpisodeContextV1.model_validate_json(json.dumps(context_payload))


def test_package_exports_only_the_approved_step_6_3_public_functions() -> None:
    exported_functions = {
        name
        for name in evaluation_api.__all__
        if inspect.isfunction(getattr(evaluation_api, name))
    }
    assert exported_functions == {
        "build_static_mechanics_catalog_v1",
        "build_evaluation_episode_context_v1",
        "build_evaluation_observer_v1",
        "build_replay_artifact_reference_v1",
        "build_replay_artifact_v1",
        "build_replay_bundle_v1",
        "build_scenario_evaluation_record_v1",
        "build_scenario_evaluation_record_v2",
        "canonical_actor_pov_content_json_bytes_v1",
        "canonical_actor_pov_replay_json_bytes_v1",
        "canonical_metric_report_artifact_json_bytes_v1",
        "canonical_replay_json_bytes_v1",
        "canonical_scenario_evaluation_record_json_bytes_v1",
        "canonical_scenario_evaluation_record_json_bytes_v2",
        "capture_initial_evaluation_frame_v1",
        "capture_evaluation_transition_unit_v1",
        "normalize_transition_facts_v1",
        "decode_evaluation_events_v1",
        "export_actor_pov_replay_v1",
        "iter_replay_transition_views_v1",
        "load_actor_pov_replay_artifact_v1",
        "load_replay_artifact_v1",
        "load_replay_bundle_v1",
        "load_scenario_evaluation_record_v1",
        "load_scenario_evaluation_record_v2",
        "reconstruct_actor_class_ids_by_team_v2",
        "reconstruct_class_ids_by_agent_by_team_v2",
        "resolved_initial_state_digest_sha256_v2",
        "save_actor_pov_replay_artifact_v1",
        "save_replay_bundle_v1",
        "save_scenario_evaluation_record_v1",
        "save_scenario_evaluation_record_v2",
        "validate_actor_pov_replay_against_replay_v1",
        "validate_actor_pov_replay_artifact_v1",
        "validate_actor_pov_replay_content_v1",
        "validate_class_ids_by_agent_by_team_against_context_v1",
        "validate_evaluation_processing_progress_v1",
        "validate_evaluation_transition_unit_v1",
        "validate_initial_evaluation_frame_v1",
        "validate_metric_report_artifact_against_replay_v1",
        "validate_replay_artifact_v1",
        "validate_official_scenario_evaluation_record_v2",
        "validate_scenario_evaluation_record_v1",
        "validate_scenario_evaluation_record_v2",
    }


def test_transition_fact_models_mirror_every_core_subtree_and_leaf_name() -> None:
    """Keep the host schema losslessly aligned with the 47-leaf core payload."""
    root_fields = tuple(TransitionFactsV1.model_fields)
    assert root_fields[2:] == CoreTransitionFacts._fields

    field_contracts = (
        (JointActionV1, CoreAction),
        (ActionAcceptanceFactsV1, CoreActionAcceptanceFacts),
        (CombatTransitionFactsV1, CoreCombatTransitionFacts),
        (DeathTransitionFactsV1, CoreDeathTransitionFacts),
        (SpawnShieldTransitionFactsV1, CoreSpawnShieldTransitionFacts),
        (RespawnTransitionFactsV1, CoreRespawnTransitionFacts),
        (RegenerationTransitionFactsV1, CoreRegenerationTransitionFacts),
        (PhysicalTransitionFactsV1, CorePhysicalTransitionFacts),
        (AuraTransitionFactsV1, CoreAuraTransitionFacts),
        (StatusLifecycleTransitionFactsV1, CoreStatusLifecycleTransitionFacts),
        (TeamDeathmatchTransitionFactsV1, CoreTeamDeathmatchTransitionFacts),
    )
    for host_model, core_named_tuple in field_contracts:
        assert tuple(host_model.model_fields) == core_named_tuple._fields


def test_resolved_config_has_own_sensitive_digest_and_roundtrips() -> None:
    config = evaluation_env_config()
    resolved = build_resolved_env_config_v1(config)
    changed = build_resolved_env_config_v1(config._replace(max_steps=101))

    assert resolved.schema_id == RESOLVED_ENV_CONFIG_SCHEMA_ID
    assert resolved.canonical_digest_sha256 == canonical_digest_sha256(
        resolved,
        exclude={"canonical_digest_sha256"},
    )
    assert resolved.canonical_digest_sha256 != canonical_digest_sha256(
        resolved,
        exclude={
            "canonical_digest_sha256",
            "schema_id",
            "schema_version",
        },
    )
    assert changed.canonical_digest_sha256 != resolved.canonical_digest_sha256
    assert (
        ResolvedEnvConfigV1.model_validate_json(resolved.model_dump_json()) == resolved
    )


def test_resolved_config_enforces_exact_task_mode_and_threshold_contract() -> None:
    neutral = build_resolved_env_config_v1(evaluation_env_config())
    team_deathmatch = build_resolved_env_config_v1(
        evaluation_env_config(
            task_mode=1,
            team_deathmatch_score_threshold=25,
        )
    )

    assert (neutral.task_mode, neutral.team_deathmatch_score_threshold) == (0, 0)
    assert (
        team_deathmatch.task_mode,
        team_deathmatch.team_deathmatch_score_threshold,
    ) == (1, 25)
    assert team_deathmatch.canonical_digest_sha256 != neutral.canonical_digest_sha256

    invalid_contracts = (
        {"task_mode": 0, "team_deathmatch_score_threshold": 1},
        {"task_mode": 1, "team_deathmatch_score_threshold": 0},
        {"task_mode": 1, "team_deathmatch_score_threshold": 2**24 - 3},
        {"task_mode": 2, "team_deathmatch_score_threshold": 0},
        {"task_mode": 3, "team_deathmatch_score_threshold": 0},
    )
    for changes in invalid_contracts:
        payload = neutral.model_dump(mode="python")
        payload.update(changes)
        payload["canonical_digest_sha256"] = canonical_digest_sha256(
            payload,
            exclude={"canonical_digest_sha256"},
        )
        with pytest.raises(ValidationError):
            ResolvedEnvConfigV1.model_validate(payload)


def test_team_deathmatch_context_accepts_a_remaining_artifact_horizon() -> None:
    config = evaluation_env_config(
        task_mode=1,
        team_deathmatch_score_threshold=5,
    )

    assert evaluation_context(config=config).expected_horizon == config.max_steps
    assert (
        evaluation_context(
            config=config,
            expected_horizon=config.max_steps - 1,
        ).expected_horizon
        == config.max_steps - 1
    )
    with pytest.raises(ValidationError, match="maximum_episode_steps"):
        evaluation_context(config=config, expected_horizon=config.max_steps + 1)


def test_team_deathmatch_context_requires_an_active_member_on_each_team() -> None:
    neutral_context = evaluation_context(
        config=evaluation_env_config(team_sizes=(3, 0))
    )
    payload = neutral_context.model_dump(mode="python")
    resolved_config = payload["resolved_env_config"]
    resolved_config["task_mode"] = 1
    resolved_config["team_deathmatch_score_threshold"] = 5
    resolved_config["canonical_digest_sha256"] = canonical_digest_sha256(
        resolved_config,
        exclude={"canonical_digest_sha256"},
    )

    with pytest.raises(ValidationError, match="active member on each team"):
        EvaluationEpisodeContextV1.model_validate(payload)


@pytest.mark.parametrize(
    "execution_information_mode",
    ("shared_obs", "no_shared_obs"),
)
def test_context_supports_both_execution_information_modes(
    execution_information_mode: ExecutionInformationMode,
) -> None:
    context = evaluation_context(
        execution_information_mode=execution_information_mode,
    )

    assert context.schema_id == CONTEXT_SCHEMA_ID
    assert context.expected_horizon == 100
    assert context.execution_information_mode == execution_information_mode
    assert len(context.roster) == 10
    assert len(context.policy_assignments) == 10
    assert len(context.resolved_env_config.slot_mechanics) == 10
    assert tuple(row.global_slot for row in context.roster) == tuple(range(10))
    assert (
        tuple((row.schema_id, row.schema_version) for row in context.schema_versions)
        == REQUIRED_SCHEMA_BINDINGS_V1
    )
    assert (
        EvaluationEpisodeContextV1.model_validate_json(context.model_dump_json())
        == context
    )
    assert "shared_observation" not in context.__class__.model_fields
    assert "availability" not in context.__class__.model_fields
    assert not any(
        isinstance(value, (Array, np.ndarray, np.generic))
        for value in _nested_values(context.model_dump(mode="python"))
    )


def test_policy_rows_are_discriminated_and_inactive_rows_are_minimal() -> None:
    context = evaluation_context()

    for roster_row, policy_row in zip(
        context.roster,
        context.policy_assignments,
        strict=True,
    ):
        if roster_row.configured_active:
            assert isinstance(policy_row, AssignedPolicySlotV1)
            assert policy_row.assignment_status == "assigned"
        else:
            assert isinstance(policy_row, NotApplicablePolicySlotV1)
            assert policy_row.model_dump() == {
                "assignment_status": "not_applicable",
                "global_slot": roster_row.global_slot,
            }

    policy_payload = [row.model_dump(mode="json") for row in context.policy_assignments]
    assert all("path" not in key.casefold() for key in _nested_keys(policy_payload))


def test_context_digest_is_independent_of_public_agent_identity() -> None:
    first = evaluation_context(public_agent_id_prefix="first-agent")
    second = evaluation_context(public_agent_id_prefix="second-agent")

    assert first.roster != second.roster
    assert (
        first.resolved_env_config.canonical_digest_sha256
        == second.resolved_env_config.canonical_digest_sha256
    )
    assert first.static_mechanics_catalog == second.static_mechanics_catalog


def test_context_rejects_policy_roster_and_seed_disagreement() -> None:
    context = evaluation_context()

    policy_payload = _json_payload(context)
    policy_payload["policy_assignments"][0] = {
        "assignment_status": "not_applicable",
        "global_slot": 0,
    }
    with pytest.raises(ValidationError, match="active roster rows"):
        EvaluationEpisodeContextV1.model_validate_json(json.dumps(policy_payload))

    roster_payload = _json_payload(context)
    roster_payload["resolved_env_config"]["slot_mechanics"][0]["maximum_health"] += 1.0
    roster_payload["resolved_env_config"]["canonical_digest_sha256"] = (
        canonical_digest_sha256(
            roster_payload["resolved_env_config"],
            exclude={"canonical_digest_sha256"},
        )
    )
    with pytest.raises(ValidationError, match="mechanics catalog"):
        EvaluationEpisodeContextV1.model_validate_json(json.dumps(roster_payload))

    seed_payload = _json_payload(context)
    seed_payload["seed_protocol"]["cooperative_partner_seed"] = "not_applicable"
    with pytest.raises(ValidationError, match="cooperative_partner seed"):
        EvaluationEpisodeContextV1.model_validate_json(json.dumps(seed_payload))


def test_scenario_identity_seed_and_capture_profile_remain_coherent() -> None:
    context = evaluation_context(with_scenario=True)

    assert context.identity.scenario is not None
    assert context.seed_protocol.scenario_seed != "not_applicable"
    assert context.capture_profile == "scenario_metric_complete"

    payload = _json_payload(context)
    payload["identity"]["scenario"] = None
    with pytest.raises(ValidationError, match="scenario"):
        EvaluationEpisodeContextV1.model_validate_json(json.dumps(payload))

    assert evaluation_context().seed_protocol.scenario_seed == "not_applicable"


def test_revision_provenance_enforces_source_and_dirty_patch_digests() -> None:
    digest = "1" * 64

    assert build_code_revision_v1(
        package_version="0.0.0",
        commit_sha="a" * 40,
        is_dirty=True,
        source_tree_digest=digest,
        dirty_patch_digest=digest,
    ).is_dirty
    with pytest.raises(ValidationError, match="require a patch digest"):
        CodeRevisionV1(
            package_version="0.0.0",
            commit_sha="a" * 40,
            is_dirty=True,
            source_tree_digest=digest,
            dirty_patch_digest=None,
        )
    with pytest.raises(ValidationError, match="clean revisions"):
        CodeRevisionV1(
            package_version="0.0.0",
            commit_sha="a" * 40,
            is_dirty=False,
            source_tree_digest=digest,
            dirty_patch_digest=digest,
        )


def test_models_are_strict_finite_and_frozen() -> None:
    with pytest.raises(ValidationError):
        VersionedIdentityV1.model_validate({"identifier": "identity", "version": "1"})
    with pytest.raises(ValidationError):
        VersionedIdentityV1(identifier="non-ascii-λ", version=1)

    resolved = build_resolved_env_config_v1(evaluation_env_config())
    payload = resolved.model_dump(mode="python")
    payload["map_width"] = float("inf")
    with pytest.raises(ValidationError):
        ResolvedEnvConfigV1.model_validate(payload)

    context = evaluation_context()
    with pytest.raises(ValidationError, match="frozen_instance"):
        context.execution_information_mode = "shared_obs"  # type: ignore[misc]


def test_strict_tuple_integer_and_fixed_axis_contracts_reject_coercion() -> None:
    resolved = build_resolved_env_config_v1(evaluation_env_config())

    list_payload = resolved.model_dump(mode="python")
    list_payload["team_respawn_wave_period_steps"] = list(
        resolved.team_respawn_wave_period_steps
    )
    list_payload["canonical_digest_sha256"] = canonical_digest_sha256(
        list_payload,
        exclude={"canonical_digest_sha256"},
    )
    with pytest.raises(ValidationError):
        ResolvedEnvConfigV1.model_validate(list_payload)

    bool_payload = resolved.model_dump(mode="python")
    bool_payload["maximum_episode_steps"] = True
    bool_payload["canonical_digest_sha256"] = canonical_digest_sha256(
        bool_payload,
        exclude={"canonical_digest_sha256"},
    )
    with pytest.raises(ValidationError):
        ResolvedEnvConfigV1.model_validate(bool_payload)

    length_payload = resolved.model_dump(mode="python")
    length_payload["slot_mechanics"] = resolved.slot_mechanics[:-1]
    length_payload["canonical_digest_sha256"] = canonical_digest_sha256(
        length_payload,
        exclude={"canonical_digest_sha256"},
    )
    with pytest.raises(ValidationError):
        ResolvedEnvConfigV1.model_validate(length_payload)


def test_snapshot_observation_mask_and_frame_roundtrip_without_array_objects() -> None:
    frame = _valid_frame()

    assert EvaluationFrameV1.model_validate_json(frame.model_dump_json()) == frame
    assert not any(
        isinstance(value, (Array, np.ndarray, np.generic))
        for value in _nested_values(frame.model_dump(mode="python"))
    )
    assert set(frame.snapshot.__class__.model_fields) == {
        "schema_id",
        "schema_version",
        "team_deathmatch_scores",
        "alive_mask",
        "agent_positions",
        "current_health",
        "ultimate_cooldowns",
        "slow_durations",
        "stun_durations",
        "rogue_poison_anti_heal_durations",
        "mage_burst_damage_amplification_durations",
        "priest_blessing_of_freedom_slow_floor_durations",
        "team_respawn_wave_countdowns",
        "spawn_shield_durations",
        "steps_until_out_of_combat",
        "previous_timestep_move_actions",
        "previous_timestep_select_target_actions",
        "previous_timestep_use_ultimate_actions",
        "has_previous_timestep_joint_action",
    }


def test_dynamic_models_reject_python_lists_wrong_shapes_and_mask_drift() -> None:
    snapshot_payload = _valid_snapshot().model_dump(mode="python")
    snapshot_payload["alive_mask"] = [False] * 10
    with pytest.raises(ValidationError):
        GlobalAnalysisSnapshotV1.model_validate(snapshot_payload)

    snapshot_payload = _valid_snapshot().model_dump(mode="python")
    snapshot_payload["slow_durations"] = cast(
        tuple[tuple[int, ...], ...], _filled_tuple((10, 2), 0)
    )
    with pytest.raises(ValidationError, match="slow_durations"):
        GlobalAnalysisSnapshotV1.model_validate(snapshot_payload)

    snapshot_payload = _valid_snapshot().model_dump(mode="python")
    snapshot_payload["team_deathmatch_scores"] = (0, 0, 0)
    with pytest.raises(ValidationError, match="team_deathmatch_scores"):
        GlobalAnalysisSnapshotV1.model_validate(snapshot_payload)

    snapshot_payload = _valid_snapshot().model_dump(mode="python")
    snapshot_payload["team_deathmatch_scores"] = (0, -1)
    with pytest.raises(ValidationError, match="team_deathmatch_scores"):
        GlobalAnalysisSnapshotV1.model_validate(snapshot_payload)

    mask_payload = _valid_action_mask().model_dump(mode="python")
    mask_payload["select_target_mask"] = cast(
        tuple[tuple[bool, ...], ...], _filled_tuple((10, 11), False)
    )
    with pytest.raises(ValidationError, match="joint-mask marginal"):
        ActionMaskV1.model_validate(mask_payload)


def test_facts_preserve_int32_submissions_and_validate_acceptance_and_recipient() -> (
    None
):
    facts = _valid_transition_facts()
    payload = facts.model_dump(mode="python")
    submitted_move = list(
        payload["action_acceptance_facts"]["submitted_joint_action"]["move"]
    )
    submitted_move[0] = -(2**31)
    payload["action_acceptance_facts"]["submitted_joint_action"]["move"] = tuple(
        submitted_move
    )
    assert TransitionFactsV1.model_validate(
        payload
    ).action_acceptance_facts.submitted_joint_action.move[0] == -(2**31)

    accepted_payload = facts.model_dump(mode="python")
    accepted_move = list(
        accepted_payload["action_acceptance_facts"]["accepted_joint_action"]["move"]
    )
    accepted_move[0] = 9
    accepted_payload["action_acceptance_facts"]["accepted_joint_action"]["move"] = (
        tuple(accepted_move)
    )
    with pytest.raises(ValidationError, match="category-bounded"):
        TransitionFactsV1.model_validate(accepted_payload)

    recipient_payload = facts.model_dump(mode="python")
    recipient_payload["combat_transition_facts"][
        "combat_effect_recipient_global_slot_by_source"
    ] = (0, *([None] * 9))
    with pytest.raises(ValidationError, match="has-recipient"):
        TransitionFactsV1.model_validate(recipient_payload)


def test_initialization_facts_roundtrip_but_cannot_construct_transition() -> None:
    initialization_facts = _valid_transition_facts(has_transition=False)
    assert initialization_facts.transition_start_step_count == -1
    assert (
        TransitionFactsV1.model_validate_json(initialization_facts.model_dump_json())
        == initialization_facts
    )

    with pytest.raises(ValidationError, match="has_transition"):
        EvaluationTransitionV1(
            episode_id="episode-001",
            transition_index=0,
            transition_id="episode-001:transition:0",
            start_frame_id="episode-001:frame:0",
            successor_frame_id="episode-001:frame:1",
            facts=initialization_facts,
            events=(),
            canonical_reward_by_agent=cast(
                tuple[float, ...], _filled_tuple((10,), 0.0)
            ),
            canonical_reward_by_team=None,
            terminated=False,
            truncated=False,
            owning_task_end_reason=None,
        )


@pytest.mark.parametrize(
    ("has_transition", "transition_start_step_count"),
    ((True, -1), (False, 0), (False, 2**31 - 1)),
)
def test_transition_facts_reject_step_sentinel_disagreement(
    has_transition: bool,
    transition_start_step_count: int,
) -> None:
    payload = _valid_transition_facts().model_dump(mode="python")
    payload["has_transition"] = has_transition
    payload["transition_start_step_count"] = transition_start_step_count

    with pytest.raises(ValidationError, match="-1 exactly"):
        TransitionFactsV1.model_validate(payload)


def test_transition_start_step_count_is_strict_and_int32_bounded() -> None:
    payload = _valid_transition_facts().model_dump(mode="python")
    payload["transition_start_step_count"] = True
    with pytest.raises(ValidationError):
        TransitionFactsV1.model_validate(payload)

    payload["transition_start_step_count"] = 2**31
    with pytest.raises(ValidationError):
        TransitionFactsV1.model_validate(payload)

    initialization_payload = _valid_transition_facts(has_transition=False).model_dump(
        mode="python"
    )
    initialization_payload["transition_start_step_count"] = -2
    with pytest.raises(ValidationError):
        TransitionFactsV1.model_validate(initialization_payload)


def test_event_and_transition_structural_identity_roundtrip() -> None:
    event = ActionRejectedEventV1(
        transition_id="episode-001:transition:0",
        ordinal=0,
        event_id="episode-001:transition:0:event:0000",
        actor_global_slot=0,
        rejection_component="domain",
        submitted_move_action=2**31 - 1,
        submitted_select_target_action=-(2**31),
        submitted_use_ultimate_action=0,
    )
    transition = EvaluationTransitionV1(
        episode_id="episode-001",
        transition_index=0,
        transition_id="episode-001:transition:0",
        start_frame_id="episode-001:frame:0",
        successor_frame_id="episode-001:frame:1",
        facts=_valid_transition_facts(),
        events=(event,),
        canonical_reward_by_agent=cast(tuple[float, ...], _filled_tuple((10,), 0.0)),
        canonical_reward_by_team=(0.0, 0.0),
        terminated=False,
        truncated=False,
        owning_task_end_reason=None,
    )

    assert (
        EvaluationTransitionV1.model_validate_json(transition.model_dump_json())
        == transition
    )
    assert "submitted_joint_action" not in transition.__class__.model_fields
    assert "accepted_joint_action" not in transition.__class__.model_fields


def test_event_and_transition_reject_identity_order_end_reason_and_emitter_drift() -> (
    None
):
    with pytest.raises(ValidationError, match="event_id"):
        ActionRejectedEventV1(
            transition_id="episode-001:transition:0",
            ordinal=0,
            event_id="wrong-event-id",
            actor_global_slot=0,
            rejection_component="movement",
            submitted_move_action=0,
            submitted_select_target_action=0,
            submitted_use_ultimate_action=0,
        )

    with pytest.raises(ValidationError, match="sorted and unique"):
        SourceDamageOutputEventV1(
            transition_id="episode-001:transition:0",
            ordinal=0,
            event_id="episode-001:transition:0:event:0000",
            source_global_slot=0,
            recipient_global_slot=5,
            raw_damage_output=1.0,
            source_modified_damage_output=1.0,
            recipient_damage_modifier=1.0,
            mage_damage_aura_covering_emitter_global_slots=(2, 1),
            warrior_mitigation_aura_covering_emitter_global_slots=(),
        )

    transition_payload = {
        "episode_id": "episode-001",
        "transition_index": 0,
        "transition_id": "episode-001:transition:0",
        "start_frame_id": "episode-001:frame:0",
        "successor_frame_id": "episode-001:frame:1",
        "facts": _valid_transition_facts(),
        "events": (),
        "canonical_reward_by_agent": cast(tuple[float, ...], _filled_tuple((10,), 0.0)),
        "canonical_reward_by_team": None,
        "terminated": False,
        "truncated": False,
        "owning_task_end_reason": "victory",
    }
    with pytest.raises(ValidationError, match="only when done"):
        EvaluationTransitionV1.model_validate(transition_payload)
