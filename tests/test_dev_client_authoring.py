"""Focused contracts for the private DevClient authoring host."""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import numpy as np
import pytest
import scripts.dev.visual_debugger.authoring_store as authoring_store_module
from pydantic import ValidationError
from scripts.dev.promote_dev_asset import main as promote_dev_asset_main
from scripts.dev.visual_debugger.authoring_compiler import (
    DevAuthoringValidationError,
    apply_alive_edit,
    canonicalize_inactive_rows,
    compile_dev_map,
    compile_dev_scenario,
    map_semantic_digest,
    normalize_scenario_content,
    scenario_semantic_digest,
    validate_dev_scenario,
    validate_map_content,
)
from scripts.dev.visual_debugger.authoring_models import (
    MAX_DEV_ASSET_SEQUENCE,
    DevAuthoringProblemV1,
    DevMapContentV1,
    DevMapDraftV1,
    DevPillarV1,
    DevPointV1,
    DevScenarioDraftV1,
    DevScenarioGlobalStateV1,
    DevSourceMapProvenanceV1,
    DevWallV1,
    default_spawn_pads,
    new_map_draft,
    new_scenario_draft,
)
from scripts.dev.visual_debugger.authoring_service import (
    DevAuthoringCommandRequestV1,
    DevCandidateSourceV1,
    DevClientAuthoringBinding,
    DevCurrentBufferSourceV1,
    DevSavedDraftSourceV1,
    DevScenarioLoadService,
    LoadedDevScenarioSnapshotV1,
    debugger_scenario_from_snapshot,
)
from scripts.dev.visual_debugger.authoring_store import (
    DevAssetAlreadyExistsError,
    DevAssetIntegrityError,
    DevAssetStore,
    DevDraftRevisionConflictError,
)
from scripts.dev.visual_debugger.control import create_session, reset_session
from scripts.dev.visual_debugger.protocol import (
    CommandRequestV1,
    ResetCommandV1,
    SetCombatConfigurationCommandV1,
)
from scripts.dev.visual_debugger.scenarios import get_scenario
from scripts.dev.visual_debugger.service import DebuggerService
from tests.visual_debugger_fixtures import debugger_test_launch_specification

from marl_battlegrounds.core.types import MAX_OBSTACLE_SLOTS
from marl_battlegrounds.evaluation.models import CodeRevisionV1


def _store(tmp_path: Path) -> DevAssetStore:
    return DevAssetStore(
        Path.cwd(),
        artifact_root=tmp_path / "artifacts" / "dev_client",
        configs_root=tmp_path / "configs",
    )


def _code_revision() -> CodeRevisionV1:
    return CodeRevisionV1(
        package_version="0.0.0",
        commit_sha="a" * 40,
        source_tree_digest="b" * 64,
        is_dirty=False,
    )


def _request(payload: dict[str, object]) -> DevAuthoringCommandRequestV1:
    return DevAuthoringCommandRequestV1.model_validate_json(json.dumps(payload))


def test_strict_models_reject_extra_fields_and_preserve_wire_schema_alias() -> None:
    draft = new_map_draft()
    payload = draft.model_dump(mode="json", by_alias=True)

    assert payload["schema"] == "dev-map-draft@1"
    assert "schema_id" not in payload
    payload["unexpected"] = True

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        DevMapDraftV1.model_validate_json(json.dumps(payload))

    bounded_id_payload = draft.model_dump(mode="json", by_alias=True)
    bounded_id_payload["content"]["spawn_pads"][0]["object_id"] = "x" * 65
    with pytest.raises(ValidationError, match="at most 64 characters"):
        DevMapDraftV1.model_validate_json(json.dumps(bounded_id_payload))

    scenario_payload = new_scenario_draft().model_dump(mode="json", by_alias=True)
    scenario_payload["content"]["embedded_map"]["obstacles"] = [
        {
            "kind": "pillar",
            "object_id": "agent-a1",
            "center_x": 10.0,
            "center_y": 5.0,
            "radius": 0.5,
        }
    ]
    with pytest.raises(ValidationError, match="unique across map and agent objects"):
        DevScenarioDraftV1.model_validate_json(json.dumps(scenario_payload))

    obsolete_role_payload = new_scenario_draft().model_dump(mode="json", by_alias=True)
    obsolete_role_payload["content"]["roster"][0]["role"] = "focal"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        DevScenarioDraftV1.model_validate_json(json.dumps(obsolete_role_payload))

    obsolete_study_payload = new_scenario_draft().model_dump(mode="json", by_alias=True)
    obsolete_study_payload["content"]["study"] = {}
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        DevScenarioDraftV1.model_validate_json(json.dumps(obsolete_study_payload))

    oversized_revision = draft.model_dump(mode="json", by_alias=True)
    oversized_revision["revision"] = MAX_DEV_ASSET_SEQUENCE + 1
    with pytest.raises(ValidationError, match="less than or equal"):
        DevMapDraftV1.model_validate_json(json.dumps(oversized_revision))


def test_blank_defaults_are_exact_and_map_copy_is_independent() -> None:
    map_draft = new_map_draft("source-map")
    scenario = new_scenario_draft("copied-scenario", source_map=map_draft)

    assert map_draft.content.width == 20.0
    assert map_draft.content.height == 10.0
    assert map_draft.content.obstacles == ()
    assert tuple(
        f"{pad.team}{pad.team_local_slot}" for pad in map_draft.content.spawn_pads
    ) == tuple(f"{team}{slot}" for team in ("A", "B") for slot in range(1, 6))
    assert scenario.content.team_a_size == scenario.content.team_b_size == 5
    assert scenario.content.task.score_threshold == 5
    assert scenario.content.episode.max_steps == 300
    assert scenario.content.episode.spawn_shield_duration_steps == 3
    assert scenario.content.episode.spawn_shield_movement_speed == 2.0
    assert scenario.content.episode.team_a_respawn_wave_period_steps == 5
    assert scenario.content.episode.team_b_respawn_wave_period_steps == 5
    assert tuple(slot.class_name for slot in scenario.content.roster[:5]) == (
        "mage",
        "warrior",
        "hunter",
        "rogue",
        "priest",
    )
    assert scenario.content.global_state == DevScenarioGlobalStateV1()
    assert scenario.content.embedded_map == map_draft.content
    assert scenario.content.embedded_map is not map_draft.content

    renamed_source = map_draft.model_copy(
        update={"content": map_draft.content.model_copy(update={"name": "Later edit"})}
    )
    assert renamed_source.content.name == "Later edit"
    assert scenario.content.embedded_map.name == "Untitled map"

    colliding_map = map_draft.model_copy(
        update={
            "content": map_draft.content.model_copy(
                update={
                    "obstacles": (
                        DevPillarV1(
                            object_id="agent-a1",
                            center_x=10.0,
                            center_y=5.0,
                            radius=0.5,
                        ),
                    )
                }
            )
        }
    )
    remapped = new_scenario_draft("collision-safe", source_map=colliding_map)
    assert remapped.content.embedded_map == colliding_map.content
    assert remapped.content.roster[0].object_id == "agent-a1-2"
    assert (
        len(
            {
                *(
                    obstacle.object_id
                    for obstacle in remapped.content.embedded_map.obstacles
                ),
                *(pad.object_id for pad in remapped.content.embedded_map.spawn_pads),
                *(slot.object_id for slot in remapped.content.roster),
            }
        )
        == 21
    )


def test_map_normalization_padding_order_and_semantic_digest_contract() -> None:
    map_a = DevMapContentV1(
        name="Display name A",
        description="first description",
        width=20.0,
        height=10.0,
        obstacles=(
            DevWallV1(
                object_id="wall-browser-a",
                center_x=8.0,
                center_y=5.0,
                width=2.0,
                height=0.5,
                rotation_degrees=450.0,
            ),
            DevPillarV1(
                object_id="pillar-browser-a",
                center_x=12.0,
                center_y=5.0,
                radius=0.75,
            ),
        ),
        spawn_pads=default_spawn_pads(),
    )
    compiled = compile_dev_map(map_a)
    host_obstacles = np.asarray(compiled.obstacles)

    assert host_obstacles.shape[1] == 8
    assert host_obstacles.shape[0] >= 2
    assert np.count_nonzero(host_obstacles[2:]) == 0
    assert host_obstacles[0, 7] == 1.0
    assert host_obstacles[1, 7] == 1.0
    normalized_wall = compiled.content.obstacles[0]
    assert isinstance(normalized_wall, DevWallV1)
    assert normalized_wall.rotation_degrees == pytest.approx(90.0)

    replacement_pads = tuple(
        pad.model_copy(update={"object_id": f"alternate-{index}"})
        for index, pad in enumerate(map_a.spawn_pads)
    )
    map_b = map_a.model_copy(
        update={
            "name": "Display name B",
            "description": "second description",
            "obstacles": tuple(
                obstacle.model_copy(update={"object_id": f"alternate-obstacle-{index}"})
                for index, obstacle in enumerate(map_a.obstacles)
            ),
            "spawn_pads": replacement_pads,
        }
    )
    assert map_semantic_digest(map_a) == map_semantic_digest(map_b)
    assert map_semantic_digest(map_a) != map_semantic_digest(
        map_a.model_copy(update={"obstacles": tuple(reversed(map_a.obstacles))})
    )

    half_turn = map_a.model_copy(
        update={
            "obstacles": (
                map_a.obstacles[0].model_copy(update={"rotation_degrees": 180.0}),
                map_a.obstacles[1],
            )
        }
    )
    equivalent_half_turn = half_turn.model_copy(
        update={
            "obstacles": (
                half_turn.obstacles[0].model_copy(update={"rotation_degrees": -180.0}),
                half_turn.obstacles[1],
            )
        }
    )
    assert map_semantic_digest(half_turn) == map_semantic_digest(equivalent_half_turn)


def test_map_validation_links_pad_errors_and_keeps_permitted_geometry_as_warning() -> (
    None
):
    draft = new_map_draft()
    outside = DevPillarV1(
        object_id="outside-pillar",
        center_x=30.0,
        center_y=5.0,
        radius=1.0,
    )
    warning_map = draft.content.model_copy(update={"obstacles": (outside,)})
    problems = validate_map_content(warning_map)

    assert not any(problem.severity == "error" for problem in problems)
    assert {problem.stable_code for problem in problems} == {
        "map-obstacle-outside-bounds"
    }

    too_small = draft.content.model_copy(update={"width": 2.0, "height": 2.0})
    pad_problems = validate_map_content(too_small)
    assert any(
        problem.stable_code == "map-spawn-pad-out-of-bounds"
        and problem.object_id == "pad-a2"
        for problem in pad_problems
    )


def test_scenario_compiler_uses_reset_overlay_and_neutral_history() -> None:
    draft = new_scenario_draft()
    states = list(draft.content.agent_states)
    states[0] = states[0].model_copy(
        update={
            "current_health": 70.0,
            "ultimate_cooldown_remaining": 5,
            "mage_burst_duration": 2,
        }
    )
    states[1] = states[1].model_copy(update={"warrior_charge_slow_duration": 1})
    states[4] = states[4].model_copy(update={"priest_blessing_of_freedom_duration": 1})
    content = draft.content.model_copy(
        update={
            "global_state": DevScenarioGlobalStateV1(
                step_count=10,
                team_a_score=1,
                team_b_score=2,
                team_a_respawn_countdown=3,
                team_b_respawn_countdown=2,
            ),
            "agent_states": tuple(states),
        }
    )
    content = apply_alive_edit(content, global_slot=5, alive=False)
    compiled = compile_dev_scenario(content)
    state = compiled.initial_state

    assert compiled.config.task_mode == 1
    assert compiled.config.team_deathmatch_score_threshold == 5
    assert np.asarray(state.step_count).item() == 10
    assert np.asarray(state.team_deathmatch_scores).tolist() == [1, 2]
    assert np.asarray(state.current_health)[0] == 70.0
    assert not np.asarray(state.alive_mask)[5]
    assert np.asarray(state.current_health)[5] == 0.0
    assert np.asarray(state.ultimate_cooldowns)[5] == 0
    assert not np.asarray(state.has_previous_timestep_joint_action).item()
    assert np.count_nonzero(np.asarray(state.previous_timestep_move_actions)) == 0
    assert compiled.resolved_configuration_digest
    assert compiled.resolved_initial_state_digest


def test_every_authorable_timer_family_reaches_the_exact_state_leaf() -> None:
    draft = new_scenario_draft()
    states = list(draft.content.agent_states)
    states[0] = states[0].model_copy(
        update={
            "ultimate_cooldown_remaining": 30,
            "warrior_charge_slow_duration": 5,
            "warrior_charge_stun_duration": 1,
            "mage_burst_duration": 5,
            "steps_until_out_of_combat": 1,
        }
    )
    states[1] = states[1].model_copy(
        update={
            "hunter_basic_slow_duration": 1,
            "hunter_trap_stun_duration": 4,
        }
    )
    states[2] = states[2].model_copy(
        update={
            "rogue_poison_slow_duration": 5,
            "rogue_poison_stun_duration": 1,
            "rogue_poison_anti_heal_duration": 4,
        }
    )
    states[3] = states[3].model_copy(update={"spawn_shield_duration_remaining": 3})
    states[4] = states[4].model_copy(update={"priest_blessing_of_freedom_duration": 1})
    compiled = compile_dev_scenario(
        draft.content.model_copy(update={"agent_states": tuple(states)})
    )
    state = compiled.initial_state

    assert np.asarray(state.ultimate_cooldowns)[0] == 30
    assert np.asarray(state.slow_durations)[:3].tolist() == [
        [5, 0, 0],
        [0, 1, 0],
        [0, 0, 5],
    ]
    assert np.asarray(state.stun_durations)[:3].tolist() == [
        [1, 0, 0],
        [0, 4, 0],
        [0, 0, 1],
    ]
    assert np.asarray(state.rogue_poison_anti_heal_durations)[2] == 4
    assert np.asarray(state.mage_burst_damage_amplification_durations)[0] == 5
    assert np.asarray(state.priest_blessing_of_freedom_slow_floor_durations)[4] == 1
    assert np.asarray(state.spawn_shield_durations)[3] == 3
    assert np.asarray(state.steps_until_out_of_combat)[0] == 1


def test_asymmetric_rosters_compile_with_canonical_inactive_padding() -> None:
    draft = new_scenario_draft()
    content = draft.content.model_copy(update={"team_a_size": 2, "team_b_size": 1})
    content = canonicalize_inactive_rows(content)
    compiled = compile_dev_scenario(content)

    assert np.asarray(compiled.config.agent_profile.active_mask).tolist() == [
        True,
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
    ]
    assert tuple(slot.class_name for slot in content.roster[2:5]) == (
        "not_applicable",
        "not_applicable",
        "not_applicable",
    )
    assert (
        np.count_nonzero(np.asarray(compiled.initial_state.agent_positions)[2:5]) == 0
    )
    assert np.count_nonzero(np.asarray(compiled.initial_state.agent_positions)[6:]) == 0


def test_invalid_edits_remain_representable_and_return_linked_problems() -> None:
    draft = new_scenario_draft()
    invalid_global = draft.content.global_state.model_copy(
        update={"step_count": 300, "team_a_score": 5}
    )
    invalid = draft.model_copy(
        update={
            "content": draft.content.model_copy(update={"global_state": invalid_global})
        }
    )
    problems = validate_dev_scenario(invalid)

    assert {(problem.stable_code, problem.field_path) for problem in problems} >= {
        ("scenario-step-count-out-of-range", "global_state.step_count"),
        ("scenario-score-out-of-range", "global_state.team_a_score"),
    }
    with pytest.raises(DevAuthoringValidationError):
        compile_dev_scenario(invalid)


def test_invalid_numeric_draft_saves_reopens_and_stays_out_of_debug_discovery(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    binding = DevClientAuthoringBinding(store, code_revision=_code_revision())
    payload = new_scenario_draft("invalid-numeric").model_dump(
        mode="json", by_alias=True
    )
    payload["content"]["episode"]["max_steps"] = 0
    payload["content"]["agent_states"][0]["current_health"] = -1.0

    saved = binding.apply_command(
        _request(
            {
                "command_type": "save",
                "draft": payload,
                "expected_revision": 0,
            }
        )
    )

    assert saved.ok
    assert saved.validation is not None
    assert not saved.validation.execution_valid
    assert {
        (problem.stable_code, problem.object_id, problem.field_path)
        for problem in saved.validation.problems
    } >= {
        ("scenario-max-steps-not-positive", None, "episode.max_steps"),
        (
            "scenario-agent-health-negative",
            "agent-a1",
            "agent_states.0.current_health",
        ),
    }
    reopened = binding.apply_command(
        _request(
            {
                "command_type": "open",
                "source": {
                    "source_kind": "saved_draft",
                    "asset_kind": "scenario",
                    "asset_id": "invalid-numeric",
                    "revision": 1,
                },
            }
        )
    )
    assert reopened.ok
    assert isinstance(reopened.draft, DevScenarioDraftV1)
    assert reopened.draft.content.episode.max_steps == 0
    listed = binding.apply_command(
        _request({"command_type": "list", "asset_kind": "scenario"})
    )
    assert [(asset.asset_id, asset.execution_valid) for asset in listed.assets] == [
        ("invalid-numeric", False)
    ]
    assert binding.scenario_loader.discover() == ()


def test_core_state_failure_links_the_exact_agent_inspector_field() -> None:
    draft = new_scenario_draft("linked-core-state")
    states = list(draft.content.agent_states)
    states[0] = states[0].model_copy(update={"current_health": 81.0})
    invalid = draft.model_copy(
        update={
            "content": draft.content.model_copy(update={"agent_states": tuple(states)})
        }
    )

    problems = validate_dev_scenario(invalid)

    assert any(
        problem.stable_code == "scenario-core-state-invalid"
        and problem.object_id == "agent-a1"
        and problem.field_path == "agent_states.0.current_health"
        for problem in problems
    )


def test_compile_map_wraps_float32_normalization_as_a_linked_problem() -> None:
    draft = new_map_draft("overflow-map")
    invalid = draft.model_copy(
        update={"content": draft.content.model_copy(update={"width": 1e100})}
    )

    with pytest.raises(DevAuthoringValidationError) as raised:
        compile_dev_map(invalid)

    assert [problem.stable_code for problem in raised.value.problems] == [
        "map-float32-normalization-failed"
    ]
    assert raised.value.problems[0].field_path == "width"


def test_execution_valid_scenario_is_freeze_qualified() -> None:
    draft = new_scenario_draft()

    assert validate_dev_scenario(draft) == ()
    assert compile_dev_scenario(draft).freeze_qualified


def test_scenario_digest_excludes_display_prose_provenance_and_browser_ids() -> None:
    draft = new_scenario_draft()
    roster = tuple(
        slot.model_copy(update={"object_id": f"replacement-agent-{index}"})
        for index, slot in enumerate(draft.content.roster)
    )
    states = tuple(
        state.model_copy(update={"object_id": roster[index].object_id})
        for index, state in enumerate(draft.content.agent_states)
    )
    changed = draft.content.model_copy(
        update={
            "name": "Renamed scenario",
            "description": "Display-only prose",
            "notes": "Private author notes",
            "source_map_provenance": DevSourceMapProvenanceV1(
                asset_id="some-map",
                revision=7,
                semantic_digest="a" * 64,
            ),
            "roster": roster,
            "agent_states": states,
        }
    )

    assert scenario_semantic_digest(draft.content) == scenario_semantic_digest(changed)


def test_scenario_notes_are_optional_and_bounded() -> None:
    payload = new_scenario_draft().model_dump(mode="json", by_alias=True)

    assert payload["content"]["notes"] == ""
    payload["content"]["notes"] = "x" * 8_000
    parsed = DevScenarioDraftV1.model_validate_json(json.dumps(payload))
    assert len(parsed.content.notes) == 8_000

    payload["content"]["notes"] = "x" * 8_001
    with pytest.raises(ValidationError, match="at most 8000 characters"):
        DevScenarioDraftV1.model_validate_json(json.dumps(payload))


def test_scenario_normalization_matches_float32_runtime_storage() -> None:
    draft = new_scenario_draft()
    first = draft.content.agent_states[0].model_copy(
        update={
            "position": DevPointV1(x=1.50000001, y=1.50000001),
            "current_health": 79.9999999,
        }
    )
    states = (first, *draft.content.agent_states[1:])
    content = draft.content.model_copy(update={"agent_states": states})
    normalized = normalize_scenario_content(content)
    compiled = compile_dev_scenario(content)

    assert normalized.agent_states[0].position.x == float(np.float32(1.50000001))
    assert normalized.agent_states[0].current_health == float(np.float32(79.9999999))
    assert compiled.content == normalized


def test_store_saves_exact_revisions_and_rejects_stale_or_unsafe_identity(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    revision_one = store.save_draft(
        new_map_draft("revisioned-map"), expected_revision=0
    )
    revision_two = store.save_draft(
        revision_one.model_copy(
            update={
                "content": revision_one.content.model_copy(update={"name": "Second"})
            }
        ),
        expected_revision=1,
    )

    assert revision_one.revision == 1
    assert revision_two.revision == 2
    assert store.load_draft("map", "revisioned-map", revision=1).content.name == (
        "Untitled map"
    )
    assert store.load_draft("map", "revisioned-map").content.name == "Second"
    with pytest.raises(DevDraftRevisionConflictError, match="stale"):
        store.save_draft(revision_one, expected_revision=1)
    with pytest.raises(ValueError, match="safe lowercase"):
        store.load_draft("map", "../escape")
    with pytest.raises(ValueError, match="positive 32-bit"):
        store.load_draft(
            "map",
            "revisioned-map",
            revision=MAX_DEV_ASSET_SEQUENCE + 1,
        )


def test_store_rejects_symlinks_before_draft_writes_and_reads(tmp_path: Path) -> None:
    store = _store(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    map_collection = store.artifact_root / "drafts" / "maps"
    map_collection.mkdir(parents=True)
    (map_collection / "linked-map").symlink_to(outside, target_is_directory=True)

    with pytest.raises(DevAssetIntegrityError, match="must not contain symlinks"):
        store.save_draft(new_map_draft("linked-map"), expected_revision=0)
    assert tuple(outside.iterdir()) == ()

    saved = store.save_draft(new_map_draft("read-map"), expected_revision=0)
    saved_path = map_collection / "read-map" / "r1.json"
    external_payload = outside / "r1.json"
    external_payload.write_bytes(saved_path.read_bytes())
    saved_path.unlink()
    saved_path.symlink_to(external_payload)

    with pytest.raises(DevAssetIntegrityError, match="must not contain symlinks"):
        store.load_draft("map", saved.asset_id, revision=saved.revision)


def test_candidate_freeze_is_content_addressed_and_tampering_fails_closed(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    draft = new_map_draft("candidate-map")
    candidate = store.freeze_map(draft, code_revision=_code_revision())

    assert store.load_candidate("map", candidate.candidate_id) == candidate
    assert store.freeze_map(draft, code_revision=_code_revision()) == candidate
    renamed = draft.model_copy(
        update={"content": draft.content.model_copy(update={"name": "Other name"})}
    )
    renamed_candidate = store.freeze_map(renamed, code_revision=_code_revision())
    assert renamed_candidate.candidate_id != candidate.candidate_id
    assert (
        renamed_candidate.evidence.semantic_digest == candidate.evidence.semantic_digest
    )

    candidate_path = (
        store.artifact_root / "candidates" / f"map-{candidate.candidate_id}.json"
    )
    payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    payload["content"]["width"] = 21.0
    candidate_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DevAssetIntegrityError, match="content digest mismatch"):
        store.load_candidate("map", candidate.candidate_id)


def test_qualified_candidate_promotion_revalidates_and_never_overwrites(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    candidate = store.freeze_scenario(
        new_scenario_draft("qualified-scenario"),
        code_revision=_code_revision(),
    )
    destination = store.promote_candidate(
        candidate.candidate_id,
        asset_id="tdm-controlled-one",
        version=1,
        approval_provenance="owner-test",
    )

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert destination == (
        tmp_path / "configs" / "scenarios" / "tdm-controlled-one" / "v1.json"
    )
    assert payload["semantic_digest"] == candidate.evidence.semantic_digest
    assert payload["approval_provenance"] == "owner-test"
    assert "partition" not in payload

    loader = DevScenarioLoadService(store)
    attempt = loader.load(
        DevCandidateSourceV1(
            asset_kind="scenario",
            candidate_id=candidate.candidate_id,
        )
    )
    assert attempt.ok
    assert attempt.summary is not None
    assert attempt.summary.candidate_id == candidate.candidate_id
    with pytest.raises(DevAssetAlreadyExistsError):
        store.promote_candidate(
            candidate.candidate_id,
            asset_id="tdm-controlled-one",
            version=1,
            approval_provenance="owner-test",
        )


def test_candidate_revalidation_preserves_linked_problems_and_current_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    candidate = store.freeze_scenario(
        new_scenario_draft("linked-candidate"),
        code_revision=_code_revision(),
    )
    loader = DevScenarioLoadService(store)
    source = DevCandidateSourceV1(
        asset_kind="scenario",
        candidate_id=candidate.candidate_id,
    )
    assert loader.load(source).ok
    original_snapshot = loader.current_snapshot
    linked_problem = DevAuthoringProblemV1(
        severity="error",
        stable_code="scenario-core-state-invalid",
        message="Current health is no longer valid.",
        object_id="agent-a1",
        field_path="agent_states.0.current_health",
    )

    def reject_candidate(*_args: object, **_kwargs: object) -> object:
        raise DevAuthoringValidationError((linked_problem,))

    monkeypatch.setattr(
        authoring_store_module,
        "compile_dev_scenario",
        reject_candidate,
    )
    failed = loader.load(source)

    assert not failed.ok
    assert failed.problems == (linked_problem,)
    assert loader.current_snapshot is original_snapshot


def test_owner_promotion_cli_uses_candidate_asset_and_version_flags(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = DevAssetStore(tmp_path)
    candidate = store.freeze_map(
        new_map_draft("cli-source"),
        code_revision=_code_revision(),
    )

    result = promote_dev_asset_main(
        [
            "--candidate",
            candidate.candidate_id,
            "--asset-id",
            "promoted-map",
            "--version",
            "1",
            "--repository-root",
            str(tmp_path),
        ]
    )

    assert result == 0
    assert (tmp_path / "configs" / "maps" / "promoted-map" / "v1.json").is_file()
    assert "Promoted candidate" in capsys.readouterr().out

    direct_help = subprocess.run(
        [
            sys.executable,
            str(Path.cwd() / "scripts" / "dev" / "promote_dev_asset.py"),
            "--help",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert direct_help.returncode == 0, direct_help.stderr
    assert "--candidate" in direct_help.stdout


def test_saved_scenario_discovery_and_restart_load_exact_revision(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    saved = store.save_draft(new_scenario_draft("later-load"), expected_revision=0)

    first_process = DevScenarioLoadService(store)
    assert first_process.current_snapshot is None
    second_process = DevScenarioLoadService(_store(tmp_path))
    summaries = second_process.discover()
    assert [(summary.asset_id, summary.revision) for summary in summaries] == [
        ("later-load", 1)
    ]

    attempt = second_process.load(
        DevSavedDraftSourceV1(
            asset_kind="scenario",
            asset_id=saved.asset_id,
            revision=saved.revision,
        )
    )
    assert attempt.ok
    assert attempt.summary is not None
    assert attempt.summary.scenario_name == "Untitled TDM scenario"
    assert second_process.current_snapshot is not None
    assert second_process.current_snapshot.compiled.semantic_digest == (
        attempt.summary.scenario_semantic_digest
    )


def test_current_saved_and_candidate_maps_use_the_exact_default_preview_path(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    current_map = new_map_draft("preview-map").model_copy(
        update={
            "content": new_map_draft("preview-map").content.model_copy(
                update={"name": "Preview arena"}
            )
        }
    )
    saved_map = store.save_draft(current_map, expected_revision=0)
    assert isinstance(saved_map, DevMapDraftV1)
    candidate = store.freeze_map(current_map, code_revision=_code_revision())
    binding = DevClientAuthoringBinding(store, code_revision=_code_revision())
    files_before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    sources_and_maps = (
        (
            {
                "source_kind": "current_buffer",
                "asset_kind": "map",
                "draft": current_map.model_dump(mode="json", by_alias=True),
            },
            current_map,
        ),
        (
            {
                "source_kind": "saved_draft",
                "asset_kind": "map",
                "asset_id": saved_map.asset_id,
                "revision": saved_map.revision,
            },
            saved_map,
        ),
        (
            {
                "source_kind": "candidate",
                "asset_kind": "map",
                "candidate_id": candidate.candidate_id,
            },
            candidate,
        ),
    )
    for source, exact_map in sources_and_maps:
        expected = compile_dev_scenario(
            new_scenario_draft("explicit-map-copy", source_map=exact_map)
        )
        response = binding.apply_command(
            _request({"command_type": "open_in_debug", "source": source})
        )

        assert response.ok
        assert response.debug_load is not None
        assert response.debug_load.asset_kind == "map"
        assert response.debug_load.debug_profile == "default_tdm_map_preview"
        assert response.debug_load.source_name == "Preview arena"
        assert response.debug_load.scenario_name == "Default TDM map preview"
        assert response.debug_load.resolved_configuration_digest == (
            expected.resolved_configuration_digest
        )
        assert response.debug_load.resolved_initial_state_digest == (
            expected.resolved_initial_state_digest
        )

    snapshot = binding.scenario_loader.current_snapshot
    assert snapshot is not None
    scenario = debugger_scenario_from_snapshot(snapshot)
    assert scenario.title == "Default TDM map preview"
    assert scenario.default_controlled_slot == 0
    assert scenario.provenance is not None
    assert scenario.provenance.source_identity == (
        f"map:candidate:{candidate.candidate_id}:profile:default-tdm-map-preview@1"
    )
    files_after = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert files_after == files_before

    maximum_name = "M" * 120
    long_name_map = new_map_draft("long-preview-name")
    long_name_map = long_name_map.model_copy(
        update={
            "content": long_name_map.content.model_copy(update={"name": maximum_name})
        }
    )
    long_name_response = binding.apply_command(
        _request(
            {
                "command_type": "open_in_debug",
                "source": {
                    "source_kind": "current_buffer",
                    "asset_kind": "map",
                    "draft": long_name_map.model_dump(mode="json", by_alias=True),
                },
            }
        )
    )
    assert long_name_response.ok
    assert long_name_response.debug_load is not None
    assert long_name_response.debug_load.source_name == maximum_name
    assert long_name_response.debug_load.scenario_name == "Default TDM map preview"


def test_failed_map_preview_preserves_current_debug_snapshot(tmp_path: Path) -> None:
    binding = DevClientAuthoringBinding(
        _store(tmp_path),
        code_revision=_code_revision(),
    )
    valid_scenario = new_scenario_draft("existing-debug-session")
    installed = binding.apply_command(
        _request(
            {
                "command_type": "open_in_debug",
                "source": {
                    "source_kind": "current_buffer",
                    "asset_kind": "scenario",
                    "draft": valid_scenario.model_dump(mode="json", by_alias=True),
                },
            }
        )
    )
    assert installed.ok
    original_snapshot = binding.scenario_loader.current_snapshot

    invalid_map = new_map_draft("invalid-preview")
    invalid_map = invalid_map.model_copy(
        update={
            "content": invalid_map.content.model_copy(
                update={"width": 2.0, "height": 2.0}
            )
        }
    )
    failed = binding.apply_command(
        _request(
            {
                "command_type": "open_in_debug",
                "source": {
                    "source_kind": "current_buffer",
                    "asset_kind": "map",
                    "draft": invalid_map.model_dump(mode="json", by_alias=True),
                },
            }
        )
    )

    assert not failed.ok
    assert any(
        problem.stable_code == "map-spawn-pad-out-of-bounds"
        for problem in failed.problems
    )
    assert binding.scenario_loader.current_snapshot is original_snapshot


def test_failed_revalidation_preserves_current_debug_snapshot(tmp_path: Path) -> None:
    store = _store(tmp_path)
    valid = store.save_draft(new_scenario_draft("load-guard"), expected_revision=0)
    assert isinstance(valid, DevScenarioDraftV1)
    loader = DevScenarioLoadService(store)
    first = loader.load(
        DevSavedDraftSourceV1(
            asset_kind="scenario",
            asset_id=valid.asset_id,
            revision=valid.revision,
        )
    )
    assert first.ok
    original_snapshot = loader.current_snapshot

    invalid_state = valid.content.global_state.model_copy(update={"step_count": 300})
    invalid = valid.model_copy(
        update={
            "content": valid.content.model_copy(update={"global_state": invalid_state})
        }
    )
    invalid_saved = store.save_draft(invalid, expected_revision=1)
    failed = loader.load(
        DevSavedDraftSourceV1(
            asset_kind="scenario",
            asset_id=invalid_saved.asset_id,
            revision=invalid_saved.revision,
        )
    )

    assert not failed.ok
    assert any(
        problem.stable_code == "scenario-step-count-out-of-range"
        for problem in failed.problems
    )
    assert loader.current_snapshot is original_snapshot
    assert loader.discover() == ()


def test_current_buffer_and_saved_loader_use_one_snapshot_path(tmp_path: Path) -> None:
    loader = DevScenarioLoadService(_store(tmp_path))
    draft = new_scenario_draft("current-buffer")
    with pytest.raises(ValidationError, match="asset_kind must match"):
        DevCurrentBufferSourceV1(asset_kind="map", draft=draft)
    attempt = loader.load(DevCurrentBufferSourceV1(asset_kind="scenario", draft=draft))

    assert attempt.ok
    assert attempt.summary is not None
    assert attempt.summary.source_kind == "current_buffer"
    assert loader.current_snapshot is not None
    assert loader.current_snapshot.compiled.content is not draft.content


def test_loaded_snapshot_replaces_and_resets_the_exact_debugger_scenario(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    draft = new_scenario_draft("debugger-load")
    global_state = draft.content.global_state.model_copy(update={"step_count": 7})
    saved = store.save_draft(
        draft.model_copy(
            update={
                "content": draft.content.model_copy(
                    update={"global_state": global_state}
                )
            }
        ),
        expected_revision=0,
    )
    initial_session = create_session(
        get_scenario("arena_5v5"),
        seed=0,
        evaluation_launch_specification=debugger_test_launch_specification(),
        controlled_global_slot=None,
        show_ranges=True,
        verbose_logging=False,
    )
    debugger = DebuggerService(
        initial_session,
        view_mode="researcher",
        preset="analysis",
        include_stress=False,
    )

    def install_snapshot(snapshot: LoadedDevScenarioSnapshotV1) -> None:
        debugger.load_scenario(debugger_scenario_from_snapshot(snapshot))

    loader = DevScenarioLoadService(
        store,
        install_snapshot=install_snapshot,
    )

    attempt = loader.load(
        DevSavedDraftSourceV1(
            asset_kind="scenario",
            asset_id=saved.asset_id,
            revision=saved.revision,
        )
    )

    assert attempt.ok
    assert attempt.summary is not None
    assert debugger.revision == 1
    assert int(debugger.session.state.step_count) == 7
    assert debugger.session.scenario.provenance is not None
    assert debugger.session.scenario.provenance.source_identity == (
        "scenario:saved_draft:debugger-load:revision:1"
    )
    assert debugger.session.scenario.default_controlled_slot == 0
    aggregation = {
        row.name: row.value
        for row in debugger.session.evaluation_context.aggregation_keys
    }
    assert aggregation["scenario_digest"] == attempt.summary.scenario_semantic_digest

    restarted = reset_session(debugger.session)
    assert restarted.scenario is debugger.session.scenario
    assert int(restarted.state.step_count) == 7

    reset_result = debugger.apply_command(
        CommandRequestV1(
            client_id="authored-restart-test",
            command_id="reset",
            base_revision=debugger.revision,
            command=ResetCommandV1(),
        )
    )
    assert reset_result.outcome == "response"
    configured_result = debugger.apply_command(
        CommandRequestV1(
            client_id="authored-restart-test",
            command_id="scripted-shared",
            base_revision=debugger.revision,
            command=SetCombatConfigurationCommandV1(
                team_b_controller="scripted_tdm",
                execution_information_mode="shared_obs",
            ),
        )
    )
    assert configured_result.outcome == "response"
    assert debugger.session.team_b_controller == "scripted_tdm"
    assert int(debugger.session.state.step_count) == 7


def test_single_authoring_binding_parses_whole_commands_and_shares_loader(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    binding = DevClientAuthoringBinding(store, code_revision=_code_revision())
    created = binding.apply_command(
        _request(
            {
                "command_type": "new_scenario",
                "asset_id": "binding-scenario",
            }
        )
    )
    assert created.ok
    assert created.draft is not None
    assert created.catalog.maximum_obstacle_slots == MAX_OBSTACLE_SLOTS
    assert len(created.catalog.class_mechanics) == 5
    serialized_created = json.loads(created.model_dump_json())
    assert serialized_created["draft"]["schema"] == "dev-scenario-draft@1"
    assert "schema_id" not in serialized_created["draft"]

    saved = binding.apply_command(
        _request(
            {
                "command_type": "save",
                "draft": created.draft.model_dump(mode="json", by_alias=True),
                "expected_revision": 0,
            }
        )
    )
    assert saved.ok
    assert saved.draft is not None
    loaded = binding.apply_command(
        _request(
            {
                "command_type": "open_in_debug",
                "source": {
                    "source_kind": "saved_draft",
                    "asset_kind": "scenario",
                    "asset_id": "binding-scenario",
                    "revision": 1,
                },
            }
        )
    )

    assert loaded.ok
    assert loaded.debug_load is not None
    assert binding.scenario_loader.current_snapshot is not None
    listed = binding.apply_command(
        _request({"command_type": "list", "asset_kind": "scenario"})
    )
    assert listed.ok
    assert [(asset.asset_id, asset.revision) for asset in listed.assets] == [
        ("binding-scenario", 1)
    ]


def test_authoring_binding_serializes_threaded_host_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = DevClientAuthoringBinding(
        _store(tmp_path),
        code_revision=_code_revision(),
    )
    first_entered = Event()
    release_first = Event()
    second_entered = Event()
    call_count = 0

    def blocking_list(_requested: str) -> tuple[object, ...]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            first_entered.set()
            assert release_first.wait(timeout=2.0)
        else:
            second_entered.set()
        return ()

    monkeypatch.setattr(binding, "_list_assets", blocking_list)
    request = _request({"command_type": "list", "asset_kind": "all"})
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(binding.apply_command, request)
        assert first_entered.wait(timeout=2.0)
        second = executor.submit(binding.apply_command, request)
        assert not second_entered.wait(timeout=0.1)
        release_first.set()
        assert first.result(timeout=2.0).ok
        assert second.result(timeout=2.0).ok
    assert second_entered.is_set()


def test_validate_returns_linked_capacity_and_float32_problems(tmp_path: Path) -> None:
    binding = DevClientAuthoringBinding(
        _store(tmp_path), code_revision=_code_revision()
    )
    map_draft = new_map_draft("too-many-obstacles")
    obstacles = tuple(
        DevPillarV1(
            object_id=f"pillar-{index}",
            center_x=10.0,
            center_y=5.0,
            radius=0.25,
        )
        for index in range(MAX_OBSTACLE_SLOTS + 1)
    )
    map_response = binding.apply_command(
        _request(
            {
                "command_type": "validate",
                "draft": map_draft.model_copy(
                    update={
                        "content": map_draft.content.model_copy(
                            update={"obstacles": obstacles}
                        )
                    }
                ).model_dump(mode="json", by_alias=True),
            }
        )
    )

    scenario_draft = new_scenario_draft("float32-overflow")
    first_state = scenario_draft.content.agent_states[0].model_copy(
        update={
            "position": scenario_draft.content.agent_states[0].position.model_copy(
                update={"x": 1e100}
            )
        }
    )
    scenario_response = binding.apply_command(
        _request(
            {
                "command_type": "validate",
                "draft": scenario_draft.model_copy(
                    update={
                        "content": scenario_draft.content.model_copy(
                            update={
                                "agent_states": (
                                    first_state,
                                    *scenario_draft.content.agent_states[1:],
                                )
                            }
                        )
                    }
                ).model_dump(mode="json", by_alias=True),
            }
        )
    )
    huge_integer_draft = new_scenario_draft("int32-overflow")
    huge_integer_response = binding.apply_command(
        _request(
            {
                "command_type": "validate",
                "draft": huge_integer_draft.model_copy(
                    update={
                        "content": huge_integer_draft.content.model_copy(
                            update={
                                "episode": (
                                    huge_integer_draft.content.episode.model_copy(
                                        update={"max_steps": 10**100}
                                    )
                                )
                            }
                        )
                    }
                ).model_dump(mode="json", by_alias=True),
            }
        )
    )
    nested_capacity_draft = new_scenario_draft("nested-capacity")
    nested_capacity_response = binding.apply_command(
        _request(
            {
                "command_type": "validate",
                "draft": nested_capacity_draft.model_copy(
                    update={
                        "content": nested_capacity_draft.content.model_copy(
                            update={
                                "embedded_map": (
                                    nested_capacity_draft.content.embedded_map.model_copy(
                                        update={"obstacles": obstacles}
                                    )
                                )
                            }
                        )
                    }
                ).model_dump(mode="json", by_alias=True),
            }
        )
    )

    assert not map_response.ok
    assert [problem.stable_code for problem in map_response.problems] == [
        "map-obstacle-capacity-exceeded"
    ]
    assert map_response.problems[0].field_path == "obstacles"
    assert not scenario_response.ok
    assert any(
        problem.stable_code == "scenario-float32-normalization-failed"
        and problem.object_id == "agent-a1"
        and problem.field_path == "agent_states.0.position.x"
        for problem in scenario_response.problems
    )
    assert not huge_integer_response.ok
    assert any(
        problem.stable_code == "scenario-integer-not-int32"
        and problem.field_path == "episode.max_steps"
        for problem in huge_integer_response.problems
    )
    assert not nested_capacity_response.ok
    assert any(
        problem.stable_code == "map-obstacle-capacity-exceeded"
        and problem.field_path == "embedded_map.obstacles"
        for problem in nested_capacity_response.problems
    )
    assert not any(
        "embedded_map.embedded_map" in problem.field_path
        for problem in nested_capacity_response.problems
    )


def test_binding_copies_saved_map_and_duplicates_saved_scenario(tmp_path: Path) -> None:
    store = _store(tmp_path)
    binding = DevClientAuthoringBinding(store, code_revision=_code_revision())
    saved_map = store.save_draft(new_map_draft("source-map"), expected_revision=0)
    saved_scenario = store.save_draft(
        new_scenario_draft("source-scenario"), expected_revision=0
    )

    copied = binding.apply_command(
        _request(
            {
                "command_type": "new_scenario",
                "asset_id": "map-copy",
                "creation_mode": "copy_saved_map",
                "source": {
                    "source_kind": "saved_draft",
                    "asset_kind": "map",
                    "asset_id": saved_map.asset_id,
                    "revision": saved_map.revision,
                },
            }
        )
    )
    assert copied.ok
    assert isinstance(copied.draft, DevScenarioDraftV1)
    assert copied.draft.content.embedded_map == saved_map.content
    assert copied.draft.content.embedded_map is not saved_map.content
    assert copied.draft.content.source_map_provenance is not None
    assert copied.draft.content.source_map_provenance.revision == 1
    assert copied.validation is not None
    assert copied.validation.effective_movement_speeds is not None
    assert len(copied.validation.effective_movement_speeds) == 10

    duplicated = binding.apply_command(
        _request(
            {
                "command_type": "new_scenario",
                "asset_id": "scenario-copy",
                "creation_mode": "duplicate_saved_scenario",
                "source": {
                    "source_kind": "saved_draft",
                    "asset_kind": "scenario",
                    "asset_id": saved_scenario.asset_id,
                    "revision": saved_scenario.revision,
                },
            }
        )
    )
    assert duplicated.ok
    assert isinstance(duplicated.draft, DevScenarioDraftV1)
    assert duplicated.draft.asset_id == "scenario-copy"
    assert duplicated.draft.revision == 0
    assert duplicated.draft.content == saved_scenario.content


def test_authoring_command_request_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _request(
            {
                "command_type": "list",
                "asset_kind": "all",
                "filesystem_path": "/tmp/escape",
            }
        )
