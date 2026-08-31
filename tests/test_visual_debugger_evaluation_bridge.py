"""Truthful debugger launch/context bridge contracts."""

from __future__ import annotations

from dataclasses import replace

import pytest
from pydantic import ValidationError
from scripts.dev.visual_debugger.evaluation_bridge import (
    DEBUGGER_PUBLIC_AGENT_IDS_V1,
    DebuggerActionSourceKindV1,
    DebuggerCaptureProfileV1,
    DebuggerEvaluationLaunchSpecificationV1,
    build_debugger_evaluation_context_v1,
    build_debugger_evaluation_launch_specification_v1,
)
from scripts.dev.visual_debugger.model import (
    DebuggerScenarioProvenance,
    TeamBController,
)
from scripts.dev.visual_debugger.scenarios import get_scenario

from marl_battlegrounds.core.env import initialize_scenario_state
from marl_battlegrounds.evaluation.capture import (
    capture_initial_evaluation_frame_v1,
)
from marl_battlegrounds.evaluation.models import (
    REQUIRED_SCHEMA_BINDINGS_V1,
    AssignedPolicySlotV1,
    CodeRevisionV1,
    EvaluationEpisodeContextV1,
    ExecutionInformationMode,
    NotApplicablePolicySlotV1,
    canonical_json_bytes,
)


def _code_revision(*, dirty: bool = False) -> CodeRevisionV1:
    return CodeRevisionV1(
        package_version="0.0.0",
        commit_sha="a" * 40,
        source_tree_digest="b" * 64,
        is_dirty=dirty,
        dirty_patch_digest="c" * 64 if dirty else None,
    )


def _launch(
    *,
    root_seed: int = 7,
    capture_profile: DebuggerCaptureProfileV1 = "debug",
) -> DebuggerEvaluationLaunchSpecificationV1:
    return build_debugger_evaluation_launch_specification_v1(
        root_seed=root_seed,
        code_revision=_code_revision(),
        capture_profile=capture_profile,
    )


def _context(
    *,
    root_seed: int = 7,
    run_generation: int = 0,
    action_source_kind: DebuggerActionSourceKindV1 = "manual",
    capture_profile: DebuggerCaptureProfileV1 = "debug",
    team_b_controller: TeamBController = "manual",
    execution_information_mode: ExecutionInformationMode = "no_shared_obs",
    scenario_name: str = "arena_5v5",
) -> EvaluationEpisodeContextV1:
    scenario = get_scenario(scenario_name)
    config, _state = scenario.build_scenario()
    return build_debugger_evaluation_context_v1(
        _launch(root_seed=root_seed, capture_profile=capture_profile),
        scenario=scenario,
        config=config,
        run_generation=run_generation,
        action_source_kind=action_source_kind,
        team_b_controller=team_b_controller,
        execution_information_mode=execution_information_mode,
    )


def test_launch_specification_is_strict_content_addressed_and_roundtrippable() -> None:
    first = _launch()
    second = _launch()

    assert first == second
    assert first.capture_profile == "debug"
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert first.specification_id == (
        f"debugger-evaluation-launch:{first.launch_content_digest_sha256}"
    )
    assert (
        DebuggerEvaluationLaunchSpecificationV1.model_validate_json(
            first.model_dump_json()
        )
        == first
    )
    assert _launch(root_seed=8) != first
    retaining = _launch(capture_profile="evaluation_metric_complete")
    assert retaining.capture_profile == "evaluation_metric_complete"
    assert retaining.root_seed == first.root_seed
    assert retaining.code_revision == first.code_revision
    assert retaining.specification_id != first.specification_id
    assert retaining.launch_content_digest_sha256 != (
        first.launch_content_digest_sha256
    )
    assert retaining.canonical_digest_sha256 != first.canonical_digest_sha256
    assert canonical_json_bytes(retaining) != canonical_json_bytes(first)
    assert (
        build_debugger_evaluation_launch_specification_v1(
            root_seed=7,
            code_revision=_code_revision(dirty=True),
        )
        != first
    )


def test_launch_specification_rejects_bool_future_extra_and_tampered_digest() -> None:
    with pytest.raises(ValidationError):
        build_debugger_evaluation_launch_specification_v1(
            root_seed=True,  # type: ignore[arg-type]
            code_revision=_code_revision(),
        )
    with pytest.raises(ValidationError):
        build_debugger_evaluation_launch_specification_v1(
            root_seed=7,
            code_revision=_code_revision(),
            capture_profile="training_light",  # type: ignore[arg-type]
        )

    launch = _launch()
    for field_name, value in (
        ("schema_version", 2),
        ("canonical_digest_sha256", "f" * 64),
        ("future_path", "/tmp/not-provenance"),
    ):
        payload = launch.model_dump(mode="python")
        payload[field_name] = value
        with pytest.raises(ValidationError):
            DebuggerEvaluationLaunchSpecificationV1.model_validate(payload)


def test_context_is_custom_debug_no_shared_and_keeps_exact_cp2_bindings() -> None:
    context = _context()

    assert context.capture_profile == "debug"
    assert context.execution_information_mode == "no_shared_obs"
    assert context.actor_projection.identifier == "base-observation-no-shared-obs"
    assert context.critic_information_regime.identifier == "not_applicable"
    assert context.identity.paired_comparison_key is None
    assert context.identity.scenario is not None
    assert context.identity.scenario.identifier.startswith("custom-debugger-scenario:")
    assert tuple(row.public_agent_id for row in context.roster) == (
        DEBUGGER_PUBLIC_AGENT_IDS_V1
    )
    assert (
        tuple((row.schema_id, row.schema_version) for row in context.schema_versions)
        == REQUIRED_SCHEMA_BINDINGS_V1
    )
    assert tuple(row.name for row in context.aggregation_keys) == tuple(
        sorted(row.name for row in context.aggregation_keys)
    )
    assert dict((row.name, row.value) for row in context.aggregation_keys) == {
        "action_source": "manual",
        "information_regime": "no_shared_obs",
        "scenario": "arena_5v5",
        "scenario_kind": "custom",
        "team_b_controller": "manual",
        "tool": "visual_debugger",
    }


def test_mixed_action_source_assigns_manual_team_a_and_scripted_tdm_team_b() -> None:
    context = _context(
        action_source_kind="mixed",
        team_b_controller="scripted_tdm",
    )
    assigned = tuple(
        row
        for row in context.policy_assignments
        if isinstance(row, AssignedPolicySlotV1)
    )
    inactive = tuple(
        row
        for row in context.policy_assignments
        if isinstance(row, NotApplicablePolicySlotV1)
    )

    assert tuple(row.global_slot for row in assigned) == tuple(range(10))
    assert not inactive
    assert tuple(row.evaluation_role for row in assigned) == (
        "focal",
        "cooperative_partner",
        "cooperative_partner",
        "cooperative_partner",
        "cooperative_partner",
        "adversarial_opponent",
        "adversarial_opponent",
        "adversarial_opponent",
        "adversarial_opponent",
        "adversarial_opponent",
    )
    assert tuple(row.policy_kind for row in assigned) == (
        *("manual" for _ in range(5)),
        *("scripted_tdm" for _ in range(5)),
    )
    assert tuple(row.algorithm_id for row in assigned) == (
        *("not_applicable" for _ in range(5)),
        *("canonical-scripted-team-deathmatch" for _ in range(5)),
    )
    assert {row.training_run_id for row in assigned} == {"not_applicable"}
    assert {row.training_step for row in assigned} == {0}
    assert {row.checkpoint_digest for row in assigned} == {None}
    assert {row.population_member_id for row in assigned} == {None}


def test_context_identities_join_config_scenario_action_code_and_generation() -> None:
    scenario = get_scenario("arena_5v5")
    config, _state = scenario.build_scenario()
    launch = _launch()
    first = build_debugger_evaluation_context_v1(
        launch,
        scenario=scenario,
        config=config,
        run_generation=0,
        action_source_kind="mixed",
        team_b_controller="scripted_tdm",
        execution_information_mode="no_shared_obs",
    )
    same = build_debugger_evaluation_context_v1(
        launch,
        scenario=scenario,
        config=config,
        run_generation=0,
        action_source_kind="mixed",
        team_b_controller="scripted_tdm",
        execution_information_mode="no_shared_obs",
    )
    next_generation = build_debugger_evaluation_context_v1(
        launch,
        scenario=scenario,
        config=config,
        run_generation=1,
        action_source_kind="mixed",
        team_b_controller="scripted_tdm",
        execution_information_mode="no_shared_obs",
    )
    manual = build_debugger_evaluation_context_v1(
        launch,
        scenario=scenario,
        config=config,
        run_generation=0,
        action_source_kind="manual",
        team_b_controller="manual",
        execution_information_mode="no_shared_obs",
    )
    scaled_config = config._replace(ordinary_movement_distance_scale=0.5)
    scaled = build_debugger_evaluation_context_v1(
        launch,
        scenario=scenario,
        config=scaled_config,
        run_generation=0,
        action_source_kind="mixed",
        team_b_controller="scripted_tdm",
        execution_information_mode="no_shared_obs",
    )
    other_scenario = get_scenario("ultimate_showcase")
    other_config, _other_state = other_scenario.build_scenario()
    other = build_debugger_evaluation_context_v1(
        launch,
        scenario=other_scenario,
        config=other_config,
        run_generation=0,
        action_source_kind="scripted",
        team_b_controller="manual",
        execution_information_mode="no_shared_obs",
    )

    assert first == same
    assert canonical_json_bytes(first) == canonical_json_bytes(same)
    assert first.code_revision == launch.code_revision
    assert first.identity.episode_id != next_generation.identity.episode_id
    assert first.identity.match_id != next_generation.identity.match_id
    assert first.identity.evaluation_id == next_generation.identity.evaluation_id
    assert first.identity.matchup_id == next_generation.identity.matchup_id
    assert first.identity.episode_id != manual.identity.episode_id
    assert first.identity.episode_id != scaled.identity.episode_id
    assert first.identity.episode_id != other.identity.episode_id
    assert first.resolved_env_config.canonical_digest_sha256 != (
        scaled.resolved_env_config.canonical_digest_sha256
    )
    assert first.identity.scenario != other.identity.scenario
    first_policy = first.policy_assignments[0]
    manual_policy = manual.policy_assignments[0]
    assert isinstance(first_policy, AssignedPolicySlotV1)
    assert isinstance(manual_policy, AssignedPolicySlotV1)
    assert first_policy.policy_content_digest != manual_policy.policy_content_digest


def test_named_seeds_are_deterministic_bounded_and_generation_stable() -> None:
    first = _context(root_seed=123, run_generation=0)
    same = _context(root_seed=123, run_generation=0)
    restarted = _context(root_seed=123, run_generation=1)

    assert first.seed_protocol == same.seed_protocol
    assert first.seed_protocol == restarted.seed_protocol
    values = (
        first.seed_protocol.root_seed,
        first.seed_protocol.episode_seed,
        first.seed_protocol.layout_seed,
        first.seed_protocol.environment_seed,
        first.seed_protocol.focal_policy_seed,
        first.seed_protocol.evaluation_seed,
        first.seed_protocol.cooperative_partner_seed,
        first.seed_protocol.adversarial_opponent_seed,
        first.seed_protocol.scenario_seed,
    )
    assert all(type(value) is int and 0 <= value <= 2**32 - 1 for value in values)


def test_information_mode_changes_projection_and_identity_but_not_named_seeds() -> None:
    no_shared = _context(execution_information_mode="no_shared_obs")
    shared = _context(execution_information_mode="shared_obs")

    assert shared.execution_information_mode == "shared_obs"
    assert shared.actor_projection.identifier == (
        "base-observation-plus-authorized-sensor-source-bank"
    )
    assert shared.actor_projection.version == 1
    assert no_shared.actor_projection.version == 2
    assert shared.identity.episode_id != no_shared.identity.episode_id
    assert shared.identity.evaluation_id != no_shared.identity.evaluation_id
    assert shared.seed_protocol == no_shared.seed_protocol


def test_authored_team_deathmatch_uses_independent_task_map_and_scenario_identity() -> (
    None
):
    source = get_scenario("arena_5v5")
    neutral_config, state = source.build_scenario()
    tdm_config = neutral_config._replace(
        task_mode=1,
        team_deathmatch_score_threshold=5,
    )
    authored = replace(
        source,
        name="authored-tdm",
        build_scenario=lambda: (tdm_config, state),
        provenance=DebuggerScenarioProvenance(
            source_kind="candidate",
            source_identity="candidate:" + "a" * 64,
            scenario_semantic_digest="b" * 64,
            map_semantic_digest="c" * 64,
            resolved_configuration_digest="d" * 64,
            resolved_initial_state_digest="e" * 64,
        ),
    )
    context = build_debugger_evaluation_context_v1(
        _launch(),
        scenario=authored,
        config=tdm_config,
        run_generation=0,
        action_source_kind="mixed",
        team_b_controller="scripted_tdm",
        execution_information_mode="shared_obs",
    )

    assert context.identity.task.identifier == "team_deathmatch"
    assert context.identity.layout.identifier == "authored-map"
    assert context.identity.layout.canonical_digest == "c" * 64
    assert context.identity.scenario is not None
    assert context.identity.scenario.identifier == "authored-team-deathmatch-scenario"
    assert context.identity.scenario.canonical_digest == "b" * 64


def test_context_captures_the_authored_initial_frame_through_public_cp2_api() -> None:
    scenario = get_scenario("basic_support")
    config, authored_state = scenario.build_scenario()
    context = build_debugger_evaluation_context_v1(
        _launch(capture_profile="evaluation_metric_complete"),
        scenario=scenario,
        config=config,
        run_generation=0,
        action_source_kind="scripted",
        team_b_controller="manual",
        execution_information_mode="no_shared_obs",
    )
    state, observation, action_mask, _info = initialize_scenario_state(
        authored_state,
        config,
    )

    frame = capture_initial_evaluation_frame_v1(
        context,
        state,
        observation,
        action_mask,
        None,
    )
    assert frame.episode_id == context.identity.episode_id
    assert frame.frame_index == 0
    assert context.capture_profile == "evaluation_metric_complete"


def test_context_rejects_invalid_horizon_generation_source_and_focal_slot() -> None:
    scenario = get_scenario("arena_5v5")
    config, _state = scenario.build_scenario()
    launch = _launch()

    with pytest.raises(ValueError, match="run_generation"):
        build_debugger_evaluation_context_v1(
            launch,
            scenario=scenario,
            config=config,
            run_generation=True,  # type: ignore[arg-type]
            action_source_kind="mixed",
            team_b_controller="scripted_tdm",
            execution_information_mode="no_shared_obs",
        )
    with pytest.raises(ValueError, match="expected_horizon"):
        build_debugger_evaluation_context_v1(
            launch,
            scenario=scenario,
            config=config,
            run_generation=0,
            action_source_kind="mixed",
            team_b_controller="scripted_tdm",
            execution_information_mode="no_shared_obs",
            expected_horizon=config.max_steps + 1,
        )
    with pytest.raises(ValueError, match="action_source_kind"):
        build_debugger_evaluation_context_v1(
            launch,
            scenario=scenario,
            config=config,
            run_generation=0,
            action_source_kind="policy",  # type: ignore[arg-type]
            team_b_controller="scripted_tdm",
            execution_information_mode="no_shared_obs",
        )
    sparse_scenario = get_scenario("basic_support")
    sparse_config, _sparse_state = sparse_scenario.build_scenario()
    inactive_default = replace(sparse_scenario, default_controlled_slot=3)
    with pytest.raises(ValueError, match="default actor"):
        build_debugger_evaluation_context_v1(
            launch,
            scenario=inactive_default,
            config=sparse_config,
            run_generation=0,
            action_source_kind="scripted",
            team_b_controller="manual",
            execution_information_mode="no_shared_obs",
        )


def test_serialized_bridge_records_contain_no_local_path_or_browser_token() -> None:
    launch = _launch(capture_profile="evaluation_metric_complete")
    payload = canonical_json_bytes(
        {
            "launch": launch,
            "context": _context(capture_profile="evaluation_metric_complete"),
        }
    )
    assert b"/home/" not in payload
    assert b"file://" not in payload
    assert b"capability_token" not in payload
    assert b"browser_token" not in payload
