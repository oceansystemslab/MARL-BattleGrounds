"""Truthful launch and episode provenance for debugger evaluation capture.

This module is intentionally narrower than the live debugger service.  It
constructs immutable CP2 context records from explicit launch inputs, but owns
neither simulator execution nor CP3 observer lifecycle.  In particular, code
revision provenance is supplied by the launcher after one launch-scoped source
inspection; this module never discovers Git state, local paths, or browser
tokens on its own.
"""

from __future__ import annotations

import hashlib
from typing import Annotated, Literal

import numpy as np
from pydantic import Field, StringConstraints, model_validator

from marl_battlegrounds.core.config import validate_env_config
from marl_battlegrounds.core.types import (
    MAX_AGENT_SLOTS,
    TASK_MODE_TDM,
    TEAM_B_ID,
    EnvConfig,
)
from marl_battlegrounds.evaluation.actor_projection import (
    NO_SHARED_OBS_ACTOR_PROJECTION_V2,
    SHARED_OBS_ACTOR_PROJECTION_V1,
)
from marl_battlegrounds.evaluation.catalog import (
    build_evaluation_episode_context_v1,
    build_evaluation_seed_protocol_v1,
    build_resolved_env_config_v1,
)
from marl_battlegrounds.evaluation.models import (
    AggregationKeyV1,
    AssignedPolicySlotV1,
    CodeRevisionV1,
    ContentAddressedIdentityV1,
    EvaluationEpisodeContextV1,
    EvaluationEpisodeIdentityV1,
    EvaluationModel,
    EvaluationRole,
    ExecutionInformationMode,
    NotApplicablePolicySlotV1,
    PolicyAssignmentSlotV1,
    VersionedIdentityV1,
    canonical_digest_sha256,
    canonical_json_bytes,
)
from scripts.dev.visual_debugger.model import DebuggerScenario, TeamBController

DEBUGGER_EVALUATION_BRIDGE_SCHEMA_VERSION: Literal[1] = 1
DEBUGGER_EVALUATION_LAUNCH_SPECIFICATION_SCHEMA_ID = (
    "marl_battlegrounds.visual_debugger.evaluation_launch_specification"
)
DEBUGGER_PUBLIC_AGENT_IDS_V1 = tuple(str(slot) for slot in range(MAX_AGENT_SLOTS))

type DebuggerActionSourceKindV1 = Literal["manual", "scripted", "mixed"]
type DebuggerCaptureProfileV1 = Literal["debug", "evaluation_metric_complete"]

_Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
_Seed = Annotated[int, Field(ge=0, le=2**32 - 1)]
_NonNegativeInt = Annotated[int, Field(ge=0)]


class DebuggerEvaluationLaunchSpecificationV1(EvaluationModel):
    """Explicit launch-scoped inputs shared by every restarted episode."""

    schema_id: Literal[
        "marl_battlegrounds.visual_debugger.evaluation_launch_specification"
    ] = DEBUGGER_EVALUATION_LAUNCH_SPECIFICATION_SCHEMA_ID
    schema_version: Literal[1] = DEBUGGER_EVALUATION_BRIDGE_SCHEMA_VERSION
    specification_id: Annotated[
        str,
        StringConstraints(pattern=r"^debugger-evaluation-launch:[0-9a-f]{64}$"),
    ]
    launch_content_digest_sha256: _Sha256Hex
    canonical_digest_sha256: _Sha256Hex
    root_seed: _Seed
    code_revision: CodeRevisionV1
    capture_profile: DebuggerCaptureProfileV1

    @model_validator(mode="after")
    def _validate_launch_specification(
        self,
    ) -> DebuggerEvaluationLaunchSpecificationV1:
        launch_payload = {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "root_seed": self.root_seed,
            "code_revision": self.code_revision,
            "capture_profile": self.capture_profile,
        }
        expected_content_digest = canonical_digest_sha256(launch_payload)
        if self.launch_content_digest_sha256 != expected_content_digest:
            raise ValueError("debugger launch content digest mismatch")
        if self.specification_id != (
            f"debugger-evaluation-launch:{expected_content_digest}"
        ):
            raise ValueError("debugger launch specification ID is not canonical")
        if self.canonical_digest_sha256 != canonical_digest_sha256(
            self,
            exclude={"canonical_digest_sha256"},
        ):
            raise ValueError("debugger launch specification digest mismatch")
        return self


def build_debugger_evaluation_launch_specification_v1(
    *,
    root_seed: int,
    code_revision: CodeRevisionV1,
    capture_profile: DebuggerCaptureProfileV1 = "debug",
) -> DebuggerEvaluationLaunchSpecificationV1:
    """Build one launch record without performing filesystem or Git discovery."""
    launch_payload = {
        "schema_id": DEBUGGER_EVALUATION_LAUNCH_SPECIFICATION_SCHEMA_ID,
        "schema_version": DEBUGGER_EVALUATION_BRIDGE_SCHEMA_VERSION,
        "root_seed": root_seed,
        "code_revision": code_revision,
        "capture_profile": capture_profile,
    }
    launch_content_digest = canonical_digest_sha256(launch_payload)
    payload = {
        **launch_payload,
        "specification_id": f"debugger-evaluation-launch:{launch_content_digest}",
        "launch_content_digest_sha256": launch_content_digest,
    }
    payload["canonical_digest_sha256"] = canonical_digest_sha256(payload)
    return DebuggerEvaluationLaunchSpecificationV1.model_validate(payload)


def _scenario_contract_payload(scenario: DebuggerScenario) -> dict[str, object]:
    """Project stable authored metadata without serializing its state callback."""
    if type(scenario) is not DebuggerScenario:
        raise TypeError("scenario must be the exact DebuggerScenario type")
    payload: dict[str, object] = {
        "schema_id": "marl_battlegrounds.visual_debugger.scenario_contract",
        "schema_version": 1,
        "name": scenario.name,
        "title": scenario.title,
        "description": scenario.description,
        "mode": scenario.mode,
        "audience": scenario.audience,
        "default_controlled_slot": scenario.default_controlled_slot,
        "frames": tuple(
            {
                "label": frame.label,
                "description": frame.description,
                "commands": tuple(
                    {
                        "actor_global_slot": command.actor_global_slot,
                        "move_action": command.move_action,
                        "target_global_slot": command.target_global_slot,
                        "use_ultimate": command.use_ultimate,
                    }
                    for command in frame.commands
                ),
            }
            for frame in scenario.frames
        ),
    }
    provenance = scenario.provenance
    if provenance is not None:
        payload["authored_provenance"] = {
            "source_kind": provenance.source_kind,
            "source_identity": provenance.source_identity,
            "scenario_semantic_digest": provenance.scenario_semantic_digest,
            "map_semantic_digest": provenance.map_semantic_digest,
            "resolved_configuration_digest": provenance.resolved_configuration_digest,
            "resolved_initial_state_digest": provenance.resolved_initial_state_digest,
        }
    return payload


def _content_identity(
    identifier: str,
    payload: dict[str, object],
) -> ContentAddressedIdentityV1:
    return ContentAddressedIdentityV1(
        identifier=identifier,
        version=1,
        canonical_digest=canonical_digest_sha256(payload),
    )


def _derive_named_seed(
    root_seed: int,
    *,
    namespace: str,
) -> int:
    """Derive one launch-stable seed for exact same-start comparisons."""
    payload = canonical_json_bytes(
        {
            "schema_id": "marl_battlegrounds.visual_debugger.named_seed",
            "schema_version": 1,
            "root_seed": root_seed,
            "namespace": namespace,
        }
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _action_source_contract_payload(
    *,
    action_source_kind: DebuggerActionSourceKindV1,
    team_b_controller: TeamBController,
    execution_information_mode: ExecutionInformationMode,
    scenario_contract_digest: str,
) -> dict[str, object]:
    if action_source_kind not in ("manual", "scripted", "mixed"):
        raise ValueError("action_source_kind must be manual, scripted, or mixed")
    return {
        "schema_id": "marl_battlegrounds.visual_debugger.action_source_contract",
        "schema_version": 1,
        "action_source_kind": action_source_kind,
        "team_b_controller": team_b_controller,
        "execution_information_mode": execution_information_mode,
        "manual_submission_included": action_source_kind in ("manual", "mixed"),
        "scripted_submission_included": action_source_kind in ("scripted", "mixed"),
        "scenario_contract_digest_sha256": scenario_contract_digest,
        "policy_execution_included": (
            action_source_kind == "mixed" and team_b_controller == "scripted_tdm"
        ),
    }


def _policy_assignments(
    config: EnvConfig,
    scenario: DebuggerScenario,
    *,
    action_source_kind: DebuggerActionSourceKindV1,
    team_b_controller: TeamBController,
    actor_projection: VersionedIdentityV1,
    action_contract_digest: str,
) -> tuple[PolicyAssignmentSlotV1, ...]:
    profile = config.agent_profile
    active = np.asarray(profile.active_mask, dtype=np.bool_)
    team_ids = np.asarray(profile.team_ids, dtype=np.int32)
    focal_slot = scenario.default_controlled_slot
    if not bool(active[focal_slot]):
        raise ValueError("scenario default actor must be configured active")
    focal_team_id = int(team_ids[focal_slot])
    rows: list[PolicyAssignmentSlotV1] = []
    for slot in range(MAX_AGENT_SLOTS):
        if not bool(active[slot]):
            rows.append(NotApplicablePolicySlotV1(global_slot=slot))
            continue
        if slot == focal_slot:
            role: EvaluationRole = "focal"
        elif int(team_ids[slot]) == focal_team_id:
            role = "cooperative_partner"
        else:
            role = "adversarial_opponent"
        policy_kind = (
            "scripted_tdm"
            if action_source_kind == "mixed"
            and team_b_controller == "scripted_tdm"
            and int(team_ids[slot]) == TEAM_B_ID
            else "manual"
            if action_source_kind == "mixed"
            else action_source_kind
        )
        scripted_tdm = policy_kind == "scripted_tdm"
        rows.append(
            AssignedPolicySlotV1(
                global_slot=slot,
                evaluation_role=role,
                policy_kind=policy_kind,
                policy_id=f"debugger-action-source:{policy_kind}:slot:{slot}",
                policy_content_digest=action_contract_digest,
                checkpoint_digest=None,
                algorithm_id=(
                    "canonical-scripted-team-deathmatch"
                    if scripted_tdm
                    else "not_applicable"
                ),
                training_run_id="not_applicable",
                training_step=0,
                population_member_id=None,
                parameter_sharing_group_id=(
                    f"debugger-action-source:{policy_kind}:team:{int(team_ids[slot])}"
                ),
                preprocessing=actor_projection,
                normalization=VersionedIdentityV1(
                    identifier="none",
                    version=1,
                ),
                # Manual input is not a sampled policy; this records the lack
                # of policy RNG rather than claiming repeatable human choices.
                execution_mode="deterministic",
            )
        )
    return tuple(rows)


def build_debugger_evaluation_context_v1(
    launch_specification: DebuggerEvaluationLaunchSpecificationV1,
    *,
    scenario: DebuggerScenario,
    config: EnvConfig,
    run_generation: int,
    action_source_kind: DebuggerActionSourceKindV1,
    team_b_controller: TeamBController,
    execution_information_mode: ExecutionInformationMode,
    expected_horizon: int | None = None,
) -> EvaluationEpisodeContextV1:
    """Build the truthful custom, nonofficial CP2 context for one live episode.

    The eventual ``create_session``/restart integration must construct this
    context from the effective post-override ``EnvConfig`` and use
    ``seed_protocol.environment_seed`` as the simulator RNG seed.  The same
    launch specification is reused while ``run_generation`` increments for
    each deliberate episode replacement.
    """
    if type(launch_specification) is not DebuggerEvaluationLaunchSpecificationV1:
        raise TypeError(
            "launch_specification must be the exact V1 launch specification"
        )
    launch = DebuggerEvaluationLaunchSpecificationV1.model_validate_json(
        launch_specification.model_dump_json()
    )
    if type(run_generation) is not int or run_generation < 0:
        raise ValueError("run_generation must be a nonnegative exact integer")
    if type(config) is not EnvConfig:
        raise TypeError("config must be the exact EnvConfig type")
    if team_b_controller not in ("manual", "scripted_tdm"):
        raise ValueError("team_b_controller must be manual or scripted_tdm")
    if execution_information_mode not in ("shared_obs", "no_shared_obs"):
        raise ValueError(
            "execution_information_mode must be shared_obs or no_shared_obs"
        )
    expected_action_source_kind: DebuggerActionSourceKindV1 = (
        "scripted"
        if scenario.mode == "scripted"
        else "mixed"
        if team_b_controller == "scripted_tdm"
        else "manual"
    )
    if action_source_kind != expected_action_source_kind:
        raise ValueError(
            "action_source_kind must match scenario mode and Team B controller"
        )
    validate_env_config(config)
    resolved_config = build_resolved_env_config_v1(config)
    horizon = config.max_steps if expected_horizon is None else expected_horizon
    if type(horizon) is not int or not 0 < horizon <= config.max_steps:
        raise ValueError("expected_horizon must be an exact positive config bound")

    scenario_payload = _scenario_contract_payload(scenario)
    scenario_digest = canonical_digest_sha256(scenario_payload)
    scenario_identity_digest = (
        scenario.provenance.scenario_semantic_digest
        if scenario.provenance is not None
        else scenario_digest
    )
    layout_identity_digest = (
        scenario.provenance.map_semantic_digest
        if scenario.provenance is not None
        else resolved_config.canonical_digest_sha256
    )
    actor_projection = (
        SHARED_OBS_ACTOR_PROJECTION_V1
        if execution_information_mode == "shared_obs"
        else NO_SHARED_OBS_ACTOR_PROJECTION_V2
    )
    action_payload = _action_source_contract_payload(
        action_source_kind=action_source_kind,
        team_b_controller=team_b_controller,
        execution_information_mode=execution_information_mode,
        scenario_contract_digest=scenario_digest,
    )
    action_digest = canonical_digest_sha256(action_payload)
    config_digest = resolved_config.canonical_digest_sha256
    evaluation_payload: dict[str, object] = {
        "schema_id": "marl_battlegrounds.visual_debugger.evaluation_assignment",
        "schema_version": 1,
        "launch_content_digest_sha256": launch.launch_content_digest_sha256,
        "scenario_contract_digest_sha256": scenario_digest,
        "scenario_identity_digest_sha256": scenario_identity_digest,
        "layout_identity_digest_sha256": layout_identity_digest,
        "resolved_config_digest_sha256": config_digest,
        "action_source_contract_digest_sha256": action_digest,
        "execution_information_mode": execution_information_mode,
        "expected_horizon": horizon,
    }
    evaluation_digest = canonical_digest_sha256(evaluation_payload)
    generation_payload: dict[str, object] = {
        "schema_id": "marl_battlegrounds.visual_debugger.episode_assignment",
        "schema_version": 1,
        "evaluation_assignment_digest_sha256": evaluation_digest,
        "run_generation": run_generation,
    }
    assignment_digest = canonical_digest_sha256(generation_payload)
    matchup_digest = canonical_digest_sha256(
        {
            "scenario_contract_digest_sha256": scenario_digest,
            "scenario_identity_digest_sha256": scenario_identity_digest,
            "layout_identity_digest_sha256": layout_identity_digest,
            "resolved_config_digest_sha256": config_digest,
            "action_source_contract_digest_sha256": action_digest,
            "execution_information_mode": execution_information_mode,
        }
    )

    evaluation_suite = _content_identity(
        "visual-debugger-custom-suite",
        {
            "schema_id": "marl_battlegrounds.visual_debugger.custom_suite",
            "schema_version": 1,
            "official": False,
            "audience": "researcher",
        },
    )
    experiment_manifest = _content_identity(
        "visual-debugger-custom-manifest",
        evaluation_payload,
    )
    task = _content_identity(
        "team_deathmatch"
        if config.task_mode == TASK_MODE_TDM
        else "visual-debugger-analysis-task",
        {
            "schema_id": "marl_battlegrounds.visual_debugger.task",
            "schema_version": 1,
            "official": False,
            "task_kind": (
                "team_deathmatch"
                if config.task_mode == TASK_MODE_TDM
                else "interactive_visual_analysis"
            ),
        },
    )
    layout = ContentAddressedIdentityV1(
        identifier=(
            "authored-map"
            if scenario.provenance is not None
            else "resolved-debugger-environment"
        ),
        version=1,
        canonical_digest=layout_identity_digest,
    )
    scenario_identity = ContentAddressedIdentityV1(
        identifier=(
            "authored-team-deathmatch-scenario"
            if scenario.provenance is not None and config.task_mode == TASK_MODE_TDM
            else f"custom-debugger-scenario:{scenario.name}"
        ),
        version=1,
        canonical_digest=scenario_identity_digest,
    )
    identity = EvaluationEpisodeIdentityV1(
        run_id=f"debugger-run:{launch.launch_content_digest_sha256}",
        evaluation_id=f"debugger-evaluation:{evaluation_digest}",
        matchup_id=f"debugger-matchup:{matchup_digest}",
        match_id=f"debugger-match:{assignment_digest}",
        episode_id=f"debugger-episode:{assignment_digest}",
        paired_comparison_key=None,
        evaluation_suite=evaluation_suite,
        experiment_manifest=experiment_manifest,
        task=task,
        layout=layout,
        curriculum=None,
        scenario=scenario_identity,
    )

    assignments = _policy_assignments(
        config,
        scenario,
        action_source_kind=action_source_kind,
        team_b_controller=team_b_controller,
        actor_projection=actor_projection,
        action_contract_digest=action_digest,
    )
    active_roles = {
        row.evaluation_role
        for row in assignments
        if isinstance(row, AssignedPolicySlotV1)
    }

    def seed(namespace: str) -> int:
        return _derive_named_seed(
            launch.root_seed,
            namespace=namespace,
        )

    seed_protocol = build_evaluation_seed_protocol_v1(
        seed_protocol=VersionedIdentityV1(
            identifier="debugger-namespaced-sha256-u32",
            version=1,
        ),
        root_seed=launch.root_seed,
        episode_seed=seed("episode"),
        layout_seed=seed("layout"),
        environment_seed=seed("environment"),
        focal_policy_seed=seed("focal-action-source"),
        evaluation_seed=seed("evaluation"),
        cooperative_partner_seed=(
            seed("cooperative-action-source")
            if "cooperative_partner" in active_roles
            else "not_applicable"
        ),
        adversarial_opponent_seed=(
            seed("adversarial-action-source")
            if "adversarial_opponent" in active_roles
            else "not_applicable"
        ),
        scenario_seed=seed("scenario"),
    )
    aggregation_keys = [
        AggregationKeyV1(name="action_source", value=action_source_kind),
        AggregationKeyV1(name="information_regime", value=execution_information_mode),
        AggregationKeyV1(name="scenario", value=scenario.name),
        AggregationKeyV1(name="scenario_kind", value="custom"),
        AggregationKeyV1(name="team_b_controller", value=team_b_controller),
        AggregationKeyV1(name="tool", value="visual_debugger"),
    ]
    if scenario.provenance is not None:
        aggregation_keys.extend(
            (
                AggregationKeyV1(
                    name="scenario_source",
                    value=scenario.provenance.source_identity,
                ),
                AggregationKeyV1(
                    name="scenario_digest",
                    value=scenario.provenance.scenario_semantic_digest,
                ),
                AggregationKeyV1(
                    name="map_digest",
                    value=scenario.provenance.map_semantic_digest,
                ),
            )
        )
    aggregation_keys.sort(key=lambda row: row.name)

    return build_evaluation_episode_context_v1(
        identity=identity,
        aggregation_keys=tuple(aggregation_keys),
        expected_horizon=horizon,
        config=config,
        public_agent_id_by_global_slot=DEBUGGER_PUBLIC_AGENT_IDS_V1,
        policy_assignments=assignments,
        seed_protocol=seed_protocol,
        capture_profile=launch.capture_profile,
        execution_information_mode=execution_information_mode,
        actor_projection=actor_projection,
        critic_information_regime=VersionedIdentityV1(
            identifier="not_applicable",
            version=1,
        ),
        canonical_reward_mode=VersionedIdentityV1(
            identifier="canonical-task-reward",
            version=1,
        ),
        shaping_configuration=_content_identity(
            "not_applicable-no-shaping",
            {
                "schema_id": "marl_battlegrounds.visual_debugger.shaping",
                "schema_version": 1,
                "enabled": False,
            },
        ),
        code_revision=launch.code_revision,
    )


__all__ = [
    "DEBUGGER_EVALUATION_BRIDGE_SCHEMA_VERSION",
    "DEBUGGER_EVALUATION_LAUNCH_SPECIFICATION_SCHEMA_ID",
    "DEBUGGER_PUBLIC_AGENT_IDS_V1",
    "DebuggerActionSourceKindV1",
    "DebuggerCaptureProfileV1",
    "DebuggerEvaluationLaunchSpecificationV1",
    "build_debugger_evaluation_context_v1",
    "build_debugger_evaluation_launch_specification_v1",
]
