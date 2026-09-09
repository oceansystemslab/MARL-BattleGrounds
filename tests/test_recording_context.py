"""Recorded provenance distinguishes known facts from absent external history."""

from pathlib import Path

import jax
import pytest

from marl_battlegrounds.evaluation import revision
from marl_battlegrounds.evaluation.actor_projection import (
    NO_SHARED_OBS_ACTOR_PROJECTION_V2,
)
from marl_battlegrounds.evaluation.models import AssignedPolicySlotV2, CodeRevisionV2
from marl_battlegrounds.evaluation.recording_context import build_recording_context
from marl_battlegrounds.evaluation.runtime_provenance import capture_runtime_provenance
from marl_battlegrounds.tasks import make_standard_team_deathmatch_config


@pytest.mark.parametrize("mode", ["shared_obs", "no_shared_obs"])
def test_custom_trainer_unknown_history_and_seeds_stay_absent(mode: str) -> None:
    config = make_standard_team_deathmatch_config(
        map_id=0,
        team_a_roster=("priest",),
        team_b_roster=("mage", "mage"),
        max_steps=2,
    )
    context, runtime = build_recording_context(
        jax.device_get(config),
        run_id="run-1",
        phase="training",
        pass_id="1",
        episode={"episode_id": 7},
        policies={"team_a": "Learner A", "team_b": "Learner B"},
        details={
            "code_revision": CodeRevisionV2(package_version="0.0.0").model_dump(
                mode="json"
            ),
            "runtime_provenance": capture_runtime_provenance("0.0.0").model_dump(
                mode="json"
            ),
            "execution_information_mode": mode,
        },
    )
    assert context.code_revision.commit_sha is None
    assert context.code_revision.source_tree_digest is None
    assert context.seed_protocol.root_seed is None
    assert context.seed_protocol.focal_policy_seed is None
    assert context.execution_information_mode == mode
    if mode == "no_shared_obs":
        assert context.actor_projection == NO_SHARED_OBS_ACTOR_PROJECTION_V2
    learner = context.policy_assignments[0]
    assert isinstance(learner, AssignedPolicySlotV2)
    assert learner.lifecycle == "evolving"
    assert (
        learner.training_run_id is learner.training_step is learner.algorithm_id is None
    )
    assert learner.preprocessing is learner.normalization is None
    assert runtime.package_version == "0.0.0"


def test_installed_package_records_real_content_without_inventing_git(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "marl_battlegrounds"
    module = package / "evaluation" / "revision.py"
    module.parent.mkdir(parents=True)
    module.write_text("value = 1\n")
    monkeypatch.setattr(revision, "__file__", str(module))
    first = revision.discover_code_revision_v2()
    assert isinstance(first, CodeRevisionV2)
    assert first.commit_sha is first.is_dirty is first.dirty_patch_digest is None
    assert first.source_tree_digest is not None
    assert revision.discover_code_revision_v2() == first
    module.write_text("value = 2\n")
    assert (
        revision.discover_code_revision_v2().source_tree_digest
        != first.source_tree_digest
    )


def test_scenario_metadata_preserves_actual_controller_and_execution_facts() -> None:
    from marl_battlegrounds.evaluation.evaluate import policy_description
    from marl_battlegrounds.evaluation.models import ContentAddressedIdentityV1
    from marl_battlegrounds.evaluation.policy_execution import (
        Policy,
        controller_identity,
        policy,
    )
    from marl_battlegrounds.evaluation.tdm_scenarios import (
        TDM_SCENARIO_PUBLIC_AGENT_IDS,
        build_tdm_qualification_seed_schedule,
    )

    original = policy("tdm-alpha")
    renamed = Policy("Researcher's display label", original.apply)
    identity = controller_identity(renamed)
    assert identity == controller_identity(original)
    assert identity != controller_identity(policy("tdm-beta"))
    descriptor = policy_description(renamed, (), (), include_digests=True)
    config = make_standard_team_deathmatch_config(
        map_id=0, max_steps=2, team_a_roster=("mage",) * 5, team_b_roster=("mage",) * 5
    )
    scenario_identity = ContentAddressedIdentityV1(
        identifier="approved-scenario", version=1, canonical_digest="b" * 64
    )
    schedule = build_tdm_qualification_seed_schedule()
    episode: dict[str, object] = {
        "episode_id": 7,
        "seed_id": 1,
        "scenario_identity": scenario_identity,
        "public_agent_id_by_global_slot": TDM_SCENARIO_PUBLIC_AGENT_IDS,
        "evaluation_role_by_global_slot": ("focal",) * 5
        + ("adversarial_opponent",) * 5,
        "seed_protocol": schedule.realized_seed_protocols[1],
        "team_a_controller_identity": identity,
        "team_b_controller_identity": identity,
    }
    context, _ = build_recording_context(
        config,
        run_id="r",
        phase="evaluation",
        pass_id="p",
        episode=episode,
        policies={"team_a": renamed.name, "team_b": renamed.name},
        details={
            "seed": 0,
            "rng_protocol": "episode-fold-in-v1",
            "policies": [descriptor, descriptor],
            "code_revision": CodeRevisionV2(package_version="0.0.0"),
            "runtime_provenance": capture_runtime_provenance("0.0.0"),
        },
    )
    assert context.identity.scenario == scenario_identity
    assert (
        tuple(row.public_agent_id for row in context.roster)
        == TDM_SCENARIO_PUBLIC_AGENT_IDS
    )
    assert (
        context.seed_protocol.root_seed == 0 and context.seed_protocol.episode_seed == 1
    )
    assignment = context.policy_assignments[1]
    assert isinstance(assignment, AssignedPolicySlotV2)
    assert assignment.evaluation_role == "focal"
    assert assignment.policy_content_digest == descriptor["variables_digest"]
    assert assignment.callable_name is not None
    assert assignment.training_step is None
    keys = {row.name: row.value for row in context.aggregation_keys}
    assert (
        ContentAddressedIdentityV1.model_validate_json(
            keys["team_a_controller_identity"]
        ).model_dump()
        == identity
    )


@pytest.mark.parametrize(
    "conflict",
    ("root_seed", "episode_seed", "seed_protocol", "controller", "roles", "public_ids"),
)
def test_scenario_metadata_cannot_replace_execution_authority(conflict: str) -> None:
    from marl_battlegrounds.evaluation.evaluate import policy_description
    from marl_battlegrounds.evaluation.policy_execution import (
        controller_identity,
        policy,
    )
    from marl_battlegrounds.evaluation.tdm_scenarios import (
        build_tdm_qualification_seed_schedule,
    )

    team = policy("tdm-alpha")
    descriptor = policy_description(team, (), (), include_digests=True)
    seeds = (
        build_tdm_qualification_seed_schedule()
        .realized_seed_protocols[0]
        .model_dump(mode="json")
    )
    episode: dict[str, object] = {"episode_id": 1, "seed_id": 0, "seed_protocol": seeds}
    if conflict in ("root_seed", "episode_seed"):
        seeds[conflict] = 99
    elif conflict == "seed_protocol":
        seeds[conflict] = {"identifier": "different-rng", "version": 1}
    elif conflict == "controller":
        episode["team_a_controller_identity"] = controller_identity(policy("tdm-beta"))
    elif conflict == "roles":
        episode["evaluation_role_by_global_slot"] = ("not_applicable",) * 10
    else:
        episode["public_agent_id_by_global_slot"] = ("only-one",)
    with pytest.raises(ValueError):
        build_recording_context(
            make_standard_team_deathmatch_config(
                map_id=0,
                max_steps=2,
                team_a_roster=("mage",) * 5,
                team_b_roster=("mage",) * 5,
            ),
            run_id="r",
            phase="evaluation",
            pass_id="p",
            episode=episode,
            policies={"team_a": team.name, "team_b": team.name},
            details={
                "seed": 0,
                "rng_protocol": "episode-fold-in-v1",
                "policies": [descriptor, descriptor],
                "code_revision": CodeRevisionV2(package_version="0.0.0"),
                "runtime_provenance": capture_runtime_provenance("0.0.0"),
            },
        )
