"""Recorded provenance distinguishes known facts from absent external history."""

from pathlib import Path

import jax
import pytest

from marl_battlegrounds.evaluation import revision
from marl_battlegrounds.evaluation.actor_projection import (
    NO_SHARED_OBS_ACTOR_PROJECTION_V2,
)
from marl_battlegrounds.evaluation.models import (
    AggregationKeyV1,
    AssignedPolicySlotV2,
    CodeRevisionV2,
)
from marl_battlegrounds.evaluation.recording_context import build_recording_context
from marl_battlegrounds.evaluation.runtime_provenance import capture_runtime_provenance
from marl_battlegrounds.tasks import make_standard_team_deathmatch_config


@pytest.mark.parametrize("map_id", (0, 20, 24))
def test_recorded_map_names_splits_geometry_and_safe_filenames(map_id: int) -> None:
    from marl_battlegrounds.evaluation.catalog import build_resolved_env_config_v1
    from marl_battlegrounds.evaluation.map_identity import approved_map_id, recorded_map
    from marl_battlegrounds.evaluation.replay_io import generated_replay_filename
    from marl_battlegrounds.tasks import list_tdm_maps

    config = make_standard_team_deathmatch_config(
        map_id=map_id, team_a_roster=("priest",), team_b_roster=("mage", "mage")
    )
    context, _ = build_recording_context(
        jax.device_get(config),
        run_id="run-1",
        phase="evaluation",
        pass_id="1",
        episode={"episode_id": 17, "map_id": map_id, "seed_id": 23},
        policies={"team_a": "Researcher's Learned Policy " * 15, "team_b": "BETA"},
        details={
            "seed": 42,
            "code_revision": CodeRevisionV2(package_version="0.0.0").model_dump(
                mode="json"
            ),
            "runtime_provenance": capture_runtime_provenance("0.0.0").model_dump(
                mode="json"
            ),
        },
    )
    metadata = recorded_map(context)
    expected = list_tdm_maps()[map_id]
    assert (
        approved_map_id(expected.source.asset_id, expected.source.semantic_digest)
        == map_id
    )
    assert (
        approved_map_id("separate_custom_copy", expected.source.semantic_digest) is None
    )
    assert approved_map_id(expected.source.asset_id, "0" * 64) is None
    assert metadata.map_id == map_id
    assert metadata.technical_name == expected.name and metadata.split == expected.split
    if map_id == 20:
        assert metadata.display_name == "Three Body Problem"
    name = generated_replay_filename(context, "a" * 64, episode_id=17)
    assert name.startswith(f"{expected.name}__episode-17__seed-42__stream-23__a-")
    assert "__b-BETA__" in name
    assert name.endswith(f"{'a' * 64}.marlbg-replay.json")
    assert name.isascii() and len(name.encode("ascii")) <= 255
    assert name == generated_replay_filename(context, "a" * 64, episode_id=17)
    assert name != generated_replay_filename(context, "b" * 64, episode_id=17)
    longest = context.model_copy(
        update={
            "policy_assignments": tuple(
                row.model_copy(update={"policy_id": "/.. /" + "policy" * 30})
                if row.assignment_status == "assigned"
                else row
                for row in context.policy_assignments
            ),
            "seed_protocol": context.seed_protocol.model_copy(
                update={"root_seed": 2**32 - 1, "episode_seed": 2**32 - 1}
            ),
        }
    )
    bounded = generated_replay_filename(longest, "c" * 64, episode_id=2**63 - 1)
    assert bounded.isascii() and len(bounded.encode("ascii")) <= 255
    assert "/" not in bounded and ".." not in bounded
    assert bounded.startswith(expected.name) and bounded.endswith(
        "c" * 64 + ".marlbg-replay.json"
    )
    roundtrip = type(context).model_validate_json(context.model_dump_json())
    assert recorded_map(roundtrip) == metadata

    # An explicitly declared map cannot be attached to another map's geometry.
    wrong = context.model_copy(
        update={
            "resolved_env_config": build_resolved_env_config_v1(
                make_standard_team_deathmatch_config(
                    map_id=1, team_a_roster=("mage",), team_b_roster=("priest",)
                )
            )
        }
    )
    with pytest.raises(ValueError, match="geometry"):
        recorded_map(wrong)
    # Old records with no declaration never acquire a split from resemblance.
    historical = context.model_copy(
        update={
            "aggregation_keys": tuple(
                row
                for row in context.aggregation_keys
                if not row.name.startswith("map_")
            )
        }
    )
    before = historical.model_dump_json()
    unknown = recorded_map(historical)
    assert unknown.map_id is unknown.split is None
    assert unknown.display_name == "Map name unavailable"
    assert historical.model_dump_json() == before
    # Older authored replays already carry the exact source identity in layout,
    # even though their aggregation keys predate map_origin/map_name.
    authored = historical.model_copy(
        update={
            "identity": historical.identity.model_copy(
                update={
                    "layout": historical.identity.layout.model_copy(
                        update={
                            "identifier": expected.source.asset_id,
                            "version": expected.source.revision,
                            "canonical_digest": expected.source.semantic_digest,
                        }
                    )
                }
            )
        }
    )
    authored_bytes = authored.model_dump_json()
    assert recorded_map(authored) == metadata
    assert authored.model_dump_json() == authored_bytes
    for mismatch in (
        {"canonical_digest": "0" * 64},
        {"identifier": "custom-copy-of-the-same-geometry"},
        {"version": expected.source.revision + 1},
    ):
        unverified = authored.model_copy(
            update={
                "identity": authored.identity.model_copy(
                    update={
                        "layout": authored.identity.layout.model_copy(update=mismatch)
                    }
                )
            }
        )
        assert recorded_map(unverified).map_id is None
        assert recorded_map(unverified).split is None
    wrong_geometry = authored.model_copy(
        update={"resolved_env_config": wrong.resolved_env_config}
    )
    with pytest.raises(ValueError, match="geometry"):
        recorded_map(wrong_geometry)
    historical_name = historical.model_copy(
        update={
            "aggregation_keys": (
                *historical.aggregation_keys,
                AggregationKeyV1(name="map_name", value="Researcher's old arena"),
            )
        }
    )
    historical_bytes = historical_name.model_dump_json()
    named = recorded_map(historical_name)
    assert named.display_name == named.technical_name == "Researcher's old arena"
    assert named.map_id is named.split is None
    assert historical_name.model_dump_json() == historical_bytes
    arbitrary_keys = historical_name.model_copy(
        update={
            "aggregation_keys": (
                *historical_name.aggregation_keys,
                AggregationKeyV1(name="map_id", value="old-free-form-identifier"),
                AggregationKeyV1(name="map_split", value="my-own-split"),
            )
        }
    )
    assert recorded_map(arbitrary_keys) == named
    incomplete = context.model_copy(
        update={
            "aggregation_keys": tuple(
                row for row in context.aggregation_keys if row.name != "map_split"
            )
        }
    )
    with pytest.raises(ValueError, match="requires ID, name and split"):
        recorded_map(incomplete)
    conflicting = context.model_copy(
        update={
            "aggregation_keys": tuple(
                row.model_copy(update={"value": "another-map"})
                if row.name == "map_name"
                else row
                for row in context.aggregation_keys
            )
        }
    )
    with pytest.raises(ValueError, match="conflicts"):
        recorded_map(conflicting)
    custom_conflict = context.model_copy(
        update={
            "aggregation_keys": tuple(
                row.model_copy(update={"value": "custom"})
                if row.name == "map_origin"
                else row
                for row in context.aggregation_keys
            )
        }
    )
    with pytest.raises(ValueError, match="custom map metadata"):
        recorded_map(custom_conflict)


@pytest.mark.parametrize("mode", ["shared_obs", "no_shared_obs"])
def test_custom_trainer_unknown_history_and_seeds_stay_absent(mode: str) -> None:
    from marl_battlegrounds.evaluation.map_identity import recorded_map
    from marl_battlegrounds.evaluation.replay_io import generated_replay_filename

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
    assert recorded_map(context).display_name == "Custom Map"
    filename = generated_replay_filename(context, "f" * 64, episode_id=7)
    assert filename.startswith("tdm_custom_map_")
    assert "__episode-7__seed-unknown__stream-unknown__" in filename


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
