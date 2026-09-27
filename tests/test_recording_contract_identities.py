"""Check truthful recording identities, numerical start records and failure guards.

Registration must preserve legacy Policy digests, include System hooks and
adapter templates, avoid calling providers, and keep unknown evidence unknown.
Replay metadata distinguishes whole-System ownership from optional routing.
"""

import json
from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from marl_battlegrounds.evaluation.models import (
    AssignedPolicySlotV2,
    CodeRevisionV2,
)
from marl_battlegrounds.evaluation.policy_execution import (
    System,
    independent_policies,
    policy,
    shared_policy,
)
from marl_battlegrounds.evaluation.recording_context import build_recording_context
from marl_battlegrounds.evaluation.recording_identity import (
    normalize_system_registration,
    policy_description,
    tree_digest,
)
from marl_battlegrounds.evaluation.recording_types import (
    EpisodeStartRecords,
    validate_recording_errors,
)
from marl_battlegrounds.evaluation.replay_io import generated_replay_filename
from marl_battlegrounds.evaluation.runtime_provenance import capture_runtime_provenance
from marl_battlegrounds.tasks import make_standard_team_deathmatch_config


def _never_call(*_args: object) -> object:
    raise AssertionError("registration executed the method")


def _other_hook(*_args: object) -> object:
    raise AssertionError("registration executed another hook")


def test_legacy_policy_description_keeps_frozen_snapshot_digest() -> None:
    variables = {"weight": np.array([1, 2], dtype=np.float32)}
    memory = np.array([7], dtype=np.int32)
    team = replace(policy("tdm-alpha"), variables=variables, initial_carry=memory)
    digest = sha256(str(cast(object, jax.tree.structure(variables))).encode())
    for leaf in jax.tree.leaves(variables):
        header = f"{leaf.dtype.str}:{leaf.shape}".encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(np.ascontiguousarray(leaf).data)
    description = policy_description(team, variables, memory, include_digests=True)
    assert description["variables_digest"] == digest.hexdigest()
    assert description["initial_carry_digest"] == tree_digest(memory)
    assert description["controller_identity"] is not None
    _, registered = normalize_system_registration(description, phase="evaluation")
    assert registered["parameter_status"] == "frozen"
    for key, value in description.items():
        assert registered[key] == value
    _, training = normalize_system_registration(description, phase="training")
    assert training["parameter_status"] == "evolving"
    assert training["variables_frozen"] is False


def test_hooks_order_parameters_and_templates_change_identity() -> None:
    original = System("same", _never_call)
    first, metadata = normalize_system_registration(original, phase="evaluation")
    assert metadata["parameter_status"] == "unknown"
    for changed in (
        replace(original, apply=_other_hook),
        replace(original, init=_other_hook),
        replace(original, reset_memory=_other_hook),
        replace(original, variables=np.array([1], dtype=np.int32)),
    ):
        assert normalize_system_registration(changed, phase="evaluation")[0] != first
    a = replace(policy("random"), name="same", variables=np.array([1]))
    b = replace(a, initial_carry=np.array([2]))
    variants = (
        shared_policy(a),
        shared_policy(b),
        independent_policies((a, b)),
        independent_policies((b, a)),
    )
    ids = {
        normalize_system_registration(item, phase="evaluation")[0] for item in variants
    }
    assert len(ids) == len(variants)
    _, adapter = normalize_system_registration(variants[0], phase="evaluation")
    recorded = cast(list[dict[str, object]], adapter["adapter_policies"])[0]
    assert recorded["variables_digest"] == tree_digest(a.variables)
    assert recorded["initial_carry_digest"] == tree_digest(a.initial_carry)
    assert recorded["variables_frozen"] is False


def test_opaque_provider_and_closures_remain_unknown_without_serialization() -> None:
    class Provider:
        def __call__(self, *_args: object) -> object:
            raise AssertionError("provider called")

        def __array__(self, *_args: object) -> object:
            raise AssertionError("provider array conversion")

        def __repr__(self) -> str:
            raise AssertionError("provider representation serialized")

    provider = Provider()
    _, description = normalize_system_registration(
        System("host", provider, execution="host", variables=provider),
        phase="evaluation",
        frozen=True,
    )
    assert description["variables_digest"] is None
    assert description["parameter_status"] == "unknown"
    hooks = cast(dict[str, dict[str, object]], description["hooks"])
    assert hooks["apply"]["code_digest"] is None
    assert hooks["apply"]["external_state"] == "unknown"


def test_start_record_is_public_numerical_tree_and_errors_reject_all_rows() -> None:
    from marl_battlegrounds.types import EpisodeStartRecords as PublicStarts

    assert PublicStarts is EpisodeStartRecords
    integer = jnp.array([1, 2], dtype=jnp.int32)
    unknown = jnp.full(2, -1, dtype=jnp.int32)
    flags = jnp.zeros(2, dtype=jnp.bool_)
    starts = EpisodeStartRecords(
        integer,
        integer,
        jnp.zeros((2, 8), dtype=jnp.uint32),
        unknown,
        unknown,
        unknown,
        flags,
        flags,
        flags,
    )
    assert len(jax.tree.leaves(starts)) == 9
    for lifecycle, tracking, expected in (
        ([False, True], None, "lifecycle_error"),
        ([False, False], [0, 8], "episode_tracking_error"),
        ([False, False], [4, 0], "episode_tracking_error"),
    ):
        packet = SimpleNamespace(
            lifecycle_error=np.array(lifecycle, dtype=np.bool_),
            episode_tracking_error=np.array(tracking, dtype=np.int32)
            if tracking is not None
            else None,
            valid=np.zeros(2, dtype=np.bool_),
        )
        with pytest.raises(ValueError, match=expected):
            validate_recording_errors(packet)
    validate_recording_errors(
        SimpleNamespace(lifecycle_error=np.array(False), episode_tracking_error=None)
    )
    with pytest.raises(TypeError, match="dtype"):
        validate_recording_errors(SimpleNamespace(lifecycle_error=np.array(1)))


@pytest.mark.parametrize(
    ("name", "filename_label"),
    [
        ("Researcher Run 3 Step 120000", "researcher_run_3_step_12"),
        ("Équipe One", "quipe_one"),
        ("Researcher\nTeam", "researcher_team"),
    ],
)
def test_replay_system_ownership_is_separate_from_unknown_internal_choices(
    name: str,
    filename_label: str,
) -> None:
    system = System(name, _never_call, components=({"name": "expert"},))
    system_id, description = normalize_system_registration(system, phase="evaluation")
    context, _ = build_recording_context(
        make_standard_team_deathmatch_config(
            map_id=0, team_a_roster=("priest",), team_b_roster=("mage",)
        ),
        run_id="raw",
        phase="evaluation",
        pass_id="1",
        episode={"episode_id": 1},
        policies={"team_a": "researcher", "team_b": "researcher"},
        details={
            "systems": {system_id: description},
            "system_ids": {"team_a": system_id, "team_b": system_id},
            "code_revision": CodeRevisionV2(package_version="0.0.0"),
            "runtime_provenance": capture_runtime_provenance("0.0.0"),
        },
    )
    for slot in (0, 5):
        assignment = context.policy_assignments[slot]
        assert isinstance(assignment, AssignedPolicySlotV2)
        assert assignment.policy_kind == "system"
        assert assignment.policy_id == system_id
        assert assignment.lifecycle == "evolving"
    assert context.policy_assignments[1].assignment_status == "not_applicable"
    metadata = {entry.name: entry.value for entry in context.aggregation_keys}
    assert json.loads(metadata["marl_bgs.system_name.team_a"]) == name
    assert json.loads(metadata["marl_bgs.system_name.team_b"]) == name
    filename = generated_replay_filename(context, "a" * 64, episode_id=1)
    assert f"__a_{filename_label}__b_{filename_label}__" in filename
    assert filename.endswith("a" * 64 + ".marlbg-replay.json")
    assert metadata["marl_bgs.parameter_status.team_a"] == "unknown"
    assert metadata["marl_bgs.policy_assignments"] == "policy_assignments.csv"
