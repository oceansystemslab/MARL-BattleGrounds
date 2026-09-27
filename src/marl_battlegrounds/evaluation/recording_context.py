"""Build truthful replay metadata from the run that actually executed.

This host-only boundary restores config types after transfers and joins
recorded source/runtime facts, policy assignments, random-stream identities
and map metadata. Missing training history stays unknown. Supplied scenario
metadata may add details but cannot contradict executed seeds or controllers.
"""

import json
import re
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import TypeAdapter

from marl_battlegrounds.core.types import EnvConfig
from marl_battlegrounds.evaluation.actor_projection import (
    NO_SHARED_OBS_ACTOR_PROJECTION_V4,
    SHARED_OBS_ACTOR_PROJECTION_V3,
)
from marl_battlegrounds.evaluation.catalog import (
    build_evaluation_episode_context_v4,
    build_resolved_env_config_v2,
)
from marl_battlegrounds.evaluation.models import (
    AggregationKeyV1,
    AssignedPolicySlotV2,
    CodeRevisionV1,
    CodeRevisionV2,
    ContentAddressedIdentityV1,
    EvaluationEpisodeContextV4,
    EvaluationEpisodeIdentityV1,
    EvaluationRole,
    EvaluationSeedProtocolV2,
    NotApplicablePolicySlotV1,
    PolicyAssignmentSlotV2,
    VersionedIdentityV1,
    canonical_digest_sha256,
    canonical_json_bytes,
)
from marl_battlegrounds.evaluation.replay import RuntimeProvenanceV1
from marl_battlegrounds.evaluation.revision import discover_code_revision_v2
from marl_battlegrounds.evaluation.runtime_provenance import capture_runtime_provenance

_CODE_REVISION: TypeAdapter[CodeRevisionV1 | CodeRevisionV2] = TypeAdapter(
    CodeRevisionV1 | CodeRevisionV2
)


def recorded_policy_assignment(
    *,
    slot: int,
    role: EvaluationRole,
    phase: str,
    name: object,
    descriptor: dict[str, object],
    system_id: str | None,
    registration: dict[str, object],
) -> AssignedPolicySlotV2:
    """Describe one active slot using the existing recorded method evidence.

    slot is the global 0 to 9 roster row; role names its evaluation role. phase
    marks training assignments as evolving. name and descriptor are fallback
    Policy evidence; a nonempty registration takes precedence. system_id is its
    saved registration ID. Shared/independent adapters select their existing
    component. Unknown weight, training and execution facts remain unknown.
    Inputs are borrowed and not changed. No method is called or initialized.
    """
    if registration:
        descriptor = registration
    policy_kind = "callable"
    policy_id = _identifier(descriptor.get("name", name))
    if registration.get("kind") == "system":
        policy_kind = "system"
        policy_id = str(system_id)
        adapter_policies = registration.get("adapter_policies", [])
        if isinstance(adapter_policies, list) and adapter_policies:
            adapter_policies = cast(list[dict[str, object]], adapter_policies)
            component = 0 if registration.get("adapter_kind") == "shared" else slot % 5
            if component >= len(adapter_policies):
                raise ValueError("recorded adapter does not cover the active roster")
            descriptor = adapter_policies[component]
            policy_kind = "callable"
            policy_id = f"{system_id}:component-{component}"
    checkpoint = descriptor.get("checkpoint")
    checkpoint_digest = (
        checkpoint
        if isinstance(checkpoint, str) and re.fullmatch("[0-9a-f]{64}", checkpoint)
        else None
    )
    frozen = (
        registration.get("parameter_status") == "frozen"
        if registration
        else descriptor.get("variables_frozen") is True
    ) and phase != "training"
    return AssignedPolicySlotV2(
        global_slot=slot,
        evaluation_role=role,
        policy_kind=policy_kind,
        policy_id=policy_id,
        lifecycle="frozen" if frozen else "evolving",
        callable_name=(
            _identifier(descriptor["callable_name"])
            if descriptor.get("callable_name") is not None
            else None
        ),
        policy_content_digest=cast(str | None, descriptor.get("variables_digest")),
        checkpoint_digest=checkpoint_digest,
    )


def capture_recording_provenance(
    *,
    num_envs: int = 1,
    policy_execution_included: bool = True,
) -> dict[str, object]:
    """Capture reusable source and runtime metadata at recording setup.

    Parameters
    ----------
    num_envs : int
        Positive environment count to record, default 1.
    policy_execution_included : bool
        Whether policy work belongs to the declared
        runtime record, default True.

    Returns
    -------
    dict[str, object]
        Dict containing JSON-ready code_revision and runtime_provenance models.

    Raises
    ------
    ValueError
        Source discovery or metadata validation fails.
    OSError
        Source/package files cannot be read.
    RuntimeError
        The selected numerical backend cannot provide runtime data.

    Notes
    -----
        Host-only; may query Git, read package files and initialize JAX devices.
        Reuse the returned facts across a pass rather than repeating discovery
        for each replay. This function writes no artifacts.
    """
    code = discover_code_revision_v2()
    runtime = capture_runtime_provenance(
        code.package_version,
        num_envs=num_envs,
        policy_execution_included=policy_execution_included,
    )
    return {
        "code_revision": code.model_dump(mode="json"),
        "runtime_provenance": runtime.model_dump(mode="json"),
    }


def _identifier(value: object) -> str:
    """Make a nonempty path-free identifier from recorded text without inventing
    meaning.
    """
    identifier = re.sub(r"[^A-Za-z0-9._:/+\-]", "-", str(value))
    return re.sub(r"^[^A-Za-z0-9]+", "", identifier).rstrip("-") or "unknown"


def _content(name: str, value: object) -> ContentAddressedIdentityV1:
    """Build a version-1 identity whose digest covers the supplied canonical content."""
    return ContentAddressedIdentityV1(
        identifier=name,
        version=1,
        canonical_digest=canonical_digest_sha256({"content": value}),
    )


def restore_recording_config(config: EnvConfig) -> EnvConfig:
    """Restore scalar config types after packet or spool transfer.

    Parameters
    ----------
    config : EnvConfig
        One scalar EnvConfig tree with host-readable numerical leaves.
        Batched configs are not scalarized into separate episode configs.

    Returns
    -------
    EnvConfig
        The same tree structure with zero-dimensional leaves converted to Python
        scalars and non-scalar leaves converted to JAX arrays. Inputs are unchanged.

    Raises
    ------
    TypeError
        A leaf cannot be represented as a numerical array.

    Notes
    -----
        Host-only. Reading a device leaf may synchronize; returning vector leaves
        may transfer them to JAX's current device. This is type restoration, not
        physical config validation or a change to spawn-bank order.
    """

    def restore(value: object) -> object:
        """Return Python scalars for rank-zero leaves and JAX arrays for vector
        leaves.
        """
        array = np.asarray(value)
        return array.item() if array.ndim == 0 else jnp.asarray(array)

    return jax.tree.map(restore, config)


def _metadata_identity(
    metadata: dict[str, object],
    name: str,
    fallback: ContentAddressedIdentityV1 | None,
) -> ContentAddressedIdentityV1 | None:
    """Validate an explicit content identity, or use the caller's known fallback."""
    value = metadata.get(name)
    return (
        fallback if value is None else ContentAddressedIdentityV1.model_validate(value)
    )


def _scenario_roles(metadata: dict[str, object], active: np.ndarray) -> tuple[str, ...]:
    """Return ten roles consistent with the configured active slots.

    Without overrides, slot zero is focal, its active teammates are cooperative
    partners, and active opposing slots are adversaries. Inactive slots always
    use not_applicable. Reject incomplete or contradictory explicit role lists.
    """
    supplied = metadata.get("evaluation_role_by_global_slot")
    if supplied is None:
        return tuple(
            "not_applicable"
            if not active[slot]
            else "focal"
            if slot == 0
            else "cooperative_partner"
            if slot < 5
            else "adversarial_opponent"
            for slot in range(10)
        )
    if (
        not isinstance(supplied, (tuple, list))
        or len(cast(tuple[object, ...] | list[object], supplied)) != 10
    ):
        raise ValueError("evaluation_role_by_global_slot must contain ten roles")
    roles = tuple(cast(tuple[object, ...] | list[object], supplied))
    for slot, role in enumerate(roles):
        if (
            role not in ("focal", "cooperative_partner", "adversarial_opponent")
            if active[slot]
            else role != "not_applicable"
        ):
            raise ValueError("evaluation roles must match active fixed slots")
    return cast(tuple[str, ...], roles)


def _map_recording_keys(
    config: EnvConfig,
    episode: dict[str, object],
    details: dict[str, object],
) -> tuple[AggregationKeyV1, ...]:
    """Verify declared source content before preserving recorded map metadata.

    config is the actual scalar episode configuration. Optional source_config_id
    must resolve through details.configurations and match its content hash. A
    declared choice must match the whole source bank exchange with all other
    fields unchanged. Saved map keys are retained for later historical geometry
    verification; they are never silently replaced by the current catalog.
    Missing or inconsistent declarations raise ValueError. This host setup helper
    does not change the actual config, load a policy, or write files.
    """
    from marl_battlegrounds.evaluation.evaluation_conditions import (
        config_record,
        restore_config,
    )
    from marl_battlegrounds.evaluation.map_identity import registered_map_metadata
    from marl_battlegrounds.tasks import spawn_locations_for_source

    declared_digest = episode.get("configuration_digest")
    source_id = episode.get("source_config_id")
    declared_choice = episode.get("spawn_locations")
    source = config
    keys: list[AggregationKeyV1] = []
    if declared_digest is not None and config_record(config)[0] != declared_digest:
        raise ValueError(
            "recorded configuration digest differs from the actual episode"
        )
    if source_id is not None:
        contents = details.get("configurations")
        if not isinstance(source_id, str) or not isinstance(contents, dict):
            raise ValueError(
                "recorded spawn source requires saved configuration content"
            )
        content = cast(dict[str, object], contents).get(source_id)
        if not isinstance(content, dict):
            raise ValueError("recorded spawn source content is missing")
        source = restore_config(cast(dict[str, object], content), validate=False)
        if config_record(source)[0] != source_id:
            raise ValueError("recorded spawn source content differs from its identity")
        matches, choice = jax.device_get(spawn_locations_for_source(config, source))
        if not bool(matches):
            raise ValueError("recorded spawn source differs from actual episode fields")
        actual_choice = int(choice)
        if declared_choice is not None:
            if type(declared_choice) is not int or declared_choice not in (0, 1):
                raise ValueError("recorded spawn locations must be 0, 1 or None")
            if declared_choice != actual_choice:
                raise ValueError(
                    "recorded spawn locations differ from actual episode banks"
                )
            keys.extend(
                (
                    AggregationKeyV1(name="marl_bgs.source_config_id", value=source_id),
                    AggregationKeyV1(
                        name="marl_bgs.spawn_locations", value=str(declared_choice)
                    ),
                )
            )
        elif actual_choice == 1 and episode.get("map_id") is not None:
            raise ValueError(
                "an exchanged registered episode requires its spawn choice"
            )
    elif declared_choice is not None:
        raise ValueError("recorded spawn locations require explicit source content")
    map_id = episode.get("map_id")
    metadata = episode.get("map_metadata")
    if map_id is None:
        if metadata is not None:
            raise ValueError("custom episodes cannot supply registered map metadata")
        keys.append(AggregationKeyV1(name="map_origin", value="custom"))
    else:
        if type(map_id) is not int:
            raise ValueError("recorded map_id must be an integer")
        if metadata is None:
            source_geometry = build_resolved_env_config_v2(source)
            keys.extend(registered_map_metadata(map_id, source_geometry))  # pyright: ignore[reportArgumentType]
        else:
            if not isinstance(metadata, (list, tuple)):
                raise ValueError(
                    "recorded map metadata must contain four identity keys"
                )
            rows = tuple(
                AggregationKeyV1.model_validate(row)
                for row in cast(list[object] | tuple[object, ...], metadata)
            )
            names = {"map_id", "map_name", "map_origin", "map_split"}
            if len(rows) != 4 or {row.name for row in rows} != names:
                raise ValueError(
                    "recorded map metadata must contain four identity keys"
                )
            values = {row.name: row.value for row in rows}
            if values["map_id"] != str(map_id) or values["map_origin"] != "registered":
                raise ValueError(
                    "recorded map metadata conflicts with the episode map ID"
                )
            keys.extend(rows)
    return tuple(keys)


def build_recording_context(
    config: EnvConfig,
    *,
    run_id: str,
    phase: str,
    pass_id: str,
    episode: dict[str, object],
    policies: dict[str, object],
    details: dict[str, object],
) -> tuple[EvaluationEpisodeContextV4, RuntimeProvenanceV1]:
    """Build replay context using only known execution and source facts.

    Parameters
    ----------
    config : EnvConfig
        Exact scalar episode EnvConfig, possibly transferred to host arrays.
    run_id : str
        Run identity used to form stable recording identifiers.
    phase : str
        Recorded phase label. Training is evolving. Other phase names do not
        establish that parameters are frozen.
    pass_id : str
        Recorded pass identity within that phase.
    episode : dict[str, object]
        Episode metadata. Requires episode_id; may include seed_id,
        expected_horizon, map_id, authored layout/start identities, ten actor
        IDs/roles and paired-comparison metadata. Source/config content IDs and
        spawn choices are checked against the actual configuration. Saved map
        metadata may name a retained historical map revision. An optional
        scenario_qualification_key is a descriptive witness correlation string;
        it is retained as an aggregation key and never declares a spawn pair.
    policies : dict[str, object]
        Team names or labels under team_a and team_b, used when detailed
        policy descriptors are absent.
    details : dict[str, object]
        Pass metadata, including policy descriptors, optional system_ids and
        systems registration tables, seed/RNG protocol, and captured
        code_revision/runtime_provenance. Missing source
        or runtime data triggers one discovery call for this context. Registered
        display names are retained as JSON strings in system_name aggregation keys
        so Unicode names fit the existing ASCII metadata contract; IDs stay separate.

    Returns
    -------
    tuple[EvaluationEpisodeContextV4, RuntimeProvenanceV1]
        (EvaluationEpisodeContextV4, RuntimeProvenanceV1). The context records exact
        resolved conditions (resolved config V2, including the Team Deathmatch
        Red Zone depth), actor assignments, information contract, known seed
        coordinates and verified map keys. Unknown training/checkpoint facts remain
        absent rather than being guessed.

    Raises
    ------
    KeyError
        episode lacks its required episode_id.
    ValueError
        Roles, seeds, controller identity, actor IDs, map geometry or
        supplied metadata contradict the executed conditions.
    TypeError
        Config or metadata types violate their validated model.
    OSError
        Required source metadata cannot be read.

    Notes
    -----
        Host-only: restores config types and may read runtime/source provenance.
        Input dictionaries are not edited. The default execution information mode
        is shared_obs; an explicit no_shared_obs mode selects that declared actor
        projection. Seed overrides may add facts but must retain the actual root,
        episode identity and RNG protocol. This function does not execute a policy,
        generate a trajectory or write a replay.
    """

    config = restore_recording_config(config)
    if "code_revision" not in details or "runtime_provenance" not in details:
        details = {**capture_recording_provenance(), **details}
    code_payload = details["code_revision"]
    code = (
        code_payload
        if isinstance(code_payload, (CodeRevisionV1, CodeRevisionV2))
        else _CODE_REVISION.validate_python(code_payload)
    )
    runtime_payload = details["runtime_provenance"]
    runtime = (
        runtime_payload
        if isinstance(runtime_payload, RuntimeProvenanceV1)
        else RuntimeProvenanceV1.model_validate_json(json.dumps(runtime_payload))
    )
    resolved = build_resolved_env_config_v2(config)
    descriptor_rows = details.get("policies", [])
    descriptors = (
        cast(list[dict[str, object]], descriptor_rows)
        if isinstance(descriptor_rows, list)
        else []
    )
    active = np.asarray(config.agent_profile.active_mask)
    roles = _scenario_roles(episode, active)
    registered = details.get("systems", {})
    registered = (
        cast(dict[str, dict[str, object]], registered)
        if isinstance(registered, dict)
        else {}
    )
    system_ids = episode.get("system_ids", details.get("system_ids", {}))
    system_ids = (
        cast(dict[str, str], system_ids) if isinstance(system_ids, dict) else {}
    )
    assignments: list[PolicyAssignmentSlotV2] = []
    for slot in range(10):
        if not active[slot]:
            assignments.append(NotApplicablePolicySlotV1(global_slot=slot))
            continue
        team = 0 if slot < 5 else 1
        team_name = "team_a" if team == 0 else "team_b"
        name = policies.get(team_name, "unknown")
        descriptor = descriptors[team] if team < len(descriptors) else {}
        system_id = system_ids.get(team_name)
        registration = (
            registered.get(system_id, {}) if isinstance(system_id, str) else {}
        )
        assignments.append(
            recorded_policy_assignment(
                slot=slot,
                role=cast(EvaluationRole, roles[slot]),
                phase=phase,
                name=name,
                descriptor=descriptor,
                system_id=system_id,
                registration=registration,
            )
        )
    episode_number = int(cast(int, episode["episode_id"]))
    identity = (
        f"{_identifier(run_id)}:{_identifier(phase)}:{_identifier(pass_id)}:"
        f"episode-{episode_number}"
    )
    protocol = str(details.get("rng_protocol", "caller-unspecified"))
    seed = details.get("seed")
    seed_id = episode.get("seed_id")
    seeds = EvaluationSeedProtocolV2(
        seed_protocol=VersionedIdentityV1(identifier=_identifier(protocol), version=1),
        root_seed=int(seed) if isinstance(seed, int) else None,
        episode_seed=int(seed_id) if isinstance(seed_id, int) else None,
    )
    seed_override = episode.get("seed_protocol")
    if seed_override is not None:
        supplemented = EvaluationSeedProtocolV2.model_validate(seed_override)
        if (
            supplemented.root_seed != seeds.root_seed
            or supplemented.episode_seed != seeds.episode_seed
            or supplemented.seed_protocol != seeds.seed_protocol
        ):
            raise ValueError(
                "scenario seeds must match the executed RNG protocol and coordinates"
            )
        seeds = supplemented
    horizon = int(cast(int, episode.get("expected_horizon", config.max_steps)))
    mode = details.get("execution_information_mode", "shared_obs")
    if mode not in ("shared_obs", "no_shared_obs"):
        raise ValueError(
            "execution_information_mode must name an available actor interface"
        )
    aggregation_keys = [
        AggregationKeyV1(name="pass_id", value=pass_id),
        AggregationKeyV1(name="phase", value=phase),
    ]
    qualification_key = episode.get("scenario_qualification_key")
    if qualification_key is not None:
        aggregation_keys.append(
            AggregationKeyV1(
                name="scenario_qualification_key", value=cast(str, qualification_key)
            )
        )
    if system_ids:
        aggregation_keys.extend(
            (
                AggregationKeyV1(name="marl_bgs.run_id", value=run_id),
                AggregationKeyV1(
                    name="marl_bgs.policy_assignments", value="policy_assignments.csv"
                ),
            )
        )
        for team_name in ("team_a", "team_b"):
            system_id = system_ids.get(team_name)
            if not isinstance(system_id, str) or system_id not in registered:
                raise ValueError(
                    "recorded system IDs must resolve to registered systems"
                )
            registration = registered[system_id]
            aggregation_keys.extend(
                (
                    AggregationKeyV1(
                        name=f"marl_bgs.system_id.{team_name}", value=system_id
                    ),
                    AggregationKeyV1(
                        name=f"marl_bgs.system_name.{team_name}",
                        value=json.dumps(
                            registration.get(
                                "name", policies.get(team_name, "Unknown system")
                            ),
                            ensure_ascii=True,
                        ),
                    ),
                    AggregationKeyV1(
                        name=f"marl_bgs.parameter_status.{team_name}",
                        value=str(registration.get("parameter_status", "unknown")),
                    ),
                )
            )
    aggregation_keys.extend(_map_recording_keys(config, episode, details))
    for team_index, team in enumerate(("team_a", "team_b")):
        descriptor = descriptors[team_index] if team_index < len(descriptors) else {}
        name = f"{team}_controller_identity"
        supplied = episode.get(name)
        if supplied is not None:
            identity_metadata = ContentAddressedIdentityV1.model_validate(supplied)
            actual = descriptor.get("controller_identity")
            if (
                actual is None
                or ContentAddressedIdentityV1.model_validate(actual)
                != identity_metadata
            ):
                raise ValueError(
                    "scenario controller identity must match the executed controller"
                )
            aggregation_keys.append(
                AggregationKeyV1(
                    name=name,
                    value=canonical_json_bytes(identity_metadata).decode("ascii"),
                )
            )
    public_ids = episode.get(
        "public_agent_id_by_global_slot", tuple(f"agent-{slot}" for slot in range(10))
    )
    if (
        not isinstance(public_ids, (list, tuple))
        or len(cast(tuple[object, ...] | list[object], public_ids)) != 10
        or any(
            not isinstance(value, str)
            for value in cast(tuple[object, ...] | list[object], public_ids)
        )
    ):
        raise ValueError("public_agent_id_by_global_slot must contain ten string IDs")
    context = build_evaluation_episode_context_v4(
        identity=EvaluationEpisodeIdentityV1(
            run_id=_identifier(run_id),
            evaluation_id=f"{_identifier(phase)}:{_identifier(pass_id)}",
            matchup_id=f"{_identifier(run_id)}:matchup",
            match_id=identity,
            episode_id=identity,
            paired_comparison_key=cast(
                str | None, episode.get("paired_comparison_key")
            ),
            evaluation_suite=_content(
                "python-evaluation", {"phase": phase, "pass_id": pass_id}
            ),
            experiment_manifest=_content("run-manifest", details),
            task=_content(
                "tdm",
                {
                    "task_mode": int(config.task_mode),
                    "score_threshold": int(config.team_deathmatch_score_threshold),
                },
            ),
            layout=cast(
                ContentAddressedIdentityV1,
                _metadata_identity(
                    episode,
                    "layout_identity",
                    _content("resolved-layout", resolved.model_dump(mode="json")),
                ),
            ),
            scenario=_metadata_identity(
                episode,
                "scenario_identity",
                _content("authored-start", episode["initial_state_digest"])
                if episode.get("initial_state_digest") is not None
                else None,
            ),
        ),
        aggregation_keys=tuple(sorted(aggregation_keys, key=lambda row: row.name)),
        expected_horizon=horizon,
        config=config,
        public_agent_id_by_global_slot=tuple(
            cast(list[str] | tuple[str, ...], public_ids)
        ),
        policy_assignments=tuple(assignments),
        seed_protocol=seeds,
        capture_profile="debug",
        execution_information_mode=mode,
        actor_projection=(
            SHARED_OBS_ACTOR_PROJECTION_V3
            if mode == "shared_obs"
            else NO_SHARED_OBS_ACTOR_PROJECTION_V4
        ),
        critic_information_regime=VersionedIdentityV1(
            identifier="not_applicable", version=1
        ),
        canonical_reward_mode=VersionedIdentityV1(identifier="core-tdm", version=1),
        shaping_configuration=_content("none", {"shaping": None}),
        code_revision=code,
        scenario_name=cast(str | None, episode.get("scenario_name")),
    )
    from marl_battlegrounds.evaluation.map_identity import recorded_map

    recorded_map(context)
    return context, runtime
