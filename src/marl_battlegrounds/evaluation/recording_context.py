"""Truthful replay context from one recorded execution, without trainer guesses."""

import json
import re
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import TypeAdapter

from marl_battlegrounds.core.types import EnvConfig
from marl_battlegrounds.evaluation.actor_projection import (
    NO_SHARED_OBS_ACTOR_PROJECTION_V2,
    SHARED_OBS_ACTOR_PROJECTION_V1,
)
from marl_battlegrounds.evaluation.catalog import (
    build_evaluation_episode_context_v2,
    build_resolved_env_config_v1,
)
from marl_battlegrounds.evaluation.models import (
    AggregationKeyV1,
    AssignedPolicySlotV2,
    CodeRevisionV1,
    CodeRevisionV2,
    ContentAddressedIdentityV1,
    EvaluationEpisodeContextV2,
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


def capture_recording_provenance(
    *,
    num_envs: int = 1,
    policy_execution_included: bool = True,
) -> dict[str, object]:
    """Discover source/runtime facts once; callers reuse them for selected replays."""
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
    identifier = re.sub(r"[^A-Za-z0-9._:/+\-]", "-", str(value))
    return re.sub(r"^[^A-Za-z0-9]+", "", identifier).rstrip("-") or "unknown"


def _content(name: str, value: object) -> ContentAddressedIdentityV1:
    return ContentAddressedIdentityV1(
        identifier=name,
        version=1,
        canonical_digest=canonical_digest_sha256({"content": value}),
    )


def restore_recording_config(config: EnvConfig) -> EnvConfig:
    """Restore host scalar/JAX array types after a captured device or spool transfer."""

    def restore(value: object) -> object:
        array = np.asarray(value)
        return array.item() if array.ndim == 0 else jnp.asarray(array)

    return jax.tree.map(restore, config)


def _metadata_identity(
    metadata: dict[str, object],
    name: str,
    fallback: ContentAddressedIdentityV1 | None,
) -> ContentAddressedIdentityV1 | None:
    value = metadata.get(name)
    return (
        fallback if value is None else ContentAddressedIdentityV1.model_validate(value)
    )


def _scenario_roles(metadata: dict[str, object], active: np.ndarray) -> tuple[str, ...]:
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


def build_recording_context(
    config: EnvConfig,
    *,
    run_id: str,
    phase: str,
    pass_id: str,
    episode: dict[str, object],
    policies: dict[str, object],
    details: dict[str, object],
) -> tuple[EvaluationEpisodeContextV2, RuntimeProvenanceV1]:
    """Build context from known facts; custom trainer seeds/history stay unknown."""

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
    resolved = build_resolved_env_config_v1(config)
    descriptor_rows = details.get("policies", [])
    descriptors = (
        cast(list[dict[str, object]], descriptor_rows)
        if isinstance(descriptor_rows, list)
        else []
    )
    active = np.asarray(config.agent_profile.active_mask)
    roles = _scenario_roles(episode, active)
    assignments: list[PolicyAssignmentSlotV2] = []
    for slot in range(10):
        if not active[slot]:
            assignments.append(NotApplicablePolicySlotV1(global_slot=slot))
            continue
        team = 0 if slot < 5 else 1
        name = policies.get("team_a" if team == 0 else "team_b", "unknown")
        descriptor = descriptors[team] if team < len(descriptors) else {}
        policy_name = str(descriptor.get("name", name))
        checkpoint = descriptor.get("checkpoint")
        checkpoint_digest = (
            checkpoint
            if isinstance(checkpoint, str) and re.fullmatch("[0-9a-f]{64}", checkpoint)
            else None
        )
        assignments.append(
            AssignedPolicySlotV2(
                global_slot=slot,
                evaluation_role=cast(EvaluationRole, roles[slot]),
                policy_kind="callable",
                policy_id=_identifier(policy_name),
                lifecycle="frozen"
                if descriptor.get("variables_frozen") or phase != "training"
                else "evolving",
                callable_name=(
                    _identifier(descriptor["callable_name"])
                    if descriptor.get("callable_name") is not None
                    else None
                ),
                policy_content_digest=cast(
                    str | None, descriptor.get("variables_digest")
                ),
                checkpoint_digest=checkpoint_digest,
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
    if episode.get("map_id") is not None:
        from marl_battlegrounds.evaluation.map_identity import registered_map_metadata

        map_id = episode["map_id"]
        if type(map_id) is not int:
            raise ValueError("recorded map_id must be an integer")
        aggregation_keys.extend(registered_map_metadata(map_id, resolved))  # pyright: ignore[reportArgumentType]
    else:
        aggregation_keys.append(AggregationKeyV1(name="map_origin", value="custom"))
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
    context = build_evaluation_episode_context_v2(
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
            SHARED_OBS_ACTOR_PROJECTION_V1
            if mode == "shared_obs"
            else NO_SHARED_OBS_ACTOR_PROJECTION_V2
        ),
        critic_information_regime=VersionedIdentityV1(
            identifier="not_applicable", version=1
        ),
        canonical_reward_mode=VersionedIdentityV1(identifier="core-tdm", version=1),
        shaping_configuration=_content("none", {"shaping": None}),
        code_revision=code,
        scenario_name=cast(str | None, episode.get("scenario_name")),
    )
    return context, runtime
