"""Build durable episode metadata from explicit runner and Core authorities.

Runtime arrays are copied to strict host records with stable global-slot order
and canonical content digests. Call these helpers at setup or recording
boundaries, outside JAX transformations. Reconstruction restores recorded values;
it does not silently substitute current class defaults or reset the simulator.
"""

from __future__ import annotations

from typing import Literal, cast

import jax.numpy as jnp
import numpy as np
from jax import Array

from marl_battlegrounds.core.axis_mappings import (
    GLOBAL_RECIPIENT_SLOT_BY_ACTOR_AND_TARGET_ACTION,
    GLOBAL_SLOT_BY_ACTOR_AND_ALLY_OBSERVATION_ROW,
    GLOBAL_SLOT_BY_ACTOR_AND_ENEMY_OBSERVATION_ROW,
    MOVEMENT_ACTION_NAME_BY_ID,
    TARGET_ACTION_NAME_BY_ID,
    TEAM_A_END,
    TEAM_A_START,
    TEAM_B_END,
    TEAM_B_START,
    UNIT_DIRECTION_VECTOR_BY_MOVEMENT_ACTION,
)
from marl_battlegrounds.core.combat import (
    BASE_MOVEMENT_SPEED_BY_CLASS,
    BASIC_DAMAGE_BY_CLASS,
    BASIC_HEALING_BY_CLASS,
    BASIC_INTERACTION_RADIUS_BY_CLASS,
    BODY_RADIUS_BY_CLASS,
    GLOBAL_SLOW_FLOOR,
    HUNTER_BASIC_SLOW_DURATION_TICKS,
    HUNTER_BASIC_SLOW_MULTIPLIER,
    HUNTER_TRAP_STUN_DURATION_TICKS,
    MAGE_BURST_DAMAGE_DURATION_TICKS,
    MAGE_BURST_DAMAGE_MULTIPLIER,
    MAGE_DAMAGE_AMPLIFICATION_AURA_MULTIPLIER,
    MAGE_DAMAGE_AMPLIFICATION_AURA_MULTIPLIER_CEILING,
    MAGE_DAMAGE_AMPLIFICATION_AURA_RADIUS,
    MAX_HEALTH_BY_CLASS,
    OBSERVATION_RADIUS_BY_CLASS,
    OUT_OF_COMBAT_DELAY_STEPS_BY_CLASS,
    OUT_OF_COMBAT_HEALTH_REGENERATION_FRACTION_PER_STEP_BY_CLASS,
    PRIEST_HEAL_SPEED_FLOOR,
    PRIEST_HEAL_SPEED_FLOOR_DURATION_TICKS,
    ROGUE_POISON_ANTI_HEAL_DURATION_TICKS,
    ROGUE_POISON_ANTI_HEAL_MULTIPLIER,
    ROGUE_POISON_SLOW_DURATION_TICKS,
    ROGUE_POISON_SLOW_MULTIPLIER,
    ROGUE_POISON_STUN_DURATION_TICKS,
    ULTIMATE_COOLDOWN_BY_CLASS,
    ULTIMATE_DAMAGE_BY_CLASS,
    ULTIMATE_HEALING_BY_CLASS,
    ULTIMATE_INTERACTION_RADIUS_BY_CLASS,
    WARRIOR_CHARGE_SLOW_DURATION_TICKS,
    WARRIOR_CHARGE_SLOW_MULTIPLIER,
    WARRIOR_CHARGE_STUN_DURATION_TICKS,
    WARRIOR_DAMAGE_MITIGATION_AURA_MULTIPLIER,
    WARRIOR_DAMAGE_MITIGATION_AURA_MULTIPLIER_FLOOR,
    WARRIOR_DAMAGE_MITIGATION_AURA_RADIUS,
)
from marl_battlegrounds.core.config import (
    validate_env_config,
    validate_product_env_config,
    validate_scenario_initial_state,
)
from marl_battlegrounds.core.types import (
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    MAX_OBSTACLE_SLOTS,
    NUM_CLASSES,
    NUM_TEAMS,
    OBSTACLE_FEATURE_ACTIVE,
    OBSTACLE_FEATURE_HEIGHT,
    OBSTACLE_FEATURE_RADIUS,
    OBSTACLE_FEATURE_THETA,
    OBSTACLE_FEATURE_TYPE,
    OBSTACLE_FEATURE_WIDTH,
    OBSTACLE_FEATURE_X,
    OBSTACLE_FEATURE_Y,
    OBSTACLE_FEATURES,
    EnvConfig,
    ResolvedAgentProfile,
)
from marl_battlegrounds.evaluation.actor_projection import (
    SHARED_OBS_ACTOR_PROJECTION_V1,
    SHARED_OBS_ACTOR_PROJECTION_V2,
)
from marl_battlegrounds.evaluation.models import (
    CATALOG_SCHEMA_ID,
    CATALOG_SCHEMA_VERSION,
    REQUIRED_SCHEMA_BINDINGS_V1,
    REQUIRED_SCHEMA_BINDINGS_V2,
    REQUIRED_SCHEMA_BINDINGS_V3,
    RESOLVED_ENV_CONFIG_SCHEMA_ID,
    RESOLVED_ENV_CONFIG_SCHEMA_VERSION,
    AggregationKeyV1,
    AuraMechanicV1,
    CaptureProfile,
    ClassMechanicsV1,
    CodeRevisionV1,
    CodeRevisionV2,
    ContentAddressedIdentityV1,
    EvaluationEpisodeContext,
    EvaluationEpisodeContextV1,
    EvaluationEpisodeContextV2,
    EvaluationEpisodeContextV3,
    EvaluationEpisodeIdentityV1,
    EvaluationFrame,
    EvaluationFrameV1,
    EvaluationFrameV2,
    EvaluationSeedProtocolV1,
    EvaluationSeedProtocolV2,
    ExecutionInformationMode,
    PolicyAssignmentSlotV1,
    PolicyAssignmentSlotV2,
    ResolvedEnvConfigV1,
    ResolvedObstacleV1,
    ResolvedSlotMechanicsV1,
    RosterSlotV1,
    SchemaVersionEntryV1,
    SchemaVersionEntryV2,
    SchemaVersionEntryV3,
    StaticMechanicsCatalogV1,
    StatusMechanicV1,
    VersionedIdentityV1,
    canonical_digest_sha256,
)


def _python_float_tuple(values: object) -> tuple[float, ...]:
    """Copy a one-dimensional catalog array to exact float32-valued Python floats.

    This host conversion can synchronize a device array. The float32 narrowing is
    intentional because the source mechanics tables use float32.
    """
    host = np.asarray(values, dtype=np.float32)
    return tuple(float(value) for value in host)


def _class_mechanics() -> tuple[ClassMechanicsV1, ...]:
    """Project current Core class tables in class-ID order, including neutral class 0.

    Record health, world-unit distances, per-tick movement, targeting modes, cooldown
    ticks, and recovery settings. Return six strict host rows; do not infer values
    from a particular roster or modify the source tables.
    """
    class_names = ("Neutral", "Mage", "Warrior", "Hunter", "Rogue", "Priest")
    basic_target_modes: tuple[Literal["unavailable", "ally", "enemy"], ...] = (
        "unavailable",
        "enemy",
        "enemy",
        "enemy",
        "enemy",
        "ally",
    )
    ultimate_target_modes: tuple[
        Literal["unavailable", "target_none", "ally", "enemy"], ...
    ] = ("unavailable", "target_none", "enemy", "enemy", "enemy", "ally")
    maximum_health = _python_float_tuple(MAX_HEALTH_BY_CLASS)
    body_radius = _python_float_tuple(BODY_RADIUS_BY_CLASS)
    movement_speed = _python_float_tuple(BASE_MOVEMENT_SPEED_BY_CLASS)
    observation_radius = _python_float_tuple(OBSERVATION_RADIUS_BY_CLASS)
    basic_radius = _python_float_tuple(BASIC_INTERACTION_RADIUS_BY_CLASS)
    basic_damage = _python_float_tuple(BASIC_DAMAGE_BY_CLASS)
    basic_healing = _python_float_tuple(BASIC_HEALING_BY_CLASS)
    ultimate_radius = _python_float_tuple(ULTIMATE_INTERACTION_RADIUS_BY_CLASS)
    ultimate_cooldown = tuple(
        int(value) for value in np.asarray(ULTIMATE_COOLDOWN_BY_CLASS)
    )
    ultimate_damage = _python_float_tuple(ULTIMATE_DAMAGE_BY_CLASS)
    ultimate_healing = _python_float_tuple(ULTIMATE_HEALING_BY_CLASS)
    recovery_delay = tuple(
        int(value) for value in np.asarray(OUT_OF_COMBAT_DELAY_STEPS_BY_CLASS)
    )
    recovery_fraction = _python_float_tuple(
        OUT_OF_COMBAT_HEALTH_REGENERATION_FRACTION_PER_STEP_BY_CLASS
    )
    return tuple(
        ClassMechanicsV1(
            class_id=class_id,
            class_name=class_names[class_id],
            maximum_health=maximum_health[class_id],
            body_radius=body_radius[class_id],
            base_movement_speed=movement_speed[class_id],
            observation_radius=observation_radius[class_id],
            basic_target_mode=basic_target_modes[class_id],
            basic_interaction_radius=basic_radius[class_id],
            basic_raw_damage=basic_damage[class_id],
            basic_raw_healing=basic_healing[class_id],
            ultimate_target_mode=ultimate_target_modes[class_id],
            ultimate_interaction_radius=ultimate_radius[class_id],
            ultimate_cooldown_steps=ultimate_cooldown[class_id],
            ultimate_raw_damage=ultimate_damage[class_id],
            ultimate_raw_healing=ultimate_healing[class_id],
            out_of_combat_delay_steps=recovery_delay[class_id],
            out_of_combat_health_regeneration_fraction_per_step=(
                recovery_fraction[class_id]
            ),
        )
        for class_id in range(NUM_CLASSES)
    )


def _status_channels() -> tuple[StatusMechanicV1, ...]:
    """Describe the nine recorded status channels in their fixed wire order.

    Use current Core durations and multipliers, including Trap's damage-break rule.
    Return validated host records; this catalog describes rules without applying them.
    """
    rows: tuple[dict[str, object], ...] = (
        {
            "status_channel_id": 0,
            "status_id": "warrior_charge_slow",
            "family": "slow",
            "source_class_id": 2,
            "source_action_component": "ultimate",
            "duration_steps": int(WARRIOR_CHARGE_SLOW_DURATION_TICKS),
            "magnitude_kind": "movement_multiplier",
            "magnitude": float(WARRIOR_CHARGE_SLOW_MULTIPLIER),
            "breaks_on_positive_damage": False,
        },
        {
            "status_channel_id": 1,
            "status_id": "hunter_basic_slow",
            "family": "slow",
            "source_class_id": 3,
            "source_action_component": "basic",
            "duration_steps": int(HUNTER_BASIC_SLOW_DURATION_TICKS),
            "magnitude_kind": "movement_multiplier",
            "magnitude": float(HUNTER_BASIC_SLOW_MULTIPLIER),
            "breaks_on_positive_damage": False,
        },
        {
            "status_channel_id": 2,
            "status_id": "rogue_poison_slow",
            "family": "slow",
            "source_class_id": 4,
            "source_action_component": "ultimate",
            "duration_steps": int(ROGUE_POISON_SLOW_DURATION_TICKS),
            "magnitude_kind": "movement_multiplier",
            "magnitude": float(ROGUE_POISON_SLOW_MULTIPLIER),
            "breaks_on_positive_damage": False,
        },
        {
            "status_channel_id": 3,
            "status_id": "warrior_charge_stun",
            "family": "stun",
            "source_class_id": 2,
            "source_action_component": "ultimate",
            "duration_steps": int(WARRIOR_CHARGE_STUN_DURATION_TICKS),
            "magnitude_kind": "none",
            "magnitude": None,
            "breaks_on_positive_damage": False,
        },
        {
            "status_channel_id": 4,
            "status_id": "hunter_trap_stun",
            "family": "stun",
            "source_class_id": 3,
            "source_action_component": "ultimate",
            "duration_steps": int(HUNTER_TRAP_STUN_DURATION_TICKS),
            "magnitude_kind": "none",
            "magnitude": None,
            "breaks_on_positive_damage": True,
        },
        {
            "status_channel_id": 5,
            "status_id": "rogue_poison_stun",
            "family": "stun",
            "source_class_id": 4,
            "source_action_component": "ultimate",
            "duration_steps": int(ROGUE_POISON_STUN_DURATION_TICKS),
            "magnitude_kind": "none",
            "magnitude": None,
            "breaks_on_positive_damage": False,
        },
        {
            "status_channel_id": 6,
            "status_id": "rogue_poison_anti_heal",
            "family": "anti_heal",
            "source_class_id": 4,
            "source_action_component": "ultimate",
            "duration_steps": int(ROGUE_POISON_ANTI_HEAL_DURATION_TICKS),
            "magnitude_kind": "healing_multiplier",
            "magnitude": float(ROGUE_POISON_ANTI_HEAL_MULTIPLIER),
            "breaks_on_positive_damage": False,
        },
        {
            "status_channel_id": 7,
            "status_id": "mage_burst_damage_amplification",
            "family": "damage_amplification",
            "source_class_id": 1,
            "source_action_component": "ultimate",
            "duration_steps": int(MAGE_BURST_DAMAGE_DURATION_TICKS),
            "magnitude_kind": "damage_multiplier",
            "magnitude": float(MAGE_BURST_DAMAGE_MULTIPLIER),
            "breaks_on_positive_damage": False,
        },
        {
            "status_channel_id": 8,
            "status_id": "priest_blessing_of_freedom_movement_floor",
            "family": "movement_floor",
            "source_class_id": 5,
            "source_action_component": "basic",
            "duration_steps": int(PRIEST_HEAL_SPEED_FLOOR_DURATION_TICKS),
            "magnitude_kind": "movement_floor",
            "magnitude": float(PRIEST_HEAL_SPEED_FLOOR),
            "breaks_on_positive_damage": False,
        },
    )
    return tuple(StatusMechanicV1.model_validate(row) for row in rows)


def _aura_mechanics() -> tuple[AuraMechanicV1, AuraMechanicV1]:
    """Describe Mage amplification and Warrior mitigation from current Core constants.

    Return those two host records in that order, including world-unit radius,
    per-emitter multiplier, and the ceiling or floor applied to combined effects.
    """
    return (
        AuraMechanicV1(
            aura_id="mage_damage_amplification",
            emitter_class_id=1,
            radius=float(MAGE_DAMAGE_AMPLIFICATION_AURA_RADIUS),
            per_emitter_multiplier=float(MAGE_DAMAGE_AMPLIFICATION_AURA_MULTIPLIER),
            clamp_kind="ceiling",
            clamp_value=float(MAGE_DAMAGE_AMPLIFICATION_AURA_MULTIPLIER_CEILING),
        ),
        AuraMechanicV1(
            aura_id="warrior_damage_mitigation",
            emitter_class_id=2,
            radius=float(WARRIOR_DAMAGE_MITIGATION_AURA_RADIUS),
            per_emitter_multiplier=float(WARRIOR_DAMAGE_MITIGATION_AURA_MULTIPLIER),
            clamp_kind="floor",
            clamp_value=float(WARRIOR_DAMAGE_MITIGATION_AURA_MULTIPLIER_FLOOR),
        ),
    )


def build_static_mechanics_catalog_v1() -> StaticMechanicsCatalogV1:
    """Capture the current mechanics and action vocabularies in one durable catalog.

    Returns
    -------
    StaticMechanicsCatalogV1
        Validated V1 catalog with fixed axes, six class rows including neutral,
        nine status rows, two aura rows, action mappings, units, and a canonical
        SHA-256 digest over all other fields.

    Notes
    -----
    Reads current Core constants and copies arrays to the host. Use this during
    recording setup, outside JIT. It performs no simulation and does not assert
    that a historical catalog matches the current code.
    """
    payload: dict[str, object] = {
        "schema_id": CATALOG_SCHEMA_ID,
        "schema_version": CATALOG_SCHEMA_VERSION,
        "maximum_agent_slots": MAX_AGENT_SLOTS,
        "maximum_agents_per_team": MAX_AGENTS_PER_TEAM,
        "number_of_teams": NUM_TEAMS,
        "number_of_movement_actions": len(MOVEMENT_ACTION_NAME_BY_ID),
        "number_of_target_actions": len(TARGET_ACTION_NAME_BY_ID),
        "number_of_ultimate_actions": 2,
        "team_global_slot_half_open_ranges": (
            (TEAM_A_START, TEAM_A_END),
            (TEAM_B_START, TEAM_B_END),
        ),
        "class_name_by_id": (
            "Neutral",
            "Mage",
            "Warrior",
            "Hunter",
            "Rogue",
            "Priest",
        ),
        "team_name_by_id": ("No Team", "Team A", "Team B"),
        "movement_action_name_by_id": MOVEMENT_ACTION_NAME_BY_ID,
        "target_action_name_by_id": TARGET_ACTION_NAME_BY_ID,
        "use_ultimate_action_name_by_id": (
            "Do Not Use Ultimate",
            "Use Ultimate",
        ),
        "ultimate_target_mode_name_by_id": (
            "Unavailable",
            "Target None",
            "Ally",
            "Enemy",
        ),
        "spawn_lifecycle_team_axis_name_by_id": ("Own Team", "Opponent Team"),
        "class_mechanics": _class_mechanics(),
        "status_channels": _status_channels(),
        "aura_mechanics": _aura_mechanics(),
        "global_slow_floor": float(GLOBAL_SLOW_FLOOR),
        "global_recipient_slot_by_actor_and_target_action": (
            GLOBAL_RECIPIENT_SLOT_BY_ACTOR_AND_TARGET_ACTION
        ),
        "global_slot_by_actor_and_ally_observation_row": (
            GLOBAL_SLOT_BY_ACTOR_AND_ALLY_OBSERVATION_ROW
        ),
        "global_slot_by_actor_and_enemy_observation_row": (
            GLOBAL_SLOT_BY_ACTOR_AND_ENEMY_OBSERVATION_ROW
        ),
        "unit_direction_vector_by_movement_action": (
            UNIT_DIRECTION_VECTOR_BY_MOVEMENT_ACTION
        ),
        "health_unit": "hit_points",
        "spatial_unit": "world_units",
        "duration_unit": "transition_ticks",
        "health_effect_stage_name_by_id": (
            "raw_source",
            "source_modified_gross",
            "recipient_modified_gross",
            "combat_resolution_health",
            "realized_net_health_change",
            "actual_regeneration",
        ),
    }
    payload["canonical_digest_sha256"] = canonical_digest_sha256(payload)
    return StaticMechanicsCatalogV1.model_validate(payload)


def _resolved_obstacles(config: EnvConfig) -> tuple[ResolvedObstacleV1, ...]:
    """Copy all fixed obstacle slots from one scalar config to ordered host rows.

    Preserve inactive rows as well as active geometry. Coordinates and sizes use
    world units; theta uses the recorded angular value. The caller validates config.
    """
    host = np.asarray(config.obstacles, dtype=np.float32)
    return tuple(
        ResolvedObstacleV1(
            obstacle_slot=slot,
            obstacle_type_id=int(host[slot, OBSTACLE_FEATURE_TYPE]),
            x=float(host[slot, OBSTACLE_FEATURE_X]),
            y=float(host[slot, OBSTACLE_FEATURE_Y]),
            radius=float(host[slot, OBSTACLE_FEATURE_RADIUS]),
            width=float(host[slot, OBSTACLE_FEATURE_WIDTH]),
            height=float(host[slot, OBSTACLE_FEATURE_HEIGHT]),
            theta=float(host[slot, OBSTACLE_FEATURE_THETA]),
            is_active=bool(host[slot, OBSTACLE_FEATURE_ACTIVE]),
        )
        for slot in range(MAX_OBSTACLE_SLOTS)
    )


def _resolved_slot_mechanics(
    config: EnvConfig,
) -> tuple[ResolvedSlotMechanicsV1, ...]:
    """Copy resolved profile values for all ten global slots into host records.

    Use config-specific values, including overrides and inactive padding, rather
    than looking up class defaults. The caller validates the scalar config.
    """
    profile = config.agent_profile
    radii = np.asarray(profile.agent_radii, dtype=np.float32)
    speeds = np.asarray(profile.base_movement_speeds, dtype=np.float32)
    observation_radii = np.asarray(profile.observation_radii, dtype=np.float32)
    basic_radii = np.asarray(profile.basic_interaction_radii, dtype=np.float32)
    ultimate_radii = np.asarray(profile.ultimate_interaction_radii, dtype=np.float32)
    maximum_health = np.asarray(profile.max_health, dtype=np.float32)
    recovery_delay = np.asarray(profile.out_of_combat_delay_steps, dtype=np.int32)
    recovery_fraction = np.asarray(
        profile.out_of_combat_health_regen_fraction_per_step,
        dtype=np.float32,
    )
    return tuple(
        ResolvedSlotMechanicsV1(
            global_slot=slot,
            body_radius=float(radii[slot]),
            base_movement_speed=float(speeds[slot]),
            observation_radius=float(observation_radii[slot]),
            basic_interaction_radius=float(basic_radii[slot]),
            ultimate_interaction_radius=float(ultimate_radii[slot]),
            maximum_health=float(maximum_health[slot]),
            out_of_combat_delay_steps=int(recovery_delay[slot]),
            out_of_combat_health_regeneration_fraction_per_step=float(
                recovery_fraction[slot]
            ),
        )
        for slot in range(MAX_AGENT_SLOTS)
    )


def build_resolved_env_config_v1(config: EnvConfig) -> ResolvedEnvConfigV1:
    """Validate one scalar runtime config and record its exact resolved values.

    Parameters
    ----------
    config : EnvConfig
        Unbatched EnvConfig accepted by Core's host validator: Python scalar
        settings and fixed-shape JAX profile, obstacle, spawn-pad, and team arrays.

    Returns
    -------
    ResolvedEnvConfigV1
        Strict V1 host record containing geometry, ten resolved slot profiles,
        task rules, spawn settings, and a digest of all other fields.

    Raises
    ------
    TypeError
        A config field has an unsupported host type or array dtype.
    ValueError
        Core configuration validation or strict record validation fails.

    Notes
    -----
    Array conversion can synchronize device data. This is a host boundary,
    not a JIT operation. The record stores resolved mechanics separately from
    roster identity; use build_roster_v1 for class/team membership.
    """
    validate_env_config(config)
    pads = np.asarray(config.team_spawn_pad_positions, dtype=np.float32)
    periods = np.asarray(config.team_respawn_wave_period_step_count, dtype=np.int32)
    payload: dict[str, object] = {
        "schema_id": RESOLVED_ENV_CONFIG_SCHEMA_ID,
        "schema_version": RESOLVED_ENV_CONFIG_SCHEMA_VERSION,
        "task_mode": config.task_mode,
        "team_deathmatch_score_threshold": (config.team_deathmatch_score_threshold),
        "maximum_episode_steps": config.max_steps,
        "map_width": float(config.map_width),
        "map_height": float(config.map_height),
        "obstacle_slots": _resolved_obstacles(config),
        "slot_mechanics": _resolved_slot_mechanics(config),
        "ordinary_movement_distance_scale": float(
            config.ordinary_movement_distance_scale
        ),
        "team_spawn_pad_positions": tuple(
            tuple(
                (float(pads[team, local, 0]), float(pads[team, local, 1]))
                for local in range(MAX_AGENTS_PER_TEAM)
            )
            for team in range(NUM_TEAMS)
        ),
        "spawn_shield_duration_steps": config.spawn_shield_duration_steps,
        "spawn_shield_movement_speed": float(config.spawn_shield_movement_speed),
        "team_respawn_wave_period_steps": tuple(int(value) for value in periods),
    }
    payload["canonical_digest_sha256"] = canonical_digest_sha256(payload)
    return ResolvedEnvConfigV1.model_validate(payload)


def build_roster_v1(
    config: EnvConfig,
    public_agent_id_by_global_slot: tuple[str, ...],
) -> tuple[RosterSlotV1, ...]:
    """Record public names and configured membership for all ten global slots.

    Parameters
    ----------
    config : EnvConfig
        Scalar runtime config with a resolved ten-slot agent profile.
    public_agent_id_by_global_slot : tuple[str, ...]
        Exactly ten nonempty valid public identifiers,
        ordered Team A slots 0-4 then Team B slots 5-9, including inactive slots.

    Returns
    -------
    tuple[RosterSlotV1, ...]
        Ten strict roster rows preserving slot order, team-local slot, public ID,
        configured team, class, and active flag.

    Raises
    ------
    ValueError
        The public-ID tuple does not have ten entries or a row fails
        model validation.

    Notes
    -----
    Copies profile arrays to the host and does not validate the entire config.
    Row construction alone does not check public-ID uniqueness across rows;
    episode-context validation owns that join.
    """
    if len(public_agent_id_by_global_slot) != MAX_AGENT_SLOTS:
        raise ValueError("public_agent_id_by_global_slot must have length 10")
    profile = config.agent_profile
    class_ids = np.asarray(profile.class_ids, dtype=np.int32)
    team_ids = np.asarray(profile.team_ids, dtype=np.int32)
    active = np.asarray(profile.active_mask, dtype=np.bool_)
    return tuple(
        RosterSlotV1(
            global_slot=slot,
            team_local_slot=slot % MAX_AGENTS_PER_TEAM,
            public_agent_id=public_agent_id_by_global_slot[slot],
            configured_team_id=int(team_ids[slot]),
            class_id=int(class_ids[slot]),
            configured_active=bool(active[slot]),
        )
        for slot in range(MAX_AGENT_SLOTS)
    )


def build_evaluation_seed_protocol_v1(
    *,
    seed_protocol: VersionedIdentityV1,
    root_seed: int,
    episode_seed: int,
    layout_seed: int,
    environment_seed: int,
    focal_policy_seed: int,
    evaluation_seed: int,
    cooperative_partner_seed: int | Literal["not_applicable"],
    adversarial_opponent_seed: int | Literal["not_applicable"],
    scenario_seed: int | Literal["not_applicable"],
) -> EvaluationSeedProtocolV1:
    """Record the runner's complete legacy named-seed provenance.

    Parameters
    ----------
    seed_protocol : VersionedIdentityV1
        Identifier and version of the rule that assigns these seeds.
    root_seed : int
        Realized root seed, a nonnegative integer.
    episode_seed : int
        Realized episode seed, a nonnegative integer.
    layout_seed : int
        Realized layout seed, a nonnegative integer.
    environment_seed : int
        Realized simulator seed, a nonnegative integer.
    focal_policy_seed : int
        Realized focal-policy seed, a nonnegative integer.
    evaluation_seed : int
        Realized evaluation seed, a nonnegative integer.
    cooperative_partner_seed : int | Literal['not_applicable']
        Nonnegative seed, or "not_applicable" when that role is absent.
    adversarial_opponent_seed : int | Literal['not_applicable']
        Nonnegative seed, or "not_applicable" when that role is absent.
    scenario_seed : int | Literal['not_applicable']
        Nonnegative seed, or "not_applicable" when no scenario seed applies.

    Returns
    -------
    EvaluationSeedProtocolV1
        Strict V1 seed record. Every argument is required; no seed is generated.

    Raises
    ------
    ValueError
        A seed or identity fails model validation.

    Notes
    -----
    Each numeric seed must fit uint32: 0 through 2**32 - 1. This helper records
    scalar seed provenance, not JAX key arrays. Use the V2
    seed model where facts are unknown; do not invent V1 values for missing data.
    """
    return EvaluationSeedProtocolV1(
        seed_protocol=seed_protocol,
        root_seed=root_seed,
        episode_seed=episode_seed,
        layout_seed=layout_seed,
        environment_seed=environment_seed,
        focal_policy_seed=focal_policy_seed,
        evaluation_seed=evaluation_seed,
        cooperative_partner_seed=cooperative_partner_seed,
        adversarial_opponent_seed=adversarial_opponent_seed,
        scenario_seed=scenario_seed,
    )


def build_code_revision_v1(
    *,
    package_version: str,
    commit_sha: str,
    is_dirty: bool,
    source_tree_digest: str,
    dirty_patch_digest: str | None,
) -> CodeRevisionV1:
    """Record explicitly supplied legacy source and Git identities.

    Parameters
    ----------
    package_version : str
        Nonempty package version identifier.
    commit_sha : str
        Full lowercase hexadecimal Git commit identity.
    is_dirty : bool
        Whether the recorded source differs from the commit.
    source_tree_digest : str
        Lowercase SHA-256 digest of the source tree.
    dirty_patch_digest : str | None
        Lowercase SHA-256 patch digest for dirty source, or None
        for clean source. Required as an argument even when None.

    Returns
    -------
    CodeRevisionV1
        Strict V1 revision record with exactly the supplied provenance.

    Raises
    ------
    ValueError
        An identity is malformed or dirty state and patch presence disagree.

    Notes
    -----
    Does not inspect Git, the filesystem, or an installed package. Call revision
    discovery helpers when facts must be discovered rather than supplied.
    """
    return CodeRevisionV1(
        package_version=package_version,
        commit_sha=commit_sha,
        source_tree_digest=source_tree_digest,
        is_dirty=is_dirty,
        dirty_patch_digest=dirty_patch_digest,
    )


def default_schema_versions_v1() -> tuple[SchemaVersionEntryV1, ...]:
    """Return the exact ordered schema bindings for a historical V1 context.

    Returns
    -------
    tuple[SchemaVersionEntryV1, ...]
        Tuple of V1 schema entries for REQUIRED_SCHEMA_BINDINGS_V1.

    Notes
    -----
    These bindings describe the legacy serialized roots. They are not the
    current context V3/frame V2 bindings.
    """
    return tuple(
        SchemaVersionEntryV1(schema_id=schema_id)
        for schema_id, _schema_version in REQUIRED_SCHEMA_BINDINGS_V1
    )


def build_evaluation_episode_context_v1(
    *,
    identity: EvaluationEpisodeIdentityV1,
    aggregation_keys: tuple[AggregationKeyV1, ...],
    expected_horizon: int,
    config: EnvConfig,
    public_agent_id_by_global_slot: tuple[str, ...],
    policy_assignments: tuple[PolicyAssignmentSlotV1, ...],
    seed_protocol: EvaluationSeedProtocolV1,
    capture_profile: CaptureProfile,
    execution_information_mode: ExecutionInformationMode,
    actor_projection: VersionedIdentityV1,
    critic_information_regime: VersionedIdentityV1,
    canonical_reward_mode: VersionedIdentityV1,
    shaping_configuration: ContentAddressedIdentityV1,
    code_revision: CodeRevisionV1,
) -> EvaluationEpisodeContextV1:
    """Build a historical version 1 episode context from explicit experiment facts.

    Parameters
    ----------
    identity : EvaluationEpisodeIdentityV1
        Runner-owned stable episode, run, evaluation, matchup, and task identities.
    aggregation_keys : tuple[AggregationKeyV1, ...]
        Immutable named experiment coordinates; names must be unique.
    expected_horizon : int
        Positive number of artifact transitions expected from frame zero.
    config : EnvConfig
        Unbatched EnvConfig accepted by Core's host configuration validator.
    public_agent_id_by_global_slot : tuple[str, ...]
        Ten unique public IDs in global-slot order,
        including inactive slots.
    policy_assignments : tuple[PolicyAssignmentSlotV1, ...]
        Ten assignment rows in global-slot order. Active slots
        have policy provenance; inactive slots are marked not applicable.
    seed_protocol : EvaluationSeedProtocolV1
        Realized seed provenance; this builder generates no seeds.
    capture_profile : CaptureProfile
        One of training_light, evaluation_metric_complete,
        scenario_metric_complete, or debug.
    execution_information_mode : ExecutionInformationMode
        "shared_obs" or "no_shared_obs".
    actor_projection : VersionedIdentityV1
        Supported projection identity matching the information mode
        and this context version.
    critic_information_regime : VersionedIdentityV1
        Explicit identifier/version of permitted critic input.
    canonical_reward_mode : VersionedIdentityV1
        Explicit identifier/version of the task reward contract.
    shaping_configuration : ContentAddressedIdentityV1
        Named/versioned content identity for the declared shaping setup.
    code_revision : CodeRevisionV1
        Supplied package/source provenance; no repository discovery occurs.

    Returns
    -------
    EvaluationEpisodeContextV1
        Validated context V1 with its exact schema bindings, resolved config,
        current static mechanics catalog, and ordered roster. It pairs with frame V1.

    Raises
    ------
    TypeError
        The runtime config has an unsupported field type or dtype.
    ValueError
        Config validity, ten-slot assignments, metadata, roster joins,
        schema bindings, or information-projection consistency fail validation.

    Notes
    -----
    Copies runtime arrays to host records during setup, outside JIT. It does not
    invent policy, seed, reward, or revision provenance. This legacy builder requires V1
    assignments, seeds, and revision facts.
    """
    validate_env_config(config)
    if len(policy_assignments) != MAX_AGENT_SLOTS:
        raise ValueError("policy_assignments must have length 10")
    return EvaluationEpisodeContextV1(
        identity=identity,
        schema_versions=default_schema_versions_v1(),
        aggregation_keys=aggregation_keys,
        expected_horizon=expected_horizon,
        resolved_env_config=build_resolved_env_config_v1(config),
        static_mechanics_catalog=build_static_mechanics_catalog_v1(),
        roster=build_roster_v1(config, public_agent_id_by_global_slot),
        policy_assignments=policy_assignments,
        seed_protocol=seed_protocol,
        capture_profile=capture_profile,
        execution_information_mode=execution_information_mode,
        actor_projection=actor_projection,
        critic_information_regime=critic_information_regime,
        canonical_reward_mode=canonical_reward_mode,
        shaping_configuration=shaping_configuration,
        code_revision=code_revision,
    )


def build_evaluation_episode_context_v2(
    *,
    identity: EvaluationEpisodeIdentityV1,
    aggregation_keys: tuple[AggregationKeyV1, ...],
    expected_horizon: int,
    config: EnvConfig,
    public_agent_id_by_global_slot: tuple[str, ...],
    policy_assignments: tuple[PolicyAssignmentSlotV2, ...],
    seed_protocol: EvaluationSeedProtocolV1 | EvaluationSeedProtocolV2,
    capture_profile: CaptureProfile,
    execution_information_mode: ExecutionInformationMode,
    actor_projection: VersionedIdentityV1,
    critic_information_regime: VersionedIdentityV1,
    canonical_reward_mode: VersionedIdentityV1,
    shaping_configuration: ContentAddressedIdentityV1,
    code_revision: CodeRevisionV1 | CodeRevisionV2,
    scenario_name: str | None = None,
) -> EvaluationEpisodeContextV2:
    """Build a historical version 2 episode context from explicit experiment facts.

    Parameters
    ----------
    identity : EvaluationEpisodeIdentityV1
        Runner-owned stable episode, run, evaluation, matchup, and task identities.
    aggregation_keys : tuple[AggregationKeyV1, ...]
        Immutable named experiment coordinates; names must be unique.
    expected_horizon : int
        Positive number of artifact transitions expected from frame zero.
    config : EnvConfig
        Unbatched EnvConfig accepted by Core's host configuration validator.
    public_agent_id_by_global_slot : tuple[str, ...]
        Ten unique public IDs in global-slot order,
        including inactive slots.
    policy_assignments : tuple[PolicyAssignmentSlotV2, ...]
        Ten assignment rows in global-slot order. Active slots
        have policy provenance; inactive slots are marked not applicable.
    seed_protocol : EvaluationSeedProtocolV1 | EvaluationSeedProtocolV2
        Realized seed provenance; this builder generates no seeds.
    capture_profile : CaptureProfile
        One of training_light, evaluation_metric_complete,
        scenario_metric_complete, or debug.
    execution_information_mode : ExecutionInformationMode
        "shared_obs" or "no_shared_obs".
    actor_projection : VersionedIdentityV1
        Supported projection identity matching the information mode
        and this context version.
    critic_information_regime : VersionedIdentityV1
        Explicit identifier/version of permitted critic input.
    canonical_reward_mode : VersionedIdentityV1
        Explicit identifier/version of the task reward contract.
    shaping_configuration : ContentAddressedIdentityV1
        Named/versioned content identity for the declared shaping setup.
    code_revision : CodeRevisionV1 | CodeRevisionV2
        Supplied package/source provenance; no repository discovery occurs.
    scenario_name : str | None
        Optional human-readable scenario name. Defaults to None.

    Returns
    -------
    EvaluationEpisodeContextV2
        Validated context V2 with its exact schema bindings, resolved config,
        current static mechanics catalog, and ordered roster. It pairs with frame V1.

    Raises
    ------
    TypeError
        The runtime config has an unsupported field type or dtype.
    ValueError
        Config validity, ten-slot assignments, metadata, roster joins,
        schema bindings, or information-projection consistency fail validation.

    Notes
    -----
    Copies runtime arrays to host records during setup, outside JIT. It does not
    invent policy, seed, reward, or revision provenance. V2 assignments allow explicitly
    unknown policy facts; seed and revision records may use V1 or V2.
    """
    validate_env_config(config)
    if len(policy_assignments) != MAX_AGENT_SLOTS:
        raise ValueError("policy_assignments must have length 10")
    return EvaluationEpisodeContextV2(
        identity=identity,
        schema_versions=tuple(
            SchemaVersionEntryV2(schema_id=name, schema_version=version)
            for name, version in REQUIRED_SCHEMA_BINDINGS_V2
        ),
        aggregation_keys=aggregation_keys,
        expected_horizon=expected_horizon,
        resolved_env_config=build_resolved_env_config_v1(config),
        static_mechanics_catalog=build_static_mechanics_catalog_v1(),
        roster=build_roster_v1(config, public_agent_id_by_global_slot),
        policy_assignments=policy_assignments,
        seed_protocol=seed_protocol,
        capture_profile=capture_profile,
        execution_information_mode=execution_information_mode,
        actor_projection=actor_projection,
        critic_information_regime=critic_information_regime,
        canonical_reward_mode=canonical_reward_mode,
        shaping_configuration=shaping_configuration,
        code_revision=code_revision,
        scenario_name=scenario_name,
    )


def build_evaluation_episode_context_v3(
    *,
    identity: EvaluationEpisodeIdentityV1,
    aggregation_keys: tuple[AggregationKeyV1, ...],
    expected_horizon: int,
    config: EnvConfig,
    public_agent_id_by_global_slot: tuple[str, ...],
    policy_assignments: tuple[PolicyAssignmentSlotV2, ...],
    seed_protocol: EvaluationSeedProtocolV1 | EvaluationSeedProtocolV2,
    capture_profile: CaptureProfile,
    execution_information_mode: ExecutionInformationMode,
    actor_projection: VersionedIdentityV1,
    critic_information_regime: VersionedIdentityV1,
    canonical_reward_mode: VersionedIdentityV1,
    shaping_configuration: ContentAddressedIdentityV1,
    code_revision: CodeRevisionV1 | CodeRevisionV2,
    scenario_name: str | None = None,
) -> EvaluationEpisodeContextV3:
    """Build the current version 3 episode context from explicit experiment facts.

    Parameters
    ----------
    identity : EvaluationEpisodeIdentityV1
        Runner-owned stable episode, run, evaluation, matchup, and task identities.
    aggregation_keys : tuple[AggregationKeyV1, ...]
        Immutable named experiment coordinates; names must be unique.
    expected_horizon : int
        Positive number of artifact transitions expected from frame zero.
    config : EnvConfig
        Unbatched EnvConfig accepted by Core's host configuration validator.
    public_agent_id_by_global_slot : tuple[str, ...]
        Ten unique public IDs in global-slot order,
        including inactive slots.
    policy_assignments : tuple[PolicyAssignmentSlotV2, ...]
        Ten assignment rows in global-slot order. Active slots
        have policy provenance; inactive slots are marked not applicable.
    seed_protocol : EvaluationSeedProtocolV1 | EvaluationSeedProtocolV2
        Realized seed provenance; this builder generates no seeds.
    capture_profile : CaptureProfile
        One of training_light, evaluation_metric_complete,
        scenario_metric_complete, or debug.
    execution_information_mode : ExecutionInformationMode
        "shared_obs" or "no_shared_obs".
    actor_projection : VersionedIdentityV1
        Supported projection identity matching the information mode
        and this context version.
    critic_information_regime : VersionedIdentityV1
        Explicit identifier/version of permitted critic input.
    canonical_reward_mode : VersionedIdentityV1
        Explicit identifier/version of the task reward contract.
    shaping_configuration : ContentAddressedIdentityV1
        Named/versioned content identity for the declared shaping setup.
    code_revision : CodeRevisionV1 | CodeRevisionV2
        Supplied package/source provenance; no repository discovery occurs.
    scenario_name : str | None
        Optional human-readable scenario name. Defaults to None.

    Returns
    -------
    EvaluationEpisodeContextV3
        Validated context V3 with its exact schema bindings, resolved config,
        current static mechanics catalog, and ordered roster. It pairs with frame V2.

    Raises
    ------
    TypeError
        The runtime config has an unsupported field type or dtype.
    ValueError
        Config validity, ten-slot assignments, metadata, roster joins,
        schema bindings, or information-projection consistency fail validation.

    Notes
    -----
    Copies runtime arrays to host records during setup, outside JIT. It does not
    invent policy, seed, reward, or revision provenance. V2 assignments allow explicitly
    unknown policy facts; seed and revision records may use V1 or V2.
    """
    validate_env_config(config)
    if len(policy_assignments) != MAX_AGENT_SLOTS:
        raise ValueError("policy_assignments must have length 10")
    return EvaluationEpisodeContextV3(
        identity=identity,
        schema_versions=tuple(
            SchemaVersionEntryV3(schema_id=name, schema_version=version)
            for name, version in REQUIRED_SCHEMA_BINDINGS_V3
        ),
        aggregation_keys=aggregation_keys,
        expected_horizon=expected_horizon,
        resolved_env_config=build_resolved_env_config_v1(config),
        static_mechanics_catalog=build_static_mechanics_catalog_v1(),
        roster=build_roster_v1(config, public_agent_id_by_global_slot),
        policy_assignments=policy_assignments,
        seed_protocol=seed_protocol,
        capture_profile=capture_profile,
        execution_information_mode=execution_information_mode,
        actor_projection=actor_projection,
        critic_information_regime=critic_information_regime,
        canonical_reward_mode=canonical_reward_mode,
        shaping_configuration=shaping_configuration,
        code_revision=code_revision,
        scenario_name=scenario_name,
    )


_INT32_MIN = int(np.iinfo(np.int32).min)
_INT32_MAX = int(np.iinfo(np.int32).max)


def _wire_int32_array(value: object, *, field_name: str, host: bool = False) -> Array:
    """Convert exact Python integers without overflow or boolean coercion.

    Preserve shape and raise TypeError for non-int elements or ValueError outside
    int32 range. By default return a JAX int32 array; host=True returns a NumPy int32
    array despite the shared Array annotation. This host-only check cannot trace.
    """
    object_values = np.asarray(value, dtype=object)
    for item in object_values.flat:
        if type(item) is not int:
            raise TypeError(f"{field_name} must contain only exact integers")
        integer = int(item)
        if not _INT32_MIN <= integer <= _INT32_MAX:
            raise ValueError(f"{field_name} must be representable as int32")
    array = np.asarray(value, dtype=np.int32)
    return cast(Array, array) if host else jnp.asarray(array, dtype=jnp.int32)


def _wire_float32_array(value: object, *, field_name: str, host: bool = False) -> Array:
    """Convert finite numeric wire values only when float32 preserves every value.

    Preserve shape and raise ValueError for nonfinite values or lossy narrowing.
    By default return a JAX float32 array; host=True returns a NumPy array despite
    the shared Array annotation. This host-only check cannot trace.
    """
    host_values = np.asarray(value, dtype=np.float64)
    if not bool(np.all(np.isfinite(host_values))):
        raise ValueError(f"{field_name} must contain only finite values")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        narrowed = host_values.astype(np.float32)
    if not bool(np.all(np.isfinite(narrowed))) or not np.array_equal(
        host_values,
        narrowed.astype(np.float64),
    ):
        raise ValueError(f"{field_name} must be losslessly representable as float32")
    return cast(Array, narrowed) if host else jnp.asarray(narrowed, dtype=jnp.float32)


def reconstruct_env_config_v1(context: EvaluationEpisodeContext) -> EnvConfig:
    """Restore one scalar runtime config from recorded context values.

    Parameters
    ----------
    context : EvaluationEpisodeContext
        Supported, already validated episode context containing resolved
        configuration and ten ordered roster rows.

    Returns
    -------
    EnvConfig
        Unbatched EnvConfig with Python scalar rules and JAX int32, float32, and bool
        arrays. Recorded profile overrides, geometry, and spawn settings are retained.

    Raises
    ------
    TypeError
        An integer wire field contains a non-integer value.
    ValueError
        An integer overflows int32 or a numeric array cannot be represented
        exactly as finite float32.

    Notes
    -----
    Runs on the host and may allocate arrays on JAX's default device. It does not
    reset state, rederive class defaults, validate the whole context, or run Core
    config validation. Admission and official-scenario checks own those checks.
    """
    resolved = context.resolved_env_config
    roster = context.roster
    mechanics = resolved.slot_mechanics

    class_ids = _wire_int32_array(
        tuple(row.class_id for row in roster),
        field_name="context.roster.class_id",
    )
    team_ids = _wire_int32_array(
        tuple(row.configured_team_id for row in roster),
        field_name="context.roster.configured_team_id",
    )
    active_mask = jnp.asarray(
        np.asarray(tuple(row.configured_active for row in roster), dtype=np.bool_),
        dtype=jnp.bool_,
    )
    profile = ResolvedAgentProfile(
        class_ids=class_ids,
        team_ids=team_ids,
        active_mask=active_mask,
        agent_radii=_wire_float32_array(
            tuple(row.body_radius for row in mechanics),
            field_name="resolved_env_config.slot_mechanics.body_radius",
        ),
        base_movement_speeds=_wire_float32_array(
            tuple(row.base_movement_speed for row in mechanics),
            field_name="resolved_env_config.slot_mechanics.base_movement_speed",
        ),
        observation_radii=_wire_float32_array(
            tuple(row.observation_radius for row in mechanics),
            field_name="resolved_env_config.slot_mechanics.observation_radius",
        ),
        basic_interaction_radii=_wire_float32_array(
            tuple(row.basic_interaction_radius for row in mechanics),
            field_name="resolved_env_config.slot_mechanics.basic_interaction_radius",
        ),
        ultimate_interaction_radii=_wire_float32_array(
            tuple(row.ultimate_interaction_radius for row in mechanics),
            field_name="resolved_env_config.slot_mechanics.ultimate_interaction_radius",
        ),
        max_health=_wire_float32_array(
            tuple(row.maximum_health for row in mechanics),
            field_name="resolved_env_config.slot_mechanics.maximum_health",
        ),
        out_of_combat_delay_steps=_wire_int32_array(
            tuple(row.out_of_combat_delay_steps for row in mechanics),
            field_name="resolved_env_config.slot_mechanics.out_of_combat_delay_steps",
        ),
        out_of_combat_health_regen_fraction_per_step=_wire_float32_array(
            tuple(
                row.out_of_combat_health_regeneration_fraction_per_step
                for row in mechanics
            ),
            field_name=(
                "resolved_env_config.slot_mechanics."
                "out_of_combat_health_regeneration_fraction_per_step"
            ),
        ),
    )

    obstacle_type_ids = tuple(row.obstacle_type_id for row in resolved.obstacle_slots)
    _wire_int32_array(
        obstacle_type_ids,
        field_name="resolved_env_config.obstacle_slots.obstacle_type_id",
    )
    obstacle_rows = tuple(
        (
            row.obstacle_type_id,
            row.x,
            row.y,
            row.radius,
            row.width,
            row.height,
            row.theta,
            1.0 if row.is_active else 0.0,
        )
        for row in resolved.obstacle_slots
    )
    obstacles = _wire_float32_array(
        obstacle_rows,
        field_name="resolved_env_config.obstacle_slots",
    )
    if np.shape(obstacles) != (MAX_OBSTACLE_SLOTS, OBSTACLE_FEATURES):
        raise AssertionError("resolved obstacle projection changed fixed shape")

    for field_name, value in (
        ("resolved_env_config.task_mode", resolved.task_mode),
        (
            "resolved_env_config.team_deathmatch_score_threshold",
            resolved.team_deathmatch_score_threshold,
        ),
        (
            "resolved_env_config.maximum_episode_steps",
            resolved.maximum_episode_steps,
        ),
        (
            "resolved_env_config.spawn_shield_duration_steps",
            resolved.spawn_shield_duration_steps,
        ),
    ):
        _wire_int32_array(value, field_name=field_name)
    config = EnvConfig(
        task_mode=resolved.task_mode,
        team_deathmatch_score_threshold=resolved.team_deathmatch_score_threshold,
        max_steps=resolved.maximum_episode_steps,
        map_width=resolved.map_width,
        map_height=resolved.map_height,
        obstacles=obstacles,
        agent_profile=profile,
        ordinary_movement_distance_scale=resolved.ordinary_movement_distance_scale,
        team_spawn_pad_positions=_wire_float32_array(
            resolved.team_spawn_pad_positions,
            field_name="resolved_env_config.team_spawn_pad_positions",
        ),
        spawn_shield_duration_steps=resolved.spawn_shield_duration_steps,
        spawn_shield_movement_speed=resolved.spawn_shield_movement_speed,
        team_respawn_wave_period_step_count=_wire_int32_array(
            resolved.team_respawn_wave_period_steps,
            field_name="resolved_env_config.team_respawn_wave_period_steps",
        ),
    )
    return config


def _validate_official_scenario_context_v2(  # pyright: ignore[reportUnusedFunction]
    context: EvaluationEpisodeContextV1,
    initial_frame: EvaluationFrameV1,
) -> None:
    """Admit a legacy scenario V2 context only as exact context V1 and frame V1.

    Raise TypeError for a different context type, then apply the shared official
    catalog, product-config, SharedObs, and initial-state checks.
    """
    if type(context) is not EvaluationEpisodeContextV1:
        raise TypeError(
            "context must be an EvaluationEpisodeContextV1, not "
            f"{type(context).__name__}"
        )
    _validate_official_scenario_context(context, initial_frame)


def _validate_official_scenario_context_v3(  # pyright: ignore[reportUnusedFunction]
    context: EvaluationEpisodeContextV2,
    initial_frame: EvaluationFrameV1,
) -> None:
    """Admit a scenario V3 context only as exact context V2 and frame V1.

    Raise TypeError for a different context type, then apply the shared official
    catalog, product-config, SharedObs, and initial-state checks.
    """
    if type(context) is not EvaluationEpisodeContextV2:
        raise TypeError("context must be an EvaluationEpisodeContextV2")
    _validate_official_scenario_context(context, initial_frame)


def _validate_official_scenario_context_v4(  # pyright: ignore[reportUnusedFunction]
    context: EvaluationEpisodeContextV3,
    initial_frame: EvaluationFrameV2,
) -> None:
    """Admit a scenario V4 context only as exact context V3 and frame V2.

    Raise TypeError for a different context type, then apply the shared official
    catalog, product-config, SharedObs, and initial-state checks.
    """
    if type(context) is not EvaluationEpisodeContextV3:
        raise TypeError("context must be an EvaluationEpisodeContextV3")
    _validate_official_scenario_context(context, initial_frame)


def _validate_official_scenario_context(
    context: EvaluationEpisodeContext,
    initial_frame: EvaluationFrame,
) -> None:
    """Check that an official scenario can be interpreted under current product rules.

    Require matching frame type/episode, frame zero, SharedObs and the context's
    projection version, exact current mechanics catalog, and lossless config/roster
    reconstruction. Delegate product-config and scenario-state validity to Core.
    Raise TypeError or ValueError on disagreement. This host check imports capture
    only when initial-state reconstruction is needed and never steps the simulator.
    """
    expected_frame = (
        EvaluationFrameV2
        if type(context) is EvaluationEpisodeContextV3
        else EvaluationFrameV1
    )
    if type(initial_frame) is not expected_frame:
        raise TypeError(
            "initial_frame must be an EvaluationFrameV1, not "
            f"{type(initial_frame).__name__}"
        )
    if context.execution_information_mode != "shared_obs":
        raise ValueError("official scenario evaluation requires shared_obs execution")
    expected_projection = (
        SHARED_OBS_ACTOR_PROJECTION_V2
        if type(context) is EvaluationEpisodeContextV3
        else SHARED_OBS_ACTOR_PROJECTION_V1
    )
    if context.actor_projection != expected_projection:
        raise ValueError(
            "official scenario evaluation requires "
            "base-observation-plus-authorized-sensor-source-bank version 1"
        )
    if initial_frame.episode_id != context.identity.episode_id:
        raise ValueError("initial frame episode identity must match context")
    if initial_frame.frame_index != 0:
        raise ValueError("official scenario initial_frame must have frame_index zero")

    live_catalog = build_static_mechanics_catalog_v1()
    if context.static_mechanics_catalog != live_catalog:
        raise ValueError("loaded context mechanics catalog disagrees with live catalog")

    config = reconstruct_env_config_v1(context)
    resolved = context.resolved_env_config
    roster = context.roster
    validate_product_env_config(config)

    reprojected_config = build_resolved_env_config_v1(config)
    if reprojected_config != resolved:
        raise ValueError("loaded resolved config does not exactly reproject")
    public_agent_ids = tuple(row.public_agent_id for row in roster)
    if build_roster_v1(config, public_agent_ids) != roster:
        raise ValueError("loaded roster does not exactly reproject")

    from marl_battlegrounds.evaluation.capture import reconstruct_env_state_v1

    initial_state = reconstruct_env_state_v1(initial_frame)
    validate_scenario_initial_state(config, initial_state)


__all__ = [
    "build_code_revision_v1",
    "build_evaluation_episode_context_v1",
    "build_evaluation_episode_context_v2",
    "build_evaluation_episode_context_v3",
    "build_evaluation_seed_protocol_v1",
    "build_resolved_env_config_v1",
    "build_roster_v1",
    "build_static_mechanics_catalog_v1",
    "default_schema_versions_v1",
    "reconstruct_env_config_v1",
]
