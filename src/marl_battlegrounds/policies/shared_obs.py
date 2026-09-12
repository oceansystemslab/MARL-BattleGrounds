"""Structured SharedObs composition over already-authored sensor rows."""

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.axis_mappings import (
    GLOBAL_SLOT_BY_ACTOR_AND_ALLY_OBSERVATION_ROW,
    GLOBAL_SLOT_BY_ACTOR_AND_ENEMY_OBSERVATION_ROW,
    TEAM_A_START,
    TEAM_B_START,
)
from marl_battlegrounds.core.types import (
    AGENT_FEATURE_ACTIVE,
    AGENT_FEATURE_ALIVE,
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    MAX_OBJECTIVE_SLOTS,
    OBJECTIVE_FEATURES,
    TEAM_A_ID,
    UNIT_FEATURES,
    ActionMask,
    Observation,
)
from marl_battlegrounds.policies.actor import ActorAction


class SharedObsSensorSourceBankV1(NamedTuple):
    """Historical global-source layout used when reading V1 recordings."""

    unit_features_by_sensor_source_and_global_slot: Array
    unit_visibility_by_sensor_source_and_global_slot: Array
    objective_features_by_sensor_source: Array


class SharedObsSensorSourceBankV2(NamedTuple):
    """Five own-team sources, with five ally then five enemy candidates.

    An actor receives features ``(5, 10, 58)``, visibility ``(5, 10)`` and
    objectives ``(5, 8, 12)``. No axis identifies a simulator team or global slot.
    """

    unit_features_by_source_and_candidate: Array
    unit_visibility_by_source_and_candidate: Array
    objective_features_by_source: Array


SharedObsPolicy = Callable[
    [Observation, ActionMask, Array, SharedObsSensorSourceBankV2, Array],
    ActorAction,
]

_ALLY_GLOBAL_SLOTS = jnp.asarray(
    GLOBAL_SLOT_BY_ACTOR_AND_ALLY_OBSERVATION_ROW,
    dtype=jnp.int32,
)
_ENEMY_GLOBAL_SLOTS = jnp.asarray(
    GLOBAL_SLOT_BY_ACTOR_AND_ENEMY_OBSERVATION_ROW,
    dtype=jnp.int32,
)
_GLOBAL_SLOTS = jnp.arange(MAX_AGENT_SLOTS, dtype=jnp.int32)
_SOURCE_ROWS = _GLOBAL_SLOTS[:, None]


def build_shared_obs_sensor_source_bank_from_base_rows(
    ally_unit_features: Array,
    enemy_unit_features: Array,
    objective_features_by_source: Array,
    ally_visibility_mask: Array,
    enemy_visibility_mask: Array,
    source_is_living: Array,
    *,
    global_slot_by_actor_and_ally_observation_row: Array = _ALLY_GLOBAL_SLOTS,
    global_slot_by_actor_and_enemy_observation_row: Array = _ENEMY_GLOBAL_SLOTS,
) -> SharedObsSensorSourceBankV1:
    """Remap visibility-redacted base rows under one explicit source-lifecycle mask."""
    unit_features = jnp.zeros(
        (MAX_AGENT_SLOTS, MAX_AGENT_SLOTS, UNIT_FEATURES),
        dtype=jnp.float32,
    )
    unit_visibility = jnp.zeros(
        (MAX_AGENT_SLOTS, MAX_AGENT_SLOTS),
        dtype=jnp.bool_,
    )
    ally_global_slots = jnp.asarray(
        global_slot_by_actor_and_ally_observation_row,
        dtype=jnp.int32,
    )
    enemy_global_slots = jnp.asarray(
        global_slot_by_actor_and_enemy_observation_row,
        dtype=jnp.int32,
    )
    unit_features = unit_features.at[_SOURCE_ROWS, ally_global_slots].set(
        ally_unit_features
    )
    unit_features = unit_features.at[_SOURCE_ROWS, enemy_global_slots].set(
        enemy_unit_features
    )
    unit_visibility = unit_visibility.at[_SOURCE_ROWS, ally_global_slots].set(
        ally_visibility_mask
    )
    unit_visibility = unit_visibility.at[_SOURCE_ROWS, enemy_global_slots].set(
        enemy_visibility_mask
    )

    unit_visibility = jnp.logical_and(unit_visibility, source_is_living[:, None])
    unit_features = jnp.where(
        unit_visibility[:, :, None],
        unit_features,
        jnp.zeros_like(unit_features),
    ).astype(jnp.float32)
    objective_features = jnp.where(
        source_is_living[:, None, None],
        objective_features_by_source,
        jnp.zeros(
            (MAX_AGENT_SLOTS, MAX_OBJECTIVE_SLOTS, OBJECTIVE_FEATURES),
            dtype=jnp.float32,
        ),
    ).astype(jnp.float32)

    return SharedObsSensorSourceBankV1(
        unit_features_by_sensor_source_and_global_slot=unit_features,
        unit_visibility_by_sensor_source_and_global_slot=unit_visibility,
        objective_features_by_sensor_source=objective_features,
    )


def build_shared_obs_team_source_bank_from_base_rows(
    ally_unit_features: Array,
    enemy_unit_features: Array,
    objective_features_by_source: Array,
    ally_visibility_mask: Array,
    enemy_visibility_mask: Array,
    source_is_living: Array,
) -> SharedObsSensorSourceBankV2:
    """Join one team's already relative rows; dead sources provide no sensors.

    Each input has five source rows. This same builder serves live policies and
    recorded input reconstruction. It neither derives visibility nor reads state.
    """
    visibility = (
        jnp.concatenate((ally_visibility_mask, enemy_visibility_mask), axis=1)
        & source_is_living[:, None]
    )
    features = jnp.concatenate((ally_unit_features, enemy_unit_features), axis=1)
    return SharedObsSensorSourceBankV2(
        jnp.where(visibility[:, :, None], features, 0.0).astype(jnp.float32),
        visibility,
        jnp.where(
            source_is_living[:, None, None], objective_features_by_source, 0.0
        ).astype(jnp.float32),
    )


def build_shared_obs_sensor_source_bank(
    observation: Observation,
) -> SharedObsSensorSourceBankV2:
    """Prepare both teams once, with a leading two-team routing axis.

    Select a team before delivering this bank to a policy. Actor inputs never
    contain this outer routing axis. No global candidate bank is materialized.
    """
    source_is_living = jnp.logical_and(
        observation.self_features[:, AGENT_FEATURE_ACTIVE] > 0.0,
        observation.self_features[:, AGENT_FEATURE_ALIVE] > 0.0,
    )
    inputs = (
        observation.ally_unit_features,
        observation.enemy_unit_features,
        observation.objective_features,
        observation.ally_visibility_mask,
        observation.enemy_visibility_mask,
        source_is_living,
    )

    def team_rows(value: Array) -> Array:
        return value.reshape((2, MAX_AGENTS_PER_TEAM, *value.shape[1:]))

    return jax.vmap(build_shared_obs_team_source_bank_from_base_rows)(
        *jax.tree.map(team_rows, inputs)
    )


def build_default_shared_obs_information_availability(
    active_mask: Array,
    team_ids: Array,
) -> Array:
    """Authorize configured-active, same-team, off-diagonal sensor sources."""
    active = jnp.asarray(active_mask, dtype=jnp.bool_)
    teams = jnp.asarray(team_ids, dtype=jnp.int32)
    configured_pair = jnp.logical_and(active[:, None], active[None, :])
    same_team = teams[:, None] == teams[None, :]
    off_diagonal = ~jnp.eye(MAX_AGENT_SLOTS, dtype=jnp.bool_)
    return jnp.logical_and(configured_pair, jnp.logical_and(same_team, off_diagonal))


def _select_shared_unit_material(
    source_bank: SharedObsSensorSourceBankV2,
    recipient_source_availability: Array,
) -> tuple[Array, Array]:
    """Select the lowest admitted own-team source for each relative candidate."""
    admitted = jnp.logical_and(
        recipient_source_availability[:, None],
        source_bank.unit_visibility_by_source_and_candidate,
    )
    candidate_visible = jnp.any(admitted, axis=0)
    selected_source = jnp.argmax(admitted, axis=0).astype(jnp.int32)
    candidate_features = source_bank.unit_features_by_source_and_candidate[
        selected_source,
        _GLOBAL_SLOTS,
    ]
    candidate_features = jnp.where(
        candidate_visible[:, None],
        candidate_features,
        jnp.zeros_like(candidate_features),
    ).astype(jnp.float32)
    return candidate_features, candidate_visible


def compose_shared_obs_unit_features(
    recipient_observation: Observation,
    source_bank: SharedObsSensorSourceBankV2,
    recipient_source_availability: Array,
) -> tuple[Array, Array, Array, Array]:
    """Union admitted teammate rows into recipient-relative unit feature axes."""
    shared_features, shared_visible = _select_shared_unit_material(
        source_bank,
        recipient_source_availability,
    )
    shared_ally_visible = shared_visible[:MAX_AGENTS_PER_TEAM]
    shared_enemy_visible = shared_visible[MAX_AGENTS_PER_TEAM:]

    ally_visible = jnp.logical_or(
        recipient_observation.ally_visibility_mask,
        shared_ally_visible,
    )
    enemy_visible = jnp.logical_or(
        recipient_observation.enemy_visibility_mask,
        shared_enemy_visible,
    )
    ally_features = jnp.where(
        recipient_observation.ally_visibility_mask[:, None],
        recipient_observation.ally_unit_features,
        jnp.where(
            shared_ally_visible[:, None],
            shared_features[:MAX_AGENTS_PER_TEAM],
            jnp.zeros_like(recipient_observation.ally_unit_features),
        ),
    ).astype(jnp.float32)
    enemy_features = jnp.where(
        recipient_observation.enemy_visibility_mask[:, None],
        recipient_observation.enemy_unit_features,
        jnp.where(
            shared_enemy_visible[:, None],
            shared_features[MAX_AGENTS_PER_TEAM:],
            jnp.zeros_like(recipient_observation.enemy_unit_features),
        ),
    ).astype(jnp.float32)

    return ally_features, enemy_features, ally_visible, enemy_visible


def mask_source_bank_for_recipient(
    source_bank: SharedObsSensorSourceBankV2,
    recipient_source_availability: Array,
) -> SharedObsSensorSourceBankV2:
    """Remove every unavailable source row before crossing the policy ABI."""
    source_available = jnp.asarray(recipient_source_availability, dtype=jnp.bool_)
    return SharedObsSensorSourceBankV2(
        unit_features_by_source_and_candidate=jnp.where(
            source_available[:, None, None],
            source_bank.unit_features_by_source_and_candidate,
            0.0,
        ),
        unit_visibility_by_source_and_candidate=jnp.logical_and(
            source_available[:, None],
            source_bank.unit_visibility_by_source_and_candidate,
        ),
        objective_features_by_source=jnp.where(
            source_available[:, None, None],
            source_bank.objective_features_by_source,
            0.0,
        ),
    )


@jax.jit(static_argnums=5)
def execute_shared_obs_team_policy(
    observation: Observation,
    action_mask: ActionMask,
    key: Array,
    source_bank: SharedObsSensorSourceBankV2,
    information_availability: Array,
    policy: SharedObsPolicy,
    team_identity: int | Array,
) -> ActorAction:
    """Route five actors to one policy, passing only own-team source positions."""
    start_index = jnp.where(team_identity == TEAM_A_ID, TEAM_A_START, TEAM_B_START)

    def _prune_tree(leaf: Array) -> Array:
        return jax.lax.dynamic_slice_in_dim(leaf, start_index, MAX_AGENTS_PER_TEAM)

    team_observation = jax.tree.map(_prune_tree, observation)
    team_action_mask = jax.tree.map(_prune_tree, action_mask)
    team_keys = jax.tree.map(_prune_tree, key)
    team_availability = jax.lax.dynamic_slice(
        information_availability,
        (start_index, start_index),
        (MAX_AGENTS_PER_TEAM, MAX_AGENTS_PER_TEAM),
    )

    def take_bank(leaf: Array) -> Array:
        return leaf[jnp.where(team_identity == TEAM_A_ID, 0, 1)]

    team_bank = jax.tree.map(take_bank, source_bank)

    def _execute_recipient_policy(
        recipient_observation: Observation,
        recipient_action_mask: ActionMask,
        actor_key: Array,
        recipient_source_availability: Array,
    ) -> ActorAction:
        authorized_source_bank = mask_source_bank_for_recipient(
            team_bank,
            recipient_source_availability,
        )
        return policy(
            recipient_observation,
            recipient_action_mask,
            actor_key,
            authorized_source_bank,
            recipient_source_availability,
        )

    policy_vmap = jax.vmap(
        fun=_execute_recipient_policy,
        in_axes=(0, 0, 0, 0),
        out_axes=0,
    )
    return policy_vmap(
        team_observation,
        team_action_mask,
        team_keys,
        team_availability,
    )


__all__ = (
    "SharedObsPolicy",
    "SharedObsSensorSourceBankV1",
    "SharedObsSensorSourceBankV2",
    "build_default_shared_obs_information_availability",
    "build_shared_obs_sensor_source_bank",
    "build_shared_obs_team_source_bank_from_base_rows",
    "compose_shared_obs_unit_features",
    "execute_shared_obs_team_policy",
    "mask_source_bank_for_recipient",
)
