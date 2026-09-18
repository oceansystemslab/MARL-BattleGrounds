"""Prepare and deliver only the teammate sensor data an actor may receive.

A source is a teammate's existing observation row, not a new observation of
hidden simulator state. The V2 bank groups five own-team sources and their
ally/enemy candidate rows. The executor removes unavailable source data before
each actor's policy call. Core remains the owner of visibility and action masks.

The V1 builder retains the older global-row layout for historical recordings.
Do not deliver a full two-team routing bank directly to a scalar actor.
"""

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
    """Store the historical ten-source, global-candidate sensor layout.

    Attributes
    ----------
    unit_features_by_sensor_source_and_global_slot : Array
        Float32 features of shape (10, 10, 58): source, global candidate, feature.
        Invisible candidates and nonliving sources have zero features.
    unit_visibility_by_sensor_source_and_global_slot : Array
        Boolean visibility of shape (10, 10), with the same source/candidate axes.
    objective_features_by_sensor_source : Array
        Float32 objective rows of shape (10, 8, 12). Nonliving sources are zeroed.

    Notes
    -----
    This immutable JAX tuple supports reconstruction of V1 records. Current actor
    delivery uses V2 and applies recipient permissions separately.
    """

    unit_features_by_sensor_source_and_global_slot: Array
    unit_visibility_by_sensor_source_and_global_slot: Array
    objective_features_by_sensor_source: Array


class SharedObsSensorSourceBankV2(NamedTuple):
    """Store five own-team sources in the current actor-input layout.

    Attributes
    ----------
    unit_features_by_source_and_candidate : Array
        Float32 array of shape (5, 10, 58): source, candidate, feature. Candidate
        rows are five allies followed by five enemies in the observation order.
    unit_visibility_by_source_and_candidate : Array
        Boolean array of shape (5, 10). False means that source supplies no
        visible row for the candidate.
    objective_features_by_source : Array
        Float32 objective rows of shape (5, 8, 12).

    Notes
    -----
    A bank built for both teams has an extra leading axis of size 2 for routing.
    That axis is removed before policy delivery. A recipient's unavailable source
    rows must be cleared with mask_source_bank_for_recipient. This record does
    not itself check permissions or rebuild visibility.
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
    """Rebuild a historical V1 bank from already recorded sensor rows.

    Parameters
    ----------
    ally_unit_features, enemy_unit_features : Array
        Float32 arrays of shape (10, 5, 58). The first axis selects the observing
        actor; the second selects an ally or enemy in that actor's row order.
    objective_features_by_source : Array
        Float32 array of shape (10, 8, 12) for the same observing actors.
    ally_visibility_mask, enemy_visibility_mask : Array
        Boolean arrays of shape (10, 5), paired with the unit-feature rows.
    source_is_living : Array
        Boolean array of shape (10,). False clears all sensors for that source.
    global_slot_by_actor_and_ally_observation_row : Array, optional
        Int32 mapping of shape (10, 5) from an observer's ally row to a global
        slot. Defaults to the installed Core ally-row mapping.
    global_slot_by_actor_and_enemy_observation_row : Array, optional
        Matching int32 enemy-row mapping of shape (10, 5). Defaults to the
        installed Core enemy-row mapping. Supply historical mappings when
        reconstructing records whose axis contract differs.

    Returns
    -------
    SharedObsSensorSourceBankV1
        Features and visibility in global candidate order, with hidden rows and
        nonliving sources cleared. Objective data is cleared for nonliving sources.

    Notes
    -----
    The caller supplies compatible shapes and correct mappings. This helper does
    not validate historical identities or authorize delivery to a recipient.
    It uses JAX numerical operations, changes no inputs and samples no randomness.
    """
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
    """Join one team's recorded ally and enemy sensors into a V2 bank.

    Parameters
    ----------
    ally_unit_features, enemy_unit_features : Array
        Float32 arrays of shape (5, 5, 58): source, candidate, feature. Candidate
        order must already follow the current actor-relative observation contract.
    objective_features_by_source : Array
        Float32 objective data of shape (5, 8, 12).
    ally_visibility_mask, enemy_visibility_mask : Array
        Boolean arrays of shape (5, 5) for those feature rows.
    source_is_living : Array
        Boolean (5,) mask. False clears that source's unit and objective data.

    Returns
    -------
    SharedObsSensorSourceBankV2
        Five-source bank. Ally candidates precede enemy candidates. Hidden unit
        rows and all data from nonliving sources are zeroed.

    Notes
    -----
    The same numerical builder serves live inputs and recorded reconstruction.
    It does not derive visibility or read privileged state. Apply recipient source
    permissions before calling a policy. Shapes are a caller precondition.
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
    """Build a V2 sensor bank for both teams from one game's observations.

    Parameters
    ----------
    observation : Observation
        Current ten-actor observations, with Team A's five routing rows followed
        by Team B's five rows. Each actor's candidate rows are already relative.

    Returns
    -------
    SharedObsSensorSourceBankV2
        Arrays with a leading two-team routing axis. Feature, visibility and
        objective shapes are (2, 5, 10, 58), (2, 5, 10), and (2, 5, 8, 12).
        Inactive or dead sources contribute no sensor data.

    Notes
    -----
    Select one team and apply its recipient permissions before actor delivery.
    The outer team axis is internal routing data. No global-candidate bank is
    created, no inputs are changed, and no action masks are rebuilt.
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
        """Split the ten routing rows into two teams of five without changing data.

        All dimensions after the actor axis keep their order.
        """
        return value.reshape((2, MAX_AGENTS_PER_TEAM, *value.shape[1:]))

    return jax.vmap(build_shared_obs_team_source_bank_from_base_rows)(
        *jax.tree.map(team_rows, inputs)
    )


def build_default_shared_obs_information_availability(
    active_mask: Array,
    team_ids: Array,
) -> Array:
    """Allow each active actor to receive sensors from its active teammates.

    Parameters
    ----------
    active_mask : Array
        Configured activity for ten slots, shape (10,). Converted to Boolean.
        This describes roster membership, not whether an actor is currently alive.
    team_ids : Array
        Team ID for the same ten slots, shape (10,). Converted to int32.

    Returns
    -------
    Array
        Boolean (10, 10) matrix indexed by recipient, then source. True requires
        both slots to be active, to share a team ID and to be different slots.

    Notes
    -----
    Dead teammates may remain authorized; the bank builder clears their current
    sensor data. This function does not validate team IDs or infer visibility.
    Use vmap to build permissions for several games.
    """
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
    """Select one admitted visible source row for each candidate.

    The bank holds five sources; recipient_source_availability is Boolean (5,).
    For each of ten candidates, choose the lowest source index that is both allowed
    and visible. Return float32 features (10, 58) and Boolean visibility (10,).
    If none is available, return zero features and False for that candidate. The
    caller has already established which sources the recipient may access.
    """
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
    """Add permitted teammate sightings to one actor's own unit observations.

    Parameters
    ----------
    recipient_observation : Observation
        One actor's current observation, without an actor or game batch axis.
    source_bank : SharedObsSensorSourceBankV2
        That actor's own-team bank with five sources and ten candidates.
    recipient_source_availability : Array
        Boolean (5,) permission mask for the same source order. The caller must
        supply only scientifically permitted sources.

    Returns
    -------
    ally_features : Array
        Float32 (5, 58) ally rows. The actor's own visible row takes precedence.
        Otherwise use the lowest-index admitted source that sees the candidate.
    enemy_features : Array
        Float32 (5, 58) enemy rows under the same precedence rule.
    ally_visible : Array
        Boolean (5,) union of own and admitted shared ally visibility.
    enemy_visible : Array
        Boolean (5,) union of own and admitted shared enemy visibility.

    Notes
    -----
    Rows unseen by every permitted source remain zero. This combines unit rows
    only; it does not merge objectives, invent hidden features or recompute an
    action mask. Inputs are unchanged and the operation supports JAX transforms.
    """
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
    """Clear every source row the recipient is not allowed to receive.

    Parameters
    ----------
    source_bank : SharedObsSensorSourceBankV2
        One team's five-source bank.
    recipient_source_availability : Array
        Permission for its five sources, shape (5,). Converted to Boolean.
        The caller owns permission validation; this helper applies that mask.

    Returns
    -------
    SharedObsSensorSourceBankV2
        Same-shaped bank. Unavailable unit and objective values are zero;
        unavailable visibility flags are False. Available values are unchanged.

    Notes
    -----
    Call this before delivering a bank to an actor. It does not infer team
    membership or grant permission from visibility. The original bank is unchanged.
    """
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
    """Call a policy for five actors, each with its own permitted source rows.

    Parameters
    ----------
    observation : Observation
        Current ten-actor observation for one game in Core routing order.
    action_mask : ActionMask
        Matching ten-actor masks for that decision.
    key : Array
        Ten per-actor random keys in the same routing order: typed shape (10,)
        or legacy uint32 shape (10, 2). Supply distinct keys for separate draws.
    source_bank : SharedObsSensorSourceBankV2
        Bank for both teams, with a leading routing axis of size 2, as returned
        by build_shared_obs_sensor_source_bank.
    information_availability : Array
        Boolean (10, 10) recipient/source permission matrix. The caller must
        restrict it to permitted sources. This executor selects its own-team
        block and never delivers cross-team source rows.
    policy : callable
        Scalar callable accepting (observation, action_mask, key, source_bank,
        source_availability) and returning ActorAction. It must support JAX
        transforms. The callable is static for compilation.
    team_identity : int or Array
        TEAM_A_ID or TEAM_B_ID. The caller must validate this routing choice;
        other values are not checked here.

    Returns
    -------
    ActorAction
        Five actions in the selected team's slot order. Each field has shape (5,).

    Notes
    -----
    All five slots are called, including inactive slots with their Core masks.
    Before each call, unavailable source data is cleared. Team identity, other
    actors' private inputs and the outer routing axes are not policy arguments.
    Inputs are unchanged. Masks and random keys are passed through rather than
    recomputed. Apply vmap over games outside this one-game executor.
    """
    start_index = jnp.where(team_identity == TEAM_A_ID, TEAM_A_START, TEAM_B_START)

    def _prune_tree(leaf: Array) -> Array:
        """Take the selected team's five rows while keeping later axes unchanged."""
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
        """Select one team's numerical bank and remove the outer routing axis."""
        return leaf[jnp.where(team_identity == TEAM_A_ID, 0, 1)]

    team_bank = jax.tree.map(take_bank, source_bank)

    def _execute_recipient_policy(
        recipient_observation: Observation,
        recipient_action_mask: ActionMask,
        actor_key: Array,
        recipient_source_availability: Array,
    ) -> ActorAction:
        """Clear unavailable source data, then call the policy once for this actor.

        The enclosing team call supplies matched observation, mask, key and five-source
        permission rows. The scalar action is returned without repair or remapping.
        """
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
