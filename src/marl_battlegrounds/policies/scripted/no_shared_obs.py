"""NoSharedObs facts for the Team Deathmatch scripted policy."""

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.types import (
    AGENT_FEATURE_ACTIVE,
    AGENT_FEATURE_ALIVE,
    CONTEXT_FEATURE_MAP_HEIGHT,
    CONTEXT_FEATURE_MAP_WIDTH,
    ActionMask,
    Observation,
)
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.scripted.team_deathmatch import (
    FOCAL_SHIELD_KNOWN_FALSE,
    FOCAL_SHIELD_KNOWN_TRUE,
    FOCAL_SHIELD_UNKNOWN,
    TEAM_DEATHMATCH_PROFILE,
    PolicyFacts,
    ScriptedTrace,
    TeamDeathmatchProfile,
    decide_team_deathmatch,
    decide_team_deathmatch_with_profile,
)

NO_SHARED_OBS_ADAPTER_ID = "no_shared_obs"
NO_SHARED_OBS_ADAPTER_VERSION = 1


def _focal_shield_state(
    observation: Observation,
    action_mask: ActionMask,
) -> Array:
    """Return what the focal actor can prove about its current shield."""
    known_false = jnp.asarray(FOCAL_SHIELD_KNOWN_FALSE, dtype=jnp.int32)
    known_true = jnp.asarray(FOCAL_SHIELD_KNOWN_TRUE, dtype=jnp.int32)
    unknown = jnp.asarray(FOCAL_SHIELD_UNKNOWN, dtype=jnp.int32)
    focal_active = observation.self_features[AGENT_FEATURE_ACTIVE] > 0.0
    focal_alive = observation.self_features[AGENT_FEATURE_ALIVE] > 0.0
    lifecycle_eligible = jnp.logical_and(focal_active, focal_alive)

    lifecycle = observation.spawn_lifecycle
    # The public own-team row does not identify the focal slot, so use only
    # roster-wide proofs about its shield state.
    own_living = jnp.logical_and(
        lifecycle.active_mask_by_agent_by_team[0],
        lifecycle.alive_mask_by_agent_by_team[0],
    )
    own_shielded = jnp.logical_and(
        own_living,
        lifecycle.spawn_shield_actual_durations_by_agent_by_team[0] > 0,
    )
    every_living_ally_shielded = jnp.logical_and(
        jnp.any(own_living),
        jnp.all(jnp.logical_or(jnp.logical_not(own_living), own_shielded)),
    )
    no_living_ally_shielded = jnp.logical_not(jnp.any(own_shielded))

    # Any legal non-inert action proves that the focal actor is not shield-blocked.
    joint_support = action_mask.select_target_use_ultimate_joint_mask
    non_inert_support = joint_support.at[0, 0].set(False)
    proves_unshielded = jnp.logical_or(
        no_living_ally_shielded,
        jnp.any(non_inert_support),
    )

    shield_state = jnp.where(
        every_living_ally_shielded,
        known_true,
        jnp.where(
            proves_unshielded,
            known_false,
            unknown,
        ),
    )
    return jnp.where(
        lifecycle_eligible,
        shield_state,
        known_false,
    )


def _build_policy_facts(
    observation: Observation,
    action_mask: ActionMask,
) -> PolicyFacts:
    """Select only the public facts authorized for NoSharedObs."""
    lifecycle = observation.spawn_lifecycle
    own_active = lifecycle.active_mask_by_agent_by_team[0]
    enemy_active = lifecycle.active_mask_by_agent_by_team[1]

    # Keep task context out of the scorer; only map dimensions cross this boundary.
    return PolicyFacts(
        focal_features=observation.self_features,
        ally_features=observation.ally_unit_features,
        enemy_features=observation.enemy_unit_features,
        obstacles=observation.map_obstacle_features,
        ally_visible=observation.ally_visibility_mask,
        enemy_visible=observation.enemy_visibility_mask,
        own_active=own_active,
        enemy_active=enemy_active,
        own_alive=lifecycle.alive_mask_by_agent_by_team[0],
        enemy_alive=lifecycle.alive_mask_by_agent_by_team[1],
        own_spawn_shields=(lifecycle.spawn_shield_actual_durations_by_agent_by_team[0]),
        enemy_spawn_shields=(
            lifecycle.spawn_shield_actual_durations_by_agent_by_team[1]
        ),
        own_class_ids=lifecycle.class_ids_by_agent_by_team[0],
        enemy_class_ids=lifecycle.class_ids_by_agent_by_team[1],
        own_configured_count=jnp.sum(own_active, dtype=jnp.int32),
        enemy_configured_count=jnp.sum(enemy_active, dtype=jnp.int32),
        map_width=observation.context_features[CONTEXT_FEATURE_MAP_WIDTH].astype(
            jnp.float32
        ),
        map_height=observation.context_features[CONTEXT_FEATURE_MAP_HEIGHT].astype(
            jnp.float32
        ),
        focal_shield_state=_focal_shield_state(observation, action_mask),
    )


def decide_team_deathmatch_no_shared_obs(
    observation: Observation,
    action_mask: ActionMask,
    key: Array,
) -> tuple[ActorAction, ScriptedTrace]:
    """Choose one TDM action and return its compact semantic trace."""
    return decide_team_deathmatch(
        _build_policy_facts(observation, action_mask),
        action_mask,
        key,
    )


def _decide_team_deathmatch_no_shared_obs_with_profile(
    observation: Observation,
    action_mask: ActionMask,
    key: Array,
    profile: TeamDeathmatchProfile,
) -> tuple[ActorAction, ScriptedTrace]:
    """Choose one action with the immutable profile bound to the callable."""
    return decide_team_deathmatch_with_profile(
        _build_policy_facts(observation, action_mask),
        action_mask,
        key,
        profile,
    )


class _TeamDeathmatchNoSharedObsPolicy(NamedTuple):
    """Hashable policy callable with one resolved immutable profile."""

    profile: TeamDeathmatchProfile

    def __call__(
        self,
        observation: Observation,
        action_mask: ActionMask,
        key: Array,
    ) -> ActorAction:
        """Choose one scalar NoSharedObs Team Deathmatch action."""
        action, _ = _decide_team_deathmatch_no_shared_obs_with_profile(
            observation,
            action_mask,
            key,
            self.profile,
        )
        return action


team_deathmatch_no_shared_obs_policy = _TeamDeathmatchNoSharedObsPolicy(
    profile=TEAM_DEATHMATCH_PROFILE
)


__all__ = (
    "NO_SHARED_OBS_ADAPTER_ID",
    "NO_SHARED_OBS_ADAPTER_VERSION",
    "decide_team_deathmatch_no_shared_obs",
    "team_deathmatch_no_shared_obs_policy",
)
