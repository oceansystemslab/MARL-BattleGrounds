"""NoSharedObs adapter for the canonical Team Deathmatch scripted policy."""

from typing import NamedTuple

from jax import Array

from marl_battlegrounds.core.types import ActionMask, Observation
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.scripted.common import (
    build_team_deathmatch_policy_facts,
    focal_shield_state,
)
from marl_battlegrounds.policies.scripted.team_deathmatch import (
    TEAM_DEATHMATCH_PROFILE,
    PolicyFacts,
    ScriptedTrace,
    TeamDeathmatchProfile,
    decide_team_deathmatch,
    decide_team_deathmatch_with_profile,
)

NO_SHARED_OBS_ADAPTER_ID = "no_shared_obs"
NO_SHARED_OBS_ADAPTER_VERSION = 1


def _focal_shield_state(  # pyright: ignore[reportUnusedFunction]
    observation: Observation,
    action_mask: ActionMask,
) -> Array:
    """Compatibility wrapper for the regime-neutral focal proof."""
    return focal_shield_state(observation, action_mask)


def _build_policy_facts(
    observation: Observation,
    action_mask: ActionMask,
) -> PolicyFacts:
    """Select only the public facts authorized for NoSharedObs."""
    return build_team_deathmatch_policy_facts(observation, action_mask)


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
