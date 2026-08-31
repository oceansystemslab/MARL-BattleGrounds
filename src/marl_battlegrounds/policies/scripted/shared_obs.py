"""SharedObs adapter for the canonical Team Deathmatch scripted policy."""

from typing import NamedTuple

from jax import Array

from marl_battlegrounds.core.types import ActionMask, Observation
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.scripted.common import (
    build_team_deathmatch_policy_facts,
)
from marl_battlegrounds.policies.scripted.team_deathmatch import (
    TEAM_DEATHMATCH_PROFILE,
    PolicyFacts,
    ScriptedTrace,
    TeamDeathmatchProfile,
    decide_team_deathmatch,
    decide_team_deathmatch_with_profile,
)
from marl_battlegrounds.policies.shared_obs import (
    SharedObsSensorSourceBankV1,
    compose_shared_obs_unit_features,
)

SHARED_OBS_ADAPTER_ID = "shared_obs"
SHARED_OBS_ADAPTER_VERSION = 1


def _build_shared_obs_team_deathmatch_policy_facts(
    recipient_observation: Observation,
    recipient_action_mask: ActionMask,
    source_bank: SharedObsSensorSourceBankV1,
    recipient_source_availability: Array,
    recipient_global_slot: Array,
) -> PolicyFacts:
    """Project admitted source rows into the regime-neutral TDM fact contract."""
    ally_features, enemy_features, ally_visible, enemy_visible = (
        compose_shared_obs_unit_features(
            recipient_observation,
            source_bank,
            recipient_source_availability,
            recipient_global_slot,
        )
    )
    return build_team_deathmatch_policy_facts(
        recipient_observation,
        recipient_action_mask,
        ally_features=ally_features,
        enemy_features=enemy_features,
        ally_visible=ally_visible,
        enemy_visible=enemy_visible,
    )


def decide_team_deathmatch_shared_obs(
    recipient_observation: Observation,
    recipient_action_mask: ActionMask,
    key: Array,
    source_bank: SharedObsSensorSourceBankV1,
    recipient_source_availability: Array,
    recipient_global_slot: Array,
) -> tuple[ActorAction, ScriptedTrace]:
    """Choose one TDM action from the recipient base input plus admitted sources."""
    return decide_team_deathmatch(
        _build_shared_obs_team_deathmatch_policy_facts(
            recipient_observation,
            recipient_action_mask,
            source_bank,
            recipient_source_availability,
            recipient_global_slot,
        ),
        recipient_action_mask,
        key,
    )


def _decide_team_deathmatch_shared_obs_with_profile(
    recipient_observation: Observation,
    recipient_action_mask: ActionMask,
    key: Array,
    source_bank: SharedObsSensorSourceBankV1,
    recipient_source_availability: Array,
    recipient_global_slot: Array,
    profile: TeamDeathmatchProfile,
) -> tuple[ActorAction, ScriptedTrace]:
    """Choose one SharedObs action with the immutable profile bound explicitly."""
    return decide_team_deathmatch_with_profile(
        _build_shared_obs_team_deathmatch_policy_facts(
            recipient_observation,
            recipient_action_mask,
            source_bank,
            recipient_source_availability,
            recipient_global_slot,
        ),
        recipient_action_mask,
        key,
        profile,
    )


class _TeamDeathmatchSharedObsPolicy(NamedTuple):
    """Hashable SharedObs callable bound to the same canonical TDM profile."""

    profile: TeamDeathmatchProfile

    def __call__(
        self,
        recipient_observation: Observation,
        recipient_action_mask: ActionMask,
        key: Array,
        source_bank: SharedObsSensorSourceBankV1,
        recipient_source_availability: Array,
        recipient_global_slot: Array,
    ) -> ActorAction:
        """Choose one scalar SharedObs Team Deathmatch action."""
        action, _ = _decide_team_deathmatch_shared_obs_with_profile(
            recipient_observation,
            recipient_action_mask,
            key,
            source_bank,
            recipient_source_availability,
            recipient_global_slot,
            self.profile,
        )
        return action


team_deathmatch_shared_obs_policy = _TeamDeathmatchSharedObsPolicy(
    profile=TEAM_DEATHMATCH_PROFILE
)


__all__ = (
    "SHARED_OBS_ADAPTER_ID",
    "SHARED_OBS_ADAPTER_VERSION",
    "decide_team_deathmatch_shared_obs",
    "team_deathmatch_shared_obs_policy",
)
