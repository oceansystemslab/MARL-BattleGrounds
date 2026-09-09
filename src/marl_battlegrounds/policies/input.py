"""Structured actor inputs using the existing SharedObs authorization boundary."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.types import MAX_AGENT_SLOTS, EnvConfig, Observation
from marl_battlegrounds.policies.shared_obs import (
    SharedObsSensorSourceBankV1,
    build_default_shared_obs_information_availability,
    build_shared_obs_sensor_source_bank,
    mask_source_bank_for_recipient,
)


class ActorInput(NamedTuple):
    """One actor's existing observation and admitted SharedObs source material.

    ``build_actor_input`` returns a leading ten-actor axis on every leaf. Taking
    one actor row gives the observation, source bank, availability row and global
    slot already accepted by the six-argument SharedObs policy interface. Masks
    and RNG keys remain separate policy arguments. NoSharedObs stays unchanged.
    """

    observation: Observation
    source_bank: SharedObsSensorSourceBankV1
    source_availability: Array
    global_slot: Array


class Observations(NamedTuple):
    """Compact all-actor observations retained by a trainer or rollout.

    Source-bank rows are derived from these existing base observations at policy
    application, then redacted before the individual actor is called. Keeping
    ten expanded banks in every training timestep would duplicate sensor data.
    """

    observation: Observation
    source_availability: Array


def build_observations(observation: Observation, config: EnvConfig) -> Observations:
    """Retain base sensor truth and authorization without expanded actor banks."""
    return Observations(
        observation,
        build_default_shared_obs_information_availability(
            config.agent_profile.active_mask, config.agent_profile.team_ids
        ),
    )


def build_actor_input(observation: Observation, config: EnvConfig) -> ActorInput:
    """Package authorized inputs for all ten slots without inspecting Core state.

    The source bank has actor/source/candidate axes; availability has actor/source
    axes. External ``vmap`` adds an environment axis. Configuration and observations
    remain ordinary dynamic PyTrees, and no action mask is rebuilt here.
    """
    availability = build_default_shared_obs_information_availability(
        config.agent_profile.active_mask, config.agent_profile.team_ids
    )
    source_bank = build_shared_obs_sensor_source_bank(observation)
    authorized_banks = jax.vmap(mask_source_bank_for_recipient, in_axes=(None, 0))(
        source_bank, availability
    )
    return ActorInput(
        observation=observation,
        source_bank=authorized_banks,
        source_availability=availability,
        global_slot=jnp.arange(MAX_AGENT_SLOTS, dtype=jnp.int32),
    )


__all__ = ("ActorInput", "Observations", "build_actor_input", "build_observations")
