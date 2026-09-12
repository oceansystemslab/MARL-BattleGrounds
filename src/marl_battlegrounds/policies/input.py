"""Structured actor inputs using the existing SharedObs authorization boundary."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.types import EnvConfig, Observation
from marl_battlegrounds.policies.shared_obs import (
    SharedObsSensorSourceBankV2,
    build_default_shared_obs_information_availability,
    build_shared_obs_sensor_source_bank,
    mask_source_bank_for_recipient,
)


class ActorInput(NamedTuple):
    """One actor's existing observation and admitted SharedObs source material.

    ``build_actor_input`` returns a leading ten-actor routing axis on every leaf.
    Taking one actor row gives only its observation, five-source bank and local
    availability row. Masks and RNG keys remain separate policy arguments.
    """

    observation: Observation
    source_bank: SharedObsSensorSourceBankV2
    source_availability: Array


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
    team_availability = jnp.stack((availability[:5, :5], availability[5:, 5:]))
    authorized_banks = jax.vmap(
        jax.vmap(mask_source_bank_for_recipient, in_axes=(None, 0))
    )(source_bank, team_availability)

    def actor_rows(value: Array) -> Array:
        return value.reshape((10, *value.shape[2:]))

    return ActorInput(
        observation=observation,
        source_bank=jax.tree.map(actor_rows, authorized_banks),
        source_availability=team_availability.reshape((10, 5)),
    )


__all__ = ("ActorInput", "Observations", "build_actor_input", "build_observations")
