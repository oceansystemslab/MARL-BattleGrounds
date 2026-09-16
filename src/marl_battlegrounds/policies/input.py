"""Keep rollout observations compact and expand permitted actor inputs on demand.

Observations stores the ten base observation rows and a source-permission matrix.
build_actor_input constructs the larger per-actor SharedObs banks when needed.
A source is another actor's sensor row; it is not access to hidden Core state.
Policy masks and random keys remain separate arguments.
"""

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
    """Bundle an actor's observation with the sensor rows it may receive.

    Attributes
    ----------
    observation : Observation
        The actor's own current observation. In the result of build_actor_input,
        each leaf has a leading axis of ten actors; select one row before a
        scalar policy call.
    source_bank : SharedObsSensorSourceBankV2
        Shared sensor data with unavailable source rows cleared. One actor's
        feature, visibility and objective arrays have shapes (5, 10, 58),
        (5, 10) and (5, 8, 12). The builder adds a leading ten-actor axis.
    source_availability : Array
        Boolean permission for five own-team sources. Shape (5,) for one actor,
        or (10, 5) in the builder result. This is permission, not a visibility
        claim: an admitted source can have no current sensor data.

    Notes
    -----
    This immutable tuple is JAX-compatible. It contains no action masks or random
    keys. Returning actions for several actors does not authorize sharing their
    private observation rows with one another.
    """

    observation: Observation
    source_bank: SharedObsSensorSourceBankV2
    source_availability: Array


class Observations(NamedTuple):
    """Store compact observations and source permissions for one game or batch.

    Attributes
    ----------
    observation : Observation
        Core's current observation rows. One game has a leading ten-actor axis.
        A native environment batch adds a leading game axis to every leaf.
    source_availability : Array
        Boolean array of shape (10, 10), or (B, 10, 10) for B games. Axes are
        recipient and source in the simulator's routing order. Only permitted
        active same-team sources other than self may be enabled.

    Notes
    -----
    This immutable tuple avoids storing a separate expanded source bank for every
    actor at every rollout step. It is data for policy routing, not an authorization
    for a scalar actor to inspect every row. The policy executor selects and clears
    rows before delivering them to each actor.
    """

    observation: Observation
    source_availability: Array


def build_observations(observation: Observation, config: EnvConfig) -> Observations:
    """Pair base observation rows with the default source permissions.

    Parameters
    ----------
    observation : Observation
        Current observation for all ten actor slots of one game.
    config : EnvConfig
        Configuration for that game. Its active mask and team IDs determine
        which sources may be shared.

    Returns
    -------
    Observations
        The supplied observation and a Boolean (10, 10) permission matrix.
        Active same-team sources are allowed, excluding the recipient itself.

    Notes
    -----
    This function does not expand source banks or inspect privileged state. It
    does not check that observation and config belong to the same game; the caller
    must keep them paired. Use vmap for several games. Inputs are unchanged.
    """
    return Observations(
        observation,
        build_default_shared_obs_information_availability(
            config.agent_profile.active_mask, config.agent_profile.team_ids
        ),
    )


def build_actor_input(observation: Observation, config: EnvConfig) -> ActorInput:
    """Build permitted SharedObs inputs for all ten actors in one game.

    Parameters
    ----------
    observation : Observation
        Current ten-actor observations. Each actor's feature rows are already
        expressed in its observation order.
    config : EnvConfig
        Matching game configuration. Active slots and team IDs supply the
        default source permissions.

    Returns
    -------
    ActorInput
        A ten-actor bundle. Source features have shape (10, 5, 10, 58),
        visibility (10, 5, 10), objectives (10, 5, 8, 12), and source permission
        (10, 5). Unavailable source data is zeroed before actor delivery.

    Notes
    -----
    The fixed Team A/Team B routing blocks must match the supplied configuration.
    This helper expands banks for policy application; retaining its output at every
    rollout step costs more space than retaining Observations. Inputs stay dynamic
    under jit. Use vmap to add a game axis. No action mask or hidden simulator state
    is constructed here.
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
        """Flatten the two-team and five-recipient axes into ten actor rows.

        The remaining source and feature axes keep their order and values.
        """
        return value.reshape((10, *value.shape[2:]))

    return ActorInput(
        observation=observation,
        source_bank=jax.tree.map(actor_rows, authorized_banks),
        source_availability=team_availability.reshape((10, 5)),
    )


__all__ = ("ActorInput", "Observations", "build_actor_input", "build_observations")
