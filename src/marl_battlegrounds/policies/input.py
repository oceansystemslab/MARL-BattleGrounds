"""Keep rollout observations compact and expand permitted actor inputs on demand.

Observations stores the ten base observation rows and a source-permission matrix.
build_team_actor_input expands one team's permitted inputs when needed.
build_actor_input provides the compatible all-actor route with default permissions.
A source is another actor's sensor row; it is not access to hidden Core state.
Policy masks and random keys remain separate arguments.
"""

from numbers import Integral
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.types import (
    AGENT_FEATURE_ACTIVE,
    AGENT_FEATURE_ALIVE,
    EnvConfig,
    Observation,
)
from marl_battlegrounds.policies.shared_obs import (
    SharedObsSensorSourceBankV2,
    build_default_shared_obs_information_availability,
    build_shared_obs_team_source_bank_from_base_rows,
    mask_source_bank_for_recipient,
)


class ActorInput(NamedTuple):
    """Bundle an actor's observation with the sensor rows it may receive.

    Attributes
    ----------
    observation : Observation
        The actor's own current observation. The team builder adds five recipient
        rows; build_actor_input adds ten. Select one row before a scalar policy
        call. An outer vmap adds the environment axis.
    source_bank : SharedObsSensorSourceBankV2
        Shared sensor data with unavailable source rows cleared. One actor's
        feature, visibility and objective arrays have shapes (5, 10, 58),
        (5, 10) and (5, 8, 12). Builders add their leading recipient axis.
    source_availability : Array
        Boolean permission for five own-team sources. Shape (5,) for one actor,
        (5, 5) for one team, or (10, 5) for all actors. This is permission, not
        a visibility claim: an admitted source can have no current sensor data.

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


def build_team_actor_input(observations: Observations, team: int) -> ActorInput:
    """Expand one team's supplied permissions into separate actor inputs.

    Parameters
    ----------
    observations : Observations
        One game's ten current observation rows and Boolean source permissions
        shaped (10, 10). Permissions must already be a valid same-team, non-self
        subset. This helper never replaces them with broader defaults.
    team : int
        Static routing choice: 0 selects Team A and 1 selects Team B. Pass a
        Python integer, not a traced array. Booleans are not accepted.

    Returns
    -------
    ActorInput
        Five recipient observations, Boolean permissions (5, 5), and redacted
        source banks. Bank feature, visibility and objective shapes are
        (5, 5, 10, 58), (5, 5, 10), and (5, 5, 8, 12). Features are float32;
        visibility is Boolean. Dead, inactive, hidden and forbidden source
        material is cleared by the existing sensor and redaction authorities.

    Raises
    ------
    ValueError
        team is not an integer 0 or 1, or is a Boolean.

    Notes
    -----
    Only the requested team's bank is built. The five recipient rows retain
    separate information rights; they do not form a shared private observation.
    Inputs stay unchanged and numerical values stay on device. Shapes and
    permission validity are caller preconditions. Use jit with a static team
    and vmap to add environment batches. Keep the expanded result transient
    instead of storing it for every rollout step.
    """
    if isinstance(team, bool) or not isinstance(team, Integral) or team not in (0, 1):
        raise ValueError("team must be the integer 0 or 1")
    start = int(team) * 5

    def take(value: Array) -> Array:
        """Select this team's five routing rows without changing feature axes."""
        return value[start : start + 5]

    observation = jax.tree.map(take, observations.observation)
    source_is_living = (observation.self_features[:, AGENT_FEATURE_ACTIVE] > 0.0) & (
        observation.self_features[:, AGENT_FEATURE_ALIVE] > 0.0
    )
    bank = build_shared_obs_team_source_bank_from_base_rows(
        observation.ally_unit_features,
        observation.enemy_unit_features,
        observation.objective_features,
        observation.ally_visibility_mask,
        observation.enemy_visibility_mask,
        source_is_living,
    )
    availability = observations.source_availability[
        start : start + 5, start : start + 5
    ]
    return ActorInput(
        observation,
        jax.vmap(mask_source_bank_for_recipient, in_axes=(None, 0))(bank, availability),
        availability,
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
    observations = build_observations(observation, config)
    team_a = build_team_actor_input(observations, 0)
    team_b = build_team_actor_input(observations, 1)

    def join(a: Array, b: Array) -> Array:
        """Join Team A's five recipients followed by Team B's five recipients."""
        return jnp.concatenate((a, b), axis=0)

    return ActorInput(
        observation=observation,
        source_bank=jax.tree.map(join, team_a.source_bank, team_b.source_bank),
        source_availability=join(
            team_a.source_availability, team_b.source_availability
        ),
    )


__all__ = (
    "ActorInput",
    "Observations",
    "build_actor_input",
    "build_observations",
    "build_team_actor_input",
)
