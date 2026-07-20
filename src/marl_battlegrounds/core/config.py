"""Host-facing fixed-slot episode-profile resolution."""

import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.combat import (
    get_base_movement_speed_by_class_ids,
    get_basic_interaction_radius_by_class_ids,
    get_body_radius_by_class_ids,
    get_max_health_by_class_ids,
    get_observation_radius_by_class_ids,
    get_ultimate_interaction_radius_by_class_ids,
)
from marl_battlegrounds.core.types import (
    MAX_AGENTS_PER_TEAM,
    NEUTRAL_CLASS_ID,
    NO_TEAM_ID,
    TEAM_A_ID,
    TEAM_B_ID,
    ResolvedAgentProfile,
)


def resolve_agent_profile(
    requested_class_ids: Array, team_sizes: Array
) -> ResolvedAgentProfile:
    """Resolve requested team rosters into immutable padded slot arrays.

    ``requested_class_ids`` has shape ``(MAX_AGENT_SLOTS,)`` and ``team_sizes``
    has shape ``(2,)``. Padded slots receive neutral class/catalog values and
    ``NO_TEAM_ID``; active team blocks receive their explicit public team IDs.
    Host-side input validation remains future scope.
    """
    # Derive active mask
    team_local_indices = jnp.arange(MAX_AGENTS_PER_TEAM)

    team_a_active_mask = team_local_indices < team_sizes[0]
    team_b_active_mask = team_local_indices < team_sizes[1]

    active_mask = jnp.hstack((team_a_active_mask, team_b_active_mask))

    # Derive class ids
    class_ids = jnp.where(active_mask, requested_class_ids, NEUTRAL_CLASS_ID)

    # Derive team ids
    team_a_team_ids = jnp.where(team_a_active_mask, TEAM_A_ID, NO_TEAM_ID)
    team_b_team_ids = jnp.where(team_b_active_mask, TEAM_B_ID, NO_TEAM_ID)
    team_ids = jnp.hstack((team_a_team_ids, team_b_team_ids), dtype=jnp.int32)

    agent_radii = get_body_radius_by_class_ids(class_ids)

    base_movement_speeds = get_base_movement_speed_by_class_ids(class_ids)

    observation_radii = get_observation_radius_by_class_ids(class_ids)

    basic_interaction_radii = get_basic_interaction_radius_by_class_ids(class_ids)

    ultimate_interaction_radii = get_ultimate_interaction_radius_by_class_ids(class_ids)

    max_health = get_max_health_by_class_ids(class_ids)

    return ResolvedAgentProfile(
        class_ids,
        team_ids,
        active_mask,
        agent_radii,
        base_movement_speeds,
        observation_radii,
        basic_interaction_radii,
        ultimate_interaction_radii,
        max_health,
    )
