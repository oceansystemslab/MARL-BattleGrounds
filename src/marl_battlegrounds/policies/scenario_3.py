"""Scenario 3: deterministic Rogue pursuit around currently observed bodies."""

import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.axis_mappings import (
    UNIT_DIRECTION_VECTOR_BY_MOVEMENT_ACTION_ARRAY,
)
from marl_battlegrounds.core.geometry import project_movement_with_geometry
from marl_battlegrounds.core.types import (
    AGENT_FEATURE_ACTIVE,
    AGENT_FEATURE_ALIVE,
    AGENT_FEATURE_BASIC_INTERACTION_RADIUS,
    AGENT_FEATURE_CLASS_ID,
    AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED,
    AGENT_FEATURE_RADIUS,
    CONTEXT_FEATURE_MAP_HEIGHT,
    CONTEXT_FEATURE_MAP_WIDTH,
    MAX_AGENT_SLOTS,
    MAX_AGENTS_PER_TEAM,
    MOVE_STAY,
    ROGUE_CLASS_ID,
    ActionMask,
    Observation,
)
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.reactive_common import (
    MINIMUM_MOVEMENT_FRACTION,
    centers,
    living_candidates,
    lowest_health_row,
)
from marl_battlegrounds.policies.shared_obs import (
    SharedObsSensorSourceBankV1,
    compose_shared_obs_unit_features,
)


def scenario_3_controller_descriptor() -> dict[str, object]:
    """Return fresh rule data for launch-bound specialist provenance."""
    return {
        "policy_id": "scenario-3-pressure-controller",
        "version": 1,
        "information": "same-epoch SharedObs and recipient exact masks",
        "execution": "deterministic; actor key ignored",
        "classes": "Rogue only; other classes Stay/no-combat even after respawn",
        "candidates": "observed active living positive-health enemies",
        "pursuit": "lowest current HP; recomputed every decision; no target lock",
        "combat": (
            "within Basic radius: lowest-HP legal Ultimate enemy else lowest-HP "
            "legal Basic enemy else no-combat; independent of pursuit"
        ),
        "ties": "health then global slot; endpoint distance then movement action ID",
        "movement": {
            "projection": "existing static obstacle/bounds geometry; no body pairs",
            "minimum_stride_fraction_inclusive": MINIMUM_MOVEMENT_FRACTION,
            "blockers": "observed living allies/enemies excluding self and prey",
            "clearance": "projected displacement segment; radius sum; tangency allowed",
            "overlap_escape": "never deepen initial overlap and finish farther away",
            "selection": "closest safe moving endpoint to prey; detours may retreat",
            "assumption": (
                "other observed bodies stationary; not future action prediction"
            ),
            "search": "eight directions only; no lookahead or route memory",
        },
        "fallbacks": "no prey or useful safe move: Stay; dead/inactive: no-op",
        "action_priority": "Ultimate replaces Basic; exact masks override preferences",
    }


def _body_clear_moves(
    origin: Array,
    endpoints: Array,
    radius: Array,
    bodies: Array,
    body_mask: Array,
) -> Array:
    """Screen eight displacement segments against fixed observed body discs.

    This is a conservative local steering preference, not simulator collision
    resolution. Existing overlaps allow only outward motion; the target and
    self have already been removed from ``body_mask`` by the caller.
    """
    travel = endpoints - origin
    start = origin - centers(bodies)
    travel_squared = jnp.sum(jnp.square(travel), axis=-1)
    initial_alignment = jnp.sum(travel[:, None, :] * start[None, :, :], axis=-1)
    fraction = jnp.clip(
        -initial_alignment / jnp.where(travel_squared > 0, travel_squared, 1)[:, None],
        0,
        1,
    )
    closest = start[None, :, :] + fraction[:, :, None] * travel[:, None, :]
    closest_squared = jnp.sum(jnp.square(closest), axis=-1)
    start_squared = jnp.sum(jnp.square(start), axis=-1)
    end_squared = jnp.sum(
        jnp.square(endpoints[:, None, :] - centers(bodies)[None, :, :]), axis=-1
    )
    required_squared = jnp.square(radius + bodies[:, AGENT_FEATURE_RADIUS])
    escaping = (initial_alignment >= 0) & (end_squared > start_squared)
    clear = jnp.where(
        start_squared < required_squared,
        escaping,
        closest_squared >= required_squared,
    )
    return jnp.all(~body_mask[None, :] | clear, axis=1)


def _body_aware_move(
    observation: Observation,
    action_mask: ActionMask,
    prey_center: Array,
    bodies: Array,
    body_mask: Array,
) -> Array:
    """Choose the closest safe moving endpoint, even if a detour retreats."""
    origin = centers(observation.self_features)
    speed = observation.self_features[AGENT_FEATURE_EFFECTIVE_MOVEMENT_SPEED]
    radius = observation.self_features[AGENT_FEATURE_RADIUS]
    directions = UNIT_DIRECTION_VECTOR_BY_MOVEMENT_ACTION_ARRAY[1:]
    positions = jnp.broadcast_to(origin, (MAX_AGENT_SLOTS, 2))
    deltas = jnp.zeros_like(positions).at[:8].set(directions * speed)
    active = jnp.arange(MAX_AGENT_SLOTS) < 8
    no_bodies = jnp.zeros(MAX_AGENT_SLOTS, dtype=jnp.bool_)
    endpoints = project_movement_with_geometry(
        positions,
        jnp.full(MAX_AGENT_SLOTS, radius, dtype=jnp.float32),
        deltas,
        active,
        active,
        observation.context_features[CONTEXT_FEATURE_MAP_WIDTH],
        observation.context_features[CONTEXT_FEATURE_MAP_HEIGHT],
        observation.map_obstacle_features,
        no_bodies,
        no_bodies,
        agent_agent_overlap_projection_passes=0,
    )[:8]
    displacement = jnp.sqrt(jnp.sum(jnp.square(endpoints - origin), axis=-1))
    admissible = (
        action_mask.move_mask[1:]
        & (speed > 0)
        & (displacement >= MINIMUM_MOVEMENT_FRACTION * speed)
        & _body_clear_moves(origin, endpoints, radius, bodies, body_mask)
    )
    distance_squared = jnp.sum(jnp.square(endpoints - prey_center), axis=-1)
    move = 1 + jnp.argmin(jnp.where(admissible, distance_squared, jnp.inf))
    return jnp.where(jnp.any(admissible), move, MOVE_STAY).astype(jnp.int32)


def scenario_3_policy(
    recipient_observation: Observation,
    recipient_action_mask: ActionMask,
    actor_key: Array,
    source_bank: SharedObsSensorSourceBankV1,
    recipient_source_availability: Array,
    recipient_global_slot: Array,
) -> ActorAction:
    """Pursue vulnerable prey while independently choosing legal Rogue combat."""
    del actor_key
    allies, enemies, ally_visible, enemy_visible = compose_shared_obs_unit_features(
        recipient_observation,
        source_bank,
        recipient_source_availability,
        recipient_global_slot,
    )
    ally_living = living_candidates(allies, ally_visible)
    enemy_living = living_candidates(enemies, enemy_visible)
    prey_row = lowest_health_row(enemies, enemy_living)
    body_mask = jnp.concatenate((ally_living, enemy_living))
    body_mask = body_mask.at[recipient_global_slot % MAX_AGENTS_PER_TEAM].set(False)
    body_mask = body_mask.at[MAX_AGENTS_PER_TEAM + prey_row].set(False)
    move = _body_aware_move(
        recipient_observation,
        recipient_action_mask,
        centers(enemies[prey_row]),
        jnp.concatenate((allies, enemies)),
        body_mask,
    )
    own = recipient_observation.self_features
    distances_squared = jnp.sum(jnp.square(centers(enemies) - centers(own)), axis=-1)
    within_basic = distances_squared <= jnp.square(
        own[AGENT_FEATURE_BASIC_INTERACTION_RADIUS]
    )
    joint = recipient_action_mask.select_target_use_ultimate_joint_mask
    basic = enemy_living & within_basic & joint[6:11, 0]
    ultimate = enemy_living & within_basic & joint[6:11, 1]
    use_ultimate = jnp.any(ultimate)
    combat_row = lowest_health_row(enemies, jnp.where(use_ultimate, ultimate, basic))
    target = jnp.where(use_ultimate | jnp.any(basic), combat_row + 6, 0)
    participating = (
        (own[AGENT_FEATURE_ACTIVE] > 0)
        & (own[AGENT_FEATURE_ALIVE] > 0)
        & (own[AGENT_FEATURE_CLASS_ID] == ROGUE_CLASS_ID)
    )
    return ActorAction(
        jnp.where(participating & jnp.any(enemy_living), move, MOVE_STAY).astype(
            jnp.int32
        ),
        jnp.where(participating, target, 0).astype(jnp.int32),
        (participating & use_ultimate).astype(jnp.int32),
    )
