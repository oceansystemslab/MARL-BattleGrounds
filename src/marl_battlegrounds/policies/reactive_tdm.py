"""Deterministic five-class reactive TDM decisions from current SharedObs."""

import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.types import (
    AGENT_FEATURE_ACTIVE,
    AGENT_FEATURE_ALIVE,
    AGENT_FEATURE_CLASS_ID,
    AGENT_FEATURE_CURRENT_HEALTH,
    CONTEXT_FEATURE_MAP_HEIGHT,
    CONTEXT_FEATURE_MAP_WIDTH,
    HUNTER_CLASS_ID,
    MAGE_CLASS_ID,
    MAX_AGENTS_PER_TEAM,
    PRIEST_CLASS_ID,
    WARRIOR_CLASS_ID,
    ActionMask,
    Observation,
)
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.reactive_common import (
    MINIMUM_MOVEMENT_FRACTION,
    centers,
    living_candidates,
    lowest_health_row,
    nearest_row,
    refine_movement,
)
from marl_battlegrounds.policies.shared_obs import (
    SharedObsSensorSourceBankV1,
    compose_shared_obs_unit_features,
)

PRIEST_CLOSE_DISTANCE = 1.5
PRIEST_FOLLOW_DISTANCE = 2.0
PRIEST_ENEMY_VISIBLE_APPROACH_DISTANCE = 3.0
PRIEST_ULTIMATE_HEALTH = 30.0
MAGE_DISTANCE = 2.0
HUNTER_CLOSE_DISTANCE = 3.0
HUNTER_FAR_DISTANCE = 3.5
HUNTER_TRAP_DISTANCE = 2.0
WARRIOR_CHARGE_HEALTH = 40.0


def reactive_tdm_controller_descriptor() -> dict[str, object]:
    """Return fresh canonical rule data for launch-bound controller provenance."""
    return {
        "policy_id": "reactive-team-deathmatch-controller",
        "version": 1,
        "information": "same-epoch-shared-obs; recipient exact masks",
        "execution": "deterministic; actor key ignored",
        "candidates": "observed living active positive-health rows",
        "ties": {
            "priest_health": "current HP, maximum HP, global slot",
            "other_targets": "criterion, global slot",
            "movement": "normalized direction alignment, movement action ID",
        },
        "priest": {
            "no_enemy_distance_band": [PRIEST_CLOSE_DISTANCE, PRIEST_FOLLOW_DISTANCE],
            "band_endpoints": "lower exclusive; upper inclusive",
            "no_enemy_movement": (
                "nearest nonself ally: retreat at/below lower, approach above upper, "
                "else Stay; no ally: map center"
            ),
            "enemy_visible_approach_distance": PRIEST_ENEMY_VISIBLE_APPROACH_DISTANCE,
            "enemy_visible_movement": (
                "lowest-HP ally including self; approach nonself ally above distance, "
                "otherwise retreat nearest enemy"
            ),
            "combat": (
                "lowest-HP legal Ultimate ally at/below threshold else lowest-HP "
                "legal Basic ally including self/full health else no-combat"
            ),
            "ultimate_hp_threshold_inclusive": PRIEST_ULTIMATE_HEALTH,
        },
        "mage": {
            "distance": MAGE_DISTANCE,
            "movement": (
                "nearest enemy: approach above, retreat below, Stay at equality"
            ),
            "combat": (
                "legal Burst if legal Basic enemy exists else lowest-HP legal Basic"
            ),
        },
        "rogue": {
            "movement": "approach nearest enemy center even at body contact",
            "combat": "lowest-HP legal Ultimate enemy else lowest-HP legal Basic enemy",
        },
        "hunter": {
            "distance_band": [HUNTER_CLOSE_DISTANCE, HUNTER_FAR_DISTANCE],
            "band_endpoints": "lower exclusive; upper inclusive",
            "movement": (
                "nearest enemy: retreat at/below lower, approach above upper, else Stay"
            ),
            "trap_distance_inclusive": HUNTER_TRAP_DISTANCE,
            "combat": (
                "nearest legal Trap enemy within distance "
                "else lowest-HP legal Basic enemy"
            ),
        },
        "warrior": {
            "movement": "approach nearest enemy center even at body contact",
            "charge_hp_threshold_exclusive": WARRIOR_CHARGE_HEALTH,
            "combat": (
                "lowest-HP legal Charge enemy below threshold "
                "else lowest-HP legal Basic enemy"
            ),
        },
        "movement": {
            "projection": "existing static obstacle/bounds geometry; no body pairs",
            "minimum_stride_fraction_inclusive": MINIMUM_MOVEMENT_FRACTION,
            "search": "eight directions; preserve intended Stay; no route memory",
            "fallback": (
                "Stay if zero intent, no legal direction, or insufficient displacement"
            ),
        },
        "fallbacks": (
            "no enemy: non-Priest map center; no legal combat: no-combat; "
            "dead/inactive: no-op; exact masks override intent"
        ),
    }


def _priest_direction(
    self_features: Array,
    allies: Array,
    ally_living: Array,
    enemies: Array,
    enemy_living: Array,
    recipient_global_slot: Array,
    center_direction: Array,
) -> Array:
    """Keep healing-oriented retreat separate from peaceful nearest-ally spacing."""
    origin = centers(self_features)
    self_row = recipient_global_slot % MAX_AGENTS_PER_TEAM
    ally_row = lowest_health_row(allies, ally_living, break_ties_by_max_health=True)
    ally_delta = centers(allies[ally_row]) - origin
    ally_distance = jnp.sqrt(jnp.sum(jnp.square(ally_delta)))
    enemy_row = nearest_row(enemies, enemy_living, origin)
    retreat_enemy = origin - centers(enemies[enemy_row])
    with_enemy = jnp.where(
        jnp.any(ally_living)
        & (ally_row != self_row)
        & (ally_distance > PRIEST_ENEMY_VISIBLE_APPROACH_DISTANCE),
        ally_delta,
        retreat_enemy,
    )

    other_allies = ally_living & (jnp.arange(MAX_AGENTS_PER_TEAM) != self_row)
    nearest_ally = nearest_row(allies, other_allies, origin)
    follow_delta = centers(allies[nearest_ally]) - origin
    follow_distance = jnp.sqrt(jnp.sum(jnp.square(follow_delta)))
    without_enemy = jnp.where(
        follow_distance <= PRIEST_CLOSE_DISTANCE,
        -follow_delta,
        jnp.where(follow_distance > PRIEST_FOLLOW_DISTANCE, follow_delta, 0.0),
    )
    without_enemy = jnp.where(jnp.any(other_allies), without_enemy, center_direction)
    return jnp.where(jnp.any(enemy_living), with_enemy, without_enemy)


def reactive_tdm_policy(
    recipient_observation: Observation,
    recipient_action_mask: ActionMask,
    actor_key: Array,
    source_bank: SharedObsSensorSourceBankV1,
    recipient_source_availability: Array,
    recipient_global_slot: Array,
) -> ActorAction:
    """Choose one complete team-agnostic action; no RNG or history is consumed."""
    del actor_key
    allies, enemies, ally_visible, enemy_visible = compose_shared_obs_unit_features(
        recipient_observation,
        source_bank,
        recipient_source_availability,
        recipient_global_slot,
    )
    ally_living = living_candidates(allies, ally_visible)
    enemy_living = living_candidates(enemies, enemy_visible)
    self_features = recipient_observation.self_features
    origin = centers(self_features)
    class_id = self_features[AGENT_FEATURE_CLASS_ID].astype(jnp.int32)
    is_priest = class_id == PRIEST_CLASS_ID
    is_mage = class_id == MAGE_CLASS_ID
    is_hunter = class_id == HUNTER_CLASS_ID
    is_warrior = class_id == WARRIOR_CLASS_ID

    enemy_distances = jnp.sqrt(jnp.sum(jnp.square(centers(enemies) - origin), axis=-1))
    nearest_enemy = nearest_row(enemies, enemy_living, origin)
    enemy_delta = centers(enemies[nearest_enemy]) - origin
    enemy_distance = enemy_distances[nearest_enemy]
    mage_direction = jnp.sign(enemy_distance - MAGE_DISTANCE) * enemy_delta
    hunter_direction = jnp.where(
        enemy_distance <= HUNTER_CLOSE_DISTANCE,
        -enemy_delta,
        jnp.where(enemy_distance > HUNTER_FAR_DISTANCE, enemy_delta, 0.0),
    )
    attack_direction = jnp.where(
        is_mage, mage_direction, jnp.where(is_hunter, hunter_direction, enemy_delta)
    )
    map_center = (
        recipient_observation.context_features[
            jnp.asarray([CONTEXT_FEATURE_MAP_WIDTH, CONTEXT_FEATURE_MAP_HEIGHT])
        ]
        * 0.5
    )
    center_direction = map_center - origin
    attack_direction = jnp.where(
        jnp.any(enemy_living), attack_direction, center_direction
    )
    direction = jnp.where(
        is_priest,
        _priest_direction(
            self_features,
            allies,
            ally_living,
            enemies,
            enemy_living,
            recipient_global_slot,
            center_direction,
        ),
        attack_direction,
    )
    move = refine_movement(recipient_observation, recipient_action_mask, direction)

    joint = recipient_action_mask.select_target_use_ultimate_joint_mask
    allies_basic = ally_living & joint[1:6, 0]
    allies_ultimate = ally_living & joint[1:6, 1]
    enemies_basic = enemy_living & joint[6:11, 0]
    enemies_ultimate = enemy_living & joint[6:11, 1]
    enemies_ultimate &= ~is_warrior | (
        enemies[:, AGENT_FEATURE_CURRENT_HEALTH] < WARRIOR_CHARGE_HEALTH
    )
    enemies_ultimate &= ~is_hunter | (enemy_distances <= HUNTER_TRAP_DISTANCE)
    ally_basic_row = lowest_health_row(
        allies, allies_basic, break_ties_by_max_health=True
    )
    ally_ultimate_row = lowest_health_row(
        allies, allies_ultimate, break_ties_by_max_health=True
    )
    enemy_basic_row = lowest_health_row(enemies, enemies_basic)
    enemy_ultimate_row = jnp.where(
        is_hunter,
        nearest_row(enemies, enemies_ultimate, origin),
        lowest_health_row(enemies, enemies_ultimate),
    )

    priest_ultimate = jnp.any(allies_ultimate) & (
        allies[ally_ultimate_row, AGENT_FEATURE_CURRENT_HEALTH]
        <= PRIEST_ULTIMATE_HEALTH
    )
    mage_ultimate = joint[0, 1] & jnp.any(enemies_basic)
    use_ultimate = jnp.where(
        is_priest,
        priest_ultimate,
        jnp.where(is_mage, mage_ultimate, jnp.any(enemies_ultimate)),
    )
    basic_target = jnp.where(
        is_priest,
        jnp.where(jnp.any(allies_basic), ally_basic_row + 1, 0),
        jnp.where(jnp.any(enemies_basic), enemy_basic_row + 6, 0),
    )
    ultimate_target = jnp.where(
        is_priest, ally_ultimate_row + 1, jnp.where(is_mage, 0, enemy_ultimate_row + 6)
    )
    target = jnp.where(use_ultimate, ultimate_target, basic_target)
    participating = (self_features[AGENT_FEATURE_ACTIVE] > 0) & (
        self_features[AGENT_FEATURE_ALIVE] > 0
    )
    return ActorAction(
        jnp.where(participating, move, 0).astype(jnp.int32),
        jnp.where(participating, target, 0).astype(jnp.int32),
        (participating & use_ultimate).astype(jnp.int32),
    )
