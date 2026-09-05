"""Scenario 1: reactive Priest, Rogue, and Mage pressure from SharedObs only."""

import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.types import (
    AGENT_FEATURE_ACTIVE,
    AGENT_FEATURE_ALIVE,
    AGENT_FEATURE_CLASS_ID,
    AGENT_FEATURE_CURRENT_HEALTH,
    MAGE_CLASS_ID,
    MAX_AGENTS_PER_TEAM,
    PRIEST_CLASS_ID,
    ROGUE_CLASS_ID,
    ActionMask,
    Observation,
)
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.scenario_controllers.common import (
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
PRIEST_FAR_DISTANCE = 3.0
PRIEST_ULTIMATE_HEALTH = 30.0
MAGE_DISTANCE = 2.0


def scenario_1_controller_descriptor() -> dict[str, object]:
    """Return fresh canonical rule data for launch-bound controller provenance."""
    return {
        "policy_id": "scenario-1-pressure-controller",
        "version": 1,
        "information": "same-epoch-shared-obs; recipient exact masks",
        "execution": "deterministic; actor key ignored",
        "candidates": "observed living active positive-health rows",
        "ties": {
            "priest": "current HP, maximum HP, global slot",
            "other_targets": "criterion, global slot",
            "movement": "normalized direction alignment, movement action ID",
        },
        "priest": {
            "distance_band": [PRIEST_CLOSE_DISTANCE, PRIEST_FAR_DISTANCE],
            "band_endpoints": "lower exclusive; upper inclusive",
            "movement": (
                "lowest-HP ally including self; self: retreat enemy else Stay; "
                "other farther than upper: approach; within upper with enemy: "
                "retreat nearest enemy; no enemy at/below lower: retreat ally; "
                "otherwise Stay"
            ),
            "combat": (
                "lowest-HP legal Ultimate ally at threshold else lowest-HP legal "
                "Basic ally including self/full health else no-combat"
            ),
            "ultimate_hp_threshold_inclusive": PRIEST_ULTIMATE_HEALTH,
        },
        "rogue": {
            "movement": "approach nearest enemy center even at body contact",
            "combat": (
                "lowest-HP legal Ultimate enemy else lowest-HP legal Basic enemy "
                "else no-combat"
            ),
        },
        "mage": {
            "distance": MAGE_DISTANCE,
            "movement": (
                "nearest enemy: approach above distance, retreat below, "
                "Stay at equality"
            ),
            "combat": (
                "legal Burst when legal Basic enemy exists else lowest-HP legal "
                "Basic enemy else no-combat"
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
            "no enemy: Mage/Rogue Stay; dead/inactive/unsupported: no-op; "
            "exact masks override intent"
        ),
    }


def _priest_direction(
    self_features: Array,
    allies: Array,
    ally_living: Array,
    enemies: Array,
    enemy_living: Array,
    recipient_global_slot: Array,
) -> Array:
    """Express the ordered approach/retreat rules without movement quantization."""
    origin = centers(self_features)
    ally_row = lowest_health_row(allies, ally_living, break_ties_by_max_health=True)
    ally_delta = centers(allies[ally_row]) - origin
    enemy_row = nearest_row(enemies, enemy_living, origin)
    retreat_enemy = origin - centers(enemies[enemy_row])
    enemy_present = jnp.any(enemy_living)
    ally_distance = jnp.sqrt(jnp.sum(jnp.square(ally_delta)))
    ally_is_self = ally_row == recipient_global_slot % MAX_AGENTS_PER_TEAM
    zero = jnp.zeros(2, dtype=jnp.float32)
    without_enemy = jnp.where(ally_distance <= PRIEST_CLOSE_DISTANCE, -ally_delta, zero)
    other_ally = jnp.where(
        ally_distance > PRIEST_FAR_DISTANCE,
        ally_delta,
        jnp.where(enemy_present, retreat_enemy, without_enemy),
    )
    return jnp.where(
        ally_is_self | ~jnp.any(ally_living),
        jnp.where(enemy_present, retreat_enemy, zero),
        other_ally,
    )


def scenario_1_policy(
    recipient_observation: Observation,
    recipient_action_mask: ActionMask,
    actor_key: Array,
    source_bank: SharedObsSensorSourceBankV1,
    recipient_source_availability: Array,
    recipient_global_slot: Array,
) -> ActorAction:
    """Choose one complete same-epoch action; no key or action history is read."""
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
    class_id = self_features[AGENT_FEATURE_CLASS_ID].astype(jnp.int32)
    is_priest = class_id == PRIEST_CLASS_ID
    is_mage = class_id == MAGE_CLASS_ID
    is_rogue = class_id == ROGUE_CLASS_ID

    nearest_enemy = nearest_row(enemies, enemy_living, centers(self_features))
    enemy_delta = centers(enemies[nearest_enemy]) - centers(self_features)
    enemy_distance = jnp.sqrt(jnp.sum(jnp.square(enemy_delta)))
    mage_direction = jnp.sign(enemy_distance - MAGE_DISTANCE) * enemy_delta
    attack_direction = jnp.where(is_mage, mage_direction, enemy_delta)
    attack_direction = jnp.where(jnp.any(enemy_living), attack_direction, 0.0)
    direction = jnp.where(
        is_priest,
        _priest_direction(
            self_features,
            allies,
            ally_living,
            enemies,
            enemy_living,
            recipient_global_slot,
        ),
        attack_direction,
    )
    move = refine_movement(recipient_observation, recipient_action_mask, direction)

    joint = recipient_action_mask.select_target_use_ultimate_joint_mask
    allies_basic = ally_living & joint[1:6, 0]
    allies_ultimate = ally_living & joint[1:6, 1]
    enemies_basic = enemy_living & joint[6:11, 0]
    enemies_ultimate = enemy_living & joint[6:11, 1]
    ally_basic_row = lowest_health_row(
        allies, allies_basic, break_ties_by_max_health=True
    )
    ally_ultimate_row = lowest_health_row(
        allies, allies_ultimate, break_ties_by_max_health=True
    )
    enemy_basic_row = lowest_health_row(enemies, enemies_basic)
    enemy_ultimate_row = lowest_health_row(enemies, enemies_ultimate)

    priest_ultimate = jnp.any(allies_ultimate) & (
        allies[ally_ultimate_row, AGENT_FEATURE_CURRENT_HEALTH]
        <= PRIEST_ULTIMATE_HEALTH
    )
    rogue_ultimate = jnp.any(enemies_ultimate)
    mage_ultimate = joint[0, 1] & jnp.any(enemies_basic)
    use_ultimate = jnp.where(
        is_priest, priest_ultimate, jnp.where(is_mage, mage_ultimate, rogue_ultimate)
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
    participating = (
        (self_features[AGENT_FEATURE_ACTIVE] > 0)
        & (self_features[AGENT_FEATURE_ALIVE] > 0)
        & (is_priest | is_mage | is_rogue)
    )
    return ActorAction(
        jnp.where(participating, move, 0).astype(jnp.int32),
        jnp.where(participating, target, 0).astype(jnp.int32),
        (participating & use_ultimate).astype(jnp.int32),
    )
