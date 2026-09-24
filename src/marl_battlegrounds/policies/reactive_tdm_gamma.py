"""Choose TDM-GAMMA actions: BETA plus spawn search, Trap holds and Trap order.

GAMMA plays exactly like BETA except for three rules:

1. Search. With no enemy in view, Warriors, Mages, Hunters and Rogues walk to
   the middle of the enemy's five spawn pads instead of the map centre, and
   stay once they are within Core's geometry tolerance of it. Priests keep
   BETA's behaviour.
2. Trap hold. GAMMA never damages an enemy that has 2 or more Hunter Trap ticks
   left: no Basic, Warrior Charge, Rogue Ultimate or Mage Burst reason on it.
   At 1 tick the Trap is ending anyway, so GAMMA hits it again.
3. Trap order. A Hunter starts a Trap only on an untrapped enemy (0 Trap
   ticks) whose Trap bit is legal in its exact mask. Among those it takes the
   first class in the order Priest, Mage, Rogue, Warrior, Hunter (lowest
   health, then lowest row) and stays still that tick. With no legal Trap but
   its Trap ready, it walks toward the first untrapped enemy in view in that
   order while still shooting. Otherwise it plays BETA's Hunter and does not
   Trap; ALPHA's 2-unit Trap rule is not used.

reactive_tdm_gamma_policy chooses actions from one actor's permitted inputs.
reactive_tdm_gamma_controller_descriptor returns the versioned rule data used to
identify the controller in recordings. The policy is registered as
policy("tdm-gamma") in evaluation.policy_execution, and the DevClient runs it
for either team's ``tdm_gamma`` choice. These are diagnostic controller rules,
not trained behaviour or a path-planning guarantee.
"""

import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.geometry import GEOMETRY_TOLERANCE
from marl_battlegrounds.core.types import (
    AGENT_FEATURE_ACTIVE,
    AGENT_FEATURE_ALIVE,
    AGENT_FEATURE_BASIC_INTERACTION_RADIUS,
    AGENT_FEATURE_CLASS_ID,
    AGENT_FEATURE_CURRENT_HEALTH,
    AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION,
    AGENT_FEATURE_STUN_ROGUE_POISON_DURATION,
    AGENT_FEATURE_STUN_WARRIOR_CHARGE_DURATION,
    AGENT_FEATURE_ULTIMATE_COOLDOWN_REMAINING,
    CONTEXT_FEATURE_MAP_HEIGHT,
    CONTEXT_FEATURE_MAP_WIDTH,
    HUNTER_CLASS_ID,
    MAGE_CLASS_ID,
    MAX_AGENTS_PER_TEAM,
    PRIEST_CLASS_ID,
    ROGUE_CLASS_ID,
    WARRIOR_CLASS_ID,
    ActionMask,
    Observation,
)
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.reactive_common import (
    _body_aware_move,  # pyright: ignore[reportPrivateUsage]
    centers,
    living_candidates,
    lowest_health_row,
    nearest_row,
    refine_movement,
)
from marl_battlegrounds.policies.reactive_tdm_alpha import (
    HUNTER_CLOSE_DISTANCE,
    HUNTER_FAR_DISTANCE,
    MAGE_DISTANCE,
    PRIEST_ULTIMATE_HEALTH,
    WARRIOR_CHARGE_HEALTH,
    _priest_direction,  # pyright: ignore[reportPrivateUsage]
)
from marl_battlegrounds.policies.reactive_tdm_beta import (
    reactive_tdm_beta_controller_descriptor,
)
from marl_battlegrounds.policies.shared_obs import (
    SharedObsSensorSourceBankV2,
    compose_shared_obs_unit_features,
)

TRAP_DAMAGE_MAX_TICKS = 1
"""Most Trap ticks an enemy may have and still take GAMMA damage (inclusive)."""

TRAP_ORDER = (
    PRIEST_CLASS_ID,
    MAGE_CLASS_ID,
    ROGUE_CLASS_ID,
    WARRIOR_CLASS_ID,
    HUNTER_CLASS_ID,
)
"""Core class IDs in the order a GAMMA Hunter picks new Trap targets."""


def reactive_tdm_gamma_controller_descriptor() -> dict[str, object]:
    """Return fresh versioned rule data describing GAMMA.

    Returns
    -------
    dict[str, object]
        GAMMA's identity, version, information rights and its three rules.
        ``inherited_controller`` is an unmodified fresh BETA descriptor (which
        nests ALPHA's): GAMMA runs BETA's rules for everything its own three
        rules do not change. Nested containers are created for this call.

    Notes
    -----
    Recording code uses this data as controller identity, and training exposure
    records compare ``inherited_controller`` with protected scenario
    controllers. Any behaviour change needs a new version. It is descriptive
    data, not a configurable controller or a performance claim.
    """
    return {
        "policy_id": "reactive-team-deathmatch-gamma-controller",
        "version": 2,
        "information": (
            "same-epoch SharedObs, the actor's own status and the map size, the "
            "opposing spawn pads, the actor's own public spawn-shield time, "
            "recipient exact masks"
        ),
        "execution": "deterministic; actor key ignored; no memory",
        "inherited_controller": reactive_tdm_beta_controller_descriptor(),
        "base": "BETA for every choice the three rules below do not change",
        "search": (
            "no visible living enemy: non-Priests move toward the mean of all five "
            "opposing spawn pads, using the full vector and simple steering; "
            "within geometry tolerance of the mean, Stay; Priests keep BETA"
        ),
        "trap_hold": {
            "trap_ticks": "observed Hunter Trap stun duration on the enemy row",
            "rule": (
                "non-Priests never choose damage on an enemy with more trap_ticks "
                "than the limit: Basic, Warrior Charge, Rogue Ultimate and Mage "
                "Burst qualification all skip it"
            ),
            "damage_max_trap_ticks_inclusive": TRAP_DAMAGE_MAX_TICKS,
            "unchanged": "movement targets, spacing, Rogue prey and Priest healing",
        },
        "hunter_trap": {
            "order": ["Priest", "Mage", "Rogue", "Warrior", "Hunter"],
            "new_trap": "only enemies with trap_ticks exactly 0",
            "cast": (
                "a legal new-Trap enemy exists: Trap the first class present in "
                "the order, lowest current HP then global slot; full mask range; "
                "Stay on that tick"
            ),
            "approach": (
                "no legal cast, Trap ready and a new-Trap enemy visible: walk to "
                "the first class present in the order, lowest HP, simple "
                "steering; keep the Trap-hold Basic"
            ),
            "ready": (
                "own cooldown 0, no own stun, no own spawn shield, active and alive"
            ),
            "otherwise": "BETA Hunter movement; Trap-hold Basic; no Trap",
        },
        "restrictions": (
            "dead or inactive: no-op; exact masks, stun and spawn shield come from "
            "Core; Ultimate replaces Basic"
        ),
    }


def _first_class_row(
    enemies: Array, classes: Array, pool: Array
) -> tuple[Array, Array]:
    """Pick the lowest-health member of the first ``TRAP_ORDER`` class in a pool.

    Parameters
    ----------
    enemies : Array
        Float32 (5, 58) composed enemy rows.
    classes : Array
        Int32 (5,) class IDs of those rows.
    pool : Array
        Boolean (5,) candidate rows.

    Returns
    -------
    row : Array
        Scalar int32 enemy row: lowest current HP, then lowest row, within the
        first class of Priest, Mage, Rogue, Warrior, Hunter that has a member.
        Meaningless when ``found`` is False.
    found : Array
        Scalar Boolean saying the pool held any member of those classes.
    """
    tier = jnp.zeros_like(pool)
    found = jnp.asarray(False)
    for class_id in TRAP_ORDER:
        members = pool & (classes == class_id)
        tier = jnp.where(found, tier, members)
        found = found | jnp.any(members)
    return lowest_health_row(enemies, tier), found


def _trap_ready(observation: Observation) -> Array:
    """Say whether this actor could use its Ultimate now, ignoring targets.

    Parameters
    ----------
    observation : Observation
        One actor's observation.

    Returns
    -------
    Array
        Scalar Boolean: own Ultimate cooldown is 0, all three own stun columns
        are 0, own spawn shield is 0, and the actor is active and alive. These
        are Core's own combat and cooldown conditions without target legality,
        so "no legal target now" never reads as "not ready".
    """
    own = observation.self_features
    shield = observation.spawn_lifecycle.spawn_shield_actual_durations_by_agent_by_team[
        0, observation.self_ally_index
    ]
    return (
        (own[AGENT_FEATURE_ULTIMATE_COOLDOWN_REMAINING] == 0)
        & (own[AGENT_FEATURE_STUN_WARRIOR_CHARGE_DURATION] == 0)
        & (own[AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION] == 0)
        & (own[AGENT_FEATURE_STUN_ROGUE_POISON_DURATION] == 0)
        & (shield == 0)
        & (own[AGENT_FEATURE_ACTIVE] > 0)
        & (own[AGENT_FEATURE_ALIVE] > 0)
    )


def _search_intent(observation: Observation, origin: Array) -> Array:
    """Point a searching non-Priest at the mean of the opposing spawn pads.

    Parameters
    ----------
    observation : Observation
        One actor's observation; spawn_lifecycle bank 1 holds the opponent's
        five ordered pads in world coordinates, unused slots included.
    origin : Array
        Float32 (2,) own centre.

    Returns
    -------
    Array
        Float32 (2,) full destination-minus-position vector. It is not
        normalized, because its length moves the wall-steering goal. It is
        exactly zero (so the move is Stay) when the actor is within Core's
        GEOMETRY_TOLERANCE of the mean.

    Notes
    -----
    Compiled code may fuse the mean and the subtraction and leave a residue of
    about 1e-7 at the exact mean; the tolerance makes eager and compiled calls
    agree there.
    """
    pads = observation.spawn_lifecycle.spawn_pad_positions_by_agent_by_team[1]
    intent = jnp.mean(pads, axis=0) - origin
    at_mean = jnp.sum(jnp.square(intent)) <= GEOMETRY_TOLERANCE**2
    return jnp.where(at_mean, 0.0, intent)


def reactive_tdm_gamma_policy(
    recipient_observation: Observation,
    recipient_action_mask: ActionMask,
    actor_key: Array,
    source_bank: SharedObsSensorSourceBankV2,
    recipient_source_availability: Array,
) -> ActorAction:
    """Choose one GAMMA action from the actor's current allowed inputs.

    Parameters
    ----------
    recipient_observation : Observation
        One actor's current observation, without an actor or game batch axis:
        self row (58,), ally and enemy rows (5, 58), public spawn facts.
    recipient_action_mask : ActionMask
        Exact mask for the same actor and decision: movement (9,) and
        target/Ultimate joint mask (11, 2); target 0 is None, 1..5 own roster
        rows, 6..10 opposing rows.
    actor_key : Array
        JAX random key accepted for the common policy interface. Ignored.
    source_bank : SharedObsSensorSourceBankV2
        Own-team bank with five sources, cleared for this recipient.
    recipient_source_availability : Array
        Boolean (5,) permission mask in the bank's source order.

    Returns
    -------
    ActorAction
        Scalar int32 movement, target and Ultimate choices. Dead or inactive
        actors return the no-op action. The action is exactly BETA's when no
        visible enemy has 2 or more Trap ticks, an enemy is in view, and (for a
        Hunter) no untrapped enemy has a legal Trap and its Trap is not ready.

    Notes
    -----
    The body re-expresses ALPHA's rules and BETA's Rogue override in one pass
    so the three GAMMA rules can reach every class; everything else keeps
    ALPHA/BETA's thresholds, tie-breaks and steering. Every choice is
    recomputed from this decision's inputs; there is no memory or random draw.
    The chosen action is a preference: Core applies masks, stun, spawn shield
    and collisions. Use the shared executor or vmap for teams and games.
    """
    del actor_key
    observation = recipient_observation
    allies, enemies, ally_visible, enemy_visible = compose_shared_obs_unit_features(
        observation, source_bank, recipient_source_availability
    )
    ally_living = living_candidates(allies, ally_visible)
    enemy_living = living_candidates(enemies, enemy_visible)
    own = observation.self_features
    origin = centers(own)
    class_id = own[AGENT_FEATURE_CLASS_ID].astype(jnp.int32)
    is_priest = class_id == PRIEST_CLASS_ID
    is_mage = class_id == MAGE_CLASS_ID
    is_hunter = class_id == HUNTER_CLASS_ID
    is_warrior = class_id == WARRIOR_CLASS_ID
    classes = enemies[:, AGENT_FEATURE_CLASS_ID].astype(jnp.int32)
    health = enemies[:, AGENT_FEATURE_CURRENT_HEALTH]
    trap_ticks = enemies[:, AGENT_FEATURE_STUN_HUNTER_TRAP_DURATION]
    joint = recipient_action_mask.select_target_use_ultimate_joint_mask

    # Trap hold: damage skips enemies with more than one Trap tick left.
    untrapped = enemy_living & (trap_ticks == 0)
    damage_living = enemy_living & (trap_ticks <= TRAP_DAMAGE_MAX_TICKS)

    # ALPHA movement toward the nearest enemy, with Mage and Hunter spacing.
    enemy_distances = jnp.sqrt(jnp.sum(jnp.square(centers(enemies) - origin), axis=-1))
    move_row = nearest_row(enemies, enemy_living, origin)
    enemy_delta = centers(enemies[move_row]) - origin
    enemy_distance = enemy_distances[move_row]
    shootable = enemy_living & joint[6:11, 0]
    mage_direction = jnp.sign(enemy_distance - MAGE_DISTANCE) * enemy_delta
    hunter_holds = (
        (enemy_distance > HUNTER_CLOSE_DISTANCE)
        & (enemy_distance <= HUNTER_FAR_DISTANCE)
        & shootable[move_row]
    )
    hunter_direction = jnp.where(
        enemy_distance <= HUNTER_CLOSE_DISTANCE,
        -enemy_delta,
        jnp.where(hunter_holds, 0.0, enemy_delta),
    )
    hunter_approach = (enemy_distance > HUNTER_CLOSE_DISTANCE) & ~hunter_holds

    # Trap order: cast on the first reachable class, else walk to it when ready.
    cast_row, has_cast = _first_class_row(enemies, classes, untrapped & joint[6:11, 1])
    approach_row, has_approach = _first_class_row(enemies, classes, untrapped)
    trap_approach = ~has_cast & has_approach & _trap_ready(observation)
    hunter_direction = jnp.where(
        trap_approach,
        centers(enemies[approach_row]) - origin,
        jnp.where(has_cast, 0.0, hunter_direction),
    )
    hunter_approach = jnp.where(
        trap_approach, True, jnp.where(has_cast, False, hunter_approach)
    )

    attack_direction = jnp.where(
        is_mage, mage_direction, jnp.where(is_hunter, hunter_direction, enemy_delta)
    )
    map_center = (
        observation.context_features[
            jnp.asarray([CONTEXT_FEATURE_MAP_WIDTH, CONTEXT_FEATURE_MAP_HEIGHT])
        ]
        * 0.5
    )
    center_direction = map_center - origin
    attack_direction = jnp.where(
        jnp.any(enemy_living), attack_direction, _search_intent(observation, origin)
    )
    priest_direction, priest_approach = _priest_direction(
        own,
        allies,
        ally_living,
        enemies,
        enemy_living,
        observation.self_ally_index,
        center_direction,
    )
    direction = jnp.where(is_priest, priest_direction, attack_direction)
    attack_approach = ~jnp.any(enemy_living) | jnp.where(
        is_mage,
        enemy_distance > MAGE_DISTANCE,
        jnp.where(is_hunter, hunter_approach, True),
    )
    move = refine_movement(
        observation,
        recipient_action_mask,
        direction,
        approach=jnp.where(is_priest, priest_approach, attack_approach),
    )

    # ALPHA combat on the Trap-hold pools; the Hunter's Ultimate is the Trap order.
    allies_basic = ally_living & joint[1:6, 0]
    allies_ultimate = ally_living & joint[1:6, 1]
    basic_pool = damage_living & joint[6:11, 0]
    enemies_ultimate = damage_living & joint[6:11, 1]
    enemies_ultimate &= ~is_warrior | (health < WARRIOR_CHARGE_HEALTH)
    ally_basic_row = lowest_health_row(
        allies, allies_basic, break_ties_by_max_health=True
    )
    ally_ultimate_row = lowest_health_row(
        allies, allies_ultimate, break_ties_by_max_health=True
    )
    enemy_basic_row = lowest_health_row(enemies, basic_pool)
    enemy_ultimate_row = jnp.where(
        is_hunter, cast_row, lowest_health_row(enemies, enemies_ultimate)
    )
    priest_ultimate = jnp.any(allies_ultimate) & (
        allies[ally_ultimate_row, AGENT_FEATURE_CURRENT_HEALTH]
        <= PRIEST_ULTIMATE_HEALTH
    )
    mage_ultimate = joint[0, 1] & jnp.any(basic_pool)
    use_ultimate = jnp.where(
        is_priest,
        priest_ultimate,
        jnp.where(
            is_mage,
            mage_ultimate,
            jnp.where(is_hunter, has_cast, jnp.any(enemies_ultimate)),
        ),
    )
    basic_target = jnp.where(
        is_priest,
        jnp.where(jnp.any(allies_basic), ally_basic_row + 1, 0),
        jnp.where(jnp.any(basic_pool), enemy_basic_row + 6, 0),
    )
    ultimate_target = jnp.where(
        is_priest,
        ally_ultimate_row + 1,
        jnp.where(is_mage, 0, enemy_ultimate_row + 6),
    )
    target = jnp.where(use_ultimate, ultimate_target, basic_target)
    participating = (own[AGENT_FEATURE_ACTIVE] > 0) & (own[AGENT_FEATURE_ALIVE] > 0)
    alpha_move = jnp.where(participating, move, 0).astype(jnp.int32)
    alpha_target = jnp.where(participating, target, 0).astype(jnp.int32)
    alpha_ultimate = (participating & use_ultimate).astype(jnp.int32)

    # BETA's Rogue: body-aware pursuit of Priest, Mage, Hunter; combat in radius.
    priests = enemy_living & (enemies[:, AGENT_FEATURE_CLASS_ID] == PRIEST_CLASS_ID)
    mages = enemy_living & (enemies[:, AGENT_FEATURE_CLASS_ID] == MAGE_CLASS_ID)
    hunters = enemy_living & (enemies[:, AGENT_FEATURE_CLASS_ID] == HUNTER_CLASS_ID)
    prey_candidates = jnp.where(
        jnp.any(priests), priests, jnp.where(jnp.any(mages), mages, hunters)
    )
    prey_row = lowest_health_row(enemies, prey_candidates)
    body_mask = jnp.concatenate((ally_living, enemy_living))
    body_mask = body_mask.at[observation.self_ally_index].set(False)
    body_mask = body_mask.at[MAX_AGENTS_PER_TEAM + prey_row].set(False)
    pursuit_move = _body_aware_move(
        observation,
        recipient_action_mask,
        centers(enemies[prey_row]),
        jnp.concatenate((allies, enemies)),
        body_mask,
    )
    distances_squared = jnp.sum(jnp.square(centers(enemies) - origin), axis=-1)
    within_basic = distances_squared <= jnp.square(
        own[AGENT_FEATURE_BASIC_INTERACTION_RADIUS]
    )
    rogue_basic = damage_living & within_basic & joint[6:11, 0]
    rogue_ultimate = damage_living & within_basic & joint[6:11, 1]
    rogue_uses_ultimate = jnp.any(rogue_ultimate)
    combat_row = lowest_health_row(
        enemies, jnp.where(rogue_uses_ultimate, rogue_ultimate, rogue_basic)
    )
    rogue_target = jnp.where(
        rogue_uses_ultimate | jnp.any(rogue_basic), combat_row + 6, 0
    )
    participating_rogue = (
        (own[AGENT_FEATURE_CLASS_ID] == ROGUE_CLASS_ID)
        & (own[AGENT_FEATURE_ACTIVE] > 0)
        & (own[AGENT_FEATURE_ALIVE] > 0)
    )
    return ActorAction(
        jnp.where(
            participating_rogue & jnp.any(prey_candidates), pursuit_move, alpha_move
        ).astype(jnp.int32),
        jnp.where(participating_rogue, rogue_target, alpha_target).astype(jnp.int32),
        jnp.where(participating_rogue, rogue_uses_ultimate, alpha_ultimate).astype(
            jnp.int32
        ),
    )
