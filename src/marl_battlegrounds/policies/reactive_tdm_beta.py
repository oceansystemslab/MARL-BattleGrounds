"""Reactive TDM BETA: Rogue shoulder pursuit prioritizes Priest, Mage, then Hunter."""

import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.types import (
    AGENT_FEATURE_ACTIVE,
    AGENT_FEATURE_ALIVE,
    AGENT_FEATURE_BASIC_INTERACTION_RADIUS,
    AGENT_FEATURE_CLASS_ID,
    HUNTER_CLASS_ID,
    MAGE_CLASS_ID,
    MAX_AGENTS_PER_TEAM,
    PRIEST_CLASS_ID,
    ROGUE_CLASS_ID,
    ActionMask,
    Observation,
)
from marl_battlegrounds.policies.actor import ActorAction
from marl_battlegrounds.policies.reactive_common import (
    BYPASS_MINIMUM_CONTACT_ANGLE_DEGREES,
    MINIMUM_MOVEMENT_FRACTION,
    _body_aware_move,  # pyright: ignore[reportPrivateUsage]
    centers,
    living_candidates,
    lowest_health_row,
)
from marl_battlegrounds.policies.reactive_tdm_alpha import (
    reactive_tdm_alpha_controller_descriptor,
    reactive_tdm_alpha_policy,
)
from marl_battlegrounds.policies.shared_obs import (
    SharedObsSensorSourceBankV1,
    compose_shared_obs_unit_features,
)


def reactive_tdm_beta_controller_descriptor() -> dict[str, object]:
    """Return fresh rule data, including the inherited general controller rules."""
    return {
        "policy_id": "scenario-5-pressure-controller",
        "version": 4,
        "information": "same-epoch SharedObs and recipient exact masks",
        "execution": "deterministic; actor key ignored",
        "inherited_controller": reactive_tdm_alpha_controller_descriptor(),
        "classes": "non-Rogues delegate unchanged to Reactive TDM",
        "candidates": "observed active living positive-health enemies",
        "pursuit": (
            "observed living enemy Priest first, else Mage, else Hunter; "
            "lowest current HP within class then global slot; recompute every decision"
        ),
        "combat": (
            "within Basic radius: lowest-HP legal Ultimate enemy else lowest-HP "
            "legal Basic enemy else no-combat; independent of pursuit"
        ),
        "ties": (
            "health then global slot; active preferred wall moves: phase alignment, "
            "endpoint distance, movement action ID; otherwise endpoint distance "
            "then movement action ID"
        ),
        "movement": {
            "projection": "existing static obstacle/bounds geometry; no body pairs",
            "minimum_stride_fraction_inclusive": MINIMUM_MOVEMENT_FRACTION,
            "blockers": "observed living allies/enemies excluding self and prey",
            "contact": "clear segment or glancing first contact at physical radius sum",
            "minimum_contact_angle_degrees": BYPASS_MINIMUM_CONTACT_ANGLE_DEGREES,
            "overlap_escape": (
                "existing geometry tolerance counts as contact; deeper overlap "
                "must never deepen and must finish farther away"
            ),
            "selection": (
                "outside wall steering: closest admissible moving endpoint to prey; "
                "active wall steering: useful phase progress preserving end "
                "clearance, ranked by alignment then prey distance then action ID; "
                "with a fitting end but no preferred move, restore all original "
                "body-admissible moves ranked by prey distance, including during "
                "corner crossing; no escape when neither end fits; no preference "
                "for contact-free detours; detours may retreat; screening unchanged"
            ),
            "wall_steering": (
                "nearby vertical-wall and projected-candidate strips; exclude "
                "same-physical-end goals and far-face-cleared groups before nearest "
                "anchor selection; group only directly overlapping body-expanded "
                "vertical walls, not transitive closure; exit East for prey at or "
                "beyond expanded East face within geometry tolerance, else West; "
                "prefer fitting SOUTH, else NORTH, avoiding occupied boundary-width "
                "passages unless an admissible South corner move exists; if both "
                "occupied retain boundary-fitting preference; turn horizontally "
                "on clearance or admissible phase progress; release beyond far face"
            ),
            "assumption": "other observed bodies stationary; no action prediction",
            "search": "eight directions only; no lookahead or route memory",
        },
        "fallbacks": (
            "no Priest, Mage or Hunter: unchanged Reactive TDM Rogue movement toward "
            "nearest enemy or map center, without body screening; "
            "no useful safe move: Stay; "
            "dead/inactive: no-op"
        ),
        "action_priority": "Ultimate replaces Basic; exact masks override preferences",
    }


def reactive_tdm_beta_policy(
    recipient_observation: Observation,
    recipient_action_mask: ActionMask,
    actor_key: Array,
    source_bank: SharedObsSensorSourceBankV1,
    recipient_source_availability: Array,
    recipient_global_slot: Array,
) -> ActorAction:
    """Inherit TDM, replacing only Rogue pursuit and radius-bounded combat."""
    baseline = reactive_tdm_alpha_policy(
        recipient_observation,
        recipient_action_mask,
        actor_key,
        source_bank,
        recipient_source_availability,
        recipient_global_slot,
    )
    allies, enemies, ally_visible, enemy_visible = compose_shared_obs_unit_features(
        recipient_observation,
        source_bank,
        recipient_source_availability,
        recipient_global_slot,
    )
    ally_living = living_candidates(allies, ally_visible)
    enemy_living = living_candidates(enemies, enemy_visible)
    priests = enemy_living & (enemies[:, AGENT_FEATURE_CLASS_ID] == PRIEST_CLASS_ID)
    mages = enemy_living & (enemies[:, AGENT_FEATURE_CLASS_ID] == MAGE_CLASS_ID)
    hunters = enemy_living & (enemies[:, AGENT_FEATURE_CLASS_ID] == HUNTER_CLASS_ID)
    prey_candidates = jnp.where(
        jnp.any(priests), priests, jnp.where(jnp.any(mages), mages, hunters)
    )
    prey_row = lowest_health_row(enemies, prey_candidates)
    body_mask = jnp.concatenate((ally_living, enemy_living))
    body_mask = body_mask.at[recipient_global_slot % MAX_AGENTS_PER_TEAM].set(False)
    body_mask = body_mask.at[MAX_AGENTS_PER_TEAM + prey_row].set(False)
    pursuit_move = _body_aware_move(
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
    participating_rogue = (
        (own[AGENT_FEATURE_CLASS_ID] == ROGUE_CLASS_ID)
        & (own[AGENT_FEATURE_ACTIVE] > 0)
        & (own[AGENT_FEATURE_ALIVE] > 0)
    )
    return ActorAction(
        jnp.where(
            participating_rogue & jnp.any(prey_candidates), pursuit_move, baseline.move
        ).astype(jnp.int32),
        jnp.where(participating_rogue, target, baseline.select_target).astype(
            jnp.int32
        ),
        jnp.where(participating_rogue, use_ultimate, baseline.use_ultimate).astype(
            jnp.int32
        ),
    )
