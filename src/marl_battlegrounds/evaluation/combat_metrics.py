"""Shared per-transition combat quantities over authoritative simulator facts."""

# Core is protected. Its existing private helpers preserve one simulator
# authority for hypothetical healing without changing Core's public API.
# pyright: reportPrivateUsage=false

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.axis_mappings import (
    GLOBAL_RECIPIENT_SLOT_INDEX_BY_ACTOR_AND_TARGET_ACTION,
)
from marl_battlegrounds.core.combat import (
    get_basic_healing_by_class_ids,
    get_ultimate_healing_by_class_ids,
)
from marl_battlegrounds.core.env import (
    _aggregate_health_effects_and_basic_passives_by_global_slot,
    _build_global_pairwise_actor_and_recipient_target_one_hot_matrix,
    _compute_health_after_simultaneous_damage_and_healing,
)
from marl_battlegrounds.core.types import (
    MAX_AGENT_SLOTS,
    PRIEST_CLASS_ID,
    Action,
    ActionMask,
    EnvConfig,
    EnvState,
    Info,
)


class CombatQuantities(NamedTuple):
    """Source/recipient matrices, plus unique recipient rescue opportunities.

    Amount matrices are float32 ``[10, 10]``. Contribution matrices are boolean
    ``[10, 10]`` and recipient opportunity/save vectors are boolean ``[10]``.
    Priest contribution counts indicate useful participation, not solo causation.
    """

    damage: Array
    healing: Array
    wasted_healing: Array
    kill_contributions: Array
    rescue_opportunities: Array
    rescues: Array
    rescue_contributions: Array


def _available_priest_healing(
    config: EnvConfig, state: EnvState, action_mask: ActionMask
) -> Array:
    """Resolve each recipient's best legal combined healing through Core.

    Catalog amounts rank Basic/Ultimate choices; the same recipient modifier
    applies to both. Core still computes the selected combined amount and its
    float32 reduction order. Each recipient is an independent counterfactual,
    not a claim that one Priest could heal several allies simultaneously.
    """
    slots = jnp.arange(MAX_AGENT_SLOTS)
    priest = config.agent_profile.class_ids == PRIEST_CLASS_ID
    raw_healing = jnp.stack(
        (
            get_basic_healing_by_class_ids(config.agent_profile.class_ids),
            get_ultimate_healing_by_class_ids(config.agent_profile.class_ids),
        ),
        axis=-1,
    )
    neutral_multipliers = jnp.ones(MAX_AGENT_SLOTS, dtype=jnp.float32)

    def healing_for(recipient: Array) -> Array:
        targets = jnp.argmax(
            recipient == GLOBAL_RECIPIENT_SLOT_INDEX_BY_ACTOR_AND_TARGET_ACTION,
            axis=-1,
        )
        legal = (
            action_mask.select_target_use_ultimate_joint_mask[slots, targets]
            & priest[:, None]
        )
        available = jnp.where(legal, raw_healing, 0)
        choice = jnp.argmax(available, axis=-1).astype(jnp.int32)
        can_heal = jnp.any(available > 0, axis=-1)
        action = Action(
            move=jnp.zeros(MAX_AGENT_SLOTS, dtype=jnp.int32),
            select_target=jnp.where(can_heal, targets, 0).astype(jnp.int32),
            use_ultimate=jnp.where(can_heal, choice, 0),
        )
        routing, _, _ = (
            _build_global_pairwise_actor_and_recipient_target_one_hot_matrix(
                action.select_target
            )
        )
        # Aura arguments affect damage only. Returning only healing lets JIT
        # discard damage, passive and status work from this existing helper.
        effects = _aggregate_health_effects_and_basic_passives_by_global_slot(
            state, config, action, routing, neutral_multipliers, neutral_multipliers
        )
        return effects.total_effective_healing_by_recipient[recipient]

    return jax.vmap(healing_for)(slots)


def combat_quantities(
    config: EnvConfig, state: EnvState, action_mask: ActionMask, info: Info
) -> CombatQuantities:
    """Compute delivered effects and truthful direct/support credit once.

    ``state`` and ``action_mask`` are the decision-start pair that produced
    ``info``. All routing, modifiers, deaths and actual combat health come from
    Core facts. Metric waste allocation is proportional to delivered healing;
    natural regeneration never earns healing, kill-support or rescue credit.
    """
    facts = info.transition_facts
    combat = facts.combat_transition_facts
    routing = (
        jax.nn.one_hot(
            combat.combat_effect_recipient_global_slot_by_source,
            MAX_AGENT_SLOTS,
            dtype=jnp.float32,
        )
        * combat.combat_effect_has_recipient_by_source[:, None]
    )
    damage = (
        routing
        * (
            combat.source_modified_damage_output_by_source
            * combat.recipient_damage_modifier_by_source
        )[:, None]
    )
    healing = (
        routing
        * (
            combat.source_modified_healing_output_by_source
            * combat.recipient_healing_modifier_by_source
        )[:, None]
    )

    total_healing = combat.total_effective_healing_by_recipient
    total_damage = combat.total_effective_damage_by_recipient
    uncapped_health = state.current_health + (total_healing - total_damage)
    waste = jnp.clip(
        uncapped_health - combat.health_after_combat_resolution_by_recipient,
        min=0,
        max=total_healing,
    )
    wasted_fraction = waste / jnp.where(total_healing > 0, total_healing, 1)
    wasted_healing = healing * wasted_fraction[None, :]
    # Decide useful participation from recipient waste, rather than subtracting
    # two source-rounded amounts and creating tiny false useful-healing residues.
    useful_healing = (
        (healing > 0)
        & (total_healing > waste)[None, :]
        & (config.agent_profile.class_ids == PRIEST_CLASS_ID)[:, None]
    )
    direct = (
        routing.astype(jnp.bool_)
        & facts.death_facts.contributed_to_new_death_by_source[:, None]
    )
    priest_support = jnp.any(useful_healing[:, :, None] & direct[None, :, :], axis=1)

    without_healing = _compute_health_after_simultaneous_damage_and_healing(
        total_damage, jnp.zeros_like(total_healing), state, config
    )
    threatened = (
        config.agent_profile.active_mask & state.alive_mask & (without_healing == 0)
    )
    available_healing = _available_priest_healing(config, state, action_mask)
    with_available_healing = _compute_health_after_simultaneous_damage_and_healing(
        total_damage, available_healing, state, config
    )
    rescue_opportunities = threatened & (with_available_healing > 0)
    rescues = (
        threatened
        & (total_healing > 0)
        & (combat.health_after_combat_resolution_by_recipient > 0)
    )
    return CombatQuantities(
        damage=damage,
        healing=healing,
        wasted_healing=wasted_healing,
        kill_contributions=direct | priest_support,
        rescue_opportunities=rescue_opportunities,
        rescues=rescues,
        rescue_contributions=useful_healing & rescues[None, :],
    )
