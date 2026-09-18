"""Derive combat measurements from one authoritative Core transition.

combat_quantities supplies directed amounts and useful kill/rescue contribution
for full metrics. combat_credit shares the observed credit rule with capture.
Hypothetical healing calls existing Core helpers; this module does not change
combat physics. All numerical functions use one scalar environment and support
outer JAX batching. Admission of real transitions belongs to the caller.
"""

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
    """Directed combat amounts and recipient-level rescue facts for one transition.

    Attributes
    ----------
    routing : Array
        float32 (10, 10) source/recipient routing from Core.
    damage : Array
        float32 (10, 10) delivered damage after source/recipient modifiers,
        including amounts beyond remaining health.
    healing : Array
        float32 (10, 10) delivered Priest healing, including excess.
    excess_healing : Array
        float32 (10, 10) excess allocated among healing sources.
    kill_contributions : Array
        bool (10, 10) direct damage or useful one-hop Priest
        support on a newly dead recipient's transition.
    rescue_opportunities : Array
        bool (10,) recipients legal healing could have saved.
    rescues : Array
        bool (10,) recipients actually saved by this transition's healing.
    rescue_contributions : Array
        bool (10, 10) Priests giving useful healing in rescues.

    Notes
    -----
    Matrix axes use global source then recipient slots. Team A occupies 0..4
    and Team B 5..9. Contribution marks useful participation, not sole causation.
    No natural regeneration receives Priest healing/support credit.
    """

    routing: Array
    damage: Array
    healing: Array
    excess_healing: Array
    kill_contributions: Array
    rescue_opportunities: Array
    rescues: Array
    rescue_contributions: Array


class CombatCredit(NamedTuple):
    """Observed excess healing and nonrecursive contribution masks.

    Attributes
    ----------
    excess_healing_by_recipient : Array
        float32 (10,) healing beyond post-combat capacity.
    useful_healing : Array
        bool (10, 10) Priest source/recipient pairs with some useful
        healing on this transition.
    kill_contributions : Array
        bool (10, 10) direct contributors plus Priests healing
        those direct contributors usefully. Healing support is not recursive.

    Notes
    -----
    All slots are global. This immutable numerical tree contains no hypothetical
    rescue computation and no attribution to unseen earlier status casters.
    """

    excess_healing_by_recipient: Array
    useful_healing: Array
    kill_contributions: Array


def combat_credit(
    class_ids: Array,
    health_before: Array,
    healing: Array,
    routing: Array,
    total_healing: Array,
    total_damage: Array,
    health_after: Array,
    contributed_to_death: Array,
) -> CombatCredit:
    """Compute observed damage/Priest kill credit without hypothetical rescues.

    Parameters
    ----------
    class_ids : Array
        int32 (10,) configured classes in global slot order.
    health_before : Array
        float32 (10,) health at the decision start.
    healing : Array
        float32 (10, 10) delivered source/recipient healing amounts.
    routing : Array
        (10, 10) source/recipient routing mask or zero/one amounts.
    total_healing : Array
        float32 (10,) Core recipient healing totals.
    total_damage : Array
        float32 (10,) Core recipient damage totals.
    health_after : Array
        float32 (10,) health immediately after simultaneous combat,
        before natural regeneration.
    contributed_to_death : Array
        bool (10,) Core source flags for newly caused deaths.

    Returns
    -------
    CombatCredit
        CombatCredit with recipient excess and source/recipient useful-healing and
        kill-contribution masks. Entirely excess healing earns no support.

    Notes
    -----
    Inputs must describe one transition in the same decision epoch. This pure
    JAX calculation does not validate, mutate or transfer them to the host.
    Priest support reaches direct attackers once; healing a support-only Priest
    does not recursively earn credit.
    """
    uncapped_health = health_before + (total_healing - total_damage)
    excess = jnp.clip(uncapped_health - health_after, min=0, max=total_healing)
    # Compare recipient totals before source rounding can create tiny false
    # useful-healing residues. Entirely excess healing never earns support.
    useful_healing = (
        (healing > 0)
        & (total_healing > excess)[None, :]
        & (class_ids == PRIEST_CLASS_ID)[:, None]
    )
    direct = routing.astype(jnp.bool_) & contributed_to_death[:, None]
    priest_support = jnp.any(useful_healing[:, :, None] & direct[None, :, :], axis=1)
    return CombatCredit(excess, useful_healing, direct | priest_support)


def _available_priest_healing(
    config: EnvConfig, state: EnvState, action_mask: ActionMask
) -> Array:
    """Return each recipient's best legally available combined Priest healing.

    config, state and action_mask describe the same scalar decision start. Rank
    legal Basic/Ultimate choices by catalog amounts, then let existing Core helpers
    compute their combined result and float32 reduction order. Return float32 (10,).
    Each recipient is a separate hypothetical case; one Priest is not claimed to
    heal all of them at once. The numerical path supports jit and outer vmap.
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
        """Ask Core for the best legal Priest-healing combination on one recipient."""
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
    """Compute directed combat effects and useful support from one Core transition.

    Parameters
    ----------
    config : EnvConfig
        Exact scalar EnvConfig for the transition.
    state : EnvState
        Core EnvState at decision start, before combat and regeneration.
    action_mask : ActionMask
        Matching decision-start masks, including joint target/Ultimate
        legality used by the healing-opportunity calculation.
    info : Info
        Core Info containing that transition's routing, modifiers, health
        resolution, death and action-acceptance facts.

    Returns
    -------
    CombatQuantities
        CombatQuantities. Amounts use float32 global (source, recipient) matrices;
        contribution masks are bool (10, 10), and rescue vectors are bool (10,).
        Excess healing is shared in proportion to delivered healing.

    Notes
    -----
    Pure numerical JAX route with no host I/O or input mutation. Core remains
    the authority for actual and hypothetical health resolution. The caller
    decides whether a transition is real before accumulating results.
    Regeneration contributes no Priest healing, kill-support or rescue credit.
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
    credit = combat_credit(
        config.agent_profile.class_ids,
        state.current_health,
        healing,
        routing,
        total_healing,
        total_damage,
        combat.health_after_combat_resolution_by_recipient,
        facts.death_facts.contributed_to_new_death_by_source,
    )
    excess_fraction = credit.excess_healing_by_recipient / jnp.where(
        total_healing > 0, total_healing, 1
    )
    excess_healing = healing * excess_fraction[None, :]

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
        routing=routing,
        damage=damage,
        healing=healing,
        excess_healing=excess_healing,
        kill_contributions=credit.kill_contributions,
        rescue_opportunities=rescue_opportunities,
        rescues=rescues,
        rescue_contributions=credit.useful_healing & rescues[None, :],
    )
