"""Fixed-shape JAX collection for the scalar TDM metric catalog.

Only sufficient statistics survive a transition. Dictionary keys are static
PyTree structure, and all values are arrays; no replay, report or host callback
is required. Final and prefix analysis use this same collection authority.
"""

from typing import cast

import jax
import jax.numpy as jnp
from jax import Array

from marl_battlegrounds.core.combat import derive_effective_movement_speeds
from marl_battlegrounds.core.env import (
    _aggregate_health_effects_and_basic_passives_by_global_slot,  # pyright: ignore[reportPrivateUsage]
)
from marl_battlegrounds.core.types import (
    ActionMask,
    EnvConfig,
    EnvState,
    Info,
)
from marl_battlegrounds.evaluation.combat_metrics import combat_quantities
from marl_battlegrounds.evaluation.episode_metrics import MetricValues
from marl_battlegrounds.evaluation.metric_catalog import (
    METRIC_COLUMNS,
    PRIORITY_METRIC_COLUMNS,
    STATUS_NAMES,
    MetricScope,
)

type FullTotals = dict[str, Array]

_COUNTER_SHAPES = {
    "deaths": (10,),
    "dead_steps": (10,),
    "kills": (10, 10),
    "activations": (10, 2),
    "actions": (10, 6),
    "applications": (10, 9),
    "active_steps": (10, 9),
    "controlled_kills": (2, 7),
    "controlled_contributions": (10, 7),
    "single_kills": (2,),
    "multi_kills": (2,),
    "focus_steps": (2,),
    "trap_periods": (2,),
    "trap_breaks": (2,),
    "trap_remaining": (2,),
    "trap_contributions": (10,),
    "waves": (2,),
    "respawns": (2,),
    "burst_kills": (10,),
    "mage_coverage": (10,),
    "mage_eligible": (10,),
    "mage_team_coverage": (2,),
    "mage_team_eligible": (2,),
    "warrior_coverage": (10,),
    "warrior_eligible": (10,),
    "warrior_team_coverage": (2,),
    "warrior_team_eligible": (2,),
    "rescue_opportunities": (2,),
    "rescues": (2,),
    "rescue_contributions": (10,),
    "freedom_protected": (10,),
    "freedom_eligible": (10,),
    "distance_count": (10, 10),
}
_AMOUNT_SHAPES = {
    "damage": (10, 10),
    "healing": (10, 10),
    "waste": (10, 10),
    "regeneration": (10,),
    "controlled_damage": (10, 7),
    "controlled_healing": (10, 7),
    "focus_sum": (2,),
    "burst_damage": (10,),
    "burst_lethal_damage": (10,),
    "mage_damage_gain": (2,),
    "warrior_damage_prevented": (2,),
    "poison_prevention": (10,),
    "distance_sum": (10, 10),
}


def _team_sum(values: Array) -> Array:
    return values.reshape((2, 5, *values.shape[1:])).sum(axis=1)


def initialize_full(config: EnvConfig, state: EnvState) -> FullTotals:
    """Count initially observed Trap periods; all other evidence starts empty."""
    totals = {
        name: jnp.zeros(shape, jnp.int32) for name, shape in _COUNTER_SHAPES.items()
    }
    totals.update(
        {name: jnp.zeros(shape, jnp.float32) for name, shape in _AMOUNT_SHAPES.items()}
    )
    initial_traps = (
        config.agent_profile.active_mask
        & state.alive_mask
        & (state.stun_durations[:, 1] > 0)
    )
    totals["trap_periods"] = _team_sum(initial_traps.astype(jnp.int32))[::-1]
    return totals


def _aura_counts(
    config: EnvConfig, state: EnvState, coverage: Array, class_id: int
) -> tuple[Array, Array, Array, Array]:
    available = (
        config.agent_profile.active_mask
        & state.alive_mask
        & (state.spawn_shield_durations == 0)
    )
    emitter = available & (config.agent_profile.class_ids == class_id)
    team = config.agent_profile.team_ids
    eligible = emitter[:, None] & available[None, :] & (team[:, None] == team[None, :])
    return (
        coverage.sum(axis=1, dtype=jnp.int32),
        eligible.sum(axis=1, dtype=jnp.int32),
        _team_sum(coverage.any(axis=0).astype(jnp.int32)),
        _team_sum(eligible.any(axis=0).astype(jnp.int32)),
    )


def update_full(
    totals: FullTotals,
    config: EnvConfig,
    state: EnvState,
    action_mask: ActionMask,
    info: Info,
) -> FullTotals:
    """Collect one real transition; no-transition facts preserve all counters."""
    facts = info.transition_facts
    combat = facts.combat_transition_facts
    quantities = combat_quantities(config, state, action_mask, info)
    active = config.agent_profile.active_mask
    living = active & state.alive_mask
    damage = quantities.damage
    healing = quantities.healing
    deaths = facts.death_facts.is_newly_dead_by_recipient
    contributions = quantities.kill_contributions
    routed = jax.nn.one_hot(combat.combat_effect_recipient_global_slot_by_source, 10)
    routed = routed * combat.combat_effect_has_recipient_by_source[:, None]
    statuses = (
        jnp.concatenate(
            (
                state.slow_durations,
                state.stun_durations,
                state.rogue_poison_anti_heal_durations[:, None],
                state.mage_burst_damage_amplification_durations[:, None],
                state.priest_blessing_of_freedom_slow_floor_durations[:, None],
            ),
            axis=1,
        )
        > 0
    )
    hostile_status = statuses[:, :7]
    acceptance = facts.action_acceptance_facts
    domain_rejection = acceptance.submitted_action_tuple_is_out_of_domain_by_actor
    move_rejection = acceptance.in_domain_move_action_is_rejected_by_actor
    combat_rejection = acceptance.in_domain_combat_action_pair_is_rejected_by_actor
    rejected = domain_rejection | move_rejection | combat_rejection
    damaging = damage > 0
    attacking_sources = damaging.any(axis=1).reshape(2, 5).sum(axis=1)
    most_shared_target = damaging.reshape(2, 5, 10).sum(axis=1).max(axis=1)
    focus_eligible = attacking_sources >= 2
    contributor_count = contributions.sum(axis=0)
    trap_breaks = (
        facts.status_lifecycle_facts.broken_by_damage_by_recipient_and_status_channel[
            :, 4
        ]
    )
    trap_applications = (
        routed * combat.stun_is_applied_by_source_and_channel[:, 1, None]
    ).sum(axis=0) > 0
    # A naturally expired or damage-broken period is closed before fresh control.
    trap_closed = (state.stun_durations[:, 1] <= 1) | trap_breaks
    new_traps = trap_applications & trap_closed
    burst = (state.mage_burst_damage_amplification_durations > 0) & (
        config.agent_profile.class_ids == 1
    )
    no_aura = _aggregate_health_effects_and_basic_passives_by_global_slot(
        state,
        config,
        acceptance.accepted_joint_action,
        routed,
        jnp.ones(10, jnp.float32),
        jnp.ones(10, jnp.float32),
    )
    mage_gain = (
        combat.source_modified_damage_output_by_source
        - no_aura.source_modified_damage_output_by_source
    ) * combat.recipient_damage_modifier_by_source
    prevented = (
        routed
        * (
            combat.source_modified_damage_output_by_source
            * (1 - combat.recipient_damage_modifier_by_source)
        )[:, None]
    )
    poison_prevention = (
        routed
        * (
            combat.source_modified_healing_output_by_source
            * (1 - combat.recipient_healing_modifier_by_source)
        )[:, None]
    )
    freedom = state.priest_blessing_of_freedom_slow_floor_durations
    freedom_eligible = (
        living
        & (state.spawn_shield_durations == 0)
        & ~(state.stun_durations > 0).any(axis=1)
        & (freedom > 0)
    )

    # Unit speed/scale compares the applicable restriction, not movement chosen.
    def restricted_speed(duration: Array) -> Array:
        return derive_effective_movement_speeds(
            state.slow_durations,
            duration,
            state.stun_durations,
            state.spawn_shield_durations,
            jnp.ones(10, jnp.float32),
            1.0,
            living,
            1.0,
        )

    freedom_protected = freedom_eligible & (
        restricted_speed(freedom) > restricted_speed(jnp.zeros_like(freedom))
    )
    same_team = (
        config.agent_profile.team_ids[:, None] == config.agent_profile.team_ids[None, :]
    )
    ally_pairs = (
        living[:, None]
        & living[None, :]
        & same_team
        & jnp.triu(jnp.ones((10, 10), bool), k=1)
    )
    distances = cast(
        Array,
        jnp.linalg.norm(
            state.agent_positions[:, None, :] - state.agent_positions[None, :, :],
            axis=-1,
        ),
    )
    delta: FullTotals = {
        "damage": damage,
        "healing": healing,
        "waste": quantities.wasted_healing,
        "regeneration": (
            facts.regeneration_facts.actual_health_regenerated_this_step_by_agent
        ),
        "deaths": deaths,
        "dead_steps": active & ~state.alive_mask,
        "kills": contributions,
        "activations": jnp.stack(
            (
                combat.basic_effect_is_activated_by_source,
                combat.ultimate_effect_is_activated_by_source,
            ),
            axis=1,
        ),
        "actions": jnp.stack(
            (
                active,
                active & ~rejected,
                active & rejected,
                active & domain_rejection,
                active & move_rejection,
                active & combat_rejection,
            ),
            axis=1,
        ),
        "applications": jnp.concatenate(
            (
                combat.slow_is_applied_by_source_and_channel,
                combat.stun_is_applied_by_source_and_channel,
                combat.rogue_poison_anti_heal_is_applied_by_source[:, None],
                combat.mage_burst_damage_amplification_is_applied_by_source[:, None],
                combat.priest_blessing_of_freedom_is_applied_by_source[:, None],
            ),
            axis=1,
        ),
        "active_steps": statuses & living[:, None],
        # Reduced-precision GPU dot products round delivered amounts before
        # accumulation; keep these scientific measurements in full float32.
        "controlled_damage": jnp.matmul(
            damage,
            hostile_status.astype(jnp.float32),
            precision=jax.lax.Precision.HIGHEST,
        ),
        "controlled_healing": jnp.matmul(
            healing,
            hostile_status.astype(jnp.float32),
            precision=jax.lax.Precision.HIGHEST,
        ),
        "controlled_kills": _team_sum(
            (deaths[:, None] & hostile_status).astype(jnp.int32)
        )[::-1],
        "controlled_contributions": contributions.astype(jnp.int32)
        @ hostile_status.astype(jnp.int32),
        "single_kills": _team_sum(
            (deaths & (contributor_count == 1)).astype(jnp.int32)
        )[::-1],
        "multi_kills": _team_sum((deaths & (contributor_count >= 2)).astype(jnp.int32))[
            ::-1
        ],
        "focus_sum": jnp.where(
            focus_eligible, most_shared_target / jnp.maximum(attacking_sources, 1), 0
        ),
        "focus_steps": focus_eligible,
        "trap_periods": _team_sum(new_traps.astype(jnp.int32))[::-1],
        "trap_breaks": _team_sum(trap_breaks.astype(jnp.int32))[::-1],
        "trap_remaining": _team_sum(
            jnp.where(trap_breaks, state.stun_durations[:, 1] - 1, 0)
        )[::-1],
        "trap_contributions": (
            (routed > 0)
            & (combat.raw_damage_output_by_source > 0)[:, None]
            & trap_breaks[None, :]
        ).sum(axis=1),
        "waves": facts.respawn_facts.respawn_wave_occurred_this_transition_by_team,
        "respawns": _team_sum(
            facts.respawn_facts.was_respawned_this_transition_by_agent.astype(jnp.int32)
        ),
        "burst_damage": jnp.where(burst, damage.sum(axis=1), 0),
        "burst_lethal_damage": (damage * deaths[None, :] * burst[:, None]).sum(axis=1),
        "burst_kills": (contributions & burst[:, None]).sum(axis=1),
        "mage_damage_gain": _team_sum(mage_gain),
        "warrior_damage_prevented": _team_sum(prevented.sum(axis=0)),
        "poison_prevention": poison_prevention.sum(axis=0),
        "rescue_opportunities": _team_sum(
            quantities.rescue_opportunities.astype(jnp.int32)
        ),
        "rescues": _team_sum(quantities.rescues.astype(jnp.int32)),
        "rescue_contributions": quantities.rescue_contributions.sum(axis=1),
        "freedom_protected": freedom_protected,
        "freedom_eligible": freedom_eligible,
        "distance_sum": jnp.where(ally_pairs, distances, 0),
        "distance_count": ally_pairs,
    }
    for name, class_id, coverage in (
        (
            "mage",
            1,
            facts.aura_facts.is_covered_by_mage_damage_aura_by_emitter_and_beneficiary,
        ),
        (
            "warrior",
            2,
            facts.aura_facts.is_covered_by_warrior_mitigation_aura_by_emitter_and_beneficiary,
        ),
    ):
        covered, eligible, team_covered, team_eligible = _aura_counts(
            config, state, coverage, class_id
        )
        delta[f"{name}_coverage"] = covered
        delta[f"{name}_eligible"] = eligible
        delta[f"{name}_team_coverage"] = team_covered
        delta[f"{name}_team_eligible"] = team_eligible

    return {
        name: total
        + jnp.where(facts.has_transition, delta[name], 0).astype(total.dtype)
        for name, total in totals.items()
    }


def _available(value: Array) -> MetricValues:
    return MetricValues(value, jnp.ones(value.shape, bool))


def _ratio(numerator: Array, denominator: Array) -> MetricValues:
    valid = denominator > 0
    return MetricValues(numerator / jnp.where(valid, denominator, 1), valid)


def full_values(
    totals: FullTotals, config: EnvConfig, priority: MetricValues
) -> MetricValues:
    """Reduce sufficient statistics to the fixed catalog, copying priority values.

    The mappings are static tracing structure, not host dispatch in a rollout.
    Only numerical reductions, ratios and fixed gathers enter compiled execution.
    """
    data: dict[tuple[MetricScope, str], MetricValues] = {}

    def add(
        name: str,
        *,
        agent: MetricValues | None = None,
        team: MetricValues | None = None,
        pair: MetricValues | None = None,
    ) -> None:
        if agent is not None:
            data[("agent", name)] = agent
        if team is not None:
            data[("team", name)] = team
        if pair is not None:
            data[("source_recipient", name)] = pair
            data[("ally_pair", name)] = pair

    def amount(name: str, agents: Array, teams: Array | None = None) -> None:
        add(
            name,
            agent=_available(agents),
            team=_available(_team_sum(agents) if teams is None else teams),
        )

    def fraction_of_team(name: str, agents: Array) -> None:
        add(name, agent=_ratio(agents, jnp.repeat(_team_sum(agents), 5)))

    deaths = totals["deaths"]
    kills = _team_sum(deaths)[::-1]
    contributions = totals["kills"].sum(axis=1)
    for index, name in enumerate(("basic_activations", "ultimate_activations")):
        amount(name, totals["activations"][:, index])
    add("deaths", agent=_available(deaths))
    fraction_of_team("death_fraction", deaths)
    amount("dead_steps", totals["dead_steps"])
    fraction_of_team("dead_step_fraction", totals["dead_steps"])
    add("kill_contributions", agent=_available(contributions))
    add("kill_contributions_per_death", agent=_ratio(contributions, deaths))
    add("kill_participation", agent=_ratio(contributions, jnp.repeat(kills, 5)))
    for effect in ("damage", "healing"):
        matrix = totals[effect]
        sources = matrix.sum(axis=1)
        amount(f"{effect}_done", sources)
        fraction_of_team(f"{effect}_done_fraction", sources)
        add(f"{effect}_done", pair=_available(matrix))
        team_recipient = jnp.repeat(matrix.reshape(2, 5, 10).sum(axis=1), 5, axis=0)
        add(f"{effect}_done_fraction", pair=_ratio(matrix, team_recipient))
    damage_received = totals["damage"].sum(axis=0)
    priest_received = totals["healing"].sum(axis=0)
    regeneration = totals["regeneration"]
    healing_received = priest_received + regeneration
    amount("damage_received", damage_received)
    fraction_of_team("damage_received_fraction", damage_received)
    amount("priest_healing_received", priest_received)
    amount("regenerated_healing", regeneration)
    amount("healing_received", healing_received)
    for name, component in (
        ("priest", priest_received),
        ("regeneration", regeneration),
    ):
        add(
            f"{name}_healing_received_fraction",
            agent=_ratio(component, healing_received),
            team=_ratio(_team_sum(component), _team_sum(healing_received)),
        )
    fraction_of_team("healing_received_fraction", healing_received)
    amount("wasted_healing", totals["waste"].sum(axis=1))
    add("wasted_healing_received", agent=_available(totals["waste"].sum(axis=0)))
    for channel, status in enumerate(STATUS_NAMES[:7]):
        for effect in ("damage", "healing"):
            amount(
                f"{effect}_to_{status}_recipient",
                totals[f"controlled_{effect}"][:, channel],
            )
        team_kills = totals["controlled_kills"][:, channel]
        agent_kills = totals["controlled_contributions"][:, channel]
        add(f"kills_of_{status}_recipient", team=_available(team_kills))
        add(f"kill_contributions_to_{status}_recipient", agent=_available(agent_kills))
        add(
            f"kill_participation_in_{status}_recipient",
            agent=_ratio(agent_kills, jnp.repeat(team_kills, 5)),
        )
    add("single_contributor_kills", team=_available(totals["single_kills"]))
    add("multi_contributor_kills", team=_available(totals["multi_kills"]))
    add("single_contributor_kill_fraction", team=_ratio(totals["single_kills"], kills))
    add("multi_contributor_kill_fraction", team=_ratio(totals["multi_kills"], kills))
    add(
        "focus_fire_concentration",
        team=_ratio(totals["focus_sum"], totals["focus_steps"]),
    )
    add("focus_fire_steps", team=_available(totals["focus_steps"]))
    for index, name in enumerate(
        (
            "actions_submitted",
            "actions_accepted",
            "actions_rejected",
            "action_domain_rejections",
            "action_movement_rejections",
            "action_combat_rejections",
        )
    ):
        amount(name, totals["actions"][:, index])
    add(
        "action_acceptance_rate",
        agent=_ratio(totals["actions"][:, 1], totals["actions"][:, 0]),
        team=_ratio(
            _team_sum(totals["actions"][:, 1]), _team_sum(totals["actions"][:, 0])
        ),
    )
    for channel, status in enumerate(STATUS_NAMES):
        amount(f"{status}_applications", totals["applications"][:, channel])
        amount(f"{status}_active_steps", totals["active_steps"][:, channel])
    add("trap_intervals", team=_available(totals["trap_periods"]))
    add("trap_breaks", team=_available(totals["trap_breaks"]))
    add("trap_break_rate", team=_ratio(totals["trap_breaks"], totals["trap_periods"]))
    add(
        "trap_mean_remaining_steps_at_break",
        team=_ratio(totals["trap_remaining"], totals["trap_breaks"]),
    )
    add("trap_break_contributions", agent=_available(totals["trap_contributions"]))
    add(
        "trap_break_participation",
        agent=_ratio(
            totals["trap_contributions"], jnp.repeat(totals["trap_breaks"], 5)
        ),
    )
    add("respawn_waves", team=_available(totals["waves"]))
    add(
        "mean_agents_per_respawn_wave", team=_ratio(totals["respawns"], totals["waves"])
    )
    amount("burst_damage", totals["burst_damage"])
    amount("burst_damage_contributing_to_kill", totals["burst_lethal_damage"])
    amount("burst_kill_contributions", totals["burst_kills"])
    mage_damage = jnp.where(
        config.agent_profile.class_ids == 1, totals["damage"].sum(axis=1), 0
    )
    add(
        "burst_damage_fraction",
        agent=_ratio(totals["burst_damage"], mage_damage),
        team=_ratio(_team_sum(totals["burst_damage"]), _team_sum(mage_damage)),
    )
    for name in ("mage", "warrior"):
        amount(
            f"{name}_aura_covered_steps",
            totals[f"{name}_coverage"],
            totals[f"{name}_team_coverage"],
        )
        amount(
            f"{name}_aura_eligible_steps",
            totals[f"{name}_eligible"],
            totals[f"{name}_team_eligible"],
        )
        add(
            f"{name}_aura_coverage",
            agent=_ratio(totals[f"{name}_coverage"], totals[f"{name}_eligible"]),
            team=_ratio(
                totals[f"{name}_team_coverage"], totals[f"{name}_team_eligible"]
            ),
        )
    add("damage_from_mage_aura", team=_available(totals["mage_damage_gain"]))
    add(
        "damage_prevented_by_warrior_aura",
        team=_available(totals["warrior_damage_prevented"]),
    )
    amount("healing_prevented_by_poison", totals["poison_prevention"])
    add("rescue_opportunities", team=_available(totals["rescue_opportunities"]))
    add("rescues", team=_available(totals["rescues"]))
    add("rescue_rate", team=_ratio(totals["rescues"], totals["rescue_opportunities"]))
    add("rescue_contributions", agent=_available(totals["rescue_contributions"]))
    add(
        "rescue_participation",
        agent=_ratio(totals["rescue_contributions"], jnp.repeat(totals["rescues"], 5)),
    )
    amount("freedom_protected_steps", totals["freedom_protected"])
    amount("freedom_eligible_steps", totals["freedom_eligible"])
    add(
        "freedom_protection_fraction",
        agent=_ratio(totals["freedom_protected"], totals["freedom_eligible"]),
        team=_ratio(
            _team_sum(totals["freedom_protected"]),
            _team_sum(totals["freedom_eligible"]),
        ),
    )
    distances = totals["distance_sum"]
    distance_count = totals["distance_count"]
    add(
        "ally_distance_mean",
        pair=_ratio(distances, distance_count),
        team=_ratio(
            _team_sum(distances.sum(axis=1)), _team_sum(distance_count.sum(axis=1))
        ),
    )
    add(
        "ally_distance_observations",
        pair=_available(distance_count),
        team=_available(_team_sum(distance_count.sum(axis=1))),
    )

    values: list[Array] = []
    valid: list[Array] = []
    active = config.agent_profile.active_mask
    classes = config.agent_profile.class_ids
    for column in METRIC_COLUMNS[len(PRIORITY_METRIC_COLUMNS) :]:
        if column.scope == "team":
            prefix = "team_a_" if column.subjects[0] == 1 else "team_b_"
            index = (column.subjects[0] - 1,)
            subject_available = active.reshape(2, 5)[index[0]].any()
            class_available = (
                ((classes == column.required_class_id) & active)
                .reshape(2, 5)[index[0]]
                .any()
                if column.required_class_id is not None
                else jnp.asarray(True)
            )
        else:
            prefix = "".join(f"agent_{slot}_" for slot in column.subjects)
            index = column.subjects
            subject_available = active[jnp.asarray(index)].all()
            class_available = (
                classes[index[0]] == column.required_class_id
                if column.required_class_id is not None
                else jnp.asarray(True)
            )
        measurement = data[(column.scope, column.name[len(prefix) :])]
        values.append(measurement.values[index])
        valid.append(measurement.valid[index] & subject_available & class_available)
    return MetricValues(
        jnp.concatenate((priority.values, jnp.stack(values).astype(jnp.float32))),
        jnp.concatenate((priority.valid, jnp.stack(valid))),
    )
